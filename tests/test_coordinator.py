"""
crystal_search/coordinator.py: shards, the basin registry, the stopping rule, and the run_search hooks.

1. rdf_distance_matrix equals analysis.crystal_rdf.compute_rdf_distance.
2. Search jobs with cfg:coord_dir write one shard per completed batch carrying the energy-model id and the effort, and a
   shard write that fails never stops the search (fail-open).
3. A curate pass assigns identical structures to one basin and distinct ones to separate basins, counts effort per stream,
   is idempotent (re-running ingests nothing twice), and refuses shards scored by another energy model.
4. Stopping rule: the exact Poisson bound (f1_hi(0) = 2.30); a stream stops only after passes_to_confirm consecutive
   passes that all say stop; the campaign STOP follows when every stream has stopped; a job then stops between batches.
5. The real optimiser (eLJ, CPU) produces shards with row-evaluations and energies.

CPU only.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import mxtaltools.crystal_search.coordinator as co
import mxtaltools.crystal_search.run_search as rs
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.dataset_utils.data_classes import MolCrystalData

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']


def _strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


@pytest.fixture(scope='module')
def acridine():
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    data = torch.load(ACRIDINE, weights_only=False)
    zp1 = [_strip(c) for c in data if int(c.sg_ind) == 14 and int(c.z_prime) == 1]
    zp2 = [_strip(c) for c in data if int(c.sg_ind) == 14 and int(c.z_prime) == 2]
    return zp1, zp2


def _campaign(tmp_path, zp1, **over):
    conf = tmp_path / 'conformer.pt'
    torch.save(zp1[0], conf)
    cfg = dict(mol_path=str(conf), sg=14, z_prime=2, energy_key='elj', energy_model_id='elj', kT=10.0,
               bands_kT={'b2': 2.0}, identity_cut=0.085, identity_calibrated=False, window_kT=100.0,
               streams={'random': dict(Z={'b2': 1e12}, min_relaxations=0, min_hits=1)}, passes_to_confirm=2)
    cfg.update(over)
    d = tmp_path / 'campaign'
    d.mkdir()
    (d / 'coord.yaml').write_text(yaml.safe_dump(cfg))
    return d, co.CampaignConfig.load(d / 'coord.yaml')


def _seeds(cfg, zp2, n, scale_step=0.04):
    """n Z'=2 crystals built from the campaign conformer: copies of a mini acridine cell, every other one stretched
    (a different packing by RDF)."""
    b = zp2[0]
    p = torch.cat([b.cell_lengths.reshape(-1), b.cell_angles.reshape(-1), b.aunit_centroid.reshape(-1)[:6],
                   b.aunit_orientation.reshape(-1)[:6]]).float()
    params, hand = [], []
    for i in range(n):
        q = p.clone()
        q[:3] = q[:3] * (1 + scale_step * (i % 2))
        params.append(q)
        hand.append(b.aunit_handedness.reshape(-1)[:2].float())
    return co.rebuild_crystals(cfg, torch.stack(params), torch.stack(hand))


def _search_config(tmp_path, n, coord_dir, stream='random', **kw):
    cfg = {'device': 'cpu', 'target_path': None, 'out_dir': str(tmp_path), 'run_name': f'run_{stream}',
           'force_restart_run': False, 'batch_size': 3, 'grow_batch_size': False, 'save_trajs': False,
           'num_samples': n, 'init_sample_method': 'data', 'init_reduced': True, 'init_target_cp': 'std',
           'opt_seed': 0, 'mol_seed': 0, 'dataset_path': None, 'lease_settle_s': 0.0,
           'coord_dir': None if coord_dir is None else str(coord_dir), 'coord_stream': stream,
           'opt': [{'optim_target': 'elj', 'init_lr': 0.01, 'max_num_steps': 3, 'optimizer_func': 'rprop',
                    'show_tqdm': False}]}
    cfg.update(kw)
    return dict2namespace(cfg)


def _fake_optimiser(energy_of):
    def fake(self, return_record=False, **kwargs):
        out = self.batch_to_list()
        for c in out:
            c.elj = torch.tensor([energy_of(c)])
            c.lj = torch.tensor([-500.0])
        rec = dict(loss=torch.zeros(4, len(out)))  # 4 steps x rows
        return (out, rec) if return_record else out
    return fake


def _energy(c):
    return -1000.0 if float(c.cell_lengths.reshape(-1)[0]) < 6.1 else -995.0


# ---------------------------------------------------------------------------

def test_rdf_distance_matrix_equals_the_library_distance(acridine, tmp_path):
    from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
    _, cfg = _campaign(tmp_path, acridine[0])
    r = co.compute_rdfs(_seeds(cfg, acridine[1], 4, scale_step=0.02))
    D = co.rdf_distance_matrix(r, r)
    bins = torch.linspace(0, 10, r.shape[-1])
    for i in range(4):
        ref = compute_rdf_distance(r[i], r, bins).flatten()
        assert torch.allclose(D[i], ref.float(), atol=1e-6), (D[i], ref)


def test_shards_registry_idempotence_and_model_refusal(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    shards = sorted((coord_dir / 'shards' / 'run_random').glob('*.pt'))
    assert len(shards) == 2, 'one shard per completed batch'
    s0 = torch.load(shards[0], weights_only=False)
    assert s0['energy_model_id'] == 'elj' and s0['row_evals'] == 12 and s0['n_relaxations'] == 3
    assert s0['params'].shape == (3, 18)

    stats, decisions, stop = co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(reg['basin_E']) == 2, 'three copies of each of two packings: two basins'
    assert sorted(np.bincount(reg['hits']['basin']).tolist()) == [3, 3]
    assert stats['streams']['random']['relaxations'] == 6 and stats['streams']['random']['row_evals'] == 24

    co.curate(str(coord_dir))  # idempotent: nothing new
    reg2 = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(reg2['hits']['basin']) == 6

    bad = dict(s0, energy_model_id='mace:other.model:1:abc', run='intruder')
    os.makedirs(coord_dir / 'shards' / 'intruder')
    torch.save(bad, coord_dir / 'shards' / 'intruder' / '00000000_000000.pt')
    co.curate(str(coord_dir))
    reg3 = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(reg3['hits']['basin']) == 6, 'a shard from another energy model must not enter the registry'
    assert any('intruder' == r['run'] for r in reg3['refused'])
    assert 'Refused inputs' in (coord_dir / 'stats.md').read_text()


def test_shard_write_is_fail_open(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    (coord_dir / 'shards').write_text('not a directory')  # every shard write will fail
    seeds = _seeds(cfg, acridine[1], 4)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    outs = rs.crystal_search(_search_config(tmp_path, 4, coord_dir))
    assert len(outs) == 4, 'the search must complete although no shard could be written'


def test_poisson_bound_and_confirmed_stop_then_job_stops(acridine, tmp_path, monkeypatch):
    assert abs(co.poisson_upper(0) - 2.3026) < 1e-3
    coord_dir, cfg = _campaign(tmp_path, acridine[0],
                               streams={'random': dict(Z={'b2': 1.0}, min_relaxations=0, min_hits=1)})
    seeds = _seeds(cfg, acridine[1], 12)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))  # half the budget
    _, d1, stop1 = co.curate(str(coord_dir))
    assert not d1['random']['stop'] and d1['random']['streak'] == 1, 'one pass is not enough to stop'
    assert not (coord_dir / 'STOP.random').exists()
    _, d2, stop2 = co.curate(str(coord_dir))
    assert d2['random']['stop'] and stop2, 'second consecutive pass confirms the stop'
    assert (coord_dir / 'STOP.random').exists() and (coord_dir / 'STOP').exists()
    outs = rs.crystal_search(_search_config(tmp_path, 12, coord_dir))  # resumes, but must stop before a new batch
    assert len(outs) == 6, 'a job must stop between batches once its stream is stopped'


def test_run_loop_terminates_after_the_campaign_stop(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0],
                               streams={'random': dict(Z={'b2': 1.0}, min_relaxations=0, min_hits=1)})
    seeds = _seeds(cfg, acridine[1], 3)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 3, coord_dir))
    co.run_loop(str(coord_dir), interval_s=0.0, max_hours=0.01)  # must return: stop confirmed, then one more pass
    hb = json.loads((coord_dir / 'heartbeat.json').read_text())
    assert hb['passes'] >= 3


def test_real_optimiser_writes_usable_shards(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    seeds = _seeds(cfg, acridine[1], 2)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    rs.crystal_search(_search_config(tmp_path, 2, coord_dir, batch_size=2))
    s = torch.load(next((coord_dir / 'shards' / 'run_random').glob('*.pt')), weights_only=False)
    assert s['row_evals'] > 0 and torch.isfinite(s['energy']).all() and len(s['energy']) == 2
    co.curate(str(coord_dir))
    assert (coord_dir / 'stats.md').exists()


def test_effort_excludes_row_steps_skipped_by_the_cascade(acridine, tmp_path, monkeypatch):
    """A row the cascade retires records +inf for the steps it skipped; those steps were never evaluated, so the shard's
    row_evals (the stopping rule's effort) must not count them."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    seeds = _seeds(cfg, acridine[1], 3)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])

    def fake(self, return_record=False, **kwargs):
        out = self.batch_to_list()
        for c in out:
            c.elj = torch.tensor([_energy(c)])
            c.lj = torch.tensor([-500.0])
        loss = torch.zeros(4, len(out))  # 4 steps x 3 rows
        loss[2:, 0] = float('inf')  # row 0 retired after step 1: 2 of its 4 steps never ran
        return (out, dict(loss=loss)) if return_record else out
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', fake)
    rs.crystal_search(_search_config(tmp_path, 3, coord_dir, batch_size=3))
    s = torch.load(next((coord_dir / 'shards' / 'run_random').glob('*.pt')), weights_only=False)
    assert s['row_evals'] == 4 * 3 - 2


# ---------------------------------------------------------------------------
# hop stream
# ---------------------------------------------------------------------------

def test_hop_stream_quota_generations_and_clean_stop(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=3,
                                                              max_generation=2, log_noise=[-1.0, -1.0]))
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))  # random stream: two basins
    co.curate(str(coord_dir))
    hp = torch.load(coord_dir / 'hop_parents.pt', weights_only=False)
    assert sorted(hp['basin'].tolist()) == [0, 1] and hp['hop_starts'].tolist() == [0, 0]

    with pytest.raises(ValueError, match='coord_dir'):
        rs.crystal_search(_search_config(tmp_path, 3, None, stream='hops', init_sample_method='hops',
                                         run_name='nocampaign'))
    outs = rs.crystal_search(_search_config(tmp_path, 6, coord_dir, stream='hops', init_sample_method='hops'))
    parents = sorted(set(int(o.dataset_index) for o in outs))
    assert set(parents) <= {0, 1} and len(outs) == 6, 'every hop output names its parent basin'
    a_par = {0: float(seeds[0].cell_lengths.reshape(-1)[0]), 1: float(seeds[1].cell_lengths.reshape(-1)[0])}
    assert any(abs(float(o.cell_lengths.reshape(-1)[0]) - a_par[int(o.dataset_index)]) > 1e-4 for o in outs), \
        'hop starts must be kicked, not copies of their parents'
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert sum(reg['hop_starts'].values()) == 6
    new = [j for j, f in enumerate(reg['basin_first']) if f['stream'] == 'hops']
    assert all(reg['basin_gen'][j] == 1 for j in new), 'a basin first found by a hop is one generation down'
    hp = torch.load(coord_dir / 'hop_parents.pt', weights_only=False)
    assert all(reg['hop_starts'].get(int(b), 0) < 3 for b in hp['basin']), 'parents past their quota drop out'

    # exhaust every quota: parents vanish and a hop job then stops without relaxing anything
    for j in range(len(reg['basin_E'])):
        reg['hop_starts'][j] = 99
    torch.save(reg, coord_dir / 'registry.pt')
    co.curate(str(coord_dir))
    assert len(torch.load(coord_dir / 'hop_parents.pt', weights_only=False)['basin']) == 0
    before = len(outs)
    outs2 = rs.crystal_search(_search_config(tmp_path, 12, coord_dir, stream='hops', init_sample_method='hops'))
    assert len(outs2) == before, 'no eligible parent: the job stops cleanly between batches'


def test_search_jobs_curate_between_batches_one_at_a_time(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir, coord_curate_every_s=0.0))
    hb = json.loads((coord_dir / 'heartbeat.json').read_text())
    assert hb['passes'] >= 1 and (coord_dir / 'registry.pt').exists(), 'the search job ran curate passes'
    # a pass younger than the interval is skipped; a held curator lease is skipped, never waited on
    assert co.maybe_curate(str(coord_dir), 3600.0) is False
    from mxtaltools.crystal_search.run_state import RunLease
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    held = RunLease(str(coord_dir / 'curator'), stale_after_s=3600, settle_s=0)
    held.acquire()
    monkeypatch.setenv('SLURM_JOB_ID', '78')
    assert co.maybe_curate(str(coord_dir), 0.0) is False, 'another job is curating: skip'
    held_rec = json.loads((coord_dir / 'curator_owner.json').read_text())
    assert held_rec['job'] == '77'
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    held.release()
    monkeypatch.setenv('SLURM_JOB_ID', '78')
    assert co.maybe_curate(str(coord_dir), 0.0) is True


def test_export_writes_one_reduced_crystal_per_basin(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    co.curate(str(coord_dir))
    n = co.export(str(coord_dir), 100.0, str(tmp_path / 'final.pt'))
    out = torch.load(tmp_path / 'final.pt', weights_only=False)
    assert n == len(out) == 2
    assert [float(c.elj) for c in out] == sorted(float(c.elj) for c in out), 'lowest first'
    assert float(collate_data_list(out).compute_cell_reduction_penalty().max()) < 1e-6, 'reduced cells'
    assert (tmp_path / 'final.csv').read_text().count('\n') == 3
