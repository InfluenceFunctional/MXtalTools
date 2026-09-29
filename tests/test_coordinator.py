"""
crystal_search/coordinator.py: shards, the basin registry, the stopping rule, and the run_search hooks.

1. rdf_distance_matrix equals analysis.crystal_rdf.compute_rdf_distance.
2. Search jobs with cfg:coord_dir write one shard per completed batch carrying the energy-model id and the effort, and a
   shard write that fails never stops the search (fail-open).
3. A curate pass assigns identical structures to one basin and distinct ones to separate basins, counts effort per stream,
   is idempotent (re-running ingests nothing twice), and refuses shards scored by another energy model.
4. Stopping rule: the exact Poisson bound (f1_hi(0) = 2.30); a stream stops only after passes_to_confirm consecutive
   passes that all say stop, each after new work from the stream (two passes over the same data are one look); a stream
   whose STOP.<s> exists stays stopped; the campaign STOP follows when every stream has stopped (a hop stream with no
   eligible parent counts as stopped); a job then stops between batches.
5. The real optimiser (eLJ, CPU) produces shards with row-evaluations and energies.
6. A hop state that relaxed back into its parent basin, or an ancestor of it, is left out of the Good-Turing counts.
7. A hop job waits (bounded) while hop_parents.pt is missing or stale, and stops only on an empty list from a pass that
   followed its own last shard.
8. export first ingests the shards written after the last pass.

CPU only.
"""
import json
import math
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
    _, d1b, _ = co.curate(str(coord_dir))
    assert not d1b['random']['stop'] and d1b['random']['streak'] == 1, 'a second look at the same data confirms nothing'
    rs.crystal_search(_search_config(tmp_path, 9, coord_dir))  # new work from the stream
    _, d2, stop2 = co.curate(str(coord_dir))
    assert d2['random']['stop'] and stop2, 'a second ok pass after new work confirms the stop'
    assert (coord_dir / 'STOP.random').exists() and (coord_dir / 'STOP').exists()
    outs = rs.crystal_search(_search_config(tmp_path, 12, coord_dir))  # resumes, but must stop before a new batch
    assert len(outs) == 9, 'a job must stop between batches once its stream is stopped'


def test_run_loop_terminates_after_the_campaign_stop(acridine, tmp_path, monkeypatch, capsys):
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hard_cap=1.0,
                               streams={'random': dict(Z={'b2': 1e12}, min_relaxations=0, min_hits=1)})
    seeds = _seeds(cfg, acridine[1], 3)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 3, coord_dir))
    co.run_loop(str(coord_dir), interval_s=0.0, max_hours=1.0)  # the hard cap stops the campaign: two passes, then out
    assert 'campaign stopped' in capsys.readouterr().out
    assert json.loads((coord_dir / 'heartbeat.json').read_text())['passes'] == 2


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

def test_hop_stream_quota_generations_and_clean_stop(acridine, tmp_path, monkeypatch, capsys):
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=3,
                                                              max_generation=2, log_noise=[-1.0, -1.0]))
    seeds = _seeds(cfg, acridine[1], 12)
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
    capsys.readouterr()
    outs2 = rs.crystal_search(_search_config(tmp_path, 12, coord_dir, stream='hops', init_sample_method='hops'))
    assert 'the stream is exhausted' in capsys.readouterr().out, 'the job reached the exhaustion exit'
    assert len(outs2) == before, 'no eligible parent: the job stops cleanly between batches'
    progress = json.loads((tmp_path / 'run_hops_progress.json').read_text())
    assert progress['cursor'] == 6 < 12, 'it stopped with samples left, not because the budget was done'


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


# ---------------------------------------------------------------------------
# lineage returns, the confirmation streak, sticky stops, hop waits, export catch-up
# ---------------------------------------------------------------------------

def _registry_with_hits(cfg, hits):
    """A registry whose basins and hits are given directly: hits = [(basin, stream, energy, parent)]; basin_parent is
    set from the first hit of each basin."""
    reg = co.new_registry(cfg)
    nb = max(b for b, *_ in hits) + 1
    reg['basin_E'] = [0.0] * nb
    reg['basin_first'] = [None] * nb
    reg['basin_parent'] = [-1] * nb
    for b, st, e, par in hits:
        if reg['basin_first'][b] is None:
            reg['basin_first'][b] = dict(stream=st, run='r', cursor=0)
            reg['basin_parent'][b] = par
        for k, v in zip(('basin', 'stream', 'energy', 'run', 'cursor', 'parent'), (b, st, e, 'r', 0, par)):
            reg['hits'][k].append(v)
    reg['effort'] = {'random': dict(row_evals=1000.0, relaxations=10), 'hops': dict(row_evals=1000.0, relaxations=10)}
    return reg


def test_hop_returns_into_their_own_lineage_leave_the_counts(acridine, tmp_path):
    _, cfg = _campaign(tmp_path, acridine[0], energy_ref=0.0,
                       streams={'random': dict(Z={'b2': 1.0}, min_relaxations=0, min_hits=1)})
    # random finds basins 0 and 1 once each; hops from 0 open basin 2; then:
    #   a hop from 0 relaxes back into 0 (a return: must not erase random's singleton 0),
    #   a hop from 2 relaxes into 0 (a return into 2's ancestor: likewise),
    #   a hop from 0 lands in 1 (an independent re-find: erases random's singleton 1)
    reg = _registry_with_hits(cfg, [(0, 'random', 0.0, -1), (1, 'random', 0.0, -1), (2, 'hops', 0.0, 0),
                                    (0, 'hops', 0.0, 0), (0, 'hops', 0.0, 2), (1, 'hops', 0.0, 0)])
    assert co.lineage_returns(reg).tolist() == [False, False, False, True, True, False]
    st = co.compute_stats(reg, cfg)
    assert st['streams']['random']['bands']['b2']['f1'] == 1, 'basin 0 stays random\'s singleton; basin 1 does not'
    assert st['streams']['hops']['bands']['b2']['f1'] == 1 and st['lineage_returns'] == 2
    reg['hits'].pop('parent')  # a registry from before the lineage record counts every hit, as it used to
    assert co.compute_stats(reg, cfg)['streams']['random']['bands']['b2']['f1'] == 0


def test_the_stop_streak_advances_only_on_new_work(acridine, tmp_path):
    _, cfg = _campaign(tmp_path, acridine[0], energy_ref=0.0,
                       streams={'random': dict(Z={'b2': 1.0}, min_relaxations=0, min_hits=1)})
    reg = _registry_with_hits(cfg, [(0, 'random', 0.0, -1), (0, 'random', 0.0, -1)])
    st = co.compute_stats(reg, cfg)
    d, stop, _ = co.verdicts(st, reg, cfg)
    assert d['random']['streak'] == 1 and not d['random']['stop']
    d, stop, _ = co.verdicts(st, reg, cfg)
    assert d['random']['streak'] == 1 and not stop, 'the same data looked at twice'
    reg['effort']['random']['row_evals'] += 10.0
    d, stop, _ = co.verdicts(co.compute_stats(reg, cfg), reg, cfg)
    assert d['random']['streak'] == 2 and d['random']['stop'] and stop


def test_a_stopped_stream_stays_stopped_and_an_exhausted_hop_stream_lets_the_campaign_stop(acridine, tmp_path,
                                                                                          monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=1,
                                                              exhaust_settle_s=0.5),
                               streams={'random': dict(Z={'b2': 1e12}, min_relaxations=0, min_hits=1),
                                        'hops': dict(Z={'b2': 1e12}, min_relaxations=0, min_hits=1)})
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    (coord_dir / 'STOP.random').write_text('by hand\n')
    _, d, stop = co.curate(str(coord_dir))
    assert d['random']['stop'] and 'STOP.random present' in d['random']['why'], 'the verdict says run; the file wins'
    assert not stop, 'hops still has parents to kick'
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    reg['hop_starts'] = {j: 5 for j in range(len(reg['basin_E']))}  # every quota used up
    reg['hop_effort_seen'] = -5.0  # hop work arrived since the previous pass: batches may still be in flight
    torch.save(reg, coord_dir / 'registry.pt')
    _, d, stop = co.curate(str(coord_dir))
    assert not stop, 'the pass that finds the list empty after new hop work may still have hop batches in flight'
    _, d, stop = co.curate(str(coord_dir))
    assert not stop, 'a second pass at once is not enough: no new hop work must hold for exhaust_settle_s'
    import time
    time.sleep(0.6)
    _, d, stop = co.curate(str(coord_dir))
    assert not d['hops']['stop'] and stop and (coord_dir / 'STOP').exists(), \
        'no parent left, no new hop work since the last pass, every other stream stopped: the campaign stops'
    assert not (coord_dir / 'STOP.hops').exists(), 'exhaustion is not a hops verdict: a new parent could revive it'


def test_hop_batch_waits_for_a_missing_or_stale_parent_list(acridine, tmp_path):
    coord = dict(dir=str(tmp_path), model_id='elj')
    cfg = _search_config(tmp_path, 3, tmp_path, stream='hops', init_sample_method='hops')
    assert rs._hop_batch(cfg, [], 0, 3, coord, 'cpu', 0) is rs.HOP_WAIT, 'no file yet'
    (tmp_path / 'hop_parents.pt').write_bytes(b'truncated')
    assert rs._hop_batch(cfg, [], 0, 3, coord, 'cpu', 0) is rs.HOP_WAIT, 'unreadable'
    torch.save(dict(basin=torch.zeros(0, dtype=torch.long), passes=4, energy_model_id='elj'),
               tmp_path / 'hop_parents.pt')
    assert rs._hop_batch(cfg, [], 0, 3, coord, 'cpu', 0, after_pass=6) is rs.HOP_WAIT, 'empty, but from an old pass'
    assert rs._hop_batch(cfg, [], 0, 3, coord, 'cpu', 0, after_pass=4) is None, 'empty from a recent pass: exhausted'


def test_a_hop_job_waits_for_the_first_pass_and_then_runs(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=8))
    seeds = _seeds(cfg, acridine[1], 12)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))  # shards exist, but no pass has run: no parent file
    naps = []

    def first_pass_arrives(seconds):  # stands in for another job finishing the first curate pass during the wait
        naps.append(seconds)
        co.curate(str(coord_dir))
    monkeypatch.setattr(rs, 'sleep', first_pass_arrives)
    outs = rs.crystal_search(_search_config(tmp_path, 3, coord_dir, stream='hops', init_sample_method='hops',
                                            coord_hop_poll_s=0.01))
    assert len(naps) == 1 and len(outs) == 3, 'waited once, then relaxed its hop starts'

    (coord_dir / 'hop_parents.pt').unlink()
    monkeypatch.setattr(rs, 'sleep', lambda seconds: None)
    with pytest.raises(rs.HopParentsUnavailable):  # the wait is bounded
        rs.crystal_search(_search_config(tmp_path, 3, coord_dir, stream='hops', init_sample_method='hops',
                                         run_name='hops_bounded', coord_hop_wait_s=0.05, coord_hop_poll_s=0.01))


def test_export_ingests_the_shards_written_after_the_last_pass(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 3, coord_dir))
    co.curate(str(coord_dir))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))  # a last batch after the last pass
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(co.pending_shards(str(coord_dir), reg)) == 1
    co.export(str(coord_dir), 100.0, str(tmp_path / 'final.pt'))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert co.pending_shards(str(coord_dir), reg) == [] and len(reg['hits']['basin']) == 6



# ---------------------------------------------------------------------------
# relaunch from an older registry: lineage backfill, content-keyed priors, run_loop and the curator lease
# ---------------------------------------------------------------------------

def _hop_campaign(acridine, tmp_path, monkeypatch):
    """A random job, a curate pass, a hop job and a second pass: a registry whose hits carry hop parents."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=8,
                                                              log_noise=[-1.0, -1.0]))
    seeds = _seeds(cfg, acridine[1], 12)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    co.curate(str(coord_dir))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir, stream='hops', init_sample_method='hops'))
    co.curate(str(coord_dir))
    return coord_dir, cfg


def test_an_older_registry_gets_its_lineage_back_from_the_shards(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _hop_campaign(acridine, tmp_path, monkeypatch)
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    want_hits, want_basins = list(reg['hits']['parent']), list(reg['basin_parent'])
    assert sum(p >= 0 for p in want_hits) == 6, 'setup: six hop hits with parents'
    # the registry as the code before the lineage record wrote it
    reg['hits'].pop('parent')
    reg.pop('basin_parent')
    torch.save(reg, coord_dir / 'registry.pt')
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert reg['hits']['parent'] == want_hits and reg['basin_parent'] == want_basins
    assert reg['migrations'][-1]['hop_hits'] == 6 and reg['migrations'][-1]['unmatched'] == 0
    # older code extended the registry since: hits without a parent entry are backfilled again, not misaligned
    for k in ('basin', 'stream', 'energy', 'run', 'cursor'):
        reg['hits'][k].append(reg['hits'][k][-1])
    torch.save(reg, coord_dir / 'registry.pt')
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(reg['hits']['parent']) == len(reg['hits']['basin'])


def test_a_prior_file_whose_content_changed_is_ingested_again(acridine, tmp_path):
    prior = tmp_path / 'known.pth'
    coord_dir, cfg = _campaign(tmp_path, acridine[0], priors=[str(prior)])
    seeds = _seeds(cfg, acridine[1], 4)
    b = collate_data_list(seeds[:2])

    def write(rows):
        n = len(rows)
        torch.save(dict(params=b.full_cell_parameters()[rows].detach().float(),
                        handedness=b.aunit_handedness.reshape(2, -1)[rows].float(),
                        energy=torch.full((n,), -1000.0), energy_model_id='elj'), prior)
    write([0])
    co.curate(str(coord_dir))
    n1 = len(torch.load(coord_dir / 'registry.pt', weights_only=False)['hits']['basin'])
    co.curate(str(coord_dir))
    assert len(torch.load(coord_dir / 'registry.pt', weights_only=False)['hits']['basin']) == n1, 'same content: once'
    write([0, 1])
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(reg['hits']['basin']) == n1 + 2, 'new content at the same path: its states are added'


def test_run_loop_skips_a_pass_while_a_job_holds_the_curator_lease(acridine, tmp_path, monkeypatch, capsys):
    from mxtaltools.crystal_search.run_state import RunLease
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    held = RunLease(str(coord_dir / 'curator'), stale_after_s=3600, settle_s=0)
    held.acquire()  # a search job is curating
    monkeypatch.setenv('SLURM_JOB_ID', '5151')
    co.run_loop(str(coord_dir), interval_s=0.0, max_hours=1e-9)  # must return, not raise and not wait
    assert 'pass skipped, curator lease held' in capsys.readouterr().out
    assert not (coord_dir / 'heartbeat.json').exists()
    held.release()



def test_old_rule_streaks_do_not_carry_over(acridine, tmp_path, monkeypatch):
    """A registry from the older rule, where a pass needed no new work: its streaks are reset at migration, so the first
    pass of the fixed code cannot confirm a stop by itself."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0],
                               streams={'random': dict(Z={'b2': 1.0}, min_relaxations=0, min_hits=1)})
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    reg['confirm'] = {'random': 1}  # as the older rule left it: one ok look
    reg.pop('confirm_effort')
    torch.save(reg, coord_dir / 'registry.pt')
    _, d, stop = co.curate(str(coord_dir))
    assert d['random']['streak'] == 1 and not d['random']['stop'] and not (coord_dir / 'STOP.random').exists()


def test_an_older_registry_with_no_hits_is_migrated(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    reg = co.new_registry(cfg)
    reg['hits'].pop('parent')  # the older layout, before any state arrived
    reg.pop('basin_parent')
    torch.save(reg, coord_dir / 'registry.pt')
    seeds = _seeds(cfg, acridine[1], 3)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 3, coord_dir))
    co.curate(str(coord_dir))  # used to raise KeyError('parent') at the first admitted row
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert len(reg['hits']['parent']) == len(reg['hits']['basin']) == 3


def test_a_rewritten_hop_shard_leaves_its_hits_unmatched_not_misassigned(acridine, tmp_path, monkeypatch):
    coord_dir, cfg = _hop_campaign(acridine, tmp_path, monkeypatch)
    shard = sorted((coord_dir / 'shards' / 'run_hops').glob('*.pt'))[0]
    blob = torch.load(shard, weights_only=False)
    blob['dataset_index'] = blob['dataset_index'].flip(0)  # the redo drew other parents...
    blob['energy'] = blob['energy'].flip(0)                  # ...and its rows landed in another order
    torch.save(blob, shard)
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    reg['hits'].pop('parent')
    reg.pop('basin_parent')
    torch.save(reg, coord_dir / 'registry.pt')
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    m = reg['migrations'][-1]
    n_group = len(blob['energy'])
    assert m['unmatched'] == n_group, 'the rewritten shard does not match as a whole: its hits stay unmatched'
    assert sum(p >= 0 for p in reg['hits']['parent']) == m['hop_hits'] - n_group


def test_a_prior_that_cannot_be_read_is_skipped_not_fatal(acridine, tmp_path):
    coord_dir, cfg = _campaign(tmp_path, acridine[0], priors=[str(tmp_path / 'moved_away.pth')])
    co.curate(str(coord_dir))
    assert (coord_dir / 'heartbeat.json').exists()


def test_run_loop_survives_a_failed_pass(acridine, tmp_path, monkeypatch, capsys):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])

    def broken(coord_dir):
        raise OSError('scratch filesystem hiccup')
    monkeypatch.setattr(co, 'curate_locked', broken)
    co.run_loop(str(coord_dir), interval_s=0.0, max_hours=1e-9)  # returns through max_hours, not the exception
    assert 'pass failed (OSError' in capsys.readouterr().out



def test_a_prior_that_fails_to_load_is_refused_and_a_corrected_one_is_ingested(acridine, tmp_path):
    prior = tmp_path / 'known.pth'
    prior.write_bytes(b'not a torch file')
    coord_dir, cfg = _campaign(tmp_path, acridine[0], priors=[str(prior)])
    co.curate(str(coord_dir))  # used to raise out of every pass
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert any('prior not ingested' in r['reason'] for r in reg['refused']) and len(reg['hits']['basin']) == 0
    b = collate_data_list(_seeds(cfg, acridine[1], 2)[:1])
    torch.save(dict(params=b.full_cell_parameters().detach().float(), handedness=b.aunit_handedness.reshape(1, -1).float(),
                    energy=torch.full((1,), -1000.0), energy_model_id='elj'), prior)
    co.curate(str(coord_dir))
    assert len(torch.load(coord_dir / 'registry.pt', weights_only=False)['hits']['basin']) == 1


def test_backfill_matches_the_admitted_rows_whatever_the_reference_was(acridine, tmp_path):
    """Rows admitted at ingestion are a shard's physical rows at or below the window's edge at that time; the match
    takes the lowest physical rows, as many as the hits, so a reference that has fallen since does not break it."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops'))
    d = coord_dir / 'shards' / 'hops_0'
    d.mkdir(parents=True)
    params = torch.zeros(4, 18)
    params[:, :3], params[:, 3:6] = 10.0, math.pi / 2  # physical cells
    torch.save(dict(run='hops_0', stream='hops', cursor=0, batch_idx=0, params=params, handedness=torch.ones(4, 2),
                    energy=torch.tensor([-5.0, -3.0, -9.0, -1.0]), lj=torch.zeros(4),
                    dataset_index=torch.tensor([7, 8, 9, 6]), energy_model_id='elj'), d / '00000000_000000.pt')
    reg = co.new_registry(cfg)
    reg['hits'].pop('parent')
    reg.pop('basin_parent')
    for e in (-5.0, -3.0, -9.0):  # admitted then: every physical row at or below -3, in row order
        for k, v in zip(('basin', 'stream', 'energy', 'run', 'cursor'), (0, 'hops', e, 'hops_0', 0)):
            reg['hits'][k].append(v)
    reg['basin_E'] = [-12.0]  # the reference has fallen since: a replayed window would now drop -3 and -5
    reg['basin_parent'] = [-1]
    hop_hits, unmatched = co.backfill_lineage(reg, cfg, str(coord_dir))
    assert (hop_hits, unmatched) == (3, 0) and reg['hits']['parent'] == [7, 8, 9]


def test_hop_exhaustion_clock_starts_for_a_registry_without_it(acridine, tmp_path, monkeypatch):
    """A registry written before hop_effort_since existed carries hop_effort_seen alone. If hop work never changes
    again, the settle clock must still start, so an exhausted hop stream can count as stopped."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=1,
                                                              exhaust_settle_s=0.5),
                               streams={'random': dict(Z={'b2': 1e12}, min_relaxations=0, min_hits=1),
                                        'hops': dict(Z={'b2': 1e12}, min_relaxations=0, min_hits=1)})
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in seeds[:config.num_samples]])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    (coord_dir / 'STOP.random').write_text('by hand\n')
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    reg['hop_starts'] = {j: 5 for j in range(len(reg['basin_E']))}  # no eligible parent left
    reg['hop_effort_seen'] = float(reg['effort'].get('hops', {}).get('row_evals', 0.0))  # as the older code wrote it
    reg.pop('hop_effort_since', None)
    torch.save(reg, coord_dir / 'registry.pt')
    import time
    _, _, stop = co.curate(str(coord_dir))  # the clock starts here
    time.sleep(0.6)
    _, _, stop = co.curate(str(coord_dir))
    assert stop and (coord_dir / 'STOP').exists(), 'no new hop work for the settle span: the campaign can stop'


def test_an_atomwise_campaign_records_its_mode_and_refuses_a_change(acridine, tmp_path, monkeypatch):
    """rdf_mode 'atomwise' clusters on per-atom channels (acridine has more of them than environment classes), is written
    into the registry, and a coord.yaml that later changes it is refused, as a changed identity cut is."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0], rdf_mode='atomwise')
    seeds = _seeds(cfg, acridine[1], 6)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 6, coord_dir))
    co.curate(str(coord_dir))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    assert reg['rdf_mode'] == 'atomwise' and len(reg['basin_E']) == 2
    env = co.compute_rdfs(seeds[:1])
    assert reg['leaders'].shape[1] > env.shape[1], 'atomwise has a channel per atom pair type, envwise fewer'
    assert 'atomwise' in (coord_dir / 'stats.md').read_text()
    raw = yaml.safe_load((coord_dir / 'coord.yaml').read_text())
    (coord_dir / 'coord.yaml').write_text(yaml.safe_dump(dict(raw, rdf_mode='envwise')))
    with pytest.raises(ValueError, match='rdf_mode'):
        co.curate(str(coord_dir))
    (coord_dir / 'coord.yaml').write_text(yaml.safe_dump(dict(raw, rdf_mode='elementwise')))
    with pytest.raises(ValueError, match="rdf_mode must be"):
        co.CampaignConfig.load(coord_dir / 'coord.yaml')


def test_a_registry_without_a_recorded_mode_counts_as_envwise(acridine, tmp_path):
    coord_dir, cfg = _campaign(tmp_path, acridine[0])
    reg = co.new_registry(cfg)
    reg.pop('rdf_mode')  # as written before the setting existed
    torch.save(reg, coord_dir / 'registry.pt')
    co.curate(str(coord_dir))  # envwise campaign: accepted
    raw = yaml.safe_load((coord_dir / 'coord.yaml').read_text())
    (coord_dir / 'coord.yaml').write_text(yaml.safe_dump(dict(raw, rdf_mode='atomwise')))
    reg = torch.load(coord_dir / 'registry.pt', weights_only=False)
    reg.pop('rdf_mode', None)
    torch.save(reg, coord_dir / 'registry.pt')
    with pytest.raises(ValueError, match='rdf_mode'):
        co.curate(str(coord_dir))


def test_a_campaign_without_priors_makes_hop_jobs_wait_for_its_first_basin(acridine, tmp_path, monkeypatch):
    """A curate pass before any shard (no priors) writes an empty parent list from an empty registry: hop jobs wait on
    it rather than stopping as exhausted; once a random shard is in, the next pass lists parents."""
    coord_dir, cfg = _campaign(tmp_path, acridine[0], hops=dict(stream='hops', window_kT=100.0, max_per_basin=8))
    co.curate(str(coord_dir))  # the curator's first pass, before any job has written a shard
    hp = torch.load(coord_dir / 'hop_parents.pt', weights_only=False)
    assert len(hp['basin']) == 0 and hp['n_basins'] == 0
    coord = dict(dir=str(coord_dir), model_id='elj')
    scfg = _search_config(tmp_path, 3, coord_dir, stream='hops', init_sample_method='hops')
    assert rs._hop_batch(scfg, [], 0, 3, coord, 'cpu', 0, after_pass=0) is rs.HOP_WAIT, 'nothing found yet: wait'
    seeds = _seeds(cfg, acridine[1], 3)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake_optimiser(_energy))
    rs.crystal_search(_search_config(tmp_path, 3, coord_dir))
    co.curate(str(coord_dir))
    hp = torch.load(coord_dir / 'hop_parents.pt', weights_only=False)
    assert len(hp['basin']) > 0 and hp['n_basins'] > 0
