"""
crystal_search (run_search.py): out-of-memory recovery, resume, and seed attribution.

1. A retry may resume from the saved intermediates only when they were written by an attempt at the SAME batch
   position; a retry after a step-0 OOM at a later position starts from its own fresh draws (the acr_wrap_sep26
   defect: 46-55% of relaxations repeated earlier starts). A file left before the run is never used.
2. Seeded (init_sample_method 'data') runs: every output carries dataset_index, the index of the seed it came from,
   even when a stage drops rows (the enforce_reduced filter); they never resume from saved states.
3. A resumed run continues from the recorded cursor and batch index: no sample is relaxed twice and no random start
   is redrawn, although dropped rows make len(outputs) smaller than the cursor.
4. dataset_index survives the real optimiser (eLJ, CPU).

CPU only.
"""
from pathlib import Path

import pytest
import torch

import mxtaltools.crystal_search.run_search as rs
from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.dataset_utils.data_classes import MolCrystalData

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
MARK = 111.0
OOM = 'CUDA out of memory. Tried to allocate 2.00 GiB'


def _templates(n):
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    base = [c for c in torch.load(ACRIDINE, weights_only=False) if int(c.sg_ind) == 14 and int(c.z_prime) == 2][0]
    base = base.clone()
    for k in DROP:
        if k in base.keys():
            delattr(base, k)
    return [base.clone() for _ in range(n)]


def _seeds(n):
    """distinct seeds: seed i has cell length a = 10 + i (traceable through the fake optimiser)."""
    out = _templates(n)
    for i, c in enumerate(out):
        L = c.cell_lengths.clone()
        L.reshape(-1)[0] = 10.0 + i
        c.cell_lengths = L
    return out


def _config(tmp_path, n, data=False, max_steps=5):
    cfg = {'device': 'cpu', 'target_path': None, 'out_dir': str(tmp_path), 'run_name': 'oomtest',
           'force_restart_run': True, 'batch_size': 4, 'grow_batch_size': False, 'save_trajs': False,
           'num_samples': n, 'init_sample_method': 'data' if data else 'random', 'init_reduced': True,
           'init_target_cp': 'std', 'opt_seed': 0, 'mol_seed': 0, 'dataset_path': None,
           'opt': [{'optim_target': 'elj', 'init_lr': 0.01, 'max_num_steps': max_steps, 'optimizer_func': 'rprop',
                    'show_tqdm': False}]}
    return dict2namespace(cfg)


def _fake(script, seen, drop_first_row=False):
    """stand-in for MolCrystalData.optimize_crystal_parameters, driven by a list of actions."""
    def fake_optimize(self, return_record=False, **kwargs):
        params = self.full_cell_parameters().detach().clone()
        seen.append(dict(params=params, index=getattr(self, 'dataset_index', None)))
        action = next(script)
        if action == 'oom_after_step':
            marked = params.clone()
            marked[:, 0] = MARK
            torch.save(marked, kwargs['intermediates_path'])  # what gradient_descent_optimization writes
            raise RuntimeError(OOM)
        if action == 'oom_at_step0':
            raise RuntimeError(OOM)  # no file written
        if action == 'crash':
            raise RuntimeError('node lost')  # not an OOM: the run stops here
        out = self.batch_to_list()
        if drop_first_row:
            out = out[1:]  # as the enforce_reduced filter can
        return (out, {}) if return_record else out
    return fake_optimize


@pytest.mark.parametrize('stale_file_before_run', [False, True])
def test_retry_never_resumes_another_batchs_states(tmp_path, monkeypatch, stale_file_before_run):
    n = 7
    cfg = _config(tmp_path, n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: _templates(n))
    inter, _ = rs.run_files(Path(cfg.out_dir) / f'{cfg.run_name}.pt')
    if stale_file_before_run:
        torch.save(torch.full((8, 18), MARK), inter)
    seen = []
    script = iter(['oom_after_step', 'ok', 'oom_at_step0', 'ok', 'ok', 'ok', 'ok'])
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake(script, seen))
    outs = rs.crystal_search(cfg)
    marked = [bool((s['params'][:, 0] == MARK).any()) for s in seen]
    if stale_file_before_run:
        assert not marked[0], 'a file left before the run must not seed its first batch'
    assert marked[1], 'a retry at the same cursor should resume from the states its own OOM saved'
    assert not any(marked[3:]), 'a retry at a later cursor must not resume from an earlier batch (stale file)'
    assert len(outs) == n and not Path(inter).exists()


def test_seeded_outputs_carry_their_seed_index_and_never_resume(tmp_path, monkeypatch):
    n = 10
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    seen = []
    script = iter(['oom_after_step'] + ['ok'] * 20)
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters', _fake(script, seen, drop_first_row=True))
    outs = rs.crystal_search(_config(tmp_path, n, data=True))
    assert not any(bool((s['params'][:, 0] == MARK).any()) for s in seen), 'data mode must redo seeds, not resume'
    idx = [int(o.dataset_index) for o in outs]
    a = [float(o.cell_lengths.reshape(-1)[0]) for o in outs]
    assert len(outs) < n, 'the fake drops a row per batch'
    assert all(abs(ai - (10.0 + i)) < 1e-4 for ai, i in zip(a, idx)), 'dataset_index must name the seed each output came from'
    assert len(set(idx)) == len(idx)


@pytest.mark.parametrize('data', [False, True])
def test_resume_continues_without_rerelaxing(tmp_path, monkeypatch, data):
    n = 12
    seeds = _seeds(n)
    monkeypatch.setattr(rs, 'init_samples_to_optim',
                        lambda config, target=None: [s.clone() for s in (seeds if data else _templates(n))])
    drawn = []
    real_init = rs.get_initial_state

    def recording_init(config, crystal_batch, device, batch_idx):
        drawn.append(batch_idx)
        return real_init(config, crystal_batch, device, batch_idx)
    monkeypatch.setattr(rs, 'get_initial_state', recording_init)

    seen1 = []
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters',
                        _fake(iter(['ok', 'oom_at_step0', 'ok', 'crash']), seen1, drop_first_row=True))
    cfg = _config(tmp_path, n, data=data)
    with pytest.raises(RuntimeError, match='node lost'):
        rs.crystal_search(cfg)
    first_run_draws = list(drawn)
    _, progress = rs.run_files(Path(cfg.out_dir) / f'{cfg.run_name}.pt')
    import json
    last_completed = json.loads(Path(progress).read_text())['batch_idx']  # the crashed attempt saved nothing

    seen2 = []
    monkeypatch.setattr(MolCrystalData, 'optimize_crystal_parameters',
                        _fake(iter(['ok'] * 20), seen2, drop_first_row=True))
    cfg2 = _config(tmp_path, n, data=data)
    cfg2.force_restart_run = False
    outs = rs.crystal_search(cfg2)
    second_run_draws = drawn[len(first_run_draws):]
    assert last_completed < max(first_run_draws), 'the first run must have crashed mid-batch for this test to mean anything'
    assert min(second_run_draws) > last_completed, 'resumed run redrew the starts of a batch it had already completed'
    if data:
        idx = [int(o.dataset_index) for o in outs]
        assert len(set(idx)) == len(idx), 'a seed was relaxed twice across the resume'
    _, progress = rs.run_files(Path(cfg.out_dir) / f'{cfg.run_name}.pt')
    assert Path(progress).exists()
    cfg3 = _config(tmp_path, n, data=data)
    cfg3.force_restart_run = False
    assert len(rs.crystal_search(cfg3)) == len(outs), 'a finished run must return without relaxing more'


def test_dataset_index_survives_the_real_optimiser(tmp_path, monkeypatch):
    seeds = _templates(3)
    monkeypatch.setattr(rs, 'init_samples_to_optim', lambda config, target=None: [s.clone() for s in seeds])
    outs = rs.crystal_search(_config(tmp_path, 3, data=True, max_steps=3))
    assert sorted(int(o.dataset_index) for o in outs) == list(range(len(outs)))
