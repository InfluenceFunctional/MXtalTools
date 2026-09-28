"""
crystal_opt_utils.gradient_descent_optimization early_stop cascade (the stage-2 energy triage).

1. Without early_stop_ref (or early_stop) the optimisation is bit-for-bit today's.
2. At a listed step, rows whose loss exceeds ref + margin * kT are retired: they are flagged early_stopped, later steps
   evaluate only the remaining rows (the compute saving), their losses after retirement are +inf in the records, and
   each returns its best state from before retirement.
3. check_convergence's 95% quorum counts active rows only; retired rows count as converged.

CPU only (eLJ).
"""
from pathlib import Path

import pytest
import torch

import mxtaltools.crystal_search.crystal_opt_utils as cou
from mxtaltools.dataset_utils.data_class_methods import crystal_analysis
from mxtaltools.dataset_utils.utils import collate_data_list

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']


def _batch():
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    base = [c for c in torch.load(ACRIDINE, weights_only=False) if int(c.sg_ind) == 14 and int(c.z_prime) == 2][0]
    base = base.clone()
    for k in DROP:
        if k in base.keys():
            delattr(base, k)
    out = []
    for f in (1.0, 1.02, 0.80, 1.35):  # two near the minimum; one crushed, one blown up: both far above it
        c = base.clone()
        c.cell_lengths = c.cell_lengths * f
        out.append(c)
    return collate_data_list(out)


def _run(batch, **kw):
    return cou.gradient_descent_optimization(batch.full_cell_parameters(), batch, optimizer_func='rprop',
                                             init_lr=0.01, max_num_steps=30, optim_target='elj', cutoff=10,
                                             show_tqdm=False, centroid_boundary='wrap', **kw)


def test_default_path_is_unchanged():
    b = _batch()
    _, r0 = _run(b.clone())
    _, r1 = _run(b.clone(), early_stop=[[5, 1.0]])  # no reference: inert
    assert torch.equal(r0['loss'], r1['loss'])


def test_retired_rows_leave_the_batch_and_keep_their_best_state(monkeypatch):
    b = _batch()
    _, ref_rec = _run(b.clone())
    e5 = ref_rec['loss'][5]
    ref = float(e5[:2].max())  # the two good rows sit at or below the reference at step 5; the bad ones far above
    assert (e5[2:] > ref + 50).all(), 'setup: rows 2 and 3 must be far above the good ones'
    sizes = []
    real_analyze = crystal_analysis.MolCrystalAnalysis.analyze

    def counting_analyze(self, *a, **k):
        sizes.append(self.num_graphs)
        return real_analyze(self, *a, **k)
    monkeypatch.setattr(crystal_analysis.MolCrystalAnalysis, 'analyze', counting_analyze)
    samples, rec = _run(b.clone(), early_stop=[[5, 10.0]], early_stop_ref=ref, early_stop_kT=1.0)
    steps = len(rec['loss'])
    assert sizes[:6] == [4] * 6 and all(s == 2 for s in sizes[6:steps]), f'evaluated rows per step: {sizes[:steps]}'
    flags = [bool(s.early_stopped) for s in samples]
    assert flags == [False, False, True, True]
    assert torch.isinf(rec['loss'][6:, 2:]).all() and torch.isfinite(rec['loss'][:6, 2:]).all()
    best = rec['loss'].amin(0)
    assert torch.equal(best[2:], rec['loss'][:6, 2:].amin(0)), 'a retired row returns its best state before retirement'
    returned = torch.tensor([float(s.elj) for s in samples])
    assert torch.allclose(returned[2:], best[2:], rtol=1e-4, atol=1e-3), 'returned energy is that best state'


def test_quorum_counts_active_rows_only():
    T, n = 60, 20
    rec = torch.zeros(T, n, 18)
    rec[:, :10] = torch.cumsum(torch.full((T, 10, 18), 1e-2), 0)  # rows 0-9 still moving, rows 10-19 flat
    active = torch.zeros(n, dtype=torch.bool)
    active[:10] = True  # the flat rows are retired ones
    conv = cou.check_convergence(rec, T, 1e-5, None, None, active=active)
    assert not conv[:10].any() and conv[10:].all(), 'retired rows must not make up a quorum for moving ones'
    rec2 = torch.zeros(T, 21, 18)
    rec2[:, 0] = torch.cumsum(torch.full((T, 18), 1e-2), 0)  # 20 of 21 active rows converged: above 95%
    assert cou.check_convergence(rec2, T, 1e-5, None, None, active=torch.ones(21, dtype=torch.bool)).all()
    assert cou.check_convergence(rec2, T, 1e-5, None, None).all(), 'the default path closes the same batch'


def test_ema_trajectory_is_the_plain_normalised_ema_and_check_convergence_compares_it():
    """A constant trajectory smooths to itself (it used to come out x alpha), and a row converges when the mean absolute
    step of its smoothed trajectory over the last 50 steps is below convergence_eps, in the parameters' own units."""
    torch.manual_seed(0)
    c = torch.full((80, 3, 18), 2.5)
    assert torch.allclose(cou.ema_trajectory(c), c)
    x = torch.randn(80, 3, 18).cumsum(0)
    alpha, want = 0.1, torch.empty_like(x)
    for t in range(80):  # the definition, term by term
        w = (1 - alpha) ** torch.arange(t, -1, -1, dtype=x.dtype)
        want[t] = (w.view(-1, 1, 1) * x[:t + 1]).sum(0) / w.sum()
    assert torch.allclose(cou.ema_trajectory(x, alpha), want, atol=1e-5)
    rec = torch.zeros(80, 4, 18)
    rec[:, 0] = torch.linspace(0, 0.5, 80)[:, None]  # moves 0.5 / 79 per step: smoothed step ~6.3e-3
    rec[:, 1] = torch.linspace(0, 0.005, 80)[:, None]  # ~6.3e-5 per step
    step = (cou.ema_trajectory(rec)[30:80].diff(dim=0).abs().mean((0, 2)))
    conv = cou.check_convergence(rec, 80, 1e-3, None, None)
    assert conv.tolist() == (step < 1e-3).tolist() == [False, True, True, True]
