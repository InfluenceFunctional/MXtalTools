"""gradient_descent_optimization(convergence='grad'): per-row stopping on the gradient of the row's own loss.

Pins: row_gradient_blocks reads each row's own gradient before the clip, in the documented block units; a row
converges after grad_patience steps at or below grad_tol (not before min_num_steps), leaves the batch and reports
converged_step; convergence_eps < 0 still means never; 'params' keeps check_convergence; an unknown mode raises.
CPU, sg 1, a handful of rows of the test molecule.
"""
import math

import pytest
import torch

from mxtaltools.crystal_search import crystal_opt_utils as cou

from tests.test_fd_refold import _case, _rows, mol  # noqa: F401  (the fixture and row builders of the refold tests)


def _batch(mol):
    _, p, h = _case('normal', n=4, seed=31)
    return _rows(mol, 1, p, h)


def _opt(batch, steps, **kw):
    kw.setdefault('centroid_boundary', 'wrap')
    kw.setdefault('convergence_eps', 1.0)
    kw.setdefault('convergence', 'grad')
    return cou.gradient_descent_optimization(batch.full_cell_parameters(), batch, optimizer_func='rprop',
                                             init_lr=0.01, max_num_steps=steps, optim_target='elj', cutoff=10,
                                             show_tqdm=False, **kw)


def test_params_stays_the_default():
    import inspect
    sig = inspect.signature(cou.gradient_descent_optimization).parameters
    assert sig['convergence'].default == 'params' and cou.CONVERGENCE_MODES == ('grad', 'params')


def test_row_gradient_blocks_are_each_rows_own_gradient(mol):
    b = _batch(mol)
    b.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
    pm = cou.CrystalParams(b.full_cell_parameters().detach())
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = pm.stacked()
    loss = (w[:, None] * x ** 2).sum(1)  # row i's loss depends on row i only
    loss.mean().backward()
    g = cou.row_gradient_blocks(pm, b, None, 4)
    full = (2 * w[:, None] * x.detach()).abs()  # d loss_i / d x_i
    L = b.cell_lengths.detach().float()
    expect = torch.stack([full[:, 0:3].amax(1), full[:, 3:6].amax(1), (full[:, 6:9] / L).amax(1),
                          full[:, 9:12].amax(1)], 1)
    assert torch.allclose(g, expect, rtol=1e-5)


def test_rows_converge_after_patience_and_leave_the_batch(mol):
    out, rec = _opt(_batch(mol), steps=80, grad_tol=1e9, grad_patience=5)  # every step is below the tolerance
    # checked from min_num_steps (50): all four rows converge at step 49 and the stage ends there
    assert rec['loss'].shape[0] == 50
    assert [int(o.converged_step) for o in out] == [49] * 4
    assert rec['grad'].shape == (50, 4, 4) and bool(torch.isfinite(rec['grad']).all())
    assert all('early_stopped' not in o.keys() for o in out)


def test_never_below_runs_to_the_cap(mol):
    out, rec = _opt(_batch(mol), steps=55, grad_tol=-1.0)
    assert rec['loss'].shape[0] == 54
    assert [int(o.converged_step) for o in out] == [-1] * 4


@pytest.mark.parametrize('eps', [0.0, -1.0])
def test_nonpositive_convergence_eps_still_means_never(mol, eps):
    out, rec = _opt(_batch(mol), steps=55, grad_tol=1e9, convergence_eps=eps)
    assert rec['loss'].shape[0] == 54 and all(int(o.converged_step) == -1 for o in out)


def test_one_row_converges_the_rest_continue(mol, monkeypatch):
    real = cou.row_gradient_blocks

    def only_row_0_small(pm, cb, active_idx, n_active):  # global row 0 reads 0, every other row a large gradient
        g = real(pm, cb, active_idx, n_active)
        rows = torch.arange(len(pm.params)) if active_idx is None else active_idx.cpu()
        return torch.where((rows == 0)[:, None], torch.zeros_like(g), torch.full_like(g, 1e6))
    monkeypatch.setattr(cou, 'row_gradient_blocks', only_row_0_small)
    out, rec = _opt(_batch(mol), steps=60, grad_tol=0.1)
    assert [int(o.converged_step) for o in out] == [49, -1, -1, -1]
    assert rec['loss'].shape[0] == 59
    assert bool(torch.isinf(rec['loss'][50:, 0]).all()), 'row 0 left the batch'
    assert bool(torch.isfinite(rec['loss'][50:, 1:]).all())
    # row 0 returns its best state from before it left
    assert int(torch.argmin(rec['loss'][:, 0])) <= 49


def test_params_mode_keeps_check_convergence(mol, monkeypatch):
    called = []
    real = cou.check_convergence
    monkeypatch.setattr(cou, 'check_convergence', lambda *a, **k: called.append(1) or real(*a, **k))
    out, rec = _opt(_batch(mol), steps=55, convergence='params', convergence_eps=-1.0)
    assert called and 'grad' not in rec and all('converged_step' not in o.keys() for o in out)
    called.clear()
    _opt(_batch(mol), steps=55, grad_tol=-1.0)
    assert not called


def test_unknown_mode_raises(mol):
    with pytest.raises(ValueError, match='convergence'):
        _opt(_batch(mol), steps=5, convergence='energy')
