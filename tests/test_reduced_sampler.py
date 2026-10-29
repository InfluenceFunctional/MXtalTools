"""MolCrystalOps.sample_random_reduced_crystal_parameters: per-row rejection sampling of reduced cells.

Pins: every row's cell has zero reduction penalty; a seed fixes the draw; each row's packing coefficient is the one
asked for that row (the candidates of one row never take another row's target); a row with no zero-penalty
candidate within max_rounds raises. CPU, sg 1 and 2 rows of the test molecule (as the refold tests).
"""
import pytest
import torch

from tests.test_fd_refold import _case, _rows, mol  # noqa: F401  (the fixture and row builders of the refold tests)


def _batch(mol, n=24):
    sg, p, h = _case('normal', n=n, seed=3)  # sg 1 and 2 alternating
    return _rows(mol, sg, p, h)


def test_every_row_reduced_and_seeded(mol):
    a, b = _batch(mol), _batch(mol)
    tpc = torch.linspace(0.5, 0.8, a.num_graphs)
    a.sample_random_reduced_crystal_parameters(target_packing_coeff=tpc, seed=11)
    b.sample_random_reduced_crystal_parameters(target_packing_coeff=tpc, seed=11)
    assert bool((a.compute_cell_reduction_penalty() == 0).all())
    assert torch.equal(a.cell_lengths, b.cell_lengths) and torch.equal(a.cell_angles, b.cell_angles)
    assert torch.equal(a.aunit_centroid, b.aunit_centroid)


def test_each_row_gets_its_own_packing_coefficient(mol):
    b = _batch(mol)
    tpc = torch.linspace(0.5, 0.8, b.num_graphs)
    b.sample_random_reduced_crystal_parameters(target_packing_coeff=tpc, seed=5)
    cp = (b.mol_volume * b.sym_mult / b.cell_volume).reshape(-1)  # the packing coefficient of the built cell
    assert torch.allclose(cp.float(), tpc, rtol=1e-3)


def test_no_round_left_raises(mol):
    b = _batch(mol, n=4)
    with pytest.raises(RuntimeError):
        b.sample_random_reduced_crystal_parameters(target_packing_coeff=0.6, seed=1, max_rounds=0)
