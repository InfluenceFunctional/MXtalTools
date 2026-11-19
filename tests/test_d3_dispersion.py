"""D3Dispersion (mlip_interfaces/dispersion.py) against torch-dftd on ASE's neighbour list, and its gradient."""
import numpy as np
import pytest
import torch

pytest.importorskip('torch_dftd')
from ase.neighborlist import primitive_neighbor_list  # noqa: E402
from torch_dftd.dftd3_xc_params import get_dftd3_default_params  # noqa: E402
from torch_dftd.nn.dftd3_module import DFTD3Module  # noqa: E402

from mxtaltools.mlip_interfaces.dispersion import D3Dispersion  # noqa: E402

CUT = 8.0


def _system(seed=0):
    """Two copies of a 6-atom C/H/N/O cluster in a triclinic cell (row-vector lattice), Angstrom."""
    g = np.random.default_rng(seed)
    cell = np.array([[7.1, 0.0, 0.0], [1.2, 6.4, 0.0], [-0.8, 0.9, 6.9]])
    mol = np.array([[0, 0, 0], [1.4, 0, 0], [2.1, 1.2, 0], [1.4, 2.4, 0], [0, 2.4, 0.1], [-0.7, 1.2, 0.1]], float)
    pos = np.concatenate([mol + 0.1 * g.standard_normal(mol.shape), mol[::-1] + [3.0, 3.1, 3.3]])
    z = np.array([6, 6, 7, 8, 1, 6] * 2)
    return z, pos, cell


def _reference(z, pos, cell, pbc):
    mod = DFTD3Module(get_dftd3_default_params('bj', 'pbe'), cutoff=CUT, cnthr=CUT, dtype=torch.float64,
                      bidirectional=True)
    i, j, S = primitive_neighbor_list('ijS', [pbc] * 3, cell, pos, CUT)
    shift = torch.tensor(S, dtype=torch.float64) @ torch.tensor(cell) if pbc else torch.zeros(len(i), 3, dtype=torch.float64)
    return float(mod.calc_energy_batch(torch.tensor(z), torch.tensor(pos), torch.tensor(np.stack([i, j])),
                                       cell=torch.tensor(cell)[None], shift_pos=shift,
                                       batch=torch.zeros(len(z), dtype=torch.long),
                                       batch_edge=torch.zeros(len(i), dtype=torch.long), damping='bj').sum())


@pytest.mark.parametrize('pbc', [True, False])
def test_matches_torch_dftd_on_ase_neighbours(pbc):
    z, pos, cell = _system()
    d3 = D3Dispersion(cutoff=CUT, cnthr=CUT)
    e = d3(torch.tensor(z), torch.tensor(pos), torch.zeros(len(z), dtype=torch.long),
           torch.tensor(cell.T)[None], pbc=pbc)
    assert e.shape == (1,)
    assert float(e) == pytest.approx(_reference(z, pos, cell, pbc), rel=1e-9, abs=1e-12)


def test_two_graphs_sum_independently():
    z, pos, cell = _system()
    z2, pos2, cell2 = _system(seed=1)
    d3 = D3Dispersion(cutoff=CUT, cnthr=CUT)
    e = d3(torch.tensor(np.concatenate([z, z2])), torch.tensor(np.concatenate([pos, pos2])),
           torch.tensor([0] * len(z) + [1] * len(z2)), torch.tensor(np.stack([cell.T, cell2.T])))
    assert e.tolist() == pytest.approx([_reference(z, pos, cell, True), _reference(z2, pos2, cell2, True)], rel=1e-9)


def test_gradient_in_positions_and_cell():
    z, pos, cell = _system()
    d3 = D3Dispersion(cutoff=CUT, cnthr=CUT)
    p = torch.tensor(pos, requires_grad=True)
    T = torch.tensor(cell.T)[None].requires_grad_(True)
    zt, bi = torch.tensor(z), torch.zeros(len(z), dtype=torch.long)
    gp, gT = torch.autograd.grad(d3(zt, p, bi, T).sum(), [p, T])
    torch.manual_seed(0)
    vp, vT, h = torch.randn_like(p), 0.01 * torch.randn_like(T), 1e-5
    with torch.no_grad():
        fd = (d3(zt, p + h * vp, bi, T + h * vT).sum() - d3(zt, p - h * vp, bi, T - h * vT).sum()) / (2 * h)
    assert float((gp * vp).sum() + (gT * vT).sum()) == pytest.approx(float(fd), rel=1e-5)
