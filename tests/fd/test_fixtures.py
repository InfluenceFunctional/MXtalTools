"""Self-checks of the tests/fd fixtures: cells per family and the float64 literal-atom oracle."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
from mxtaltools.constants.space_group_info import LATTICE_TYPE, SYM_OPS
from mxtaltools.crystal_fd.tables import group_table

FAMILIES = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'hexagonal', 'cubic']


@pytest.mark.parametrize('family', FAMILIES)
def test_cells_per_family(family, cell_sampler, literal_oracle):
    """Each family's constraints hold, and the oracle's cell matrix is MXtalTools' T_fc."""
    cells = cell_sampler(family, 50, seed=3)
    a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
    deg = np.degrees(cells[:, 3:])
    if family in ('tetragonal', 'hexagonal', 'cubic'):
        assert np.all(a == b)
    if family == 'cubic':
        assert np.all(a == c)
    if family != 'triclinic':
        assert np.allclose(deg[:, 0], 90) and np.allclose(deg[:, 2], 120 if family == 'hexagonal' else 90)
    T_fc, _, vol = batch_compute_fractional_transform(torch.tensor(cells[:, :3]), torch.tensor(cells[:, 3:]))
    for k, cell in enumerate(cells):
        T = literal_oracle.cell_matrix(cell)
        assert np.abs(T - T_fc[k].numpy()).max() < 1e-12
        assert abs(np.linalg.det(T) - vol[k].item()) < 1e-9 * vol[k].item()


@pytest.mark.parametrize('sg', [2, 14, 19, 92, 148, 198])
def test_oracle_equivalences(sg, chiral_molecule, cell_sampler, literal_oracle):
    """G+ elements give the same literal crystal; Gamma+ elements give its image under the element.

    The crystal is built from the table's exact G (SYM_OPS stores thirds to 8
    decimals, a floor near 1e-7 Angstrom in trigonal groups). Reference: float64
    noise (about 1e-12 Angstrom). The control (a Gamma+ element outside G, read
    without applying it) must differ by more than 0.1 Angstrom.
    """
    X = chiral_molecule['pos']
    cell = cell_sampler(LATTICE_TYPE[sg], 1, seed=sg)[0]
    T = literal_oracle.cell_matrix(cell)
    tab = group_table(sg)
    W, w = tab.g_W.numpy(), tab.g_w.numpy()
    assert np.abs(w - np.stack([np.asarray(o, float)[:3, 3] for o in SYM_OPS[sg]])).max() <= 1e-8
    rng = np.random.default_rng(sg)
    gW, gw = tab.gamma_W.numpy(), tab.gamma_w.numpy()
    n_g = len(SYM_OPS[sg])
    worst, control = 0.0, np.inf
    for _ in range(4):
        t = rng.random(3)
        R = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        ref = literal_oracle.unit_cell(cell, t, R, X, W, w)
        for k in tab.gamma_plus.tolist():
            Q = T @ gW[k] @ np.linalg.inv(T)
            assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-12) and np.linalg.det(Q) > 0
            img = literal_oracle.unit_cell(cell, gW[k] @ t + gw[k], Q @ R, X, W, w)
            moved = ref @ Q.T + T @ gw[k]
            worst = max(worst, literal_oracle.max_atom_deviation(img, moved, cell))
            if k < n_g:
                worst = max(worst, literal_oracle.max_atom_deviation(img, ref, cell))
            else:
                control = min(control, literal_oracle.max_atom_deviation(img, ref, cell))
    assert worst < 1e-9
    if len(tab.gamma_plus) > len([d for d in tab.gamma_det[:n_g].tolist() if d == 1]):
        assert control > 0.1
