"""
Rotation-vector fold: clean_cell_parameters(canonicalize_orientations=True) and the crystal-search optimiser's
returned rows.

Contracts.
1. clean_cell_parameters with canonicalize_orientations=True leaves every unit's rotation vector on the +z
   half-space and its rotation unchanged, at Z'=1 and at Z'=2; with the flag off a z < 0 vector stays as given.
2. gradient_descent_optimization returns rows whose rotation vectors are on the +z half-space in both centroid
   modes, each the rotation recorded at that row's lowest-loss step; the per-step record itself is not folded, so
   a vector that crosses z = 0 does not jump in it.

CPU only.
"""
import math
from pathlib import Path

import pytest
import torch

from mxtaltools.common.geometry_utils import rotvec2rotmat
from mxtaltools.crystal_search.crystal_opt_utils import gradient_descent_optimization
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parent
ACRIDINE = HERE / 'datasets' / 'mini_acridine.pt'
CSD = HERE.parent / 'mini_datasets' / 'mini_new_csd.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
ROT_TOL = 1e-5  # float32 rotation matrices of the same rotation written two ways


def _load(path, n, zp):
    if not path.exists():
        pytest.skip(f'{path} not present')
    rows = []
    for c in torch.load(path, weights_only=False):
        if int(c.z_prime) != zp:
            continue
        c = c.clone()
        for k in DROP:
            if k in c.keys():
                delattr(c, k)
        rows.append(c)
        if len(rows) == n:
            break
    if not rows:
        pytest.skip(f"no Z'={zp} crystal in {path.name}")
    return collate_data_list(rows)


def _flip(v):
    """[n, 3] -> the same rotations written with the axis reversed: -v/|v| (2 pi - |v|)."""
    n = v.norm(dim=1, keepdim=True)
    return -v / n * (2 * math.pi - n)


def _units(batch):
    """[(rows, column slice)] of the present units."""
    zp = batch.z_prime.reshape(-1)
    return [(torch.nonzero(zp > k).flatten(), slice(3 * k, 3 * k + 3)) for k in range(batch.max_z_prime)]


def _negative_z_batch(path, zp):
    b = _load(path, 8, zp)
    o = b.aunit_orientation.clone()
    for rows, sl in _units(b):
        v = o[rows, sl]
        o[rows, sl] = torch.where((v[:, 2:] > 0), _flip(v), v)  # every present unit written with z < 0
    b.aunit_orientation = o
    return b


@pytest.mark.parametrize('path,zp', [(CSD, 1), (ACRIDINE, 2)], ids=['zp1_csd', 'zp2_acridine'])
def test_clean_folds_negative_z_without_changing_the_rotation(path, zp):
    b = _negative_z_batch(path, zp)
    before = b.aunit_orientation.clone()
    assert all((before[rows, sl][:, 2] < 0).all() for rows, sl in _units(b)), 'setup: every present unit z < 0'

    off = b.clone()
    off.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
    on = b.clone()
    on.clean_cell_parameters(mode='hard', canonicalize_orientations=True)

    for rows, sl in _units(b):
        assert (off.aunit_orientation[rows, sl][:, 2] < 0).all(), 'control: with the flag off, z < 0 stays'
        assert (on.aunit_orientation[rows, sl][:, 2] >= 0).all(), 'the fold must put every unit on +z'
        r_on = rotvec2rotmat(on.aunit_orientation[rows, sl].double())
        r_off = rotvec2rotmat(off.aunit_orientation[rows, sl].double())
        assert (r_on - r_off).abs().max() < ROT_TOL, 'the fold must not change the rotation'


@pytest.mark.parametrize('boundary', ['clamp', 'wrap'])
def test_optimiser_returns_folded_rotations_of_the_best_step(boundary):
    b = _negative_z_batch(CSD, 1)
    out, rec = gradient_descent_optimization(b.full_cell_parameters(), b, optimizer_func='rprop', init_lr=0.01,
                                             max_num_steps=6, optim_target='elj', cutoff=10, show_tqdm=False,
                                             convergence_eps=0.0, centroid_boundary=boundary)
    got = collate_data_list(out).aunit_orientation[:, :3]
    assert (got[:, 2] >= 0).all(), 'returned rows must be on the +z half-space'

    params = rec['params']  # [steps, rows, 6 + 6 * max_z_prime], the per-step record
    o = 6 + 3 * b.max_z_prime  # first orientation column
    assert (params[0, :, o + 2] < 0).all(), 'the record is not folded: the z < 0 start stays as given'
    best = torch.argmin(rec['loss'], dim=0).flatten()
    rec_rot = rotvec2rotmat(torch.stack([params[best[i], i, o:o + 3] for i in range(len(best))]).double())
    if boundary == 'clamp':  # wrap mode may re-express a row through a space-group operation, rotating it
        assert (rotvec2rotmat(got.double()) - rec_rot).abs().max() < ROT_TOL
