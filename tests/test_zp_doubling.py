"""
crystal_building/zp_doubling.py: a Z'=1 sg14 crystal re-described as Z'=2 on its index-2 supercell.

Contracts.
1. Every sg14 Z'=1 crystal of the mini acridine set doubles to the same crystal: the built-in atom check passes
   (every doubled unit-cell atom on a same-element parent atom, every parent atom hit exactly twice), eLJ per
   molecule equals the parent's, the doubled cell has zero monoclinic reduction penalty, both centres sit in the
   asymmetric-unit box. The set must exercise the basis-change and inversion (b' = -b, handedness flip) paths.
2. reduce=False keeps the plain (2a, b, c) basis and is equally exact.
3. The atom check can fail: a doubled crystal with one molecule rotated by 2 degrees, or with the handedness flip
   withheld, is caught.
4. Wrong inputs (Z'=2, another space group) are refused.

CPU only.
"""
from pathlib import Path

import pytest
import torch

from mxtaltools.common.sym_utils import cell_reduction_penalty
from mxtaltools.crystal_building.zp_doubling import doubling_deviation, double_zp1_crystal
from mxtaltools.common.geometry_utils import rotvec2rotmat, rotmat2rotvec
from mxtaltools.dataset_utils.utils import collate_data_list

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
M0 = torch.diag(torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64))


def _strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


@pytest.fixture(scope='module')
def parents():
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    out = [_strip(c) for c in torch.load(ACRIDINE, weights_only=False) if int(c.sg_ind) == 14 and int(c.z_prime) == 1]
    assert len(out) >= 20
    return out


def _elj(crystals):
    b = collate_data_list([c.clone() for c in crystals])
    with torch.no_grad():
        return b.analyze(['elj'], cutoff=10, supercell_size=10, std_orientation=True)['elj'].double()


def test_doubles_are_exact(parents):
    doubled, infos = zip(*[double_zp1_crystal(c) for c in parents])  # raises unless the atom check passes
    n_basis = sum(not torch.allclose(i['M'], M0) for i in infos)
    n_flip = sum(float(i['handedness'][0]) != float(c.aunit_handedness.reshape(-1)[0]) for i, c in zip(infos, parents))
    assert n_basis >= 5 and n_flip >= 5, (n_basis, n_flip)  # the basis-change and inversion paths both ran
    e1, e2 = _elj(parents), _elj(doubled)
    assert torch.allclose(e2 / 2, e1, rtol=1e-3), float(((e2 / 2 - e1) / e1).abs().max())
    b = collate_data_list([c.clone() for c in doubled])
    assert (cell_reduction_penalty(b.cell_angles, b.cell_lengths, b.sg_ind, 0.0) <= 1e-9).all()
    u = b.aunit_centroid / torch.tensor([1.0, 0.25, 1.0]).repeat(2)
    assert ((u >= -1e-6) & (u <= 1 + 1e-6)).all()
    assert (b.z_prime == 2).all() and (b.sg_ind == 14).all()


def test_plain_basis_is_exact(parents):
    for c in parents[:6]:
        _, info = double_zp1_crystal(c, reduce=False)
        assert torch.equal(info['M'], M0)


def _perturbed(doubled, info, rotate_deg=0.0, withhold_flip=False):
    d = doubled.clone()
    ori = d.aunit_orientation.clone().double().reshape(1, -1)
    if rotate_deg:
        axis = torch.tensor([[0.3, 0.8, 0.52]], dtype=torch.float64)
        kick = rotvec2rotmat(axis / axis.norm() * rotate_deg * torch.pi / 180)[0]
        ori[0, 3:6] = rotmat2rotvec((kick @ rotvec2rotmat(ori[:, 3:6])[0])[None])[0]
    d.aunit_orientation = ori.reshape(d.aunit_orientation.shape).to(d.aunit_orientation.dtype)
    if withhold_flip:
        d.aunit_handedness = -d.aunit_handedness
    return d


def test_the_atom_check_can_fail(parents):
    c = parents[0]
    doubled, info = double_zp1_crystal(c)
    worst, twice = doubling_deviation(c, doubled, info['M'], info['origin'])
    assert worst < 1e-3 and twice
    worst, _ = doubling_deviation(c, _perturbed(doubled, info, rotate_deg=2.0), info['M'], info['origin'])
    assert worst > 1e-2, 'a 2 degree rotation of one molecule must be caught'
    worst, _ = doubling_deviation(c, _perturbed(doubled, info, withhold_flip=True), info['M'], info['origin'])
    # acridine is nearly flat: a wrong handedness moves atoms only ~0.004-0.014 A, still well above the default tol 1e-3
    assert worst > 2e-3, 'a wrong handedness must be caught at the default tolerance'


def test_output_schema_matches_search_outputs(parents):
    """Summed per-crystal molecule attributes (packing_coeff and density equal the parent's, not half), the shapes
    of saved search outputs, collation with native Z'=2 crystals, latent_params() on the mixed batch, and rotvecs on
    canonicalize_rotvec's +z hemisphere (off it, the latent transform clips theta and decodes another crystal)."""
    doubled = [double_zp1_crystal(c)[0] for c in parents[:8]]
    for p, d in zip(parents[:8], doubled):
        assert torch.allclose(d.packing_coeff.reshape(-1), p.packing_coeff.reshape(-1).to(d.packing_coeff.dtype), rtol=1e-4)
        assert d.packing_coeff.numel() == 1 and d.mass.numel() == 1
        assert torch.allclose(d.mass.reshape(-1), 2 * p.mass.reshape(-1).to(d.mass.dtype))
    native = [_strip(c) for c in torch.load(ACRIDINE, weights_only=False) if int(c.z_prime) == 2 and int(c.sg_ind) == 14]
    b = collate_data_list([c.clone() for c in doubled] + [c.clone() for c in native])
    lat = b.latent_params()
    assert lat.shape[0] == len(doubled) + len(native) and torch.isfinite(lat).all()
    rv = collate_data_list([c.clone() for c in doubled]).aunit_orientation.reshape(-1, 3)
    assert (rv[:, 2] >= -1e-6).all(), 'doubled rotvecs must be canonical (z >= 0)'


def test_duplicated_molecule_is_caught(parents):
    """molecule 1 stacked onto molecule 0 duplicates atoms modulo the PARENT lattice, so a check modulo the parent
    lattice passes it; the one-to-one check in the doubled basis must not."""
    c = parents[0]
    doubled, info = double_zp1_crystal(c)
    wrong = doubled.clone()
    cen = wrong.aunit_centroid.clone().reshape(1, -1)
    ori = wrong.aunit_orientation.clone().reshape(1, -1)
    cen[0, 3:6] = cen[0, 0:3]
    ori[0, 3:6] = ori[0, 0:3]
    wrong.aunit_centroid = cen.reshape(wrong.aunit_centroid.shape)
    wrong.aunit_orientation = ori.reshape(wrong.aunit_orientation.shape)
    _, one_to_one = doubling_deviation(c, wrong, info['M'], info['origin'])
    assert not one_to_one


def test_near_pi_rotation_parent_is_exact(parents):
    """A parent rotated by almost pi: rotmat2rotvec amplifies a ~1e-7 non-orthogonality there by 1/sin(angle)."""
    c = parents[1].clone()
    axis = torch.tensor([0.36, -0.48, 0.8], dtype=torch.float64)
    ori = c.aunit_orientation.clone().double().reshape(1, -1)
    ori[0, :3] = axis / axis.norm() * (torch.pi - 1e-4)
    c.aunit_orientation = ori.reshape(c.aunit_orientation.shape).to(c.aunit_orientation.dtype)
    doubled, info = double_zp1_crystal(c)  # raises unless atom-for-atom exact
    worst, one_to_one = doubling_deviation(c, doubled, info['M'], info['origin'])
    assert one_to_one and worst < 1e-4, worst


def test_refuses_wrong_input(parents):
    zp2 = [_strip(c) for c in torch.load(ACRIDINE, weights_only=False) if int(c.z_prime) == 2]
    with pytest.raises(ValueError):
        double_zp1_crystal(zp2[0])
    other = parents[0].clone()
    other.sg_ind = torch.tensor([19])
    with pytest.raises(ValueError):
        double_zp1_crystal(other)
