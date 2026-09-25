"""
Crystal-search centroid boundary: crystal_opt_utils.wrap_centroid, canonicalize_aunit and
gradient_descent_optimization(centroid_boundary=...).

Contracts.
1. wrap_centroid lattice-translates centres into [0, CELL_EDGE] (the builders' clip) and passes the
   gradient through unchanged everywhere, including where the value is clamped.
2. The builders accept a centre anywhere in the unit cell: a molecule re-expressed through any
   space-group operation (centre wrapped into the cell, often outside the asymmetric-unit box) gives the
   same unit cell atom for atom. canonicalize_aunit brings every such description back inside the box,
   again atom for atom. Checked on a deliberately puckered (chiral) conformer, where a handedness
   error cannot hide: the same comparison with the handedness flip withheld must fail on improper ops.
3. The default boundary is the asymmetric-unit clamp: a centre placed exactly on a face gets zero
   gradient and never moves. With centroid_boundary='wrap' the same centre moves off the face.

CPU only.
"""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_search.crystal_opt_utils import (CELL_EDGE, canonicalize_aunit,
                                                        gradient_descent_optimization, wrap_centroid)
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parent
ACRIDINE = HERE / 'datasets' / 'mini_acridine.pt'
CSD = HERE.parent / 'mini_datasets' / 'mini_new_csd.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
ATOM_TOL = 1e-3  # Angstrom; float32 builds agree to ~1e-5


def _strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


def _load(path):
    if not path.exists():
        pytest.skip(f'{path} not present')
    return [_strip(c) for c in torch.load(path, weights_only=False)]


def _pucker(c, dz=0.4):
    """Push one off-axis atom of every molecule dz Angstrom out of that molecule's mean plane: the
    conformer loses every mirror, so its mirror image is a different object."""
    c = c.clone()
    pos = c.pos.clone()
    zp = int(c.z_prime)
    na = pos.shape[0] // zp
    for m in range(zp):
        idx = torch.arange(m * na, (m + 1) * na)
        p = pos[idx].double()
        centred = p - p.mean(0)
        _, _, vt = torch.linalg.svd(centred)
        u, v = centred @ vt[0], centred @ vt[1]
        j = int(torch.nonzero((u.abs() > 0.6) & (v.abs() > 0.3))[0])
        pos[idx[j]] = (p[j] + dz * vt[2]).to(pos.dtype)
    c.pos = pos
    return c


def _cases():
    acr = _load(ACRIDINE)
    csd = _load(CSD)
    by_sg = {}
    for c in csd:
        if str(int(c.sg_ind)) in ASYM_UNITS and not bool(c.nonstandard_symmetry):
            by_sg.setdefault(int(c.sg_ind), []).append(c)
    picked = [c for c in acr if int(c.z_prime) > 1 and str(int(c.sg_ind)) in ASYM_UNITS]
    picked += [c for cs in by_sg.values() for c in cs[:2]]
    return picked


def _unit_cells(crystals):
    """Per crystal: fractional unit-cell atoms wrapped into [0,1) and T_fc, via mol2ucell (the geometry every
    energy route scores). Atoms are sliced by count: num_atoms (all aunit molecules) x sym_mult per crystal."""
    b = collate_data_list([c.clone() for c in crystals])
    with torch.no_grad():
        b.mol2ucell(std_orientation=True)
    counts = (b.num_atoms * b.sym_mult).tolist()
    out, start = [], 0
    for i, n in enumerate(counts):
        f = b.unit_cell_pos[start:start + n].double() @ b.T_cf[i].double().T
        out.append((f - torch.floor(f), b.T_fc[i].double()))
        start += n
    assert start == b.unit_cell_pos.shape[0]
    return out


def _elements(c):
    """Elements of the unit-cell atoms in mol2ucell's order (each aunit molecule repeated sym_mult times)."""
    zp, sm = int(c.z_prime), int(c.sym_mult)
    z = c.z.reshape(zp, -1)
    return z[:, None, :].expand(zp, sm, z.shape[1]).reshape(-1)


def _max_atom_deviation(c1, c2):
    """Largest distance (Angstrom) between matched atoms of two unit cells: same element, periodic minimum image."""
    # collated one at a time: a dataset item and a batch_to_list item need not share attribute shapes
    (a, t_fc), = _unit_cells([c1])
    (b, _), = _unit_cells([c2])
    z = _elements(c1).numpy()
    worst = 0.0
    for e in np.unique(z):
        d = a[z == e][:, None] - b[z == e][None]
        d = d - torch.round(d)
        dist = torch.linalg.norm(d @ t_fc.T, dim=-1).numpy()
        r, cidx = linear_sum_assignment(dist)
        worst = max(worst, float(dist[r, cidx].max()))
    return worst


def _redescribe(c, k, op, flip=True):
    """Molecule k of crystal c re-expressed through one 4x4 space-group operation (centre wrapped into the cell)."""
    b = collate_data_list([c.clone()])
    sl = slice(3 * k, 3 * k + 3)
    op = torch.as_tensor(op, dtype=b.aunit_centroid.dtype)
    cen, ori, h = b._transform_aunit_params(b.aunit_centroid[:, sl], b.aunit_orientation[:, sl],
                                            b.aunit_handedness[:, k], op[None, :3, :3], op[None, :3, 3], wrap=True)
    b.aunit_centroid[:, sl] = cen
    b.aunit_orientation[:, sl] = ori
    b.aunit_handedness[:, k] = torch.sign(h) if flip else b.aunit_handedness[:, k]
    return b.batch_to_list()[0]


# ---------------------------------------------------------------------------
# 1. wrap_centroid
# ---------------------------------------------------------------------------

def test_wrap_centroid_values_and_gradient():
    raw = torch.tensor([-1.3, -1e-9, 0.0, 0.2, 0.99995, 1.0, 2.7, -0.75], requires_grad=True)
    out = wrap_centroid(raw)
    assert (out >= 0).all() and (out <= CELL_EDGE).all()
    for n in (-2.0, 1.0, 3.0):
        assert torch.allclose(wrap_centroid(raw.detach() + n), out.detach(), atol=1e-6)
    out.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw)), 'gradient must pass unchanged, clamped entries included'


# ---------------------------------------------------------------------------
# 2. out-of-box descriptions build the same crystal; canonicalize_aunit restores the box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pucker', [False, True], ids=['stored_conformer', 'puckered_chiral_conformer'])
def test_redescription_and_canonicalisation_are_exact(pucker):
    cases = [_pucker(c) if pucker else c for c in _cases()]
    assert len({int(c.sg_ind) for c in cases}) >= 5, 'need several space groups for this to mean anything'
    n_checked, n_out_of_box, n_improper = 0, 0, 0
    for c in cases:
        box = torch.tensor(ASYM_UNITS[str(int(c.sg_ind))])
        for k in range(int(c.z_prime)):
            for op in c.symmetry_operators:
                r = _redescribe(c, k, op)
                u = r.aunit_centroid.reshape(-1)[3 * k:3 * k + 3] / box
                n_out_of_box += bool((u > 1).any())
                assert _max_atom_deviation(c, r) < ATOM_TOL, 're-described crystal differs from the original'

                cb = collate_data_list([r.clone()])
                canonicalize_aunit(cb)
                canon = cb.batch_to_list()[0]
                u_all = canon.aunit_centroid.reshape(-1)[:3 * int(c.z_prime)].reshape(-1, 3) / box
                assert (u_all >= -1e-6).all() and (u_all <= 1 + 1e-6).all(), 'canonical centre left the box'
                assert _max_atom_deviation(c, canon) < ATOM_TOL, 'canonicalisation changed the crystal'

                det = float(np.linalg.det(np.asarray(op, dtype=np.float64)[:3, :3]))
                if pucker and det < 0:  # the comparison must be able to fail: withhold the handedness flip
                    n_improper += 1
                    wrong = _redescribe(c, k, op, flip=False)
                    assert _max_atom_deviation(c, wrong) > 0.1, 'atom comparison cannot see a handedness error'
                n_checked += 1
    assert n_checked >= 40 and n_out_of_box >= n_checked // 4, (n_checked, n_out_of_box)
    if pucker:
        assert n_improper >= 10, n_improper


# ---------------------------------------------------------------------------
# 3. the optimiser: clamp freezes a centre on a face, wrap does not
# ---------------------------------------------------------------------------

def _face_start(axis):
    acr = [c for c in _load(ACRIDINE) if int(c.sg_ind) == 14 and int(c.z_prime) == 2]
    if not acr:
        pytest.skip('no sg14 Z\'=2 crystal in the mini acridine set')
    c = acr[0].clone()
    cen = c.aunit_centroid.clone()
    cen.reshape(-1)[axis] = 0.0  # molecule 0 exactly on the x=0 (periodic) or y=0 (inversion) face
    c.aunit_centroid = cen
    return collate_data_list([c])


def _run(batch, **kw):
    return gradient_descent_optimization(batch.full_cell_parameters(), batch, optimizer_func='rprop', init_lr=0.01,
                                         max_num_steps=12, optim_target='elj', cutoff=10, show_tqdm=False, **kw)


def test_default_boundary_is_the_clamp():
    assert inspect.signature(gradient_descent_optimization).parameters['centroid_boundary'].default == 'clamp'
    with pytest.raises(ValueError):
        _run(_face_start(0), centroid_boundary='fold')


@pytest.mark.parametrize('axis', [0, 1], ids=['x0_periodic_face', 'y0_inversion_face'])
def test_clamp_freezes_a_face_centre_and_wrap_frees_it(axis):
    _, rec_clamp = _run(_face_start(axis))
    _, rec_wrap = _run(_face_start(axis), centroid_boundary='wrap')
    clamp_traj = rec_clamp['params'][:, 0, 6 + axis]
    wrap_traj = rec_wrap['params'][:, 0, 6 + axis]
    assert torch.equal(clamp_traj, torch.zeros_like(clamp_traj)), 'default clamp should hold the face (documented trap)'
    assert (wrap_traj != 0).any(), 'wrap mode must let the centre leave the face'
