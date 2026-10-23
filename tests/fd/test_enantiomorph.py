"""The enantiomorph of Z' = 1 Sohncke rows (``crystal_fd.enantiomorph``) and the importer's
``off_sheet='enantiomorph'`` (``io_mxt``), owner decision 2026-10-21.

1. The map, all 65 Sohncke groups: sg_out is sg exactly where Gamma has an improper element and the ITA partner in the
   11 pairs (22 groups); P = I; p = 0 except sg 80 (0, 1/2, 0), 98 (0, 1/2, 1/4) and 210 (1/4, 1/4, 1/4); (P, p)
   re-expresses the inverted operators as SYM_OPS[sg_out] (``settings.setting_matches``); where sg_out = sg the map
   records the element (-I, -p) of Gamma; the map of sg_out leads back to sg; P and p are read-only. io_mxt's
   docstring and the README name the origin shift -p and exactly the groups where the map derives one.
2. Atoms, every group, random float64 crystals of a random asymmetric molecule: the output's unit cell (built from
   ``group_table(sg_out)``'s exact G by ``checks.unit_cell_fractional``) is the inversion of the input's up to a
   translation, every atom within 1e-9 Angstrom of a same-element atom and back (minimum image over the 27
   neighbouring cells), and the translations that match are exactly -p plus the pure translations (lattice and
   centring) of SYM_OPS[sg_out]. Controls, each more than 0.1 Angstrom off under every translation: h flipped alone
   (the input's own pose: not the inversion), p dropped (sg 80, 98, 210), the source space group kept (partner
   groups). Also h' = -h, cell and R' = R diag(1, -1, -1) bit for bit, t' in [0, 1).
3. Twice is the identity: sg, cell, R, h bit for bit, t within 1e-15 (mod 1).
4. ``canon.canonicalize(sheet=1)`` of the output of h = -1 rows: canonical in every group (sg >= 16 has a cell step
   since 2026-10-21; pose_only before), on the +1 sheet, no improper move; the inputs are off the sheet.
5. The strict wrap: tiny and signed-zero coordinates (5e-324, 1e-17, -0.0, 1.0, 1 - 1e-16) give t' in [0, 1), where
   the unguarded ``x - floor(x)`` gives 1.0, and t' = -t - p (mod 1) within 1e-15.
6. The ``settings.re_express`` path (for P != I, which no current table gives) equals the direct path to 1e-12.
7. Refusals: non-Sohncke groups and sg outside 1-230 (ValueError), Z' > 1 (NotImplementedError); padded unit slots of
   Z' = 1 rows come back bit for bit; an empty batch.
8. Importer on mini_new_prot_csd.pt (local data, opened read-only; skipped when absent), ``off_sheet='enantiomorph'``
   against the D2 default: the same rows and reasons; enantiomorph exactly where D2 mirrored; reference X = X'
   diag(-1, 1, 1) bit for bit (the unmirrored reference's hash); the row = ``enantiomorph`` of the
   ``mirror_reference=False`` row; its unit cell the inversion of the D2 row's (1e-9 Angstrom); h = +1, on the sheet,
   reference_handedness +1, sg_source the source group; every other row bit for bit the D2 row. Parameters:
   contradicting ``mirror_reference`` / ``off_sheet`` raise, consistent spellings give the same rows; sheet=None maps
   nothing; sheet=-1 maps the h = +1 Sohncke rows; with ``allow_zp_gt1`` (padded slots) Z' = 2 rows are never mapped.
9. Importer on synthetic float32 rows of partner, origin-shift and same-group cases (76, 92, 144, 169, 212, 80, 98,
   210, 19), each as a pair of mirror-image molecules: the mapped row's unit cell is the inversion of the source
   crystal and the other row the source crystal itself (1e-3 Angstrom, float32 storage), sg the partner where paired.
   A synthetic Z' = 2 row of sg 19 off the sheet (``allow_zp_gt1``), in one batch with a Z' = 1 row off the sheet:
   only the Z' = 1 row is mapped (and, in the default mode, D2-mirrored); the Z' = 2 row is the ``'keep'`` row.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_fd import canon, checks, io_mxt, settings, so3
from mxtaltools.crystal_fd import enantiomorph as en
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
SOHNCKE = [s for s in range(1, 231) if io_mxt.is_sohncke(s)]
NB = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)
DATA = Path(os.environ.get('MXT_CSD_DATA', 'D:/crystal_datasets'))
#: the expected origin shifts (stated by the 2026-10-21 design; derived by the module, checked here)
SHIFTS = {80: (0.0, 0.5, 0.0), 98: (0.0, 0.5, 0.25), 210: (0.25, 0.25, 0.25)}
ATOM_TOL = 1e-9
CONTROL_TOL = 0.1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_molecule(g: torch.Generator, n: int = 8) -> MoleculeRef:
    """n generic points (N(0, 1.2^2) Angstrom), elements with repeats and at least 4 heavy atoms: asymmetric."""
    pool = torch.tensor([1, 6, 7, 8, 9, 16, 17])
    while True:
        z = pool[torch.randint(0, len(pool), (n,), generator=g)]
        if int((z > 1).sum()) >= 4:
            break
    return MoleculeRef.from_coords(1.2 * torch.randn(n, 3, generator=g, dtype=F64), z)


def _random_rows(sg: int, n: int, seed: int, h=-1) -> CrystalDescription:
    g = torch.Generator().manual_seed(seed)
    cell = checks.random_cells(LATTICE_TYPE[sg], n, g)
    t = torch.rand(n, 3, generator=g, dtype=F64)
    R = so3.random_rotations(n, g)
    hh = torch.full((n,), h, dtype=torch.long) if isinstance(h, int) else torch.as_tensor(h, dtype=torch.long)
    return CrystalDescription.from_parts(sg, cell, t, R, hh)


def _nearest(A, B, T, chunk: int = 64) -> torch.Tensor:
    """[len(A)] Angstrom: distance from each fractional point of A to the nearest of B, minimum image over the 27
    cells around the rounded difference (T the fractional-to-Cartesian matrix)."""
    if len(B) == 0:
        return torch.full((len(A),), float('inf'), dtype=F64)
    out = []
    for lo in range(0, len(A), chunk):
        d = A[lo:lo + chunk, None] - B[None]
        d = d - torch.round(d)
        out.append(((d[:, :, None] + NB) @ T.T).norm(dim=-1).amin((1, 2)))
    return torch.cat(out)


def _set_dev(A, zA, B, zB, T) -> float:
    """Largest distance from an atom of either set to the nearest same-element atom of the other (Angstrom)."""
    if len(A) != len(B):
        return float('inf')
    dev = 0.0
    for el in torch.unique(torch.cat([zA, zB])).tolist():
        a, b = A[zA == el], B[zB == el]
        if len(a) != len(b):
            return float('inf')
        dev = max(dev, float(_nearest(a, b, T).max()), float(_nearest(b, a, T).max()))
    return dev


def inversion_match(F_in, F_out, z, T):
    """Element-aware test that the atoms ``F_out`` [nG', n, 3] are the inversion of ``F_in`` [nG, n, 3] (fractional,
    one basis, cell matrix T) up to a translation v: F_out = -F_in + v (mod Z^3).

    Candidate v send output atom 0 onto a same-element inverted atom. A candidate whose atoms 1-4 already lie further
    than ``CONTROL_TOL`` from every same-element atom is out (its full deviation is at least that); the rest get the
    full set deviation. Returns (smallest deviation, [v whose deviation is below ATOM_TOL]).
    """
    A = F_out.reshape(-1, 3)
    B = (-F_in).reshape(-1, 3)
    zA = z.repeat(F_out.shape[0])
    zB = z.repeat(F_in.shape[0])
    if A.shape != B.shape:
        return float('inf'), []
    V = A[0] - B[zB == zA[0]]                                                  # [c, 3]
    bound = torch.zeros(len(V), dtype=F64)
    for i in range(1, min(5, len(A))):
        Bi = B[zB == zA[i]]
        d = A[i][None, None] - Bi[None] - V[:, None]                           # [c, m, 3]
        d = d - torch.round(d)
        bound = torch.maximum(bound, ((d[:, :, None] + NB) @ T.T).norm(dim=-1).amin((1, 2)))
    best, hits = float('inf'), []
    for k in torch.nonzero(bound <= CONTROL_TOL)[:, 0].tolist():
        dev = _set_dev(A, zA, B + V[k], zB, T)
        best = min(best, dev)
        if dev < ATOM_TOL:
            hits.append(V[k])
    return best, hits


def _pure_translations(sg: int) -> torch.Tensor:
    tab = group_table(sg)
    eye = (tab.g_W == torch.eye(3, dtype=F64)).all(-1).all(-1)
    w = tab.g_w[eye]
    return w - torch.floor(w)


def _mod1(x: torch.Tensor) -> torch.Tensor:
    return x - torch.round(x)


# ---------------------------------------------------------------------------
# 1: the map
# ---------------------------------------------------------------------------

def test_the_map_of_every_sohncke_group():
    assert len(SOHNCKE) == 65
    paired = []
    for sg in SOHNCKE:
        m = en.enantiomorph_map(sg)
        tab = group_table(sg)
        improper = bool((tab.gamma_det < 0).any())
        assert m.sg == sg and m.sg_out == (sg if improper else en.PARTNER[sg]), sg
        assert np.array_equal(m.P, np.eye(3)) and m.identity_basis, sg
        assert np.array_equal(m.p, np.array(SHIFTS.get(sg, (0.0, 0.0, 0.0)))), (sg, m.p)
        assert settings.setting_matches(en.inverted_ops(sg), m.sg_out, m.P, m.p), sg
        assert en.enantiomorph_map(m.sg_out).sg_out == sg, sg
        assert not m.P.flags.writeable and not m.p.flags.writeable
        if m.sg_out == sg:
            k = m.gamma_index
            assert k is not None and int(tab.gamma_det[k]) == -1
            assert torch.equal(tab.gamma_W[k], -torch.eye(3, dtype=F64))
            d = _mod1(tab.gamma_w[k] + torch.as_tensor(np.array(m.p)))
            assert float(torch.where(tab.free, 0.0, d).abs().max()) < 1e-12, sg
        else:
            assert m.gamma_index is None
            paired.append(sg)
    assert sorted(paired) == sorted(en.PARTNER) and len(paired) == 22
    assert all(io_mxt.is_sohncke(s) for s in en.PARTNER)


def test_docs_state_the_origin_shift():
    """io_mxt's module docstring and the README state the enantiomorph row's atoms as the D2 row's inverted and
    shifted by -p, naming exactly the groups where the map derives p != 0 (dropping p is off by Angstroms there:
    test_enantiomorph_is_the_inversion)."""
    shifted = [s for s in SOHNCKE if bool(np.any(en.enantiomorph_map(s).p != 0))]
    assert shifted == sorted(SHIFTS)
    readme = (Path(io_mxt.__file__).parent / 'README.md').read_text(encoding='utf-8')
    for name, text in (('io_mxt', io_mxt.__doc__), ('README', readme)):
        flat = ' '.join(text.split())
        assert 'shifted by the origin shift -p' in flat, name
        assert f"nonzero only in sg {', '.join(map(str, shifted))}" in flat, name


# ---------------------------------------------------------------------------
# 2-4: atoms, involution, canonicalisation, every group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', SOHNCKE)
def test_enantiomorph_is_the_inversion(sg):
    n = 6
    d = _random_rows(sg, n, 1000 + sg, h=[-1, -1, -1, -1, 1, -1])
    gm = torch.Generator().manual_seed(2000 + sg)
    mols = [_random_molecule(gm) for _ in range(n)]
    m = en.enantiomorph_map(sg)
    e = en.enantiomorph(d)
    e.validate(tol=1e-12)
    assert e.sg.tolist() == [m.sg_out] * n
    assert torch.equal(e.cell, d.cell) and torch.equal(e.h, -d.h)
    assert torch.equal(e.R, d.R * torch.tensor([1.0, -1.0, -1.0], dtype=F64))
    assert bool((e.t >= 0).all()) and bool((e.t < 1).all())
    p = torch.as_tensor(np.array(m.p))
    assert float(_mod1(e.t[:, 0] - (-d.t[:, 0] - p)).abs().max()) < 1e-15
    taus = _pure_translations(m.sg_out)
    for i in range(n):
        T = d[i].cell_matrix()[0]
        F_in = checks.unit_cell_fractional(d[i], mols[i])[0]
        best, hits = inversion_match(F_in, checks.unit_cell_fractional(e[i], mols[i])[0], mols[i].z, T)
        assert best < ATOM_TOL, (sg, i, best)
        # the translations that match: exactly -p plus the lattice and centring translations of SYM_OPS[sg_out]
        assert len(hits) == len(taus), (sg, len(hits), len(taus))
        for v in hits:
            assert float(_mod1(v + p - taus).abs().amax(1).min()) < 1e-9, (sg, v)
        if i >= 2:
            continue
        controls = {'h_flipped': d[i].replace(h=-d[i].h)}
        if sg in SHIFTS:
            controls['p_dropped'] = e[i].replace(t=en._wrap(-d[i].t))
        if m.sg_out != sg:
            controls['sg_kept'] = e[i].replace(sg=d[i].sg)
        for name, wrong in controls.items():
            bad, _ = inversion_match(F_in, checks.unit_cell_fractional(wrong, mols[i])[0], mols[i].z, T)
            assert bad > CONTROL_TOL, (sg, name, bad)
    # twice is the identity
    ee = en.enantiomorph(e)
    assert torch.equal(ee.sg, d.sg) and torch.equal(ee.cell, d.cell) and torch.equal(ee.R, d.R)
    assert torch.equal(ee.h, d.h)
    assert float(_mod1(ee.t - d.t).abs().max()) < 1e-15
    # canonicalisation: the h = -1 inputs are off the +1 sheet, their enantiomorphs on it
    minus = d.h[:, 0] == -1
    c_in = canon.canonicalize(d[minus], sheet=1)
    c_out = canon.canonicalize(e[minus], sheet=1)
    assert not bool(c_in.on_sheet.any())
    assert c_out.names() == ['canonical'] * int(minus.sum()), (sg, c_out.names())
    assert bool(c_out.on_sheet.all()) and bool((c_out.desc.h[:, 0] == 1).all()) and not bool(c_out.sheet_moved.any())


# ---------------------------------------------------------------------------
# 5: the strict wrap
# ---------------------------------------------------------------------------

#: rows of centroids at the wrap's edges: -t - p is exactly 0, -0.0, an integer, or a negative number of magnitude
#: below half an ulp of 1 (5e-324 and 1e-17 where p = 0; -1/4 + 2^-55 and -1/2 + 2^-54 where p is 1/4 or 1/2), which
#: ``x - floor(x)`` rounds to 1.0
EDGE_T = [[5e-324, 1e-17, -0.0], [0.0, 1.0, 1e-300], [-1e-17, 0.5, 1 - 1e-16], [1e-16, 0.75, 0.25],
          [1.0, 5e-324, 0.5], [0.5, 0.5, 0.5], [-0.25 + 2 ** -55, -0.5 + 2 ** -54, -0.25 + 2 ** -55]]


@pytest.mark.parametrize('sg', [1, 4, 19, 80, 98, 144, 210])
def test_wrap_is_strict(sg):
    t = torch.tensor(EDGE_T, dtype=F64)
    n = len(t)
    g = torch.Generator().manual_seed(sg)
    d = CrystalDescription.from_parts(sg, checks.random_cells(LATTICE_TYPE[sg], n, g), t, so3.random_rotations(n, g),
                                      -torch.ones(n, dtype=torch.long))
    p = torch.as_tensor(np.array(en.enantiomorph_map(sg).p))
    naive = (-t - p) - torch.floor(-t - p)
    assert bool((naive == 1.0).any())                          # the case the guard exists for
    e = en.enantiomorph(d)
    assert bool((e.t >= 0).all()) and bool((e.t < 1).all())
    assert float(_mod1(e.t[:, 0] - (-t - p)).abs().max()) < 1e-15
    assert float(_mod1(en.enantiomorph(e).t[:, 0] - t).abs().max()) < 1e-15


# ---------------------------------------------------------------------------
# 6: the re_express path
# ---------------------------------------------------------------------------

def test_re_express_path_equals_the_direct_path():
    worst = 0.0
    for sg in SOHNCKE:
        d = _random_rows(sg, 3, 3000 + sg)
        m = en.enantiomorph_map(sg)
        a = en._invert(d, m)
        b = en._invert(d, m, via_re_express=True)
        worst = max(worst, float((a[0] - b[0]).abs().max()), float(_mod1(a[1] - b[1]).abs().max()),
                    float((a[2] - b[2]).abs().max()))
        assert torch.equal(a[3], b[3])
        assert bool((b[1] >= 0).all()) and bool((b[1] < 1).all())
    assert worst < 1e-12, worst


# ---------------------------------------------------------------------------
# 7: refusals and shapes
# ---------------------------------------------------------------------------

def test_refusals():
    for sg in (2, 14, 61, 230):
        with pytest.raises(ValueError, match='not a Sohncke group'):
            en.enantiomorph_map(sg)
    for sg in (0, 231):
        with pytest.raises(ValueError):
            en.enantiomorph_map(sg)
    mixed = CrystalDescription.cat([_random_rows(19, 2, 1), _random_rows(14, 1, 2)])
    with pytest.raises(ValueError, match='not a Sohncke group'):
        en.enantiomorph(mixed)
    zp2 = CrystalDescription.from_parts(19, _random_rows(19, 1, 3).cell, torch.rand(1, 2, 3, dtype=F64),
                                        so3.random_rotations(2).reshape(1, 2, 3, 3), torch.tensor([[-1, -1]]))
    with pytest.raises(NotImplementedError):
        en.enantiomorph(zp2)


def test_padded_slots_and_empty_batches():
    d1 = CrystalDescription.cat([_random_rows(19, 2, 4), _random_rows(169, 2, 5), _random_rows(80, 1, 6)])
    B = d1.num_rows
    t = torch.full((B, 2, 3), float('nan'), dtype=F64)
    R = torch.zeros(B, 2, 3, 3, dtype=F64)
    h = torch.zeros(B, 2, dtype=torch.long)
    t[:, 0], R[:, 0], h[:, 0] = d1.t[:, 0], d1.R[:, 0], d1.h[:, 0]
    padded = CrystalDescription.from_parts(d1.sg, d1.cell, t, R, h, torch.ones(B, dtype=torch.long))
    e1, ep = en.enantiomorph(d1), en.enantiomorph(padded)
    for f in ('sg', 'cell', 'z_prime'):
        assert torch.equal(getattr(ep, f), getattr(e1, f)), f
    assert torch.equal(ep.t[:, 0], e1.t[:, 0]) and torch.equal(ep.R[:, 0], e1.R[:, 0]) and torch.equal(ep.h[:, 0],
                                                                                                      e1.h[:, 0])
    assert bool(torch.isnan(ep.t[:, 1]).all()) and bool((ep.R[:, 1] == 0).all()) and bool((ep.h[:, 1] == 0).all())
    assert e1.sg.tolist() == [19, 19, 170, 170, 80]
    empty = en.enantiomorph(d1[torch.zeros(0, dtype=torch.long)])
    assert empty.num_rows == 0 and empty.t.dtype == F64


# ---------------------------------------------------------------------------
# 8: the importer on the mini CSD file
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def prot():
    path = DATA / 'mini_new_prot_csd.pt'
    if not path.exists():
        pytest.skip(f'{path} not on this machine')
    return torch.load(path, weights_only=False)


FIELDS = ('sg', 'cell', 'z_prime', 'unit_mask', 't', 'R', 'h')


def test_importer_enantiomorph_mode_against_d2(prot):
    d2, i2 = io_mxt.from_mol_crystal_data(prot)
    dk, ik = io_mxt.from_mol_crystal_data(prot, mirror_reference=False)
    de, ie = io_mxt.from_mol_crystal_data(prot, off_sheet='enantiomorph')
    assert ie.reason == i2.reason and torch.equal(ie.row, i2.row) and torch.equal(ie.reason_mask, i2.reason_mask)
    assert torch.equal(ie.enantiomorph, i2.mirrored) and int(ie.enantiomorph.sum()) >= 5
    assert not bool(ie.mirrored.any()) and bool((ie.reference_handedness == 1).all())
    assert not bool(i2.enantiomorph.any()) and not bool(ik.enantiomorph.any())
    assert torch.equal(i2.sg_source, d2.sg) and torch.equal(ik.sg_source, dk.sg)
    assert bool(ie.on_sheet.all()) and bool((de.h[:, 0] == 1).all())
    assert torch.equal(ie.h_input, i2.h_input) and torch.equal(ie.setting_P, i2.setting_P)
    assert torch.equal(ie.sg_source, d2.sg)
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64)
    ent = torch.nonzero(ie.enantiomorph)[:, 0]
    rest = torch.nonzero(~ie.enantiomorph)[:, 0]
    for f in FIELDS:
        assert torch.equal(getattr(de, f)[rest], getattr(d2, f)[rest]), f
    assert all(ie.mols[j].hash == i2.mols[j].hash for j in rest.tolist())
    want = en.enantiomorph(dk[ent])
    for f in FIELDS:
        assert torch.equal(getattr(de, f)[ent], getattr(want, f)), f
    assert de.sg[ent].tolist() == [en.enantiomorph_map(s).sg_out for s in d2.sg[ent].tolist()]
    for j in ent.tolist():
        X, Xp = ie.mols[j], i2.mols[j]
        assert torch.equal(X.pos, Xp.pos * flip) and torch.equal(X.z, Xp.z)
        assert X.hash == ik.mols[j].hash and X.hash != Xp.hash
        T = d2[j].cell_matrix()[0]
        F_in = checks.unit_cell_fractional(d2[j], Xp)[0]
        best, hits = inversion_match(F_in, checks.unit_cell_fractional(de[j], X)[0], X.z, T)
        assert best < ATOM_TOL and hits, (ie.identifier[int(ie.row[j])], best)
        ctrl, _ = inversion_match(F_in, checks.unit_cell_fractional(d2[j], X)[0], X.z, T)
        assert ctrl > CONTROL_TOL                                     # X at the D2 pose: not the inversion


def test_importer_off_sheet_parameters(prot):
    rows = [prot[i] for i in range(40)]
    base_d, base_i = io_mxt.from_mol_crystal_data(rows)
    for kw in (dict(mirror_reference=True), dict(off_sheet='mirror_reference'),
               dict(mirror_reference=True, off_sheet='mirror_reference')):
        d, i = io_mxt.from_mol_crystal_data(rows, **kw)
        assert all(torch.equal(getattr(d, f), getattr(base_d, f)) for f in FIELDS), kw
        assert torch.equal(i.mirrored, base_i.mirrored) and [m.hash for m in i.mols] == [m.hash for m in base_i.mols]
    keep_d, keep_i = io_mxt.from_mol_crystal_data(rows, mirror_reference=False)
    for kw in (dict(off_sheet='keep'), dict(mirror_reference=False, off_sheet='keep')):
        d, i = io_mxt.from_mol_crystal_data(rows, **kw)
        assert all(torch.equal(getattr(d, f), getattr(keep_d, f)) for f in FIELDS), kw
        assert torch.equal(i.on_sheet, keep_i.on_sheet) and not bool(i.mirrored.any())
    a_d, _ = io_mxt.from_mol_crystal_data(rows, off_sheet='enantiomorph')
    b_d, _ = io_mxt.from_mol_crystal_data(rows, off_sheet='enantiomorph', mirror_reference=False)
    assert all(torch.equal(getattr(a_d, f), getattr(b_d, f)) for f in FIELDS)
    for kw in (dict(mirror_reference=True, off_sheet='enantiomorph'), dict(mirror_reference=True, off_sheet='keep'),
               dict(mirror_reference=False, off_sheet='mirror_reference')):
        with pytest.raises(ValueError, match='contradicts'):
            io_mxt.from_mol_crystal_data(rows, **kw)
    for kw in (dict(off_sheet='mirror'), dict(off_sheet=True), dict(mirror_reference='yes'), dict(mirror_reference=1)):
        with pytest.raises(ValueError):
            io_mxt.from_mol_crystal_data(rows, **kw)


def test_importer_enantiomorph_sheets(prot):
    none_d, none_i = io_mxt.from_mol_crystal_data(prot, sheet=None)
    d, i = io_mxt.from_mol_crystal_data(prot, sheet=None, off_sheet='enantiomorph')
    assert not bool(i.enantiomorph.any()) and torch.equal(i.sg_source, d.sg)
    assert all(torch.equal(getattr(d, f), getattr(none_d, f)) for f in FIELDS)
    dm, im = io_mxt.from_mol_crystal_data(prot, sheet=-1, off_sheet='enantiomorph')
    km, ki = io_mxt.from_mol_crystal_data(prot, sheet=-1, off_sheet='keep')
    assert torch.equal(im.enantiomorph, ki.sohncke & ~ki.on_sheet & (km.z_prime == 1))
    assert bool(im.enantiomorph.any()) and bool((im.h_input[im.enantiomorph, 0] == 1).all())
    assert bool(im.on_sheet.all()) and bool((dm.h[:, 0] == -1).all())
    ent = torch.nonzero(im.enantiomorph)[:, 0]
    want = en.enantiomorph(km[ent])
    assert all(torch.equal(getattr(dm, f)[ent], getattr(want, f)) for f in FIELDS)


def test_importer_enantiomorph_with_zp_gt1(prot):
    d1, i1 = io_mxt.from_mol_crystal_data(prot, off_sheet='enantiomorph')
    d2, i2 = io_mxt.from_mol_crystal_data(prot, off_sheet='enantiomorph', allow_zp_gt1=True)
    k2, _ = io_mxt.from_mol_crystal_data(prot, off_sheet='keep', allow_zp_gt1=True)
    if d2.max_z_prime < 2:
        pytest.skip("no Z' > 1 row imports from the mini file")
    assert not bool(i2.enantiomorph[d2.z_prime > 1].any())
    assert bool((d2.h[d2.z_prime > 1] == k2.h[k2.z_prime > 1]).all())
    rows1 = i1.row.tolist()
    for j in torch.nonzero(i2.enantiomorph)[:, 0].tolist():
        k = rows1.index(int(i2.row[j]))
        assert bool(i1.enantiomorph[k]) and int(d2.sg[j]) == int(d1.sg[k])
        assert torch.equal(d2.t[j, 0], d1.t[k, 0]) and torch.equal(d2.R[j, 0], d1.R[k, 0])
        assert int(d2.h[j, 0]) == 1
        assert torch.equal(d2.t[j, 1:], k2.t[j, 1:]) and torch.equal(d2.R[j, 1:], k2.R[j, 1:])
        assert torch.equal(d2.h[j, 1:], k2.h[j, 1:])
    assert bool(i2.enantiomorph.any())


# ---------------------------------------------------------------------------
# 9: the importer on synthetic rows of partner and origin-shift groups
# ---------------------------------------------------------------------------

class _Row(dict):
    """A minimal MolCrystalData-like row: the fields the importer reads, as attributes."""
    __getattr__ = dict.__getitem__

    def keys(self):
        return list(super().keys())


def _row_from_desc(d: CrystalDescription, mol: MoleculeRef) -> _Row:
    """A Z' = 1 float32 row (standard setting) whose stored atoms are the literal asymmetric unit of ``d``, with the
    pose re-derived in the importer's aligned frame (as tests/fd/test_io_settings.py writes them)."""
    from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
    from mxtaltools.crystal_fd.io_mxt import _Units
    T = d.cell_matrix()[0]
    pos = (T @ d.t[0, 0] + (d.pose[0, 0] @ mol.pos.T).T).float()
    u = _Units(pos.clone(), mol.z, torch.zeros(len(mol.z), dtype=torch.long), 1)
    align_mol_batch_to_standard_axes(u, handedness=None)
    X = u.pos.double()
    Y = pos.double() - pos.double()[mol.z > 1].mean(0)
    U, _, Vt = torch.linalg.svd(X.T @ Y)
    R, h = so3.split_o3((Vt.T @ U.T)[None])
    c = torch.linalg.solve(T, pos.double()[mol.z > 1].mean(0))
    return _Row(sg_ind=torch.tensor(int(d.sg[0])), z_prime=torch.tensor([1]),
                cell_lengths=d.cell[0, :3].float()[None], cell_angles=d.cell[0, 3:].float()[None],
                aunit_centroid=c.float()[None], aunit_orientation=so3.matrix_to_rotvec(R).float(),
                aunit_handedness=h.reshape(1, 1).float(), pos=pos, z=mol.z.clone(),
                symmetry_operators=settings.std_ops(int(d.sg[0])), identifier='SYNTH')


def test_importer_on_synthetic_rows(chiral_molecule):
    mol = MoleculeRef.from_coords(chiral_molecule['pos'], chiral_molecule['z'])
    groups = (76, 92, 144, 169, 212, 80, 98, 210, 19)
    rows, src = [], []
    for sg in groups:
        d = _random_rows(sg, 2, 4000 + sg, h=1)
        for i in range(2):
            for hs in (1, -1):                                  # a molecule and its mirror image at one pose
                di = d[i].replace(h=torch.tensor([[hs]]))
                rows.append(_row_from_desc(di, mol))
                src.append(di)
    desc, info = io_mxt.from_mol_crystal_data(rows, special_tol=0.0, off_sheet='enantiomorph')
    assert info.imported.all(), info.reason
    assert int(info.enantiomorph.sum()) == len(rows) // 2       # one of each pair
    worst_in, worst_same = 0.0, 0.0
    for j in range(desc.num_rows):
        s = src[int(info.row[j])]
        sg = int(s.sg[0])
        T = s.cell_matrix()[0]
        F_src = checks.unit_cell_fractional(s, mol)[0]
        F_new = checks.unit_cell_fractional(desc[j], info.mols[j])[0]
        Td = desc[j].cell_matrix()[0]
        if bool(info.enantiomorph[j]):
            assert int(desc.sg[j]) == en.enantiomorph_map(sg).sg_out and int(info.sg_source[j]) == sg
            assert torch.equal(info.mols[j].z, mol.z)
            dev, _ = inversion_match(F_src, F_new, mol.z, Td)
            worst_in = max(worst_in, dev)
        else:
            assert int(desc.sg[j]) == sg
            dev = _set_dev(F_new.reshape(-1, 3), info.mols[j].z.repeat(F_new.shape[0]), F_src.reshape(-1, 3),
                           mol.z.repeat(F_src.shape[0]), T)
            worst_same = max(worst_same, dev)
    assert worst_in < 1e-3 and worst_same < 1e-3, (worst_in, worst_same)


def _zp2_row(d0: CrystalDescription, d1: CrystalDescription, mol: MoleculeRef) -> _Row:
    """A Z' = 2 float32 row (one cell, d0's): unit k is ``_row_from_desc(d_k, mol)``, as the importer reads Z' > 1 rows
    (equal contiguous atom blocks, parameter block k for unit k)."""
    a, b = _row_from_desc(d0, mol), _row_from_desc(d1.replace(cell=d0.cell), mol)
    r = _Row(a)
    r.update(z_prime=torch.tensor([2]), pos=torch.cat([a.pos, b.pos]), z=torch.cat([a.z, b.z]),
             **{k: torch.cat([a[k], b[k]], 1) for k in ('aunit_centroid', 'aunit_orientation', 'aunit_handedness')})
    return r


def test_importer_enantiomorph_skips_zp_gt1_rows(chiral_molecule):
    """A synthetic Z' = 2 row of P2_12_12_1, both units at h = -1 (off the +1 sheet), imported with ``allow_zp_gt1`` in
    one batch with a Z' = 1 row off the sheet: the Z' = 1 row is mapped, the Z' = 2 row is not (``enantiomorph``
    raises on Z' > 1 rows): off the sheet, enantiomorph False, sg_source 19, its description and reference bit for bit
    the ``off_sheet='keep'`` import's. D2 likewise mirrors only the Z' = 1 row."""
    mol = MoleculeRef.from_coords(chiral_molecule['pos'], chiral_molecule['z'])
    d = _random_rows(19, 3, 5019, h=-1)
    rows = [_zp2_row(d[1], d[2], mol), _row_from_desc(d[0], mol)]
    kw = dict(special_tol=0.0, allow_zp_gt1=True)
    de, ie = io_mxt.from_mol_crystal_data(rows, off_sheet='enantiomorph', **kw)
    dk, ik = io_mxt.from_mol_crystal_data(rows, off_sheet='keep', **kw)
    _, im = io_mxt.from_mol_crystal_data(rows, **kw)
    assert ie.reason == [None, None] and de.z_prime.tolist() == [2, 1], ie.reason
    assert bool((dk.h[0] == -1).all()) and ik.on_sheet.tolist() == [False, False]
    assert ie.enantiomorph.tolist() == [False, True] and ie.on_sheet.tolist() == [False, True]
    assert ie.sg_source.tolist() == [19, 19] and not bool(ie.mirrored.any())
    assert im.mirrored.tolist() == [False, True] and im.on_sheet.tolist() == [False, True]
    for f in FIELDS:
        assert torch.equal(getattr(de, f)[0], getattr(dk, f)[0]), f
    assert ie.mols[0].hash == ik.mols[0].hash and torch.equal(ie.mols[0].pos, ik.mols[0].pos)
    want = en.enantiomorph(dk[torch.tensor([1])])
    for f in FIELDS:
        assert torch.equal(getattr(de, f)[1:], getattr(want, f)), f
    assert int(de.h[1, 0]) == 1
