"""Generate the crystal fundamental-domain tables: ``data/fd_tables_v1.json`` and ``data/MANIFEST.json``.

OFFLINE ONLY. This module imports cctbx (2025.11 was used); the runtime loader
``tables.py`` never does.

Per space group 1-230, in the standard setting of ``SYM_OPS[sg]``, the table holds

* ``G``: the operators of SYM_OPS[sg] as exact rationals, index-aligned (translations
  not reduced);
* ``N_reps``: one representative per coset of N_E(G) / G, the identity first, taken
  from NORMALIZER_OPS where the list has one;
* ``gamma``: N_E(G) modulo Z^3 and free-axis shifts, as (W, w, det), ordered by
  coset; ``gamma_plus`` indexes its det +1 elements, ``kernel`` the subgroup K of
  Gamma+ acting trivially on the centroid once free axes are projected out;
* ``lattice_basis``: the translation lattice L+ of Gamma+ (columns, HNF);
* ``pplus``: the rotation group P+ of Gamma+ (name, rotation axes as lattice
  directions, principal and secondary axes);
* ``g_asu`` and ``asu``: cctbx's asymmetric units of G and of Gamma+ (Gamma+/K
  acting on the non-free coordinates, as a prism over the free axes), in our basis,
  with cut planes, inclusion flags and tie-break sub-cuts, exact vertices and volume;
* ``facets``: for every facet of ``asu``, the elements gluing it (Gamma+ index,
  lattice shift, kind, landing facet, area share);
* ``ar``: an autoregressive coordinate order with Fourier-Motzkin bounds for ``asu``;
* ``trainer_box``: ASYM_UNITS[sg], its coverage of a chiral molecule's crystals on
  the +1 sheet and the images per covered crystal (k_box);
* ``validation``: the checks below, all of which must pass for the file to be written.

Usage
-----
``python -m mxtaltools.crystal_fd.generate_fd_tables``          all groups, writes data/
``python -m mxtaltools.crystal_fd.generate_fd_tables --sgs 2,14 --dry-run``
"""
import argparse
import datetime
import itertools
import json
import subprocess
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from cctbx import sgtbx
from cctbx.sgtbx.direct_space_asu.cut_plane import cut as CctbxCut, cut_expression as CctbxCutExpression

from mxtaltools.constants.asymmetric_units import ASYM_UNITS, RAW_ASYM_UNITS
from mxtaltools.constants.space_group_info import (CONTINUOUS_DIMS, LATTICE_TYPE, NORMALIZER_OPS, SPACE_GROUPS,
                                                   SYM_OPS)
from mxtaltools.crystal_fd import exact as X
from mxtaltools.crystal_fd.exact import DEN, Op

SCHEMA_VERSION = 1
DATA_DIR = Path(__file__).with_name('data')
TABLE_NAME = 'fd_tables_v1.json'
SEED = 20261015
N_ORBIT_POINTS = 2000
FACE_TOL = 1e-10


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def g_rationals(sg):
    """SYM_OPS[sg] as (W int [3, 3], w exact [3]) pairs, index-aligned, translations unreduced."""
    out = []
    for o in SYM_OPS[sg]:
        o = np.asarray(o, dtype=float)
        W = np.round(o[:3, :3]).astype(np.int64)
        assert np.abs(o[:3, :3] - W).max() == 0, f'sg {sg}: non-integer rotation'
        out.append((W, [X.rationalize(v) for v in o[:3, 3]]))
    return out


def listed_normalizer(sg):
    """NORMALIZER_OPS[str(sg)] as exact (W, w) pairs, in list order."""
    out = []
    for W, w in NORMALIZER_OPS.get(str(sg), []):
        Wi = np.round(np.asarray(W, dtype=float)).astype(np.int64)
        out.append((Wi, [X.rationalize(v) for v in w]))
    return out


def generic_metric(sg, rng):
    """Gram matrix of a random generic cell of the group's family (MXtalTools frame)."""
    lt = LATTICE_TYPE[sg]
    a, b, c = rng.uniform(5, 15, 3)
    al, be, ga = {'triclinic': rng.uniform(70, 110, 3), 'monoclinic': (90., rng.uniform(95, 125), 90.),
                  'orthorhombic': (90., 90., 90.), 'tetragonal': (90., 90., 90.), 'hexagonal': (90., 90., 120.),
                  'cubic': (90., 90., 90.)}[lt]
    if lt == 'tetragonal' or lt == 'hexagonal':
        b = a
    if lt == 'cubic':
        b = c = a
    al, be, ga = np.radians([al, be, ga])
    cx, cy = c * np.cos(be), c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    B = np.array([[a, b * np.cos(ga), cx], [0, b * np.sin(ga), cy], [0, 0, np.sqrt(c ** 2 - cx ** 2 - cy ** 2)]])
    return B.T @ B


# ---------------------------------------------------------------------------
# cctbx bridge
# ---------------------------------------------------------------------------

def cctbx_group(ops):
    g = sgtbx.space_group('P 1', t_den=DEN)
    for o in ops:
        g.expand_smx(sgtbx.rt_mx(sgtbx.rot_mx([int(v) for v in o.W.ravel()], 1),
                                 sgtbx.tr_vec([int(v) for v in o.t], DEN)))
    return g


def op_from_rt_mx(op, free=X.NO_FREE):
    r_den = op.r().den()
    W = np.array(op.r().num(), dtype=np.int64).reshape(3, 3)
    assert r_den == 1 and np.all(W % 1 == 0)
    t = [Fraction(int(v), op.t().den()) for v in op.t().num()]
    return Op.from_rationals(W, t, free)


def cut_json(node):
    """A cctbx cut or cut_expression as a JSON tree with integer normals and a rational offset."""
    if isinstance(node, CctbxCutExpression):
        return {'op': node.op, 'lhs': cut_json(node.lhs), 'rhs': cut_json(node.rhs)}
    assert isinstance(node, CctbxCut)
    n = [Fraction(str(v)) for v in node.n]
    c = Fraction(str(node.c))
    L = 1
    for q in n + [c]:
        L = X.lcm(L, q.denominator)
    if any(q.denominator != 1 for q in n):
        n, c = [q * L for q in n], c * L
    return {'n': [int(q) for q in n], 'c': str(c), 'inclusive': bool(node.inclusive),
            'sub': cut_json(node.cut_expr) if node.has_cuts() else None}


def asu_entry(info):
    """cctbx direct-space asymmetric unit of a space_group_info, in its own basis, with exact geometry."""
    asu = info.direct_space_asu()
    cuts = [cut_json(c) for c in asu.cuts]
    normals, offsets = X.cut_planes(cuts)
    verts = X.polytope_vertices(normals, offsets)
    return dict(type_number=int(info.type().number()), symbol=str(info),
                change_of_basis_to_reference=info.change_of_basis_op_to_reference_setting().as_xyz(),
                cuts=cuts, xyz=[c.as_xyz() for c in asu.cuts],
                facet=[X.is_facet(n, c, verts, 3) for n, c in zip(normals, offsets)],
                vertices=[[str(v) for v in p] for p in verts],
                volume=str(X.polytope_volume(normals, offsets, verts)))


def cctbx_normalizer(sg):
    """N_E(G) from cctbx alone: additional generators plus discrete seminvariant shifts; (elements, free)."""
    sgi = sgtbx.space_group_info(number=sg)
    ne = sgi.type().expand_addl_generators_of_euclidean_normalizer(True, True)
    free = np.zeros(3, dtype=bool)
    shifts = []
    for vm in sgtbx.structure_seminvariants(sgi.group()).vectors_and_moduli():
        v = np.array(vm.v, dtype=np.int64)
        if vm.m == 0:
            free |= v != 0
        else:
            shifts.append([Fraction(int(x), vm.m) for x in v])
    gens = [op_from_rt_mx(ne(i), free) for i in range(ne.order_z())]
    gens += [Op.from_rationals(np.eye(3), s, free) for s in shifts]
    return X.closure(gens, free), free


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------

_ORDER_BY_TRACE = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}


def rotation_axis(W):
    """Primitive integer direction (fractional coordinates) of the axis of a proper rotation W != I."""
    A = W - np.eye(3, dtype=np.int64)
    for i, j in itertools.combinations(range(3), 2):
        v = np.cross(A[i], A[j])
        if np.any(v):
            break
    g = np.gcd.reduce(np.abs(v))
    v = v // g
    if v[np.nonzero(v)[0][0]] < 0:
        v = -v
    assert not np.any(A @ v)
    return tuple(int(x) for x in v)


def classify_pplus(Ws):
    """Name and rotation axes of a finite proper rotation group given by distinct integer matrices."""
    n = len(Ws)
    orders = [_ORDER_BY_TRACE[int(np.trace(W))] for W in Ws]
    cnt = Counter(orders)
    maxo = max(orders)
    if n == 1:
        name = 'C1'
    elif maxo == n:
        name = f'C{n}'
    elif n == 2 * maxo:
        name = f'D{maxo}'
    elif n == 12 and cnt[3] == 8:
        name = 'T'
    elif n == 24 and cnt[4] == 6:
        name = 'O'
    else:
        raise ValueError(f'unclassified rotation group of order {n}: {dict(cnt)}')
    axes = {}
    for W, o in zip(Ws, orders):
        if o > 1:
            ax = rotation_axis(W)
            axes[ax] = max(axes.get(ax, 1), o)
    axes_sorted = sorted(axes.items(), key=lambda kv: (-kv[1], kv[0][::-1]))
    principal = secondary = None
    if name[0] in 'CD' and n > 1:
        top = [ax for ax, o in axes_sorted if o == maxo]
        principal = (0, 0, 1) if (0, 0, 1) in top else top[0]
        if name[0] == 'D':
            secondary = next(ax for ax, o in axes_sorted if o == 2 and ax != principal)
    return dict(name=name, order=n, axes=[dict(axis=list(ax), order=o) for ax, o in axes_sorted],
                principal_axis=list(principal) if principal else None,
                secondary_axis=list(secondary) if secondary else None)


def project(op, free):
    """Action on the non-free coordinates, as an element acting trivially along free axes."""
    W = op.W.copy()
    keep = ~free
    assert not W[np.ix_(keep, free)].any() and not W[np.ix_(free, keep)].any(), 'W mixes free and fixed axes'
    W[np.ix_(free, free)] = np.eye(int(free.sum()), dtype=np.int64)
    return Op(W, op.t, free)


# ---------------------------------------------------------------------------
# facet gluings (exact)
# ---------------------------------------------------------------------------

def _poly2d(points, drop):
    keep = [i for i in range(3) if i != drop]
    pts = [(p[keep[0]], p[keep[1]]) for p in points]
    cx = sum(float(p[0]) for p in pts) / len(pts)
    cy = sum(float(p[1]) for p in pts) / len(pts)
    return sorted(pts, key=lambda p: np.arctan2(float(p[1]) - cy, float(p[0]) - cx))


def _area2(poly):
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))


def _clip(subject, clipper):
    """Sutherland-Hodgman intersection of convex counter-clockwise polygons, exact."""
    out = list(subject)
    for a, b in zip(clipper, clipper[1:] + clipper[:1]):
        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        inp, out = out, []
        for P, Q in zip(inp, inp[1:] + inp[:1]):
            sp, sq = side(P), side(Q)
            if sq >= 0:
                if sp < 0:
                    s = sp / (sp - sq)
                    out.append((P[0] + s * (Q[0] - P[0]), P[1] + s * (Q[1] - P[1])))
                out.append(Q)
            elif sp >= 0:
                s = sp / (sp - sq)
                out.append((P[0] + s * (Q[0] - P[0]), P[1] + s * (Q[1] - P[1])))
        if not out:
            break
    return out


def facet_gluings(asu, gamma_ops, plus_idx, free):
    """For each facet of the Gamma+ asymmetric unit, the elements carrying the region just outside it back in.

    Exact: every tile g(P) + s of the tiling by Gamma+/K (projected) that lies on the
    far side of a facet's plane and meets the facet in positive area contributes one
    piece. The gluing element is the inverse of that tile map.
    """
    normals, offsets = X.cut_planes(asu['cuts'])
    verts = [tuple(Fraction(v) for v in p) for p in asu['vertices']]
    vf = np.array([[float(v) for v in p] for p in verts])
    lo, hi = vf.min(0), vf.max(0)
    facets = [i for i, f in enumerate(asu['facet']) if f]
    # projected Gamma+ elements, deduplicated over K, each with its first Gamma+ index
    proj = {}
    for gi in plus_idx:
        p = project(gamma_ops[gi], free)
        proj.setdefault(p.key, (p, gi))
    out = {i: [] for i in facets}
    face_poly = {}
    for i in facets:
        drop = int(np.argmax([abs(v) for v in normals[i]]))
        poly = _poly2d(X.tight(normals[i], offsets[i], verts), drop)
        face_poly[i] = (drop, poly, abs(_area2(poly)))
    nf = np.array([[float(v) for v in n] for n in normals])
    cf = np.array([float(c) for c in offsets])
    for p, gi in proj.values():
        Wf, wf = p.W.astype(float), p.t / DEN
        tv = vf @ Wf.T + wf
        tlo, thi = tv.min(0), tv.max(0)
        rng_s = [range(int(np.ceil(lo[d] - thi[d] - 1e-9)), int(np.floor(hi[d] - tlo[d] + 1e-9)) + 1) for d in range(3)]
        for s in itertools.product(*rng_s):
            tvs = tv + np.array(s, dtype=float)
            vals = tvs @ nf.T + cf
            for i in facets:
                col = vals[:, i]
                if col.max() > 1e-9 or (np.abs(col) < 1e-9).sum() < 3:
                    continue
                w = [Fraction(int(t), DEN) + int(sj) for t, sj in zip(p.t, s)]
                tile = [tuple(sum(int(p.W[r, k]) * v[k] for k in range(3)) + w[r] for r in range(3)) for v in verts]
                if any(X.evaluate(normals[i], offsets[i], q) > 0 for q in tile):
                    continue
                on = X.tight(normals[i], offsets[i], tile)
                if len(on) < 3:
                    continue
                drop, fpoly, farea = face_poly[i]
                tpoly = _poly2d(on, drop)
                if _area2(tpoly) == 0:
                    continue
                inter = _clip(fpoly, tpoly)
                if len(inter) < 3 or _area2(inter) == 0:
                    continue
                share = abs(_area2(inter)) / farea
                # lift the piece's vertex centroid to 3D on the facet plane
                keep = [k for k in range(3) if k != drop]
                c2 = [sum(q[j] for q in inter) / len(inter) for j in range(2)]
                c3 = [None] * 3
                c3[keep[0]], c3[keep[1]] = c2
                c3[drop] = -(offsets[i] + sum(Fraction(normals[i][k]) * c3[k] for k in keep)) / normals[i][drop]
                # gluing element g = (tile map)^-1
                Wg = X.int_inverse(p.W)
                wg = [-sum(int(Wg[r, k]) * w[k] for k in range(3)) for r in range(3)]
                y = [sum(int(Wg[r, k]) * c3[k] for k in range(3)) + wg[r] for r in range(3)]
                land = [j for j in facets if X.evaluate(normals[j], offsets[j], y) == 0]
                assert len(land) == 1, f'piece lands on facets {land}'
                g = Op.from_rationals(Wg, [q - (q.numerator // q.denominator) for q in wg], free)
                elem = proj.get(g.key)
                assert elem is not None, 'gluing element is not in Gamma+'
                e_op, e_idx = elem
                shift = [wg[r] - e_op.w()[r] for r in range(3)]
                assert all(q.denominator == 1 for q in shift)
                normal_free = all(normals[i][k] == 0 for k in range(3) if not free[k])
                if normal_free:
                    kind = 'free_axis'
                elif np.array_equal(Wg, np.eye(3, dtype=np.int64)):
                    kind = 'translation'
                elif land[0] == i:
                    kind = 'self_twisted'
                else:
                    kind = 'pair_twisted'
                out[i].append(dict(element=int(e_idx), shift=[int(q) for q in shift], kind=kind,
                                   lands_on=int(land[0]), share=str(share)))
    return [dict(cut=i, gluings=sorted(out[i], key=lambda d: (-Fraction(d['share']), d['element'], d['shift'])))
            for i in facets]


# ---------------------------------------------------------------------------
# trainer box and orbit counts
# ---------------------------------------------------------------------------

def trainer_box(sg, gamma_ops, plus_idx, free):
    """Coverage of ASYM_UNITS[sg] on the +1 sheet and Gamma+ images per covered crystal, exactly.

    The count of Gamma+ images of a centroid inside the box is piecewise constant on
    an arrangement of planes. When every rotation is a signed permutation and every
    translation and box bound lies on the 1/DEN grid, the arrangement is axis-aligned
    on that grid and the midpoints of a 1/DEN grid sample each cell exactly. A box
    equal to the whole cell holds every image of every crystal.
    """
    box = [X.rationalize(v) for v in ASYM_UNITS[str(sg)]]
    keep = ~free
    D = int(keep.sum())
    n_plus = len(plus_idx)
    box_vol_keep = Fraction(1)
    for d in range(3):
        if keep[d]:
            box_vol_keep *= box[d]
    entry = dict(box=[str(b) for b in box], placeholder=str(sg) not in RAW_ASYM_UNITS,
                 mean_images=str(n_plus * box_vol_keep))
    W = np.stack([gamma_ops[i].W[np.ix_(keep, keep)] for i in plus_idx]) if D else None
    t = np.stack([gamma_ops[i].t[keep] for i in plus_idx]) if D else None
    signed_perm = D == 0 or bool(np.all((np.abs(W).sum(1) == 1) & (np.abs(W).sum(2) == 1)))
    if D == 0:
        counts = np.full(1, n_plus)
        method = 'no_fixed_coordinates'
    elif signed_perm and all((b * DEN).denominator == 1 for b in box):
        grid = np.array(list(itertools.product(range(DEN), repeat=D)), dtype=np.int64) * 2 + 1   # over 2*DEN
        img = np.mod(np.einsum('kij,pj->pki', W, grid) + 2 * t[None], 2 * DEN)
        bound = np.array([int(b * 2 * DEN) for b, k in zip(box, keep) if k])
        counts = np.all(img <= bound, axis=2).sum(1)
        method = 'exact_grid'
    elif all(b == 1 for b in box):
        counts = np.full(1, n_plus)
        method = 'whole_cell'
    else:
        raise ValueError(f'sg {sg}: no exact coverage method for box {box}')
    total = len(counts)
    covered = counts > 0
    hist = Counter(int(c) for c in counts)
    k_vals = sorted({int(c) for c in counts[covered]})
    entry.update(method=method, coverage=str(Fraction(int(covered.sum()), total)),
                 k_box=k_vals[0] if len(k_vals) == 1 else None,
                 images_histogram={str(k): str(Fraction(v, total)) for k, v in sorted(hist.items())})
    entry['mean_consistent'] = Fraction(int(counts.sum()), total) == n_plus * box_vol_keep
    return entry


def orbit_count(asu, gamma_ops, plus_idx, free, rng, n=N_ORBIT_POINTS):
    """Images of generic centroids under Gamma+/K inside the asymmetric unit (floats; generic points)."""
    keep = ~free
    D = int(keep.sum())
    if D == 0:
        return dict(points=n, exactly_one=n, on_face=0)
    proj = {}
    for gi in plus_idx:
        p = project(gamma_ops[gi], free)
        proj.setdefault(p.key, p)
    W = np.stack([p.W[np.ix_(keep, keep)] for p in proj.values()]).astype(float)
    w = np.stack([p.t[keep] for p in proj.values()]) / DEN
    normals, offsets = X.cut_planes(asu['cuts'])
    rows = [i for i, nn in enumerate(normals) if any(nn[k] != 0 for k in range(3) if keep[k])]
    N = np.array([[float(normals[i][k]) for k in range(3) if keep[k]] for i in rows])
    C = np.array([float(offsets[i]) for i in rows])
    x = rng.random((n, D))
    img = np.einsum('kij,pj->pki', W, x) + w[None]
    img = img - np.floor(img)
    cnt = np.zeros(n, dtype=np.int64)
    face = np.zeros(n, dtype=bool)
    for s in itertools.product((-1, 0, 1), repeat=D):
        vals = (img + np.array(s, dtype=float)) @ N.T + C
        cnt += np.all(vals > FACE_TOL, axis=2).sum(1)
        face |= np.any(np.all(vals > -FACE_TOL, axis=2) & np.any(np.abs(vals) <= FACE_TOL, axis=2), axis=1)
    return dict(points=n, exactly_one=int((cnt == 1).sum()), on_face=int(face.sum()))


# ---------------------------------------------------------------------------
# one group
# ---------------------------------------------------------------------------

def group_entry(sg):
    rng = np.random.default_rng(SEED + sg)
    free = X.free_mask(CONTINUOUS_DIMS[str(sg)])
    keep = ~free
    val = {}

    G_rat = g_rationals(sg)
    G_exact = [Op.from_rationals(W, [q % 1 for q in w]) for W, w in G_rat]          # mod Z^3 only
    G_mod = {}
    for W, w in G_rat:
        o = Op.from_rationals(W, [q % 1 for q in w], free)
        G_mod.setdefault(o.key, o)
    G_list = list(G_mod.values())
    N_list = listed_normalizer(sg)

    # N_E = closure(G u listed representatives), modulo Z^3 and free shifts
    N_ops = [Op.from_rationals(W, w, free) for W, w in N_list]
    gamma = X.closure(G_list + N_ops, free)

    # cosets r G, ordered: identity, listed representatives in list order, the rest by key
    def coset(r):
        return frozenset(r.compose(g, free).key for g in G_list)
    cosets, reps, rep_src = [], [], []
    ident = Op(np.eye(3), np.zeros(3), free)
    for src, r in [(-1, ident)] + list(enumerate(N_ops)):
        cs = coset(r)
        if cs not in cosets:
            cosets.append(cs)
            reps.append(r)
            rep_src.append(src)
    covered = set().union(*cosets)
    for key in sorted(k for k in gamma if k not in covered):
        if key in covered:
            continue
        cs = coset(gamma[key])
        cosets.append(cs)
        reps.append(gamma[key])
        rep_src.append(None)
        covered |= cs
    gamma_ops, gamma_coset, gamma_g = [], [], []
    for ci, r in enumerate(reps):
        for gj, g in enumerate(G_list):
            gamma_ops.append(r.compose(g, free))
            gamma_coset.append(ci)
            gamma_g.append(gj)
    assert len({o.key for o in gamma_ops}) == len(gamma_ops) == len(gamma)
    plus_idx = [i for i, o in enumerate(gamma_ops) if o.det == 1]
    plus_ops = [gamma_ops[i] for i in plus_idx]
    G_index_of = {o.key: j for j, o in enumerate(G_list)}
    sym_index = [next(k for k, (W, w) in enumerate(G_rat) if Op.from_rationals(W, [q % 1 for q in w], free).key == g.key)
                 for g in G_list]

    # validation: groups
    Wg, tg = X.stack_ops(gamma_ops)
    val['gamma_closed'] = X.is_closed(Wg, tg, free)
    Wp, tp = X.stack_ops(plus_ops)
    val['gamma_plus_closed'] = X.is_closed(Wp, tp, free)
    val['gamma_plus_det_one'] = all(o.det == 1 for o in plus_ops)
    plus_keys = {o.key for o in plus_ops}
    val['gamma_plus_contains_g_plus'] = all(g.key in plus_keys for g in G_list if g.det == 1)
    Gx = {o.key for o in G_exact}
    val['normalizer_ops_normalise_g'] = all(
        all(n.compose(g).compose(n.inverse()).key in Gx for g in G_exact)
        for n in (Op.from_rationals(W, w) for W, w in N_list))
    M = generic_metric(sg, rng)
    val['normalizer_ops_isometric'] = all(np.abs(W.T @ M @ W - M).max() < 1e-8 * np.abs(M).max() for W, _ in N_list)
    brute = X.brute_force_normalizer(G_exact, free, M)
    val['gamma_equals_brute_force'] = set(brute) == set(gamma)
    val['gamma_plus_equals_brute_force'] = {k for k, o in brute.items() if o.det == 1} == plus_keys
    cct, cct_free = cctbx_normalizer(sg)
    val['free_axes_equal_cctbx'] = bool(np.array_equal(cct_free, free))
    val['gamma_equals_cctbx'] = set(cct) == set(gamma)
    val['gamma_plus_equals_cctbx'] = {k for k, o in cct.items() if o.det == 1} == plus_keys
    sgi = sgtbx.space_group_info(number=sg)
    cct_G = {op_from_rt_mx(sgi.group()(i)).key for i in range(sgi.group().order_z())}
    val['g_equals_cctbx_standard_setting'] = cct_G == Gx

    # K, lattice, P+
    kernel = [i for i in plus_idx
              if np.array_equal(gamma_ops[i].W[np.ix_(keep, keep)], np.eye(int(keep.sum()), dtype=np.int64))
              and not gamma_ops[i].t[keep].any()]
    lattice = X.translation_lattice(plus_ops)
    lat_index = Fraction(1) / abs(X.det3(*lattice))
    assert lat_index.denominator == 1
    Ws = {}
    for o in plus_ops:
        Ws.setdefault(tuple(o.W.ravel().tolist()), o.W)
    pplus = classify_pplus(list(Ws.values()))

    # asymmetric units
    g_info = sgtbx.space_group_info(group=cctbx_group(G_exact))
    g_asu = asu_entry(g_info)
    proj = {}
    for i in plus_idx:
        p = project(gamma_ops[i], free)
        proj.setdefault(p.key, p)
    p_group = cctbx_group(list(proj.values()))
    val['projected_order_matches'] = p_group.order_z() == len(proj) == len(plus_idx) // max(len(kernel), 1)
    asu = asu_entry(sgtbx.space_group_info(group=p_group))
    normals, offsets = X.cut_planes(asu['cuts'])
    prism = all(all(n[k] == 0 for k in range(3) if free[k]) or all(n[k] == 0 for k in range(3) if keep[k])
                for n in normals)
    for d in np.nonzero(free)[0]:
        lo_hi = sorted((Fraction(-to) / n[d], n[d] > 0) for n, to in zip(normals, offsets)
                       if n[d] != 0 and all(n[k] == 0 for k in range(3) if k != d))
        prism &= [q for q, _ in lo_hi] == [0, 1]
    val['asu_is_prism_over_free_axes'] = prism
    val['asu_volume_times_order'] = str(Fraction(asu['volume']) * len(proj))
    val['g_asu_volume_times_order'] = str(Fraction(g_asu['volume']) * len(G_exact))

    facets = facet_gluings(asu, gamma_ops, plus_idx, free)
    val['every_facet_glued'] = all(f['gluings'] for f in facets)
    val['gluing_shares_sum_to_one'] = all(sum(Fraction(g['share']) for g in f['gluings']) == 1 for f in facets)

    # AR bounds over the fixed coordinates, natural order
    order = [d for d in range(3) if keep[d]]
    stages = X.ar_stages(normals, offsets, order, eliminate=[d for d in range(3) if free[d]])
    if order:
        hn, ho = X.ar_halfspaces(stages, order)
        ar_verts = set(X.polytope_vertices(hn, ho))
        p_verts = {tuple(Fraction(p[d]) for d in order) for p in asu['vertices']}
        val['ar_bounds_reproduce_asu'] = ar_verts == p_verts
    else:
        val['ar_bounds_reproduce_asu'] = True

    box = trainer_box(sg, gamma_ops, plus_idx, free)
    val['trainer_box_mean_consistent'] = box.pop('mean_consistent')
    val['orbit_count'] = orbit_count(asu, gamma_ops, plus_idx, free, rng)
    val['kernel_trivial_outside_6_9'] = (len(kernel) > 1) == (sg in (6, 7, 8, 9))

    def rat_list(q):
        return [str(v) for v in q]

    entry = dict(
        sg=sg, symbol=SPACE_GROUPS[sg], lattice_type=LATTICE_TYPE[sg], free=[bool(v) for v in free],
        G=dict(W=[W.ravel().tolist() for W, _ in G_rat], w=[rat_list(w) for _, w in G_rat],
               det=[int(round(np.linalg.det(W))) for W, _ in G_rat]),
        N_reps=[dict(W=r.W.ravel().tolist(), w=rat_list(r.w()), source=s) for r, s in zip(reps, rep_src)],
        normalizer_ops_listed=len(N_list),
        gamma=dict(W=[o.W.ravel().tolist() for o in gamma_ops], w=[rat_list(o.w()) for o in gamma_ops],
                   det=[o.det for o in gamma_ops], coset=gamma_coset,
                   sym_op=[sym_index[j] if c == 0 else -1 for c, j in zip(gamma_coset, gamma_g)]),
        gamma_plus=plus_idx,
        kernel=kernel,
        lattice_basis=[rat_list(row) for row in lattice],
        lattice_index=int(lat_index),
        pplus=pplus,
        g_asu=g_asu,
        asu=asu,
        facets=facets,
        ar=dict(order=order, stages=[dict(coord=s['coord'],
                                          lower=[dict(coef=rat_list(c), const=str(k)) for c, k in s['lower']],
                                          upper=[dict(coef=rat_list(c), const=str(k)) for c, k in s['upper']])
                                     for s in stages]),
        trainer_box=box,
        orders=dict(G=len(G_exact), G_plus=sum(1 for o in G_exact if o.det == 1), N_E=len(gamma_ops),
                    gamma_plus=len(plus_idx), K=len(kernel), cosets=len(reps)),
        validation=val,
    )
    return entry


def failures(entry):
    v = entry['validation']
    bad = [k for k, x in v.items() if x is False]
    if v['asu_volume_times_order'] != '1':
        bad.append('asu_volume_times_order')
    if v['g_asu_volume_times_order'] != '1':
        bad.append('g_asu_volume_times_order')
    oc = v['orbit_count']
    if oc['exactly_one'] != oc['points']:
        bad.append('orbit_count')
    return bad


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def dump_tables(entries) -> str:
    """One group per line, so a regeneration diff is read group by group."""
    head = dict(schema_version=SCHEMA_VERSION, den=DEN,
                generator='mxtaltools/crystal_fd/generate_fd_tables.py')
    lines = ['{'] + [f' {json.dumps(k)}: {json.dumps(v)},' for k, v in head.items()] + [' "groups": {']
    keys = sorted(entries)
    for i, sg in enumerate(keys):
        sep = ',' if i < len(keys) - 1 else ''
        lines.append(f'  "{sg}": {json.dumps(entries[sg], separators=(",", ":"))}{sep}')
    lines += [' }', '}', '']
    return '\n'.join(lines)


def _git(*args):
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write(entries, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / TABLE_NAME
    table_path.write_bytes(dump_tables(entries).encode('utf-8'))
    from libtbx.version import get_version
    status = _git('status', '--porcelain')
    manifest = dict(schema_version=SCHEMA_VERSION, table_file=TABLE_NAME, table_sha256=X.file_sha256(table_path),
                    generator='mxtaltools/crystal_fd/generate_fd_tables.py', generator_commit=_git('rev-parse', 'HEAD'),
                    working_tree_dirty=None if status is None else bool(status), cctbx_version=get_version(),
                    date=datetime.date.today().isoformat(), den=DEN, groups=len(entries),
                    input_hashes=X.input_hashes())
    (out_dir / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    return table_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sgs', default='1-230', help="'1-230' or a comma list")
    ap.add_argument('--out', default=str(DATA_DIR))
    ap.add_argument('--dry-run', action='store_true', help='build and validate, write nothing')
    args = ap.parse_args(argv)
    if '-' in args.sgs:
        a, b = (int(v) for v in args.sgs.split('-'))
        sgs = list(range(a, b + 1))
    else:
        sgs = [int(v) for v in args.sgs.split(',')]
    entries, bad = {}, {}
    t0 = time.time()
    for sg in sgs:
        t = time.time()
        e = group_entry(sg)
        entries[sg] = e
        f = failures(e)
        if f:
            bad[sg] = f
        o = e['orders']
        print(f"sg {sg:3d} {e['symbol']:10s} |G| {o['G']:3d} |N_E| {o['N_E']:4d} |Gamma+| {o['gamma_plus']:3d} "
              f"|K| {o['K']} P+ {e['pplus']['name']:3s} facets {len(e['facets'])} "
              f"box cov {e['trainer_box']['coverage']:>5s} k {e['trainer_box']['k_box']} "
              f"{'FAIL ' + ','.join(f) if f else 'ok'} ({time.time() - t:.1f} s)", flush=True)
    print(f'{len(sgs)} groups in {time.time() - t0:.0f} s; {len(bad)} failing')
    if bad:
        print('failures:', bad)
        return 1
    if not args.dry_run:
        if sgs != list(range(1, 231)):
            print('refusing to write a partial table; pass --dry-run')
            return 1
        path = write(entries, Path(args.out))
        print(f'wrote {path} and MANIFEST.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
