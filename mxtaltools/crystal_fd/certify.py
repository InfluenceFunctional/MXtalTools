"""Exact tiling certificate for the centroid part of the pose domain ASU(Gamma+) x SO(3).

Gate 1 counts orbit representatives by sampling. This module proves the centroid
part in exact rational arithmetic, from the table alone (no cctbx, no sampling, and
no other gate: the certificate checks its own premises).

Setting
-------
d is the number of non-free axes. P in R^d is the table's ASU(Gamma+) on those
coordinates: its cuts with a normal on free axes only (the walls) are dropped and
checked separately, and a cut mixing free and non-free axes is refused. Gamma-bar is
Gamma+ acting on R^d through the non-free blocks, ``x -> W x + w + s`` (s in Z^d):
the same projection the fold uses, which sets free coordinates to the gauge. It is
the union over n representatives k of Gamma+/K modulo Z^d (the identity first) of
the cosets k + Z^d. K is trivial outside sg 6-9, where therefore n = |Gamma+|. That
the representatives are Gamma+/K of the space group's Euclidean normaliser is a table
fact (gate 0's brute-force N_E); the certificate takes the list as given and proves
that P tiles R^d under the group it generates.

Certificate (``certify_tiling``; ``ok`` requires every item)
------------------------------------------------------------
(G)   The representatives are a group modulo Z^d (``group_defects``): every W is an
      integer matrix with |det W| = 1 (else ValueError), the first is the identity,
      no two are equal modulo Z^d, and every product of two is one of them modulo Z^d
      (all n^2 compositions, n <= 384). A finite set closed under composition is a
      subgroup, so Gamma-bar is a group in which Z^d has index n, and the pairs (k, s)
      index it without repetition.
(B)   P is bounded and d-dimensional (``pair_intersections`` raises otherwise).
(i)   vol(P) x n = 1, with vol(P) computed here from P's exact vertices.
(ii)  No examined pair (k, s) != (id, 0) meets P in positive volume.
(iii) Facet coverage (``facet_coverage``): for every facet F of P, the contacts lying
      in F (intersections P n (k.P + s) of zero volume and dimension d - 1) have
      measures summing exactly to F's, and every two of them meet in measure zero.
      Measures are (d-1)-dimensional on F's plane, projected along the axis where F's
      normal is largest (a fixed positive multiple of area on that plane, so equality
      is exact): area for d = 3, length for d = 2. For d = 1, F is an endpoint and the
      measure counts points: exactly one examined image touches each endpoint.
(W)   The walls bound each free axis f by exactly a_f <= x_f <= a_f + 1
      (``wall_defects``), so the table's ASU(Gamma+) is the height-1 prism over P.

Examined pairs: s runs over the integer box where the bounding boxes of P and
k.P + s meet, computed exactly from the vertices (outside it the two are disjoint).
The proof below does not need that range to be complete: a range missing a facet's
neighbour fails (iii) rather than passing. For each (k, s) the vertex set of P n Q
(Q = k.P + s) is enumerated exactly: vertices of P in Q, vertices of Q in P, and the
crossings of an edge of either with a cut plane of the other (every vertex of P n Q
is one of these). The volume is 0 iff the vertices span fewer than d dimensions (no
vertices: empty); otherwise it is computed by exact facet fans. Sign tests run on
int64 scaled by the common denominator, with every product bounded before it is
formed; ranks, volumes and measures use ``fractions.Fraction``.

Why (G), (B), (i) and (iii) make P a fundamental region of Gamma-bar
--------------------------------------------------------------------
For d = 0, P is the point R^0 and (G) with (i) means n = 1. For d >= 1:

Covering. Gamma-bar is n cosets of Z^d and P is compact, so the images g.P are
locally finite and their union U is closed. Let S be the union of the faces of
dimension <= d - 2 of all images: closed, nowhere dense, and with connected
complement (a locally finite union of codimension-2 pieces; empty for d = 1). Take x
in U outside S, in an image T = g.P. If x is interior to T, a ball about x lies in U.
Otherwise x is in the relative interior of a facet g.F of T. The contacts of T in
g.F are g applied to those of P in F (by (G), g composed with a contact's element is
in Gamma-bar), and g, affine, scales measure on F's plane by one constant; so by
(iii) they have the measure of g.F and overlap in measure zero:
their union, closed, has full measure in g.F and is therefore all of g.F, and x lies
in an image T' meeting T in zero volume. T' lies on the far side of g.F's plane (a
point of T' on T's side would give T' interior points inside T near g.F), and x,
outside S, is in the relative interior of a facet of T' in that plane; so a ball
about x lies in T u T'. Hence U minus S is open and closed in the connected
complement of S and contains int(P) minus S: U contains the complement of S and,
being closed, is R^d.

Disjoint interiors. Let m(x) count the images containing x. Its integral over the
unit cell is the sum over the n representatives of vol(k.P) = n vol(P) = 1 (|det W| =
1, and the (k, s) index Gamma-bar without repetition), the cell's volume, while
m >= 1 everywhere by covering. So m = 1 almost everywhere: two distinct images meet
in zero volume, i.e. have disjoint interiors (two convex bodies meet in positive
volume iff their interiors meet). (ii) follows, and stays in ``ok`` as a cross-check
of the enumeration.

Why that makes ASU(Gamma+) x SO(3) a fundamental domain (K trivial)
-------------------------------------------------------------------
By (W) the free coordinates contribute one period, which the fold gauges, so the
centroid part of the domain is P. g = (W, w) in Gamma+ acts on poses by
g.(x, R) = (g-bar x, Q_g R), with Q_g = T W T^-1 in SO(3) (W is a proper isometry of
the family's generic metric: a table fact, gate 0).

1. Every orbit meets P x SO(3): by covering, some g has g-bar x in P.
2. At most one point of an orbit lies in int(P) x SO(3): if (x, R) and g.(x, R)
   both do with g != id, then g-bar != id (K is trivial) and g-bar x lies in
   int(P) n int(g-bar P), which is empty.
3. The action on poses is free: g.(x, R) = (x, R) forces Q_g = I, hence W = I
   (T is invertible); a g != id with W = I is a nonzero translation (a nonzero
   element modulo Z^3 and free shifts, or a nonzero lattice shift), and fixes no
   centroid. So no element other than the identity fixes a pose.

By 1 and 2, P x SO(3) is a fundamental domain up to the measure-zero set
dP x SO(3): Haar(dR) x Lebesgue(dx) restricted to it is the quotient measure with
multiplicity one. By 3, the points of one orbit in P x SO(3) are pairwise distinct
poses: they differ in centroid (on dP, where the cctbx cut trees keep one point of
each Gamma-bar orbit) or, over a centroid with a nontrivial stabiliser, in rotation
(the stabiliser acts on SO(3) without fixed points, so the fold's lexicographic
rule on Q_g R, a strict order on distinct matrices, keeps exactly one). Given cut
trees that keep one centroid per Gamma-bar orbit, the strict domain therefore holds
exactly one pose per orbit.

Not proved here: that the representatives are Gamma+/K of N_E(G) (gate 0); that the
cut trees keep exactly one point of each boundary orbit (gate 0 checks them against
cctbx; gate 1 checks exact special points); and the fold's behaviour within its float
tolerance band (README, Known limits). In sg 6-9, K = {1, k} with k-bar = id but
Q_k != I: the certificate holds for Gamma+/K, but every pose has a second
representative k.(x, R) over the same centroid, so the domain there is
P x (a K cut of SO(3)), not P x SO(3), and the K cut is gate 1's.

ASU(G) (``g_asu_domain``, ``g_asu_certificate``)
------------------------------------------------
The same certificate on cctbx's asymmetric unit of G under G modulo Z^3, improper
elements included, on all three coordinates (no walls): the centroid part of the
domain of every unit but the anchor at Z' > 1 (energy_sampling/docs/design/zprime_fd.md
3.4, gate G3). The pose part there, with the conformer mirror on improper elements,
is ``domains.fold_unit``'s.

Usage
-----
``python -m mxtaltools.crystal_fd.certify``             all 230 groups, one table row each
``python -m mxtaltools.crystal_fd.certify --sgs 2,14``
"""
import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

import numpy as np

from mxtaltools.crystal_fd import exact as X
from mxtaltools.crystal_fd.tables import load_raw

#: bound on every int64 intermediate; each product is bounded against it before it is formed
_INT_LIMIT = 2 ** 62


# ---------------------------------------------------------------------------
# the projected domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectedDomain:
    """ASU(Gamma+) and Gamma+/K on the non-free coordinates, exactly.

    Attributes
    ----------
    sg : int
    keep : tuple of int
        The non-free axes in order; d = len(keep).
    normals : tuple of tuple of int
        Half-spaces ``n . x + c >= 0`` over the non-free coordinates.
    offsets : tuple of Fraction
    elements : tuple of (numpy.ndarray, tuple)
        One representative per element of Gamma+/K: integer [d, d] ``W`` and ``w``,
        d Fractions in [0, 1). The identity is first.
    source : tuple of int
        Position in ``GroupTable.gamma_plus`` of each representative (in G's element
        order for ``g_asu_domain``, which holds ASU(G) and G).
    walls : tuple of (tuple of int, Fraction)
        The dropped cuts ``n . x_free + c >= 0``, over the free axes in order.
    """
    sg: int
    keep: tuple
    normals: tuple
    offsets: tuple
    elements: tuple
    source: tuple
    walls: tuple = ()

    @property
    def dim(self) -> int:
        return len(self.keep)

    @property
    def free(self) -> tuple:
        return tuple(i for i in range(3) if i not in self.keep)


def projected_domain(sg: int, entry: dict = None) -> ProjectedDomain:
    """ASU(Gamma+) and Gamma+/K of space group ``sg`` on the non-free coordinates.

    Parameters
    ----------
    sg : int
    entry : dict, optional
        A table entry (``tables.load_raw()[str(sg)]`` when omitted).

    Raises
    ------
    ValueError
        If a cut or an element mixes free and non-free coordinates.
    """
    e = load_raw()[str(sg)] if entry is None else entry
    free = np.array(e['free'], dtype=bool)
    keep = tuple(int(i) for i in np.nonzero(~free)[0])
    fr = tuple(int(i) for i in np.nonzero(free)[0])
    normals, offsets, walls = [], [], []
    for n, c in zip(*X.cut_planes(e['asu']['cuts'])):
        nk, nf = [int(n[i]) for i in keep], [int(n[i]) for i in fr]
        if any(nk) and any(nf):
            raise ValueError(f'sg {sg}: cut {n} mixes free and non-free coordinates')
        if any(nk):
            normals.append(tuple(nk))
            offsets.append(X.to_fraction(c))
        else:
            walls.append((tuple(nf), X.to_fraction(c)))
    elements, source, seen = [], [], set()
    for pos, i in enumerate(e['gamma_plus']):
        W = np.array(e['gamma']['W'][i], dtype=np.int64).reshape(3, 3)
        if W[np.ix_(keep, fr)].any() or W[np.ix_(fr, keep)].any():
            raise ValueError(f'sg {sg}: Gamma+ element {i} mixes free and non-free coordinates')
        Wk = W[np.ix_(keep, keep)]
        wk = tuple(X.to_fraction(e['gamma']['w'][i][j]) % 1 for j in keep)
        key = (tuple(Wk.ravel().tolist()), wk)
        if key in seen:
            continue
        seen.add(key)
        elements.append((Wk, wk))
        source.append(pos)
    d = len(keep)
    if not np.array_equal(elements[0][0], np.eye(d, dtype=np.int64)) or any(elements[0][1]):
        raise ValueError(f'sg {sg}: the first Gamma+ element is not the identity')
    return ProjectedDomain(sg=int(sg), keep=keep, normals=tuple(normals), offsets=tuple(offsets),
                           elements=tuple(elements), source=tuple(source), walls=tuple(walls))


# ---------------------------------------------------------------------------
# premises: the group and the walls
# ---------------------------------------------------------------------------

def _int_det(W: np.ndarray) -> np.ndarray:
    """Exact determinants of integer [n, d, d] matrices, d <= 3."""
    d = W.shape[-1]
    if d == 0:
        return np.ones(len(W), dtype=np.int64)
    if d == 1:
        return W[:, 0, 0]
    if d == 2:
        return W[:, 0, 0] * W[:, 1, 1] - W[:, 0, 1] * W[:, 1, 0]
    return (W[:, 0, 0] * (W[:, 1, 1] * W[:, 2, 2] - W[:, 1, 2] * W[:, 2, 1])
            - W[:, 0, 1] * (W[:, 1, 0] * W[:, 2, 2] - W[:, 1, 2] * W[:, 2, 0])
            + W[:, 0, 2] * (W[:, 1, 0] * W[:, 2, 1] - W[:, 1, 1] * W[:, 2, 0]))


def group_defects(dom: ProjectedDomain) -> tuple:
    """Why the representatives are not a group modulo Z^d, as (kind, detail) pairs; () when they are.

    Kinds: ``'identity'`` (the first is not the identity), ``'duplicate'`` (two are equal
    modulo Z^d), ``'closure'`` (a product of two is none of them modulo Z^d). Every one of
    the n^2 compositions is formed, exactly, on integers over the translations' common
    denominator.

    Raises
    ------
    ValueError
        If some W is not an integer matrix with |det W| = 1.
    """
    d, n = dom.dim, len(dom.elements)
    W = np.array([np.asarray(w).reshape(d, d) for w, _ in dom.elements], dtype=np.int64).reshape(n, d, d)
    det = _int_det(W)
    if (np.abs(det) != 1).any():
        k = int(np.nonzero(np.abs(det) != 1)[0][0])
        raise ValueError(f'sg {dom.sg}: representative {k} has det W = {int(det[k])}')
    L = lcm(1, *[Fraction(q).denominator for _, w in dom.elements for q in w])
    t = np.array([[int(Fraction(q) * L) % L for q in w] for _, w in dom.elements], dtype=np.int64).reshape(n, d)
    out = []
    if not np.array_equal(W[0], np.eye(d, dtype=np.int64)) or t[0].any():
        out.append(('identity', 'the first representative is not the identity'))

    # one int64 code per element modulo Z^d: W entries in [-B, B], translations in [0, L)
    B = max(1, _absmax(W))
    R = 2 * B + 1
    _bounded(R ** (d * d) * L ** d)

    def code(Wx, tx):
        c = np.zeros(Wx.shape[:-2], dtype=np.int64)
        for v in np.moveaxis(Wx.reshape(*Wx.shape[:-2], d * d) + B, -1, 0):
            c = c * R + v
        for v in np.moveaxis(tx, -1, 0):
            c = c * L + v
        return c

    codes = code(W, t)
    first = {}
    for k, c in enumerate(codes.tolist()):
        if c in first:
            out.append(('duplicate', f'representative {k} equals representative {first[c]} modulo Z^d'))
        else:
            first[c] = k
    _bounded(d * B * B, d * B * L + L)
    Wab = np.einsum('aij,bjk->abik', W, W)
    tab = np.mod(np.einsum('aij,bj->abi', W, t) + t[:, None, :], L)
    member = (np.abs(Wab) <= B).all((-1, -2))
    member &= np.isin(code(np.clip(Wab, -B, B), tab), codes)
    if not member.all():
        a, b = (int(v[0]) for v in np.nonzero(~member))
        out.append(('closure', f'{int((~member).sum())} of {n * n} products are not representatives modulo Z^d '
                               f'(first: {a} o {b})'))
    return tuple(out)


def wall_defects(dom: ProjectedDomain) -> tuple:
    """Why the dropped free-axis cuts are not height-1 walls, as ('wall', detail) pairs; () when they are.

    Height-1 walls: every dropped cut is on one free axis, and on each free axis f the
    tightest of them give exactly a_f <= x_f <= a_f + 1.
    """
    out, lo, hi = [], {}, {}
    for n, c in dom.walls:
        nz = [j for j, a in enumerate(n) if a]
        if len(nz) != 1:
            out.append(('wall', f'cut {tuple(n)} . x_free + {c} >= 0 is not on one free axis'))
            continue
        j, a = nz[0], Fraction(n[nz[0]])
        b = -Fraction(c) / a
        if a > 0:
            lo[j] = max(lo.get(j, b), b)
        else:
            hi[j] = min(hi.get(j, b), b)
    for j, ax in enumerate(dom.free):
        if j not in lo or j not in hi:
            out.append(('wall', f'free axis {ax} is not bounded on both sides'))
        elif hi[j] - lo[j] != 1:
            out.append(('wall', f'free axis {ax} has height {hi[j] - lo[j]}'))
    return tuple(out)


# ---------------------------------------------------------------------------
# exact convex geometry
# ---------------------------------------------------------------------------

def affine_rank(points) -> int:
    """Dimension of the affine hull of exact points (-1 for none)."""
    points = list(points)
    if not points:
        return -1
    return X.rank([[a - b for a, b in zip(p, points[0])] for p in points[1:]])


def _cross(o, a, b) -> Fraction:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _ring(points, axes=(0, 1)) -> list:
    """Vertices of the convex hull of exact points in the plane of ``axes``, counter-clockwise (monotone chain)."""
    i, j = axes
    pts = sorted(set(points), key=lambda p: (p[i], p[j]))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and _cross((out[-2][i], out[-2][j]), (out[-1][i], out[-1][j]), (p[i], p[j])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def convex_volume(points, normals=(), offsets=()) -> Fraction:
    """Exact d-volume (d = 1, 2, 3) of the convex hull of exact points.

    For d = 3 the facets are read from the half-spaces ``n . x + c >= 0`` that the
    points satisfy and that hold every facet of their hull (an H-representation of
    the hull); each facet is ordered by an exact planar hull and fanned from the
    vertex centroid. A hull of lower dimension has volume 0.
    """
    points = sorted(set(tuple(Fraction(v) for v in p) for p in points))
    d = len(points[0]) if points else 0
    if affine_rank(points) < d:
        return Fraction(0)
    if d == 1:
        return points[-1][0] - points[0][0]
    if d == 2:
        ring = _ring(points)
        return abs(sum(_cross(ring[0], a, b) for a, b in zip(ring[1:], ring[2:]))) / 2
    if d != 3:
        raise ValueError(f'dimension {d} is not supported')
    apex = [sum(p[i] for p in points) / len(points) for i in range(3)]
    vol, seen = Fraction(0), set()
    for n, c in zip(normals, offsets):
        face = [p for p in points if X.evaluate(n, c, p) == 0]
        key = tuple(face)
        if key in seen or affine_rank(face) != 2:
            continue
        seen.add(key)
        drop = max(range(3), key=lambda i: abs(Fraction(n[i])))
        ring = _ring(face, tuple(i for i in range(3) if i != drop))
        a = [ring[0][i] - apex[i] for i in range(3)]
        for p, q in zip(ring[1:], ring[2:]):
            vol += abs(X.det3(a, [p[i] - apex[i] for i in range(3)], [q[i] - apex[i] for i in range(3)])) / 6
    return vol


def edges(vertices, normals, offsets) -> list:
    """Index pairs of adjacent vertices: the half-spaces tight at both have rank d - 1."""
    d = len(vertices[0])
    tight = [frozenset(k for k, (n, c) in enumerate(zip(normals, offsets)) if X.evaluate(n, c, v) == 0)
             for v in vertices]
    return [(i, j) for i, j in itertools.combinations(range(len(vertices)), 2)
            if X.rank([normals[k] for k in tight[i] & tight[j]]) == d - 1]


def is_bounded(normals) -> bool:
    """True when {x : n . x + c >= 0} (normals of rank d <= 3) is bounded: no u != 0 has n . u >= 0 for all n.

    Such a u can be taken on an extreme ray of that cone, where d - 1 independent
    normals vanish: the cross product of two normals (d = 3), a normal turned by 90
    degrees (d = 2), or +-1 (d = 1).
    """
    d = len(normals[0])
    if d == 1:
        rays = [(1,)]
    elif d == 2:
        rays = [(-n[1], n[0]) for n in normals]
    else:
        rays = [(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
                for a, b in itertools.combinations(normals, 2)]
    for u in rays:
        vals = [sum(a * b for a, b in zip(n, u)) for n in normals]
        if any(u) and (min(vals) >= 0 or max(vals) <= 0):
            return False
    return True


def _clip(subject: list, clip: list) -> list:
    """Convex polygon ``subject`` n ``clip`` (counter-clockwise rings of exact 2D points), Sutherland-Hodgman."""
    out = list(subject)
    for p, q in zip(clip, clip[1:] + clip[:1]):
        pts, out = out, []
        for a, b in zip(pts, pts[1:] + pts[:1]):
            sa, sb = _cross(p, q, a), _cross(p, q, b)
            if sa >= 0:
                out.append(a)
            if sa * sb < 0:
                f = sa / (sa - sb)
                out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        if not out:
            break
    return out


@dataclass(frozen=True)
class FacetCoverage:
    """One facet F of P against the contacts lying in it (module docstring, item iii).

    Measures are (d-1)-dimensional on F's plane projected along axis ``drop`` (where
    F's normal is largest): area (d = 3), length (d = 2), or a point count (d = 1).

    Attributes
    ----------
    cut : int
        Index into ``ProjectedDomain.normals`` of a cut defining F.
    measure : Fraction
        F's measure.
    covered : Fraction
        Sum of the measures of the contacts in F.
    overlap : Fraction
        Sum over pairs of those contacts of the measure of their intersection.
    n_contacts : int
    """
    cut: int
    measure: Fraction
    covered: Fraction
    overlap: Fraction
    n_contacts: int

    @property
    def ok(self) -> bool:
        return self.covered == self.measure and self.overlap == 0


def _flat(points, e: int):
    """A convex set in R^e (e <= 2) from exact points: () for a point, (lo, hi), or a counter-clockwise ring."""
    if e == 0:
        return ()
    if e == 1:
        return min(p[0] for p in points), max(p[0] for p in points)
    return _ring(points)


def _flat_measure(s, e: int) -> Fraction:
    if e == 0:
        return Fraction(1)
    if e == 1:
        return s[1] - s[0]
    return convex_volume(s)


def _flat_overlap(a, b, e: int) -> Fraction:
    if e == 0:
        return Fraction(1)
    if e == 1:
        return max(Fraction(0), min(a[1], b[1]) - max(a[0], b[0]))
    return convex_volume(_clip(a, b)) if len(a) > 2 and len(b) > 2 else Fraction(0)


def facet_coverage(vertices, normals, offsets, faces) -> tuple:
    """Item (iii) of the certificate: each facet of P against the contacts lying in it.

    Parameters
    ----------
    vertices : list of tuple of Fraction
        P's exact vertices.
    normals, offsets : sequences
        P's half-spaces ``n . x + c >= 0``.
    faces : sequence of collections of points
        Vertex sets of the contacts: intersections with images of zero volume and
        dimension d - 1. Each lies in exactly one facet's plane.

    Returns
    -------
    facets : tuple of FacetCoverage
        One per facet of P (cuts with the same tight vertices counted once).
    stray : int
        Contacts in no facet's plane (none can be, for a contact of zero volume).
    """
    d = len(vertices[0])
    facets, seen = [], set()
    for i, (n, c) in enumerate(zip(normals, offsets)):
        tight = tuple(v for v in vertices if X.evaluate(n, c, v) == 0)
        if tight not in seen and affine_rank(tight) == d - 1:
            seen.add(tight)
            facets.append((i, n, c, tight, []))
    stray = 0
    for face in faces:
        hit = [f for f in facets if all(X.evaluate(f[1], f[2], v) == 0 for v in face)]
        if len(hit) == 1:
            hit[0][4].append(face)
        else:
            stray += 1
    out = []
    for i, n, c, tight, pieces in facets:
        drop = max(range(d), key=lambda j: abs(Fraction(n[j])))

        def flat(points):
            return _flat(sorted({tuple(p[j] for j in range(d) if j != drop) for p in points}), d - 1)

        shapes = [flat(p) for p in pieces]
        out.append(FacetCoverage(
            cut=i, measure=_flat_measure(flat(tight), d - 1),
            covered=sum((_flat_measure(s, d - 1) for s in shapes), Fraction(0)),
            overlap=sum((_flat_overlap(a, b, d - 1) for a, b in itertools.combinations(shapes, 2)), Fraction(0)),
            n_contacts=len(pieces)))
    return tuple(out), stray


# ---------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TilingCertificate:
    """Result of ``certify_tiling`` (see the module docstring).

    Attributes
    ----------
    sg, dim : int
        Space group and number of non-free coordinates.
    n_elements : int
        |Gamma+/K|, the representatives modulo Z^d.
    volume : Fraction
        vol(P) on the non-free coordinates, from P's exact vertices.
    identity_volume : Fraction
        vol(P n P) by the pairwise routine (must equal ``volume``).
    shift_min, shift_max : tuple of int
        Per non-free axis, the extreme lattice shifts over all elements at which
        the bounding boxes of P and k.P + s meet (the enumerated range).
    n_pairs : int
        (k, s) pairs examined, the identity pair included.
    contacts : tuple of int
        Pairs other than (id, 0) whose intersection is nonempty with zero volume,
        by the dimension of the intersection (0 a point, 1 a segment, 2 a polygon).
    overlaps : tuple
        (representative index, shift, volume) of every pair other than (id, 0) with
        positive volume.
    facets : tuple of FacetCoverage
        Item (iii) per facet of P (empty for d = 0).
    defects : tuple of (str, str)
        (kind, detail) for each failed premise: ``group_defects``, ``wall_defects``,
        and ``'contact'`` for contacts in no facet.
    seconds : float
    """
    sg: int
    dim: int
    n_elements: int
    volume: Fraction
    identity_volume: Fraction
    shift_min: tuple
    shift_max: tuple
    n_pairs: int
    contacts: tuple
    overlaps: tuple
    facets: tuple
    defects: tuple
    seconds: float

    @property
    def uncovered(self) -> tuple:
        """The facets failing item (iii)."""
        return tuple(f for f in self.facets if not f.ok)

    @property
    def ok(self) -> bool:
        """Items (G), (i), (ii), (iii) and (W) of the module docstring ((B) raises when it fails)."""
        return (self.volume * self.n_elements == 1 and self.identity_volume == self.volume and not self.overlaps
                and not self.defects and not self.uncovered)


def _bounded(*bounds):
    for b in bounds:
        if b >= _INT_LIMIT:
            raise OverflowError(f'an int64 intermediate could reach {b}')


def _absmax(a) -> int:
    return int(np.abs(a).max(initial=0))


def _int_inverse(W: np.ndarray) -> np.ndarray:
    d = W.shape[0]
    Wi = np.round(np.linalg.inv(W.astype(float))).astype(np.int64)
    if not np.array_equal(W @ Wi, np.eye(d, dtype=np.int64)):
        raise ValueError(f'{W.tolist()} has no integer inverse')
    return Wi


def _crossings(V, H, E, N, C):
    """Points where an edge of a polytope crosses a cut plane of another, kept when inside the other.

    V [p, v, d] vertices (scaled by L), H [p, v, m] the other's cut values at them,
    E [e, 2] edges, N [p, m, d] and C [p, m] the other's cuts (scaled). Returns
    (num [p, e, m, d], den [p, e, m], ok [p, e, m]): point num / (den L) where ok.
    """
    hi, hj = H[:, E[:, 0]], H[:, E[:, 1]]                                   # [p, e, m]
    cross = (hi > 0) & (hj < 0) | (hi < 0) & (hj > 0)
    _bounded(2 * _absmax(V) * _absmax(H))
    num = V[:, E[:, 1], None, :] * hi[..., None] - V[:, E[:, 0], None, :] * hj[..., None]
    den = hi - hj
    sgn = np.where(den < 0, -1, 1)
    num, den = num * sgn[..., None], den * sgn
    _bounded(N.shape[-1] * _absmax(N) * _absmax(num) + _absmax(C) * _absmax(den))
    inside = (np.einsum('pld,pemd->peml', N, num) + C[:, None, None, :] * den[..., None] >= 0).all(-1)
    return num, den, cross & inside


@dataclass(frozen=True)
class PairIntersection:
    """P n (k.P + s) for one representative ``element`` k and lattice ``shift`` s.

    ``normals``, ``offsets``: the half-spaces of k.P + s (exact); ``vertices``: the
    exact vertices of the intersection, empty when the two are disjoint.
    """
    element: int
    shift: tuple
    normals: tuple
    offsets: tuple
    vertices: frozenset


@dataclass(frozen=True)
class PairSweep:
    """Every (k, s) whose bounding box meets P's, with P's vertices and the enumerated shift range."""
    vertices: list
    pairs: list
    shift_min: tuple
    shift_max: tuple


def pair_intersections(dom: ProjectedDomain, shifts=None) -> PairSweep:
    """The exact vertices of P n (k.P + s) for every representative k and every s whose boxes meet (d >= 1).

    s runs, per element and axis, over the integers at which the bounding boxes of
    P and k.P + s meet; outside that range the two are disjoint. Vertices of the
    intersection are P's vertices in Q, Q's vertices in P and the crossings of an
    edge of either with a cut plane of the other (module docstring).

    Parameters
    ----------
    dom : ProjectedDomain
    shifts : collection of tuple of int, optional
        Examine only these shifts within the range (a control: the certificate's
        proof does not need the range complete). ``shift_min`` / ``shift_max`` still
        report the full range.

    Raises
    ------
    ValueError
        If P is not a bounded d-dimensional polytope (item B).
    """
    d = dom.dim
    if d == 0:
        raise ValueError('pair_intersections needs at least one non-free coordinate')
    normals, offsets = [list(v) for v in dom.normals], list(dom.offsets)
    verts = X.polytope_vertices(normals, offsets)
    if affine_rank(verts) != d or not is_bounded(normals):
        raise ValueError(f'sg {dom.sg}: the domain is not a bounded {d}-dimensional polytope')
    shifts = None if shifts is None else {tuple(int(v) for v in s) for s in shifts}
    E = np.array(edges(verts, normals, offsets), dtype=np.int64).reshape(-1, 2)

    # integers over the common denominator L
    L = lcm(*[q.denominator for p in verts for q in p], *[Fraction(c).denominator for c in offsets],
            *[q.denominator for _, w in dom.elements for q in w])
    Vp = np.array([[int(q * L) for q in p] for p in verts], dtype=np.int64)              # [v, d]
    Np = np.array(normals, dtype=np.int64)                                               # [m, d]
    Cp = np.array([int(Fraction(c) * L) for c in offsets], dtype=np.int64)               # [m]
    lo, hi = Vp.min(0), Vp.max(0)

    # every (k, s) whose bounding boxes meet P's
    Ws, offs, pairs = [], [], []
    smin = [None] * d
    smax = [None] * d
    for k, (W, w) in enumerate(dom.elements):
        wL = np.array([int(q * L) for q in w], dtype=np.int64)
        img = Vp @ W.T + wL
        s_lo = -((img.max(0) - lo) // L)                       # ceil((lo - max img) / L)
        s_hi = (hi - img.min(0)) // L
        for j in range(d):
            smin[j] = int(s_lo[j]) if smin[j] is None else min(smin[j], int(s_lo[j]))
            smax[j] = int(s_hi[j]) if smax[j] is None else max(smax[j], int(s_hi[j]))
        for s in itertools.product(*[range(int(a), int(b) + 1) for a, b in zip(s_lo, s_hi)]):
            if shifts is not None and s not in shifts:
                continue
            Ws.append(W)
            offs.append(wL + L * np.array(s, dtype=np.int64))
            pairs.append((k, s))
    W = np.stack(Ws)                                                                     # [p, d, d]
    off = np.stack(offs)                                                                 # [p, d]
    Minv = np.stack([_int_inverse(w) for w in Ws])

    # Q = k.P + s = {x : (M^T n) . x + c - (M^T n) . (w + s) >= 0}, M = W^-1
    _bounded(d * _absmax(W) * _absmax(Vp) + _absmax(off), d * _absmax(Minv) * _absmax(Np))
    Vq = np.einsum('pij,vj->pvi', W, Vp) + off[:, None, :]                              # [p, v, d]
    Nq = np.einsum('pji,mj->pmi', Minv, Np)                                              # [p, m, d]
    _bounded(d * _absmax(Nq) * _absmax(off) + _absmax(Cp))
    Cq = Cp[None] - np.einsum('pmi,pi->pm', Nq, off)                                     # [p, m]
    _bounded(d * _absmax(Np) * _absmax(Vq) + _absmax(Cp), d * _absmax(Nq) * _absmax(Vp) + _absmax(Cq))
    Hp_at_q = np.einsum('pvi,mi->pvm', Vq, Np) + Cp                                      # P's cuts at Q's vertices
    Hq_at_p = np.einsum('vi,pmi->pvm', Vp, Nq) + Cq[:, None, :]                          # Q's cuts at P's vertices
    a_ok = (Hq_at_p >= 0).all(-1)                                                        # P's vertices in Q
    b_ok = (Hp_at_q >= 0).all(-1)                                                        # Q's vertices in P
    P_cuts = (np.broadcast_to(Np, (len(pairs),) + Np.shape), np.broadcast_to(Cp, (len(pairs),) + Cp.shape))
    c_num, c_den, c_ok = _crossings(np.broadcast_to(Vp, (len(pairs),) + Vp.shape), Hq_at_p, E, Nq, Cq)
    d_num, d_den, d_ok = _crossings(Vq, Hp_at_q, E, *P_cuts)
    out = []
    for p, (k, s) in enumerate(pairs):
        pts = {tuple(Fraction(int(q), L) for q in Vp[v]) for v in np.nonzero(a_ok[p])[0]}
        pts |= {tuple(Fraction(int(q), L) for q in Vq[p, v]) for v in np.nonzero(b_ok[p])[0]}
        for num, den, ok in ((c_num, c_den, c_ok), (d_num, d_den, d_ok)):
            for e, m in zip(*np.nonzero(ok[p])):
                pts.add(tuple(Fraction(int(q), int(den[p, e, m]) * L) for q in num[p, e, m]))
        out.append(PairIntersection(element=k, shift=s, normals=tuple(tuple(int(v) for v in r) for r in Nq[p]),
                                    offsets=tuple(Fraction(int(c), L) for c in Cq[p]), vertices=frozenset(pts)))
    return PairSweep(vertices=verts, pairs=out, shift_min=tuple(smin), shift_max=tuple(smax))


def certify_tiling(dom: ProjectedDomain, shifts=None) -> TilingCertificate:
    """The exact tiling certificate of ``dom`` (module docstring).

    ``shifts`` restricts the examined lattice shifts (``pair_intersections``), for controls.
    """
    t0 = time.perf_counter()
    d, n = dom.dim, len(dom.elements)
    defects = group_defects(dom) + wall_defects(dom)
    if d == 0:
        # R^0 is one point: every representative maps P onto P, and P has no facets
        return TilingCertificate(sg=dom.sg, dim=0, n_elements=n, volume=Fraction(1), identity_volume=Fraction(1),
                                 shift_min=(), shift_max=(), n_pairs=n, contacts=(),
                                 overlaps=tuple((k, (), Fraction(1)) for k in range(1, n)), facets=(),
                                 defects=defects, seconds=time.perf_counter() - t0)
    normals, offsets = [list(v) for v in dom.normals], list(dom.offsets)
    sweep = pair_intersections(dom, shifts)
    volume = convex_volume(sweep.vertices, normals, offsets)
    contacts = [0] * d
    overlaps, faces = [], []
    identity_volume = Fraction(0)
    for pi in sweep.pairs:
        if not pi.vertices:
            continue
        r = affine_rank(pi.vertices)
        vol = Fraction(0) if r < d else convex_volume(pi.vertices, normals + [list(v) for v in pi.normals],
                                                      offsets + list(pi.offsets))
        if pi.element == 0 and not any(pi.shift):
            identity_volume = vol
        elif vol > 0:
            overlaps.append((pi.element, pi.shift, vol))
        else:
            contacts[r] += 1
            if r == d - 1:
                faces.append(pi.vertices)
    facets, stray = facet_coverage(sweep.vertices, normals, offsets, faces)
    if stray:
        defects += (('contact', f'{stray} contacts of dimension d - 1 lie in no facet of P'),)
    return TilingCertificate(sg=dom.sg, dim=d, n_elements=n, volume=volume, identity_volume=identity_volume,
                             shift_min=sweep.shift_min, shift_max=sweep.shift_max, n_pairs=len(sweep.pairs),
                             contacts=tuple(contacts), overlaps=tuple(overlaps), facets=facets, defects=defects,
                             seconds=time.perf_counter() - t0)


def tiling_certificate(sg: int) -> TilingCertificate:
    """``certify_tiling`` of the table's ASU(Gamma+) and Gamma+/K for space group ``sg``."""
    return certify_tiling(projected_domain(sg))


def g_asu_domain(sg: int, entry: dict = None) -> ProjectedDomain:
    """cctbx's ASU(G) and G modulo Z^3 of space group ``sg``, exactly (zprime_fd.md 3.4, gate G3).

    All three coordinates are kept (d = 3, no walls). The representatives are G's
    elements, improper ones included, in SYM_OPS order with w modulo 1, so ``source``
    indexes ``GroupTable.g_W``. They are not deduplicated: ``group_defects`` reports a
    repeated element. ``certify_tiling`` of this domain proves that ASU(G) tiles R^3
    under G, the centroid part of the domain ASU(G) x SO(3) x Q of every unit but the
    anchor at Z' > 1.

    Parameters
    ----------
    sg : int
    entry : dict, optional
        A table entry (``tables.load_raw()[str(sg)]`` when omitted).
    """
    e = load_raw()[str(sg)] if entry is None else entry
    normals, offsets = X.cut_planes(e['g_asu']['cuts'])
    elements = tuple((np.array(W, dtype=np.int64).reshape(3, 3), tuple(X.to_fraction(q) % 1 for q in w))
                     for W, w in zip(e['G']['W'], e['G']['w']))
    return ProjectedDomain(sg=int(sg), keep=(0, 1, 2), normals=tuple(tuple(int(v) for v in n) for n in normals),
                           offsets=tuple(offsets), elements=elements, source=tuple(range(len(elements))))


def g_asu_certificate(sg: int) -> TilingCertificate:
    """``certify_tiling`` of cctbx's ASU(G) under G for space group ``sg``."""
    return certify_tiling(g_asu_domain(sg))


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

CAPTION = ('Exact tiling certificate of ASU(Gamma+) on the non-free coordinates, one row per space group, from '
           'data/fd_tables_v1.json in exact rationals (no sampling). n = |Gamma+/K| representatives modulo the '
           'lattice, checked to be a group (closure and distinctness over all n^2 products); pairs = (element, '
           'lattice shift) whose bounding boxes meet P (all others are disjoint); contacts = pairs meeting P in a '
           'polygon / segment / point (zero volume); facets covered = facets of P whose contacts of dimension d - 1 '
           'sum exactly to the facet\'s measure without overlapping; defects = failed premises (group, free-axis '
           'walls). Reference: a tiling has all facets covered, 0 overlaps, 0 defects and vol x n = 1 exactly; '
           'sg 6-9 (|K| = 2) tile by Gamma+/K only.')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sgs', default='1-230', help="'1-230' or a comma list")
    args = ap.parse_args(argv)
    if '-' in args.sgs:
        a, b = (int(v) for v in args.sgs.split('-'))
        sgs = list(range(a, b + 1))
    else:
        sgs = [int(v) for v in args.sgs.split(',')]
    raw = load_raw()
    print(CAPTION)
    print(f"{'sg':>3} {'symbol':10} {'d (axes)':>8} {'|K|':>3} {'n (elements)':>12} {'shift range (per axis)':>24} "
          f"{'pairs (count)':>13} {'contacts polygon/segment/point (count)':>38} {'facets covered (count)':>22} "
          f"{'overlaps (count)':>16} {'defects (count)':>15} {'vol x n (exact)':>15} {'time (s)':>8}")
    bad, t0 = [], time.perf_counter()
    for sg in sgs:
        c = tiling_certificate(sg)
        if not c.ok:
            bad.append(sg)
        rng = ' '.join(f'{a}..{b}' for a, b in zip(c.shift_min, c.shift_max)) or '-'
        con = '/'.join(str(c.contacts[i]) if i < c.dim else '-' for i in (2, 1, 0))
        cov = f'{len(c.facets) - len(c.uncovered)}/{len(c.facets)}'
        print(f"{sg:3d} {raw[str(sg)]['symbol']:10} {c.dim:8d} {len(raw[str(sg)]['kernel']):3d} {c.n_elements:12d} "
              f"{rng:>24} {c.n_pairs:13d} {con:>38} {cov:>22} {len(c.overlaps):16d} {len(c.defects):15d} "
              f"{str(c.volume * c.n_elements):>15} {c.seconds:8.2f}", flush=True)
        for kind, detail in c.defects:
            print(f'    {kind}: {detail}')
    print(f'{len(sgs)} groups in {time.perf_counter() - t0:.1f} s; failing: {bad or "none"}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
