"""GATE 0, tiling certificate: ASU(Gamma+) tiles the centroid space under Gamma+, proved exactly.

``certify.certify_tiling`` checks, in exact rationals from the table alone, that the
representatives of Gamma+/K are a group modulo Z^d (closure, distinctness), that
vol(P) x |Gamma+/K| = 1, that every facet of P is covered exactly, without overlap,
by its face-to-face contacts with images, that no examined image overlaps P, and
that the dropped free-axis cuts are height-1 walls. Together these prove that the
images of P cover the non-free coordinates with disjoint interiors, and, with K
trivial and the free action of Gamma+ on poses, that ASU(Gamma+) x SO(3) is a
fundamental domain (derivation: the ``certify`` module docstring). It turns gate 1's
sampled orbit counts into a proof for the centroid part; gate 1 still covers the
cut trees' boundary choices and the orientation rule.

- every group but sg 6-9: the certificate holds with |Gamma+/K| = |Gamma+|;
- sg 6-9: it holds for Gamma+/K, and K's other element turns poses (the K cut);
- the fast vertex enumeration equals brute-force enumeration over every subset of
  d cut planes, on every examined pair of every group;
- the enumerated shifts: a brute force one shift beyond the range on every side
  finds no other intersecting pair, and the range is re-derived in Fractions;
- mutation controls: a translated domain (same volume) overlaps, except where
  Gamma+ is pure translations (sg 2); a pushed-out facet overlaps; a pulled-in
  facet loses volume; the G asymmetric unit overlaps (in sg 227 at equal volume);
- premise controls, each invisible to the volume and overlap items: a duplicated
  element, a translation nudged off the group, a wrong domain examined at shift 0
  only (rejected by facet coverage), a free-axis wall of the wrong height.

The certificate sweep, the shift-range re-derivation and the premise controls over
their named groups take a few seconds and run in the fast tier. The brute-force
cross-checks outside ``FAST_SG`` and ``BOUND_SG`` and the grid of wrong domains are
marked ``slow`` (about 4 min in all).
"""
import copy
import itertools
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from mxtaltools.crystal_fd import certify as C
from mxtaltools.crystal_fd import exact as X
from mxtaltools.crystal_fd.tables import load_raw

K_GROUPS = (6, 7, 8, 9)
SUPPORTED = [sg for sg in range(1, 231) if sg not in K_GROUPS]
LARGEST = tuple(sg for sg in SUPPORTED if len(load_raw()[str(sg)]['gamma_plus']) == 384)
#: brute-force lane: the named representatives, the largest Gamma+ (384), free-axis groups (d = 2: 4, 75, 143, 168)
#: and groups whose neighbours sit two cells away (88, 148, 152)
FAST_SG = tuple(sorted({1, 2, 4, 14, 19, 75, 88, 92, 143, 148, 152, 168, 198, 227} | set(LARGEST)))
#: the shift-bound brute force costs about n x (range + 2)^d x C(2m, d) exact solves; kept to small groups
BOUND_SG = (2, 4, 75, 143)
BOUND_SG_SLOW = (14, 88, 148, 152)


def _params(sgs, fast):
    return [sg if sg in fast else pytest.param(sg, marks=pytest.mark.slow) for sg in sgs]


def _entry(sg):
    return load_raw()[str(sg)]


def _image_cuts(dom, W, w, s):
    """Half-spaces of k.P + s in Fractions, written independently of ``certify``."""
    Mi = np.round(np.linalg.inv(W.astype(float))).astype(np.int64)
    assert np.array_equal(W @ Mi, np.eye(len(W), dtype=np.int64))
    normals = [[int(v) for v in Mi.T @ np.array(n, dtype=np.int64)] for n in dom.normals]
    offsets = [Fraction(c) - sum(Fraction(a) * (q + si) for a, q, si in zip(n, w, s))
               for n, c in zip(normals, dom.offsets)]
    return normals, offsets


def _brute_vertices(dom, normals, offsets):
    return X.polytope_vertices([list(n) for n in dom.normals] + normals, list(dom.offsets) + offsets)


def _kinds(cert):
    return {kind for kind, _ in cert.defects}


def _volume_and_overlaps_pass(cert):
    """The items the certificate had before its premises were checked: vol x n = 1 and no examined overlap."""
    return cert.volume * cert.n_elements == 1 and cert.identity_volume == cert.volume and not cert.overlaps


# ---------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', SUPPORTED)
def test_tiling_certificate(sg):
    """vol(P) x |Gamma+| = 1 and no (k, s) other than (identity, 0) overlaps P in positive volume, exactly."""
    e = _entry(sg)
    cert = C.tiling_certificate(sg)
    assert len(e['kernel']) == 1
    assert cert.n_elements == len(e['gamma_plus'])
    assert cert.dim == 3 - sum(e['free'])
    # the projected volume is the table's prism volume (height 1 along free axes)
    assert cert.volume == X.to_fraction(e['asu']['volume'])
    assert cert.volume * cert.n_elements == 1
    assert cert.identity_volume == cert.volume
    assert cert.overlaps == ()
    assert cert.defects == ()
    assert cert.ok
    if cert.dim:
        assert cert.n_pairs >= 1 + sum(cert.contacts)
        assert sum(cert.contacts) >= cert.dim + 1, 'a tile of R^d has at least d + 1 neighbours'
        assert len(cert.facets) >= cert.dim + 1 and not cert.uncovered
        assert sum(f.n_contacts for f in cert.facets) == cert.contacts[cert.dim - 1]


@pytest.mark.parametrize('sg', K_GROUPS)
def test_kernel_groups_tile_by_the_quotient(sg):
    """In sg 6-9 the certificate holds for Gamma+/K; K's other element has W != I, so P x SO(3) holds two poses."""
    e = _entry(sg)
    cert = C.tiling_certificate(sg)
    assert cert.ok and cert.n_elements * 2 == len(e['gamma_plus'])
    (k,) = [i for i in e['kernel'] if i != e['gamma_plus'][0]]
    assert not np.array_equal(np.array(e['gamma']['W'][k]).reshape(3, 3), np.eye(3, dtype=int))


def test_trivial_and_guard_cases():
    """P1 has no non-free coordinate (one element); a second element there overlaps; int64 bounds refuse overflow."""
    dom = C.projected_domain(1)
    assert dom.dim == 0 and C.certify_tiling(dom).ok
    two = C.certify_tiling(replace(dom, elements=dom.elements * 2))
    assert not two.ok and _kinds(two) == {'duplicate'}
    with pytest.raises(OverflowError):
        C._bounded(2 ** 62)


def test_convex_volume_on_known_shapes():
    """Exact volumes of a unit cube, a corner simplex, a triangle and a segment; flat point sets have volume 0."""
    cube = list(itertools.product((0, 1), repeat=3))
    planes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    assert C.convex_volume(cube, planes, [0, 1, 0, 1, 0, 1]) == 1
    simplex = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert C.convex_volume(simplex, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], [0, 0, 0, 1]) == Fraction(1, 6)
    assert C.convex_volume([(0, 0), (Fraction(1, 2), 0), (0, Fraction(1, 3)), (Fraction(1, 6), Fraction(1, 9))]) \
        == Fraction(1, 12)
    assert C.convex_volume([(Fraction(1, 3),), (Fraction(-1, 4),)]) == Fraction(7, 12)
    assert C.convex_volume([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], planes, [0, 1, 0, 1, 0, 1]) == 0


# ---------------------------------------------------------------------------
# cross-checks of the enumeration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', _params([sg for sg in range(2, 231)], FAST_SG))
def test_vertex_enumeration_matches_brute_force(sg):
    """Every examined pair's vertex set equals brute-force enumeration over all d-subsets of both polytopes' cuts."""
    dom = C.projected_domain(sg)
    if dom.dim == 0:
        pytest.skip('no non-free coordinate')
    sweep = C.pair_intersections(dom)
    assert any(p.element == 0 and not any(p.shift) for p in sweep.pairs)
    for p in sweep.pairs:
        W, w = dom.elements[p.element]
        normals, offsets = _image_cuts(dom, W, w, p.shift)
        assert [list(n) for n in p.normals] == normals and list(p.offsets) == offsets
        assert sorted(p.vertices) == _brute_vertices(dom, normals, offsets), (p.element, p.shift)


@pytest.mark.parametrize('sg', SUPPORTED)
def test_shift_range_rederived(sg):
    """Per element, the enumerated shifts are exactly the integer s at which the boxes of P and k.P + s meet."""
    dom = C.projected_domain(sg)
    if dom.dim == 0:
        pytest.skip('no non-free coordinate')
    sweep = C.pair_intersections(dom)
    verts = X.polytope_vertices([list(n) for n in dom.normals], list(dom.offsets))
    lo = [min(v[j] for v in verts) for j in range(dom.dim)]
    hi = [max(v[j] for v in verts) for j in range(dom.dim)]
    window = [range(a - 2, b + 3) for a, b in zip(sweep.shift_min, sweep.shift_max)]
    listed = {(p.element, p.shift) for p in sweep.pairs}
    for k, (W, w) in enumerate(dom.elements):
        img = [[sum(int(W[i, j]) * v[j] for j in range(dom.dim)) + w[i] for i in range(dom.dim)] for v in verts]
        ilo = [min(p[j] for p in img) for j in range(dom.dim)]
        ihi = [max(p[j] for p in img) for j in range(dom.dim)]
        want = {s for s in itertools.product(*window)
                if all(ilo[j] + s[j] <= hi[j] and ihi[j] + s[j] >= lo[j] for j in range(dom.dim))}
        assert want == {s for e, s in listed if e == k}, k


@pytest.mark.parametrize('sg', list(BOUND_SG) + [pytest.param(sg, marks=pytest.mark.slow) for sg in BOUND_SG_SLOW])
def test_no_contact_beyond_the_shift_range(sg):
    """Brute force over every element and every shift up to one beyond the range finds exactly the meeting pairs."""
    dom = C.projected_domain(sg)
    sweep = C.pair_intersections(dom)
    meet = {(p.element, p.shift) for p in sweep.pairs if p.vertices}
    found = set()
    window = [range(a - 1, b + 2) for a, b in zip(sweep.shift_min, sweep.shift_max)]
    for k, (W, w) in enumerate(dom.elements):
        for s in itertools.product(*window):
            if _brute_vertices(dom, *_image_cuts(dom, W, w, s)):
                found.add((k, s))
    assert found == meet


# ---------------------------------------------------------------------------
# mutation controls: the certificate fails on wrong domains
# ---------------------------------------------------------------------------

MUTATION_SG = (2, 14, 75, 92, 143, 148, 198, 227)


def _moved(dom, delta):
    """P + delta: offsets c - n . delta."""
    return replace(dom, offsets=tuple(c - sum(Fraction(a) * b for a, b in zip(n, delta))
                                      for n, c in zip(dom.normals, dom.offsets)))


def _facet(dom):
    verts = X.polytope_vertices([list(n) for n in dom.normals], list(dom.offsets))
    return next(i for i, (n, c) in enumerate(zip(dom.normals, dom.offsets)) if X.is_facet(n, c, verts, dom.dim))


@pytest.mark.parametrize('sg', MUTATION_SG)
def test_translated_domain_overlaps(sg):
    """P moved by a small rational vector keeps vol x n = 1 exactly, so only the overlap test can reject it; it does.

    Where Gamma+ is pure translations (sg 2), a translate of a fundamental domain is
    one, and the certificate accepts it.
    """
    dom = C.projected_domain(sg)
    cert = C.certify_tiling(_moved(dom, [Fraction(k, 97) for k in (1, 2, 3)][:dom.dim]))
    assert cert.volume * cert.n_elements == 1
    if all(np.array_equal(W, np.eye(dom.dim, dtype=np.int64)) for W, _ in dom.elements):
        assert cert.ok
        return
    assert cert.overlaps and all(v > 0 for _, _, v in cert.overlaps)
    assert not cert.ok


@pytest.mark.parametrize('sg', MUTATION_SG)
def test_moved_facet_is_rejected(sg):
    """A facet pushed out by 1/97 overlaps a neighbour; pulled in by 1/97 it overlaps nothing but loses volume."""
    dom = C.projected_domain(sg)
    i = _facet(dom)
    for sign in (1, -1):
        off = list(dom.offsets)
        off[i] += sign * Fraction(1, 97)
        cert = C.certify_tiling(replace(dom, offsets=tuple(off)))
        assert not cert.ok
        if sign > 0:
            assert cert.overlaps and cert.volume * cert.n_elements > 1
        else:
            assert not cert.overlaps and cert.volume * cert.n_elements < 1


@pytest.mark.parametrize('sg', (14, 92, 198, 227))
def test_space_group_asu_is_rejected(sg):
    """The asymmetric unit of G overlaps its Gamma+ images; in sg 227 (|Gamma+| = |G|) only the overlap test sees it."""
    e = dict(_entry(sg))
    e['asu'] = e['g_asu']
    cert = C.certify_tiling(C.projected_domain(sg, e))
    assert cert.volume * cert.n_elements == Fraction(len(e['gamma_plus']), len(e['G']['W']))
    assert cert.overlaps and not cert.ok


# ---------------------------------------------------------------------------
# premise controls: wrong inputs the volume and overlap items cannot see
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', (2, 14, 92, 198, 227))
def test_duplicated_element_is_rejected(sg):
    """A non-identity representative replaced by a copy of another (n unchanged) is refused as a duplicate.

    vol x n and every examined overlap still pass: without the group check these lists were certified.
    """
    dom = C.projected_domain(sg)
    n = len(dom.elements)
    for j in sorted({1, n // 2, n - 1}):
        els = list(dom.elements)
        els[j] = els[n - 1 if j == 1 else 1]
        cert = C.certify_tiling(replace(dom, elements=tuple(els)))
        assert _volume_and_overlaps_pass(cert), j
        assert not cert.ok and 'duplicate' in _kinds(cert), j


@pytest.mark.parametrize('sg', (2, 14, 92, 198))
def test_nudged_translation_is_rejected(sg):
    """Every non-identity representative's translation moved by 1/24 on each axis (a non-member) breaks closure.

    Most of these lists pass vol x n and every examined overlap, so only the group check rejects them.
    """
    dom = C.projected_domain(sg)
    members = {(tuple(W.ravel().tolist()), w) for W, w in dom.elements}
    tried, blind = 0, 0
    for j, ax in itertools.product(range(1, len(dom.elements)), range(dom.dim)):
        W, w = dom.elements[j]
        w2 = tuple((q + Fraction(1, 24)) % 1 if i == ax else q for i, q in enumerate(w))
        if (tuple(W.ravel().tolist()), w2) in members:
            continue
        els = list(dom.elements)
        els[j] = (W, w2)
        cert = C.certify_tiling(replace(dom, elements=tuple(els)))
        assert not cert.ok and 'closure' in _kinds(cert), (j, ax)
        tried += 1
        blind += _volume_and_overlaps_pass(cert)
    assert tried and blind


def test_restricted_shifts_on_the_true_domains():
    """Examining shift 0 only, the volume and overlap items pass in every group; facet coverage then fails wherever a
    facet's neighbour sits at another shift, and holds (the certificate is still a proof) where none does."""
    failing = []
    for sg in range(2, 231):
        dom = C.projected_domain(sg)
        cert = C.certify_tiling(dom, shifts={(0,) * dom.dim})
        assert _volume_and_overlaps_pass(cert) and cert.defects == (), sg
        assert cert.ok == (not cert.uncovered), sg
        if cert.uncovered:
            failing.append(sg)
    assert 92 in failing and 152 in failing and len(failing) < 229


#: equal-volume translates of P that do not tile, yet pass vol x n and every overlap when only shift 0 is examined
WRONG_AT_SHIFT_0 = {
    92: [(Fraction(7, 24), Fraction(7, 24), Fraction(-5, 24)), (Fraction(3, 8), Fraction(0), Fraction(-1, 6)),
         (Fraction(-1, 2), Fraction(-1, 2), Fraction(-5, 12))],
    152: [(Fraction(5, 24), Fraction(-7, 24), Fraction(1, 12)), (Fraction(1, 24), Fraction(1, 24), Fraction(1, 24)),
          (Fraction(-11, 24), Fraction(1, 8), Fraction(1, 24)), (Fraction(-1, 2), Fraction(1, 12), Fraction(-1, 2))],
}


@pytest.mark.parametrize('sg', sorted(WRONG_AT_SHIFT_0))
def test_restricted_shifts_reject_a_wrong_domain(sg):
    """A wrong domain examined at shift 0 only passes vol x n and the overlaps; facet coverage rejects it anyway."""
    dom = C.projected_domain(sg)
    for delta in WRONG_AT_SHIFT_0[sg]:
        moved = _moved(dom, delta)
        cert = C.certify_tiling(moved, shifts={(0,) * dom.dim})
        assert _volume_and_overlaps_pass(cert) and cert.defects == (), delta
        assert cert.uncovered and not cert.ok, delta
        assert not C.certify_tiling(moved).ok, delta


@pytest.mark.slow
@pytest.mark.parametrize('sg', (92, 152))
def test_restricted_shifts_never_certify_a_wrong_domain(sg):
    """Over every translate of P on the 1/12 grid in [-1/2, 1/2]^3: a certificate at shift 0 only implies the full one,
    and the full certificate agrees with vol x n and the overlaps over the complete range."""
    dom = C.projected_domain(sg)
    blind = 0
    for delta in itertools.product([Fraction(k, 12) for k in range(-6, 7)], repeat=dom.dim):
        moved = _moved(dom, delta)
        cert = C.certify_tiling(moved, shifts={(0,) * dom.dim})
        full = C.certify_tiling(moved)
        assert full.ok == _volume_and_overlaps_pass(full), delta
        if cert.ok:
            assert full.ok, delta
        blind += _volume_and_overlaps_pass(cert) and not full.ok
    assert blind


def _wall_entry(sg, c=None, tilt=False):
    """``sg``'s entry with its first upper free-axis wall (-x_f + 1 >= 0) given offset ``c``, or tilted onto the
    next free axis."""
    e = copy.deepcopy(_entry(sg))
    free = [i for i in range(3) if e['free'][i]]
    f = free[0]
    cut = next(cut for cut in e['asu']['cuts'] if cut['n'][f] < 0 and not any(cut['n'][i] for i in range(3) if i != f))
    if c is not None:
        cut['c'] = c
    if tilt:
        cut['n'] = list(cut['n'])
        cut['n'][free[1]] = -1
    return e


@pytest.mark.parametrize('sg, c, tilt', [(1, '1/2', False), (4, '1/2', False), (4, '2', False), (75, '2', False),
                                         (143, '1/2', False), (6, '1/2', False), (6, None, True), (1, None, True)])
def test_wrong_wall_is_rejected(sg, c, tilt):
    """A free-axis wall of the wrong height, or one tilted onto two free axes, is refused; nothing else changes."""
    cert = C.certify_tiling(C.projected_domain(sg, _wall_entry(sg, c, tilt)))
    assert _volume_and_overlaps_pass(cert) and not cert.uncovered
    assert not cert.ok and _kinds(cert) == {'wall'}


def test_facet_coverage_needs_disjoint_contacts():
    """Contacts whose measures sum to a facet's still fail when two overlap (leaving a gap), in d = 1, 2 and 3."""
    h, q, z, one = Fraction(1, 2), Fraction(1, 4), Fraction(0), Fraction(1)

    def facet(vertices, normals, offsets, cut, faces):
        facets, stray = C.facet_coverage(vertices, normals, offsets, faces)
        assert stray == 0
        (f,) = [f for f in facets if f.cut == cut]
        return f

    seg = [(z,), (one,)]
    f = facet(seg, [[1], [-1]], [0, 1], 0, [{(z,)}])
    assert f.ok and f.measure == 1
    f = facet(seg, [[1], [-1]], [0, 1], 0, [{(z,)}, {(z,)}])
    assert not f.ok and f.overlap == 1

    sq = list(itertools.product((z, one), repeat=2))
    planes, offs = [[1, 0], [-1, 0], [0, 1], [0, -1]], [0, 1, 0, 1]
    assert facet(sq, planes, offs, 2, [{(z, z), (h, z)}, {(h, z), (one, z)}]).ok
    f = facet(sq, planes, offs, 2, [{(z, z), (h, z)}, {(q, z), (q + h, z)}])
    assert f.covered == f.measure == 1 and f.overlap == q and not f.ok

    cube = list(itertools.product((z, one), repeat=3))
    planes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    offs = [0, 1, 0, 1, 0, 1]
    tri = [{(z, z, z), (one, z, z), (z, one, z)}, {(one, z, z), (one, one, z), (z, one, z)}]
    assert facet(cube, planes, offs, 4, tri).ok
    strips = [{(a, b, z) for a in (z, h) for b in (z, one)}, {(a, b, z) for a in (q, q + h) for b in (z, one)}]
    f = facet(cube, planes, offs, 4, strips)
    assert f.covered == f.measure == 1 and f.overlap == q and not f.ok


def test_unbounded_and_non_unimodular_inputs_are_refused():
    """An unbounded domain whose vertices span R^d, and a representative with |det W| != 1, raise."""
    assert C.is_bounded([[1, 0], [-1, 0], [0, 1], [0, -1]])
    assert not C.is_bounded([[0, 1], [1, 0], [1, 1], [2, -1]])         # x, y >= 0, x + y >= 1, y <= 2x + 2
    assert C.is_bounded([[1], [-1]]) and not C.is_bounded([[1], [2]])
    assert not C.is_bounded([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]])
    ident = (np.eye(2, dtype=np.int64), (Fraction(0), Fraction(0)))
    dom = C.ProjectedDomain(sg=0, keep=(0, 1), normals=((0, 1), (1, 0), (1, 1), (2, -1)),
                            offsets=(Fraction(0), Fraction(0), Fraction(-1), Fraction(2)), elements=(ident,),
                            source=(0,), walls=(((1,), Fraction(0)), ((-1,), Fraction(1))))
    assert C.affine_rank(X.polytope_vertices([list(n) for n in dom.normals], list(dom.offsets))) == 2
    with pytest.raises(ValueError, match='bounded'):
        C.certify_tiling(dom)
    sq = replace(dom, normals=((1, 0), (-1, 0), (0, 1), (0, -1)), offsets=tuple(Fraction(v) for v in (0, 1, 0, 1)))
    assert C.certify_tiling(sq).ok
    with pytest.raises(ValueError, match='det'):
        C.certify_tiling(replace(sq, elements=(ident, (np.array([[2, 0], [0, 1]]), ident[1]))))
