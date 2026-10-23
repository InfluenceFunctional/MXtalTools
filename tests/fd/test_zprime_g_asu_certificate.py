"""GATE G3 of the Z' > 1 design (energy_sampling/docs/design/zprime_fd.md section 8): ASU(G) tiles under G, exactly.

``certify.g_asu_certificate`` runs ``certify_tiling`` on cctbx's asymmetric unit of G (``GroupTable.g_asu``) under
G modulo Z^3, improper elements included, with all three coordinates kept (no free axes, no walls). In exact
rationals from the table alone: G is a group modulo Z^3 (identity first, no duplicates, closure over all |G|^2
products, |det W| = 1); vol(ASU(G)) x |G| = 1, the volume from the polytope's own vertices; no examined image
g.P + s other than (identity, 0) overlaps P; every facet is covered exactly, without overlap, by its contacts. This
proves the centroid part of each non-anchor unit's domain ASU(G) x SO(3) x Q (zprime_fd.md 3.4). The fold's tie
rules on the boundary and the pose rule at special positions are stage Z2's (``test_zprime_unit_fold.py``).

- all 230 groups: the certificate holds, n = |G| = len(SYM_OPS[sg]), and the volume equals the table's;
- controls: G with its improper elements dropped (G+, non-Sohncke groups) leaves half the volume uncovered; the
  anchor's ASU(Gamma+) under G covers less than the cell; a facet pushed out overlaps and pulled in loses volume; a
  duplicated element is refused by the group check.
"""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from mxtaltools.constants.space_group_info import SYM_OPS
from mxtaltools.crystal_fd import certify as C
from mxtaltools.crystal_fd import exact as X
from mxtaltools.crystal_fd.tables import load_raw


def _entry(sg):
    return load_raw()[str(sg)]


@pytest.mark.parametrize('sg', range(1, 231))
def test_g_asu_certificate(sg):
    """vol(ASU(G)) x |G| = 1, G a group modulo Z^3, no overlaps and every facet covered exactly."""
    e = _entry(sg)
    dom = C.g_asu_domain(sg)
    assert dom.keep == (0, 1, 2) and dom.walls == ()
    assert len(dom.elements) == len(SYM_OPS[sg]) == len(e['G']['W'])
    cert = C.certify_tiling(dom)
    assert cert.dim == 3
    assert cert.volume == X.to_fraction(e['g_asu']['volume'])
    assert cert.volume * cert.n_elements == 1
    assert cert.identity_volume == cert.volume
    assert cert.overlaps == ()
    assert cert.defects == ()
    assert len(cert.facets) >= 4 and not cert.uncovered
    assert sum(f.n_contacts for f in cert.facets) == cert.contacts[2]
    assert cert.ok
    assert C.g_asu_certificate(sg).ok


def test_g_asu_domain_elements_are_the_table_g():
    """The representatives are G in SYM_OPS order (W exact, w modulo 1), the identity first; ``source`` indexes G."""
    for sg in (1, 2, 14, 19, 148, 227):
        e, dom = _entry(sg), C.g_asu_domain(sg)
        assert dom.source == tuple(range(len(e['G']['W'])))
        for (W, w), W0, w0 in zip(dom.elements, e['G']['W'], e['G']['w']):
            assert np.array_equal(W, np.array(W0, dtype=np.int64).reshape(3, 3))
            assert w == tuple(X.to_fraction(q) % 1 for q in w0)
        assert np.array_equal(dom.elements[0][0], np.eye(3, dtype=np.int64)) and not any(dom.elements[0][1])


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', (2, 14, 61, 148, 227))
def test_proper_part_of_g_is_rejected(sg):
    """ASU(G) under G+ only (improper elements dropped): vol x n = 1/2 and facets glued by improper elements open."""
    dom = C.g_asu_domain(sg)
    proper = tuple((W, w) for W, w in dom.elements if round(np.linalg.det(W.astype(float))) == 1)
    assert 2 * len(proper) == len(dom.elements)
    cert = C.certify_tiling(replace(dom, elements=proper, source=tuple(range(len(proper)))))
    assert cert.volume * cert.n_elements == Fraction(1, 2)
    assert cert.uncovered and not cert.ok


@pytest.mark.parametrize('sg', (2, 14, 19, 148))
def test_anchor_asu_under_g_is_rejected(sg):
    """The anchor's ASU(Gamma+) examined under G (no free axes in these groups): vol x |G| < 1."""
    e = dict(_entry(sg))
    assert not any(e['free'])
    e['g_asu'] = e['asu']
    cert = C.certify_tiling(C.g_asu_domain(sg, e))
    assert cert.volume * cert.n_elements < 1 and not cert.ok


@pytest.mark.parametrize('sg', (2, 14, 19, 148, 227))
def test_moved_facet_is_rejected(sg):
    """A facet of ASU(G) pushed out by 1/97 overlaps a neighbour; pulled in by 1/97 it loses volume."""
    dom = C.g_asu_domain(sg)
    cert0 = C.certify_tiling(dom)
    i = cert0.facets[0].cut
    for sign in (1, -1):
        off = list(dom.offsets)
        off[i] += sign * Fraction(1, 97)
        cert = C.certify_tiling(replace(dom, offsets=tuple(off)))
        assert not cert.ok
        if sign > 0:
            assert cert.overlaps and cert.volume * cert.n_elements > 1
        else:
            assert not cert.overlaps and cert.volume * cert.n_elements < 1


@pytest.mark.parametrize('sg', (2, 14, 227))
def test_duplicated_element_is_rejected(sg):
    """A non-identity element replaced by a copy of another (of the identity in sg 2): refused as a duplicate."""
    dom = C.g_asu_domain(sg)
    els = list(dom.elements)
    els[1] = els[0] if len(els) == 2 else els[2]
    cert = C.certify_tiling(replace(dom, elements=tuple(els)))
    assert not cert.ok and 'duplicate' in {kind for kind, _ in cert.defects}
