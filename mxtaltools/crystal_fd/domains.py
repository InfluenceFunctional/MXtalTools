"""Fold Z' = 1 poses into the fundamental domain ASU(Gamma+) x SO(3) of the +1-sheet action.

A ``Section`` is that domain for one space group: the elements of Gamma+ (the
det +1 part of the Euclidean normaliser, modulo Z^3 and free-axis shifts), the
cctbx asymmetric unit of Gamma+ on the non-free coordinates with its tie-break
cut trees, and the gauge of the free axes.

The rules, for (t, R) with free coordinates at the gauge:

1. centroid: slack ``s = min_i (n_i . x + c_i) / |n_i|`` over the top-level cuts.
   ``s > tol`` is inside, ``s < -tol`` outside; for ``|s| <= tol`` the cctbx cut
   trees decide, each plane test again taken within ``tol``;
2. orientation: among candidates that pass (1), the lexicographically largest
   rotated pose ``Q R`` wins, entries compared within ``tol``, and then the
   largest centroid. Candidates passing together are the stabiliser of a special
   position (which in sg 6-9 always contains K: this is the K cut in SO(3)).

``fold`` applies every Gamma+ element with the lattice shifts that can reach the
asymmetric unit, keeps the candidates that pass (1) and chooses by (2). If none
passes (possible only within about ``tol`` of a facet, edge or vertex, where
plane tests within a tolerance are not equivariant), the candidates within
``tol`` of the largest slack are chosen among by (2). Either way the choice is a
function of the orbit, not of the input description, so every image of a crystal
folds to the same description (up to rounding of a slack that is exactly
``+-tol``, where an image's plane test can flip). ``in_domain`` is membership by that definition:
a description is in the domain iff it is its own fold. ``in_asu`` evaluates (1)
and (2) on one description, independently of ``fold``, to validate the domain
(``checks.orbit_check``); the two agree outside that band.

The fold never builds atoms. Free axes are gauged to half the trainer box
(``trainer_box / 2``), the value ``MolCrystalOps.canonicalize_free_axes`` sets.
Elements act on poses through ``Q = T_fc W T_fc^-1`` (projected onto SO(3)).

``fold_unit`` (Z' > 1, stage Z2 of energy_sampling/docs/design/zprime_fd.md)
folds one unit per row into ASU(G) x SO(3) under all of G by the same rules on
cctbx's ASU(G): an element (W, w) acts by (t, R) -> (W t + w, M R D^s),
s = [det W = -1], M = T_fc W T_fc^-1 (projected onto O(3)), D the body
reflection with X(sigma q) = D X(q); an improper element is reported as
``mirrored`` and the caller applies sigma to the unit's conformer.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import torch

from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.exact import to_fraction
from mxtaltools.crystal_fd.ops import _lex_select, cartesian_rotation, cell_matrices, real_cells
from mxtaltools.crystal_fd.so3 import split_o3
from mxtaltools.crystal_fd.tables import GroupTable, group_table

F64 = torch.float64
#: margin when enumerating the lattice shifts that can reach a region (fractional)
_PAD = 1e-6
#: largest number of (row, element, shift) candidates evaluated at once
_CHUNK = 2_000_000


# ---------------------------------------------------------------------------
# regions: polytope + cctbx tie-break cut trees, evaluated within a tolerance
# ---------------------------------------------------------------------------

def _compile(node, device):
    if 'op' in node:
        return ('op', node['op'], _compile(node['lhs'], device), _compile(node['rhs'], device))
    n = torch.tensor([float(v) for v in node['n']], dtype=F64, device=device)
    norm = n.norm()
    c = float(to_fraction(node['c']))
    sub = None if node['sub'] is None else _compile(node['sub'], device)
    return ('cut', n / norm, c / norm.item(), bool(node['inclusive']), sub)


def _tree_contains(node, x: torch.Tensor, tol: float) -> torch.Tensor:
    """cctbx is_inside on x [..., 3], each plane test within ``tol`` (fractional distance)."""
    if node[0] == 'op':
        a = _tree_contains(node[2], x, tol)
        b = _tree_contains(node[3], x, tol)
        return a & b if node[1] == '&' else a | b
    _, n, c, inclusive, sub = node
    v = x @ n + c
    on = v.abs() <= tol
    if sub is None:
        on_ok = torch.full_like(on, inclusive)
    else:
        on_ok = _tree_contains(sub, x, tol)
    return (v > tol) | (on & on_ok)


@dataclass(frozen=True, eq=False)
class Region:
    """A polytope {x : n . x + c >= 0} (unit normals) with cctbx tie-break trees, and its bounding box."""
    normals: torch.Tensor   # [c, 3] unit
    offsets: torch.Tensor   # [c]
    trees: tuple
    lo: torch.Tensor        # [3]
    hi: torch.Tensor        # [3]
    cuts: tuple             # the exact JSON trees

    @classmethod
    def from_cuts(cls, cuts, vertices: torch.Tensor, device=None) -> 'Region':
        cuts = tuple(cuts)
        trees = tuple(_compile(c, device) for c in cuts)
        return cls(normals=torch.stack([t[1] for t in trees]), offsets=torch.tensor([t[2] for t in trees], dtype=F64,
                                                                                    device=device),
                   trees=trees, lo=vertices.amin(0).to(F64), hi=vertices.amax(0).to(F64), cuts=cuts)

    def slack(self, x: torch.Tensor) -> torch.Tensor:
        """[..., 3] -> [...] smallest signed distance (fractional units) to a top-level cut; > 0 inside."""
        return (x @ self.normals.T + self.offsets).amin(-1)

    def contains(self, x: torch.Tensor, tol: float, slack: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Membership with tie rules: slack > tol, or |slack| <= tol and every cut tree accepts."""
        s = self.slack(x) if slack is None else slack
        inside = s > tol
        on = s.abs() <= tol
        if bool(on.any()):
            xo = x[on]
            ok = torch.ones(len(xo), dtype=torch.bool, device=x.device)
            for tr in self.trees:
                ok &= _tree_contains(tr, xo, tol)
            inside = inside.clone()
            inside[on] = ok
        return inside


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Section:
    """The translation cut ASU(Gamma+) x SO(3) of one space group.

    Attributes
    ----------
    sg : int
    W, w : torch.Tensor
        Gamma+ elements, [m, 3, 3] and [m, 3] float64 (w in [0, 1), zero on free
        axes); index 0 is the identity. ``element`` indices refer to this order,
        which is ``GroupTable.gamma_plus``.
    region : Region
        ASU(Gamma+) on the non-free coordinates (a prism along free axes).
    free : torch.Tensor
        [3] bool free axes.
    gauge : torch.Tensor
        [3] value of each free coordinate in the domain (zero elsewhere).
    shifts : torch.Tensor
        [S, 3] long lattice-shift offsets enumerated per element.
    table_hash : str
    """
    sg: int
    W: torch.Tensor
    w: torch.Tensor
    region: Region
    free: torch.Tensor
    gauge: torch.Tensor
    shifts: torch.Tensor
    table_hash: str

    @property
    def order(self) -> int:
        return int(self.W.shape[0])

    @classmethod
    def from_table(cls, tab: GroupTable, region: Optional[Region] = None) -> 'Section':
        dev = tab.gamma_W.device
        W = tab.gamma_plus_W.clone()
        w = tab.gamma_plus_w.clone()
        free = tab.free.clone()
        if not torch.equal(W[0], torch.eye(3, dtype=F64, device=dev)) or bool(w[0].abs().max() > 0):
            raise ValueError(f'sg {tab.sg}: the first Gamma+ element is not the identity')
        if bool(W[:, ~free][:, :, free].ne(0).any()):
            raise ValueError(f'sg {tab.sg}: a Gamma+ element mixes free coordinates into non-free ones')
        if region is None:
            region = Region.from_cuts(tab.asu.cuts, tab.asu.vertices, dev)
        return cls(sg=tab.sg, W=W, w=w, region=region, free=free,
                   gauge=torch.where(free, tab.trainer_box / 2, torch.zeros_like(tab.trainer_box)),
                   shifts=_shift_grid(region, free), table_hash=tab.table_hash)

    def with_region(self, region: Region) -> 'Section':
        """The same group acting on another region (e.g. a mutated asymmetric unit)."""
        return replace(self, region=region, shifts=_shift_grid(region, self.free))


def _shift_grid(region: Region, free: torch.Tensor) -> torch.Tensor:
    m = torch.floor(region.hi - region.lo + 2 * _PAD).to(torch.long) + 1
    m = torch.where(free, torch.ones_like(m), m)
    grids = torch.meshgrid(*[torch.arange(int(k), device=m.device) for k in m], indexing='ij')
    return torch.stack([g.reshape(-1) for g in grids], 1)


@lru_cache(maxsize=None)
def _cached_section(sg: int, device: str) -> Section:
    return Section.from_table(group_table(sg, device))


def translation_section(sg: int, device=None) -> Section:
    """The default section of space group ``sg``: ASU(Gamma+) x SO(3), cached per device."""
    return _cached_section(int(sg), str(torch.device('cpu' if device is None else device)))


# ---------------------------------------------------------------------------
# fold / unfold / membership
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FoldResult:
    """Result of ``fold``.

    Attributes
    ----------
    desc : CrystalDescription
        The canonical descriptions (float64).
    element : torch.Tensor
        [B] long index into ``Section.W`` of the element applied.
    lattice : torch.Tensor
        [B, 3] long lattice shift added after the element (zero on free axes).
    free_shift : torch.Tensor
        [B, 3] float64 continuous shift that moved free coordinates to the gauge.
    slack : torch.Tensor
        [B] slack of the canonical centroid.
    tie : torch.Tensor
        [B] int8: 0 interior; 1 centroid on a face, resolved by the cut trees;
        2 several candidates passed and the orientation rule chose (stabiliser,
        K); 3 no candidate passed the cut trees (the ``tol`` band of a facet,
        edge or vertex) and the orientation rule chose among the candidates
        within ``tol`` of the largest slack: an image of the input within about
        ``tol`` of ASU(Gamma+).
    """
    desc: CrystalDescription
    element: torch.Tensor
    lattice: torch.Tensor
    free_shift: torch.Tensor
    slack: torch.Tensor
    tie: torch.Tensor


def _sections_for(desc: CrystalDescription, section: Optional[Section]):
    if section is not None:
        if bool((desc.sg != section.sg).any()):
            raise ValueError(f'section is for sg {section.sg}; the description holds other groups')
        return [(section, torch.arange(desc.num_rows, device=desc.device))]
    out = []
    for sg in torch.unique(desc.sg).tolist():
        out.append((translation_section(sg, desc.device), torch.nonzero(desc.sg == sg)[:, 0]))
    return out


def _check_zp1(desc: CrystalDescription):
    if bool((desc.z_prime != 1).any()):
        raise NotImplementedError("Z' > 1 folding is a later stage (fd_tooling.md section 2.8)")


def _check_finite(d: CrystalDescription, T: torch.Tensor):
    ok = torch.isfinite(d.t[:, 0]).all(1) & torch.isfinite(d.R[:, 0]).all(-1).all(-1) & \
        torch.isfinite(T).all(-1).all(-1) & real_cells(d.cell)
    if not bool(ok.all()):
        bad = torch.nonzero(~ok)[:, 0].tolist()
        raise ValueError(f'{len(bad)} rows have a non-finite centroid or rotation, or no real cell: '
                         f'rows {bad[:10]}{" ..." if len(bad) > 10 else ""}')


def _rotations(sec: Section, T: torch.Tensor) -> torch.Tensor:
    """[b, m, 3, 3] Q = T W T^-1 for every row and element, projected onto SO(3)."""
    b, m = T.shape[0], sec.order
    return cartesian_rotation(T[:, None].expand(b, m, 3, 3).reshape(-1, 3, 3),
                              sec.W[None].expand(b, m, 3, 3).reshape(-1, 3, 3)).reshape(b, m, 3, 3)


def _fold_block(sec: Section, t: torch.Tensor, R: torch.Tensor, T: torch.Tensor, tol: float):
    b, m, S = t.shape[0], sec.order, sec.shifts.shape[0]
    free = sec.free
    y = torch.einsum('mij,bj->bmi', sec.W, t) + sec.w                     # [b, m, 3]
    free_shift = torch.where(free, sec.gauge - y, torch.zeros_like(y))
    y = torch.where(free, sec.gauge, y)
    nlo = torch.ceil(sec.region.lo - _PAD - y)
    nlo = torch.where(free, torch.zeros_like(nlo), nlo)
    lat = nlo[:, :, None, :] + sec.shifts.to(F64)                          # [b, m, S, 3]
    x = y[:, :, None, :] + lat
    s = sec.region.slack(x)                                                # [b, m, S]
    inset = sec.region.contains(x, tol, s)
    count = inset.sum((1, 2))
    flat = inset.reshape(b, m * S)
    pick = flat.to(torch.int8).argmax(1)                                   # the single candidate when count == 1
    s_pick = s.reshape(b, -1).gather(1, pick[:, None])[:, 0]
    tie = torch.full((b,), 3, dtype=torch.int8, device=t.device)
    tie[(count == 1) & (s_pick > tol)] = 0
    tie[(count == 1) & (s_pick <= tol)] = 1
    Qall = None
    multi = count > 1
    none = count == 0
    if bool((multi | none).any()):
        Qall = _rotations(sec, T)
        sf = s.reshape(b, m * S)
        valid = torch.where(none[:, None], sf >= sf.amax(1, keepdim=True) - tol, flat)
        r = multi | none
        kR = (Qall[r] @ R[r][:, None]).reshape(-1, m, 1, 9).expand(-1, m, S, 9)
        keys = torch.cat([kR, x[r]], -1).reshape(-1, m * S, 12)              # pose, then centroid
        pick[r] = _lex_select(keys, valid[r], tol, largest=True)
        tie[multi] = 2
    e = pick // S
    k = pick % S
    rows = torch.arange(b, device=t.device)
    t_c = x[rows, e, k]
    Q = cartesian_rotation(T, sec.W[e]) if Qall is None else Qall[rows, e]
    return (t_c, Q @ R, e, lat[rows, e, k].round().to(torch.long), free_shift[rows, e], s[rows, e, k], tie)


def fold(desc: CrystalDescription, section: Optional[Section] = None, tol: float = 1e-9) -> FoldResult:
    """Canonical Z' = 1 descriptions: one representative per Gamma+ orbit (see module docstring).

    Parameters
    ----------
    desc : CrystalDescription
        Z' = 1 rows, any handedness (h is unchanged: Gamma+ is proper).
    section : Section, optional
        Domain to fold into; default ``translation_section(sg)`` per row.
    tol : float
        Face tolerance (fractional distance, and rotation-matrix entries).

    Raises
    ------
    ValueError
        For rows with a non-finite centroid or rotation, or no real cell
        (``ops.real_cells``), naming them.
    """
    _check_zp1(desc)
    d = desc.double()
    B = d.num_rows
    dev = d.device
    t_out = torch.empty(B, 3, dtype=F64, device=dev)
    R_out = torch.empty(B, 3, 3, dtype=F64, device=dev)
    element = torch.empty(B, dtype=torch.long, device=dev)
    lattice = torch.empty(B, 3, dtype=torch.long, device=dev)
    free_shift = torch.empty(B, 3, dtype=F64, device=dev)
    slack = torch.empty(B, dtype=F64, device=dev)
    tie = torch.empty(B, dtype=torch.int8, device=dev)
    T_all = d.cell_matrix()
    _check_finite(d, T_all)
    for sec, idx in _sections_for(d, section):
        step = max(1, _CHUNK // (sec.order * sec.shifts.shape[0]))
        for lo in range(0, len(idx), step):
            r = idx[lo:lo + step]
            out = _fold_block(sec, d.t[r, 0], d.R[r, 0], T_all[r], tol)
            t_out[r], R_out[r], element[r], lattice[r], free_shift[r], slack[r], tie[r] = out
    canon = d.replace(t=t_out[:, None], R=R_out[:, None])
    return FoldResult(desc=canon, element=element, lattice=lattice, free_shift=free_shift, slack=slack, tie=tie)


def unfold(desc: CrystalDescription, element: torch.Tensor, lattice: torch.Tensor,
           free_shift: Optional[torch.Tensor] = None, section: Optional[Section] = None) -> CrystalDescription:
    """The exact inverse of ``fold``: t = W^-1 (t_c - w - lattice - free_shift), R = Q^T R_c."""
    _check_zp1(desc)
    d = desc.double()
    B = d.num_rows
    lattice = lattice.to(F64)
    free_shift = torch.zeros(B, 3, dtype=F64, device=d.device) if free_shift is None else free_shift.to(F64)
    t = torch.empty(B, 3, dtype=F64, device=d.device)
    R = torch.empty(B, 3, 3, dtype=F64, device=d.device)
    T_all = d.cell_matrix()
    for sec, idx in _sections_for(d, section):
        e = element[idx]
        W, w = sec.W[e], sec.w[e]
        t[idx] = torch.einsum('bij,bj->bi', torch.linalg.inv(W), d.t[idx, 0] - w - lattice[idx] - free_shift[idx])
        Q = cartesian_rotation(T_all[idx], W)
        R[idx] = Q.transpose(1, 2) @ d.R[idx, 0]
    return d.replace(t=t[:, None], R=R[:, None])


def in_domain(desc: CrystalDescription, section: Optional[Section] = None, tol: float = 1e-9):
    """(inside [B] bool, slack [B]) of Z' = 1 descriptions: inside iff the row is its own fold (t, R within ``tol``).

    Every orbit has exactly one such description, since ``fold`` is a function of
    the orbit (except where a slack lies within rounding of exactly +-``tol``).
    ``slack`` is the centroid's slack (free coordinates at the gauge). See
    ``in_asu`` for the rules checked one description at a time.
    """
    _check_zp1(desc)
    d = desc.double()
    fr = fold(d, section, tol)
    inside = ((fr.desc.t[:, 0] - d.t[:, 0]).abs() <= tol).all(1) & \
        ((fr.desc.R[:, 0] - d.R[:, 0]).abs() <= tol).all(-1).all(-1)
    slack = torch.empty(d.num_rows, dtype=F64, device=d.device)
    for sec, idx in _sections_for(d, section):
        slack[idx] = sec.region.slack(torch.where(sec.free, sec.gauge, d.t[idx, 0]))
    return inside, slack


def in_asu(desc: CrystalDescription, section: Optional[Section] = None, tol: float = 1e-9):
    """(inside [B] bool, slack [B]) of Z' = 1 descriptions by the rules (module docstring), independently of ``fold``.

    Rule (2) is applied to the elements that fix the centroid within ``tol``.
    Free coordinates must equal the gauge within ``tol``. Outside a band about
    ``tol`` wide around facets, edges and vertices this equals ``in_domain``;
    inside it an orbit can count 0 or 2 images. For validating a domain (orbit
    counts, mutated regions).
    """
    _check_zp1(desc)
    d = desc.double()
    B = d.num_rows
    inside = torch.zeros(B, dtype=torch.bool, device=d.device)
    slack = torch.empty(B, dtype=F64, device=d.device)
    T_all = d.cell_matrix()
    for sec, idx in _sections_for(d, section):
        t, R = d.t[idx, 0], d.R[idx, 0]
        free = sec.free
        gauge_ok = ((t - sec.gauge).abs() <= tol)[:, free].all(1)
        x = torch.where(free, sec.gauge, t)
        s = sec.region.slack(x)
        slack[idx] = s
        ok = gauge_ok & sec.region.contains(x, tol, s)
        cand = torch.nonzero(ok)[:, 0]
        if len(cand):
            xc, Rc = x[cand], R[cand]
            y = torch.einsum('mij,bj->bmi', sec.W, xc) + sec.w
            y = torch.where(free, sec.gauge, y)
            dist = (y + torch.round(xc[:, None] - y) - xc[:, None]).abs()
            stab = (torch.where(free, torch.zeros_like(dist), dist) <= tol).all(-1)      # [b, m]
            if bool(stab[:, 1:].any()):
                Q = _rotations(sec, T_all[idx][cand])
                keys = (Q @ Rc[:, None]).reshape(len(cand), sec.order, 9)
                ok[cand] = _lex_select(keys, stab, tol, largest=True) == 0
        inside[idx] = ok
    return inside, slack


# ---------------------------------------------------------------------------
# free axes, images, handedness
# ---------------------------------------------------------------------------

def gauge_free_axes(desc: CrystalDescription) -> CrystalDescription:
    """Z' = 1 rows with every free coordinate at the gauge (``canonicalize_free_axes``' value)."""
    _check_zp1(desc)
    d = desc.double()
    t = d.t.clone()
    for sec, idx in _sections_for(d, None):
        t[idx, 0] = torch.where(sec.free, sec.gauge, t[idx, 0])
    return d.replace(t=t)


def images(desc: CrystalDescription, group: str = 'gamma_plus', within: Optional[str] = None, tol: float = 1e-9):
    """Every description of each Z' = 1 row under a group, optionally restricted to a region.

    Parameters
    ----------
    group : str
        ``'gamma_plus'`` (proper; h unchanged) or ``'gamma'`` (the whole
        normaliser table; a det -1 element flips h).
    within : str, optional
        None: one image per element, centroid wrapped to [0, 1)^3.
        ``'g_asu'``: the images whose centroid lies in cctbx's asymmetric unit of
        G (tie rules within ``tol``). ``'latent_v1_box'``: the images on the +1
        sheet whose non-free coordinates lie in [0, ASYM_UNITS[sg]) with free
        coordinates at the trainer gauge.

    Returns
    -------
    (CrystalDescription, source [N] long row index, element [N] long index into
    the group's element list: ``GroupTable.gamma_plus`` order, or ``gamma``)
    """
    _check_zp1(desc)
    if group not in ('gamma_plus', 'gamma'):
        raise ValueError(group)
    if within not in (None, 'g_asu', 'latent_v1_box'):
        raise ValueError(f'within={within!r}: expected None, "g_asu" or "latent_v1_box"')
    d = desc.double()
    T_all = d.cell_matrix()
    parts, sources, elements = [], [], []
    for sg in torch.unique(d.sg).tolist():
        tab = group_table(sg, d.device)
        idx = torch.nonzero(d.sg == sg)[:, 0]
        if group == 'gamma_plus':
            W, w = tab.gamma_plus_W, tab.gamma_plus_w
        else:
            W, w = tab.gamma_W, tab.gamma_w
        b, m = len(idx), W.shape[0]
        T = T_all[idx]
        y = torch.einsum('mij,bj->bmi', W, d.t[idx, 0]) + w
        Q = cartesian_rotation(T[:, None].expand(b, m, 3, 3).reshape(-1, 3, 3),
                               W[None].expand(b, m, 3, 3).reshape(-1, 3, 3))
        R, h = split_o3(Q @ d.pose[idx, 0][:, None].expand(b, m, 3, 3).reshape(-1, 3, 3))
        R, h = R.reshape(b, m, 3, 3), h.reshape(b, m)
        src = idx[:, None].expand(b, m)
        el = torch.arange(m, device=d.device)[None].expand(b, m)
        if within is None:
            t_img, keep = y - torch.floor(y), torch.ones(b, m, dtype=torch.bool, device=d.device)
            t_img, R_s, h_s, src_s, el_s = t_img[keep], R[keep], h[keep], src[keep], el[keep]
        elif within == 'g_asu':
            region = Region.from_cuts(tab.g_asu.cuts, tab.g_asu.vertices, d.device)
            grid = _shift_grid(region, torch.zeros(3, dtype=torch.bool, device=d.device)).to(F64)
            nlo = torch.ceil(region.lo - _PAD - y)
            x = y[:, :, None] + nlo[:, :, None] + grid                         # [b, m, S, 3]
            keep = region.contains(x, tol)
            S = grid.shape[0]
            t_img = x[keep]
            R_s = R[:, :, None].expand(b, m, S, 3, 3)[keep]
            h_s = h[:, :, None].expand(b, m, S)[keep]
            src_s = src[:, :, None].expand(b, m, S)[keep]
            el_s = el[:, :, None].expand(b, m, S)[keep]
        else:
            box = tab.trainer_box
            t_img = torch.where(tab.free, box / 2, y - torch.floor(y))
            keep = (h == 1) & (t_img < box).all(-1)
            t_img, R_s, h_s, src_s, el_s = t_img[keep], R[keep], h[keep], src[keep], el[keep]
        parts.append(CrystalDescription.from_parts(d.sg[src_s], d.cell[src_s], t_img, R_s, h_s))
        sources.append(src_s)
        elements.append(el_s)
    return CrystalDescription.cat(parts), torch.cat(sources), torch.cat(elements)


def to_sheet(desc: CrystalDescription, sheet: int = 1):
    """Move every Z' = 1 row to handedness ``sheet`` through an improper element of G.

    Returns (desc, ok [B] bool). Rows already on the sheet are unchanged. A row
    of a Sohncke group (no improper element in G) cannot change sheet: it is
    returned unchanged with ok = False, since its other-handed description is a
    crystal of the mirror-image molecule.
    """
    _check_zp1(desc)
    if sheet not in (1, -1):
        raise ValueError('sheet must be +1 or -1')
    d = desc.double()
    ok = d.h[:, 0] == sheet
    out = d
    for sg in torch.unique(d.sg[~ok]).tolist():
        tab = group_table(sg, d.device)
        det = torch.round(torch.linalg.det(tab.g_W))
        improper = torch.nonzero(det < 0)[:, 0]
        if len(improper) == 0:
            continue
        rows = torch.nonzero((d.sg == sg) & ~ok)[:, 0]
        k = int(improper[0])
        from mxtaltools.crystal_fd.ops import apply
        moved = apply(d[rows], tab.g_W[k], tab.g_w[k], wrap=True)
        out = out.replace(t=out.t.index_put((rows,), moved.t), R=out.R.index_put((rows,), moved.R),
                          h=out.h.index_put((rows,), moved.h))
        ok = ok.index_put((rows,), torch.ones(len(rows), dtype=torch.bool, device=d.device))
    return out, ok


# ---------------------------------------------------------------------------
# Z' > 1: one unit in ASU(G) x SO(3), the conformer mirror on improper elements
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class UnitSection:
    """ASU(G) x SO(3) of one space group, the domain of every unit but the anchor (zprime_fd.md 3.4).

    Attributes
    ----------
    sg : int
    W, w : torch.Tensor
        G, [m, 3, 3] and [m, 3] float64: ``GroupTable.g_W`` / ``g_w`` (SYM_OPS order, w
        unreduced). ``element`` indices refer to this order.
    det : torch.Tensor
        [m] long, det W (+-1).
    translation : torch.Tensor
        [m] bool, W = I: these elements leave poses exactly unchanged.
    region : Region
        cctbx's ASU(G) with its tie-break cut trees (``GroupTable.g_asu``).
    shifts : torch.Tensor
        [S, 3] long lattice-shift offsets enumerated per element.
    table_hash : str
    """
    sg: int
    W: torch.Tensor
    w: torch.Tensor
    det: torch.Tensor
    translation: torch.Tensor
    region: Region
    shifts: torch.Tensor
    table_hash: str

    @property
    def order(self) -> int:
        return int(self.W.shape[0])

    @classmethod
    def from_table(cls, tab: GroupTable) -> 'UnitSection':
        dev = tab.g_W.device
        region = Region.from_cuts(tab.g_asu.cuts, tab.g_asu.vertices, dev)
        eye = torch.eye(3, dtype=F64, device=dev)
        return cls(sg=tab.sg, W=tab.g_W.clone(), w=tab.g_w.clone(),
                   det=torch.round(torch.linalg.det(tab.g_W)).to(torch.long),
                   translation=(tab.g_W == eye).all(-1).all(-1), region=region,
                   shifts=_shift_grid(region, torch.zeros(3, dtype=torch.bool, device=dev)),
                   table_hash=tab.table_hash)


@lru_cache(maxsize=None)
def _cached_unit_section(sg: int, device: str) -> UnitSection:
    return UnitSection.from_table(group_table(sg, device))


def unit_section(sg: int, device=None) -> UnitSection:
    """ASU(G) x SO(3) of space group ``sg``, cached per device."""
    return _cached_unit_section(int(sg), str(torch.device('cpu' if device is None else device)))


@dataclass(frozen=True, eq=False)
class UnitFold:
    """Result of ``fold_unit``.

    Attributes
    ----------
    t : torch.Tensor
        [N, 3] float64 folded centroids ``W t + w + lattice``, in ASU(G).
    R : torch.Tensor
        [N, 3, 3] float64 folded proper rotations ``M R D^s``.
    element : torch.Tensor
        [N] long index into ``UnitSection.W`` (``GroupTable.g_W``) of the element applied.
    W, w : torch.Tensor
        [N, 3, 3] and [N, 3]: that element.
    lattice : torch.Tensor
        [N, 3] long lattice shift added after the element.
    mirrored : torch.Tensor
        [N] bool, det W = -1: R has been mapped to M R D, and the caller must apply
        sigma to the unit's conformer.
    slack : torch.Tensor
        [N] slack of the folded centroid in ASU(G).
    tie : torch.Tensor
        [N] int8, as ``FoldResult.tie``: 0 interior; 1 on a face, resolved by the
        cut trees; 2 several candidates passed and the pose rule chose: the stabiliser of a
        special position of G, or, within about ``tol`` of a glued facet pair whose gluing does
        not preserve fractional distance (hexagonal and trigonal groups), a facet point and its
        glued image. Not a special-position flag: test the stabiliser directly for that. 3 none
        passed (the ``tol`` band of a facet, edge or vertex).
    """
    t: torch.Tensor
    R: torch.Tensor
    element: torch.Tensor
    W: torch.Tensor
    w: torch.Tensor
    lattice: torch.Tensor
    mirrored: torch.Tensor
    slack: torch.Tensor
    tie: torch.Tensor


def _reflection(D, device) -> torch.Tensor:
    """The body reflection: diag(1, 1, -1) by default; otherwise checked orthogonal with D D = I and det D = -1."""
    if D is None:
        return torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=F64, device=device))
    D = torch.as_tensor(D, dtype=F64, device=device)
    eye = torch.eye(3, dtype=F64, device=device)
    if D.shape != (3, 3) or not bool(torch.isfinite(D).all()) or float((D @ D.T - eye).abs().max()) > 1e-12 \
            or float((D @ D - eye).abs().max()) > 1e-12 or float(torch.linalg.det(D)) > 0:
        raise ValueError('D must be a reflection: orthogonal, D @ D = I and det D = -1')
    return D


def _unit_poses(sec: UnitSection, T: torch.Tensor, R: torch.Tensor, D: torch.Tensor, e=None) -> torch.Tensor:
    """M R D^s: [b, m, 3, 3] for every element of ``sec``, or [b, 3, 3] for the elements ``e`` [b].

    M = T W T^-1 projected onto O(3), and exactly I for the translations.
    """
    if e is None:
        b, m = T.shape[0], sec.order
        M = cartesian_rotation(T[:, None].expand(b, m, 3, 3).reshape(-1, 3, 3),
                               sec.W[None].expand(b, m, 3, 3).reshape(-1, 3, 3)).reshape(b, m, 3, 3)
        transl, improper, R = sec.translation[None], sec.det[None] < 0, R[:, None]
    else:
        M = cartesian_rotation(T, sec.W[e])
        transl, improper = sec.translation[e], sec.det[e] < 0
    M = torch.where(transl[..., None, None], torch.eye(3, dtype=F64, device=T.device), M)
    P = M @ R
    return torch.where(improper[..., None, None], P @ D, P)


def _fold_unit_block(sec: UnitSection, t: torch.Tensor, R: torch.Tensor, T: torch.Tensor, D: torch.Tensor,
                     tol: float):
    b, m, S = t.shape[0], sec.order, sec.shifts.shape[0]
    y = torch.einsum('mij,bj->bmi', sec.W, t) + sec.w                     # [b, m, 3]
    lat = torch.ceil(sec.region.lo - _PAD - y)[:, :, None, :] + sec.shifts.to(F64)   # [b, m, S, 3]
    x = y[:, :, None, :] + lat
    s = sec.region.slack(x)                                                # [b, m, S]
    inset = sec.region.contains(x, tol, s)
    count = inset.sum((1, 2))
    flat = inset.reshape(b, m * S)
    pick = flat.to(torch.int8).argmax(1)                                   # the single candidate when count == 1
    s_pick = s.reshape(b, -1).gather(1, pick[:, None])[:, 0]
    tie = torch.full((b,), 3, dtype=torch.int8, device=t.device)
    tie[(count == 1) & (s_pick > tol)] = 0
    tie[(count == 1) & (s_pick <= tol)] = 1
    multi = count > 1
    r = multi | (count == 0)
    if bool(r.any()):
        sf = s[r].reshape(-1, m * S)
        valid = torch.where(multi[r][:, None], flat[r], sf >= sf.amax(1, keepdim=True) - tol)
        kR = _unit_poses(sec, T[r], R[r], D).reshape(-1, m, 1, 9).expand(-1, m, S, 9)
        keys = torch.cat([kR, x[r]], -1).reshape(-1, m * S, 12)              # pose, then centroid
        pick[r] = _lex_select(keys, valid, tol, largest=True)
        tie[multi] = 2
    e = pick // S
    k = pick % S
    rows = torch.arange(b, device=t.device)
    return (x[rows, e, k], _unit_poses(sec, T, R, D, e), e, lat[rows, e, k].round().to(torch.long), s[rows, e, k],
            tie)


def fold_unit(sg, cell, t, R, D=None, tol: float = 1e-9) -> UnitFold:
    """Fold units of a Z' > 1 crystal, one per row, into ASU(G) x SO(3), with the conformer mirror.

    Every element of G applies, improper ones included: (W, w) with lattice shift L
    acts on a unit by (t, R, q) -> (W t + w + L, M R D^s, sigma^s q), s = [det W = -1],
    M = T_fc W T_fc^-1 (projected onto O(3); exactly I for translations). The rules are
    ``fold``'s (module docstring) on cctbx's ASU(G) and its cut trees: (1) the centroid
    by slack and cut trees within ``tol``; (2) among candidates passing together (the
    stabiliser of a special position of G, or the two images of a point in the ``tol`` band
    of glued facets), the lexicographically largest folded pose
    M R D^s, then the largest centroid; if none passes, the candidates within ``tol``
    of the largest slack, chosen by (2). The choice is a function of the unit's G-orbit,
    so every G image of a unit, with the mirror applied to an improper image, folds to
    the same (t, R) (outside rounding of a slack at exactly +-``tol``). The fold reads no
    conformer: where an improper element fixes both the centroid and the pose
    (M R D = R, a set of measure zero) the candidates differ only in q, and the first in
    element order is returned.

    Parameters
    ----------
    sg : int or torch.Tensor
        Space group, or [N] long per row.
    cell : torch.Tensor
        [N, 6] or [6]: a, b, c (Angstrom), alpha, beta, gamma (rad).
    t : torch.Tensor
        [N, 3] fractional centroids.
    R : torch.Tensor
        [N, 3, 3] proper rotations (Cartesian), seed frame to lab.
    D : torch.Tensor, optional
        The body reflection with X(sigma q) = D X(q): orthogonal, D @ D = I, det -1.
        Default diag(1, 1, -1), the reflection through the conformer seed plane (xy).
    tol : float
        Face tolerance (fractional distance, and rotation-matrix entries).

    Raises
    ------
    ValueError
        For rows with a non-finite centroid or rotation, an improper rotation, or no
        real cell (``ops.real_cells``), naming them; for a D that is not a reflection.
    """
    t = torch.as_tensor(t).to(F64)
    dev = t.device
    N = t.shape[0]
    R = torch.as_tensor(R).to(device=dev, dtype=F64)
    cell = torch.as_tensor(cell).to(device=dev, dtype=F64)
    cell = cell.expand(N, 6) if cell.ndim == 1 else cell
    sg = torch.as_tensor(sg, device=dev).to(torch.long).reshape(-1)
    sg = sg.expand(N) if sg.numel() == 1 else sg
    if t.shape != (N, 3) or R.shape != (N, 3, 3) or cell.shape != (N, 6) or sg.shape != (N,):
        raise ValueError(f'expected t [N, 3], R [N, 3, 3], cell [N, 6] or [6] and sg [N] or scalar; got '
                         f'{tuple(t.shape)}, {tuple(R.shape)}, {tuple(cell.shape)}, {tuple(sg.shape)}')
    D = _reflection(D, dev)
    T = cell_matrices(cell)
    ok = torch.isfinite(t).all(1) & torch.isfinite(R).all(-1).all(-1) & torch.isfinite(T).all(-1).all(-1) & \
        real_cells(cell)
    eye = torch.eye(3, dtype=F64, device=dev)
    ok &= torch.linalg.det(torch.where(ok[:, None, None], R, eye)) > 0
    if not bool(ok.all()):
        bad = torch.nonzero(~ok)[:, 0].tolist()
        raise ValueError(f'{len(bad)} rows have a non-finite centroid or rotation, an improper rotation, or no real '
                         f'cell: rows {bad[:10]}{" ..." if len(bad) > 10 else ""}')
    t_out = torch.empty(N, 3, dtype=F64, device=dev)
    R_out = torch.empty(N, 3, 3, dtype=F64, device=dev)
    element = torch.empty(N, dtype=torch.long, device=dev)
    lattice = torch.empty(N, 3, dtype=torch.long, device=dev)
    slack = torch.empty(N, dtype=F64, device=dev)
    tie = torch.empty(N, dtype=torch.int8, device=dev)
    W_out = torch.empty(N, 3, 3, dtype=F64, device=dev)
    w_out = torch.empty(N, 3, dtype=F64, device=dev)
    mirrored = torch.empty(N, dtype=torch.bool, device=dev)
    for g in torch.unique(sg).tolist():
        sec = unit_section(g, dev)
        idx = torch.nonzero(sg == g)[:, 0]
        step = max(1, _CHUNK // (sec.order * sec.shifts.shape[0]))
        for lo in range(0, len(idx), step):
            r = idx[lo:lo + step]
            out = _fold_unit_block(sec, t[r], R[r], T[r], D, tol)
            t_out[r], R_out[r], element[r], lattice[r], slack[r], tie[r] = out
            W_out[r], w_out[r], mirrored[r] = sec.W[out[2]], sec.w[out[2]], sec.det[out[2]] < 0
    return UnitFold(t=t_out, R=R_out, element=element, W=W_out, w=w_out, lattice=lattice, mirrored=mirrored,
                    slack=slack, tie=tie)
