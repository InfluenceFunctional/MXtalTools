"""Charts: coordinates on a domain of Z' = 1 crystal descriptions, with exact maps and log-Jacobians.

Protocol
--------
A chart has

- ``dims``: one ``Dim(name, kind, period, ar_position)`` per coordinate, in column
  order, which is the autoregressive (AR) order. ``kind`` is ``'interval'``,
  ``'circle'`` (``y`` and ``y + period`` give the same crystal on the +1 sheet),
  ``'fixed'`` (a row the gauge fixes; it carries no information) or ``'embedded'``
  (the quaternion chart, a 3-manifold in R^4);
- ``forward(y, ctx) -> ChartMap(desc, logabsdet)``;
- ``inverse(desc) -> (y, logabsdet)``;
- ``bounds(y_prefix, i) -> (lo, hi)``: the range [B] of coordinate ``i`` given the
  first ``i`` columns ``y_prefix`` [B, i].

``logabsdet`` is the log-density, relative to Lebesgue measure on ``y``, of the
natural measure of what the chart describes: Lebesgue d^k t on the non-free
fractional centroid coordinates, normalised Haar measure on SO(3), and on the cell
d of each free log length and of each free angle (``CellChart``, per family):
d(log a) d(log b) d(log c) d(alpha) d(beta) d(gamma) triclinic; d(log a) d(log b)
d(log c) d(beta) monoclinic; d(log a) d(log b) d(log c) orthorhombic; d(log a) d(log c)
tetragonal and hexagonal; d(log a) cubic. A length or angle the family fixes (an
angle of 90 or 120 deg, b = a, c = a) has no coordinate and no factor. ``forward``
returns it at ``y``; ``inverse`` returns its negative (the log-Jacobian of the
inverse map). ``measure.log_reference_density`` adds the cell-measure and V terms.

Block charts (``PolytopeARChart`` for t, ``OrientationChart`` for R, ``CellChart``
for the cell) also offer ``map(y) -> (x, logabsdet)`` and ``unmap(x)``; their
``forward`` fills one field of a context description. ``CrystalChart`` is the
product of the three on the +1 sheet, for every space group (sg 6-9 with the K cut of
``KCutHemisphereChart`` in the hemisphere chart, or as the double cover of
``orientation_cover='double'`` in any orientation chart). ``LatentV1Chart`` wraps the
trainer's latent maps (``MolCrystalOps.latent_to_cell_params`` / ``latent_transform``),
clamp included, and returns the clamp as a mask; its log-Jacobian is built for sg 1
and 2 only.
"""
import math
from typing import NamedTuple, Optional

import torch

from mxtaltools.common.sym_utils import MONO_CLASS, MONO_FREE
from mxtaltools.crystal_fd import axis_order, so3
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.domains import translation_section
from mxtaltools.crystal_fd.exact import to_fraction
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64
_EYE = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # identity, as the table stores W (row-major)
KINDS = ('interval', 'circle', 'fixed', 'embedded')

#: space groups of each cell family; trigonal groups, P and R alike, are 'hexagonal' (SYM_OPS holds the R groups on
#: hexagonal obverse axes, and their exact metric is the hexagonal one)
FAMILY_SG = {'triclinic': (1, 2), 'monoclinic': (3, 15), 'orthorhombic': (16, 74), 'tetragonal': (75, 142),
             'hexagonal': (143, 194), 'cubic': (195, 230)}
#: the cell families from orthorhombic up (one coordinate per free cell length; every angle fixed)
HIGHER_FAMILIES = ('orthorhombic', 'tetragonal', 'hexagonal', 'cubic')
HALF_PI = math.pi / 2
#: gamma of the hexagonal cell: math.acos(-0.5), the float64 nearest 2 pi / 3 (2 * math.pi / 3 is one ulp below it;
#: ``canon``'s snap returns either, from the rounding of the snapped metric)
HEX_GAMMA = math.acos(-0.5)


def cell_family(sg: int) -> str:
    """The cell family of space group ``sg`` (``FAMILY_SG``): the ``CellChart`` family its ``CrystalChart`` takes."""
    sg = int(sg)
    for fam, (lo, hi) in FAMILY_SG.items():
        if lo <= sg <= hi:
            return fam
    raise ValueError(f'space group {sg} is outside 1-230')


class Dim(NamedTuple):
    """One chart coordinate: its name, kind (``KINDS``), period (circles only, chart units) and AR position."""
    name: str
    kind: str
    period: Optional[float]
    ar_position: int


class ChartMap(NamedTuple):
    """``forward`` output: descriptions [B] and log-density [B] of the natural measure relative to Lebesgue on y."""
    desc: CrystalDescription
    logabsdet: torch.Tensor


class LatentMap(NamedTuple):
    """``LatentV1Chart.forward`` output: as ``ChartMap``, plus ``clamped`` [B], True where the live clamp moved y."""
    desc: CrystalDescription
    logabsdet: torch.Tensor
    clamped: torch.Tensor


class Chart:
    """Base class; see the module docstring for the protocol."""
    dims: tuple = ()

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def forward(self, y, ctx=None):
        raise NotImplementedError

    def inverse(self, desc):
        raise NotImplementedError

    def bounds(self, y_prefix, i):
        raise NotImplementedError


def from_unit(chart: Chart, u: torch.Tensor):
    """Fill chart coordinates from u in [0, 1]^n through the AR bounds: y_i = lo_i + u_i (hi_i - lo_i).

    Returns (y [B, n], log_volume [B]) with ``log_volume = sum_i log(hi_i - lo_i)``;
    y is then distributed with density exp(-log_volume) when u is uniform.
    """
    u = torch.as_tensor(u, dtype=F64)
    cols, logv = [], torch.zeros(u.shape[0], dtype=F64, device=u.device)
    for i in range(chart.ndim):
        prefix = torch.stack(cols, 1) if cols else u.new_zeros(u.shape[0], 0)
        lo, hi = chart.bounds(prefix, i)
        cols.append(lo + u[:, i] * (hi - lo))
        logv = logv + torch.log(hi - lo)
    y = torch.stack(cols, 1) if cols else u.new_zeros(u.shape[0], 0)
    return y, logv


def to_unit(chart: Chart, y: torch.Tensor) -> torch.Tensor:
    """The inverse of ``from_unit``: u_i = (y_i - lo_i(y_<i)) / (hi_i - lo_i(y_<i)), [B, n] float64.

    A circle coordinate is first wrapped into its [lo, hi) period. u is not clamped: a
    row outside the chart's bounds lands outside [0, 1].
    """
    y = torch.as_tensor(y, dtype=F64)
    u = torch.empty_like(y)
    for i in range(chart.ndim):
        lo, hi = chart.bounds(y[:, :i], i)
        yi = y[:, i]
        if chart.dims[i].kind == 'circle':
            yi = lo + torch.remainder(yi - lo, float(chart.dims[i].period))
        u[:, i] = (yi - lo) / (hi - lo)
    return u


def _full(B, value, like):
    return torch.full((B,), float(value), dtype=F64, device=like.device)


# ---------------------------------------------------------------------------
# block charts
# ---------------------------------------------------------------------------

class _BlockChart(Chart):
    """A chart of one field of a Z' = 1 description: ``field`` is 'cell', 't' or 'R'."""
    field = None

    def map(self, y):
        raise NotImplementedError

    def unmap(self, x):
        raise NotImplementedError

    def forward(self, y, ctx=None) -> ChartMap:
        """Fill this chart's field of ``ctx`` (a Z' = 1 CrystalDescription) from ``y``."""
        if ctx is None:
            raise ValueError(f'{type(self).__name__} fills the {self.field!r} field of a context description')
        x, ld = self.map(y)
        d = ctx.double()
        if self.field == 'cell':
            d = d.replace(cell=x)
        elif self.field == 't':
            d = d.replace(t=x[:, None])
        else:
            d = d.replace(R=x[:, None])
        return ChartMap(d, ld)

    def inverse(self, desc):
        d = desc.double()
        x = d.cell if self.field == 'cell' else (d.t[:, 0] if self.field == 't' else d.R[:, 0])
        return self.unmap(x)


class TranslationSeam(NamedTuple):
    """Two facets of ASU(Gamma+) glued by a pure translation: circle coordinate ``dim`` of the centroid chart.

    The facet with cut index ``lower`` (``asu.cuts``) is u_dim = 0, the facet ``upper`` is u_dim = 1, and the exact
    translation ``w`` (fractional, ``fractions.Fraction`` [3], lattice shift included) takes the first onto the second
    with every other chart coordinate unchanged.
    """
    dim: int
    lower: int
    upper: int
    w: tuple


class PolytopeARChart(_BlockChart):
    """Centroid chart of ASU(Gamma+): u in [0, 1]^k -> t through Fourier-Motzkin bounds.

    The AR coordinates are the non-free fractional coordinates t (``basis`` None) or, in a
    sheared group, s = M t with M = ``basis`` (exact integer rows, unit rows on free
    axes). Coordinate i is AR coordinate ``order[i]``:
    ``s_c = lo_i(s_prev) + u_i (hi_i - lo_i)``. log|det dt/du| = sum log(hi_i - lo_i)
    - ``log_det_basis`` (log|det M|, 0 without a basis): the density of d^k t, the
    natural measure, in either case. Free axes are set to the gauge
    (``domains.translation_section(sg).gauge``). A box (all bounds constant) is the
    special case (sg 2: the 3-torus of side 1/2).

    Every pair of facets glued by a pure translation (``tab.facets`` kind
    'translation', W = I) is a circle of period 1 in u (``seams``, ``TranslationSeam``).
    Without a basis (the table's AR order and bounds, every group whose seams run along
    an axis) the circle is a coordinate c whose faces are glued by (hi - lo) e_c. In sg
    88, 91, 92, 95, 96, 98, 122, 141 and 142 the seam translation is (1/2, +-1/2, 0),
    across the facets x +- y = const: there M has rows (1, +-1, 0), (1, -+1, 0) and
    (0, 0, 1) (the seam normal first in AR order, then rows orthogonal to the
    translation, preferring axes and facet normals), ASU(Gamma+) is a box in s, |det M| = 2,
    and the translation moves s_0 by its period alone (``_centroid_basis``).

    Parameters
    ----------
    sg : int
    device : optional
    stages : sequence of tables.ARStage, optional
        Replacement bounds over the AR coordinates (mutation controls); default the table's (without a basis) or
        the sheared basis's. Build them from ``PolytopeARChart(sg).stages``, not ``group_table(sg).ar``: in a sheared
        group the table's bounds are over t and these are over s.
    stages_in : 't' or 's'
        The coordinates ``stages`` is written in: 't' (default; the table's) or 's' (``self.stages``, the AR
        coordinates). Without a basis s = t and either is accepted; in a sheared group 't' is refused (ValueError):
        bounds over t are no bounds over s, and reading them as such would chart a different polytope.
    """
    field = 't'

    def __init__(self, sg: int, device=None, stages=None, stages_in: str = 't'):
        if stages_in not in ('t', 's'):
            raise ValueError(f"stages_in must be 't' or 's', not {stages_in!r}")
        tab = group_table(sg, device)
        self.sg = int(sg)
        self.free = tab.free
        self.gauge = translation_section(sg, device).gauge
        shear = _centroid_basis(self.sg)
        pairs = _translation_pairs(tab)
        if shear is None:
            self.basis = self.basis_inv = None
            self.log_det_basis = 0.0
            self.order = tab.ar_order
            self.stages = tab.ar if stages is None else tuple(stages)
            circles = _translation_circles(tab)
            names = tuple('t_' + 'xyz'[c] for c in self.order)
            slots = [next(k for k in range(3) if n[k] != 0) for _, _, n, _ in pairs]
        else:
            if stages is not None and stages_in != 's':
                raise ValueError(f"sg {self.sg}: the centroid chart's AR coordinates are s = M t (a sheared basis), so "
                                 "replacement stages over t cannot be read as bounds over s; build them from "
                                 "PolytopeARChart(sg).stages and pass stages_in='s'")
            dev = tab.free.device
            self.basis = torch.tensor([[float(v) for v in r] for r in shear.rows], dtype=F64, device=dev)
            self.basis_inv = torch.tensor([[float(v) for v in r] for r in shear.inv], dtype=F64, device=dev)
            self.log_det_basis = shear.log_det
            self.order = shear.order
            self.stages = tuple(_ar_stage(st, k, dev) for k, st in enumerate(shear.stages)) if stages is None \
                else tuple(stages)
            circles = set(shear.circle_slots)
            names = tuple('t_' + _row_name(shear.rows[c]) for c in self.order)
            slots = list(shear.circle_slots)
        self.dims = tuple(Dim(name, 'circle' if c in circles else 'interval', 1.0 if c in circles else None, i)
                          for i, (c, name) in enumerate(zip(self.order, names)))
        self.seams = tuple(TranslationSeam(tuple(self.order).index(c), lo, up, w)
                           for c, (lo, up, _, w) in zip(slots, pairs))

    def t_bounds(self, t_prev: torch.Tensor, i: int):
        """(lo, hi) [B] of AR coordinate ``order[i]`` given the earlier AR coordinates ``t_prev`` [B, i]: fractional t,
        or s = M t in a sheared group (``ar_coordinates``)."""
        st = self.stages[i]
        lo = (t_prev @ st.lower_coef.T + st.lower_const).amax(-1)
        hi = (t_prev @ st.upper_coef.T + st.upper_const).amin(-1)
        return lo, hi

    def ar_coordinates(self, t) -> torch.Tensor:
        """[B, k] the AR coordinates of fractional centroids ``t`` [B, 3], in AR order: t[:, order], or (M t)[:, order]
        in a sheared group."""
        t = torch.as_tensor(t, dtype=F64)
        s = t if self.basis is None else t @ self.basis.to(t).T
        return s[:, list(self.order)]

    def bounds(self, y_prefix, i):
        B = y_prefix.shape[0]
        return _full(B, 0, y_prefix), _full(B, 1, y_prefix)

    def map(self, u):
        u = torch.as_tensor(u, dtype=F64)
        B = u.shape[0]
        prev, ld = [], torch.zeros(B, dtype=F64, device=u.device)
        for i in range(len(self.order)):
            lo, hi = self.t_bounds(torch.stack(prev, 1) if prev else u.new_zeros(B, 0), i)
            prev.append(lo + u[:, i] * (hi - lo))
            ld = ld + torch.log(hi - lo)
        cols = [self.gauge[c].expand(B) for c in range(3)]
        for i, c in enumerate(self.order):
            cols[c] = prev[i]
        if self.basis is None:
            return torch.stack(cols, 1), ld
        t = torch.stack(cols, 1) @ self.basis_inv.to(u).T
        t = torch.where(self.free.to(t.device), self.gauge.to(t), t)
        return t, ld - self.log_det_basis

    def unmap(self, t):
        t = torch.as_tensor(t, dtype=F64)
        B = t.shape[0]
        s = t if self.basis is None else t @ self.basis.to(t).T
        us, ld = [], torch.zeros(B, dtype=F64, device=t.device)
        for i, c in enumerate(self.order):
            lo, hi = self.t_bounds(s[:, list(self.order[:i])], i)
            us.append((s[:, c] - lo) / (hi - lo))
            ld = ld - torch.log(hi - lo)
        if self.basis is not None:
            ld = ld + self.log_det_basis
        return (torch.stack(us, 1) if us else t.new_zeros(B, 0)), ld


def _translation_pairs(tab) -> tuple:
    """Every pair of facets of ASU(Gamma+) glued by a pure translation, once: (lower cut, upper cut, n, w).

    From ``tab.facets`` (kind 'translation'; its element must have W = I). ``n`` is the integer normal of the lower
    facet (first nonzero entry positive) and ``w`` (exact) the translation, lattice shift included, taking the lower
    facet onto the upper one (n . w > 0). RuntimeError if a pair is not two whole antiparallel facets.
    """
    raw = tab.raw
    cuts = raw['asu']['cuts']
    out, seen = [], set()
    for f in raw['facets']:
        for g in f['gluings']:
            if g['kind'] != 'translation':
                continue
            lower, upper = f['cut'], g['lands_on']
            if frozenset((lower, upper)) in seen:
                continue
            seen.add(frozenset((lower, upper)))
            n, n_up = list(cuts[lower]['n']), list(cuts[upper]['n'])
            if list(raw['gamma']['W'][g['element']]) != _EYE or len(f['gluings']) != 1 \
                    or to_fraction(g['share']) != 1 or n_up != [-v for v in n]:
                raise RuntimeError(f'sg {tab.sg}: the translation gluing of cut {lower} is not a whole facet glued to '
                                   'its antiparallel facet by a pure translation')
            w = [to_fraction(v) + s for v, s in zip(raw['gamma']['w'][g['element']], g['shift'])]
            if next(v for v in n if v != 0) < 0:
                lower, upper, n, w = upper, lower, n_up, [-v for v in w]
            out.append((lower, upper, tuple(n), tuple(w)))
    return tuple(out)


class _Shear(NamedTuple):
    rows: tuple          # M, exact integer rows (s = M t)
    inv: tuple           # M^-1, exact
    order: tuple         # AR order over s (non-free slots, seam rows first)
    stages: tuple        # exact.ar_stages of ASU(Gamma+) in s
    circle_slots: tuple  # the s slot of each _translation_pairs entry
    log_det: float       # log |det M|


def _primitive(v) -> tuple:
    from math import gcd
    g = 0
    for x in v:
        g = gcd(g, abs(int(x)))
    return tuple(int(x) // g for x in v)


def _row_name(row) -> str:
    """'x+y', 'x-y', 'z', '-x+y', '2x-y': the linear form of an integer row in t."""
    out = ''
    for v, a in zip(row, 'xyz'):
        v = int(v)
        if v:
            out += ('-' if v < 0 else ('+' if out else '')) + ('' if abs(v) == 1 else str(abs(v))) + a
    return out


def _ar_stage(st: dict, k: int, device):
    """tables.ARStage from an ``exact.ar_stages`` stage over k earlier AR coordinates."""
    from mxtaltools.crystal_fd.tables import ARStage

    def side(rows):
        coef = torch.tensor([[float(v) for v in coef] for coef, _ in rows], dtype=F64, device=device)
        return coef.reshape(len(rows), k), torch.tensor([float(c) for _, c in rows], dtype=F64, device=device)
    lc, lk = side(st['lower'])
    uc, uk = side(st['upper'])
    return ARStage(coord=int(st['coord']), lower_coef=lc, lower_const=lk, upper_coef=uc, upper_const=uk)


_SHEARS = {}


def _centroid_basis(sg: int):
    """The sheared AR basis of ``sg``'s centroid chart (``_Shear``), or None when ``_translation_circles`` already
    makes every translation seam a circle (the table's AR order and bounds are then used unchanged).

    Rows of M: each seam normal n (primitive), then rows r with r . w = 0 for every seam translation w, axes e_k
    first and then the other facet normals of ASU(Gamma+), until M is nonsingular on the non-free axes (unit rows on
    free axes). A unit row e_k sits in slot k, the others fill the remaining non-free slots in turn; the AR order is
    the seam slots, then the other non-free slots ascending. The bounds are ``exact.ar_stages`` of ASU(Gamma+) in s.
    Checked exactly: M w = (n . w) e_slot for every seam (the translation moves its circle coordinate by the period
    alone) and the seam coordinate's bounds are the two glued facets, constant, n . w apart.
    """
    sg = int(sg)
    if sg in _SHEARS:
        return _SHEARS[sg]
    from fractions import Fraction
    from mxtaltools.crystal_fd import exact
    tab = group_table(sg)
    pairs = _translation_pairs(tab)
    axis = _translation_circles(tab)
    if all(sum(1 for v in n if v != 0) == 1 and next(k for k in range(3) if n[k] != 0) in axis
           for _, _, n, _ in pairs):
        _SHEARS[sg] = None
        return None
    cuts = tab.raw['asu']['cuts']
    free = [bool(v) for v in tab.raw['free']]
    nonfree = [k for k in range(3) if not free[k]]
    ws = [w for *_, w in pairs]
    seam_rows = [_primitive(n) for _, _, n, _ in pairs]
    dot = lambda r, w: sum(Fraction(a) * b for a, b in zip(r, w))
    for i, r in enumerate(seam_rows):
        if any(dot(r, w) != 0 for j, w in enumerate(ws) if j != i):
            raise RuntimeError(f'sg {sg}: seam {i} moves under another seam translation; no sheared basis')
    units = [tuple(int(k == j) for j in range(3)) for k in nonfree]
    chosen = list(seam_rows)
    for r in units + [_primitive(c['n']) for c in cuts]:
        if len(chosen) == len(nonfree):
            break
        if any(r[k] != 0 for k in range(3) if free[k]) or any(dot(r, w) != 0 for w in ws):
            continue
        if exact.rank(chosen + [r]) > exact.rank(chosen):
            chosen.append(r)
    if len(chosen) != len(nonfree):
        raise RuntimeError(f'sg {sg}: no basis of rows orthogonal to the seam translations')
    rows, slot_of = [None] * 3, []
    for k in range(3):
        if free[k]:
            rows[k] = tuple(int(k == j) for j in range(3))
    is_unit = lambda r: sum(1 for v in r if v != 0) == 1 and max(r) == 1
    for r in chosen:
        k = r.index(1) if is_unit(r) else None
        slot_of.append(k if k is not None and rows[k] is None else None)
        if slot_of[-1] is not None:
            rows[k] = r
    open_slots = iter(k for k in nonfree if rows[k] is None)
    for j, r in enumerate(chosen):
        if slot_of[j] is None:
            slot_of[j] = next(open_slots)
            rows[slot_of[j]] = r
    seam_slots = tuple(slot_of[:len(pairs)])
    order = seam_slots + tuple(k for k in nonfree if k not in seam_slots)
    inv_cols = [exact.solve(rows, [int(i == j) for i in range(3)]) for j in range(3)]
    inv = tuple(tuple(inv_cols[j][i] for j in range(3)) for i in range(3))
    normals = [[sum(Fraction(c['n'][i]) * inv[i][j] for i in range(3)) for j in range(3)] for c in cuts]
    offsets = [to_fraction(c['c']) for c in cuts]
    stages = exact.ar_stages(normals, offsets, list(order), eliminate=[k for k in range(3) if free[k]])
    for (lo_cut, _, n, w), r, slot in zip(pairs, seam_rows, seam_slots):
        Mw = [dot(rows[k], w) for k in range(3)]
        period = dot(r, w)
        st = stages[order.index(slot)]
        if not (period > 0 and Mw == [period if k == slot else 0 for k in range(3)]
                and len(st['lower']) == 1 and len(st['upper']) == 1
                and all(v == 0 for side in ('lower', 'upper') for v in st[side][0][0])
                and st['upper'][0][1] - st['lower'][0][1] == period):
            raise RuntimeError(f'sg {sg}: the sheared basis does not make the seam of cut {lo_cut} a circle')
    det = exact.det3(*[[Fraction(v) for v in r] for r in rows])
    _SHEARS[sg] = _Shear(tuple(rows), inv, order, tuple(stages), seam_slots, math.log(abs(det)))
    return _SHEARS[sg]


def _translation_circles(tab) -> set:
    """Coordinates c whose AR bounds are constant and whose lower facet is glued to the upper by t + (hi - lo) e_c."""
    raw = tab.raw
    cuts = raw['asu']['cuts']
    out = set()
    for i, (c, st) in enumerate(zip(raw['ar']['order'], raw['ar']['stages'])):
        if len(st['lower']) != 1 or len(st['upper']) != 1:
            continue
        if any(to_fraction(v) != 0 for side in ('lower', 'upper') for v in st[side][0]['coef']):
            continue
        lo, hi = to_fraction(st['lower'][0]['const']), to_fraction(st['upper'][0]['const'])
        for f in raw['facets']:
            n = cuts[f['cut']]['n']
            if not (n[c] > 0 and all(n[k] == 0 for k in range(3) if k != c)) or len(f['gluings']) != 1:
                continue
            g = f['gluings'][0]
            if g['kind'] != 'translation':
                continue
            W = raw['gamma']['W'][g['element']]
            w = [to_fraction(v) + s for v, s in zip(raw['gamma']['w'][g['element']], g['shift'])]
            n_land = cuts[g['lands_on']]['n']
            if (list(W) == _EYE and w[c] == hi - lo
                    and all(w[k] == 0 for k in range(3) if k != c)
                    and n_land[c] < 0 and all(n_land[k] == 0 for k in range(3) if k != c)):
                out.add(c)
    return out


_ORIENTATION_DIMS = {
    'hemisphere': (Dim('theta', 'interval', None, 0), Dim('phi', 'circle', 2 * math.pi, 1),
                   Dim('r', 'interval', None, 2)),
    'euler_zyz': (Dim('alpha', 'circle', 2 * math.pi, 0), Dim('u', 'interval', None, 1),
                  Dim('gamma', 'circle', 2 * math.pi, 2)),
    'axis_angle': (Dim('v_x', 'interval', None, 0), Dim('v_y', 'interval', None, 1),
                   Dim('v_z', 'interval', None, 2)),
    'quaternion': (Dim('w', 'embedded', None, 0), Dim('x', 'embedded', None, 1), Dim('y', 'embedded', None, 2),
                   Dim('z', 'embedded', None, 3)),
}


class OrientationChart(_BlockChart):
    """Orientation chart ``name`` (one of ``so3.CHARTS``; there is no default): ``so3`` conversions and Haar densities.

    - ``hemisphere`` (theta, phi, r): theta in [0, pi/2] (the equator is glued to
      itself by (phi, r) -> (phi + pi, 2 pi - r)), phi a circle, r in [0, 2 pi];
    - ``euler_zyz`` (alpha, u = cos beta, gamma): alpha, gamma circles, u in [-1, 1];
    - ``axis_angle`` v in the ball |v| <= pi (antipodes glued on the sphere), AR bounds
      v_y in +-sqrt(pi^2 - v_x^2), v_z in +-sqrt(pi^2 - v_x^2 - v_y^2);
    - ``quaternion`` (w, x, y, z), w >= 0: an embedding, density relative to the
      surface measure of the hemisphere of S^3; it has no AR bounds.
    """
    field = 'R'

    def __init__(self, name: str):
        if name not in so3.CHARTS:
            raise ValueError(f'unknown orientation chart {name!r}; choose one of {so3.CHARTS}')
        self.name = name
        self.dims = _ORIENTATION_DIMS[name]

    def map(self, y):
        y = torch.as_tensor(y, dtype=F64)
        return so3.from_chart(self.name, y), so3.haar_log_density(self.name, y)

    def unmap(self, R):
        y = so3.to_chart(self.name, torch.as_tensor(R, dtype=F64))
        return y, -so3.haar_log_density(self.name, y)

    def bounds(self, y_prefix, i):
        B = y_prefix.shape[0]
        pi = math.pi
        if self.name == 'hemisphere':
            lo, hi = ((0, pi / 2), (-pi, pi), (0, 2 * pi))[i]
        elif self.name == 'euler_zyz':
            lo, hi = ((0, 2 * pi), (-1, 1), (0, 2 * pi))[i]
        elif self.name == 'axis_angle':
            rad = torch.sqrt((pi ** 2 - (y_prefix[:, :i] ** 2).sum(1)).clamp(min=0))
            return -rad, rad
        else:
            raise NotImplementedError('the quaternion chart is an embedding of S^3; it has no AR bounds')
        return _full(B, lo, y_prefix), _full(B, hi, y_prefix)


#: Q_K of sg 6-9 in MXtalTools' Cartesian frame (a along x, b in the xy plane): the 2-fold about b, which is Cartesian
#: y for every cell with alpha = gamma = 90 deg. ``KCutHemisphereChart``'s bound assumes it; ``CrystalChart`` checks it
#: per row against ``k_rotation``.
K_B_ROTATION = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
#: the fractional W of K's 2-fold in sg 6-9 (row-major, as the table stores it)
_W_K_B = [-1, 0, 0, 0, 1, 0, 0, 0, -1]


def k_element(sg: int) -> int:
    """Index into ``group_table(sg).gamma_W`` of the non-identity element of K (sg 6-9); ValueError where K is
    trivial, NotImplementedError where K has more than two elements (no group in the table)."""
    tab = group_table(int(sg))
    ks = [int(i) for i in tab.kernel if int(i) != int(tab.gamma_plus[0])]
    if not ks:
        raise ValueError(f'sg {sg}: K is trivial')
    if len(ks) > 1:
        raise NotImplementedError(f'sg {sg}: |K| = {len(ks) + 1}; the K cut is built for |K| = 2')
    return ks[0]


def k_rotation(sg: int, cell) -> torch.Tensor:
    """[B, 3, 3] Q_K = T_fc W_K T_fc^-1 of each cell [B, 6] (projected onto SO(3), as ``domains.fold`` acts), W_K the
    group table's non-identity element of K (``k_element``). For a monoclinic cell (alpha = gamma = 90 deg) in
    MXtalTools' frame this is ``K_B_ROTATION``: with W_K = diag(-1, 1, -1) = -I + 2 e_b e_b^T,
    Q_K = -I + 2 b b*^T (b = T e_b, b* = e_b^T T^-1 the reciprocal row), and b* = b / |b|^2 when b is orthogonal to a
    and c, so Q_K = -I + 2 b^ b^T, the 2-fold about b^ = e_y (b lies along y when gamma = 90 deg)."""
    from mxtaltools.crystal_fd.ops import cartesian_rotation, cell_matrices
    W = group_table(int(sg)).gamma_W[k_element(sg)]
    cell = torch.as_tensor(cell, dtype=F64)
    return cartesian_rotation(cell_matrices(cell), W.to(cell.device))


class KCutHemisphereChart(OrientationChart):
    """SO(3)/K for sg 6-9 (K = {I, Q_K}, Q_K = ``K_B_ROTATION``): the hemisphere chart with r read as a signed angle
    and cut to the fold's half of SO(3). Every orientation of the fundamental domain once (up to its boundary).

    Coordinates (AR order) ``(theta, phi, r)``: the axis n = (sin theta cos phi, sin theta sin phi, cos theta),
    theta in [0, pi/2], phi a circle of period 2 pi, and the signed angle r in [-rho(theta, phi), rho(theta, phi)],
    R = rotation by r about n (r < 0 is the trainer's r + 2 pi, the same rotation: ``so3.matrix_to_hemisphere``'s
    r in (pi, 2 pi] is read as r - 2 pi, so r runs over (-pi, pi]).

    The cut. ``domains.fold``'s rule 2 keeps, of R and Q_K R, the lexicographically larger matrix (row-major). Q_K's
    first row is (-1, 0, 0), so (Q_K R)[0, :] = -R[0, :] and the first key compares R00 with -R00: the fold keeps
    R00 > 0 (within its tie band |R00| <= tol/2 the next entries decide; measure zero, and both points lie on this
    chart's boundary). Gamma+ of sg 6-9 is {I, t(0, 1/2, 0), 2_b, 2_b t(0, 1/2, 0)}: the elements outside K translate
    t_y by 1/2 with rotation parts I and Q_K, so no t_y has a larger stabiliser in SO(3), and the fold cuts the same
    half R00 >= 0 at every t_y, the faces of ASU(Gamma+) included. With
    u_x = n_x = sin theta cos phi, R00 = cos r + u_x^2 (1 - cos r) = (1 - u_x^2) cos r + u_x^2, so R00 >= 0 iff
    cos r >= -u_x^2 / (1 - u_x^2), i.e. |r| <= rho = arccos(-u_x^2 / (1 - u_x^2)) where u_x^2 < 1/2, and every r
    where u_x^2 >= 1/2 (rho = pi). Written without the ill-conditioned arccos near -1:
    rho = 2 atan2(1, sqrt(max(0, 1 - 2 u_x^2))) (arccos c = 2 atan2(sqrt(1 - c), sqrt(1 + c)), and here
    1 - c = 1 / (1 - u_x^2), 1 + c = (1 - 2 u_x^2) / (1 - u_x^2)). rho runs from pi/2 (axis along y or z) to pi,
    computed in float64 whatever the prefix's dtype. It has a square-root cusp where u_x^2 = 1/2 (infinite slope in
    theta and phi): there a perturbation eps of the axis moves rho by about sqrt(eps), so a float32-stored row can
    lie up to about 1e-3 rad in r outside the bound of its own float32 prefix, and d rho / d(theta, phi) is
    unbounded; a float32 or differentiable consumer keeps the orientation row in float64 or clamps r to its own
    bound. The bound holds for every K whose 2-fold axis is perpendicular to Cartesian x; ``CrystalChart`` checks
    Q_K = ``K_B_ROTATION`` per row within ``FRAME_TOL``.

    Measure. Haar's density in (theta, phi, r) is sin^2(r/2) sin(theta) / (2 pi^2), even in r, so the hemisphere
    one (``so3.haar_log_density('hemisphere', y)``) holds for signed r; the cut has Haar mass 1/2:
    (1 / 2 pi^2) * integral of (rho - sin rho) sin theta dtheta dphi = 1/2.

    Gluings (seams for a consumer): the equator theta = pi/2 to itself by (phi, r) -> (phi + pi, -r) (the trainer's
    (phi + pi, 2 pi - r); rho is unchanged by it); r = -pi to r = pi where rho = pi (one rotation); the cut faces
    |r| = rho, where R00 = 0, to themselves by R -> Q_K R. A point on the cut face is fold-canonical only when the
    fold's later keys favour it (about half of them): a consumer whose outputs can land on |r| = rho (a bounded AR
    row saturating at its end) canonicalises them before using them as canonical forms.

    ``unmap`` (``inverse``) decides between R and Q_K R by the fold's own rule (``fold_keeps``, rule 2 at
    ``FOLD_TOL``, on R as given), and charts (1) R where the fold keeps it, (2) Q_K R where the fold keeps Q_K R
    by a later entry (R00 within the fold's tie band, R00 >= -``BAND_TOL``), and (3) R as it is otherwise: an image
    the fold rejects by its first key (R00 < -``BAND_TOL``), outside the bounds. In (1) and (2) the coordinates are
    those of the rotation nearest the charted matrix (``so3.nearest_rotation``), turned onto the face by
    ``onto_face`` where its R00 is below 0 (the fold's choice lies within the closed cut, R00 >= -tol/2, so this is
    the tie band, rounding, or the non-orthogonality of R; geodesic move arcsin|R00|), and |r| left above rho by
    rounding (up to about 1e-8, at the cusp u_x^2 = 1/2, |r| = pi) is put on rho. Snapping r alone instead would move
    R by about 2 sqrt|R00| at the cusp, where r runs along the face. So every row the fold keeps charts inside the
    bounds, and ``forward(inverse(row))`` returns the row itself within its distance from SO(3), plus |R00| for a
    tie-band row (put on the face), plus about 1e-8.
    """
    #: ``domains.fold``'s default tolerance (on R entries); ``unmap`` applies the fold's rule 2 at this tolerance
    FOLD_TOL = 1e-9
    #: the fold's tie band on R00 at ``FOLD_TOL``: its first key (R00 against -R00) is within tolerance where
    #: |R00| <= FOLD_TOL / 2, and a later entry decides
    BAND_TOL = FOLD_TOL / 2
    #: ``CrystalChart`` refuses rows whose Q_K (``k_rotation``) differs from ``K_B_ROTATION`` by more than this. The
    #: chart reads every row at alpha = gamma = 90 deg (the monoclinic cell chart ignores both, and ``canonicalize``
    #: snaps them within its ``angle_tol`` = 1e-6 rad before the fold), where Q_K is ``K_B_ROTATION`` exactly. On
    #: random monoclinic cells (beta 90-130 deg) Q_K's deviation measured 1.31 times an alpha offset and 1.0 times a
    #: gamma offset; float32 storage leaves both angles about 4.4e-8 rad off 90 deg and Q_K up to 9.3e-8 off.
    FRAME_TOL = 1e-6
    cut = 'K'

    def __init__(self, sg: int):
        sg = int(sg)
        k = k_element(sg)
        W = group_table(sg).raw['gamma']['W'][k]
        if list(W) != _W_K_B:
            raise NotImplementedError(f'sg {sg}: K is {W}, not the 2-fold about b this chart cuts')
        super().__init__('hemisphere')
        self.sg = sg
        self.k_element = k
        self.Q_K = torch.tensor(K_B_ROTATION, dtype=F64)
        self.dims = (Dim('theta', 'interval', None, 0), Dim('phi', 'circle', 2 * math.pi, 1),
                     Dim('r_signed', 'interval', None, 2))

    @staticmethod
    def rho(theta, phi) -> torch.Tensor:
        """[B] float64: the bound on |r| at the axis (theta, phi): 2 atan2(1, sqrt(max(0, 1 - 2 u_x^2))),
        u_x = sin theta cos phi (computed in float64 whatever the inputs' dtype)."""
        theta, phi = torch.as_tensor(theta, dtype=F64), torch.as_tensor(phi, dtype=F64)
        ux = torch.sin(theta) * torch.cos(phi)
        return 2 * torch.atan2(torch.ones_like(ux), torch.sqrt((1 - 2 * ux * ux).clamp(min=0)))

    def k_image(self, R) -> torch.Tensor:
        """[B, 3, 3] Q_K R."""
        R = torch.as_tensor(R, dtype=F64)
        return self.Q_K.to(R.device) @ R

    def fold_keeps(self, R) -> torch.Tensor:
        """[B] bool: whether ``domains.fold``'s rule 2 (at ``FOLD_TOL``) keeps R rather than Q_K R. The two poses
        share their centroid (K acts trivially on it), so the pose keys decide, R first in candidate order."""
        from mxtaltools.crystal_fd.ops import _lex_select
        R = torch.as_tensor(R, dtype=F64)
        keys = torch.stack([R.reshape(-1, 9), self.k_image(R).reshape(-1, 9)], 1)
        valid = torch.ones(keys.shape[:2], dtype=torch.bool, device=R.device)
        return _lex_select(keys, valid, self.FOLD_TOL, largest=True) == 0

    def bounds(self, y_prefix, i):
        B = y_prefix.shape[0]
        if i == 0:
            return _full(B, 0, y_prefix), _full(B, math.pi / 2, y_prefix)
        if i == 1:
            return _full(B, -math.pi, y_prefix), _full(B, math.pi, y_prefix)
        rho = self.rho(y_prefix[:, 0], y_prefix[:, 1])
        return -rho, rho

    @staticmethod
    def onto_face(R) -> torch.Tensor:
        """[B, 3, 3] the nearest rotation to each R [B, 3, 3] on the cut face R00 = 0: R turned by the smallest
        rotation taking its first column c = R e_x to (0, c_y, c_z) / |(c_y, c_z)|, a geodesic distance arcsin|R00|
        (a rotation by w moves c by at most w, and c is arcsin|c_x| from the yz plane). Rows with c = +-e_x (R00 =
        +-1, far from the face) are returned unchanged."""
        R = torch.as_tensor(R, dtype=F64)
        c = R[:, :, 0]
        s = torch.sqrt(c[:, 1] ** 2 + c[:, 2] ** 2)
        ok = s > 0
        s = torch.where(ok, s, torch.ones_like(s))
        cp = torch.stack([torch.zeros_like(s), c[:, 1] / s, c[:, 2] / s], 1)
        v = torch.linalg.cross(c, cp)                            # sin(angle) times the axis
        cos = (c * cp).sum(1)                                    # = |(c_y, c_z)| > 0
        z = torch.zeros_like(s)
        V = torch.stack([z, -v[:, 2], v[:, 1], v[:, 2], z, -v[:, 0], -v[:, 1], v[:, 0], z], 1).reshape(-1, 3, 3)
        eye = torch.eye(3, dtype=F64, device=R.device).expand_as(V)
        turn = eye + V + (V @ V) / (1 + cos)[:, None, None]      # Rodrigues: the rotation taking c to cp
        return torch.where(ok[:, None, None], turn @ R, R)

    def unmap(self, R):
        R = torch.as_tensor(R, dtype=F64)
        keeps = self.fold_keeps(R)
        tie = ~keeps & (R[:, 0, 0] >= -self.BAND_TOL)          # the fold took Q_K R by a later entry
        charted = keeps | tie
        Rc = torch.where(tie[:, None, None], self.k_image(R), R)
        if Rc.shape[0]:
            Rc = torch.where(charted[:, None, None], so3.nearest_rotation(Rc), Rc)
            Rc = torch.where((charted & (Rc[:, 0, 0] < 0))[:, None, None], self.onto_face(Rc), Rc)
        hs = so3.matrix_to_hemisphere(Rc)
        theta, phi, r = hs.unbind(1)
        r = torch.where(r > math.pi, r - 2 * math.pi, r)
        rho = self.rho(theta, phi)
        snap = charted & (r.abs() > rho)                         # rounding only; |r| > rho >= pi/2, sign(r) = +-1
        r = torch.where(snap, torch.sign(r) * rho, r)
        y = torch.stack([theta, phi, r], 1)
        return y, -so3.haar_log_density('hemisphere', y)


class CellGluing(NamedTuple):
    """Two faces of an orthorhombic cell chart glued by an allowed axis order (a seam; each side an interval end).

    A cell y on face ``face`` (``faces[face] . y = 0``) is the lattice ``y[perm]`` on face ``lands_on``, re-described
    by order ``order`` of ``axis_order.AxisOrders`` (``perm = perm[order]``; its (N, s) carries the pose along).
    """
    face: int
    lands_on: int
    order: int
    perm: tuple


def _face_point(A: torch.Tensor, i: int) -> torch.Tensor:
    """A generic point of face i of {A y >= 0}: the first small-integer y on it strictly inside every other face."""
    import itertools
    for v in itertools.product((1., 2., 3.), repeat=A.shape[1]):
        y = torch.tensor(v, dtype=F64)
        f = A @ y
        if float(f[i]) == 0 and all(float(f[k]) > 0 for k in range(A.shape[0]) if k != i):
            return y
    raise RuntimeError(f'no generic point on face {i} of {A.tolist()}')


def _cell_gluings(orders) -> tuple:
    """(face_kinds, gluings) of an orthorhombic class, derived from its faces and allowed orders.

    For each face, a generic point on it is moved by every allowed order: an order other than the identity that fixes
    it makes the face a mirror (a real wall); otherwise an order landing it on another face of the closed domain glues
    the two (a seam, ``CellGluing``). RuntimeError if a face is neither, or if the kinds disagree with
    ``AxisOrders.mirror``.
    """
    A = orders.faces
    kinds, gluings = [], []
    for i in range(A.shape[0]):
        y = _face_point(A, i)
        moved = [y[orders.perm[j]] for j in range(orders.k)]
        if any(torch.equal(moved[j], y) for j in range(1, orders.k)):
            kinds.append('mirror')
            continue
        for j in range(1, orders.k):
            f = A @ moved[j]
            on = [k for k in range(A.shape[0]) if k != i and float(f[k]) == 0]
            if on and bool((f >= 0).all()):
                gluings.append(CellGluing(i, on[0], j, tuple(int(p) for p in orders.perm[j])))
                break
        else:
            raise RuntimeError(f'sg {orders.sg}: face {i} of the axis-order domain is neither a mirror nor glued')
        kinds.append('seam')
    if any((k == 'mirror') != orders.mirror for k in kinds):
        raise RuntimeError(f'sg {orders.sg}: face kinds {kinds} disagree with AxisOrders.mirror = {orders.mirror}')
    return tuple(kinds), tuple(gluings)


class CellChart(_BlockChart):
    """Canonical-cell chart of one family within a length box: triclinic and monoclinic, the margin-0 zero set of the
    ``sym_utils`` penalty; from orthorhombic up, the cells ``canon``'s cell step returns (exact metric, orthorhombic
    axis order).

    **Triclinic** (``family='triclinic'``, R3's map; sg 1, 2). Coordinates (AR order):
    ``log c``; ``r2 = b/c``; ``r1 = a/b``; ``s_beta``, ``s_gamma`` in [-1, 0]; ``v`` in
    [0, 1], with ``Lambda = -1 + r1^2 max(0, -1 - s_beta - s_gamma)``,
    ``s_alpha = Lambda + (1 - Lambda) v`` and cos alpha = s_alpha r2 / 2,
    cos beta = s_beta r1 r2 / 2, cos gamma = s_gamma r1 / 2. These are exactly the walls of
    ``sym_utils.tri_niggli_reduction_penalty`` at margin 0 (a <= b <= c, the three
    pairwise bounds, beta and gamma obtuse, the three-vector cut). The length box adds
    a in [lo_a, hi_a], b in [lo_b, hi_b], c in [lo_c, hi_c].

    log|det d(log a, log b, log c, alpha, beta, gamma)/dy|
    = log((1 - Lambda) r1 r2 / (8 sin alpha sin beta sin gamma)); the Jacobian is lower
    triangular in the order (log c, log b, log a, beta, gamma, alpha).

    **Monoclinic** (``family='monoclinic'``; ``sg`` 3-15 names the setting class of
    ``sym_utils.MONO_CLASS``, and one chart serves every group of a class). Coordinates
    (AR order): ``log c``, ``log a``, ``log b``, ``s`` in [0, 1], with alpha = gamma = pi/2
    and cos beta = -s m(a, c), m the class bound of ``ops.mono_bound`` (GLIDE
    min(a/c, c/2a), CENTRED min(c/a, a/2c), FREE a/2c). In the trainer's angle latent
    x_4 (beta = pi/2 + 0.3 pi x_4) this is s = sin(0.3 pi x_4) / m. The margin-0 zero
    set of ``sym_utils.mono_reduction_penalty`` is alpha = gamma = pi/2,
    -m <= cos beta <= 0, and a <= c for FREE (``ops.mono_conditions``); s in [0, 1]
    is the beta part, and the length bounds carry the rest: log c in
    [log max(lo_c, lo_a), log hi_c] and log a in [log lo_a, log min(hi_a, c)] for FREE,
    the box alone otherwise; log b in [log lo_b, log hi_b]. ``unmap`` puts s and (FREE)
    log a on their bound when they lie within ``SLACK`` outside it, which is where
    rounding leaves a canonical cell on a face (beta = pi/2 gives s = -6e-17 / m).

    Its natural measure has no alpha or gamma row (both are fixed):
    d(log a) d(log b) d(log c) d(beta). Derivation: (log c, log a, log b) are copied and
    beta = arccos(-s m(a, c)) depends on s and, through m, on log a and log c only, so
    the Jacobian of y -> (log c, log a, log b, beta) is lower triangular with diagonal
    (1, 1, 1, d beta / d s), and d beta / d s = m / sqrt(1 - s^2 m^2) = m / sin beta:
    log|det| = log m - log sin beta. For GLIDE and CENTRED m is the smaller of two smooth
    terms (a kink at c^2 = 2 a^2, resp. a^2 = 2 c^2); the kink enters below the diagonal
    only, so the determinant is continuous across it.

    **From orthorhombic up** (2026-10-21; owner: one coordinate per free cell parameter,
    designed for efficiency, not to mirror the trainer's latent). Every angle is fixed
    (exactly ``HALF_PI``, gamma ``HEX_GAMMA`` in the hexagonal family) and the
    coordinates are the free log lengths, so ``map`` is exact (b = a, c = a bit for bit)
    and log|det| = 0 relative to the natural measure, d of each free log length:

    - ``'orthorhombic'`` (``sg`` 16-74 names the axis-order class of
      ``axis_order.axis_orders``; one chart serves every group of a class): (log a,
      log b, log c), the columns of ``axis_order.OrthoCellChart``, with the canonical
      domain {y : ``faces`` y >= 0} of ``axis_order.choose``: no face (one allowed order),
      log a <= log b (a <-> b), log a <= log b <= log c (all six orders), Pbca (61) the
      shortest-first wedge log a <= log b, log a <= log c. AR bounds, the length box cut
      by the faces: log a in [log lo_a, log hi_a], the top lowered to log hi_b where
      a <= b and to log hi_c where a <= c; log b in [log lo_b, log hi_b], the bottom
      raised to log a where a <= b and the top lowered to log hi_c where b <= c; log c in
      [log lo_c, log hi_c], the bottom raised to log b (six orders) or log a (Pbca). Mirror
      faces (``face_kinds`` 'mirror': an allowed transposition fixes the face pointwise)
      and Pbca's two seams are interval ends (owner O1 (a)); the seams are glued by the
      3-cycle (``gluings``: y on ``face`` is the lattice y[perm] on ``lands_on``, and the
      pose goes with the order's (N, s)), and they meet at the cone a = b = c.
    - ``'tetragonal'`` (75-142): (log a, log c), b = a; ``'hexagonal'`` (143-194: trigonal
      P and R groups, on hexagonal axes, and hexagonal groups): (log a, log c), b = a,
      gamma = 2 pi / 3; ``'cubic'`` (195-230): (log a,), b = c = a. Bounds: log a in
      [log max(lo_a, lo_b), log min(hi_a, hi_b)] (cubic: over all three axes), log c in
      [log lo_c, log hi_c]. No canonicaliser step chooses among these cells (``canon``
      only snaps them to the exact metric), so the domain is the box.

    ``unmap`` reads the log lengths (a for b = a and c = a) and refuses cells further than
    ``FAMILY_TOL`` from the family's metric (relative lengths, angles in rad); it puts a
    log length within ``SLACK`` below its face bound on it (the rounding of a canonical
    cell on a face, as for the monoclinic chart). ``faces`` [m, n] holds the face rows of
    the shape bounds (none outside orthorhombic).

    Parameters
    ----------
    family : str
        One of ``FAMILIES``; ``cell_family(sg)`` names the family of a space group.
    length_box : (lo, hi)
        [3] or [B, 3] bounds on (a, b, c) in Angstrom (``latent_v1_length_box`` gives
        the trainer's; from orthorhombic up it is used as a box only).
    sg : int, optional
        Required for the monoclinic (3-15: a group of the setting class) and orthorhombic
        (16-74: a group of the axis-order class) charts; elsewhere optional and checked
        against the family; ignored for triclinic.
    """
    field = 'cell'
    FAMILIES = ('triclinic', 'monoclinic') + HIGHER_FAMILIES
    #: monoclinic ``unmap``: s within SLACK below 0 or above 1, and (FREE) log a within SLACK above log c, are put on
    #: the bound (rounding of a canonical cell on a face); larger excursions are returned unchanged. Orthorhombic: a log
    #: length within SLACK below its face bound is put on it.
    SLACK = 1e-9
    #: orthorhombic and up: ``unmap`` refuses a cell whose fixed angles or equal lengths are further than this from the
    #: family's metric (lengths relative, angles in rad)
    FAMILY_TOL = 1e-6

    def __init__(self, family: str, length_box, sg: Optional[int] = None):
        if family not in self.FAMILIES:
            raise ValueError(f'CellChart({family!r}): the families are {self.FAMILIES}')
        lo, hi = (torch.as_tensor(v, dtype=F64) for v in length_box)
        self.family = family
        if family in HIGHER_FAMILIES:
            self._init_higher(lo, hi, sg)
            return
        if family == 'triclinic':
            if bool((hi[..., 2] < torch.maximum(lo[..., :2].amax(-1), lo[..., 2])).any()) \
                    or bool((hi[..., 1] < lo[..., :2].amax(-1)).any()) or bool((hi < lo).any()):
                raise ValueError('the length box leaves no reduced cell (need hi_c >= lo_a, lo_b, lo_c and hi_b >= lo_a, lo_b)')
            self.lo, self.hi = lo, hi
            self.dims = tuple(Dim(n, 'interval', None, i) for i, n in
                              enumerate(('log_c', 'r2', 'r1', 's_beta', 's_gamma', 'v')))
            return
        if sg is None or not 3 <= int(sg) <= 15:
            raise ValueError("CellChart('monoclinic') needs sg in 3-15 (its setting class)")
        self.sg = int(sg)
        self.mono_class = int(MONO_CLASS[self.sg])
        if bool((hi < lo).any()) or (self.mono_class == MONO_FREE and bool((hi[..., 2] < lo[..., 0]).any())):
            raise ValueError('the length box leaves no reduced cell (need lo <= hi, and hi_c >= lo_a for a <= c)')
        self.lo, self.hi = lo, hi
        self.dims = tuple(Dim(n, 'interval', None, i) for i, n in enumerate(('log_c', 'log_a', 'log_b', 's')))

    def _init_higher(self, lo: torch.Tensor, hi: torch.Tensor, sg: Optional[int]):
        fam = self.family
        if fam == 'orthorhombic':
            if sg is None or cell_family(sg) != 'orthorhombic':
                raise ValueError("CellChart('orthorhombic') needs sg in 16-74 (its axis-order class)")
            self.sg = int(sg)
            self.orders = axis_order.axis_orders(self.sg)
            self.axis_order_kind = self.orders.kind
            self.faces = self.orders.faces.clone()
            self.face_kinds, self.gluings = _cell_gluings(self.orders)
            names = ('log_a', 'log_b', 'log_c')
        else:
            if sg is not None and cell_family(sg) != fam:
                raise ValueError(f'sg {sg} is not {fam}')
            self.sg = None if sg is None else int(sg)
            self.axis_order_kind = None
            names = ('log_a',) if fam == 'cubic' else ('log_a', 'log_c')
            self.faces = torch.zeros(0, len(names), dtype=F64)
            self.face_kinds, self.gluings = (), ()
        ok = bool((hi >= lo).all())
        if fam in ('tetragonal', 'hexagonal'):
            ok &= bool((torch.maximum(lo[..., 0], lo[..., 1]) <= torch.minimum(hi[..., 0], hi[..., 1])).all())
        elif fam == 'cubic':
            ok &= bool((lo.amax(-1) <= hi.amin(-1)).all())
        elif self.axis_order_kind != 'one':    # log a below hi_b (and hi_c); six orders also log lo_b below hi_c
            ok &= bool((lo[..., 0] <= hi[..., 1]).all())
            if self.axis_order_kind in ('all', 'cyclic'):
                ok &= bool((lo[..., 0] <= hi[..., 2]).all())
            if self.axis_order_kind == 'all':
                ok &= bool((lo[..., 1] <= hi[..., 2]).all())
        if not ok:
            raise ValueError(f'the length box leaves no {fam} cell of the chart (the bounds of every coordinate must be '
                             'nonempty: see the CellChart docstring)')
        self.lo, self.hi = lo, hi
        self.dims = tuple(Dim(n, 'interval', None, i) for i, n in enumerate(names))

    def _bounds_higher(self, y_prefix, i):
        B = y_prefix.shape[0]
        lo, hi = self._box(B, y_prefix.device)
        llo, lhi = torch.log(lo), torch.log(hi)
        fam, kind = self.family, self.axis_order_kind
        if fam == 'cubic':
            return llo.amax(1), lhi.amin(1)
        if fam in ('tetragonal', 'hexagonal'):
            if i == 0:
                return torch.maximum(llo[:, 0], llo[:, 1]), torch.minimum(lhi[:, 0], lhi[:, 1])
            return llo[:, 2], lhi[:, 2]
        if i == 0:
            up = {'one': lhi[:, 0], 'swap_ab': lhi[:, :2].amin(1)}.get(kind, None)
            return llo[:, 0], lhi.amin(1) if up is None else up
        if i == 1:
            return (llo[:, 1] if kind == 'one' else torch.maximum(llo[:, 1], y_prefix[:, 0]),
                    torch.minimum(lhi[:, 1], lhi[:, 2]) if kind == 'all' else lhi[:, 1])
        below = {'all': y_prefix[:, 1], 'cyclic': y_prefix[:, 0]}.get(kind, None)
        return llo[:, 2] if below is None else torch.maximum(llo[:, 2], below), lhi[:, 2]

    def _map_higher(self, y):
        B, dev = y.shape[0], y.device
        half = torch.full((B,), HALF_PI, dtype=F64, device=dev)
        L = torch.exp(y)
        if self.family == 'orthorhombic':
            cols = [L[:, 0], L[:, 1], L[:, 2], half, half, half]
        elif self.family == 'cubic':
            cols = [L[:, 0], L[:, 0], L[:, 0], half, half, half]
        else:
            gamma = half if self.family == 'tetragonal' else torch.full((B,), HEX_GAMMA, dtype=F64, device=dev)
            cols = [L[:, 0], L[:, 0], L[:, 1], half, half, gamma]
        return torch.stack(cols, -1), torch.zeros(B, dtype=F64, device=dev)

    def family_deviation(self, cell) -> torch.Tensor:
        """[B] orthorhombic and up: largest deviation of ``cell`` [B, 6] from the family's metric (equal lengths
        relative to a, fixed angles in rad); 0 on the chart's image."""
        cell = torch.as_tensor(cell, dtype=F64)
        a, b, c = cell[:, 0], cell[:, 1], cell[:, 2]
        gamma = HEX_GAMMA if self.family == 'hexagonal' else HALF_PI
        dev = torch.stack([(cell[:, 3] - HALF_PI).abs(), (cell[:, 4] - HALF_PI).abs(), (cell[:, 5] - gamma).abs()], 1)
        if self.family != 'orthorhombic':
            dev = torch.cat([dev, ((b - a) / a).abs()[:, None]], 1)
        if self.family == 'cubic':
            dev = torch.cat([dev, ((c - a) / a).abs()[:, None]], 1)
        return dev.amax(1)

    def _unmap_higher(self, cell):
        dev = self.family_deviation(cell)
        if bool((dev > self.FAMILY_TOL).any()):
            bad = torch.nonzero(dev > self.FAMILY_TOL)[:, 0]
            raise ValueError(f'CellChart({self.family!r}): {len(bad)} cells (rows {bad[:8].tolist()}) are further than '
                             f'FAMILY_TOL = {self.FAMILY_TOL} from the family metric (largest {float(dev.max()):.3g}); '
                             "canonicalise them first (canon snaps cells to the group's exact metric)")
        logL = torch.log(cell[:, :3])
        la, lb, lc = logL.unbind(-1)
        B = cell.shape[0]
        zero = torch.zeros(B, dtype=F64, device=cell.device)
        if self.family == 'cubic':
            return la[:, None], zero
        if self.family != 'orthorhombic':
            return torch.stack([la, lc], -1), zero
        kind = self.axis_order_kind
        if kind != 'one':
            lb = self._onto(lb, la)
        if kind == 'all':
            lc = self._onto(lc, lb)
        elif kind == 'cyclic':
            lc = self._onto(lc, la)
        return torch.stack([la, lb, lc], -1), zero

    def _onto(self, x, bound):
        """x, with values within SLACK below ``bound`` put on it (rounding of a canonical cell on a face)."""
        return torch.where((x < bound) & (x >= bound - self.SLACK), bound, x)

    @staticmethod
    def lam(r1, s_beta, s_gamma):
        """Lower bound of s_alpha: the three-vector cut b^2 (1 + s_alpha) + a^2 (1 + s_beta + s_gamma) >= 0."""
        return -1 + r1 ** 2 * (-1 - s_beta - s_gamma).clamp(min=0)

    def _box(self, B, dev):
        return self.lo.to(dev).expand(B, 3), self.hi.to(dev).expand(B, 3)

    def _mono_bound(self, a, c):
        from mxtaltools.crystal_fd.ops import mono_bound
        return mono_bound(a, c, torch.full(a.shape, self.mono_class, dtype=torch.long, device=a.device))

    def bounds(self, y_prefix, i):
        if self.family in HIGHER_FAMILIES:
            return self._bounds_higher(y_prefix, i)
        B = y_prefix.shape[0]
        lo, hi = self._box(B, y_prefix.device)
        if self.family == 'monoclinic':
            free = self.mono_class == MONO_FREE
            if i == 0:
                return torch.log(torch.maximum(lo[:, 2], lo[:, 0]) if free else lo[:, 2]), torch.log(hi[:, 2])
            if i == 1:
                return torch.log(lo[:, 0]), (torch.minimum(torch.log(hi[:, 0]), y_prefix[:, 0]) if free
                                             else torch.log(hi[:, 0]))
            if i == 2:
                return torch.log(lo[:, 1]), torch.log(hi[:, 1])
            return _full(B, 0, y_prefix), _full(B, 1, y_prefix)
        if i == 0:
            return torch.log(lo.amax(1)), torch.log(hi[:, 2])
        if i == 1:
            c = torch.exp(y_prefix[:, 0])
            return lo[:, :2].amax(1) / c, torch.minimum(hi[:, 1], c) / c
        if i == 2:
            b = torch.exp(y_prefix[:, 0]) * y_prefix[:, 1]
            return lo[:, 0] / b, torch.minimum(hi[:, 0], b) / b
        if i in (3, 4):
            return _full(B, -1, y_prefix), _full(B, 0, y_prefix)
        return _full(B, 0, y_prefix), _full(B, 1, y_prefix)

    def map(self, y):
        y = torch.as_tensor(y, dtype=F64)
        if self.family in HIGHER_FAMILIES:
            return self._map_higher(y)
        if self.family == 'monoclinic':
            log_c, log_a, log_b, s = y.unbind(-1)
            a, b, c = torch.exp(log_a), torch.exp(log_b), torch.exp(log_c)
            m = self._mono_bound(a, c)
            cb = -s * m
            half = torch.full_like(s, math.pi / 2)
            cell = torch.stack([a, b, c, half, torch.arccos(cb), half], -1)
            return cell, torch.log(m) - 0.5 * torch.log1p(-cb * cb)
        log_c, r2, r1, sb, sg, v = y.unbind(-1)
        c = torch.exp(log_c)
        b = r2 * c
        a = r1 * b
        lam = self.lam(r1, sb, sg)
        sa = lam + (1 - lam) * v
        ca, cb, cg = sa * r2 / 2, sb * r1 * r2 / 2, sg * r1 / 2
        cell = torch.stack([a, b, c, torch.arccos(ca), torch.arccos(cb), torch.arccos(cg)], -1)
        sins = torch.sqrt(1 - ca ** 2) * torch.sqrt(1 - cb ** 2) * torch.sqrt(1 - cg ** 2)
        ld = torch.log(1 - lam) + torch.log(r1) + torch.log(r2) - math.log(8) - torch.log(sins)
        return cell, ld

    def unmap(self, cell):
        cell = torch.as_tensor(cell, dtype=F64)
        if self.family in HIGHER_FAMILIES:
            return self._unmap_higher(cell)
        a, b, c, al, be, ga = cell.unbind(-1)
        if self.family == 'monoclinic':
            m = self._mono_bound(a, c)
            s = -torch.cos(be) / m
            log_a, log_c = torch.log(a), torch.log(c)
            # a canonical cell on a face comes back within rounding outside it (cos(pi/2) = 6e-17): those excursions,
            # and only those, are put on the bound; anything further out is returned as it is
            s = torch.where((s < 0) & (s >= -self.SLACK), torch.zeros_like(s),
                            torch.where((s > 1) & (s <= 1 + self.SLACK), torch.ones_like(s), s))
            if self.mono_class == MONO_FREE:
                log_a = torch.where((log_a > log_c) & (log_a <= log_c + self.SLACK), log_c, log_a)
            y = torch.stack([log_c, log_a, torch.log(b), s], -1)
            return y, -(torch.log(m) - torch.log(torch.sin(be)))
        r2, r1 = b / c, a / b
        sa, sb, sg = 2 * torch.cos(al) / r2, 2 * torch.cos(be) / (r1 * r2), 2 * torch.cos(ga) / r1
        lam = self.lam(r1, sb, sg)
        v = (sa - lam) / (1 - lam)
        y = torch.stack([torch.log(c), r2, r1, sb, sg, v], -1)
        ld = -(torch.log(1 - lam) + torch.log(r1) + torch.log(r2) - math.log(8)
               - torch.log(torch.sin(al) * torch.sin(be) * torch.sin(ga)))
        return y, ld


# ---------------------------------------------------------------------------
# the product chart
# ---------------------------------------------------------------------------

class CrystalChart(Chart):
    """Product chart cell x centroid x orientation of Z' = 1 descriptions of one space group, on the +1 sheet.

    Columns: ``cell`` (6 triclinic, 4 monoclinic, 3 orthorhombic, 2 tetragonal and
    hexagonal, 1 cubic), then ``centroid`` (the non-free AR coordinates), then
    ``orientation``; ``logabsdet`` is the sum of the blocks'. ``inverse`` does not fold:
    the description must already be in the domain (see ``canon.canonicalize``). Every
    space group whose K is trivial: the image is the fundamental domain (``domain``
    'fundamental', ``multiplicity`` 1).

    **sg 6-9** (K = C2(b) acts on R alone; the fundamental domain cuts it out of SO(3)).
    With ``orientation_cover=None`` (since 2026-10-26) the orientation block is
    ``KCutHemisphereChart(sg)``, the hemisphere chart with a signed r cut to
    |r| <= rho(theta, phi) (the fold's half of SO(3)): ``domain`` 'fundamental',
    ``multiplicity`` 1. An ``OrientationChart('hemisphere')`` passed in is replaced by it
    (``k_cut`` True); the other orientation charts have no SO(3)/K chart and are refused
    (NotImplementedError). ``forward`` and ``inverse`` check per row that Q_K
    (``k_rotation``, T W_K T^-1 of the row's cell) is the 2-fold about Cartesian y the
    cut assumes (within ``KCutHemisphereChart.FRAME_TOL`` = 1e-6, which float32-stored
    rows pass; ValueError otherwise). ``inverse`` charts the fold's choice between R and
    Q_K R (``KCutHemisphereChart.unmap``).

    With ``orientation_cover='double'`` the chart instead covers ASU(Gamma+) x all of
    SO(3) in any orientation chart, both K images of every
    crystal, so ``domain`` is 'double_cover', ``multiplicity`` exactly 2 and
    ``log_multiplicity`` log 2 (``measure.log_multiplicity(sg, 'fundamental',
    'double_cover')``): log Z over the chart is log Z of the fundamental domain plus log 2.
    A density on this chart must be K-invariant as a density relative to the reference
    measure (Haar on R): log p_y(y) - ``log_reference_density`` (equivalently log p_y(y)
    - ``logabsdet``(y); the cell, the non-free t and V are the same at both points) is
    equal at y and at the chart point of ``k_image``. In ``euler_zyz`` the Haar density
    is uniform, so p_y itself is equal there; in ``hemisphere`` and ``axis_angle`` it is
    not, and the two Lebesgue densities differ by the Haar ratio. Train on data
    augmented with the K image of each row (``k_image``); a model that is not
    K-invariant puts unequal mass on two descriptions of one crystal.

    Parameters
    ----------
    sg : int
    cell : CellChart
        Of ``cell_family(sg)``; monoclinic of the group's setting class, orthorhombic of its axis-order class.
    centroid : PolytopeARChart
        Of ``sg``.
    orientation : OrientationChart
        sg 6-9 without a cover: ``OrientationChart('hemisphere')`` or ``KCutHemisphereChart(sg)`` (above).
    orientation_cover : None or 'double'
        'double' admits sg 6-9 as the double cover (above); ValueError elsewhere (K is trivial).
    """
    COVERS = (None, 'double')

    def __init__(self, sg: int, cell: CellChart, centroid: PolytopeARChart, orientation: OrientationChart,
                 orientation_cover: Optional[str] = None):
        if centroid.sg != int(sg):
            raise ValueError(f'centroid chart is for sg {centroid.sg}, not {sg}')
        if orientation_cover not in self.COVERS:
            raise ValueError(f'orientation_cover must be one of {self.COVERS}, not {orientation_cover!r}')
        if cell.family == 'triclinic' and int(sg) not in (1, 2):
            raise ValueError(f'sg {sg} is not triclinic')
        if cell.family in HIGHER_FAMILIES:
            if cell_family(sg) != cell.family:
                raise ValueError(f'sg {sg} is not {cell.family}')
            if cell.family == 'orthorhombic' and axis_order.axis_orders(sg).kind != cell.axis_order_kind:
                raise ValueError(f'sg {sg} is not in the axis-order class of the cell chart (sg {cell.sg}, '
                                 f'{cell.axis_order_kind!r})')
        n_k = len(group_table(int(sg)).kernel)
        if orientation_cover == 'double' and n_k == 1:
            raise ValueError(f"sg {sg}: K is trivial, so the chart is already the fundamental domain; "
                             "orientation_cover='double' is for sg 6-9")
        k_cut = False
        if cell.family == 'monoclinic':
            if not 3 <= int(sg) <= 15 or int(MONO_CLASS[int(sg)]) != cell.mono_class:
                raise ValueError(f'sg {sg} is not in the monoclinic setting class of the cell chart (sg {cell.sg})')
            if n_k > 1 and orientation_cover is None:
                if isinstance(orientation, KCutHemisphereChart):
                    if orientation.sg != int(sg):
                        raise ValueError(f'the K-cut orientation chart is for sg {orientation.sg}, not {sg}')
                elif type(orientation) is OrientationChart and orientation.name == 'hemisphere':
                    orientation = KCutHemisphereChart(int(sg))
                else:
                    raise NotImplementedError(
                        f"sg {sg}: the domain cuts K out of SO(3); its chart of SO(3)/K is built in the hemisphere "
                        f"chart only (KCutHemisphereChart), not {getattr(orientation, 'name', orientation)!r}; "
                        "orientation_cover='double' charts the double cover in any orientation chart")
                k_cut = True
        if isinstance(orientation, KCutHemisphereChart) and not k_cut:
            raise ValueError(f"sg {sg}: KCutHemisphereChart is the orientation block of sg 6-9 without a cover (K is "
                             f"{'trivial' if n_k == 1 else 'covered by orientation_cover=' + repr(orientation_cover)})")
        self.sg = int(sg)
        self.k_cut = k_cut
        self.cell, self.centroid, self.orientation = cell, centroid, orientation
        self.orientation_cover = orientation_cover
        self.multiplicity = n_k if orientation_cover == 'double' else 1
        self.log_multiplicity = math.log(self.multiplicity)
        self.domain = 'double_cover' if orientation_cover == 'double' else 'fundamental'
        self.blocks = (cell, centroid, orientation)
        self.offsets = []
        dims, k = [], 0
        for blk in self.blocks:
            self.offsets.append(k)
            dims += [d._replace(ar_position=k + j) for j, d in enumerate(blk.dims)]
            k += blk.ndim
        self.dims = tuple(dims)

    @classmethod
    def for_length_box(cls, sg: int, length_box, orientation: str, device=None,
                       orientation_cover: Optional[str] = None) -> 'CrystalChart':
        """The domain chart of space group ``sg`` with an explicit length box (lo, hi), [3] or [B, 3] Angstrom: the cell
        chart of ``cell_family(sg)``, ``PolytopeARChart(sg)`` and ``OrientationChart(orientation)``."""
        fam = cell_family(sg)
        cell = CellChart(fam, length_box) if fam == 'triclinic' else CellChart(fam, length_box, sg=sg)
        return cls(sg, cell, PolytopeARChart(sg, device), OrientationChart(orientation),
                   orientation_cover=orientation_cover)

    @classmethod
    def for_latent_v1(cls, sg: int, radius, orientation: str, z_prime: int = 1, device=None,
                      orientation_cover: Optional[str] = None) -> 'CrystalChart':
        """The domain chart of space group ``sg`` with the trainer's length box at ``radius``
        (``latent_v1_length_box``). From orthorhombic up the box is used as a box only (lo, hi per axis): no part of
        the trainer's cell latent (its averaged length rows, its angle rows) enters the chart."""
        box = latent_v1_length_box(sg, radius, z_prime)
        return cls.for_length_box(sg, box, orientation, device=device, orientation_cover=orientation_cover)

    def k_image(self, desc) -> CrystalDescription:
        """The other K image of each row (sg 6-9: the double cover, or the K cut): the same crystal, at the other
        point of the cover; on the K cut, outside the chart (the image the fold rejects).

        K's element acts trivially on the non-free centroid coordinates (kept bit for bit), free coordinates are set
        to the gauge, and R becomes Q_K R with Q_K = T W_K T^-1 of each row's cell.
        """
        if self.orientation_cover != 'double' and not self.k_cut:
            raise ValueError("k_image is defined where K is nontrivial (sg 6-9: orientation_cover='double', or the "
                             "K cut) only")
        from mxtaltools.crystal_fd.ops import apply
        tab = group_table(self.sg)
        (k,) = [int(i) for i in tab.kernel if int(i) != int(tab.gamma_plus[0])]
        d = desc.double()
        img = apply(d, tab.gamma_W[k].to(d.device), tab.gamma_w[k].to(d.device))
        free = self.centroid.free.to(d.device)
        t = torch.where(free, self.centroid.gauge.to(d.device), d.t)
        return d.replace(t=t, R=img.R, h=img.h)

    def _check_k_frame(self, cell):
        """K cut: ValueError unless every row's Q_K (``k_rotation``) is ``K_B_ROTATION`` within ``FRAME_TOL``."""
        Q = k_rotation(self.sg, cell)
        dev = (Q - torch.tensor(K_B_ROTATION, dtype=F64, device=Q.device)).abs().amax((1, 2))
        bad = ~(dev <= KCutHemisphereChart.FRAME_TOL)
        if bool(bad.any()):
            rows = torch.nonzero(bad)[:, 0]
            raise ValueError(f'sg {self.sg}: {len(rows)} rows (rows {rows[:8].tolist()}) have Q_K = T W_K T^-1 further '
                             f'than FRAME_TOL = {KCutHemisphereChart.FRAME_TOL} from the 2-fold about y the K cut '
                             f'assumes (largest {float(dev[bad].max()):.3g}); canonicalise them first (alpha = gamma = '
                             '90 deg)')

    def _split(self, y):
        return [y[:, o:o + b.ndim] for o, b in zip(self.offsets, self.blocks)]

    def forward(self, y, ctx=None) -> ChartMap:
        y = torch.as_tensor(y, dtype=F64)
        yc, yt, yr = self._split(y)
        cell, l0 = self.cell.map(yc)
        if self.k_cut:
            self._check_k_frame(cell)
        t, l1 = self.centroid.map(yt)
        R, l2 = self.orientation.map(yr)
        return ChartMap(CrystalDescription.from_parts(self.sg, cell, t, R), l0 + l1 + l2)

    def inverse(self, desc):
        d = desc.double()
        if bool((d.z_prime != 1).any()) or bool((d.sg != self.sg).any()):
            raise ValueError(f"CrystalChart(sg {self.sg}) takes Z' = 1 rows of sg {self.sg}")
        if bool((d.h[:, 0] != 1).any()):
            raise ValueError('CrystalChart is on the +1 sheet; move rows there first (domains.to_sheet)')
        if self.k_cut:
            self._check_k_frame(d.cell)
        y0, l0 = self.cell.unmap(d.cell)
        y1, l1 = self.centroid.unmap(d.t[:, 0])
        y2, l2 = self.orientation.unmap(d.R[:, 0])
        return torch.cat([y0, y1, y2], 1), l0 + l1 + l2

    def bounds(self, y_prefix, i):
        for o, blk in zip(self.offsets, self.blocks):
            if i < o + blk.ndim:
                return blk.bounds(y_prefix[:, o:i], i - o)
        raise IndexError(i)


# ---------------------------------------------------------------------------
# the trainer's latent chart
# ---------------------------------------------------------------------------

def _latent_view_class():
    from mxtaltools.dataset_utils.data_class_methods.crystal_ops import MolCrystalOps

    class _LatentView:
        """The attributes MolCrystalOps' latent maps read, and those maps, unchanged.

        ``box_analysis`` is a no-op (no cell-derived attribute is needed by the
        maps); ``inv_latent_transform`` records the clamped latents it receives.
        """
        latent_to_cell_params = MolCrystalOps.latent_to_cell_params
        latent_transform = MolCrystalOps.latent_transform
        set_cell_parameters = MolCrystalOps.set_cell_parameters
        build_asym_unit_tensor = MolCrystalOps.build_asym_unit_tensor
        build_free_axis_tensor = MolCrystalOps.build_free_axis_tensor
        canonicalize_free_axes = MolCrystalOps.canonicalize_free_axes

        def __init__(self, sg_ind, radius, z_prime, max_z_prime, device):
            self.sg_ind, self.radius, self.z_prime = sg_ind, radius, z_prime
            self.max_z_prime, self.device = max_z_prime, device
            self.clamped_latents = None

        def inv_latent_transform(self, latents):
            self.clamped_latents = latents
            return MolCrystalOps.inv_latent_transform(self, latents)

        def box_analysis(self):
            pass

    return _LatentView


_VIEW = None

#: why ``LatentV1Chart`` has no log-Jacobian for sg >= 16, and will not get one
LATENT_ABOVE_MONOCLINIC = (
    "LatentV1Chart log-Jacobian: not built for sg >= 16, and not to be. Above orthorhombic the trainer's cell latent "
    "averages length rows (a = b; a = b = c in cubic) and is to be replaced in the trainer, not mirrored here (owner, "
    "2026-10-21: 'Averaging is probably dumb'); in orthorhombic groups it fixes no axis order, and from orthorhombic "
    "up its angle rows are overwritten, so it is no chart of the canonical cells. Use CrystalChart (one coordinate "
    "per free cell parameter: CrystalChart.for_length_box, or for_latent_v1 with the trainer's length box used as a "
    "box); describe and latents still read the live maps.")


def _make_view(sg: torch.Tensor, radius: torch.Tensor, device):
    """A latent view of len(sg) Z' = 1 rows (sg long, radius float32)."""
    global _VIEW
    if _VIEW is None:
        _VIEW = _latent_view_class()
    return _VIEW(sg.to(device=device, dtype=torch.long).clone(), radius.to(device=device, dtype=torch.float32).clone(),
                 torch.ones(len(sg), dtype=torch.long, device=device), 1, device)


class LatentV1Chart(Chart):
    """The trainer's latent (``MolCrystalOps.latent_transform`` layout, Z' = 1) as a chart, read-only and exact.

    ``describe`` (and ``forward``) runs the live ``latent_to_cell_params`` (clamp,
    inverse map, ``enforce_crystal_system``) on a view holding only what it reads, in
    float32 as the trainer does, and returns the clamp as a mask instead of applying
    it silently; ``latents`` (and ``inverse``) runs the live ``canonicalize_free_axes``
    and ``latent_transform`` (no clip; ``latent_params`` would also clip to [-1, 1]).
    The description is float64 from the float32 parameters, at handedness +1. The
    maps serve any space group; ``forward`` and ``inverse``, which carry the
    log-Jacobian, serve sg 1 and 2 (stage 0). For sg >= 16 they refuse for good
    (``LATENT_ABOVE_MONOCLINIC``): the trainer's latent there is to be replaced, and no
    averaged-row Jacobian is built; ``CrystalChart`` is the chart of those groups.

    ``logabsdet`` (triclinic groups) is the log-density of the natural measure
    relative to the live rows of the latent, at the built (clamped) latent: the
    slopes of the live map (probed in float64, so they carry its float32 constants)
    plus the Haar density sin^2(r/2) sin(theta) / (2 pi^2) in (theta, phi, r).
    Rows that ``canonicalize_free_axes`` fixes are ``'fixed'`` and excluded.

    Parameters
    ----------
    sg : int or torch.Tensor
        Space group, or [B] per row.
    radius : float or torch.Tensor
        Molecule radius (``MolData.radius``), or [B]; kept as float32, as the batch holds it.
    z_prime : int
        1 (Z' > 1 is a later stage).
    """

    def __init__(self, sg, radius, z_prime: int = 1, device=None):
        if int(z_prime) != 1:
            raise NotImplementedError("LatentV1Chart: Z' > 1 is a later stage")
        self.device = torch.device('cpu' if device is None else device)
        self.sg = torch.as_tensor(sg, dtype=torch.long, device=self.device).reshape(-1)
        self.radius = torch.as_tensor(radius, dtype=torch.float32, device=self.device).reshape(-1)
        self.z_prime = 1
        self.width = 6 + 6 * self.z_prime
        sgs = torch.unique(self.sg).tolist()
        self._tables = {sg: group_table(sg, self.device) for sg in sgs}
        self.dims = self._dims(sgs[0]) if len(sgs) == 1 else tuple(
            Dim(d.name, 'interval', None, d.ar_position) for d in self._dims(sgs[0]))
        self._slopes = {}
        self._luts = None

    # -- live view --------------------------------------------------------------

    def _rows(self, B):
        sg = self.sg.expand(B) if self.sg.numel() == 1 else self.sg
        rad = self.radius.expand(B) if self.radius.numel() == 1 else self.radius
        if sg.shape[0] != B or rad.shape[0] != B:
            raise ValueError(f'chart holds {self.sg.numel()} space groups and {self.radius.numel()} radii for {B} rows')
        return sg, rad

    def _view(self, B):
        """A view of B rows carrying the live lookup tables, built once by the live builders and then shared."""
        v = _make_view(*self._rows(B), self.device)
        if self._luts is None:
            v.build_asym_unit_tensor()
            v.build_free_axis_tensor()
            self._luts = (v.asym_unit_lut, v.free_axis_lut)
        v.asym_unit_lut, v.free_axis_lut = self._luts
        return v

    def clamp_bounds(self):
        """(lo, hi) [width] float32: the live clamp of ``latent_to_cell_params``, read off the live code."""
        if not hasattr(self, '_clamp'):
            v = _make_view(self.sg[:1].expand(2), torch.ones(2), self.device)
            probe = torch.tensor([[-1e30] * self.width, [1e30] * self.width], dtype=torch.float32, device=self.device)
            v.latent_to_cell_params(probe, skip_box_analysis=True, skip_enforce_crystal_system=True)
            self._clamp = (v.clamped_latents[0].clone(), v.clamped_latents[1].clone())
        return self._clamp

    def cell_params(self, y):
        """Live ``latent_to_cell_params`` on latents y [B, width], any space group.

        Returns (params [B, width] float32 in the full_cell_parameters layout, clamped [B] bool, the clamped
        latents [B, width]).
        """
        y32 = torch.as_tensor(y).to(device=self.device, dtype=torch.float32)
        v = self._view(y32.shape[0])
        v.latent_to_cell_params(y32, skip_box_analysis=True)
        clamped = (v.clamped_latents != y32).any(1)
        params = torch.cat([v.cell_lengths, v.cell_angles, v.aunit_centroid, v.aunit_orientation], 1)
        return params, clamped, v.clamped_latents

    def describe(self, y):
        """(descriptions [B] of the latents y as the trainer builds them, clamped [B]); any space group."""
        params, clamped, _ = self.cell_params(y)
        return self._desc(params), clamped

    def _desc(self, params):
        sg, _ = self._rows(params.shape[0])
        return CrystalDescription.from_crystal_params(sg, params[:, :3], params[:, 3:6], params[:, 6:9],
                                                      params[:, 9:12])

    def latents(self, desc):
        """[B, width] float32 latents of descriptions: live ``canonicalize_free_axes`` then ``latent_transform``.

        Any space group; no clip (``latent_params`` would also clip to [-1, 1]).
        """
        d = desc.double()
        if bool((d.z_prime != 1).any()):
            raise NotImplementedError("LatentV1Chart: Z' > 1 is a later stage")
        if bool((d.h[:, 0] != 1).any()):
            raise ValueError('the trainer latent is on the +1 sheet; move rows there first (domains.to_sheet)')
        if not torch.equal(d.sg.to(self.device), self._rows(d.num_rows)[0]):
            raise ValueError('description space groups differ from the chart')
        L, A, c, rv, _ = d.crystal_params()
        v = self._view(d.num_rows)
        v.aunit_centroid = c.float()
        v.canonicalize_free_axes()
        return v.latent_transform(torch.cat([L.float(), A.float(), v.aunit_centroid, rv.float()], 1))

    # -- protocol (log-Jacobians: triclinic groups) ------------------------------------

    def forward(self, y, ctx=None) -> LatentMap:
        params, clamped, yc = self.cell_params(y)
        return LatentMap(self._desc(params), self.log_abs_det(yc), clamped)

    def inverse(self, desc):
        lat = self.latents(desc)
        return lat, -self.log_abs_det(lat)

    def bounds(self, y_prefix, i):
        lo, hi = self.clamp_bounds()
        B = y_prefix.shape[0]
        return _full(B, lo[i], y_prefix), _full(B, hi[i], y_prefix)

    def in_clamp_box(self, y):
        """[B] True where the live clamp leaves every entry of y unchanged."""
        return ~self.cell_params(y)[1]

    # -- log-Jacobian ------------------------------------------------------------------

    def live_rows(self, sg: int) -> torch.Tensor:
        """[width] bool: rows that are not fixed by the gauge (free-axis centroid rows are fixed)."""
        live = torch.ones(self.width, dtype=torch.bool)
        live[6:9] = ~self._tables[sg].free.cpu()
        return live

    def _log_slopes(self, sg: int) -> float:
        """Sum over live rows of log|d(natural coordinate)/d(latent)| of the live map, probed at +-1 in float64."""
        if sg not in self._slopes:
            from mxtaltools.common.geometry_utils import lat2sph_rotvec
            v = _make_view(torch.full((2,), sg), torch.ones(2), self.device)
            ends = torch.tensor([[1.0] * self.width, [-1.0] * self.width], dtype=F64, device=self.device)
            p = v.inv_latent_transform(ends)
            nat = torch.cat([torch.log(p[:, :3]), p[:, 3:9], lat2sph_rotvec(ends[:, 9:12], 1)], 1)
            slope = (nat[0] - nat[1]) / 2
            self._slopes[sg] = float(torch.log(slope.abs())[self.live_rows(sg)].sum())
        return self._slopes[sg]

    def log_abs_det(self, y) -> torch.Tensor:
        """[B] log-density of the natural measure relative to Lebesgue on the live latent rows, at ``y``."""
        from mxtaltools.common.geometry_utils import lat2sph_rotvec
        y = torch.as_tensor(y).to(device=self.device, dtype=F64)
        B = y.shape[0]
        sg, _ = self._rows(B)
        if not bool(((sg == 1) | (sg == 2)).all()):
            if bool((sg >= 16).any()):
                raise NotImplementedError(LATENT_ABOVE_MONOCLINIC)
            raise NotImplementedError('LatentV1Chart log-Jacobian: triclinic groups only (stage 0)')
        const = torch.zeros(B, dtype=F64, device=self.device)
        for s in torch.unique(sg).tolist():
            const = torch.where(sg == s, self._log_slopes(int(s)), const)
        sph = lat2sph_rotvec(y[:, 9:12], 1)
        haar = so3.haar_log_density('hemisphere', sph)
        return const + haar

    def _dims(self, sg: int):
        tab = self._tables[sg]
        names = ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 't_x', 't_y', 't_z', 'theta', 'phi', 'r')
        out = []
        for i, n in enumerate(names):
            kind, period = 'interval', None
            if 6 <= i < 9:
                ax = i - 6
                if bool(tab.free[ax]):
                    kind = 'fixed'
                elif _has_translation(tab, ax, to_fraction(tab.raw['trainer_box']['box'][ax])):
                    kind, period = 'circle', 2.0
            elif n == 'phi':
                kind, period = 'circle', 2.0
            out.append(Dim(n, kind, period, i))
        return tuple(out)


def _has_translation(tab, axis: int, length) -> bool:
    """True if Gamma+ holds the pure translation length * e_axis modulo Z^3."""
    target = [length % 1 if k == axis else 0 for k in range(3)]
    for k in tab.raw['gamma_plus']:
        if list(tab.raw['gamma']['W'][k]) == _EYE \
                and [to_fraction(v) % 1 for v in tab.raw['gamma']['w'][k]] == target:
            return True
    return False


def latent_v1_length_box(sg: int, radius, z_prime: int = 1):
    """(lo, hi) [B, 3] float64 cell lengths (Angstrom) the trainer's clamp allows at ``radius``.

    The live inverse latent map evaluated in float64 at the live clamp bounds of
    the three length rows.
    """
    ch = LatentV1Chart(sg, radius, z_prime)
    lo_l, hi_l = ch.clamp_bounds()
    B = max(ch.radius.numel(), ch.sg.numel())
    sgv, rad = ch._rows(B)
    v = _make_view(torch.cat([sgv, sgv]), torch.cat([rad, rad]), ch.device)
    ends = torch.zeros(2 * B, ch.width, dtype=F64, device=ch.device)
    ends[:B, :3] = lo_l[:3].double()
    ends[B:, :3] = hi_l[:3].double()
    p = v.inv_latent_transform(ends)
    return p[:B, :3].clone(), p[B:, :3].clone()
