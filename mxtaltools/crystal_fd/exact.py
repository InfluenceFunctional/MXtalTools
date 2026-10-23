"""Exact rational arithmetic for the crystal fundamental-domain tables.

Shared by the offline generator (``generate_fd_tables.py``), the loader
(``tables.py``) and the gate tests. Nothing here imports cctbx.

Conventions
-----------
An affine element acts on fractional column vectors as ``x -> W x + w``. ``W`` is
an integer [3, 3] array and ``w`` is held as integer numerators over ``DEN``.
Group elements are taken modulo Z^3 and modulo the continuous origin shifts along
free axes: translations are reduced to [0, 1) and zeroed on free axes.

A polytope is a list of half-spaces ``n . x + c >= 0`` with exact rational
(``fractions.Fraction``) or integer entries.
"""
import hashlib
import itertools
import json
from fractions import Fraction
from math import gcd

import numpy as np

#: Common denominator of every translation in SYM_OPS, NORMALIZER_OPS and their
#: closure. Rationals off this grid raise.
DEN = 24

NO_FREE = np.zeros(3, dtype=bool)


# ---------------------------------------------------------------------------
# rationals
# ---------------------------------------------------------------------------

def to_fraction(x) -> Fraction:
    """Fraction from a 'p/q' string, an int or a Fraction. Floats are refused."""
    if isinstance(x, (float, np.floating)):
        raise TypeError(f'refusing to read float {x!r} as an exact rational; use rationalize')
    return Fraction(x)


def rationalize(x: float, den: int = DEN, tol: float = 1e-8) -> Fraction:
    """The rational on the 1/den grid within ``tol`` of the float ``x``; raises otherwise."""
    q = Fraction(round(float(x) * den), den)
    if abs(float(q) - float(x)) > tol:
        raise ValueError(f'{x!r} is not within {tol} of the 1/{den} grid')
    return q


def fraction_str(q) -> str:
    return str(Fraction(q))


def free_mask(continuous_dims) -> np.ndarray:
    """Boolean [3] mask of the free axes from a CONTINUOUS_DIMS entry."""
    v = np.asarray(continuous_dims, dtype=float).reshape(-1, 3)
    return np.abs(v).sum(0) > 0


# ---------------------------------------------------------------------------
# affine elements modulo Z^3 and free-axis shifts
# ---------------------------------------------------------------------------

class Op:
    """Affine map ``x -> W x + t / DEN`` modulo Z^3 and free-axis shifts.

    Parameters
    ----------
    W : array-like
        Integer rotation part, [3, 3].
    t : array-like
        Integer translation numerators over ``DEN``, [3].
    free : numpy.ndarray, optional
        Boolean [3] mask of free axes, whose translation components are zeroed.
    """
    __slots__ = ('W', 't', 'key')

    def __init__(self, W, t, free=NO_FREE):
        W = np.asarray(W, dtype=np.int64).reshape(3, 3)
        t = np.mod(np.asarray(t, dtype=np.int64).reshape(3), DEN)
        t[np.asarray(free, dtype=bool)] = 0
        self.W, self.t = W, t
        self.key = (tuple(W.ravel().tolist()), tuple(t.tolist()))

    @classmethod
    def from_rationals(cls, W, w, free=NO_FREE):
        """From an integer [3, 3] ``W`` and three exact rationals ``w`` on the 1/DEN grid."""
        num = []
        for q in w:
            q = to_fraction(q) * DEN
            if q.denominator != 1:
                raise ValueError(f'translation {w} is off the 1/{DEN} grid')
            num.append(int(q))
        return cls(W, num, free)

    @property
    def det(self) -> int:
        return int(round(np.linalg.det(self.W)))

    def w(self) -> list:
        """Translation as exact rationals in [0, 1)."""
        return [Fraction(int(v), DEN) for v in self.t]

    def compose(self, other: 'Op', free=NO_FREE) -> 'Op':
        """``self o other``: x -> W_s (W_o x + w_o) + w_s."""
        return Op(self.W @ other.W, self.W @ other.t + self.t, free)

    def inverse(self, free=NO_FREE) -> 'Op':
        Wi = int_inverse(self.W)
        return Op(Wi, -(Wi @ self.t), free)

    def __repr__(self):
        return f'Op({xyz_string(self.W, self.w())})'


def int_inverse(W: np.ndarray) -> np.ndarray:
    """Exact inverse of an integer matrix with determinant +-1."""
    Wi = np.round(np.linalg.inv(W.astype(float))).astype(np.int64)
    if not np.array_equal(W @ Wi, np.eye(3, dtype=np.int64)):
        raise ValueError(f'{W.tolist()} has no integer inverse')
    return Wi


def xyz_string(W, w) -> str:
    """'x,y,z'-style string of an affine map with exact translation ``w``."""
    out = []
    for i in range(3):
        s = ''
        for j, v in enumerate('xyz'):
            c = int(W[i][j])
            if c == 0:
                continue
            s += ('+' if c > 0 and s else '') + ('-' if c < 0 else '') + (f'{abs(c)}*' if abs(c) != 1 else '') + v
        q = Fraction(w[i])
        if q != 0:
            s += ('+' if q > 0 and s else '') + str(q)
        out.append(s or '0')
    return ','.join(out)


def stack_ops(ops):
    """(W [n, 3, 3] int64, t [n, 3] int64) from a sequence of Op."""
    ops = list(ops)
    return (np.stack([o.W for o in ops]).astype(np.int64),
            np.stack([o.t for o in ops]).astype(np.int64))


def op_codes(W: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unique int64 code per element; ``W`` entries must lie in [-2, 2]."""
    W = np.asarray(W, dtype=np.int64)
    if W.size and (W.min() < -2 or W.max() > 2):
        raise ValueError('rotation entries outside [-2, 2]')
    flat = (W.reshape(*W.shape[:-2], 9) + 2)
    code = np.zeros(W.shape[:-2], dtype=np.int64)
    for k in range(9):
        code = code * 5 + flat[..., k]
    t = np.asarray(t, dtype=np.int64)
    return (code * DEN + t[..., 0]) * DEN * DEN + t[..., 1] * DEN + t[..., 2]


def closure(gens, free=NO_FREE, cap: int = 4096) -> dict:
    """All elements generated by ``gens`` (the identity included), keyed by ``Op.key``."""
    ident = Op(np.eye(3), np.zeros(3), free)
    els = {ident.key: ident}
    frontier = [ident]
    gens = [Op(g.W, g.t, free) for g in gens]
    while frontier:
        new = []
        for a in frontier:
            for g in gens:
                c = g.compose(a, free)
                if c.key not in els:
                    els[c.key] = c
                    new.append(c)
                    if len(els) > cap:
                        raise RuntimeError(f'closure exceeded {cap} elements')
        frontier = new
    return els


def compose_all(W: np.ndarray, t: np.ndarray, free=NO_FREE):
    """All pairwise products a o b of the elements (W, t): ([n, n, 3, 3], [n, n, 3])."""
    Wab = np.einsum('aij,bjk->abik', W, W)
    tab = np.einsum('aij,bj->abi', W, t) + t[:, None, :]
    tab = np.mod(tab, DEN)
    tab[..., np.asarray(free, dtype=bool)] = 0
    return Wab, tab


def is_closed(W: np.ndarray, t: np.ndarray, free=NO_FREE) -> bool:
    """True when every product of two elements is again an element (vectorised)."""
    codes = op_codes(W, t)
    Wab, tab = compose_all(W, t, free)
    return bool(np.isin(op_codes(Wab, tab), codes).all())


def generating_set(ops, free=NO_FREE) -> list:
    """A small subset of ``ops`` whose closure contains all of ``ops`` (greedy, in order)."""
    gens, span = [], closure([], free)
    for o in ops:
        if o.key not in span:
            gens.append(o)
            span = closure(gens, free)
    return gens


# ---------------------------------------------------------------------------
# brute-force Euclidean normaliser
# ---------------------------------------------------------------------------

_SIGNED_UNIMODULAR = None


def signed_unimodular_matrices() -> np.ndarray:
    """Every 3x3 matrix with entries in {-1, 0, 1} and determinant +-1, [n, 3, 3]."""
    global _SIGNED_UNIMODULAR
    if _SIGNED_UNIMODULAR is None:
        allw = np.array(list(itertools.product((-1, 0, 1), repeat=9)), dtype=np.int64).reshape(-1, 3, 3)
        det = np.round(np.linalg.det(allw.astype(float))).astype(np.int64)
        _SIGNED_UNIMODULAR = allw[np.abs(det) == 1]
    return _SIGNED_UNIMODULAR


def brute_force_normalizer(group_ops, free, metric: np.ndarray) -> dict:
    """Euclidean normaliser of a space group by exhaustive search.

    Searches every W in {-1, 0, 1}^{3x3} with |det W| = 1 that is an isometry of
    ``metric`` and every w on the 1/DEN grid, keeping (W, w) that normalises the
    group. The normalising test is applied to a generating set of the group.

    Parameters
    ----------
    group_ops : list of Op
        The space group modulo Z^3, NOT reduced by free-axis shifts.
    free : numpy.ndarray
        Free-axis mask; the result is reduced modulo these shifts.
    metric : numpy.ndarray
        Gram matrix [3, 3] of a generic cell of the group's family.

    Returns
    -------
    dict
        ``Op.key`` -> Op for every normaliser element modulo Z^3 and free shifts.
    """
    allw = signed_unimodular_matrices().astype(float)
    M = np.asarray(metric, dtype=float)
    dev = np.abs(np.einsum('kji,jl,klm->kim', allw, M, allw) - M).max(axis=(1, 2))
    isos = signed_unimodular_matrices()[dev < 1e-8 * np.abs(M).max()]
    pg = {tuple(g.W.ravel().tolist()) for g in group_ops}
    trans_by_W = {}
    for g in group_ops:
        trans_by_W.setdefault(tuple(g.W.ravel().tolist()), set()).add(int((g.t[0] * DEN + g.t[1]) * DEN + g.t[2]))
    gens = generating_set(group_ops)
    grid = np.array(list(itertools.product(range(DEN), repeat=3)), dtype=np.int64)
    eye = np.eye(3, dtype=np.int64)
    found = {}
    for W in isos:
        Wi = int_inverse(W)
        if {tuple((W @ g.W @ Wi).ravel().tolist()) for g in group_ops} != pg:
            continue
        ok = np.ones(len(grid), dtype=bool)
        for g in gens:
            Wg2 = W @ g.W @ Wi
            T = np.mod((W @ g.t)[None, :] + grid @ (eye - Wg2).T, DEN)
            code = (T[:, 0] * DEN + T[:, 1]) * DEN + T[:, 2]
            ok &= np.isin(code, list(trans_by_W[tuple(Wg2.ravel().tolist())]))
            if not ok.any():
                break
        for w in grid[ok]:
            n = Op(W, w, free)
            found[n.key] = n
    return found


# ---------------------------------------------------------------------------
# lattices
# ---------------------------------------------------------------------------

def hermite_basis(vectors) -> list:
    """Canonical row-style Hermite normal form of a full-rank integer row lattice in Z^3.

    Returns three integer rows: upper triangular, positive pivots, and entries
    above each pivot reduced to [0, pivot).
    """
    A = [[int(v) for v in row] for row in vectors]
    r = 0
    for col in range(3):
        while True:
            nz = [i for i in range(r, len(A)) if A[i][col] != 0]
            if not nz:
                raise ValueError('lattice is not full rank')
            i0 = min(nz, key=lambda i: abs(A[i][col]))
            A[r], A[i0] = A[i0], A[r]
            if A[r][col] < 0:
                A[r] = [-v for v in A[r]]
            done = True
            for i in range(r + 1, len(A)):
                if A[i][col]:
                    q = A[i][col] // A[r][col]
                    A[i] = [a - q * b for a, b in zip(A[i], A[r])]
                    done = done and A[i][col] == 0
            if done:
                break
        r += 1
    H = A[:3]
    for j in range(3):
        for i in range(j):
            q = H[i][j] // H[j][j]
            H[i] = [a - q * b for a, b in zip(H[i], H[j])]
    return H


def translation_lattice(ops) -> list:
    """Basis (columns, exact rationals) of Z^3 + {w : (I, w) in ops}, in canonical HNF."""
    eye = np.eye(3, dtype=np.int64)
    rows = [list(o.t) for o in ops if np.array_equal(o.W, eye)] + [[DEN if i == j else 0 for j in range(3)] for i in range(3)]
    H = hermite_basis(rows)
    return [[Fraction(H[j][i], DEN) for j in range(3)] for i in range(3)]


# ---------------------------------------------------------------------------
# exact linear algebra and polytopes
# ---------------------------------------------------------------------------

def solve(A, b):
    """Exact solution of the square system A x = b, or None when singular."""
    n = len(A)
    M = [[Fraction(v) for v in row] + [Fraction(bi)] for row, bi in zip(A, b)]
    for col in range(n):
        piv = next((i for i in range(col, n) if M[i][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        M[col] = [v / p for v in M[col]]
        for i in range(n):
            if i != col and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * c for a, c in zip(M[i], M[col])]
    return [M[i][n] for i in range(n)]


def rank(rows) -> int:
    """Exact rank of a list of rational vectors."""
    M = [[Fraction(v) for v in row] for row in rows]
    if not M:
        return 0
    r = 0
    for col in range(len(M[0])):
        piv = next((i for i in range(r, len(M)) if M[i][col] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(r + 1, len(M)):
            if M[i][col] != 0:
                f = M[i][col] / M[r][col]
                M[i] = [a - f * c for a, c in zip(M[i], M[r])]
        r += 1
    return r


def evaluate(n, c, x) -> Fraction:
    return sum((Fraction(a) * b for a, b in zip(n, x)), Fraction(c))


def polytope_vertices(normals, offsets) -> list:
    """Exact vertices of {x : n . x + c >= 0}, in any dimension, sorted."""
    d = len(normals[0])
    verts = set()
    for idx in itertools.combinations(range(len(normals)), d):
        x = solve([normals[i] for i in idx], [-Fraction(offsets[i]) for i in idx])
        if x is None:
            continue
        if all(evaluate(n, c, x) >= 0 for n, c in zip(normals, offsets)):
            verts.add(tuple(x))
    return sorted(verts)


def tight(n, c, points) -> list:
    return [p for p in points if evaluate(n, c, p) == 0]


def is_facet(n, c, points, dim: int) -> bool:
    """True when the inequality is tight on an affinely (dim-1)-dimensional subset of ``points``."""
    if all(Fraction(a) == 0 for a in n):
        return False
    tp = tight(n, c, points)
    if dim == 1:
        return len(tp) >= 1
    if len(tp) < dim:
        return False
    return rank([[a - b for a, b in zip(p, tp[0])] for p in tp[1:]]) == dim - 1


def ordered_polygon(points, normal) -> list:
    """Vertices of a convex planar polygon in 3D, ordered around its centroid."""
    d = int(np.argmax([abs(float(a)) for a in normal]))
    keep = [i for i in range(3) if i != d]
    cx = sum(float(p[keep[0]]) for p in points) / len(points)
    cy = sum(float(p[keep[1]]) for p in points) / len(points)
    return sorted(points, key=lambda p: np.arctan2(float(p[keep[1]]) - cy, float(p[keep[0]]) - cx))


def det3(a, b, c) -> Fraction:
    return (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def polytope_volume(normals, offsets, vertices=None) -> Fraction:
    """Exact volume of a bounded 3D polytope {x : n . x + c >= 0}, by facet fans."""
    verts = polytope_vertices(normals, offsets) if vertices is None else vertices
    p = [sum(v[i] for v in verts) / len(verts) for i in range(3)]
    vol = Fraction(0)
    seen = set()
    for n, c in zip(normals, offsets):
        if not is_facet(n, c, verts, 3):
            continue
        face = tuple(sorted(tight(n, c, verts)))
        if face in seen:
            continue
        seen.add(face)
        poly = ordered_polygon(list(face), n)
        a = [poly[0][i] - p[i] for i in range(3)]
        for k in range(1, len(poly) - 1):
            b = [poly[k][i] - p[i] for i in range(3)]
            e = [poly[k + 1][i] - p[i] for i in range(3)]
            vol += abs(det3(a, b, e)) / 6
    return vol


# ---------------------------------------------------------------------------
# Fourier-Motzkin autoregressive bounds
# ---------------------------------------------------------------------------

def _canonical_row(a, c):
    piv = next(abs(v) for v in a if v != 0)
    return tuple(v / piv for v in a), c / piv


def _fm_eliminate(rows, k):
    pos = [r for r in rows if r[0][k] > 0]
    neg = [r for r in rows if r[0][k] < 0]
    out = [r for r in rows if r[0][k] == 0]
    for ap, cp in pos:
        for an, cn in neg:
            s, t = -an[k], ap[k]
            out.append((tuple(s * x + t * y for x, y in zip(ap, an)), s * cp + t * cn))
    return out


def _prune(rows, points, coords):
    """Facet-defining rows of conv(points) over ``coords`` (rows vanish elsewhere), deduplicated."""
    proj = sorted({tuple(p[i] for i in coords) for p in points})
    out = {}
    for a, c in rows:
        if all(a[i] == 0 for i in coords):
            if c < 0:
                raise ValueError('infeasible constant row')
            continue
        sub = [a[i] for i in coords]
        vals = [evaluate(sub, c, q) for q in proj]
        if min(vals) < 0:
            raise ValueError('Fourier-Motzkin row is not valid on the polytope')
        if is_facet(sub, c, proj, len(coords)):
            key = _canonical_row(a, c)
            out.setdefault(key, key)
    return list(out.values())


def ar_stages(normals, offsets, order, eliminate=()):
    """Autoregressive bounds of {x : n . x + c >= 0} by Fourier-Motzkin elimination.

    Parameters
    ----------
    normals, offsets : sequences of exact rationals
        Half-spaces n . x + c >= 0 in 3D.
    order : sequence of int
        AR order of the kept coordinates: coordinate ``order[k]`` is bounded by
        affine functions of ``order[:k]``.
    eliminate : sequence of int
        Coordinates projected out first (free axes).

    Returns
    -------
    list of dict
        One stage per kept coordinate, ``{'coord': i, 'lower': [(coef, const)],
        'upper': [(coef, const)]}`` with ``coef`` over ``order[:k]``:
        x_i >= max(coef . x_prev + const) and x_i <= min(coef . x_prev + const).
    """
    rows = [(tuple(Fraction(v) for v in n), Fraction(c)) for n, c in zip(normals, offsets)]
    verts = polytope_vertices(normals, offsets)
    remaining = [0, 1, 2]
    for v in eliminate:
        rows = _fm_eliminate(rows, v)
        remaining.remove(v)
        rows = _prune(rows, verts, remaining)
    assert sorted(remaining) == sorted(order)
    stages = []
    remaining = list(order)
    for k in reversed(range(len(order))):
        var = order[k]
        rows = _prune(rows, verts, remaining)
        lower, upper = [], []
        for a, c in rows:
            if a[var] == 0:
                continue
            s = a[var]
            coef = tuple(-a[j] / s for j in order[:k])
            if s > 0:
                lower.append((coef, -c / s))
            else:
                upper.append((coef, -c / s))
        stages.insert(0, dict(coord=var, lower=sorted(lower), upper=sorted(upper)))
        rows = _fm_eliminate(rows, var)
        remaining.remove(var)
    return stages


def ar_halfspaces(stages, order):
    """The AR stages as half-spaces over the kept coordinates (in ``order``)."""
    D = len(order)
    normals, offsets = [], []
    for k, st in enumerate(stages):
        for side, sign in (('lower', 1), ('upper', -1)):
            for coef, const in st[side]:
                n = [Fraction(0)] * D
                n[k] = Fraction(sign)
                for j, a in enumerate(coef):
                    n[j] -= sign * Fraction(a)
                normals.append(n)
                offsets.append(-sign * Fraction(const))
    return normals, offsets


# ---------------------------------------------------------------------------
# cctbx-style cut trees (exact evaluation; cctbx semantics)
# ---------------------------------------------------------------------------

def cut_contains(node: dict, p) -> bool:
    """cctbx ``cut.is_inside`` / ``cut_expression.is_inside`` on a JSON cut tree, exactly.

    A cut node is ``{'n', 'c', 'inclusive', 'sub'}``: strictly positive side in,
    strictly negative out, and on the plane ``sub`` decides, else ``inclusive``.
    An expression node is ``{'op': '&' | '|', 'lhs', 'rhs'}``.
    """
    if 'op' in node:
        if node['op'] == '&':
            return cut_contains(node['lhs'], p) and cut_contains(node['rhs'], p)
        if node['op'] == '|':
            return cut_contains(node['lhs'], p) or cut_contains(node['rhs'], p)
        raise ValueError(node['op'])
    v = evaluate(node['n'], to_fraction(node['c']), p)
    if v != 0:
        return v > 0
    if node['sub'] is None:
        return bool(node['inclusive'])
    return cut_contains(node['sub'], p)


def asu_contains(cuts, p) -> bool:
    """True when the exact point ``p`` lies in the asymmetric unit, ties broken as cctbx does."""
    return all(cut_contains(c, p) for c in cuts)


def cut_planes(cuts):
    """(normals, offsets) of the top-level cuts, exact."""
    return [list(c['n']) for c in cuts], [to_fraction(c['c']) for c in cuts]


# ---------------------------------------------------------------------------
# canonical serialisation of the generator inputs
# ---------------------------------------------------------------------------

def _canon(obj):
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _canon(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        raise TypeError('unexpected bool in a numeric table')
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return float(obj)
    raise TypeError(f'cannot canonicalise {type(obj)}')


def canonical_sha256(obj) -> str:
    """sha256 of a numeric table: keys as str, every number as float, sorted compact JSON."""
    text = json.dumps(_canon(obj), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def input_hashes() -> dict:
    """sha256 of SYM_OPS, NORMALIZER_OPS, CONTINUOUS_DIMS and ASYM_UNITS as currently imported."""
    from mxtaltools.constants.asymmetric_units import ASYM_UNITS
    from mxtaltools.constants.space_group_info import CONTINUOUS_DIMS, NORMALIZER_OPS, SYM_OPS
    return {'SYM_OPS': canonical_sha256(SYM_OPS),
            'NORMALIZER_OPS': canonical_sha256(NORMALIZER_OPS),
            'CONTINUOUS_DIMS': canonical_sha256(CONTINUOUS_DIMS),
            'ASYM_UNITS': canonical_sha256(ASYM_UNITS)}


def text_sha256(data: bytes) -> str:
    """sha256 of text bytes with CRLF line endings read as LF, so a CRLF checkout of a file keeps its digest."""
    return hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()


def file_sha256(path) -> str:
    """``text_sha256`` of a file's bytes."""
    with open(path, 'rb') as f:
        return text_sha256(f.read())


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)
