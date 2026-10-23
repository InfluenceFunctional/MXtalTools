"""Per-space-group fundamental-domain tables, loaded from ``data/fd_tables_v1.json``.

The table is generated offline by ``generate_fd_tables.py`` (cctbx); this loader
never imports cctbx. At import it reads the table file once, recomputes the
sha256 of SYM_OPS, NORMALIZER_OPS, CONTINUOUS_DIMS and ASYM_UNITS and of those
bytes, and raises ``TableMismatchError`` if any differs from
``data/MANIFEST.json``: a table built from other constants is never used. The
parsed table is the verified bytes, never a later read of the file. The table
digest reads CRLF line endings as LF (``exact.text_sha256``), and
``data/.gitattributes`` keeps git from converting them.

All group arithmetic is float64. Exact values (rationals as 'p/q' strings) stay
available in ``GroupTable.raw``; ``exact.py`` reads them.

Notation: G is the space group in the setting of SYM_OPS[sg]; Gamma is its
Euclidean normaliser N_E(G) modulo Z^3 and free-axis shifts; Gamma+ its det +1
part; K the subgroup of Gamma+ acting trivially on the centroid once free axes
are projected out; ASU(Gamma+) cctbx's asymmetric unit of Gamma+/K on the
non-free coordinates (a prism of height 1 along free axes).
"""
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from mxtaltools.crystal_fd.exact import input_hashes, text_sha256, to_fraction

SCHEMA_VERSION = 1
DATA_DIR = Path(__file__).with_name('data')
TABLE_PATH = DATA_DIR / 'fd_tables_v1.json'
MANIFEST_PATH = DATA_DIR / 'MANIFEST.json'


class TableMismatchError(RuntimeError):
    """The table or its manifest does not match the constants it was generated from."""


def verify_manifest(manifest: Optional[dict] = None, table_path: Path = TABLE_PATH) -> dict:
    """Check the manifest against the imported constants and the table file; return it.

    Parameters
    ----------
    manifest : dict, optional
        Manifest to check; read from ``data/MANIFEST.json`` when omitted.
    table_path : pathlib.Path
        Table file whose sha256 (``exact.text_sha256``) must equal ``manifest['table_sha256']``.

    Raises
    ------
    TableMismatchError
        On a schema version, input hash or table hash mismatch.
    """
    return _verified(manifest, table_path.read_bytes())[0]


def _verified(manifest: Optional[dict], table_bytes: bytes):
    """(manifest, table_bytes) after the checks of ``verify_manifest`` on these bytes."""
    if manifest is None:
        if not MANIFEST_PATH.exists():
            raise TableMismatchError(f'{MANIFEST_PATH} is missing; run generate_fd_tables')
        manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise TableMismatchError(f"manifest schema {manifest.get('schema_version')} != {SCHEMA_VERSION}")
    now = input_hashes()
    stale = sorted(k for k in now if manifest.get('input_hashes', {}).get(k) != now[k])
    if stale:
        raise TableMismatchError(f'{stale} changed since the fundamental-domain tables were generated; '
                                 'regenerate with mxtaltools.crystal_fd.generate_fd_tables')
    if text_sha256(table_bytes) != manifest.get('table_sha256'):
        raise TableMismatchError('the table does not match its manifest digest')
    return manifest, table_bytes


MANIFEST, _TABLE_BYTES = _verified(None, TABLE_PATH.read_bytes())
#: Digest of the table file; consumers (priors, flow artifacts) store it as ``table_hash``.
TABLE_HASH = MANIFEST['table_sha256']


@lru_cache(maxsize=1)
def load_raw() -> dict:
    """The parsed table, keyed by space-group number as ``str``: the bytes verified at import."""
    data = json.loads(_TABLE_BYTES.decode('utf-8'))
    if data.get('schema_version') != SCHEMA_VERSION:
        raise TableMismatchError(f"table schema {data.get('schema_version')} != {SCHEMA_VERSION}")
    return data['groups']


def _f64(x, device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64, device=device)


def _rat(rows) -> list:
    """Nested lists of 'p/q' strings to floats."""
    if isinstance(rows, (list, tuple)):
        return [_rat(r) for r in rows]
    return float(to_fraction(rows))


@dataclass(frozen=True, eq=False)
class Polytope:
    """cctbx asymmetric unit {x : n . x + c >= 0} in fractional coordinates.

    ``normals`` [m, 3] and ``offsets`` [m] are the top-level cut planes (integer
    normals, rational offsets, as float64); ``inclusive`` [m] is cctbx's inclusion
    flag and ``facet`` [m] marks facet-defining cuts. Tie-break sub-cuts are in
    ``cuts`` (exact JSON trees; see ``exact.cut_contains``). ``volume`` is exact, as
    a fraction of the conventional cell.
    """
    normals: torch.Tensor
    offsets: torch.Tensor
    inclusive: torch.Tensor
    facet: torch.Tensor
    vertices: torch.Tensor
    volume: Fraction
    cuts: tuple
    symbol: str


@dataclass(frozen=True, eq=False)
class ARStage:
    """Bounds of one AR coordinate: x[coord] in [max(lower_coef @ x_prev + lower_const), min(upper_...)].

    ``x_prev`` are the earlier coordinates in AR order; ``*_coef`` is [r, k].
    """
    coord: int
    lower_coef: torch.Tensor
    lower_const: torch.Tensor
    upper_coef: torch.Tensor
    upper_const: torch.Tensor


@dataclass(frozen=True, eq=False)
class GroupTable:
    """Fundamental-domain data for one space group on one device.

    Attributes
    ----------
    sg, symbol, lattice_type
        Space-group number, symbol and family, as in space_group_info.
    free : torch.Tensor
        Bool [3], the free (continuous origin-shift) axes.
    g_W, g_w : torch.Tensor
        G, index-aligned with SYM_OPS[sg]: rotations [nG, 3, 3] and unreduced
        translations [nG, 3], float64.
    gamma_W, gamma_w, gamma_det : torch.Tensor
        N_E(G) modulo Z^3 and free shifts: [nE, 3, 3], [nE, 3] in [0, 1) (zero on
        free axes), and det [nE] (long). The first |G| elements are G.
    gamma_plus : torch.Tensor
        Long indices into ``gamma`` of the det +1 elements; the first is the identity.
    kernel : torch.Tensor
        Long indices into ``gamma`` of K; nontrivial only in sg 6-9.
    lattice_basis : torch.Tensor
        [3, 3] columns spanning the translation lattice L+ of Gamma+.
    pplus : dict
        Rotation group of Gamma+: ``name`` (C1, C2, C3, C4, C6, D2, D3, D4, D6, T,
        O), ``order``, ``axes`` and ``principal_axis`` / ``secondary_axis`` as
        integer lattice directions.
    g_asu, asu : Polytope
        cctbx asymmetric units of G and of Gamma+.
    facets : tuple of dict
        Per facet of ``asu``: ``cut`` (index into ``asu.cuts``) and ``gluings``, each
        ``{element (index into gamma), shift (int [3]), kind ('translation',
        'self_twisted', 'pair_twisted', 'free_axis'), lands_on (cut index), share}``.
        ``element`` + ``shift`` maps the region just outside the facet back inside.
    ar_order : tuple of int
        Non-free coordinates in autoregressive order.
    ar : tuple of ARStage
        Fourier-Motzkin bounds of ``asu`` over ``ar_order``.
    trainer_box : torch.Tensor
        ASYM_UNITS[sg] [3].
    box_coverage : fractions.Fraction
        Fraction of a chiral molecule's crystals with a Gamma+ image in the box.
    k_box : int or None
        Gamma+ images per covered crystal (None if not constant).
    table_hash : str
        sha256 of the table file.
    raw : dict
        The exact JSON entry.
    """
    sg: int
    symbol: str
    lattice_type: str
    free: torch.Tensor
    g_W: torch.Tensor
    g_w: torch.Tensor
    gamma_W: torch.Tensor
    gamma_w: torch.Tensor
    gamma_det: torch.Tensor
    gamma_plus: torch.Tensor
    kernel: torch.Tensor
    lattice_basis: torch.Tensor
    pplus: dict
    g_asu: Polytope
    asu: Polytope
    facets: tuple
    ar_order: tuple
    ar: tuple
    trainer_box: torch.Tensor
    box_coverage: Fraction
    k_box: Optional[int]
    table_hash: str
    raw: dict

    @property
    def gamma_plus_W(self) -> torch.Tensor:
        return self.gamma_W[self.gamma_plus]

    @property
    def gamma_plus_w(self) -> torch.Tensor:
        return self.gamma_w[self.gamma_plus]


def _polytope(e: dict, device) -> Polytope:
    cuts = tuple(e['cuts'])
    return Polytope(normals=_f64([c['n'] for c in cuts], device),
                    offsets=_f64([float(to_fraction(c['c'])) for c in cuts], device),
                    inclusive=torch.tensor([c['inclusive'] for c in cuts], dtype=torch.bool, device=device),
                    facet=torch.tensor(e['facet'], dtype=torch.bool, device=device),
                    vertices=_f64(_rat(e['vertices']), device).reshape(-1, 3),
                    volume=to_fraction(e['volume']), cuts=cuts, symbol=e['symbol'])


def _stage(s: dict, k: int, device) -> ARStage:
    def side(rows):
        coef = _f64([_rat(r['coef']) for r in rows], device).reshape(len(rows), k)
        return coef, _f64([float(to_fraction(r['const'])) for r in rows], device)
    lc, lk = side(s['lower'])
    uc, uk = side(s['upper'])
    return ARStage(coord=int(s['coord']), lower_coef=lc, lower_const=lk, upper_coef=uc, upper_const=uk)


def _build(sg: int, device: torch.device) -> GroupTable:
    raw = load_raw()[str(sg)]
    g, gm = raw['G'], raw['gamma']
    box = raw['trainer_box']
    return GroupTable(
        sg=sg, symbol=raw['symbol'], lattice_type=raw['lattice_type'],
        free=torch.tensor(raw['free'], dtype=torch.bool, device=device),
        g_W=_f64(g['W'], device).reshape(-1, 3, 3), g_w=_f64(_rat(g['w']), device).reshape(-1, 3),
        gamma_W=_f64(gm['W'], device).reshape(-1, 3, 3), gamma_w=_f64(_rat(gm['w']), device).reshape(-1, 3),
        gamma_det=torch.tensor(gm['det'], dtype=torch.long, device=device),
        gamma_plus=torch.tensor(raw['gamma_plus'], dtype=torch.long, device=device),
        kernel=torch.tensor(raw['kernel'], dtype=torch.long, device=device),
        lattice_basis=_f64(_rat(raw['lattice_basis']), device),
        pplus=raw['pplus'],
        g_asu=_polytope(raw['g_asu'], device), asu=_polytope(raw['asu'], device),
        facets=tuple(raw['facets']),
        ar_order=tuple(raw['ar']['order']),
        ar=tuple(_stage(s, k, device) for k, s in enumerate(raw['ar']['stages'])),
        trainer_box=_f64(_rat(box['box']), device),
        box_coverage=to_fraction(box['coverage']), k_box=box['k_box'],
        table_hash=TABLE_HASH, raw=raw)


_CACHE = {}


def group_table(sg: int, device=None) -> GroupTable:
    """The fundamental-domain table of space group ``sg`` (1-230), cached per device.

    Parameters
    ----------
    sg : int
        International Tables number, in the setting of SYM_OPS[sg].
    device : torch.device or str, optional
        Device for the tensors; CPU when omitted.
    """
    sg = int(sg)
    if not 1 <= sg <= 230:
        raise ValueError(f'space group {sg} is not in 1-230')
    device = torch.device('cpu' if device is None else device)
    key = (sg, str(device))
    if key not in _CACHE:
        _CACHE[key] = _build(sg, device)
    return _CACHE[key]
