"""GATE 0: the fundamental-domain tables, all 230 space groups (fd_tooling.md section 4).

Every fact is recomputed here from the table and the MXtalTools constants in exact
rational arithmetic; the generator's own ``validation`` record is not trusted.
Tests marked ``cctbx`` re-derive the cctbx facts and skip without cctbx.

For a group with free axes, Gamma+ acts on the non-free coordinates through
Gamma+/K, and its asymmetric unit is a prism of height 1 along the free axes;
"vol x |Gamma+| = 1" is then vol x |Gamma+/K| = 1 (K is trivial outside sg 6-9).

Without cctbx (the ``cctbx`` tests skip), the tie-break sub-cut trees are checked
here only for well-formedness and by the pinned table digest; their decisions are
checked against cctbx only under the marker, and at run time by gate 1's exact
special points (``test_pose_special_points``), which need no cctbx.
"""
import ast
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from mxtaltools.constants.asymmetric_units import ASYM_UNITS, RAW_ASYM_UNITS
from mxtaltools.constants.space_group_info import LATTICE_TYPE, NORMALIZER_OPS, SYM_OPS
from mxtaltools.crystal_fd import exact as X
from mxtaltools.crystal_fd.exact import DEN, Op
from mxtaltools.crystal_fd.tables import (MANIFEST, TABLE_HASH, TABLE_PATH, TableMismatchError, group_table,
                                          load_raw, verify_manifest)

SGS = list(range(1, 231))

#: sha256 of data/fd_tables_v1.json. A regenerated table changes it: review the
#: table diff, then edit this line by hand. Never update it from a test run.
PINNED_TABLE_SHA256 = '0dddede823729a1ca2bebcad55dccf77294b81bc1845e0d4a8bdf7c7e3b1bafd'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def entry(sg):
    return load_raw()[str(sg)]


def free_of(e):
    return np.array(e['free'], dtype=bool)


def gamma_ops(e):
    free = free_of(e)
    return [Op.from_rationals(np.array(W).reshape(3, 3), w, free) for W, w in zip(e['gamma']['W'], e['gamma']['w'])]


def sym_ops_exact(sg, free=X.NO_FREE):
    """SYM_OPS[sg] modulo Z^3 (and free shifts), rationalised independently of the table."""
    out = []
    for o in SYM_OPS[sg]:
        o = np.asarray(o, float)
        out.append(Op.from_rationals(np.round(o[:3, :3]), [X.rationalize(v) % 1 for v in o[:3, 3]], free))
    return out


def project(op, free):
    """Action on the non-free coordinates, written as an element acting trivially along free axes."""
    keep = ~free
    W = op.W.copy()
    assert not W[np.ix_(keep, free)].any() and not W[np.ix_(free, keep)].any()
    W[np.ix_(free, free)] = np.eye(int(free.sum()), dtype=np.int64)
    return Op(W, op.t, free)


def frac_points(rows):
    return [tuple(Fraction(v) for v in p) for p in rows]


def generic_metric(sg, cell_sampler, literal_oracle):
    T = literal_oracle.cell_matrix(cell_sampler(LATTICE_TYPE[sg], 1, seed=1000 + sg)[0])
    return T.T @ T


# ---------------------------------------------------------------------------
# manifest, digest, independence
# ---------------------------------------------------------------------------

def test_manifest_hashes_match():
    """The manifest's input hashes equal the constants imported now, and the table matches its digest."""
    m = verify_manifest()
    assert m['input_hashes'] == X.input_hashes()
    assert m['table_sha256'] == X.file_sha256(TABLE_PATH) == TABLE_HASH
    for k in ('schema_version', 'generator', 'cctbx_version', 'date', 'table_file'):
        assert m.get(k), f'manifest lacks {k}'


def test_table_digest_pinned():
    """Golden pin: regeneration is reviewed and this digest edited by hand."""
    assert X.file_sha256(TABLE_PATH) == PINNED_TABLE_SHA256


def test_manifest_mismatch_raises(tmp_path):
    """Changed inputs, a changed table or a changed schema refuse to load."""
    stale = dict(MANIFEST, input_hashes=dict(MANIFEST['input_hashes'], SYM_OPS='0' * 64))
    with pytest.raises(TableMismatchError, match='SYM_OPS'):
        verify_manifest(stale)
    with pytest.raises(TableMismatchError, match='digest'):
        verify_manifest(dict(MANIFEST, table_sha256='0' * 64))
    with pytest.raises(TableMismatchError, match='schema'):
        verify_manifest(dict(MANIFEST, schema_version=2))
    tampered = tmp_path / 'fd_tables_v1.json'
    tampered.write_bytes(TABLE_PATH.read_bytes().replace(b'"P21/c"', b'"P21/n"', 1))
    with pytest.raises(TableMismatchError, match='digest'):
        verify_manifest(MANIFEST, table_path=tampered)
    verify_manifest(MANIFEST)


def test_crlf_checkout_keeps_the_digest(tmp_path):
    """A CRLF copy of the table (a Windows checkout) verifies and parses to the same table; git keeps LF anyway."""
    raw = TABLE_PATH.read_bytes()
    assert b'\r\n' not in raw
    crlf = tmp_path / 'fd_tables_v1.json'
    crlf.write_bytes(raw.replace(b'\n', b'\r\n'))
    assert crlf.read_bytes() != raw
    verify_manifest(MANIFEST, table_path=crlf)
    assert json.loads(crlf.read_bytes().decode('utf-8'))['groups'] == load_raw()
    attrs = (TABLE_PATH.parent / '.gitattributes').read_text(encoding='utf-8').splitlines()
    assert '*.json -text' in attrs


def test_package_imports():
    """Only the offline generator imports cctbx; nothing in the package imports GFN."""
    pkg = Path(X.__file__).parent
    for path in sorted(pkg.glob('*.py')):
        mods = set()
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                mods |= {a.name.split('.')[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split('.')[0])
        assert not mods & {'gfn_diffusion', 'energy_sampling', 'energies', 'models'}, path.name
        if path.name != 'generate_fd_tables.py':
            assert not mods & {'cctbx', 'libtbx', 'scitbx', 'boost_adaptbx'}, path.name


_FRESH = '''
import sys, torch
import mxtaltools.crystal_fd.tables as T
from mxtaltools.crystal_fd import certify, charts, checks, description, domains, exact, measure, ops, so3
[T.group_table(s) for s in (1, 2, 14, 230)]
assert certify.tiling_certificate(14).ok
ch = charts.CrystalChart.for_latent_v1(2, 1.7, 'euler_zyz')
m = ch.forward(charts.from_unit(ch, torch.rand(4, ch.ndim, dtype=torch.float64))[0])
domains.fold(m.desc)
ops.reduce_cell(m.desc)
measure.log_reference_density(m.desc, ch, 'metric_invariant')
bad = [k for k in sys.modules if k.split('.')[0] in ('cctbx', 'libtbx', 'scitbx', 'boost_adaptbx')]
assert not bad, bad
gfn = sys.argv[1]
files = [str(getattr(v, '__file__', None) or '').replace(chr(92), '/').lower() for v in list(sys.modules.values())]
loaded = [f for f in files if f.startswith(gfn)]
assert not loaded, loaded
'''


def test_loader_runs_without_cctbx():
    """In a fresh interpreter, importing every runtime module and using it loads neither cctbx nor any GFN module."""
    root = Path(X.__file__).parents[2]
    gfn = str(root.parent / 'gfn_diffusion').replace(chr(92), '/').lower()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1',
               PYTHONPATH=os.pathsep.join([str(root), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-c', _FRESH, gfn], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_all_groups_present():
    assert sorted(int(k) for k in load_raw()) == SGS
    g = group_table(14)
    assert group_table(14) is g and g.table_hash == TABLE_HASH


# ---------------------------------------------------------------------------
# consistency with SYM_OPS and NORMALIZER_OPS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', SGS)
def test_rationals_round_to_sym_ops(sg):
    """G is index-aligned with SYM_OPS[sg] and its rationals round to it within 1e-8."""
    e = entry(sg)
    ops = [np.asarray(o, float) for o in SYM_OPS[sg]]
    assert len(e['G']['W']) == len(ops)
    for k, o in enumerate(ops):
        assert np.array_equal(np.array(e['G']['W'][k]).reshape(3, 3), o[:3, :3])
        w = np.array([float(X.to_fraction(q)) for q in e['G']['w'][k]])
        assert np.abs(w - o[:3, 3]).max() <= 1e-8, (sg, k)
        assert e['G']['det'][k] == round(np.linalg.det(o[:3, :3]))
    # identity coset of gamma is G, pointing back at SYM_OPS
    free = free_of(e)
    gam = gamma_ops(e)
    G_free = sym_ops_exact(sg, free)
    for i, (c, s) in enumerate(zip(e['gamma']['coset'], e['gamma']['sym_op'])):
        assert (c == 0) == (s >= 0)
        if s >= 0:
            assert gam[i].key == G_free[s].key
    # listed normaliser representatives round to NORMALIZER_OPS within 1e-8
    listed = NORMALIZER_OPS.get(str(sg), [])
    assert e['normalizer_ops_listed'] == len(listed)
    for rep in e['N_reps']:
        if rep['source'] is not None and rep['source'] >= 0:
            W, w = listed[rep['source']]
            assert np.array_equal(np.array(rep['W']).reshape(3, 3), np.asarray(W))
            d = np.array([float(X.to_fraction(q)) for q in rep['w']]) - np.asarray(w, float)
            d = (d - np.round(d))[~free]
            assert np.abs(d).max(initial=0) <= 1e-8


# ---------------------------------------------------------------------------
# group checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', SGS)
def test_gamma_plus_is_a_group(sg):
    """Gamma and Gamma+ are closed; Gamma+ is the det +1 part, contains G+ and Z^3; L+ is its lattice."""
    e = entry(sg)
    free = free_of(e)
    ops = gamma_ops(e)
    W, t = X.stack_ops(ops)
    det = np.round(np.linalg.det(W.astype(float))).astype(int)
    assert det.tolist() == e['gamma']['det'] and set(np.abs(det).tolist()) == {1}
    assert len({o.key for o in ops}) == len(ops)
    plus = e['gamma_plus']
    assert plus == [i for i, d in enumerate(det) if d == 1]
    assert ops[plus[0]].key == Op(np.eye(3), np.zeros(3), free).key
    assert X.is_closed(W, t, free)
    assert X.is_closed(W[plus], t[plus], free)
    plus_keys = {ops[i].key for i in plus}
    assert {g.key for g in sym_ops_exact(sg, free) if g.det == 1} <= plus_keys
    assert {g.key for g in sym_ops_exact(sg, free)} <= {o.key for o in ops}
    # L+ = Z^3 + pure translations of Gamma+, and it contains Z^3
    L = [[X.to_fraction(v) for v in row] for row in e['lattice_basis']]
    assert X.translation_lattice([ops[i] for i in plus]) == L
    for j in range(3):
        coeff = X.solve(L, [Fraction(int(i == j)) for i in range(3)])
        assert all(c.denominator == 1 for c in coeff), 'Z^3 is not in L+'
    assert Fraction(1) / abs(X.det3(*L)) == e['lattice_index']
    o = e['orders']
    assert (o['N_E'], o['gamma_plus'], o['G']) == (len(ops), len(plus), len(SYM_OPS[sg]))


@pytest.mark.parametrize('sg', SGS)
def test_gamma_equals_brute_force(sg, cell_sampler, literal_oracle):
    """Gamma = closure(SYM_OPS u NORMALIZER_OPS) = brute-force N_E on the 1/24 grid; same for the det +1 parts."""
    e = entry(sg)
    free = free_of(e)
    table = {o.key: o for o in gamma_ops(e)}
    listed = [Op.from_rationals(np.round(np.asarray(W)), [X.rationalize(v) for v in w], free)
              for W, w in NORMALIZER_OPS.get(str(sg), [])]
    assert set(X.closure(sym_ops_exact(sg, free) + listed, free)) == set(table)
    brute = X.brute_force_normalizer(sym_ops_exact(sg), free, generic_metric(sg, cell_sampler, literal_oracle))
    assert set(brute) == set(table)
    assert {k for k, o in brute.items() if o.det == 1} == {k for k, o in table.items() if o.det == 1}


@pytest.mark.parametrize('sg', SGS)
def test_kernel(sg):
    """K (trivial on the projected centroid) is recomputed exactly and is nontrivial exactly in sg 6-9."""
    e = entry(sg)
    free = free_of(e)
    keep = ~free
    ops = gamma_ops(e)
    K = [i for i in e['gamma_plus']
         if np.array_equal(ops[i].W[np.ix_(keep, keep)], np.eye(int(keep.sum()), dtype=np.int64))
         and not ops[i].t[keep].any()]
    assert K == e['kernel']
    assert (len(K) > 1) == (6 <= sg <= 9)
    if 6 <= sg <= 9:
        assert len(K) == 2 and e['pplus']['name'] == 'C2'


# ---------------------------------------------------------------------------
# domain checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sg', SGS)
def test_asu_volumes_exact(sg):
    """Rational vertex enumeration: vol(ASU(Gamma+)) x |Gamma+/K| = 1 and vol(ASU(G)) x |G| = 1, exactly."""
    e = entry(sg)
    for name, order in (('asu', Fraction(len(e['gamma_plus']), len(e['kernel']))), ('g_asu', len(SYM_OPS[sg]))):
        a = e[name]
        normals, offsets = X.cut_planes(a['cuts'])
        assert all(isinstance(v, int) for n in normals for v in n), 'cut normals must be integers'
        verts = X.polytope_vertices(normals, offsets)
        assert verts == sorted(frac_points(a['vertices']))
        assert [X.is_facet(n, c, verts, 3) for n, c in zip(normals, offsets)] == a['facet']
        vol = X.polytope_volume(normals, offsets, verts)
        assert vol == X.to_fraction(a['volume'])
        assert vol * order == 1, f'{name}: vol {vol} x {order}'


def _well_formed(node):
    if 'op' in node:
        assert node['op'] in ('&', '|')
        _well_formed(node['lhs'])
        _well_formed(node['rhs'])
        return
    assert len(node['n']) == 3 and all(isinstance(v, int) for v in node['n']) and any(node['n'])
    X.to_fraction(node['c'])
    assert isinstance(node['inclusive'], bool)
    if node['sub'] is not None:
        _well_formed(node['sub'])


@pytest.mark.parametrize('sg', SGS)
def test_every_facet_glued_with_tie_breaks(sg):
    """Every facet of ASU(Gamma+) has gluing elements whose pieces tile it, and a tie-break record.

    Each gluing element (projected to the non-free coordinates, plus its lattice
    shift) maps the facet's plane onto its landing facet's plane and the outside
    of the facet to the inside of the landing facet. A single-piece facet's
    centroid lands in the closed asymmetric unit.
    """
    e = entry(sg)
    free = free_of(e)
    ops = gamma_ops(e)
    a = e['asu']
    normals, offsets = X.cut_planes(a['cuts'])
    verts = frac_points(a['vertices'])
    plus = set(e['gamma_plus'])
    assert [f['cut'] for f in e['facets']] == [i for i, f in enumerate(a['facet']) if f]
    for f in e['facets']:
        i = f['cut']
        _well_formed(a['cuts'][i])
        assert f['gluings'], f'facet {i} has no gluing'
        shares = [X.to_fraction(g['share']) for g in f['gluings']]
        assert all(0 < s <= 1 for s in shares) and sum(shares) == 1
        n_i, c_i = [Fraction(v) for v in normals[i]], offsets[i]
        for g in f['gluings']:
            assert g['element'] in plus
            p = project(ops[g['element']], free)
            Wp = p.W
            wp = [q + s for q, s in zip(p.w(), g['shift'])]
            j = g['lands_on']
            n_j, c_j = [Fraction(v) for v in normals[j]], offsets[j]
            lhs = [sum(n_j[r] * int(Wp[r, k]) for r in range(3)) for k in range(3)]       # n_j^T W
            k0 = next(k for k in range(3) if n_i[k] != 0)
            lam = -lhs[k0] / n_i[k0]
            assert lam > 0 and all(lhs[k] == -lam * n_i[k] for k in range(3)), (i, g)
            assert sum(n_j[r] * wp[r] for r in range(3)) + c_j == -lam * c_i, (i, g)
            free_normal = not any(n_i[k] for k in range(3) if not free[k])
            identity = np.array_equal(Wp, np.eye(3, dtype=np.int64))
            expect = ('free_axis' if free_normal else 'translation' if identity
                      else 'self_twisted' if j == i else 'pair_twisted')
            assert g['kind'] == expect, (i, g)
        if len(f['gluings']) == 1:
            g = f['gluings'][0]
            face = X.tight(normals[i], offsets[i], verts)
            x = [sum(v[k] for v in face) / len(face) for k in range(3)]
            p = project(ops[g['element']], free)
            y = [sum(int(p.W[r, k]) * x[k] for k in range(3)) + p.w()[r] + g['shift'][r] for r in range(3)]
            assert X.evaluate(normals[g['lands_on']], offsets[g['lands_on']], y) == 0
            assert all(X.evaluate(n, c, y) >= 0 for n, c in zip(normals, offsets))
    # a translation-glued pair without sub-cuts keeps exactly one of its two faces
    for f in e['facets']:
        for g in f['gluings']:
            j = g['lands_on']
            ci, cj = a['cuts'][f['cut']], a['cuts'][j]
            if g['kind'] in ('translation', 'free_axis') and ci['sub'] is None and cj['sub'] is None:
                assert ci['inclusive'] != cj['inclusive'], (f['cut'], j)


@pytest.mark.parametrize('sg', SGS)
def test_ar_bounds_reproduce_asu(sg):
    """The Fourier-Motzkin AR bounds, read as half-spaces, have exactly the vertices of ASU(Gamma+)."""
    e = entry(sg)
    free = free_of(e)
    order = e['ar']['order']
    assert order == [d for d in range(3) if not free[d]]
    stages = e['ar']['stages']
    assert [s['coord'] for s in stages] == order
    if not order:
        return
    st = [dict(coord=s['coord'],
               lower=[([X.to_fraction(v) for v in r['coef']], X.to_fraction(r['const'])) for r in s['lower']],
               upper=[([X.to_fraction(v) for v in r['coef']], X.to_fraction(r['const'])) for r in s['upper']])
          for s in stages]
    for k, s in enumerate(st):
        assert s['lower'] and s['upper']
        assert all(len(c) == k for c, _ in s['lower'] + s['upper'])
    hn, ho = X.ar_halfspaces(st, order)
    got = set(X.polytope_vertices(hn, ho))
    want = {tuple(v[d] for d in order) for v in X.polytope_vertices(*X.cut_planes(e['asu']['cuts']))}
    assert got == want


@pytest.mark.parametrize('sg', SGS)
def test_trainer_box(sg):
    """ASYM_UNITS box, its +1 coverage and k_box: exact identities plus a Monte Carlo cross-check.

    The mean number of Gamma+ images in the box over all crystals is exactly
    |Gamma+| x vol(box) on the non-free coordinates.
    """
    e = entry(sg)
    free = free_of(e)
    keep = ~free
    b = e['trainer_box']
    box = [X.to_fraction(v) for v in b['box']]
    assert [float(v) for v in box] == [float(v) for v in ASYM_UNITS[str(sg)]]
    assert b['placeholder'] == (str(sg) not in RAW_ASYM_UNITS)
    plus = e['gamma_plus']
    vol = Fraction(1)
    for d in range(3):
        if keep[d]:
            vol *= box[d]
    hist = {int(k): X.to_fraction(v) for k, v in b['images_histogram'].items()}
    assert sum(hist.values()) == 1
    assert X.to_fraction(b['mean_images']) == len(plus) * vol == sum(k * v for k, v in hist.items())
    assert X.to_fraction(b['coverage']) == 1 - hist.get(0, 0)
    nonzero = sorted(k for k in hist if k > 0)
    assert (b['k_box'] == nonzero[0]) if len(nonzero) == 1 else b['k_box'] is None
    if not keep.any():
        return
    ops = gamma_ops(e)
    W = np.stack([ops[i].W[np.ix_(keep, keep)] for i in plus]).astype(float)
    w = np.stack([ops[i].t[keep] for i in plus]) / DEN
    rng = np.random.default_rng(sg)
    n = 4000
    x = rng.random((n, int(keep.sum())))
    img = np.einsum('kij,pj->pki', W, x) + w[None]
    img -= np.floor(img)
    hi = np.array([float(box[d]) for d in range(3) if keep[d]])
    counts = np.all(img <= hi, axis=2).sum(1)
    assert set(np.unique(counts).tolist()) <= set(hist)
    cov = float(X.to_fraction(b['coverage']))
    assert abs((counts > 0).mean() - cov) <= 5 * np.sqrt(cov * (1 - cov) / n) + 1e-12


# ---------------------------------------------------------------------------
# cctbx re-derivation
# ---------------------------------------------------------------------------

@pytest.mark.cctbx
@pytest.mark.parametrize('sg', SGS)
def test_gamma_equals_cctbx(sg):
    """Gamma from cctbx alone (normaliser generators + discrete seminvariant shifts) equals the table's."""
    from cctbx import sgtbx
    from mxtaltools.crystal_fd import generate_fd_tables as gen
    e = entry(sg)
    free = free_of(e)
    cct, cfree = gen.cctbx_normalizer(sg)
    assert np.array_equal(cfree, free)
    table = {o.key: o for o in gamma_ops(e)}
    assert set(cct) == set(table)
    assert {k for k, o in cct.items() if o.det == 1} == {k for k, o in table.items() if o.det == 1}
    g = sgtbx.space_group_info(number=sg).group()
    assert {gen.op_from_rt_mx(g(i)).key for i in range(g.order_z())} == {o.key for o in sym_ops_exact(sg)}


@pytest.mark.cctbx
@pytest.mark.parametrize('sg', SGS)
def test_asus_equal_cctbx(sg):
    """cctbx re-derives both asymmetric units, cut for cut, with the same tie-break trees."""
    from cctbx import sgtbx
    from mxtaltools.crystal_fd import generate_fd_tables as gen
    e = entry(sg)
    free = free_of(e)
    g_info = sgtbx.space_group_info(group=gen.cctbx_group(sym_ops_exact(sg)))
    ops = gamma_ops(e)
    proj = {}
    for i in e['gamma_plus']:
        p = project(ops[i], free)
        proj.setdefault(p.key, p)
    p_info = sgtbx.space_group_info(group=gen.cctbx_group(list(proj.values())))
    for name, info in (('g_asu', g_info), ('asu', p_info)):
        asu = info.direct_space_asu()
        assert [gen.cut_json(c) for c in asu.cuts] == e[name]['cuts']
        assert [c.as_xyz() for c in asu.cuts] == e[name]['xyz']
        assert int(info.type().number()) == e[name]['type_number']


@pytest.mark.cctbx
@pytest.mark.parametrize('sg', SGS)
def test_tie_break_evaluator_matches_cctbx(sg):
    """exact.asu_contains agrees with cctbx's is_inside on exact points on, just inside and just outside facets."""
    from boost_adaptbx.boost import rational
    from cctbx import sgtbx
    from mxtaltools.crystal_fd import generate_fd_tables as gen
    e = entry(sg)
    free = free_of(e)
    ops = gamma_ops(e)
    proj = {}
    for i in e['gamma_plus']:
        p = project(ops[i], free)
        proj.setdefault(p.key, p)
    infos = {'g_asu': sgtbx.space_group_info(group=gen.cctbx_group(sym_ops_exact(sg))),
             'asu': sgtbx.space_group_info(group=gen.cctbx_group(list(proj.values())))}
    rng = np.random.default_rng(sg)
    eps = Fraction(1, 997)
    for name, info in infos.items():
        cc = info.direct_space_asu()
        cuts = e[name]['cuts']
        normals, offsets = X.cut_planes(cuts)
        verts = frac_points(e[name]['vertices'])
        pts = list(verts)
        for i, f in enumerate(e[name]['facet']):
            if not f:
                continue
            face = X.tight(normals[i], offsets[i], verts)
            for a, b in zip(face, face[1:] + face[:1]):
                pts.append(tuple((x + y) / 2 for x, y in zip(a, b)))
            for _ in range(12):
                wts = [Fraction(int(k), 1) for k in rng.integers(0, 4, len(face))]
                if sum(wts) == 0:
                    continue
                p = tuple(sum(wt * v[k] for wt, v in zip(wts, face)) / sum(wts) for k in range(3))
                n = normals[i]
                pts += [p, tuple(x + eps * n[k] for k, x in enumerate(p)), tuple(x - eps * n[k] for k, x in enumerate(p))]
        assert len(pts) > 20
        for p in pts:
            want = cc.is_inside(tuple(rational.int(q.numerator, q.denominator) for q in p))
            assert X.asu_contains(cuts, p) == bool(want), (name, p)


@pytest.mark.cctbx
@pytest.mark.slow
def test_regeneration_is_byte_identical(tmp_path):
    """Running the generator again reproduces the checked-in table byte for byte (about a minute)."""
    from mxtaltools.crystal_fd import generate_fd_tables as gen
    assert gen.main(['--out', str(tmp_path)]) == 0
    assert (tmp_path / gen.TABLE_NAME).read_bytes() == TABLE_PATH.read_bytes()
