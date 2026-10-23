"""R space groups filed on rhombohedral axes are read on hexagonal axes (obverse), as SYM_OPS lists them.

The rhombohedral-axes operators come from spglib's database (an independent source), never from
SYM_OPS, and the "same crystal" check compares interatomic distances, which need no basis
change, so neither depends on the transformation under test. Before the re-expression existed
every case in `test_rhombohedral_axes_read_on_hexagonal_axes` raised
CifInconsistentSymmetryError (|G| / 3 operators under an R group).
"""
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS
from mxtaltools.dataset_utils.construction.cif import (CifInconsistentSymmetryError, parse_cif, read_cif,
                                                       reexpress_rhombohedral_axes, rhombohedral)
from mxtaltools.dataset_utils.construction.cif.rhombohedral import HALL_SYMBOLS
from mxtaltools.dataset_utils.construction.cif.symmetry import symops_from_block

spglib = pytest.importorskip('spglib')

FIXTURES = Path(__file__).parent / 'fixtures'

#: spglib Hall numbers of the R groups: (rhombohedral axes, hexagonal axes).
_HALL_NUMBERS = {146: (434, 433), 148: (437, 436), 155: (445, 444), 160: (451, 450), 161: (453, 452),
                 166: (459, 458), 167: (461, 460)}
#: Four sites in general position, rhombohedral fractional coordinates.
_SITES = ((('C1', 'C'), (0.1123, 0.2047, 0.3311)), (('C2', 'C'), (0.1734, 0.2411, 0.3790)),
          (('N1', 'N'), (0.0912, 0.2853, 0.4102)), (('H1', 'H'), (0.1450, 0.1902, 0.2817)))


def _op_text(rot, trans) -> str:
    rows = []
    for i in range(3):
        s = ''.join(('+' if k > 0 else '-') + ('' if abs(k) == 1 else f'{abs(k)}*') + ax
                    for k, ax in zip(rot[i], 'xyz') if k)
        t = Fraction(float(trans[i])).limit_denominator(12)
        if t:
            s += f'+{t}' if t > 0 else f'-{-t}'
        rows.append(s.lstrip('+'))
    return ','.join(rows)


def _rhombohedral_ops(number):
    """spglib's operators for the rhombohedral-axes setting: [m, 4, 4]."""
    sym = spglib.get_symmetry_from_database(_HALL_NUMBERS[number][0])
    ops = np.zeros((len(sym['rotations']), 4, 4))
    ops[:, :3, :3], ops[:, :3, 3], ops[:, 3, 3] = sym['rotations'], sym['translations'], 1
    return ops


def rhombohedral_cif(number, *, a=7.3, alpha=81.5, cell=None, hall='rhombohedral', ops=None, sites=_SITES,
                     z_value='one', aniso=False, name='rh'):
    """CIF text of a crystal of ``number`` on rhombohedral axes, Z' = 1 by default."""
    ops = _rhombohedral_ops(number) if ops is None else ops
    cell = (a, a, a, alpha, alpha, alpha) if cell is None else cell
    hall = HALL_SYMBOLS[number][0] if hall == 'rhombohedral' else hall
    lines = [f'data_{name}', '_symmetry_cell_setting rhombohedral',
             f"_symmetry_space_group_name_H-M '{SPACE_GROUPS[number]}'", f'_symmetry_Int_Tables_number {number}']
    if hall is not None:
        lines.append(f"_space_group_name_Hall '{hall}'")
    lines += ['loop_', '_symmetry_equiv_pos_site_id', '_symmetry_equiv_pos_as_xyz']
    lines += [f'{i + 1} {_op_text(o[:3, :3].astype(int), o[:3, 3])}' for i, o in enumerate(ops)]
    for tag, v in zip(('a', 'b', 'c', 'alpha', 'beta', 'gamma'), cell):
        lines.append(f'_cell_{"length" if len(tag) == 1 else "angle"}_{tag} {v}')
    lines.append(f'_cell_volume 380.5')
    if z_value is not None:
        lines.append(f'_cell_formula_units_Z {len(ops) if z_value == "one" else z_value}')
    lines += ['loop_', '_atom_site_label', '_atom_site_type_symbol',
              '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
    lines += [f'{lab} {el} {x} {y} {z}' for (lab, el), (x, y, z) in sites]
    if aniso:
        lines += ['loop_', '_atom_site_aniso_label', '_atom_site_aniso_U_11', 'C1 0.05']
    lines += ['loop_', '_geom_bond_atom_site_label_1', '_geom_bond_atom_site_label_2',
              '_geom_bond_site_symmetry_1', '_geom_bond_site_symmetry_2',
              'C1 C2 1_555 1_555', 'C2 N1 1_555 1_555', 'C1 H1 1_555 1_555']
    return '\n'.join(lines) + '\n'


def _lattice(a, b, c, alpha, beta, gamma):
    """Lattice vectors as rows (a along x, b in the xy plane); lengths in A, angles in degrees."""
    ca, cb, cg = np.cos(np.radians([alpha, beta, gamma]))
    sg = np.sin(np.radians(gamma))
    cx, cy = c * cb, c * (ca - cb * cg) / sg
    return np.array([[a, 0, 0], [b * cg, b * sg, 0], [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]])


def _distances(lattice, frac_sites, ops, cutoff=6.0):
    """Per deposited site, its sorted distances (A) to every atom of the crystal within ``cutoff``."""
    images = (np.einsum('mij,nj->nmi', ops[:, :3, :3], frac_sites) + ops[None, :, :3, 3]).reshape(-1, 3)
    shifts = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)])
    every = (images[:, None, :] + shifts[None]).reshape(-1, 3) @ lattice
    out = []
    for x in frac_sites @ lattice:
        d = np.linalg.norm(every - x, axis=1)
        out.append(np.sort(d[(d > 1e-6) & (d < cutoff)]))
    return out


@pytest.mark.parametrize('number', sorted(HALL_SYMBOLS))
def test_rhombohedral_axes_read_on_hexagonal_axes(number):
    """The read crystal states hexagonal axes: SYM_OPS[n] exactly, the tripled cell and Z, the same Z'."""
    ops_r = _rhombohedral_ops(number)
    assert 3 * len(ops_r) == len(SYM_OPS[number])
    c = read_cif(rhombohedral_cif(number))[0]
    assert c.reexpression is not None and c.reexpression.startswith('re-expressed from rhombohedral axes')
    assert c.symmetry.number == number and c.symmetry.operators_are_standard
    assert len(c.symmetry.operators) == len(SYM_OPS[number])
    assert all(np.array_equal(o, np.asarray(s, dtype=float)) for o, s in zip(c.symmetry.operators, SYM_OPS[number]))
    a, alpha = 7.3, np.radians(81.5)
    np.testing.assert_allclose(c.cell_lengths, [2 * a * np.sin(alpha / 2)] * 2 + [a * np.sqrt(3 + 6 * np.cos(alpha))],
                               rtol=1e-14)
    assert np.array_equal(c.cell_angles, np.radians([90.0, 90.0, 120.0]))
    assert c.z_value == 3 * len(ops_r) and c.z_prime == 1
    frac_r = np.array([x for _, x in _SITES])
    p_inv = np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]]) / 3
    np.testing.assert_allclose(c.sites.frac, frac_r @ p_inv.T, atol=1e-15)
    assert c.sites.labels == [lab for (lab, _), _ in _SITES] and c.components.n_components == 1


@pytest.mark.parametrize('number', sorted(HALL_SYMBOLS))
def test_reexpressed_crystal_is_the_same_crystal(number):
    """Distances from every deposited atom to its neighbours agree between the two descriptions, and spglib
    finds the declared group in the hexagonal cell built from SYM_OPS."""
    a, alpha = 7.3, 81.5
    c = read_cif(rhombohedral_cif(number, a=a, alpha=alpha))[0]
    lat_r = _lattice(a, a, a, alpha, alpha, alpha)
    lat_h = _lattice(*c.cell_lengths, *np.degrees(c.cell_angles))
    assert abs(np.linalg.det(lat_h) / np.linalg.det(lat_r) - 3) < 1e-12
    frac_r = np.array([x for _, x in _SITES])
    ops_h = np.stack(c.symmetry.operators)
    for d_r, d_h in zip(_distances(lat_r, frac_r, _rhombohedral_ops(number)), _distances(lat_h, c.sites.frac, ops_h)):
        assert len(d_r) == len(d_h) > 0
        np.testing.assert_allclose(d_h, d_r, atol=1e-6)     # SYM_OPS holds 2/3 to 8 decimals
    images = np.mod(np.einsum('mij,nj->nmi', ops_h[:, :3, :3], c.sites.frac) + ops_h[None, :, :3, 3], 1).reshape(-1, 3)
    numbers = np.tile(c.sites.z, (len(ops_h), 1)).T.reshape(-1)
    data = spglib.get_symmetry_dataset((lat_h, images, numbers), symprec=1e-4)
    assert data.number == number


def test_hall_symbols_are_spglibs():
    for number, (rh, hx) in HALL_SYMBOLS.items():
        r, h = (spglib.get_spacegroup_type(k) for k in _HALL_NUMBERS[number])
        assert (r.number, r.choice, r.hall_symbol) == (number, 'R', rh)
        assert (h.number, h.choice, h.hall_symbol) == (number, 'H', hx)


def test_other_blocks_are_returned_as_they_are():
    """Every fixture, an R group on hexagonal axes, and an R group on rhombohedral axes whose coordinates
    cannot be read: the same block object comes back, with no note."""
    blocks = [b for p in sorted(FIXTURES.glob('*.cif')) for b in parse_cif(p.read_text(errors='ignore'))]
    assert len(blocks) == 16
    hexagonal = rhombohedral_cif(148, cell=(10.9, 10.9, 12.9, 90, 90, 120), hall='-R 3', ops=np.stack(SYM_OPS[148]),
                                 z_value=18)
    blocks += parse_cif(hexagonal)
    unreadable = rhombohedral_cif(148, sites=((('C1', 'C'), (0.1, '?', 0.3)),))
    blocks += parse_cif(unreadable)
    for b in blocks:
        out, note = reexpress_rhombohedral_axes(b, b.name)
        assert out is b and note is None, b.name
    c = read_cif(hexagonal)[0]
    assert c.reexpression is None and c.symmetry.operators_are_standard
    with pytest.raises(CifInconsistentSymmetryError, match='do not factorise'):   # refused as before
        read_cif(unreadable)


def test_the_block_is_not_modified():
    block = parse_cif(rhombohedral_cif(167, aniso=True))[0]
    tags, loops = dict(block.tags), [(list(l.tags), [list(r) for r in l.rows]) for l in block.loops]
    out, note = reexpress_rhombohedral_axes(block, 'rh')
    assert out is not block and note is not None
    assert block.tags == tags and [(list(l.tags), [list(r) for r in l.rows]) for l in block.loops] == loops
    assert out.get('_cell_formula_units_Z') == str(3 * 12) and out.get('_cell_volume') == repr(3 * 380.5)
    assert out.get('_space_group_name_Hall') == HALL_SYMBOLS[167][1]
    assert out.get('_symmetry_cell_setting') == 'trigonal'
    assert out.loop_with('_atom_site_aniso_label') is None          # displacements on the old axes are dropped
    assert out.loop_with('_geom_bond_atom_site_label_1') is block.loop_with('_geom_bond_atom_site_label_1')


def _shifted_origin(number, t=(0.1, 0.0, 0.0)):
    """spglib's rhombohedral operators with the origin moved by ``t``: the same group, another origin."""
    ops = _rhombohedral_ops(number)
    ops[:, :3, 3] += (ops[:, :3, :3] - np.eye(3)) @ np.asarray(t)
    return ops


@pytest.mark.parametrize('kwargs,match', [
    (dict(cell=(7.3, 7.31, 7.3, 81.5, 81.5, 81.5)), 'is not rhombohedral'),
    (dict(cell=(7.3, 7.3, 7.3, 81.5, 81.5, 82.0)), 'is not rhombohedral'),
    (dict(hall='-R 3'), 'Hall symbol'),
    (dict(ops=_shifted_origin(148)), 'are not SYM_OPS'),
])
def test_rhombohedral_count_that_does_not_reexpress_is_refused(kwargs, match):
    with pytest.raises(CifInconsistentSymmetryError, match=match):
        read_cif(rhombohedral_cif(148, **kwargs))


def test_without_z_or_hall_symbol():
    """Z and the Hall symbol are optional: absent before, absent after."""
    block = parse_cif(rhombohedral_cif(161, hall=None, z_value=None))[0]
    out, note = reexpress_rhombohedral_axes(block, 'rh')
    assert note is not None and out.get('_cell_formula_units_Z') is None and out.get('_space_group_name_Hall') is None


def _overflowing(text: str) -> str:
    """``text`` with its last operator's z component shifted by 1e400, a term no float holds."""
    lines = text.split('\n')
    i = next(k for k, line in enumerate(lines) if line.startswith('_cell_length_a'))
    lines[i - 1] += '+1e400'
    return '\n'.join(lines)


@pytest.mark.parametrize('cell', [None, (10.9, 10.9, 12.9, 90, 90, 120)], ids=['rhombohedral', 'hexagonal'])
def test_operators_that_overflow_are_left_to_the_regular_read(cell):
    """An operator term that overflows a float under an R group, on either axes: the block comes back as it is
    (the regular read then refuses it as for any group). The operator parser raises OverflowError, which the
    re-expression let through before it caught it."""
    block = parse_cif(_overflowing(rhombohedral_cif(148, cell=cell)))[0]
    with pytest.raises(OverflowError):
        symops_from_block(block, 'rh')
    out, note = reexpress_rhombohedral_axes(block, 'rh')
    assert out is block and note is None


@pytest.mark.parametrize('number', sorted(HALL_SYMBOLS))
def test_hm_symbol_alone_names_the_group(number):
    """No IT number: the group is read from the H-M symbol, as space_group_from_block reads it."""
    text = rhombohedral_cif(number).replace(f'_symmetry_Int_Tables_number {number}\n', '')
    assert 'Int_Tables_number' not in text
    c = read_cif(text)[0]
    assert c.reexpression is not None and c.symmetry.number == number and c.symmetry.operators_are_standard
    assert c.z_value == 3 * len(_rhombohedral_ops(number))


def _deviations(message: str):
    """(metric deviation, image deviation) as the refusal states them."""
    m = re.search(r'metric deviation (\S+) relative, image deviation (\S+) A', message)
    return float(m.group(1)), float(m.group(2))


def test_a_defective_coordinate_transform_does_not_rebuild(monkeypatch):
    """The rebuild check catches a wrong x_H = P^-1 x_R (here P^-T, the transpose); the metric check does not."""
    monkeypatch.setattr(rhombohedral, '_hexagonal_frac', lambda f: f @ rhombohedral._P_INV3 / 3)
    with pytest.raises(CifInconsistentSymmetryError, match='does not rebuild') as e:
        read_cif(rhombohedral_cif(148))
    metric, image = _deviations(str(e.value))
    assert metric <= rhombohedral._METRIC_TOL < rhombohedral._REBUILD_TOL < image


def test_a_defective_cell_formula_fails_the_metric_check(monkeypatch):
    """The metric check catches a_H off by one part in 10^7; the rebuild check (fractional) does not."""
    real = rhombohedral._hexagonal_lengths
    monkeypatch.setattr(rhombohedral, '_hexagonal_lengths',
                        lambda a, alpha: (real(a, alpha)[0] * (1 + 1e-7), real(a, alpha)[1]))
    with pytest.raises(CifInconsistentSymmetryError, match='does not rebuild') as e:
        read_cif(rhombohedral_cif(167))
    metric, image = _deviations(str(e.value))
    assert image <= rhombohedral._REBUILD_TOL and metric > rhombohedral._METRIC_TOL
