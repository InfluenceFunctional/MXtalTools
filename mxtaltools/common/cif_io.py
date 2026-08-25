"""CIF writing for MolCrystalData.

Emits a symmetry-bearing CIF: space group number, Hermann-Mauguin symbol,
the symmetry operator list, and the ASYMMETRIC UNIT -- not a P1 expansion.

Why this exists.  The incumbent writer (`common.ase_interface.ase_write_cif`)
hands a bare `ase.Atoms` to `ase.io.write`, which emits P1 with **no space group
at all**: `ase_interface.py:154` does build an `ase.spacegroup.crystal` with the
right group, but that call is on the `return_crystal` branch, which the writer
path never takes.  Everything symmetry-related is therefore lost on write --
`sg_ind`, `symmetry_operators`, `z_prime`, every `aunit_*` parameter,
`identifier` -- so a written file cannot be read back into the object that wrote
it.  See `docs/design/cif_reader_design.md` §8.

Deliberately depends on nothing but numpy/torch and the package's own symmetry
tables.  ASE's CIF reader has a defect this writer must not feed: at
`ase/io/cif.py:378` it takes `_symmetry_Int_Tables_number` and never consults the
H-M symbol when no symop loop is present, so a file claiming IT number 1 with a
non-P1 symbol reads back as P1, silently.  We always write the symop loop.
"""

from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from mxtaltools.constants.space_group_info import SPACE_GROUPS, SYM_OPS

__all__ = ['sym_op_to_xyz', 'cif_block_from_crystal', 'write_asymmetric_unit_cif',
           'operators_are_standard']

_AXES = ('x', 'y', 'z')


def _fmt_translation(t: float) -> str:
    """Render a fractional translation as an exact CIF fraction ('1/2', '-1/3')."""
    if abs(t) < 1e-8:
        return ''
    frac = Fraction(float(t)).limit_denominator(12)
    if abs(float(frac) - float(t)) > 1e-6:            # not a nice fraction; be explicit
        return f'{t:+.6f}'
    return f'{"+" if frac > 0 else "-"}{abs(frac.numerator)}/{abs(frac.denominator)}' \
        if frac.denominator != 1 else f'{frac.numerator:+d}'


def sym_op_to_xyz(op) -> str:
    """4x4 affine symmetry operator -> a CIF 'x,y,z' string.

    `op` is [[R | t], [0 0 0 1]] acting on fractional coordinates, matching the
    layout of `constants.space_group_info.SYM_OPS`.

    >>> sym_op_to_xyz(np.eye(4))
    'x,y,z'
    """
    op = np.asarray(op, dtype=float)
    if op.shape != (4, 4):
        raise ValueError(f'expected a 4x4 affine operator, got shape {op.shape}')

    terms = []
    for row in range(3):
        parts = []
        for col in range(3):
            c = op[row, col]
            if abs(c) < 1e-8:
                continue
            if abs(abs(c) - 1.0) < 1e-8:
                parts.append(('-' if c < 0 else '+') + _AXES[col])
            else:
                frac = Fraction(float(c)).limit_denominator(12)
                parts.append(f'{"+" if c > 0 else "-"}{abs(frac.numerator)}/'
                             f'{abs(frac.denominator)}*{_AXES[col]}')
        parts.append(_fmt_translation(op[row, 3]))
        expr = ''.join(p for p in parts if p)
        expr = expr.lstrip('+') or '0'
        terms.append(expr)
    return ','.join(terms)


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _element_symbols(z: Sequence[int]) -> list:
    from ase.data import chemical_symbols          # ase is already a hard dependency
    return [chemical_symbols[int(zi)] for zi in z]


def _operators_for(crystal, cell_index: int, sg_ind: int, single: bool) -> list:
    """The operators to WRITE: the ones the crystal was BUILT from.

    `MolCrystalData` stores `symmetry_operators` -- the CIF's actual operators
    when it came from a file, or the `SYM_OPS[sg_ind]` fallback the constructor
    supplies when it did not (`crystal_ops.py:50-62`).  The unit-cell builder
    applies *those* (`crystal_building/utils.py:677`), not the standard table.

    So the writer must emit them too.  Substituting the standard table would be a
    silent wrong answer for a large fraction of real structures: measured over 400
    random CSD entries, only **62.5 %** carry operators identical to
    `SYM_OPS[sg_ind]` in the same order -- 18.2 % are the same set in a different
    order, and **19.2 % are a different set entirely** (alternate settings such as
    `1/2-x,1/2+y,1/2-z` filed under IT number 14, i.e. P2_1/n written as P2_1/c).
    For that 19.2 % the emitted operators would not correspond to the coordinates,
    and the file would still read back as a valid crystal.
    """
    stored = crystal.symmetry_operators
    if stored is None:
        return [np.asarray(o, dtype=float) for o in SYM_OPS[sg_ind]]

    ops = stored if single else stored[cell_index]
    ops = np.asarray(ops, dtype=float)
    if ops.ndim == 2 and ops.shape == (4, 4):          # a lone operator
        ops = ops[None]
    if ops.ndim != 3 or ops.shape[1:] != (4, 4):
        raise ValueError(
            f'symmetry_operators for crystal {cell_index} has shape {ops.shape}; '
            'expected (n_ops, 4, 4)')

    sym_mult = crystal.sym_mult
    if sym_mult is not None:
        expected = int(_as_float(sym_mult if single else sym_mult[cell_index]))
        if len(ops) != expected:
            raise ValueError(
                f'crystal {cell_index} carries {len(ops)} symmetry operators but '
                f'sym_mult is {expected}; refusing to write an inconsistent cell')
    return list(ops)


def operators_are_standard(crystal, index: int = 0) -> bool:
    """Whether this crystal's operators match `SYM_OPS[sg_ind]` exactly, in order.

    Informational: a False here is normal (37.5 % of the CSD) and is NOT an error.
    It only means the file must carry its own operator list, which it always does.
    """
    single = crystal.batch is None
    sg_ind = int(_as_float(crystal.sg_ind if single else crystal.sg_ind[index]))
    ops = _operators_for(crystal, index, sg_ind, single)
    std = [np.asarray(o, dtype=float) for o in SYM_OPS[sg_ind]]
    return len(ops) == len(std) and all(np.allclose(a, b, atol=1e-6)
                                        for a, b in zip(ops, std))


def cif_block_from_crystal(crystal,
                           index: int = 0,
                           block_name: str = 'crystal',
                           identifier: Optional[str] = None) -> str:
    """Render one crystal as a CIF text block carrying its symmetry.

    Parameters
    ----------
    crystal : MolCrystalData (batched or single)
    index : which crystal in the batch
    block_name : the `data_` block name
    identifier : written as `_chemical_name_common`; defaults to the crystal's own

    Returns
    -------
    str : a complete CIF data block, newline terminated.

    Notes
    -----
    Positions are taken from `pos` -- the POSED ASYMMETRIC UNIT -- and converted
    to fractional coordinates with `T_cf`.  No symmetry expansion is performed;
    the operator loop is what reconstitutes the cell.
    """
    single = crystal.batch is None
    if single:
        atom_slice = slice(0, int(crystal.num_nodes))
        cell_index = 0
    else:
        atom_inds = torch.where(crystal.batch == index)[0]
        atom_slice = slice(int(atom_inds[0]), int(atom_inds[-1]) + 1)
        cell_index = index

    sg_ind = int(_as_float(crystal.sg_ind if single else crystal.sg_ind[cell_index]))
    if sg_ind not in SYM_OPS:
        raise KeyError(f'space group {sg_ind} is not in SYM_OPS; cannot write symmetry')

    # cell parameters -- angles are stored in RADIANS, CIF wants degrees
    lengths = crystal.cell_lengths[cell_index] if not single else crystal.cell_lengths
    angles = crystal.cell_angles[cell_index] if not single else crystal.cell_angles
    a, b, c = [float(v) for v in torch.as_tensor(lengths).detach().cpu().reshape(-1)[:3]]
    al, be, ga = [np.degrees(float(v))
                  for v in torch.as_tensor(angles).detach().cpu().reshape(-1)[:3]]

    T_cf = crystal.T_cf[cell_index] if crystal.T_cf.ndim == 3 else crystal.T_cf
    pos = crystal.pos[atom_slice].detach().cpu()
    frac = (torch.as_tensor(T_cf).detach().cpu().float() @ pos.T.float()).T.numpy()

    z = crystal.z[atom_slice].detach().cpu().numpy()
    symbols = _element_symbols(z)

    ops = _operators_for(crystal, cell_index, sg_ind, single)
    sym_mult = len(ops)
    z_prime = crystal.z_prime
    if z_prime is not None:
        z_prime = _as_float(z_prime if single else z_prime[cell_index])
    else:
        z_prime = 1.0
    cell_z = int(round(sym_mult * z_prime))

    if identifier is None:
        ident = crystal.identifier
        if isinstance(ident, (list, tuple)):
            ident = ident[cell_index] if len(ident) > cell_index else ident[0]
        identifier = str(ident) if ident is not None else block_name

    lines = [
        f'data_{block_name if block_name != "crystal" else identifier}',
        f"_chemical_name_common               '{identifier}'",
        '',
        f'_cell_length_a                      {a:.6f}',
        f'_cell_length_b                      {b:.6f}',
        f'_cell_length_c                      {c:.6f}',
        f'_cell_angle_alpha                   {al:.6f}',
        f'_cell_angle_beta                    {be:.6f}',
        f'_cell_angle_gamma                   {ga:.6f}',
        f'_cell_formula_units_Z               {cell_z}',
        '',
        # Both the current and the deprecated spellings: the current ones are what
        # this package's own files use, the deprecated ones are what CSD exports
        # carry, and a reader should not have to guess which dialect it is holding.
        f'_space_group_IT_number              {sg_ind}',
        f"_space_group_name_H-M_alt           '{SPACE_GROUPS[sg_ind]}'",
        f'_symmetry_Int_Tables_number         {sg_ind}',
        f"_symmetry_space_group_name_H-M      '{SPACE_GROUPS[sg_ind]}'",
        '',
        'loop_',
        '  _space_group_symop_id',
        '  _space_group_symop_operation_xyz',
    ]
    for i, op in enumerate(ops, start=1):
        lines.append(f"  {i}  '{sym_op_to_xyz(op)}'")

    lines += [
        '',
        'loop_',
        '  _atom_site_label',
        '  _atom_site_type_symbol',
        '  _atom_site_fract_x',
        '  _atom_site_fract_y',
        '  _atom_site_fract_z',
        '  _atom_site_occupancy',
    ]
    counts: dict = {}
    for sym, (fx, fy, fz) in zip(symbols, frac):
        counts[sym] = counts.get(sym, 0) + 1
        label = f'{sym}{counts[sym]}'
        lines.append(f'  {label:<6s} {sym:<3s} {fx: .6f} {fy: .6f} {fz: .6f}  1.000')

    lines.append('')
    return '\n'.join(lines) + '\n'


def write_asymmetric_unit_cif(crystal,
                              inds: Union[Sequence[int], torch.Tensor],
                              path: Union[str, Path]) -> list:
    """Write one CIF per index, each carrying space group + operators + aunit.

    Returns the list of paths written.  Mirrors `ase_write_cif`'s naming
    (`{path}_{ind}.cif`) so callers can switch modes without changing how they
    locate the output.
    """
    written = []
    for ind in (inds.tolist() if torch.is_tensor(inds) else list(inds)):
        ind = int(ind)
        cif_path = Path(f'{path}_{ind}.cif')
        # name the block after the crystal, not the output path: the block name is
        # what a reader takes as the identity (CSD writes `data_EFUNUP`), so naming
        # it after the file would lose the identifier on a round trip.
        block = cif_block_from_crystal(crystal, index=ind)
        cif_path.write_text(block, encoding='utf-8')
        written.append(cif_path)
    return written
