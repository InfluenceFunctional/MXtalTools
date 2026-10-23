import warnings
from typing import Union

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import compute_Ip_handedness, rotvec2rotmat, batch_molecule_principal_axes_torch, \
    extract_rotmat, apply_rotation_to_batch, rotmat2rotvec, \
    fractional_transform, center_batch
from mxtaltools.common.utils import block_repeat_interleave

###############################################################################
# the atom -> unit map of a crystal row (audit #1, stage 2)
###############################################################################

#: Stored key of the atom -> unit map: per atom of a crystal ROW, the row-local index (0 .. z_prime - 1, long) of the
#: asymmetric-unit molecule ("unit") the atom belongs to; the single authority for which unit an atom belongs to.
#: The name avoids 'index', 'face' and 'batch': collation offsets those keys (MXtalBase.__inc__ / __cat_dim__) and
#: canonicalize_zp_aunits renumbers 'index' / 'face' keys as atom indices. Cluster batches do not carry it (their
#: ``mol_ind`` numbers the cluster's molecules). ``mol_ind`` on rows is still written where it was; the builders,
#: analyses and ingestion steps no longer read it as the unit map. Since stage 3 (2026-10-23) ``find_molecules`` and
#: ``from_arrays`` write the map, ``MolCrystalChem.sanitize`` fills it for the hydrogens it places and reads units (as
#: ``features`` does) through ``unit_layout``, and ``parameterize_pose`` checks and keeps it (it writes it from
#: ``mol_ind`` only on a batch written before). The batch helpers file it per atom (``data_classes.KEY_LEVELS``).
UNIT_MAP_KEY = 'unit_ind'

#: ``unit_ind`` on every atom of a row whose units are not known (``z_prime < 1``, atoms that do not split into
#: ``z_prime`` equal blocks, or units found that are not those blocks): outside 0 .. z_prime - 1 for every z_prime,
#: so every reader of the map refuses the row, as it refuses such a row without a map.
NO_UNIT = -1


class UnitMapError(ValueError):
    """A row's units cannot be read at this stage. ``reasons``: {row index: why}, every refused row (the message
    names at most ``_UNIT_ROWS_NAMED`` of them); empty when the refusal is not about particular rows."""

    def __init__(self, message: str, reasons: dict):
        super().__init__(message)
        self.reasons = reasons


#: unit-map refusal messages name at most this many rows, then count the rest
_UNIT_ROWS_NAMED = 8


def _unit_row_name(batch, b: int) -> str:
    """'row 3 (ABC)': the row's index and, where one is stored, its identifier."""
    ident = batch['identifier'] if 'identifier' in batch._store else None
    if isinstance(ident, str):
        ident = [ident]
    return f'row {b}' + (f' ({ident[b]})' if isinstance(ident, (list, tuple)) and b < len(ident) else '')


def equal_block_unit_ind(counts: Tensor, z_prime: Tensor, fits: OptTensor = None,
                         total: Union[int, None] = None) -> Tensor:
    """The layout of a row without a unit map: ``z_prime`` equal contiguous blocks, block ``k`` unit ``k``.

    Parameters
    ----------
    counts, z_prime : torch.Tensor
        [B] long: each row's atom count and number of units.
    fits : torch.Tensor, optional
        [B] bool, the rows whose atoms split into their units; default ``z_prime >= 1`` and ``counts`` divisible
        by ``z_prime``. Every atom of another row gets unit 0.
    total : int, optional
        ``counts.sum()`` when the caller knows it on the host (saves a device sync).

    Returns
    -------
    torch.Tensor
        [sum(counts)] long on ``counts``' device: the row-local unit of each atom, rows in order.
    """
    counts = counts.long()
    z_prime = z_prime.long()
    if fits is None:
        fits = (z_prime >= 1) & (counts % z_prime.clamp(min=1) == 0)
    n = int(counts.sum()) if total is None else int(total)
    device = counts.device
    atom_row = torch.repeat_interleave(torch.arange(len(counts), device=device), counts, output_size=n)
    local = torch.arange(n, device=device) - (torch.cumsum(counts, 0) - counts)[atom_row]
    per_unit = torch.where(fits, counts // z_prime.clamp(min=1), torch.ones_like(counts))
    per_unit = per_unit.clamp(min=1)    # an empty row that 'fits' holds no atom to divide
    return torch.where(fits[atom_row], local // per_unit[atom_row], torch.zeros_like(local))


def unit_map_problem(unit_ind, n_atoms: Union[int, None] = None) -> Union[str, None]:
    """Why ``unit_ind`` cannot be a unit map, or None: it must be a tensor of an integer dtype (not bool), one-dimensional,
    and, when ``n_atoms`` is given, hold one entry per atom. Shape and dtype only; the values are checked against
    ``z_prime`` by ``refuse_unit_map_disagreeing_with_z_prime`` (construction) and ``unit_layout`` (readers)."""
    if not torch.is_tensor(unit_ind):
        return f'it is a {type(unit_ind).__name__}, not a tensor'
    if unit_ind.dtype == torch.bool or unit_ind.is_floating_point() or unit_ind.is_complex():
        return f'its dtype is {unit_ind.dtype}, not an integer dtype (long)'
    if unit_ind.ndim != 1:
        return f'its shape is {tuple(unit_ind.shape)}, not one entry per atom [num_atoms]'
    if n_atoms is not None and len(unit_ind) != int(n_atoms):
        return f'it holds {len(unit_ind)} entries for {int(n_atoms)} atoms'
    return None


def derived_unit_ind(data, caller: str = 'derived_unit_ind') -> Tensor:
    """The unit map of crystal rows that carry none, written out: what every reader reads them as (``unit_layout``'s
    absence path). ``z_prime`` equal contiguous blocks per row, block ``k`` unit ``k``; ``NO_UNIT`` (-1) on every atom
    of a row whose units that path cannot read (``z_prime < 1``, or atoms not divisible by ``z_prime``), so the
    written map is refused wherever the absent one was. Used to give a map to the map-less side when rows with and
    without one are joined (``MXtalBase.append_batch``; ``collate_data_list`` since stage 3, 2026-10-23).

    Parameters
    ----------
    data : MolCrystalData (single row or batch) without ``unit_ind``
        Reads ``num_atoms`` (``ptr`` without it), ``z_prime`` (one unit per row without it) and the atoms held
        (``pos``, else ``z``). Not modified.
    caller : str
        Named in the refusal message.

    Returns
    -------
    torch.Tensor
        [atoms held] long, on the device of the rows' atom counts.

    Raises
    ------
    ValueError
        If ``data`` already carries a map, or its rows' atom counts do not sum to the atoms it holds (no map can be
        laid over its atoms). Explicit raises: they hold under ``python -O``.
    """
    store = data._store
    if UNIT_MAP_KEY in store:
        raise ValueError(f'{caller}: the rows already carry a unit map ({UNIT_MAP_KEY}); nothing to derive')
    layout, bad, _ = _unit_layout_unchecked(data)
    held = store['pos'] if 'pos' in store and store['pos'] is not None else store.get('z')
    n = len(held) if held is not None else 0
    total = int(layout.counts.sum())
    if total != n:
        raise ValueError(f'{caller}: the rows\' atom counts (num_atoms) sum to {total}, but {n} atoms are held; '
                         f'their unit map cannot be derived')
    counts = layout.counts
    atom_row = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts, output_size=n)
    derived = equal_block_unit_ind(counts, layout._z_prime, layout._fits, total=n)
    return torch.where(bad[atom_row], torch.full_like(derived, NO_UNIT), derived)


class UnitLayout:
    """The units of a batch's crystal rows, as ``unit_layout`` reads them. Nothing here is stored on the batch.

    Attributes
    ----------
    counts : torch.Tensor
        [B] long: atoms of each row's asymmetric unit (``num_atoms``; ``ptr`` differences without it).
    units_per_row : torch.Tensor
        [B] long: ``z_prime`` (clamped at 0), the number of units of each row (``num_units``).
    atoms_per_unit : torch.Tensor
        [B] long: ``counts // z_prime``, the atom count of every unit of the row (units are equal at this stage).
        Meaningful on the rows ``unit_layout`` checked.
    stored : bool
        Whether the batch carries a unit map (``unit_ind``); otherwise the equal-block layout is derived.
    unit_ind : torch.Tensor (property)
        [sum(counts)] long: each atom's row-local unit, the stored map or the derived layout (equal on the rows that
        pass).
    unit_ptr : torch.Tensor (property)
        [B + 1] long: cumulative unit offsets, ``[0, z'_0, z'_0 + z'_1, ...]``.
    unit_atom_counts : torch.Tensor (property)
        [sum(z_prime)] long: atoms of each unit, rows in order.
    unit_batch : torch.Tensor (property)
        [sum(counts)] long: the batch-wide unit of each atom, ``unit_ptr[row] + unit_ind``.

    On a row batch ``counts`` are its atoms; on a cluster they are the asymmetric units it was built from (a
    cluster keeps the rows' ``num_atoms`` and ``z_prime``), so ``unit_ind`` / ``unit_batch`` index those, not the
    cluster's atoms.
    """

    def __init__(self, counts, units_per_row, atoms_per_unit, z_prime, stored_map, fits):
        self.counts = counts
        self.units_per_row = units_per_row
        self.atoms_per_unit = atoms_per_unit
        self.stored = stored_map is not None
        self._z_prime = z_prime
        self._fits = fits
        self._unit_ind = stored_map

    @property
    def unit_ind(self) -> Tensor:
        if self._unit_ind is None:
            self._unit_ind = equal_block_unit_ind(self.counts, self._z_prime, self._fits)
        return self._unit_ind

    @property
    def unit_ptr(self) -> Tensor:
        return torch.cat([self.units_per_row.new_zeros(1), torch.cumsum(self.units_per_row, 0)])

    @property
    def unit_atom_counts(self) -> Tensor:
        return self.atoms_per_unit.repeat_interleave(self.units_per_row)

    @property
    def unit_batch(self) -> Tensor:
        unit_ind = self.unit_ind
        start = self.unit_ptr[:-1].repeat_interleave(self.counts, output_size=len(unit_ind))
        return start + unit_ind


def _unit_layout_unchecked(batch):
    """``(layout, bad, reason)``: the layout of every row, a [B] bool tensor of the rows whose units cannot be read
    (no device sync), and ``reason(b) -> str`` for a refused row (host reads; call only when refusing).

    A row without a map (no ``unit_ind`` stored) is refused when ``z_prime < 1`` or its atoms do not divide by
    ``z_prime``. A row with a map is refused unless its values are 0 .. z_prime - 1, every unit holds an atom, the
    units hold equal atom counts, and they are contiguous blocks in unit order: at this stage the map must equal
    the equal-block layout. Raises UnitMapError (no rows named) when a stored map is not a one-dimensional tensor
    of an integer dtype (not bool), or does not hold one entry per atom of the rows (``sum(num_atoms)``), e.g. a
    cluster that kept a row map.
    """
    store = batch._store
    if 'num_atoms' in store and store['num_atoms'] is not None:
        counts = store['num_atoms'].reshape(-1).long()
    elif 'ptr' in store and store['ptr'] is not None:
        counts = (store['ptr'][1:] - store['ptr'][:-1]).long()
    else:
        counts = torch.tensor([len(store['z'])], dtype=torch.long, device=store['z'].device)
    B = len(counts)
    zp_stored = store['z_prime'] if 'z_prime' in store else None
    z_prime = torch.ones_like(counts) if zp_stored is None else zp_stored.reshape(-1).long().to(counts.device)
    divides = counts % z_prime.clamp(min=1) == 0
    fits = (z_prime >= 1) & divides
    atoms_per_unit = counts // z_prime.clamp(min=1)
    units_per_row = z_prime.clamp(min=0)
    stored_map = store[UNIT_MAP_KEY] if UNIT_MAP_KEY in store else None
    if stored_map is None:
        def reason(b):
            zp, n = int(z_prime[b]), int(counts[b])
            if zp < 1:
                return f'z_prime {zp} (a row holds at least one unit)'
            return (f'{n} atoms, not divisible by z_prime {zp} (it carries no unit_ind, so its units are z_prime '
                    f'equal contiguous blocks)')

        return UnitLayout(counts, units_per_row, atoms_per_unit, z_prime, None, fits), ~fits, reason

    problem = unit_map_problem(stored_map)
    if problem is not None:     # a float map would be truncated and an [N, 1] one broadcast: refused, not cast
        raise UnitMapError(f'{UNIT_MAP_KEY} cannot be read as a unit map: {problem}', {})
    total = len(stored_map)
    # the rows' atom count, from the host where it is known: a row batch holds sum(num_atoms) atoms in pos (the
    # split checks it); a cluster (aux_ind stored) holds more, and then num_atoms is summed (one device sync)
    if 'pos' in store and store['pos'] is not None and 'aux_ind' not in store:
        expected = len(store['pos'])
    else:
        expected = int(counts.sum())
    if total != expected or (not counts.is_cuda and int(counts.sum()) != total):
        raise UnitMapError(f'unit_ind holds {total} entries, but the rows hold {expected} atoms (num_atoms sum '
                           f'{int(counts.sum())}); a cluster or another per-atom layout carries no row unit map', {})
    derived = equal_block_unit_ind(counts, z_prime, fits, total=total).to(stored_map.device)
    atom_row = torch.repeat_interleave(torch.arange(B, device=counts.device), counts, output_size=total)
    differs = torch.zeros(B, dtype=torch.long, device=counts.device).index_add_(
        0, atom_row, (stored_map.long() != derived).long().to(counts.device))
    bad = ~fits | (counts < z_prime) | (differs > 0)

    def reason(b):
        zp, n = int(z_prime[b]), int(counts[b])
        start = int((torch.cumsum(counts, 0) - counts)[b])
        u = stored_map[start:start + n].long().cpu()
        if zp < 1:
            return f'z_prime {zp} (a row holds at least one unit)'
        if n and bool((u == NO_UNIT).all()):
            return f'{n} atoms, z_prime {zp}: unit_ind {NO_UNIT} on every atom (its units are not known)'
        outside = sorted({int(v) for v in u.tolist() if not 0 <= v < zp})
        if outside:
            return f'{n} atoms, z_prime {zp}: unit_ind values {outside[:8]} outside 0..{zp - 1}'
        sizes = torch.bincount(u, minlength=zp).tolist()
        missing = [k for k, s in enumerate(sizes) if s == 0]
        if missing:
            return f'{n} atoms, z_prime {zp}: unit(s) {missing[:8]} of its unit_ind hold no atom'
        if len(set(sizes)) > 1:
            return (f'{n} atoms in units of {sizes} atoms at z_prime {zp} (units of different sizes are refused at '
                    f'this stage)')
        return (f'{n} atoms, z_prime {zp}: its unit_ind units are not contiguous blocks in unit order (required at '
                f'this stage)')

    return UnitLayout(counts, units_per_row, atoms_per_unit, z_prime, stored_map.long(), fits), bad, reason


def unit_layout(batch, caller: str = 'unit_layout', rows=None) -> UnitLayout:
    """The units of a crystal batch (or single crystal): THE reader of the atom -> unit map.

    A row carrying ``unit_ind`` has the units its map says; a row without one (an older file, GFN's blank batch; a
    bare ``from_arrays`` batch, or a posed one before stage 3) has ``z_prime`` equal contiguous blocks of
    ``num_atoms // z_prime`` atoms,
    today's layout (``derived_unit_ind`` writes it out). Every reader of a crystal's units calls this. At this stage units must be equal contiguous blocks in
    unit order whether or not a map is stored, so with a valid map the layout is the equal-block one bit for bit.

    Parameters
    ----------
    batch : MolCrystalData (batch or single)
        Reads ``num_atoms`` (``ptr`` without it), ``z_prime`` (one unit per row without it) and ``unit_ind``. Not
        modified.
    caller : str
        Named in the refusal message.
    rows : sequence of int, optional
        The rows to check; default every row. Rows not checked still get a layout entry, which is meaningful only
        where their units are readable.

    Returns
    -------
    UnitLayout

    Raises
    ------
    UnitMapError (a ValueError)
        Naming each refused row and why: ``z_prime < 1``; without a map, atoms not divisible by ``z_prime``; with
        one, values outside 0 .. z_prime - 1, a unit without atoms, units of unequal size, or units that are not
        contiguous blocks in unit order. Also when a stored map is not a one-dimensional integer (non-bool) tensor
        or does not hold one entry per atom of the rows. Explicit raises: they hold under ``python -O``.
    """
    layout, bad, reason = _unit_layout_unchecked(batch)
    if rows is not None:
        sel = torch.as_tensor(list(rows), dtype=torch.long, device=bad.device)
        mask = torch.zeros_like(bad)
        mask[sel] = True
        bad = bad & mask
    if bool(bad.any()):     # one device sync on the valid path; the row list only when refusing
        bad_rows = torch.nonzero(bad).flatten().tolist()
        reasons = {b: f'{_unit_row_name(batch, b)}: {reason(b)}' for b in bad_rows}
        named = '; '.join(reasons[b] for b in bad_rows[:_UNIT_ROWS_NAMED])
        more = f'; and {len(bad_rows) - _UNIT_ROWS_NAMED} more' if len(bad_rows) > _UNIT_ROWS_NAMED else ''
        raise UnitMapError(f"{caller}: {len(bad_rows)} row(s) whose units cannot be read (a unit is what the row's "
                           f"unit_ind says, or one of z_prime equal contiguous blocks without one; units must be "
                           f"equal contiguous blocks in unit order at this stage): {named}{more}", reasons)
    return layout


def refuse_unit_map_disagreeing_with_z_prime(unit_ind: Tensor, z_prime, where: str = 'MolCrystalData',
                                             n_atoms: Union[int, None] = None):
    """Construction check of ONE crystal: raise ValueError unless its map is a unit map (``unit_map_problem``: an
    integer, non-bool, one-dimensional tensor, one entry per atom when ``n_atoms`` is given) holding exactly the units
    0 .. z_prime - 1, and ``z_prime`` holds one value (a crystal row has one ``z_prime``; since the stage-2 review,
    2026-10-23, a [2]-entry z_prime on one row is refused rather than left unchecked).

    Cheap (a min, a max and a bincount over the row's atoms); unit sizes and contiguity are checked when a reader
    reads the map (``unit_layout``).
    """
    problem = unit_map_problem(unit_ind, n_atoms)
    if problem is not None:
        raise ValueError(f'{where}: {UNIT_MAP_KEY} is not a unit map: {problem}')
    zp = z_prime.reshape(-1) if torch.is_tensor(z_prime) else torch.as_tensor([z_prime]).reshape(-1)
    if zp.numel() != 1:
        raise ValueError(f'{where}: z_prime holds {zp.numel()} values ({zp.tolist()[:8]}); one crystal has one z_prime')
    zp = int(zp[0])
    u = unit_ind.long()
    if zp < 1:
        raise ValueError(f'{where}: z_prime {zp}, but its unit map (unit_ind) holds {len(torch.unique(u))} unit(s); '
                         f'a crystal holds at least one unit')
    if len(u) == 0:
        raise ValueError(f'{where}: z_prime {zp}, but the crystal has no atoms for its units')
    lo, hi = int(u.min()), int(u.max())
    if lo < 0 or hi >= zp:
        raise ValueError(f'{where}: z_prime {zp}, but its unit map (unit_ind) holds unit {hi if hi >= zp else lo}: '
                         f'the molecules given do not make z_prime units (a one-molecule crystal is one unit; pass '
                         f'one MolData per unit, or unit_ind for a merged molecule)')
    sizes = torch.bincount(u, minlength=zp)
    missing = torch.nonzero(sizes == 0).flatten().tolist()
    if missing:
        raise ValueError(f'{where}: z_prime {zp}, but unit(s) {missing[:8]} of its unit map (unit_ind) hold no atom: '
                         f'the molecules given make {zp - len(missing)} unit(s)')


def generate_sorted_fractional_translations(supercell_size):
    """Integer lattice translations in [-supercell_size, supercell_size]^3, sorted by L1 norm.

    Parameters
    ----------
    supercell_size : int
        Largest absolute translation index on each axis.

    Returns
    -------
    torch.Tensor
        [(2 * supercell_size + 1) ** 3, 3] float32 tensor on the CPU, rows ordered by
        ascending sum of absolute components (torch.argsort, so the order among rows with
        equal sums is not specified). The first row is (0, 0, 0).
    """
    xrange = torch.arange(-supercell_size, supercell_size + 1, dtype=torch.int16)
    xx, yy, zz = torch.meshgrid(xrange, xrange, xrange,
                                indexing='ij'
                                )

    # Flatten the meshgrid and stack the results into a 2D tensor
    fractional_translations = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).float()
    # sort fractional vectors from closest to furthest from central unit cell
    return fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]


#: Angstrom added to every cluster reach (ucell2cluster): covers the float32 rounding of the centroids, reaches and
#: distances the reach test is computed from, so that rounding can add a molecule but never drop one holding a pair
#: within the cutoff. It replaced the 0.1 A that stood beside the heavy-atom radius until 2026-10-23.
REACH_MARGIN = 1e-3


def molecule_reach(pos, batch, num_graphs: int):
    """Unweighted all-atom centroid and all-atom reach of each graph: ``(centroid [num_graphs, 3], reach
    [num_graphs])``, ``reach`` the largest distance from the centroid to an atom of the graph.

    The quantities the cluster reach test is exact for: an atom of a molecule M lies within ``reach_M`` of M's
    centroid, so two molecules hold an atom pair closer than ``cutoff`` only if their centroids are closer than
    ``cutoff + reach_A + reach_B``, hydrogens included. ``reach`` is computed from ``pos`` without its autograd
    graph (it only selects); ``centroid`` keeps it, as ucell2cluster's centre always has.
    """
    centroid = scatter(pos, batch, dim_size=num_graphs, dim=0, reduce='mean')
    dist = (pos.detach() - centroid.detach()[batch]).norm(dim=1)
    reach = scatter(dist, batch, dim_size=num_graphs, dim=0, reduce='max')
    return centroid, reach


def reference_molecule_reach(crystal_batch):
    """``molecule_reach`` of molecule 0 of each graph's built unit cell: ``(centroid [num_graphs, 3], reach
    [num_graphs])``, both without autograd.

    Molecule 0 is the image of the asymmetric unit under the first symmetry operator, the first ``num_atoms`` rows of
    the graph's ``unit_cell_pos`` block (aunit2ucell's image-major layout of a one-unit graph); ucell2cluster makes it,
    at the zero translation, the cluster's ``aux_ind == 0`` molecule. It is not ``pos``: aunit2ucell places the unit
    cell about ``aunit_centroid`` clipped to [0, 1 - 1e-4] (assign_aunit_centroid), while ``pos`` is posed about the
    unclipped value, so a centroid component at 1.0 puts molecule 0 1e-4 of that lattice vector away from ``pos``
    (more than REACH_MARGIN for a vector longer than 10 A), and a nonstandard first operator moves it further. The
    cluster's reach test is centred here so that it is exact (stage-5 review, 2026-10-23).
    """
    n = crystal_batch.num_atoms.reshape(-1).long()
    num_graphs = crystal_batch.num_graphs
    ucell_atoms = n * crystal_batch.sym_mult.reshape(-1).long()
    total = int(n.sum())
    graph = torch.repeat_interleave(torch.arange(num_graphs, device=n.device), n, output_size=total)
    within = torch.arange(total, device=n.device) - (torch.cumsum(n, 0) - n)[graph]
    rows = (torch.cumsum(ucell_atoms, 0) - ucell_atoms)[graph] + within     # image 0 of each graph's block
    centroid, reach = molecule_reach(crystal_batch.unit_cell_pos[rows].detach(), graph, num_graphs)
    return centroid, reach


def ucell2cluster(crystal_batch, cutoff: float = 6,
                  supercell_size: int = 10,
                  zp_buffer: Union[float, torch.Tensor] = 0):  # todo this could be modularized/simplified
    """Build a finite cluster of explicit molecules around each crystal's asymmetric unit.

    1) select the lattice translations whose unit cells hold at least one molecule within
    range of the asymmetric unit, in any cell basis (get_cart_translations, given the unit
    cell's molecule centroids); 2) instantiate those unit cells as explicit atoms
    (instantiate_cluster); 3) flag, without removing, every molecule whose unweighted
    all-atom centroid lies farther than ``rho = cutoff + zp_buffer + 2 * reach + REACH_MARGIN``
    from the all-atom centroid of the cluster's molecule 0 (the first image in the unit cell,
    ``reference_molecule_reach``; not ``pos``, which is posed about the unclipped
    ``aunit_centroid``), ``reach`` that molecule's all-atom reach about its centroid. Every
    molecule of the cluster is a symmetry or lattice image of it, with the same reach, so a
    flagged molecule holds no atom within ``cutoff + zp_buffer`` of any atom of molecule 0:
    the test is exact, hydrogens included. Until 2026-10-23 it read ``2 * radius + 0.1``,
    ``radius`` the heavy-atom radius about the heavy-atom centroid (MolData.radius_calculation),
    which hydrogens reach past: it missed pairs within the cutoff (audit #16; phenacetin P21/c,
    4 of 2305 at 6 A), and at Z'>1 the split carried the row's summed radius (audit #11).
    Step 1 uses the same distance (plus a 1e-4 margin), so every unflagged molecule's cell is
    built, and a cell is built only if it holds a molecule within that distance; before
    2026-10-18 step 1 also built cells whose molecules were all flagged, and missed needed
    cells in oblique bases. A crystal keeps at most get_cart_translations'
    MAX_CELLS_PER_CRYSTAL cells, the nearest ones. A crystal with a non-finite or singular
    cell or a non-finite pose keeps only its own unit cell, with a warning
    (MolCrystalData.analyze returns NaN for it).

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Collated batch of Z'=1 crystals whose unit cell has been built (build_unit_cell).
        Reads ``batch``, ``ptr``, ``num_graphs``, ``num_atoms``, ``sym_mult``,
        ``z_prime``, ``T_fc``, ``unit_cell_pos``, ``unit_cell_batch``, ``z``, ``x`` and
        ``device``; ``cell_lengths`` and ``cell_angles`` only for the error message. Not
        modified. ``radius`` and ``pos`` are not read (since 2026-10-23).
    cutoff : float, default 6
        Interaction cutoff, in the length units of ``pos`` and ``T_fc``.
    supercell_size : int, default 10
        Ignored; passed to get_cart_translations, which no longer caps its range.
    zp_buffer : float or torch.Tensor, default 0
        Added to ``cutoff`` in both the translation selection and the molecule flagging;
        a scalar or a [num_graphs] tensor. A Z'>1 build passes each unit-graph's
        ``MolCrystalBuilding._unit_reach_buffer``, which makes the cluster of each unit
        complete for every asymmetric-unit molecule of its crystal.

    Returns
    -------
    MolCrystalData
        A clone of ``crystal_batch`` in which ``pos`` [num_cluster_atoms, 3], ``z``, ``x``,
        ``batch`` and ``ptr`` describe the cluster, and ``mol_ind`` and ``aux_ind``
        [num_cluster_atoms] are set. ``num_atoms`` is not updated and still counts the
        asymmetric-unit atoms. Atoms are ordered by crystal, then by translation (in the
        order get_cart_translations returns, zero translation first), then in
        unit-cell atom order. ``mol_ind`` numbers the molecules within each cluster from 0,
        molecule 0 being the image under the first symmetry operator at the zero
        translation. ``aux_ind`` is 0 for molecule 0, 1 for the other molecules and 2 for
        flagged molecules.

    Raises
    ------
    ValueError
        If any crystal has ``z_prime > 1`` (from _index_mols).
    AssertionError
        If molecule 0 of any cluster is itself beyond the flagging distance (raised
        explicitly, with per-crystal diagnostics).
    """
    """
    Generate set of unit cells to instantiate
    """
    # the centre is the cluster's own molecule 0 (the first image in the unit cell), not pos: they differ where the
    # unit cell clips aunit_centroid (reference_molecule_reach)
    cc_centroids, reach = reference_molecule_reach(crystal_batch)
    # the molecule-centroid distance within which a molecule can hold a pair within cutoff + zp_buffer (exact)
    rho = cutoff + zp_buffer + 2 * reach + REACH_MARGIN
    # all-atom centroid of every molecule of each unit cell (sym_mult consecutive copies of the asymmetric unit), the
    # quantity _pare_cluster_molwise tests, so that get_cart_translations keeps the cell of every molecule it will not
    # flag (and of molecules within a 1e-4 A band beyond)
    mols_per_crystal = crystal_batch.sym_mult.long()
    ucell_atoms = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()
    ucell_batch = crystal_batch.unit_cell_batch
    atom_in_crystal = (torch.arange(crystal_batch.unit_cell_pos.shape[0], device=ucell_batch.device)
                       - (torch.cumsum(ucell_atoms, 0) - ucell_atoms)[ucell_batch])
    mol_id = ((torch.cumsum(mols_per_crystal, 0) - mols_per_crystal)[ucell_batch]
              + torch.div(atom_in_crystal, crystal_batch.num_atoms[ucell_batch], rounding_mode='floor'))
    mol_centroids = scatter(crystal_batch.unit_cell_pos, mol_id, dim=0, reduce='mean')
    good_translations, ucells_per_cluster = get_cart_translations(cc_centroids,
                                                                  crystal_batch.T_fc,
                                                                  None,
                                                                  None,
                                                                  supercell_size,
                                                                  mol_centroids=mol_centroids,
                                                                  mols_per_crystal=mols_per_crystal,
                                                                  rho=rho)

    """
    Build constants & indices
    """
    # unit cell
    atoms_per_ucell = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()

    # cluster
    atoms_per_cluster = atoms_per_ucell * ucells_per_cluster  # total atoms per crystal
    aunits_per_cluster = ucells_per_cluster * crystal_batch.sym_mult

    # get atomwise translations
    atoms_per_translation = atoms_per_ucell.repeat_interleave(ucells_per_cluster, dim=0)
    atomwise_translation = good_translations.repeat_interleave(atoms_per_translation, dim=0)

    """
    Instantiate explicit clusters
    """
    cluster_batch = instantiate_cluster(atoms_per_ucell,
                                        atoms_per_cluster,
                                        atomwise_translation,
                                        ucells_per_cluster,
                                        crystal_batch,
                                        )

    """
    flag the molecules of kept cells that lie beyond the convolution window (aux_ind = 2); each kept cell holds at least
    one molecule within it, the others are flagged here
    """
    cluster_batch = _pare_cluster_molwise(atoms_per_cluster, cc_centroids,
                                          cluster_batch, crystal_batch,
                                          rho, aunits_per_cluster)

    return cluster_batch
    # zpg_inds = torch.argwhere(cluster_batch.z_prime > 1).flatten().tolist()
    # cluster_batch.visualize(zpg_inds, mode='distance', cutoff=4, highlight_canonical_conformer=True)
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='unit cell')
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='convolve with')


def _pare_cluster_molwise(atoms_per_cluster, cc_centroids, cluster_batch,
                          crystal_batch, rho, aunits_per_cluster):
    """Set ``aux_ind = 2``, in place on ``cluster_batch``, for atoms of molecules whose centroid is farther than
    ``rho`` [num_graphs] (ucell2cluster's reach) from ``cc_centroids``; raise AssertionError if molecule 0 of any
    cluster is."""
    # get all mol centroids
    # indexes which molecule each atom belongs to, globally (not within-crystal)
    aunit_ptr = torch.cat([torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
                           torch.cumsum(aunits_per_cluster, dim=0)])[:-1]
    atom2aunit_ind = cluster_batch.mol_ind + aunit_ptr.repeat_interleave(atoms_per_cluster, dim=0)
    # this is extremely memory heavy with float32, we can make huge savings with half/float16, but it can overflow
    aunit_centroids = scatter(cluster_batch.pos, #.half(),
                              atom2aunit_ind, reduce='mean', dim=0)
    # cc_centroids = scatter(crystal_batch.pos.half(), crystal_batch.batch, reduce='mean', dim=0) # already compute this above
    # get the mol indices within the widest conv cutoff
    molwise_batch = (torch.arange(crystal_batch.num_graphs, device=cc_centroids.device)
                     .repeat_interleave(aunits_per_cluster, dim=0))
    mol_inds = torch.arange(len(molwise_batch), device=cluster_batch.device)
    # get the distances
    mol_to_cc_dist_sq = (aunit_centroids - cc_centroids.repeat_interleave(aunits_per_cluster, dim=0)).square().sum(1)
    cluster_cutoff = torch.as_tensor(rho, device=cc_centroids.device).reshape(-1).expand(
        crystal_batch.num_graphs).square()
    mol_cutoff_sq = cluster_cutoff[molwise_batch]
    # molecules to discard
    bad_mol_inds = mol_inds[mol_to_cc_dist_sq > mol_cutoff_sq]
    # convert back to atomic indices
    # atom for each molecule in the kept list
    # atoms_in_bad_mols_bool = torch.isin(atom_to_flat_mol_ind, bad_mol_inds)
    # dramatically faster alternative to isin
    # bad_sorted = bad_mol_inds.sort().values
    # atom_to_mol index of each atom which is in a 'bad' molecule
    # this works because atom_to_flat_mol_ind is already sorted
    idx = torch.searchsorted(bad_mol_inds, atom2aunit_ind)  # sorting step
    valid = idx < bad_mol_inds.numel()  # safety check for wayward searchsorted outputs > length of tensors
    atoms_in_bad_mols_bool = torch.zeros_like(atom2aunit_ind, dtype=torch.bool)
    atoms_in_bad_mols_bool[valid] = bad_mol_inds[idx[valid]] == atom2aunit_ind[valid]  # atom is actually in a bad mol
    # --- diagnostic: central aunit must never be marked outside the conv window ---
    # By construction, the canonical conformer (mol_ind == 0 within each cluster)
    # is at zero distance from cc_centroids, so it should never appear in bad_mol_inds.
    # If it does, something upstream produced inf/garbage in cluster_batch.pos or
    # aunit_centroids, or the centroid invariant has been violated.

    # Per-cluster index of the central aunit in the global aunit_centroids tensor:
    central_aunit_inds = aunit_ptr  # mol_ind == 0 within each cluster, plus the cluster's offset

    central_dist_sq = mol_to_cc_dist_sq[central_aunit_inds]
    central_marked_bad = mol_to_cc_dist_sq[central_aunit_inds] > mol_cutoff_sq[central_aunit_inds]

    if central_marked_bad.any():
        bad_graphs = central_marked_bad.nonzero(as_tuple=True)[0]
        msg_lines = ["central aunit marked outside conv window — invariant violated"]
        for g in bad_graphs.tolist():
            atoms_mask = cluster_batch.batch == g
            graph_pos = cluster_batch.pos[atoms_mask]
            graph_aux_zero_count = (cluster_batch.aux_ind[atoms_mask] == 0).sum().item()
            msg_lines.append(
                f"  graph={g}: "
                f"central_dist_sq={central_dist_sq[g].item():.4g}, "
                f"cutoff_sq={mol_cutoff_sq[central_aunit_inds[g]].item():.4g}, "
                f"pos_finite={graph_pos.isfinite().all().item()}, "
                f"pos_absmax={graph_pos.abs().max().item():.4g}, "
                f"aunit_centroid_finite={aunit_centroids[central_aunit_inds[g]].isfinite().all().item()}, "
                f"aunit_centroid={aunit_centroids[central_aunit_inds[g]].tolist()}, "
                f"cc_centroid={cc_centroids[g].tolist()}, "
                f"cell_lengths={crystal_batch.cell_lengths[g].tolist()}, "
                f"cell_angles={crystal_batch.cell_angles[g].tolist()}, "
                f"reach_cutoff={cluster_cutoff[g].sqrt().item():.4g}, "
                f"current_aux_zeros_in_graph={graph_aux_zero_count}, "
            )
        raise AssertionError("\n".join(msg_lines))
    # --- end diagnostic ---

    cluster_batch.aux_ind[atoms_in_bad_mols_bool] = 2  # marker for "outside convolutional field"

    return cluster_batch


def instantiate_cluster(atoms_per_ucell,
                        atoms_per_cluster,
                        atomwise_translation,
                        ucells_per_cluster,
                        crystal_batch):
    """Instantiate explicit cluster atoms by tiling each unit cell and adding per-atom translations.

    Parameters
    ----------
    atoms_per_ucell : torch.Tensor
        [num_graphs] atoms per unit cell (``num_atoms * sym_mult``), long.
    atoms_per_cluster : torch.Tensor
        [num_graphs] atoms per cluster (``atoms_per_ucell * ucells_per_cluster``).
    atomwise_translation : torch.Tensor
        [sum(atoms_per_cluster), 3] Cartesian translation added to each cluster atom.
    ucells_per_cluster : torch.Tensor
        [num_graphs] number of unit-cell copies in each cluster.
    crystal_batch : MolCrystalData
        Collated batch with the unit cell built. Reads ``unit_cell_pos``,
        ``unit_cell_batch``, ``ptr``, ``num_graphs``, ``num_atoms``, ``sym_mult``,
        ``z_prime``, ``z``, ``x`` and ``device``. Not modified.

    Returns
    -------
    MolCrystalData
        A clone of ``crystal_batch`` in which ``pos`` is the unit-cell positions repeated
        ``ucells_per_cluster`` times per crystal plus ``atomwise_translation``; ``z``, ``x``,
        ``batch`` and ``ptr`` describe the cluster; ``mol_ind`` is the within-cluster
        molecule index; and ``aux_ind`` is 0 where ``mol_ind == 0`` and 1 elsewhere. The
        row unit map ``unit_ind`` is not carried (it indexes the rows' atoms).

    Raises
    ------
    ValueError
        If any crystal has ``z_prime > 1`` (from _index_mols).
    """
    """
    Generate necessary indices to instantiate clusters in parallel
    """
    unit_cell_batch = crystal_batch.unit_cell_batch
    unit_cell_ptr = torch.cat(
        [torch.tensor([0], device=crystal_batch.device),
         torch.cumsum(atoms_per_ucell, dim=0)]).long()

    atom_in_unit_cell_batch, molwise_nodes_within_unit_cell, supercell_batch = _index_cluster_nodes(atoms_per_cluster,
                                                                                                    atoms_per_ucell,
                                                                                                    crystal_batch,
                                                                                                    unit_cell_ptr)

    # add ptr offset per graph
    node_within_mol_in_ucell = molwise_nodes_within_unit_cell + crystal_batch.ptr[:-1][unit_cell_batch]
    # this indexes from the full cluster to the atom in each aunit
    cluster_node2aunit_node = node_within_mol_in_ucell[atom_in_unit_cell_batch]

    mol_ind = _index_mols(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell, crystal_batch,
                          ucells_per_cluster)

    """
    do the instantiation
    """
    cluster_batch = _instantiate_cluster(atom_in_unit_cell_batch,
                                         atoms_per_cluster, atomwise_translation,
                                         crystal_batch, cluster_node2aunit_node,
                                         supercell_batch,
                                         mol_ind)

    return cluster_batch


def _index_cluster_nodes(atoms_per_cluster, atoms_per_ucell, crystal_batch, unit_cell_ptr):
    tot_num_atoms = atoms_per_cluster.sum()
    cluster_atom_ind = torch.arange(tot_num_atoms, device=crystal_batch.device)
    supercell_batch = torch.repeat_interleave(torch.arange(crystal_batch.num_graphs,
                                                           device=crystal_batch.device),
                                              atoms_per_cluster)
    supercell_ptr = torch.cumsum(torch.cat([torch.tensor([0], device=crystal_batch.device),
                                            atoms_per_cluster[:-1]]), dim=0)
    atom_in_crystal_index = cluster_atom_ind - supercell_ptr[supercell_batch]
    atom_in_unit_cell_batch = (atom_in_crystal_index % atoms_per_ucell[supercell_batch]) + unit_cell_ptr[
        supercell_batch]
    # within-block index (0..num_atoms[i]-1), repeated for sym_mult[i].
    # BATCHED, and the reason matters: the previous form called .tolist() on two CUDA
    # tensors, which extracts element by element and so costs 2 x num_graphs DEVICE SYNCS
    # per call, then built num_graphs tiny arange/repeat kernels and cat'd them. Profiled
    # 2026-09-22 on a conditional run it was the single largest source of scalar readback --
    # 244,362 of the ~357,000 device syncs in the sample, at ~1,500 graphs per call. Block i
    # has length num_atoms[i] * sym_mult[i] == atoms_per_ucell[i], and within it position p
    # is simply p % num_atoms[i], so the whole thing is one arange, one cumsum, one
    # repeat_interleave and one modulo. The single .sum() sync below is one the caller has
    # already paid at `torch.arange(tot_num_atoms)` above. Verified bit-identical to the loop
    # on uniform, ragged, sym_mult>1, n=1 and single-graph batches; 93.2 ms -> 0.3 ms at 3000
    # graphs.
    _total_ucell_atoms = int(atoms_per_ucell.sum())
    _ucell_graph_idx = torch.repeat_interleave(
        torch.arange(crystal_batch.num_graphs, device=crystal_batch.device),
        atoms_per_ucell, output_size=_total_ucell_atoms)
    _ucell_block_start = torch.cumsum(atoms_per_ucell, dim=0) - atoms_per_ucell
    molwise_nodes_within_unit_cell = (
        (torch.arange(_total_ucell_atoms, device=crystal_batch.device)
         - _ucell_block_start[_ucell_graph_idx])
        % crystal_batch.num_atoms[_ucell_graph_idx])
    ''' test and compare this method on a large batch
        num_atoms = crystal_batch.num_atoms
        sym_mult = crystal_batch.sym_mult
        
        # total length = sum(num_atoms * sym_mult)
        lengths = num_atoms * sym_mult
        total = lengths.sum()
        
        # assign each position to its "unit cell"
        ucell_idx = torch.arange(total, device=crystal_batch.device) \
            .repeat_interleave(1)  # just a flat arange
        
        # expand each unit cell according to lengths
        ucell_idx = torch.repeat_interleave(torch.arange(len(lengths), device=total.device), lengths)
        
        # within each block, cycle through atom indices
        within = torch.arange(total, device=total.device) - torch.repeat_interleave(
            torch.cumsum(torch.cat([torch.zeros(1, device=total.device, dtype=num_atoms.dtype),
                                    lengths[:-1]]), 0),
            lengths
        )
        
        # reduce modulo num_atoms[i]
        molwise_nodes_within_unit_cell = within % torch.repeat_interleave(num_atoms, lengths)
        '''
    return atom_in_unit_cell_batch, molwise_nodes_within_unit_cell, supercell_batch


def _index_mols(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell, crystal_batch, ucells_per_cluster):
    """Within-cluster molecule index of each cluster atom, for Z'=1 batches.

    A graph with ``z_prime > 1`` raises ValueError naming its rows (an explicit raise, so it holds under
    ``python -O``; until 2026-10-22 this was ``assert False``, and ``-O`` fell through to _zp_gr1_mol_indexing, which
    gave every unit of an asymmetric unit the same ``mol_ind``). Split such batches first: mol2cluster does
    (split_to_zp1_batch, build, join_zp1_cluster_batch)."""
    if crystal_batch.z_prime.amax() > 1:
        rows = torch.nonzero(crystal_batch.z_prime.reshape(-1) > 1).flatten().tolist()
        raise ValueError(f"cluster construction needs Z'=1 graphs, but {len(rows)} graph(s) have z_prime > 1: "
                         + ', '.join(f'row {i} (z_prime {int(crystal_batch.z_prime.reshape(-1)[i])})' for i in rows[:8])
                         + (f' and {len(rows) - 8} more' if len(rows) > 8 else '')
                         + "; split the batch into one graph per unit first (MolCrystalData.mol2cluster does)")
    else:  # much simpler to do it this way
        mols_per_ucell = crystal_batch.sym_mult
        mols_per_cluster = ucells_per_cluster * mols_per_ucell
        tot_num_mols = torch.sum(mols_per_cluster)

        global_mol_index = torch.arange(tot_num_mols, device=crystal_batch.device)
        mol_offset = torch.cat([torch.zeros(1, device=crystal_batch.device, dtype=torch.long),
                                torch.cumsum(mols_per_cluster, dim=0)[:-1]])
        local_mol_index = global_mol_index - mol_offset.repeat_interleave(mols_per_cluster)

        atoms_per_mol_in_cluster = crystal_batch.num_atoms.repeat_interleave(mols_per_cluster)
        atomwise_mol_ind = local_mol_index.repeat_interleave(
            atoms_per_mol_in_cluster
        )

    return atomwise_mol_ind


def _zp_gr1_mol_indexing(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell, crystal_batch,
                         ucells_per_cluster):
    """
    This appears technically functional, but is an avoidable monstrosity
    :param atom_in_unit_cell_batch:
    :param atoms_per_cluster:
    :param atoms_per_ucell:
    :param crystal_batch:
    :param ucells_per_cluster:
    :return:
    """
    if not getattr(_zp_gr1_mol_indexing, '_warned', False):
        warnings.warn(
            "WARNING: You have done cluster building directly on a crystal batch with some Z prime greater than 1."
            "This calls a complicated and avoidable indexing function. We recommend instead splitting to separate Zp=1 "
            "graphs then recombining.", UserWarning, stacklevel=2)
        _zp_gr1_mol_indexing._warned = True

    mols_per_ucell = atoms_per_ucell // crystal_batch.num_atoms * crystal_batch.z_prime
    ucell_mol_offset = torch.cumsum(torch.cat([torch.zeros(1, device=mols_per_ucell.device, dtype=mols_per_ucell.dtype),
                                               mols_per_ucell[:-1]]), 0)
    within_ucell_mol_ind = (torch.arange(mols_per_ucell.sum(), device=mols_per_ucell.device, dtype=torch.long)
                            - ucell_mol_offset.repeat_interleave(mols_per_ucell))
    # unreachable since 2026-10-22 (_index_mols refuses Z'>1 graphs); units read from the unit map all the same
    atoms_per_mol = unit_layout(crystal_batch, '_zp_gr1_mol_indexing').atoms_per_unit
    molwise_ucell_sym_mult = crystal_batch.sym_mult.repeat_interleave(mols_per_ucell)
    # indexes the asymmetric units within each unit cell
    within_ucell_aunit_ind = within_ucell_mol_ind % molwise_ucell_sym_mult
    # ditto, atomwise
    within_ucell_aunit_atom_ind = within_ucell_aunit_ind.repeat_interleave(
        atoms_per_mol.repeat_interleave(mols_per_ucell))
    # we then need to expand this to the cluster batch
    cluster_aunit_inner_ind = within_ucell_aunit_atom_ind[atom_in_unit_cell_batch]
    # and add offsets to account for the new tiles in the cluster
    tot_num_ucells = ucells_per_cluster.sum()
    aunits_per_cluster = crystal_batch.sym_mult * ucells_per_cluster
    ucell_num_aunits = crystal_batch.sym_mult.repeat_interleave(ucells_per_cluster)
    aunits_per_cluster_ucell = crystal_batch.sym_mult.repeat_interleave(ucells_per_cluster)
    aunit_cum = torch.cumsum(
        torch.cat([torch.zeros(1, device=ucells_per_cluster.device, dtype=ucells_per_cluster.dtype),
                   ucell_num_aunits[:-1]]), 0)
    cluster_aunit_offset = torch.cumsum(
        torch.cat([torch.zeros(1, device=ucells_per_cluster.device, dtype=ucells_per_cluster.dtype),
                   aunits_per_cluster[:-1]]), 0)
    cluster_nodewise_offset = cluster_aunit_offset.repeat_interleave(atoms_per_cluster)
    aunit2ucell = torch.arange(tot_num_ucells, device=crystal_batch.device).repeat_interleave(
        aunits_per_cluster_ucell
    )
    aunit_offset = aunit_cum[aunit2ucell]
    atoms_per_aunit = crystal_batch.num_atoms.repeat_interleave(aunits_per_cluster)
    nodewise_offset = aunit_offset.repeat_interleave(atoms_per_aunit)
    mol_ind = cluster_aunit_inner_ind + nodewise_offset - cluster_nodewise_offset
    return mol_ind


def _instantiate_cluster(atom_in_unit_cell_batch,
                         atoms_per_cluster,
                         atomwise_translation,
                         crystal_batch,
                         cluster_node2aunit_node,
                         supercell_batch,
                         mol_ind):
    cluster_batch = crystal_batch.clone()
    cluster_batch._store.pop(UNIT_MAP_KEY, None)    # the unit map is per row atom: a cluster carries none
    # init cluster with right number of duplicated unit cells
    cluster_batch.pos = crystal_batch.unit_cell_pos[atom_in_unit_cell_batch]
    # build cluster by translating all atoms manually
    cluster_batch.pos += atomwise_translation
    # pattern out atom properties
    cluster_batch.z = crystal_batch.z[cluster_node2aunit_node]
    cluster_batch.x = crystal_batch.x[cluster_node2aunit_node]
    # rebuild batch indices
    cluster_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=crystal_batch.device),
                                   torch.cumsum(atoms_per_cluster, dim=0)]).long()
    cluster_batch.batch = supercell_batch

    # asymmetric unit indices
    cluster_batch.mol_ind = mol_ind
    cluster_batch.aux_ind = torch.ones_like(mol_ind)
    cluster_batch.aux_ind[mol_ind == 0] = 0

    return cluster_batch


def _molwise_indexing(cluster_batch,
                      crystal_batch,
                      mols_per_subunit,
                      tot_num_subunits):
    """
    Indexing for downstream analysis
    aux_ind labels 0: asymmetric unit, 1: convolutional window, 2: outside convolutional window
    mol_ind indexes the molecule (within cluster) for each atom
    """
    # which subunit each entry belongs to
    mol2subunit = torch.arange(tot_num_subunits, device=mols_per_subunit.device).repeat_interleave(mols_per_subunit)

    # index *within* the cluster
    offsets = torch.cumsum(torch.cat([torch.zeros(1, device=mols_per_subunit.device, dtype=mols_per_subunit.dtype),
                                      mols_per_subunit[:-1]]), 0)
    mol_within_subunit = torch.arange(mols_per_subunit.sum(), device=mols_per_subunit.device) - offsets[mol2subunit]

    atoms_per_mol_by_crystal = unit_layout(crystal_batch, '_molwise_indexing').atoms_per_unit  # no caller (2026-10-22)
    atoms_per_mol_by_subunit = atoms_per_mol_by_crystal.repeat_interleave(crystal_batch.z_prime)
    atoms_per_mol = atoms_per_mol_by_subunit.repeat_interleave(mols_per_subunit)

    cluster_batch.mol_ind = mol_within_subunit.repeat_interleave(
        atoms_per_mol
    )

    cluster_batch.aux_ind = torch.zeros_like(cluster_batch.z)
    cluster_batch.aux_ind[cluster_batch.mol_ind != 0] = 1  # MK this may have to be adjusted for Z'>1

    return cluster_batch


#: get_cart_translations: candidate translations examined per pass (roughly 250 bytes each at peak on CPU); crystals
#: are taken in contiguous chunks so that one pass holds at most this many (one crystal is never split)
TRANSLATION_CANDIDATES_PER_PASS = 2 ** 22
#: get_cart_translations: upper bound on the sweeps of pairwise size reduction applied to each lattice basis before
#: the search (it stops early once a sweep changes nothing); the result is exact after any number of sweeps, only the
#: size of the search boxes depends on it
BASIS_REDUCTION_SWEEPS = 24
#: get_cart_translations: most candidates one crystal's search may examine. A crystal whose boxes would hold more (a
#: nearly flat cell: a box holds about (2 rho / d + 1)^3 candidates, d the interplanar spacings of the reduced basis)
#: is searched with a smaller radius chosen to fit, which still finds far more than MAX_CELLS_PER_CRYSTAL cells.
MAX_SEARCH_CANDIDATES_PER_CRYSTAL = 2 ** 21
#: get_cart_translations: most translations kept for one crystal (owner decision 2026-10-18: cap, nearest first, memory
#: bounded like the old code, whose cube held at most 21^3 = 9,261). Beyond it the cells whose nearest molecule is
#: closest to the asymmetric unit are kept. Only physically impossible cells reach it (a cell much smaller than a
#: molecule, packing coefficient far above 1, eLJ far above 1e3 raw); a trained policy's crystals keep a few hundred.
MAX_CELLS_PER_CRYSTAL = 2 ** 13


def _reduce_lattice_bases(basis):
    """[n, 3, 3] float64 bases (columns are lattice vectors) -> (reduced [n, 3, 3], U [n, 3, 3]) with reduced = basis @ U.

    Sweeps over the ordered pairs (i, j) of b_i <- b_i - round(b_i.b_j / b_j.b_j) b_j, which never lengthen b_i, until a
    sweep changes nothing or BASIS_REDUCTION_SWEEPS sweeps have run; U is unimodular, so the reduced basis spans the same
    lattice. Meant for CPU tensors (the early-exit test reads a value).
    """
    red = basis.clone()
    U = torch.eye(3, dtype=basis.dtype, device=basis.device).expand(basis.shape[0], 3, 3).clone()
    for _ in range(BASIS_REDUCTION_SWEEPS):
        changed = False
        for i, j in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
            bj = red[:, :, j]
            mu = torch.round((red[:, :, i] * bj).sum(1) / bj.square().sum(1))
            if bool((mu != 0).any()):
                changed = True
                red[:, :, i] -= mu[:, None] * bj
                U[:, :, i] -= mu[:, None] * U[:, :, j]
        if not changed:
            break
    return red, U


def _box_count(m_point, reach_per_rho, rho):
    """Candidates in the integer boxes of spheres of radius rho [units] (float64 counts, no overflow)."""
    reach = rho[:, None] * reach_per_rho
    return (torch.floor(m_point + reach) - torch.ceil(m_point - reach) + 1).clamp(min=0).prod(1)


def get_cart_translations(cc_centroids,
                          T_fc,
                          mol_radii,
                          cutoff,
                          supercell_size: int = 9,
                          mol_centroids=None,
                          mols_per_crystal=None,
                          rho=None,
                          ):
    """Select the lattice translations whose unit cells hold a molecule within range of each asymmetric unit.

    With ``mol_centroids``: a lattice translation t (an integer combination of the columns of
    ``T_fc``) is kept for a crystal when some molecule of its unit cell, moved by t, has its
    centroid within ``rho`` of ``cc_centroids``: the ``rho`` given (as ucell2cluster passes
    it, the distance its molecule flagging ``_pare_cluster_molwise`` uses), or else
    ``2 * mol_radii + cutoff + 0.1``. Without them, the older cell test is used: t is kept when
    ``|cc_centroids - t| < 2 * mol_radii + cutoff + 0.1 + D``, D the largest norm among the
    seven non-origin corners of the cell parallelepiped. In both, the zero translation is
    always kept, every translation the test admits is found however oblique the basis the
    cell is written in, and a crystal's result does not depend on the other crystals of the
    batch. A crystal keeps at most MAX_CELLS_PER_CRYSTAL translations: the ones whose nearest
    molecule (or, in the cell test, whose own vector) is closest to the centre; a warning
    says how many crystals hit the cap.

    Parameters
    ----------
    cc_centroids : torch.Tensor
        [num_graphs, 3] Cartesian centre of each crystal's asymmetric unit.
    T_fc : torch.Tensor
        [num_graphs, 3, 3] fractional-to-Cartesian transforms; column i is lattice vector i.
    mol_radii : torch.Tensor
        [num_graphs] molecular radius of each crystal; not read when ``rho`` is given.
    cutoff : float or torch.Tensor
        Interaction cutoff; a scalar or a [num_graphs] tensor, in the length units of
        ``T_fc``; not read when ``rho`` is given.
    supercell_size : int, default 9
        Ignored. It used to cap the integer range searched, which silently dropped
        translations the test keeps (cells written in oblique bases got wrong,
        batch-dependent energies); it is accepted so that existing calls keep working.
    mol_centroids : torch.Tensor, optional
        [sum(mols_per_crystal), 3] Cartesian centroids of the molecules of each crystal's
        unit cell, crystal by crystal, as placed at the zero translation.
    mols_per_crystal : torch.Tensor, optional
        [num_graphs] number of molecules per unit cell (positive integers); required with
        ``mol_centroids``.
    rho : float or torch.Tensor, optional
        The molecule-centroid distance of the test, a scalar or [num_graphs]; only with
        ``mol_centroids``. ucell2cluster passes its exact all-atom reach.

    Returns
    -------
    good_translations : torch.Tensor
        [sum(cells_per_cluster), 3] kept Cartesian translation vectors (dtype of ``T_fc``,
        built from ``T_fc`` so they carry its gradient), grouped by crystal in batch order.
        Within a crystal they are ordered by ascending |m|_1, m the integer indices in a
        reduced basis of the lattice, ties in lexicographic order of m; the zero translation
        comes first.
    cells_per_cluster : torch.Tensor
        [num_graphs] number of translations kept for each crystal.

    Raises
    ------
    ValueError
        If ``mols_per_crystal`` does not match ``mol_centroids``, or ``rho`` is given without
        ``mol_centroids``.

    Notes
    -----
    A crystal whose cell, centre, radius, cutoff or molecule centroids are not finite, or
    whose cell is singular, has no defined cluster: it keeps only its zero translation
    (with NaN entries when its cell is not finite), with a warning, and the rest of the batch is built as
    usual. ``MolCrystalData.analyze`` returns NaN for such crystals.

    Each lattice basis is first size-reduced (``_reduce_lattice_bases``): the lattice, hence
    the set of translations, is unchanged, but the search boxes become small. For a sphere
    of radius rho about a point p, every lattice point t = B m inside it has
    |m_i - (B^-1 p)_i| < rho |B^-1[i]| = rho / d_i, d_i the spacing of the lattice planes
    normal to reciprocal vector i of the reduced basis B, so the box searched is
    m_i in [ceil((B^-1 p)_i - rho / d_i), floor((B^-1 p)_i + rho / d_i)]. With
    ``mol_centroids`` there is one sphere per molecule k, about p = cc - c_k, and the
    crystal's translations are the union; the cost is about Z (2 rho / d + 1)^3 candidates,
    whatever the basis. The test runs in float64 with radius rho + 1e-4, so float32
    differences against the flagging step add cells rather than drop them (for bases whose
    farthest corner is below about 1e3 A; beyond, float32 rounding of the flagging distance
    can exceed the margin). A kept cell may therefore hold only molecules flagged within
    that 1e-4 band. A crystal whose boxes would exceed MAX_SEARCH_CANDIDATES_PER_CRYSTAL is
    searched with the largest radius below rho that fits; its nearest cells are still found.

    Before 2026-10-18 this function searched one cube of indices for the whole batch, in the
    basis given, bounded through R / |lattice vector i| and capped at ``supercell_size``, with
    the cell test above. Since d_i <= |lattice vector i|, that range is too small for an
    oblique cell: a MIPCAS crystal written in a basis with a 17 deg angle needs indices up to
    84 on one axis of that basis and got 5, and its eLJ came out 63 raw units high, by an
    amount that depended on the other crystals in the batch. The cell test's radius grows
    with D, so in an oblique basis it admits cells holding nothing in range; the molecule test
    does not.

    Host synchronisation: one device-to-host copy of the inputs, one host-to-device copy of
    the plan, and per pass of at most TRANSLATION_CANDIDATES_PER_PASS candidates a nonzero
    and a unique (each returns a data-dependent size).
    """
    num_graphs = T_fc.shape[0]
    device, dtype = T_fc.device, T_fc.dtype
    if num_graphs == 0:
        return T_fc.new_zeros((0, 3)), torch.zeros(0, dtype=torch.long, device=device)
    if rho is not None and mol_centroids is None:
        raise ValueError('get_cart_translations: rho is the molecule test\'s distance; pass mol_centroids with it')
    if mol_centroids is not None:
        if mols_per_crystal is None:
            raise ValueError('get_cart_translations: mols_per_crystal is required with mol_centroids')
        per = torch.as_tensor(mols_per_crystal, device=device).reshape(-1)
        if per.shape[0] != num_graphs:
            raise ValueError(f'get_cart_translations: mols_per_crystal has {per.shape[0]} entries for {num_graphs} '
                             f'crystals')
        if rho is None:
            rho = mol_radii * 2 + cutoff + 0.1                                 # the molecule test's older distance
        rho = torch.as_tensor(rho, dtype=dtype, device=device).reshape(-1).expand(num_graphs)
        units = mol_centroids.reshape(-1, 3).to(dtype)
        per = per.to(dtype)
    else:
        cell_vectors = T_fc.permute(0, 2, 1)
        # cell diagonals: the farthest point of a cell from its origin corner is one of its other 7 corners
        # (|a+b+c| alone collapses for oblique cells, where a and c nearly cancel)
        corners = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                               dtype=dtype, device=device)
        cell_diag = torch.einsum('ki,nij->nkj', corners, cell_vectors).norm(dim=-1).amax(1)
        rho = mol_radii * 2 + cutoff + 0.1 + cell_diag                         # cell test (no molecule centroids)
        rho = torch.as_tensor(rho, dtype=dtype, device=device).reshape(-1).expand(num_graphs)
        units = torch.zeros(num_graphs, 3, dtype=dtype, device=device)         # one sphere about cc itself
        per = torch.ones(num_graphs, dtype=dtype, device=device)

    # the one device-to-host copy: cells, centres, radii, unit counts and unit centroids, planned on the CPU in float64
    head = torch.cat([T_fc.reshape(num_graphs, 9), cc_centroids.reshape(num_graphs, 3).to(dtype), rho[:, None],
                      per[:, None]], dim=1).reshape(-1)
    host = torch.cat([head, units.reshape(-1)]).detach().double().cpu()
    crystal = host[:num_graphs * 14].reshape(num_graphs, 14)
    unit_xyz = host[num_graphs * 14:].reshape(-1, 3)
    basis, centre, radius, count = crystal[:, :9].reshape(num_graphs, 3, 3), crystal[:, 9:12], crystal[:, 12], crystal[:, 13]
    n_units = count.round().long()
    if bool((count != n_units).any()) or bool((n_units < 1).any()) or int(n_units.sum()) != unit_xyz.shape[0]:
        raise ValueError(f'get_cart_translations: mols_per_crystal must be positive integers summing to the '
                         f'{unit_xyz.shape[0]} molecule centroids given (sum {float(count.sum())})')
    unit_graph = torch.repeat_interleave(torch.arange(num_graphs), n_units)
    # crystals with no defined cluster (a non-finite cell, centre, radius, cutoff or molecule centroid, or a singular
    # cell) keep only their zero translation, so one bad row does not stop a batch; MolCrystalData.analyze returns NaN
    # for them
    bad_units = torch.zeros(num_graphs, dtype=torch.long).index_add_(0, unit_graph, (~torch.isfinite(unit_xyz).all(1)).long())
    undefined = ~torch.isfinite(crystal).all(1) | (torch.linalg.det(basis).abs() <= 0) | (bad_units > 0)
    if bool(undefined.any()):
        warnings.warn(f'get_cart_translations: crystals {torch.nonzero(undefined)[:, 0].tolist()[:20]} have a '
                      f'non-finite or singular cell, or a non-finite centre, radius, cutoff or molecule centroid; each '
                      f'keeps only its zero translation')
        basis = torch.where(undefined[:, None, None], torch.eye(3, dtype=torch.float64), basis)
        centre = torch.where(undefined[:, None], 0.0, centre)
        radius = torch.where(undefined, 0.0, radius)
        unit_xyz = torch.where(undefined[unit_graph, None], 0.0, unit_xyz)

    reduced, U = _reduce_lattice_bases(basis)
    reduced_cf = torch.linalg.inv(reduced)
    spacing_inv = reduced_cf.square().sum(-1).sqrt()                           # 1 / d_i, [num_graphs, 3]
    point = centre[unit_graph] - unit_xyz                                     # sphere centres, [units, 3]
    m_point = torch.einsum('uij,uj->ui', reduced_cf[unit_graph], point)
    per_rho = spacing_inv[unit_graph] * (1 + 1e-9) + 1e-12                    # reach per unit radius, rounding outward
    search_rho = radius + 1e-4                                                # [num_graphs]
    per_crystal_f = torch.zeros(num_graphs, dtype=torch.float64).index_add_(
        0, unit_graph, _box_count(m_point, per_rho, search_rho[unit_graph]))
    big = per_crystal_f > MAX_SEARCH_CANDIDATES_PER_CRYSTAL
    if bool(big.any()):                                                       # nearly flat cells: shrink the search
        lo_r, hi_r = torch.zeros(num_graphs, dtype=torch.float64), search_rho.clone()
        for _ in range(40):                                                   # bisection, bounded
            mid = 0.5 * (lo_r + hi_r)
            n_mid = torch.zeros(num_graphs, dtype=torch.float64).index_add_(
                0, unit_graph, _box_count(m_point, per_rho, mid[unit_graph]))
            fits = n_mid <= MAX_SEARCH_CANDIDATES_PER_CRYSTAL
            lo_r = torch.where(fits, mid, lo_r)
            hi_r = torch.where(fits, hi_r, mid)
        search_rho = torch.where(big, lo_r, search_rho)
    reach = search_rho[unit_graph, None] * per_rho
    lo = torch.ceil(m_point - reach)
    sizes = (torch.floor(m_point + reach) - lo + 1).clamp(min=0)
    per_unit = sizes.prod(1)
    per_crystal = torch.zeros(num_graphs, dtype=torch.float64).index_add_(0, unit_graph, per_unit).long().tolist()
    l1_bound = (torch.maximum(lo.abs(), (lo + sizes - 1).abs()).sum(1) + 1).long()   # bounds |m|_1 per unit
    unit_ptr = (torch.cumsum(n_units, 0) - n_units).tolist()
    n_units_l = n_units.tolist()
    lo_l, hi_l = lo.long(), (lo + sizes - 1).long()

    # the one host-to-device copy of the plan: per unit (lo, sizes, sphere centre, radius^2, crystal, candidates) and
    # per crystal (reduced basis, U)
    n_u = unit_xyz.shape[0]
    plan = torch.cat([torch.cat([lo, sizes, point, (search_rho ** 2)[unit_graph, None], unit_graph[:, None].double(),
                                 per_unit[:, None]], dim=1).reshape(-1),
                      torch.cat([reduced.reshape(num_graphs, 9), U.reshape(num_graphs, 9)], dim=1).reshape(-1)])
    plan = plan.to(device)
    plan_u = plan[:n_u * 12].reshape(n_u, 12)
    plan_c = plan[n_u * 12:].reshape(num_graphs, 18)
    T_fc64 = T_fc.to(torch.float64)                                            # keeps T_fc's autograd graph
    translations, counts, n_capped = [], [], 0
    start = 0
    while start < num_graphs:
        stop, total = start, 0
        while stop < num_graphs and (stop == start or total + per_crystal[stop] <= TRANSLATION_CANDIDATES_PER_PASS):
            total += per_crystal[stop]
            stop += 1
        u0, u1 = unit_ptr[start], unit_ptr[stop - 1] + n_units_l[stop - 1]
        pu = plan_u[u0:u1]
        k = pu[:, 11].long()
        unit = torch.repeat_interleave(torch.arange(u1 - u0, device=device), k, output_size=total)
        flat = torch.arange(total, device=device) - (torch.cumsum(k, 0) - k)[unit]
        s1, s2 = pu[unit, 4].long(), pu[unit, 5].long()
        m = torch.stack((torch.div(flat, s1 * s2, rounding_mode='floor'),
                         torch.div(flat, s2, rounding_mode='floor') % s1,
                         flat % s2), dim=1) + pu[unit, :3].long()             # [total, 3] reduced-basis indices
        del flat, s1, s2
        graph = pu[unit, 10].long()                                           # crystal index in the batch
        cart = torch.einsum('tij,tj->ti', plan_c[graph, :9].reshape(-1, 3, 3), m.to(torch.float64))
        d2 = (pu[unit, 6:9] - cart).square().sum(-1)
        keep = d2 <= pu[unit, 9]
        del cart
        sel = torch.nonzero(keep)[:, 0]                                       # sync: data-dependent size
        zero = torch.arange(start, stop, device=device)
        graph = torch.cat([graph[sel], zero])
        m = torch.cat([m[sel], torch.zeros(stop - start, 3, dtype=torch.long, device=device)])
        d2 = torch.cat([d2[sel], torch.zeros(stop - start, dtype=torch.float64, device=device)])
        del keep, sel
        # union over each crystal's molecules through a 1-D key that sorts by crystal, then m lexicographically
        m_lo = lo_l[u0:u1].min(0).values.clamp(max=0)                        # the appended zeros must fit too
        m_hi = hi_l[u0:u1].max(0).values.clamp(min=0)
        base = (m_hi - m_lo + 1).tolist()
        m_lo = m_lo.tolist()
        if (stop - start) * base[0] * base[1] * base[2] >= 2 ** 62:
            raise ValueError('get_cart_translations: index range too large for the union key')
        mo = m - torch.tensor(m_lo, device=device)
        key = ((graph - start) * base[0] + mo[:, 0]) * base[1] * base[2] + mo[:, 1] * base[2] + mo[:, 2]
        uniq, inverse = torch.unique(key, return_inverse=True)                # sync: data-dependent size
        dmin = torch.full(uniq.shape, float('inf'), dtype=torch.float64, device=device).scatter_reduce(
            0, inverse, d2, reduce='amin')
        g = torch.div(uniq, base[0] * base[1] * base[2], rounding_mode='floor')
        rem = uniq - g * base[0] * base[1] * base[2]
        mm = torch.stack([torch.div(rem, base[1] * base[2], rounding_mode='floor'),
                          torch.div(rem, base[2], rounding_mode='floor') % base[1],
                          rem % base[2]], dim=1) + torch.tensor(m_lo, device=device)
        # the cap: within each crystal keep the MAX_CELLS_PER_CRYSTAL translations with the nearest molecule
        n_per = torch.zeros(stop - start, dtype=torch.long, device=device).index_add_(0, g, torch.ones_like(g))
        if int(n_per.max()) > MAX_CELLS_PER_CRYSTAL:
            n_capped += int((n_per > MAX_CELLS_PER_CRYSTAL).sum())
            by_dist = torch.argsort(g.double() * (float(dmin.max()) + 1.0) + dmin, stable=True)
            first = torch.cumsum(n_per, 0) - n_per
            rank = torch.empty_like(by_dist)
            rank[by_dist] = torch.arange(by_dist.shape[0], device=device) - first[g[by_dist]]
            keep_c = rank < MAX_CELLS_PER_CRYSTAL
            g, mm = g[keep_c], mm[keep_c]
        # order within each crystal: ascending |m|_1, ties lexicographic in m (unique's order); zero first
        l1 = mm.abs().sum(1)
        span = int(l1_bound[u0:u1].max())
        order = torch.argsort(g * span + l1, stable=True)
        g, mm = g[order] + start, mm[order]
        n = torch.einsum('tij,tj->ti', plan_c[g, 9:].reshape(-1, 3, 3), mm.to(torch.float64))
        translations.append(torch.einsum('tij,tj->ti', T_fc64[g], n).to(dtype))
        counts.append(torch.zeros(stop - start, dtype=torch.long, device=device)
                      .index_add_(0, g - start, torch.ones_like(g)))
        start = stop
    if n_capped:
        warnings.warn(f'get_cart_translations: {n_capped} of {num_graphs} crystals kept only their '
                      f'{MAX_CELLS_PER_CRYSTAL} nearest cells (MAX_CELLS_PER_CRYSTAL); such cells are far smaller '
                      f'than their molecules, and their energies come from those nearest cells')
    return torch.cat(translations), torch.cat(counts)


def find_coord_in_box_torch(coords, box, epsilon=0):
    """Row indices of ``coords`` with ``0 <= coords[:, k] <= box[k] + epsilon`` for k = 0, 1, 2.

    Parameters
    ----------
    coords : torch.Tensor
        [n, 3] coordinates.
    box : sequence or torch.Tensor
        Three upper bounds.
    epsilon : float, default 0
        Added to the upper bounds only; the lower bound is 0 on every axis.

    Returns
    -------
    torch.Tensor
        [m] long tensor of the indices of the rows inside the box.
    """
    # which of the given coords is inside the specified box, with option for a little leeway
    return torch.where((coords[:, 0] <= (box[0] + epsilon)) *
                       (coords[:, 1] <= (box[1] + epsilon)) *
                       (coords[:, 2] <= (box[2] + epsilon)) *
                       (coords[:, 0] >= 0) *
                       (coords[:, 1] >= 0) *
                       (coords[:, 2] >= 0)
                       )[0]


def parameterize_crystal_batch(crystal_batch,
                               asym_unit_dict,
                               enforce_right_handedness: bool = False,
                               return_aunit: bool = False,
                               ):
    """Recover centroid, orientation and handedness parameters of each crystal from its unit cell.

    For each crystal, the unit cell is split into ``sym_mult`` molecules and
    identify_canonical_asymmetric_unit picks the "canonical" one, whose heavy-atom centroid
    lies in the space group's asymmetric-unit box (with the fallback described there when no
    centroid or more than one lies in it). ``crystal_batch.pos`` is then replaced by
    the chosen molecules' Cartesian coordinates, from which extract_aunit_orientation (with
    ``canonicalize_orientation=True``) reads the orientation and handedness.

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Collated Z'=1 batch whose unit cell is laid out, per crystal, as ``sym_mult``
        consecutive copies of the asymmetric unit in the atom order of ``z`` (as
        aunit2ucell builds it). Reads ``unit_cell_pos``, ``unit_cell_batch``, ``z``,
        ``batch``, ``num_atoms``, ``sym_mult``, ``T_cf``, ``sg_ind``, ``num_graphs`` and
        ``device``. ``pos`` is reassigned in place.
    asym_unit_dict : dict
        Maps ``str(sg_ind)`` to the three upper bounds of the fractional asymmetric-unit box.
    enforce_right_handedness : bool, default False
        Forwarded to extract_aunit_orientation.
    return_aunit : bool, default False
        Also return the chosen molecules' coordinates.

    Returns
    -------
    mol_position_list : torch.Tensor
        [num_graphs, 3] fractional heavy-atom centroid of each chosen molecule, wrapped into
        [0, 1).
    rotvec_list : torch.Tensor
        [num_graphs, 3] rotation vectors from extract_aunit_orientation.
    handedness_list : torch.Tensor
        [num_graphs] handedness from extract_aunit_orientation: +1 or -1 (0 for a degenerate
        principal-axis basis); all +1 when ``enforce_right_handedness``.
    well_defined_asym_unit_list : list of bool
        Per crystal, whether exactly one molecule's centroid was inside the box.
    canonical_conformer_coords_list : list of torch.Tensor
        Per crystal, [num_atoms, 3] Cartesian coordinates of the chosen molecule; returned
        only when ``return_aunit`` is True.
    """
    canonical_conformer_coords_list = []
    mol_position_list = []
    well_defined_asym_unit_list = []
    for i, (T_cf, sg_ind) in enumerate(zip(crystal_batch.T_cf, crystal_batch.sg_ind)):
        unit_cell_coords = crystal_batch.unit_cell_pos[crystal_batch.unit_cell_batch == i]
        unit_cell_atom_types = crystal_batch.z[crystal_batch.batch == i].repeat(crystal_batch.sym_mult[i])
        # filter for heavy only
        num_heavy_atoms = sum(crystal_batch.z[crystal_batch.batch == i] > 1)
        heavy_ucell_coords = unit_cell_coords[unit_cell_atom_types > 1]
        heavy_ucell_coords = heavy_ucell_coords.reshape(crystal_batch.sym_mult[i], num_heavy_atoms, 3)
        (canonical_conformer_index,
         centroids_fractional,
         well_defined_asym_unit) = (
            identify_canonical_asymmetric_unit(
                T_cf,
                asym_unit_dict,
                sg_ind,
                heavy_ucell_coords))

        unit_cell_coords = unit_cell_coords.reshape(crystal_batch.sym_mult[i], crystal_batch.num_atoms[i], 3)
        canonical_conformer_coords_list.append(unit_cell_coords[canonical_conformer_index[0]])
        mol_position_list.append(centroids_fractional[canonical_conformer_index[0]])
        well_defined_asym_unit_list.extend([well_defined_asym_unit])

    mol_position_list = torch.stack(mol_position_list)
    crystal_batch.pos = torch.cat(canonical_conformer_coords_list)

    '''
    Pose Analysis
    compute the inverse of the rotation required to align the molecule with the cartesian axes
    help from http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html#:~:text=Another%20popular%20rotation%20representation%20is,be%20represented%20in%20this%20form!
    
    '''

    rotvec_list, handedness_list = extract_aunit_orientation(crystal_batch,
                                                             enforce_right_handedness,
                                                             canonicalize_orientation=True)

    if return_aunit:
        return mol_position_list, rotvec_list, handedness_list, well_defined_asym_unit_list, canonical_conformer_coords_list
    else:
        return mol_position_list, rotvec_list, handedness_list, well_defined_asym_unit_list


def extract_aunit_orientation(mol_batch,
                              enforce_right_handedness,
                              canonicalize_orientation: bool = True):
    """Rotation vectors taking each molecule from the standard orientation to its current orientation.

    Ip holds each molecule's heavy-atom principal axes as rows, in ascending order of
    principal moment (batch_molecule_principal_axes_torch). The rotation matrix is
    ``(diag(h, 1, 1) @ Ip)^T``, the transpose of the matrix align_mol_batch_to_standard_axes
    applies for the same h. It is converted with rotmat2rotvec and cleanup_invalid_rotvecs.

    Parameters
    ----------
    mol_batch : MolData or MolCrystalData
        Collated batch. Reads ``pos``, ``batch``, ``num_graphs``, ``num_atoms``, ``z`` and
        ``device``. Not modified.
    enforce_right_handedness : bool
        If False, h is each molecule's principal-axis handedness (compute_Ip_handedness)
        and the matrix is a proper rotation. If True, h = 1 for every molecule; for a
        molecule whose principal-axis frame is left-handed the matrix then has determinant
        -1, and rotmat2rotvec prints a warning.
    canonicalize_orientation : bool, default True
        Apply canonicalize_rotvec to the result.

    Returns
    -------
    rotvec_list : torch.Tensor
        [num_graphs, 3] rotation vectors.
    handedness : torch.Tensor
        [num_graphs] the h used for each molecule, as a float tensor: +1 or -1, or 0 where
        compute_Ip_handedness returns 0 for a degenerate principal-axis basis.
    """
    Ip, Ipm, I = batch_molecule_principal_axes_torch(
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        mol_batch.num_atoms,
        heavy_atoms_only=True,
        atom_types=mol_batch.z
    )

    handedness_list = compute_Ip_handedness(Ip).long()
    eye = torch.eye(3, device=mol_batch.device).tile(mol_batch.num_graphs, 1, 1)
    if not enforce_right_handedness:
        eye[:, 0, 0] = handedness_list.flatten()

    # det should all be 1, as we should have correct handedness
    rotation_matrix_list = extract_rotmat(eye, torch.linalg.inv(Ip)).permute(0, 2, 1)

    rotvec_list = rotmat2rotvec(rotation_matrix_list)
    rotvec_list = cleanup_invalid_rotvecs(rotation_matrix_list, rotvec_list)

    if canonicalize_orientation:
        canonicalize_rotvec(rotvec_list)

    return rotvec_list, eye[:, 0, 0]  # the actual handedness


def canonicalize_rotvec(rotvecs: torch.Tensor):
    """Move rotation vectors to the +z half-space without changing the rotations they encode.

    A rotation by angle theta about unit axis n is the same rotation as one by
    2 * pi - theta about -n, so constraining the axis to a half-sphere gives uniqueness;
    the +z direction is taken as canonical. Rows with a negative z component are replaced by
    ``-v / |v| * (2 * pi - |v|)``, which has z > 0 whenever ``|v| < 2 * pi``. Rows with
    z >= 0 are unchanged.

    Parameters
    ----------
    rotvecs : torch.Tensor
        [n, 3] rotation vectors; modified in place.

    Returns
    -------
    torch.Tensor
        ``rotvecs`` itself.
    """
    flip_inds = torch.where(rotvecs[:, -1] < 0)[0]
    flip_vecs = rotvecs[flip_inds]
    flip_norms = torch.linalg.norm(flip_vecs, dim=-1)
    new_norms = 2 * torch.pi - flip_norms
    new_vecs = -flip_vecs / flip_norms[:, None] * new_norms[:, None]
    rotvecs[flip_inds] = new_vecs
    return rotvecs


def cleanup_invalid_rotvecs(rotation_matrix_list, rotvec_list):
    """Rescale rotation vectors to the angle given by the matrix trace, patching degenerate rows.

    Parameters
    ----------
    rotation_matrix_list : torch.Tensor
        [n, 3, 3] rotation matrices.
    rotvec_list : torch.Tensor
        [n, 3] rotation vectors supplying the axis directions. Rows patched below are
        overwritten in place.

    Returns
    -------
    torch.Tensor
        [n, 3] new tensor: the direction of ``rotvec_list`` scaled to the angle
        ``arccos((tr(R) - 1) / 2)``. Rows with ``(tr(R) - 1) / 2 >= 1`` (rotation angle zero
        to rounding) become angle pi about (1, 1, 1) / sqrt(3). Rows whose angle is not
        finite get angle pi with their direction kept. Rows whose direction is not finite
        or is all zeros become angle pi about (1, 1, 1) / sqrt(3).
    """
    r_arg_list = (torch.einsum('ijj->i', rotation_matrix_list) - 1) / 2
    bad_rotations = torch.argwhere(r_arg_list >= 1).flatten()
    # if we are close enough to one, the arccos will NaN
    # situation corresponds to a rotation by ~pi, with an unknown direction
    # fortunately, either direction results in the same transformation (C2)
    # some ill conditioned rotations may also throw |args| greater than 1 -
    # for safety must set these as something, may as well be this  # todo look at such ill-conditioned cases
    # set rotation then as pi
    rotvec_list[bad_rotations, :] = 1
    r_list = torch.arccos(r_arg_list)
    r_list[bad_rotations] = torch.pi
    more_bad_rotations = torch.argwhere(~torch.isfinite(r_list)).flatten()
    if len(more_bad_rotations) > 0:  # manually catch other edge cases
        r_list[more_bad_rotations] = torch.pi
    # more manual edge cases
    bad_directions = torch.argwhere(torch.logical_or(
        (~torch.isfinite(rotvec_list.sum(1))),
        (rotvec_list.abs().sum(1) == 0))
    ).flatten()
    if len(bad_directions) > 0:
        r_list[bad_directions] = torch.pi
        rotvec_list[bad_directions] = 1
    rotvec_list = rotvec_list / torch.linalg.norm(rotvec_list, dim=-1)[:, None] * r_list[:, None]
    return rotvec_list


def identify_canonical_asymmetric_unit(T_cf, asym_unit_dict, sg_ind, unit_cell_coords):
    """Pick the molecule of one unit cell whose centroid lies in the space group's asymmetric-unit box.

    Parameters
    ----------
    T_cf : torch.Tensor
        [3, 3] Cartesian-to-fractional transform of the crystal.
    asym_unit_dict : dict
        Maps ``str(sg_ind)`` to the three upper bounds of the fractional asymmetric-unit box.
    sg_ind : int or torch.Tensor
        Space group number.
    unit_cell_coords : torch.Tensor
        [Z, n_atoms, 3] Cartesian coordinates of the Z molecules of the unit cell.

    Returns
    -------
    canonical_conformer_index : list or torch.Tensor
        One-element container of the chosen molecule's index: a one-element long tensor
        when exactly one centroid is in the box, otherwise a list holding one 0-d tensor.
    centroids_fractional : torch.Tensor
        [Z, 3] fractional centroids (unweighted means over the given atoms), wrapped into
        [0, 1).
    well_defined_asym_unit : bool
        True only when exactly one centroid is in the box.

    Notes
    -----
    A centroid is in the box when ``0 <= f_k <= bound_k`` for k = 0, 1, 2
    (find_coord_in_box_torch with epsilon 0). With no centroid in the box, the molecule
    whose wrapped fractional centroid has the smallest Euclidean norm is chosen from all Z;
    with several in the box, the smallest norm among those (torch.argmin, first index on
    exact ties). The dimension-by-dimension branch for tied norms is not reached, because
    its test ``len(set(dists)) < len(dists)`` builds a set of 0-d tensors, which hash by
    object identity.
    """
    # cartesian mol centroids -> fractional mol centroids -> confirm inside box
    centroids_cartesian = unit_cell_coords.mean(-2)  # assume a shape [Z, n_atoms, 3]
    centroids_fractional = fractional_transform(centroids_cartesian, T_cf)
    centroids_fractional -= torch.floor(centroids_fractional)
    aunit_size = asym_unit_dict[str(int(sg_ind))]
    canonical_conformer_index_i = find_coord_in_box_torch(centroids_fractional, aunit_size)
    # if we didn't find one, or found more than one pick the closest.
    # In some cases, they are truly indistinguishable.
    if len(canonical_conformer_index_i) == 0:
        well_defined_asym_unit = False
        canonical_conformer_index = [torch.argmin(torch.linalg.norm(centroids_fractional, axis=1))]
    elif len(canonical_conformer_index_i) == 1:
        well_defined_asym_unit = True  # if there is any ambiguity, it is not 'well defined'
        canonical_conformer_index = canonical_conformer_index_i * 1
    elif len(canonical_conformer_index_i) > 1:
        # if we find more than one inside
        well_defined_asym_unit = False
        dists = torch.linalg.norm(centroids_fractional[canonical_conformer_index_i], axis=1)
        if len(set(dists)) < len(dists):  # if any are equal, go dimension-by-dimension
            canonical_conformer_index = [canonical_conformer_index_i[0]]  # pre-set a random ind, overwrite later
            for dim in range(3):
                dists = centroids_fractional[canonical_conformer_index_i][:, dim]
                if len(set(dists)) < len(dists):
                    pass  # if are equal, go to next dim
                else:
                    canonical_conformer_index = [canonical_conformer_index_i[torch.argmin(dists)]]
                    break  # stop at this dim
        else:
            canonical_conformer_index = [canonical_conformer_index_i[torch.argmin(dists)]]
    return canonical_conformer_index, centroids_fractional, well_defined_asym_unit


def align_mol_batch_to_standard_axes(mol_batch, handedness=None, return_rot=False):
    """Align each molecule's principal inertial axes to the (x, y, z) or (-x, y, z) axes, in place.

    The axes are geometric principal axes (every atom has mass 1) over heavy atoms (z > 1),
    from batch_molecule_principal_axes_torch: rows of Ip in ascending order of principal
    moment, with signs fixed by that function. After recentring each molecule on its
    heavy-atom centroid, the matrix ``diag(h, 1, 1) @ Ip`` is applied, so the
    smallest-moment axis maps to h * x, the middle axis to y and the largest to z. The
    matrix is a proper rotation only where h equals the molecule's own principal-axis
    handedness (compute_Ip_handedness); elsewhere it includes a reflection and the molecule
    is mirrored.

    Parameters
    ----------
    mol_batch : MolData or MolCrystalData
        Collated batch. Reads ``pos``, ``batch``, ``num_graphs``, ``num_atoms`` and ``z``.
        ``pos`` is recentred (recenter_molecules) and then reassigned to the aligned
        positions.
    handedness : torch.Tensor, optional
        [num_graphs, k]; column 0 gives h (+1 or -1) for each molecule. None sets h = +1 for
        every molecule. A non-tensor value is indexed as ``handedness[:, :1]`` and assigned
        without flattening.
    return_rot : bool, default False
        Also return the applied matrices.

    Returns
    -------
    mol_batch : MolData or MolCrystalData
        The input object, modified in place.
    std_rotations : torch.Tensor
        [num_graphs, 3, 3] applied matrices ``diag(h, 1, 1) @ Ip``; returned only when
        ``return_rot`` is True.
    """
    Ip, Ipm, I = batch_molecule_principal_axes_torch(
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        mol_batch.num_atoms,
        heavy_atoms_only=True,
        atom_types=mol_batch.z
    )

    eye = torch.tile(torch.eye(3,
                               device=mol_batch.pos.device,
                               dtype=torch.float32),
                     (mol_batch.num_graphs, 1, 1)
                     )  # set as right-handed by default

    if handedness is not None:  # otherwise, custom
        # alignment vector is (x,y,z) by default, or (-x,y,z) for left-handed structures
        if torch.is_tensor(handedness):
            eye[:, 0, 0] = handedness[:, :1].flatten()  # flatten - in case it's higher dimensional [n,1]
        else:
            eye[:, 0, 0] = handedness[:, :1]

    # rotation is p_align = R @ p_orig,
    # R = p_align \ p_orig

    std_rotations = extract_rotmat(eye, torch.linalg.inv(Ip))
    mol_batch.recenter_molecules()  # ensures molecules are centered
    mol_batch.pos = apply_rotation_to_batch(mol_batch.pos,
                                            std_rotations,
                                            mol_batch.batch)
    # test
    # Ip, Ipm, I = batch_molecule_principal_axes_torch(
    #     mol_batch.pos,
    #     mol_batch.batch,
    #     mol_batch.num_graphs,
    #     mol_batch.num_atoms
    # )
    # assert torch.all(torch.isclose(Ip, eye, atol=1e-4))
    if return_rot:
        return mol_batch, std_rotations
    else:
        return mol_batch


def unit_cell_gather_index(batch_ind, z_prime, sym_mult, num_graphs: int):
    """The asymmetric-unit atom each ``unit_cell_pos`` row is an image of, in the builder's layout.

    ``build_unit_cell`` (``aunit2ucell``) writes a crystal's unit cell image by image, ``sym_mult`` copies of its
    asymmetric unit; a Z'>1 crystal is built unit by unit (``split_to_zp1_batch``, then ``join_zp1_ucell_batch``), so
    its unit cell is ``[unit][image][atom]``: every image of unit 0, then every image of unit 1. At Z'=1 that is the
    image-major tiling ``z[batch_ind == i].repeat(sym_mult[i])``, and the index is that tiling's, element for element.
    Every reader of unit-cell elements gathers through this index (the MLIP input builders, the ASE and spglib
    assemblies, zp_doubling), so that elements follow the positions whatever the units hold. Until 2026-10-23 they
    tiled the whole asymmetric unit image by image, which gives a two-compound Z'>1 crystal the other compound's
    element on some rows (audit #15).

    Parameters
    ----------
    batch_ind : torch.Tensor
        [n] graph index of each asymmetric-unit atom, graphs contiguous and in order (a batch's ``batch``, or its
        ``aux_ind == 0`` atoms on a cluster).
    z_prime : torch.Tensor
        [num_graphs] units per crystal. A crystal's units are ``z_prime`` equal contiguous blocks of its atoms, as
        ``split_to_zp1_batch`` requires at this stage.
    sym_mult : torch.Tensor
        [num_graphs] symmetry images per unit.
    num_graphs : int

    Returns
    -------
    index : torch.Tensor
        [sum_i n_i * sym_mult_i] long index into the asymmetric-unit atoms (``z[index]`` are the unit-cell
        elements).
    out_n : torch.Tensor
        [num_graphs] unit-cell atoms per crystal, ``n_i * sym_mult_i``.

    Raises
    ------
    ValueError
        If a crystal has ``z_prime < 1`` or atoms that do not divide into its ``z_prime`` units (an explicit raise).
    """
    n = torch.bincount(batch_ind, minlength=num_graphs)                       # asymmetric-unit atoms per crystal
    zp = z_prime.reshape(-1).to(n.dtype)
    m = sym_mult.reshape(-1).to(n.dtype)
    out_n = n * m
    bad = (zp < 1) | (n % zp.clamp(min=1) != 0)
    n_units, total, n_bad = torch.stack([zp.clamp(min=0).sum(), out_n.sum(), bad.sum()]).tolist()   # the one sync
    if n_bad:
        rows = torch.nonzero(bad).flatten()[:8].tolist()
        raise ValueError(f'unit_cell_gather_index: crystals {rows} have z_prime < 1 or atoms that do not divide into '
                         f'their z_prime units')
    unit_row = torch.repeat_interleave(torch.arange(num_graphs, device=n.device), zp, output_size=n_units)
    unit_n = (n // zp)[unit_row]                                              # atoms per unit
    unit_out = unit_n * m[unit_row]                                           # unit-cell atoms per unit
    src_off = torch.cumsum(unit_n, 0) - unit_n                                # first atom of each unit
    dst_off = torch.cumsum(unit_out, 0) - unit_out
    slot = torch.arange(total, device=n.device) - torch.repeat_interleave(dst_off, unit_out, output_size=total)
    within = slot % torch.repeat_interleave(unit_n, unit_out, output_size=total)   # the tile, per unit
    return torch.repeat_interleave(src_off, unit_out, output_size=total) + within, out_n


def unit_cell_z(z, z_prime, sym_mult):
    """One crystal's unit-cell elements in the builder's layout, by arithmetic independent of unit_cell_gather_index
    (the per-crystal list paths use this, the vectorised paths the index, so that the equivalence tests compare two
    derivations): ``z`` (the crystal's asymmetric-unit atoms) as ``z_prime`` equal units, each tiled ``sym_mult``
    times, unit by unit, ``[unit][image][atom]``; at Z'=1 ``z.repeat(sym_mult)``. ``reshape`` raises when the atoms
    do not divide into the units."""
    return z.reshape(int(z_prime), -1).repeat(1, int(sym_mult)).reshape(-1)


def aunit2ucell(mol_batch):
    """Build each crystal's unit cell by applying its symmetry operators to the posed asymmetric unit.

    For each operator (a 4x4 fractional affine matrix), the molecule's heavy-atom centroid
    is placed at ``op @ (c, 1)``, where c is ``aunit_centroid`` clipped to [0, 1 - 1e-4]
    (assign_aunit_centroid); the result is wrapped into [0, 1), except that a component
    exactly equal to 1.0 is left at 1.0. The molecule's coordinates, centred on its
    heavy-atom centroid, are converted to fractional, multiplied by the operator's 3x3
    part, shifted to the new centroid and converted back to Cartesian.

    Parameters
    ----------
    mol_batch : MolCrystalData
        Collated Z'=1 batch (``aunit_centroid`` of shape [num_graphs, 3]) whose ``pos`` holds
        the posed asymmetric units. Reads ``pos``, ``batch``, ``num_graphs``, ``num_atoms``,
        ``sym_mult``, ``z``, ``aunit_centroid``, ``T_fc``, ``T_cf``, ``symmetry_operators``
        (per crystal, [sym_mult, 4, 4]) and ``device``. Not modified.

    Returns
    -------
    unit_cell_pos : torch.Tensor
        [sum(num_atoms * sym_mult), 3] Cartesian positions, ordered by crystal, then by
        symmetry operator, then in the atom order of ``pos``.
    unit_cell_batch : torch.Tensor
        [sum(num_atoms * sym_mult)] crystal index of each atom.
    ucell_node2ucell_mol : torch.Tensor
        [sum(num_atoms * sym_mult)] molecule index of each atom, counted across the whole
        batch (0 to sum(sym_mult) - 1), not restarted per crystal.
    """
    atoms_per_mol = mol_batch.num_atoms
    sym_mult = mol_batch.sym_mult
    atoms_per_unit_cell = mol_batch.num_atoms * sym_mult
    tot_num_mols = torch.sum(sym_mult)

    aunit_coords_c = mol_batch.pos
    centroids_f = mol_batch.aunit_centroid

    # repeat each molecule Z times
    mol2ucell = torch.arange(len(sym_mult), device=sym_mult.device).repeat_interleave(sym_mult)

    ucell_centroids_f = mol_batch.assign_aunit_centroid(mol_batch.aunit_centroid[mol2ucell])
    fc_transform_list = mol_batch.T_fc
    cf_transform_list = mol_batch.T_cf

    # same, atomwise
    ucell_node2mol_node = block_repeat_interleave(mol_batch.num_atoms, sym_mult)

    unit_cell_coords_c = aunit_coords_c[ucell_node2mol_node]
    unit_cell_batch = mol_batch.batch[ucell_node2mol_node]

    # assign which molecule each atom is in
    ucell_node2ucell_mol = torch.arange(tot_num_mols, device=mol_batch.device).repeat_interleave(
        atoms_per_mol.repeat_interleave(sym_mult))

    # MK todo this tensor(np.concatenate) can't be very efficient
    molwise_sym_ops = torch.tensor(np.concatenate(mol_batch.symmetry_operators, axis=0),
                                   device=centroids_f.device, dtype=torch.float32)

    # add 4th dimension as a dummy for affine transforms, should have shape (num_aunits, 4, 4)
    unit_cell_affine_centroids_f = torch.cat(
        (
            ucell_centroids_f,
            torch.ones(ucell_centroids_f.shape[:-1] + (1,), dtype=torch.float32)
            .to(aunit_coords_c.device)),
        dim=-1)

    # get all molecule centroids via symmetry ops
    ucell_centroids_f = torch.einsum('nij,nj->ni', (molwise_sym_ops, unit_cell_affine_centroids_f))[..., :-1]

    # force centroids within unit cell # note this disallows centroids at fractional 1.0 and moves them to 0.0, which causes some artifacting
    # centroids_f_in_cell = ucell_centroids_f - torch.floor(ucell_centroids_f)
    # this one lets them sit at 1.0 exactly
    mask = (ucell_centroids_f != 1.0)
    centroids_f_in_cell = torch.where(mask,
                                      ucell_centroids_f - torch.floor(ucell_centroids_f),
                                      ucell_centroids_f)

    # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
    centered_coords_c = center_batch(unit_cell_coords_c,
                                     unit_cell_batch,
                                     mol_batch.num_graphs,
                                     atoms_per_unit_cell,
                                     center_on_heavy_atoms=True,
                                     atom_types=mol_batch.z[ucell_node2mol_node])

    centered_coords_f = fractional_transform(centered_coords_c,
                                             cf_transform_list[unit_cell_batch])
    rot_coords_f = torch.einsum('nj,nij->ni', (centered_coords_f, molwise_sym_ops[ucell_node2ucell_mol, :3, :3]))

    # translate rotated aunits to their correct position
    unit_cell_pos = torch.einsum('nij,nj->ni',
                                 (fc_transform_list[unit_cell_batch],
                                  rot_coords_f + centroids_f_in_cell[ucell_node2ucell_mol, :]))

    return unit_cell_pos, unit_cell_batch, ucell_node2ucell_mol


def descale_asymmetric_unit(asym_unit_dict, mol_position, sg_inds):
    """Map positions scaled to [0, 1] onto each space group's fractional asymmetric-unit box.

    Only space groups present in ``asym_unit_dict`` work, and the box form cannot represent
    asymmetric units that are not neat parallelepipeds [0, b_0] x [0, b_1] x [0, b_2].

    Parameters
    ----------
    asym_unit_dict : dict
        Maps ``str(sg_ind)`` to a [3] tensor of the box's upper bounds (e.g. from
        build_asym_unit_dict). A missing space group raises KeyError.
    mol_position : torch.Tensor
        [n, 3] positions scaled so that the box spans [0, 1] on each axis.
    sg_inds : iterable
        n space-group numbers, one per row of ``mol_position``.

    Returns
    -------
    torch.Tensor
        [n, 3] ``mol_position`` multiplied elementwise by each row's box bounds.
    """

    return mol_position * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])


def rescale_asymmetric_unit(asym_unit_dict, mol_position, sg_inds):
    """Map fractional positions in each space group's asymmetric-unit box onto [0, 1].

    Inverse of descale_asymmetric_unit.

    Only space groups present in ``asym_unit_dict`` work, and the box form cannot represent
    asymmetric units that are not neat parallelepipeds [0, b_0] x [0, b_1] x [0, b_2].

    Parameters
    ----------
    asym_unit_dict : dict
        Maps ``str(sg_ind)`` to a [3] tensor of the box's upper bounds (e.g. from
        build_asym_unit_dict). A missing space group raises KeyError.
    mol_position : torch.Tensor
        [n, 3] fractional positions, spanning [0, b_k] on axis k.
    sg_inds : iterable
        n space-group numbers, one per row of ``mol_position``.

    Returns
    -------
    torch.Tensor
        [n, 3] ``mol_position`` divided elementwise by each row's box bounds.
    """
    return mol_position / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])


def get_aunit_positions(mol_batch,
                        std_orientation: bool = True,
                        mol_handedness: OptTensor = None,
                        ) -> Tensor:
    """Cartesian positions of each crystal's asymmetric unit, posed by its rotation and fractional centroid.

    Returns ``R(aunit_orientation[:, :3]) @ p + T_fc @ aunit_centroid[:, :3]`` per atom,
    where R is rotvec2rotmat and p is the molecule recentred on its heavy-atom centroid
    (recenter_molecules). With ``std_orientation`` True, p is first aligned by
    align_mol_batch_to_standard_axes with handedness ``mol_handedness[:, :1]``; with it
    False, ``pos`` is only recentred and no handedness is applied. Only the first three
    columns of ``aunit_centroid`` and ``aunit_orientation`` (the first asymmetric unit) are
    used.

    Parameters
    ----------
    mol_batch : MolCrystalData
        Collated batch. Reads ``pos``, ``batch``, ``num_graphs``, ``num_atoms``, ``z``,
        ``T_fc``, ``aunit_centroid`` (fractional) and ``aunit_orientation`` (rotation
        vectors). ``mol_batch.pos`` is modified in place: recentred and, with
        ``std_orientation``, aligned.
    std_orientation : bool, default True
        Align to the standard axes before rotating.
    mol_handedness : torch.Tensor, optional
        [num_graphs, k]; column 0 is used. Required when ``std_orientation`` is True.

    Returns
    -------
    torch.Tensor
        [num_nodes, 3] Cartesian positions; not assigned to ``mol_batch``.

    Raises
    ------
    ValueError
        If ``std_orientation`` is True and ``mol_handedness`` is None (an explicit
        raise, so it holds under ``python -O``; before 2026-10-22 a bare assert).
    """
    rotations_list = rotvec2rotmat(mol_batch.aunit_orientation[:, :3])

    if std_orientation:  # align canonical conformers principal axes to cartesian axes - not usually done here, but allowed
        if mol_handedness is None:
            raise ValueError("get_aunit_positions: std_orientation=True aligns each molecule to the standard axes "
                             "with its handedness; mol_handedness is None")
        mol_batch = align_mol_batch_to_standard_axes(mol_batch, handedness=mol_handedness[:, :1])
    else:
        mol_batch.recenter_molecules()

    # apply rotations and fractional translations
    pos = (apply_rotation_to_batch(mol_batch.pos,
                                   rotations_list,
                                   mol_batch.batch)
           + fractional_transform(mol_batch.aunit_centroid[:, :3], mol_batch.T_fc)[mol_batch.batch])

    # # ensure embeddings rotate along with the aunit positions
    # this is a bit risky - leave it for now
    # if hasattr(mol_batch, 'embedding'):
    #     if mol_batch.embedding.shape[1] == 3:  # if there is a vector embedding, we should also rotate it
    #         mol_batch.embedding = torch.einsum('nij, njk -> njk', rotations_list, mol_batch.embedding)

    return pos


def canonicalize_aunit_order(batch):
    """Reorder each crystal's asymmetric units by the norm of their fractional centroids.

    Slots at index >= ``z_prime`` are unused. For sorting, an unused slot's centroid is
    replaced by (1, 1, 1); the order is ascending Euclidean norm of the fractional centroid
    (torch.argsort, so the order of exact ties is not specified). In the outputs, unused
    slots carry centroid 0.5 and orientation pi / 2 in every component, while handedness
    values are reordered with no placeholder fill. All operations are out of place.

    Parameters
    ----------
    batch : MolCrystalData
        Collated batch. Reads ``num_graphs``, ``max_z_prime``, ``z_prime``, ``device``,
        ``aunit_centroid`` and ``aunit_orientation`` ([num_graphs, 3 * max_z_prime]) and
        ``aunit_handedness`` ([num_graphs, max_z_prime]). Not modified.

    Returns
    -------
    aunit_centroid : torch.Tensor
        [num_graphs, 3 * max_z_prime]
    aunit_orientation : torch.Tensor
        [num_graphs, 3 * max_z_prime]
    aunit_handedness : torch.Tensor
        [num_graphs, max_z_prime]
    """
    n, k = batch.num_graphs, batch.max_z_prime
    per_aunit_centroids = batch.aunit_centroid.reshape(n, k, 3)

    idx = torch.arange(k, device=batch.device)[None, ...]
    mask = (idx >= batch.z_prime[..., None])[..., None].expand(-1, -1, 3)

    # ordering uses the '1' fill; out-of-place
    centroids_for_sort = torch.where(mask, torch.ones_like(per_aunit_centroids), per_aunit_centroids)
    origin_dists = centroids_for_sort.norm(dim=2)
    canonical_order = origin_dists.argsort(dim=1)
    order_exp = canonical_order.unsqueeze(-1).expand(-1, -1, 3)

    # gathered values use the 0.5 placeholder fill; out-of-place, differentiable source
    centroids_for_gather = torch.where(mask, torch.full_like(per_aunit_centroids, 0.5), per_aunit_centroids)
    sorted_centroids = torch.gather(centroids_for_gather, 1, order_exp)
    aunit_centroid = sorted_centroids.reshape(n, k * 3)

    per_aunit_orient = batch.aunit_orientation.reshape(n, k, 3)
    orient_for_gather = torch.where(mask, torch.full_like(per_aunit_orient, torch.pi / 2), per_aunit_orient)
    sorted_orient = torch.gather(orient_for_gather, 1, order_exp)
    aunit_orientation = sorted_orient.reshape(n, k * 3)

    aunit_handedness = torch.gather(batch.aunit_handedness, 1, canonical_order)
    return aunit_centroid, aunit_orientation, aunit_handedness

# def canonicalize_aunit_order(batch):  # old
#     # canonical aunit ordering according to distance to the origin, with x, y, z, tiebreak
#     # TODO implement tiebreak
#     per_aunit_centroids = batch.aunit_centroid.reshape(batch.num_graphs, batch.max_z_prime, 3)
#     # mask out elements with lower z prime than max
#     idx = torch.arange(batch.max_z_prime, device=batch.device)[None, ...]
#     mask = (idx >= (batch.z_prime[..., None]))[..., None].expand(-1, -1, 3)
#     per_aunit_centroids[mask] = 1  # this will put lower Z' options always at the end
#     origin_dists = per_aunit_centroids.norm(dim=2)
#     canonical_order = origin_dists.argsort(dim=1)
#     canonical_order_exp = canonical_order.unsqueeze(-1).expand(-1, -1, 3)  # [n, k, 3]
#     per_aunit_centroids[mask] = 0.5  # this is the placeholder value for 'not being used'
#     sorted_centroids = torch.gather(per_aunit_centroids, dim=1, index=canonical_order_exp)
#
#     aunit_centroid = sorted_centroids.reshape(batch.num_graphs, batch.max_z_prime * 3)
#
#     # also reorder the orientations accordingly
#     per_aunit_orient = batch.aunit_orientation.reshape(batch.num_graphs, batch.max_z_prime, 3)
#     per_aunit_orient[mask] = torch.pi / 2  # this places all unused samples in the same state
#     canonical_order_exp = canonical_order.unsqueeze(-1).expand(-1, -1, 3)  # [n, k, 3]
#     sorted_orient = torch.gather(per_aunit_orient, dim=1, index=canonical_order_exp)
#     aunit_orientation = sorted_orient.reshape(batch.num_graphs, batch.max_z_prime * 3)
#
#     aunit_handedness = torch.gather(batch.aunit_handedness, dim=1, index=canonical_order)
#     return aunit_centroid, aunit_orientation, aunit_handedness


def protonate_mol(atom_types, coords):
    """Add hydrogens to a set of atoms with RDKit, keeping the input atom positions.

    An RDKit molecule is built from the atoms and coordinates alone, with no bonds added,
    sanitized with ``SANITIZE_ADJUSTHS`` only, and passed to ``Chem.AddHs(addCoords=True)``.
    The number of hydrogens added to each atom is therefore the implicit-hydrogen count
    RDKit assigns to an atom with no explicit bonds.

    Parameters
    ----------
    atom_types : sequence of int
        [N] atomic numbers.
    coords : array-like
        [N, 3] positions in Angstrom.

    Returns
    -------
    new_atom_types : np.ndarray
        [N + num_H] int atomic numbers: the input atoms in input order, then the added
        hydrogens.
    new_coords : np.ndarray
        [N + num_H, 3] float positions.
    """

    # 1. Build an RDKit mol without sanitization
    mol = Chem.RWMol()
    for Z in atom_types:
        mol.AddAtom(Chem.Atom(int(Z)))

    # 2. Add a conformer with your coords
    conf = Chem.Conformer(len(atom_types))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol = mol.GetMol()
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    mol.AddConformer(conf, assignId=True)

    # 3. Add hydrogens (fast, valence-based)
    molH = Chem.AddHs(mol, addCoords=True)

    # If your input had 3D coordinates, this preserves heavy atom positions
    # and places hydrogens using RDKit's built-in valence geometry rules.
    # No re-embedding needed for most molecules.

    # 4. Extract arrays back out
    confH = molH.GetConformer()
    n = molH.GetNumAtoms()

    new_coords = np.array([list(confH.GetAtomPosition(i)) for i in range(n)], dtype=float)
    new_atom_types = np.array([molH.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=int)

    return new_atom_types, new_coords

    """
    
    m1 = Atoms(positions=coords, numbers=atom_types)
    m2 = Atoms(positions=new_coords, numbers=new_atom_types)
    view([m1, m2])
    
    """
