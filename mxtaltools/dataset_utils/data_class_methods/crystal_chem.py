"""Chemistry full-chain steps mixed into MolCrystalData: sanitize and features.

The two steps of the ingestion contract (``docs/design/crystal_ingestion.md`` §4, Tables 7
and 8, in the GFN repository) that need a chemistry backend. In the chain they run as

    read -> find_molecules -> **sanitize** -> parameterize_pose -> check_rebuild
    -> check_validity -> **features**

``sanitize``
    units -> chemistry: bonds with bond orders, formal charges, hydrogens (each atom's
    ``h_source``), per-unit SMILES. Backend ``'ccdc'`` (the CSD Python API, read lazily,
    with RDKit reading its perception) or ``'none'`` (hydrogen choices that need no
    chemistry). Runs before the pose, so the posed blocks carry the hydrogens.
``features``
    posed rows with chemistry -> per unit an RDKit molecule rebuilt from the stored
    chemistry: SMILES (where sanitize stored none), a packed Morgan fingerprint and
    Gasteiger charges in ``x``. A unit outside RDKit's valence model (an element outside
    ``LEWIS_ELEMENTS``, every problem RDKit names near one) falls back to a partial
    sanitization, flagged in ``features_fallback``.

Both follow the step contract of ``crystal_steps`` (new batch, pass-through rows
bit-identical, one status per failing row, ``done`` bit, ``steps_run`` record, bad
arguments raise, one pass per batch). Neither CCDC nor RDKit is imported until a step needs
it, so ``MolCrystalData`` stays importable without them.

Status codes: ``STATUS`` is ``crystal_steps.STATUS`` followed by ``CHEM_STATUS``; codes
are only ever appended. ``crystal_store.status_table`` reads ``STATUS`` from here. Until
``crystal_steps`` accepts these codes, its steps refuse (``ValueError``) a batch holding
one: keep the rows that pass with ``select_rows`` before ``parameterize_pose``.
"""
import os
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mxtaltools.crystal_building.utils import UNIT_MAP_KEY, unit_layout
from mxtaltools.dataset_utils.construction.cif.rhombohedral import _P as _RH_P
from mxtaltools.dataset_utils.construction.cif.rhombohedral import _P_INV3 as _RH_P_INV3
from mxtaltools.dataset_utils.construction.cif.rhombohedral import _metric as _rh_metric
from mxtaltools.dataset_utils.data_class_methods.crystal_io import POSE_KEYS
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import STATUS as _STEPS_STATUS
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import (DONE_FIND_MOLECULES, DONE_SANITIZE,
                                                                       _copy_batch, _finish, _isolate, _kabsch,
                                                                       _ok_rows, _refuse_second_pass, _require_done,
                                                                       _set, _transforms64, _write_unit_properties)

__all__ = ['MolCrystalChem', 'STATUS', 'CHEM_STATUS', 'BOND_ORDERS', 'H_SOURCES', 'DONE_FEATURES',
           'OLD_FINGERPRINT', 'FEATURES_FALLBACK', 'LEWIS_ELEMENTS']

#: Codes of the chemistry steps, appended to ``crystal_steps.STATUS``. ``sanitize``:
#: ``chemistry_failed``, ``template_mismatch`` (reserved for the template backend, not
#: built), ``hydrogens_failed``, ``hydrogen_count_mismatch``, ``component_mismatch``,
#: ``backend_unavailable``; ``features``: ``features_failed``.
CHEM_STATUS = ('chemistry_failed', 'template_mismatch', 'hydrogens_failed', 'hydrogen_count_mismatch',
               'component_mismatch', 'backend_unavailable', 'features_failed')
STATUS = tuple(_STEPS_STATUS) + CHEM_STATUS
_STATUS = {name: i for i, name in enumerate(STATUS)}
assert len(_STATUS) == len(STATUS), 'a chemistry status repeats an earlier one'
_OK = _STATUS['ok']

#: ``done`` bit of ``features`` (``DONE_SANITIZE`` = 4 is ``crystal_steps``').
DONE_FEATURES = 1 << 6

#: ``mol_bond_order`` codes (index into this tuple). A stored bond ``(i, j)`` has i < j;
#: ``dative`` points from i to j, ``dative_reverse`` from j to i. ``aromatic``,
#: ``delocalised`` and ``pi`` occur only on units RDKit could not read (there the orders
#: are CCDC's own); elsewhere the orders are a Kekule structure.
BOND_ORDERS = ('unspecified', 'single', 'double', 'triple', 'quadruple', 'aromatic', 'delocalised', 'pi', 'dative',
               'dative_reverse')
_BOND = {name: i for i, name in enumerate(BOND_ORDERS)}
_CCDC_BOND = {'Unknown': 'unspecified', 'Single': 'single', 'Double': 'double', 'Triple': 'triple',
              'Quadruple': 'quadruple', 'Aromatic': 'aromatic', 'Delocalised': 'delocalised', 'Pi': 'pi'}

#: ``h_source`` codes: an atom as deposited, or a hydrogen placed by the backend.
H_SOURCES = ('deposited', 'regenerated')

#: The old CSD featurizer's Morgan fingerprint (``featurization_utils.py:30``:
#: ``GetMorganGenerator(radius=2, includeChirality=False)``, whose ``fpSize`` default is
#: 2048), as a ``features(fingerprint=...)`` argument.
OLD_FINGERPRINT = dict(radius=2, n_bits=2048, chirality=False)

#: ``features_fallback`` codes (index into this tuple): how ``features`` sanitized a row's
#: units, the furthest any unit needed. ``none``: RDKit's full sanitization; ``valence``:
#: every sanitization step except the valence check; ``kekule``: also without
#: kekulization and aromaticity perception (the stored aromatic flags are kept).
FEATURES_FALLBACK = ('none', 'valence', 'kekule')

#: Atomic numbers whose bonding RDKit's valence model is meant to hold: H, C, N, O, F, P,
#: S, Cl, Se, Br, I. A unit of these alone that RDKit cannot sanitize has stored chemistry
#: that is no Lewis structure (a missing charge or hydrogen, or a ring called aromatic
#: that has no Kekule form), so ``features`` gives it no fallback.
LEWIS_ELEMENTS = frozenset((1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53))

#: Bonds: the fallback's reach. A unit with an element outside ``LEWIS_ELEMENTS`` gets the
#: fallback only when every problem RDKit names lies near such an atom: an atom the valence
#: check refuses within ``_VALENCE_REACH`` bonds of one (the atom itself, a ligand atom bound
#: to it, or the atom beyond: a carbonyl's O), an aromatic system with no Kekule form within
#: ``_KEKULE_REACH`` (it holds one or is bonded to one: a ring bound to a metal, a ring holding
#: boron). A problem further out is the stored chemistry's own (a missing charge, a ring called
#: aromatic that has no Kekule form), which a unit of ``LEWIS_ELEMENTS`` alone is refused for.
_VALENCE_REACH = 2
_KEKULE_REACH = 1

_CHEMISTRIES = ('ccdc', 'none')
_HYDROGENS = ('regenerate', 'add_missing', 'as_read', 'remove')
_ON_CHANGE = ('flag', 'refuse')

#: Angstrom: a backend atom and a row atom of the same element closer than this, modulo
#: the lattice, are the same site. CCDC keeps the file's coordinates (float64 parse), so
#: the measured differences are rounding, far below it; two sites of one element this
#: close are superposed atoms, which make the map ambiguous (``chemistry_failed``).
_MAP_TOL = 0.01

#: Angstrom: a backend hydrogen and a deposited hydrogen closer than this, modulo the lattice,
#: are the same atom. A hydrogen CCDC keeps sits on the file's coordinates (under 1e-8 A on
#: the 316 gate CIFs, every mode); one it places lands at least 1e-3 A away (two of 5,054 in
#: 'regenerate' within 0.01 A), so ``_MAP_TOL`` would take placed hydrogens for deposited ones.
_SAME_H_TOL = 1e-4

#: Relative to the row's largest squared cell length: a row whose metric is hexagonal and equals
#: P^T G P, G the metric of the cell CCDC reads and P the reader's rhombohedral-to-hexagonal
#: matrix (``cif/rhombohedral.py``), is a row the reader re-expressed from rhombohedral axes,
#: and CCDC's fractional coordinates are on those axes. The two metrics come from the same
#: file values (the reader's cell formulae, float64), so they agree to rounding, far below it.
_RH_METRIC_TOL = 1e-6

#: Angstrom: RDKit's mol2 coordinates must equal CCDC's to this (the mol2 block writes four
#: decimals), else RDKit's atom order is not CCDC's.
_MOL2_TOL = 1e-3

#: Attributes ``sanitize`` writes (a batch holding any of them has been sanitized).
_SANITIZE_KEYS = ('h_source', 'formal_charge', 'mol_bond_order', 'unit_smiles', 'has_chemistry',
                  'protonation_checked', 'protonation_changed', 'h_count_change', 'removed_deposit_site',
                  'removed_deposit_frac', 'charge_imbalance')
#: Attributes only ``features`` writes.
_FEATURE_KEYS = ('unit_fingerprint', 'charges_nonfinite', 'features_fallback')

#: Per-atom attributes ``sanitize`` can fill for a hydrogen it places: gathered for kept
#: atoms, set for new ones (``mol_ind`` and the unit map ``unit_ind`` are set for every atom of
#: a changed row, to its unit; the map since stage 3, 2026-10-23).
_FILLABLE = ('z', 'pos', 'deposit_frac', 'deposit_site', 'occupancy', 'mol_ind', UNIT_MAP_KEY)
#: Per-atom attributes known by name, for a batch whose rows each hold one atom (where a
#: per-atom and a per-row slice entry coincide).
_KNOWN_ATOM_KEYS = frozenset(_FILLABLE + ('x', 'rebuild_dev', 'h_source', 'formal_charge', 'h_count_change'))

_F64 = torch.float64


class _Refusal(Exception):
    """Internal: one row fails a chemistry step; carries the ``STATUS`` name and a message."""

    def __init__(self, code: str, message: str, **info):
        assert code in _STATUS, code
        self.code = code
        self.info = info
        super().__init__(message)


# noinspection PyAttributeOutsideInit
class MolCrystalChem:
    """Mixin providing the chemistry full-chain steps (module docstring)."""

    def sanitize(self, *, chemistry: str, hydrogens: str, on_protonation_change: str):
        """Give each row's units their chemistry: bond orders, formal charges, hydrogens, SMILES.

        Runs after ``find_molecules`` and before ``parameterize_pose`` (the posed blocks then
        carry the hydrogens; a posed batch is refused).

        Parameters
        ----------
        chemistry : {'ccdc', 'none'}
            ``'ccdc'``: each ok row's own CIF block (``source_path`` / ``block_name``, kept
            by ``read``) is read with the CSD Python API, ``assign_bonds`` is run, then the
            hydrogen choice; the molecules (``Crystal.molecule``) are mapped onto the row
            and each CCDC component is read into RDKit through its mol2 block (the old
            featurizer's route: RDKit's mol2 reader sets formal charges from CCDC's SYBYL
            atom types, which CCDC's own ``set_formal_charges`` leaves at 0 on nitro,
            iminium and borate atoms). ``'none'``: only the hydrogen choices that need no
            chemistry (``'as_read'``, ``'remove'``); no bond orders, charges or SMILES.
            (The contract's ``'template'`` backend is not built.)
        hydrogens : {'regenerate', 'add_missing', 'as_read', 'remove'}
            ``'regenerate'``: every hydrogen placed by CCDC (``add_hydrogens(mode='all')``,
            which discards the deposited ones); ``'add_missing'``: deposited hydrogens kept
            and those CCDC deems missing added (``mode='missing'``); ``'as_read'``: the
            deposited hydrogens; ``'remove'``: none. In every mode a CCDC hydrogen on a
            deposited hydrogen's coordinates (within 1e-4 A modulo the lattice) is that
            deposited atom: CCDC's ``add_hydrogens(mode='all')`` and ``remove_hydrogens``
            keep a hydrogen bonded to two atoms. With ``'remove'`` any hydrogen CCDC leaves refuses
            the row (``hydrogens_failed``).
        on_protonation_change : {'flag', 'refuse'}
            Where the backend's hydrogen count on some heavy atom differs from the
            deposited one: ``'flag'`` keeps the row (``protonation_changed``), ``'refuse'``
            gives it ``hydrogen_count_mismatch``.

        Returns
        -------
        MolCrystalDataBatch
            A new batch. On rows that pass with ``'ccdc'``, each unit is its heavy atoms in
            ``find_molecules``' order followed by its hydrogens, grouped by parent heavy
            atom (in unit 0 in parent order, then CCDC's order; units 1+ re-indexed onto
            unit 0's by parent and, among one parent's hydrogens, by assignment after a
            Kabsch fit of the unit's heavy atoms onto unit 0's, proper unless the mirror
            fit assigns the hydrogens clearly closer), so units stay contiguous
            with one atom order; heavy atoms keep their coordinates exactly (each Z' > 1
            unit its own conformer); a hydrogen sits at its parent's position plus CCDC's
            parent-to-hydrogen vector (minimum image), so every unit stays whole. With
            ``'none'`` and ``'remove'`` the hydrogens are dropped and the order is kept;
            with ``'none'`` and ``'as_read'`` the atoms are unchanged. Every per-atom
            attribute follows the atoms; ``file_bonds`` is renumbered (bonds to removed
            atoms dropped); ``num_atoms`` and the per-unit and row radius, mass and volume
            are recomputed on rows that changed. Added (rows not processed, or failing,
            keep their atoms and get the neutral value):

            - ``h_source`` [N] int8: index into ``H_SOURCES``; a placed hydrogen has
              ``deposit_site`` -1, occupancy 1, its CCDC label, and ``deposit_frac`` its
              position in the deposited frame (parent's deposited coordinates plus the
              same vector), the reference ``check_rebuild`` expands; a CCDC hydrogen on a
              deposited hydrogen's coordinates is ``'deposited'`` and keeps that atom's
              values;
            - ``h_count_change`` [N] int8: on each heavy atom, the backend's hydrogen count
              minus the deposited one (0 elsewhere, and where not compared; kept on rows
              refused for it);
            - ``protonation_checked``, ``protonation_changed`` [B] bool: the comparison
              ran (``'ccdc'`` rows whose deposited set holds hydrogen, any mode but
              ``'remove'``), and some count differs;
            - ``has_chemistry`` [B] bool: the row holds ``'ccdc'`` chemistry;
            - ``charge_imbalance`` [B] bool: on a row with chemistry, some unit's formal
              charges do not sum to zero (each unit's net charge in the message). An ok row
              holds copies of one molecule (``find_molecules`` gives several kinds of
              molecule ``multi_component``), so a charged unit means a crystal that is not
              neutral: charges from SYBYL atom types on metals, a hydrogen CCDC added or
              dropped, or counter-ions missing from the file. Unlike
              ``protonation_changed`` it also covers files without hydrogens;
            - ``removed_deposit_site`` (long [k]) and ``removed_deposit_frac`` (float64
              [k, 3]): list[B], the deposited atoms the step removed (None where none), so
              the deposited set stays recoverable;
            - with ``'ccdc'`` only: ``mol_bonds`` replaced by CCDC's bonds (row-local, new
              order, hydrogens included); ``mol_bond_order`` list[B] of int8 [E] codes into
              ``BOND_ORDERS`` (RDKit's Kekule structure of CCDC's bonds; None on rows
              without chemistry); ``formal_charge`` [N] int8 (RDKit's, read from CCDC's
              perception; 0 on rows without chemistry); ``unit_smiles`` list[B] of per-unit
              lists of RDKit canonical SMILES (explicit hydrogens, stereo from the unit's
              deposited coordinates; None for a unit RDKit cannot read, None on rows
              without chemistry).

        Raises
        ------
        ValueError
            An argument outside its choices, ``chemistry='none'`` with a hydrogen choice
            that needs chemistry, ``chemistry='template'`` (not built), a posed batch, a
            batch already sanitized, an ok row that ``find_molecules`` has not processed,
            ``'ccdc'`` on a batch without ``source_path`` / ``block_name``, a per-atom
            attribute the step cannot fill for a placed hydrogen (``'regenerate'``,
            ``'add_missing'``), or not a batch from ``read`` / ``from_arrays``.

        Notes
        -----
        Status codes, the first that applies:

        ``backend_unavailable``
            ``'ccdc'``: the CSD Python API (or RDKit) cannot be imported or licensed.
        ``chemistry_failed``
            no source file (text or arrays), the block is not in the file, CCDC cannot read
            it or assign bonds, CCDC's heavy atoms cannot be mapped onto the row's (same
            element, fractional coordinates modulo the lattice, within 0.01 A, one to one),
            or processing raised a ``RuntimeError`` / ``ValueError``. On a row ``read``
            re-expressed from rhombohedral axes, CCDC reads the file on those axes (its cell
            is the row's hexagonal cell carried by the reader's P: ``_ccdc_on_rhombohedral_axes``),
            so the map, the hydrogen match and CCDC's parent-to-hydrogen vectors run there
            (row coordinates x_R = P x_H, modulo the rhombohedral lattice) and the vectors
            come back as x_H = P^-1 x_R.
        ``hydrogens_failed``
            ``add_hydrogens`` / ``remove_hydrogens`` raised; hydrogens left in CCDC's
            molecule with ``'remove'``; a hydrogen with no heavy-atom neighbour; deposited
            hydrogens missing from CCDC's molecule (``'as_read'``, ``'add_missing'``); or
            units whose matched heavy atoms carry different numbers of hydrogens (one atom
            order cannot hold them).
        ``component_mismatch``
            CCDC's components (``assign_bonds`` ignores file bonds) do not partition the
            heavy atoms as ``find_molecules``' units do.
        ``hydrogen_count_mismatch``
            ``on_protonation_change='refuse'`` and some heavy atom's count changed.

        Units RDKit cannot read keep CCDC's own bond types and formal charges (0) and get
        no SMILES; the row stays ok, and ``features`` rebuilds the unit from CCDC's bond types.
        A unit RDKit's full sanitization refuses there fails ``features`` unless it holds an
        element outside ``LEWIS_ELEMENTS`` and every problem lies near one (then the flagged
        partial-sanitization fallback; ``features``' Notes). Deposited hydrogen counts
        come from ``mol_bonds`` (``find_molecules``' bonds): a hydrogen's one bonded heavy
        atom, the nearest of several; an unbonded deposited hydrogen is not counted.
        """
        _check_batch(self)
        if chemistry == 'template':
            raise ValueError("chemistry='template' is not built; use 'ccdc' or 'none'")
        if chemistry not in _CHEMISTRIES:
            raise ValueError(f'chemistry must be one of {_CHEMISTRIES}, got {chemistry!r}')
        if hydrogens not in _HYDROGENS:
            raise ValueError(f'hydrogens must be one of {_HYDROGENS}, got {hydrogens!r}')
        if chemistry == 'none' and hydrogens not in ('as_read', 'remove'):
            raise ValueError(f"chemistry='none' takes hydrogens 'as_read' or 'remove' (placing hydrogens needs "
                             f'chemistry), got {hydrogens!r}')
        if on_protonation_change not in _ON_CHANGE:
            raise ValueError(f'on_protonation_change must be one of {_ON_CHANGE}, got {on_protonation_change!r}')
        posed = [k for k in POSE_KEYS[1:] if k in self._store]
        if posed:
            raise ValueError(f'sanitize runs before parameterize_pose; this batch is posed (holds {posed})')
        _refuse_second_pass(self, DONE_SANITIZE, _SANITIZE_KEYS, 'sanitize',
                            'its hydrogens are already the backend\'s, and a second pass would compare them with '
                            'themselves')
        rows = _ok_rows(self)
        _require_done(self, rows, DONE_FIND_MOLECULES, 'sanitize', 'find_molecules')
        needed = ('mol_ind', 'deposit_frac', 'deposit_site', 'mol_bonds', 'z_prime')
        missing = [k for k in needed if k not in self._store]
        if rows and missing:
            raise ValueError(f'sanitize needs find_molecules\' output; this batch lacks {missing}')
        if chemistry == 'ccdc' and rows:
            absent = [k for k in ('source_path', 'block_name') if k not in self._store]
            if absent:
                raise ValueError(f"chemistry='ccdc' reads each row's CIF block, but this batch lacks {absent} "
                                 '(a batch from from_arrays has no source file)')
        atom_keys, atom_lists = _atom_level_keys(self)
        if chemistry == 'ccdc' and hydrogens in ('regenerate', 'add_missing'):
            unfillable = [k for k in atom_keys if k not in _FILLABLE and k != 'batch'] + \
                         [k for k in atom_lists if k != 'atom_labels']
            if unfillable:
                raise ValueError(f'sanitize cannot fill per-atom attribute(s) {unfillable} for the hydrogens it places')

        B, ptr = self.num_graphs, self.ptr.detach().to('cpu').long()
        T_fc64, _ = _transforms64(self)
        rowdata = _RowData(self, ptr)
        backend, why = (None, '')
        if chemistry == 'ccdc' and rows:
            backend, why = _backends()

        def run(idx):
            # each row's units through the unit map (the one reader: unit_ind, or z_prime equal contiguous blocks
            # without one; a row whose units cannot be read raises here, and _isolate gives it chemistry_failed)
            lay = unit_layout(self, 'sanitize', rows=idx)
            units = lay.unit_ind.detach().to('cpu').long().numpy()
            per_unit = lay.atoms_per_unit.detach().to('cpu').long()
            out = {}
            for b in idx:
                try:
                    row = rowdata.row(b)
                    s, e = int(ptr[b]), int(ptr[b + 1])
                    row['units'], row['n_u'] = units[s:e], int(per_unit[b])
                    if chemistry == 'none':
                        out[b] = _none_row(row, hydrogens)
                    elif backend is None:
                        out[b] = _Refusal('backend_unavailable', why)
                    else:
                        out[b] = _ccdc_row(backend, row, T_fc64[b].numpy(), self['source_path'][b],
                                           self['block_name'][b], hydrogens, on_protonation_change)
                except _Refusal as r:
                    out[b] = r
                except (RuntimeError, ValueError) as e:     # a row's own failure; never another row's
                    out[b] = _Refusal('chemistry_failed', f'{type(e).__name__}: {str(e)[:300]}')
            return out

        results, failed = _isolate(run, rows)
        for b, msg in failed.items():
            results[b] = _Refusal('chemistry_failed', msg)

        # rows whose atoms change (with 'none' and 'as_read' none do: they keep every value exactly)
        changed = {b: r for b, r in results.items() if not isinstance(r, _Refusal) and not r.get('same', False)}
        out = _rebuild_atoms(self, changed, ptr, atom_keys, atom_lists)
        new_ptr = out.ptr.detach().to('cpu').long()
        device, dtype = self.pos.device, self.pos.dtype
        n_new = int(new_ptr[-1])

        h_source = torch.zeros(n_new, dtype=torch.int8)
        h_change = torch.zeros(n_new, dtype=torch.int8)
        charge = torch.zeros(n_new, dtype=torch.int8)
        checked = torch.zeros(B, dtype=torch.bool)
        prot_changed = torch.zeros(B, dtype=torch.bool)
        has_chem = torch.zeros(B, dtype=torch.bool)
        imbalance = torch.zeros(B, dtype=torch.bool)
        removed_site, removed_frac = [None] * B, [None] * B
        mol_bonds = list(self['mol_bonds']) if 'mol_bonds' in self._store else [None] * B
        bond_order, smiles = [None] * B, [None] * B
        status = out.status.detach().to('cpu').clone()
        message = list(out.message)
        for b, res in results.items():
            s = int(new_ptr[b])
            if isinstance(res, _Refusal):
                status[b], message[b] = _STATUS[res.code], str(res)
                if 'h_count_change' in res.info:            # refused for its protonation: the detail stays
                    h_change[s:s + len(res.info['h_count_change'])] = torch.as_tensor(res.info['h_count_change'])
                    checked[b] = prot_changed[b] = True
                continue
            n = len(res['src'])
            h_source[s:s + n] = torch.as_tensor(res['h_source'])
            h_change[s:s + n] = torch.as_tensor(res['h_count_change'])
            checked[b], prot_changed[b] = res['checked'], res['changed']
            mol_bonds[b] = res['bonds']
            if len(res['removed']):
                old = rowdata.row(b)
                removed_site[b] = old['dep_site'][res['removed']].astype(np.int64)
                removed_frac[b] = old['dep_frac'][res['removed']].copy()
            if chemistry == 'ccdc':
                has_chem[b] = True
                imbalance[b] = res['imbalance']
                charge[s:s + n] = torch.as_tensor(res['formal_charge'])
                bond_order[b] = res['bond_order']
                smiles[b] = res['smiles']
            if res['note']:
                message[b] = res['note'] if not message[b] else f"{message[b]}; {res['note']}"

        _set(out, 'h_source', h_source.to(device), 'atom')
        _set(out, 'h_count_change', h_change.to(device), 'atom')
        _set(out, 'protonation_checked', checked.to(device), 'row')
        _set(out, 'protonation_changed', prot_changed.to(device), 'row')
        _set(out, 'has_chemistry', has_chem.to(device), 'row')
        _set(out, 'charge_imbalance', imbalance.to(device), 'row')
        _set(out, 'removed_deposit_site', removed_site, 'row')
        _set(out, 'removed_deposit_frac', removed_frac, 'row')
        if 'mol_bonds' in self._store:
            _set(out, 'mol_bonds', mol_bonds, 'row')
        if chemistry == 'ccdc':
            _set(out, 'formal_charge', charge.to(device), 'atom')
            _set(out, 'mol_bond_order', bond_order, 'row')
            _set(out, 'unit_smiles', smiles, 'row')
        if changed and 'z_prime' in self._store:
            _update_unit_properties(out, self, set(changed), new_ptr, dtype, device)
        _finish(out, self, status, message, results, DONE_SANITIZE, 'sanitize',
                dict(chemistry=chemistry, hydrogens=hydrogens, on_protonation_change=on_protonation_change))
        return out

    def features(self, *, fingerprint: Optional[dict], charges: Optional[str]):
        """Per unit: an RDKit molecule from the sanitized chemistry; SMILES, Morgan fingerprint, Gasteiger charges.

        Runs on a posed batch (after ``parameterize_pose``; ``x`` exists from there) whose
        rows went through ``sanitize(chemistry='ccdc')``. Each unit's molecule is rebuilt
        from ``z``, ``formal_charge``, ``mol_bonds`` and ``mol_bond_order`` (every hydrogen
        an explicit atom, no implicit ones), sanitized by RDKit, with stereo from the
        unit's posed coordinates.

        Where RDKit's full sanitization refuses a unit that holds an element outside
        ``LEWIS_ELEMENTS`` (a metal, boron, silicon, ...) and every problem RDKit names lies
        near such an atom (an over-valent atom within 2 bonds, an aromatic system with no
        Kekule form holding one or bonded to one), the unit is sanitized again without the
        valence check (``UpdatePropertyCache(strict=False)``, then every other step) and, if
        that is refused too, also without kekulization and aromaticity perception, keeping
        the stored aromatic flags (RDKit's partial sanitization). These are the units
        ``sanitize`` could not read into RDKit, which keep CCDC's own bond types and formal
        charges (0): coordination, cluster and multicentre bonds (carbonyl and isocyanide
        ligands, bridging hydrides and halides, carboranes, rings bound to a metal) that
        RDKit's valence and Kekule models do not hold. The row says so in
        ``features_fallback`` and its message.

        Parameters
        ----------
        fingerprint : dict or None
            ``dict(radius=int >= 0, n_bits=int, a positive multiple of 8, chirality=bool)``
            for RDKit's Morgan generator (``radius``, ``fpSize``, ``includeChirality``), or
            None for no fingerprint. ``OLD_FINGERPRINT`` is the old featurizer's
            (radius 2, 2048 bits, no chirality).
        charges : {'gasteiger'} or None
            Gasteiger charges (``ComputeGasteigerCharges``) into ``x``, or None to leave
            ``x`` as it is.

        Returns
        -------
        MolCrystalDataBatch
            A new batch. On rows that pass: ``unit_smiles`` filled where sanitize stored
            none (RDKit canonical SMILES of the rebuilt molecule); ``unit_fingerprint``
            [B, Z'max, n_bits / 8] uint8, each unit's bits packed (``np.packbits``, first
            bit in the high bit of byte 0; zeros on rows not featurized), with
            ``fingerprint``; ``x`` [N] each atom's charge (batch dtype) and
            ``charges_nonfinite`` [B] bool, with ``charges``: a non-finite charge is kept
            as NaN and flags its row, never set to zero; ``features_fallback`` [B] int8,
            an index into ``FEATURES_FALLBACK``: how far the sanitization of the row's
            units fell back (0 on rows sanitized in full and on rows not featurized), the
            units and RDKit's refusal of the full sanitization named in the message (for a
            ``'kekule'`` row also its refusal of the sanitization without the valence
            check, which names the atoms it could not kekulize). A unit featurized by the
            fallback gets no SMILES here (RDKit could not read one back), so its
            ``unit_smiles`` entry stays None.

        Raises
        ------
        ValueError
            An argument of the wrong form, a batch without a pose, a batch already
            featurized, or not a batch from ``read`` / ``from_arrays``.

        Notes
        -----
        Status ``features_failed``: a row without chemistry (``sanitize`` not run with
        ``'ccdc'``); a unit RDKit's full sanitization refuses whose atoms are all in
        ``LEWIS_ELEMENTS``; such a unit holding an element outside them where a problem
        RDKit names (``Chem.DetectChemistryProblems``, every unkekulizable system in turn)
        lies beyond the fallback's reach: an over-valent (or otherwise refused) atom more
        than 2 bonds from every atom outside ``LEWIS_ELEMENTS``, or an aromatic system with
        no Kekule form none of whose atoms is within 1 bond of one (a missing charge or a
        ring called aromatic that has no Kekule form, as in a unit of ``LEWIS_ELEMENTS``
        alone); a unit the fallback refuses too (or whose stored aromatic bonds are not all
        in rings); or processing raised a ``RuntimeError`` / ``ValueError``. The message
        says which.

        SMILES stored by ``sanitize`` carry the stereo of the deposited unit; a unit whose
        posed image comes from an improper operator is its mirror image, so a SMILES
        computed here, or a fingerprint with ``chirality=True``, carries the opposite
        stereo for a chiral unit.

        ``canonicalize_zp_aunits`` (``crystal_ops``) moves the atom blocks of a Z' > 1 row
        but not yet ``unit_fingerprint`` or ``unit_smiles``: after it, slot ``j``'s values
        can belong to another unit (``tests/test_crystal_chem.py`` pins this as a strict
        xfail).
        """
        _check_batch(self)
        fp = _fingerprint_args(fingerprint)
        if charges not in (None, 'gasteiger'):
            raise ValueError(f"charges must be 'gasteiger' or None, got {charges!r}")
        missing = [k for k in POSE_KEYS if k not in self._store]
        if missing or 'x' not in self._store:
            raise ValueError(f'features needs a posed batch (after parameterize_pose); this one lacks '
                             f'{missing or ["x"]}')
        _refuse_second_pass(self, DONE_FEATURES, _FEATURE_KEYS, 'features',
                            'its charges are already in x')
        rows = _ok_rows(self)
        B, ptr = self.num_graphs, self.ptr.detach().to('cpu').long()
        zp_all = self['z_prime'].detach().to('cpu').long()
        zmax = max(int(zp_all.max()) if B else 1, 1)
        has_chem = self['has_chemistry'].detach().to('cpu') if 'has_chemistry' in self._store else \
            torch.zeros(B, dtype=torch.bool)
        chem_keys = all(k in self._store for k in ('formal_charge', 'mol_bond_order', 'mol_bonds'))
        z_all = self.z.detach().to('cpu').long().numpy()
        pos_all = self.pos.detach().to('cpu', _F64).numpy()
        q_all = self['formal_charge'].detach().to('cpu').long().numpy() if chem_keys else None
        stored = self['unit_smiles'] if 'unit_smiles' in self._store else [None] * B
        generator = _morgan(fp) if fp is not None else None

        def run(idx):
            # each row's atoms per unit through the unit map (unit_layout; n // z_prime before stage 3, 2026-10-23); a
            # row whose units cannot be read raises here, and _isolate gives it features_failed
            per_unit = unit_layout(self, 'features', rows=idx).atoms_per_unit.detach().to('cpu').long()
            out = {}
            for b in idx:
                try:
                    if not bool(has_chem[b]) or not chem_keys:
                        raise _Refusal('features_failed', "the row has no chemistry: sanitize(chemistry='ccdc') "
                                                          'has not given it any')
                    s, e = int(ptr[b]), int(ptr[b + 1])
                    out[b] = _features_row(z_all[s:e], q_all[s:e], pos_all[s:e], self['mol_bonds'][b],
                                           self['mol_bond_order'][b], int(zp_all[b]), stored[b], generator,
                                           charges, n_u=int(per_unit[b]))
                except _Refusal as r:
                    out[b] = r
                except (RuntimeError, ValueError) as e:
                    out[b] = _Refusal('features_failed', f'{type(e).__name__}: {str(e)[:300]}')
            return out

        results, failed = _isolate(run, rows)
        for b, msg in failed.items():
            results[b] = _Refusal('features_failed', msg)

        out = _copy_batch(self)
        device, dtype = self.pos.device, self.pos.dtype
        status = out.status.detach().to('cpu').clone()
        message = list(out.message)
        smiles = [None if v is None else list(v) for v in stored]
        n_bytes = fp['n_bits'] // 8 if fp is not None else 0
        bits = torch.zeros(B, zmax, n_bytes, dtype=torch.uint8)
        nonfinite = torch.zeros(B, dtype=torch.bool)
        fallback = torch.zeros(B, dtype=torch.int8)
        x = self.x.detach().to('cpu').clone()
        for b, res in results.items():
            if isinstance(res, _Refusal):
                status[b], message[b] = _STATUS[res.code], str(res)
                continue
            smiles[b] = res['smiles']
            if res['fallback']:
                fallback[b] = max(res['fallback'].values())
                units = ', '.join(f'unit {u} {FEATURES_FALLBACK[t]!r}' for u, t in sorted(res['fallback'].items()))
                note = (f'features_fallback: {units} (RDKit refuses the full sanitization of unit '
                        f"{min(res['fallback'])}: {res['fallback_reason']}")
                if res['fallback_tier1'] is not None:
                    note += (f"; and the sanitization of unit {res['fallback_tier1'][0]} without the valence check: "
                             f"{res['fallback_tier1'][1]}")
                note += '); no SMILES for them'
                message[b] = note if not message[b] else f'{message[b]}; {note}'
            if fp is not None:
                bits[b, :len(res['bits'])] = torch.as_tensor(np.stack(res['bits']))
            if charges is not None:
                s = int(ptr[b])
                x[s:s + len(res['charges'])] = torch.as_tensor(res['charges']).to(x.dtype)
                nonfinite[b] = res['nonfinite']
                if res['nonfinite']:
                    note = f"charges_nonfinite: {res['n_nonfinite']} atom charge(s) are not finite (kept as NaN)"
                    message[b] = note if not message[b] else f'{message[b]}; {note}'
        _set(out, 'unit_smiles', smiles, 'row')
        if fp is not None:
            _set(out, 'unit_fingerprint', bits.to(device), 'row')
        if charges is not None:
            out['x'] = x.to(dtype).to(device)
            _set(out, 'charges_nonfinite', nonfinite.to(device), 'row')
        _set(out, 'features_fallback', fallback.to(device), 'row')
        _finish(out, self, status, message, results, DONE_FEATURES, 'features',
                dict(fingerprint=None if fp is None else dict(fp), charges=charges))
        return out


###############################################################################
# shared machinery
###############################################################################

def _check_batch(batch):
    """Raise ValueError unless ``batch`` is a batch with the ingestion bookkeeping and codes in ``STATUS``."""
    if not getattr(batch, 'is_batch', False):
        raise ValueError(f'the steps take a batch from MolCrystalData.read / from_arrays, not a {type(batch).__name__}')
    missing = [k for k in ('ptr', 'batch', 'status', 'message', 'done', 'z', 'pos', 'cell_lengths', 'cell_angles',
                           'sg_ind', 'symmetry_operators') if k not in batch._store]
    if missing or '_slice_dict' not in batch.__dict__:
        raise ValueError(f'the steps take a batch from MolCrystalData.read / from_arrays; this one lacks '
                         f'{missing or ["_slice_dict"]}')
    bad = sorted({int(s) for s in batch.status.detach().to('cpu').tolist() if not 0 <= int(s) < len(STATUS)})
    if bad:
        raise ValueError(f'status codes {bad} are outside STATUS')


def _backends():
    """``(ccdc.io, '')`` when the CSD Python API and RDKit import (the API checks its licence on import),
    else ``(None, reason)``."""
    try:
        from ccdc import io as ccdc_io
    except Exception as e:      # ImportError, or the API's own licence error: either way no backend
        return None, f'the CSD Python API cannot be imported or licensed: {type(e).__name__}: {str(e)[:200]}'
    try:
        import rdkit  # noqa: F401
    except ImportError as e:
        return None, f'RDKit cannot be imported: {e}'
    return ccdc_io, ''


def _atom_level_keys(batch):
    """``(tensor keys, list keys)`` stored per atom: tensors sliced by ``ptr`` with N rows, and row-level
    lists whose entries are lists of the row's atom count (``atom_labels``)."""
    ptr = batch.ptr.detach().to('cpu').long()
    B, N = len(ptr) - 1, int(ptr[-1])
    row_slices = torch.arange(B + 1)
    counts = (ptr[1:] - ptr[:-1]).tolist()
    ambiguous = torch.equal(ptr, row_slices)
    tensors, lists = [], []
    for key, value in batch._store.items():
        slices = batch._slice_dict.get(key)
        if key in ('ptr',) or slices is None:
            continue
        slices = slices.to('cpu')
        if torch.is_tensor(value) and torch.equal(slices, ptr) and value.shape[:1] == (N,):
            if ambiguous and key not in _KNOWN_ATOM_KEYS and key != 'batch':
                continue                        # one atom per row: only known names are per atom
            tensors.append(key)
        elif isinstance(value, list) and torch.equal(slices, row_slices) and key != 'file_bonds' and not ambiguous:
            if all(isinstance(v, (list, tuple)) and len(v) == counts[b] for b, v in enumerate(value)) and \
                    any(c > 1 for c in counts):
                lists.append(key)
    if ambiguous and 'atom_labels' in batch._store:
        lists.append('atom_labels')
    return tensors, lists


class _RowData:
    """Host copies of the per-atom inputs, sliced per row on request."""

    def __init__(self, batch, ptr):
        self.ptr = ptr
        self.z = batch.z.detach().to('cpu').long().numpy()
        self.pos = batch.pos.detach().to('cpu', _F64).numpy()
        self.dep_frac = batch['deposit_frac'].detach().to('cpu', _F64).numpy() if 'deposit_frac' in batch._store \
            else None
        self.dep_site = batch['deposit_site'].detach().to('cpu').long().numpy() if 'deposit_site' in batch._store \
            else None
        self.mol_ind = batch['mol_ind'].detach().to('cpu').long().numpy() if 'mol_ind' in batch._store else None
        self.zp = batch['z_prime'].detach().to('cpu').long().numpy() if 'z_prime' in batch._store else None
        self.bonds = batch['mol_bonds'] if 'mol_bonds' in batch._store else None
        self.labels = batch['atom_labels'] if 'atom_labels' in batch._store else None

    def row(self, b) -> dict:
        s, e = int(self.ptr[b]), int(self.ptr[b + 1])
        n = e - s
        return dict(z=self.z[s:e], pos=self.pos[s:e], dep_frac=self.dep_frac[s:e], dep_site=self.dep_site[s:e],
                    mol_ind=self.mol_ind[s:e], zp=int(self.zp[b]),
                    bonds=np.asarray(self.bonds[b], dtype=np.int64).reshape(2, -1),
                    labels=list(self.labels[b]) if self.labels is not None else [''] * n)


def _rebuild_atoms(inp, changed, ptr, atom_keys, atom_lists):
    """A copy of ``inp`` whose rows in ``changed`` hold their new atoms; every other row bit-identical.

    ``changed[b]`` gives ``src`` (per new atom its row-local index in ``inp``, -1 for a
    placed atom), ``mol_ind`` (every new atom's unit, written to ``mol_ind`` and to the unit
    map ``unit_ind``), ``fill`` (per fillable key, values for the
    placed atoms in order), ``pos`` (float64, every new atom) with ``pos_keep`` (True where
    the old coordinates are kept exactly), ``labels`` (every new atom). Per-atom slice
    entries, ``ptr``, ``batch``, ``num_atoms`` and ``file_bonds`` follow.
    """
    out = _copy_batch(inp)
    if not changed:
        return out
    B, N = len(ptr) - 1, int(ptr[-1])
    counts = (ptr[1:] - ptr[:-1]).clone()
    gsrc_parts = []
    for b in range(B):
        if b in changed:
            src = np.asarray(changed[b]['src'], dtype=np.int64)
            counts[b] = len(src)
            gsrc_parts.append(np.where(src >= 0, src + int(ptr[b]), -1))
        else:
            gsrc_parts.append(np.arange(int(ptr[b]), int(ptr[b + 1])))
    new_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    gsrc = torch.as_tensor(np.concatenate(gsrc_parts) if gsrc_parts else np.zeros(0, np.int64), dtype=torch.long)
    take = gsrc.clamp(min=0)
    device = inp.pos.device
    for key in atom_keys:
        if key == 'batch':
            continue
        value = inp[key]
        new = value[take.to(value.device)] if N else value.new_zeros((len(gsrc),) + tuple(value.shape[1:]))
        for b, res in changed.items():
            s = int(new_ptr[b])
            n = len(res['src'])
            local_placed = np.asarray(res['src']) < 0
            if key in ('mol_ind', UNIT_MAP_KEY):    # every atom of the row: its unit (the map since stage 3)
                new[s:s + n] = torch.as_tensor(res['mol_ind'], dtype=new.dtype).to(new.device)
            elif key == 'pos':
                fresh = ~np.asarray(res['pos_keep'])
                if fresh.any():
                    idx = torch.as_tensor(np.nonzero(fresh)[0] + s)
                    new[idx.to(new.device)] = torch.as_tensor(res['pos'][fresh]).to(new.dtype).to(new.device)
            elif local_placed.any():
                idx = torch.as_tensor(np.nonzero(local_placed)[0] + s)
                new[idx.to(new.device)] = torch.as_tensor(res['fill'][key]).to(new.dtype).to(new.device)
        out[key] = new
        out._slice_dict[key] = new_ptr.clone()
    for key in atom_lists:
        value = list(out[key])
        for b, res in changed.items():
            if key == 'atom_labels':
                value[b] = type(inp[key][b])(res['labels'])
            else:
                old = inp[key][b]
                value[b] = type(old)(old[i] for i in res['src'])
        out[key] = value
    if 'file_bonds' in out._store:
        fb = list(out['file_bonds'])
        for b, res in changed.items():
            if fb[b] is None:
                continue
            n_old = int(ptr[b + 1] - ptr[b])
            new_index = np.full(n_old, -1, dtype=np.int64)
            src = np.asarray(res['src'])
            kept = src >= 0
            new_index[src[kept]] = np.nonzero(kept)[0]
            e = new_index[np.asarray(fb[b], dtype=np.int64).reshape(2, -1)]
            fb[b] = e[:, (e >= 0).all(0)].copy()
        out['file_bonds'] = fb
    out['ptr'] = new_ptr.to(device)
    out['batch'] = torch.arange(B).repeat_interleave(counts).to(device)
    if 'num_atoms' in out._store:
        out['num_atoms'] = counts.to(inp['num_atoms'].dtype).to(inp['num_atoms'].device)
    return out


def _update_unit_properties(out, inp, changed_rows, new_ptr, dtype, device):
    """Per-unit and row radius / mass / volume recomputed (``_write_unit_properties``) on the changed rows only."""
    keys = ('unit_radius', 'unit_mass', 'unit_volume', 'radius', 'mass', 'mol_volume')
    previous = {k: inp[k] for k in keys if k in inp._store}
    z_prime = out['z_prime'].detach().to('cpu').long()
    mol_ind = out['mol_ind'].detach().to('cpu').long()
    _write_unit_properties(out, new_ptr, z_prime, mol_ind, dtype, device)
    keep = torch.tensor([b not in changed_rows for b in range(len(z_prime))], dtype=torch.bool)
    for k, old in previous.items():
        new = out[k]
        if new.shape == old.shape:
            new[keep.to(new.device)] = old[keep.to(old.device)].to(new.dtype)
        out[k] = new


def _min_image(d):
    """Fractional differences reduced to [-0.5, 0.5]."""
    return d - np.round(d)


###############################################################################
# sanitize: backend 'none'
###############################################################################

def _none_row(row, hydrogens) -> dict:
    """``chemistry='none'``: the deposited hydrogens kept (``'as_read'``) or dropped (``'remove'``)."""
    z, n = row['z'], len(row['z'])
    keep = np.arange(n) if hydrogens == 'as_read' else np.nonzero(z > 1)[0]
    new_index = np.full(n, -1, dtype=np.int64)
    new_index[keep] = np.arange(len(keep))
    e = new_index[row['bonds']]
    bonds = e[:, (e >= 0).all(0)].copy()
    removed = np.setdiff1d(np.arange(n), keep)
    note = f'{len(removed)} deposited hydrogen(s) removed' if len(removed) else ''
    return dict(src=keep, mol_ind=row['units'][keep], fill={}, pos=row['pos'][keep],
                pos_keep=np.ones(len(keep), dtype=bool), labels=[row['labels'][i] for i in keep],
                h_source=np.zeros(len(keep), dtype=np.int8), h_count_change=np.zeros(len(keep), dtype=np.int8),
                checked=False, changed=False, bonds=bonds, removed=removed, same=len(keep) == n, note=note)


###############################################################################
# sanitize: backend 'ccdc'
###############################################################################

def _ccdc_crystal(ccdc_io, path, block):
    """The CCDC crystal of ``block`` in ``path``; raises ``_Refusal('chemistry_failed')``."""
    if path is None or block is None:
        raise _Refusal('chemistry_failed', 'the row has no source file for CCDC to read (read from CIF text, or a '
                                           'block that did not parse)')
    if not os.path.isfile(path):
        raise _Refusal('chemistry_failed', f'the source file {path!r} is not there any more')
    try:
        reader = ccdc_io.CrystalReader(path, format='cif')
    except (RuntimeError, ValueError, OSError) as e:
        raise _Refusal('chemistry_failed', f'CCDC cannot open {path!r}: {type(e).__name__}: {str(e)[:200]}')
    try:
        names = []
        for i in range(len(reader)):            # bounded by the file's blocks
            crystal = reader[i]
            names.append(crystal.identifier)
            if str(crystal.identifier).strip().lower() == str(block).strip().lower():
                return crystal
    except (RuntimeError, ValueError) as e:
        raise _Refusal('chemistry_failed', f'CCDC cannot read {path!r}: {type(e).__name__}: {str(e)[:200]}')
    finally:
        close = getattr(reader, 'close', None)
        if close is not None:
            close()
    raise _Refusal('chemistry_failed', f'block {block!r} is not among the {len(names)} block(s) CCDC reads in {path!r}')


def _ccdc_on_rhombohedral_axes(crystal, T_fc) -> bool:
    """True when CCDC reads the row's file on the rhombohedral axes the reader re-expressed it from.

    ``read`` re-expresses an R group filed on rhombohedral axes on hexagonal axes, obverse
    (``cif/rhombohedral.py``: a_H = a_R - b_R, b_H = b_R - c_R, c_H = a_R + b_R + c_R, i.e.
    x_R = P x_H); CCDC reads the file as written. Decided from the cells alone: the row's
    metric (from ``T_fc``) is hexagonal (a = b, alpha = beta = 90, gamma = 120 degrees, to
    ``_RH_METRIC_TOL``) and equals P^T G P, G the metric of CCDC's cell, which makes CCDC's
    cell rhombohedral (a = b = c, alpha = beta = gamma) with three times smaller volume.
    CCDC's cell is read only on a hexagonal row; a cell it cannot give means False.
    """
    g_row = T_fc.T @ T_fc
    scale = float(max(g_row[0, 0], g_row[1, 1], g_row[2, 2]))
    tol = _RH_METRIC_TOL * scale
    hexagonal = (abs(g_row[0, 0] - g_row[1, 1]) <= tol and abs(g_row[0, 2]) <= tol and abs(g_row[1, 2]) <= tol
                 and abs(2 * g_row[0, 1] + g_row[0, 0]) <= tol)
    if not hexagonal:
        return False
    try:
        lengths = [float(v) for v in crystal.cell_lengths]
        angles = [float(v) for v in crystal.cell_angles]
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return False
    if len(lengths) != 3 or len(angles) != 3 or not np.isfinite(lengths + angles).all():
        return False
    g_ccdc = _rh_metric(*lengths, *angles)
    return bool(np.abs(g_row - _RH_P.T @ g_ccdc @ _RH_P).max() <= tol)


def _rdkit_component(comp, cc_z, cc_xyz):
    """``(mol, None)`` with ``mol`` the component read by RDKit from CCDC's mol2 block (RDKit atom order =
    CCDC's), or ``(None, reason)``."""
    from rdkit import Chem, rdBase

    block = rdBase.BlockLogs()                  # RDKit logs stay quiet while it lives
    try:
        text = comp.to_string('mol2')
        mol = Chem.MolFromMol2Block(text, sanitize=True, removeHs=False)
        if mol is None:
            reason = "RDKit's mol2 reader refuses CCDC's perception"
            raw = Chem.MolFromMol2Block(text, sanitize=False, removeHs=False)
            if raw is not None:
                try:
                    Chem.SanitizeMol(raw)
                except Exception as e:          # only to name the reason; the outcome is the refusal above
                    reason = f'{reason}: {str(e)[:120]}'
            return None, reason
        if [a.GetAtomicNum() for a in mol.GetAtoms()] != list(cc_z):
            return None, "RDKit's atom order differs from CCDC's"
        xyz = mol.GetConformer().GetPositions()
        if len(xyz) and float(np.abs(xyz - cc_xyz).max()) > _MOL2_TOL:
            return None, "RDKit's coordinates differ from CCDC's"
        return mol, None
    finally:
        del block


def _kekule_chemistry(mol):
    """``(bonds [k, 2] local, codes [k], charges [n])`` of an RDKit molecule's Kekule structure; ValueError
    for a bond type outside ``BOND_ORDERS``."""
    from rdkit import Chem

    k = Chem.Mol(mol)
    Chem.Kekulize(k, clearAromaticFlags=True)
    types = {Chem.BondType.UNSPECIFIED: 'unspecified', Chem.BondType.SINGLE: 'single',
             Chem.BondType.DOUBLE: 'double', Chem.BondType.TRIPLE: 'triple', Chem.BondType.QUADRUPLE: 'quadruple',
             Chem.BondType.DATIVE: 'dative'}
    bonds, codes = [], []
    for bond in k.GetBonds():
        name = types.get(bond.GetBondType())
        if name is None:
            raise ValueError(f'RDKit bond type {bond.GetBondType()} has no code in BOND_ORDERS')
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        codes.append(_BOND[name])
    charges = [a.GetFormalCharge() for a in k.GetAtoms()]
    return np.array(bonds, dtype=np.int64).reshape(-1, 2), np.array(codes, dtype=np.int8), \
        np.array(charges, dtype=np.int64)


def _ccdc_row(ccdc_io, row, T_fc, path, block, hydrogens, on_change) -> dict:
    """One row through CCDC; raises ``_Refusal`` (see ``sanitize``)."""
    crystal = _ccdc_crystal(ccdc_io, path, block)
    try:
        crystal.assign_bonds()
    except (RuntimeError, ValueError) as e:
        raise _Refusal('chemistry_failed', f'CCDC assign_bonds: {type(e).__name__}: {str(e)[:200]}')
    try:
        if hydrogens == 'regenerate':
            crystal.add_hydrogens(mode='all')
        elif hydrogens == 'add_missing':
            crystal.add_hydrogens(mode='missing')
        elif hydrogens == 'remove':
            crystal.remove_hydrogens()
    except (RuntimeError, ValueError) as e:
        raise _Refusal('hydrogens_failed', f'CCDC {hydrogens} hydrogens: {type(e).__name__}: {str(e)[:200]}')
    try:
        molecule = crystal.molecule
        comps = list(molecule.components)
    except (RuntimeError, ValueError) as e:
        raise _Refusal('chemistry_failed', f"CCDC's molecule: {type(e).__name__}: {str(e)[:200]}")

    # the row's units as the unit map gives them (unit_layout, in sanitize): each atom's unit, and the atoms per unit
    # (equal contiguous blocks at this stage); they were mol_ind and n // z_prime before stage 3 (2026-10-23)
    z, pos, dep_frac, mol_ind = row['z'], row['pos'], row['dep_frac'], row['units']
    n, zp = len(z), row['zp']
    n_u = row['n_u']

    # every CCDC atom, component by component, with its chemistry (RDKit's reading where it reads)
    cc_z, cc_frac, cc_label, cc_comp, cc_charge = [], [], [], [], []
    cc_bonds, cc_codes = [], []
    unreadable = {}
    offset = 0
    for ci, comp in enumerate(comps):
        atoms = list(comp.atoms)
        zc = np.array([a.atomic_number for a in atoms], dtype=np.int64)
        if any(a.fractional_coordinates is None or a.coordinates is None for a in atoms):
            raise _Refusal('chemistry_failed', f'CCDC component {ci} holds atoms without coordinates')
        fc = np.array([tuple(a.fractional_coordinates) for a in atoms], dtype=np.float64).reshape(-1, 3)
        xyz = np.array([tuple(a.coordinates) for a in atoms], dtype=np.float64).reshape(-1, 3)
        if not (np.isfinite(fc).all() and np.isfinite(xyz).all()):
            raise _Refusal('chemistry_failed', f'CCDC component {ci} holds atoms without finite coordinates')
        mol, reason = _rdkit_component(comp, zc, xyz)
        if mol is not None:
            try:
                bonds, codes, charges = _kekule_chemistry(mol)
            except ValueError as e:
                mol, reason = None, str(e)
        if mol is None:
            unreadable[ci] = reason
            index = {a.index: i for i, a in enumerate(atoms)}
            bonds = np.array([(index[b.atoms[0].index], index[b.atoms[1].index]) for b in comp.bonds],
                             dtype=np.int64).reshape(-1, 2)
            codes = np.array([_BOND[_CCDC_BOND.get(str(b.bond_type), 'unspecified')] for b in comp.bonds],
                             dtype=np.int8)
            charges = np.array([int(a.formal_charge) for a in atoms], dtype=np.int64)
        cc_z.append(zc)
        cc_frac.append(fc)
        cc_label += [a.label for a in atoms]
        cc_comp.append(np.full(len(atoms), ci, dtype=np.int64))
        cc_charge.append(charges)
        cc_bonds.append(bonds + offset)
        cc_codes.append(codes)
        offset += len(atoms)
    if not comps:
        raise _Refusal('chemistry_failed', "CCDC's molecule holds no atoms")
    cc_z, cc_frac, cc_comp, cc_charge = (np.concatenate(v) for v in (cc_z, cc_frac, cc_comp, cc_charge))
    cc_bonds, cc_codes = np.concatenate(cc_bonds), np.concatenate(cc_codes)

    # CCDC's frame: the row's, or for a row read from rhombohedral axes the file's own axes, where the comparisons
    # run (the row's deposited coordinates carried there, x_R = P x_H; modulo the rhombohedral lattice, the
    # crystal's, on which the hexagonal cell's centring translations are integer) and from which CCDC's
    # parent-to-hydrogen vectors come back (x_H = P^-1 x_R)
    rhombohedral = _ccdc_on_rhombohedral_axes(crystal, T_fc)
    if rhombohedral:
        dep_cc, T_cc = dep_frac @ _RH_P.T, T_fc @ _RH_P_INV3 / 3
    else:
        dep_cc, T_cc = dep_frac, T_fc

    # heavy atoms: CCDC's onto the row's deposited ones
    cc_heavy = np.nonzero(cc_z > 1)[0]
    heavy = np.nonzero(z > 1)[0]
    heavy_of = _map_sites(cc_z[cc_heavy], cc_frac[cc_heavy], z[heavy], dep_cc[heavy], T_cc, 'heavy atom')
    our = np.full(len(cc_z), -1, dtype=np.int64)
    our[cc_heavy] = heavy[heavy_of]

    # components against units
    unit_of_comp = {}
    for ci in range(len(comps)):
        members = our[(cc_comp == ci) & (cc_z > 1)]
        units = sorted(set(mol_ind[members].tolist()))
        if len(units) > 1:
            raise _Refusal('component_mismatch', f'CCDC component {ci} spans units {units}')
        if units:
            unit_of_comp[ci] = units[0]
    covered = sorted(unit_of_comp.values())
    if covered != list(range(zp)):
        raise _Refusal('component_mismatch', f"CCDC's {len(comps)} components map onto units {covered}; the row has "
                                             f'{zp} unit(s)')

    # hydrogens: parent, position, source
    adj = [[] for _ in range(len(cc_z))]
    for i, j in cc_bonds.tolist():
        adj[i].append(j)
        adj[j].append(i)
    cc_H = np.nonzero(cc_z == 1)[0]
    if hydrogens == 'remove' and len(cc_H):         # CCDC's remove_hydrogens keeps a hydrogen bonded to two atoms
        h0 = int(cc_H[0])
        raise _Refusal('hydrogens_failed', f"CCDC's remove_hydrogens left {len(cc_H)} hydrogen(s), e.g. "
                                           f'{cc_label[h0]} bonded to {[cc_label[j] for j in adj[h0]]}, with '
                                           "hydrogens='remove'")
    parent = np.full(len(cc_z), -1, dtype=np.int64)
    for h in cc_H.tolist():
        nb = [j for j in adj[h] if cc_z[j] > 1]
        if not nb:
            raise _Refusal('hydrogens_failed', f"hydrogen {cc_label[h]} has no heavy-atom neighbour in CCDC's "
                                               'molecule')
        if len(nb) > 1:
            d = [np.linalg.norm(_min_image(cc_frac[h] - cc_frac[j]) @ T_cc.T) for j in nb]
            nb = [nb[int(np.argmin(d))]]
        parent[h] = nb[0]
    h_vec = _min_image(cc_frac[cc_H] - cc_frac[parent[cc_H]]) if len(cc_H) else np.zeros((0, 3))
    h_parent = our[parent[cc_H]]                         # row-local heavy atom of each CCDC hydrogen
    h_pos = pos[h_parent] + h_vec @ T_cc.T
    h_dep = dep_frac[h_parent] + (h_vec @ _RH_P_INV3.T / 3 if rhombohedral else h_vec)
    dep_H = np.nonzero(z == 1)[0]
    # the deposited hydrogen each CCDC hydrogen is, or -1; in every mode, as add_hydrogens(mode='all') keeps a
    # hydrogen bonded to two atoms on its deposited site
    h_src = np.full(len(cc_H), -1, dtype=np.int64)
    if len(cc_H) and len(dep_H):
        h_src = _match_hydrogens(cc_frac[cc_H], dep_cc[dep_H], T_cc, dep_H)
    if hydrogens == 'as_read' and (h_src < 0).any():
        raise _Refusal('chemistry_failed', f'{int((h_src < 0).sum())} hydrogen(s) of CCDC\'s molecule are not '
                                           "deposited hydrogens, with hydrogens='as_read'")
    if hydrogens in ('as_read', 'add_missing'):
        lost = np.setdiff1d(dep_H, h_src[h_src >= 0])
        if len(lost):
            raise _Refusal('hydrogens_failed', f"{len(lost)} deposited hydrogen(s) are missing from CCDC's molecule, "
                                               f"e.g. {row['labels'][int(lost[0])]}")

    # protonation: hydrogens per heavy atom, the backend's against the deposited ones
    checked = hydrogens != 'remove' and len(dep_H) > 0
    change = np.zeros(n, dtype=np.int64)
    note_parts = []
    if checked:
        dep_parent, orphans = _deposited_parents(z, pos, row['bonds'])
        dep_count = np.bincount(dep_parent[dep_parent >= 0], minlength=n)
        new_count = np.bincount(h_parent, minlength=n) if len(h_parent) else np.zeros(n, dtype=np.int64)
        change = (new_count - dep_count) * (z > 1)
        if orphans:
            note_parts.append(f'{orphans} deposited hydrogen(s) bonded to no heavy atom are not counted')
    changed_atoms = np.nonzero(change)[0]
    if len(changed_atoms):
        detail = ', '.join(f"{row['labels'][i]} {change[i]:+d}" for i in changed_atoms[:8].tolist())
        more = f' and {len(changed_atoms) - 8} more' if len(changed_atoms) > 8 else ''
        text = f'protonation_changed: hydrogens per heavy atom differ from the deposited set ({detail}{more})'
        if on_change == 'refuse':
            raise _Refusal('hydrogen_count_mismatch', text, h_count_change=change.astype(np.int8))
        note_parts.append(text)

    # one atom order: per unit, its heavy atoms in order, then its hydrogens by parent
    order_H = _order_hydrogens(h_parent, h_pos, mol_ind, z, pos, zp, n_u, row['labels'])
    cc_of_row = np.full(n, -1, dtype=np.int64)
    cc_of_row[our[cc_heavy]] = cc_heavy
    src, new_mol_ind, new_pos, pos_keep, labels, h_source, cc_of_new = [], [], [], [], [], [], []
    for u in range(zp):                                 # bounded by Z'
        block = np.arange(u * n_u, (u + 1) * n_u)
        for i in block[z[block] > 1].tolist():
            src.append(i)
            new_mol_ind.append(u)
            new_pos.append(pos[i])
            pos_keep.append(True)
            labels.append(row['labels'][i])
            h_source.append(0)
            cc_of_new.append(int(cc_of_row[i]))
        for k in order_H[u]:
            s = int(h_src[k])
            same_place = s >= 0 and float(np.linalg.norm(pos[s] - h_pos[k])) < _MAP_TOL
            src.append(s)
            new_mol_ind.append(u)
            new_pos.append(pos[s] if same_place else h_pos[k])
            pos_keep.append(bool(same_place))
            labels.append(row['labels'][s] if s >= 0 else cc_label[cc_H[k]])
            h_source.append(0 if s >= 0 else 1)
            cc_of_new.append(int(cc_H[k]))
    src = np.array(src, dtype=np.int64)
    cc_of_new = np.array(cc_of_new, dtype=np.int64)
    new_index = np.empty(len(cc_z), dtype=np.int64)
    new_index[cc_of_new] = np.arange(len(cc_of_new))
    placed = src < 0
    placed_k = [k for u in range(zp) for k in order_H[u] if h_src[k] < 0]
    fill = dict(z=np.ones(int(placed.sum()), dtype=np.int64),
                deposit_frac=h_dep[placed_k].reshape(-1, 3),
                deposit_site=np.full(int(placed.sum()), -1, dtype=np.int64),
                occupancy=np.ones(int(placed.sum())))

    # bonds and orders in the new order (a stored pair has i < j; a dative bond keeps its direction in its code)
    e = new_index[cc_bonds]
    flip = e[:, 0] > e[:, 1]
    codes = cc_codes.copy()
    codes[flip & (codes == _BOND['dative'])] = _BOND['dative_reverse']
    e = np.sort(e, axis=1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    bonds, codes = e[order].T.copy(), codes[order]
    charge = cc_charge[cc_of_new]

    # the new units, as written above (each unit's heavy atoms then its hydrogens, contiguous in unit order): their
    # bounds read from the new map rather than len(src) // zp (equal while units are equal)
    new_mol_ind = np.array(new_mol_ind, dtype=np.int64)
    bounds = np.concatenate([[0], np.cumsum(np.bincount(new_mol_ind, minlength=zp))])

    # units of one molecule are neutral: a net formal charge on any of them is flagged
    nets = [int(charge[bounds[u]:bounds[u + 1]].sum()) for u in range(zp)]
    imbalance = any(nets)
    if imbalance:
        note_parts.append(f'charge_imbalance: the units\' net formal charges are {nets} (units of one molecule '
                          'should be neutral)')

    # SMILES per unit, from the stored chemistry and the unit's (deposited) coordinates
    new_z = np.where(placed, 1, z[np.clip(src, 0, None)])
    new_pos = np.array(new_pos).reshape(-1, 3)
    smiles = []
    comp_of_unit = {u: ci for ci, u in unit_of_comp.items()}
    for u in range(zp):
        ci = comp_of_unit[u]
        if ci in unreadable:
            smiles.append(None)
            continue
        lo, hi = int(bounds[u]), int(bounds[u + 1])
        smiles.append(_unit_smiles(new_z[lo:hi], charge[lo:hi], bonds, codes, new_pos[lo:hi], lo, hi))
    if unreadable:
        units = sorted(unit_of_comp[ci] for ci in unreadable)
        note_parts.append(f'RDKit cannot read unit(s) {units} ({unreadable[min(unreadable)]}): their bond orders and '
                          'formal charges are CCDC\'s, no SMILES')
    if any(s is None for s in smiles) and not unreadable:
        note_parts.append('RDKit cannot rebuild some unit from the stored chemistry: no SMILES there')
    if len({s for s in smiles if s is not None}) > 1:
        note_parts.append('the units\' SMILES differ (charges, bond orders or stereo)')
    removed = np.setdiff1d(np.arange(n), src[src >= 0])
    if len(removed):
        note_parts.append(f'{len(removed)} deposited hydrogen(s) replaced or removed')
    new_change = np.where(placed, 0, change[np.clip(src, 0, None)]).astype(np.int8)
    return dict(src=src, mol_ind=new_mol_ind, fill=fill, pos=new_pos,
                pos_keep=np.array(pos_keep, dtype=bool), labels=labels, h_source=np.array(h_source, dtype=np.int8),
                h_count_change=new_change, checked=bool(checked), changed=bool(len(changed_atoms)),
                imbalance=bool(imbalance), bonds=bonds, bond_order=codes.astype(np.int8),
                formal_charge=charge.astype(np.int8),
                smiles=smiles, removed=removed, note='; '.join(note_parts))


def _map_sites(cc_z, cc_frac, z, frac, T_fc, what) -> np.ndarray:
    """Per backend site, the index of the row site of its element within ``_MAP_TOL`` modulo the lattice
    (one to one, every row site used); raises ``_Refusal('chemistry_failed')``."""
    if len(cc_z) != len(z):
        raise _Refusal('chemistry_failed', f"CCDC's molecule holds {len(cc_z)} {what}s, the row {len(z)}: they cannot "
                                           'be mapped one to one')
    if not len(z):
        return np.zeros(0, dtype=np.int64)
    d = np.linalg.norm(_min_image(cc_frac[:, None] - frac[None]) @ T_fc.T, axis=-1)
    d[cc_z[:, None] != z[None]] = np.inf
    hits = (d < _MAP_TOL).sum(1)
    best = d.argmin(1)
    if (hits != 1).any() or len(np.unique(best)) != len(best):
        k = int(np.nonzero(hits != 1)[0][0]) if (hits != 1).any() else 0
        raise _Refusal('chemistry_failed', f"CCDC's {what}s cannot be mapped onto the row's (same element within "
                                           f'{_MAP_TOL} A modulo the lattice, one to one): {what} {k} has '
                                           f'{int(hits[k])} candidate(s), nearest {float(d[k].min()):.3g} A')
    return best


def _match_hydrogens(cc_frac, dep_frac, T_fc, dep_index) -> np.ndarray:
    """Per backend hydrogen, the deposited hydrogen it is (within ``_SAME_H_TOL`` modulo the lattice), or -1."""
    d = np.linalg.norm(_min_image(cc_frac[:, None] - dep_frac[None]) @ T_fc.T, axis=-1)
    best = d.argmin(1)
    out = np.where(d[np.arange(len(best)), best] < _SAME_H_TOL, dep_index[best], -1)
    _, first = np.unique(out[out >= 0], return_index=True)
    if len(first) != int((out >= 0).sum()):
        raise _Refusal('chemistry_failed', 'two of CCDC\'s hydrogens sit on one deposited hydrogen')
    return out


def _deposited_parents(z, pos, bonds):
    """``(parent [n], orphans)``: per deposited hydrogen its heavy atom from ``mol_bonds`` (the one bonded heavy
    atom, the nearest of several; -1 for other atoms and unbonded hydrogens), and how many hydrogens have none."""
    n = len(z)
    parent = np.full(n, -1, dtype=np.int64)
    nbrs = [[] for _ in range(n)]
    for i, j in bonds.T.tolist():
        nbrs[i].append(j)
        nbrs[j].append(i)
    orphans = 0
    for h in np.nonzero(z == 1)[0].tolist():
        heavy = [j for j in nbrs[h] if z[j] > 1]
        if not heavy:
            orphans += 1
            continue
        parent[h] = heavy[int(np.argmin([np.linalg.norm(pos[j] - pos[h]) for j in heavy]))]
    return parent, orphans


def _order_hydrogens(h_parent, h_pos, mol_ind, z, pos, zp, n_u, labels):
    """Per unit, its hydrogens (indices into the backend's hydrogens) in one order across units.

    Unit 0: by parent (row order), then the backend's order. Unit k: one parent's
    hydrogens assigned to unit 0's hydrogens of the matching parent (same position in
    the unit's heavy atoms) by least total distance after a Kabsch fit of unit k's heavy
    atoms onto unit 0's; of the proper and the improper fit, the one whose assignment
    has the lower total distance (the proper one unless the other is lower by more than
    1e-6 A). Raises ``_Refusal('hydrogens_failed')`` when matched parents carry
    different counts.
    """
    unit_of_h = mol_ind[h_parent] if len(h_parent) else np.zeros(0, dtype=np.int64)
    within = h_parent - unit_of_h * n_u                  # the parent's position in its unit
    first = np.nonzero(unit_of_h == 0)[0]
    first = first[np.lexsort((first, within[first]))]   # by parent, then the backend's order
    out = [first.tolist()]
    c0 = np.bincount(within[first], minlength=n_u)
    heavy0 = np.nonzero(z[:n_u] > 1)[0]
    for u in range(1, zp):                              # bounded by Z'
        mine = np.nonzero(unit_of_h == u)[0]
        cu = np.bincount(within[mine], minlength=n_u)
        if not np.array_equal(c0, cu):
            i = int(np.nonzero(c0 != cu)[0][0])
            raise _Refusal('hydrogens_failed', f'unit {u} carries {int(cu[i])} hydrogen(s) on heavy atom '
                                               f'{labels[u * n_u + i]}, unit 0 {int(c0[i])} on {labels[i]}: one atom '
                                               'order cannot hold both')
        # the heavy atoms alone cannot always tell a proper fit from its mirror (a planar skeleton fits both
        # exactly, and the mirror swaps the hydrogens above and below the plane): the hydrogens decide
        best = None
        for improper in (False, True):
            apply = _kabsch(pos[u * n_u + heavy0], pos[heavy0], improper)[1]
            ordered, total = [], 0.0
            for i in np.unique(within[first]).tolist():     # unit 0's parents in increasing position, as ``first``
                k0 = first[within[first] == i]
                ku = mine[within[mine] == i]
                cost = np.linalg.norm(h_pos[k0][:, None] - apply(h_pos[ku])[None], axis=-1)
                rr, cc = linear_sum_assignment(cost)
                assigned = np.empty(len(k0), dtype=np.int64)
                assigned[rr] = ku[cc]
                ordered += assigned.tolist()
                total += float(cost[rr, cc].sum())
            if best is None or total < best[0] - 1e-6:      # a mirror fit only when clearly better
                best = (total, ordered)
        out.append(best[1])
    return out


def _raw_molecule(z, charge, bonds_local, codes, pos):
    """The unsanitized RDKit molecule of stored chemistry: every hydrogen an explicit atom (no implicit ones),
    stored aromatic bonds flagged aromatic with their atoms, a 3D conformer from ``pos``."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    types = {'unspecified': Chem.BondType.UNSPECIFIED, 'single': Chem.BondType.SINGLE,
             'double': Chem.BondType.DOUBLE, 'triple': Chem.BondType.TRIPLE, 'quadruple': Chem.BondType.QUADRUPLE,
             'aromatic': Chem.BondType.AROMATIC, 'delocalised': Chem.BondType.ONEANDAHALF,
             'pi': Chem.BondType.UNSPECIFIED, 'dative': Chem.BondType.DATIVE, 'dative_reverse': Chem.BondType.DATIVE}
    m = Chem.RWMol()
    for zz, q in zip(np.asarray(z).tolist(), np.asarray(charge).tolist()):
        atom = Chem.Atom(int(zz))
        atom.SetFormalCharge(int(q))
        atom.SetNoImplicit(True)
        m.AddAtom(atom)
    for (i, j), c in zip(np.asarray(bonds_local).reshape(2, -1).T.tolist(), np.asarray(codes).tolist()):
        name = BOND_ORDERS[int(c)]
        if name == 'dative_reverse':
            i, j = j, i
        m.AddBond(int(i), int(j), types[name])
        if name == 'aromatic':
            m.GetBondBetweenAtoms(int(i), int(j)).SetIsAromatic(True)
            m.GetAtomWithIdx(int(i)).SetIsAromatic(True)
            m.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(np.asarray(pos, dtype=np.float64).tolist()):
        conf.SetAtomPosition(i, Point3D(*p))
    conf.Set3D(True)
    m.AddConformer(conf, assignId=True)
    return m.GetMol()


def _rdkit_molecule(z, charge, bonds_local, codes, pos, fallback: int = 0):
    """An RDKit molecule from stored chemistry (``_raw_molecule``), sanitized, stereo from ``pos``.

    ``fallback`` 0: ``Chem.SanitizeMol`` in full. 1 or 2 (``FEATURES_FALLBACK``): RDKit's
    partial sanitization, ``UpdatePropertyCache(strict=False)`` then every step but the
    valence check (1) and also but kekulization and aromaticity perception (2, which
    raises ValueError when a stored aromatic bond is not in a ring).
    """
    from rdkit import Chem

    mol = _raw_molecule(z, charge, bonds_local, codes, pos)
    if not fallback:
        Chem.SanitizeMol(mol)
    else:
        S = Chem.SanitizeFlags
        ops = S.SANITIZE_ALL ^ S.SANITIZE_PROPERTIES
        if fallback == 2:
            ops = ops ^ S.SANITIZE_KEKULIZE ^ S.SANITIZE_SETAROMATICITY
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, ops)
        if fallback == 2:
            loose = [b.GetIdx() for b in mol.GetBonds() if b.GetIsAromatic() and not b.IsInRing()]
            if loose:
                raise ValueError(f'{len(loose)} stored aromatic bond(s) are in no ring, e.g. bond {loose[0]}')
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def _remote_problems(raw, z) -> list:
    """The problems RDKit's sanitization finds in the unsanitized unit ``raw`` that lie beyond ``features``'
    fallback reach of every atom outside ``LEWIS_ELEMENTS``, one sentence each (empty: every problem is near one).

    Problems are ``Chem.DetectChemistryProblems``' (which runs RDKit's cleanup first, so an uncharged nitro group is
    none). An atom it refuses (valence, or any other per-atom problem) is near when at most ``_VALENCE_REACH``
    bonds from such an atom; an aromatic system with no Kekule form (the stored aromatic bonds connected to the
    atoms it names) when one of the system's atoms is at most ``_KEKULE_REACH`` bonds from one. RDKit names one
    unkekulizable system per call, so each named system's aromatic flags are cleared on a copy (its bonds made
    single) and the call repeats: every repeat clears at least one aromatic bond, so the loop ends within the
    bond count.
    """
    from rdkit import Chem

    z = np.asarray(z)
    n = len(z)
    adj = [[] for _ in range(n)]
    for bond in raw.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i].append(j)
        adj[j].append(i)
    far = n + 1                                 # no bond path to an atom outside LEWIS_ELEMENTS
    dist = np.full(n, far, dtype=np.int64)
    frontier = [i for i in range(n) if int(z[i]) not in LEWIS_ELEMENTS]
    dist[frontier] = 0
    step = 0
    while frontier:                             # breadth-first: each atom enters the frontier once
        step += 1
        nxt = []
        for i in frontier:
            for j in adj[i]:
                if dist[j] == far:
                    dist[j] = step
                    nxt.append(j)
        frontier = nxt

    def how_far(d):
        return 'no bond path' if d >= far else f'{d} bonds'

    out = []
    for p in Chem.DetectChemistryProblems(Chem.Mol(raw)):
        if p.GetType() != 'KekulizeException' and hasattr(p, 'GetAtomIdx'):
            i = p.GetAtomIdx()
            if dist[i] > _VALENCE_REACH:
                out.append(f'{p.GetType()} at atom {i} ({raw.GetAtomWithIdx(i).GetSymbol()}), '
                           f'{how_far(int(dist[i]))} from the nearest atom outside LEWIS_ELEMENTS')
    mol = Chem.RWMol(raw)
    for _ in range(mol.GetNumBonds() + 1):      # bounded: every pass clears an aromatic bond, or ends
        named = [p for p in Chem.DetectChemistryProblems(mol) if p.GetType() == 'KekulizeException']
        if not named:
            break
        system = set(named[0].GetAtomIndices())
        stack, aromatic = list(system), set()
        while stack:                            # the stored aromatic system: bounded by the atoms
            i = stack.pop()
            for bond in mol.GetAtomWithIdx(i).GetBonds():
                if bond.GetIsAromatic():
                    aromatic.add(bond.GetIdx())
                    j = bond.GetOtherAtomIdx(i)
                    if j not in system:
                        system.add(j)
                        stack.append(j)
        d = int(min(dist[i] for i in system))
        if d > _KEKULE_REACH:
            out.append(f'the aromatic system of atoms {sorted(system)} has no Kekule form and is '
                       f'{how_far(d)} from the nearest atom outside LEWIS_ELEMENTS')
        if not aromatic:
            break
        for k in aromatic:
            mol.GetBondWithIdx(k).SetIsAromatic(False)
            mol.GetBondWithIdx(k).SetBondType(Chem.BondType.SINGLE)
        for i in system:
            mol.GetAtomWithIdx(i).SetIsAromatic(False)
    return out


def _fallback_molecule(z, charge, bonds_local, codes, pos):
    """``(mol, tier, tier1_refusal)`` from RDKit's partial sanitization (``_rdkit_molecule`` fallback 1, else 2) of
    a unit whose full sanitization failed (``tier1_refusal``: RDKit's refusal of fallback 1 when 2 was used, else
    ''), or ``(None, 0, why)``: the unit's atoms are all in ``LEWIS_ELEMENTS``, a problem RDKit names lies beyond
    the fallback's reach of every atom outside them (``_remote_problems``), or both tiers are refused."""
    if set(np.asarray(z).tolist()) <= LEWIS_ELEMENTS:
        return None, 0, ('no fallback: every atom of the unit is H, C, N, O, F, P, S, Cl, Se, Br or I, whose '
                         'stored chemistry must sanitize in full')
    remote = _remote_problems(_raw_molecule(z, charge, bonds_local, codes, pos), z)
    if remote:
        more = f' (and {len(remote) - 2} more)' if len(remote) > 2 else ''
        return None, 0, (f"no fallback: {'; '.join(remote[:2])}{more}; the fallback holds only problems within "
                         f'{_VALENCE_REACH} bonds (an atom) or {_KEKULE_REACH} bond (an aromatic system) of an atom '
                         'outside LEWIS_ELEMENTS')
    why = ''
    for tier in (1, 2):
        try:
            return _rdkit_molecule(z, charge, bonds_local, codes, pos, fallback=tier), tier, why
        except Exception as e:  # RDKit's sanitize exceptions, and the ring check above: the next tier, or refused
            why = f'{type(e).__name__}: {str(e)[:150]}'
    return None, 0, f'the fallback without valence check and kekulization is refused too: {why}'


def _unit_bonds(bonds, codes, lo, hi):
    """The bonds of atoms [lo, hi) (both ends inside), renumbered from lo, and their codes."""
    inside = ((bonds >= lo) & (bonds < hi)).all(0)
    return bonds[:, inside] - lo, codes[inside]


def _unit_smiles(z, charge, bonds, codes, pos, lo, hi) -> Optional[str]:
    """RDKit canonical SMILES of one unit, or None when RDKit cannot sanitize it."""
    from rdkit import Chem, rdBase

    ub, uc = _unit_bonds(bonds, codes, lo, hi)
    block = rdBase.BlockLogs()
    try:
        return Chem.MolToSmiles(_rdkit_molecule(z, charge, ub, uc, pos), canonical=True)
    except Exception:       # RDKit's sanitize exceptions: the unit has no SMILES, said in the row's message
        return None
    finally:
        del block


###############################################################################
# features
###############################################################################

def _fingerprint_args(fingerprint) -> Optional[dict]:
    """The validated fingerprint dict, or None."""
    if fingerprint is None:
        return None
    if not isinstance(fingerprint, dict) or set(fingerprint) != {'radius', 'n_bits', 'chirality'}:
        raise ValueError(f'fingerprint must be None or dict(radius=..., n_bits=..., chirality=...), '
                         f'got {fingerprint!r}')
    radius, n_bits, chirality = fingerprint['radius'], fingerprint['n_bits'], fingerprint['chirality']
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError(f'fingerprint radius must be an int >= 0, got {radius!r}')
    if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)) or n_bits < 8 or n_bits % 8:
        raise ValueError(f'fingerprint n_bits must be a positive multiple of 8 (bits are packed), got {n_bits!r}')
    if not isinstance(chirality, bool):
        raise ValueError(f'fingerprint chirality must be a bool, got {chirality!r}')
    return dict(radius=int(radius), n_bits=int(n_bits), chirality=chirality)


def _morgan(fp):
    from rdkit.Chem import rdFingerprintGenerator
    return rdFingerprintGenerator.GetMorganGenerator(radius=fp['radius'], fpSize=fp['n_bits'],
                                                     includeChirality=fp['chirality'])


def _features_row(z, charge, pos, bonds, codes, zp, stored, generator, charges, n_u=None) -> dict:
    """One row's features; raises ``_Refusal('features_failed')``. ``n_u``: atoms per unit, as ``features`` reads them
    from the unit map (``unit_layout``); a direct caller that passes none gets ``len(z) // zp``, equal units."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    if bonds is None or codes is None:
        raise _Refusal('features_failed', 'the row has no bond orders (sanitize gave it no chemistry)')
    bonds = np.asarray(bonds, dtype=np.int64).reshape(2, -1)
    codes = np.asarray(codes, dtype=np.int64)
    n = len(z)
    if n_u is None:
        n_u = n // zp
    smiles, bits, q = [], [], np.zeros(n, dtype=np.float64)
    fallback, reason, tier1 = {}, '', None
    block = rdBase.BlockLogs()
    try:
        for u in range(zp):
            lo, hi = u * n_u, (u + 1) * n_u
            ub, uc = _unit_bonds(bonds, codes, lo, hi)
            try:
                mol = _rdkit_molecule(z[lo:hi], charge[lo:hi], ub, uc, pos[lo:hi])
            except Exception as e:      # RDKit's sanitize exceptions name the problem
                full = f'{type(e).__name__}: {str(e)[:200]}'
                mol, tier, why = _fallback_molecule(z[lo:hi], charge[lo:hi], ub, uc, pos[lo:hi])
                if mol is None:
                    raise _Refusal('features_failed', f'unit {u}: RDKit cannot rebuild the stored chemistry: '
                                                      f'{full}; {why}')
                fallback[u] = tier
                reason = reason or full[:150]
                if tier == 2 and tier1 is None:
                    tier1 = (u, why)
            have = stored[u] if stored is not None and u < len(stored) else None
            if have is None and u not in fallback:
                have = Chem.MolToSmiles(mol, canonical=True)
            smiles.append(have)
            if generator is not None:
                bits.append(np.packbits(generator.GetFingerprintAsNumPy(mol).astype(np.uint8)))
            if charges is not None:
                AllChem.ComputeGasteigerCharges(mol)
                q[lo:hi] = [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()]
    finally:
        del block
    bad = ~np.isfinite(q)
    q[bad] = np.nan
    return dict(smiles=smiles, bits=bits, charges=q, nonfinite=bool(bad.any()), n_nonfinite=int(bad.sum()),
                fallback=fallback, fallback_reason=reason, fallback_tier1=tier1)
