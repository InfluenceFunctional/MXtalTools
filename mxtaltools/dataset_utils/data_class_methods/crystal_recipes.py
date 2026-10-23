"""Recipes of the ingestion chain, and the fundamental-domain wrapper, mixed into MolCrystalData.

The convenience layer of the ingestion contract (``docs/design/crystal_ingestion.md`` §4, in
the GFN repository) over the steps of ``crystal_steps`` and ``crystal_chem``:

``Recipe``
    A frozen record of one run of the chain: ``read`` and then steps of
    ``find_molecules -> sanitize -> parameterize_pose -> check_rebuild -> check_validity ->
    features`` (each at most once, in that order), every one with EVERY argument written
    out, so a call that names a recipe shows exactly what runs, and several recipes can
    coexist (CSD, generator output, tests). ``spec()`` is its JSON form and ``digest`` the
    sha256 of that form.
``CSD_V1``
    The CSD recipe (owner answers of 2026-10-20, §7a): file bonds, CCDC chemistry with
    regenerated hydrogens and a flag where the protonation changed, the old featurizer's
    validity thresholds, fingerprint and charges. ``to_standard_setting`` (an optional step of
    the featurization recipes, owner answer Q4) is not built and is not in it.
``run_chain``
    ``batch.run_chain(recipe)`` runs a recipe's steps after ``read`` on a freshly read batch;
    ``MolCrystalData.read(..., full='csd_v1')`` (or a ``Recipe``) reads and then runs it.
    ``read`` re-analyses nothing unless a recipe is named (owner answer Q7): ``full=True``
    raises. The ``run_chain`` record names the recipe's ``read`` arguments and whether they
    were checked against the call that read the batch (``read_checked``: True through
    ``read(full=...)``, which refuses other arguments; False for ``batch.run_chain``, whose
    batch carries no record of how it was read beyond its dtype).
``canonicalize``
    Posed, Z' = 1, rebuild-checked rows to their fundamental-domain form: the importer
    ``crystal_fd.io_mxt.from_mol_crystal_data``, then ``crystal_fd.canon.canonicalize``.
    It returns a derived product (``FDProduct``: descriptions, molecules, reference
    handedness, identifiers, statuses, the outcome of every input row, and a record of its
    parent batch), never a batch, and can write it with ``crystal_fd.dataset.save_dataset``.
    ``crystal_fd`` is imported inside the method: its table check can raise at import,
    which importing MolCrystalData must not inherit.

Status codes. A recipe adds none: every row's code is a step's. Until ``crystal_steps``
accepts the codes of the later modules (its input check reads ``crystal_steps.STATUS`` only, so
one row refused by ``sanitize`` makes ``parameterize_pose`` raise for the whole batch),
``run_chain`` hands the structural steps a view of the batch in which such rows carry a
placeholder code of ``crystal_steps.STATUS`` that is not ``ok``; the steps pass them through
untouched (they are not ok), and the row's own code is put back. The rows' other values are
the step's pass-through values, bit-identical.

A row's outcome never depends on the other rows of its batch. ``check_validity`` runs
``composition`` on each ok row that holds a moiety that parses (and ``z_value``), and leaves
it unchecked on the others (``validity_checked`` says which), but raises for the whole batch
when no ok row holds one. Until it skips that case per row too, ``run_chain`` runs
``check_validity`` without ``composition`` on such a batch and records the step with the
recipe's ``composition``: every row's values and the record are those a batch with one
moiety-bearing ok row would give the same rows.
"""
import copy
import csv
import hashlib
import inspect
import json
import math
import numbers
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

import numpy as np
import torch
from torch_geometric.data.storage import GlobalStorage

from mxtaltools.dataset_utils.data_class_methods import crystal_steps
from mxtaltools.dataset_utils.data_class_methods.crystal_chem import FEATURES_FALLBACK, OLD_FINGERPRINT, MolCrystalChem
from mxtaltools.dataset_utils.data_class_methods.crystal_io import _CIF_OPTIONS, DONE_READ
from mxtaltools.dataset_utils.data_class_methods.crystal_steps import (DONE_POSE, DONE_REBUILD, MolCrystalSteps,
                                                                       _copy_batch, _set)

__all__ = ['MolCrystalRecipes', 'Recipe', 'CSD_V1', 'RECIPES', 'CHAIN', 'FDProduct', 'get_recipe', 'check_read_call',
           'run_chain_after_read', 'FD_IMPORT_TOLS', 'FD_CANON_TOLS']

#: The chain, in order. A recipe starts with ``read`` and lists a subsequence of the rest.
CHAIN = ('read', 'find_molecules', 'sanitize', 'parameterize_pose', 'check_rebuild', 'check_validity', 'features')

#: Where each step after ``read`` is defined (its keyword-only parameters are the arguments a
#: recipe must list).
_STEP_OWNERS = dict(find_molecules=MolCrystalSteps, sanitize=MolCrystalChem, parameterize_pose=MolCrystalSteps,
                    check_rebuild=MolCrystalSteps, check_validity=MolCrystalSteps, features=MolCrystalChem)

#: Steps of the contract that a recipe cannot name yet, and why.
_NOT_BUILT = dict(
    to_standard_setting="to_standard_setting is not built: it is the optional re-expression step the owner placed in the "
                        "featurization recipes (crystal_ingestion.md §7a, Q4); until it exists, canonicalize "
                        "re-expresses (D1)")

#: ``crystal_steps``' own status table: its input check refuses any later code (module docstring).
_N_STEP_CODES = len(crystal_steps.STATUS)
#: The placeholder a row with a later code carries while a structural step runs: a code of
#: ``crystal_steps.STATUS`` that is not ``ok``, so the step passes the row through.
_PLACEHOLDER = crystal_steps.STATUS.index('validity_failed')
_STRUCTURAL = ('find_molecules', 'parameterize_pose', 'check_rebuild', 'check_validity')

#: Thresholds ``canonicalize`` passes to ``crystal_fd.io_mxt.from_mol_crystal_data``: its
#: defaults, which built ``D:/crystal_datasets/csd_fd_v1b`` (its meta.json, ``importer``).
FD_IMPORT_TOLS = MappingProxyType(dict(pose_tol=1e-2, special_tol=0.1, op_tol=1e-6, metric_tol=1e-6))
#: Thresholds ``canonicalize`` passes to ``crystal_fd.canon.canonicalize``: ``reduce_cell``'s
#: tol 1e-10, angle_tol 1e-6 and face_tol 1e-5 (``FACE_TOL``, owner decision D4), fold tol 1e-9,
#: near-face tol 1e-5 (as csd_fd_v1b), rot_tol 1e-6 and metric_tol 1e-4 (``canonicalize``'s
#: defaults). ``moved_tol`` follows the batch dtype (1e-9 float64, 1e-5 otherwise). For sg 16-74
#: cell_tol and face_tol are also the axis order's tie and automorphism tolerances, and near_tol
#: its face tolerance, all on log lengths (``crystal_fd.axis_order``).
FD_CANON_TOLS = MappingProxyType(dict(tol=1e-9, cell_tol=1e-10, angle_tol=1e-6, face_tol=1e-5, near_tol=1e-5,
                                      rot_tol=1e-6, metric_tol=1e-4))

_SETTINGS = ('reexpress', 'refuse')
_OFF_SHEET = ('mirror_reference', 'enantiomorph', 'keep')
_NEAR_FACE = ('flag', 'nudge')
_ZP_GT1 = ('refuse',)


###############################################################################
# recipes
###############################################################################

def _check_arguments(name, step, args):
    """Raise unless ``args`` are exactly the arguments ``step`` takes (``read``: format, dtype, format options)."""
    if step == 'read':
        if args.get('format') != 'cif':
            raise ValueError(f"recipe {name!r}: read format must be 'cif' (the only format), got {args.get('format')!r}")
        dtype = args.get('dtype')
        if not (isinstance(dtype, torch.dtype) and dtype.is_floating_point):
            raise ValueError(f'recipe {name!r}: read dtype must be a floating torch dtype, got {dtype!r}')
        wanted = {'format', 'dtype'} | set(_CIF_OPTIONS)
    else:
        params = inspect.signature(getattr(_STEP_OWNERS[step], step)).parameters.values()
        wanted = {p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY}
    missing, unknown = sorted(wanted - set(args)), sorted(set(args) - wanted)
    if missing or unknown:
        raise ValueError(f'recipe {name!r}: {step} must list every argument it takes ({sorted(wanted)}); '
                         f'missing {missing}, unknown {unknown}')


def _freeze(where, value):
    """An immutable copy of a recipe argument: lists and tuples as tuples, dicts read-only."""
    if value is None or isinstance(value, (bool, int, float, str, torch.dtype)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(where, v) for v in value)
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f'{where}: a dict argument needs str keys')
        return MappingProxyType({k: _freeze(f'{where}.{k}', v) for k, v in value.items()})
    raise TypeError(f'{where}: a recipe argument must be None, bool, int, float, str, a torch dtype, or a list, '
                    f'tuple or dict of them; got {type(value).__name__}')


def _thaw(value):
    """A frozen argument as the step takes it: read-only dicts as dicts (tuples kept)."""
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(v) for v in value)
    return value


def _jsonable(value):
    if isinstance(value, torch.dtype):
        return str(value).replace('torch.', '')
    if isinstance(value, MappingProxyType):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _canonical_json(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)



@dataclass(frozen=True, eq=False)
class Recipe:
    """A frozen chain: ``read`` and the steps after it, each with every argument.

    Parameters
    ----------
    name : str
        Non-empty; recorded with every product of the recipe.
    steps : sequence of (str, dict)
        ``('read', {...})`` first, then steps of ``CHAIN`` in chain order, each at most
        once. ``read`` lists ``format`` (``'cif'``), ``dtype`` (a floating torch dtype) and
        every format option of that format (``max_atomic_number`` for ``'cif'``); every
        other step lists exactly its keyword-only parameters (none may be left to a
        default). Values: None, bool, int, float, str, a torch dtype (``read``), and
        lists, tuples and str-keyed dicts of them; stored frozen (lists as tuples, dicts
        read-only).

    Attributes
    ----------
    digest : str
        sha256 of ``spec()`` (canonical JSON); recipes are equal when their digests are.

    Raises
    ------
    ValueError
        An empty name; no steps; a first step other than ``read``; an unknown step, one
        out of chain order or repeated; ``canonicalize`` (not a chain step: its output is a
        derived product); missing or unknown arguments; a ``read`` format other than
        ``'cif'`` or a non-floating dtype.
    NotImplementedError
        A step of the contract that is not built (``to_standard_setting``).
    TypeError
        Arguments that are not a str-keyed dict, or a value of another type.
    """
    name: str
    steps: tuple

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f'a recipe needs a non-empty str name, got {self.name!r}')
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, (list, tuple)) or not self.steps:
            raise ValueError('a recipe needs a non-empty sequence of (step, args) pairs')
        frozen, last = [], -1
        for i, item in enumerate(self.steps):
            if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)):
                raise ValueError(f'step {i} of recipe {self.name!r} must be a (name, args) pair, got {item!r:.80}')
            step, args = item
            if step in _NOT_BUILT:
                raise NotImplementedError(_NOT_BUILT[step])
            if step == 'canonicalize':
                raise ValueError("canonicalize is not a chain step: its output is a derived product "
                                 "(batch.canonicalize), not a batch")
            if step not in CHAIN:
                raise ValueError(f'unknown step {step!r} in recipe {self.name!r}; the chain is {CHAIN}')
            if i == 0 and step != 'read':
                raise ValueError(f"recipe {self.name!r} must start with 'read', not {step!r}")
            k = CHAIN.index(step)
            if k <= last:
                raise ValueError(f'recipe {self.name!r}: {step!r} is out of chain order or repeated ({CHAIN})')
            last = k
            if not isinstance(args, dict) or not all(isinstance(a, str) for a in args):
                raise TypeError(f'the arguments of {step!r} must be a dict with str keys, got {args!r:.80}')
            _check_arguments(self.name, step, args)
            frozen.append((step, MappingProxyType({a: _freeze(f'{step}.{a}', v) for a, v in args.items()})))
        object.__setattr__(self, 'steps', tuple(frozen))
        object.__setattr__(self, 'digest', hashlib.sha256(_canonical_json(self.spec()).encode()).hexdigest())

    @property
    def step_names(self) -> tuple:
        """The steps, in order (``'read'`` first)."""
        return tuple(step for step, _ in self.steps)

    def args(self, step: str) -> dict:
        """A mutable copy of one step's arguments, as the step takes them (dicts thawed, tuples kept).

        Raises
        ------
        KeyError
            The recipe does not run ``step``.
        """
        for name, args in self.steps:
            if name == step:
                return {a: _thaw(v) for a, v in args.items()}
        raise KeyError(f'recipe {self.name!r} does not run {step!r}; it runs {self.step_names}')

    def spec(self) -> dict:
        """The recipe as JSON-ready data: ``{'name', 'steps': [{'step', 'args'}, ...]}`` (dtypes as 'float64' etc.)."""
        return dict(name=self.name, steps=[dict(step=step, args={a: _jsonable(v) for a, v in args.items()})
                                           for step, args in self.steps])

    def __eq__(self, other):
        return isinstance(other, Recipe) and other.digest == self.digest

    def __hash__(self):
        return hash(self.digest)

    def __repr__(self):
        return f'Recipe({self.name!r}, steps={self.step_names}, digest={self.digest[:12]})'


#: The CSD recipe (owner answers of 2026-10-20, crystal_ingestion.md §7a). Per step:
#:
#: - ``read``: float64, so the deposited and posed coordinates and the cell are stored as
#:   parsed (the store's float64 fields, §5); the native reader's element limit 100.
#: - ``find_molecules``: the file's bond loop (every CSD export carries one); covalent-radius
#:   scales (1.15 heavy, 1.35 hydrogen) for bonds to symmetry images and ``unlisted_bond``
#:   (the generator benchmark's ingestion values, used in every gate of the steps);
#:   isomorphism cap 5000; special-position tolerance 0.1 A (the FD importer's).
#: - ``sanitize``: CCDC chemistry (Q2: kept until a CCDC-free chain matches it), hydrogens
#:   regenerated with changed protonation flagged, not refused (Q1).
#: - ``parameterize_pose``: measured handedness (D2 excludes forcing); degenerate-axes gap 1e-4,
#:   handedness-stability tolerance 1e-3 (the gates' values).
#: - ``check_rebuild``: the contract's thresholds (0.05 A largest, 0.01 A mean), elements
#:   matched.
#: - ``check_validity``: the old featurizer's ``crystal_filter`` as ``featurize_cif_chunks``
#:   called it (atomic number at most 100, at most 100 heavy atoms per molecule, every heavy
#:   atom's nearest heavy neighbour within 0.9-3.0 A); ``occupancy`` 1e-3 (the read's own
#:   partial-occupancy threshold, stated so the record shows it); ``disorder`` not applied
#:   (the CSD dump excludes disordered entries, ``dump_CSD.py``, and no reader writes
#:   ``has_disorder``: the database flag is in the provenance table, not joined to rows);
#:   ``cell_constraints`` 1e-4 and ``min_contact`` 0.5 A (the structural gates' values:
#:   operators that do not preserve the cell's metric, and superposed atoms, which nothing
#:   else catches in files without disorder tags); ``composition='heavy'`` (the deposited
#:   set against the moiety formula, hydrogens excepted since they are regenerated).
#: - ``features``: the old featurizer's Morgan fingerprint (radius 2, 2048 bits, no
#:   chirality) and Gasteiger charges.
CSD_V1 = Recipe('csd_v1', (
    ('read', dict(format='cif', dtype=torch.float64, max_atomic_number=100)),
    ('find_molecules', dict(bonds='file', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)),
    ('sanitize', dict(chemistry='ccdc', hydrogens='regenerate', on_protonation_change='flag')),
    ('parameterize_pose', dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)),
    ('check_rebuild', dict(max_dev=0.05, mean_dev=0.01, match_elements=True)),
    ('check_validity', dict(max_atomic_number=100, max_heavy_atoms=100, heavy_nn_range=(0.9, 3.0), occupancy=1e-3,
                            disorder=None, cell_constraints=1e-4, min_contact=0.5, composition='heavy')),
    ('features', dict(fingerprint=dict(OLD_FINGERPRINT), charges='gasteiger')),
))

#: Recipes by name (what ``read(full=...)`` and the chunk runner's ``--recipe`` accept).
RECIPES = MappingProxyType({CSD_V1.name: CSD_V1})


def get_recipe(recipe) -> Recipe:
    """The ``Recipe`` named by ``recipe`` (a name in ``RECIPES``, or a ``Recipe`` itself).

    Raises
    ------
    NotImplementedError
        ``recipe`` is True: no recipe is the default (owner answer Q7); True may become an
        alias once there is a single agreed recipe.
    ValueError
        An unknown name.
    TypeError
        Anything else.
    """
    if isinstance(recipe, Recipe):
        return recipe
    if recipe is True:
        raise NotImplementedError("full=True names no recipe: the owner's answer (Q7) is that read re-analyses nothing "
                                  f'unless a recipe is named; pass one of {sorted(RECIPES)} or a Recipe')
    if isinstance(recipe, str):
        if recipe not in RECIPES:
            raise ValueError(f'unknown recipe {recipe!r}; known: {sorted(RECIPES)}')
        return RECIPES[recipe]
    raise TypeError(f'a recipe is a name ({sorted(RECIPES)}) or a Recipe, got {type(recipe).__name__}')


def check_read_call(recipe: Recipe, *, format: str, dtype, format_options: dict):
    """Raise ValueError unless a ``read`` call's arguments are the recipe's ``read`` arguments.

    ``format_options`` are the call's options merged over the format's defaults. The recipe
    fixes them all, so a call cannot silently read otherwise than the recipe says.
    """
    want = recipe.args('read')
    got = dict(format=format, dtype=dtype, **format_options)
    differ = sorted(k for k in set(want) | set(got) if want.get(k, '<absent>') != got.get(k, '<absent>'))
    if differ:
        raise ValueError(f'recipe {recipe.name!r} reads with ' + ', '.join(f'{k}={want.get(k)!r}' for k in differ)
                         + '; this call passes ' + ', '.join(f'{k}={got.get(k)!r}' for k in differ)
                         + ' (pass the recipe\'s values)')


def run_chain_after_read(batch, recipe):
    """``batch.run_chain(recipe)`` for ``MolCrystalData.read(..., full=recipe)``, which called ``check_read_call``.

    The same chain; its record says ``read_checked=True``. Only ``read`` calls it, after
    checking its own arguments against the recipe's.
    """
    return _run_chain(batch, get_recipe(recipe), read_checked=True)


###############################################################################
# the fundamental-domain product
###############################################################################

@dataclass(frozen=True, eq=False)
class FDProduct:
    """What ``canonicalize`` returns: the canonicalised rows, and the outcome of every input row.

    Attributes
    ----------
    desc : CrystalDescription
        [K] Z' = 1, float64, standard setting ``SYM_OPS[sg]``: each canonicalised row's
        canonical form (``crystal_fd.canon.Canonicalization.done``: ``canonical``,
        ``special_metric``, ``off_sheet``; ``pose_only`` is no longer produced: since
        2026-10-21 rows of sg >= 16 have a cell step too), in input-row order.
    mols : list of MoleculeRef
        [K] each row's reference molecule from the importer (unit 0 aligned, hydrogens
        included; the mirror image where ``columns['mirrored']``), bit-exact.
    reference_handedness : torch.Tensor
        [K] long: -1 where the reference is mirrored (D2), +1 elsewhere.
    identifier : list of str
        [K].
    status : torch.Tensor
        [K] long, index into ``status_names``.
    status_names : tuple of str
        ``crystal_fd.canon.STATUS``.
    columns : dict
        [K] per-row columns (``column_docs`` describes each): ``batch_row`` (the input
        batch's row), the importer's flags and diagnostics, the cell and fold records,
        ``near_face`` / ``near_face_kind``, and ``features_fallback`` (the batch's flag of a
        row featurized by the partial-sanitization fallback, an index into
        ``crystal_chem.FEATURES_FALLBACK``; 0 on every row of a batch without the
        attribute); with ``near_face='nudge'`` also ``nudged``,
        ``nudge_cell``, ``nudge_t``, ``nudge_shift`` and ``nudge_message``; with
        ``off_sheet='enantiomorph'`` also ``enantiomorph`` and ``sg_source``.
    column_docs : dict
    manifest : dict
        One entry per input batch row, ``crystal_fd.dataset``'s manifest layout:
        ``source_row`` (the batch row), ``identifier``, ``code`` (the canonical status name
        for a kept row; ``skipped:<status>`` for a row not ok in the batch;
        ``refused:<reason>`` for a row the importer or ``zp_gt1`` refused; ``failed:<status>``
        for a row ``canon.canonicalize`` could not canonicalise), ``dataset_row`` (-1 when not
        kept) and ``detail`` (``batch_status``, ``message``).
    args : dict
        The choices and thresholds ``canonicalize`` ran with.
    parent : dict
        The batch the product was made from (``canonicalize``, Returns).
    """
    desc: object
    mols: list
    reference_handedness: torch.Tensor
    identifier: list
    status: torch.Tensor
    status_names: tuple
    columns: dict
    column_docs: dict
    manifest: dict
    args: dict
    parent: dict

    def __len__(self) -> int:
        return len(self.identifier)

    def code_counts(self) -> dict:
        """{manifest code: input rows}, most frequent first."""
        counts = {}
        for c in self.manifest['code']:
            counts[c] = counts.get(c, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def save(self, path, *, provenance: Optional[dict] = None) -> dict:
        """Write the product with ``crystal_fd.dataset.save_dataset`` to a new folder ``path``.

        Parameters
        ----------
        path : str or os.PathLike
            Must not exist.
        provenance : dict, optional
            Merged into the dataset's ``provenance``; the product records its own
            arguments under ``'canonicalize'`` and its parent batch under ``'parent'``
            (neither key may be given here).

        Returns
        -------
        dict
            The dataset's meta.json.

        Raises
        ------
        FileExistsError
            ``path`` exists.
        TypeError
            ``path`` is not a str or path, or ``provenance`` is not a dict.
        ValueError
            ``provenance`` names ``'canonicalize'`` or ``'parent'``.
        """
        from mxtaltools.crystal_fd.dataset import save_dataset
        path = _new_path(path)
        _check_provenance(provenance)
        record = dict(provenance or {})
        record['canonicalize'] = self.args
        record['parent'] = self.parent
        return save_dataset(path, self.desc, self.mols, self.identifier, reference_handedness=self.reference_handedness,
                            status=self.status, status_names=self.status_names, columns=self.columns,
                            column_docs=self.column_docs, manifest=self.manifest, provenance=record)


_COLUMN_DOCS = dict(
    batch_row='row of the input batch this crystal came from',
    smiles="RDKit canonical SMILES of the reference molecule mols[k]: unit 0's sanitized chemistry (z, formal_charge, "
           "mol_bonds, mol_bond_order) with stereo from mols[k]'s coordinates, so it describes the molecule a "
           "consumer places with the FD pose, mirrored references included; None where the row has no chemistry or "
           "RDKit cannot rebuild it",
    smiles_deposited="unit 0's SMILES as the batch holds it (unit_smiles: stereo of the deposited unit); the "
                     "enantiomer of smiles for a chiral molecule whose reference is an improper image of the "
                     "deposited unit (the pose's image, or a D2 mirror); None where the batch holds none",
    mirrored="D2: the reference molecule is the mirror image diag(-1,1,1) X of the aligned stored molecule "
             "(reference_handedness -1)",
    setting_reexpressed="D1: the batch's operators were not SYM_OPS[sg] as a set; re-expressed by (setting_P, setting_p)",
    ops_reordered='the operators equal SYM_OPS[sg] as a set, in another order',
    setting_P='x_stored = P x_std + p (identity for standard rows)',
    setting_p='x_stored = P x_std + p (zero for standard rows)',
    h_input="unit 0's handedness in the batch, before to_sheet and D2",
    pose_dev='importer pose check: literal asymmetric unit against the batch pos (Angstrom)',
    metric_dev="importer: the standard-setting cell's largest relative departure from the group's metric constraints",
    min_centroid_sep='importer: closest pair of molecular centroids in the unit cell (Angstrom)',
    on_sheet='canonicalize: on the requested sheet (every row when sheet is None)',
    sheet_moved='canonicalize: h changed through an improper element of G',
    reduce_N='cell step basis change, x_std = N x_red + s: reduce_cell (sg 1-15); the canonical axis order of '
             'sg 16-74 (crystal_fd.axis_order: a signed axis permutation with det +1, so the permutation applied is '
             'explicit; identity where the cell was already in canonical order or the group allows one order); '
             'identity for sg 75-230 (the snapped standard cell is canonical)',
    reduce_s='cell step origin shift: reduce_cell (sg 1-15), the axis order\'s quarter-grid shift (sg 16-74); zero for '
             'sg 75-230',
    cell_n_auto='cell step n_auto: proper automorphisms of the reduced metric (sg 1-15); the order (1, 2, 3 or 6) '
                'of the subgroup generated by the allowed axis orders giving the chosen cell within face_tol in log '
                'lengths (sg 16-74); 1 for sg 75-230; above 1 is special_metric',
    cell_n_tied='cell step n_tied: reduce_cell candidates (sg 1-15) or allowed axis orders (sg 16-74) within tol of '
                'the canonical domain; 1 for sg 75-230',
    cell_violation='cell step violation of the chosen cell (reduce_cell, or the axis order\'s faces in log lengths); 0 '
                   'for sg 75-230',
    cell_snap="sg >= 16: size of the snap of the imported cell to the group's exact metric (the mean of W^T G W over "
              "G's rotations), the largest change of a length (relative) or angle (rad); NaN for sg 1-15",
    fold_element='domains.fold element', fold_lattice='domains.fold lattice shift',
    fold_free_shift='domains.fold free-axis shift',
    fold_slack='domains.fold slack of the canonical centroid (fractional distance to the nearest face of ASU(Gamma+))',
    fold_tie='domains.fold tie code (3: tolerance-band fallback)',
    near_face="near_face_kind != 0: within near_tol of a face, so a re-description rounded at float32 level can "
              "canonicalise to a different row (near_face='flag')",
    near_face_kind='bitmask: 1 cell near a cell-chart face (sg 16-74: a face of the axis order in log lengths) or tied '
                   'cell-step candidates; 2 centroid near a face of ASU(Gamma+); 4 fold tolerance-band fallback',
    moved='the canonical form differs from the imported description by more than moved_tol',
    nudged="near_face='nudge': the row was moved nudge_step off the faces it was near (cell-chart units for the cell, "
           'sg 16-74 log lengths, with t and R kept; fractional for the centroid); another crystal than the imported '
           'one: rescore its energy',
    nudge_cell="near_face='nudge': length of the cell's move in cell-chart coordinates (sg 16-74: log lengths); 0 if "
               'not moved',
    nudge_t="near_face='nudge': largest absolute change of a fractional centroid coordinate; 0 if not moved",
    nudge_shift="near_face='nudge': largest atom displacement (Angstrom) of the unit cell built from mols[k] "
                '(crystal_fd.checks.unit_cell_fractional, each form in its own cell), nudged against unnudged '
                'canonical form; 0 if not moved',
    nudge_message="near_face='nudge': why a row with near_face_kind bit 1 or 2 was not nudged; '' otherwise",
    features_fallback=f'int8 index into crystal_chem.FEATURES_FALLBACK = {FEATURES_FALLBACK!r}: how far the features '
                      "step's RDKit sanitization of the row's units fell back (the batch's features_fallback); 1 "
                      "'valence' or 2 'kekule': RDKit refused the full sanitization of a unit holding an element "
                      'outside its valence model (mostly organometallic), so the batch fingerprint and charges of '
                      'that unit come from a partial sanitization; 0 on rows sanitized in full and rows the batch '
                      'did not featurize, and on every row of a batch without the attribute (its features step '
                      'predates the flag and had no fallback, refusing such rows as features_failed, or it ran no '
                      'features step)',
)

#: ``near_face='nudge'``: these columns describe the stored (nudged) row
_NUDGE_COLUMN_DOCS = dict(
    near_face="near_face_kind != 0: the stored row is within near_tol of a face; nudged rows are cleared, and special "
              "metrics, off-sheet rows, fold-band rows and nudge refusals (nudge_message) keep their flag",
    near_face_kind='bitmask of the stored row: 1 cell near a cell-chart face (sg 16-74: a face of the axis order in '
                   'log lengths) or tied cell-step candidates; 2 centroid near a face of ASU(Gamma+); 4 fold '
                   'tolerance-band fallback',
    moved='the stored form (nudged rows: after the nudge) differs from the imported description by more than moved_tol',
    **{k: _COLUMN_DOCS[k] + '; nudged rows: of the unmoved canonical form, not of the stored row (which is at least '
                            'nudge_step from every face)'
       for k in ('reduce_N', 'reduce_s', 'cell_n_auto', 'cell_n_tied', 'cell_violation', 'cell_snap', 'fold_element',
                 'fold_lattice', 'fold_free_shift', 'fold_slack', 'fold_tie')},
)

#: ``off_sheet='enantiomorph'``: the columns it adds, and those whose meaning it changes
_ENANTIOMORPH_COLUMN_DOCS = dict(
    enantiomorph="off_sheet='enantiomorph': a Z' = 1 Sohncke row the importer left off the sheet, stored as its "
                 'mirror-image crystal (crystal_fd.enantiomorph: inversion through the origin written in the standard '
                 'setting (origin shift p in sg 80, 98, 210), reference X, h flipped, the partner space group in the '
                 '11 enantiomorphic pairs): not the deposited crystal but its enantiomorph (energies are required to '
                 'be mirror-invariant)',
    sg_source="off_sheet='enantiomorph': the space group before the enantiomorph map (the batch's, standard setting); "
              'differs from desc.sg on enantiomorph rows of the 22 groups of the enantiomorphic pairs, equal to it '
              'on every other row',
    mirrored="D2 mirrored reference: never set with off_sheet='enantiomorph' (all False; reference_handedness +1)",
    smiles="RDKit canonical SMILES of the reference molecule mols[k] (the aligned molecule X on every row in this "
           "mode): unit 0's sanitized chemistry (z, formal_charge, mol_bonds, mol_bond_order) with stereo from "
           "mols[k]'s coordinates; for a chiral molecule the enantiomer of the posed unit 0's (the batch's "
           'coordinates) exactly where h_input is -1. The posed unit is the deposited unit or an image of it under G: '
           'in a Sohncke group a proper image, so there smiles is the enantiomer of smiles_deposited exactly where '
           'h_input is -1 (on the +1 sheet: every enantiomorph row); in other groups the image may be improper (no '
           'column records it), so h_input alone does not decide whether smiles is smiles_deposited or its '
           'enantiomer; None where the row has no chemistry or RDKit cannot rebuild it',
    smiles_deposited="unit 0's SMILES as the batch holds it (unit_smiles: stereo of the deposited unit); for a chiral "
                     'molecule in a Sohncke group the enantiomer of smiles exactly where h_input is -1; in other '
                     'groups the enantiomer where the reference is an improper image of the deposited unit (see '
                     'smiles); None where the batch holds none',
    h_input="unit 0's handedness in the batch, before to_sheet and the enantiomorph map",
    setting_P='x_stored = P x_std + p (identity for standard rows); on enantiomorph rows, of the source crystal before '
              'the map',
    setting_p='x_stored = P x_std + p (zero for standard rows); on enantiomorph rows, of the source crystal before the '
              'map',
)


###############################################################################
# the mixin
###############################################################################

# noinspection PyAttributeOutsideInit
class MolCrystalRecipes:
    """Mixin providing ``run_chain`` and ``canonicalize`` (module docstring)."""

    def run_chain(self, recipe):
        """Run a recipe's steps after ``read`` on this freshly read batch.

        Parameters
        ----------
        recipe : str or Recipe
            A name in ``RECIPES`` (``'csd_v1'``) or a ``Recipe``.

        Returns
        -------
        MolCrystalDataBatch
            A new batch: each step's output in turn (every step's own contract: rows not ok
            pass through, a failing row gets a status and message, ``done`` bits,
            ``steps_run``); ``steps_run`` ends with ``{'step': 'run_chain', 'args': {'recipe':
            name, 'digest': ..., 'read': the recipe's read arguments, 'read_checked':
            False}}``: this batch carries no record of the arguments it was read with beyond
            its dtype, so the record says they were not checked (``read(..., full=recipe)``
            checks them and records True).

        Raises
        ------
        ValueError
            Not a batch from ``read`` / ``from_arrays``; a row with any ``done`` bit besides
            the read's, or a batch with ``steps_run`` (the recipe runs once, on a fresh
            batch); a batch whose dtype is not the recipe's ``read`` dtype; and whatever a
            step raises for its arguments.
        NotImplementedError, TypeError
            From ``get_recipe``.
        """
        return _run_chain(self, get_recipe(recipe), read_checked=False)

    def canonicalize(self, *, setting: str, sheet, off_sheet: str, near_face: str, zp_gt1: str, path=None,
                     provenance: Optional[dict] = None, parent_store=None, nudge_step=None) -> FDProduct:
        """Fundamental-domain forms of the posed, Z' = 1, rebuild-checked rows.

        The rows that are ``ok``, through ``parameterize_pose`` and ``check_rebuild``, with
        Z' = 1 go to ``crystal_fd.io_mxt.from_mol_crystal_data`` and then
        ``crystal_fd.canon.canonicalize``; the thresholds are ``FD_IMPORT_TOLS`` and
        ``FD_CANON_TOLS`` (those ``D:/crystal_datasets/csd_fd_v1b`` was built with), every
        one passed explicitly. The importer aligns each reference molecule with the live
        builder's function, which runs in float32 only, so it is given the posed positions
        in float32 (cell and pose in the batch dtype), as csd_fd_v1b's float32 rows gave
        them. The batch is not changed.

        Parameters
        ----------
        setting : {'reexpress', 'refuse'}
            The importer's ``nonstandard``: a row whose operators are not ``SYM_OPS[sg]`` as
            a set is re-expressed in the standard setting (D1), or refused.
        sheet : {1, -1, None}
            The handedness sheet rows are moved to (importer and ``canonicalize``); None
            keeps each row's.
        off_sheet : {'mirror_reference', 'enantiomorph', 'keep'}
            A Z' = 1 Sohncke row that cannot reach ``sheet`` through G: re-described with its
            own enantiomer as the reference molecule (D2; ``reference_handedness`` -1), stored
            as its mirror-image crystal (owner decision 2026-10-21; the importer's
            ``off_sheet='enantiomorph'``: reference X, h flipped, the partner space group in the
            11 enantiomorphic pairs; columns ``enantiomorph`` and ``sg_source``, ``mirrored``
            all False), or kept off the sheet (status ``off_sheet``).
        near_face : {'flag', 'nudge'}
            ``'flag'``: rows within ``near_tol`` of a face keep their canonical form and are
            flagged (``near_face``, ``near_face_kind``). ``'nudge'``: ``canonical`` rows near a
            cell-chart face (sg 16-74: a face of the axis order in log lengths, such as Pbca's
            a = b seam) or a face of ASU(Gamma+) are moved
            ``nudge_step`` off it (``crystal_fd.canon.canonicalize(nudge=...)``), with the
            columns ``nudged``, ``nudge_cell``, ``nudge_t``, ``nudge_shift``, ``nudge_message``;
            a nudged row is another crystal, so its energy must be rescored.
        zp_gt1 : {'refuse'}
            Z' > 1 rows are refused (D3: the FD data model holds one molecule per row).
        path : str or os.PathLike, optional
            When given, the product is also written there (``FDProduct.save``); it must not
            exist, which is checked before any work.
        provenance : dict, optional
            Passed to ``FDProduct.save`` with ``path`` (keys other than ``'canonicalize'``
            and ``'parent'``, which this method writes).
        parent_store : str or os.PathLike, optional
            The ingest store this batch was loaded from (``MolCrystalData.load``): its
            schema, file digests, steps and runner record go into ``parent['store']``, after
            checking that its rows, atoms, identifiers and statuses are this batch's.
        nudge_step : float
            Required with ``near_face='nudge'`` and refused otherwise (no default): the step
            off the face, in ``near_face`` units (cell-chart units for the cell, sg 16-74 log
            lengths; fractional for the centroid); it must exceed the near-face tolerance
            ``FD_CANON_TOLS['near_tol']``.
            Recorded in ``args`` (``nudge_step``, and ``canonicalize['nudge']``).

        Returns
        -------
        FDProduct
            ``parent`` records the batch whatever its origin: rows, atoms, the digests of its
            identifiers and source paths, its status counts, its ``steps_run`` (with the
            ``run_chain`` record: recipe, digest, read arguments), the ingestion code digest
            (``crystal_store.code_digest``), and the store (``parent_store``) or None. The
            column ``features_fallback`` carries the batch's flag of each kept row (0 on
            every row when the batch has no ``features_fallback``).

        Raises
        ------
        ValueError
            A choice outside its values; ``near_face='nudge'`` without a finite ``nudge_step``
            above the near-face tolerance, or a ``nudge_step`` with ``near_face='flag'``; an ok
            row that has not been through ``parameterize_pose`` and ``check_rebuild``; not a
            batch; a ``provenance`` key this method writes; a ``parent_store`` that does not
            hold this batch's rows; a batch ``features_fallback`` that is not [B] integer
            codes of ``FEATURES_FALLBACK``.
        FileExistsError
            ``path`` exists.
        FileNotFoundError, StoreError
            ``parent_store`` is not a store folder (``crystal_store.read_meta``).
        """
        if setting not in _SETTINGS:
            raise ValueError(f'setting must be one of {_SETTINGS}, got {setting!r}')
        if isinstance(sheet, bool) or sheet not in (1, -1, None):
            raise ValueError(f'sheet must be 1, -1 or None, got {sheet!r}')
        if off_sheet not in _OFF_SHEET:
            raise ValueError(f'off_sheet must be one of {_OFF_SHEET}, got {off_sheet!r}')
        if near_face not in _NEAR_FACE:
            raise ValueError(f'near_face must be one of {_NEAR_FACE}, got {near_face!r}')
        if near_face == 'nudge':
            if nudge_step is None:
                raise ValueError("near_face='nudge' needs an explicit nudge_step (the step off the face, in near_face "
                                 'units); there is no default')
            if isinstance(nudge_step, bool) or not isinstance(nudge_step, numbers.Real) \
                    or not math.isfinite(nudge_step) or not nudge_step > FD_CANON_TOLS['near_tol']:
                raise ValueError(f"nudge_step must be a finite number above the near-face tolerance "
                                 f"{FD_CANON_TOLS['near_tol']}, got {nudge_step!r}")
            nudge_step = float(nudge_step)
        elif nudge_step is not None:
            raise ValueError(f"nudge_step is for near_face='nudge'; near_face={near_face!r} moves no row")
        if zp_gt1 not in _ZP_GT1:
            raise ValueError(f"zp_gt1 must be 'refuse' (D3: Z' > 1 stays refused for the FD), got {zp_gt1!r}")
        if path is not None:
            path = _new_path(path)
        _check_provenance(provenance)
        from mxtaltools.dataset_utils.data_class_methods.crystal_store import status_table
        if not getattr(self, 'is_batch', False) or any(k not in self._store for k in ('status', 'done', 'ptr')):
            raise ValueError('canonicalize takes a batch from MolCrystalData.read / from_arrays and the steps')
        parent = _parent_record(self, parent_store)
        table = status_table()
        B = self.num_graphs
        fallback = _features_fallback(self)
        status = self.status.detach().to('cpu').long()
        done = self.done.detach().to('cpu').long()
        ok = (status == 0).tolist()
        unchecked = [b for b in range(B) if ok[b] and not (int(done[b]) & DONE_POSE and int(done[b]) & DONE_REBUILD)]
        if unchecked:
            raise ValueError(f'canonicalize takes posed, rebuild-checked rows; ok rows {unchecked[:10]} have not been '
                             'through parameterize_pose and check_rebuild')
        identifiers = [str(x) for x in self['identifier']] if 'identifier' in self._store else [str(b) for b in range(B)]
        zp = self['z_prime'].detach().to('cpu').long() if 'z_prime' in self._store else torch.zeros(B, dtype=torch.long)
        code, message = [''] * B, [''] * B
        for b in range(B):
            if not ok[b]:
                code[b] = f'skipped:{table[int(status[b])]}'
            elif int(zp[b]) != 1:
                code[b] = 'refused:z_prime_gt1'
        eligible = [b for b in range(B) if ok[b] and int(zp[b]) == 1]
        args = dict(setting=setting, sheet=sheet, off_sheet=off_sheet, near_face=near_face, zp_gt1=zp_gt1,
                    importer=dict(FD_IMPORT_TOLS, allow_zp_gt1=False, nonstandard=setting,
                                  mirror_reference=off_sheet == 'mirror_reference', sheet=sheet),
                    canonicalize=dict(FD_CANON_TOLS, sheet=sheet,
                                      moved_tol=1e-9 if self.pos.dtype == torch.float64 else 1e-5))
        docs = dict(_COLUMN_DOCS)
        if off_sheet == 'enantiomorph':
            args['importer']['off_sheet'] = off_sheet
            docs.update(_ENANTIOMORPH_COLUMN_DOCS)
        if near_face == 'nudge':
            args['nudge_step'] = nudge_step
            args['canonicalize']['nudge'] = nudge_step
            docs.update(_NUDGE_COLUMN_DOCS)
        product = _canonical_rows(self, eligible, identifiers, code, message, args, fallback)
        manifest = dict(source_row=torch.arange(B), identifier=identifiers, code=code,
                        dataset_row=product.pop('dataset_row'),
                        detail=dict(batch_status=[table[int(s)] for s in status.tolist()], message=message))
        out = FDProduct(manifest=manifest, args=args, column_docs={k: docs[k] for k in product['columns']},
                        parent=parent, **product)
        if path is not None:
            out.save(path, provenance=provenance)
        return out


###############################################################################
# machinery
###############################################################################

def _new_path(path) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f'path must be a str or os.PathLike, got {type(path).__name__}')
    path = os.path.abspath(os.fspath(path))
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists; a product is never overwritten')
    return path


def _check_provenance(provenance):
    if provenance is None:
        return
    if not isinstance(provenance, dict):
        raise TypeError(f'provenance must be a dict, got {type(provenance).__name__}')
    taken = sorted({'canonicalize', 'parent'} & set(provenance))
    if taken:
        raise ValueError(f'provenance keys {taken} are written by canonicalize itself (its arguments and the parent '
                         'batch; name the parent store with parent_store)')


def _sha_strings(values) -> str:
    """sha256 of one line per value (None as an empty line)."""
    return hashlib.sha256('\n'.join('' if v is None else str(v) for v in values).encode('utf-8')).hexdigest()


def _parent_record(batch, parent_store) -> dict:
    """``FDProduct.parent``: what the batch is, and the store it was loaded from when named (checked against it)."""
    from mxtaltools.dataset_utils.data_class_methods.crystal_store import code_digest, read_meta, status_table
    table = status_table()
    B = batch.num_graphs
    N = int(batch.ptr[-1])
    status = batch.status.detach().to('cpu').long()
    counts = {}
    for s in status.tolist():
        counts[table[s]] = counts.get(table[s], 0) + 1
    identifiers = list(batch['identifier']) if 'identifier' in batch._store else [None] * B
    sources = list(batch['source_path']) if 'source_path' in batch._store else [None] * B
    steps = json.loads(json.dumps(list(batch['steps_run']) if 'steps_run' in batch._store else []))
    chain = [s for s in steps if isinstance(s, dict) and s.get('step') == 'run_chain']
    record = dict(rows=B, atoms=N, identifiers_sha256=_sha_strings(identifiers), source_paths_sha256=_sha_strings(sources),
                  status_counts=counts, steps_run=steps, run_chain=chain[-1]['args'] if chain else None,
                  ingestion_code=code_digest()['digest'], store=None)
    if parent_store is None:
        return record
    if not isinstance(parent_store, (str, os.PathLike)):
        raise TypeError(f'parent_store must be a str or os.PathLike, got {type(parent_store).__name__}')
    store = os.path.abspath(os.fspath(parent_store))
    meta = read_meta(store)
    with open(os.path.join(store, 'manifest.csv'), newline='', encoding='utf-8') as f:
        lines = list(csv.reader(f))[1:]
    stored_ids = [line[1] for line in lines]
    stored_status = [line[4] for line in lines]
    differ = [name for name, same in (
        ('rows', meta.get('rows') == B), ('atoms', meta.get('atoms') == N),
        ('identifiers', stored_ids == ['' if v is None else str(v) for v in identifiers]),
        ('statuses', stored_status == [table[s] for s in status.tolist()])) if not same]
    if differ:
        raise ValueError(f'parent_store {store} does not hold this batch: its {differ} differ (rows {meta.get("rows")} '
                         f'there, {B} here)')
    record['store'] = dict(path=store, schema=meta['schema'], data_sha256=meta['files']['data.pt']['sha256'],
                           manifest_sha256=meta['files']['manifest.csv']['sha256'], written_utc=meta.get('written_utc'),
                           code=meta.get('code', {}).get('digest'), steps_run=meta.get('steps_run'),
                           featurize_cifs=(meta.get('extra') or {}).get('featurize_cifs'))
    return record


def _reference_smiles(batch, b, mol) -> Optional[str]:
    """RDKit SMILES of the reference molecule ``mol`` with row ``b``'s unit 0 chemistry; None without one.

    The importer's reference keeps unit 0's atom order (``io_mxt``: unit 0 aligned, or its
    mirror image), so the row's bonds apply to it; a row whose atomic numbers do not match
    gets None rather than a wrong molecule.
    """
    from mxtaltools.dataset_utils.data_class_methods.crystal_chem import _unit_smiles
    if not all(k in batch._store for k in ('has_chemistry', 'formal_charge', 'mol_bonds', 'mol_bond_order')):
        return None
    bonds, codes = batch['mol_bonds'][b], batch['mol_bond_order'][b]
    if not bool(batch['has_chemistry'][b]) or bonds is None or codes is None:
        return None
    s, e = int(batch.ptr[b]), int(batch.ptr[b + 1])
    z = batch.z[s:e].detach().to('cpu').long()              # Z' = 1: the row is unit 0
    if len(mol.z) != len(z) or not torch.equal(mol.z.long(), z):
        return None
    charge = batch['formal_charge'][s:e].detach().to('cpu').long().numpy()
    bonds = np.asarray(bonds, dtype=np.int64).reshape(2, -1)
    return _unit_smiles(z.numpy(), charge, bonds, np.asarray(codes, dtype=np.int64), mol.pos.detach().numpy(), 0, len(z))


def _check_fresh(batch, recipe):
    """Raise ValueError unless ``batch`` is a batch only ``read`` has processed, in the recipe's dtype."""
    crystal_steps._check_input_batch(batch)
    done = batch.done.detach().to('cpu').long()
    other = [b for b in range(len(done)) if int(done[b]) != DONE_READ]
    if other or ('steps_run' in batch._store and batch['steps_run']):
        raise ValueError(f'run_chain runs a recipe once, on a batch only read has processed; '
                         + (f'rows {other[:10]} carry done bits {sorted({int(done[b]) for b in other})[:5]}' if other
                            else 'this batch has steps_run'))
    dtype = recipe.args('read')['dtype']
    if batch.pos.dtype != dtype:
        raise ValueError(f'recipe {recipe.name!r} reads with dtype {dtype}; this batch is {batch.pos.dtype}')


def _run_chain(batch, recipe, *, read_checked: bool):
    """The steps of ``recipe`` after ``read`` on a fresh ``batch``, and the ``run_chain`` record."""
    _check_fresh(batch, recipe)
    out = batch
    for step in recipe.step_names[1:]:
        out = _run_step(out, step, recipe.args(step))
    if out is batch:
        out = _copy_batch(batch)
    previous = list(out['steps_run']) if 'steps_run' in out._store else []
    read = recipe.spec()['steps'][0]['args']
    _set(out, 'steps_run', previous + [dict(step='run_chain', args=dict(recipe=recipe.name, digest=recipe.digest,
                                                                        read=read, read_checked=read_checked))],
         'batch')
    return out


def _no_ok_row_has_a_moiety(batch) -> bool:
    """True when ``check_validity(composition=...)`` would raise only because no ok row holds a moiety that parses.

    False when there is no ok row (nothing raises), or when the batch holds no ``moiety`` /
    ``z_value`` at all (then every batch of its reader raises alike: not a row's matter).
    """
    rows = crystal_steps._ok_rows(batch)
    if not rows or 'moiety' not in batch._store or 'z_value' not in batch._store:
        return False
    return all(crystal_steps._moiety_counts(batch['moiety'][b], int(batch['z_value'][b]),
                                            len(batch.symmetry_operators[b])) is None for b in rows)


def _call_step(batch, step, args):
    """``batch.<step>(**args)``, with ``check_validity``'s composition kept per row (module docstring)."""
    if step == 'check_validity' and args.get('composition') in ('heavy', 'all') and _no_ok_row_has_a_moiety(batch):
        out = batch.check_validity(**dict(args, composition=None))
        steps = copy.deepcopy(list(out['steps_run']))
        if steps[-1].get('step') != 'check_validity' or 'composition' in steps[-1]['args']:
            raise RuntimeError(f'check_validity recorded {steps[-1]!r:.200}; run_chain cannot restore its composition')
        steps[-1]['args'] = dict(steps[-1]['args'], composition=args['composition'])     # last, as check_validity orders it
        _set(out, 'steps_run', steps, 'batch')
        return out
    return getattr(batch, step)(**args)


def _run_step(batch, step, args):
    """``batch.<step>(**args)``; a structural step sees rows with later codes under a placeholder (module docstring)."""
    if step not in _STRUCTURAL:
        return _call_step(batch, step, args)
    status = batch.status.detach().to('cpu').long()
    later = status >= _N_STEP_CODES
    if not bool(later.any()):
        return _call_step(batch, step, args)
    placeholder = torch.where(later, torch.full_like(status, _PLACEHOLDER), status)
    view = _view(batch, status=placeholder.to(batch.status.dtype).to(batch.status.device))
    out = _call_step(view, step, args)
    got = out.status.detach().to('cpu').long()
    if not bool((got[later] == _PLACEHOLDER).all()):
        raise RuntimeError(f'{step} changed the status of rows it should have passed through: '
                           f'{torch.nonzero(later & (got != _PLACEHOLDER)).reshape(-1).tolist()[:10]}')
    restored = torch.where(later, status, got)
    out['status'] = restored.to(out.status.dtype).to(out.status.device)
    return out


def _view(batch, **values):
    """A batch sharing every value of ``batch`` but those given (the steps and the importer never write to
    their input); the slice bookkeeping is shared, so a replacement keeps its key's level."""
    view = type(batch).__new__(type(batch))
    view.__dict__['_store'] = GlobalStorage(_parent=view)
    for key, value in batch._store.items():
        view[key] = values.get(key, value)
    view.__dict__['_num_graphs'] = batch.__dict__.get('_num_graphs')
    for name in ('_slice_dict', '_inc_dict'):
        entries = batch.__dict__.get(name)
        view.__dict__[name] = None if entries is None else dict(entries)
    return view


def _features_fallback(batch) -> torch.Tensor:
    """[B] int8 CPU: the batch's ``features_fallback``, or zeros when it has none (a features step that predates the
    flag, or none); raises ValueError for anything but [B] integer codes of ``FEATURES_FALLBACK``."""
    B = batch.num_graphs
    if 'features_fallback' not in batch._store:
        return torch.zeros(B, dtype=torch.int8)
    v = batch['features_fallback']
    if not torch.is_tensor(v) or v.shape != (B,) or v.is_floating_point() or v.is_complex() or v.dtype == torch.bool:
        raise ValueError(f'features_fallback must be a [B] = [{B}] integer tensor, got '
                         + (f'{v.dtype} {tuple(v.shape)}' if torch.is_tensor(v) else type(v).__name__))
    v = v.detach().to('cpu')
    bad = torch.nonzero((v < 0) | (v >= len(FEATURES_FALLBACK))).reshape(-1).tolist()
    if bad:
        raise ValueError(f'features_fallback rows {bad[:10]} hold codes outside FEATURES_FALLBACK {FEATURES_FALLBACK}')
    return v.to(torch.int8)


def _canonical_rows(batch, eligible, identifiers, code, message, args, fallback) -> dict:
    """Import and canonicalise the ``eligible`` rows; fill ``code`` / ``message`` for them; the product's fields
    (``fallback``: ``_features_fallback(batch)``)."""
    from mxtaltools.crystal_fd import canon, io_mxt
    from mxtaltools.crystal_fd.description import CrystalDescription
    B = len(code)
    dataset_row = torch.full((B,), -1, dtype=torch.long)
    F64 = torch.float64
    empty = dict(desc=CrystalDescription.from_parts(torch.zeros(0, dtype=torch.long), torch.zeros(0, 6, dtype=F64),
                                                    torch.zeros(0, 1, 3, dtype=F64), torch.zeros(0, 1, 3, 3, dtype=F64),
                                                    torch.zeros(0, 1, dtype=torch.long)),
                 mols=[], reference_handedness=torch.zeros(0, dtype=torch.long), identifier=[],
                 status=torch.zeros(0, dtype=torch.long), status_names=tuple(canon.STATUS), columns={},
                 dataset_row=dataset_row)
    if not eligible:
        return empty
    sub = batch.select_rows(torch.tensor(eligible, dtype=torch.long))
    # the importer aligns each molecule with the live builder's function, which runs in float32 only
    sub = _view(sub, pos=sub.pos.to(torch.float32))
    imp = args['importer']
    extra = {'off_sheet': imp['off_sheet']} if 'off_sheet' in imp else {}
    desc, info = io_mxt.from_mol_crystal_data(sub, sheet=imp['sheet'], pose_tol=imp['pose_tol'],
                                              special_tol=imp['special_tol'], op_tol=imp['op_tol'],
                                              allow_zp_gt1=imp['allow_zp_gt1'], nonstandard=imp['nonstandard'],
                                              mirror_reference=imp['mirror_reference'], metric_tol=imp['metric_tol'],
                                              **extra)
    for i, reason in enumerate(info.reason):
        if reason is not None:
            code[eligible[i]] = f'refused:{reason}'
    if desc.num_rows == 0:
        return empty
    c = canon.canonicalize(desc, **args['canonicalize'])
    names = c.names()
    rows = info.row.tolist()                       # imported row j is sub row rows[j]
    keep = []
    for j, r in enumerate(rows):
        b = eligible[r]
        if bool(c.done[j]):
            code[b] = names[j]
            keep.append(j)
        else:
            code[b] = f'failed:{names[j]}'
        message[b] = c.message[j]
    kj = torch.tensor(keep, dtype=torch.long)
    in_rows = torch.tensor([rows[j] for j in keep], dtype=torch.long)
    batch_rows = torch.tensor([eligible[rows[j]] for j in keep], dtype=torch.long)
    dataset_row[batch_rows] = torch.arange(len(keep))
    smiles = batch['unit_smiles'] if 'unit_smiles' in batch._store else [None] * B
    columns = dict(
        batch_row=batch_rows,
        smiles=[_reference_smiles(batch, b, info.mols[j]) for j, b in zip(keep, batch_rows.tolist())],
        smiles_deposited=[(smiles[b][0] if smiles[b] else None) for b in batch_rows.tolist()],
        mirrored=info.mirrored[kj], setting_reexpressed=info.setting_reexpressed[in_rows],
        ops_reordered=info.ops_reordered[in_rows], setting_P=info.setting_P[kj], setting_p=info.setting_p[kj],
        h_input=info.h_input[kj, 0], pose_dev=info.pose_dev[in_rows], metric_dev=info.metric_dev[in_rows],
        min_centroid_sep=info.min_centroid_sep[in_rows], on_sheet=c.on_sheet[kj], sheet_moved=c.sheet_moved[kj],
        reduce_N=c.N[kj], reduce_s=c.s[kj], cell_n_auto=c.n_auto[kj], cell_n_tied=c.n_tied[kj],
        cell_violation=c.violation[kj], cell_snap=c.cell_snap[kj], fold_element=c.element[kj],
        fold_lattice=c.lattice[kj], fold_free_shift=c.free_shift[kj], fold_slack=c.slack[kj], fold_tie=c.tie[kj],
        near_face=c.near_face[kj] != 0, near_face_kind=c.near_face[kj], moved=c.moved[kj])
    if imp.get('off_sheet') == 'enantiomorph':
        columns.update(enantiomorph=info.enantiomorph[kj], sg_source=info.sg_source[kj])
    if 'nudge' in args['canonicalize']:
        columns.update(nudged=c.nudged[kj], nudge_cell=c.nudge_cell[kj], nudge_t=c.nudge_t[kj],
                       nudge_shift=_nudge_shifts(desc, c, keep, info.mols, args['canonicalize']),
                       nudge_message=[c.nudge_message[j] for j in keep])
    columns['features_fallback'] = fallback[batch_rows].clone()
    return dict(desc=c.desc[kj], mols=[info.mols[j] for j in keep],
                reference_handedness=info.reference_handedness[kj].clone(),
                identifier=[identifiers[b] for b in batch_rows.tolist()], status=c.status[kj].long(),
                status_names=tuple(canon.STATUS), columns=columns, dataset_row=dataset_row)


def _nudge_shifts(desc, c, keep, mols, canon_args) -> torch.Tensor:
    """[len(keep)] float64 ``nudge_shift``: for each nudged row, its unnudged canonical form (``canon.canonicalize``
    without ``nudge`` on the nudged rows, the same thresholds) with the nudged row's R against the nudged one
    (``_nudge_shift``); 0 elsewhere. The nudge keeps R except at a special position (sg >= 16, fold tie 2), where it
    takes the image of the unnudged pose under a stabiliser element that fixes the unnudged centroid: the reference is
    then that image, so the shift is the move alone."""
    from mxtaltools.crystal_fd import canon
    out = torch.zeros(len(keep), dtype=torch.float64)
    picked = [q for q, j in enumerate(keep) if bool(c.nudged[j])]
    if not picked:
        return out
    rows = torch.tensor([keep[q] for q in picked], dtype=torch.long)
    plain = canon.canonicalize(desc[rows], **{k: v for k, v in canon_args.items() if k != 'nudge'})
    if not torch.equal(plain.status, c.status[rows]) or bool((plain.near_face == 0).any()):
        raise RuntimeError('canonicalize without nudge does not give the unnudged forms of the nudged rows')
    for n, (q, j) in enumerate(zip(picked, rows.tolist())):
        ref = plain.desc[n:n + 1]
        out[q] = _nudge_shift(c.desc[j:j + 1], ref.replace(R=c.desc.R[j:j + 1]), mols[j])
    return out


def _nudge_shift(nudged, plain, mol) -> float:
    """Largest distance (Angstrom) between the same atom of the same molecule in the unit cells of two descriptions
    of one row (``checks.unit_cell_fractional``, each in its own cell matrix), wrapped centroids matched by the
    nearest whole lattice vector."""
    from mxtaltools.crystal_fd.checks import unit_cell_fractional
    fn = unit_cell_fractional(nudged, mol)[0]                                    # [|G|, n, 3]
    f0 = unit_cell_fractional(plain, mol)[0]
    fn = fn - torch.round(fn - f0)
    xn = torch.einsum('ij,gnj->gni', nudged.cell_matrix()[0], fn)
    x0 = torch.einsum('ij,gnj->gni', plain.cell_matrix()[0], f0)
    return float((xn - x0).norm(dim=-1).max())

