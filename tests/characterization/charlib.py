"""Shared machinery of the stage-0 characterization suite (``tests/characterization``).

Imported by the test modules and by the two generator scripts (``build_fixtures.py``,
``build_goldens.py``), so a pin is computed by exactly one piece of code whether it is being
written or checked. Not collected by pytest (no ``test_`` prefix). Reads only
``tests/characterization/data``; never D:.

Contents
--------
- ``deterministic()``: the thread / determinism / default-dtype settings every pin is computed
  under (one thread, deterministic algorithms, float32 default), restored on exit.
- fixtures: ``fixtures()`` loads ``data/fixtures.pt``; ``build_case(name)`` rebuilds a fresh
  crystal batch for a named case from those plain arrays through the public constructors.
- pins: ``CASES``, ``case_groups(case)`` and ``compute_group(case, group)``, the quantities pinned
  bit for bit; ``encode`` / ``compare`` store and check them (small tensors in full, large ones as
  a sha256 of their canonical bytes). ``group_environment(kind)``: the package versions a pin
  group depends on beyond the core ones (``environment()``).
- MLIP stubs (``StubUMA``, ``StubMACE``): record every input they are handed and return a
  deterministic, non-zero energy per graph computed from that input, so the lattice-energy
  arithmetic downstream of the model is pinned too.
- oracle: ``oracle_pairs`` / ``builder_pairs`` / ``compare_pair_sets``, a float64 brute-force
  lattice sum (every atom pair closer than the cutoff between the asymmetric-unit molecules and
  every lattice and symmetry image), independent of the cluster builder.
- ``element_truth``: the element of every unit-cell row found geometrically (which asymmetric-unit
  atom it is an image of), independent of any assumed row layout; ``assembled_elements``: what
  each MLIP input builder hands a model.
- known-failure helpers: ``deliberate_refusal`` (an exception raised by a ``raise`` statement in
  mxtaltools, not an accidental crash) and ``raised_in`` (where an exception came from).
"""
import hashlib
import math
import os
import platform
import re
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FIXTURES_PATH = DATA / 'fixtures.pt'
FIXTURES_MANIFEST = DATA / 'fixtures_manifest.json'
GOLDEN_MANIFEST = DATA / 'goldens_manifest.json'

#: tensors larger than this are pinned by the sha256 of their canonical bytes, not stored in full
HASH_ABOVE_BYTES = 16384

#: the cutoffs every cluster quantity is pinned at (the live GFN analyze cutoff is 10)
CUTOFFS = (6.0, 10.0)
RDF_MODES = (None, 'elementwise', 'envwise', 'atomwise')

#: |d - cutoff| below this (Angstrom) is a boundary pair: reported, excluded from pair-set equality
ORACLE_BAND = 2e-4
#: per-pair distance agreement between the builder (float32 geometry) and the oracle (float64)
ORACLE_DIST_TOL = 1e-3

#: MACE stub: everything the input builders read from a model
STUB_R_MAX = 6.0


def assert_inside_data(path) -> Path:
    """Generators may write only under tests/characterization/data; refuse anything else."""
    p = Path(path).resolve()
    if p.parent != DATA:
        raise RuntimeError(f'refusing to write {p}: generators write only into {DATA}')
    return p


# ---------------------------------------------------------------------------------------------
# determinism and environment
# ---------------------------------------------------------------------------------------------

@contextmanager
def deterministic():
    """One thread, deterministic algorithms, float32 default dtype; the previous settings come
    back on exit, so nothing leaks into other test modules of the same session."""
    prev = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(), torch.get_default_dtype())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_num_threads(prev[0])
        torch.use_deterministic_algorithms(prev[1], warn_only=prev[2])
        torch.set_default_dtype(prev[3])


def environment() -> dict:
    """The library versions the goldens depend on, the platform, and every MXT_* variable (the
    MLIP builders and the triclinic walls read MXT_* switches at import)."""
    import torch_geometric
    import torch_scatter
    import torch_cluster
    return {
        'torch': str(torch.__version__),        # plain str: TorchVersion does not load with weights_only
        'numpy': str(np.__version__),
        'torch_geometric': str(torch_geometric.__version__),
        'torch_scatter': str(torch_scatter.__version__),
        'torch_cluster': str(torch_cluster.__version__),
        'python': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'mxt_env': {k: v for k, v in sorted(os.environ.items())
                    if k.startswith('MXT_') and k not in MXT_ENV_IGNORED},
    }


#: per pin-group kind, the packages beyond the core ones (``environment()``) whose code the group's values
#: pass through; the group's pins skip when any of these versions differs from the goldens'
GROUP_PACKAGES = {
    'mlip': ('fairchem.core', 'mace', 'ase'),       # the MLIP data classes the assembly pins pass through
    'elements': ('fairchem.core', 'mace', 'ase'),
    'exp6': ('rdkit',),                             # atom typing and MMFF charges
    'stdcell': ('spglib',),
    'ingest': ('networkx', 'scipy', 'ase'),         # isomorphism, assignment, covalent radii
}


def _versions(names) -> dict:
    out = {}
    for name in names:
        try:
            mod = __import__(name, fromlist=['__version__'])
            out[name] = str(getattr(mod, '__version__', 'unknown'))
        except Exception as e:  # pragma: no cover - recorded, never raised
            out[name] = f'unavailable: {type(e).__name__}'
    return out


def group_environment(kind: str) -> dict:
    """Versions of the packages a pin group of ``kind`` depends on beyond the core environment ({} for most)."""
    return _versions(GROUP_PACKAGES.get(kind, ()))


def all_group_environments() -> dict:
    return {kind: group_environment(kind) for kind in GROUP_PACKAGES}


#: MXT_* variables that change no computed value (a test-harness switch and a timing hook); every other
#: MXT_* variable is recorded and must match, since several select MLIP builders or penalty forms at import
MXT_ENV_IGNORED = ('MXT_SKIP_GPU_PREFLIGHT', 'MXT_UMA_GRAPH_TIMER')

ENV_KEYS_THAT_SKIP = ('torch', 'numpy', 'torch_geometric', 'torch_scatter', 'torch_cluster', 'mxt_env')


def environment_mismatch(recorded: dict) -> list:
    """[(key, recorded, current)] for the environment entries whose change invalidates the goldens."""
    now = environment()
    return [(k, recorded.get(k), now.get(k)) for k in ENV_KEYS_THAT_SKIP if recorded.get(k) != now.get(k)]


def source_digest() -> dict:
    """sha256 of every mxtaltools/*.py file, recorded with the goldens for information only (a
    mismatch does not skip anything: changing the code is exactly what the pins exist to catch)."""
    import mxtaltools
    root = Path(mxtaltools.__file__).resolve().parent
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        h.update(str(p.relative_to(root)).replace('\\', '/').encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
        n += 1
    return {'root': str(root), 'files': n, 'sha256': h.hexdigest()}


# ---------------------------------------------------------------------------------------------
# fixtures: plain arrays -> crystals, through the public constructors only
# ---------------------------------------------------------------------------------------------

_FIXTURES = None


def fixtures() -> dict:
    """``data/fixtures.pt`` (tensors, strings, lists and dicts only; read with weights_only=True)."""
    global _FIXTURES
    if _FIXTURES is None:
        if not FIXTURES_PATH.exists():
            raise FileNotFoundError(f'{FIXTURES_PATH} is missing; run build_fixtures.py (see README.md)')
        _FIXTURES = torch.load(FIXTURES_PATH, weights_only=True)
    return _FIXTURES


def _mol(rec, x_default_2d=True, **extra):
    """MolData from a molecule record {'z', 'pos', 'x'?}; analysis (radius, mass, volume) computed."""
    from mxtaltools.dataset_utils.data_classes import MolData
    z = rec['z'].clone()
    if rec.get('x') is not None:
        x = rec['x'].clone()
    else:
        x = torch.zeros(len(z), 1) if x_default_2d else torch.zeros(len(z))
    return MolData(z=z, pos=rec['pos'].clone(), x=x, do_mol_analysis=True, **extra)


def f1_row(rec, max_z_prime: int = 1, reference_handedness: bool = True):
    """One Z'=1 csd_fd_v5 row exactly as ``crystal_fd.dataset.to_mol_crystal_data(mode='body')``
    builds it (build_fixtures.py asserts the equality when the fixture is written)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    z = rec['z'].clone()
    m = MolData(z=z, pos=rec['pos'].clone(), x=torch.zeros(len(z), 1), do_mol_analysis=True)
    row = MolCrystalData(molecule=m, sg_ind=int(rec['sg']), cell_lengths=rec['cell_lengths'].clone(),
                         cell_angles=rec['cell_angles'].clone(), aunit_centroid=rec['aunit_centroid'].clone(),
                         aunit_orientation=rec['aunit_orientation'].clone(),
                         aunit_handedness=rec['aunit_handedness'].clone(), identifier=rec['identifier'],
                         do_box_analysis=True, max_z_prime=max_z_prime)
    if reference_handedness:
        row.reference_handedness = rec['reference_handedness'].clone()
    return row


def f3_batch(recs):
    """The ingestion-chain Z'=2 rows as one batch, through ``MolCrystalData.from_arrays`` (the
    ingestion constructor) with each row's own symmetry operators, float32."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    cell = torch.stack([torch.cat([r['cell_lengths'], r['cell_angles']]) for r in recs])
    xyz = torch.cat([r['pos'] for r in recs])
    z = torch.cat([r['z'] for r in recs])
    counts = torch.tensor([len(r['z']) for r in recs])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return MolCrystalData.from_arrays(
        cell, xyz, z, ptr=ptr, sg=torch.tensor([int(r['sg']) for r in recs]),
        ops=[r['ops'].numpy() for r in recs], dtype=torch.float32,
        z_prime=torch.stack([r['z_prime'] for r in recs]),
        aunit_centroid=torch.stack([r['aunit_centroid'] for r in recs]),
        aunit_orientation=torch.stack([r['aunit_orientation'] for r in recs]),
        aunit_handedness=torch.stack([r['aunit_handedness'] for r in recs]),
        identifier=[r['identifier'] for r in recs], is_well_defined=[True] * len(recs),
        unit_radius=torch.stack([r['unit_radius'] for r in recs]).float(),
        unit_mass=torch.stack([r['unit_mass'] for r in recs]).float(),
        unit_volume=torch.stack([r['unit_volume'] for r in recs]).float(),
        unit_smiles=[list(r['unit_smiles']) for r in recs])


def f3_list_row(rec):
    """One ingestion-chain Z'=2 row written through the LIST constructor (one MolData per unit,
    each with its own posed conformer), max_z_prime 2: the form a Z'=2 row takes beside a
    constructor-built Z'=1 row in one collated batch (from_arrays rows carry other keys)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    zp = int(rec['z_prime'])
    n = len(rec['z']) // zp
    units = [_mol({'z': rec['z'][k * n:(k + 1) * n], 'pos': rec['pos'][k * n:(k + 1) * n].float()})
             for k in range(zp)]
    # z_prime as a [1] tensor: until stage 2 an int was stored 0-dim, which did not collate beside the
    # default ones(1) of a constructor-built Z'=1 row (audit #7); an int is stored [1] since, and the
    # tensor form is kept so the fixture rows are built exactly as before
    return MolCrystalData(molecule=units, sg_ind=int(rec['sg']), z_prime=torch.tensor([zp]), max_z_prime=zp,
                          cell_lengths=rec['cell_lengths'].float(), cell_angles=rec['cell_angles'].float(),
                          aunit_centroid=rec['aunit_centroid'].float(),
                          aunit_orientation=rec['aunit_orientation'].float(),
                          aunit_handedness=rec['aunit_handedness'].clone(), identifier=rec['identifier'],
                          do_box_analysis=True)


def f3b_row(rec):
    """One identical-copy Z'=2 row exactly as ``zp_doubling.double_zp1_crystal`` builds it: the
    molecule duplicated through the list constructor (0-dim radius / mass / volume, so they sum),
    then round-tripped through a batch (the attribute shapes search outputs carry)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    m = rec['mol']
    kw = {k: m[k].clone() for k in ('x', 'mol_volume', 'mass', 'radius')}
    mol = MolData(z=m['z'].clone(), pos=m['pos'].clone(), identifier=m['identifier'], **kw)
    doubled = MolCrystalData(
        molecule=[mol, mol.clone()], sg_ind=int(rec['sg']), z_prime=2, max_z_prime=2,
        cell_lengths=rec['cell_lengths'].clone(), cell_angles=rec['cell_angles'].clone(),
        aunit_centroid=rec['aunit_centroid'].clone(), aunit_orientation=rec['aunit_orientation'].clone(),
        aunit_handedness=rec['aunit_handedness'].clone(), identifier=rec['identifier'], do_box_analysis=True)
    return collate_data_list([doubled]).batch_to_list()[0]


def synthetic_row(rec, max_z_prime=None):
    """A hand-posed crystal of the audit probes from its stored molecules (RDKit-embedded once by
    build_fixtures.py): list constructor when it has several units."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    mols = [_mol(m, x_default_2d=False) for m in rec['mols']]
    zp = len(mols)
    return MolCrystalData(molecule=mols if zp > 1 else mols[0], sg_ind=int(rec['sg']), z_prime=torch.tensor([zp]),
                          max_z_prime=max_z_prime or zp, cell_lengths=rec['cell_lengths'].clone(),
                          cell_angles=rec['cell_angles'].clone(), aunit_centroid=rec['aunit_centroid'].clone(),
                          aunit_orientation=rec['aunit_orientation'].clone(),
                          aunit_handedness=rec['aunit_handedness'].clone(), do_box_analysis=True)


def collate(rows, max_z_prime=None):
    from mxtaltools.dataset_utils.utils import collate_data_list
    return collate_data_list(rows, max_z_prime=max_z_prime)


def build_F1():
    return collate([f1_row(r) for r in fixtures()['F1']])


def build_F3():
    return f3_batch(fixtures()['F3'])


def build_F3b():
    return collate([f3b_row(r) for r in fixtures()['F3b']])


def build_F4a():
    """Phenacetin (Z'=1, constructor width 1, padded by collate) then an ingestion Z'=2 row."""
    fx = fixtures()
    return collate([f1_row(fx['F1'][0], reference_handedness=False), f3_list_row(fx['F3'][1])], max_z_prime=2)


def build_F4b():
    """As F4a, with the Z'=1 row built at width 2 by the constructor (its own pad values)."""
    fx = fixtures()
    return collate([f1_row(fx['F1'][0], max_z_prime=2, reference_handedness=False), f3_list_row(fx['F3'][1])],
                   max_z_prime=2)


def build_two(name):
    """A two-compound audit crystal (``fixtures()['AUDIT'][name]``) as a one-row batch at max_z_prime 2."""
    return collate([synthetic_row(fixtures()['AUDIT'][name])], max_z_prime=2)


def f3_deposited_batch():
    """The F3 rows as BARE ingestion input (``from_arrays`` without a pose, float64): each row's stored atoms
    wrapped atom by atom into the cell (so molecules are cut at the cell faces), the two units interleaved atom
    by atom, and unit 1's atoms in reverse order -- so ``find_molecules`` must find the components, make them
    whole and re-index unit 1 onto unit 0's atom order before ``parameterize_pose`` and ``check_rebuild``."""
    from mxtaltools.common.geometry_utils import batch_compute_fractional_transform
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    recs = fixtures()['F3']
    xyz, z, counts = [], [], []
    for r in recs:
        n = len(r['z']) // 2
        order = torch.stack([torch.arange(n), n + torch.arange(n).flip(0)], 1).reshape(-1)
        T_fc, T_cf, _ = batch_compute_fractional_transform(r['cell_lengths'][None].double(),
                                                           r['cell_angles'][None].double())
        f = r['pos'][order].double() @ T_cf[0].T
        xyz.append((f - torch.floor(f)) @ T_fc[0].T)
        z.append(r['z'][order])
        counts.append(len(order))
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), torch.tensor(counts).cumsum(0)])
    cell = torch.stack([torch.cat([r['cell_lengths'], r['cell_angles']]) for r in recs])
    return MolCrystalData.from_arrays(cell, torch.cat(xyz), torch.cat(z), ptr=ptr,
                                      sg=torch.tensor([int(r['sg']) for r in recs]),
                                      ops=[r['ops'].numpy() for r in recs], dtype=torch.float64,
                                      identifier=[r['identifier'] for r in recs])


#: the ingestion step arguments of the csd_v1 recipe (crystal_recipes.CSD_V1), except bonds='distance' (no file)
INGEST_ARGS = (('find_molecules', dict(bonds='distance', bond_scale=(1.15, 1.35), iso_cap=5000, special_tol=0.1)),
               ('parameterize_pose', dict(handedness='measured', degenerate_tol=1e-4, handedness_tol=1e-3)),
               ('check_rebuild', dict(max_dev=0.05, mean_dev=0.01, match_elements=True)))

#: groups a main case carries besides box / canonicalize / latent and the per-std / per-cutoff ones
_MAIN_EXTRA = {
    'F1': ('reduction', 'helpers', 'zp_doubling', 'stdcell', 'reparam'),
    'F3': ('reduction', 'helpers', 'stdcell', 'reparam', 'ingest'),
    'F3b': ('helpers', 'zp_doubling', 'stdcell', 'reparam'),
    'F4a': ('helpers',),
    'F4b': (),
}
#: main cases whose exp6 energy is pinned (per std_orientation and cutoff)
_EXP6_CASES = ('F1', 'F3', 'F3b')

#: name -> (builder, std_orientation values pinned, description, kind); kind 'main' or 'two' (two compounds)
CASES = OrderedDict([
    ('F1', (build_F1, (True, False),
            "4 Z'=1 csd_fd_v5 rows (C1 molecules; PYRAZB21 phenacetin sg14, YIRLEU sg2, OYIQIX sg9, HIFQUJ sg14), "
            "body frame, so both std_orientation values reproduce the crystal", 'main')),
    ('F3', (build_F3, (True,),
            "2 Z'=2 rows of the ingestion chain (r2 store, status ok, per-unit conformers, mixed handedness; "
            "AHUMIZ sg2, AFEVOA sg14) via from_arrays; posed atoms, so std_orientation=True only", 'main')),
    ('F3b', (build_F3b, (True, False),
             "2 identical-copy Z'=2 rows (zp_doubling.double_zp1_crystal of the two sg14 F1 rows, the layout "
             "the crystal search makes). std_orientation=True reproduces both parents; False reproduces HIFQUJ "
             "(handedness +1) but not PYRAZB21, whose doubled units have handedness -1 that a False build ignores "
             "(audit #30): that row's False pins are the builder's own behaviour, not the crystal", 'main')),
    ('F4a', (build_F4a, (True,),
             "mixed batch at max_z_prime=2: phenacetin Z'=1 (collate-padded slot) then AFEVOA Z'=2 (list constructor)",
             'main')),
    ('F4b', (build_F4b, (True,),
             "as F4a with the Z'=1 row padded by the constructor (centroid 0, orientation 1, handedness 1)", 'main')),
    # two compounds (audit #1, #15, #19): today's WRONG outputs, pinned so that a change from one wrong value to
    # another is visible; stages 2, 4 and 7 re-pin them deliberately when they change them on purpose
    ('two_eth_meth', (lambda: build_two('two_eth_meth'), (True, False),
                      "ethanol (9) + methane (5) at Z'=2, sg 2 (audit probe P5): split 7 + 7 until stage 2, refused "
                      "since (its unit map holds units of 9 and 5 atoms; #1)", 'two')),
    ('two_eth_prop', (lambda: build_two('two_eth_prop'), (True, False),
                      "ethanol (9) + propane (11) at Z'=2, sg 2 (two_eth_meth's cell and slots; >= 3 heavy atoms "
                      "per unit, so each unit's frame is defined): split 10 + 10 until stage 2, refused since "
                      "(units of 9 and 11 atoms; #1)", 'two')),
    ('two_eth_ether', (lambda: build_two('two_eth_ether'), (True, False),
                       "ethanol + dimethyl ether at Z'=2, sg 2 (audit probe P4): equal counts, 4 of 36 MLIP unit-cell "
                       "elements wrong today (#15)", 'two')),
    ('two_pyr_flb', (lambda: build_two('two_pyr_flb'), (True, False),
                     "2-pyridone + fluorobenzene at Z'=2, sg 14 (audit probe2 a): equal counts, 8 of 96 MLIP "
                     "unit-cell elements wrong today (#15)", 'two')),
])


def build_case(name):
    return CASES[name][0]()


# ---------------------------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------------------------

def _std_tag(std):
    return f'std{int(bool(std))}'


def _cut_tag(cutoff):
    return f'cut{int(cutoff)}'


def case_groups(case) -> list:
    """Pin groups of a case, in order. A group is computed by one call of compute_group."""
    stds, kind = CASES[case][1], CASES[case][3]
    if kind == 'two':
        out = ['box', 'canonicalize', 'split']
        for s in stds:
            out.append(f'ucell/{_std_tag(s)}')
            out.append(f'elements/{_std_tag(s)}')
            for c in CUTOFFS:
                out.append(f'analyze/{_std_tag(s)}/{_cut_tag(c)}')
        return out
    out = ['box', 'canonicalize', 'latent'] + list(_MAIN_EXTRA[case])
    for s in stds:
        out.append(f'ucell/{_std_tag(s)}')
        out.append(f'mlip/{_std_tag(s)}')
        for c in CUTOFFS:
            out.append(f'cluster/{_std_tag(s)}/{_cut_tag(c)}')
            out.append(f'analyze/{_std_tag(s)}/{_cut_tag(c)}')
            if case in _EXP6_CASES:
                out.append(f'exp6/{_std_tag(s)}/{_cut_tag(c)}')
            for mode in RDF_MODES:
                out.append(f'rdf/{_std_tag(s)}/{_cut_tag(c)}/{mode}')
    return out


def group_kind(group: str) -> str:
    return group.split('/')[0]


def _parse(group):
    parts = group.split('/')
    std = cutoff = mode = None
    for p in parts[1:]:
        if p.startswith('std'):
            std = bool(int(p[3:]))
        elif p.startswith('cut'):
            cutoff = float(p[3:])
        else:
            mode = None if p == 'None' else p
    return parts[0], std, cutoff, mode


class Raised:
    """A pinned exception: the type is pinned, the message kept for information."""

    def __init__(self, exc):
        self.type = type(exc).__name__
        self.message = str(exc)[:400]


def _guard(out, prefix, fn):
    """Run fn() -> {name: value}; on an exception pin it as ``prefix/raises``."""
    try:
        res = fn()
    except Exception as e:  # pinned, deliberately broad: a new exception type is a behaviour change
        out[f'{prefix}raises'] = Raised(e)
        return
    for k, v in res.items():
        out[f'{prefix}{k}'] = v


_ROW_KEYS = ('num_atoms', 'z_prime', 'sym_mult', 'sg_ind', 'radius', 'mass', 'mol_volume', 'cell_lengths',
             'cell_angles', 'T_fc', 'T_cf', 'cell_volume', 'packing_coeff', 'density', 'aunit_centroid',
             'aunit_orientation', 'aunit_handedness', 'pos', 'z', 'x', 'ptr', 'batch')


def _attrs(b, keys):
    out = OrderedDict()
    for k in keys:
        v = b[k] if k in b._store else None
        out[k] = v
    out['max_z_prime'] = int(b.max_z_prime)
    return out


def _sorted_edges(ei, n_nodes):
    if ei.numel() == 0:
        return ei.clone()
    key = ei[1].long() * (int(n_nodes) + 1) + ei[0].long()
    return ei[:, torch.argsort(key)]


def _per_graph_counts(index, batch, num_graphs):
    return torch.bincount(batch[index], minlength=num_graphs)


def _group_box(b):
    out = _attrs(b, _ROW_KEYS)
    if 'packing_coeff' not in b._store or b.packing_coeff is None:
        # from_arrays (F3) does not run box_analysis: pin it on a clone, where the #6 sum convention of the
        # row's radius / mass / mol_volume over its units shows
        c = b.clone()
        c.box_analysis()
        for k in ('cell_volume', 'packing_coeff', 'density', 'T_fc', 'T_cf'):
            out[f'box_analysis_on_clone/{k}'] = c[k]
    return out


#: cell variants for the reduction penalty: (name, lengths permutation, angle overrides in degrees (None keeps),
#: length scale); every fixture cell is a reduced CSD cell (penalty 0), so variants off the reduced setting pin it
_REDUCTION_VARIANTS = (
    ('own', (0, 1, 2), (None, None, None), (1.0, 1.0, 1.0)),
    ('swap_ac', (2, 1, 0), (None, None, None), (1.0, 1.0, 1.0)),
    ('swap_ab', (1, 0, 2), (None, None, None), (1.0, 1.0, 1.0)),
    ('alpha80', (0, 1, 2), (80.0, None, None), (1.0, 1.0, 1.0)),
    ('beta80', (0, 1, 2), (None, 80.0, None), (1.0, 1.0, 1.0)),
    ('obtuse_100_110_120', (0, 1, 2), (100.0, 110.0, 120.0), (1.0, 1.0, 1.0)),
    ('acute_80_85_75', (0, 1, 2), (80.0, 85.0, 75.0), (1.0, 1.0, 1.0)),
    ('stretched', (0, 1, 2), (None, None, None), (1.0, 1.6, 0.7)),
)
#: None = each row's own space group; then one space group per crystal system (two monoclinic settings)
_REDUCTION_SGS = (None, 1, 2, 5, 14, 19, 92, 146, 169, 198)


def _group_reduction(b):
    """analyze(['reduction_en']) (no build: it reads cell, sg_ind) on every row under cell variants and space
    groups that take the cell off its reduced setting, so the penalty's walls are pinned, not only its zero."""
    out = OrderedDict()
    for name, perm, angles, scale in _REDUCTION_VARIANTS:
        lengths = b.cell_lengths[:, list(perm)] * torch.tensor(scale, dtype=b.cell_lengths.dtype)
        ang = b.cell_angles[:, list(perm)].clone()
        for i, a in enumerate(angles):
            if a is not None:
                ang[:, i] = a * math.pi / 180
        for sg in _REDUCTION_SGS:
            c = b.clone()
            c.cell_lengths = lengths.clone()
            c.cell_angles = ang.clone()
            if sg is not None:
                c.sg_ind = torch.full_like(b.sg_ind, sg)
            out[f'{name}/sg{"own" if sg is None else sg}'] = c.analyze(['reduction_en'])['reduction_en']
    return out


def _all_keys(b, prefix=''):
    """Every stored value of a batch or data object, and its PyG slice / increment bookkeeping."""
    out = OrderedDict()
    for k in sorted(b._store.keys()):
        out[f'{prefix}{k}'] = b._store[k]
    for name in ('_slice_dict', '_inc_dict'):
        d = b.__dict__.get(name)
        if d is not None:
            for k in sorted(d):
                out[f'{prefix}{name}/{k}'] = d[k]
    out[f'{prefix}num_graphs'] = int(b.num_graphs) if getattr(b, 'is_batch', False) else None
    return out


def _group_helpers(b):
    """The batch helpers on an ordinary crystal batch (audit #3; stage 3 rewrites them): subsample in reverse
    order and to one row, append a clone, split to a list, and collate that list back, every key pinned."""
    out = OrderedDict()
    _guard(out, 'subsample_10/', lambda: _all_keys(b.clone().subsample_new_batch([1, 0])))
    _guard(out, 'subsample_0/', lambda: _all_keys(b.clone().subsample_new_batch([0])))
    _guard(out, 'append_clone/', lambda: _all_keys(b.clone().append_batch(b.clone())))

    def to_list():
        res = OrderedDict()
        rows = b.clone().batch_to_list()
        res['n'] = len(rows)
        for i, r in enumerate(rows):
            res.update(_all_keys(r, f'row{i}/'))
        return res
    _guard(out, 'batch_to_list/', to_list)
    _guard(out, 'relist/', lambda: _all_keys(collate(b.clone().batch_to_list(), max_z_prime=int(b.max_z_prime))))
    return out


def _group_zp_doubling(case):
    """zp_doubling (audit #15 site): F1 -> double_zp1_crystal of each sg 14 row (params and the doubled crystal);
    F3b -> _unit_cell_frac of each doubled row (fractional unit cell, its element tiling, T_fc)."""
    from mxtaltools.crystal_building.zp_doubling import double_zp1_crystal, _unit_cell_frac
    out = OrderedDict()
    if case == 'F1':
        for r in fixtures()['F1']:
            if int(r['sg']) != 14:
                continue
            tag = f'{r["identifier"]}/'

            def run(r=r):
                doubled, params = double_zp1_crystal(f1_row(r), reduce=True, origin='best', check=True)
                res = OrderedDict((f'params/{k}', v) for k, v in params.items())
                res.update(_attrs(doubled, ('cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation',
                                            'aunit_handedness', 'pos', 'z', 'x', 'num_atoms', 'z_prime', 'radius',
                                            'mass', 'mol_volume', 'T_fc', 'packing_coeff', 'density')))
                return res
            _guard(out, tag, run)
    else:
        for i, r in enumerate(fixtures()['F3b']):
            def run(r=r):
                frac, z, t_fc = _unit_cell_frac(f3b_row(r))
                return OrderedDict(frac=frac, z=z, t_fc=t_fc)
            _guard(out, f'row{i}/', run)
    return out


def _group_stdcell(b):
    """compute_standard_cell() on a clone (spglib; it labels unit-cell rows by tiling z, an audit #15 site)."""
    out = OrderedDict()

    def run():
        s = b.clone().compute_standard_cell()
        res = _attrs(s, ('cell_lengths', 'cell_angles', 'T_fc', 'aunit_centroid', 'aunit_orientation',
                         'aunit_handedness', 'pos', 'z', 'is_well_defined', 'unit_cell_pos', 'unit_cell_batch'))
        res['symmetry_operators'] = [np.asarray(o) for o in s.symmetry_operators]
        return res
    _guard(out, '', run)
    return out


def _group_reparam(b):
    """mol2ucell(std_orientation=True) then reparameterize_unit_cell() (audit #18), on a clone."""
    def run():
        c = b.clone()
        c.mol2ucell(std_orientation=True)
        cen, ori, hand, wd, pos = c.reparameterize_unit_cell()
        return OrderedDict(aunit_centroid=cen, aunit_orientation=ori, handedness=hand,
                           is_well_defined=wd if torch.is_tensor(wd) else list(wd), pos=pos)
    out = OrderedDict()
    _guard(out, '', run)
    return out


def _group_ingest():
    """The ingestion chain's structural steps (audit #9, #10: equal blocks at crystal_steps.py parameterize_pose
    and check_rebuild) on the F3 rows given as bare deposited sets (``f3_deposited_batch``): every key after each
    step."""
    out = OrderedDict()
    state = {'b': f3_deposited_batch()}
    for step, kw in INGEST_ARGS:
        def run(step=step, kw=kw):
            state['b'] = getattr(state['b'], step)(**kw)
            return _all_keys(state['b'])
        _guard(out, f'{step}/', run)
        if f'{step}/raises' in out:
            break
    return out


def _group_exp6(b, std, cutoff):
    def run():
        return OrderedDict(exp6=b.clone().analyze(['exp6'], cutoff=cutoff, std_orientation=std)['exp6'])
    out = OrderedDict()
    _guard(out, '', run)
    return out


def _group_split(b):
    """split_to_zp1_batch() (audit #1, #11): the per-unit graphs the builder makes."""
    s = b.clone().split_to_zp1_batch()
    return _attrs(s, ('num_atoms', 'pos', 'z', 'x', 'batch', 'ptr', 'z_prime', 'aunit_centroid', 'aunit_orientation',
                      'aunit_handedness', 'radius', 'mass', 'mol_volume', 'sym_mult'))


def _group_elements(b, std):
    """Per MLIP input builder (and ase 'unit cell'): the elements and positions it hands a model for each crystal,
    and how many rows carry the wrong element against the geometric truth (``element_truth``)."""
    out = OrderedDict()
    for which in ASSEMBLERS:
        def run(which=which):
            c = b.clone()
            c.mol2ucell(std_orientation=std)
            res = OrderedDict()
            for g, (pos, z) in enumerate(assembled_elements(c, which, std)):
                a, e = int(c.ptr[g]), int(c.ptr[g + 1])
                truth = element_truth(pos, c.pos[a:e], c.z[a:e], c.T_fc[g], c.symmetry_operators[g])
                res[f'g{g}/elements'] = z.long()
                res[f'g{g}/positions'] = pos
                res[f'g{g}/n_wrong'] = int((truth != z.long()).sum())
            return res
        _guard(out, f'{which}/', run)
    return out


def _group_canonicalize(b):
    c = b.clone()
    c.canonicalize_zp_aunits()
    keys = ['pos', 'z', 'x', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness']
    keys += [k for k in ('mol_ind', 'unit_radius', 'unit_mass', 'unit_volume') if k in c._store]
    out = _attrs(c, keys)
    if 'unit_smiles' in c._store:
        out['unit_smiles'] = [list(v) for v in c.unit_smiles]
    return out


def _group_latent(b):
    out = OrderedDict()
    for gauge in (True, False):
        tag = f'gauge{int(gauge)}/'

        def run():
            c = b.clone()
            lat = c.latent_params(gauge_fix_free_axes=gauge)
            res = OrderedDict(latents=lat)
            for k in ('aunit_centroid', 'aunit_orientation', 'aunit_handedness', 'pos', 'z'):
                res[f'after_forward/{k}'] = c[k]
            inv = c.clone()
            inv.latent_to_cell_params(lat)
            for k in ('cell_lengths', 'cell_angles', 'aunit_centroid', 'aunit_orientation', 'aunit_handedness',
                      'T_fc', 'cell_volume', 'packing_coeff'):
                res[f'inverse/{k}'] = inv[k]
            return res

        _guard(out, tag, run)
    return out


def _group_ucell(b, std):
    c = b.clone()
    c.mol2ucell(std_orientation=std)
    return OrderedDict(pos=c.pos, z=c.z, unit_cell_pos=c.unit_cell_pos, unit_cell_batch=c.unit_cell_batch,
                       unit_cell_mol_ind=c.unit_cell_mol_ind)


def _group_cluster(b, std, cutoff):
    cl = b.clone().mol2cluster(cutoff=cutoff, std_orientation=std)
    out = OrderedDict()
    for k in ('pos', 'z', 'x', 'aux_ind', 'mol_ind', 'ptr', 'batch'):
        out[f'cluster/{k}'] = cl[k] if k in cl._store else None
    out['cluster/unit_cell_pos'] = cl.unit_cell_pos
    out['cluster/unit_cell_batch'] = cl.unit_cell_batch
    cl.construct_radial_graph(cutoff=cutoff)
    ed = cl.edges_dict
    n = cl.pos.shape[0]
    out['graph/intra_edges_sorted'] = _sorted_edges(ed['edge_index'], n)
    out['graph/inter_edges_sorted'] = _sorted_edges(ed['edge_index_inter'], n)
    out['graph/intra_count_per_graph'] = _per_graph_counts(ed['edge_index'][1], cl.batch, cl.num_graphs)
    out['graph/inter_count_per_graph'] = _per_graph_counts(ed['edge_index_inter'][1], cl.batch, cl.num_graphs)
    out['graph/inter_dist_raw_order'] = ed['intermolecular_dist']
    return out


def _group_analyze(b, std, cutoff):
    res = b.clone().analyze(['reduction_en', 'elj'], cutoff=cutoff, std_orientation=std)
    return OrderedDict(reduction_en=res['reduction_en'], elj=res['elj'])


def _rdf_pairs(d):
    return [[str(k), str(v)] for k, v in d.items()] if isinstance(d, dict) else str(d)


def _group_rdf(b, std, cutoff, mode):
    rdf, bins, pairs = b.clone().analyze(['rdf'], cutoff=cutoff, std_orientation=std, rdf_mode=mode)['rdf']
    return OrderedDict(rdf=rdf, bins=bins.detach(), pairs=_rdf_pairs(pairs))


def stub_energy(pos, z, graph, num_graphs, cell, n_edges):
    """A deterministic, non-zero 'energy' per graph computed from a model's input, in float64:
    sum over the graph's atoms of 1e-3 (x + 2y + 3z) + 1e-2 Z, plus 1e-3 times the sum of its cell entries,
    plus 1e-4 times its edge count. Orientation-, element-, cell- and graph-sensitive, so the gas leg (one
    molecule, vacuum cell) and the crystal leg (the unit cell) get different values, and the arithmetic that
    turns them into a lattice energy (audit #26) is pinned through the result."""
    w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    per_atom = 1e-3 * (pos.detach().double() @ w) + 1e-2 * z.detach().double()
    e = torch.zeros(num_graphs, dtype=torch.float64).index_add_(0, graph.detach().long(), per_atom)
    if isinstance(cell, (list, tuple)):     # the per-graph MACE builders hand over a list of [3, 3] arrays
        cell = torch.stack([torch.as_tensor(np.asarray(c)) for c in cell])
    e = e + 1e-3 * cell.detach().double().reshape(num_graphs, -1).sum(1)
    return e + 1e-4 * n_edges.detach().double()


class StubMACE:
    """Stands in for a MACE model: the two attributes the input builders read, and a forward that
    records the input dict it is given and returns ``stub_energy`` of it (dtype of the positions)."""

    def __init__(self, r_max=STUB_R_MAX, atomic_numbers=tuple(range(1, 101))):
        self.r_max = torch.tensor(float(r_max))
        self.atomic_numbers = torch.tensor(list(atomic_numbers))
        self.calls = []

    def __call__(self, input_data, compute_force=False, compute_stress=False):
        self.calls.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in input_data.items()})
        g = int(input_data['ptr'].numel() - 1)
        z = self.atomic_numbers[input_data['node_attrs'].argmax(dim=1)]
        n_edges = torch.bincount(input_data['batch'][input_data['edge_index'][0]], minlength=g)
        e = stub_energy(input_data['positions'], z, input_data['batch'], g, input_data['cell'], n_edges)
        return {'energy': e.to(input_data['positions'].dtype)}


#: the backbone cutoff the UMA stub reports (the external neighbour list is built at it)
STUB_UMA_CUTOFF = 6.0


class StubUMA:
    """Stands in for a fairchem predict unit: ``predict`` records the AtomicData batch it is given and
    returns float32 ``stub_energy`` of it. It carries the attributes the adapter reads to attach the
    EXTERNAL neighbour list (``inference_settings.external_graph_gen`` True, as the real crystal
    predictor's default USE_UMA_EXTERNAL_GRAPH sets it; ``model.module.backbone.cutoff``), so the graph
    UMA is handed on the live route is pinned too."""

    def __init__(self):
        self.calls = []
        self.inference_settings = SimpleNamespace(external_graph_gen=True)
        self.model = SimpleNamespace(module=SimpleNamespace(backbone=SimpleNamespace(cutoff=STUB_UMA_CUTOFF)))

    def predict(self, uma_batch):
        self.calls.append(uma_batch)
        g = int(uma_batch.natoms.numel())
        e = stub_energy(uma_batch.pos, uma_batch.atomic_numbers, uma_batch.batch, g, uma_batch.cell,
                        uma_batch.nedges)
        return {'energy': e.float()}


#: 'dataset' is the task name, which selects UMA's output head; 'sid' is fairchem's sample id
_UMA_FIELDS = ('pos', 'atomic_numbers', 'cell', 'pbc', 'natoms', 'batch', 'edge_index', 'cell_offsets', 'nedges',
               'charge', 'spin', 'fixed', 'tags', 'dataset', 'sid')
_MACE_FIELDS = ('edge_index', 'positions', 'shifts', 'unit_shifts', 'cell', 'node_attrs', 'batch', 'ptr', 'head',
                'pbc')


def _uma_fields(ad):
    out = OrderedDict()
    for f in _UMA_FIELDS:
        v = getattr(ad, f)
        out[f] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def _mace_fields(d, z_table):
    out = OrderedDict((f, d.get(f)) for f in _MACE_FIELDS)
    out['element_from_node_attrs'] = z_table[d['node_attrs'].argmax(dim=1)]
    return out


def _group_mlip(b, std):
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    from mace.tools.torch_geometric.dataloader import Collater
    from mxtaltools.mlip_interfaces.uma_utils import batch_to_fairchem_batch, batch_to_fairchem_atomicdata
    from mxtaltools.mlip_interfaces.AL_mace_utils import batch_to_mace_input_dict, batch_to_mace_atomicdata

    out = OrderedDict()
    stub = StubMACE()
    z_table = stub.atomic_numbers

    def built():
        c = b.clone()
        c.mol2ucell(std_orientation=std)
        return c

    def uma_vec():
        return _uma_fields(batch_to_fairchem_batch(built(), std, pbc=True))

    def uma_list():
        return _uma_fields(atomicdata_list_to_batch(batch_to_fairchem_atomicdata(built(), std, pbc=True)))

    def uma_rebuild():
        return _uma_fields(batch_to_fairchem_batch(b.clone(), std, pbc=True, force_rebuild=True))

    def mace_dict():
        return _mace_fields(batch_to_mace_input_dict(built(), False, stub, std, pbc=True), z_table)

    def mace_list():
        coll = Collater([None], [None])(batch_to_mace_atomicdata(built(), False, stub, std, pbc=True))
        return _mace_fields(coll.to_dict(), z_table)

    def route(key, computes, stub, fields, **kw):
        """analyze(computes) with a stub predictor: what it returns, the predictor energies it stores on the
        evaluated batch (``<key>_pot`` crystal leg, ``<key>_gas_pot`` gas leg), and every input the stub was
        handed (gas leg, then crystal leg) -- recorded even when the route raises."""
        res = OrderedDict()
        try:
            got, evaluated = b.clone().analyze(computes, predictor=stub, std_orientation=std, return_cluster=True,
                                               **kw)
            for k in computes:
                res[f'result/{k}'] = got[k]
            for k in (f'{key}_pot', f'{key}_gas_pot'):
                res[k] = evaluated[k] if k in evaluated._store else None
        except Exception as e:  # pinned, deliberately broad: a new exception type is a behaviour change
            res['raises'] = Raised(e)
        res['n_calls'] = len(stub.calls)
        for i, c in enumerate(stub.calls):
            for k, v in fields(c).items():
                res[f'call{i}/{k}'] = v
        return res

    def uma_route():
        return route('uma', ['uma'], StubUMA(), _uma_fields)

    def mace_route():
        return route('mace', ['mace'], StubMACE(), lambda d: _mace_fields(d, z_table))

    # the MLIP-on-cluster branch (_resolve_aunit_z / _mace_resolve_aunit_z read the aux_ind == 0 atoms), as
    # calibrate_qm9_energy.py and the paper evaluation use it: elj and the MLIP in one analyze call, cutoff 6
    def uma_cluster_route():
        return route('uma', ['elj', 'uma'], StubUMA(), _uma_fields, cutoff=6.0)

    def mace_cluster_route():
        return route('mace', ['elj', 'mace'], StubMACE(), lambda d: _mace_fields(d, z_table), cutoff=6.0)

    def ase_unit_cell():
        """common.ase_interface 'unit cell' assembly (CIF / figures; an audit #15 site): per crystal."""
        from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
        c = built()
        res = OrderedDict()
        for g in range(c.num_graphs):
            atoms = ase_mol_from_crystaldata(c, index=g, mode='unit cell')
            res[f'g{g}/positions'] = torch.as_tensor(np.asarray(atoms.get_positions()))
            res[f'g{g}/numbers'] = torch.as_tensor(np.asarray(atoms.numbers)).long()
            res[f'g{g}/cell'] = torch.as_tensor(np.asarray(atoms.cell))
        return res

    for name, fn in (('uma_vec', uma_vec), ('uma_list', uma_list), ('uma_rebuild', uma_rebuild),
                     ('mace_dict', mace_dict), ('mace_list', mace_list), ('uma_route', uma_route),
                     ('mace_route', mace_route), ('uma_cluster_route', uma_cluster_route),
                     ('mace_cluster_route', mace_cluster_route), ('ase_unit_cell', ase_unit_cell)):
        _guard(out, f'{name}/', fn)
    return out


def compute_group(case, group) -> OrderedDict:
    """{key: value} of one pin group; keys are prefixed with the group. Values are tensors, python
    values or ``Raised``. A fresh batch is built for every group, so groups cannot leak state."""
    kind, std, cutoff, mode = _parse(group)
    b = None if kind in ('ingest', 'zp_doubling') else build_case(case)
    out = OrderedDict()
    if kind == 'box':
        _guard(out, '', lambda: _group_box(b))
    elif kind == 'canonicalize':
        _guard(out, '', lambda: _group_canonicalize(b))
    elif kind == 'latent':
        out.update(_group_latent(b))
    elif kind == 'ucell':
        _guard(out, '', lambda: _group_ucell(b, std))
    elif kind == 'cluster':
        _guard(out, '', lambda: _group_cluster(b, std, cutoff))
    elif kind == 'analyze':
        _guard(out, '', lambda: _group_analyze(b, std, cutoff))
    elif kind == 'rdf':
        _guard(out, '', lambda: _group_rdf(b, std, cutoff, mode))
    elif kind == 'mlip':
        out.update(_group_mlip(b, std))
    elif kind == 'reduction':
        _guard(out, '', lambda: _group_reduction(b))
    elif kind == 'helpers':
        out.update(_group_helpers(b))
    elif kind == 'zp_doubling':
        out.update(_group_zp_doubling(case))
    elif kind == 'stdcell':
        out.update(_group_stdcell(b))
    elif kind == 'reparam':
        out.update(_group_reparam(b))
    elif kind == 'ingest':
        out.update(_group_ingest())
    elif kind == 'exp6':
        out.update(_group_exp6(b, std, cutoff))
    elif kind == 'split':
        _guard(out, '', lambda: _group_split(b))
    elif kind == 'elements':
        out.update(_group_elements(b, std))
    else:
        raise KeyError(group)
    return OrderedDict((f'{group}/{k}', v) for k, v in out.items())


# ---------------------------------------------------------------------------------------------
# encoding and comparison (bit for bit)
# ---------------------------------------------------------------------------------------------

def _canon(t: torch.Tensor) -> torch.Tensor:
    """Contiguous CPU copy; floating NaNs given one bit pattern and -0.0 made +0.0, so that byte
    equality is exactly torch.equal with NaN == NaN."""
    t = t.detach().to('cpu').contiguous()
    if t.is_floating_point():
        t = torch.where(torch.isnan(t), torch.full_like(t, float('nan')), t) + 0.0
    return t.clone()


def tensor_sha256(t: torch.Tensor) -> str:
    t = _canon(t)
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes() if t.numel() else b'')
    return h.hexdigest()


def _summary(t: torch.Tensor) -> dict:
    t = _canon(t)
    flat = t.reshape(-1)
    s = {'shape': list(t.shape), 'dtype': str(t.dtype)}
    if t.numel():
        tf = flat.double() if (t.is_floating_point() or t.dtype in (torch.int64, torch.int32, torch.int16, torch.int8,
                                                                     torch.uint8, torch.bool)) else None
        if tf is not None:
            fin = tf[torch.isfinite(tf)]
            s['sum'] = float(fin.sum()) if fin.numel() else 0.0
            s['n_nonfinite'] = int((~torch.isfinite(tf)).sum())
            s['head'] = [float(v) for v in tf[:8]]
    return s


def encode(value):
    """The stored form of one pinned value."""
    if isinstance(value, Raised):
        return {'kind': 'raises', 'type': value.type, 'message': value.message}
    if torch.is_tensor(value):
        t = _canon(value)
        if t.numel() * t.element_size() <= HASH_ABOVE_BYTES:
            return {'kind': 'tensor', 'value': t}
        return {'kind': 'digest', 'sha256': tensor_sha256(t), **_summary(t)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {'kind': 'value', 'value': value}
    if isinstance(value, (list, tuple, dict)):
        return {'kind': 'value', 'value': _plain(value)}
    if isinstance(value, np.ndarray):
        return encode(torch.as_tensor(value))
    if isinstance(value, np.generic):
        return {'kind': 'value', 'value': value.item()}
    raise TypeError(f'cannot pin a {type(value).__name__}')


def _plain(v):
    """A nested python value of lists, str-keyed dicts and plain scalars (tensors and arrays as nested lists);
    the type of each tensor / array is kept in front of its values, so a dtype change is a change."""
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if torch.is_tensor(v) or isinstance(v, np.ndarray):
        return [f'{type(v).__name__}:{v.dtype}:{list(v.shape)}', _plain(v.tolist())]
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def describe(rec) -> str:
    k = rec['kind']
    if k == 'tensor':
        t = rec['value']
        return f'tensor {str(t.dtype)} {list(t.shape)}'
    if k == 'digest':
        return f'tensor {rec["dtype"]} {rec["shape"]} (sha256 {rec["sha256"][:12]}..., sum {rec.get("sum")})'
    if k == 'raises':
        return f'raises {rec["type"]}: {rec["message"][:120]}'
    v = rec['value']
    s = repr(v)
    return f'value {s[:120]}{"..." if len(s) > 120 else ""}'


def compare(golden: dict, value) -> tuple:
    """(equal, detail). Exact: tensors must match in dtype, shape and every bit (NaN == NaN,
    -0.0 == 0.0); exceptions must have the same type; plain values must be equal."""
    new = encode(value)
    gk, nk = golden['kind'], new['kind']
    if gk == 'raises' or nk == 'raises':
        if gk == nk and golden['type'] == new['type']:
            return True, ''
        return False, f'golden {describe(golden)} | now {describe(new)}'
    if gk == 'value' or nk == 'value':
        if gk == nk and golden['value'] == new['value']:
            return True, ''
        return False, f'golden {describe(golden)} | now {describe(new)}'
    # tensors (either stored form)
    if gk == 'tensor' and nk == 'tensor':
        a, b = golden['value'], new['value']
        if a.dtype != b.dtype or a.shape != b.shape:
            return False, f'golden {describe(golden)} | now {describe(new)}'
        if torch.equal(a, b) or (a.is_floating_point() and torch.equal(torch.isnan(a), torch.isnan(b))
                                 and torch.equal(a[~torch.isnan(a)], b[~torch.isnan(b)])):
            return True, ''
        diff = (a.double() - b.double()).abs()
        diff = diff[torch.isfinite(diff)]
        return False, (f'{int((a != b).sum())} of {a.numel()} elements differ'
                       f'{"; max |diff| %.3e" % float(diff.max()) if diff.numel() else ""}')
    gsha = golden['sha256'] if gk == 'digest' else tensor_sha256(golden['value'])
    nsha = new['sha256'] if nk == 'digest' else tensor_sha256(new['value'])
    if gsha == nsha:
        return True, ''
    gs = golden if gk == 'digest' else _summary(golden['value'])
    ns = new if nk == 'digest' else _summary(new['value'])
    return False, (f'digest differs: golden {gs.get("dtype")} {gs.get("shape")} sum {gs.get("sum")} head '
                   f'{gs.get("head")} | now {ns.get("dtype")} {ns.get("shape")} sum {ns.get("sum")} head {ns.get("head")}')


# ---------------------------------------------------------------------------------------------
# oracle: float64 brute-force lattice sum
# ---------------------------------------------------------------------------------------------

def vdw_table(dtype=torch.float64):
    from mxtaltools.constants.atom_properties import VDW_RADII
    return torch.tensor(list(VDW_RADII.values()), dtype=dtype)


def elj_pair64(d, s, stiffness=2.5):
    """The eLJ pair energy of vdw_analysis.exponential_edgewise_lj_energy, in float64."""
    lj = 4.0 * ((s / d) ** 12 - (s / d) ** 6)
    k = stiffness / s
    a = (24.0 / s) / k
    rep = a * torch.exp(-k * (d - s)) - a
    return torch.where(d < s, rep, lj)


def oracle_pairs(units_pos, units_z, T_fc, ops, cutoff: float, band: float = ORACLE_BAND) -> dict:
    """Every intermolecular pair within ``cutoff`` of a crystal, by brute force in float64.

    The crystal is the set of molecules ``op_k(unit_u) + n``: every unit ``u`` of the asymmetric
    unit (``units_pos[u]``, Cartesian, posed), every symmetry operator ``k`` of ``ops`` (affine
    [m, 4, 4] on fractional coordinates) and every lattice translation ``n``. The translations are
    all those for which some pair can be within the cutoff (|df_i| <= cutoff * |row i of T_cf|).
    A reference molecule is ``identity(unit_u) + 0``; its atoms are the targets. A pair is (target
    atom, atom of any OTHER molecule) with distance strictly below ``cutoff`` (the radial graph's
    convention: torch_cluster.radius keeps d < r). Pairs with ``|d - cutoff| < band`` are counted
    as boundary pairs and kept apart, since float32 builders may fall on either side.

    Returns {'targets': [(dist [k] float64 sorted, partner z [k])] per target atom (units
    concatenated), 'boundary': int, 'elj': float64 eLJ over the pairs with d < cutoff,
    'n_translations': int}.
    """
    T_fc = T_fc.detach().double().cpu()
    T_cf = torch.linalg.inv(T_fc)
    ops = torch.as_tensor(np.asarray(ops), dtype=torch.float64)
    R, t = ops[:, :3, :3], ops[:, :3, 3]
    eye = torch.eye(3, dtype=torch.float64)
    ident = [k for k in range(len(ops)) if torch.allclose(R[k], eye) and float(t[k].abs().max()) < 1e-12]
    if len(ident) != 1:
        raise ValueError(f'expected one identity operator, found {ident}')
    k0 = ident[0]
    frac = [P.detach().double().cpu() @ T_cf.T for P in units_pos]
    zs = [torch.as_tensor(z).long().cpu() for z in units_z]
    images = [(u, k, f @ R[k].T + t[k]) for u, f in enumerate(frac) for k in range(len(ops))]
    all_f = torch.cat([g for _, _, g in images])
    ref_f = torch.cat(frac)
    reach = cutoff * T_cf.norm(dim=1) + 1e-9
    lo = torch.floor(ref_f.min(0).values - all_f.max(0).values - reach).long()
    hi = torch.ceil(ref_f.max(0).values - all_f.min(0).values + reach).long()
    trans = torch.cartesian_prod(*[torch.arange(int(lo[i]), int(hi[i]) + 1) for i in range(3)]).double()
    zero = int(torch.nonzero((trans == 0).all(1))[0])
    vdw = vdw_table()
    targets, boundary, energy = [], 0, 0.0
    for u, f in enumerate(frac):
        P = f @ T_fc.T                                                      # [n_u, 3]
        per_atom = [[] for _ in range(len(P))]
        for (u2, k, g) in images:
            Q = (g[None, :, :] + trans[:, None, :]) @ T_fc.T                # [T, n_u2, 3]
            d = (P[:, None, None, :] - Q[None]).norm(dim=-1)                # [n_u, T, n_u2]
            keep = d < cutoff + band
            if u2 == u and k == k0:
                keep[:, zero, :] = False
            if not bool(keep.any()):
                continue
            ia, _, ja = torch.nonzero(keep, as_tuple=True)
            dd = d[keep]
            zz = zs[u2][ja]
            for i in range(len(P)):
                m = ia == i
                if bool(m.any()):
                    per_atom[i].append((dd[m], zz[m]))
            inside = dd < cutoff
            boundary += int(((dd - cutoff).abs() < band).sum())
            s = vdw[zs[u][ia]] + vdw[zz]
            energy += float(elj_pair64(dd[inside], s[inside]).sum())
        for i in range(len(P)):
            if per_atom[i]:
                dd = torch.cat([a for a, _ in per_atom[i]])
                zz = torch.cat([b for _, b in per_atom[i]])
                o = torch.argsort(dd)
                targets.append((dd[o], zz[o]))
            else:
                targets.append((torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.long)))
    return {'targets': targets, 'boundary': boundary, 'elj': energy, 'n_translations': len(trans)}


def builder_pairs(cluster, g: int, posed_pos) -> dict:
    """The builder's intermolecular pairs of crystal ``g`` of an analysed cluster (``edges_dict``
    built), per target atom, ordered as ``posed_pos`` (the crystal's posed asymmetric unit, the
    oracle's targets): each inside atom is matched to the posed atom it sits on. Distances are
    recomputed in float64 from the cluster's float32 positions."""
    ei = cluster.edges_dict['edge_index_inter']
    inside = torch.nonzero((cluster.aux_ind == 0) & (cluster.batch == g)).flatten()
    P = posed_pos.detach().double().cpu()
    X = cluster.pos.detach().double().cpu()
    dmat = (X[inside][:, None, :] - P[None]).norm(dim=-1)
    match = dmat.argmin(dim=1)
    if float(dmat.min(dim=1).values.max()) > 1e-3 or len(set(match.tolist())) != len(P) or len(inside) != len(P):
        # RuntimeError, not AssertionError: a harness failure must not pass as a strict xfail's known failure
        raise RuntimeError(f'cluster inside atoms of crystal {g} do not coincide one-to-one with its posed atoms')
    targets = [None] * len(P)
    src_all, dst_all = ei[0].cpu(), ei[1].cpu()
    for a, i in zip(inside.tolist(), match.tolist()):
        src = src_all[dst_all == a]
        d = (X[src] - X[a]).norm(dim=1)
        o = torch.argsort(d)
        targets[i] = (d[o], cluster.z[src].cpu().long()[o])
    return {'targets': targets}


def compare_pair_sets(builder: dict, oracle: dict, cutoff: float, band: float = ORACLE_BAND,
                      tol: float = ORACLE_DIST_TOL) -> dict:
    """Per target atom, the builder's and the oracle's partners with |d - cutoff| >= band: counts,
    sorted distances (within ``tol``) and partner-element multisets. Returns {'missing': pairs
    the oracle has and the builder lacks, 'extra': the reverse, 'distance_mismatch': atoms whose
    sorted distances disagree beyond tol, 'element_mismatch': atoms whose partner elements
    disagree, 'n_oracle', 'n_builder', 'boundary_builder'}."""
    missing = extra = n_o = n_b = boundary_b = 0
    dist_bad, elem_bad, worst = [], [], []
    for i, ((db, zb), (do, zo)) in enumerate(zip(builder['targets'], oracle['targets'])):
        mb = (db - cutoff).abs() >= band
        mo = (do - cutoff).abs() >= band
        boundary_b += int((~mb).sum())
        db, zb, do, zo = db[mb], zb[mb], do[mo], zo[mo]
        n_o += len(do)
        n_b += len(db)
        if len(db) != len(do):
            if len(do) > len(db):
                missing += len(do) - len(db)
            else:
                extra += len(db) - len(do)
            # name the unmatched oracle pairs (nearest-distance greedy)
            worst.append((i, len(do) - len(db),
                          [round(float(x), 4) for x in do if bool(((db - x).abs() > tol).all())][:4]))
            continue
        if len(db) and float((db - do).abs().max()) > tol:
            dist_bad.append((i, float((db - do).abs().max())))
        if not torch.equal(torch.sort(zb).values, torch.sort(zo).values):
            elem_bad.append(i)
    return {'missing': missing, 'extra': extra, 'distance_mismatch': dist_bad, 'element_mismatch': elem_bad,
            'n_oracle': n_o, 'n_builder': n_b, 'boundary_builder': boundary_b, 'count_mismatch_atoms': worst}


def library_pair_energy64(d, z_target, z_partner):
    """eLJ pair energies by the library's own pair function (vdw_analysis.exponential_edgewise_lj_energy,
    stiffness 2.5), evaluated in float64."""
    from mxtaltools.analysis.vdw_analysis import exponential_edgewise_lj_energy
    dd = {'intermolecular_dist': d.double(), 'intermolecular_dist_atoms': [z_partner.long(), z_target.long()]}
    return exponential_edgewise_lj_energy(vdw_table(), dd, 2.5)


def interior_energy(targets, target_z, cutoff: float, band: float = ORACLE_BAND, library: bool = False) -> tuple:
    """(sum, sum of |.|) of the eLJ energy of the pairs of ``targets`` (per target atom (d, partner z))
    with d < cutoff and |d - cutoff| >= band, in float64; by the library's pair function when
    ``library``, else by ``elj_pair64``."""
    vdw = vdw_table()
    tot = absum = 0.0
    for (d, zp), zi in zip(targets, target_z):
        m = (d < cutoff) & ((d - cutoff).abs() >= band)
        if not bool(m.any()):
            continue
        zt = torch.full((int(m.sum()),), int(zi), dtype=torch.long)
        e = library_pair_energy64(d[m], zt, zp[m]) if library else elj_pair64(d[m], vdw[zt] + vdw[zp[m]])
        tot += float(e.sum())
        absum += float(e.abs().sum())
    return tot, absum


def boundary_energy(targets, target_z, cutoff: float, band: float = ORACLE_BAND) -> float:
    """float64 eLJ (library pair function) of the pairs of ``targets`` inside the boundary band."""
    tot = 0.0
    for (d, zp), zi in zip(targets, target_z):
        m = (d - cutoff).abs() < band
        if bool(m.any()):
            zt = torch.full((int(m.sum()),), int(zi), dtype=torch.long)
            tot += float(library_pair_energy64(d[m], zt, zp[m]).sum())
    return tot


def crystal_units(batch, g: int, unit_sizes=None):
    """(posed units [list of [n_u, 3]], their z) of crystal ``g`` from a batch on which mol2ucell
    ran; units are consecutive blocks of ``unit_sizes`` (default: z_prime equal blocks)."""
    a, b = int(batch.ptr[g]), int(batch.ptr[g + 1])
    pos, z = batch.pos[a:b], batch.z[a:b]
    if unit_sizes is None:
        zp = int(batch.z_prime[g])
        unit_sizes = [(b - a) // zp] * zp
    out_p, out_z, s = [], [], 0
    for n in unit_sizes:
        out_p.append(pos[s:s + n])
        out_z.append(z[s:s + n])
        s += n
    return out_p, out_z


# ---------------------------------------------------------------------------------------------
# element truth for the MLIP assembly
# ---------------------------------------------------------------------------------------------

def element_truth(unit_cell_pos, aunit_pos, aunit_z, T_fc, ops, tol: float = 2e-3) -> torch.Tensor:
    """The element of every unit-cell row of one crystal, found geometrically: the row must sit on
    ``op_k(aunit atom a) + n`` for some operator k, atom a and lattice vector n (within ``tol``
    Angstrom), and takes atom a's element. Independent of how the rows are ordered. Raises
    RuntimeError if a row matches no atom or atoms of different elements."""
    T_fc = T_fc.detach().double().cpu()
    T_cf = torch.linalg.inv(T_fc)
    ops = torch.as_tensor(np.asarray(ops), dtype=torch.float64)
    fa = aunit_pos.detach().double().cpu() @ T_cf.T                                   # [n, 3]
    img = torch.einsum('kij,nj->kni', ops[:, :3, :3], fa) + ops[:, None, :3, 3]       # [m, n, 3]
    img = img.reshape(-1, 3)
    zimg = torch.as_tensor(aunit_z).long().cpu().repeat(len(ops))
    fr = unit_cell_pos.detach().double().cpu() @ T_cf.T
    df = fr[:, None, :] - img[None]
    df = df - torch.round(df)
    dist = (df @ T_fc.T).norm(dim=-1)                                                  # [rows, m n]
    hit = dist < tol
    if not bool(hit.any(1).all()):
        raise RuntimeError(f'{int((~hit.any(1)).sum())} unit-cell rows sit on no image of an asymmetric-unit atom')
    out = torch.empty(len(fr), dtype=torch.long)
    for r in range(len(fr)):
        zs = torch.unique(zimg[hit[r]])
        if len(zs) != 1:
            raise RuntimeError(f'unit-cell row {r} sits on images of different elements {zs.tolist()}')
        out[r] = zs[0]
    return out


#: the MLIP input builders (and the ase 'unit cell' assembly) whose element layout audit #15 names
ASSEMBLERS = ('uma_vectorised', 'uma_list', 'mace_dict', 'mace_list', 'ase_unit_cell')


def assembled_elements(batch, which, std):
    """[(positions [n, 3], elements [n])] per crystal, as the MLIP input builder ``which`` hands them to a
    model (``batch`` must already carry its unit cell, e.g. after ``mol2ucell``)."""
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    from mace.tools.torch_geometric.dataloader import Collater
    from mxtaltools.mlip_interfaces.uma_utils import batch_to_fairchem_batch, batch_to_fairchem_atomicdata
    from mxtaltools.mlip_interfaces.AL_mace_utils import batch_to_mace_input_dict, batch_to_mace_atomicdata
    from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
    stub = StubMACE()
    out = []
    if which in ('uma_vectorised', 'uma_list'):
        ad = (batch_to_fairchem_batch(batch, std, pbc=True) if which == 'uma_vectorised'
              else atomicdata_list_to_batch(batch_to_fairchem_atomicdata(batch, std, pbc=True)))
        ptr = torch.cat([torch.zeros(1, dtype=torch.long), ad.natoms.cumsum(0)])
        for g in range(batch.num_graphs):
            out.append((ad.pos[ptr[g]:ptr[g + 1]], ad.atomic_numbers[ptr[g]:ptr[g + 1]]))
    elif which in ('mace_dict', 'mace_list'):
        d = (batch_to_mace_input_dict(batch, False, stub, std, pbc=True) if which == 'mace_dict'
             else Collater([None], [None])(batch_to_mace_atomicdata(batch, False, stub, std, pbc=True)).to_dict())
        z = stub.atomic_numbers[d['node_attrs'].argmax(dim=1)]
        for g in range(batch.num_graphs):
            out.append((d['positions'][d['ptr'][g]:d['ptr'][g + 1]], z[d['ptr'][g]:d['ptr'][g + 1]]))
    elif which == 'ase_unit_cell':
        for g in range(batch.num_graphs):
            atoms = ase_mol_from_crystaldata(batch, index=g, mode='unit cell')
            out.append((torch.as_tensor(np.asarray(atoms.get_positions())), torch.as_tensor(np.asarray(atoms.numbers))))
    else:
        raise KeyError(which)
    return out


# ---------------------------------------------------------------------------------------------
# known-failure helpers: where an exception came from, and whether it is a deliberate refusal
# ---------------------------------------------------------------------------------------------

def _mxt_root() -> Path:
    import mxtaltools
    return Path(mxtaltools.__file__).resolve().parent


def innermost_frame(exc) -> traceback.FrameSummary:
    return traceback.extract_tb(exc.__traceback__)[-1]


def raised_in(exc, function: str, module_suffix: str = None) -> bool:
    """True when the innermost frame of ``exc`` is the mxtaltools function ``function`` (and, if given, in a
    file whose path ends with ``module_suffix``): how a test names the library line it expects to fail."""
    f = innermost_frame(exc)
    path = Path(f.filename).resolve()
    if _mxt_root() not in path.parents:
        return False
    if module_suffix is not None and not path.as_posix().endswith(module_suffix):
        return False
    return f.name == function


#: the exception types a deliberate refusal may have (each survives ``python -O``)
REFUSAL_TYPES = (ValueError, RuntimeError, NotImplementedError)


def deliberate_refusal(exc) -> bool:
    """True when ``exc`` is a refusal the library MEANT: one of ``REFUSAL_TYPES`` (or a subclass, e.g. a library
    error class), raised by a ``raise`` statement in mxtaltools source. An accidental crash -- a shape error
    from a torch / numpy call (RuntimeError from einsum, cat or view), an IndexError, a bare ``assert`` (which
    ``python -O`` removes) -- is not one, wherever it surfaces."""
    if not isinstance(exc, REFUSAL_TYPES):
        return False
    f = innermost_frame(exc)
    path = Path(f.filename).resolve()
    return _mxt_root() in path.parents and _RAISE_LINE.search((f.line or '').strip()) is not None


_RAISE_LINE = re.compile(r'(^|[:;]\s*)raise(\s|\(|$)')


def describe_exception(exc) -> str:
    f = innermost_frame(exc)
    return f'{type(exc).__name__}: {str(exc)[:160]} (at {Path(f.filename).name}:{f.lineno} in {f.name})'
