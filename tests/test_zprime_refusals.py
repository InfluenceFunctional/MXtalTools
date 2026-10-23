"""Stage 1 of the Z'>1 units plan: refuse instead of guessing (gfn_diffusion
energy_sampling/docs/design/crystal_units_audit.md, section 5; items #1, #4, #12, #13 and section 7).

What is pinned here:

- every refusal on the Z'>1 build path is a ``ValueError`` raised by a ``raise`` statement in mxtaltools,
  naming the row and the reason, and it is still raised under ``python -O`` (a child interpreter runs every
  refusal with ``-O``; a bare ``assert`` would vanish there):
  ``split_to_zp1_batch`` (a single object; a row whose atoms are not z_prime equal blocks; a row with fewer pose
  slots than units), ``_index_mols`` (cluster construction on an unsplit Z'>1 graph), ``join_zp1_cluster_batch``
  (a split cluster with the wrong number of graphs), ``mol2ucell`` / ``mol2cluster`` (a row whose z_prime exceeds
  the batch's max_z_prime, or below 1, at any width), ``pose_aunit`` / ``get_aunit_positions`` (std_orientation
  without a handedness), ``build_unit_cell`` (a wide batch holding a row that is not one unit),
  ``collate_data_list`` (a max_z_prime below some row's z_prime, or wider than the rows' pose slots), and on the
  std_orientation path in common/geometry_utils.py ``batch_molecule_principal_axes_torch`` (atom counts that
  disagree with ``batch``) and ``scatter_compute_Ip`` (a failed eigh, e.g. a NaN coordinate: a RuntimeError);
- a batch of only Z'=1 rows at max_z_prime 2 builds the clusters, the unit cell and the scores of the width-1
  batch bit for bit, through mol2cluster / mol2ucell, through pose_aunit + build_unit_cell called directly, and
  through each MLIP input builder's own rebuild branch;
- the Z'>1 cutoff buffer is computed per row from that row's own units, never a pad slot: a Z'=1 row, or a Z'>1
  row narrower than its batch, gets the cluster it gets alone (stage 1 read the row's own centroid slots; since
  stage 5, 2026-10-23, it is the per-unit all-atom reach buffer, whose value tests/test_cluster_reach.py pins);
- a std_orientation=False build does not need aunit_handedness at any width;
- ``collate_data_list`` leaves its inputs untouched when it widens them, and its output is the one collating
  pre-widened rows gives.

The crystals are synthetic (two small molecules without symmetry, a triclinic cell); the characterization suite
(tests/characterization) pins the builder's outputs on real rows.
"""
import copy
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]

# two small molecules without point symmetry (>= 3 non-collinear heavy atoms each, so std_orientation has a frame);
# 'C' is 'B' without its last H: A (6) + C (7) = 13 atoms do not divide into two units
_ZA = torch.tensor([6, 6, 8, 7, 6, 1])
_PA = torch.tensor([[0., 0, 0], [1.5, 0, 0], [2.2, 1.2, 0], [-0.7, 1.2, 0.3], [2.1, -1.3, 0.8], [-0.5, -0.9, -0.4]])
_ZB = torch.tensor([6, 6, 6, 6, 8, 8, 1, 1])
_PB = torch.tensor([[0., 0, 0], [1.4, 0, 0], [2.1, 1.2, 0], [1.4, 2.4, 0.2], [3.5, 1.2, 0.1], [0.0, 2.4, -0.3],
                    [-0.5, -0.9, 0], [3.9, 0.3, 0]])
_MOLS = {'A': (_ZA, _PA), 'B': (_ZB, _PB), 'C': (_ZB[:7], _PB[:7])}
#: cell lengths (Angstrom) and angles (radians) of every synthetic crystal
_LENGTHS = torch.tensor([10.5, 11.0, 12.5])
_ANGLES = torch.tensor([1.62, 1.74, 1.52])
_CUTOFF = 6.0


def _mol(name):
    from mxtaltools.dataset_utils.data_classes import MolData
    z, pos = _MOLS[name]
    return MolData(z=z.clone(), pos=pos.clone(), x=torch.zeros(len(z), 1), do_mol_analysis=True)


def zp1_row(name='A', identifier='ONE', max_z_prime=1, sg=2, centroid=(0.23, 0.31, 0.27),
            orientation=(0.4, -0.3, 1.1)):
    """A Z'=1 crystal built by the constructor at ``max_z_prime`` slots (pads: centroid 0, orientation 1,
    handedness 1)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    return MolCrystalData(molecule=_mol(name), sg_ind=sg, cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                          aunit_centroid=torch.tensor(centroid), aunit_orientation=torch.tensor(orientation),
                          aunit_handedness=torch.tensor([1]), identifier=identifier, max_z_prime=max_z_prime,
                          do_box_analysis=True)


def zp2_row(names=('B', 'B'), identifier='TWO', max_z_prime=2, sg=2,
            centroids=(0.12, 0.20, 0.31, 0.62, 0.55, 0.71), orientations=(0.3, 0.2, 1.0, -1.2, 0.5, 0.4)):
    """A Z'=2 crystal through the list constructor (one MolData per unit, fresh 0-dim row sums)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    return MolCrystalData(molecule=[_mol(n) for n in names], sg_ind=sg, z_prime=torch.tensor([2]),
                          max_z_prime=max_z_prime, cell_lengths=_LENGTHS.clone(), cell_angles=_ANGLES.clone(),
                          aunit_centroid=torch.tensor(centroids), aunit_orientation=torch.tensor(orientations),
                          aunit_handedness=torch.tensor([1, -1]), identifier=identifier, do_box_analysis=True)


def _collate(rows, max_z_prime=None):
    from mxtaltools.dataset_utils.utils import collate_data_list
    return collate_data_list(rows, max_z_prime=max_z_prime)


# ------------------------------------------------------------------------------------------------
# the refusals: name -> (call, innermost mxtaltools function that must raise, words the message must hold)
# ------------------------------------------------------------------------------------------------

def _split_unequal_units():
    _collate([zp1_row(identifier='FIRST'), zp2_row(('A', 'C'), identifier='UNEQUAL')],
             max_z_prime=2).split_to_zp1_batch()


def _split_single_object():
    zp2_row().split_to_zp1_batch()


def _split_more_units_than_slots():
    b = _collate([zp2_row(identifier='WIDE')])
    b.aunit_centroid = b.aunit_centroid[:, :3]          # the row keeps z_prime 2 with one slot
    b.split_to_zp1_batch()


def _index_mols_on_an_unsplit_row():
    from mxtaltools.crystal_building.utils import _index_mols
    _index_mols(None, None, None, SimpleNamespace(z_prime=torch.tensor([1, 2])), None)


def _join_a_foreign_split():
    parent = _collate([zp2_row(identifier='PARENT')])            # 2 units
    foreign = _collate([zp1_row(identifier='OTHER')]).mol2cluster(cutoff=_CUTOFF)   # 1 graph
    parent.join_zp1_cluster_batch(foreign)


def _wider_than_batch(route):
    b = _collate([zp1_row(identifier='OK'), zp2_row(identifier='WIDE')], max_z_prime=2)
    b.max_z_prime = 1                                   # the batch claims one slot; row 1 holds two units
    if route == 'mol2ucell':
        b.mol2ucell()
    else:
        b.mol2cluster(cutoff=_CUTOFF)


def _pose_without_handedness():
    b = _collate([zp1_row()])
    del b.aunit_handedness
    b.pose_aunit(std_orientation=True)


def _collate_truncating_a_zp2_row():
    _collate([zp1_row(identifier='OK'), zp2_row(identifier='WIDE')], max_z_prime=1)


def _collate_wider_than_the_slots():
    _collate([zp1_row(identifier='NARROW')], max_z_prime=2)


def _z_prime_below_one(route, width, z0):
    """A row whose z_prime is below 1 (it holds no unit). At width 1 it used to be posed and scored as a Z'=1
    crystal, at width 2 refused by the split; both widths now refuse it before the build."""
    if width == 1:
        b = _collate([zp1_row(identifier='ZERO'), zp1_row('B', identifier='OK')])
    else:
        b = _collate([zp1_row(identifier='ZERO'), zp2_row(identifier='OK')], max_z_prime=2)
    b.z_prime = torch.tensor([z0, int(b.z_prime[1])])
    if route == 'mol2ucell':
        b.mol2ucell()
    else:
        b.mol2cluster(cutoff=_CUTOFF)


def _std_build_without_handedness_at_width2():
    """std_orientation=True at width 2 on a batch without aunit_handedness: refused by pose_aunit, as at width 1
    (the split used to subscript the missing value: TypeError)."""
    b = _collate([zp1_row(), zp2_row()], max_z_prime=2)
    del b.aunit_handedness
    b.mol2cluster(cutoff=_CUTOFF, std_orientation=True)


def _aunit_positions_without_handedness():
    """get_aunit_positions called directly (pose_aunit checks first) with std_orientation and no handedness: it used
    to stop at a bare assert, which -O strips."""
    from mxtaltools.crystal_building.utils import get_aunit_positions
    get_aunit_positions(_collate([zp1_row()]), std_orientation=True, mol_handedness=None)


def _build_unit_cell_on_an_unsplit_zp2_row():
    """build_unit_cell called directly on a width-2 batch holding a Z'=2 row (the MLIP input builders' rebuild
    branch does this only for z_prime.amax() == 1, but nothing stopped a caller): it used to stop in torch einsum,
    which met six slot columns, an accidental shape error."""
    b = _collate([zp1_row(identifier='ONE'), zp2_row(identifier='TWO')], max_z_prime=2)
    b.pose_aunit()
    b.build_unit_cell()


def _std_pose_with_a_nan_coordinate(width):
    """std_orientation=True aligns each unit to its heavy-atom principal axes; one non-finite coordinate makes the
    CPU eigh of the inertia tensors fail in scatter_compute_Ip. That used to stop at a bare `assert False` there, and
    under -O fall through to an UnboundLocalError."""
    if width == 1:
        b = _collate([zp1_row(identifier='NAN'), zp1_row('B', identifier='OK')])
    else:
        b = _collate([zp1_row(identifier='NAN'), zp2_row(identifier='OK')], max_z_prime=2)
    b.pos[1, 0] = float('nan')
    b.mol2cluster(cutoff=_CUTOFF, std_orientation=True)


def _principal_axes_with_wrong_atom_counts():
    """batch_molecule_principal_axes_torch (the std_orientation alignment) handed atom counts that disagree with
    `batch`: it used to stop at a bare assert, which -O strips."""
    from mxtaltools.common.geometry_utils import batch_molecule_principal_axes_torch
    batch = torch.tensor([0] * len(_ZA) + [1] * len(_ZB))
    batch_molecule_principal_axes_torch(torch.cat([_PA, _PB]), batch, 2, torch.tensor([7, 7]),
                                        atom_types=torch.cat([_ZA, _ZB]))


#: the function that refuses a row whose z_prime is outside 1..max_z_prime (mol2ucell / mol2cluster)
_SLOT_GUARD = '_refuse_z_prime_outside_slots'


REFUSALS = {
    'split_unequal_units': (_split_unequal_units, 'split_to_zp1_batch',
                            ('row 1 (UNEQUAL)', '13 atoms', 'z_prime 2')),
    'split_single_object': (_split_single_object, 'split_to_zp1_batch', ('single data object',)),
    'split_more_units_than_slots': (_split_more_units_than_slots, 'split_to_zp1_batch',
                                    ('row 0 (WIDE)', 'z_prime 2', '1 pose slot')),
    'index_mols_unsplit': (_index_mols_on_an_unsplit_row, '_index_mols', ('row 1', 'z_prime > 1')),
    'join_foreign_split': (_join_a_foreign_split, 'join_zp1_cluster_batch', ('1 graph', 'sum(z_prime) = 2')),
    'mol2ucell_row_wider_than_batch': (lambda: _wider_than_batch('mol2ucell'), _SLOT_GUARD,
                                       ('mol2ucell', 'row 1 (WIDE)', 'z_prime 2', 'max_z_prime = 1')),
    'mol2cluster_row_wider_than_batch': (lambda: _wider_than_batch('mol2cluster'), _SLOT_GUARD,
                                         ('mol2cluster', 'row 1 (WIDE)', 'z_prime 2', 'max_z_prime = 1')),
    'mol2cluster_row_with_z_prime_0': (lambda: _z_prime_below_one('mol2cluster', 1, 0), _SLOT_GUARD,
                                       ('mol2cluster', 'row 0 (ZERO)', 'z_prime 0', 'at least one unit')),
    'mol2ucell_row_with_z_prime_minus_1': (lambda: _z_prime_below_one('mol2ucell', 1, -1), _SLOT_GUARD,
                                           ('mol2ucell', 'row 0 (ZERO)', 'z_prime -1', 'at least one unit')),
    'mol2cluster_row_with_z_prime_0_at_width2': (lambda: _z_prime_below_one('mol2cluster', 2, 0), _SLOT_GUARD,
                                                 ('mol2cluster', 'row 0 (ZERO)', 'z_prime 0', 'at least one unit')),
    'pose_without_handedness': (_pose_without_handedness, 'pose_aunit', ('handedness',)),
    'pose_without_handedness_at_width2': (_std_build_without_handedness_at_width2, 'pose_aunit', ('handedness',)),
    'collate_truncating_zp2_row': (_collate_truncating_a_zp2_row, 'collate_data_list',
                                   ('max_z_prime=1', 'row 1 (WIDE)', 'z_prime 2')),
    'collate_wider_than_slots': (_collate_wider_than_the_slots, 'collate_data_list',
                                 ('max_z_prime=2', '1 pose slot')),
    'aunit_positions_without_handedness': (_aunit_positions_without_handedness, 'get_aunit_positions',
                                           ('handedness',)),
    'build_unit_cell_on_unsplit_zp2_row': (_build_unit_cell_on_an_unsplit_zp2_row, 'build_unit_cell',
                                           ('row 1 (TWO)', 'z_prime 2', 'split_to_zp1_batch')),
    'std_pose_with_nan_coordinate': (lambda: _std_pose_with_a_nan_coordinate(1), 'scatter_compute_Ip',
                                     ('eigh', 'non-finite', 'graph(s) [0]'), 'RuntimeError'),
    'std_pose_with_nan_coordinate_at_width2': (lambda: _std_pose_with_a_nan_coordinate(2), 'scatter_compute_Ip',
                                               ('eigh', 'non-finite', 'graph(s) [0]'), 'RuntimeError'),
    'principal_axes_wrong_atom_counts': (_principal_axes_with_wrong_atom_counts,
                                         'batch_molecule_principal_axes_torch',
                                         ('bincount', 'num_atoms', '[6, 8]', '[7, 7]')),
}


def _asserts_active() -> bool:
    try:
        assert False
    except AssertionError:
        return True
    return False


def run_refusals() -> dict:
    """Run every refusal; per case the exception type, the innermost frame's function and file, and the message.
    Run in this process by the tests below and, by ``under_O``, in a child interpreter started with ``-O``."""
    torch.set_num_threads(1)
    out = {'optimize': sys.flags.optimize, 'asserts_active': _asserts_active()}
    for name, (call, *_) in REFUSALS.items():
        try:
            call()
            out[name] = {'raised': None}
        except Exception as e:  # recorded, judged by the tests
            f = traceback.extract_tb(e.__traceback__)[-1]
            out[name] = {'raised': type(e).__name__, 'function': f.name, 'file': Path(f.filename).as_posix(),
                         'line': (f.line or '').strip(), 'message': str(e)}
    return out


def _judge(name, rec):
    _, where, words, *kind = REFUSALS[name]           # kind: the exception type, when not ValueError
    assert rec['raised'] == (kind[0] if kind else 'ValueError'), rec
    assert rec['function'] == where, rec
    assert '/mxtaltools/' in rec['file'], rec
    assert rec['line'].startswith('raise'), rec          # a raise statement, not an accidental crash
    for w in words:
        assert w in rec['message'], (w, rec['message'])


@pytest.fixture(scope='module')
def here():
    return run_refusals()


_CHILD = (
    "import importlib.util, json\n"
    "spec = importlib.util.spec_from_file_location('zprime_refusals_child', {path!r})\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "print('RESULT ' + json.dumps(mod.run_refusals()))\n"
)


@pytest.fixture(scope='module')
def under_O():
    """``run_refusals()`` in a child interpreter started with ``python -O`` (asserts stripped), CPU only, writing no
    bytecode."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(REPO)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    proc = subprocess.run([sys.executable, '-O', '-c', _CHILD.format(path=str(HERE))], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
    assert proc.returncode == 0 and lines, f'child failed ({proc.returncode}):\n{proc.stderr[-4000:]}'
    return json.loads(lines[-1][len('RESULT '):])


def test_child_runs_with_asserts_stripped(under_O):
    assert under_O['optimize'] >= 1 and under_O['asserts_active'] is False, under_O


def test_this_process_runs_with_asserts(here):
    assert here['asserts_active'] is True


@pytest.mark.parametrize('name', list(REFUSALS))
def test_refusal(here, name):
    _judge(name, here[name])


@pytest.mark.parametrize('name', list(REFUSALS))
def test_refusal_survives_python_O(under_O, name):
    _judge(name, under_O[name])


# ------------------------------------------------------------------------------------------------
# an all-Z'=1 batch at max_z_prime 2 (audit #12): the join is the identity
# ------------------------------------------------------------------------------------------------

def _two_zp1_rows(width):
    return [zp1_row('A', identifier='A1', max_z_prime=width),
            zp1_row('B', identifier='B1', max_z_prime=width, centroid=(0.61, 0.18, 0.44), orientation=(-1.0, 0.7, 0.2))]


_CLUSTER_KEYS = ('pos', 'z', 'x', 'aux_ind', 'mol_ind', 'batch', 'ptr', 'unit_cell_pos', 'unit_cell_batch')


def test_all_zp1_batch_at_width2_builds_the_width1_clusters():
    w1 = _collate(_two_zp1_rows(1))
    w2 = _collate(_two_zp1_rows(2), max_z_prime=2)
    assert w1.max_z_prime == 1 and w2.max_z_prime == 2
    c1, c2 = w1.mol2cluster(cutoff=_CUTOFF), w2.mol2cluster(cutoff=_CUTOFF)
    for k in _CLUSTER_KEYS:
        a, b = getattr(c1, k), getattr(c2, k)
        assert a.dtype == b.dtype and torch.equal(a, b), k
    # the returned cluster carries the batch's own pose slots, not the split's slot-0 view
    assert c2.max_z_prime == 2
    for k in ('aunit_centroid', 'aunit_orientation', 'aunit_handedness'):
        assert torch.equal(getattr(c2, k), getattr(w2, k)), k
    e1 = w1.analyze(['elj'], cutoff=_CUTOFF)['elj']
    e2 = w2.analyze(['elj'], cutoff=_CUTOFF)['elj']
    assert bool(torch.isfinite(e1).all()) and torch.equal(e1, e2), (e1, e2)


def test_identity_join_leaves_its_argument():
    b = _collate(_two_zp1_rows(2), max_z_prime=2)
    s = b.split_to_zp1_batch()
    s.pose_aunit()
    s.build_unit_cell()
    cl = s.build_cluster(cutoff=_CUTOFF)
    slots = cl.aunit_centroid.clone()
    out = b.join_zp1_cluster_batch(cl)
    assert out is not cl
    assert cl.aunit_centroid.shape == (2, 3) and torch.equal(cl.aunit_centroid, slots)
    assert out.aunit_centroid.shape == (2, 6) and torch.equal(out.pos, cl.pos)
    out.aunit_centroid[0, 0] += 1.0                    # the returned slots are its own, not the batch's
    assert not torch.equal(out.aunit_centroid, b.aunit_centroid)


_UCELL_KEYS = ('pos', 'unit_cell_pos', 'unit_cell_batch', 'unit_cell_mol_ind')


@pytest.mark.parametrize('std', [True, False], ids=['std1', 'std0'])
def test_all_zp1_batch_at_width2_builds_its_unit_cell_directly(std):
    """pose_aunit + build_unit_cell called directly on an all-Z'=1 batch at max_z_prime 2, as the MLIP input
    builders' own rebuild branch calls them whenever z_prime.amax() == 1: the width-1 batch's unit cell, and
    mol2ucell's at width 2, bit for bit (it used to stop in torch einsum, which met six slot columns). The batch keeps
    its own full-width slots."""
    w1, w2, u2 = _collate(_two_zp1_rows(1)), _collate(_two_zp1_rows(2), max_z_prime=2), \
        _collate(_two_zp1_rows(2), max_z_prime=2)
    slots = w2.aunit_centroid.clone()
    for b in (w1, w2):
        b.pose_aunit(std_orientation=std)
        b.build_unit_cell()
    u2.mol2ucell(std_orientation=std)
    for k in _UCELL_KEYS:
        a, b, c = getattr(w1, k), getattr(w2, k), getattr(u2, k)
        assert a.dtype == b.dtype == c.dtype and torch.equal(a, b) and torch.equal(a, c), k
    assert w2.aunit_centroid.shape == (2, 6) and torch.equal(w2.aunit_centroid, slots)


def _mlip_inputs(builder, b, std):
    """What MLIP input builder ``builder`` hands a model for batch ``b`` built through its own rebuild branch
    (force_rebuild=True; ``b`` carries no unit cell), as the characterization suite's field lists read them."""
    cl = _charlib()
    if builder in ('fairchem_batch', 'fairchem_atomicdata'):
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
        from mxtaltools.mlip_interfaces.uma_utils import batch_to_fairchem_batch, batch_to_fairchem_atomicdata
        ad = (batch_to_fairchem_batch(b, std, pbc=True, force_rebuild=True) if builder == 'fairchem_batch'
              else atomicdata_list_to_batch(batch_to_fairchem_atomicdata(b, std, pbc=True, force_rebuild=True)))
        return cl._uma_fields(ad)
    if builder == 'ase_ucell_list':
        from mxtaltools.mlip_interfaces.uma_utils import batch_to_ase_ucell_list
        return [{'positions': torch.as_tensor(np.asarray(a.get_positions())), 'numbers': np.asarray(a.numbers),
                 'cell': np.asarray(a.cell[:]), 'pbc': np.asarray(a.pbc)}
                for a in batch_to_ase_ucell_list(b, std, True, force_rebuild=True)]
    from mace.tools.torch_geometric.dataloader import Collater
    from mxtaltools.mlip_interfaces.AL_mace_utils import (batch_to_mace_input_dict, batch_to_mace_atomicdata,
                                                          batch_to_mace_atomicdata_hoisted)
    stub = cl.StubMACE()
    if builder == 'mace_dict':
        d = batch_to_mace_input_dict(b, True, stub, std, pbc=True)
    else:
        make = batch_to_mace_atomicdata if builder == 'mace_atomicdata' else batch_to_mace_atomicdata_hoisted
        d = Collater([None], [None])(make(b, True, stub, std, pbc=True)).to_dict()
    return cl._mace_fields(d, stub.atomic_numbers)


@pytest.mark.parametrize('builder', ['fairchem_batch', 'fairchem_atomicdata', 'ase_ucell_list', 'mace_dict',
                                     'mace_atomicdata', 'mace_atomicdata_hoisted'])
def test_mlip_rebuild_branch_takes_an_all_zp1_batch_at_width2(builder):
    """Each MLIP input builder's own rebuild branch (force_rebuild=True) dispatches on z_prime.amax(), not
    max_z_prime, so an all-Z'=1 batch at max_z_prime 2 reaches pose_aunit + build_unit_cell unsplit. It now hands the
    model exactly what the width-1 batch gives (it used to stop in torch einsum)."""
    with _charlib().deterministic():
        ref = _mlip_inputs(builder, _collate(_two_zp1_rows(1)), True)
        got = _mlip_inputs(builder, _collate(_two_zp1_rows(2), max_z_prime=2), True)
    assert _same(ref, got)


# ------------------------------------------------------------------------------------------------
# the Z'>1 cutoff buffer, per row from the row's own units (audit #13; the reach buffer since stage 5)
# ------------------------------------------------------------------------------------------------

def _all_slot_buffer(b):
    """The buffer as mol2cluster computed it before stage 1: the largest Cartesian distance between ANY two of the
    max_z_prime centroid slots, pads included, repeated over the row's units."""
    from mxtaltools.common.geometry_utils import fractional_transform
    frac = b.aunit_centroid.reshape(b.num_graphs * b.max_z_prime, 3)
    cart = fractional_transform(frac, b.T_fc.repeat_interleave(b.max_z_prime, dim=0)
                                ).reshape(b.num_graphs, b.max_z_prime, 3)
    d = (cart[:, :, None, :] - cart[:, None, :, :]).norm(dim=-1)
    return d.amax(dim=(1, 2)).repeat_interleave(b.z_prime, dim=0)


def _reach_buffer(b, std=True):
    """The buffer mol2cluster passes since stage 5 (2026-10-23): MolCrystalBuilding._unit_reach_buffer on the posed
    split. Stage 1 pinned its predecessor (the row's own-slot centroid distances, _zp_buffer); the pad and
    batch-mate properties below carry over, the value does not (tests/test_cluster_reach.py pins the definition). It
    reads each unit's molecule 0 from the split's built unit cell, as mol2cluster calls it."""
    s = b.split_to_zp1_batch()
    s.pose_aunit(std_orientation=std)
    s.build_unit_cell()
    return b._unit_reach_buffer(s)


def test_zp_buffer_of_a_zp1_row_ignores_its_pad():
    for rows in ([zp1_row(), zp2_row()],                  # pad written by collate (centroid 0)
                 [zp1_row(max_z_prime=2), zp2_row()]):    # pad written by the constructor (centroid 0)
        b = _collate(rows, max_z_prime=2)
        new, old = _reach_buffer(b), _all_slot_buffer(b)
        assert new.shape == (3,)
        assert float(new[0]) == 0.0 and float(old[0]) > 0.0       # the pad used to widen the Z'=1 row
        assert torch.equal(new[1:], _reach_buffer(_collate([zp2_row()])))   # the Z'=2 row: as alone


def test_zp_buffer_ignores_a_pad_beside_two_units():
    w2 = _collate([zp2_row()])
    w3 = _collate([zp2_row(max_z_prime=3)], max_z_prime=3)
    assert torch.equal(_reach_buffer(w3), _reach_buffer(w2))
    assert bool((_all_slot_buffer(w3) > _all_slot_buffer(w2)).all())   # the pad slot is the farthest centroid


def test_zp1_row_in_a_mixed_batch_builds_its_alone_cluster():
    alone = _collate([zp1_row()]).mol2cluster(cutoff=_CUTOFF)
    mixed = _collate([zp1_row(), zp2_row()], max_z_prime=2).mol2cluster(cutoff=_CUTOFF)
    m = mixed.batch == 0
    for k in ('pos', 'z', 'x', 'aux_ind', 'mol_ind'):
        assert torch.equal(getattr(mixed, k)[m], getattr(alone, k)), k


def _row_cluster_as_alone(wide_cluster, g, alone_cluster):
    """Row g of a cluster batch equals a one-row cluster: atoms, elements, features and aux_ind bit for bit, and
    mol_ind up to the constant offset of the molecules of the rows before it (it is numbered across the batch)."""
    m = wide_cluster.batch == g
    for k in ('pos', 'z', 'x', 'aux_ind'):
        a, b = getattr(wide_cluster, k)[m], getattr(alone_cluster, k)
        assert a.dtype == b.dtype and torch.equal(a, b), k
    mi = wide_cluster.mol_ind[m]
    assert torch.equal(mi - mi.min(), alone_cluster.mol_ind - alone_cluster.mol_ind.min())


def test_zp2_row_in_a_wider_batch_builds_its_alone_cluster():
    """A Z'=2 row in a width-3 batch (one pad slot): its buffer, cluster and eLJ are those it gets alone at width 2.
    Before stage 1 the pad's centroid (0, 0, 0) widened its buffer, so its cluster grew (README regeneration log)."""
    alone = _collate([zp2_row()])
    wide = _collate([zp1_row(max_z_prime=3), zp2_row(max_z_prime=3)], max_z_prime=3)
    assert torch.equal(_reach_buffer(wide)[1:], _reach_buffer(alone))
    _row_cluster_as_alone(wide.mol2cluster(cutoff=_CUTOFF), 1, alone.mol2cluster(cutoff=_CUTOFF))
    ea = alone.analyze(['elj'], cutoff=_CUTOFF)['elj']
    ew = _collate([zp1_row(max_z_prime=3), zp2_row(max_z_prime=3)], max_z_prime=3).analyze(['elj'], cutoff=_CUTOFF)['elj']
    assert bool(torch.isfinite(ea).all()) and torch.equal(ew[1:], ea), (ew, ea)


def _charlib():
    """The characterization suite's shared module (tests/characterization/charlib.py), loaded by path under its own
    name: its fixtures are real rows (data/fixtures.pt, no D:)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('charlib_for_zprime_refusals',
                                                  str(REPO / 'tests' / 'characterization' / 'charlib.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _f3_list_row_at(cl, rec, width):
    """charlib.f3_list_row (an ingestion Z'=2 row through the list constructor) built at ``width`` pose slots; the
    constructor pads the slots past z_prime."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    zp = int(rec['z_prime'])
    n = len(rec['z']) // zp
    units = [cl._mol({'z': rec['z'][k * n:(k + 1) * n], 'pos': rec['pos'][k * n:(k + 1) * n].float()})
             for k in range(zp)]
    return MolCrystalData(molecule=units, sg_ind=int(rec['sg']), z_prime=torch.tensor([zp]), max_z_prime=width,
                          cell_lengths=rec['cell_lengths'].float(), cell_angles=rec['cell_angles'].float(),
                          aunit_centroid=rec['aunit_centroid'].float(),
                          aunit_orientation=rec['aunit_orientation'].float(),
                          aunit_handedness=rec['aunit_handedness'].clone(), identifier=rec['identifier'],
                          do_box_analysis=True)


@pytest.mark.parametrize('k', [0, 1], ids=['AHUMIZ', 'AFEVOA'])
def test_real_zp2_row_in_a_width3_batch_scores_as_alone(k):
    """The characterization F3 rows (real ingestion Z'=2 rows, two different conformers per crystal) beside the F1
    phenacetin row in a max_z_prime=3 batch: each row's cluster and eLJ equal its own alone build bit for bit, at
    cutoffs 6 and 10 (std_orientation True; the F3 atoms are posed). Before stage 1 the Z'=2 row's pad slot widened
    its cluster; the measured shifts on store rows are in the characterization README's stage-1 log."""
    cl = _charlib()
    fx = cl.fixtures()
    rec = fx['F3'][k]
    with cl.deterministic():
        for cutoff in cl.CUTOFFS:
            make_wide = lambda: _collate([cl.f1_row(fx['F1'][0], max_z_prime=3, reference_handedness=False),
                                          _f3_list_row_at(cl, rec, 3)], max_z_prime=3)
            alone_zp2 = lambda: _collate([cl.f3_list_row(rec)])
            alone_zp1 = lambda: _collate([cl.f1_row(fx['F1'][0], reference_handedness=False)])
            wide = make_wide().mol2cluster(cutoff=cutoff)
            _row_cluster_as_alone(wide, 0, alone_zp1().mol2cluster(cutoff=cutoff))
            _row_cluster_as_alone(wide, 1, alone_zp2().mol2cluster(cutoff=cutoff))
            ew = make_wide().analyze(['elj'], cutoff=cutoff)['elj']
            e1 = alone_zp1().analyze(['elj'], cutoff=cutoff)['elj']
            e2 = alone_zp2().analyze(['elj'], cutoff=cutoff)['elj']
            assert bool(torch.isfinite(ew).all()), ew
            assert torch.equal(ew, torch.cat([e1, e2])), (cutoff, ew, e1, e2)


# ------------------------------------------------------------------------------------------------
# a build that does not read aunit_handedness (std_orientation=False) does not need it, at any width
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('rows', ['all_zp1', 'mixed'])
def test_std_false_build_at_width2_does_not_need_handedness(rows):
    """std_orientation=False never reads aunit_handedness; at width 2 a batch without it builds and scores exactly as
    the same batch with it, and no handedness is invented on the way (the split used to subscript the missing value:
    TypeError; at width 1 it already built). std_orientation=True without it is refused by pose_aunit (REFUSALS)."""
    def make():
        if rows == 'all_zp1':
            return _collate(_two_zp1_rows(2), max_z_prime=2)
        return _collate([zp1_row(), zp2_row()], max_z_prime=2)

    def bare():
        b = make()
        del b.aunit_handedness
        assert b.aunit_handedness is None
        return b

    c1, c2 = make().mol2cluster(cutoff=_CUTOFF, std_orientation=False), \
        bare().mol2cluster(cutoff=_CUTOFF, std_orientation=False)
    for k in _CLUSTER_KEYS:
        a, b = getattr(c1, k), getattr(c2, k)
        assert a.dtype == b.dtype and torch.equal(a, b), k
    assert c2.aunit_handedness is None
    u1, u2 = make(), bare()
    u1.mol2ucell(std_orientation=False)
    u2.mol2ucell(std_orientation=False)
    for k in ('pos', 'unit_cell_pos', 'unit_cell_batch', 'unit_cell_mol_ind'):
        assert torch.equal(getattr(u1, k), getattr(u2, k)), k
    e1 = make().analyze(['elj'], cutoff=_CUTOFF, std_orientation=False)['elj']
    e2 = bare().analyze(['elj'], cutoff=_CUTOFF, std_orientation=False)['elj']
    assert bool(torch.isfinite(e1).all()) and torch.equal(e1, e2), (e1, e2)


# ------------------------------------------------------------------------------------------------
# collate_data_list (audit #4 and section 7)
# ------------------------------------------------------------------------------------------------

def _same(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _snapshot(d):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in d._store.items()}


def test_collate_leaves_its_inputs_unchanged():
    rows = [zp1_row(), zp2_row()]
    listed = list(rows)
    snaps = [_snapshot(r) for r in rows]
    held = [{k: id(v) for k, v in r._store.items()} for r in rows]
    _collate(rows, max_z_prime=2)
    assert len(rows) == 2 and all(a is b for a, b in zip(rows, listed))
    for r, snap, ids in zip(rows, snaps, held):
        assert set(r._store.keys()) == set(snap)
        for k in snap:
            assert _same(snap[k], r._store[k]), k
            assert id(r._store[k]) == ids[k], k
    assert rows[0].max_z_prime == 1 and rows[0].aunit_centroid.shape == (1, 3)


def test_collate_output_equals_collating_prewidened_rows():
    """Today's output bit for bit: collating a narrow row beside a wider one gives what collating a copy padded
    as collate always padded it (zeros: 3 columns per missing slot, 1 for handedness) gives."""
    out = _collate([zp1_row(), zp2_row()], max_z_prime=2)
    p = zp1_row()
    p.aunit_centroid = F.pad(p.aunit_centroid, (0, 3))
    p.aunit_orientation = F.pad(p.aunit_orientation, (0, 3))
    p.aunit_handedness = F.pad(p.aunit_handedness, (0, 1))
    p.max_z_prime = 2
    ref = _collate([p, zp2_row()], max_z_prime=2)
    assert type(out) is type(ref)
    assert set(out._store.keys()) == set(ref._store.keys())
    for k in ref._store.keys():
        assert _same(ref._store[k], out._store[k]), k
    for d in ('_slice_dict', '_inc_dict'):
        a, b = getattr(ref, d), getattr(out, d)
        assert a.keys() == b.keys(), d
        for k in a:
            assert _same(a[k], b[k]), (d, k)


def test_collate_still_drops_the_pad_slots_of_zp1_rows():
    b = _collate([zp1_row(max_z_prime=2)], max_z_prime=1)
    assert b.max_z_prime == 1 and b.aunit_centroid.shape == (1, 3) and b.aunit_handedness.shape == (1, 1)
