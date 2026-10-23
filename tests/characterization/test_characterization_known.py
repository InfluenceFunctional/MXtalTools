"""Stage 0: known wrong behaviour as strict expected failures, and the controls that must keep passing.

Each ``xfail(strict=True)`` test states the RIGHT behaviour; it fails today for the reason named
(audit item, docs/design/crystal_units_audit.md in gfn_diffusion) and must start passing at the
stage named, at which point pytest reports XPASS(strict) as a failure and the marker has to be
removed in that change. Nothing here is a pin of a value: the values, today's wrong ones included,
are pinned in test_characterization_pins.py.

How the known failure is recognised. Every strict xfail expects ``pytest.fail`` (``raises=FAILED``)
and nothing else: the test calls ``pytest.fail`` when its own check finds today's wrong value, and
when the library stops at the known place -- the exception TYPE and the innermost FUNCTION are
both named (``_known_crash``). Any other exception propagates and is reported as a failure, so a
regression cannot hide behind an xfail.

"Refused" means a deliberate refusal (``charlib.deliberate_refusal``): a ValueError, RuntimeError
or NotImplementedError (or a library subclass) raised by a ``raise`` statement in mxtaltools. An
accidental crash -- a torch shape error (RuntimeError from einsum / cat / view), an IndexError, a
bare ``assert`` that ``python -O`` removes -- is not a refusal wherever it surfaces.
"""
from pathlib import Path

import pytest
import torch

import charlib

FAILED = pytest.fail.Exception   # what pytest.fail (and an unmet pytest.raises) raise


def _xfail(item, stage, why):
    """Strict xfail whose failure must be the test's own ``pytest.fail`` (the failure mode measured today)."""
    return pytest.mark.xfail(strict=True, raises=FAILED,
                             reason=f'audit {item}: {why}; expected to flip at stage {stage}')


def _known_crash(exc, exc_type, function, path_part=None):
    """``pytest.fail`` when ``exc`` is the known accidental crash -- an ``exc_type`` whose innermost frame is
    ``function`` (in a file whose path contains ``path_part``) -- else re-raise it: an exception of any other
    kind or origin is a failure, not the known one."""
    f = charlib.innermost_frame(exc)
    where = Path(f.filename).as_posix()
    if isinstance(exc, exc_type) and f.name == function and (path_part is None or path_part in where):
        pytest.fail(f'known failure: {charlib.describe_exception(exc)}')
    raise exc


def _refused(exc) -> bool:
    return charlib.deliberate_refusal(exc)


def _fx():
    return charlib.fixtures()


def _audit_row(name, **kw):
    return charlib.synthetic_row(_fx()['AUDIT'][name], **kw)


# ------------------------------------------------------------------------------------------------
# the refusal rule itself (controls)
# ------------------------------------------------------------------------------------------------

def test_deliberate_refusal_rule():
    """Control on the harness: a ``raise`` in mxtaltools counts as a refusal; torch's einsum RuntimeError and a
    bare library assert do not. Since stage 1 the two negative examples are the module function aunit2ucell called
    directly on an unsplit Z'=2 batch (its einsum meets six slot columns; the build_unit_cell method, mol2ucell and
    mol2cluster split first or refuse) and collate_data_list([]) (a bare assert). Stage 0 used two crashes that
    stage 1 turned into refusals, and stage 1's first control, the build_unit_cell method on the same batch, became
    a refusal in the stage-1 review fixes."""
    from mxtaltools.crystal_building.utils import aunit2ucell
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    with pytest.raises(ValueError) as deliberate:
        MolCrystalData.from_arrays(torch.zeros(1, 6), torch.zeros(1, 3), torch.ones(1, dtype=torch.long),
                                   ptr=torch.tensor([0, 1]), sg=999, setting='standard')
    assert charlib.deliberate_refusal(deliberate.value), charlib.describe_exception(deliberate.value)
    with pytest.raises(RuntimeError) as accidental:
        aunit2ucell(charlib.build_case('F3'))
    assert charlib.innermost_frame(accidental.value).name == 'einsum', charlib.describe_exception(accidental.value)
    assert not charlib.deliberate_refusal(accidental.value), charlib.describe_exception(accidental.value)
    with pytest.raises(AssertionError) as bare:
        charlib.collate([])
    assert charlib.raised_in(bare.value, 'collate_data_list'), charlib.describe_exception(bare.value)
    assert not charlib.deliberate_refusal(bare.value)


# ------------------------------------------------------------------------------------------------
# #1: a unit is found by arithmetic, num_atoms // z_prime
# ------------------------------------------------------------------------------------------------

def _true_units_energy(rec, std, cutoff):
    """Brute-force eLJ of a two-compound crystal posed unit by unit: each true unit is built alone as a
    Z'=1 crystal with its own slot (the builder's per-unit pose, which test_unit_major_layout checks),
    then the lattice sum runs over the true units. Returns (eLJ, sum |e|)."""
    units, zs = [], []
    for u, m in enumerate(rec['mols']):
        sub = dict(rec, mols=[m], aunit_centroid=rec['aunit_centroid'][3 * u:3 * u + 3],
                   aunit_orientation=rec['aunit_orientation'][3 * u:3 * u + 3],
                   aunit_handedness=rec['aunit_handedness'][u:u + 1])
        b = charlib.collate([charlib.synthetic_row(sub)])
        b.mol2ucell(std_orientation=std)
        units.append(b.pos)
        zs.append(b.z)
        T_fc, ops = b.T_fc[0], b.symmetry_operators[0]
    o = charlib.oracle_pairs(units, zs, T_fc, ops, cutoff)
    return charlib.interior_energy(o['targets'], torch.cat(zs), cutoff)


def _unequal(name):
    """(record, row) of an unequal-unit crystal: an audit fixture, or ethanol (9) + 2-pyridone (12) = 21 atoms
    ('eth_pyr', two_eth_meth's cell and slots), whose total does not divide by Z'=2."""
    A = _fx()['AUDIT']
    if name == 'eth_pyr':
        rec = dict(A['two_eth_meth'], mols=[A['two_eth_meth']['mols'][0], A['pyr_p21c']['mols'][0]])
    else:
        rec = A[name]
    return rec, charlib.collate([charlib.synthetic_row(rec)], max_z_prime=2)


def _refused_or_scored_right(name, std):
    rec, b = _unequal(name)
    try:
        res = b.analyze(['elj'], cutoff=10.0, std_orientation=std)
    except Exception as e:
        if _refused(e):
            return
        _known_crash(e, AssertionError, 'split_to_zp1_batch', 'data_class_methods/crystal_building.py')
    e_true, scale = _true_units_energy(rec, std, 10.0)
    if abs(float(res['elj'][0]) - e_true) > 2e-4 * scale:
        pytest.fail(f"eLJ {float(res['elj'][0]):.3f}; the true units give {e_true:.3f} (sum|e| {scale:.1f})")


_UNEQUAL = [
    # methane (one heavy atom) has no defined frame at std_orientation=True (audit section 7), so its true-unit
    # reference is only meaningful at std_orientation=False; ethanol + propane has >= 3 heavy atoms per unit.
    # flipped at stage 2 (2026-10-22): ethanol (9) + methane (5) was split 7 + 7 and scored with no error (eLJ -68.71
    # against -58.90 for the true units, std_orientation False), ethanol (9) + propane (11) 10 + 10 (eLJ -124.86 vs
    # -130.64 at std True, -159.18 vs -163.42 at std False); the list constructor now writes the unit map (9 + 5,
    # 9 + 11) and split_to_zp1_batch refuses units of different sizes (ValueError naming the row) until stage 7
    pytest.param('two_eth_meth', False, id='eth_meth-std0'),
    pytest.param('two_eth_prop', True, id='eth_prop-std1'),
    pytest.param('two_eth_prop', False, id='eth_prop-std0'),
    # flipped at stage 1: ethanol (9) + 2-pyridone (12) = 21 atoms at Z'=2 used to stop at the bare assert in
    # split_to_zp1_batch, which python -O removes (the audit then measured a 12/12 split of 25 atoms and a later
    # IndexError); split_to_zp1_batch now raises ValueError naming the row (tests/test_zprime_refusals.py runs it
    # under -O)
    pytest.param('eth_pyr', True, id='eth_pyr21-std1'),
]


@pytest.mark.parametrize('name,std', _UNEQUAL)
def test_unequal_units_are_refused_or_scored_right(name, std):
    _refused_or_scored_right(name, std)


@pytest.mark.parametrize('name,sizes', [
    # flipped at stage 2 (2026-10-22): split_to_zp1_batch cut ethanol + methane into two 7-atom units and ethanol +
    # propane into two 10-atom units; it now reads the unit map and refuses unequal units (ValueError) until stage 7
    pytest.param('two_eth_meth', [9, 5], id='eth_meth'),
    pytest.param('two_eth_prop', [9, 11], id='eth_prop'),
])
def test_unequal_units_split_by_their_atoms_or_refused(name, sizes):
    _, b = _unequal(name)
    try:
        s = b.split_to_zp1_batch()
    except Exception as e:
        if _refused(e):
            return
        raise
    if s.num_atoms.tolist() != sizes:
        pytest.fail(f'split into {s.num_atoms.tolist()} atoms, the units hold {sizes}')


# ------------------------------------------------------------------------------------------------
# #2: z_prime is a label nothing checks against the atoms
# ------------------------------------------------------------------------------------------------

def _phenacetin_mol():
    r = _fx()['F1'][0]
    assert r['identifier'] == 'PYRAZB21'
    return charlib._mol({'z': r['z'], 'pos': r['pos']})


def _two_slots():
    r = _fx()['F1'][0]
    cen = torch.cat([r['aunit_centroid'], (r['aunit_centroid'] + 0.37) % 1.0])
    ori = torch.cat([r['aunit_orientation'], r['aunit_orientation'].flip(0)])
    return r, cen, ori


def _one_molecule(which):
    """(molecule record, cell lengths, cell angles, sg, two pose slots) of a one-molecule crystal."""
    if which == 'phenacetin':
        r, cen, ori = _two_slots()
        return {'z': r['z'], 'pos': r['pos']}, r['cell_lengths'], r['cell_angles'], 14, cen, ori
    a = _fx()['AUDIT']['pyr_p21c']          # 2-pyridone, 12 atoms
    cen = torch.cat([a['aunit_centroid'], (a['aunit_centroid'] + 0.37) % 1.0])
    ori = torch.cat([a['aunit_orientation'], a['aunit_orientation'].flip(0)])
    return a['mols'][0], a['cell_lengths'], a['cell_angles'], int(a['sg']), cen, ori


def _relabel_outcome(score):
    """Run ``score`` (one-molecule row at Z'=2): a deliberate refusal passes; phenacetin's known IndexError
    (geometry_utils.batch_get_furthest_node_vector, after the 13 + 13 cut) and a silent score are the known
    failures; anything else propagates."""
    try:
        out = score()
    except Exception as e:
        if _refused(e):
            return
        _known_crash(e, IndexError, 'batch_get_furthest_node_vector', 'common/geometry_utils.py')
    pytest.fail(f"scored as two half-molecule units: eLJ {float(out['elj'][0]):.3f}")


#: flipped at stage 2 (2026-10-22): a one-molecule row labelled Z'=2 was cut into two half-molecule 'units' (2-pyridone,
#: 12 atoms: scored, eLJ 412.4 at cutoff 6, no error; phenacetin, 26 atoms: an IndexError in
#: geometry_utils.batch_get_furthest_node_vector). The constructor now writes the unit map (one molecule, one unit):
#: relabelled on the batch, split_to_zp1_batch refuses the row (unit 1 holds no atom); given z_prime=2 at
#: construction, the constructor refuses it
_RELABEL = ['pyridone', 'phenacetin']


@pytest.mark.parametrize('which', _RELABEL)
def test_one_molecule_row_relabelled_zp2_is_refused(which):
    """z_prime set to 2 on a batch whose row holds one molecule, the relabelling GFN condition_samples does, on a row
    built by the constructor, which writes its unit map (one unit). Rows that carry no map are not covered: GFN's own
    blank batch (set_mol_attrs' batch path), from_arrays rows and files written before stage 2 are still cut into two
    half-molecule units and scored when so relabelled (README, stage-2 flip note; owner decision)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    mol, lengths, angles, sg, cen, ori = _one_molecule(which)

    def score():
        row = MolCrystalData(molecule=charlib._mol(mol), sg_ind=sg, max_z_prime=2, cell_lengths=lengths.clone(),
                             cell_angles=angles.clone(), aunit_centroid=cen[:3].clone(),
                             aunit_orientation=ori[:3].clone(), aunit_handedness=torch.tensor([1]),
                             do_box_analysis=True)
        b = charlib.collate([row], max_z_prime=2)
        b.aunit_centroid = cen[None].clone()
        b.aunit_orientation = ori[None].clone()
        b.aunit_handedness = torch.tensor([[1, 1]])
        b.z_prime = torch.tensor([2])
        return b.analyze(['elj'], cutoff=6.0)
    _relabel_outcome(score)


@pytest.mark.parametrize('which', _RELABEL)
def test_one_molecule_given_zp2_at_construction_is_refused(which):
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    mol, lengths, angles, sg, cen, ori = _one_molecule(which)

    def score():
        row = MolCrystalData(molecule=charlib._mol(mol), sg_ind=sg, z_prime=torch.tensor([2]), max_z_prime=2,
                             cell_lengths=lengths.clone(), cell_angles=angles.clone(), aunit_centroid=cen,
                             aunit_orientation=ori, aunit_handedness=torch.tensor([1, 1]), do_box_analysis=True)
        return charlib.collate([row], max_z_prime=2).analyze(['elj'], cutoff=6.0)
    _relabel_outcome(score)


# ------------------------------------------------------------------------------------------------
# #4 / #12 / #13 and section 7: one batch-wide max_z_prime
# ------------------------------------------------------------------------------------------------

def test_collate_refuses_to_truncate_a_zp2_row():
    """Flipped at stage 1 (audit #4): collate_data_list(max_z_prime=1) used to truncate a Z'=2 row's slots to one
    and keep z_prime 2, silently; it now raises ValueError naming the row."""
    rows = [charlib.f3_list_row(_fx()['F3'][1])]
    try:
        b = charlib.collate(rows, max_z_prime=1)
    except Exception as e:
        if _refused(e):
            return
        raise
    pytest.fail(f"collated: z_prime {b.z_prime.tolist()}, aunit_centroid {tuple(b.aunit_centroid.shape)}")


def _snapshot(d):
    """Every stored value of a data object in its pinned form (charlib.encode: tensors bit for bit)."""
    return {k: charlib.encode(v.clone() if torch.is_tensor(v) else v) for k, v in d._store.items()}


def test_collate_leaves_its_inputs_unchanged():
    """Flipped at stage 1 (audit section 7): collate_data_list used to widen a narrower input row IN PLACE
    (phenacetin's aunit_centroid (1, 3) -> (1, 6), max_z_prime 1 -> 2 in the caller's list), so pad values
    depended on collation history (#5); it now pads a shallow copy."""
    r1 = charlib.f1_row(_fx()['F1'][0], reference_handedness=False)
    r2 = charlib.f3_list_row(_fx()['F3'][1])
    before = [_snapshot(r1), _snapshot(r2)]
    charlib.collate([r1, r2], max_z_prime=2)
    changed = []
    for i, (r, snap) in enumerate(zip((r1, r2), before)):
        for k in sorted(set(snap) | set(r._store.keys())):
            if k not in snap or k not in r._store:
                changed.append(f'row {i} {k}: {"added" if k not in snap else "removed"}')
                continue
            ok, detail = charlib.compare(snap[k], r._store[k])
            if not ok:
                changed.append(f'row {i} {k}: {detail}')
    if changed:
        pytest.fail(f'inputs modified by collate: {changed}')


def _f1_rows(width):
    return [charlib.f1_row(r, max_z_prime=width, reference_handedness=False) for r in _fx()['F1']]


def _score_all_zp1_width2():
    try:
        return charlib.collate(_f1_rows(2), max_z_prime=2).analyze(['elj'], cutoff=10.0)['elj']
    except Exception as e:
        _known_crash(e, AssertionError, 'join_zp1_cluster_batch', 'data_class_methods/crystal_building.py')


def test_all_zp1_batch_at_width2_is_scored():
    """Flipped at stage 1 (audit #12): a batch of only Z'=1 rows at max_z_prime=2 used to stop at the bare assert in
    join_zp1_cluster_batch; the join now returns the split's clusters (splitting a Z'=1 row is the identity)."""
    e = _score_all_zp1_width2()
    if not bool(torch.isfinite(e).all()):
        pytest.fail(f'non-finite eLJ {e.tolist()}')


def test_all_zp1_batch_at_width2_equals_width1():
    """Flipped at stage 1 (audit #12, #13): as above, and the pad slots no longer widen the cluster, so the scores
    equal the width-1 batch's bit for bit."""
    w2 = _score_all_zp1_width2()
    w1 = charlib.collate(_f1_rows(1)).analyze(['elj'], cutoff=10.0)['elj']
    if not torch.equal(w2, w1):
        pytest.fail(f'width 2 {w2.tolist()} vs width 1 {w1.tolist()}')


_WIDTH1_ROUTES = ['mol2ucell', 'analyze']


@pytest.mark.parametrize('route', _WIDTH1_ROUTES)
def test_zp2_row_in_a_width1_batch_is_refused(route):
    """Flipped at stage 1 (audit #12): a Z'=2 row in a batch whose max_z_prime is 1 (F3 with max_z_prime set to 1)
    used to stop in torch.einsum (RuntimeError 'subscript j has size 7 ... size 4', an accidental shape error);
    mol2ucell / mol2cluster now refuse a row wider than the batch (ValueError naming the row)."""
    b = charlib.build_case('F3')
    b.max_z_prime = 1
    try:
        if route == 'mol2ucell':
            b.mol2ucell(std_orientation=True)
        else:
            b.analyze(['elj'], cutoff=6.0)
    except Exception as e:
        if _refused(e):
            return
        _known_crash(e, RuntimeError, 'einsum', 'torch/functional.py')
    pytest.fail(f"{route} ran on a Z'=2 row in a max_z_prime=1 batch with no error")


def _alone_and_mixed(case):
    """(Z'=1 row scored alone at width 1, the same row as row 0 of a max_z_prime=2 batch beside a Z'=2 row)."""
    fx = _fx()
    if case == 'pyridone':           # audit verify/v1 (c)
        alone = charlib.collate([_audit_row('pyr_p21c')])
        mixed = charlib.collate([_audit_row('pyr_p21c', max_z_prime=1), _audit_row('pyr_zp2')], max_z_prime=2)
    else:
        alone = charlib.collate([charlib.f1_row(fx['F1'][0], reference_handedness=False)])
        mixed = charlib.build_case(case)
    return alone, mixed


def _row0(batch, cutoff=10.0):
    """Row 0's cluster size, cluster atoms with aux_ind < 2, sorted inter-pair distances, and eLJ."""
    res, cl = batch.clone().analyze(['elj'], cutoff=cutoff, return_cluster=True)
    m = cl.batch == 0
    ei = cl.edges_dict['edge_index_inter']
    d = torch.sort(cl.edges_dict['intermolecular_dist'][cl.batch[ei[1]] == 0]).values
    return {'atoms': int(m.sum()), 'aux_lt2': int((m & (cl.aux_ind < 2)).sum()), 'dist': d, 'elj': res['elj'][0]}


@pytest.mark.parametrize('case', ['pyridone', 'F4a', 'F4b'])
def test_zp1_row_cluster_in_mixed_batch_same_as_alone(case):
    """Flipped at stage 1 (audit #13): the Z'=1 row's zp_buffer used to be the largest distance between its centroid
    slots, pad slots included, which widened its cluster in a max_z_prime=2 batch (phenacetin 9152 cluster atoms,
    6110 with aux_ind < 2, in F4a / F4b against 5928 (4056) alone; pyridone 3264 (2184) against 2256 (1368)).
    zp_buffer now reads each row's own slots only."""
    alone, mixed = (_row0(b) for b in _alone_and_mixed(case))
    a, m = (alone['atoms'], alone['aux_lt2']), (mixed['atoms'], mixed['aux_lt2'])
    if a != m:
        pytest.fail(f'row-0 cluster atoms (all, aux_ind < 2): {m} in the mixed batch, {a} alone')


#: float32 units in the last place allowed between the eLJ of the same pair set summed in two cluster orders
ELJ_ULPS = 8


@pytest.mark.parametrize('case', [
    # flipped at stage 1: the pyridone row had 4995 inter pairs in the mixed batch and 4993 alone (eLJ -200.4179 vs
    # -200.4142); 4995 is the true count, the row built alone missing 2 pairs to the reach fault (#16) that the
    # pad-widened mixed cluster happened to recover. Pads no longer widen it: mixed = alone, 4993 from stage 1 and
    # the true 4995 since stage 5 fixed the reach (2026-10-23; the oracle's pyr_p21c-row0-*-cut10 cases pass)
    'pyridone', 'F4a', 'F4b'])
def test_zp1_row_pairs_and_elj_in_mixed_batch_equal_alone(case):
    """The row's pair-distance multiset is identical in both builds, and its eLJ equal within a few float32 ulps
    (today phenacetin's differs by one ulp: -637.1484985 vs -637.1484375, a summation-order difference, not a
    physical one; the eLJ values themselves are pinned bit for bit in the pins)."""
    alone, mixed = (_row0(b) for b in _alone_and_mixed(case))
    da, dm = alone['dist'], mixed['dist']
    if da.shape != dm.shape or not torch.equal(da, dm):
        pytest.fail(f'row-0 inter pairs: {len(dm)} in the mixed batch, {len(da)} alone '
                    f'(eLJ {float(mixed["elj"]):.4f} vs {float(alone["elj"]):.4f})')
    ea, em = float(alone['elj']), float(mixed['elj'])
    tol = ELJ_ULPS * float(torch.finfo(torch.float32).eps) * max(abs(ea), 1.0)
    if abs(ea - em) > tol:
        pytest.fail(f'eLJ {em:.7f} in the mixed batch, {ea:.7f} alone: |diff| {abs(ea - em):.2e} > {tol:.2e}')


# ------------------------------------------------------------------------------------------------
# #7: the list constructor merges molecules by the type of each key
# ------------------------------------------------------------------------------------------------

def test_list_constructor_sums_or_refuses_collated_rows():
    """Flipped at stage 2 (2026-10-22; audit #7): molecules taken out of a collated batch carry [1]-shaped num_atoms /
    radius / mass / mol_volume, which the list constructor concatenated instead of summing (num_atoms [26, 26],
    packing halved); it now sums one value per molecule, 0-dim or [1], and refuses anything else."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    m = _phenacetin_mol()
    rows = charlib.collate([m.clone(), m.clone()]).to_data_list()
    r, cen, ori = _two_slots()
    try:
        c = MolCrystalData(molecule=[rows[0], rows[1]], sg_ind=14, z_prime=torch.tensor([2]), max_z_prime=2,
                           cell_lengths=r['cell_lengths'].clone(), cell_angles=r['cell_angles'].clone(),
                           aunit_centroid=cen, aunit_orientation=ori, aunit_handedness=torch.tensor([1, 1]),
                           do_box_analysis=True)
    except Exception as e:
        if _refused(e):
            return
        raise
    n = int(m.num_atoms)
    bad = []
    if not (c.num_atoms.numel() == 1 and int(c.num_atoms) == 2 * n):
        bad.append(f'num_atoms {c.num_atoms.tolist()}')
    for k in ('radius', 'mass', 'mol_volume'):
        v = getattr(c, k)
        if not (v.numel() == 1 and torch.allclose(v.reshape(()), 2 * getattr(m, k).reshape(()))):
            bad.append(f'{k} {v.tolist()}')
    if c.packing_coeff.numel() != 1:
        bad.append(f'packing_coeff {c.packing_coeff.tolist()}')
    if bad:
        pytest.fail('; '.join(bad))


_ZA = torch.tensor([6, 6, 8, 7, 6, 1])
_PA = torch.tensor([[0., 0, 0], [1.5, 0, 0], [2.2, 1.2, 0], [-0.7, 1.2, 0.3], [2.1, -1.3, 0.8], [-0.5, -0.9, -0.4]])
_ZB = torch.tensor([6, 6, 6, 6, 8, 8, 1, 1])
_PB = torch.tensor([[0., 0, 0], [1.4, 0, 0], [2.1, 1.2, 0], [1.4, 2.4, 0.2], [3.5, 1.2, 0.1], [0.0, 2.4, -0.3],
                    [-0.5, -0.9, 0], [3.9, 0.3, 0]])


def _chain_index(n):
    return torch.stack([torch.arange(n - 1), torch.arange(1, n)])


def _two_molecules(key):
    from mxtaltools.dataset_utils.data_classes import MolData
    a = MolData(z=_ZA.clone(), pos=_PA.clone(), do_mol_analysis=True)
    b = MolData(z=_ZB.clone(), pos=_PB.clone(), do_mol_analysis=True)
    a[key] = _chain_index(6)
    b[key] = _chain_index(8)
    return a, b


def test_list_constructor_offsets_edge_index():
    """Flipped at stage 3 (2026-10-23; audit #7): the list constructor concatenated two molecules' edge_index on dim 0
    without an offset (two 6-atom molecules gave [4, 5], both from index 0); it now joins them along the last axis,
    the second offset by 6, as collation does."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    a, _ = _two_molecules('edge_index')
    a2, _ = _two_molecules('edge_index')
    r, cen, ori = _two_slots()
    try:
        c = MolCrystalData(molecule=[a, a2], sg_ind=14, z_prime=torch.tensor([2]), max_z_prime=2,
                           cell_lengths=r['cell_lengths'].clone(), cell_angles=r['cell_angles'].clone(),
                           aunit_centroid=cen, aunit_orientation=ori, aunit_handedness=torch.tensor([1, 1]),
                           do_box_analysis=True)
    except Exception as e:
        if _refused(e):
            return
        raise
    want = torch.cat([_chain_index(6), _chain_index(6) + 6], dim=1)
    if not (c.edge_index.shape == want.shape and torch.equal(c.edge_index, want)):
        pytest.fail(f'edge_index {tuple(c.edge_index.shape)}: {c.edge_index.tolist()}')


def test_int_z_prime_row_collates_beside_a_default_row():
    """Flipped at stage 2 (2026-10-22; audit #7): z_prime given to the list constructor as an int was stored 0-dim, and
    collate_data_list could not concatenate it beside a constructor-built Z'=1 row's [1]-shaped default (IndexError in
    torch_geometric's collate); an int is now stored [1]."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    rec = _fx()['F3'][1]
    n = len(rec['z']) // 2
    units = [charlib._mol({'z': rec['z'][k * n:(k + 1) * n], 'pos': rec['pos'][k * n:(k + 1) * n].float()})
             for k in range(2)]
    try:
        r2 = MolCrystalData(molecule=units, sg_ind=int(rec['sg']), z_prime=2, max_z_prime=2,
                            cell_lengths=rec['cell_lengths'].float(), cell_angles=rec['cell_angles'].float(),
                            aunit_centroid=rec['aunit_centroid'].float(),
                            aunit_orientation=rec['aunit_orientation'].float(),
                            aunit_handedness=rec['aunit_handedness'].clone(), identifier=rec['identifier'],
                            do_box_analysis=True)
        r1 = charlib.f1_row(_fx()['F1'][0], max_z_prime=2, reference_handedness=False)
        b = charlib.collate([r1, r2], max_z_prime=2)
    except Exception as e:
        if _refused(e):
            return
        _known_crash(e, IndexError, '<listcomp>', 'torch_geometric/data/collate.py')
    if b.z_prime.tolist() != [1, 2]:
        pytest.fail(f'collated z_prime {b.z_prime.tolist()}')


# ------------------------------------------------------------------------------------------------
# #3: the batch helpers know two sliced levels, atom and row
# ------------------------------------------------------------------------------------------------

def _with_unit_value(batch, offset=0.0):
    n_units = int(batch.z_prime.sum())
    batch['unit_flat'] = torch.arange(n_units, dtype=torch.float32) * 10 + offset
    batch._slice_dict['unit_flat'] = torch.cat([torch.zeros(1, dtype=torch.long), batch.z_prime.cumsum(0)])
    batch._inc_dict['unit_flat'] = torch.zeros(batch.num_graphs, dtype=torch.long)
    return batch


@pytest.mark.parametrize('case,row,want', [
    pytest.param('F3', 1, [20.0, 30.0], id='F3-zp2-2'),
    pytest.param('F4a', 1, [10.0, 20.0], id='F4a-ragged-row1'),
    pytest.param('F4a', 0, [0.0], id='F4a-ragged-row0'),
])
def test_subsample_slices_a_per_unit_value(case, row, want):
    """Flipped at stage 3 (2026-10-23; audit #3): a per-unit value with a unit_ptr slice entry was carried whole by
    subsample_new_batch (F3 row 1 of 2 kept all 4 units' values; on the ragged Z' [1, 2] F4a batch row 1 and row 0
    each kept all 3); it is now the UNIT level, sliced by the unit pointer of z_prime."""
    sub = _with_unit_value(charlib.build_case(case)).subsample_new_batch([row])
    if sub.unit_flat.tolist() != want:
        pytest.fail(f'unit_flat {sub.unit_flat.tolist()}, want {want}')


def test_append_concatenates_a_per_unit_value():
    """Flipped at stage 3 (2026-10-23; audit #3): append_batch filed a per-unit value as shared metadata (two batches
    with equal values kept ONE copy, 4 values for 8 units, silently; different values raised 'Shared tensor field
    differs'); it now concatenates it per unit."""
    a = _with_unit_value(charlib.build_case('F3'))
    b = _with_unit_value(charlib.build_case('F3'))
    j = a.append_batch(b)
    if j.unit_flat.tolist() != [0.0, 10.0, 20.0, 30.0, 0.0, 10.0, 20.0, 30.0]:
        pytest.fail(f'unit_flat {j.unit_flat.tolist()}')


@pytest.mark.parametrize('case,want', [('F3', [[0.0, 10.0], [20.0, 30.0]]), ('F4a', [[0.0], [10.0, 20.0]])])
def test_batch_to_list_splits_a_per_unit_value(case, want):
    """Control: PyG's own split honours the unit_ptr slice entry (audit #3, 'what already works'), ragged too."""
    rows = _with_unit_value(charlib.build_case(case)).batch_to_list()
    assert [r.unit_flat.tolist() for r in rows] == want


@pytest.mark.parametrize('n_graphs', [2, 3])
def test_subsample_slices_an_index_tensor(n_graphs):
    """Flipped at stage 3 (2026-10-23; audit #3): on a 2-graph batch the length rule read a [2, E] '*_index' tensor
    (concatenated on its last dim) as per graph and gathered ROW 1 (tree_index [2, 12] became [1, 12]); on a 3-graph
    batch it matched no level and was carried whole and un-offset (molecule 1 of 3 kept all 17 bonds, indices up to 19
    on an 8-atom batch). An 'index' key is now the EDGE level: its rows' columns gathered, re-offset."""
    a, b = _two_molecules('tree_index')
    mols = [a, b] if n_graphs == 2 else [a, b, _two_molecules('tree_index')[0]]
    sub = charlib.collate(mols).subsample_new_batch([1])
    if not (sub.tree_index.shape == _chain_index(8).shape and torch.equal(sub.tree_index, _chain_index(8))):
        pytest.fail(f'tree_index {tuple(sub.tree_index.shape)}: {sub.tree_index.tolist()}')


def test_append_offsets_edge_index():
    """Flipped at stage 3 (2026-10-23; audit #3): append_batch did not offset a second batch's edge_index by the
    first's atom count (on these 2-graph batches it concatenated the [2, E] tensors as rows, [4, 12])."""
    a1, b1 = _two_molecules('edge_index')
    a2, b2 = _two_molecules('edge_index')
    x = charlib.collate([a1, b1])
    y = charlib.collate([a2, b2])
    j = x.append_batch(y)
    want = torch.cat([_chain_index(6), _chain_index(8) + 6, _chain_index(6) + 14, _chain_index(8) + 20], dim=1)
    if not (j.edge_index.shape == want.shape and torch.equal(j.edge_index, want)):
        pytest.fail(f'edge_index {j.edge_index.tolist()}')


# ------------------------------------------------------------------------------------------------
# #15: MLIP element assignment tiles the asymmetric unit's z image by image
# ------------------------------------------------------------------------------------------------

def _element_errors(batch, which, std):
    b = batch.clone()
    b.mol2ucell(std_orientation=std)
    wrong = []
    for g, (pos, z) in enumerate(charlib.assembled_elements(b, which, std)):
        a, e = int(b.ptr[g]), int(b.ptr[g + 1])
        truth = charlib.element_truth(pos, b.pos[a:e], b.z[a:e], b.T_fc[g], b.symmetry_operators[g])
        wrong.append(int((truth != z.long()).sum()))
    return wrong


_ONE_COMPOUND = [('F1', True), ('F1', False), ('F3', True), ('F3b', True), ('F3b', False), ('F4a', True)]
_TWO_COMPOUNDS = [('two_eth_ether', True), ('two_eth_ether', False), ('two_pyr_flb', True), ('two_pyr_flb', False)]


@pytest.mark.parametrize('which', charlib.ASSEMBLERS)
@pytest.mark.parametrize('case,std', _ONE_COMPOUND, ids=[f'{c}-std{int(s)}' for c, s in _ONE_COMPOUND])
def test_mlip_elements_right_for_one_compound(case, std, which):
    """Control: with one compound every unit carries unit 0's atom order, so the image-by-image tiling
    happens to give every unit-cell row its own element. Stage 4 must keep this bit-identical."""
    assert _element_errors(charlib.build_case(case), which, std) == [0] * charlib.build_case(case).num_graphs


@pytest.mark.parametrize('which', charlib.ASSEMBLERS)
@pytest.mark.parametrize('case,std', _TWO_COMPOUNDS, ids=[f'{c}-std{int(s)}' for c, s in _TWO_COMPOUNDS])
def test_mlip_elements_right_for_two_compounds(case, std, which):
    """Flipped at stage 4 (2026-10-23; audit #15): the unit cell is laid out [unit][image][atom], and the MLIP
    builders tiled the asymmetric unit's z image by image, so with two compounds some rows got the other compound's
    element (4 of 36 for ethanol + dimethyl ether, 8 of 96 for pyridone + fluorobenzene); every reader now gathers
    through crystal_building.utils.unit_cell_gather_index / unit_cell_z."""
    errors = _element_errors(charlib.build_case(case), which, std)
    if errors != [0]:
        pytest.fail(f'{errors[0]} unit-cell rows carry the wrong element')


@pytest.mark.parametrize('case', [pytest.param('F3b', id='F3b-control'), pytest.param('two_pyr_flb', id='two_pyr_flb')])
def test_zp_doubling_unit_cell_elements_right(case):
    """two_pyr_flb flipped at stage 4 (2026-10-23; audit #15): zp_doubling._unit_cell_frac tiled unit 0's z over the
    whole unit cell; it now reads the unit cell's layout (unit_cell_z)."""
    from mxtaltools.crystal_building.zp_doubling import _unit_cell_frac
    row = charlib.f3b_row(_fx()['F3b'][0]) if case == 'F3b' else _audit_row(case)
    frac, z, t_fc = _unit_cell_frac(row)
    b = charlib.collate([row.clone()], max_z_prime=2)
    b.mol2ucell(std_orientation=True)
    truth = charlib.element_truth(frac @ t_fc.T, b.pos, b.z, t_fc, b.symmetry_operators[0])
    wrong = int((truth != z.long()).sum())
    if wrong:
        pytest.fail(f'{wrong} of {len(z)} unit-cell rows carry the wrong element')


# ------------------------------------------------------------------------------------------------
# #15 (layout): the unit cell is unit-major at Z' > 1
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('case,std', [('F3', True), ('F3b', True), ('F3b', False), ('F4a', True)])
def test_unit_major_layout(case, std):
    """Pins today's layout geometrically (tests/test_unit_cell_layout.py covers Z'=1 only): each crystal's
    unit_cell_pos is [unit][image][atom]; every n_u-row block is a rigid copy of ITS unit's posed
    conformer (internal distances within 1e-4 A), and blocks of unit u come before those of unit u + 1."""
    b = charlib.build_case(case)
    b.mol2ucell(std_orientation=std)
    for g in range(b.num_graphs):
        zp, m = int(b.z_prime[g]), int(b.sym_mult[g])
        a, e = int(b.ptr[g]), int(b.ptr[g + 1])
        n_u = (e - a) // zp
        rows = b.unit_cell_pos[b.unit_cell_batch == g].double()
        assert rows.shape[0] == zp * m * n_u
        blocks = rows.reshape(zp, m, n_u, 3)
        for u in range(zp):
            unit = b.pos[a + u * n_u:a + (u + 1) * n_u].double()
            ref = torch.cdist(unit, unit)
            for k in range(m):
                dev = float((torch.cdist(blocks[u, k], blocks[u, k]) - ref).abs().max())
                assert dev < 1e-4, (case, g, u, k, dev)


# ------------------------------------------------------------------------------------------------
# #19: canonical unit order treats every unit as exchangeable
# ------------------------------------------------------------------------------------------------

@_xfail('#19', 7, "canonicalize_zp_aunits sorts the slots by centroid norm across compounds: with ethanol in the slot of "
                  "larger norm, ethanol and dimethyl ether trade slots and atom blocks (block 0 becomes the ether, "
                  "z [6, 8, 6, ...]); stage 7 orders units within a compound only")
def test_two_compounds_do_not_trade_slots():
    rec = dict(_fx()['AUDIT']['two_eth_ether'])
    rec['aunit_centroid'] = torch.cat([rec['aunit_centroid'][3:], rec['aunit_centroid'][:3]])
    rec['aunit_orientation'] = torch.cat([rec['aunit_orientation'][3:], rec['aunit_orientation'][:3]])
    b = charlib.collate([charlib.synthetic_row(rec)], max_z_prime=2)
    c = b.clone()
    c.canonicalize_zp_aunits()
    if not (torch.equal(c.z, b.z) and torch.equal(c.aunit_centroid, b.aunit_centroid)):
        pytest.fail(f'unit 0 was {b.z[:9].tolist()} at {b.aunit_centroid[0, :3].tolist()}; after canonicalisation '
                    f'{c.z[:9].tolist()} at {c.aunit_centroid[0, :3].tolist()}')


# ------------------------------------------------------------------------------------------------
# controls that pin no value but must hold
# ------------------------------------------------------------------------------------------------

def test_ingest_chain_reproduces_the_stored_f3_pose():
    """The structural ingestion steps re-derive, from the F3 rows given as bare deposited sets (wrapped,
    units interleaved, unit 1 reversed; charlib.f3_deposited_batch), the units, pose slots and handedness the
    r2 store holds for them, every row status ok (the full outputs are pinned in the 'ingest' group)."""
    b = charlib.f3_deposited_batch()
    for step, kw in charlib.INGEST_ARGS:
        b = getattr(b, step)(**kw)
    assert b.status.tolist() == [0, 0], b.message
    for g, rec in enumerate(_fx()['F3']):
        assert int(b.z_prime[g]) == int(rec['z_prime'])
        assert torch.equal(b.aunit_handedness[g, :2].long(), rec['aunit_handedness'].long())
        assert torch.allclose(b.aunit_centroid[g, :6].double(), rec['aunit_centroid'], atol=1e-6)
        a, e = int(b.ptr[g]), int(b.ptr[g + 1])
        assert torch.equal(b.mol_ind[a:e], rec['mol_ind']) and torch.equal(b.z[a:e], rec['z'])
