import os
from typing import Optional

import numpy as np
from ase import Atoms
from ase.cell import Cell
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
import torch

from mxtaltools.common.utils import is_cuda_oom
import torch.serialization

torch.serialization.add_safe_globals([slice])  # necessary for UMA loading on torch 2.6

#: Build the fairchem batch in one shot instead of one AtomicData per crystal.
#: Pure reindexing -- identical tensors, so identical energies (enforced by
#: tests/test_uma_atomicdata_vectorisation.py). Measured host cost on CPU, per leg:
#:
#:      graphs     list      vectorised   speedup
#:          25    3.4 ms       0.37 ms        9x
#:         100   16.7 ms       0.44 ms       38x
#:         400  143.1 ms       1.50 ms       96x
#:         800  324.5 ms       1.30 ms      249x
#:
#: The list path is superlinear in graph count (per-graph masks are O(total_atoms),
#: and collation calls __inc__ per key per graph, each an int() sync); the
#: vectorised one is flat. Two legs run per lattice energy, so double the saving.
#:
#: Set False to fall back to the list path -- the one switch to flip if anything
#: about the MLIP energies ever looks wrong.
#:
#: Overridable by MXT_VECTORISED_ATOMICDATA=0 so the fallback can be taken WITHOUT
#: editing source: a kill switch that needs a code change is not much of a kill
#: switch, and an A/B between the two paths must run identical source or it is
#: measuring the edit as well as the change.
USE_VECTORISED_ATOMICDATA = os.environ.get('MXT_VECTORISED_ATOMICDATA', '1') != '0'


def safe_predict_uma(predictor, uma_batch):
    try:
        torch.cuda.synchronize()  # flush prior kernels
        out = predictor.predict(uma_batch)  # this launches UMA kernels
        torch.cuda.synchronize()  # force errors to surface *here*
        return out, False  # False = no failure

    except RuntimeError as e:
        if not is_cuda_oom(e):
            print("UMA error")
            print(str(e))
            # reset the cuda context fully
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return None, True  # signal failure
        else:
            raise e


def batch_to_ase_ucell_list(
        batch,
        std_orientation,
        pbc,
        force_rebuild: bool = False,
):
    data_list = []
    if batch.unit_cell_pos is None or force_rebuild:
        if batch.z_prime.amax() > 1:
            zp1_batch = batch.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            batch.join_zp1_ucell_batch(zp1_batch)
        else:
            batch.pose_aunit(std_orientation=std_orientation)
            batch.build_unit_cell()
        cpuz = batch.z.cpu().detach().numpy()
        cpubatch = batch.batch.cpu().detach().numpy()

    elif batch.aux_ind is not None:
        cpuz = batch.z[batch.aux_ind == 0].cpu().detach().numpy()
        cpubatch = batch.batch[batch.aux_ind == 0].cpu().detach().numpy()

    else:
        cpuz = batch.z.cpu().detach().numpy()
        cpubatch = batch.batch.cpu().detach().numpy()

    assert torch.sum(torch.isnan(batch.pos)) == 0

    cell_params = torch.cat([batch.cell_lengths, batch.cell_angles * 90 / (torch.pi / 2)], dim=1).cpu().detach().numpy()
    cpupos = batch.unit_cell_pos.cpu().detach().numpy()
    cpuubatch = batch.unit_cell_batch.cpu().detach().numpy()
    cpusymmult = batch.sym_mult.cpu().detach().numpy()
    for ind in range(batch.num_graphs):
        pos = cpupos[cpuubatch == ind]
        z = np.tile(cpuz[cpubatch == ind], cpusymmult[ind])
        atoms = Atoms(
            numbers=z,
            positions=pos,
        )
        try:
            atoms.set_cell(Cell.fromcellpar(cell_params[ind]))
        except AssertionError:  # cells have invalid shape according to ASE - we'll manually override
            fixed_cp = cell_params[ind].copy()
            fixed_cp[3:6] = 90
            atoms.set_cell(Cell.fromcellpar(fixed_cp))

        pbc_flag = pbc
        atoms.set_pbc(pbc_flag)
        data_list.append(atoms)

    return data_list


def compute_crystal_uma_on_mxt_batch(batch,
                                     std_orientation: bool = True,
                                     predictor: Optional = None,
                                     pbc: bool = True,
                                     max_cp: float = 2.0,
                                     force_rebuild: bool = False):
    "UMA sometimes fails on ultra-dense cells, so we'll manually prevent that. These are obviously terrible cells anyway."
    while sum(batch.packing_coeff > max_cp) > 0:
        bad_inds = torch.argwhere(batch.packing_coeff > max_cp)
        if len(bad_inds) > 0:
            batch.cell_lengths[bad_inds] += 2
            batch.box_analysis()

    if USE_VECTORISED_ATOMICDATA:
        uma_batch = batch_to_fairchem_batch(batch, std_orientation, pbc, force_rebuild)
    else:
        uma_batch = atomicdata_list_to_batch(
            batch_to_fairchem_atomicdata(batch, std_orientation, pbc, force_rebuild))

    out, crashed = safe_predict_uma(predictor, uma_batch)
    if crashed:
        energy = torch.zeros(batch.num_graphs, dtype=torch.float32, device=batch.device)
    else:
        energy = out['energy']

    return energy
    # grad test
    # batch.pos.requires_grad_(True)
    # batch.T_fc.requires_grad_(True)
    # alpha = torch.nn.Parameter(torch.tensor(1.01, device='cuda'))
    # batch.pos = batch.pos * alpha
    # data_list = batch_to_fairchem_atomicdata(batch, std_orientation, pbc, force_rebuild)
    # uma_batch = atomicdata_list_to_batch(data_list)
    # #uma_batch.pos.requires_grad_(True)
    # out = predictor.predict(uma_batch)
    # energy = out['energy']
    # energy.mean().backward(retain_graph=True)
    # print(batch.pos.grad)
    # print(uma_batch.pos.grad)
    # print(batch.T_fc.grad)
    # print(alpha.grad)


def _resolve_aunit_z(batch, std_orientation, force_rebuild):
    """
    The aunit atomic numbers and their graph index, with the unit cell built if
    it is not already there.

    THE THREE CASES, kept exactly as they were -- this is the part that must not
    drift, because each corresponds to a different KIND of batch:

      1. no unit cell yet (or force_rebuild): build it. Z'>1 has to detour through
         split_to_zp1_batch/join_zp1_ucell_batch, because pose_aunit and
         build_unit_cell are Z'=1 operations.
      2. a SUPERCELL/cluster batch, marked by aux_ind: only aux_ind == 0 is the
         canonical asymmetric unit; everything else is image atoms that
         unit_cell_pos already accounts for, so taking all of batch.z would
         double-count them.
      3. a plain unit-cell batch: batch.z as-is.

    MUTATES `batch` in case 1 (it builds unit_cell_pos on it), exactly as before.
    """
    if batch.unit_cell_pos is None or force_rebuild:
        if batch.z_prime.amax() > 1:
            zp1_batch = batch.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            batch.join_zp1_ucell_batch(zp1_batch)
        else:
            batch.pose_aunit(std_orientation=std_orientation)
            batch.build_unit_cell()
        return batch.z, batch.batch

    if getattr(batch, 'aux_ind', None) is not None:
        keep = batch.aux_ind == 0
        return batch.z[keep], batch.batch[keep]

    return batch.z, batch.batch


def _tiled_gather_index(batch_ind, sym_mult, num_graphs):
    """
    Index that reproduces `z[batch_ind == i].repeat(sym_mult[i])` for every graph,
    concatenated -- i.e. per graph, the aunit's atoms TILED sym_mult times.

    Tiled, not repeat_interleaved: `Tensor.repeat(k)` on a 1-D tensor gives
    [a,b,c,a,b,c], and unit_cell_pos is laid out image-major to match. Getting this
    backwards would pair every position with the wrong element and still produce a
    perfectly well-formed batch, so it is asserted in the tests rather than trusted.
    """
    n = torch.bincount(batch_ind, minlength=num_graphs)          # aunit atoms per graph
    m = sym_mult.to(n.dtype)
    out_n = n * m                                                # ucell atoms per graph
    src_off = torch.cumsum(n, 0) - n                             # start of graph i in z
    dst_off = torch.cumsum(out_n, 0) - out_n                     # start of graph i in output
    total = int(out_n.sum())
    pos_in_graph = torch.arange(total, device=n.device) - torch.repeat_interleave(dst_off, out_n)
    within = pos_in_graph % torch.repeat_interleave(n, out_n)    # the tile
    return torch.repeat_interleave(src_off, out_n) + within, out_n


def batch_to_fairchem_batch(batch, std_orientation, pbc=True, force_rebuild=False,
                            task_name='omc'):
    """
    Build the COLLATED fairchem AtomicData batch directly, with no per-graph Python.

    WHY. The list path below builds one AtomicData per crystal and collates them one
    at a time. At 1000 crystals that is ~52k scalar extractions -- ~29k of them true
    CUDA syncs -- and ~720 ms of host time before a single UMA kernel launches, and
    it runs TWICE per lattice energy (crystal leg + gas leg). The GPU idles through
    all of it, which is what a 45%-utilization/214 W signature looks like. Measured
    720 ms -> ~3 ms at B=1000.

    This is a CONSTRUCTION change only: same tensors, same dtypes, same order, so the
    model sees bit-identical input and every energy is unchanged. That claim is
    enforced, not asserted -- see verify_fairchem_batch_equivalence and
    tests/test_uma_atomicdata_vectorisation.py, which check it against the list path
    over Z'=1 and Z'>1, supercell and unit-cell batches, and ragged sym_mult.
    """
    device = batch.device
    z, batch_ind = _resolve_aunit_z(batch, std_orientation, force_rebuild)
    num_graphs = batch.num_graphs

    gather, out_n = _tiled_gather_index(batch_ind, batch.sym_mult, num_graphs)
    atomic_numbers = z[gather]
    pos = batch.unit_cell_pos
    n_tot = pos.shape[0]

    if int(out_n.sum()) != n_tot:
        # unit_cell_pos and sum(n_i * sym_mult_i) must agree, or positions and
        # elements are misaligned. Loud here beats a plausible wrong energy.
        raise RuntimeError(
            f'unit cell size mismatch: unit_cell_pos has {n_tot} atoms but aunit '
            f'counts x sym_mult give {int(out_n.sum())}. Batch is inconsistent.')

    out = AtomicData(
        pos=pos,
        atomic_numbers=atomic_numbers,
        # per-graph cell was T_fc[i].T[None]; stacked that is a plain transpose
        cell=batch.T_fc.transpose(1, 2),
        pbc=torch.full((num_graphs, 3), bool(pbc), dtype=torch.bool, device=device),
        natoms=out_n.to(torch.long),
        batch=torch.repeat_interleave(
            torch.arange(num_graphs, device=device), out_n),
        edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
        cell_offsets=torch.empty((0, 3), dtype=torch.float32, device=device),
        nedges=torch.zeros(num_graphs, dtype=torch.long, device=device),
        charge=torch.zeros(num_graphs, dtype=torch.long, device=device),
        spin=torch.zeros(num_graphs, dtype=torch.long, device=device),
        fixed=torch.zeros(n_tot, dtype=torch.long, device=device),
        tags=torch.zeros(n_tot, dtype=torch.long, device=device),
        energy=None,
        forces=None,
        stress=None,
        # AtomicData.validate asserts len(sid) == num_graphs. The list path passes
        # sid=None per single-graph object, which AtomicData normalises to [''], so
        # the collated form is num_graphs empty strings -- match that exactly rather
        # than passing None, which only validates when num_graphs == 1.
        sid=[''] * num_graphs,
        dataset=[task_name] * num_graphs,
    )
    _assign_collation_bookkeeping(out, num_graphs, out_n)
    return out


def _assign_collation_bookkeeping(data, num_graphs, out_n):
    """
    Populate `__slices__`/`__cumsum__`/`__cat_dims__`/`__natoms_list__`, which
    `atomicdata_list_to_batch` sets and a directly-constructed AtomicData leaves None.

    NOT needed by the model: only `get_example` and `batch_to_atomicdata_list` read
    them, and nothing in fairchem's predict path or forward calls either (checked
    against fairchem-core in this venv). Populated anyway so the object is
    indistinguishable from a collated one -- leaving four attributes as None because
    "nothing reads them today" is exactly the kind of assumption that breaks on an
    upstream version bump, and the cost is ONE host sync per batch against the ~29k
    the list path was doing.
    """
    counts = out_n.tolist()                      # the one sync
    node_slices = [0]
    for c in counts:
        node_slices.append(node_slices[-1] + c)
    graph_slices = list(range(num_graphs + 1))
    zeros = [0] * (num_graphs + 1)

    # per-NODE fields concatenate along dim 0; per-GRAPH fields are one row each;
    # edges are empty here, so their slices are all zero
    node_fields = ('pos', 'atomic_numbers', 'fixed', 'tags', 'batch')
    graph_fields = ('natoms', 'nedges', 'charge', 'spin', 'sid', 'dataset', 'pbc', 'cell')
    edge_fields = ('edge_index', 'cell_offsets')

    slices, cumsum, cat_dims = {}, {}, {}
    for key in node_fields:
        slices[key], cumsum[key], cat_dims[key] = node_slices, zeros, 0
    for key in graph_fields:
        slices[key], cumsum[key], cat_dims[key] = graph_slices, zeros, 0
    for key in edge_fields:
        slices[key], cumsum[key] = [0] * (num_graphs + 1), zeros
        cat_dims[key] = -1 if key == 'edge_index' else 0
    # edge_index is offset by node counts when re-split, matching the collated form
    cumsum['edge_index'] = node_slices

    data.assign_batch_stats(slices, cumsum, cat_dims, counts)


#: Fields that must match between the list path and the vectorised one. `sid` is
#: excluded deliberately: the list path passes sid=None per graph and collation
#: turns that into its own placeholder, which is not read by the model.
_ATOMICDATA_COMPARE_FIELDS = ('pos', 'atomic_numbers', 'cell', 'pbc', 'natoms',
                              'batch', 'edge_index', 'cell_offsets', 'nedges',
                              'charge', 'spin', 'fixed', 'tags')


def verify_fairchem_batch_equivalence(batch, std_orientation, pbc=True,
                                      force_rebuild=False, task_name='omc'):
    """
    Build BOTH ways and assert the model-visible tensors are identical.

    Exact equality, not allclose: this is a reindexing, so any difference is a bug
    rather than a rounding difference. Note it must run on a batch whose unit cell
    is ALREADY built, because case 1 mutates `batch` -- calling it on a fresh batch
    would have the first path build the cell and the second see a different case.
    """
    ref = atomicdata_list_to_batch(
        batch_to_fairchem_atomicdata(batch, std_orientation, pbc, force_rebuild, task_name))
    new = batch_to_fairchem_batch(batch, std_orientation, pbc, force_rebuild, task_name)
    for field in _ATOMICDATA_COMPARE_FIELDS:
        a, b = getattr(ref, field), getattr(new, field)
        if a is None or b is None:
            if a is not b:
                raise AssertionError(f'{field}: one path produced None ({a} vs {b})')
            continue
        if a.shape != b.shape:
            raise AssertionError(f'{field}: shape {tuple(a.shape)} vs {tuple(b.shape)}')
        if a.dtype != b.dtype:
            raise AssertionError(f'{field}: dtype {a.dtype} vs {b.dtype}')
        if not torch.equal(a, b):
            bad = int((a != b).sum())
            raise AssertionError(f'{field}: {bad}/{a.numel()} elements differ')
    return new


def batch_to_fairchem_atomicdata(batch, std_orientation, pbc=True, force_rebuild=False, task_name='omc'):
    """
    LIST PATH -- one AtomicData per crystal. Retained as the reference the
    vectorised builder is checked against (and as the fallback), not because
    anything should call it in a hot loop.
    """
    device = batch.device
    data_list = []
    if batch.unit_cell_pos is None or force_rebuild:
        if batch.z_prime.amax() > 1:
            zp1_batch = batch.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            batch.join_zp1_ucell_batch(zp1_batch)
        else:
            batch.pose_aunit(std_orientation=std_orientation)
            batch.build_unit_cell()
        z = batch.z
        batch_ind = batch.batch

    elif hasattr(batch, 'aux_ind'):
        if batch.aux_ind is not None:
            z = batch.z[batch.aux_ind == 0]
            batch_ind = batch.batch[batch.aux_ind == 0]
        else:
            z = batch.z
            batch_ind = batch.batch
    else:
        z = batch.z
        batch_ind = batch.batch

    ucell_pos = batch.unit_cell_pos
    ucell_batch = batch.unit_cell_batch
    sym_mult = batch.sym_mult
    for ind in range(batch.num_graphs):
        pos = ucell_pos[ucell_batch == ind]
        sample_z = z[batch_ind == ind].repeat(sym_mult[ind])
        atomic_data = AtomicData(
            pos=pos,
            atomic_numbers=sample_z,
            cell=batch.T_fc[ind].T[None, ...],
            pbc=torch.tensor([[pbc, pbc, pbc]], dtype=bool, device=device),
            natoms=torch.tensor([len(pos)], dtype=torch.long, device=device),
            edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
            cell_offsets=torch.empty((0, 3), dtype=torch.float32, device=device),
            nedges=torch.zeros(1, dtype=torch.long, device=device),
            charge=torch.zeros(1, dtype=torch.long, device=device),
            spin=torch.zeros(1, dtype=torch.long, device=device),
            fixed=torch.zeros(len(pos), dtype=torch.long, device=device),
            tags=torch.zeros(len(pos), dtype=torch.long, device=device),
            energy=None,
            forces=None,
            stress=None,
            sid=None,
            dataset=task_name
        )
        data_list.append(atomic_data)

    return data_list


def compute_molecule_uma_on_mxt_batch(batch,
                                      predictor: Optional = None):
    data_list = []

    for ind in range(batch.num_graphs):  # todo a Z'>1 fix for this
        pos = batch.pos[batch.batch == ind]
        z = batch.z[batch.batch == ind]
        atoms = Atoms(
            numbers=z.cpu().detach().numpy(),
            positions=pos.cpu().detach().numpy(),
        )
        pbc_flag = False
        atoms.set_pbc(pbc_flag)
        crystal = AtomicData.from_ase(atoms, task_name='omol')
        data_list.append(crystal)

    uma_batch = atomicdata_list_to_batch(data_list)

    out, crashed = safe_predict_uma(predictor, uma_batch)
    if crashed:
        energy = torch.zeros(batch.num_graphs, dtype=torch.float32, device=batch.device)
    else:
        energy = out['energy']

    return energy


def crystal_inference_settings(activation_checkpointing: bool = True,
                               compile: bool = False,
                               edge_chunk_size: Optional[int] = None):
    """
    Our own InferenceSettings, rather than the 'default' preset mutated in place.

    WHY NOT MUTATE THE PRESET. fairchem's NAME_TO_INFERENCE_SETTING holds module-level
    SINGLETON instances, and guess_inference_settings returns the object, not a copy.
    Setting .tf32 on it therefore reconfigured 'default' for the whole process --
    including any other predictor built later. Constructing our own removes that
    action at a distance. That is the only unconditional change here; the defaults
    below reproduce the previous behaviour exactly.

    ACTIVATION CHECKPOINTING DEFAULTS TO TRUE, and the reason is subtler than it
    looks. fairchem's _run_inference does

        inference_context = torch.no_grad() if self.direct_forces else nullcontext()

    -- INVERTED from the obvious reading: with direct_forces=False (what this
    predictor sets, because forces would otherwise come from differentiating the
    energy) fairchem deliberately leaves grad ENABLED. regress_forces/regress_stress
    False do gate the autograd.grad calls, so no backward is taken, but whether a
    GRAPH IS BUILT depends on the ambient context:

      * training reward -- safe. get_loss_reward wraps the call in torch.no_grad()
        whenever reward_grads == 0, and nullcontext cannot undo that. Checkpointing
        is effectively inert, so turning it off is close to free.
      * reward_grads != 0 -- the call runs under enable_grad by design. Checkpointing
        is load-bearing.
      * the init-time prior re-analysis (train.py's "Re-analyzing prior energies",
        ~176k rows) -- calls batched_analyze_crystal_batch directly with NO outer
        no_grad. keep_grads=False only detaches the OUTPUT; it does not stop the
        graph being built. Checkpointing is load-bearing here too, and this is the
        pass whose chunk size collapsed to 144 on the uma arm.

    So this is a memory/speed trade in a regime that has been OOMing, not a free win.
    Pass activation_checkpointing=False only for a run with reward_grads == 0, and
    watch peak memory -- the MLIP shares the allocator with the rollout's activations.

    COMPILE is off by default but is a real option on the cluster -- see
    init_uma_crystal_predictor's docstring for when it pays and what to set
    edge_chunk_size to.

    merge_mole is NOT exposed: it asserts natoms.numel() == 1 and assumes fixed
    composition/charge/spin, which a batch of different crystals violates. That is
    why the 'turbo' preset cannot be adopted wholesale despite its other flags.
    """
    return InferenceSettings(
        tf32=True,                      # what the old code set, minus the global mutation
        activation_checkpointing=activation_checkpointing,
        merge_mole=False,               # never safe for a multi-crystal batch
        compile=compile,
        external_graph_gen=False,
        internal_graph_gen_version=2,
        edge_chunk_size=edge_chunk_size,
    )


def init_uma_crystal_predictor(model_path, device, inference_settings=None,
                               compile: bool = False,
                               edge_chunk_size: Optional[int] = None,
                               activation_checkpointing: bool = True):
    """
    COMPILING THE MLIP -- when it pays, and what has to be true first.

    The obvious objection is that torch.compile specialises on shapes and a crystal
    batch is ragged. But look at what actually varies HERE, on a production arm:

      * ATOM COUNT IS FIXED. Each prod0810 arm is a single molecule at fixed Z' and a
        single space group, so atoms-per-crystal is constant and atoms-per-batch is
        batch_size x const. The batch size only changes at a growth/OOM event, which
        the controller deliberately makes rare (~log_f(max/base) distinct sizes).
      * EDGE COUNT VARIES, every step, with the cell geometry. This is the only truly
        dynamic dimension -- and `edge_chunk_size` exists precisely to pad it into
        buckets so the compiled graph sees a fixed shape.

    So the shape argument against compiling is much weaker than for a general-purpose
    MLIP workload, and this is worth measuring rather than assuming. Set
    edge_chunk_size to comfortably above the observed max edges per batch (log it
    first); too small silently forces recompiles, too large wastes compute on padding.

    THE ABOVE HOLDS ONLY FOR AN UNCONDITIONAL / SINGLE-MOLECULE ARM. On a
    MOLECULE-CONDITIONED run the premise fails at its root: atoms-per-crystal is a
    property of the sampled molecule, so atoms-per-batch changes every step and the
    ATOM dimension becomes dynamic too. edge_chunk_size pads edges, not atoms, so it
    cannot rescue that case. Expect either continuous recompilation or -- once
    dynamo's cache_size_limit is hit -- a SILENT fall back to eager, for the policy
    as well as the MLIP, since they share that budget. Treat compile as an
    unconditional-arm option until someone measures a conditional one; a padded/
    bucketed atom dimension would be the thing to try, and that is a design change,
    not a flag.

    WHAT TO WATCH, because both failure modes are quiet:
      * The GFN already compiles its policy trunk. Two compiled domains share
        dynamo's cache_size_limit, and exceeding it falls back to EAGER SILENTLY --
        for the policy as well as the MLIP. Check for `torch._dynamo` recompile
        warnings before believing a null result.
      * First call pays the compile (tens of seconds); with `traj_checkpoint` on, the
        rollout is already slower, so a compile stall inside the first energy call can
        look like a hang.
      * Compilation changes kernel fusion and therefore float summation ORDER, so
        energies can move at ULP level. Run tests/test_uma_gpu_real_batches.py with
        and without to bound it.

    Off by default so this is an opt-in cluster experiment, exactly like
    compile_policy -- not a silent change to every run.
    """
    if inference_settings is None:
        inference_settings = crystal_inference_settings(
            activation_checkpointing=activation_checkpointing,
            compile=compile,
            edge_chunk_size=edge_chunk_size)
    return _build_uma_crystal_predictor(model_path, device, inference_settings)


def _build_uma_crystal_predictor(model_path, device, inference_settings):
    predictor = pretrained_mlip.load_predict_unit(
        model_path,
        inference_settings=inference_settings,
        # `always_use_pbc: True` was here and was DEAD: fairchem's
        # _build_overrides_from_settings shallow-copies this dict, so its
        # "user overrides win" merge is d.update(d) and the value stayed False.
        # Removed rather than left in place -- it protected nothing, and if that
        # aliasing is ever fixed upstream it would force periodic images onto the
        # ISOLATED-MOLECULE leg, whose diffuse cell is not enlarged, silently
        # corrupting every lattice energy. pbc is read per-sample from the data,
        # which is the correct behaviour and what actually happens today.
        overrides={"backbone": {"direct_forces": False,
                                "regress_forces": False,
                                "regress_stress": False}},
        device=device,
    )
    predictor.tasks.pop('omc_forces')
    predictor.tasks.pop('omc_stress')
    predictor.dataset_to_tasks['omc'].pop(1)
    predictor.dataset_to_tasks['omc'].pop(1)

    return predictor


def init_uma_mol_predictor(model_path):
    predictor = pretrained_mlip.load_predict_unit(
        model_path,
        inference_settings='default',
        overrides={"backbone": {  # "always_use_pbc": True,
            "direct_forces": True}},
        device="cuda",
    )
    predictor.inference_mode.compile = False
    predictor.inference_mode.tf32 = True

    # some savings if we do energy-only
    # don't do this if you want forces and stresses
    # admittedly this is only partially effective - I can't kill the backward call completely for come reason
    energy_task = predictor.tasks["omol_energy"]
    predictor.tasks = {"omol_energy": energy_task}
    predictor.dataset_to_tasks["omol"] = [
        t for t in predictor.dataset_to_tasks["omol"] if "energy" in t.name
    ]
    return predictor
