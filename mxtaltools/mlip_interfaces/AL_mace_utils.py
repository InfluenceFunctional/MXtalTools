from mace.tools.torch_geometric.dataloader import Collater

import time

import torch
from mace.data import AtomicData, get_neighborhood, utils
from mace.tools import to_one_hot, atomic_numbers_to_indices

from mxtaltools.common.geometry_utils import fractional_transform
from mxtaltools.common.utils import is_cuda_oom


def load_mace_model(model_path, device, dtype):
    import torch.fx._symbolic_trace as _st
    if not hasattr(_st, 'is_fx_symbolic_tracing'):
        _st.is_fx_symbolic_tracing = _st.is_fx_tracing
    _original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        # Force weights_only=False if not specified
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    model = torch.load(f=model_path, map_location=device)
    model.to(dtype=dtype)

    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
        import cuequivariance_ops_torch  # noqa: F401
        _CUEQ_AVAILABLE = True
    except ImportError:
        _CUEQ_AVAILABLE = False

    use_cueq = _CUEQ_AVAILABLE and torch.cuda.is_available()
    if use_cueq:
        print("cuequivariance (with ops) detected, enabling...")
        model = run_e3nn_to_cueq(model)

    model.to(device, dtype=dtype)
    model.eval()

    return model


def mxt_crystal_to_mace_atomicdata(batch,
                                   unit_cell_batch,
                                   unit_cell_pos,
                                   mol_z,
                                   sym_mult,
                                   T_fc,
                                   dtype, cutoff, z_table, ind, pbc):
    mask = batch == ind
    ucell_mask = unit_cell_batch == ind
    pos = unit_cell_pos[ucell_mask]
    cell = T_fc[ind].T
    sample_z = mol_z[mask].repeat(sym_mult[ind])

    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=pos.cpu().detach().numpy(), cutoff=cutoff,
        pbc=[pbc, pbc, pbc], cell=cell.cpu().detach().numpy()
    )
    indices = atomic_numbers_to_indices(sample_z.cpu().numpy(), z_table=z_table)
    one_hot = to_one_hot(torch.tensor(indices, dtype=torch.long).unsqueeze(-1), num_classes=len(z_table))
    num_atoms = len(sample_z)

    mace_data = AtomicData(
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        positions=pos,
        shifts=torch.as_tensor(shifts, dtype=dtype),
        unit_shifts=torch.as_tensor(unit_shifts, dtype=dtype),
        cell=cell,
        node_attrs=one_hot,
        head=torch.tensor(0, dtype=torch.long),  # → tensor(0)
        pbc=torch.tensor([[pbc, pbc, pbc]], dtype=torch.bool),
        weight=torch.tensor(1.0, dtype=dtype),
        energy_weight=torch.tensor(1.0, dtype=dtype),
        forces_weight=torch.tensor(1.0, dtype=dtype),
        stress_weight=torch.tensor(1.0, dtype=dtype),
        virials_weight=torch.tensor(1.0, dtype=dtype),
        dipole_weight=torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype),
        charges_weight=torch.tensor(1.0, dtype=dtype),
        polarizability_weight=torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=dtype),
        energy=torch.tensor(0.0, dtype=dtype),
        forces=torch.zeros(num_atoms, 3, dtype=dtype),
        stress=torch.zeros(1, 3, 3, dtype=dtype),
        virials=torch.zeros(1, 3, 3, dtype=dtype),
        dipole=torch.zeros(1, 3, dtype=dtype),
        charges=torch.zeros(num_atoms, dtype=dtype),
        polarizability=torch.zeros(1, 3, 3, dtype=dtype),
        elec_temp=torch.tensor(0.0, dtype=dtype),
        total_charge=torch.tensor(0.0, dtype=dtype),
        total_spin=torch.tensor(1.0, dtype=dtype),
    )
    return mace_data


#: Where a mace energy call's seconds go, accumulated across calls and drained by
#: MolecularCrystal.drain_energy_timing into energy/mace_* metrics.
#:
#: THE QUESTION IT ANSWERS. `build` + `collate` is the part a vectorised AtomicData
#: builder would replace; `forward` is the part it cannot touch. Vectorising is worth
#: doing iff the first is a large share of the total -- and that is NOT safe to assume.
#: The same argument for UMA produced a 52-249x microbenchmark and 1.38x end to end,
#: because the UMA forward turned out to be >99% of the call. MEASURE, then decide.
#:
#: Honest because safe_predict_mace synchronises: without that drain the forward
#: returns immediately and its time is charged to whichever later statement touches
#: the result, which would put the split exactly backwards.
_PHASE_SECONDS = {'build': 0.0, 'collate': 0.0, 'xfer': 0.0, 'forward': 0.0}
_PHASE_CALLS = 0


def drain_mace_phase_timing():
    """Pop the accumulated per-phase seconds. {} when nothing was timed, so a stage
    that never calls mace (bwd/dataset MLE) logs nothing rather than zeros."""
    global _PHASE_CALLS
    if not _PHASE_CALLS:
        return {}
    total = sum(_PHASE_SECONDS.values())
    out = {f'energy/mace_{k}_s': v for k, v in _PHASE_SECONDS.items()}
    out['energy/mace_calls'] = _PHASE_CALLS
    if total > 0:
        # THE decision number: what fraction of a mace call is vectorisable host work
        out['energy/mace_host_frac'] = (_PHASE_SECONDS['build']
                                        + _PHASE_SECONDS['collate']) / total
        out['energy/mace_forward_frac'] = _PHASE_SECONDS['forward'] / total
    for k in _PHASE_SECONDS:
        _PHASE_SECONDS[k] = 0.0
    _PHASE_CALLS = 0
    return out


def compute_crystal_mace_on_mxt_batch(batch, model,
                                      std_orientation=True, pbc: bool=True, force_rebuild: bool = False):
    global _PHASE_CALLS
    _t = time.perf_counter()
    dataset = batch_to_mace_atomicdata(batch, force_rebuild, model, std_orientation, pbc=pbc)
    _PHASE_SECONDS['build'] += time.perf_counter() - _t

    _t = time.perf_counter()
    collater = Collater([None], [None])
    mbatch = collater(dataset)
    _PHASE_SECONDS['collate'] += time.perf_counter() - _t

    _t = time.perf_counter()
    mbatch = mbatch.to(batch.device)
    input_data = mbatch.to_dict()

    frac_pos = fractional_transform(batch.unit_cell_pos, batch.T_cf[batch.unit_cell_batch])
    cart_pos = fractional_transform(frac_pos, batch.T_fc[batch.unit_cell_batch])
    input_data['positions'] = cart_pos.to(batch.device)

    graph_ind = batch.unit_cell_batch[mbatch.edge_index[0].cpu()]
    unit_shifts = input_data['unit_shifts']
    input_data['shifts'] = fractional_transform(unit_shifts, batch.T_fc[graph_ind].to(batch.device))
    _PHASE_SECONDS['xfer'] += time.perf_counter() - _t

    _t = time.perf_counter()
    output, crashed = safe_predict_mace(model, input_data)
    _PHASE_SECONDS['forward'] += time.perf_counter() - _t
    _PHASE_CALLS += 1

    if crashed:
        energy = torch.zeros(batch.num_graphs, dtype=torch.float32, device=batch.device)
    else:
        energy = output['energy']
    return energy


def safe_predict_mace(model, input_data):
    try:
        # RESTORED 2026-08-14, to match safe_predict_uma, which never lost them.
        #
        # These were commented out here and left live on the UMA path, and that is the
        # one structural difference between the two routes -- which is also how they
        # behave in production: the UMA arms run, the MACE arms collapse their batch
        # and get cancelled for low occupancy.
        #
        # They do more than surface errors. Without a drain the host runs ahead and
        # queues several calls' worth of kernels before any of their intermediates are
        # released, so peak RESERVED reflects several in-flight chunks rather than one.
        # `cuda_memory_fraction` is applied as a hard per-process cap, so that inflated
        # peak is not merely untidy -- it is the thing a later allocation fails against,
        # and the caller then reads the failure as "this batch is too big" and shrinks
        # a batch that was never the problem.
        #
        # Cost is one drain per energy call on a route where the MLIP forward is >99%
        # of that call. Measure it against vram/cached_mb and vram/peak_reserved_mb
        # (train.py vram_ledger) before concluding either way.
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # flush prior kernels
        out = model(input_data, compute_force=False, compute_stress=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # force errors to surface *here*
        return out, False  # False = no failure

    except RuntimeError as e:
        if not is_cuda_oom(e):
            print("MACE error")
            print(str(e))
            # reset the cuda context fully
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return None, True  # signal failure
        else:
            raise e


def batch_to_mace_atomicdata(batch, force_rebuild, model, std_orientation, pbc: bool=True):
    do_rebuild = False
    if not hasattr(batch, 'unit_cell_pos'):
        do_rebuild = True
    elif batch.unit_cell_pos is None:
        do_rebuild = True
    if force_rebuild:
        do_rebuild = True

    if do_rebuild:
        if batch.z_prime.amax() > 1:
            zp1_batch = batch.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            batch.join_zp1_ucell_batch(zp1_batch)
        else:
            batch.pose_aunit(std_orientation=std_orientation)
            batch.build_unit_cell()

        mol_z = batch.z
        mol_batch_inds = batch.batch

    elif batch.aux_ind is not None:
        mol_z = batch.z[batch.aux_ind == 0]
        mol_batch_inds = batch.batch[batch.aux_ind == 0]

    else:
        mol_z = batch.z
        mol_batch_inds = batch.batch

    cutoff = float(model.r_max)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    dataset = [mxt_crystal_to_mace_atomicdata(
        mol_batch_inds,
        batch.unit_cell_batch,
        batch.unit_cell_pos,
        mol_z,
        batch.sym_mult,
        batch.T_fc,
        batch.unit_cell_pos.dtype, cutoff, z_table, ind, pbc=pbc)
        for ind in range(batch.num_graphs)]
    return dataset
