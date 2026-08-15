from mace.tools.torch_geometric.dataloader import Collater

import os
import time

import numpy as np
import torch
from mace.data import AtomicData, get_neighborhood, utils
from mace.tools import to_one_hot, atomic_numbers_to_indices

from mxtaltools.common.geometry_utils import fractional_transform
from mxtaltools.common.utils import is_cuda_oom
# SHARED ON PURPOSE: the tiling rule is the one piece of index maths both MLIP
# builders must get identically right, and it is already pinned by
# tests/test_uma_atomicdata_vectorisation.py::test_tiling_is_not_repeat_interleave.
# A second copy here would be a second thing to get wrong.
from mxtaltools.mlip_interfaces.uma_utils import _tiled_gather_index


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
    builder = (batch_to_mace_atomicdata_hoisted if USE_HOISTED_MACE_ATOMICDATA
               else batch_to_mace_atomicdata)
    dataset = builder(batch, force_rebuild, model, std_orientation, pbc=pbc)
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


#: Use the hoisted AtomicData builder. Same kill-switch shape as uma_utils'
#: USE_VECTORISED_ATOMICDATA: overridable from the environment so the fallback can be
#: taken WITHOUT editing source, because an A/B between the two paths must run
#: identical source or it measures the edit as well as the change.
USE_HOISTED_MACE_ATOMICDATA = os.environ.get('MXT_HOISTED_MACE_ATOMICDATA', '1') != '0'

#: Replace the per-graph matscipy neighbour list with the batched on-device one
#: (pbc_neighbours.batched_pbc_neighbour_list).
#:
#: DEFAULT OFF, deliberately. The edge SETS are verified identical to matscipy on
#: synthetic and real crystals, but as of 2026-08-14 no ENERGY has ever been computed
#: through this path -- and an edge-set match is a necessary, not sufficient,
#: condition for an unchanged energy. Flip it on only once
#: test_mace_gpu_real_batches.py::test_energies_match_with_batched_neighbours has run
#: on hardware that can execute a MACE forward.
#:
#: Measured 10.8x over matscipy at 256 graphs on GPU, and the gap widens with batch
#: size because the GPU cost is launch-dominated and nearly flat.
USE_BATCHED_MACE_NEIGHBOURS = os.environ.get('MXT_BATCHED_MACE_NEIGHBOURS', '0') != '0'

#: Fields the two builders must agree on EXACTLY. Everything the model reads, plus
#: the bookkeeping that decides how it is read.
_MACE_COMPARE_FIELDS = (
    'edge_index', 'positions', 'shifts', 'unit_shifts', 'cell', 'node_attrs',
    'batch', 'ptr', 'head', 'pbc', 'weight', 'energy_weight', 'forces_weight',
    'stress_weight', 'virials_weight', 'dipole_weight', 'charges_weight',
    'polarizability_weight', 'energy', 'forces', 'stress', 'virials', 'dipole',
    'charges', 'polarizability', 'elec_temp', 'total_charge', 'total_spin',
)


def _mace_resolve_aunit_z(batch, force_rebuild, std_orientation):
    """
    The aunit atomic numbers and their graph index, unit cell built if absent.

    Lifted verbatim from batch_to_mace_atomicdata so the two builders cannot drift.
    Deliberately NOT uma_utils._resolve_aunit_z, despite the same three cases: that
    one tests `batch.unit_cell_pos is None` and would raise AttributeError on a batch
    that has no such attribute at all, which this path explicitly handles via hasattr.
    Sharing it would have silently changed the behaviour of the attribute-less case.
    """
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
        return batch.z, batch.batch

    if batch.aux_ind is not None:
        keep = batch.aux_ind == 0
        return batch.z[keep], batch.batch[keep]

    return batch.z, batch.batch


def batch_to_mace_atomicdata_hoisted(batch, force_rebuild, model, std_orientation,
                                     pbc: bool = True):
    """
    The same AtomicData list, with every per-graph operation that does NOT need to be
    per-graph hoisted out of the loop.

    WHAT MOVES, and why these and not others. Measured split of the build at 128
    graphs (scratch profile, 2026-08-14): get_neighborhood 64.9%, collate 10.4%,
    AtomicData construction 9.5%, tensor slicing 8.5%, one-hot 5.8%, .cpu()/.numpy()
    0.9%. The neighbour list genuinely cannot be batched -- every graph has its own
    cell, so every graph needs its own call -- but everything around it can:

      * ONE host transfer for the whole batch instead of three per graph
        (positions, cell, atomic numbers). At 1000 graphs that is 3 syncs, not 3000.
      * ONE tiled gather for the elements, via uma_utils._tiled_gather_index, which
        reproduces `z[batch == i].repeat(sym_mult[i])` for every graph at once. TILED,
        not repeat_interleaved -- unit_cell_pos is laid out image-major and getting
        this backwards pairs every position with the wrong element while still
        producing a perfectly well-formed batch.
      * ONE one-hot over the whole batch, sliced per graph.
      * The ~14 CONSTANT per-graph tensors (weights, zeroed targets) built once and
        shared, rather than reallocated per crystal.

    Bit-identical to batch_to_mace_atomicdata by construction: same tensors, same
    dtypes, same order. Enforced by verify_mace_atomicdata_equivalence and
    tests/test_mace_atomicdata_vectorisation.py, not merely claimed.

    NOT YET HOISTED: the Collater call (10.4%). Building the collated Batch directly
    means reproducing PyG's _slice_dict/_inc_dict internals, which is a separate and
    riskier change; this one is a pure loop-invariant hoist.
    """
    z, batch_ind = _mace_resolve_aunit_z(batch, force_rebuild, std_orientation)

    cutoff = float(model.r_max)
    z_table = utils.AtomicNumberTable([int(zz) for zz in model.atomic_numbers])

    gather, out_n = _tiled_gather_index(batch_ind, batch.sym_mult, batch.num_graphs)
    sample_z_all = z[gather]

    pos_all = batch.unit_cell_pos
    if int(out_n.sum()) != pos_all.shape[0]:
        # positions and elements must agree, or every energy is quietly wrong
        raise RuntimeError(
            f'unit_cell_pos has {pos_all.shape[0]} atoms but sum(n_i * sym_mult_i) '
            f'is {int(out_n.sum())} -- positions and elements are misaligned')

    # --- the three host transfers, once each for the whole batch ---
    pos_np_all = pos_all.detach().cpu().numpy()
    cell_np_all = batch.T_fc.transpose(-2, -1).detach().cpu().numpy()
    z_np_all = sample_z_all.detach().cpu().numpy()

    # --- one one-hot for the batch ---
    indices_all = atomic_numbers_to_indices(z_np_all, z_table=z_table)
    one_hot_all = to_one_hot(torch.tensor(indices_all, dtype=torch.long).unsqueeze(-1),
                             num_classes=len(z_table))

    dtype = pos_all.dtype
    # --- constants, built once. Shared across graphs: nothing mutates them, and
    #     Collater concatenates rather than aliases, so sharing is safe. ---
    k_head = torch.tensor(0, dtype=torch.long)
    k_pbc = torch.tensor([[pbc, pbc, pbc]], dtype=torch.bool)
    k_one = torch.tensor(1.0, dtype=dtype)
    k_zero = torch.tensor(0.0, dtype=dtype)
    k_spin = torch.tensor(1.0, dtype=dtype)
    k_dipw = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
    k_polw = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], dtype=dtype)
    k_33 = torch.zeros(1, 3, 3, dtype=dtype)
    k_13 = torch.zeros(1, 3, dtype=dtype)

    offsets_t = torch.cumsum(out_n, 0) - out_n
    offsets = offsets_t.detach().cpu().numpy()
    counts = out_n.detach().cpu().numpy()

    # --- optional: one batched, on-device neighbour list for the whole batch ---
    #
    # SCOPED TO pbc=True ON PURPOSE. The non-periodic branch of get_neighborhood
    # REWRITES the cell in place (cell[ax] = max_positions * 5 * cutoff * I[ax]) and
    # the caller uses the returned cell, so the gas leg is not a pure neighbour-list
    # question. Leaving it on the reference keeps that side effect exactly as it is.
    nl = None
    if USE_BATCHED_MACE_NEIGHBOURS and pbc:
        from mxtaltools.mlip_interfaces.pbc_neighbours import batched_pbc_neighbour_list
        cells_t = batch.T_fc.transpose(-2, -1)
        ucell_graph = torch.repeat_interleave(
            torch.arange(batch.num_graphs, device=pos_all.device), out_n)
        e_i, e_s, e_g = batched_pbc_neighbour_list(
            pos_all, ucell_graph, cells_t, cutoff, pbc=True)
        nl = (e_i, e_s, e_g)

    dataset = []
    for ind in range(batch.num_graphs):
        lo = int(offsets[ind])
        hi = lo + int(counts[ind])
        pos = pos_all[lo:hi]
        if nl is None:
            edge_index, shifts, unit_shifts, cell = get_neighborhood(
                positions=pos_np_all[lo:hi], cutoff=cutoff,
                pbc=[pbc, pbc, pbc], cell=cell_np_all[ind])
        else:
            e_i, e_s, e_g = nl
            m = e_g == ind
            edge_index = (e_i[:, m] - lo).cpu().numpy()      # global -> per-graph
            unit_shifts = e_s[m].to(pos_all.dtype)
            cell = cell_np_all[ind]
            shifts = (unit_shifts @ torch.as_tensor(
                cell, dtype=pos_all.dtype, device=pos_all.device)).cpu().numpy()
            unit_shifts = unit_shifts.cpu().numpy()
        num_atoms = hi - lo
        dataset.append(AtomicData(
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            positions=pos,
            shifts=torch.as_tensor(shifts, dtype=dtype),
            unit_shifts=torch.as_tensor(unit_shifts, dtype=dtype),
            cell=cell,
            node_attrs=one_hot_all[lo:hi],
            head=k_head, pbc=k_pbc,
            weight=k_one, energy_weight=k_one, forces_weight=k_one,
            stress_weight=k_one, virials_weight=k_one,
            dipole_weight=k_dipw, charges_weight=k_one,
            polarizability_weight=k_polw,
            energy=k_zero, forces=torch.zeros(num_atoms, 3, dtype=dtype),
            stress=k_33, virials=k_33, dipole=k_13,
            charges=torch.zeros(num_atoms, dtype=dtype),
            polarizability=k_33,
            elec_temp=k_zero, total_charge=k_zero, total_spin=k_spin,
        ))
    return dataset


def verify_mace_atomicdata_equivalence(batch, force_rebuild, model, std_orientation,
                                       pbc: bool = True):
    """
    Build BOTH ways, collate both, and assert every model-visible tensor is identical.

    EXACT equality, not allclose: this is a loop-invariant hoist, so any difference at
    all is a bug rather than a rounding difference.

    Must run on a batch whose unit cell is ALREADY built. The no-unit-cell branch
    MUTATES `batch` (it builds unit_cell_pos), so on a fresh batch the second path
    would take a different branch than the first and the two would not be comparable.
    """
    collate = Collater([None], [None])
    ref = collate(batch_to_mace_atomicdata(batch, force_rebuild, model,
                                           std_orientation, pbc=pbc))
    new = collate(batch_to_mace_atomicdata_hoisted(batch, force_rebuild, model,
                                                   std_orientation, pbc=pbc))
    bad = mace_batch_differences(ref, new)
    if bad:
        raise AssertionError('hoisted mace AtomicData differs from the list path:\n  '
                             + '\n  '.join(bad))
    return True


def mace_batch_differences(ref, new, fields=None):
    """
    Field-by-field differences between two collated mace batches, as a list of
    human-readable strings ([] means identical).

    ONE comparator, exported, because the obvious hand-rolled version is wrong in a
    way that looks right: not every collated field is a tensor -- PyG leaves some as
    PYTHON LISTS -- so `a.dtype` raises AttributeError instead of comparing. That was
    written twice and got caught twice; there is now one copy.
    """
    bad = []
    for field in (fields or _MACE_COMPARE_FIELDS):
        a, b = getattr(ref, field, None), getattr(new, field, None)
        if a is None or b is None:
            if a is not b:
                bad.append(f'{field}: one path has it, the other does not')
            continue
        # NOT every collated field is a tensor. PyG leaves some as PYTHON LISTS
        # (per-graph values it has no cat_dim for), and assuming `.dtype` here made
        # this checker raise AttributeError instead of comparing -- i.e. it failed
        # loudly on the fields it was least able to check, which is the good
        # direction to fail but useless as a verifier.
        if torch.is_tensor(a) != torch.is_tensor(b):
            bad.append(f'{field}: {type(a).__name__} vs {type(b).__name__}')
        elif torch.is_tensor(a):
            if a.dtype != b.dtype:
                bad.append(f'{field}: dtype {a.dtype} vs {b.dtype}')
            elif a.shape != b.shape:
                bad.append(f'{field}: shape {tuple(a.shape)} vs {tuple(b.shape)}')
            elif not torch.equal(a, b):
                bad.append(f'{field}: values differ (max |d| '
                           f'{(a - b).abs().max().item() if a.is_floating_point() else "n/a"})')
        else:
            if len(a) != len(b):
                bad.append(f'{field}: length {len(a)} vs {len(b)}')
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    same = (torch.equal(x, y) if torch.is_tensor(x) and torch.is_tensor(y)
                            else bool(np.array_equal(np.asarray(x), np.asarray(y))))
                    if not same:
                        bad.append(f'{field}[{i}]: differs')
                        break
    return bad


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
