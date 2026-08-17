"""
Profiling harness for the MACE / UMA single-point energy calls used in
MolCrystalAnalysis.compute_lattice_mace / compute_lattice_uma
(mxtaltools/dataset_utils/data_class_methods/crystal_analysis.py).

This is a standalone script - it is NOT wired into any test suite. Run it
directly in an environment with torch + mace-torch + fairchem-core + a GPU
(this sandbox does not have those installed, so this could not be executed
or verified here - check the printed numbers carefully the first time you run it).

What it measures
-----------------
1. Whether the `mol2cluster()` + `construct_radial_graph()` step that
   `analyze(['mace'])` / `analyze(['uma'])` currently performs unconditionally
   (see COMPUTES_REQUIRE_CLUSTER in crystal_analysis.py) is wasted work.
   Neither compute_crystal_mace nor compute_crystal_uma ever reads
   `edges_dict` or the exploded cluster positions - they only use
   `unit_cell_pos` / `unit_cell_batch` / `sym_mult` / `T_fc`, which come from
   the much cheaper `build_unit_cell()`. This block cross-checks that a
   "direct" call (skip mol2cluster, call compute_lattice_mace/uma straight
   on the un-clustered batch) gives the SAME energies, then times both paths.
2. Within MACE's per-crystal AtomicData construction, how much wall time is
   spent inside `mace.data.get_neighborhood` (the CPU/numpy neighbor list
   build) versus everything else (masking, one-hot encoding, tensor
   creation). This is the "duplicative neighbor list construction" you
   suspected - it's not literally duplicated against our own radial graph
   (the two use incompatible representations, see notes below), but it IS
   a serial, per-crystal, CPU-bound Python loop that doesn't batch across
   the crystal batch at all.
3. Same breakdown for UMA, via torch.profiler on the predictor.predict call,
   since UMA builds its own neighbor list internally (we pass empty
   edge_index/cell_offsets into fairchem's AtomicData) rather than in our
   code - so there's nothing to "de-duplicate" on our side for UMA.

Notes on why the vdW radial graph can't just be reused for MACE
-----------------------------------------------------------------
`construct_radial_graph` (used for lj/vdw/es/etc.) operates on the exploded,
non-periodic "cluster" (`mol2cluster` replicates whole unit cells out to
`supercell_size` and does a plain distance cutoff - no PBC bookkeeping
needed because periodic images are already explicit atoms).
MACE instead wants the compact single unit cell + true PBC (cell matrix,
pbc flags) and computes its own minimum-image neighbor list + per-edge
integer cell-shift vectors via `get_neighborhood`. Reusing the cluster's
edge list would require recovering, for every replica atom, which integer
cell offset it came from - `ucell2cluster` doesn't currently expose that,
so it isn't a drop-in swap. It IS possible (build the cluster only once at
a cutoff >= max(mace_r_max, uma_cutoff, classical_cutoff) and tag each
replica with its (nx, ny, nz) offset during instantiate_cluster, then filter
edges <= r_max per potential) but that's a real refactor of
crystal_building/utils.py, not a quick fix - only worth it if this profiling
shows neighbor-list construction, not #1 above, actually dominates.

Edit the constants below to match your machine, then:
    python examples/profile_mlip_energy.py
"""
import time
from contextlib import contextmanager

import torch

from mxtaltools.dataset_utils.utils import collate_data_list

# --- edit these ---
MACE_PATH = r"C:\Users\mikem\Downloads\acr_112025_mh1_stagetwo.model"
UMA_PATH = r"D:\crystal_datasets\esen_s.pt"
MINI_DATASET_PATH = "../mini_datasets/mini_CSD_dataset.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = (8, 32, 128)  # sweep - cluster overhead should grow faster than
# direct-path cost as this increases, if hypothesis #1 above is correct
N_REPS = 5  # first rep is warmup (cuda context / cudnn autotune) and excluded
RUN_TORCH_PROFILER = True  # extra torch.profiler table for the UMA forward pass
# ------------------


@contextmanager
def timer(label, store):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    store.setdefault(label, []).append(time.perf_counter() - t0)


def summarize(store, indent="  "):
    for label, times in store.items():
        used = times[1:] if len(times) > 1 else times  # drop warmup rep
        mean = sum(used) / len(used)
        print(f"{indent}{label:55s} mean={mean * 1000:9.2f} ms  (n={len(used)})")


def make_batch(n):
    example_crystals = torch.load(MINI_DATASET_PATH, weights_only=False)
    assert len(example_crystals) >= n, f"mini dataset only has {len(example_crystals)} crystals"
    batch = collate_data_list(example_crystals[:n])
    return batch.to(DEVICE) if DEVICE == "cuda" else batch


@contextmanager
def patch_get_neighborhood():
    """Isolate time spent in MACE's own per-crystal CPU neighbor-list build."""
    import mxtaltools.mlip_interfaces.AL_mace_utils as mace_mod
    store = {}
    original = mace_mod.get_neighborhood

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        out = original(*args, **kwargs)
        store.setdefault('  -> get_neighborhood only (CPU, per-crystal)', []).append(time.perf_counter() - t0)
        return out

    mace_mod.get_neighborhood = wrapped
    try:
        yield store
    finally:
        mace_mod.get_neighborhood = original


def profile_cluster_overhead_and_correctness(batch, mace_model, uma_predictor):
    """A/B: current analyze(['mace'/'uma']) path (builds cluster + edges_dict,
    unused by either MLIP) vs. calling compute_lattice_mace/uma directly on the
    un-clustered batch. Also checks the two give matching energies."""
    n = batch.num_graphs
    print(f"\n--- cluster-overhead check (n_crystals={n}) ---")

    store = {}
    mace_via_analyze = mace_direct = uma_via_analyze = uma_direct = None
    for rep in range(N_REPS):
        with timer('analyze([\'mace\']) [current path: mol2cluster + radial graph + mace]', store):
            b = batch.clone()
            out = b.analyze(['mace'], predictor=mace_model, cutoff=10, supercell_size=10)
            mace_via_analyze = out['mace'].detach()

        with timer('direct compute_lattice_mace (no mol2cluster at all)', store):
            b = batch.clone()
            mace_direct = b.compute_lattice_mace(mace_model).detach()

        with timer('analyze([\'uma\']) [current path: mol2cluster + radial graph + uma]', store):
            b = batch.clone()
            out = b.analyze(['uma'], predictor=uma_predictor, cutoff=10, supercell_size=10)
            uma_via_analyze = out['uma'].detach()

        with timer('direct compute_lattice_uma (no mol2cluster at all)', store):
            b = batch.clone()
            uma_direct = b.compute_lattice_uma(uma_predictor).detach()

    summarize(store)

    mace_ok = torch.allclose(mace_via_analyze, mace_direct, atol=1e-2, rtol=1e-3)
    uma_ok = torch.allclose(uma_via_analyze, uma_direct, atol=1e-2, rtol=1e-3)
    print(f"  mace energies match (analyze vs direct): {mace_ok}"
          f"  max abs diff={ (mace_via_analyze - mace_direct).abs().max().item():.4g}")
    print(f"  uma  energies match (analyze vs direct): {uma_ok}"
          f"  max abs diff={ (uma_via_analyze - uma_direct).abs().max().item():.4g}")
    if not (mace_ok and uma_ok):
        print("  !! energies differ more than tolerance - do NOT skip mol2cluster without investigating why")


def profile_mace_internals(batch, mace_model):
    n = batch.num_graphs
    print(f"\n--- MACE internal breakdown (n_crystals={n}) ---")
    store = {}
    with patch_get_neighborhood() as nl_store:
        for rep in range(N_REPS):
            b = batch.clone()
            with timer('1. build_unit_cell', store):
                b.pose_aunit(std_orientation=True)
                b.build_unit_cell()
            with timer('2. atomicdata list build (masking + neighbor list + one-hot)', store):
                from mxtaltools.mlip_interfaces.AL_mace_utils import batch_to_mace_atomicdata
                dataset = batch_to_mace_atomicdata(b, force_rebuild=False, model=mace_model, std_orientation=True)
            with timer('3. collate + move to device', store):
                from mace.tools.torch_geometric.dataloader import Collater
                collater = Collater([None], [None])
                mbatch = collater(dataset).to(b.device)
            with timer('4. model forward (compute_crystal_mace_on_mxt_batch)', store):
                b.compute_crystal_mace(mace_model)

    summarize(store)
    print("  sub-breakdown of stage 2:")
    summarize(nl_store, indent="    ")
    stage2 = sum(store['2. atomicdata list build (masking + neighbor list + one-hot)'][1:])
    nl_time = sum(nl_store.get('  -> get_neighborhood only (CPU, per-crystal)', [0])[1:])
    if stage2 > 0:
        print(f"    => neighbor list is {100 * nl_time / stage2:.1f}% of stage-2 wall time")


def profile_uma_internals(batch, uma_predictor):
    n = batch.num_graphs
    print(f"\n--- UMA internal breakdown (n_crystals={n}) ---")
    store = {}
    for rep in range(N_REPS):
        b = batch.clone()
        with timer('1. build_unit_cell', store):
            b.pose_aunit(std_orientation=True)
            b.build_unit_cell()
        with timer('2. atomicdata list build (masking only - no neighbor list on our side)', store):
            from mxtaltools.mlip_interfaces.uma_utils import batch_to_fairchem_atomicdata
            data_list = batch_to_fairchem_atomicdata(b, std_orientation=True)
        with timer('3. collate + move to device', store):
            from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
            uma_batch = atomicdata_list_to_batch(data_list)
        with timer('4. model forward (predictor.predict, incl. its own internal neighbor search)', store):
            b.compute_crystal_uma(uma_predictor)
    summarize(store)

    if RUN_TORCH_PROFILER and DEVICE == "cuda":
        print("  torch.profiler breakdown of one predictor.predict call (top 15 by self CUDA time):")
        from mxtaltools.mlip_interfaces.uma_utils import batch_to_fairchem_atomicdata
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
        b = batch.clone()
        b.pose_aunit(std_orientation=True)
        b.build_unit_cell()
        data_list = batch_to_fairchem_atomicdata(b, std_orientation=True)
        uma_batch = atomicdata_list_to_batch(data_list)
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            with torch.no_grad():
                uma_predictor.predict(uma_batch)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))


def profile_batch_construction_scaling(batch):
    """Cheap, dependency-free check of the O(N_atoms * N_graphs) boolean-mask
    pattern (`batch.batch == ind`) used inside mxt_crystal_to_mace_atomicdata /
    batch_to_fairchem_atomicdata's per-graph loop, isolated from any MLIP call."""
    n = batch.num_graphs
    print(f"\n--- masking-loop micro-benchmark (n_crystals={n}) ---")
    store = {}
    for rep in range(N_REPS):
        with timer('current: boolean mask per graph (batch.batch == ind)', store):
            for ind in range(batch.num_graphs):
                _ = batch.pos[batch.batch == ind]
        with timer('alternative: single torch.split by num_atoms (assumes sorted batch)', store):
            counts = torch.bincount(batch.batch, minlength=batch.num_graphs).tolist()
            _ = torch.split(batch.pos, counts)
    summarize(store)


if __name__ == '__main__':
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

    mace_model = load_mace_model(MACE_PATH, device=DEVICE, dtype=torch.float32)
    uma_predictor = init_uma_crystal_predictor(UMA_PATH, device=DEVICE)

    for n in BATCH_SIZES:
        print(f"\n{'=' * 20} batch size = {n} {'=' * 20}")
        batch = make_batch(n)
        profile_batch_construction_scaling(batch)
        profile_cluster_overhead_and_correctness(batch, mace_model, uma_predictor)
        profile_mace_internals(batch, mace_model)
        profile_uma_internals(batch, uma_predictor)

    print("\nDone. Read the '--- cluster-overhead check ---' sections first: "
          "if 'analyze([...])' is much slower than the 'direct compute_lattice_*' "
          "call at every batch size, and the energies match, that mol2cluster + "
          "construct_radial_graph step is pure waste for MLIP-only requests and "
          "is the highest-value, lowest-risk fix (change COMPUTES_REQUIRE_CLUSTER "
          "in crystal_analysis.py for the uma/mace keys to False).")
