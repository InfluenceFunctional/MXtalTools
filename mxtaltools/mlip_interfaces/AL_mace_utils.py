from typing import Optional

import numpy as np
import torch
from ase import Atoms

from mxtaltools.common.utils import is_cuda_oom

# NOTE: MACE / e3nn imports are deferred to init_* functions so that importing
# this module does not require MACE to be installed.


# ---------------------------------------------------------------------------
# torch.load monkeypatch (MACE checkpoints predate weights_only=True default)
# ---------------------------------------------------------------------------
_original_torch_load = torch.load
_patched = False


def _patch_torch_load():
    """Force weights_only=False for MACE checkpoint compatibility."""
    global _patched
    if _patched:
        return

    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    _patched = True


# ---------------------------------------------------------------------------
# Safe predict
# ---------------------------------------------------------------------------
def safe_predict_mace(predictor,
                      mace_batch):
    """
    Run MACE forward with CUDA OOM handling, mirroring safe_predict_uma.
    Returns (output_dict, crashed_flag).
    """
    try:
        torch.cuda.synchronize()
        out = predictor["model"](
            mace_batch,
            compute_force=False,
            compute_stress=False,
        )
        torch.cuda.synchronize()
        return out, False

    except RuntimeError as e:
        if not is_cuda_oom(e):
            print("MACE error")
            print(str(e))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return None, True
        else:
            raise e


# ---------------------------------------------------------------------------
# Batch -> MACE AtomicData
# ---------------------------------------------------------------------------
def batch_to_mace_atomicdata(batch,
                             predictor,
                             std_orientation: bool = True,
                             pbc: bool = True,
                             force_rebuild: bool = False):
    """
    Convert an mxt batch to a batched MACE AtomicData object, routed through ASE.
    Reuses batch_to_ase_ucell_list from the UMA utils to avoid duplication.
    """
    # Deferred MACE imports
    from mace import data as mace_data
    from mace.tools import torch_geometric

    # Reuse the existing ASE conversion from the UMA utils
    from mxtaltools.models.functions.AL_uma_utils import batch_to_ase_ucell_list

    atoms_list = batch_to_ase_ucell_list(
        batch,
        std_orientation=std_orientation,
        pbc=pbc,
        force_rebuild=force_rebuild,
    )

    configs = [mace_data.config_from_atoms(atoms) for atoms in atoms_list]

    atomic_data_list = [
        mace_data.AtomicData.from_config(
            config,
            z_table=predictor["z_table"],
            cutoff=predictor["r_max"],
            heads=predictor["heads"],
        )
        for config in configs
    ]

    # Collate into a single batch (equivalent to a DataLoader with batch_size=len)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data_list,
        batch_size=len(atomic_data_list),
        shuffle=False,
        drop_last=False,
    )
    mace_batch = next(iter(loader)).to(predictor["device"])
    return mace_batch.to_dict()


def molecule_batch_to_mace_atomicdata(batch, predictor, pbc: bool = False):
    """
    Non-periodic molecule variant: convert an mxt batch's ASU/molecule positions
    directly to MACE AtomicData without any unit-cell construction.
    """
    from mace import data as mace_data
    from mace.tools import torch_geometric

    device = predictor["device"]

    cpu_z = batch.z.cpu().detach().numpy()
    cpu_pos = batch.pos.cpu().detach().numpy()
    cpu_batch_ind = batch.batch.cpu().detach().numpy()

    atoms_list = []
    for ind in range(batch.num_graphs):
        mask = cpu_batch_ind == ind
        atoms = Atoms(
            numbers=cpu_z[mask],
            positions=cpu_pos[mask],
        )
        atoms.set_pbc(pbc)
        atoms_list.append(atoms)

    configs = [mace_data.config_from_atoms(atoms) for atoms in atoms_list]
    atomic_data_list = [
        mace_data.AtomicData.from_config(
            config,
            z_table=predictor["z_table"],
            cutoff=predictor["r_max"],
            heads=predictor["heads"],
        )
        for config in configs
    ]

    loader = torch_geometric.dataloader.DataLoader(
        dataset=atomic_data_list,
        batch_size=len(atomic_data_list),
        shuffle=False,
        drop_last=False,
    )
    mace_batch = next(iter(loader)).to(device)
    return mace_batch.to_dict()


# ---------------------------------------------------------------------------
# Top-level entry points (mirror UMA signatures)
# ---------------------------------------------------------------------------
def compute_crystal_mace_on_mxt_batch(batch,
                                      std_orientation: bool = True,
                                      predictor: Optional = None,
                                      pbc: bool = True,
                                      max_cp: float = 2.0,
                                      force_rebuild: bool = False):
    """MACE crystal energy prediction on an mxt batch. Energy only."""
    # Guard against ultra-dense cells (mirrors UMA entry point)
    while sum(batch.packing_coeff > max_cp) > 0:
        bad_inds = torch.argwhere(batch.packing_coeff > max_cp)
        if len(bad_inds) > 0:
            batch.cell_lengths[bad_inds] += 2
            batch.box_analysis()

    mace_batch = batch_to_mace_atomicdata(
        batch,
        predictor=predictor,
        std_orientation=std_orientation,
        pbc=pbc,
        force_rebuild=force_rebuild,
    )

    out, crashed = safe_predict_mace(predictor, mace_batch)
    if crashed:
        energy = torch.zeros(batch.num_graphs, dtype=torch.float32, device=batch.device)
    else:
        energy = out["energy"]
        if energy.ndim == 2:
            energy = energy.flatten()

    return energy


def compute_molecule_mace_on_mxt_batch(batch,
                                       predictor: Optional = None):
    """MACE molecule energy prediction on an mxt batch (non-periodic). Energy only."""
    mace_batch = molecule_batch_to_mace_atomicdata(batch, predictor=predictor, pbc=False)

    out, crashed = safe_predict_mace(predictor, mace_batch)
    if crashed:
        energy = torch.zeros(batch.num_graphs, dtype=torch.float32, device=batch.device)
    else:
        energy = out["energy"]
        if energy.ndim == 2:
            energy = energy.flatten()

    return energy


# ---------------------------------------------------------------------------
# Predictor initializers
# ---------------------------------------------------------------------------
def _init_mace_predictor(model_path: str,
                         device: str,
                         default_dtype: str = 'float64',
                         enable_cueq: bool = False,
                         head: Optional[str] = None):
    """Shared predictor init: loads model, extracts z_table/r_max/heads."""
    _patch_torch_load()

    # Deferred imports
    from mace.tools import torch_tools, utils

    torch_tools.set_default_dtype(default_dtype)
    torch_device = torch_tools.init_device(device)

    model = torch.load(f=model_path, map_location=torch_device)

    if enable_cueq:
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
        model = run_e3nn_to_cueq(model)

    model.to(torch_device)
    model.eval()

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    predictor = {
        "model": model,
        "z_table": z_table,
        "r_max": r_max,
        "heads": heads,
        "device": torch_device,
        "head": head,
    }
    return predictor


def init_mace_crystal_predictor(model_path: str,
                                device: str = 'cuda',
                                default_dtype: str = 'float64',
                                enable_cueq: bool = False,
                                head: Optional[str] = None):
    """Initialize a MACE predictor for periodic crystal energy evaluation."""
    return _init_mace_predictor(
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
        head=head,
    )


def init_mace_mol_predictor(model_path: str,
                            device: str = 'cuda',
                            default_dtype: str = 'float64',
                            enable_cueq: bool = False,
                            head: Optional[str] = None):
    """Initialize a MACE predictor for non-periodic molecule energy evaluation."""
    return _init_mace_predictor(
        model_path=model_path,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
        head=head,
    )