from typing import Optional

import numpy as np
from ase import Atoms
from ase.cell import Cell
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
import torch

from mxtaltools.common.utils import is_cuda_oom
import torch.serialization

torch.serialization.add_safe_globals([slice])  # necessary for UMA loading on torch 2.6


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
    if not hasattr(batch, 'unit_cell_pos') or force_rebuild:
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

    elif hasattr(batch, 'aux_ind'):
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
    # data_list = batch_to_ase_ucell_list(batch, std_orientation, pbc, force_rebuild)
    # uma_batch = atomicdata_list_to_batch(
    #     [AtomicData.from_ase(atoms, task_name='omc') for atoms in data_list])
    data_list = batch_to_fairchem_atomicdata(batch, std_orientation, pbc, force_rebuild)
    uma_batch = atomicdata_list_to_batch(data_list)

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


def batch_to_fairchem_atomicdata(batch, std_orientation, pbc=True, force_rebuild=False, task_name='omc'):
    device = batch.device
    data_list = []
    if not hasattr(batch, 'unit_cell_pos') or force_rebuild:
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
        z = batch.z[batch.aux_ind == 0]
        batch_ind = batch.batch[batch.aux_ind == 0]

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
            natoms=torch.tensor([len(pos)], dtype=torch.long),
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


def init_uma_crystal_predictor(model_path, device):
    predictor = pretrained_mlip.load_predict_unit(
        model_path,
        inference_settings='default',
        overrides={"backbone": {"always_use_pbc": True,
                                "direct_forces": False,
                                "regress_forces": False,
                                "regress_stress": False}
            , },
        device=device,
    )
    predictor.inference_mode.compile = False
    predictor.inference_mode.tf32 = True
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
