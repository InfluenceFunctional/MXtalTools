from typing import Optional
from ase import Atoms
from ase.cell import Cell
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
import torch

def compute_crystal_uma_on_mxt_batch(batch,
                                     std_orientation: bool = True,
                                     predictor: Optional = None,
                                     pbc: bool = True):
    data_list = []
    "UMA sometimes fails on ultra-dense cells, so we'll manually prevent that. These are obviously terrible cells anyway."
    bad_inds = torch.argwhere(batch.packing_coeff > 10)
    if len(bad_inds) > 0:
        batch.cell_lengths[bad_inds] *= 2
        batch.box_analysis()
    batch.pose_aunit(std_orientation=std_orientation)
    assert torch.sum(torch.isnan(batch.pos)) == 0
    batch.build_unit_cell()
    cell_params = torch.cat([batch.cell_lengths, batch.cell_angles * 90 / (torch.pi / 2)], dim=1).cpu().detach().numpy()

    for ind in range(batch.num_graphs):
        pos = batch.unit_cell_pos[batch.unit_cell_batch == ind]
        z = batch.z[batch.batch==ind].repeat(batch.sym_mult[ind])
        atoms = Atoms(
            numbers=z.cpu().detach().numpy(),
            positions=pos.cpu().detach().numpy(),
        )
        atoms.set_cell(Cell.fromcellpar(cell_params[ind]))
        pbc_flag = pbc
        atoms.set_pbc(pbc_flag)
        crystal = AtomicData.from_ase(atoms, task_name='omc')
        data_list.append(crystal)

    uma_batch = atomicdata_list_to_batch(data_list)

    out = predictor.predict(uma_batch)  # output in total eV per sample
    assert torch.sum(torch.isnan(batch.pos)) == 0

    return out['energy']


def compute_molecule_uma_on_mxt_batch(batch,
                                     predictor: Optional = None):
    data_list = []

    for ind in range(batch.num_graphs):
        pos = batch.pos[batch.batch == ind]
        z = batch.z[batch.batch==ind]
        atoms = Atoms(
            numbers=z.cpu().detach().numpy(),
            positions=pos.cpu().detach().numpy(),
        )
        pbc_flag = False
        atoms.set_pbc(pbc_flag)
        crystal = AtomicData.from_ase(atoms, task_name='omol')
        data_list.append(crystal)

    uma_batch = atomicdata_list_to_batch(data_list)

    out = predictor.predict(uma_batch)  # output in total eV per sample

    return out['energy']

def init_uma_crystal_predictor(model_path, device):
    predictor = pretrained_mlip.load_predict_unit(
        model_path,
        inference_settings='default',
        overrides={"backbone": {"always_use_pbc": True,
                                "direct_forces": True}},
        device=device,
    )
    predictor.inference_mode.compile = False
    predictor.inference_mode.tf32 = True
    #predictor.inference_mode.activation_checkpointing=False

    # # some savings if we do energy-only
    # # don't do this if you want forces and stresses
    # # admittedly this is only partially effective - I can't kill the backward call completely for come reason
    # energy_task = predictor.tasks["omc_energy"]
    # predictor.tasks = {"omc_energy": energy_task}
    # predictor.dataset_to_tasks["omc"] = [
    #     t for t in predictor.dataset_to_tasks["omc"] if "energy" in t.name
    # ]
    return predictor


def init_uma_mol_predictor(model_path):
    predictor = pretrained_mlip.load_predict_unit(
        model_path,
        inference_settings='default',
        overrides={"backbone": {#"always_use_pbc": True,
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
