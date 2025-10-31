from typing import Optional
from ase import Atoms
from ase.cell import Cell
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
import torch

def compute_crystal_uma_on_mxt_batch(batch,
                                     std_orientation: bool = True,
                                     predictor: Optional = None):
    data_list = []
    batch.pose_aunit(std_orientation=std_orientation)
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
        pbc_flag = True
        atoms.set_pbc(pbc_flag)
        crystal = AtomicData.from_ase(atoms, task_name='omc')
        data_list.append(crystal)

    uma_batch = atomicdata_list_to_batch(data_list)

    out = predictor.predict(uma_batch)  # output in total eV per sample

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

def init_uma_crystal_predictor(model_path):
    predictor = pretrained_mlip.load_predict_unit(
        model_path,
        inference_settings='default',
        overrides={"backbone": {#"always_use_pbc": True,
                                "direct_forces": True}},
        device="cuda",
    )
    predictor.inference_mode.compile = False
    predictor.inference_mode.tf32 = True

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

def scale_uma_to_silu_range(uma_energy):
    """
    Scales uma energy in kJ/mol to units of our SiLU potential
    uma_en.append((cry_en/batch.sym_mult)/96.485)  # eV/atom to kJ/mol lattice energy
    uma_fixed = (uma - uma.mean())/uma.std() * lj.std() + lj.mean()

    conversion values extracted from 10k CSD crystals evaluated under both metrics,
    with excellent resulting range overlap
    :param uma_energy:
    :return:
    """
    uma_mean = -2.2629
    uma_std = 1.0233
    silu_mean = -5.0319
    silu_std = 1.8549

    return (uma_energy - uma_mean)/uma_std* silu_std + silu_mean


def scale_uma_to_lj_range(uma_energy):
    """
    Scales uma energy in kJ/mol to units of our LJ potential
    uma_en.append((cry_en/batch.sym_mult)/96.485)  # eV/atom to kJ/mol lattice energy
    uma_fixed = (uma - uma.mean())/uma.std() * lj.std() + lj.mean()

    note LJ energy has a longer range, comes with a 10 angstrom cutoff, as compared to SiLU at 6
    conversion values extracted from 10k CSD crystals evaluated under both metrics,
    with excellent resulting range overlap
    :param uma_energy:
    :return:
    """
    uma_mean = -2.2629
    uma_std = 1.0233
    lj_mean = -16.4616
    lj_std = 5.0916

    return (uma_energy - uma_mean)/uma_std* lj_std + lj_mean
