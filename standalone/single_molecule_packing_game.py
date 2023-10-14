import numpy as np
import torch
import os
from common.config_processing import get_config
from common.geometry_calculations import compute_fractional_transform_torch
from crystal_modeller import Modeller
from dataset_management.CrystalData import CrystalData
from torch_geometric.loader.dataloader import Collater
from constants.space_group_info import SYM_OPS
from common.ase_interface import crystals_to_ase_mols, ase_mol_from_crystaldata
from ase.visualize import view
import ase
from tqdm import tqdm
from models.vdw_overlap import vdw_overlap

batch_size = 1000
num_iters = 1000
SGS_TO_SEARCH = [i for i in range(1,21)]
os.chdir(r'C:\Users\mikem\crystals\toys')

"""get molecule"""
atom_numbers = [6, 9, 17, 35, 53]
coords = np.stack([[-1.27665, 0.04371, -1.09742],
                   [0.08215, 0.03560, - 1.10428],
                   [-1.87222, -1.60535, -1.32478],
                   [-1.92029, 1.17139, -2.54270],
                   [-1.95420, 0.78568, 0.72402]])

mol_data = CrystalData(
    x=torch.tensor(atom_numbers, dtype=torch.long)[:, None],
    pos=torch.Tensor(coords),
    sg_ind=torch.ones(1),
    symmetry_operators=torch.ones(1),
    mol_size=5
)

'''init cell generator'''
config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/crystal_building.yaml'
user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'
config = get_config(user_yaml_path=user_path, main_yaml_path=config_path)

supercell_size = 5
rotation_basis = 'spherical'

modeller = Modeller(config)
_, _, _ = modeller.load_dataset_and_dataloaders(override_test_fraction=1)  # need to initialize statistics
modeller.misc_pre_training_items()  # initialize generator

"""propose cell parameters"""
reasonable_cell_params = [[] for _ in SGS_TO_SEARCH]
for sg_search_index, sg_ind in enumerate(SGS_TO_SEARCH):  # search space groups
    sym_ops = SYM_OPS[sg_ind] * 1
    mol_data.sg_ind = sg_ind
    mol_data.symmetry_operators = sym_ops
    mol_data.mult = len(sym_ops)
    collater = Collater(None, None)
    mol_batch = collater([mol_data for _ in range(batch_size)])
    for ii in tqdm(range(num_iters)):

        '''rescale cell density'''
        cell_parameters = modeller.gaussian_generator(batch_size, sg_ind=torch.ones(batch_size))
        cell_lengths, cell_angles, mol_position, mol_rotation_i = (
            cell_parameters[:, :3], cell_parameters[:, 3:6], cell_parameters[:, 6:9], cell_parameters[:, 9:])
        T_fc_list, T_cf_list, generated_cell_volumes = compute_fractional_transform_torch(cell_lengths, cell_angles)

        # mol volume of FC(Cl)(Br)I is approx 98.856
        # at 0.7 packing coefficient, the asymmetric unit volume should be ~141 cubic angstrom

        asym_unit_volume = generated_cell_volumes / len(sym_ops)
        generated_packing_coeff = 98.856 / asym_unit_volume
        target_packings = (torch.randn(batch_size) * 0.04257 + 0.6732).clip(min=0.5)  # real dataset statistics for packing coefficients
        scaling_factor = (generated_packing_coeff / target_packings) ** (1 / 3)
        cell_parameters[:, 0:3] *= scaling_factor[:, None]

        '''build samples'''
        supercells, cell_volumes = modeller.supercell_builder.build_supercells(
            molecule_data=mol_batch,
            cell_parameters=cell_parameters,
            supercell_size=1,
            graph_convolution_cutoff=6,
            pare_to_convolution_cluster=False,
            skip_refeaturization=True
        )

        vdw_scores = vdw_overlap(modeller.vdw_radii, crystaldata=supercells, return_score_only=True)

        good_samples = torch.argwhere(vdw_scores >= -0.1)
        if len(good_samples) > 0:
            reasonable_cell_params[sg_search_index].extend([cell_parameters[sample][0].tolist() + [sg_ind] for sample in good_samples])
            print(len(reasonable_cell_params[sg_search_index]))

        if len(reasonable_cell_params[sg_search_index]) > 10 or (ii == num_iters - 1):
            rebuild_samples_i = torch.Tensor(reasonable_cell_params[sg_search_index])

            rebuild_samples = rebuild_samples_i[:, :12]
            rebuild_sg = int(rebuild_samples_i[0,-1])

            sym_ops = SYM_OPS[rebuild_sg] * 1
            mol_data.sg_ind = rebuild_sg
            mol_data.symmetry_operators = sym_ops
            mol_data.mult = len(sym_ops)
            collater = Collater(None, None)
            mol_batch = collater([mol_data for _ in range(len(rebuild_samples))])

            supercells, cell_volumes = modeller.supercell_builder.build_supercells(
                molecule_data=mol_batch,
                cell_parameters=rebuild_samples,
                supercell_size=1,
                graph_convolution_cutoff=6,
                pare_to_convolution_cluster=False,
                skip_refeaturization=True
            )

            structures = [ase_mol_from_crystaldata(supercells, highlight_canonical_conformer=False,
                                                 index=i, exclusion_level='none', inclusion_distance=4, return_crystal=True)
                        for i in range(supercells.num_graphs)]

            for i in range(len(structures)):
                #ase.io.write(f'space_group_{sg_ind}_crystal_{i}.cif', structures[i][1])  # these are distorted
                ase.io.write(f'space_group_{sg_ind}_cluster_{i}.cif', structures[i][0])


            break
    np.save(f'good_params_for_{sg_ind}', good_samples.cpu().detach().numpy())
aa = 0
