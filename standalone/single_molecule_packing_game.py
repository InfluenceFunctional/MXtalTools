import warnings
import numpy as np
import torch
import os
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore w&b error
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from common.config_processing import get_config
from common.geometry_calculations import compute_fractional_transform_torch
from crystal_modeller import Modeller
from dataset_management.CrystalData import CrystalData
from torch_geometric.loader.dataloader import Collater
from constants.space_group_info import SYM_OPS
from common.ase_interface import ase_mol_from_crystaldata
from ase.visualize import view
import ase
from tqdm import tqdm
from models.vdw_overlap import vdw_overlap

batch_size = 1000  # how many samples per batch
num_iters = 250  # how many batches to try before giving up on this space group
vdw_threshold = 0.25  # maximum allowed average normalized vdw overlap per-molecule
SGS_TO_SEARCH = [i for i in range(1, 230 + 1)]

os.chdir(r'C:\Users\mikem\crystals\toys')  # where you want everything to save
config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/crystal_building.yaml'
user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/users/mkilgour.yaml'

"""get molecule"""
atom_numbers = [6, 9, 17, 35, 53]
coords = np.stack([[-1.27665, 0.04371, -1.09742],
                   [0.08215, 0.03560, - 1.10428],
                   [-1.87222, -1.60535, -1.32478],
                   [-1.92029, 1.17139, -2.54270],
                   [-1.95420, 0.78568, 0.72402]])

coords -= coords.mean(0)  # subtract centroid

# from ccdc.molecule import Molecule, Atom
# mol = Molecule(identifier='my molecule')
# a1 = Atom('C', coordinates=(-1.27665, 0.04371, -1.09742))
# a2 = Atom('F', coordinates=(0.08215, 0.03560, - 1.10428))
# a3 = Atom('Cl', coordinates=(-1.87222, -1.60535, -1.32478))
# a4 = Atom('Br', coordinates=(-1.92029, 1.17139, -2.54270))
# a5 = Atom('I', coordinates=(-1.95420, 0.78568, 0.72402))
# a1_id = mol.add_atom(a1)
# a2_id = mol.add_atom(a2)
# a3_id = mol.add_atom(a3)
# a4_id = mol.add_atom(a4)

"""convert to molecular crystal data object"""
mol_data = CrystalData(
    x=torch.tensor(atom_numbers, dtype=torch.long)[:, None],
    pos=torch.Tensor(coords),
    sg_ind=torch.ones(1),
    symmetry_operators=torch.ones(1),
    mol_size=5)

'''init cell generator - uses gaussian noise and dataset statistics to propose crystal parameters'''
config = get_config(user_yaml_path=user_path, main_yaml_path=config_path)

supercell_size = 5
rotation_basis = 'spherical'

modeller = Modeller(config)
_, _, _ = modeller.load_dataset_and_dataloaders(override_test_fraction=1)  # need this to initialize some statistics
modeller.misc_pre_training_items()  # initialize generator

reasonable_cell_params = [[] for _ in SGS_TO_SEARCH]  # initialize record
for sg_search_index, sg_ind in enumerate(SGS_TO_SEARCH):  # loop over space groups
    if not os.path.exists(f'good_params_for_{sg_ind}.npy'):
        """retrieve and assign symmetry operations"""
        sym_ops = SYM_OPS[sg_ind] * 1
        mol_data.sg_ind = sg_ind
        mol_data.symmetry_operators = sym_ops
        mol_data.mult = len(sym_ops)

        "generate batch"
        collater = Collater(None, None)
        mol_batch = collater([mol_data for _ in range(batch_size)])

        """over a certain number of batches / attempts"""
        for ii in tqdm(range(num_iters)):
            vdw_threshold_i = vdw_threshold + (0.15 * ii / num_iters)  # slowly increase over num_iters
            '''rescale cell density'''
            cell_parameters = modeller.gaussian_generator(batch_size, sg_ind=torch.ones(batch_size) * sg_ind)
            cell_lengths, cell_angles, mol_position, mol_rotation_i = (
                cell_parameters[:, :3], cell_parameters[:, 3:6], cell_parameters[:, 6:9], cell_parameters[:, 9:])
            _, _, generated_cell_volumes = compute_fractional_transform_torch(cell_lengths, cell_angles)

            # mol volume of FC(Cl)(Br)I is approx 98.856 A^3
            # at 0.7 packing coefficient, the asymmetric unit volume should be ~141 cubic angstrom

            asym_unit_volume = generated_cell_volumes / len(sym_ops)  # unit cell volume divided by number of asymmetric units
            generated_packing_coeff = 98.856 / asym_unit_volume
            target_packings = (torch.randn(batch_size) * 0.04257 + 0.6732).clip(min=0.5)  # real dataset statistics for packing coefficients
            scaling_factor = (generated_packing_coeff / target_packings) ** (1 / 3)
            cell_parameters[:, 0:3] *= scaling_factor[:, None]

            '''build supercells'''
            supercells, cell_volumes = modeller.supercell_builder.build_supercells(
                molecule_data=mol_batch,
                cell_parameters=cell_parameters,
                supercell_size=1,
                graph_convolution_cutoff=6,
                pare_to_convolution_cluster=False,
                skip_refeaturization=True
            )

            '''get vdw clashes'''
            vdw_scores = vdw_overlap(modeller.vdw_radii, crystaldata=supercells, return_score_only=True)

            '''determine which samples are 'reasonable'''
            good_samples = torch.argwhere(vdw_scores >= -vdw_threshold_i)
            if len(good_samples) > 0:
                reasonable_cell_params[sg_search_index].extend([cell_parameters[sample][0].tolist() + [sg_ind] for sample in good_samples])
                print(len(reasonable_cell_params[sg_search_index]))

            """
            if we have 10 reasonable samples for this space group OR we have run out of attempts
            save what we have and move on
            """
            if len(reasonable_cell_params[sg_search_index]) >= 10 or (ii == num_iters - 1) and (len(reasonable_cell_params[sg_search_index]) > 0):
                '''rebuild good samples'''
                rebuild_samples_i = torch.Tensor(reasonable_cell_params[sg_search_index])

                rebuild_samples = rebuild_samples_i[:, :12]
                rebuild_sg = int(rebuild_samples_i[0, -1])

                sym_ops = SYM_OPS[rebuild_sg] * 1
                mol_data.sg_ind = rebuild_sg
                mol_data.symmetry_operators = sym_ops
                mol_data.mult = len(sym_ops)
                collater = Collater(None, None)
                mol_batch = collater([mol_data for _ in range(len(rebuild_samples))])

                unit_cells, cell_volumes = modeller.supercell_builder.build_supercells(
                    molecule_data=mol_batch,
                    cell_parameters=rebuild_samples,
                    supercell_size=0,
                    graph_convolution_cutoff=6,
                    pare_to_convolution_cluster=False,
                    skip_refeaturization=True
                )

                cells = unit_cells.pos.reshape(len(unit_cells.pos) // 5, 5, 3)
                dists = torch.stack([torch.cdist(cells[i], cells[i]) for i in range(min(10, len(cells)))])
                dmat = torch.zeros((len(dists), len(dists)))
                for i in range(len(dists)):
                    for j in range(len(dists)):
                        dmat[i, j] = torch.mean(torch.abs(dists[i] - dists[j]))

                if dmat.max() > 1e-3:
                    print(f"Warping {dmat.max():.3f} in space group {sg_ind}")

                # supercells, cell_volumes = modeller.supercell_builder.build_supercells(
                #     molecule_data=mol_batch,
                #     cell_parameters=rebuild_samples,
                #     supercell_size=1,
                #     graph_convolution_cutoff=6,
                #     pare_to_convolution_cluster=False,
                #     skip_refeaturization=True
                # )

                '''save supercells as cifs'''
                print(f"Saving SG={sg_ind} structures")
                cell_structures = [ase_mol_from_crystaldata(unit_cells, highlight_canonical_conformer=False,
                                                            index=i, exclusion_level='none', inclusion_distance=4, return_crystal=False)
                                   for i in range(unit_cells.num_graphs)]
                # supercell_structures = [ase_mol_from_crystaldata(supercells, highlight_canonical_conformer=False,
                #                                                  index=i, exclusion_level='none', inclusion_distance=4, return_crystal=False)
                #                         for i in range(supercells.num_graphs)]

                for i in range(min(10, len(cell_structures))):
                    # ase.io.write(f'space_group_{sg_ind}_crystal_{i}.cif', structures[i][1])  # these are distorted
                    ase.io.write(f'space_group_{sg_ind}_unit_cell_{i}.cif', cell_structures[i])
                    #ase.io.write(f'space_group_{sg_ind}_supercell_{i}.cif', supercell_structures[i])

                break

        '''save also the raw cell params in case we want them in the future'''
        np.save(f'good_params_for_{sg_ind}', good_samples.cpu().detach().numpy())
