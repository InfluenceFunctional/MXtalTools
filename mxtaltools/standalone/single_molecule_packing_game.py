import os
import warnings

import numpy as np
import pandas as pd
import torch

from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.models.task_models.generator_models import CSDPrior
from mxtaltools.models.utils import denormalize_generated_cell_params, get_intermolecular_dists_dict

warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore w&b error
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from mxtaltools.common.config_processing import process_main_config
from mxtaltools.common.geometry_calculations import compute_fractional_transform_torch
from mxtaltools.modeller import Modeller
from mxtaltools.dataset_management.CrystalData import CrystalData
from torch_geometric.loader.dataloader import Collater
from mxtaltools.constants.space_group_info import SYM_OPS, POINT_GROUPS, LATTICE_TYPE, SPACE_GROUPS
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
import ase
from tqdm import tqdm
from mxtaltools.models.functions.vdw_overlap import vdw_overlap, vdw_analysis

batch_size = 1000  # how many samples per batch
num_iters = 50  # how many batches to try before giving up on this space group
vdw_threshold = 0.35  # maximum allowed average normalized vdw overlap per-molecule
#SGS_TO_SEARCH = [i for i in range(1, 230 + 1)]
SGS_TO_SEARCH = [123, 183, 191, 200, 215, 221, 225, 226, 227, 228, 229, 230]

os.chdir(r'C:\Users\mikem\crystals\toys')  # where you want everything to save
config_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/test_configs/crystal_building.yaml'
user_path = r'C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/Users/mikem.yaml'

"""get molecule"""
atom_numbers = [6, 9, 17, 35, 53]
coords = np.stack([[-1.27665, 0.04371, -1.09742],
                   [0.08215, 0.03560, - 1.10428],
                   [-1.87222, -1.60535, -1.32478],
                   [-1.92029, 1.17139, -2.54270],
                   [-1.95420, 0.78568, 0.72402]])

coords -= coords.mean(0)  # subtract centroid

supercell_size = 5
rotation_basis = 'cartesian'
sym_ops = SYM_OPS
point_groups = POINT_GROUPS
lattice_type = LATTICE_TYPE
space_groups = SPACE_GROUPS
sym_info = {  # collect space group info into single dict
    'sym_ops': sym_ops,
    'point_groups': point_groups,
    'lattice_type': lattice_type,
    'space_groups': space_groups}
vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device='cpu')

prior_generator = CSDPrior(
                sym_info=sym_info, device='cpu',
                cell_means=None,
                cell_stds=None,
                lengths_cov_mat=None, )

supercell_builder = SupercellBuilder(device='cpu', rotation_basis=rotation_basis)

reasonable_cell_params = [[] for _ in SGS_TO_SEARCH]  # initialize record
for sg_search_index, sg_ind in enumerate(SGS_TO_SEARCH):  # loop over space groups
    if not os.path.exists(f'good_params_for_{sg_ind}.npy'):
        """convert to molecular crystal data object with relevant symmetry info"""
        mol_data = CrystalData(
            x=torch.tensor(atom_numbers, dtype=torch.long)[:, None],
            pos=torch.Tensor(coords),
            sg_ind=sg_ind,
            z_prime=1,
            mol_size=5)

        "generate batch"
        collater = Collater(None, None)
        mol_batch = collater([mol_data for _ in range(batch_size)])

        """over a certain number of batches / attempts"""
        for ii in tqdm(range(num_iters)):
            vdw_threshold_i = vdw_threshold + (0.15 * ii / num_iters)  # slowly increase over num_iters

            raw_samples = prior_generator(mol_data.num_graphs, mol_data.sg_ind)
            cell_parameters = denormalize_generated_cell_params(
                raw_samples,
                mol_data,
                supercell_builder.asym_unit_dict
            )

            cell_lengths, cell_angles, mol_position, mol_rotation_i = (
                cell_parameters[:, :3], cell_parameters[:, 3:6], cell_parameters[:, 6:9], cell_parameters[:, 9:])
            _, _, generated_cell_volumes = compute_fractional_transform_torch(cell_lengths, cell_angles)

            # mol volume of FC(Cl)(Br)I is approx 98.856 A^3
            # at 0.7 packing coefficient, the asymmetric unit volume should be ~141 cubic angstrom

            # unit cell volume divided by number of asymmetric units
            asym_unit_volume = generated_cell_volumes / mol_data.sym_mult
            generated_packing_coeff = 98.856 / asym_unit_volume
            target_packings = (torch.randn(batch_size) * 0.04257 + 0.6732).clip(min=0.5, max=0.9)  # real dataset statistics for packing coefficients
            scaling_factor = (generated_packing_coeff / target_packings) ** (1 / 3)
            cell_parameters[:, 0:3] *= scaling_factor[:, None]

            '''build supercells'''
            supercell_data, cell_volumes = supercell_builder.build_zp1_supercells(
                molecule_data=mol_batch,
                cell_parameters=cell_parameters,
                supercell_size=5,
                graph_convolution_cutoff=6,
                pare_to_convolution_cluster=False,
                skip_refeaturization=True
            )

            dist_dict = get_intermolecular_dists_dict(supercell_data, 6, 100)

            '''get vdw clashes'''
            _, _, _, vdw_loss, _ = vdw_analysis(vdw_radii_tensor, dist_dict, mol_data.num_graphs, 0)

            '''determine which samples are 'reasonable'''
            good_samples = torch.argwhere(vdw_loss <= vdw_threshold_i)
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
                mol_data.sym_mult = len(sym_ops)
                collater = Collater(None, None)
                mol_batch = collater([mol_data for _ in range(len(rebuild_samples))])

                unit_cells, cell_volumes = modeller.supercell_builder.build_zp1_supercells(
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
