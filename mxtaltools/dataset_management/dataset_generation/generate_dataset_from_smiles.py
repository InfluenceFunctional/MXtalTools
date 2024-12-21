import gzip
import multiprocessing as mp
import os
from pathlib import Path
from time import time
from typing import Optional

from rdkit import RDLogger
from torch_geometric.loader.dataloader import Collater

from mxtaltools.common.geometry_calculations import batch_molecule_vdW_volume
from mxtaltools.crystal_search.sampling import Sampler
from mxtaltools.models.task_models.generator_models import CSDPrior

RDLogger.DisableLog('rdApp.*')

import numpy as np
import torch
from tqdm import tqdm

from mxtaltools.common.utils import chunkify, init_sym_info
from mxtaltools.conformer_generation.conformer_generator import generate_random_conformers_from_smiles
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.dataset_management.data_manager import DataManager


def process_smiles_list_to_file(lines: list, file_path, allowed_atom_types, **conf_kwargs):
    samples = []
    for line in lines:
        sample, reason = process_smiles(line, allowed_atom_types, to_dict=False, **conf_kwargs)
        if sample is not None:
            samples.append(sample)

    print(f"finished processing smiles list with {len(samples)} samples")
    torch.save(samples, file_path)


def process_smiles_list(lines: list, allowed_atom_types, **conf_kwargs):
    samples = []
    for line in lines:
        sample, reason = process_smiles(line, allowed_atom_types, to_dict=False, **conf_kwargs)
        if sample is not None:
            samples.append(sample)

    #print(f"finished processing smiles list with {len(samples)} samples")
    return samples


def process_smiles_to_crystal_opt(lines: list,
                                  file_path,
                                  allowed_atom_types,
                                  space_group,
                                  **conf_kwargs):
    """"""
    '''build molecules'''
    collater = Collater(0, 0)
    mol_samples = process_smiles_list(lines,allowed_atom_types, **conf_kwargs)
    if len(mol_samples) == 0:
        torch.save([], file_path)
        return None

    mol_batch = collater(mol_samples)

    '''sample random crystals'''
    crystal_generator = CSDPrior(
        sym_info=init_sym_info(),
        device="cpu",
        cell_means=None, cell_stds=None, lengths_cov_mat=None)
    normed_cell_params = crystal_generator(mol_batch.num_graphs, space_group * torch.ones(mol_batch.num_graphs))
    mol_batch.sg_ind = space_group * torch.ones(mol_batch.num_graphs)

    '''optimize crystals and save opt trajectory'''
    sampler = Sampler(0,
                      'cpu',
                      'local',
                      None,
                      None,
                      None,
                      None,
                      show_tqdm=False,
                      skip_rdf=True,
                      gd_score_func='vdW',
                      num_cpus=1,
                      )

    mol_batch.mol_volume = batch_molecule_vdW_volume(mol_batch.x.flatten(),
                                                     mol_batch.pos,
                                                     mol_batch.batch,
                                                     mol_batch.num_graphs,
                                                     sampler.vdw_radii_tensor)

    opt_vdw_pot, opt_vdw_loss, opt_packing_coeff, opt_cell_params, opt_aunits = sampler.local_opt_for_proxy_discrim(
        mol_batch.clone().cpu(),
        normed_cell_params.cpu(),
        opt_eps=1e-1,
    )
    samples = []
    for graph_ind in range(mol_batch.num_graphs):
        graph_inds = mol_batch.batch == graph_ind
        for sample_ind in range(len(opt_vdw_pot)):
            cell_params = opt_cell_params[sample_ind, graph_ind]
            sample = CrystalData(
                x=mol_batch.x[graph_inds],
                pos=opt_aunits[sample_ind, graph_inds],
                smiles=mol_batch.smiles[graph_ind],
                identifier=mol_batch.smiles[graph_ind],
                y=torch.zeros(1, dtype=torch.float32),
                require_crystal_features=True,
                sg_ind=int(mol_batch.sg_ind[graph_ind]),
                z_prime=1,
                cell_lengths=cell_params[:3],
                cell_angles=cell_params[3:6],
                pose_parameters=cell_params[None, 6:],
                vdw_pot=opt_vdw_pot[sample_ind, graph_ind],
                vdw_loss=opt_vdw_loss[sample_ind, graph_ind],
                packing_coeff=opt_packing_coeff[sample_ind, graph_ind]
            )

            samples.append(sample)

    print(
        f"finished processing smiles list with {mol_batch.num_graphs} molecules and optimizing crystals with {len(samples)} samples")
    torch.save(samples, file_path)


""" script to confirm the above sampled structures are the same on being rebuilt


from mxtaltools.crystal_building.builder import CrystalBuilder
from mxtaltools.crystal_building.utils import overwrite_symmetry_info
from mxtaltools.models.utils import denormalize_generated_cell_params

supercell_builder = CrystalBuilder(device='cpu',
                                   rotation_basis='cartesian')
rebuild_pack = torch.zeros_like(opt_packing_coeff)
rebuild_pot = torch.zeros_like(opt_vdw_pot)
rebuild_aunit=torch.zeros_like(opt_aunits)
for sample_ind in range(len(opt_cell_params)):
    sample = opt_cell_params[sample_ind]
    mol_batch2 = mol_batch.clone()
    mol_batch2.pos = opt_aunits[sample_ind]
    mol_batch2 = overwrite_symmetry_info(mol_batch2,
                                       mol_batch2.sg_ind,
                                       supercell_builder.symmetries_dict,
                                       randomize_sgs=False)
    supercell_batch, generated_cell_volumes = (
        supercell_builder.build_zp1_supercells(
            mol_batch=mol_batch2,
            cell_parameters=sample,
            supercell_size=5,
            graph_convolution_cutoff=6,
            align_to_standardized_orientation=False,
            skip_refeaturization=True,
            skip_molecule_posing=True,
        ))
    reduced_volume = generated_cell_volumes / supercell_batch.sym_mult
    packing_coeff = mol_batch2.mol_volume / reduced_volume
    dist_dict, loss, vdw_potential = sampler._score_crystal_batch(
        mol_batch2, 'vdW', supercell_batch,
    )
    rebuild_pot[sample_ind] = vdw_potential
    rebuild_pack[sample_ind] = packing_coeff
    rebuild_aunit[sample_ind] = supercell_batch.pos[supercell_batch.aux_ind == 0].detach()
    if (rebuild_aunit[sample_ind] - opt_aunits[sample_ind]).abs().sum() > 1e-3:
        print('aa!')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=min(5, mol_batch.num_graphs))
for ind in range(min(5, mol_batch.num_graphs)):
    fig.add_scatter(y=(rebuild_pot[:, ind] - opt_vdw_pot[:, ind].amin() + 0.1).flatten().log(), row=1, col=ind + 1)
    fig.add_scatter(y=(opt_vdw_pot[:, ind] - opt_vdw_pot[:, ind].amin() + 0.1).flatten().log(), row=1, col=ind + 1)
fig.show()

fig = make_subplots(rows=1, cols=min(5, mol_batch.num_graphs))
for ind in range(min(5, mol_batch.num_graphs)):
    fig.add_scatter(y=(rebuild_pack[:, ind]).flatten().log(), row=1, col=ind + 1)
    fig.add_scatter(y=(opt_packing_coeff[:, ind]).flatten().log(), row=1, col=ind + 1)
fig.show()

"""


def process_smiles(smile: str,
                   allowed_atom_types,
                   max_num_atoms: int = 1000,
                   max_num_heavy_atoms: int = 100,
                   to_dict: bool = True,
                   max_radius: float = 15,
                   protonate: bool = True,
                   rotamers_per_sample: int = 1,
                   allow_simple_hydrogen_rotations: bool = False,
                   pare_to_size: Optional[int] = None):
    if rotamers_per_sample > 1:
        assert False, "Multiple rotamers not implemented"
    coords, atom_types, mask_rotate, mask_edges = generate_random_conformers_from_smiles(
        smile,
        protonate=protonate,
        max_rotamers_per_samples=rotamers_per_sample,
        allow_simple_hydrogen_rotations=allow_simple_hydrogen_rotations)

    if coords is False:
        return None, 'no coordinates'

    coords = coords[0]
    atom_types = atom_types[0]

    # use rotatable bonds as fragmentation sites to pare the molecule
    if pare_to_size is not None:
        while np.sum(atom_types > 1) > pare_to_size and len(mask_rotate) > 0:
            fragment_size = np.sum(mask_rotate, axis=1)
            min_atoms_to_remove = len(coords) - pare_to_size
            fragment_to_pare = np.argmin(np.abs(min_atoms_to_remove - fragment_size))
            atoms_to_pare = mask_rotate[fragment_to_pare, :]
            coords, atom_types = coords[~atoms_to_pare], atom_types[~atoms_to_pare]
            mask_rotate = np.delete(mask_rotate, fragment_to_pare, axis=0)
            mask_rotate = mask_rotate[:, ~atoms_to_pare]

    # molecule sizes filter
    if np.sum(atom_types > 1) > max_num_heavy_atoms:
        return None, "too many heavy atoms"
    elif len(atom_types) < 6:
        return None, "too few atoms"
    elif len(atom_types) > max_num_atoms:
        return None, "too many atoms"

    # atom types filter
    if not set(atom_types).issubset(allowed_atom_types):  #[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]):
        return None, "invalid atom types"

    sample = CrystalData(
        x=torch.tensor(atom_types, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        smiles=smile,
        identifier=smile,
        y=torch.zeros(1, dtype=torch.float32),
        require_crystal_features=False,
    )

    # molecule radius filter
    if sample.radius > max_radius:
        return None, "molecule too large"

    if to_dict:
        return sample.to_dict(), None
    else:
        return sample, None

#
# if __name__ == '__main__':
#     parent_directory = r'D:\crystal_datasets\zinc22'
#     chunks_dir = os.path.join(Path(parent_directory), 'chunks')
#
#     os.chdir(parent_directory)
#     dirs = os.listdir()
#
#     chunk_ind = - 1
#     min_chunk = 0
#     max_chunk = min(100000, len(dirs))
#
#     pool = mp.Pool(mp.cpu_count() - 1)
#
#     datapoint_counter = 0
#     dataset_length = 100000
#     t0 = time()
#     with tqdm(total=max_chunk) as pbar:
#         while chunk_ind < max_chunk - 1:
#             pbar.update(1)
#             chunk_ind += 1
#
#             if not (max_chunk > chunk_ind >= min_chunk):
#                 continue
#
#             if dirs[chunk_ind][0] == 'H':
#                 dirpath = Path(dirs[chunk_ind])
#                 for file_ind, file in enumerate(tqdm(os.listdir(dirpath))):
#                     chunkpath = os.path.join(chunks_dir, fr'chunk_{chunk_ind}_{file_ind}.pkl')
#                     if not os.path.exists(chunkpath):
#                         filepath = Path(file)
#                         combo_path = os.path.join(dirpath, filepath)
#
#                         if combo_path[-3:] == '.gz':
#                             with gzip.open(combo_path, 'r') as f:
#                                 lines = f.readlines()
#                         elif combo_path[-4:] == '.smi':
#                             with open(combo_path, 'r') as f:
#                                 lines = f.readlines()
#                         else:
#                             pass
#
#                         chunks = chunkify(lines, int(np.ceil(len(lines) / 1000)))
#                         del lines
#
#                         for chunk_ind2, chunk in enumerate(chunks):
#                             chunk_path = chunks_dir + f'/chunk_{chunk_ind}_{file_ind}_{chunk_ind2}.pkl'
#                             # process_smiles_list(chunk, chunk_path)
#                             if not os.path.exists(chunk_path):
#                                 pool.apply_async(process_smiles_list_to_file, args=(chunk, chunk_path, {}))
#                                 datapoint_counter += len(chunk)
#
#                             if datapoint_counter >= dataset_length:
#                                 print('Hit required number of samples')
#                                 break
#
#     pool.close()
#     pool.join()
#
#     print(time() - t0)
#
#     miner = DataManager(device='cpu',
#                         datasets_path=r"D:\crystal_datasets/",
#                         chunks_path=chunks_dir,
#                         dataset_type='molecule')
#     miner.process_new_dataset(new_dataset_name='temp_zinc_conf_dataset')
