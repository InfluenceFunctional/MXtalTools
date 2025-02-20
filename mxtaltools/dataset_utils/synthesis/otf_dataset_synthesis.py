import gzip
import os
from pathlib import Path
from random import shuffle
from typing import Optional, List, Any, Union

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.data import Batch

from mxtaltools.common.geometry_utils import batch_molecule_vdW_volume
from mxtaltools.common.utils import chunkify
from mxtaltools.common.training_utils import init_sym_info

from mxtaltools.conformer_generation.conformer_generator import generate_random_conformers_from_smiles
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_search.standalone_crystal_opt import standalone_opt_random_crystals
from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.models.task_models.generator_models import CSDPrior
from mxtaltools.analysis.crystals_analysis import get_intermolecular_dists_dict
from mxtaltools.analysis.vdw_analysis import electrostatic_analysis, vdw_analysis
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

"""
Functions for generating molecule and molecular crystal datasets on-the-fly given molecule SMILES
"""

def otf_synthesize_molecules(dataset_length: int,
                             smiles_source: Union[str, Path],
                             workdir: Union[str, Path],
                             allowed_atom_types: list,
                             num_processes: int,
                             mp_pool,
                             max_num_atoms: int,
                             max_num_heavy_atoms: int,
                             pare_to_size: int,
                             max_radius: float,
                             synchronize: bool = True):

    chunks_path = Path(workdir)  # where to save outputs
    smiles_path = Path(smiles_source)  # where to get inputs
    os.chdir(smiles_path)
    chunks = generate_smiles_dataset(dataset_length, num_processes, smiles_path)  # get batch of smiles and chunkify
    os.chdir(chunks_path)

    # generate samples
    min_ind = 0  # always overwrite any existing chunks
    outputs = []
    for ind, chunk in enumerate(chunks):
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
        mp_pool.apply_async(process_smiles_list,
                            args=(chunk,
                                  allowed_atom_types,
                                  chunk_path,
                                  False
                                  ),
                            kwds={
                             'max_num_atoms': max_num_atoms,
                             'max_num_heavy_atoms': max_num_heavy_atoms,
                             'pare_to_size': pare_to_size,
                             'max_radius': max_radius,
                             'protonate': True,
                             'rotamers_per_sample': 1,
                             'allow_simple_hydrogen_rotations': False,
                                'do_partial_charges': False,
                         })
    mp_pool.close()
    if synchronize:  # force function to complete before returning
        mp_pool.join()
        [out.get() for out in outputs]  # check for output messages

    return mp_pool


def otf_synthesize_crystals(dataset_length: int,
                            smiles_source: Union[str, Path],
                            workdir: Union[str, Path],
                            allowed_atom_types: list,
                            num_processes: int,
                            mp_pool,
                            max_num_atoms: int,
                            max_num_heavy_atoms: int,
                            pare_to_size: int,
                            max_radius: float,
                            synchronize: bool = True):

    chunks_path = Path(workdir)  # where to save outputs
    smiles_path = Path(smiles_source)  # where to get inputs
    os.chdir(smiles_path)
    chunks = generate_smiles_dataset(dataset_length, num_processes, smiles_path)  # get batch of smiles and chunkify
    os.chdir(chunks_path)

    ## optional synchronous evaluation for debugging
    # chunk_ind = 0
    # chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
    # print('running test chunk')
    # print("Checking if cuda available...")
    # import torch
    # print(torch.cuda.is_available())
    # process_smiles_to_crystal_opt(
    #     chunks[chunk_ind], chunk_path, allowed_atom_types, 1, False,
    #     **{
    #         'max_num_atoms': max_num_atoms,
    #         'max_num_heavy_atoms': max_num_heavy_atoms,
    #         'pare_to_size': pare_to_size,
    #         'max_radius': max_radius,
    #         'protonate': True,
    #         'rotamers_per_sample': 1,
    #         'allow_simple_hydrogen_rotations': False,
    #         'do_partial_charges': True
    #                      })

    # generate samples
    outputs = []
    min_ind = 0  # always overwrite any existing chunks
    for ind, chunk in enumerate(chunks):
        print(f'starting chunk {ind} with {len(chunk)} smiles')
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
        outputs.append(
        mp_pool.apply_async(process_smiles_to_crystal_opt,
                            args=(chunk, chunk_path, allowed_atom_types, 1, False),
                            kwds={
                             'max_num_atoms': max_num_atoms,
                             'max_num_heavy_atoms': max_num_heavy_atoms,
                             'pare_to_size': pare_to_size,
                             'max_radius': max_radius,
                             'protonate': True,
                             'rotamers_per_sample': 1,
                             'allow_simple_hydrogen_rotations': False,
                                'do_partial_charges': True,
                         }))

    mp_pool.close()
    if synchronize:
        mp_pool.join()
        [out.get() for out in outputs]  # check for output messages

    return mp_pool


def generate_smiles_dataset(dataset_length, num_processes, smiles_dirs_path, seed: int = 1):
    #np.random.seed(seed)
    h_dirs = os.listdir(smiles_dirs_path)
    h_dirs = [elem for elem in h_dirs if elem[0] == 'H']
    smiles_paths = []
    for dir in h_dirs:
        files = os.listdir(dir)
        smiles_paths.extend(
            [os.path.join(Path(dir), Path(file)) for file in files]
        )
    file_sizes = np.zeros(len(smiles_paths))
    for i, path in enumerate(smiles_paths):
        file_sizes[i] = os.path.getsize(path)
    paths_to_keep = []
    for i in range(len(file_sizes)):
        if file_sizes[i] > 0:
            paths_to_keep.append(i)
    smiles_paths = [smiles_paths[ind] for ind in paths_to_keep]
    file_sizes = file_sizes[paths_to_keep]
    # sample proportional to their size
    # select random set of smiles files
    smiles_list = []
    while len(smiles_list) < dataset_length:
        file_to_add = np.random.choice(
            range(len(smiles_paths)),
            size=1,
            replace=False,
            p=file_sizes / np.sum(file_sizes)
        )[0]
        filename = smiles_paths[file_to_add]
        if filename[-3:] == '.gz':
            with gzip.open(filename, 'r') as f:
                for line in f:
                    smiles_list.append(line.rstrip())
        elif filename[-4:] == '.txt':
            with open(filename, 'r') as f:
                for line in f:
                    smiles_list.append(line.rstrip())

    shuffle(smiles_list)
    chunks = chunkify(smiles_list[:dataset_length], num_processes)
    return chunks


class Collater:
    """
    simplified version of the PyG collater function
    """
    def __init__(
            self,
    ):
        self.abc = 1

    def __call__(self, batch: List[Any]) -> Any:
        return Batch.from_data_list(
            batch,
        )


def process_smiles_list(lines: list,
                        allowed_atom_types,
                        file_path: Optional[Union[str, Path]] = None,
                        return_samples: bool = True,
                        **conf_kwargs):
    samples = []
    for line in lines:
        sample, reason = process_smiles(line, allowed_atom_types, to_dict=False, **conf_kwargs)
        if sample is not None:
            samples.append(sample)

    print(f"finished processing smiles list with {len(samples)} samples")
    if file_path is not None:
        torch.save(samples, file_path)

    if return_samples:
        return samples


def process_smiles_to_crystal_opt(lines: list,
                                  file_path,
                                  allowed_atom_types,
                                  space_group,
                                  run_tests=False,
                                  **conf_kwargs):

    """starting chunk pool"""
    mol_samples = process_smiles_list(lines, allowed_atom_types, return_samples=True, **conf_kwargs)
    if len(mol_samples) == 0:
        assert False, "Zero valid molecules in batch, increase crystal generation batch size"

    collater = Collater()
    mol_batch = collater(mol_samples)

    #print('''sample random crystals''')
    crystal_generator = CSDPrior(
        sym_info=init_sym_info(),
        device="cpu",
        cell_means=None,
        cell_stds=None,
        lengths_cov_mat=None)
    normed_cell_params = crystal_generator(mol_batch.num_graphs, space_group * torch.ones(mol_batch.num_graphs))
    mol_batch.sg_ind = space_group * torch.ones(mol_batch.num_graphs)

    #print('''batch compute vdw volume''')
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device='cpu')
    mol_batch.mol_volume = batch_molecule_vdW_volume(mol_batch.x.flatten(),
                                                     mol_batch.pos,
                                                     mol_batch.batch,
                                                     mol_batch.num_graphs,
                                                     vdw_radii_tensor)

    #print('''do local opt''')
    opt_vdw_pot, opt_vdw_loss, opt_packing_coeff, opt_cell_params, opt_aunits = standalone_opt_random_crystals(
        mol_batch.clone().cpu(),
        normed_cell_params.cpu(),
        opt_eps=1e-1,
        post_scramble_each=10,
        device='cpu',
    )

    #print('''extract samples''')
    samples = []
    for graph_ind in range(mol_batch.num_graphs):
        graph_inds = mol_batch.batch == graph_ind
        for sample_ind in range(len(opt_vdw_pot)):
            cell_params = opt_cell_params[sample_ind, graph_ind]
            sample = CrystalData(  # CRYTODO
                x=mol_batch.x[graph_inds],
                p_charges=mol_batch.p_charges[graph_inds],
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

    print(f"finished processing smiles list with {mol_batch.num_graphs} "
          f"molecules and optimizing crystals with {len(samples)} samples")
    if run_tests:
        test_crystal_rebuild_from_embedding(
            mol_batch,
            opt_vdw_pot,
            opt_vdw_loss,
            opt_aunits,
            opt_cell_params,
            denorm=False,
            destd=False,
            renorm=False,
            restd=False,
            make_figs=True,
        )
    torch.save(samples, file_path)


def test_crystal_rebuild_from_embedding(mol_batch,
                                        opt_vdw_pot,
                                        opt_vdw_loss,
                                        opt_aunits,
                                        opt_cell_params,
                                        denorm=False,
                                        destd=False,
                                        renorm=False,
                                        restd=False,
                                        make_figs=True,
                                        ):

    """
    todo - we do functions like this in several places, essentially confirming that our parameterizations are correct
    we should instead have a single workflow for this, maybe even built into the data classes
    """
    def _score_crystal_batch(mol_batch, supercell_batch, vdw_radii_tensor):

        dist_dict = get_intermolecular_dists_dict(supercell_batch,
                                                  6, 100)
        molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, lj_pot \
            = vdw_analysis(vdw_radii_tensor.cpu(),
                           dist_dict,
                           mol_batch.num_graphs, )
        estat_energy = electrostatic_analysis(dist_dict, mol_batch.num_graphs)
        vdw_potential += estat_energy
        vdw_loss += estat_energy
        return vdw_potential, vdw_loss

    from mxtaltools.crystal_building.builder import CrystalBuilder
    from mxtaltools.crystal_building.utils import overwrite_symmetry_info
    from mxtaltools.models.utils import denormalize_generated_cell_params, renormalize_generated_cell_params
    from mxtaltools.constants.atom_properties import VDW_RADII
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device='cpu')

    supercell_builder = CrystalBuilder(device='cpu',
                                       rotation_basis='cartesian')

    lattice_means = torch.tensor([
        1, 1, 1,
        torch.pi / 2, torch.pi / 2, torch.pi / 2,
        0.5, 0.5, 0.5,
        torch.pi / 4, 0, torch.pi / 2
    ], device='cpu', dtype=torch.float32)
    lattice_stds = torch.tensor([
        .35, .35, .35,
        .45, .45, .45,
        0.25, 0.25, 0.25,
        0.33, torch.pi / 2, torch.pi / 2
    ], device='cpu', dtype=torch.float32)
    rebuild_pot = torch.zeros_like(opt_vdw_pot)
    rebuild_loss = torch.zeros_like(rebuild_pot)
    rebuild_aunit = torch.zeros_like(opt_aunits)
    for sample_ind in range(len(opt_cell_params)):
        sample = opt_cell_params[sample_ind]

        if renorm:
            sample = renormalize_generated_cell_params(
                sample,
                mol_batch,
                supercell_builder.asym_unit_dict
            )
        if restd:
            sample = (sample - lattice_means[None, :]) / lattice_stds[None, :]

        if destd:
            sample = sample * lattice_stds[None, :] + lattice_means[None, :]
        if denorm:
            sample = denormalize_generated_cell_params(sample,
                                                       mol_batch,
                                                       supercell_builder.asym_unit_dict
                                                       )

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
        vdw_potential, vdw_loss = _score_crystal_batch(
            mol_batch2, supercell_batch, vdw_radii_tensor
        )
        rebuild_pot[sample_ind] = vdw_potential
        rebuild_aunit[sample_ind] = supercell_batch.pos[supercell_batch.aux_ind == 0].detach()
        rebuild_loss[sample_ind] = vdw_loss
        # if ((vdw_loss - opt_vdw_loss[sample_ind]).abs().log10() > 0.1).any():
        #     aa = 1
        #     bad_inds = torch.argwhere((vdw_potential - opt_vdw_pot[sample_ind]).abs() > 0.1).flatten()
        #
        # if ((rebuild_aunit[sample_ind] - opt_aunits[sample_ind]).abs() > 0.1).any():
        #     aa = 1

    if make_figs:
        fig = make_subplots(rows=1, cols=min(5, mol_batch.num_graphs))
        for ind in range(min(5, mol_batch.num_graphs)):
            fig.add_scatter(y=(rebuild_pot[:, ind] - opt_vdw_pot[:, ind].amin() + 0.1).flatten().log(), row=1,
                            col=ind + 1)
            fig.add_scatter(y=(opt_vdw_pot[:, ind] - opt_vdw_pot[:, ind].amin() + 0.1).flatten().log(), row=1,
                            col=ind + 1)
        fig.update_layout(title='rebuild vs original trajectories')
        fig.show()

        fig = go.Figure()
        fig.add_scatter(y=(((rebuild_pot.flatten() - opt_vdw_pot.flatten()).flatten()).abs() + 1).log10(), )
        fig.add_scatter(y=(rebuild_pot.flatten() - opt_vdw_pot.amin() + 1).log10(), )
        fig.add_scatter(y=(opt_vdw_pot.flatten() - opt_vdw_pot.amin() + 1).log10(), )
        fig.update_layout(title='more rebuild vs original trajectories')
        fig.show()
        fig = go.Figure()
        fig.add_scatter(
            y=(((rebuild_pot.flatten() - opt_vdw_pot.flatten()) / opt_vdw_pot.flatten()).abs() + 1).log10(), )
        fig.update_layout(title='relative errors')
        fig.show()

        fig = go.Figure()
        fig.add_scatter(y=(((rebuild_loss.flatten() - opt_vdw_loss.flatten())).abs()), )
        fig.add_scatter(y=(((rebuild_loss.flatten()))), )
        fig.add_scatter(y=(((opt_vdw_loss.flatten()))), )
        fig.update_layout(title='raw trajectories')
        fig.show()

        deviations = torch.zeros(mol_batch.num_graphs)
        for ind in range(mol_batch.num_graphs):
            deviations[ind] = (
                    rebuild_aunit[0, mol_batch.batch == ind] - opt_aunits[0, mol_batch.batch == ind]).abs().std()
        go.Figure(go.Scatter(y=deviations, name='deviations', showlegend=True)).show()

    mae = (rebuild_loss.flatten() - opt_vdw_loss.flatten()).abs()
    print(mae)
    print(mae.sum())

    return rebuild_pot, rebuild_loss, rebuild_aunit


def process_smiles(smile: str,
                   allowed_atom_types,
                   max_num_atoms: int = 1000,
                   max_num_heavy_atoms: int = 100,
                   to_dict: bool = True,
                   max_radius: float = 15,
                   protonate: bool = True,
                   rotamers_per_sample: int = 1,
                   allow_simple_hydrogen_rotations: bool = False,
                   pare_to_size: Optional[int] = None,
                   min_num_heavy_atoms: int = 5,
                   do_partial_charges: bool = False):
    if rotamers_per_sample > 1:
        assert False, "Multiple rotamers not implemented"
    coords, atom_types, mask_rotate, mask_edges, charges = generate_random_conformers_from_smiles(
        smile,
        protonate=protonate,
        max_rotamers_per_samples=rotamers_per_sample,
        allow_simple_hydrogen_rotations=allow_simple_hydrogen_rotations)

    if coords is False:
        return None, 'no coordinates'

    coords = coords[0]
    atom_types = atom_types[0]
    num_atoms = len(coords[0])

    # use rotatable bonds as fragmentation sites to pare the molecule down to an acceptable size
    atoms_kept = np.arange(len(coords))
    iter = 0
    if pare_to_size is not None:
        while np.sum(atom_types > 1) > pare_to_size and len(mask_rotate) > 0 and iter < 10:
            fragment_size = np.sum(mask_rotate[:, atom_types > 1], axis=1)  # how many heavy atoms in the fragment
            fragment_to_pare = \
                np.random.choice(len(fragment_size), 1, p=np.exp(-fragment_size) / np.sum(np.exp(-fragment_size)))[0]
            atoms_to_pare = mask_rotate[fragment_to_pare, :]
            coords, atom_types = coords[~atoms_to_pare], atom_types[~atoms_to_pare]
            mask_rotate = np.delete(mask_rotate, fragment_to_pare, axis=0)
            mask_rotate = mask_rotate[:, ~atoms_to_pare]
            atoms_kept = atoms_kept[~atoms_to_pare]
            iter += 1

    # molecule sizes filter
    if np.sum(atom_types > 1) > max_num_heavy_atoms:
        return None, "too many heavy atoms"
    elif np.sum(atom_types > 1) < min_num_heavy_atoms:
        return None, "too few heavy atoms"
    elif len(atom_types) > max_num_atoms:
        return None, "too many atoms"

    # atom types filter
    if not set(atom_types).issubset(allowed_atom_types):
        return None, "invalid atom types"

    sample = CrystalData(  # CRYTODO
        x=torch.tensor(atom_types, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        p_charges=charges[atoms_kept],
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
