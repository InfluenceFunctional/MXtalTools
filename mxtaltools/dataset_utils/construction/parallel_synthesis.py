import gzip
import os
from pathlib import Path
from random import shuffle
from typing import Optional, Union

import numpy as np
import torch
from rdkit import RDLogger

from mxtaltools.common.utils import chunkify
from mxtaltools.crystal_search.standalone_crystal_opt import sample_about_crystal
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import embed_crystal_list

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
                                'protonate': True,
                                'allow_methyl_rotations': True,
                                'compute_partial_charges': False,
                                'max_radius': max_radius,
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
                            space_group: int,
                            do_embedding: bool = False,
                            embedding_type: Optional[str] = None,
                            encoder_checkpoint_path=None,
                            post_scramble_each: Optional[int] = None,
                            synchronize: bool = True,
                            num_chunks: Optional[int] = None,
                            debug: bool = False,
                            do_mace_energy: bool = False, ):
    chunks_path = Path(workdir)  # where to save outputs
    smiles_path = Path(smiles_source)  # where to get inputs
    os.chdir(smiles_path)
    if num_chunks is None:
        num_chunks = num_processes
    chunks = generate_smiles_dataset(dataset_length, num_chunks, smiles_path)  # get batch of smiles and chunkify
    os.chdir(chunks_path)

    conf_kwargs = {
        'max_num_atoms': max_num_atoms,
        'max_num_heavy_atoms': max_num_heavy_atoms,
        'pare_to_size': pare_to_size,
        'max_radius': max_radius,
        'protonate': True,
        'allow_methyl_rotations': True,
        'compute_partial_charges': True,
    }
    # for synchronous debugging
    if debug:
        min_ind = 0
        ind = 0
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')

        process_smiles_to_crystal_opt(chunks[0],
                                      chunk_path,
                                      allowed_atom_types,
                                      space_group,
                                      post_scramble_each,
                                      do_embedding,
                                      embedding_type,
                                      encoder_checkpoint_path,
                                      do_mace_energy,
                                      **conf_kwargs)

    # generate samples
    outputs = []
    min_ind = 0  # always overwrite any existing chunks
    for ind, chunk in enumerate(chunks):
        print(f'starting chunk {ind} with {len(chunk)} smiles')
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
        outputs.append(
            mp_pool.apply_async(process_smiles_to_crystal_opt,
                                args=(chunk,
                                      chunk_path,
                                      allowed_atom_types,
                                      space_group,
                                      post_scramble_each,
                                      do_embedding,
                                      embedding_type,
                                      encoder_checkpoint_path,
                                      do_mace_energy
                                      ),
                                kwds=conf_kwargs))

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


def process_smiles_list(lines: list,
                        allowed_atom_types,
                        file_path: Optional[Union[str, Path]] = None,
                        return_samples: bool = True,
                        max_num_heavy_atoms: int = 100,
                        max_num_atoms: int = 100,
                        max_radius: float = 15,
                        min_num_heavy_atoms: int = 4,
                        **conf_kwargs):
    base_data = MolData()
    samples = []
    for line in lines:
        try:
            samples.append(base_data.from_smiles(line, **conf_kwargs))
        except:  # bare exeption is rough but what can you do, sometimes RDKit just fails in weird ways
            pass

    valid_samples = []
    if len(samples) > 0:
        for sample in samples:
            if sample is not None:
                if torch.sum(sample.z > 1) > max_num_heavy_atoms:
                    continue
                if torch.sum(sample.z > 1) < min_num_heavy_atoms:
                    continue
                if sample.num_nodes > max_num_atoms:
                    continue
                if not set(sample.z.tolist()).issubset(allowed_atom_types):
                    continue
                if sample.radius > max_radius:
                    continue

                valid_samples.append(sample)

        print(f"finished processing smiles list with {len(valid_samples)} samples")
        if file_path is not None:
            torch.save(valid_samples, file_path)

        if return_samples:
            return valid_samples


def process_smiles_to_crystal_opt(lines: list,
                                  file_path,
                                  allowed_atom_types,
                                  space_group,
                                  post_scramble_each: int = None,
                                  do_embedding: bool = False,
                                  embedding_type: Optional[str] = None,
                                  encoder_checkpoint_path: Optional[str] = None,
                                  do_mace_energy: bool = False,
                                  **conf_kwargs):
    try:
        """starting chunk pool"""
        mol_samples = process_smiles_list(lines,
                                          allowed_atom_types,
                                          return_samples=True,
                                          **conf_kwargs)
        if len(mol_samples) == 0:
            assert False, "Zero valid molecules in batch, increase crystal generation batch size"

        mol_batch = collate_data_list(mol_samples)
        aunit_handedness = torch.LongTensor([
            np.random.choice([-1, 1], size=len(mol_samples), replace=True)
        ])[0]
        crystals = []
        for ind, sample in enumerate(mol_samples):
            crystal = MolCrystalData(
                molecule=sample,
                sg_ind=space_group,
                cell_lengths=torch.ones(3),
                cell_angles=torch.ones(3) * torch.pi / 2,
                aunit_centroid=torch.ones(3) * 0.5,
                aunit_orientation=torch.ones(3),
                aunit_handedness=int(aunit_handedness[ind]),
                identifier=sample.smiles,
            )
            crystals.append(crystal)
        crystal_batch = collate_data_list(crystals)
        crystal_batch.sample_reasonable_random_parameters(
            target_packing_coeff=0.5,  # diffuse target
            tolerance=3,
            max_attempts=500
        )

        # print("doing opt")
        optimization_trajectory = crystal_batch.optimize_crystal_parameters(
            optim_target='LJ',
            show_tqdm=False,
            convergence_eps=1e-5,
            do_box_restriction=True,
            cutoff=10,
            compression_factor=0.1,
        )

        # extract optimized samples
        samples = optimization_trajectory[-1]

        # filter unbound states
        samples = [sample for sample in samples if sample.scaled_lj_pot < 3]

        # sample noisily about optimized minima
        nearby_samples = sample_about_crystal(samples,
                                              noise_level=0.05,  # empirically gets us an LJ std about 3
                                              num_samples=post_scramble_each,
                                              cutoff=10)

        for ind in range(post_scramble_each):
            samples.extend(nearby_samples[ind])

        samples = [sample for sample in samples if sample.scaled_lj_pot < 3]  # filter bound states

        # do embedding
        if do_embedding:
            print('embedding crystals')
            samples = embed_crystal_list(
                mol_batch.num_graphs,
                samples,
                embedding_type,
                encoder_checkpoint_path
            )

        if do_mace_energy:
            print('doing mace energy')
            samples = add_mace_energy(samples)
            samples = [sample for sample in samples if hasattr(sample, 'mace_mol_pot')]

        print(f"finished processing smiles list with {mol_batch.num_graphs} "
              f"molecules and optimizing crystals with {len(samples)} samples")

        torch.save(samples, file_path)

    except Exception as e:
        print(str(e))
        print("Crystal synthesis job failed!")
        raise e


def add_mace_energy(samples):
    # calculating crystal mace energies
    from mxtaltools.mace_sp.utils import SPMaceCalculator
    calculator = SPMaceCalculator('cpu')
    for s_ind, sample in enumerate(samples):
        try:
            sample.pose_aunit()
            sample.build_unit_cell()
            sample.mace_mol_pot, sample.mace_lattice_pot = calculator.lattice_energy_calculation(
                sample.unit_cell_pos[0].flatten(0, 1).cpu().detach().numpy(),
                sample.z.repeat(sample.sym_mult.flatten()).cpu().detach().numpy(),
                sample.pos.cpu().detach().numpy(),
                sample.z.cpu().detach().numpy(),
                sample.cell_lengths.flatten().cpu().detach().numpy(),
                sample.cell_angles.flatten().cpu().detach().numpy() * 90 / (np.pi / 2),
                int(sample.sym_mult)
            )
        except AssertionError as e:  # Ase will not allow inside-out cells and throws an assertion to this effect
            sample.mace_mol_pot, sample.mace_lattice_pot = None, None

    torch.set_default_dtype(torch.float32)  # calculator may change the default dtype, causing problems later

    return samples
