import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader.dataloader import Collater

from mxtaltools.dataset_management.data_manager import DataManager
from mxtaltools.dataset_management.dataset_generation.generate_dataset_from_smiles import process_smiles_to_crystal_opt
from mxtaltools.dataset_management.dataset_generation.generate_dataset_from_smiles import test_crystal_rebuild_from_embedding
from mxtaltools.dataset_management.otf_conf_gen import get_smiles_list

if __name__ == '__main__':
    test=False
    smiles_path = r'D:\crystal_datasets\zinc22'
    chunks_path = Path(r'D:\crystal_datasets')
    new_dataset_name ='otf_dataset'
    os.chdir(smiles_path)
    chunks = get_smiles_list(10000, 8, smiles_path)
    os.chdir(chunks_path)

    min_ind = 0
    pool = mp.Pool(8)
    # ind = 0
    # chunk_ind = min_ind + ind
    # chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
    # process_smiles_to_crystal_opt(chunks[ind], chunk_path, np.array([1, 6, 7, 8, 9]), 1, True,** {
    #     'max_num_atoms': 30,
    #     'max_num_heavy_atoms': 9,
    #     'pare_to_size': None,
    #     'max_radius': 15,
    #     'protonate': True,
    #     'rotamers_per_sample': 1,
    #     'allow_simple_hydrogen_rotations': False
    # })

    outputs = []

    for ind, chunk in enumerate(chunks):
        print(f'starting chunk {ind} with {len(chunk)} smiles')
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
        outputs.append(pool.apply_async(process_smiles_to_crystal_opt,
                                        args=(chunk, chunk_path, np.array([1, 6, 7, 8, 9]), 1, False),
                                        kwds={
                                            'max_num_atoms': 30,
                                            'max_num_heavy_atoms': 9,
                                            'pare_to_size': None,
                                            'max_radius': 15,
                                            'protonate': True,
                                            'rotamers_per_sample': 1,
                                            'allow_simple_hydrogen_rotations': False
                                        }))

    pool.close()
    pool.join()

    '''process and save dataset'''
    miner = DataManager(device='cpu',
                        config=Namespace(**{'seed': 0,
                                            'max_dataset_length': 1000000,
                                            'filter_protons': False}),
                        datasets_path=chunks_path,
                        chunks_path=chunks_path,
                        dataset_type='crystal',
                        do_crystal_indexing=False)
    miner.process_new_dataset(new_dataset_name=new_dataset_name,
                              chunks_patterns=['chunk'])

    if test:
        '''check sample quality'''
        collater = Collater(0, 0)
        mol_batch = collater([elem for elem in miner.dataset[:1000]]).clone().cpu()
        opt_vdw_pot = mol_batch.vdw_pot[None, ...]
        opt_vdw_loss = mol_batch.vdw_loss[None, ...]
        opt_aunits = mol_batch.pos[None, ...]
        cell_params = torch.cat([
            mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
        ], dim=1)
        opt_cell_params = cell_params[None, ...]

        r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
            mol_batch,
            opt_vdw_pot,
            opt_vdw_loss,
            opt_aunits,
            opt_cell_params,
            denorm=False,
            destd=False,
            renorm=False,
            restd=False,
            make_figs=False,
        )

    del miner.dataset

    '''reload for testing'''
    conv_cutoff = 6
    # nonzero_positional_noise = sum(list(self.config.positional_noise.__dict__.values()))
    miner.regression_target = None
    miner.load_dataset_for_modelling(
        new_dataset_name + '.pt',
        filter_conditions=None,
        filter_polymorphs=False,
        filter_duplicate_molecules=False,
        filter_protons=False,
        conv_cutoff=conv_cutoff,
        do_shuffle=True,
        precompute_edges=False,
        single_identifier=None,
    )

    if test:
        '''check sample quality'''
        collater = Collater(0, 0)
        mol_batch = collater([elem for elem in miner.dataset[:1000]]).clone().cpu()
        opt_vdw_pot = mol_batch.vdw_pot[None, ...]
        opt_vdw_loss = mol_batch.vdw_loss[None, ...]
        opt_aunits = mol_batch.pos[None, ...]
        cell_params = torch.cat([
            mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
        ], dim=1)
        opt_cell_params = cell_params[None, ...]

        r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
            mol_batch,
            opt_vdw_pot,
            opt_vdw_loss,
            opt_aunits,
            opt_cell_params,
            denorm=False,
            destd=False,
            renorm=False,
            restd=False,
            make_figs=False,
        )

    aa = 1
