import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader.dataloader import Collater

from mxtaltools.dataset_utils.dataset_manager import DataManager
from mxtaltools.dataset_utils.synthesis.otf_conf_gen import get_smiles_list, process_smiles_to_crystal_opt, \
    test_crystal_rebuild_from_embedding

if __name__ == '__main__':
    test = True
    num_smiles = 20
    num_processes = 1
    num_chunks = max(num_processes, num_smiles // 500)
    smiles_path = '/home/mkilgour/crystal_datasets/zinc22'#r'D:\crystal_datasets\zinc22'
    chunks_path = Path('/home/mkilgour/crystal_datasets') #Path(r'D:\crystal_datasets')
    new_dataset_name = 'pd_dataset_w_h_bonds'
    os.chdir(smiles_path)
    chunks = get_smiles_list(num_smiles, num_chunks, smiles_path)
    os.chdir(chunks_path)
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(num_processes)

    min_ind = 0
    ind = 0
    for ind, chunk in enumerate(chunks):
        chunk_ind = min_ind + ind
        chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
        process_smiles_to_crystal_opt(chunks[ind], chunk_path, np.array([1, 6, 7, 8, 9]), 1, False,
                                      ** {
                                          'max_num_atoms': 30,
                                            'max_num_heavy_atoms': 9,
                                            'pare_to_size': 9,
                                            'max_radius': 15,
                                            'protonate': True,
                                            'rotamers_per_sample': 1,
                                            'allow_simple_hydrogen_rotations': False,
                                          'do_partial_charges': True
        })
    #
    # outputs = []
    # min_ind = 0
    # for ind, chunk in enumerate(chunks):
    #     print(f'starting chunk {ind} with {len(chunk)} smiles')
    #     chunk_ind = min_ind + ind
    #     chunk_path = os.path.join(chunks_path, f'chunk_{chunk_ind}.pkl')
    #     outputs.append(pool.apply_async(process_smiles_to_crystal_opt,
    #                                     args=(chunk, chunk_path, np.array([1, 6, 7, 8, 9]), 1, False),
    #                                     kwds={
    #                                         'max_num_atoms': 30,
    #                                         'max_num_heavy_atoms': 9,
    #                                         'pare_to_size': 9,
    #                                         'max_radius': 15,
    #                                         'protonate': True,
    #                                         'rotamers_per_sample': 1,
    #                                         'allow_simple_hydrogen_rotations': False,
    #                                         'do_partial_charges': True,
    #                                     }))
    # [print(out.get()) for out in outputs]

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
            make_figs=True,
        )

    if test:
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
            make_figs=True,
        )

    aa = 1
