import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path

from mxtaltools.dataset_utils.dataset_manager import DataManager
from mxtaltools.dataset_utils.construction.parallel_synthesis import otf_synthesize_crystals

if __name__ == '__main__':
    # initialize
    debug = True
    space_group = 1
    num_smiles = 10
    num_processes = 1
    new_dataset_name = f'pd_dataset_sg{space_group}'

    num_chunks = max(num_processes, num_smiles // 100)
    smiles_path = r'D:\crystal_datasets\zinc22'  #'/home/mkilgour/crystal_datasets/zinc22'#
    chunks_path = Path(r'D:\crystal_datasets')  # Path('/home/mkilgour/crystal_datasets') #
    os.chdir(chunks_path)
    mp.set_start_method('spawn', force=True)
    mp_pool = mp.Pool(num_processes)

    """synthesize random crystal dataset"""
    mp_pool = otf_synthesize_crystals(
        num_smiles,
        smiles_path,
        chunks_path,
        allowed_atom_types=[1, 6, 7, 8, 9],
        num_processes=num_processes,
        num_chunks=num_chunks,
        mp_pool=mp_pool,
        max_num_atoms=30,
        max_num_heavy_atoms=9,
        pare_to_size=9,
        max_radius=15,
        post_scramble_each=10,
        space_group=space_group,
        synchronize=True,
        do_embedding=False,
        do_mace_energy=True,
        debug=debug,
        embedding_type='principal_axes',
        encoder_checkpoint_path=r'C:\Users\mikem\crystals\CSP_runs\models\cluster/best_autoencoder_experiments_autoencoder_tests_otf_zinc_test3_7_05-12-14-03-45'
    )
    mp_pool.close()
    mp_pool.join()

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

