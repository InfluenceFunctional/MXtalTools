import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path

from mxtaltools.dataset_utils.dataset_manager import DataManager
from mxtaltools.dataset_utils.synthesis.utils import otf_synthesize_crystals

if __name__ == '__main__':
    # initialize
    debug=True
    num_smiles = 50
    num_processes = 1
    num_chunks = max(num_processes, num_smiles // 500)
    smiles_path = r'D:\crystal_datasets\zinc22'  #'/home/mkilgour/crystal_datasets/zinc22'#
    chunks_path = Path(r'D:\crystal_datasets')  # Path('/home/mkilgour/crystal_datasets') #
    new_dataset_name = 'pd_dataset_toy'
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
        space_group=1,
        synchronize=True,
        do_embedding=False,
        debug=debug,
        embedding_type='principal_axes',
        encoder_checkpoint_path= r'C:\Users\mikem\crystals\CSP_runs\models\cluster/best_autoencoder_experiments_autoencoder_tests_otf_zinc_test3_7_05-12-14-03-45'
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

    aa = 1

    # if test:
    #     '''check sample quality'''
    #     mol_batch = collate_data_list([elem for elem in miner.dataset[:1000]]).clone().cpu()
    #     opt_vdw_pot = mol_batch.vdw_pot[None, ...]
    #     opt_vdw_loss = mol_batch.vdw_loss[None, ...]
    #     opt_aunits = mol_batch.pos[None, ...]
    #     cell_params = torch.cat([
    #         mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
    #     ], dim=1)
    #     opt_cell_params = cell_params[None, ...]
    #
    #     r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
    #         mol_batch,
    #         opt_vdw_pot,
    #         opt_vdw_loss,
    #         opt_aunits,
    #         opt_cell_params,
    #         denorm=False,
    #         destd=False,
    #         renorm=False,
    #         restd=False,
    #         make_figs=True,
    #     )
    #
    # if test:
    #     del miner.dataset
    #
    #     '''reload for testing'''
    #     conv_cutoff = 6
    #     # nonzero_positional_noise = sum(list(self.config.positional_noise.__dict__.values()))
    #     miner.regression_target = None
    #     miner.load_dataset_for_modelling(
    #         new_dataset_name + '.pt',
    #         filter_conditions=None,
    #         filter_polymorphs=False,
    #         filter_duplicate_molecules=False,
    #         filter_protons=False,
    #         conv_cutoff=conv_cutoff,
    #         do_shuffle=True,
    #         precompute_edges=False,
    #         single_identifier=None,
    #     )
    #     '''check sample quality'''
    #     collater = Collater(0, 0)
    #     mol_batch = collater([elem for elem in miner.dataset[:1000]]).clone().cpu()
    #     opt_vdw_pot = mol_batch.vdw_pot[None, ...]
    #     opt_vdw_loss = mol_batch.vdw_loss[None, ...]
    #     opt_aunits = mol_batch.pos[None, ...]
    #     cell_params = torch.cat([
    #         mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
    #     ], dim=1)
    #     opt_cell_params = cell_params[None, ...]
    #
    #     r_pot, r_loss, r_au = test_crystal_rebuild_from_embedding(
    #         mol_batch,
    #         opt_vdw_pot,
    #         opt_vdw_loss,
    #         opt_aunits,
    #         opt_cell_params,
    #         denorm=False,
    #         destd=False,
    #         renorm=False,
    #         restd=False,
    #         make_figs=True,
    #     )

