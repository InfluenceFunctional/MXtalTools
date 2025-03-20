import os
from argparse import Namespace
from pathlib import Path

from mxtaltools.dataset_utils.dataset_manager import DataManager

if __name__ == '__main__':
    # initialize
    space_group = 1
    # new_dataset_name = f'pd_dataset_sg{space_group}'
    # chunks_path = Path(r'/scratch/mk8347/csd_runs/datasets')
    new_dataset_name = f'pd_dataset_sg{space_group}_test'
    chunks_path = Path(r'D:\crystal_datasets')
    os.chdir(chunks_path)

    '''process and save dataset'''
    miner = DataManager(device='cpu',
                        config=Namespace(**{'seed': 0,
                                            'max_dataset_length': 100000000,
                                            'filter_protons': False}),
                        datasets_path=chunks_path,
                        chunks_path=chunks_path,
                        dataset_type='crystal',
                        do_crystal_indexing=False)
    miner.process_new_dataset(new_dataset_name=new_dataset_name,
                              chunks_patterns=[f'sg_{space_group}_chunk'],
                              build_stats=False)
    print(f"finished processing datast with {len(miner.dataset)} samples")
