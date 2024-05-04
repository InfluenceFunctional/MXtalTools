# collate dataset from processed chunks
from mxtaltools.dataset_management.data_manager import DataManager

# 'CSD crystal dataset'
# miner = DataManager(device='cpu',
#                     datasets_path=r"D:\crystal_datasets/",
#                     chunks_path=r"D:\crystal_datasets/CSD_featurized_chunks/",
#                     dataset_type='crystal')
# miner.process_new_dataset(new_dataset_name='dataset')

'QM9 molecules dataset'
miner = DataManager(device='cpu',
                    datasets_path=r"D:\crystal_datasets/",
                    chunks_path=r"D:\crystal_datasets/QM9_chunks/",
                    dataset_type='molecule')
miner.process_new_dataset(new_dataset_name='qm9_molecules_dataset')
