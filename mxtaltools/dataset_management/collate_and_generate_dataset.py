# collate dataset from processed chunks
from mxtaltools.dataset_management.data_manager import DataManager
#
# 'CSD crystal dataset'
# miner = DataManager(device='cpu',
#                     datasets_path=r"D:\crystal_datasets/",
#                     chunks_path=r"D:\crystal_datasets/CSD_featurized_chunks/",
#                     dataset_type='crystal')
# miner.process_new_dataset(new_dataset_name='CSD_dataset')


'CSD/QM9 crystal dataset'
miner = DataManager(device='cpu',
                    datasets_path=r"D:\crystal_datasets/",
                    chunks_path=r"D:\crystal_datasets/CSD_QM9_featurized_chunks/",
                    dataset_type='crystal')
miner.process_new_dataset(new_dataset_name='CSD_QM9_dataset')


# 'QM9 molecules dataset'
# miner = DataManager(device='cpu',
#                     datasets_path=r"D:\crystal_datasets/",
#                     chunks_path=r"D:\crystal_datasets/QM9_chunks/",
#                     dataset_type='molecule')
# miner.process_new_dataset(new_dataset_name='qm9_dataset')

# 'GEOM Drugs dataset - partial'  # DEPRECATED
# miner = DataManager(device='cpu',
#                     datasets_path=r"D:\crystal_datasets/",
#                     chunks_path=r"D:\crystal_datasets/drugs_crude.msgpack/drugs_chunks",
#                     dataset_type='molecule')
# miner.process_new_dataset(new_dataset_name='GEOM_QM9_DRUGS_dataset', max_chunks=100, samples_per_chunk=10000)
