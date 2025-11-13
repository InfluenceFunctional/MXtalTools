from mxtaltools.dataset_utils.dataset_manager import DataManager

#
# collate final dataset from preprocessed chunks
#


if __name__ == '__main__':
    'CSD crystal dataset'
    miner = DataManager(device='cpu',
                        datasets_path=r"D:\crystal_datasets/",
                        chunks_path=r"D:\crystal_datasets/CSD_featurized_chunks/",
                        dataset_type='crystal')
    miner.process_new_dataset(new_dataset_name='new_new_csd',
                              chunks_patterns=None)

    # 'QM9 molecules dataset'
    # miner = DataManager(device='cpu',
    #                     datasets_path=r"D:\crystal_datasets/",
    #                     chunks_path=r"D:\crystal_datasets/qm9_featurized_chunks/",
    #                     dataset_type='molecule')
    # miner.process_new_dataset(new_dataset_name='qm9_dataset')


    # 'QM9s molecules dataset'
    # miner = DataManager(device='cpu',
    #                     datasets_path=r"D:\crystal_datasets/",
    #                     chunks_path=r"D:\crystal_datasets/qm9s_featurized_chunks/",
    #                     dataset_type='molecule')
    # miner.process_new_dataset(new_dataset_name='qm9s_dataset')