import pickle
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import lmdb


def write_lmdb(database, current_map_size, dict_to_write):
    try:
        with lmdb.open(database, map_size=current_map_size) as db:
            with db.begin(write=True) as txn:
                for key, value in dict_to_write.items():
                    txn.put(key.encode('ascii'), pickle.dumps(value))

        return current_map_size

    except lmdb.MapFullError:
        assert current_map_size < 2e11
        new_map_size = int(current_map_size * 1.25)
        print(f'Boosting map size from {current_map_size / 1e9:.1f}GB to {new_map_size / 1e9:.1f}GB')
        current_map_size = new_map_size
        return write_lmdb(database, current_map_size, dict_to_write)


if __name__ == '__main__':
    """
    Combine GEOM_drugs and zinc datasets into a single lmdb dataset.
    """
    zinc_directory = r'D:\crystal_datasets\zinc22_3d'
    #zinc_directory = r'/vast/mk8347/zinc22_3d'
    # geom_train_directory = r'/vast/mk8347/geom_drugs/train.lmdb'
    # geom_test_directory = r'/vast/mk8347/geom_drugs/test.lmdb'
    map_size = int(1e9)

    lmdb_database = 'zinc3d.lmdb'

    chunks_dir = os.path.join(Path(zinc_directory), 'chunks')

    keys_path = zinc_directory + '/' + lmdb_database.split('.lmdb')[0] + '_keys'
    database_path = zinc_directory + '/' + lmdb_database

    '''build zinc database'''
    os.chdir(chunks_dir)
    sample_counter = 0
    for chunk in tqdm(os.listdir()):
        chunk_path = Path(chunk)
        with open(chunk_path, 'rb') as f:
            samples = pickle.load(f)
            samples_dict = {str(sample_counter + ind): sample for ind, sample in enumerate(samples)}
            map_size = write_lmdb(database_path, map_size, samples_dict)
            sample_counter += len(samples_dict)
    #
    # '''combine with prebuilt geom database'''
    # with lmdb.open(database_path, map_size=map_size) as db:
    #     with db.begin(write=True) as txn:
    #         database_counter = 0
    #         with lmdb.open(geom_train_directory, map_size=map_size) as db2:
    #             with db.begin(write=True) as txn2:
    #                 txn.put(str(sample_counter).encode('ascii'), txn2.get(str(database_counter).encode('ascii')))
    #                 sample_counter += 1
    #                 database_counter += 1

    np.save(keys_path, sample_counter)
    print(f"Database saved with {sample_counter} samples")
