import numpy as np
import lmdb
from tqdm import tqdm

root = r'D:\crystal_datasets\drugs_crude.msgpack\\'
keys_filename = root + 'test_full_keys.npy'
db_filename = root + 'test_full.lmdb'

if __name__ == '__main__':
    env = lmdb.open(
        db_filename,
        readonly=True,
    )
    txn = env.begin()
    data_keys = list(txn.cursor().iternext(values=False))

    keys_dict = np.load(keys_filename, allow_pickle=True).item()

    samplewise_index = [np.concatenate([np.zeros(1), np.cumsum([chunk[s_ind] for s_ind in chunk.keys()])]).astype(int)
                        for chunk in keys_dict.values()]
    chunk_tail_index = np.concatenate([np.zeros(1), np.cumsum([s[-1] for s in samplewise_index])]).astype(int)
    
    for idx in tqdm(range(len(data_keys))):
        chunk_ind = np.digitize(idx, chunk_tail_index) - 1
        index_in_chunk = idx - chunk_tail_index[chunk_ind]
        sample_index = np.digitize(index_in_chunk, samplewise_index[chunk_ind]) - 1
        index_in_sample = index_in_chunk - samplewise_index[chunk_ind][sample_index]

        sample_key = f'{chunk_ind}_{sample_index}_{index_in_sample}'
    
        assert sample_key.encode('ascii') in data_keys, f'{idx} mismatch for {sample_key}'

