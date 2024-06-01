import msgpack
import os
import numpy as np
from tqdm import tqdm
import torch
import lmdb
import pickle

from mxtaltools.dataset_management.CrystalData import CrystalData


def write_lmdb(database, current_map_size, dict_to_write):
    try:
        with lmdb.open(database, map_size=current_map_size) as db:
            with db.begin(write=True) as txn:
                for key, value in dict_to_write.items():
                    txn.put(key.encode('ascii'), pickle.dumps(value))

        return current_map_size

    except lmdb.MapFullError:
        assert current_map_size < 2e10
        new_map_size = int(current_map_size * 1.25)
        print(f'Boosting map size from {current_map_size / 1e9:.1f}GB to {new_map_size / 1e9:.1f}GB')
        current_map_size = new_map_size
        return write_lmdb(database, current_map_size, dict_to_write)


if __name__ == '__main__':
    direc = r'D:\crystal_datasets\drugs_crude.msgpack'
    data_type = 'drugs'  # 'drugs 'or 'qm9'
    filename = os.path.join(direc, f"{data_type}_crude.msgpack")
    file = open(filename, "rb")
    unpacker = msgpack.Unpacker(file)
    lmdb_database = 'train_full.lmdb'
    map_size = int(30e9)  # map size in bytes

    min_chunk = 5
    max_chunk = 400  # qm9 has 135 chunks, drugs has 296
    'test dataset approx 500k samples from chunks 0-5 in qm9 and drugs'
    'train dataset from subsequent 5 chunks'

    os.chdir(direc)
    if not os.path.exists(lmdb_database.split('.lmdb')[0] + '_keys.npy'):
        keys_dict = {}
    else:
        keys_dict = np.load(lmdb_database.split('.lmdb')[0] + '_keys.npy', allow_pickle=True).item()

    chunk_ind = - 1
    with tqdm(total=max_chunk) as pbar:
        while chunk_ind < max_chunk - 1:
            pbar.update(1)
            chunk_ind += 1
            # todo replace this by checking the largest chunk in the keys dict and incrementing by one
            if keys_dict != {}:
                chunk_to_print = max(list(keys_dict.keys())) + 1
            else:
                chunk_to_print = 0

            if not (max_chunk > chunk_ind >= min_chunk):
                unpacker.skip()
                continue

            data_batch = unpacker.unpack()
            data_dict = {}
            smiles_ind = 0
            keys_dict[chunk_to_print] = {}
            for s_ind, (smiles, entry) in enumerate((data_batch.items())):
                samples = []
                for conformer_ind, conformer in enumerate((entry['conformers'])):
                    atoms_arr = np.array(conformer['xyz'])
                    if len(atoms_arr) < 6 or len(atoms_arr) > 100:
                        continue
                    sample = CrystalData(
                        x=torch.tensor(atoms_arr[:, 0], dtype=torch.long),
                        pos=torch.tensor(atoms_arr[:, 1:], dtype=torch.float32),
                        smiles=smiles,
                        identifier=smiles + '_' + str(conformer_ind),
                        y=torch.zeros(1, dtype=torch.float32),
                        require_crystal_features=False,
                    )
                    if sample.radius > 15:
                        continue

                    samples.append(sample.to_dict())

                if len(samples) > 0:
                    sample_inds = [f'{chunk_to_print}_' + str(smiles_ind) + '_' + str(cc_idx) for cc_idx in
                                   range(len(samples))]
                    data_dict.update({k: v for k, v in zip(sample_inds, samples)})
                    keys_dict[chunk_to_print][smiles_ind] = len(samples)
                    smiles_ind += 1

            np.save(lmdb_database.split('.lmdb')[0] + '_keys', keys_dict)

            map_size = write_lmdb(lmdb_database, map_size, data_dict)

''' # how to index the database
if True:
    data_type = 'qm9'
    samplewise_index = [np.concatenate([np.zeros(1), np.cumsum([chunk[s_ind] for s_ind in chunk.keys()])]).astype(int) for chunk in keys_dict[data_type].values()]
    chunk_tail_index = np.concatenate([np.zeros(1),np.cumsum([s[-1] for s in samplewise_index])]).astype(int)
    
    for idx in range(len(keys)):
    
        chunk_ind = np.digitize(idx, chunk_tail_index) - 1
        index_in_chunk = idx - chunk_tail_index[chunk_ind]
        sample_index = np.digitize(index_in_chunk, samplewise_index[chunk_ind]) - 1
        index_in_sample = index_in_chunk - samplewise_index[chunk_ind][sample_index]
    
        sample_key = f'{data_type}_chunk_{chunk_ind}_{sample_index}_{index_in_sample}'
        assert sample_key.encode('ascii') in keys, f'{idx} mismatch for {sample_key}'

'''
'''
n_samples = 0
n_smiles = 0
for iter, data_batch in enumerate(tqdm.tqdm(unpacker)):
    if iter * 1000 < max_dataset_size:
        for smiles, entry in data_batch.items():
            n_smiles += 1
            for conformer in entry['conformers']:
                n_samples += 1

print(iter)
print(n_smiles)
print(n_samples)

295it [21:32,  4.38s/it]
294 ITERS
248090 MOLECULES
26105787 CONFORMERS
'''

#
# def geom_msgpack_to_minimal_dataset(filename, max_dataset_size=1000000):
#     file = open(filename, "rb")
#     unpacker = msgpack.Unpacker(file)
#     #atomic_numbers, coordinates, num_atoms, identifier, smileses, num_samples, radius = [], [], [], [], [], [], []
#     num_bins = 100
#     min_atomic_numbers, max_atomic_numbers, min_radius, max_radius, min_num_atoms, max_num_atoms, min_num_samples, max_num_samples = (
#         10000000, 0, 10000000, 0, 10000000, 0, 10000000, 0)
#     for iter, data_batch in enumerate(tqdm.tqdm(unpacker)):
#         if iter * 1000 < max_dataset_size:
#             atomic_numbers, coordinates, num_atoms, identifier, smileses, num_samples, radius = [], [], [], [], [], [], []
#             for smiles, entry in data_batch.items():
#                 for conformer in entry['conformers']:
#                     atoms_arr = np.array(conformer['xyz'])
#                     atomic_numbers.extend(atoms_arr[:, 0].astype(int))
#                     coordinates.append(atoms_arr[:, 1:].astype(float))
#                     radius.append(np.amax(np.linalg.norm(atoms_arr[:, 1:] - atoms_arr[:, 1:].mean(0), axis=1)))
#                     num_atoms.append(int(len(atoms_arr)))
#                     identifier.append(conformer['geom_id'])
#                     smileses.append(smiles)
#                     num_samples.append(len(entry['conformers']))
#             min_atomic_numbers, max_atomic_numbers, min_radius, max_radius, min_num_atoms, max_num_atoms, min_num_samples, max_num_samples = (
#                 min(min_atomic_numbers, min(atomic_numbers)), max(max_atomic_numbers, max(atomic_numbers)),
#                 min(min_radius, min(radius)), max(max_radius, max(radius)), min(min_num_atoms, min(num_atoms)),
#                 max(max_num_atoms, max(num_atoms)), min(min_num_samples, min(num_samples)),
#                 max(max_num_samples, max(num_samples)))
#         else:
#             break
#     file.close()
#     unpacker = msgpack.Unpacker(open(filename, "rb"))
#     overall_hist_atomic_numbers, overall_hist_radius, overall_hist_num_atoms, overall_hist_num_samples = (
#         None, None, None, None)
#     for iter, data_batch in enumerate(tqdm.tqdm(unpacker)):
#         if iter * 1000 < max_dataset_size:
#             atomic_numbers, coordinates, num_atoms, identifier, smileses, num_samples, radius = [], [], [], [], [], [], []
#             for smiles, entry in data_batch.items():
#                 for conformer in entry['conformers']:
#                     atoms_arr = np.array(conformer['xyz'])
#                     atomic_numbers.extend(atoms_arr[:, 0].astype(int))
#                     coordinates.append(atoms_arr[:, 1:].astype(float))
#                     radius.append(np.amax(np.linalg.norm(atoms_arr[:, 1:] - atoms_arr[:, 1:].mean(0), axis=1)))
#                     num_atoms.append(int(len(atoms_arr)))
#                     identifier.append(conformer['geom_id'])
#                     smileses.append(smiles)
#                     num_samples.append(len(entry['conformers']))
#             atomic_numbers_hist = np.histogram(atomic_numbers, range=[min_atomic_numbers, max_atomic_numbers],
#                                                bins=num_bins)
#             radius_hist = np.histogram(radius, range=[min_radius, max_radius], bins=num_bins)
#             num_atoms_hist = np.histogram(num_atoms, range=[min_num_atoms, max_num_atoms], bins=num_bins)
#             num_samples_hist = np.histogram(num_samples, range=[min_num_samples, max_num_samples], bins=num_bins)
#             if not isinstance(overall_hist_atomic_numbers, np.ndarray):
#                 overall_hist_atomic_numbers = atomic_numbers_hist
#             else:
#                 overall_hist_atomic_numbers += atomic_numbers_hist
#             if not isinstance(overall_hist_radius, np.ndarray):
#                 overall_hist_radius = radius_hist
#             else:
#                 overall_hist_radius += radius_hist
#             if not isinstance(overall_hist_num_atoms, np.ndarray):
#                 overall_hist_num_atoms = num_atoms_hist
#             else:
#                 overall_hist_num_atoms += num_atoms_hist
#             if not isinstance(overall_hist_num_samples, np.ndarray):
#                 overall_hist_num_samples = num_samples_hist
#             else:
#                 overall_hist_num_samples += num_samples_hist
#         else:
#             break
#     return overall_hist_atomic_numbers, overall_hist_radius, overall_hist_num_atoms, overall_hist_num_samples
#
#
# direc = 'D:\crystal_datasets\drugs_crude.msgpack/'
# qm9_crude = os.path.join(direc, "drugs_crude.msgpack")
# overall_hist_atomic_numbers, overall_hist_radius, overall_hist_num_atoms, overall_hist_num_samples = (
#     geom_msgpack_to_minimal_dataset(qm9_crude, max_dataset_size=250000))
# plt.figure()
# plt.bar(overall_hist_atomic_numbers[1][:-1], np.log10(overall_hist_atomic_numbers[0]))
# plt.figure()
# plt.bar(overall_hist_radius[1][:-1], np.log10(overall_hist_radius[0]))
# plt.figure()
# plt.bar(overall_hist_num_atoms[1][:-1], np.log10(overall_hist_num_atoms[0]))
# plt.figure()
# plt.bar(overall_hist_num_samples[1][:-1], np.log10(overall_hist_num_samples[0]))
# plt.show()

aa = 1
# import msgpack
# import os
# import numpy as np
# import pandas as pd
# import tqdm
#
#
# # filename = 'qm9_crude.msgpack'
# # datasets_directory = 'D:/crystal_datasets/'
#
# def geom_msgpack_to_minimal_dataset(filename, datasets_directory, max_dataset_size=1000000):
#     datafile = os.path.join(datasets_directory, filename)
#     unpacker = msgpack.Unpacker(open(datafile, "rb"))
#     df = pd.DataFrame(columns=['atom_atomic_numbers', 'atom_coordinates', 'molecule_num_atoms', 'identifier',
#                                'molecule_smiles', 'molecule_num_duplicates', 'molecule_raidus'])  # the minimal set of information needed for training
#     atomic_numbers, coordinates, num_atoms, identifier, smileses, num_samples, radius = [], [], [], [], [], [], []
#
#     for iter, data_batch in enumerate(tqdm.tqdm(unpacker)):
#         if iter * 1000 < max_dataset_size:
#             for smiles, entry in data_batch.items():
#                 for conformer in entry['conformers']:
#                     atoms_arr = np.array(conformer['xyz'])
#                     atomic_numbers.append(atoms_arr[:, 0].astype(int))
#                     coordinates.append(atoms_arr[:, 1:].astype(float))
#                     radius.append(np.amax(np.linalg.norm(atoms_arr[:, 1:] - atoms_arr[:, 1:].mean(0), axis=1)))
#                     num_atoms.append(int(len(atoms_arr)))
#                     identifier.append(conformer['geom_id'])
#                     smileses.append(smiles)
#                     num_samples.append(len(entry['conformers']))
#         else:
#             break
#
#     df['atom_atomic_numbers'] = atomic_numbers
#     df['atom_coordinates'] = coordinates
#     df['molecule_num_atoms'] = num_atoms
#     df['identifier'] = identifier
#     df['molecule_smiles'] = smileses
#     df['molecule_num_duplicates'] = num_samples
#     df['molecule_radius'] = radius
#
#     return df


# DATASET STATS
