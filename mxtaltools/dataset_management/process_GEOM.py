import msgpack
import os
import numpy as np
import tqdm
import torch

from mxtaltools.dataset_management.CrystalData import CrystalData

if __name__ == '__main__':
    direc = 'D:\crystal_datasets\drugs_crude.msgpack/'
    filename = os.path.join(direc, "drugs_crude.msgpack")
    #filename = os.path.join(direc, "qm9_crude.msgpack")
    file = open(filename, "rb")
    unpacker = msgpack.Unpacker(file)

    skip_steps = 190
    #skip_steps = 100
    #skip_steps = 200
    #skip_steps = 225

    iter = -1
    with tqdm.tqdm(total=300) as pbar:
        while iter <= 300:
            pbar.update(1)
            iter += 1
            if iter < skip_steps:
                unpacker.skip()
                continue

            if not os.path.exists(direc + f"drugs_chunks/drugs_chunk_{iter}.pt"):
                data_batch = unpacker.unpack()
                data_list = []
                for smiles, entry in data_batch.items():
                    for conformer_ind, conformer in enumerate(entry['conformers']):
                        atoms_arr = np.array(conformer['xyz'])
                        #
                        # if np.amax(atoms_arr[:, 0]) > max_atom_type:  # filter samples with big atoms
                        #     continue
                        # radius = np.amax(np.linalg.norm(atoms_arr[:, 1:] - atoms_arr[:, 1:].mean(0), axis=1))
                        # if radius > max_mol_radius:  # filter beyond a max radius
                        #     continue

                        data_list.append(CrystalData(
                            x=torch.tensor(atoms_arr[:, 0], dtype=torch.long),
                            pos=torch.tensor(atoms_arr[:, 1:], dtype=torch.float32),
                            smiles=smiles,
                            identifier=smiles + '_' + str(conformer_ind),
                            y=torch.zeros(1, dtype=torch.float32)
                        ))

                #torch.save(data_list, direc + f"drugs_chunks/qm9_chunk_{iter}.pt")
                torch.save(data_list, direc + f"drugs_chunks/drugs_chunk_{iter}.pt")

            else:
                unpacker.skip()

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
