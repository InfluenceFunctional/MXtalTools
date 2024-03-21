import msgpack
import os
import numpy as np
import pandas as pd
import tqdm


# filename = 'qm9_crude.msgpack'
# datasets_directory = 'D:/crystal_datasets/'

def geom_msgpack_to_minimal_dataset(filename, datasets_directory, max_dataset_size=1000000):
    datafile = os.path.join(datasets_directory, filename)
    unpacker = msgpack.Unpacker(open(datafile, "rb"))
    df = pd.DataFrame(columns=['atom_atomic_numbers', 'atom_coordinates', 'molecule_num_atoms', 'identifier',
                               'molecule_smiles', 'molecule_num_duplicates', 'molecule_raidus'])  # the minimal set of information needed for training
    atomic_numbers, coordinates, num_atoms, identifier, smileses, num_samples, radius = [], [], [], [], [], [], []

    for iter, data_batch in enumerate(tqdm.tqdm(unpacker)):
        if iter * 1000 < max_dataset_size:
            for smiles, entry in data_batch.items():
                for conformer in entry['conformers']:
                    atoms_arr = np.array(conformer['xyz'])
                    atomic_numbers.append(atoms_arr[:, 0].astype(int))
                    coordinates.append(atoms_arr[:, 1:].astype(float))
                    radius.append(np.amax(np.linalg.norm(atoms_arr[:, 1:] - atoms_arr[:, 1:].mean(0), axis=1)))
                    num_atoms.append(int(len(atoms_arr)))
                    identifier.append(conformer['geom_id'])
                    smileses.append(smiles)
                    num_samples.append(len(entry['conformers']))
        else:
            break

    df['atom_atomic_numbers'] = atomic_numbers
    df['atom_coordinates'] = coordinates
    df['molecule_num_atoms'] = num_atoms
    df['identifier'] = identifier
    df['molecule_smiles'] = smileses
    df['molecule_num_duplicates'] = num_samples
    df['molecule_radius'] = radius

    return df
