import os
import pandas as pd
import tqdm
import warnings
from random import shuffle
from common.utils import chunkify
import numpy as np
from constants.atom_properties import NUMBERS

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

n_chunks = 100  # too many chunks can cause problems e.g., if some have zero valid entries
chunks_path = r'D:/crystal_datasets/QM9_chunks/'  # where you would like processed dataset chunks to be stored before collation into final dataset

chunk_prefix = ''
xyzs_path = r'D:\crystal_datasets\Molecule_Datasets\QM9/'

os.chdir(xyzs_path)
xyzs_list = os.listdir()

if not os.path.exists(chunks_path):
    os.mkdir(chunks_path)

n_chunks = min(n_chunks, len(xyzs_list))
print(f"Breaking dataset into {n_chunks} chunks")
chunks_list = chunkify(xyzs_list, n_chunks)
chunk_inds = [i for i in range(len(chunks_list))]
start_ind, stop_ind = 0, len(chunks_list)

shuffle(chunk_inds)  # optionally do it in random order
chunks_list = [chunks_list[ind] for ind in chunk_inds]

for chunk_ind, chunk in zip(chunk_inds, chunks_list[start_ind:stop_ind]):  # todo consider adding indexing over multiple or nested directories
    increment_df = None
    if not os.path.exists(chunks_path + chunk_prefix + f"_chunk_{chunk_ind}.pkl"):
        print(f"Starting chunk {chunk_ind} with {len(chunk)} xyzs")
        for ind, xyz_path in enumerate(tqdm.tqdm(chunk)):
            with open(xyz_path, "r") as f:
                text = f.read().split('\n')

            try:
                props = text[1].split('\t')
                molecule_dict = {
                    "molecule_num_atoms": int(text[0]),
                    "identifier": int(props[0].split()[1]),
                    "molecule_rotational_constant_a": float(props[1]),
                    "molecule_rotational_constant_b": float(props[2]),
                    "molecule_rotational_constant_c": float(props[3]),
                    "molecule_dipole_moment": float(props[4]),
                    "molecule_isotropic_polarizability": float(props[5]),
                    "molecule_HOMO_energy": float(props[6]),
                    "molecule_LUMO_energy": float(props[7]),
                    "molecule_gap_energy": float(props[8]),
                    "molecule_el_spatial_extent": float(props[9]),
                    "molecule_zpv_energy": float(props[10]),
                    "molecule_internal_energy_0": float(props[11]),
                    "molecule_internal_energy_STP": float(props[12]),
                    "molecule_enthalpy_STP": float(props[13]),
                    "molecule_free_energy_STP": float(props[14]),
                    "molecule_heat_capacity_STP": float(props[15]),
                }

                atoms_block_text = text[2:molecule_dict['molecule_num_atoms'] + 2]
                atom_types = np.zeros(molecule_dict['molecule_num_atoms'], dtype=np.int_)
                atom_coords = np.zeros((molecule_dict['molecule_num_atoms'], 3))
                atom_charges = np.zeros(molecule_dict['molecule_num_atoms'])

                for ind, line in enumerate(atoms_block_text):
                    line = line.split('\t')
                    atom_types[ind] = int(NUMBERS[line[0]])
                    atom_coords[ind, :] = float(line[1]), float(line[2]), float(line[3])
                    atom_charges[ind] = float(line[4])

                molecule_dict['atom_coordinates'] = atom_coords
                molecule_dict['atom_atomic_numbers'] = atom_types
                molecule_dict['atom_partial_charges'] = atom_charges

                molecule_dict['molecule_radius'] = np.amax(np.linalg.norm(molecule_dict['atom_coordinates'] - molecule_dict['atom_coordinates'].mean(0), axis=-1))
                molecule_dict['molecule_volume'] = np.random.uniform(0)  # explicit dummy value
                molecule_dict['molecule_mass'] = np.random.uniform(0)  # explicit dummy value

                new_df = pd.DataFrame()
                for key in molecule_dict.keys():
                    new_df[key] = [molecule_dict[key]]

                if increment_df is None:
                    increment_df = new_df
                else:
                    increment_df = pd.concat([increment_df, new_df])

            except ValueError:
                pass

        if increment_df is not None:
            increment_df.to_pickle(chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")
