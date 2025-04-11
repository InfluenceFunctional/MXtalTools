import os
import warnings

import torch
import tqdm
from rdkit import Chem as Chem

from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII, ATOMIC_SYMBOLS, \
    ELECTRONEGATIVITY, PERIOD, GROUP
from mxtaltools.dataset_utils.construction.featurization_utils import chunkify_path_list, get_qm9_properties, \
    featurize_xyz_molecule
from mxtaltools.dataset_utils.data_classes import MolData

HDonorSmarts = Chem.MolFromSmarts(
    '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
HAcceptorSmarts = Chem.MolFromSmarts(
    '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
    '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
    '$([nH0,o,s;+0])]')

vdw_radii_dict = VDW_RADII
element_symbols_dict = ATOMIC_SYMBOLS
electronegativity_dict = ELECTRONEGATIVITY
period_dict = PERIOD
group_dict = GROUP
atomic_weight_dict = ATOM_WEIGHTS

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

"""
Convert xyz files to chunks of crystaldata objects
"""


def process_xyzs_to_chunks(xyzs_path, chunks_path, n_chunks):
    os.chdir(xyzs_path)
    xyzs_list = os.listdir()
    if not os.path.exists(chunks_path):
        os.mkdir(chunks_path)
    chunk_inds, chunks_list, start_ind, stop_ind = chunkify_path_list(xyzs_list, n_chunks)
    for chunk_ind, chunk in zip(chunk_inds, chunks_list[
                                            start_ind:stop_ind]):  # todo consider adding indexing over multiple or nested directories
        data_list = []
        if not os.path.exists(chunks_path + chunk_prefix + f"_chunk_{chunk_ind}.pkl"):
            print(f"Starting chunk {chunk_ind} with {len(chunk)} xyzs")
            for ind, xyz_path in enumerate(tqdm.tqdm(chunk)):
                with open(xyz_path, "r") as f:
                    text = f.read().split('\n')
                try:
                    if 'qm9' in chunks_path.lower():
                        molecule_dict, props = get_qm9_properties(text)
                    else:
                        molecule_dict = {'num_atoms': int(text[0])}

                    molecule_dict = featurize_xyz_molecule(molecule_dict, text)

                    data = MolData(
                        z=torch.tensor(molecule_dict['atom_atomic_numbers'], dtype=torch.long),
                        pos=torch.tensor(molecule_dict['atom_coordinates'], dtype=torch.float32),
                        smiles=molecule_dict['molecule_smiles'],
                        identifier=molecule_dict['molecule_smiles'],  #molecule_dict['identifier'],
                        x=torch.tensor(molecule_dict['partial_charges'], dtype=torch.float32),
                        y=torch.tensor([float(prop) for prop in props[1:-1]], dtype=torch.float32)[None,
                          :] if 'qm9' in chunks_path.lower() else None,
                        skip_mol_analysis=False,
                    )

                    batch = collate_data_batch([mol1, mol2, mol3, mol4])

                    data_list.append(data)

                except ValueError:
                    pass

            torch.save(data_list, chunks_path + f"{chunk_prefix}_chunk_{chunk_ind}.pkl")


if __name__ == '__main__':
    n_chunks = 100  # too many chunks can cause problems e.g., if some have zero valid entries
    chunks_path = r'D:/crystal_datasets/QM9_chunks/'  # where you would like processed dataset chunks to be stored before collation into final dataset

    chunk_prefix = ''
    xyzs_path = r'D:\crystal_datasets\Molecule_Datasets\QM9/'

    process_xyzs_to_chunks(xyzs_path=xyzs_path,
                           chunks_path=chunks_path,
                           n_chunks=n_chunks)
