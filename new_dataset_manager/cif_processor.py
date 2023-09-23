import os
from ccdc import io
import rdkit.Chem as Chem
import pandas as pd
import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

from new_dataset_manager.featurization_utils import extract_crystal_data, featurize_molecule


def crystal_filter(crystal):
    """
    apply checks to see if this is a valid crystal to be featurized and put in the dataset
    - disorder
    """
    passed_crystal_checks = True
    passed_molecule_checks = True
    # crystal checks
    if any([crystal.has_disorder,
            crystal.molecule.is_polymeric,
            len(crystal.molecule.atoms) == 0,
            not crystal.molecule.all_atoms_have_sites,
            len(crystal.molecule.components) != crystal.z_prime,
            crystal.z_prime < 1,
            int(crystal.z_prime) != crystal.z_prime,  # integer z-prime only
            len(crystal.molecule.components) == 0,
            all([len(component.atoms) > 0 for component in crystal.molecule.components]),
            ]):
        passed_crystal_checks = False

    # molecule check via RDKit. If RDKit doesn't see it as a real molecule, don't accept it to the dataset.
    rd_mols = []
    for component in crystal.asymmetric_unit_molecule.components:
        mol = Chem.MolFromMol2Block(component.to_string('mol2'), sanitize=True, removeHs=True)
        try:
            rd_mols.append(Chem.RemoveAllHs(mol))
        except:
            passed_molecule_checks = False
        if mol is None:
            passed_molecule_checks = False

    if not passed_crystal_checks:
        print(f'{crystal.identifier} failed crystal checks')
    if not passed_molecule_checks:
        print(f'{crystal.identifier} failed molecule checks')

    return all([passed_molecule_checks, passed_crystal_checks]), rd_mols

dataset_path = r'D:/crystal_datasets/dataset.pkl'

if os.path.exists(dataset_path):
    df = pd.read_pickle(dataset_path)
else:
    df = None

cifs_path = r'D:/CSD_dump/'
os.chdir(cifs_path)
cifs_list = os.listdir()

'''
todo:
asymmetric unit parameterization
'''
start_point = 0
for ind, cif_path in enumerate(tqdm.tqdm(cifs_list)):
    do_featurize = False
    if df is not None:
        if cif_path.split('.')[0] not in df['crystal_identifier']:
            do_featurize = True
    else:
        do_featurize = True

    if do_featurize:
        reader = io.CrystalReader(cif_path, format='cif')

        for crystal in reader:  # one cif file may have many crystals in it
            passed_filter, rd_mols = crystal_filter(crystal)
            if passed_filter:  # filter various undesirable traits
                crystal_dict = extract_crystal_data(crystal)
                molecules = []
                for i_c, rd_mol in enumerate(rd_mols):
                    molecules.append(featurize_molecule(crystal, crystal_dict, rd_mol, component_num=i_c))

                crystal_keys = list(crystal_dict.keys())
                for key in crystal_keys:
                    crystal_dict['crystal_' + key] = crystal_dict[key]
                    del crystal_dict[key]

                for key in molecules[0].keys():
                    crystal_dict[key] = []
                    for molecule in molecules:
                        crystal_dict[key].append(molecule[key])

                new_df = pd.DataFrame()
                for key in crystal_dict.keys():
                    new_df[key] = [crystal_dict[key]]

                if df is None:
                    df = new_df
                else:
                    df = pd.concat([df, new_df])

                if len(df) % 100 == 0:  # save each k iters
                    df.to_pickle(dataset_path)

df.to_pickle(dataset_path)

aa = 1
