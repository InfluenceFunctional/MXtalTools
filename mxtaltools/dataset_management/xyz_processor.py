import os
import pandas as pd
import tqdm
import warnings
from random import shuffle

from mxtaltools.common.geometry_calculations import compute_principal_axes_np
from mxtaltools.common.utils import chunkify
import numpy as np
from mxtaltools.constants.atom_properties import ATOMIC_NUMBERS, ATOM_WEIGHTS, VDW_RADII, ATOMIC_SYMBOLS, ELECTRONEGATIVITY, PERIOD, GROUP
from rdkit import Chem as Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, rdFreeSASA

from mxtaltools.dataset_management.featurization_utils import get_dipole
from mxtaltools.dataset_management.utils import get_fraction

HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
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
                    atom_types[ind] = int(ATOMIC_NUMBERS[line[0]])
                    atom_coords[ind, :] = float(line[1]), float(line[2]), float(line[3])
                    atom_charges[ind] = float(line[4])

                molecule_dict['atom_coordinates'] = atom_coords
                molecule_dict['atom_atomic_numbers'] = atom_types
                molecule_dict['atom_partial_charges'] = atom_charges
                molecule_dict['atom_group'] = [group_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
                molecule_dict['atom_period'] = [period_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
                molecule_dict['atom_vdW_radius'] = [vdw_radii_dict[number] for number in molecule_dict['atom_atomic_numbers']]
                molecule_dict['atom_electronegativity'] = [electronegativity_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
                molecule_dict['atom_mass'] = [atomic_weight_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]

                molecule_dict['molecule_smiles'] = text[-3].split('\t')[0]
                molecule_dict['molecule_radius'] = np.amax(np.linalg.norm(molecule_dict['atom_coordinates'] - molecule_dict['atom_coordinates'].mean(0), axis=-1))
                molecule_dict['molecule_volume'] = np.random.uniform(0)  # explicit dummy value
                molecule_dict['molecule_mass'] = np.sum(molecule_dict['atom_mass'])

                mol = Chem.MolFromSmiles(molecule_dict['molecule_smiles'])

                h_donors = list(sum(mol.GetSubstructMatches(HDonorSmarts, uniquify=1), ()))
                h_acceptors = list(sum(mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1), ()))

                '''molecule-wise features'''
                radii = rdFreeSASA.classifyAtoms(mol)
                #molecule_dict['molecule_freeSASA'] = rdFreeSASA.CalcSASA(mol, radii)
                molecule_dict['molecule_mass'] = Descriptors.MolWt(mol)  # includes implicit protons
                molecule_dict['molecule_num_atoms'] = len(molecule_dict['atom_coordinates'])  # mol.GetNumAtoms()
                molecule_dict['molecule_num_rings'] = mol.GetRingInfo().NumRings()
                molecule_dict['molecule_num_donors'] = len(h_donors)
                molecule_dict['molecule_num_acceptors'] = len(h_acceptors)
                molecule_dict['molecule_polarity'], _ = get_dipole(molecule_dict['atom_coordinates'], molecule_dict['atom_electronegativity'])
                #molecule_dict['molecule_spherical_defect'] = rdMolDescriptors.CalcAsphericity(mol)
                #molecule_dict['molecule_eccentricity'] = rdMolDescriptors.CalcEccentricity(mol)
                molecule_dict['molecule_num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                #molecule_dict['molecule_planarity'] = rdMolDescriptors.CalcPBF(mol)
                #molecule_dict['molecule_radius_of_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol)
                molecule_dict['molecule_radius'] = np.amax(np.linalg.norm(molecule_dict['atom_coordinates'] - molecule_dict['atom_coordinates'].mean(0), axis=-1))

                for anum in range(1, 10):
                    molecule_dict[f'molecule_{element_symbols_dict[anum]}_fraction'] = get_fraction(molecule_dict['atom_atomic_numbers'], anum)

                for frag in Fragments.__dict__.keys():  # for all the class methods
                    if frag[0:3] == 'fr_':  # if it's a functional group analysis methodad
                        molecule_dict[f'molecule_{frag[3:]}_count'] = Fragments.__dict__[frag](mol, countUnique=False)

                molecule_dict['molecule_chemical_formula'] = rdMolDescriptors.CalcMolFormula(mol)

                Ip, Ipm, _ = compute_principal_axes_np(np.asarray(molecule_dict['atom_coordinates']))  # rdMolTransforms.ComputePrincipalAxesAndMoments(mol.GetConformer(), ignoreHs=False) # this does it column-wise
                molecule_dict['molecule_principal_axes'] = Ip  # row-wise principal_axes
                molecule_dict['molecule_principal_moment_1'] = Ipm[0]
                molecule_dict['molecule_principal_moment_2'] = Ipm[1]
                molecule_dict['molecule_principal_moment_3'] = Ipm[2]
                molecule_dict['molecule_is_spherical_top'] = Ipm[0] == Ipm[1] == Ipm[2]
                molecule_dict['molecule_is_symmetric_top'] = any([
                    Ipm[0] != Ipm[1] == Ipm[2],
                    Ipm[0] == Ipm[1] != Ipm[2],
                    Ipm[0] == Ipm[2] != Ipm[1]
                ])
                molecule_dict['molecule_is_asymmetric_top'] = not Ipm[0] == Ipm[1] == Ipm[2]


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
