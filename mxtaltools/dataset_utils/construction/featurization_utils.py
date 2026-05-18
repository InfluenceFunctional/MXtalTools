from random import shuffle

import numpy as np
import rdkit.Chem.AllChem as AllChem
import torch
from rdkit import Chem as Chem
from rdkit import rdBase
from rdkit.Chem import rdMolDescriptors
from scipy.spatial.distance import cdist

from mxtaltools.common.geometry_utils import coor_trans_matrix_np
from mxtaltools.common.utils import chunkify
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import ELECTRONEGATIVITY, PERIOD, GROUP, VDW_RADII, ATOMIC_SYMBOLS, \
    ATOMIC_NUMBERS
from mxtaltools.constants.space_group_info import SPACE_GROUPS
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData
from mxtaltools.dataset_utils.mol_building import get_partial_charges
from mxtaltools.dataset_utils.utils import collate_data_list

"""
Utilities for featurizing molecules and crystals for construction of MXtalTools Data objects.
combining RDKit and custom analysis functions
"""

# block rdkit logs
blocker = rdBase.BlockLogs()

'''setup fingerprint generator'''
fingerprint_generator = AllChem.GetMorganGenerator(radius=2, includeChirality=False)

'''set up some constants'''
vdw_radii_dict = VDW_RADII
element_symbols_dict = ATOMIC_SYMBOLS
electronegativity_dict = ELECTRONEGATIVITY
period_dict = PERIOD
group_dict = GROUP

for key in electronegativity_dict.keys():
    if electronegativity_dict[key] is None:
        electronegativity_dict[key] = 0

HDonorSmarts = Chem.MolFromSmarts(
    '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
HAcceptorSmarts = Chem.MolFromSmarts(
    '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' + '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' + '$([nH0,o,s;+0])]')

sg_numbers = {}
for i in range(1, 231):
    sg_numbers[SPACE_GROUPS[i]] = i

asym_unit_dict = ASYM_UNITS.copy()
for key in asym_unit_dict:
    asym_unit_dict[key] = torch.tensor(asym_unit_dict[key], dtype=torch.float32)


def get_dipole(coords, charges):
    """
    get the absolute difference between the center of charge and center of geometry
    """
    center_of_geometry = np.average(np.asarray(coords), axis=0)
    center_of_charge = np.average(np.asarray(coords), weights=charges, axis=0)
    return np.linalg.norm(center_of_charge - center_of_geometry), center_of_charge - center_of_geometry


def get_crystal_sym_ops(crystal):
    """
    for a CSD crystal object
    convert symmetry operators to affine transform matrices
    """
    sym_ops = crystal.symmetry_operators  # get symmetry operators
    sym_elements = [np.eye(4) for _ in range(len(sym_ops))]
    for j in range(1, len(sym_ops)):  # convert to affine transform
        sym_elements[j][:3, :3] = np.asarray(crystal.symmetry_rotation(sym_ops[j])).reshape(3, 3)
        sym_elements[j][:3, -1] = np.asarray(crystal.symmetry_translation(sym_ops[j]))

    return sym_elements


def extract_crystal_data(identifier, crystal, reduced_crystal, unit_cell):
    """
    crystal is a csd crystal object loaded from cif file or directly from csd
    extracts key information for crystal modelling
    """
    crystal_dict = {}
    crystal_dict['identifier'] = identifier

    # extract crystal features
    crystal_dict['z_prime'] = crystal.z_prime
    crystal_dict['z_value'] = crystal.z_value
    crystal_dict['symmetry_operators'] = get_crystal_sym_ops(reduced_crystal)
    crystal_dict['symmetry_multiplicity'] = len(crystal_dict['symmetry_operators'])
    assert (crystal.z_value // crystal.z_prime) == crystal_dict['symmetry_multiplicity']
    crystal_dict['space_group_number'], crystal_dict[
        'space_group_setting'] = reduced_crystal.spacegroup_number_and_setting
    crystal_dict['lattice_a'], crystal_dict['lattice_b'], crystal_dict['lattice_c'] = np.asarray(
        reduced_crystal.cell_lengths,
        dtype=float)
    crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'], crystal_dict['lattice_gamma'] = np.asarray(
        reduced_crystal.cell_angles, dtype=float) / 180 * np.pi
    # NOTE this calls a (probably mol volume) calculation which is by far the heaviest part of this function
    # it differs from the below method by usually less than 1% but sometimes up to 5%. Molecule volume in general is not straightforward to accurately estimate
    # crystal_dict['packing_coefficient'] = crystal.packing_coefficient

    crystal_dict['fc_transform'], crystal_dict['cell_volume'] = (
        coor_trans_matrix_np('f_to_c', np.asarray(reduced_crystal.cell_lengths),
                             np.asarray(reduced_crystal.cell_angles) / 180 * np.pi,
                             return_vol=True))

    # this uses a pattern over the asymmetric unit molecule, which is sometimes different from the 'molecule' molecule
    # e.g., extra (erroneous or dubious) atoms in silly places
    # currently, we just toss such structures in the filter
    crystal_dict['unit_cell_coordinates'] = [
        np.asarray([np.asarray(atom.coordinates) for atom in component.atoms]) for component in
        unit_cell.components]
    # crystal_dict['unit_cell_fractional_coordinates'] = [np.asarray([heavy_atom.fractional_coordinates for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]
    crystal_dict['unit_cell_atomic_numbers'] = [
        np.asarray([atom.atomic_number for atom in component.atoms]) for component in
        unit_cell.components]

    # confirm packing above has correct number of components
    assert len(crystal_dict['unit_cell_coordinates']) == int(crystal_dict['symmetry_multiplicity'] * crystal_dict[
        'z_prime']), "crystal multiplicity error in unit cell packing"

    if crystal_dict[
        'space_group_number'] != 0:  # sometimes the sg number is broken, but if not, assign a consistent canonical SG symbol
        crystal_dict['space_group_symbol'] = SPACE_GROUPS[crystal_dict['space_group_number']]
    else:  # in which case, try reverse assigning the number, given the space group
        crystal_dict['space_group_number'] = sg_numbers[crystal_dict['space group symbol']]

    # somehow a C-1 in SG#2 got passed us here at some point
    assert crystal_dict['space_group_symbol'] in list(SPACE_GROUPS.values())

    return crystal_dict


def featurize_molecule(crystal, rd_mol, component_num, protonation_state='deprotonated'):
    """
    extract atom & molecule-scale features
    """

    molecule_dict = {}

    # extract a single asymmetric unit features (not necessarily the canonical unit)
    component = crystal.molecule.components[component_num]  # opt for Z': int>= 1 systems
    if protonation_state == 'protonated':
        molecule_dict['atom_coordinates'] = np.asarray([atom.coordinates for atom in component.atoms])
        molecule_dict['atom_atomic_numbers'] = np.asarray([atom.atomic_number for atom in component.atoms])
    elif protonation_state == 'deprotonated':
        molecule_dict['atom_coordinates'] = np.asarray([atom.coordinates for atom in component.heavy_atoms])
        molecule_dict['atom_atomic_numbers'] = np.asarray([atom.atomic_number for atom in component.heavy_atoms])

    if protonation_state == 'deprotonated':
        assert sum(np.asarray(molecule_dict['atom_atomic_numbers']) == 1) == 0

    '''molecule-wise features'''
    molecule_dict['molecule_fingerprint'] = fingerprint_generator.GetFingerprintAsNumPy(rd_mol)
    molecule_dict['molecule_smiles'] = Chem.MolToSmiles(rd_mol, canonical=True)

    charges = get_partial_charges(rd_mol)
    if np.all(np.isfinite(charges)):
        molecule_dict['atom_partial_charge'] = charges
    else:
        molecule_dict['atom_partial_charge'] = np.zeros_like(charges)

    return molecule_dict


def old_featurize_molecule(crystal, rd_mol, component_num, protonation_state='deprotonated'):
    """
    extract atom & molecule-scale features
    """

    molecule_dict = {}

    # extract a single asymmetric unit features (not necessarily the canonical unit)
    component = crystal.molecule.components[component_num]  # opt for Z': int>= 1 systems
    if protonation_state == 'protonated':
        molecule_dict['atom_coordinates'] = np.asarray([atom.coordinates for atom in component.atoms])
        molecule_dict['atom_atomic_numbers'] = np.asarray(
            [atom.atomic_number for atom in component.atoms])
    elif protonation_state == 'deprotonated':
        molecule_dict['atom_coordinates'] = np.asarray([atom.coordinates for atom in component.heavy_atoms])
        molecule_dict['atom_atomic_numbers'] = np.asarray(
            [atom.atomic_number for atom in component.heavy_atoms])

    # confirm mol2 file as read by rdkit agrees with CSD output
    atoms = rd_mol.GetAtoms()
    conformer = rd_mol.GetConformer()

    coords = conformer.GetPositions()
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in rd_mol.GetAtoms()])

    # confirm RDKit and CSD agree on order of atoms
    # we do this with both RDKit and CSD to double-check they agree. Probably unnecessary
    assert np.mean(np.abs(coords - molecule_dict['atom_coordinates'])) < 1e-3
    assert np.sum(np.abs(atomic_numbers - molecule_dict['atom_atomic_numbers'])) == 0

    h_donors = list(sum(rd_mol.GetSubstructMatches(HDonorSmarts, uniquify=1), ()))  # convert tuple to list
    h_acceptors = list(sum(rd_mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1), ()))

    '''atom-wise features'''
    # molecule_dict['atom_mass'] = [atom.GetMass() for atom in atoms]
    # molecule_dict['atom_is_H_bond_donor'] = np.array([1 if ind in list(h_donors) else 0 for ind in range(len(atoms))])
    # molecule_dict['atom_is_H_bond_acceptor'] = np.array([1 if ind in list(h_acceptors) else 0 for ind in range(len(atoms))])
    # molecule_dict['atom_valence'] = [atom.GetTotalValence() for atom in atoms]
    # molecule_dict['atom_vdW_radius'] = [vdw_radii_dict[number] for number in molecule_dict['atom_atomic_numbers']]
    # molecule_dict['atom_on_a_ring'] = [atom.IsInRing() for atom in atoms]
    # molecule_dict['atom_chirality'] = [atom.GetChiralTag().real for atom in atoms]
    # molecule_dict['atom_is_aromatic'] = [atom.GetIsAromatic() for atom in atoms]
    # molecule_dict['atom_degree'] = [atom.GetDegree() for atom in atoms]
    # molecule_dict['atom_electronegativity'] = [electronegativity_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
    # molecule_dict['atom_group'] = [group_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
    # molecule_dict['atom_period'] = [period_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]

    if protonation_state == 'deprotonated':
        assert sum(np.asarray(molecule_dict['atom_atomic_numbers']) == 1) == 0

    '''molecule-wise features'''
    molecule_dict['molecule_fingerprint'] = fingerprint_generator.GetFingerprintAsNumPy(rd_mol)
    # radii = rdFreeSASA.classifyAtoms(rd_mol)
    # molecule_dict['molecule_freeSASA'] = rdFreeSASA.CalcSASA(rd_mol, radii)
    # molecule_dict['molecule_mass'] = Descriptors.MolWt(rd_mol)  # includes implicit protons
    # molecule_dict['molecule_num_atoms'] = len(molecule_dict['atom_coordinates'])  # rd_mol.GetNumAtoms()
    # molecule_dict['molecule_num_rings'] = rd_mol.GetRingInfo().NumRings()
    # molecule_dict['molecule_point group'] = pointGroupAnalysis(molecule_dict['atom Z'], molecule_dict['atom coords'])  # this is also slow, approx 30% of total effort
    # molecule_dict['molecule_volume'] = AllChem.ComputeMolVolume(rd_mol)  # this is very slow - approx 50% of total effort - fill this in later from the CSD
    # molecule_dict['molecule_volume'] = component.molecular_volume  # this is much faster
    # molecule_dict['molecule_volume'] = mol_volume
    # molecule_dict['molecule_num_donors'] = len(h_donors)
    # molecule_dict['molecule_num_acceptors'] = len(h_acceptors)
    # molecule_dict['molecule_polarity'], _ = get_dipole(molecule_dict['atom_coordinates'], molecule_dict['atom_electronegativity'])
    # molecule_dict['molecule_spherical_defect'] = rdMolDescriptors.CalcAsphericity(rd_mol)
    # molecule_dict['molecule_eccentricity'] = rdMolDescriptors.CalcEccentricity(rd_mol)
    # molecule_dict['molecule_num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(rd_mol)
    # molecule_dict['molecule_planarity'] = rdMolDescriptors.CalcPBF(rd_mol)
    # molecule_dict['molecule_radius_of_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(rd_mol)
    # molecule_dict['molecule_radius'] = np.amax(np.linalg.norm(molecule_dict['atom_coordinates'] - molecule_dict['atom_coordinates'].mean(0), axis=-1))

    # for anum in range(1, 36):
    #     molecule_dict[f'molecule_{element_symbols_dict[anum]}_fraction'] = get_fraction(molecule_dict['atom_atomic_numbers'], anum)

    # for frag in Fragments.__dict__.keys():  # for all the class methods
    #     if frag[0:3] == 'fr_':  # if it's a functional group analysis methodad
    #         molecule_dict[f'molecule_{frag[3:]}_count'] = Fragments.__dict__[frag](rd_mol, countUnique=False)

    molecule_dict['molecule_smiles'] = Chem.MolToSmiles(rd_mol, canonical=True)
    # molecule_dict['molecule_chemical_formula'] = rdMolDescriptors.CalcMolFormula(rd_mol)

    # Ip, Ipm, _ = compute_principal_axes_np(np.asarray(molecule_dict['atom_coordinates']))
    # molecule_dict['molecule_principal_axes'] = Ip  # row-wise principal_axes
    # molecule_dict['molecule_principal_moment_1'] = Ipm[0]
    # molecule_dict['molecule_principal_moment_2'] = Ipm[1]
    # molecule_dict['molecule_principal_moment_3'] = Ipm[2]
    # molecule_dict['molecule_is_spherical_top'] = Ipm[0] == Ipm[1] == Ipm[2]
    # molecule_dict['molecule_is_symmetric_top'] = any([
    #     Ipm[0] != Ipm[1] == Ipm[2],
    #     Ipm[0] == Ipm[1] != Ipm[2],
    #     Ipm[0] == Ipm[2] != Ipm[1]
    # ])
    # molecule_dict['molecule_is_asymmetric_top'] = not Ipm[0] == Ipm[1] == Ipm[2]

    charges = get_partial_charges(rd_mol)
    if np.all(np.isfinite(charges)):
        molecule_dict['atom_partial_charge'] = charges
    else:
        molecule_dict['atom_partial_charge'] = np.zeros_like(charges)

    return molecule_dict


# noinspection PyUnreachableCode
def crystal_filter(crystal,
                   reduced_crystal,
                   max_heavy_atoms=1000,
                   protonation_state='deprotonated',
                   max_atomic_number=1000,
                   max_z_prime: int = 100,
                   ):
    """
    apply checks to see if this is a valid crystal to be featurized and put in the dataset
    disorder
    polymers
    no atoms
    missing sites
    wrong number of components
    non integer z prime
    zero components
    all components have atoms
    asymmetric unit and molecule have same number of atoms
    wrong number of asymmetric unit components
    failed packing
    failed to be recognized by RDKit
    """
    # crystal checks
    if crystal.has_disorder:
        return False, None, None, "Disordered"
    if crystal.molecule.is_polymeric:
        return False, None, None, "Polymer"
    if len(crystal.molecule.atoms) == 0:
        return False, None, None, "No atoms"
    if not crystal.molecule.all_atoms_have_sites:
        return False, None, None, "Missing sites"
    # TODO note the CSD convention for z_prime is such that it doesn't count cocrystals as multiple molecules
    # therefore the below two filters always kill cocrystals, which show up with more components than z_prime
    # consider relaxing / reprocessing in the future. Though we must be careful

    if crystal.z_prime < 1:
        return False, None, None, "Z' < 1"
    if crystal.z_prime > max_z_prime:
        return False, None, None, "Z' > max_z_prime"
    if int(crystal.z_prime) != crystal.z_prime:  # integer z-prime only
        return False, None, None, "Z' not integer"
    # if int(crystal.z_prime) != 1: # Z'=1 only
    #     return False, None, None
    if len(reduced_crystal.molecule.components) == 0:
        return False, None, None, "No components"
    if any([len(component.atoms) == 0 for component in crystal.molecule.components]):
        return False, None, None, "Not all components have atoms"
    if len(reduced_crystal.asymmetric_unit_molecule.heavy_atoms) != len(
            crystal.asymmetric_unit_molecule.heavy_atoms):  # could make this done Z'-by-Z'
        return False, None, None, "Aunit and mol heavy atoms disagree"
    if len(reduced_crystal.molecule.components) != crystal.z_prime:
        return False, None, None, "Z' != mol components"
    if len(reduced_crystal.asymmetric_unit_molecule.components) != crystal.z_prime:  # can relax this if we build our own reference cells
        return False, None, None, "Z' != components"

    try:  # some entries have invalid SG information, and this will fail
        _ = reduced_crystal.spacegroup_number_and_setting
    except RuntimeError:
        return False, None, None, "Invalid SG"

    try:  # sometimes packing fails
        unit_cell = reduced_crystal.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='CentroidIncluded')
    except RuntimeError:
        return False, None, None, "Packing failed"

    # crystal has to have the right number of molecules
    if len(unit_cell.components) != int(len(crystal.symmetry_operators) * crystal.z_prime):
        return False, None, None, "Components != Z"

    for zp in range(int(crystal.z_prime)):  # confirm unit cell symmetry images have each the right number of atoms
        l1 = len(reduced_crystal.molecule.components[zp].heavy_atoms)
        mult = len(reduced_crystal.symmetry_operators)
        for z in range(mult):
            l2 = len(unit_cell.components[zp * mult + z].heavy_atoms)
            if l1 != l2:
                return False, None, None, "Symmetry images unequal sizes"

    # molecule check via RDKit. If RDKit doesn't see it as a real molecule, don't accept it to the dataset.
    rd_mols = []
    for component in reduced_crystal.molecule.components:
        mol = Chem.MolFromMol2Block(component.to_string('mol2'),
                                    sanitize=True,
                                    removeHs=True if protonation_state == 'deprotonated' else False)
        if mol is None:
            return False, None, None, "Failed RDKit embedding"
        else:
            try:
                # number of heavy atoms
                num_heavy_atoms = mol.GetNumHeavyAtoms()
                if num_heavy_atoms > max_heavy_atoms:
                    return False, None, None, "Too many heavy atoms"

                # largest atomic number
                atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
                if np.max(atomic_numbers) > max_atomic_number:
                    return False, None, None, "Too many atoms"

                if protonation_state == 'protonated':  # this block may still need work
                    mol_formula = rdMolDescriptors.CalcMolFormula(mol)

                    # correct number of hydrogens
                    if 'H' in mol_formula:
                        H_count = mol.GetNumAtoms() - mol.GetNumHeavyAtoms()

                        num_expected_hydrogens = H_count
                        num_actual_hydrogens = np.sum(atomic_numbers == 1)

                        if not num_expected_hydrogens == num_actual_hydrogens:
                            return False, None, None, "Wrong number of hydrogens"

                if protonation_state == 'deprotonated':
                    rd_mols.append(Chem.RemoveAllHs(mol))
                else:
                    rd_mols.append(mol)

            except Exception as e:
                return False, None, None, f"Unknown RDKit error {str(e)}"

    # do this last as it's expensive
    for component in reduced_crystal.molecule.components:  # check for overlapping atoms or unconnected fragments
        coords = np.asarray([np.asarray(heavy_atom.coordinates) for heavy_atom in component.heavy_atoms])
        distmat = cdist(coords, coords)
        np.fill_diagonal(distmat, 100)
        min_interatomic_distance = distmat.min(1)
        # if any atoms are too close, or have no neighbors, in a very generous range (3 angstroms and 0.9 angstroms)
        if any(min_interatomic_distance > 3) or any(min_interatomic_distance < 0.9):
            return False, None, None, "Unphysical molecules"

    return True, unit_cell, rd_mols, None


def chunkify_path_list(cifs_list, n_chunks, do_shuffle=True):
    n_chunks = min(n_chunks, len(cifs_list))
    print(f"Breaking dataset into {n_chunks} chunks")
    chunks_list = chunkify(cifs_list, n_chunks)
    chunk_inds = [i for i in range(len(chunks_list))]
    start_ind, stop_ind = 0, len(chunks_list)
    if do_shuffle:
        shuffle(chunk_inds)  # optionally do it in random order
    chunks_list = [chunks_list[ind] for ind in chunk_inds]
    return chunk_inds, chunks_list, start_ind, stop_ind


def extract_custom_cif_data(cif_path, crystal_dict):
    """
    for extracting information from Nikos' old cif format
    customize for any future usages
    """
    with open(cif_path, 'r') as f:
        text = f.read()

        if 'zzp' in text:
            lines = text.split('\n')
            for line_ind, line in enumerate(lines):
                if 'zzp' in line:
                    break
            prop_line = lines[line_ind + 2]
            crystal_dict['zzp_cost'] = prop_line.split()[0]
            crystal_dict['contact_overlap_cost'] = prop_line.split()[-1]

    return crystal_dict


def featurize_xyz_molecule(molecule_dict, text):
    """
    Featurize a molecule directly from an xyz file
    """
    atoms_block_text = text[2:molecule_dict['num_atoms'] + 2]
    atom_types = np.zeros(molecule_dict['num_atoms'], dtype=np.int_)
    atom_coords = np.zeros((molecule_dict['num_atoms'], 3))
    atom_charges = np.zeros(molecule_dict['num_atoms'])
    for ind, line in enumerate(atoms_block_text):
        line = line.split('\t')
        atom_types[ind] = int(ATOMIC_NUMBERS[line[0]])
        atom_coords[ind, :] = float(line[1]), float(line[2]), float(line[3])
        # atom_charges[ind] = float(line[4])
    molecule_dict['atom_coordinates'] = atom_coords
    molecule_dict['atom_atomic_numbers'] = atom_types
    molecule_dict['molecule_smiles'] = text[-3].split('\t')[0]
    molecule_dict['partial_charges'] = custom_molecule_partial_charges(atom_types, atom_coords)
    return molecule_dict


def custom_molecule_partial_charges(types, coords):
    """
    to ensure proper indexing, we have to set up a custom rdkit mol object
    then rdkit will compute the charges for us
    """
    mol = Chem.RWMol()
    for ind, atom in enumerate(types):
        idx = mol.AddAtom(Chem.Atom(int(atom)))

    conf = Chem.Conformer(len(types))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))

    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)  # we don't have to add implicit Hs as this whole workflow is protonated

    return get_partial_charges(mol)


def get_qm9_properties(text):
    """extract the 15 QM9 scalar properties from the standard xyz format"""
    props = text[1].split('\t')
    molecule_dict = {
        "num_atoms": int(text[0]),
        "identifier": int(props[0].split()[1]),
        "rotational_constant_a": float(props[1]),
        "rotational_constant_b": float(props[2]),
        "rotational_constant_c": float(props[3]),
        "dipole_moment": float(props[4]),
        "isotropic_polarizability": float(props[5]),
        "HOMO_energy": float(props[6]),
        "LUMO_energy": float(props[7]),
        "gap_energy": float(props[8]),
        "el_spatial_extent": float(props[9]),
        "zpv_energy": float(props[10]),
        "internal_energy_0": float(props[11]),
        "internal_energy_STP": float(props[12]),
        "enthalpy_STP": float(props[13]),
        "free_energy_STP": float(props[14]),
        "heat_capacity_STP": float(props[15]),
    }
    return molecule_dict, props


def canonical_atom_order(pos, z):
    centroid = np.mean(pos, axis=0)
    dists = np.linalg.norm(pos - centroid, axis=1)
    atom_types = np.unique(z)

    indices = []
    for type in atom_types:
        elems = np.argwhere(dists[z == type])
        indices.extend(elems[np.argsort(dists[elems], descending=True)])

    return np.array(indices)


def instantiate_crystal(aunit_centroid, aunit_handedness,
                        aunit_orientation, max_z_prime,
                        mols, rebuild_batch, z_prime):
    if z_prime > 1:
        # instantiate Z'>1 crystal object
        mol = MolData(
            z=rebuild_batch.z,
            pos=rebuild_batch.pos,
            x=rebuild_batch.x,
            smiles='|'.join(rebuild_batch.smiles),
            fingerprint=rebuild_batch.fingerprint.sum(0, keepdim=True),
            do_mol_analysis=False,  # combine manually below
        )
        mol.mass = torch.stack([m.mass for m in mols]).sum()
        mol.mol_volume = torch.stack([m.mol_volume for m in mols]).sum()
        mol.radius = torch.stack([m.radius for m in mols]).sum()

        zp1_crystal = MolCrystalData(
            molecule=mol,
            sg_ind=rebuild_batch.sg_ind[0],
            cell_lengths=rebuild_batch.cell_lengths[0],
            cell_angles=rebuild_batch.cell_angles[0],
            aunit_orientation=aunit_orientation.flatten(),
            aunit_centroid=aunit_centroid.flatten(),
            aunit_handedness=aunit_handedness.flatten(),
            identifier=rebuild_batch.identifier[0],
            nonstandard_symmetry=any(rebuild_batch.nonstandard_symmetry),
            symmetry_operators=rebuild_batch.symmetry_operators[0],
            z_prime=torch.ones(1, dtype=torch.long) * z_prime,
            do_box_analysis=True,
            cocrystal=False,
            max_z_prime=max_z_prime,
            mol_ind=torch.arange(z_prime, dtype=torch.long).repeat_interleave(
                torch.stack([m.num_atoms for m in mols])),
        )

    else:
        mol = MolData(
            z=rebuild_batch.z,
            pos=rebuild_batch.pos,
            x=rebuild_batch.x,
            smiles=rebuild_batch.smiles[0],
            fingerprint=rebuild_batch.fingerprint,
            do_mol_analysis=True,  # combine manually below
        )
        zp1_crystal = MolCrystalData(
            molecule=mol,
            sg_ind=rebuild_batch.sg_ind[0],
            cell_lengths=rebuild_batch.cell_lengths[0],
            cell_angles=rebuild_batch.cell_angles[0],
            aunit_orientation=aunit_orientation.flatten(),
            aunit_centroid=aunit_centroid.flatten(),
            aunit_handedness=aunit_handedness.flatten(),
            identifier=rebuild_batch.identifier[0],
            nonstandard_symmetry=any(rebuild_batch.nonstandard_symmetry),
            symmetry_operators=rebuild_batch.symmetry_operators[0],
            z_prime=torch.ones(1, dtype=torch.long),
            do_box_analysis=True,
            cocrystal=False,
            max_z_prime=max_z_prime,
            mol_ind=torch.zeros(len(mol.z), dtype=torch.long),
        )
    return zp1_crystal


def init_zp1_crystals(crystal_dict, mols, sym_ops, sym_ops_are_standard):
    zp1_crystals = []
    for mol in mols:
        cell_lengths = torch.tensor(
            np.stack([
                crystal_dict['lattice_a'], crystal_dict['lattice_b'], crystal_dict['lattice_c']
            ]), dtype=torch.float32)
        cell_angles = torch.tensor(
            np.stack([
                crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'],
                crystal_dict['lattice_gamma']
            ]), dtype=torch.float32)

        crystal = MolCrystalData(
            molecule=mol,
            sg_ind=int(crystal_dict['space_group_number']),
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            aunit_orientation=torch.zeros(3, dtype=torch.float32),  # dummy
            aunit_centroid=torch.zeros(3, dtype=torch.float32),  # dummy
            aunit_handedness=torch.zeros(1, dtype=torch.bool),  # dummy
            identifier=crystal_dict['identifier'],
            nonstandard_symmetry=not sym_ops_are_standard,
            symmetry_operators=sym_ops,
            z_prime=torch.ones(1, dtype=torch.long),
            do_box_analysis=True,
            cocrystal=False,
            max_z_prime=1
        )
        zp1_crystals.append(crystal)
    return zp1_crystals


def extract_zp1_pose_info(molwise_ucell_coords, z_prime, zp1_crystals):
    crystal_batch = collate_data_list(zp1_crystals, max_z_prime=1)
    if len(zp1_crystals) > 1:
        crystal_batch.unit_cell_pos = torch.tensor(np.concatenate(molwise_ucell_coords),
                                                   dtype=torch.float32)
        crystal_batch.unit_cell_batch = torch.arange(z_prime, dtype=torch.float32).repeat_interleave(
            crystal_batch.num_atoms * crystal_batch.sym_mult)

    else:
        crystal_batch.unit_cell_pos = torch.tensor(np.concatenate(molwise_ucell_coords),
                                                   dtype=torch.float32)
        crystal_batch.unit_cell_batch = torch.zeros(len(crystal_batch.unit_cell_pos), dtype=torch.float32)
    aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()
    return aunit_centroid, aunit_handedness, aunit_orientation, crystal_batch, is_well_defined, pos


def typed_pair_fingerprint(mol):
    n = mol.num_atoms
    d = torch.cdist(mol.pos, mol.pos)
    z = mol.z
    iu = torch.triu_indices(n, n, offset=1)
    zi, zj = z[iu[0]], z[iu[1]]
    a = torch.minimum(zi, zj)  # unordered pair label
    b = torch.maximum(zi, zj)
    dist = d[iu[0], iu[1]]
    # lexsort by (a, b, distance) — pack into one key
    key = a.long() * 200_000 + b.long() * 1_000 + (dist * 100).long()
    order = key.argsort()
    return torch.stack([a[order].float(), b[order].float(), dist[order]], dim=1)


def cocrystal_check(mols):
    # molecule size
    if not all([mol.num_atoms == mols[0].num_atoms for mol in mols]):
        return False

    # check for the same molecule
    fp = mols[0].smiles
    for m in mols[1:]:
        if m.smiles != fp:  # not np.all(fp == m['molecule_fingerprint']):
            return False
    #
    # # same molecular graph — try to align atom orderings to mols[0]
    # for m in mols:
    #     perm = torch.argsort(m.z, descending=True, stable=True)
    #     m.z = m.z[perm]
    #     m.pos = m.pos[perm]
    #     m.x = m.x[perm]
    #
    # # molecule atom order
    # if not torch.all(torch.stack([mol.z for mol in mols]).diff(dim=0) == 0):
    #     return False

    f1 = typed_pair_fingerprint(mols[0])
    for mol in mols[1:]:
        if not torch.allclose(f1, typed_pair_fingerprint(mol), atol=1e-1):
            return False

    # # check if they are identical conformers (this is only a rough method)
    # d1 = torch.cdist(mols[0].pos, mols[0].pos)
    # f1 = d1.mean(1).sort(dim=0).values
    # for mol in mols:
    #     f2 = torch.cdist(mol.pos, mol.pos).mean(1).sort(dim=0).values
    #     if not torch.allclose(f1, f2, atol=1e-1):
    #         return False

    return True


def overwrite_conformer_to_zp(mols, rebuild_batch, z_prime):
    mol = mols[0]
    for zp in range(1, z_prime):
        mols[zp] = mol.clone()
    # we then have to confirm this doesn't overly distort crystal construction
    crystal_pos = torch.cat([m.pos for m in mols])
    crystal_z = torch.cat([m.z for m in mols])
    crystal_x = torch.cat([m.x for m in mols])
    aligned_batch = rebuild_batch.clone()
    aligned_batch.pos = crystal_pos
    aligned_batch.z = crystal_z
    aligned_batch.x = crystal_x
    return aligned_batch


def check_zp_mol_alignment(aligned_batch, rebuild_batch):
    same_atom_types = (torch.bincount(rebuild_batch.z) == torch.bincount(aligned_batch.z)).all()
    aligned_batch.pose_aunit()
    rebuild_batch.pose_aunit()
    distmat = torch.cdist(aligned_batch.pos, rebuild_batch.pos)
    furthest_nn = distmat.amin(0).amax()
    e1 = aligned_batch.analyze(['elj'])['elj']
    e2 = rebuild_batch.analyze(['elj'])['elj']
    diff = (e1 - e2).abs() / (e2 + 1e-3).abs()
    failed = (not same_atom_types) or (furthest_nn > 0.5) or (diff.amax() > 0.01)
    return not failed
