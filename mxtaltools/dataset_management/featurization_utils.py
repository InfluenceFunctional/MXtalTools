import numpy as np
from rdkit import Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, rdFreeSASA
from scipy.spatial.distance import cdist

from mxtaltools.common.geometry_calculations import compute_principal_axes_np, coor_trans_matrix
from mxtaltools.constants.atom_properties import ELECTRONEGATIVITY, PERIOD, GROUP, VDW_RADII, SYMBOLS
from mxtaltools.constants.space_group_info import SPACE_GROUPS
from mxtaltools.dataset_management.utils import get_fraction

'''setup fingerprint generator'''
fingerprint_generator = AllChem.GetMorganGenerator(radius=2, includeChirality=False)

'''set up some constants'''
vdw_radii_dict = VDW_RADII
element_symbols_dict = SYMBOLS
electronegativity_dict = ELECTRONEGATIVITY
period_dict = PERIOD
group_dict = GROUP

for key in electronegativity_dict.keys():
    if electronegativity_dict[key] is None:
        electronegativity_dict[key] = 0

HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
HAcceptorSmarts = Chem.MolFromSmarts(
    '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
    '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
    '$([nH0,o,s;+0])]')

sg_numbers = {}
for i in range(1, 231):
    sg_numbers[SPACE_GROUPS[i]] = i


def chunkify(lst: list, n: int):
    """
    break up a list into n chunks of equal size (up to last chunk)
    """
    return [lst[ind::n] for ind in range(n)]


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


def extract_crystal_data(identifier, crystal, unit_cell):
    """
    crystal is a csd crystal object loaded from cif file or directly from csd
    extracts key information for crystal modelling
    """
    crystal_dict = {}
    crystal_dict['identifier'] = identifier

    # extract crystal features
    crystal_dict['z_prime'] = crystal.z_prime
    crystal_dict['z_value'] = crystal.z_value
    crystal_dict['symmetry_operators'] = get_crystal_sym_ops(crystal)
    crystal_dict['symmetry_operator_symbols'] = crystal.symmetry_operators
    crystal_dict['symmetry_multiplicity'] = len(crystal_dict['symmetry_operators'])
    assert (crystal.z_value // crystal.z_prime) == crystal_dict['symmetry_multiplicity']
    crystal_dict['space_group_number'], crystal_dict['space_group_setting'] = crystal.spacegroup_number_and_setting
    crystal_dict['space_group_symbol'] = crystal.spacegroup_symbol
    crystal_dict['system'] = crystal.crystal_system
    crystal_dict['lattice_a'], crystal_dict['lattice_b'], crystal_dict['lattice_c'] = np.asarray(crystal.cell_lengths, dtype=float)
    crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'], crystal_dict['lattice_gamma'] = np.asarray(crystal.cell_angles, dtype=float) / 180 * np.pi
    # NOTE this calls a (probably mol volume) calculation which is by far the heaviest part of this function
    # it differs from the below method by usually less than 1% but sometimes up to 5%. Molecule volume in general is not straightforward to accurately estimate
    # crystal_dict['packing_coefficient'] = crystal.packing_coefficient
    crystal_dict['is_organic'] = crystal.molecule.is_organic
    crystal_dict['is_organometallic'] = crystal.molecule.is_organometallic

    crystal_dict['fc_transform'], crystal_dict['cell_volume'] = (
        coor_trans_matrix('f_to_c', np.asarray(crystal.cell_lengths), np.asarray(crystal.cell_angles) / 180 * np.pi, return_vol=True))
    crystal_dict['cf_transform'] = (
        coor_trans_matrix('c_to_f', np.asarray(crystal.cell_lengths), np.asarray(crystal.cell_angles) / 180 * np.pi, return_vol=False))
    crystal_dict['density'] = crystal.calculated_density
    crystal_dict['reduced_volume'] = crystal_dict['cell_volume'] / crystal_dict['symmetry_multiplicity']
    mol_volumes = [component.molecular_volume for component in crystal.molecule.components]
    crystal_dict['packing_coefficient'] = (sum(mol_volumes) * crystal.z_value / crystal.z_prime / crystal_dict['cell_volume'])

    # extract a complete unit cell. Leave outputs as lists, since they may have different lengths for different molecules
    # this uses a pattern over the asymmetric unit molecule, which is sometimes different from the 'molecule' molecule
    # e.g., extra (erroneous or dubous) atoms in silly places
    # TODO do this with our code using the 'molecule' as a basis, and ignore possibly erroneous asymmetric units
    # currently, we just toss such structures in the filter
    crystal_dict['unit_cell_coordinates'] = [np.asarray([np.asarray(heavy_atom.coordinates) for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]
    crystal_dict['unit_cell_fractional_coordinates'] = [np.asarray([heavy_atom.fractional_coordinates for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]
    crystal_dict['unit_cell_atomic_numbers'] = [np.asarray([heavy_atom.atomic_number for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]
    # confirm packing above has correct number of components
    assert len(crystal_dict['unit_cell_coordinates']) == int(crystal_dict['symmetry_multiplicity'] * crystal_dict['z_prime']), "crystal multiplicity error in unit cell packing"

    if crystal_dict['space_group_number'] != 0:  # sometimes the sg number is broken, but if not, assign a consistent canonical SG symbol
        crystal_dict['space_group_symbol'] = SPACE_GROUPS[crystal_dict['space_group_number']]
    else:  # in which case, try reverse assigning the number, given the space group
        crystal_dict['space_group_number'] = sg_numbers[crystal_dict['space group symbol']]

    # somehow a C-1 in SG#2 got passed us here at some point
    assert crystal_dict['space_group_symbol'] in list(SPACE_GROUPS.values())

    return crystal_dict, mol_volumes


def featurize_molecule(crystal, rd_mol, mol_volume, component_num):
    """
    extract atom & molecule-scale features
    """

    molecule_dict = {}

    # extract a single asymmetric unit features (not necessarily the canonical unit)
    component = crystal.molecule.components[component_num]  # opt for Z': int>= 1 systems
    molecule_dict['atom_coordinates'] = np.asarray([heavy_atom.coordinates for heavy_atom in component.heavy_atoms])
    molecule_dict['atom_fractional_coordinates'] = np.asarray([heavy_atom.fractional_coordinates for heavy_atom in component.heavy_atoms])
    molecule_dict['atom_atomic_numbers'] = np.asarray([heavy_atom.atomic_number for heavy_atom in component.heavy_atoms])

    atoms = rd_mol.GetAtoms()
    conformer = rd_mol.GetConformer()

    coords = conformer.GetPositions()
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in rd_mol.GetAtoms()])

    # confirm RDKit and CSD agree on order of atoms
    assert np.mean(np.abs(coords - molecule_dict['atom_coordinates'])) < 1e-3  # we do this with both RDKit and CSD to double-check they agree. Probably unnecessary
    assert np.sum(np.abs(atomic_numbers - molecule_dict['atom_atomic_numbers'])) == 0

    h_donors = list(sum(rd_mol.GetSubstructMatches(HDonorSmarts, uniquify=1), ()))  # convert tuple to list
    h_acceptors = list(sum(rd_mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1), ()))

    '''atom-wise features'''
    molecule_dict['atom_mass'] = [atom.GetMass() for atom in atoms]
    molecule_dict['atom_is_H_bond_donor'] = [1 if ind in list(h_donors) else 0 for ind in range(len(atoms))]
    molecule_dict['atom_is_H_bond_acceptor'] = [1 if ind in list(h_acceptors) else 0 for ind in range(len(atoms))]
    molecule_dict['atom_valence'] = [atom.GetTotalValence() for atom in atoms]
    molecule_dict['atom_vdW_radius'] = [vdw_radii_dict[number] for number in molecule_dict['atom_atomic_numbers']]
    molecule_dict['atom_on_a_ring'] = [atom.IsInRing() for atom in atoms]
    molecule_dict['atom_chirality'] = [atom.GetChiralTag().real for atom in atoms]
    molecule_dict['atom_is_aromatic'] = [atom.GetIsAromatic() for atom in atoms]
    molecule_dict['atom_degree'] = [atom.GetDegree() for atom in atoms]
    molecule_dict['atom_electronegativity'] = [electronegativity_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
    molecule_dict['atom_group'] = [group_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]
    molecule_dict['atom_period'] = [period_dict[atom] for atom in molecule_dict['atom_atomic_numbers']]

    assert sum(np.asarray(molecule_dict['atom_atomic_numbers']) == 1) == 0  # positively assert there are absolutely no protons in the dataset

    '''molecule-wise features'''
    molecule_dict['molecule_fingerprint'] = fingerprint_generator.GetFingerprintAsNumPy(rd_mol)
    radii = rdFreeSASA.classifyAtoms(rd_mol)
    molecule_dict['molecule_freeSASA'] = rdFreeSASA.CalcSASA(rd_mol, radii)
    molecule_dict['molecule_mass'] = Descriptors.MolWt(rd_mol)  # includes implicit protons
    molecule_dict['molecule_num_atoms'] = len(molecule_dict['atom_coordinates'])  # rd_mol.GetNumAtoms()
    molecule_dict['molecule_num_rings'] = rd_mol.GetRingInfo().NumRings()
    # molecule_dict['molecule_point group'] = pointGroupAnalysis(molecule_dict['atom Z'], molecule_dict['atom coords'])  # this is also slow, approx 30% of total effort
    # molecule_dict['molecule_volume'] = AllChem.ComputeMolVolume(rd_mol)  # this is very slow - approx 50% of total effort - fill this in later from the CSD
    # molecule_dict['molecule_volume'] = component.molecular_volume  # this is much faster
    molecule_dict['molecule_volume'] = mol_volume
    molecule_dict['molecule_num_donors'] = len(h_donors)
    molecule_dict['molecule_num_acceptors'] = len(h_acceptors)
    molecule_dict['molecule_polarity'], _ = get_dipole(molecule_dict['atom_coordinates'], molecule_dict['atom_electronegativity'])
    molecule_dict['molecule_spherical_defect'] = rdMolDescriptors.CalcAsphericity(rd_mol)
    molecule_dict['molecule_eccentricity'] = rdMolDescriptors.CalcEccentricity(rd_mol)
    molecule_dict['molecule_num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(rd_mol)
    molecule_dict['molecule_planarity'] = rdMolDescriptors.CalcPBF(rd_mol)
    molecule_dict['molecule_radius_of_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(rd_mol)
    molecule_dict['molecule_radius'] = np.amax(np.linalg.norm(molecule_dict['atom_coordinates'] - molecule_dict['atom_coordinates'].mean(0), axis=-1))

    for anum in range(1, 36):
        molecule_dict[f'molecule_{element_symbols_dict[anum]}_fraction'] = get_fraction(molecule_dict['atom_atomic_numbers'], anum)

    for frag in Fragments.__dict__.keys():  # for all the class methods
        if frag[0:3] == 'fr_':  # if it's a functional group analysis methodad
            molecule_dict[f'molecule_{frag[3:]}_count'] = Fragments.__dict__[frag](rd_mol, countUnique=False)

    molecule_dict['molecule_smiles'] = Chem.MolToSmiles(rd_mol)
    molecule_dict['molecule_chemical_formula'] = rdMolDescriptors.CalcMolFormula(rd_mol)

    Ip, Ipm, _ = compute_principal_axes_np(np.asarray(molecule_dict['atom_coordinates']))  # rdMolTransforms.ComputePrincipalAxesAndMoments(rd_mol.GetConformer(), ignoreHs=False) # this does it column-wise
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

    return molecule_dict


def crystal_filter(crystal):
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
    passed_crystal_checks = True
    passed_molecule_checks = True
    # crystal checks  # I believe one at a time like this should be faster than checking all of them
    if crystal.has_disorder:
        return False, None, None
    if crystal.molecule.is_polymeric:
        return False, None, None
    if len(crystal.molecule.atoms) == 0:
        return False, None, None
    if not crystal.molecule.all_atoms_have_sites:
        return False, None, None
    if len(crystal.molecule.components) != crystal.z_prime:
        return False, None, None
    if crystal.z_prime < 1:
        return False, None, None
    if int(crystal.z_prime) != crystal.z_prime:  # integer z-prime only
        return False, None, None
    if len(crystal.molecule.components) == 0:
        return False, None, None
    if any([len(component.atoms) == 0 for component in crystal.molecule.components]):
        return False, None, None
    if len(crystal.asymmetric_unit_molecule.heavy_atoms) != len(crystal.molecule.heavy_atoms):  # could make this done Z'-by-Z'
        return False, None, None
    if len(crystal.asymmetric_unit_molecule.components) != crystal.z_prime:  # can relax this if we build our own reference cells
        return False, None, None

    try:  # some entries have invalid SG information
        _ = crystal.spacegroup_number_and_setting
    except RuntimeError:
        return False, None, None

    try:
        unit_cell = crystal.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='CentroidIncluded')
    except RuntimeError:  # sometimes packing fails
        return False, None, None

    if len(unit_cell.components) != int(len(crystal.symmetry_operators) * crystal.z_prime):
        return False, None, None

    for zp in range(int(crystal.z_prime)):  # confirm unit cell symmetry images have each the right number of atoms
        l1 = len(crystal.molecule.components[zp].heavy_atoms)
        mult = len(crystal.symmetry_operators)
        for z in range(mult):
            l2 = len(unit_cell.components[zp * mult + z].heavy_atoms)
            if l1 != l2:
                return False, None, None

    # molecule check via RDKit. If RDKit doesn't see it as a real molecule, don't accept it to the dataset.
    rd_mols = []
    for component in crystal.molecule.components:
        mol = Chem.MolFromMol2Block(component.to_string('mol2'), sanitize=True, removeHs=True)
        try:
            rd_mols.append(Chem.RemoveAllHs(mol))
        except:  # todo add exception type
            passed_molecule_checks = False
        if mol is None:
            passed_molecule_checks = False

    if not passed_molecule_checks:
        pass  # print(f'{crystal.identifier} failed molecule checks')

    for component in crystal.molecule.components:  # check for overlapping atoms or unconnected fragments
        coords = np.asarray([np.asarray(heavy_atom.coordinates) for heavy_atom in component.heavy_atoms])
        distmat = cdist(coords, coords) + np.eye(len(coords)) * 100
        min_interatomic_distance = distmat.min(1)
        # if any atoms are too close, or have no neighbors, in a very generous range (3 angstroms and 0.9 angstroms)
        if any(min_interatomic_distance > 3) or any(min_interatomic_distance < 0.9):
            return False, None, None

    return all([passed_molecule_checks, passed_crystal_checks]), unit_cell, rd_mols
