import numpy as np
import rdkit.Chem as Chem
from rdkit import Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, rdFreeSASA
from mendeleev import element as element_table
from constants.space_group_info import SPACE_GROUPS

from crystal_building.coordinate_transformations import coor_trans_matrix

'''setup fingerprint generator'''
fingerprint_generator = AllChem.GetMorganGenerator(radius=2, includeChirality=False)

'''set up some constants'''
periodic_table = Chem.GetPeriodicTable()
vdw_radii = {}
element_symbols = {}
for i in range(1, 119):
    vdw_radii[str(i)] = periodic_table.GetRvdw(i)
    element_symbols[str(i)] = periodic_table.GetElementSymbol(i)

electronegativity_dict = {}
period_dict = {}
group_dict = {}
for i in range(1, 101):  # this is weirdly slow
    electronegativity_dict[i] = element_table(i).electronegativity('pauling')
    period_dict[i] = element_table(i).period
    group_dict[i] = element_table(i).group_id
    if group_dict[i] is None:
        group_dict[i] = 19  # assign F-block groups to unique class

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
    return [lst[i::n] for i in range(n)]

def compute_Ip_handedness(Ip):
    """
    determine the right or left handedness from the cross products of principal inertial axes
    np.array or torch.tensor input, single or multiple samples
    """
    if Ip.ndim == 2:
        return np.sign(np.dot(Ip[0], np.cross(Ip[1], Ip[2])).sum())
    elif Ip.ndim == 3:
        return np.sign(np.dot(Ip[:, 0], np.cross(Ip[:, 1], Ip[:, 2], axis=1).T).sum(1))


def compute_principal_axes_np(coords):
    """
    compute the principal axes for a given set of particle coordinates, ignoring particle mass
    use our overlap rules to ensure a fixed direction for all axes under almost all circumstances
    """  # todo harmonize with torch version - currently disagrees ~0.5% of the time
    points = coords - coords.mean(0)

    x, y, z = points.T
    Ixx = np.sum((y ** 2 + z ** 2))
    Iyy = np.sum((x ** 2 + z ** 2))
    Izz = np.sum((x ** 2 + y ** 2))
    Ixy = -np.sum(x * y)
    Iyz = -np.sum(y * z)
    Ixz = -np.sum(x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
    Ipm, Ip = np.linalg.eig(I)  # principal inertial tensor
    Ipm, Ip = np.real(Ipm), np.real(Ip)
    sort_inds = np.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to the farthest atom
    dists = np.linalg.norm(points, axis=1)
    max_ind = np.argmax(dists)
    max_equivs = np.argwhere(np.round(dists, 8) == np.round(dists[max_ind], 8))[:, 0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(np.amin(max_equivs))
    direction = points[max_ind]
    direction = np.divide(direction, np.linalg.norm(direction))
    overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
    signs = np.sign(overlaps)  # returns zero for zero overlap, but we want it to default to +1 in this case
    signs[signs == 0] = 1

    Ip = (Ip.T * signs).T  # if the vectors have negative overlap, flip the direction
    if np.any(np.abs(overlaps) < 1e-3):  # if any overlaps are vanishing, determine the direction via the RHR (if two overlaps are vanishing, this will not work)
        # align the 'good' vectors
        fix_ind = np.argmin(np.abs(overlaps))  # vector with vanishing overlap
        if compute_Ip_handedness(Ip) < 0:  # make sure result is right handed
            Ip[fix_ind] = -Ip[fix_ind]

    return Ip, Ipm, I


def get_fraction(atomic_numbers, target):
    return np.sum(atomic_numbers == target) / len(atomic_numbers)


def get_dipole(coords, charges):
    center_of_geometry = np.average(np.asarray(coords), axis=0)
    center_of_charge = np.average(np.asarray(coords), weights=charges, axis=0)
    return np.linalg.norm(center_of_charge - center_of_geometry), center_of_geometry


def get_crystal_sym_ops(crystal):
    """
    for a CSD crystal object
    convert symmetry operators to affine transform matrices
    """
    sym_ops = crystal.symmetry_operators  # get symmetry operators
    sym_elements = [np.eye(4) for m in range(len(sym_ops))]
    for j in range(1, len(sym_ops)):  # convert to affine transform
        sym_elements[j][:3, :3] = np.asarray(crystal.symmetry_rotation(sym_ops[j])).reshape(3, 3)
        sym_elements[j][:3, -1] = np.asarray(crystal.symmetry_translation(sym_ops[j]))

    return sym_elements


def extract_crystal_data(crystal, unit_cell):
    """
    crystal is a csd python crystal object loaded from cif file or directly from csd
    extracts key information
    """

    crystal_dict = {}
    crystal_dict['identifier'] = crystal.identifier

    # extract crystal features
    crystal_dict['z_prime'] = crystal.z_prime
    crystal_dict['z_value'] = crystal.z_value
    crystal_dict['symmetry_operators'] = get_crystal_sym_ops(crystal)
    crystal_dict['symmetry_multiplicity'] = len(crystal_dict['symmetry_operators'])
    crystal_dict['space_group_number'], crystal_dict['space_group_setting'] = crystal.spacegroup_number_and_setting
    crystal_dict['space_group_symbol'] = crystal.spacegroup_symbol
    crystal_dict['system'] = crystal.crystal_system
    crystal_dict['lattice_a'], crystal_dict['lattice_a'], crystal_dict['lattice_a'] = np.asarray(crystal.cell_lengths, dtype=float)
    crystal_dict['lattice_alpha'], crystal_dict['lattice_beta'], crystal_dict['lattice_gamma'] = np.asarray(crystal.cell_angles, dtype=float) / 180 * np.pi
    # NOTE this calls a (probably mol volume) calculation which is by far the heaviest part of this function - but it
    # differs from the below method by usually less than 1% but sometimes up to 5%
    #crystal_dict['packing_coefficient'] = crystal.packing_coefficient  # we do it here and back out the implied volume later, as this is much faster than the RDKit method
    crystal_dict['is_organic'] = crystal.molecule.is_organic
    crystal_dict['is_organometallic'] = crystal.molecule.is_organometallic

    crystal_dict['fc_transform'], crystal_dict['cell_volume'] = coor_trans_matrix('f_to_c', np.asarray(crystal.cell_lengths), np.asarray(crystal.cell_angles) / 180 * np.pi, return_vol=True)
    crystal_dict['cf_transform'] = coor_trans_matrix('c_to_f', np.asarray(crystal.cell_lengths), np.asarray(crystal.cell_angles) / 180 * np.pi)
    crystal_dict['density'] = crystal.calculated_density
    mol_volumes = [component.molecular_volume for component in crystal.molecule.components]

    crystal_dict['packing_coefficient'] = (sum(mol_volumes) * crystal.z_value / crystal.z_prime / crystal_dict['cell_volume'])

    # extract a complete unit cell. Leave outputs as lists, since they may have different lengths for different molecules
    crystal_dict['unit_cell_coordinates'] = [np.asarray([np.asarray(heavy_atom.coordinates) for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]
    crystal_dict['unit_cell_fractional_coordinates'] = [np.asarray([heavy_atom.fractional_coordinates for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]
    crystal_dict['unit_cell_atomic_numbers'] = [np.asarray([heavy_atom.atomic_number for heavy_atom in component.heavy_atoms]) for component in unit_cell.components]

    if crystal_dict['space_group_number'] != 0:  # sometimes the sg number is broken, but if not, assign a consistent canonical SG symbol
        crystal_dict['space group symbol'] = SPACE_GROUPS[crystal_dict['space_group_number']]
    else:  # in which case, try reverse assigning the number, given the space group
        crystal_dict['space_group_number'] = sg_numbers[crystal_dict['space group symbol']]

    return crystal_dict, mol_volumes


def featurize_molecule(crystal, crystal_dict, rd_mol, mol_volume, component_num):
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
    assert np.mean(np.abs(coords - molecule_dict['atom_coordinates'])) < 1e-3
    assert np.sum(np.abs(atomic_numbers - molecule_dict['atom_atomic_numbers'])) == 0

    h_donors = list(sum(rd_mol.GetSubstructMatches(HDonorSmarts, uniquify=1), ()))  # convert tuple to list
    h_acceptors = list(sum(rd_mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1), ()))

    '''atom-wise features'''
    molecule_dict['atom_mass'] = [atom.GetMass() for atom in atoms]
    molecule_dict['atom_is_H_bond_donor'] = [1 if i in list(h_donors) else 0 for i in range(len(atoms))]
    molecule_dict['atom_is_H_bond_acceptor'] = [1 if i in list(h_acceptors) else 0 for i in range(len(atoms))]
    molecule_dict['atom_valence'] = [atom.GetTotalValence() for atom in atoms]
    molecule_dict['atom_vdW_radius'] = [vdw_radii[str(number)] for number in molecule_dict['atom_atomic_numbers']]
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
    molecule_dict['molecule_spherical defect'] = rdMolDescriptors.CalcAsphericity(rd_mol)
    molecule_dict['molecule_eccentricity'] = rdMolDescriptors.CalcEccentricity(rd_mol)
    molecule_dict['molecule_num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds((rd_mol))
    molecule_dict['molecule_planarity'] = rdMolDescriptors.CalcPBF(rd_mol)
    molecule_dict['molecule_radius_of_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(rd_mol)
    molecule_dict['molecule_radius'] = np.amax(np.linalg.norm(molecule_dict['atom_coordinates'] - molecule_dict['atom_coordinates'].mean(0), axis=-1))

    for anum in range(1, 36):
        molecule_dict[f'molecule_{element_symbols[str(anum)]}_fraction'] = get_fraction(molecule_dict['atom_atomic_numbers'], anum)

    for key in Fragments.__dict__.keys():  # for all the class methods
        if key[0:3] == 'fr_':  # if it's a functional group analysis methodad
            molecule_dict[f'molecule_has_{key[3:]}'] = Fragments.__dict__[key](rd_mol, countUnique=False)

    molecule_dict['molecule_smiles'] = Chem.MolToSmiles(rd_mol)
    molecule_dict['molecule_chemical_formula'] = rdMolDescriptors.CalcMolFormula(rd_mol)

    Ip, Ipm, _ = compute_principal_axes_np(np.asarray(molecule_dict['atom_coordinates']))  # rdMolTransforms.ComputePrincipalAxesAndMoments(rd_mol.GetConformer(), ignoreHs=False) # this does it column-wise
    molecule_dict['molecule_principal_axes'] = Ip  # row-wise principal_axes
    molecule_dict['molecule_principal_moment_1'] = Ipm[0]
    molecule_dict['molecule_principal_moment_2'] = Ipm[1]
    molecule_dict['molecule_principal_moment_3'] = Ipm[2]
    molecule_dict['molecule_is_asymmetric_top'] = not Ipm[0] == Ipm[1] == Ipm[2]

    return molecule_dict


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
            any([len(component.atoms) == 0 for component in crystal.molecule.components]),
            ]):
        #print(f'{crystal.identifier} failed crystal checks')
        return False, None, None

    try:
        unit_cell = crystal.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='CentroidIncluded')
    except RuntimeError:  # sometimes packing fails
        #print(f'{crystal.identifier} failed crystal checks')
        return False, None, None

    # molecule check via RDKit. If RDKit doesn't see it as a real molecule, don't accept it to the dataset.
    rd_mols = []
    for component in crystal.molecule.components:
        mol = Chem.MolFromMol2Block(component.to_string('mol2'), sanitize=True, removeHs=True)
        try:
            rd_mols.append(Chem.RemoveAllHs(mol))
        except:
            passed_molecule_checks = False
        if mol is None:
            passed_molecule_checks = False

    if not passed_molecule_checks:
        pass #print(f'{crystal.identifier} failed molecule checks')

    return all([passed_molecule_checks, passed_crystal_checks]), unit_cell, rd_mols
