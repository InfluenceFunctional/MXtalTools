import numpy as np
from models.utils import enforce_1d_bound
from common.geometry_calculations import compute_principal_axes_np, single_molecule_principal_axes_torch, batch_molecule_principal_axes_torch, compute_Ip_handedness
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import sys


def ref_to_supercell(reference_cell_list: list, cell_vector_list: list, T_fc_list: list,
                     atoms_list: list, z_values, supercell_scale=5, cutoff=5,
                     sorted_fractional_translations=None):
    """
    1) generate fractional translations for full supercell
    for each sample
    2) generate cartesian coordinates
    3) identify canonical conformer, inside inds, outside inds
    4) kick out molecules which are outside
    """
    if sorted_fractional_translations is None:
        n_cells = (2 * supercell_scale + 1) ** 3
        fractional_translations = torch.zeros((n_cells, 3), device=T_fc_list.device)  # initialize the translations in fractional coords
        i = 0
        for xx in range(-supercell_scale, supercell_scale + 1):
            for yy in range(-supercell_scale, supercell_scale + 1):
                for zz in range(-supercell_scale, supercell_scale + 1):
                    fractional_translations[i] = torch.tensor((xx, yy, zz), device=T_fc_list.device)
                    i += 1
        sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]
    else:
        n_cells = len(sorted_fractional_translations)

    device = T_fc_list.device
    supercell_coords_list = []
    supercell_atoms_list = []
    ref_mol_inds_list = []
    copies = []
    z_values = [len(ref_list) for ref_list in reference_cell_list]  # todo delete once generator method is separated
    for i, (ref_cell, unit_cell_vectors, atoms, z_value) in enumerate(zip(reference_cell_list, cell_vector_list, atoms_list, z_values)):
        if type(ref_cell) == np.ndarray:
            ref_cell = torch.tensor(ref_cell, device=device)

        mol_n_atoms = len(atoms)
        supercell_coords = ref_cell.clone().reshape(z_value * ref_cell.shape[1], 3).tile(n_cells, 1)  # duplicate over XxXxX supercell
        cart_translations_i = torch.mul(unit_cell_vectors.tile(n_cells, 1), sorted_fractional_translations.reshape(n_cells * 3, 1))  # 3 cell vectors
        # cart_translations = torch.stack(cart_translations_i.split(3, dim=0), dim=0).sum(1)
        cart_translations = cart_translations_i.reshape(n_cells, 3, 3).sum(1)  # faster

        full_supercell_coords = supercell_coords + torch.repeat_interleave(cart_translations, ref_cell.shape[1] * ref_cell.shape[0], dim=0)  # add translations throughout

        # index atoms within the 'canonical' conformer, which is always indexed first
        # TODO canonical conformer is not indexed first in real cells - shouldn't actually impact morphology but would be nice to clean up
        in_mol_inds = torch.arange(mol_n_atoms)
        molwise_supercell_coords = full_supercell_coords.reshape(n_cells * z_value, mol_n_atoms, 3)
        ref_mol_centroid = molwise_supercell_coords[0].mean(0)  # first is always the canonical conformer
        all_mol_centroids = torch.mean(molwise_supercell_coords, dim=1)  # centroids for all molecules in the supercell
        mol_centroid_dists = torch.cdist(ref_mol_centroid[None, :], all_mol_centroids, p=2)[0]
        ref_mol_radius = torch.max(torch.cdist(ref_mol_centroid[None, :], full_supercell_coords[in_mol_inds]))

        successful_gconv = False
        extra_cutoff = 0
        while successful_gconv == False:
            # ignore atoms which are more than mol_radius + conv_cutoff + buffer
            convolve_mol_inds = torch.where((mol_centroid_dists <= (2 * ref_mol_radius + cutoff + extra_cutoff + 0.1)))[0]

            if len(convolve_mol_inds) <= mol_n_atoms:  # if the crystal is too diffuse / there are no molecules close enough to convolve with, we open the window and try again
                extra_cutoff += 0.5
            else:
                successful_gconv = True

        ref_mol_inds = torch.ones(len(convolve_mol_inds) * mol_n_atoms, dtype=int, device=device)  # only index molecules which will be kept
        ref_mol_inds[in_mol_inds] = 0

        assert len(convolve_mol_inds) > mol_n_atoms

        convolve_atom_inds = (torch.arange(mol_n_atoms, device=device)[:, None] + convolve_mol_inds * mol_n_atoms).T.reshape(len(convolve_mol_inds) * mol_n_atoms)  # looks complicated but it's fast

        supercell_coords_list.append(full_supercell_coords[convolve_atom_inds])  # only the relevant molecules are now kept
        supercell_atoms = atoms.repeat(len(convolve_mol_inds), 1)  # take the number of kept molecules only

        supercell_atoms_list.append(supercell_atoms)
        ref_mol_inds_list.append(ref_mol_inds)

        copies.append(len(convolve_mol_inds))

    n_copies = torch.tensor(copies, dtype=torch.int32)

    return supercell_coords_list, supercell_atoms_list, ref_mol_inds_list, n_copies


def update_supercell_data(supercell_data, supercell_atoms_list, supercell_coords_list, ref_mol_inds_list, reference_cell_list):
    """
    copy new supercell data back onto original supercell objects, omitting symmetry information
    """
    device = supercell_data.x.device
    for i in range(supercell_data.num_graphs):
        if i == 0:
            new_batch = torch.ones(len(supercell_atoms_list[i]), device=device).int() * i
            new_ptr = torch.zeros(supercell_data.num_graphs + 1, device=device)
            new_ptr[1] = len(supercell_coords_list[0])
        else:
            new_batch = torch.cat((new_batch, torch.ones(len(supercell_atoms_list[i]), device=device).int() * i))
            new_ptr[i + 1] = new_ptr[i] + len(supercell_coords_list[i])

    # update dataloader with cell info
    supercell_data.x = torch.cat(supercell_atoms_list).type(dtype=torch.float32)
    supercell_data.pos = torch.cat(supercell_coords_list).type(dtype=torch.float32)
    supercell_data.batch = new_batch.type(dtype=torch.int64)
    supercell_data.ptr = new_ptr.type(dtype=torch.int64)
    supercell_data.aux_ind = torch.cat(ref_mol_inds_list).type(dtype=torch.int)
    supercell_data.ref_cell_pos = reference_cell_list

    return supercell_data


def clean_cell_output(cell_lengths: torch.tensor, cell_angles: torch.tensor, mol_position: torch.tensor, mol_rotation: torch.tensor,
                      lattices, dataDims,
                      enforce_crystal_system=False, rotation_type='cartesian rotvec', return_transforms=False, standardized_sample: bool=True):
    """
    constrain cell parameters to physically meaningful values
    :param cell_lengths:
    :param cell_angles:
    :param mol_position:
    :param mol_rotation:
    :return:
    """
    # todo sub datadims for raw means and stds
    if standardized_sample:
        # de-standardize everything
        means = torch.Tensor(dataDims['lattice means']).to(cell_lengths.device)
        stds = torch.Tensor(dataDims['lattice stds']).to(cell_lengths.device)

        # soft clipping to ensure correct range with finite gradients
        cell_lengths = cell_lengths * stds[0:3] + means[0:3]
        cell_angles = cell_angles * stds[3:6] + means[3:6]
        mol_position = mol_position * stds[6:9] + means[6:9]
        mol_rotation = mol_rotation * stds[9:12] + means[9:12]

    # TODO find a way to bound arbitrary shapes beyond parallelpipeds

    cell_lengths = F.softplus(cell_lengths - 0.1) + 0.1  # enforces positive nonzero

    cell_angles = enforce_1d_bound(cell_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='soft')  # prevent too-skinny cells
    mol_position = enforce_1d_bound(mol_position, 0.5, 0.5, mode='soft')

    norms = torch.linalg.norm(mol_rotation, dim=1)
    normed_norms = enforce_1d_bound(norms, torch.pi, torch.pi, mode='soft')
    mol_rotation = mol_rotation / norms[:, None] * normed_norms[:, None]  # renormalize

    for i in range(len(cell_lengths)):
        if enforce_crystal_system:  # enforce properties of crystal system
            lattice = lattices[i]

            # enforce agreement with crystal system
            if lattice.lower() == 'triclinic':
                pass
            elif lattice.lower() == 'monoclinic':  # fix alpha and gamma
                cell_angles[i, 0], cell_angles[i, 2] = torch.pi / 2, torch.pi / 2
            elif lattice.lower() == 'orthorhombic':  # fix all angles
                cell_angles[i] = torch.ones(3) * torch.pi / 2
            elif (lattice.lower() == 'tetragonal'):  # fix all angles and a & b vectors
                cell_angles[i] = torch.ones(3) * torch.pi / 2
                cell_lengths[i, 0], cell_lengths[i, 1] = torch.mean(cell_lengths[i, 0:2]) * torch.ones(2).to(cell_lengths.device)
            elif (lattice.lower() == 'hexagonal') or (lattice.lower() == 'trigonal') or (lattice.lower() == 'rhombohedral'):
                cell_lengths[i, 0], cell_lengths[i, 1] = torch.mean(cell_lengths[i, 0:2]) * torch.ones(2).to(cell_lengths.device)
                cell_angles[i, 0:2] = torch.pi / 2
                cell_angles[i, 2] = torch.pi * 2 / 3
            elif (lattice.lower() == 'cubic'):  # all angles 90 all lengths equal
                cell_lengths[i] = cell_lengths[i].mean() * torch.ones(3).to(cell_lengths.device)
                cell_angles[i] = torch.pi * torch.ones(3) / 2
            else:
                print(lattice + ' is not a valid crystal lattice!')
                sys.exit()
        else:
            # don't assume a crystal system, but snap angles close to 90, to assist in precise symmetry
            pass  # cell_angles[i, torch.abs(cell_angles[i] - torch.pi / 2) < 0.01] = torch.pi / 2

    if return_transforms:
        return cell_lengths, cell_angles, mol_position, mol_rotation, None, None, None
    else:
        return cell_lengths, cell_angles, mol_position, mol_rotation


def compute_lattice_vector_overlap(coords_list: list, T_cf_list: list, normed_lattice_vectors=None):
    """
    compute overlap between molecule principal axes and the crystal lattice vectors
    """
    if normed_lattice_vectors is None:
        # initialize fractional lattice vectors - should be exactly identical to what's in molecule_featurizer.py
        # ideally precomputed and fed to the function
        supercell_scale = 2  # t
        n_cells = (2 * supercell_scale + 1) ** 3

        fractional_translations = np.zeros((n_cells, 3))  # initialize the translations in fractional coords
        i = 0
        for xx in range(-supercell_scale, supercell_scale + 1):
            for yy in range(-supercell_scale, supercell_scale + 1):
                for zz in range(-supercell_scale, supercell_scale + 1):
                    fractional_translations[i] = np.array((xx, yy, zz))
                    i += 1
        lattice_vectors = torch.Tensor(fractional_translations[np.argsort(np.abs(fractional_translations).sum(1))][1:])  # leave out the 0,0,0 element
        normed_lattice_vectors = lattice_vectors / torch.linalg.norm(lattice_vectors, axis=1)[:, None]

    # Ip_list = compute_principal_axes_list(coords_list)
    Ip_list, _, _ = batch_molecule_principal_axes_torch(coords_list)

    # get mol axes in fractional basis
    vectors_f = torch.einsum('nij,nmj->nmi', (T_cf_list, Ip_list))

    # compute overlaps
    normed_vectors_f = vectors_f / torch.linalg.norm(vectors_f, axis=2)[:, :, None]
    return torch.einsum('ij,nmj->nmi', (normed_lattice_vectors, normed_vectors_f))


def get_cell_fractional_centroids(coords, T_cf):
    """
    input is the cartesian coordinates and the c->f transformation matrix
    """
    if isinstance(coords, np.ndarray):
        return np.einsum('nmj,ij->nmi', coords, T_cf).mean(1)
    elif torch.is_tensor(coords):
        return torch.einsum('nmj,ij->nmi', (coords, T_cf)).mean(1)


def c_f_transform(coords, T_cf):
    """
    input is the cartesian coordinates and the c->f transformation matrix
    """
    if coords.ndim == 2:  # option for extra dimension
        if isinstance(coords, np.ndarray):
            return np.einsum('nj,ij->ni', coords, T_cf)
        elif torch.is_tensor(coords):
            return torch.einsum('nj,ij->ni', (coords, T_cf))
    elif coords.ndim == 3:
        if isinstance(coords, np.ndarray):
            return np.einsum('nmj,ij->nmi', coords, T_cf)
        elif torch.is_tensor(coords):
            return torch.einsum('nmj,ij->nmi', (coords, T_cf))


def f_c_transform(coords, T_fc):
    """
    input is the fractional coordinates and the f->c transformation matrix
    """
    if coords.ndim == 2:  # option for extra dimension
        if isinstance(coords, np.ndarray):
            return np.einsum('nj,ij->ni', coords, T_fc)
        elif torch.is_tensor(coords):
            return torch.einsum('nj,ij->ni', (coords, T_fc))
    elif coords.ndim == 3:
        if isinstance(coords, np.ndarray):
            return np.einsum('nmj,ij->nmi', coords, T_fc)
        elif torch.is_tensor(coords):
            return torch.einsum('nmj,ij->nmi', (coords, T_fc))


def find_coord_in_box(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return np.where((coords[:, 0] <= (box[0] + epsilon)) * (coords[:, 1] <= (box[1] + epsilon) * (coords[:, 2] <= (box[2] + epsilon))))[0]


def unit_cell_analysis(unit_cell_coords, sg_ind, asym_unit_dict, T_cf, enforce_right_handedness=False, return_asym_unit_coords=False):
    """

    Parameters
    ----------
    unit_cell_coords: coordinates for the full unit cell. Each list entry [Z, n_atoms, 3]
    sg_ind: space group index
    asym_unit_dict: dict which defines the asymmetric unit for each space group
    T_cf : list of cartesian-to-fractional matrix transforms
    enforce_right_handedness : DEPRECATED doesn't make sense given nature of our transform
    Returns : standardized cell parameters and canonical conformer handedness
    -------

    """
    if isinstance(unit_cell_coords, np.ndarray):
        asym_unit = np.asarray(asym_unit_dict[str(int(sg_ind))])  # will only work for units which we have written down the parameterization for
    else:
        unit_cell_coords = unit_cell_coords.cpu().detach().numpy()
        asym_unit = asym_unit_dict[str(int(sg_ind))].cpu().detach().numpy()
        T_cf = T_cf.cpu().detach().numpy()

    # identify which of the Z asymmetric units is canonical # todo need a better system for when conformers are exactly or nearly exactly on the edge
    centroids_cartesian = unit_cell_coords.mean(-2)
    centroids_fractional = np.inner(T_cf, centroids_cartesian).T
    centroids_fractional -= np.floor(centroids_fractional)
    canonical_conformer_index = find_coord_in_box(centroids_fractional, asym_unit)

    if len(canonical_conformer_index) == 0:  # if we didn't find one, patch over by just picking the closest # todo delete this when we have the above fixed
        canonical_conformer_index = [np.argmin(np.linalg.norm(centroids_fractional, axis=1))]

    canonical_conformer_coords = unit_cell_coords[canonical_conformer_index[0]]  # we enforce in the filtering step that there must be exactly one centroid in the canonical asymmetric unit

    # next we need to compute the inverse of the rotation required to align the molecule with the cartesian axes
    Ip_axes, _, _ = compute_principal_axes_np(canonical_conformer_coords)
    handedness = compute_Ip_handedness(Ip_axes)

    '''
    we want the matrix which rotates from the standard to the native orientation
    native_orientation = rotation_matrix @ standard_orientation
    rotation_matrix = native_orientation @ inv(standard_orientation)

    standard_orientation = inv(rotation_matrix) @ native_orientation

    '''
    alignment = np.eye(3)
    if not enforce_right_handedness:
        alignment[0, 0] = handedness

    rotation_matrix = Ip_axes.T @ np.linalg.inv(alignment.T)

    ''' debugging 
    # check original transform on the Ip axes
    print(np.sum(np.abs(np.einsum('ij,nj->ni', rotation_matrix, alignment) - Ip_axes)) < 1e-5)
    
    # check the inverse transform
    centered_conformer = (canonical_conformer_coords - canonical_conformer_coords.mean(0))
    transformed_canonical_coords = np.einsum('ij,nj->ni', np.linalg.inv(rotation_matrix), centered_conformer)
    Ip_axes_transformed, _, _ = compute_principal_axes_np(transformed_canonical_coords)

    assert np.sum(np.abs(Ip_axes_transformed - alignment)) < 1e-5
    '''

    # test if we got it right
    if not enforce_right_handedness:
        assert np.linalg.det(rotation_matrix) > 0  # negative determinant is an improper rotation, which will not work

    mol_orientation = Rotation.from_matrix(rotation_matrix).as_rotvec()
    mol_position = centroids_fractional[canonical_conformer_index[0]]

    if return_asym_unit_coords:
        return mol_position, mol_orientation, handedness, canonical_conformer_coords

    else:
        return mol_position, mol_orientation, handedness


def flip_I3(coords: torch.tensor, Ip: torch.tensor):
    """
    flip the given coordinates such that the third principal inertial vector (row 0) direction is swapped
    thus switching the handedness of the coordinates
    """
    target_Ip = Ip.clone()
    target_Ip[0] = -target_Ip[0]
    rotation_matrix = target_Ip.T @ torch.linalg.inv(Ip).T  # find rotation matrix between given and target principal axes
    flipped_coords = torch.inner(rotation_matrix, coords).T  # apply this transformation to the coordinates

    return flipped_coords


def invert_coords(coords):
    """
    reflect point cloud about it's centroid
    """
    return -(coords - coords.mean(0)) + coords.mean(0)


def compute_principal_axes_list(coords_list):
    """
    compute principal axes for a list of multiple sets of coordinates
    """
    Ip_axes_list = torch.zeros((len(coords_list), 3, 3), device=coords_list[0].device)
    for i, coords in enumerate(coords_list):
        Ip_axes_list[i], _, _ = single_molecule_principal_axes_torch(coords)

    return Ip_axes_list


def align_crystaldata_to_principal_axes(crystaldata: object, handedness: object = None) -> object:
    """
    align principal inertial axes of molecules in a crystaldata object to the xyz or xy(-z) axes
    only works for geometric principal axes (all atoms mass = 1)
    """
    coords_list = [crystaldata.pos[crystaldata.ptr[i]:crystaldata.ptr[i + 1]] for i in range(crystaldata.num_graphs)]
    coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(crystaldata.num_graphs)]
    # principal_axes_list = compute_principal_axes_list(coords_list_centred, masses_list = None)
    principal_axes_list, _, _ = batch_molecule_principal_axes_torch(coords_list_centred)  # much faster

    eye = torch.tile(torch.eye(3, device=crystaldata.x.device), (crystaldata.num_graphs, 1, 1))  # set as right-handed in general
    if handedness is not None:  # otherwise, custom
        eye[:, 0, 0] = handedness

    # rotation2 = torch.matmul(eye2.reshape(data.num_graphs, 3, 3), torch.linalg.inv(principal_axes_list.reshape(data.num_graphs, 3, 3))) # one step

    rotation_matrix_list = [torch.matmul(torch.linalg.inv(principal_axes_list[i]), eye[i]) for i in range(crystaldata.num_graphs)]

    crystaldata.pos = torch.cat([torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in range(crystaldata.num_graphs)])

    # for debugging
    # std_coords_list = [torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in range(data.num_graphs)]
    # principal_axes_list2, _, _ = batch_molecule_principal_axes(std_coords_list)  # much faster
    # print(torch.abs(principal_axes_list2 - eye).sum((1,2))) # should be close to zero
    return crystaldata


def random_crystaldata_alignment(crystaldata):
    """
    randomize orientation of molecules in a crystaldata object
    """
    coords_list = [crystaldata.pos[crystaldata.ptr[i]:crystaldata.ptr[i + 1]] for i in range(crystaldata.num_graphs)]
    coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(crystaldata.num_graphs)]

    rotation_matrix_list = torch.tensor(Rotation.random(num=crystaldata.num_graphs).as_matrix(), device=crystaldata.x.device, dtype=crystaldata.pos.dtype)
    crystaldata.pos = torch.cat([torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in range(crystaldata.num_graphs)])

    return crystaldata


def set_sym_ops(supercell_data):
    """
    enforce known symmetry operations for given space groups
    sometimes, the samples come with nonstandard space groups, which we do not want to use by accident
    @param supercell_data:
    @return:
    """
    sym_ops_list = [torch.tensor(supercell_data.symmetry_operators[n], device=supercell_data.x.device, dtype=supercell_data.x.dtype)
                    for n in range(len(supercell_data.symmetry_operators))]

    return sym_ops_list, supercell_data