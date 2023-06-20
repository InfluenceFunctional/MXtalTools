import numpy as np
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn

from models.utils import enforce_1d_bound
from common.geometry_calculations import compute_principal_axes_np, single_molecule_principal_axes_torch, batch_molecule_principal_axes_torch, compute_Ip_handedness, rotvec2sph, sph2rotvec
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import sys


def ref_to_supercell(reference_cell_list: list, cell_vector_list: list, T_fc_list: list,
                     atoms_list: list, z_values, supercell_scale=5, cutoff=5,
                     sorted_fractional_translations=None, pare_to_convolution_cluster=True):
    """
    1) generate fractional translations for full supercell
    for each sample
    2) generate cartesian coordinates
    3) identify canonical conformer, inside inds, outside inds
    4) kick out molecules which are outside
    """
    n_cells = (2 * supercell_scale + 1) ** 3
    if sorted_fractional_translations is None:
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

        in_mol_inds = torch.arange(mol_n_atoms)  # assume canonical conformer is indexed first
        if pare_to_convolution_cluster:
            # TODO canonical conformer is not indexed first in prebuilt reference cells - shouldn't actually impact morphology but would be nice to clean up
            molwise_supercell_coords = full_supercell_coords.reshape(n_cells * z_value, mol_n_atoms, 3)

            ref_mol_centroid = molwise_supercell_coords[0].mean(0)  # first is always the canonical conformer
            all_mol_centroids = torch.mean(molwise_supercell_coords, dim=1)  # centroids for all molecules in the supercell

            mol_centroid_dists = torch.cdist(ref_mol_centroid[None, :], all_mol_centroids, p=2)[0]  # distances between canonical conformer and all other molecules
            ref_mol_radius = torch.max(torch.cdist(ref_mol_centroid[None, :], full_supercell_coords[in_mol_inds]))  # molecule radius of canonical conformer

            '''
            include atoms with molecules within the convolution window
            if there are no such molecules, boost the window so we have something to convolve with
            otherwise discriminator errors out with zero edges
            '''
            successful_gconv = False
            extra_cutoff = 0
            while successful_gconv == False:
                # ignore atoms which are more than mol_radius + conv_cutoff + buffer
                convolve_mol_inds = torch.where((mol_centroid_dists <= (2 * ref_mol_radius + cutoff + extra_cutoff + 0.1)))[0]

                if len(convolve_mol_inds) <= mol_n_atoms:  # if the crystal is too diffuse / there are no molecules close enough to convolve with, we open the window and try again
                    extra_cutoff += 0.5
                else:
                    successful_gconv = True

            assert len(convolve_mol_inds) > mol_n_atoms  # must be more than one molecule in convolution

            '''add final indexing of atoms which are canonical conformer: 0, kept symmetry images: 1, and otherwise tossed'''
            ref_mol_inds = torch.ones(len(convolve_mol_inds) * mol_n_atoms, dtype=int, device=device)  # only index molecules which will be kept
            ref_mol_inds[in_mol_inds] = 0

            convolve_atom_inds = (torch.arange(mol_n_atoms, device=device)[:, None] + convolve_mol_inds * mol_n_atoms).T.reshape(len(convolve_mol_inds) * mol_n_atoms)  # looks complicated but it's fast

            supercell_coords_list.append(full_supercell_coords[convolve_atom_inds])  # only the relevant molecules are now kept
            supercell_atoms = atoms.repeat(len(convolve_mol_inds), 1)  # take the number of kept molecules only
        else:  # keep the whole NxNxN supercell
            supercell_coords_list.append(full_supercell_coords)
            supercell_atoms = atoms.repeat(len(full_supercell_coords) // mol_n_atoms, 1)
            ref_mol_inds = torch.ones(len(supercell_atoms), dtype=int, device=device)  # only index molecules which will be kept
            ref_mol_inds[in_mol_inds] = 0
            convolve_mol_inds = torch.arange(1, ref_mol_inds.sum() / mol_n_atoms)  # index every molecule

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
                      enforce_crystal_system=False, rotation_type='cartesian rotvec', return_transforms=False, standardized_sample: bool = True):
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


def find_coord_in_box_np(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return np.where((coords[:, 0] <= (box[0] + epsilon)) * (coords[:, 1] <= (box[1] + epsilon) * (coords[:, 2] <= (box[2] + epsilon))))[0]


def find_coord_in_box_torch(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return torch.where((coords[:, 0] <= (box[0] + epsilon)) * (coords[:, 1] <= (box[1] + epsilon) * (coords[:, 2] <= (box[2] + epsilon))))[0]


def asymmetric_unit_pose_analysis_np(unit_cell_coords, sg_ind, asym_unit_dict, T_cf, enforce_right_handedness=False, return_asym_unit_coords=False, rotation_basis='cartesian'):
    """
    Parameters
    ----------
    unit_cell_coords: coordinates for the full unit cell. Each list entry [Z, n_atoms, 3]
    sg_ind: space group index
    asym_unit_dict: dict which defines the asymmetric unit for each space group
    T_cf : list of cartesian-to-fractional matrix transforms
    enforce_right_handedness : DEPRECATED doesn't make sense given nature of our transform
    Returns : standardized pose parameters and canonical conformer handedness
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
    canonical_conformer_index = find_coord_in_box_np(centroids_fractional, asym_unit)

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

    if not enforce_right_handedness:
        assert np.linalg.det(rotation_matrix) > 0  # negative determinant is an improper rotation, which will not work

    unit_vector = np.asarray([
        rotation_matrix[2, 1] - rotation_matrix[1, 2],
        rotation_matrix[0, 2] - rotation_matrix[2, 0],
        rotation_matrix[1, 0] - rotation_matrix[0, 1]])
    theta = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    rotvec = unit_vector / np.linalg.norm(unit_vector) * theta

    if rotation_basis == 'cartesian':
        mol_orientation = rotvec
        # mol_orientation = Rotation.from_matrix(rotation_matrix).as_rotvec()  # old way using scipy
    elif rotation_basis == 'spherical':
        mol_orientation = rotvec2sph(rotvec)
    else:
        print(f'{rotation_basis} is not a valid orientation parameterization!')
        sys.exit()

    mol_position = centroids_fractional[canonical_conformer_index[0]]

    if return_asym_unit_coords:
        return mol_position, mol_orientation, handedness, canonical_conformer_coords

    else:
        return mol_position, mol_orientation, handedness

    ''' debugging 
    # check original transform on the Ip axes
    print(np.sum(np.abs(np.einsum('ij,nj->ni', rotation_matrix, alignment) - Ip_axes)) < 1e-5)

    # check the inverse transform
    centered_conformer = (canonical_conformer_coords - canonical_conformer_coords.mean(0))
    transformed_canonical_coords = np.einsum('ij,nj->ni', np.linalg.inv(rotation_matrix), centered_conformer)
    Ip_axes_transformed, _, _ = compute_principal_axes_np(transformed_canonical_coords)

    assert np.sum(np.abs(Ip_axes_transformed - alignment)) < 1e-5
    '''


def batch_asymmetric_unit_pose_analysis_torch(unit_cell_coords_list, sg_ind_list, asym_unit_dict,
                                              T_fc_list, enforce_right_handedness=False,
                                              rotation_basis='cartesian', return_asym_unit_coords=False):
    """
    Parameters
    ----------
    unit_cell_coords: coordinates for the full unit cell. Each list entry [Z, n_atoms, 3]
    sg_ind: space group index
    asym_unit_dict: dict which defines the asymmetric unit for each space group
    T_cf : list of cartesian-to-fractional matrix transforms
    enforce_right_handedness : DEPRECATED doesn't make sense given nature of our transform
    rotation_basis : 'cartesian' or 'spherical' whether mol orientation rotvec should be paramterized in cartesian or spherical coordinates
    Returns : standardized pose parameters and canonical conformer handedness
    -------
    """

    T_cf_list = torch.linalg.inv(T_fc_list)
    num_samples = len(unit_cell_coords_list)
    canonical_conformer_coords_list = []
    mol_position_list = []
    for i, unit_cell_coords in enumerate(unit_cell_coords_list):
        # identify which of the Z asymmetric units is canonical # todo need a better system for when conformers are exactly or nearly exactly on the edge
        centroids_cartesian = unit_cell_coords.mean(-2)
        centroids_fractional = torch.inner(T_cf_list[i], centroids_cartesian).T
        centroids_fractional -= torch.floor(centroids_fractional)
        asym_unit = asym_unit_dict[str(int(sg_ind_list[i]))]
        canonical_conformer_index = find_coord_in_box_torch(centroids_fractional, asym_unit)

        if len(canonical_conformer_index) == 0:  # if we didn't find one, patch over by just picking the closest # todo delete this when we have the above fixed
            canonical_conformer_index = [torch.argmin(torch.linalg.norm(centroids_fractional, axis=1))]

        canonical_conformer_coords_list.append(unit_cell_coords[canonical_conformer_index[0]])
        mol_position_list.append(centroids_fractional[canonical_conformer_index[0]])

    mol_position_list = torch.stack(mol_position_list)

    # next we need to compute the inverse of the rotation required to align the molecule with the cartesian axes
    Ip_axes_list, _, _ = batch_molecule_principal_axes_torch(canonical_conformer_coords_list)
    handedness_list = compute_Ip_handedness(Ip_axes_list)

    alignment_list = torch.eye(3, device=mol_position_list.device).tile(num_samples, 1, 1)
    if not enforce_right_handedness:
        alignment_list[:, 0, 0] = handedness_list

    rotvec_list = []
    for Ip_axes, alignment in zip(Ip_axes_list, alignment_list):
        rotation_matrix = Ip_axes.T @ torch.linalg.inv(alignment.T)
        if not enforce_right_handedness:
            assert torch.linalg.det(rotation_matrix) > 0  # negative determinant is an improper rotation, which will not work

        unit_vector = torch.tensor([
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1]], device=rotation_matrix.device, dtype=torch.float32)
        theta = torch.arccos((torch.trace(rotation_matrix) - 1) / 2)
        rotvec_list.append(unit_vector / torch.linalg.norm(unit_vector) * theta)

    if rotation_basis == 'cartesian':
        mol_orientation = torch.stack(rotvec_list)
        # mol_orientation = Rotation.from_matrix(rotation_matrix).as_rotvec()  # old way using scipy
    elif rotation_basis == 'spherical':
        mol_orientation = rotvec2sph(torch.stack(rotvec_list))
    else:
        print(f'{rotation_basis} is not a valid orientation parameterization!')
        sys.exit()

    if return_asym_unit_coords:
        return mol_position_list, mol_orientation, handedness_list, canonical_conformer_coords_list
    else:
        return mol_position_list, mol_orientation, handedness_list


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
                    for n in range(len(supercell_data.symmetry_operators))]  # todo refeaturize symmetry_operators as lists

    return sym_ops_list, supercell_data


def rotvec2rotmat(mol_rotation: torch.tensor, basis='cartesian'):
    """
    get applied rotation matrix
    mol_rotation here is a list of rotation vectors [n_samples, 3]
    rotvec -> rotation matrix directly (see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
    rotvec -> quat -> rotation matrix (old way)
    """
    if basis == 'cartesian':
        theta = torch.linalg.norm(mol_rotation, dim=1)
        unit_vector = mol_rotation / theta[:, None]
    elif basis == 'spherical':  # psi, phi, (spherical unit vector) theta (rotaiton vector)
        mol_rotation = sph2rotvec(mol_rotation)
        theta = torch.linalg.norm(mol_rotation, dim=1)
        unit_vector = mol_rotation / theta[:, None]
    else:
        print(f'{basis} is not a valid orientation parameterization!')
        sys.exit()

    K = torch.stack((  # matrix representing rotation axis
        torch.stack((torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 2], unit_vector[:, 1]), dim=1),
        torch.stack((unit_vector[:, 2], torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 0]), dim=1),
        torch.stack((-unit_vector[:, 1], unit_vector[:, 0], torch.zeros_like(unit_vector[:, 0])), dim=1)
    ), dim=1)

    applied_rotation_list = torch.eye(3, device=theta.device)[None, :, :].tile(len(theta), 1, 1) + torch.sin(theta[:, None, None]) * K + (1 - torch.cos(theta[:, None, None])) * (K @ K)

    # old way via quaternion
    # q = torch.cat([torch.cos(theta / 2)[:, None], unit_vector * torch.sin(theta / 2)[:, None]], dim=1)
    #
    # applied_rotation_list = torch.stack((torch.stack((1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3], 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]), dim=1),
    #                                      torch.stack((2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]), dim=1),
    #                                      torch.stack((2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2], 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2), dim=1)),
    #                                     dim=1)

    return applied_rotation_list


def get_canonical_conformer(supercell_data, mol_position, sym_ops_list, debug=False):
    """ # todo officially deprecate - unusedc
    identify canonical conformer
    """
    # do it in batches of same z-values to allow some parallelization
    unique_z_values = torch.unique(supercell_data.Z)
    z_inds = [torch.where(supercell_data.Z == z)[0] for z in unique_z_values]

    # initialize final list
    canonical_fractional_centroids_list = torch.zeros((len(mol_position), 3)).to(mol_position.device)
    # split it into z values and do each in parallel for speed
    for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
        centroids_i = torch.zeros((z_value, len(inds), 3)).to(mol_position.device)  # initialize
        for zv in range(z_value):
            centroids_i[zv, :, :] = \
                torch.einsum('nhb,nb->nh', (torch.stack([sym_ops_list[j] for j in inds])[:, zv],  # was an index error here
                                            torch.cat((mol_position[inds],
                                                       torch.ones(mol_position[inds].shape[:-1] + (1,)).to(mol_position.device)), dim=-1)
                                            )
                             )[:, :-1]
        centroids = centroids_i - torch.floor(centroids_i)

        canonical_fractional_centroids_list[inds] = centroids[torch.argmin(torch.linalg.norm(centroids, dim=2), dim=0), torch.arange(len(inds))]

        if debug:  # todo rewrite these debug statements as tests
            assert F.l1_loss(canonical_fractional_centroids_list[inds], mol_position[inds], reduction='mean') < 0.001

    if debug:
        assert F.l1_loss(canonical_fractional_centroids_list, mol_position, reduction='mean') < 0.001

    return canonical_fractional_centroids_list


def build_unit_cell(z_values, final_coords_list, T_fc_list, T_cf_list, sym_ops_list):
    """
    use cell symmetry to pattern canonical conformer into full unit cell
    batch crystals with same Z value together for added speed in large batches
    """
    reference_cell_list_i = []

    unique_z_values = torch.unique(z_values)
    z_inds = [torch.where(z_values == z)[0] for z in unique_z_values]

    for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
        # padding allows for parallel transforms below
        lens = torch.tensor([len(final_coords_list[ii]) for ii in inds])
        padded_coords_c = rnn.pad_sequence(final_coords_list, batch_first=True)[inds]
        centroids_c = torch.stack([final_coords_list[inds[ii]].mean(0) for ii in range(len(inds))])
        centroids_f = torch.einsum('nij,nj->ni', (T_cf_list[inds], centroids_c))

        # initialize empty
        ref_cells = torch.zeros((z_value, len(inds), padded_coords_c.shape[1], 3)).to(final_coords_list[0].device)
        # get symmetry ops for this batch
        z_sym_ops = torch.stack([sym_ops_list[j] for j in inds])
        # add 4th dimension as a dummy for affine transforms
        affine_centroids_f = torch.cat((centroids_f, torch.ones(centroids_f.shape[:-1] + (1,)).to(padded_coords_c.device)), dim=-1)

        for zv in range(z_value):
            # get molecule centroids via symmetry ops
            centroids_f_z = torch.einsum('nij,nj->ni', (z_sym_ops[:, zv], affine_centroids_f))[..., :-1]

            # force centroids within unit cell
            centroids_f_z_in_cell = centroids_f_z - torch.floor(centroids_f_z)

            # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame  # todo reconfirm this is the right way to do this
            rot_coords_f = torch.einsum('nmj,nij->nmi',
                                        (torch.einsum('mij,mnj->mni',
                                                      (T_cf_list[inds],
                                                       padded_coords_c - centroids_c[:, None, :])),
                                         z_sym_ops[:, zv, :3, :3]))
            # add final centroid
            ref_cells[zv, :, :, :] = torch.einsum('mij,mnj->mni',
                                                  (T_fc_list[inds],
                                                   rot_coords_f + centroids_f_z_in_cell[:, None, :]))

        reference_cell_list_i.extend([ref_cells[:, jj, :lens[jj], :] for jj in range(len(inds))])

    sorted_z_inds = torch.argsort(torch.cat(z_inds))

    reference_cell_list = [reference_cell_list_i[ind] for ind in sorted_z_inds]

    return reference_cell_list


def scale_asymmetric_unit(asym_unit_dict, mol_position, sg_ind):
    """
    input fractional coordinates are scaled on 0-1
    rescale these for the specific ranges according to each space group
    only space groups in asym_unit_dict will work - not all have been manually encoded
    this approach will not work for asymmetric units which are not parallelpipeds
    Parameters
    ----------
    mol_position
    sg_ind

    Returns
    -------
    """  # todo vectorize
    scaled_mol_position = mol_position.clone()
    for i, ind in enumerate(sg_ind):
        scaled_mol_position[i, :] = mol_position[i, :] * asym_unit_dict[str(int(ind))]

    return scaled_mol_position
