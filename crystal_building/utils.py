import numpy as np
from torch.nn.utils import rnn as rnn

from models.utils import clean_generator_output, enforce_crystal_system
from common.geometry_calculations import single_molecule_principal_axes_torch, batch_molecule_principal_axes_torch, compute_Ip_handedness, rotvec2sph, sph2rotvec
from scipy.spatial.transform import Rotation
import torch
import sys


def generate_sorted_fractional_translations(supercell_size):
    # initialize fractional translations for supercell construction
    n_cells = (2 * supercell_size + 1) ** 3
    fractional_translations = torch.zeros((n_cells, 3))  # initialize the translations in fractional coords
    i = 0
    for xx in range(-supercell_size, supercell_size + 1):
        for yy in range(-supercell_size, supercell_size + 1):
            for zz in range(-supercell_size, supercell_size + 1):
                fractional_translations[i] = torch.tensor((xx, yy, zz))
                i += 1

    # sort fractional vectors from closest to furthest from central unit cell
    return fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]


def unit_cell_to_convolution_cluster(reference_cell_list: list,
                                     cell_vector_list: list,
                                     T_fc_list,
                                     atoms_list: list,
                                     crystal_multiplicity,
                                     supercell_scale=5,
                                     cutoff=5,
                                     sorted_fractional_translations=None,
                                     pare_to_convolution_cluster=True):
    """
    1) generate fractional translations for full supercell
    for each sample
    2) generate cartesian coordinates
    3) identify canonical conformer, inside inds, outside inds
    4) kick out molecules which are outside
    """
    device = T_fc_list.device

    if sorted_fractional_translations is None:
        sorted_fractional_translations = generate_sorted_fractional_translations(supercell_scale).to(device)

    supercell_sizes = sorted_fractional_translations.abs().max(1)[0]
    sorted_fractional_translations_f = sorted_fractional_translations[supercell_sizes <= supercell_scale]
    max_num_cells = len(sorted_fractional_translations_f)

    supercell_coords_list = []
    supercell_atoms_list = []
    ref_mol_inds_list = []
    copies = []
    for i, (ref_cell, unit_cell_vectors, atoms, sym_mult) in enumerate(zip(reference_cell_list, cell_vector_list, atoms_list, crystal_multiplicity)):
        if type(ref_cell) == np.ndarray:
            ref_cell = torch.tensor(ref_cell, device=device)

        ref_mol_radius = torch.max(torch.cdist(ref_cell[0].mean(0)[None, :], ref_cell[0]))
        cart_translations_i = torch.mul(unit_cell_vectors.tile(max_num_cells, 1), sorted_fractional_translations_f.reshape(max_num_cells * 3, 1))  # 3 cell vectors
        supercell_coords = ref_cell.clone().reshape(sym_mult * ref_cell.shape[1], 3).tile(max_num_cells, 1)  # duplicate over XxXxX supercell
        cart_translations = cart_translations_i.reshape(max_num_cells, 3, 3).sum(1)  # faster
        full_supercell_coords = supercell_coords + torch.repeat_interleave(cart_translations, ref_cell.shape[1] * ref_cell.shape[0], dim=0)  # add translations throughout

        mol_num_atoms = len(atoms)
        in_mol_inds = torch.arange(mol_num_atoms)  # assume canonical conformer is indexed first
        if pare_to_convolution_cluster:
            # NOTE canonical conformer is not always indexed first in prebuilt reference cells - shouldn't actually impact morphology but would be nice to clean up
            # will only be fixed if we build our own unit cells in dataset construction
            molwise_supercell_coords = full_supercell_coords.reshape(max_num_cells * sym_mult, mol_num_atoms, 3)

            ref_mol_centroid = molwise_supercell_coords[0].mean(0)  # first is always the canonical conformer
            all_mol_centroids = molwise_supercell_coords.mean(1)  # centroids for all molecules in the supercell

            mol_centroid_dists = torch.cdist(ref_mol_centroid[None, :], all_mol_centroids, p=2)[0]  # distances between canonical conformer and all other molecules

            '''
            include atoms with molecules within the convolution window
            if there are no such molecules, boost the window so we have something to convolve with
            otherwise discriminator errors out with zero edges
            '''
            successful_gconv = False
            extra_cutoff = 0
            while not successful_gconv:
                # ignore atoms which are more than mol_radius + conv_cutoff + buffer
                convolve_mol_inds = torch.where((mol_centroid_dists <= (ref_mol_radius + cutoff + extra_cutoff + 0.01)))[0]

                if len(convolve_mol_inds) <= 9:  # if the crystal is too diffuse / there are no molecules close enough to convolve with, we open the window and try again
                    extra_cutoff += 1
                else:
                    successful_gconv = True

            assert len(convolve_mol_inds) > 1  # must be more than one molecule in convolution

            '''add final indexing of atoms which are canonical conformer: 0, kept symmetry images: 1, and otherwise tossed'''
            ref_mol_inds = torch.ones(len(convolve_mol_inds) * mol_num_atoms, dtype=int, device=device)  # only index molecules which will be kept
            ref_mol_inds[in_mol_inds] = 0  # 0 is for 'inside', 1 is for 'outside'

            convolve_atom_inds = (torch.arange(mol_num_atoms, device=device)[:, None] + convolve_mol_inds * mol_num_atoms).T.reshape(len(convolve_mol_inds) * mol_num_atoms)  # looks complicated but it's fast

            supercell_coords_list.append(full_supercell_coords[convolve_atom_inds])  # only the relevant molecules are now kept
            supercell_atoms = atoms.repeat(len(convolve_mol_inds), 1)  # take the number of kept molecules only
        else:  # keep the whole NxNxN supercell
            supercell_coords_list.append(full_supercell_coords)
            supercell_atoms = atoms.repeat(len(full_supercell_coords) // mol_num_atoms, 1)
            ref_mol_inds = torch.ones(len(supercell_atoms), dtype=int, device=device)  # only index molecules which will be kept
            ref_mol_inds[in_mol_inds] = 0
            convolve_mol_inds = torch.arange(0, ref_mol_inds.sum() / mol_num_atoms)  # index every molecule

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


def fractional_transform(coords, T_mat):
    """
    input is the cartesian coordinates and the c-f or f-c fractional transformation matrix
    """
    if isinstance(coords, np.ndarray):
        return fractional_transform_np(coords, T_mat)
    elif torch.is_tensor(coords):
        return fractional_transform_torch(coords, T_mat)


def fractional_transform_torch(coords, T_mat):
    if coords.ndim == 2:
        return torch.einsum('nj,ij->ni', (coords, T_mat))
    elif coords.ndim == 3:
        return torch.einsum('nmj,ij->nmi', (coords, T_mat))


def find_coord_in_box_torch(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return torch.where((coords[:, 0] <= (box[0] + epsilon)) *
                       (coords[:, 1] <= (box[1] + epsilon)) *
                       (coords[:, 2] <= (box[2] + epsilon)))[0]


def batch_asymmetric_unit_pose_analysis_torch(unit_cell_coords_list, sg_ind_list, asym_unit_dict,
                                              T_fc_list, enforce_right_handedness=False,
                                              rotation_basis='cartesian', return_asym_unit_coords=False):
    """
    Parameters
    ----------
    unit_cell_coords: coordinates for the full unit cell. Each list entry [Z, num_atoms, 3]
    sg_ind: space group index
    asym_unit_dict: dict which defines the asymmetric unit for each space group
    T_cf : list of cartesian-to-fractional matrix transforms
    enforce_right_handedness : DEPRECATED doesn't make sense given nature of our transform
    rotation_basis : 'cartesian' or 'spherical' whether mol orientation rotvec should be parameterized in cartesian or spherical coordinates
    Returns : standardized pose parameters and canonical conformer handedness
    -------
    """

    '''Asymmetric Unit Centroid Analysis'''
    T_cf_list = torch.linalg.inv(T_fc_list)
    num_samples = len(unit_cell_coords_list)
    canonical_conformer_coords_list = []
    mol_position_list = []
    well_defined_asym_unit_list = []
    for i, unit_cell_coords in enumerate(unit_cell_coords_list):
        # identify which of the Z asymmetric units is canonical
        centroids_cartesian = unit_cell_coords.mean(-2)
        centroids_fractional = fractional_transform(centroids_cartesian, T_cf_list[i])
        centroids_fractional -= torch.floor(centroids_fractional)
        asym_unit = asym_unit_dict[str(int(sg_ind_list[i]))]
        canonical_conformer_index_i = find_coord_in_box_torch(centroids_fractional, asym_unit)

        well_defined_asym_unit = False
        if len(canonical_conformer_index_i) == 0:  # if we didn't find one, or found more than one pick the closest. In some cases, they are truly indistinguishable otherwise.
            canonical_conformer_index = [torch.argmin(torch.linalg.norm(centroids_fractional, axis=1))]
        elif len(canonical_conformer_index_i) > 1:
            dists = torch.linalg.norm(centroids_fractional[canonical_conformer_index_i], axis=1)
            if len(set(dists)) < len(dists):  # if any are equal, go dimension-by-dimension
                canonical_conformer_index = [canonical_conformer_index_i[0]]  # pre-set a random ind, overwrite later
                for dim in range(3):
                    dists = centroids_fractional[canonical_conformer_index_i][:, dim]
                    if len(set(dists)) < len(dists):
                        pass  # if are equal, go to next dim
                    else:
                        canonical_conformer_index = [canonical_conformer_index_i[torch.argmin(dists)]]
                        break  # stop at this dim
            else:
                canonical_conformer_index = [canonical_conformer_index_i[torch.argmin(dists)]]
        elif len(canonical_conformer_index_i) == 1:
            well_defined_asym_unit = True  # if there is any ambiguity, it is not 'well defined'
            canonical_conformer_index = canonical_conformer_index_i * 1

        canonical_conformer_coords_list.append(unit_cell_coords[canonical_conformer_index[0]])
        mol_position_list.append(centroids_fractional[canonical_conformer_index[0]])
        well_defined_asym_unit_list.extend([well_defined_asym_unit])

    mol_position_list = torch.stack(mol_position_list)

    assert torch.sum(torch.isnan(mol_position_list)) == 0

    '''Pose Analysis'''
    # compute the inverse of the rotation required to align the molecule with the cartesian axes
    Ip_axes_list, _, _ = batch_molecule_principal_axes_torch(canonical_conformer_coords_list)
    handedness_list = compute_Ip_handedness(Ip_axes_list)

    alignment_list = torch.eye(3, device=mol_position_list.device).tile(num_samples, 1, 1)
    if not enforce_right_handedness:
        alignment_list[:, 0, 0] = handedness_list

    # http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html#:~:text=Another%20popular%20rotation%20representation%20is,be%20represented%20in%20this%20form!
    assert torch.sum(torch.isnan(Ip_axes_list)) == 0, f"{Ip_axes_list}"

    rotvec_list = []
    for Ip_axes, alignment in zip(Ip_axes_list, alignment_list):
        rotation_matrix = Ip_axes.T @ torch.linalg.inv(alignment.T)
        assert torch.sum(torch.isnan(rotation_matrix)) == 0, f"{Ip_axes} {alignment} {rotation_matrix}"
        if not enforce_right_handedness:
            assert torch.linalg.det(rotation_matrix) > 0  # negative determinant is an improper rotation, which we do not want - inverts the molecule

        direction_vector = torch.tensor([
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1]],
            device=rotation_matrix.device, dtype=torch.float32)  # 32 precision is limiting here in some cases

        r_arg = (torch.trace(rotation_matrix) - 1) / 2

        if torch.abs(r_arg) >= 1:  # if we are close enough to one, the arccos will NaN
            # situation corresponds to a rotation by ~pi, with an unknown direction
            # fortunately, either direction results in the same transformation (C2)
            # some ill conditioned rotations may also throw |args| greater than 1 -
            # for safety must set these as something, may as well be this  # todo look at such ill-conditioned cases
            r = torch.pi
            direction_vector = torch.ones(3, device=rotation_matrix.device,
                                          dtype=torch.float32)
        else:  # calculate as normal
            r = torch.arccos(r_arg)
            if torch.isnan(r):  # manual catch for edge cases
                # print(f"caught bad rotation from {rotation_matrix} with direction {direction_vector} and r_arg {r_arg}")
                r = torch.pi

        if torch.sum(torch.isnan(direction_vector)) > 0 or (torch.sum(direction_vector) == 0):  # bad rotation or ~approx null rotation
            # print(f"caught bad direction from {rotation_matrix} with direction {direction_vector} and r_arg {r_arg}")

            r = torch.pi
            direction_vector = torch.ones(3, device=rotation_matrix.device,
                                          dtype=torch.float32)

        rotvec_list.append(direction_vector / torch.linalg.norm(direction_vector) * r)

    rotvec_list = torch.stack(rotvec_list)

    assert torch.sum(torch.isnan(rotvec_list)) == 0, f"{rotvec_list}"  # todo see how this could be possible

    '''
    since the direction of the axis is arbitrary, (x,y,z) is the same rotation as (-x,-y,-z),
    we can improve specificity of the model by constraining the axis to a half-sphere.
    Here we will take the +z direction as 'canonical' (equivalent to theta <pi/2).
    
    Swap the direction and take 2pi-norm to recapture the identical rotation. 
    '''
    flip_inds = torch.where(rotvec_list[:, -1] < 0)[0]
    flip_vecs = rotvec_list[flip_inds]
    flip_norms = torch.linalg.norm(flip_vecs, dim=-1)
    new_norms = 2 * torch.pi - flip_norms
    new_vecs = -flip_vecs / flip_norms[:, None] * new_norms[:, None]
    rotvec_list[flip_inds] = new_vecs

    '''
    # test for the above condition
    m1 = Rotation.from_rotvec(rotvec_list.detach().numpy()).as_matrix()
    m2 = Rotation.from_rotvec(rotvec_list0.detach().numpy()).as_matrix() # pre-inversion rotvec list
    print((m1 - m2).sum())
    '''
    if rotation_basis == 'cartesian':
        mol_orientation = rotvec_list
    elif rotation_basis == 'spherical':  # convert from cartesian to spherical coordinates
        mol_orientation = rotvec2sph(rotvec_list)
    else:
        print(f'{rotation_basis} is not a valid orientation parameterization!')
        sys.exit()

    assert torch.sum(torch.isnan(mol_orientation)) == 0, f"{mol_orientation} {rotvec_list}"
    assert torch.sum(torch.isnan(mol_position_list)) == 0, f"{mol_position_list}"

    if return_asym_unit_coords:
        return mol_position_list, mol_orientation, handedness_list, well_defined_asym_unit_list, canonical_conformer_coords_list
    else:
        return mol_position_list, mol_orientation, handedness_list, well_defined_asym_unit_list


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


def align_crystaldata_to_principal_axes(crystaldata, handedness):
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
    return symmetry operators as a list
    @param supercell_data:
    @return:
    """
    sym_ops_list = [torch.tensor(supercell_data.symmetry_operators[n], device=supercell_data.x.device, dtype=torch.float32)
                    for n in range(len(supercell_data.symmetry_operators))]

    return sym_ops_list, supercell_data


def rotvec2rotmat(mol_rotation: torch.tensor, basis='cartesian'):
    """
    get applied rotation matrix
    mol_rotation here is a list of rotation vectors [n_samples, 3]
    rotvec -> rotation matrix directly (see https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
    rotvec -> quat -> rotation matrix (old way)
    """
    if basis == 'cartesian':
        r = torch.linalg.norm(mol_rotation, dim=1)
        unit_vector = mol_rotation / r[:, None]
    elif basis == 'spherical':  # psi, phi, (spherical unit vector) theta (rotation vector)
        r = mol_rotation[:, -1]  # third dimension in spherical basis is the norm #torch.linalg.norm(mol_rotation, dim=1)
        mol_rotation = sph2rotvec(mol_rotation)
        unit_vector = mol_rotation / r[:, None]
    else:
        print(f'{basis} is not a valid orientation parameterization!')
        sys.exit()

    K = torch.stack((  # matrix representing rotation axis
        torch.stack((torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 2], unit_vector[:, 1]), dim=1),
        torch.stack((unit_vector[:, 2], torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 0]), dim=1),
        torch.stack((-unit_vector[:, 1], unit_vector[:, 0], torch.zeros_like(unit_vector[:, 0])), dim=1)
    ), dim=1)

    applied_rotation_list = torch.eye(3, device=r.device)[None, :, :].tile(len(r), 1, 1) + torch.sin(r[:, None, None]) * K + (1 - torch.cos(r[:, None, None])) * (K @ K)

    return applied_rotation_list


def build_unit_cell(symmetry_multiplicity, final_coords_list, T_fc_list, T_cf_list, sym_ops_list):
    """
    use cell symmetry to pattern canonical conformer into full unit cell
    batch crystals with same Z value together for added speed in large batches
    """
    reference_cell_list_i = []

    unique_z_values = torch.unique(symmetry_multiplicity)
    z_inds = [torch.where(symmetry_multiplicity == z)[0] for z in unique_z_values]

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

            # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
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


def clean_cell_params(samples, sg_inds, lattice_means, lattice_stds, symmetries_dict, asym_unit_dict,
                      rescale_asymmetric_unit=True, destandardize=False, mode='soft'):

    lattice_lengths, lattice_angles, mol_positions, mol_orientations \
        = clean_generator_output(samples, lattice_means, lattice_stds, destandardize=destandardize, mode=mode)

    fixed_lengths, fixed_angles = (
        enforce_crystal_system(lattice_lengths, lattice_angles, sg_inds, symmetries_dict))

    if rescale_asymmetric_unit:
        fixed_positions = scale_asymmetric_unit(asym_unit_dict, mol_positions, sg_inds)
    else:
        fixed_positions = mol_positions * 1

    '''collect'''
    final_samples = torch.cat((
        fixed_lengths,
        fixed_angles,
        fixed_positions,
        mol_orientations,
    ), dim=-1)

    return final_samples


def scale_asymmetric_unit(asym_unit_dict, mol_position, sg_inds):
    """
    input fractional coordinates are scaled on 0-1
    rescale these for the specific ranges according to each space group
    only space groups in asym_unit_dict will work - not all have been manually encoded
    this approach will not work for asymmetric units which are not parallelpipeds
    Parameters
    ----------
    asym_unit_dict
    mol_position
    sg_inds

    Returns
    -------
    """
    # scaled_mol_position = mol_position.clone()
    # for i, ind in enumerate(sg_ind):
    #     scaled_mol_position[i, :] = mol_position[i, :] * asym_unit_dict[str(int(ind))]

    # vectorized for speed
    # asym_units = torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_ind])
    # scaled_mol_position = mol_position * asym_units

    return mol_position * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])


def DEPRECATED_write_sg_to_all_crystals(override_sg, dataDims, supercell_data, symmetries_dict, sym_ops_list):
    # todo rewrite or deprecate when we update sampling
    # overwrite point group one-hot
    # overwrite space group one-hot
    # overwrite crystal system one-hot
    # overwrite z value

    sg_num = list(symmetries_dict['space_groups'].values()).index(override_sg) + 1  # indexing from 0
    sg_ind = symmetries_dict['sg_feature_ind_dict'][symmetries_dict['space_groups'][sg_num]]
    crysys_ind = symmetries_dict['crysys_ind_dict'][symmetries_dict['lattice_type'][sg_num]]
    z_value_ind = max(list(symmetries_dict['crysys_ind_dict'].values())) + 1

    supercell_data.x[:, -dataDims['num crystal generation features']] = 0  # set all crystal features to 0
    supercell_data.x[:, sg_ind] = 1  # set all molecules to the given space group
    supercell_data.x[:, crysys_ind] = 1  # set all molecules to the given crystal system

    supercell_data.mult = len(sym_ops_list[0]) * torch.ones_like(supercell_data.mult)
    supercell_data.x[:, z_value_ind] = supercell_data.mult[0] * torch.ones_like(supercell_data.x[:, 0])
    supercell_data.sg_ind = sg_num * torch.ones_like(supercell_data.sg_ind)

    return supercell_data


def update_crystal_symmetry_elements(mol_data, generate_sgs, symmetries_dict, randomize_sgs=False):
    """
    update the symmetry information in molecule-wise crystaldata objects
    """
    # identify the SG numbers we want to generate
    if type(generate_sgs[0]) == str:
        generate_sg_inds = [list(symmetries_dict['space_groups'].values()).index(SG) + 1 for SG in generate_sgs]  # indexing from 0
    elif torch.is_tensor(generate_sgs):
        generate_sg_inds = generate_sgs.cpu().detach().numpy()
    else:
        generate_sg_inds = generate_sgs

    # randomly assign SGs to samples
    if randomize_sgs:
        sample_sg_inds = np.random.choice(generate_sg_inds, size=mol_data.num_graphs, replace=True)
    else:
        sample_sg_inds = generate_sg_inds

    # update sym ops
    mol_data.symmetry_operators = [torch.Tensor(symmetries_dict['sym_ops'][sg_ind]).to(mol_data.x.device) for sg_ind in sample_sg_inds]
    mol_data.sg_ind = torch.tensor(sample_sg_inds, dtype=mol_data.sg_ind.dtype, device=mol_data.sg_ind.device)
    mol_data.mult = torch.tensor([len(ops) for ops in mol_data.symmetry_operators], dtype=torch.int32, device=mol_data.sg_ind.device)

    return mol_data


def fractional_transform_np(coords, T_mat):
    if coords.ndim == 2:
        return np.einsum('nj,ij->ni', coords, T_mat)
    elif coords.ndim == 3:
        return np.einsum('nmj,ij->nmi', coords, T_mat)


def find_coord_in_box_np(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return np.where((coords[:, 0] <= (box[0] + epsilon)) * (coords[:, 1] <= (box[1] + epsilon) * (coords[:, 2] <= (box[2] + epsilon))))[0]
