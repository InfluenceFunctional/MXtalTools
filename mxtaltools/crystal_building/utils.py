import sys
from typing import Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.nn.utils import rnn as rnn
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import single_molecule_principal_axes_torch, \
    list_molecule_principal_axes_torch, compute_Ip_handedness, rotvec2sph, batch_compute_fractional_transform, \
    get_batch_centroids, \
    rotvec2rotmat, batch_molecule_principal_axes_torch, extract_rotmat, apply_rotation_to_batch, rotmat2rotvec
from mxtaltools.models.functions.radial_graph import radius


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


def unit_cell_to_supercell_cluster(crystal_batch, cutoff: float = 6, supercell_size: int = 10):
    cluster_batch = crystal_batch.clone()

    cc_centroids = fractional_transform(crystal_batch.aunit_centroid, crystal_batch.T_fc)
    good_translations, good_translations_bool = get_cart_translations(cc_centroids,
                                                                      crystal_batch.T_fc,
                                                                      crystal_batch.radius,
                                                                      cutoff, supercell_size)

    ucell_num_atoms = crystal_batch.num_atoms * crystal_batch.sym_mult
    unit_cell_pos = torch.cat([torch.tensor(poses.reshape(ucell_num_atoms[ind], 3),
                                            dtype=torch.float32,
                                            device=crystal_batch.device
                                            )
                               for ind, poses in enumerate(crystal_batch.unit_cell_pos)])
    unit_cell_batch = torch.arange(crystal_batch.num_graphs, device=crystal_batch.device
                                   ).repeat_interleave(crystal_batch.sym_mult * crystal_batch.num_atoms)

    num_cells = good_translations_bool.sum(1)  # == torch.diff(good_translations_ptr).long()

    translations_per_atom = num_cells.repeat_interleave(ucell_num_atoms, dim=0).long()
    atoms_per_translation = ucell_num_atoms.repeat_interleave(num_cells, dim=0)
    atomwise_translation = good_translations.repeat_interleave(atoms_per_translation, dim=0)

    # index final features
    # pos, x, z, mol_ind, aux_ind, ptr, batch
    atoms_per_unit_cell = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()
    apuc_cum = torch.cat([torch.tensor([0], device=crystal_batch.device), torch.cumsum(atoms_per_unit_cell, dim=0)]).long()
    unit_cell_to_cluster_index = torch.cat([
        (torch.arange(atoms_per_unit_cell[ind], device=crystal_batch.device) + apuc_cum[ind]).repeat(num_cells[ind])
        for ind in range(crystal_batch.num_graphs)
    ])
    mol_to_unit_cell_index = torch.cat([
        (torch.arange(crystal_batch.num_atoms[ind], device=crystal_batch.device) + crystal_batch.ptr[ind]).repeat(crystal_batch.sym_mult[ind])
        for ind in range(crystal_batch.num_graphs)
    ])

    cluster_batch.pos = unit_cell_pos[unit_cell_to_cluster_index]
    cluster_batch.pos += atomwise_translation
    cluster_batch.z = crystal_batch.z[mol_to_unit_cell_index][unit_cell_to_cluster_index]
    cluster_batch.x = crystal_batch.x[mol_to_unit_cell_index][unit_cell_to_cluster_index]
    cluster_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=crystal_batch.device),
                                   torch.cumsum(ucell_num_atoms * num_cells, dim=0)]).long()
    cluster_batch.batch = unit_cell_batch.repeat_interleave(translations_per_atom, dim=0)
    cluster_batch.aux_ind = torch.ones_like(cluster_batch.z)
    cluster_batch.mol_ind = torch.ones_like(cluster_batch.z)
    for ind in range(cluster_batch.num_graphs):
        cluster_batch.aux_ind[cluster_batch.ptr[ind]:cluster_batch.ptr[ind] + cluster_batch.num_atoms[ind]] = 0
        cluster_batch.mol_ind[cluster_batch.ptr[ind]:cluster_batch.ptr[ind+1]] = (
            torch.arange(num_cells[ind]*crystal_batch.sym_mult[ind], device=crystal_batch.device).repeat_interleave(crystal_batch.num_atoms[ind]))

    #'''
    #cluster_batch.visualize_structure([1,2,3,4,5], exclusion_level='unit cell')
    #cluster_batch.visualize_structure([1,2,3,4,5], exclusion_level='distance')
    #'''
    return cluster_batch


def get_cart_translations(cc_centroids,
                          T_fc,
                          mol_radii,
                          cutoff,
                          supercell_size: int = 9):
    sorted_fractional_translations = generate_sorted_fractional_translations(supercell_size).to(T_fc.device)

    # get the set of all possible cartesian translations
    cell_vectors = T_fc.permute(0, 2, 1)
    # for the n samples, for the 3 (i) cell vectors, get the inner product with all the fractional translation vectors
    parallelpiped_centroids = torch.einsum('nij, mi -> nmj', cell_vectors, sorted_fractional_translations)

    # we need to include any parallelpiped which overlaps with a sphere of
    # radius cutoff_distance around the canonical conformer for each cell - a bounding box calculation
    # we compute this in a brute force way, adding the diagonal length of each parallelpiped to the cutoff
    # this will take extra boxes, but is fast and simple
    cell_diag = cell_vectors.sum(1).norm(dim=-1)
    cutoff_distance = mol_radii * 2 + cutoff + 0.1 + cell_diag/2  # convolutional radius plus cell diagonal

    centroid_dists = torch.linalg.norm(cc_centroids[:, None, :] - parallelpiped_centroids, dim=-1)
    good_translations_bool = centroid_dists < cutoff_distance[:, None]

    # build only the unit cells which are relevant to each cluster
    good_translations_bool[:, 0] = True  # always take the 0,0,0 element
    good_translations = parallelpiped_centroids[good_translations_bool]

    # unit_cell_ptr = torch.cat([
    #     torch.zeros(1, dtype=torch.float32, device=crystal_batch.device),
    #     torch.cumsum(ucell_num_atoms, dim=0)
    # ]).long()
    # good_translations_ptr = torch.cat([torch.zeros(1, device=cart_translations.device),
    #                                    torch.cumsum(good_translations_bool.sum(1), dim=0)])

    return good_translations, good_translations_bool


def unit_cell_to_convolution_cluster(unit_cell_pos_list: list,
                                     cell_vector_list: list,
                                     device: str,
                                     node_features_list: list,
                                     crystal_multiplicity: torch.LongTensor,
                                     sorted_fractional_translations: torch.Tensor,
                                     supercell_scale: int = 5,
                                     cutoff: float = 6,
                                     pare_to_convolution_cluster=True,
                                     ):
    """
    1) generate fractional translations for full supercell
    for each sample
    2) generate cartesian coordinates
    3) identify canonical conformer, inside inds, outside inds
    4) kick out molecules which are outside
    """
    # todo see if we can dynamically reduce the supercell size when it's behaving well empirically
    # pare down to fractional translations we will actually need for the given supercell size
    supercell_sizes = sorted_fractional_translations.abs().amax(1)
    sorted_fractional_translations_f = sorted_fractional_translations[supercell_sizes <= supercell_scale]
    max_num_cells = len(sorted_fractional_translations_f)

    supercell_coords_list = []
    supercell_atoms_list = []
    ref_mol_inds_list = []
    copies = []
    for i, (ref_cell, unit_cell_vectors, node_features, sym_mult) in enumerate(
            zip(unit_cell_pos_list, cell_vector_list, node_features_list, crystal_multiplicity)):
        if isinstance(ref_cell[0], np.ndarray):  # TODO would be nice not to have this, or to do it all at once
            ref_cell = torch.tensor(ref_cell, device=device)
        if node_features.ndim == 1:  # TODO likewise, this should be done all at once or not at all
            node_features = node_features[:, None]

        # TODO give this as an argument, as it should be precomputed when we make the object
        ref_mol_radius = torch.max(torch.cdist(ref_cell[0].mean(0)[None, :], ref_cell[0]))
        # get cartesian translations along 3 cell vectors # TODO could certainly be parallelized
        cart_translations_i = torch.mul(unit_cell_vectors.tile(max_num_cells, 1),
                                        sorted_fractional_translations_f.reshape(max_num_cells * 3, 1))
        # TODO parallelize with repeat_interleave
        # duplicate coordinates over XxXxX supercell
        supercell_coords = ref_cell.clone().reshape(sym_mult * ref_cell.shape[1], 3).tile(max_num_cells, 1)

        # todo parallelize
        cart_translations = cart_translations_i.reshape(max_num_cells, 3, 3).sum(1)  # faster

        # todo parallelize
        full_supercell_coords = supercell_coords + torch.repeat_interleave(cart_translations,
                                                                           ref_cell.shape[1] * ref_cell.shape[0],
                                                                           dim=0)  # add translations throughout

        mol_num_atoms = len(node_features)
        in_mol_inds = torch.arange(mol_num_atoms)  # assume canonical conformer is indexed first
        if pare_to_convolution_cluster:
            # NOTE canonical conformer is not always indexed first in prebuilt reference cells - shouldn't actually impact morphology but would be nice to clean up
            # will only be fixed if we build our own unit cells in dataset construction

            # todo could be parallelized by scatter op with correct batching
            molwise_supercell_coords = full_supercell_coords.reshape(max_num_cells * sym_mult, mol_num_atoms, 3)
            ref_mol_centroid = molwise_supercell_coords[0].mean(0)  # first is always the canonical conformer
            all_mol_centroids = molwise_supercell_coords.mean(1)  # centroids for all molecules in the supercell

            # todo could be parallelized via radius call
            # distances between asymmetric unit and all other molecules
            mol_centroid_dists = torch.cdist(ref_mol_centroid[None, :], all_mol_centroids, p=2)[0]

            '''
            include atoms with molecules within the convolution window
            if there are no such molecules, boost the window so we have something to convolve with
            otherwise discriminator errors out with zero edges
            '''
            # todo this is the one part that might be tricky to parallelize
            successful_gconv = False
            extra_cutoff = 0
            test_ind = 0
            while not successful_gconv and test_ind <= 5:  # this actually just sucks and shouldn't be necessary
                # ignore atoms which are more than 2 * mol_radius + conv_cutoff + buffer
                convolve_mol_inds = \
                    torch.where((mol_centroid_dists <= (2 * ref_mol_radius + cutoff + extra_cutoff + 0.01)))[0]

                if test_ind >= 5:
                    convolve_mol_inds = torch.where(mol_centroid_dists < 100)[0]  # just take everything
                elif len(
                        convolve_mol_inds) <= 2:  # if the crystal is too diffuse / there are no molecules close enough to convolve with, we open the window and try again
                    extra_cutoff += 1
                else:
                    successful_gconv = True
                test_ind += 1

            if len(convolve_mol_inds) <= 2:
                # just add the 5 closest so the code doesn't crash
                convolve_mol_inds = torch.argsort(mol_centroid_dists)[:5]

            '''add final indexing of atoms which are canonical conformer: 0, kept symmetry images: 1, and otherwise tossed'''
            ref_mol_inds = torch.ones(len(convolve_mol_inds) * mol_num_atoms, dtype=int,
                                      device=device)  # only index molecules which will be kept
            ref_mol_inds[in_mol_inds] = 0  # 0 is for 'inside', 1 is for 'outside'

            convolve_atom_inds = (torch.arange(mol_num_atoms, device=device)[:,
                                  None] + convolve_mol_inds * mol_num_atoms).T.reshape(
                len(convolve_mol_inds) * mol_num_atoms)  # looks complicated but it's fast
            # I take it back this is a monstrosity
            supercell_coords_list.append(
                full_supercell_coords[convolve_atom_inds])  # only the relevant molecules are now kept
            supercell_atoms = node_features.repeat(len(convolve_mol_inds), 1)  # take the number of kept molecules only
        else:  # keep the whole NxNxN supercell
            supercell_coords_list.append(full_supercell_coords)
            supercell_atoms = node_features.repeat(len(full_supercell_coords) // mol_num_atoms, 1)
            ref_mol_inds = torch.ones(len(supercell_atoms), dtype=int,
                                      device=device)  # only index molecules which will be kept
            ref_mol_inds[in_mol_inds] = 0
            convolve_mol_inds = torch.arange(0, ref_mol_inds.sum() / mol_num_atoms)  # index every molecule

        supercell_atoms_list.append(supercell_atoms)
        ref_mol_inds_list.append(ref_mol_inds)

        copies.append(len(convolve_mol_inds))

    n_copies = torch.tensor(copies, dtype=torch.int32)

    return supercell_coords_list, supercell_atoms_list, ref_mol_inds_list, n_copies


def update_supercell_data(supercell_batch,
                          supercell_atoms_list,
                          supercell_coords_list,
                          ref_mol_inds_list,
                          reference_cell_list,
                          ):
    """
    copy new supercell data back onto original supercell objects, omitting symmetry information
    """
    device = supercell_batch.x.device
    for i in range(supercell_batch.num_graphs):
        if i == 0:
            new_batch = torch.ones(len(supercell_atoms_list[i]), device=device).int() * i
            new_ptr = torch.zeros(supercell_batch.num_graphs + 1, device=device)
            new_ptr[1] = len(supercell_coords_list[0])
        else:
            new_batch = torch.cat((new_batch, torch.ones(len(supercell_atoms_list[i]), device=device).int() * i))
            new_ptr[i + 1] = new_ptr[i] + len(supercell_coords_list[i])

    molwise_batch = supercell_batch.batch.clone()

    # update crystaldata batch with cell info
    supercell_batch.x = torch.cat(supercell_atoms_list).type(dtype=torch.float32)
    supercell_batch.pos = torch.cat(supercell_coords_list).type(dtype=torch.float32)
    supercell_batch.batch = new_batch.type(dtype=torch.int64)
    supercell_batch.ptr = new_ptr.type(dtype=torch.int64)
    supercell_batch.aux_ind = torch.cat(ref_mol_inds_list).type(dtype=torch.int)
    supercell_batch.unit_cell_pos = reference_cell_list
    supercell_batch.mol_ind = torch.zeros_like(supercell_batch.aux_ind)  # mol ind is always 0 for zp=1 cells

    inside_batch = supercell_batch.batch[supercell_batch.aux_ind == 0]
    n_repeats = torch.tensor([int(torch.sum(supercell_batch.batch == ii) / torch.sum(inside_batch == ii)) for ii in
                              range(supercell_batch.num_graphs)])  # number of molecules in convolution region
    supercell_partial_charges = [supercell_batch.p_charges[molwise_batch == ii].repeat(n_repeats[ii]) for ii in
                                 range(supercell_batch.num_graphs)]
    supercell_batch.p_charges = torch.cat(supercell_partial_charges)

    return supercell_batch


def fractional_transform(coords, transform_matrix):
    """
    Transform between fractional/cartesian bases.
    Assumes the fractional->cartesian transform is the transpose of the box vectors
    Args:
        coords:
        transform_matrix:

    Returns: transformed_coords
    """
    if isinstance(coords, np.ndarray):
        return fractional_transform_np(coords, transform_matrix)
    elif torch.is_tensor(coords):
        return fractional_transform_torch(coords, transform_matrix)


def fractional_transform_np(coords, transform_matrix):
    if coords.ndim == 2 and transform_matrix.ndim == 2:
        return np.einsum('nj,ij->ni', coords, transform_matrix)
    elif coords.ndim == 3 and transform_matrix.ndim == 2:
        return np.einsum('nmj,ij->nmi', coords, transform_matrix)
    elif coords.ndim == 2 and transform_matrix.ndim == 3:
        return np.einsum('nj,nij->ni', coords, transform_matrix)


def fractional_transform_torch(coords, transform_matrix):
    if coords.ndim == 2 and transform_matrix.ndim == 2:
        return torch.einsum('nj,ij->ni', (coords, transform_matrix))
    elif coords.ndim == 3 and transform_matrix.ndim == 2:
        return torch.einsum('nmj,ij->nmi', (coords, transform_matrix))
    elif coords.ndim == 2 and transform_matrix.ndim == 3:
        return torch.einsum('nj,nij->ni', (coords, transform_matrix))


def find_coord_in_box_torch(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return torch.where((coords[:, 0] <= (box[0] + epsilon)) *
                       (coords[:, 1] <= (box[1] + epsilon)) *
                       (coords[:, 2] <= (box[2] + epsilon)) *
                       (coords[:, 0] >= 0) *
                       (coords[:, 1] >= 0) *
                       (coords[:, 2] >= 0)
                       )[0]


def batch_asymmetric_unit_pose_analysis_torch(unit_cell_coords_list,
                                              sg_ind_list,
                                              asym_unit_dict,
                                              T_fc_list,
                                              enforce_right_handedness=False,
                                              rotation_basis='cartesian',
                                              return_asym_unit_coords=False):
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

    # assert torch.sum(torch.isnan(mol_position_list)) == 0

    '''Pose Analysis'''
    # compute the inverse of the rotation required to align the molecule with the cartesian axes
    Ip_axes_list, _, _ = list_molecule_principal_axes_torch(canonical_conformer_coords_list)
    handedness_list = compute_Ip_handedness(Ip_axes_list)

    alignment_list = torch.eye(3, device=mol_position_list.device).tile(num_samples, 1, 1)
    if not enforce_right_handedness:
        alignment_list[:, 0, 0] = handedness_list

    # http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html#:~:text=Another%20popular%20rotation%20representation%20is,be%20represented%20in%20this%20form!
    # assert torch.sum(torch.isnan(Ip_axes_list)) == 0, f"{Ip_axes_list}"

    rotvec_list = []
    for Ip_axes, alignment in zip(Ip_axes_list, alignment_list):
        rotation_matrix = Ip_axes.T @ torch.linalg.inv(alignment.T)
        assert torch.sum(torch.isnan(rotation_matrix)) == 0, f"{Ip_axes} {alignment} {rotation_matrix}"
        if not enforce_right_handedness:
            # negative determinant is an improper rotation, which we do not want - inverts the molecule
            assert torch.linalg.det(rotation_matrix) > 0

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

        # bad rotation or ~approx null rotation
        if torch.sum(torch.isnan(direction_vector)) > 0 or (torch.sum(direction_vector) == 0):
            # print(f"caught bad direction from {rotation_matrix} with direction {direction_vector} and r_arg {r_arg}")

            r = torch.pi
            direction_vector = torch.ones(3, device=rotation_matrix.device, dtype=torch.float32)

        rotvec_list.append(direction_vector / torch.linalg.norm(direction_vector) * r)

    rotvec_list = torch.stack(rotvec_list)

    # assert torch.sum(torch.isnan(rotvec_list)) == 0, f"{rotvec_list}"

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

    # assert torch.sum(torch.isnan(mol_orientation)) == 0, f"{mol_orientation} {rotvec_list}"
    # assert torch.sum(torch.isnan(mol_position_list)) == 0, f"{mol_position_list}"

    if return_asym_unit_coords:
        return mol_position_list, mol_orientation, handedness_list, well_defined_asym_unit_list, canonical_conformer_coords_list
    else:
        return mol_position_list, mol_orientation, handedness_list, well_defined_asym_unit_list


def parameterize_crystal_batch(crystal_batch,
                               asym_unit_dict,
                               enforce_right_handedness: bool = False,
                               return_aunit: bool = False,
                               ):
    """

    """

    '''
    Asymmetric Unit Centroid Analysis
    identify "canonical" asymmetric unit out of the Z asymmetric units in the unit cell
    '''
    canonical_conformer_coords_list = []
    mol_position_list = []
    well_defined_asym_unit_list = []
    for i, (unit_cell_coords, T_cf, sg_ind) in enumerate(
            zip(crystal_batch.unit_cell_pos, crystal_batch.T_cf, crystal_batch.sg_ind)):
        if not torch.is_tensor(unit_cell_coords):
            unit_cell_coords = torch.tensor(unit_cell_coords, dtype=torch.float32, device=crystal_batch.device)
        (canonical_conformer_index,
         centroids_fractional,
         well_defined_asym_unit) = (
            identify_canonical_asymmetric_unit(
                T_cf,
                asym_unit_dict,
                sg_ind,
                unit_cell_coords))

        canonical_conformer_coords_list.append(unit_cell_coords[canonical_conformer_index[0]])
        mol_position_list.append(centroids_fractional[canonical_conformer_index[0]])
        well_defined_asym_unit_list.extend([well_defined_asym_unit])

    mol_position_list = torch.stack(mol_position_list)
    crystal_batch.pos = torch.cat(canonical_conformer_coords_list)

    '''
    Pose Analysis
    compute the inverse of the rotation required to align the molecule with the cartesian axes
    help from http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html#:~:text=Another%20popular%20rotation%20representation%20is,be%20represented%20in%20this%20form!
    
    '''

    rotvec_list, handedness_list = extract_aunit_orientation(crystal_batch,
                                                             enforce_right_handedness,
                                                             canonicalize_orientation=True)

    if return_aunit:
        return mol_position_list, rotvec_list, handedness_list, well_defined_asym_unit_list, canonical_conformer_coords_list
    else:
        return mol_position_list, rotvec_list, handedness_list, well_defined_asym_unit_list


def extract_aunit_orientation(mol_batch,
                              enforce_right_handedness,
                              canonicalize_orientation: bool = True):
    Ip, Ipm, I = batch_molecule_principal_axes_torch(
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        mol_batch.num_atoms
    )

    handedness_list = compute_Ip_handedness(Ip).long()
    eye = torch.eye(3, device=mol_batch.device).tile(mol_batch.num_graphs, 1, 1)
    if not enforce_right_handedness:
        eye[:, 0, 0] = handedness_list.flatten()

    # det should all be 1, as we should have correct handedness
    rotation_matrix_list = extract_rotmat(eye, torch.linalg.inv(Ip)).permute(0, 2, 1)

    rotvec_list = rotmat2rotvec(rotation_matrix_list)
    rotvec_list = cleanup_invalid_rotvecs(rotation_matrix_list, rotvec_list)

    if canonicalize_orientation:
        canonicalize_rotvec(rotvec_list)

    return rotvec_list, eye[:, 0, 0]  # the actual handedness


def canonicalize_rotvec(rotvecs: torch.Tensor):
    """
    since the direction of the axis is arbitrary, (x,y,z) is the same rotation as (-x,-y,-z),
    we can improve specificity of the model by constraining the axis to a half-sphere.
    Here we will take the +z direction as 'canonical'

    Swap the direction and take 2pi-norm to recapture the identical rotation.
    """
    flip_inds = torch.where(rotvecs[:, -1] < 0)[0]
    flip_vecs = rotvecs[flip_inds]
    flip_norms = torch.linalg.norm(flip_vecs, dim=-1)
    new_norms = 2 * torch.pi - flip_norms
    new_vecs = -flip_vecs / flip_norms[:, None] * new_norms[:, None]
    rotvecs[flip_inds] = new_vecs
    return rotvecs


def cleanup_invalid_rotvecs(rotation_matrix_list, rotvec_list):
    r_arg_list = (torch.einsum('ijj->i', rotation_matrix_list) - 1) / 2
    bad_rotations = torch.argwhere(r_arg_list >= 1).flatten()
    # if we are close enough to one, the arccos will NaN
    # situation corresponds to a rotation by ~pi, with an unknown direction
    # fortunately, either direction results in the same transformation (C2)
    # some ill conditioned rotations may also throw |args| greater than 1 -
    # for safety must set these as something, may as well be this  # todo look at such ill-conditioned cases
    # set rotation then as pi
    rotvec_list[bad_rotations, :] = 1
    r_list = torch.arccos(r_arg_list)
    r_list[bad_rotations] = torch.pi
    more_bad_rotations = torch.argwhere(~torch.isfinite(r_list)).flatten()
    if len(more_bad_rotations) > 0:  # manually catch other edge cases
        r_list[more_bad_rotations] = torch.pi
    # more manual edge cases
    bad_directions = torch.argwhere(torch.logical_or(
        (~torch.isfinite(rotvec_list.sum(1))),
        (rotvec_list.abs().sum(1) == 0))
    ).flatten()
    if len(bad_directions) > 0:
        r_list[bad_directions] = torch.pi
        rotvec_list[bad_directions] = 1
    rotvec_list = rotvec_list / torch.linalg.norm(rotvec_list, dim=-1)[:, None] * r_list[:, None]
    return rotvec_list


def identify_canonical_asymmetric_unit(T_cf, asym_unit_dict, sg_ind, unit_cell_coords):
    # cartesian mol centroids -> fractional mol centroids -> confirm inside box
    centroids_cartesian = unit_cell_coords.mean(-2)  # assume a shape [Z, n_atoms, 3]
    centroids_fractional = fractional_transform(centroids_cartesian, T_cf)
    centroids_fractional -= torch.floor(centroids_fractional)
    aunit_size = asym_unit_dict[str(int(sg_ind))]
    canonical_conformer_index_i = find_coord_in_box_torch(centroids_fractional, aunit_size)
    # if we didn't find one, or found more than one pick the closest.
    # In some cases, they are truly indistinguishable.
    if len(canonical_conformer_index_i) == 0:
        well_defined_asym_unit = False
        canonical_conformer_index = [torch.argmin(torch.linalg.norm(centroids_fractional, axis=1))]
    elif len(canonical_conformer_index_i) == 1:
        well_defined_asym_unit = True  # if there is any ambiguity, it is not 'well defined'
        canonical_conformer_index = canonical_conformer_index_i * 1
    elif len(canonical_conformer_index_i) > 1:
        # if we find more than one inside
        well_defined_asym_unit = False
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
    return canonical_conformer_index, centroids_fractional, well_defined_asym_unit


def flip_I3(coords: torch.tensor, Ip: torch.tensor):
    """
    flip the given coordinates such that the third principal inertial vector (row 0) direction is swapped
    thus switching the handedness of the coordinates
    """
    target_Ip = Ip.clone()
    target_Ip[0] = -target_Ip[0]
    rotation_matrix = target_Ip.T @ torch.linalg.inv(
        Ip).T  # find rotation matrix between given and target principal axes
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


def align_mol_batch_to_standard_axes(mol_batch, handedness=None):
    """
    align principal inertial axes of molecules in a crystaldata object to the xyz or xy(-z) axes
    only works for geometric principal axes (all atoms mass = 1)
    """
    Ip, Ipm, I = batch_molecule_principal_axes_torch(
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        mol_batch.num_atoms
    )

    eye = torch.tile(torch.eye(3,
                               device=mol_batch.x.device,
                               dtype=torch.float32),
                     (mol_batch.num_graphs, 1, 1)
                     )  # set as right-handed by default

    if handedness is not None:  # otherwise, custom
        # alignment vector is (x,y,z) by default, or (-x,y,z) for left-handed structures
        eye[:, 0, 0] = handedness.flatten()  # flatten - in case it's higher dimensional [n,1]

    # rotation is p_align = R @ p_orig,
    # R = p_align \ p_orig

    rotation_matrix_list = extract_rotmat(eye, torch.linalg.inv(Ip))
    mol_batch.recenter_molecules()  # ensures molecules are centered
    mol_batch.pos = apply_rotation_to_batch(mol_batch.pos,
                                            rotation_matrix_list,
                                            mol_batch.batch)
    # test
    # Ip, Ipm, I = batch_molecule_principal_axes_torch(
    #     mol_batch.pos,
    #     mol_batch.batch,
    #     mol_batch.num_graphs,
    #     mol_batch.num_atoms
    # )
    # assert torch.all(torch.isclose(Ip, eye, atol=1e-4))
    return mol_batch


def align_batch_to_principal_axes(mol_batch, handedness=None):
    """
    align principal inertial axes of molecules in a crystaldata object to the xyz or (-x)yz axes
    only works for geometric principal axes (all atoms mass = 1)
    """
    coords_list = [mol_batch.pos[mol_batch.ptr[i]:mol_batch.ptr[i + 1]] for i in range(mol_batch.num_graphs)]
    coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(mol_batch.num_graphs)]
    principal_axes_list, _, _ = (
        batch_molecule_principal_axes_torch(
            mol_batch.pos, mol_batch.batch,
            num_graphs=mol_batch.num_graphs,
        nodes_per_graph=mol_batch.num_atoms))  # much faster

    eye = torch.tile(torch.eye(3, device=mol_batch.x.device),
                     (mol_batch.num_graphs, 1, 1))  # set as right-handed in general
    if handedness is not None:  # otherwise, custom
        eye[:, 0, 0] = handedness.flatten()  # flatten - in case it's higher dimensional [n,1]

    rotation_matrix_list = [torch.matmul(torch.linalg.inv(principal_axes_list[i]), eye[i]) for i in
                            range(mol_batch.num_graphs)]

    mol_batch.pos = torch.cat([torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in
                               range(mol_batch.num_graphs)])

    return mol_batch


def random_crystaldata_alignment(crystaldata, include_inversion=False):
    """
    randomize orientation of molecules in a crystaldata object
    """
    coords_list = [crystaldata.pos[crystaldata.ptr[i]:crystaldata.ptr[i + 1]] for i in range(crystaldata.num_graphs)]

    # center at 0
    coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(crystaldata.num_graphs)]

    # optionally invert through the centroid
    if include_inversion:
        invert_inds = np.random.choice([-1, 1], size=crystaldata.num_graphs, replace=True)
        coords_list_centred = [coords * inversion for coords, inversion in zip(coords_list_centred, invert_inds)]

    # random orientation
    rotation_matrix_list = torch.tensor(Rotation.random(num=crystaldata.num_graphs).as_matrix(),
                                        device=crystaldata.x.device, dtype=crystaldata.pos.dtype)
    crystaldata.pos = torch.cat([torch.einsum('ji, mj->mi', (rotation_matrix_list[i], coords_list_centred[i])) for i in
                                 range(crystaldata.num_graphs)])

    return crystaldata


def set_sym_ops(supercell_data):
    """
    return symmetry operators as a list
    @param supercell_data:
    @return:
    """
    sym_ops_list = [
        torch.tensor(supercell_data.symmetry_operators[n], device=supercell_data.x.device, dtype=torch.float32)
        for n in range(len(supercell_data.symmetry_operators))]

    return sym_ops_list, supercell_data


def new_aunit2unit_cell(mol_batch):
    aunit_lens = mol_batch.num_atoms
    symmetry_multiplicity = mol_batch.sym_mult

    aunit_padded_coords_c = rnn.pad_sequence([mol_batch[ind].pos for ind in range(mol_batch.num_graphs)],
                                             batch_first=True)
    centroids_c = get_batch_centroids(mol_batch.pos,
                                      mol_batch.batch,
                                      mol_batch.num_graphs)

    # repeat each molecule Z times
    flat_unit_cell_centroids_c = torch.repeat_interleave(centroids_c, symmetry_multiplicity, dim=0)
    flat_unit_cell_padded_coords_c = torch.repeat_interleave(aunit_padded_coords_c, symmetry_multiplicity, dim=0)

    centroids_f = mol_batch.aunit_centroid
    flat_unit_cell_centroids_f = torch.repeat_interleave(centroids_f, symmetry_multiplicity, dim=0)

    flat_fc_transform_list = torch.repeat_interleave(mol_batch.T_fc, symmetry_multiplicity, dim=0)
    flat_cf_transform_list = torch.repeat_interleave(mol_batch.T_cf, symmetry_multiplicity, dim=0)

    # add 4th dimension as a dummy for affine transforms, should have shape (num_aunits, 4, 4)
    sym_ops = torch.tensor(np.concatenate(mol_batch.symmetry_operators, axis=0),
                           device=centroids_f.device, dtype=torch.float32)

    flat_unit_cell_affine_centroids_f = torch.cat(
        (flat_unit_cell_centroids_f,
         torch.ones(flat_unit_cell_centroids_f.shape[:-1] + (1,), dtype=torch.float32
                    ).to(aunit_padded_coords_c.device)), dim=-1)

    # get all molecule centroids via symmetry ops
    flat_unit_cell_centroids_f = torch.einsum('nij,nj->ni', (sym_ops, flat_unit_cell_affine_centroids_f))[..., :-1]

    # force centroids within unit cell
    flat_centroids_f_in_cell = flat_unit_cell_centroids_f - torch.floor(flat_unit_cell_centroids_f)

    # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
    # todo this is too complicated and important to be done on a single line
    # todo replace this and next fractional transforms with our standard utility
    flat_rot_coords_f = torch.einsum('nmj,nij->nmi',
                                     (torch.einsum('mij,mnj->mni',
                                                   (flat_cf_transform_list,
                                                    flat_unit_cell_padded_coords_c - flat_unit_cell_centroids_c[:, None,
                                                                                     :])),
                                      sym_ops[:, :3, :3]))

    # translate rotated aunits to their correct position
    padded_unit_cells = torch.einsum('mij,mnj->mni',
                                     (flat_fc_transform_list,
                                      flat_rot_coords_f + flat_centroids_f_in_cell[:, None, :]))

    reference_cell_list = []  # recombine everything into their respective crystals
    mol_ind = 0
    for crystal_ind, mult in enumerate(symmetry_multiplicity):
        reference_cell_list.append(padded_unit_cells[mol_ind:mol_ind + mult][:, :aunit_lens[crystal_ind]])
        mol_ind += mult

    return reference_cell_list


def aunit2unit_cell(symmetry_multiplicity,
                    aunit_coords_list,
                    fc_transform_list,
                    cf_transform_list,
                    sym_ops_list,
                    override_aunit=False):
    """  # NEW VERSION - faster. See scratch_5 for testing
    use cell symmetry to pattern asymmetric unit into full unit cell
    'unzip' and 'collect' by Z to do the whole thing in a single parallel pass
    """  # todo better docstring
    num_crystals = len(aunit_coords_list)
    num_aunits = torch.sum(symmetry_multiplicity)
    aunit_lens = torch.tensor([len(aunit_coords_list[ii]) for ii in range(num_crystals)])

    padded_coords_c = rnn.pad_sequence(aunit_coords_list, batch_first=True)
    flat_padded_coords_c = torch.repeat_interleave(padded_coords_c, symmetry_multiplicity, dim=0)

    # todo could be faster with scatter if the aunit_coords_list was continuous
    centroids_c = torch.stack([aunit_coords_list[ii].mean(0) for ii in range(num_crystals)])
    # repeat each molecule Z times
    flat_centroids_c = torch.repeat_interleave(centroids_c, symmetry_multiplicity, dim=0)

    centroids_f = torch.einsum('nij,nj->ni', (cf_transform_list, centroids_c))
    flat_centroids_f = torch.repeat_interleave(centroids_f, symmetry_multiplicity, dim=0)

    flat_fc_transform_list = torch.repeat_interleave(fc_transform_list, symmetry_multiplicity, dim=0)
    flat_cf_transform_list = torch.repeat_interleave(cf_transform_list, symmetry_multiplicity, dim=0)

    sym_ops = torch.cat(sym_ops_list, dim=0).reshape(num_aunits, 4, 4)
    # add 4th dimension as a dummy for affine transforms
    flat_affine_centroids_f = torch.cat(
        (flat_centroids_f, torch.ones(flat_centroids_f.shape[:-1] + (1,)).to(padded_coords_c.device)), dim=-1)

    # get molecule centroids via symmetry ops
    flat_centroids_f = torch.einsum('nij,nj->ni', (sym_ops, flat_affine_centroids_f))[..., :-1]

    # force centroids within unit cell
    flat_centroids_f_in_cell = flat_centroids_f - torch.floor(flat_centroids_f)

    # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
    flat_rot_coords_f = torch.einsum('nmj,nij->nmi',
                                     (torch.einsum('mij,mnj->mni',
                                                   (flat_cf_transform_list,
                                                    flat_padded_coords_c - flat_centroids_c[:, None, :])),
                                      sym_ops[:, :3, :3]))

    # translate rotated aunits to their correct position
    padded_unit_cells = torch.einsum('mij,mnj->mni',
                                     (flat_fc_transform_list,
                                      flat_rot_coords_f + flat_centroids_f_in_cell[:, None, :]))

    reference_cell_list = []  # recombine everything into their respective crystals
    mol_ind = 0
    for crystal_ind, mult in enumerate(symmetry_multiplicity):
        reference_cell_list.append(padded_unit_cells[mol_ind:mol_ind + mult][:, :aunit_lens[crystal_ind]])
        mol_ind += mult

    return reference_cell_list


def old_aunit2unit_cell(symmetry_multiplicity, aunit_coords_list, fc_transform_list, cf_transform_list, sym_ops_list):
    """  # OLD, pretty good, but slower
    use cell symmetry to pattern asymmetric unit into full unit cell
    batch crystals with same Z value together for added speed

    """
    reference_cell_list_i = []

    unique_z_values = torch.unique(symmetry_multiplicity)
    z_inds = [torch.where(symmetry_multiplicity == z)[0] for z in unique_z_values]

    for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
        # padding allows for parallel transforms below
        lens = torch.tensor([len(aunit_coords_list[ii]) for ii in inds])
        padded_coords_c = rnn.pad_sequence(aunit_coords_list, batch_first=True)[inds]
        centroids_c = torch.stack([aunit_coords_list[inds[ii]].mean(0) for ii in range(len(inds))])
        centroids_f = torch.einsum('nij,nj->ni', (cf_transform_list[inds], centroids_c))

        # initialize list of empty tensors [Z, n_crystals, n_atoms, 3]
        ref_cells = torch.zeros((z_value, len(inds), padded_coords_c.shape[1], 3)).to(aunit_coords_list[0].device)
        # get symmetry ops for this batch
        z_sym_ops = torch.stack([sym_ops_list[j] for j in inds])
        # add 4th dimension as a dummy for affine transforms
        affine_centroids_f = torch.cat(
            (centroids_f, torch.ones(centroids_f.shape[:-1] + (1,)).to(padded_coords_c.device)), dim=-1)

        for zv in range(z_value):
            # get molecule centroids via symmetry ops
            centroids_f_z = torch.einsum('nij,nj->ni', (z_sym_ops[:, zv], affine_centroids_f))[..., :-1]

            # force centroids within unit cell
            centroids_f_z_in_cell = centroids_f_z - torch.floor(centroids_f_z)

            # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
            rot_coords_f = torch.einsum('nmj,nij->nmi',
                                        (torch.einsum('mij,mnj->mni',
                                                      (cf_transform_list[inds],
                                                       padded_coords_c - centroids_c[:, None, :])),
                                         z_sym_ops[:, zv, :3, :3]))

            # add final centroid
            ref_cells[zv, :, :, :] = torch.einsum('mij,mnj->mni',
                                                  (fc_transform_list[inds],
                                                   rot_coords_f + centroids_f_z_in_cell[:, None, :]))

        reference_cell_list_i.extend([ref_cells[:, jj, :lens[jj], :] for jj in range(len(inds))])

    sorted_z_inds = torch.argsort(torch.cat(z_inds))

    reference_cell_list = [reference_cell_list_i[ind] for ind in sorted_z_inds]

    return reference_cell_list


def descale_asymmetric_unit(asym_unit_dict, mol_position, sg_inds):
    """
    input fractional coordinates are scaled on 0-1
    rescale these for the specific ranges according to each space group
    only space groups in asym_unit_dict will work - not all have been manually encoded
    this approach will not work for asymmetric units which are not neat parallelpipeds
    Parameters
    ----------
    asym_unit_dict
    mol_position
    sg_inds

    Returns
    -------
    """

    return mol_position * torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])


def rescale_asymmetric_unit(asym_unit_dict, mol_position, sg_inds):
    """
    input fractional coordinates are scaled on 0-max
    rescale these for the specific ranges according to each space group
    only space groups in asym_unit_dict will work - not all have been manually encoded
    this approach will not work for asymmetric units which are not neat parallelpipeds
    Parameters
    ----------
    asym_unit_dict
    mol_position
    sg_inds

    Returns
    -------
    """
    return mol_position / torch.stack([asym_unit_dict[str(int(ind))] for ind in sg_inds])


def overwrite_symmetry_info(mol_batch, generate_sgs, symmetries_dict, randomize_sgs=False):
    """
    update the symmetry information in molecule-wise crystaldata objects
    """  # todo clean this up it's a mess
    # identify the SG numbers we want to generate
    if type(generate_sgs[0]) == str:
        generate_sg_inds = [list(symmetries_dict['space_groups'].values()).index(SG) + 1 for SG in
                            generate_sgs]  # indexing from 0
    elif torch.is_tensor(generate_sgs):
        generate_sg_inds = generate_sgs.cpu().detach().numpy()
    else:
        generate_sg_inds = generate_sgs

    # randomly assign SGs to samples
    if randomize_sgs:
        sample_sg_inds = np.random.choice(generate_sg_inds, size=mol_batch.num_graphs, replace=True)
    elif isinstance(generate_sgs, int):
        sample_sg_inds = np.ones(mol_batch.num_graphs, dtype=int) * generate_sgs
    else:
        sample_sg_inds = generate_sg_inds

    assert len(sample_sg_inds) == mol_batch.num_graphs, "Must have same number of sgs as graphs"

    # update sym ops
    mol_batch.symmetry_operators = [torch.Tensor(symmetries_dict['sym_ops'][sg_ind]).to(mol_batch.x.device) for sg_ind
                                    in
                                    sample_sg_inds]
    mol_batch.sg_ind = torch.tensor(sample_sg_inds,
                                    dtype=torch.long,
                                    device=mol_batch.x.device)
    mol_batch.sym_mult = torch.tensor([len(ops) for ops in mol_batch.symmetry_operators],
                                      dtype=torch.int32,
                                      device=mol_batch.x.device)

    return mol_batch


def get_intra_mol_dists(cell_data, ind):
    # assumes molecules are indexed sequentially in blocks
    coords = cell_data.pos[cell_data.batch == ind]
    coords = coords.reshape(len(coords) // int(cell_data.num_atoms[ind]), int(cell_data.num_atoms[ind]), 3)
    return torch.stack([torch.cdist(coords[i], coords[i]) for i in range(len(coords))])


def set_molecule_alignment(mol_batch,
                           mode: str,
                           right_handed: Optional[bool] = False,
                           include_inversion: Optional[bool] = False):
    """
    set the position and orientation of the molecule with respect to the xyz axis
    'standardized' sets the molecule principal inertial axes equal to the xyz axis
    'random' sets a random orientation of the molecule
    in any case, the molecule centroid is set at (0,0,0)

    option to preserve the handedness of the molecule, e.g., by aligning with
    (x,y,-z) for a left-handed molecule
    """

    if mode == 'standardized':
        mol_batch = align_batch_to_principal_axes(mol_batch,
                                                  handedness=mol_batch.aunit_handedness
                                                  )

    elif mode == 'random':
        mol_batch = random_crystaldata_alignment(mol_batch, include_inversion=include_inversion)
        if right_handed:
            coords_list = [mol_batch.pos[mol_batch.ptr[i]:mol_batch.ptr[i + 1]] for i in range(mol_batch.num_graphs)]
            coords_list_centred = [coords_list[i] - coords_list[i].mean(0) for i in range(mol_batch.num_graphs)]
            principal_axes_list, _, _ = list_molecule_principal_axes_torch(coords_list_centred)
            handedness = compute_Ip_handedness(principal_axes_list)
            for ind, hand in enumerate(handedness):
                if hand == -1:
                    mol_batch.pos[mol_batch.batch == ind] = -mol_batch.pos[mol_batch.batch == ind]  # invert

            mol_batch.aunit_handedness = torch.ones(mol_batch.num_graphs, dtype=torch.long, device=mol_batch.x.device)

    elif mode == 'as is' or mode is None:
        pass  # do nothing

    return mol_batch


def get_aunit_positions(mol_batch,
                        align_to_standardized_orientation: bool = True,
                        mol_handedness: OptTensor = None,
                        ) -> Tensor:
    """
    simplified standalone util version of build_zp_1_asymmetric_unit
    pose asymmetric unit via fractional translations and rotations
    """
    rotations_list = rotvec2rotmat(mol_batch.aunit_orientation)

    if align_to_standardized_orientation:  # align canonical conformers principal axes to cartesian axes - not usually done here, but allowed
        assert mol_handedness is not None, "Must provide explicit handedness when aligning mol to canonical axes"
        mol_batch = align_mol_batch_to_standard_axes(mol_batch, handedness=mol_handedness)
    else:
        mol_batch.recenter_molecules()

    # apply rotations and fractional translations
    mol_batch.pos = (apply_rotation_to_batch(mol_batch.pos,
                                             rotations_list,
                                             mol_batch.batch)
                     + fractional_transform(mol_batch.aunit_centroid, mol_batch.T_fc)[mol_batch.batch])

    return mol_batch.pos

    # re-analyze rotations
    '''
    mol_batch.recenter_molecules()
    eye = torch.tile(torch.eye(3,
                               device=mol_batch.x.device,
                               dtype=torch.float32),
                     (mol_batch.num_graphs, 1, 1)
                     )  # set as right-handed by default

    # alignment vector is (x,y,z) by default, or (-x,y,z) for left-handed structures
    eye[:, 0, 0] = mol_batch.aunit_handedness.flatten()  # flatten - in case it's higher dimensional [n,1]
    from mxtaltools.common.geometry_utils import extract_rotmat

    Ip, Ipm, I = batch_molecule_principal_axes_torch(
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        mol_batch.num_atoms
    )

    rmats = extract_rotmat(eye, torch.linalg.inv(Ip))
    print((rmats.permute(0,2,1) - rotations_list).abs().sum((1,2)))
    '''


def get_symmetry_functions(cell_angles, cell_lengths, mol_position, mol_rotation, supercell_data):
    # get transformation matrices
    T_fc_list, T_cf_list, generated_cell_volumes = batch_compute_fractional_transform(cell_lengths, cell_angles)
    supercell_data.T_fc = T_fc_list
    sym_ops_list, supercell_data = set_sym_ops(supercell_data)  # assign correct symmetry options
    return T_cf_list, T_fc_list, generated_cell_volumes, supercell_data, sym_ops_list


def new_unit_cell_to_convolution_cluster(supercell_data,
                                         cell_vector_list,
                                         sorted_fractional_translations,
                                         device,
                                         cutoff=6,
                                         ):
    """
    Fast/parallel function to pattern unit cells into supercells and then pare to convolution clusters
    1. Generate cartesian translations
    2. Identify maximum required supercell for this batch
    3. Build and index this supercell

    Parameters
    ----------
    unit_cell_pos_list
    cell_vector_list
    crystal_multiplicity
    sorted_fractional_translations
    mol_num_atoms
    mol_radius
    cutoff
    device
    num_graphs

    Returns
    -------

    """
    '''preliminaries'''  # todo ideally this could be done ahead of time

    unit_cell_pos_list = supercell_data.unit_cell_pos
    crystal_multiplicity = supercell_data.sym_mult
    mol_num_atoms = supercell_data.num_atoms
    mol_radius = supercell_data.radius
    num_graphs = supercell_data.num_graphs
    molecule_nodes = supercell_data.x
    molecule_batch = supercell_data.batch

    ucell_num_atoms = mol_num_atoms * crystal_multiplicity
    ucell_atoms_per_mol = mol_num_atoms.repeat_interleave(crystal_multiplicity)
    ucell_num_mols = len(ucell_atoms_per_mol)  # equals also sum of crystal_multiplicity

    unit_cell_pos = torch.cat([torch.tensor(poses.reshape(ucell_num_atoms[ind], 3),
                                            dtype=torch.float32, device=device)
                               for ind, poses in enumerate(unit_cell_pos_list)])
    unit_cell_batch = torch.arange(num_graphs, device=device).repeat_interleave(crystal_multiplicity * mol_num_atoms)

    '''get supercell cartesian translations'''
    cart_translations = torch.einsum('nij, mi -> nmij', cell_vector_list, sorted_fractional_translations).sum(2)
    good_translations_bool = cart_translations.norm(dim=-1) < (mol_radius * 2 + cutoff + 0.1)[:, None]
    good_translations = cart_translations[:, good_translations_bool.any(0)]

    num_cells = good_translations.shape[1]
    supercell_ptr = torch.zeros(num_graphs, dtype=torch.long, device=device)
    supercell_ptr[1:] = torch.cumsum(ucell_num_atoms * num_cells, dim=0)[:-1]

    # test
    # assert torch.all(torch.where(torch.diff(supercell_batch).abs() > 0)[0] + 1 == supercell_ptr[1:]), "ptr indexing failed!"

    # get atom-CC_centroid dists
    canonical_conformer_node_inds = torch.cat([
        torch.arange(mol_num_atoms[ind], device=device) + supercell_ptr[ind] for ind in range(num_graphs)
    ])

    atomwise_cart_translations = good_translations.repeat_interleave(mol_num_atoms * crystal_multiplicity, 0)
    supercell_pos = unit_cell_pos[:, None, :] + atomwise_cart_translations
    supercell_pos = torch.cat([
        supercell_pos[unit_cell_batch == ind].permute(1, 0, 2).reshape(num_cells * ucell_num_atoms[ind], 3) for ind
        in range(num_graphs)
    ])
    supercell_batch = unit_cell_batch.repeat_interleave(num_cells, 0)
    supercell_mol_batch = torch.arange(ucell_num_mols * num_cells, device=device).repeat_interleave(
        ucell_atoms_per_mol.repeat_interleave(num_cells))

    cc_centroids = scatter(supercell_pos[canonical_conformer_node_inds],
                           supercell_batch[canonical_conformer_node_inds],
                           dim=0, dim_size=num_graphs, reduce='mean')

    canonical_conformer_mol_inds = torch.zeros(num_graphs, dtype=torch.long, device=device)
    canonical_conformer_mol_inds[1:] = torch.cumsum(crystal_multiplicity * num_cells, dim=0)[:-1]

    edge_atoms, edge_cc = radius(x=cc_centroids, y=supercell_pos,
                                 batch_x=torch.arange(num_graphs, device=device),
                                 batch_y=supercell_batch,
                                 r=cutoff * 2 * mol_radius.amax() + 0.01,
                                 max_num_neighbors=100)

    dists = torch.linalg.norm(supercell_pos[edge_atoms] - cc_centroids[edge_cc], dim=1)
    relevant_max_radii = mol_radius[supercell_batch[edge_atoms]]
    good_bools = dists < (2 * relevant_max_radii + cutoff + 0.01)

    '''generate reference indices'''
    # 2 for exclude, 1 for convolve, 0 for canonical conformer
    ref_mol_inds = torch.ones_like(supercell_mol_batch)
    # will never be duplicate entries because edges above are strictly many-to-one
    ref_mol_inds[edge_atoms[~good_bools]] = 2
    ref_mol_inds[canonical_conformer_node_inds] = 0

    good_nodes_per_graph = scatter(torch.tensor(ref_mol_inds < 2, dtype=torch.int, device=device), supercell_batch,
                                   reduce='sum', dim=0, dim_size=num_graphs)

    '''generate supercell features array'''
    supercell_nodes = torch.cat([
        molecule_nodes[molecule_batch == ind].repeat(crystal_multiplicity[ind] * num_cells, 1)
        for ind in range(num_graphs)
    ])

    '''update supercell crystaldata object'''
    supercell_data.x = supercell_nodes
    supercell_data.pos = supercell_pos
    supercell_data.aux_ind = ref_mol_inds
    supercell_data.batch = supercell_batch
    supercell_data.ptr = supercell_ptr
    supercell_data.mol_ind = torch.zeros_like(supercell_data.aux_ind)  # mol ind is always 0 for zp=1 cells

    return supercell_data, num_cells
