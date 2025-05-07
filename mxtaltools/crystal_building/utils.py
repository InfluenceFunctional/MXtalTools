import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import rnn as rnn
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import compute_Ip_handedness, get_batch_centroids, \
    rotvec2rotmat, batch_molecule_principal_axes_torch, extract_rotmat, apply_rotation_to_batch, rotmat2rotvec, \
    fractional_transform


def generate_sorted_fractional_translations(supercell_size):
    # initialize fractional translations for supercell construction
    # n_cells = (2 * supercell_size + 1) ** 3
    # fractional_translations = torch.zeros((n_cells, 3))  # initialize the translations in fractional coords
    # i = 0
    # for xx in range(-supercell_size, supercell_size + 1):
    #     for yy in range(-supercell_size, supercell_size + 1):
    #         for zz in range(-supercell_size, supercell_size + 1):
    #             fractional_translations[i] = torch.tensor((xx, yy, zz))
    #             i += 1
    #
    # # sort fractional vectors from closest to furthest from central unit cell
    # return fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]
    xx, yy, zz = torch.meshgrid(
        torch.arange(-supercell_size, supercell_size + 1),
        torch.arange(-supercell_size, supercell_size + 1),
        torch.arange(-supercell_size, supercell_size + 1),
        indexing='ij'
    )

    # Flatten the meshgrid and stack the results into a 2D tensor
    fractional_translations = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).float()
    # sort fractional vectors from closest to furthest from central unit cell
    return fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]


def unit_cell_to_supercell_cluster(crystal_batch, cutoff: float = 6, supercell_size: int = 10):
    cc_centroids = fractional_transform(crystal_batch.aunit_centroid, crystal_batch.T_fc)
    good_translations, good_translations_bool = get_cart_translations(cc_centroids,
                                                                      crystal_batch.T_fc,
                                                                      crystal_batch.radius,
                                                                      cutoff,
                                                                      supercell_size)

    ucell_num_atoms = crystal_batch.num_atoms * crystal_batch.sym_mult
    unit_cell_pos = torch.cat([poses.reshape(ucell_num_atoms[ind], 3) for ind, poses in enumerate(crystal_batch.unit_cell_pos)])
    unit_cell_batch = torch.arange(crystal_batch.num_graphs, device=crystal_batch.device
                                   ).repeat_interleave(crystal_batch.sym_mult * crystal_batch.num_atoms)

    num_cells = good_translations_bool.sum(1)  # == torch.diff(good_translations_ptr).long()

    translations_per_atom = num_cells.repeat_interleave(ucell_num_atoms, dim=0).long()
    atoms_per_translation = ucell_num_atoms.repeat_interleave(num_cells, dim=0)
    atomwise_translation = good_translations.repeat_interleave(atoms_per_translation, dim=0)

    # index cluster features

    # number of atoms per unit cell
    atoms_per_unit_cell = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()

    # counter number of atoms per unit cell, accumulated over crystals, indexed from 0
    apuc_cum = torch.cat(
        [torch.tensor([0], device=crystal_batch.device), torch.cumsum(atoms_per_unit_cell, dim=0)]).long()

    # corresponding unit cell atom for each atom in each cluster
    unit_cell_to_cluster_index = torch.cat([
        (torch.arange(atoms_per_unit_cell[ind], device=crystal_batch.device) + apuc_cum[ind]).repeat(num_cells[ind])
        for ind in range(crystal_batch.num_graphs)
    ])

    # corresponding aunit atom for each atom in each unit cell
    aunit_to_unit_cell_index = torch.cat([
        (torch.arange(crystal_batch.num_atoms[ind], device=crystal_batch.device) + crystal_batch.ptr[ind]).repeat(
            crystal_batch.sym_mult[ind])
        for ind in range(crystal_batch.num_graphs)
    ])

    # build cluster by reindexing unit cells
    cluster_batch = crystal_batch.clone()

    cluster_batch.pos = unit_cell_pos[unit_cell_to_cluster_index]
    cluster_batch.pos += atomwise_translation

    cluster_batch.z = crystal_batch.z[aunit_to_unit_cell_index][unit_cell_to_cluster_index]
    cluster_batch.x = crystal_batch.x[aunit_to_unit_cell_index][unit_cell_to_cluster_index]

    cluster_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=crystal_batch.device),
                                   torch.cumsum(ucell_num_atoms * num_cells, dim=0)]).long()

    cluster_batch.batch = unit_cell_batch.repeat_interleave(translations_per_atom, dim=0)

    cluster_batch.aux_ind = torch.ones_like(cluster_batch.z)
    cluster_batch.mol_ind = torch.ones_like(cluster_batch.z)
    for ind in range(cluster_batch.num_graphs):
        cluster_batch.aux_ind[cluster_batch.ptr[ind]:cluster_batch.ptr[ind] + cluster_batch.num_atoms[ind]] = 0
        cluster_batch.mol_ind[cluster_batch.ptr[ind]:cluster_batch.ptr[ind + 1]] = (
            torch.arange(num_cells[ind] * crystal_batch.sym_mult[ind], device=crystal_batch.device).repeat_interleave(
                crystal_batch.num_atoms[ind]))

    # we then can further pare by molecule centroid, since our original box calculation is rather permissive
    # get all mol centroids
    mols_per_cluster = num_cells * cluster_batch.sym_mult
    # indexes which molecule each atom belongs to
    flat_mol_inds = cluster_batch.mol_ind + torch.cat([torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
                                                       torch.cumsum(mols_per_cluster, dim=0)])[:-1].repeat_interleave(
        torch.diff(cluster_batch.ptr), dim=0)
    mol_centroids = scatter(cluster_batch.pos, flat_mol_inds, reduce='mean', dim=0)
    cc_centroids = scatter(cluster_batch.pos[cluster_batch.aux_ind == 0], crystal_batch.batch, reduce='mean', dim=0)

    # get the mol indices within the widest conv cutoff
    molwise_batch = torch.arange(len(cc_centroids), device=cc_centroids.device).repeat_interleave(mols_per_cluster,
                                                                                                  dim=0)
    mol_inds = torch.arange(len(molwise_batch), device=cluster_batch.device)
    mol_to_cc_dist = torch.linalg.norm(mol_centroids - cc_centroids.repeat_interleave(mols_per_cluster, dim=0), dim=1)

    # finer mol dependent cutoff
    mol_cutoff = (cutoff + 2 * crystal_batch.radius + 0.1)[molwise_batch[mol_inds]]

    # molecules to keep
    bad_mol_inds = mol_inds[mol_to_cc_dist > mol_cutoff]
    #_, filtered_mols_per_cluster = torch.unique(molwise_batch[bad_mol_inds], return_counts=True)
    # convert back to atomic indices
    # atom for each molecule in the kept list
    atoms_in_bad_mols_bool = torch.isin(flat_mol_inds, bad_mol_inds)

    cluster_batch.aux_ind[atoms_in_bad_mols_bool] = 2  # marker for "outside convolutional field"

    return cluster_batch

    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='unit cell')
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='convolve with')


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
    cutoff_distance = mol_radii * 2 + cutoff + 0.1 + cell_diag  # convolutional radius plus cell diagonal

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


def find_coord_in_box_torch(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return torch.where((coords[:, 0] <= (box[0] + epsilon)) *
                       (coords[:, 1] <= (box[1] + epsilon)) *
                       (coords[:, 2] <= (box[2] + epsilon)) *
                       (coords[:, 0] >= 0) *
                       (coords[:, 1] >= 0) *
                       (coords[:, 2] >= 0)
                       )[0]


def parameterize_crystal_batch(crystal_batch,
                               asym_unit_dict,
                               enforce_right_handedness: bool = False,
                               return_aunit: bool = False,
                               ):
    """
    Asymmetric Unit Centroid Analysis
    identify "canonical" asymmetric unit out of the Z asymmetric units in the unit cell
    """
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
                               device=mol_batch.pos.device,
                               dtype=torch.float32),
                     (mol_batch.num_graphs, 1, 1)
                     )  # set as right-handed by default

    if handedness is not None:  # otherwise, custom
        # alignment vector is (x,y,z) by default, or (-x,y,z) for left-handed structures
        if torch.is_tensor(handedness):
            eye[:, 0, 0] = handedness.flatten()  # flatten - in case it's higher dimensional [n,1]
        else:
            eye[:, 0, 0] = handedness

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


def aunit2unit_cell(mol_batch):
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
