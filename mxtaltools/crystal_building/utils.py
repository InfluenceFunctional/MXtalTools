import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import rnn as rnn
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import compute_Ip_handedness, get_batch_centroids, \
    rotvec2rotmat, batch_molecule_principal_axes_torch, extract_rotmat, apply_rotation_to_batch, rotmat2rotvec, \
    fractional_transform, center_mol_batch
from mxtaltools.common.utils import block_repeat_interleave


def generate_sorted_fractional_translations(supercell_size):
    xrange = torch.arange(-supercell_size, supercell_size + 1, dtype=torch.int16)
    xx, yy, zz = torch.meshgrid(xrange, xrange, xrange,
                                indexing='ij'
                                )

    # Flatten the meshgrid and stack the results into a 2D tensor
    fractional_translations = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1).float()
    # sort fractional vectors from closest to furthest from central unit cell
    return fractional_translations[torch.argsort(fractional_translations.abs().sum(1))]


def unit_cell_to_supercell_cluster(crystal_batch, cutoff: float = 6, supercell_size: int = 10):
    """ # TODO this is a key bottleneck. It would be great if we could speed it up or obviate it.

    """
    # pre-compute useful stuff
    cc_centroids = fractional_transform(crystal_batch.aunit_centroid, crystal_batch.T_fc)
    ucell_num_atoms = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()
    # counter number of atoms per unit cell, accumulated over crystals, indexed from 0
    apuc_cum = torch.cat(
        [torch.tensor([0], device=crystal_batch.device),
         torch.cumsum(ucell_num_atoms, dim=0)]).long()

    # get initial cluster shell
    good_translations, good_translations_bool = get_cart_translations(cc_centroids,
                                                                      crystal_batch.T_fc,
                                                                      crystal_batch.radius,
                                                                      cutoff,
                                                                      supercell_size)
    cells_per_cluster = good_translations_bool.sum(1)

    # get flat unit cell pos and batch
    # todo obviate this slow listcomp step by feeding the unit cells flat in the first place
    unit_cell_pos = torch.cat(
        [poses.reshape(ucell_num_atoms[ind], 3) for ind, poses in enumerate(crystal_batch.unit_cell_pos)])
    unit_cell_batch = torch.arange(crystal_batch.num_graphs, device=crystal_batch.device).repeat_interleave(
        crystal_batch.sym_mult * crystal_batch.num_atoms)

    # get atomwise translations
    # TODO carefully see if these are necessary
    translations_per_atom = cells_per_cluster.repeat_interleave(ucell_num_atoms, dim=0).long()
    atoms_per_translation = ucell_num_atoms.repeat_interleave(cells_per_cluster, dim=0)
    atomwise_translation = good_translations.repeat_interleave(atoms_per_translation, dim=0)

    # these two replacements are maybe marginally faster
    # # corresponding unit cell atom for each atom in each cluster
    # unit_cell_to_cluster_index = torch.cat([
    #     (torch.arange(ucell_num_atoms[ind], device=crystal_batch.device) + apuc_cum[ind]).repeat(num_cells[ind])
    #     for ind in range(crystal_batch.num_graphs)
    # ])
    atoms_per_crystal = ucell_num_atoms * cells_per_cluster  # total atoms per crystal
    tot_num_atoms = atoms_per_crystal.sum()

    # per-atom offsets
    base = torch.arange(tot_num_atoms, device=crystal_batch.device)
    # map into per-crystal block
    block_ids = torch.repeat_interleave(torch.arange(len(atoms_per_crystal), device=crystal_batch.device),
                                        atoms_per_crystal)

    # relative index within each crystal’s block
    in_block = base - \
               torch.cumsum(torch.cat([torch.tensor([0], device=crystal_batch.device), atoms_per_crystal[:-1]]), dim=0)[
                   block_ids]

    # now mod by ucell_num_atoms and add offset
    unit_cell_to_cluster_index = (in_block % ucell_num_atoms[block_ids]) + apuc_cum[block_ids]
    # corresponding aunit atom for each atom in each unit cell
    # aunit_to_unit_cell_index = torch.cat([
    #     (torch.arange(crystal_batch.num_atoms[ind], device=crystal_batch.device) + crystal_batch.ptr[ind]).repeat(
    #         crystal_batch.sym_mult[ind])
    #     for ind in range(crystal_batch.num_graphs)
    # ])
    device = crystal_batch.device
    num_graphs = crystal_batch.num_graphs

    # length of each block before repetition
    block_lens = crystal_batch.num_atoms  # [G]
    sym_mult = crystal_batch.sym_mult  # [G]
    ptrs = crystal_batch.ptr[:-1]  # [G]

    # total elements after repetition
    #tot_len = torch.sum(block_lens * sym_mult)

    # assign each output element to its graph
    graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=device),
                                        block_lens * sym_mult)

    # within-block index (0..num_atoms[i]-1), repeated for sym_mult[i]
    in_block = torch.cat([
        torch.arange(n, device=device).repeat(m)
        for n, m in zip(block_lens.tolist(), sym_mult.tolist())
    ])

    # add ptr offset per graph
    aunit_to_unit_cell_index = in_block + ptrs[graph_ids]

    """
    Construct Batch of Supercells/Clusters
    """  # TODO pare down to what is strictly necessary
    cluster_batch = crystal_batch.clone()

    # init cluster with right number of duplicated unit cells
    cluster_batch.pos = unit_cell_pos[unit_cell_to_cluster_index]
    # build cluster by translating all atoms manually
    cluster_batch.pos += atomwise_translation

    # this indexes from the molecule (aunit) to all its symmetry images
    aunit_to_cluster_index = aunit_to_unit_cell_index[unit_cell_to_cluster_index]

    # pattern out atom properties
    cluster_batch.z = crystal_batch.z[aunit_to_cluster_index]
    cluster_batch.x = crystal_batch.x[aunit_to_cluster_index]

    # rebuild batch indices
    cluster_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=crystal_batch.device),
                                   torch.cumsum(atoms_per_crystal, dim=0)]).long()
    cluster_batch.batch = unit_cell_batch.repeat_interleave(translations_per_atom, dim=0)

    mols_per_cluster = cells_per_cluster * cluster_batch.sym_mult
    # cluster_batch.mol_ind = torch.ones_like(cluster_batch.z)
    # for ind in range(cluster_batch.num_graphs):  # this part is rather slow still
    #     cluster_batch.mol_ind[cluster_batch.ptr[ind]:cluster_batch.ptr[ind + 1]] = (
    #         torch.arange(mols_per_cluster[ind], device=crystal_batch.device).repeat_interleave(
    #             crystal_batch.num_atoms[ind]))

    cry_int_mol_batch = torch.cat([torch.arange(elem) for elem in mols_per_cluster]).to(crystal_batch.device)
    cluster_batch.mol_ind = cry_int_mol_batch.repeat_interleave(
        crystal_batch.num_atoms.repeat_interleave(mols_per_cluster))

    # cluster_batch.aux_ind = torch.ones_like(cluster_batch.z)
    # mask = torch.cat([
    #     torch.arange(cluster_batch.num_atoms[i], device=cluster_batch.device) + cluster_batch.ptr[i]
    #     for i in range(cluster_batch.num_graphs)
    # ])
    # cluster_batch.aux_ind[mask] = 0
    cluster_batch.aux_ind = torch.zeros_like(cluster_batch.z)
    cluster_batch.aux_ind[cluster_batch.mol_ind != 0] = 1  # MK this may have to be adjusted for Z'>1
    """
    we then can further pare by molecule centroid, since our original box calculation is rather permissive
    """
    # get all mol centroids
    # indexes which molecule each atom belongs to, globally (not within-crystal)
    molwise_ptr = torch.cat([torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
                             torch.cumsum(mols_per_cluster, dim=0)])[:-1]
    flat_mol_inds = cluster_batch.mol_ind + molwise_ptr.repeat_interleave(torch.diff(cluster_batch.ptr), dim=0)

    mol_centroids = scatter(cluster_batch.pos, flat_mol_inds, reduce='mean', dim=0)
    cc_centroids = scatter(crystal_batch.pos, crystal_batch.batch, reduce='mean', dim=0)

    # get the mol indices within the widest conv cutoff
    molwise_batch = (torch.arange(crystal_batch.num_graphs, device=cc_centroids.device)
                     .repeat_interleave(mols_per_cluster, dim=0))
    mol_inds = torch.arange(len(molwise_batch), device=cluster_batch.device)

    mol_to_cc_dist_sq = (mol_centroids - cc_centroids.repeat_interleave(mols_per_cluster, dim=0)).square().sum(1)
    mol_cutoff_sq = (cutoff + 2 * crystal_batch.radius + 0.1)[molwise_batch[mol_inds]].square()

    # molecules to keep
    bad_mol_inds = mol_inds[mol_to_cc_dist_sq > mol_cutoff_sq]
    #_, filtered_mols_per_cluster = torch.unique(molwise_batch[bad_mol_inds], return_counts=True)
    # convert back to atomic indices
    # atom for each molecule in the kept list
    atoms_in_bad_mols_bool = torch.isin(flat_mol_inds, bad_mol_inds)

    cluster_batch.aux_ind[atoms_in_bad_mols_bool] = 2  # marker for "outside convolutional field"

    return cluster_batch

    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='unit cell')
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='convolve with')


def new_unit_cell_to_supercell_cluster(crystal_batch, cutoff: float = 6, supercell_size: int = 10):
    # pre-compute useful stuff
    cc_centroids = fractional_transform(crystal_batch.aunit_centroid, crystal_batch.T_fc)
    ucell_num_atoms = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()

    # TODO error right here - the first unit cell molecule pos is not lining up with the cc centroids
    # this is causing downstream confusion and errors
    # the aunit centroid on the other hand, is perfect
    # cc = scatter(aup, aub, reduce='mean', dim=0)
    # cc2 = scatter(crystal_batch.pos, crystal_batch.batch, reduce='mean', dim=0)
    # the issue comes from unit cell construction, when the aunit position is at the edge of the box (1)
    # we need to go back to our floor / box constraint function in unit cell construction, and make it so the aunit is never moved
    unit_cell_pos = crystal_batch.unit_cell_pos
    unit_cell_batch = crystal_batch.unit_cell_batch
    unit_cell_ptr = torch.cat(
        [torch.tensor([0], device=crystal_batch.device),
         torch.cumsum(ucell_num_atoms, dim=0)]).long()

    # get initial cluster shell
    good_translations, good_translations_bool = get_cart_translations(cc_centroids,
                                                                      crystal_batch.T_fc,
                                                                      crystal_batch.radius,
                                                                      cutoff,
                                                                      supercell_size)
    # some more constants
    cells_per_cluster = good_translations_bool.sum(1)
    atoms_per_crystal = ucell_num_atoms * cells_per_cluster  # total atoms per crystal
    tot_num_atoms = atoms_per_crystal.sum()

    # get atomwise translations
    translations_per_atom = cells_per_cluster.repeat_interleave(ucell_num_atoms, dim=0).long()
    atoms_per_translation = ucell_num_atoms.repeat_interleave(cells_per_cluster, dim=0)
    atomwise_translation = good_translations.repeat_interleave(atoms_per_translation, dim=0)


    supercell_atom_ind = torch.arange(tot_num_atoms, device=crystal_batch.device)
    supercell_batch = torch.repeat_interleave(torch.arange(crystal_batch.num_graphs,
                                                     device=crystal_batch.device),
                                              atoms_per_crystal)
    supercell_ptr = torch.cumsum(torch.cat([torch.tensor([0], device=crystal_batch.device), atoms_per_crystal[:-1]]), dim=0)
    within_crystal_index = supercell_atom_ind - supercell_ptr[supercell_batch]
    atom_in_unit_cell_batch = (within_crystal_index % ucell_num_atoms[supercell_batch]) + unit_cell_ptr[supercell_batch]

    # assign each output element to its graph
    graph_ids = torch.repeat_interleave(torch.arange(crystal_batch.num_graphs, device=crystal_batch.device),
                                        crystal_batch.num_atoms * crystal_batch.sym_mult)

    # within-block index (0..num_atoms[i]-1), repeated for sym_mult[i]
    within_crystal_index = torch.cat([
        torch.arange(n, device=crystal_batch.device).repeat(m)
        for n, m in zip(crystal_batch.num_atoms.tolist(), crystal_batch.sym_mult.tolist())
    ])

    # add ptr offset per graph
    aunit_to_unit_cell_index = within_crystal_index + crystal_batch.ptr[:-1][graph_ids]

    """
    Construct Batch of Supercells/Clusters
    """  # TODO pare down to what is strictly necessary
    cluster_batch = crystal_batch.clone()

    # init cluster with right number of duplicated unit cells
    cluster_batch.pos = unit_cell_pos[atom_in_unit_cell_batch]
    # build cluster by translating all atoms manually
    cluster_batch.pos += atomwise_translation

    # this indexes from the molecule (aunit) to all its symmetry images
    aunit_to_cluster_index = aunit_to_unit_cell_index[atom_in_unit_cell_batch]

    # pattern out atom properties
    cluster_batch.z = crystal_batch.z[aunit_to_cluster_index]
    cluster_batch.x = crystal_batch.x[aunit_to_cluster_index]

    # rebuild batch indices
    cluster_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=crystal_batch.device),
                                   torch.cumsum(atoms_per_crystal, dim=0)]).long()
    cluster_batch.batch = unit_cell_batch.repeat_interleave(translations_per_atom, dim=0)

    mols_per_cluster = cells_per_cluster * cluster_batch.sym_mult

    lengths = mols_per_cluster
    offsets = torch.cumsum(torch.cat([torch.zeros(1, device=lengths.device, dtype=lengths.dtype),
                                      lengths[:-1]]), 0)

    # total size
    total = lengths.sum()

    # which cluster each entry belongs to
    cluster_ids = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device),
                                          lengths)

    # index *within* the cluster
    cry_int_mol_batch = torch.arange(total, device=lengths.device) - offsets[cluster_ids]

    cluster_batch.mol_ind = cry_int_mol_batch.repeat_interleave(
        crystal_batch.num_atoms.repeat_interleave(mols_per_cluster))

    cluster_batch.aux_ind = torch.zeros_like(cluster_batch.z)
    cluster_batch.aux_ind[cluster_batch.mol_ind != 0] = 1  # MK this may have to be adjusted for Z'>1
    """
    we then can further pare by molecule centroid, since our original box calculation is rather permissive
    """
    # get all mol centroids
    # indexes which molecule each atom belongs to, globally (not within-crystal)
    molwise_ptr = torch.cat([torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
                             torch.cumsum(mols_per_cluster, dim=0)])[:-1]
    atom_to_flat_mol_ind = cluster_batch.mol_ind + molwise_ptr.repeat_interleave(atoms_per_crystal, dim=0)

    # this is extremely memory heavy with float32, we can make huge savings with half/float16
    mol_centroids = scatter(cluster_batch.pos.half(), atom_to_flat_mol_ind, reduce='mean', dim=0)
    cc_centroids = scatter(crystal_batch.pos.half(), crystal_batch.batch, reduce='mean', dim=0)

    # get the mol indices within the widest conv cutoff
    molwise_batch = (torch.arange(crystal_batch.num_graphs, device=cc_centroids.device)
                     .repeat_interleave(mols_per_cluster, dim=0))
    mol_inds = torch.arange(len(molwise_batch), device=cluster_batch.device)

    # get the distances
    mol_to_cc_dist_sq = (mol_centroids - cc_centroids.repeat_interleave(mols_per_cluster, dim=0)).square().sum(1)
    cluster_cutoff = (cutoff + 2 * crystal_batch.radius + 0.1).square()
    mol_cutoff_sq = cluster_cutoff[molwise_batch]

    # molecules to discard
    bad_mol_inds = mol_inds[mol_to_cc_dist_sq > mol_cutoff_sq]
    # convert back to atomic indices
    # atom for each molecule in the kept list
    # atoms_in_bad_mols_bool = torch.isin(atom_to_flat_mol_ind, bad_mol_inds)
    # dramatically faster alternative to isin
    #bad_sorted = bad_mol_inds.sort().values

    # atom_to_mol index of each atom which is in a 'bad' molecule
    # this works because atom_to_flat_mol_ind is already sorted
    idx = torch.searchsorted(bad_mol_inds, atom_to_flat_mol_ind)  # sorting step
    valid = idx < bad_mol_inds.numel()  # safety check for wayward searchsorted outputs > length of tensors
    atoms_in_bad_mols_bool = torch.zeros_like(atom_to_flat_mol_ind, dtype=torch.bool)
    atoms_in_bad_mols_bool[valid] = bad_mol_inds[idx[valid]] == atom_to_flat_mol_ind[valid] # atom is actually in a bad mol

    cluster_batch.aux_ind[atoms_in_bad_mols_bool] = 2  # marker for "outside convolutional field"

    return cluster_batch

    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='unit cell')
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='convolve with')

def get_cart_translations(cc_centroids,
                          T_fc,
                          mol_radii,
                          cutoff,
                          supercell_size: int = 9):
    # precompute some stuff
    # get the set of all possible cartesian translations
    cell_vectors = T_fc.permute(0, 2, 1)
    # cell diagonals
    cell_diag = cell_vectors.sum(1).norm(dim=-1)
    # bounding box cutoff
    cutoff_distance = mol_radii * 2 + cutoff + 0.1 + cell_diag  # convolutional radius plus cell diagonal

    # identify actual supercell size
    box_lengths = torch.linalg.norm(cell_vectors, dim=-1)
    max_translations_per_dim = int(torch.ceil(cutoff_distance[:, None] / box_lengths).amax())
    actual_supercell_size = min(supercell_size, max_translations_per_dim)  # use the argument size as a cap

    sorted_fractional_translations = generate_sorted_fractional_translations(actual_supercell_size).to(T_fc.device)

    # for the n samples, for the 3 (i) cell vectors, get the inner product with all the fractional translation vectors
    parallelpiped_centroids = torch.einsum('nij, mi -> nmj', cell_vectors, sorted_fractional_translations)

    # we need to include any parallelpiped which overlaps with a sphere of
    # radius cutoff_distance around the canonical conformer for each cell - a bounding box calculation
    # we compute this in a brute force way, adding the diagonal length of each parallelpiped to the cutoff
    # this will take extra boxes, but is fast and simple
    # centroid_dists = torch.linalg.norm(cc_centroids[:, None, :] - parallelpiped_centroids, dim=-1)
    # good_translations_bool = centroid_dists < cutoff_distance[:, None]
    # this is faster without the sqrt, and comparing against the squared dist
    centroid_dists = (cc_centroids[:, None, :] - parallelpiped_centroids).square().sum(dim=-1)
    good_translations_bool = centroid_dists < (cutoff_distance[:, None].square())

    # build only the unit cells which are relevant to each cluster
    good_translations_bool[:, 0] = True  # always take the 0,0,0 element
    good_translations = parallelpiped_centroids[good_translations_bool]

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

    # MK TODO see if this could be done without the listcomp
    aunit_padded_coords_c = rnn.pad_sequence([mol_batch[ind].pos for ind in range(mol_batch.num_graphs)],
                                             batch_first=True)
    centroids_c = get_batch_centroids(mol_batch.pos,
                                      mol_batch.batch,
                                      mol_batch.num_graphs)
    centroids_f = mol_batch.aunit_centroid

    # repeat each molecule Z times
    # this pattern with shared indexing is faster for larger samples
    repeat_index = torch.arange(len(symmetry_multiplicity), device=symmetry_multiplicity.device) \
        .repeat_interleave(symmetry_multiplicity)

    flat_unit_cell_centroids_c = centroids_c[repeat_index]
    flat_unit_cell_padded_coords_c = aunit_padded_coords_c[repeat_index]
    flat_unit_cell_centroids_f = mol_batch.aunit_centroid[repeat_index]
    flat_fc_transform_list = mol_batch.T_fc[repeat_index]
    flat_cf_transform_list = mol_batch.T_cf[repeat_index]

    # MK todo this tensor(np.concatenate) can't be efficient
    sym_ops = torch.tensor(np.concatenate(mol_batch.symmetry_operators, axis=0),
                           device=centroids_f.device, dtype=torch.float32)

    # add 4th dimension as a dummy for affine transforms, should have shape (num_aunits, 4, 4)
    flat_unit_cell_affine_centroids_f = torch.cat(
        (
            flat_unit_cell_centroids_f,
            torch.ones(flat_unit_cell_centroids_f.shape[:-1] + (1,), dtype=torch.float32)
            .to(aunit_padded_coords_c.device)),
        dim=-1)

    # get all molecule centroids via symmetry ops
    flat_unit_cell_centroids_f = torch.einsum('nij,nj->ni', (sym_ops, flat_unit_cell_affine_centroids_f))[..., :-1]

    # force centroids within unit cell
    flat_centroids_f_in_cell = flat_unit_cell_centroids_f - torch.floor(flat_unit_cell_centroids_f)

    # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
    # todo replace below with our standard transforms
    centered_padded_coords_c = flat_unit_cell_padded_coords_c - flat_unit_cell_centroids_c[:, None, :]
    centered_padded_coords_f = torch.einsum('mij,mnj->mni', (flat_cf_transform_list, centered_padded_coords_c))
    flat_rot_coords_f = torch.einsum('nmj,nij->nmi', (centered_padded_coords_f, sym_ops[:, :3, :3]))

    # translate rotated aunits to their correct position
    padded_unit_cells = torch.einsum('mij,mnj->mni',
                                     (flat_fc_transform_list,
                                      flat_rot_coords_f + flat_centroids_f_in_cell[:, None, :]))

    # this MUST be very slow
    # MK TODO replace this with a flat batched list, or at least delete the padding without this manual step
    reference_cell_list = []  # recombine everything into their respective crystals
    mol_ind = 0
    for crystal_ind, mult in enumerate(symmetry_multiplicity):
        reference_cell_list.append(padded_unit_cells[mol_ind:mol_ind + mult][:, :aunit_lens[crystal_ind]])
        mol_ind += mult

    return reference_cell_list


def new_aunit2unit_cell(mol_batch):
    """
    :param mol_batch:
    :return:
    """
    aunit_lens = mol_batch.num_atoms
    symmetry_multiplicity = mol_batch.sym_mult
    atoms_per_unit_cell = mol_batch.num_atoms * symmetry_multiplicity
    tot_num_mols = torch.sum(symmetry_multiplicity)

    aunit_coords_c = mol_batch.pos
    centroids_f = mol_batch.aunit_centroid

    # repeat each molecule Z times
    molwise_repeat_index = torch.arange(len(symmetry_multiplicity), device=symmetry_multiplicity.device) \
        .repeat_interleave(symmetry_multiplicity)

    unit_cell_centroids_f = mol_batch.assign_aunit_centroid(mol_batch.aunit_centroid[molwise_repeat_index])
    fc_transform_list = mol_batch.T_fc
    cf_transform_list = mol_batch.T_cf

    # same, atomwise
    atomwise_repeat_index = block_repeat_interleave(mol_batch.num_atoms,
                                                    symmetry_multiplicity)

    unit_cell_coords_c = aunit_coords_c[atomwise_repeat_index]
    unit_cell_batch = mol_batch.batch[atomwise_repeat_index]

    # assign which molecule each atom is in
    atom_in_mol_batch = torch.arange(tot_num_mols, device=mol_batch.device).repeat_interleave(aunit_lens.repeat_interleave(symmetry_multiplicity))

    # MK todo this tensor(np.concatenate) can't be efficient
    sym_ops = torch.tensor(np.concatenate(mol_batch.symmetry_operators, axis=0),
                           device=centroids_f.device, dtype=torch.float32)

    # add 4th dimension as a dummy for affine transforms, should have shape (num_aunits, 4, 4)
    unit_cell_affine_centroids_f = torch.cat(
        (
            unit_cell_centroids_f,
            torch.ones(unit_cell_centroids_f.shape[:-1] + (1,), dtype=torch.float32)
            .to(aunit_coords_c.device)),
        dim=-1)

    # get all molecule centroids via symmetry ops
    unit_cell_centroids_f = torch.einsum('nij,nj->ni', (sym_ops, unit_cell_affine_centroids_f))[..., :-1]

    # force centroids within unit cell
    centroids_f_in_cell = unit_cell_centroids_f - torch.floor(unit_cell_centroids_f)

    # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
    centered_coords_c = center_mol_batch(unit_cell_coords_c,
                                                unit_cell_batch,
                                                mol_batch.num_graphs,
                                                atoms_per_unit_cell)
    centered_coords_f = fractional_transform(centered_coords_c,
                                             cf_transform_list[unit_cell_batch])
    rot_coords_f = torch.einsum('nj,nij->ni', (centered_coords_f, sym_ops[atom_in_mol_batch, :3, :3]))

    # translate rotated aunits to their correct position
    unit_cell_pos = torch.einsum('nij,nj->ni',
                                     (fc_transform_list[unit_cell_batch],
                                      rot_coords_f + centroids_f_in_cell[atom_in_mol_batch, :]))

    return unit_cell_pos, unit_cell_batch, atom_in_mol_batch


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
