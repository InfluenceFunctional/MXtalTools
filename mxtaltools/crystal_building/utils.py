import warnings
from typing import Union

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import compute_Ip_handedness, rotvec2rotmat, batch_molecule_principal_axes_torch, \
    extract_rotmat, apply_rotation_to_batch, rotmat2rotvec, \
    fractional_transform, center_batch
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


def ucell2cluster(crystal_batch, cutoff: float = 6,
                  supercell_size: int = 10,
                  zp_buffer: Union[float, torch.Tensor] = 0):  # todo this could be modularized/simplified
    """
    1) generate supercell cluster including only unit cells which could plausibly interact with the asymmetric unit
    2) instantiate the cluster
    3) pare cluster to asymmetric units which actually interact with the asymmetric unit
    """
    """
    Generate set of unit cells to instantiate
    """
    # this method is simple but depends on the canonical conformer being properly set as the raw position
    cc_centroids = scatter(crystal_batch.pos, crystal_batch.batch,
                           dim_size=crystal_batch.num_graphs, dim=0, reduce='mean')
    good_translations, ucells_per_cluster = get_cart_translations(cc_centroids,
                                                                  crystal_batch.T_fc,
                                                                  crystal_batch.radius,
                                                                  cutoff + zp_buffer,
                                                                  supercell_size)

    """
    Build constants & indices
    """
    # unit cell
    atoms_per_ucell = (crystal_batch.num_atoms * crystal_batch.sym_mult).long()

    # cluster
    atoms_per_cluster = atoms_per_ucell * ucells_per_cluster  # total atoms per crystal
    aunits_per_cluster = ucells_per_cluster * crystal_batch.sym_mult

    # get atomwise translations
    atoms_per_translation = atoms_per_ucell.repeat_interleave(ucells_per_cluster, dim=0)
    atomwise_translation = good_translations.repeat_interleave(atoms_per_translation, dim=0)

    """
    Instantiate explicit clusters
    """
    cluster_batch = instantiate_cluster(atoms_per_ucell,
                                        atoms_per_cluster,
                                        atomwise_translation,
                                        ucells_per_cluster,
                                        crystal_batch,
                                        )

    """
    we then can further pare by molecule centroid, since our original box calculation is rather permissive
    """
    cluster_batch = _pare_cluster_molwise(atoms_per_cluster, cc_centroids,
                                          cluster_batch, crystal_batch,
                                          cutoff + zp_buffer, aunits_per_cluster)

    return cluster_batch
    # zpg_inds = torch.argwhere(cluster_batch.z_prime > 1).flatten().tolist()
    # cluster_batch.visualize(zpg_inds, mode='distance', cutoff=4, highlight_canonical_conformer=True)
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='unit cell')
    # cluster_batch.visualize([1, 2, 3, 4, 5], mode='convolve with')


def _pare_cluster_molwise(atoms_per_cluster, cc_centroids, cluster_batch,
                          crystal_batch, cutoff, aunits_per_cluster):
    # get all mol centroids
    # indexes which molecule each atom belongs to, globally (not within-crystal)
    aunit_ptr = torch.cat([torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
                           torch.cumsum(aunits_per_cluster, dim=0)])[:-1]
    atom2aunit_ind = cluster_batch.mol_ind + aunit_ptr.repeat_interleave(atoms_per_cluster, dim=0)
    # this is extremely memory heavy with float32, we can make huge savings with half/float16
    aunit_centroids = scatter(cluster_batch.pos.half(), atom2aunit_ind, reduce='mean', dim=0)
    # cc_centroids = scatter(crystal_batch.pos.half(), crystal_batch.batch, reduce='mean', dim=0) # already compute this above
    # get the mol indices within the widest conv cutoff
    molwise_batch = (torch.arange(crystal_batch.num_graphs, device=cc_centroids.device)
                     .repeat_interleave(aunits_per_cluster, dim=0))
    mol_inds = torch.arange(len(molwise_batch), device=cluster_batch.device)
    # get the distances
    mol_to_cc_dist_sq = (aunit_centroids - cc_centroids.repeat_interleave(aunits_per_cluster, dim=0)).square().sum(1)
    cluster_cutoff = (cutoff + 2 * crystal_batch.radius + 0.1).square()
    mol_cutoff_sq = cluster_cutoff[molwise_batch]
    # molecules to discard
    bad_mol_inds = mol_inds[mol_to_cc_dist_sq > mol_cutoff_sq]
    # convert back to atomic indices
    # atom for each molecule in the kept list
    # atoms_in_bad_mols_bool = torch.isin(atom_to_flat_mol_ind, bad_mol_inds)
    # dramatically faster alternative to isin
    # bad_sorted = bad_mol_inds.sort().values
    # atom_to_mol index of each atom which is in a 'bad' molecule
    # this works because atom_to_flat_mol_ind is already sorted
    idx = torch.searchsorted(bad_mol_inds, atom2aunit_ind)  # sorting step
    valid = idx < bad_mol_inds.numel()  # safety check for wayward searchsorted outputs > length of tensors
    atoms_in_bad_mols_bool = torch.zeros_like(atom2aunit_ind, dtype=torch.bool)
    atoms_in_bad_mols_bool[valid] = bad_mol_inds[idx[valid]] == atom2aunit_ind[valid]  # atom is actually in a bad mol
    cluster_batch.aux_ind[atoms_in_bad_mols_bool] = 2  # marker for "outside convolutional field"

    return cluster_batch


def instantiate_cluster(atoms_per_ucell,
                        atoms_per_cluster,
                        atomwise_translation,
                        ucells_per_cluster,
                        crystal_batch):
    """"""
    """
    Generate necessary indices to instantiate clusters in parallel
    """
    unit_cell_batch = crystal_batch.unit_cell_batch
    unit_cell_ptr = torch.cat(
        [torch.tensor([0], device=crystal_batch.device),
         torch.cumsum(atoms_per_ucell, dim=0)]).long()

    atom_in_unit_cell_batch, molwise_nodes_within_unit_cell, supercell_batch = _index_cluster_nodes(atoms_per_cluster,
                                                                                                    atoms_per_ucell,
                                                                                                    crystal_batch,
                                                                                                    unit_cell_ptr)

    # add ptr offset per graph
    node_within_mol_in_ucell = molwise_nodes_within_unit_cell + crystal_batch.ptr[:-1][unit_cell_batch]
    # this indexes from the full cluster to the atom in each aunit
    cluster_node2aunit_node = node_within_mol_in_ucell[atom_in_unit_cell_batch]

    mol_ind = _index_mols(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell, crystal_batch,
                          ucells_per_cluster)

    """
    do the instantiation
    """
    cluster_batch = _instantiate_cluster(atom_in_unit_cell_batch,
                                         atoms_per_cluster, atomwise_translation,
                                         crystal_batch, cluster_node2aunit_node,
                                         supercell_batch,
                                         mol_ind)

    return cluster_batch


def _index_cluster_nodes(atoms_per_cluster, atoms_per_ucell, crystal_batch, unit_cell_ptr):
    tot_num_atoms = atoms_per_cluster.sum()
    cluster_atom_ind = torch.arange(tot_num_atoms, device=crystal_batch.device)
    supercell_batch = torch.repeat_interleave(torch.arange(crystal_batch.num_graphs,
                                                           device=crystal_batch.device),
                                              atoms_per_cluster)
    supercell_ptr = torch.cumsum(torch.cat([torch.tensor([0], device=crystal_batch.device),
                                            atoms_per_cluster[:-1]]), dim=0)
    atom_in_crystal_index = cluster_atom_ind - supercell_ptr[supercell_batch]
    atom_in_unit_cell_batch = (atom_in_crystal_index % atoms_per_ucell[supercell_batch]) + unit_cell_ptr[
        supercell_batch]
    # within-block index (0..num_atoms[i]-1), repeated for sym_mult[i]
    # TODO replace with some nice batched offset, if possible
    molwise_nodes_within_unit_cell = torch.cat([
        torch.arange(n, device=crystal_batch.device).repeat(m)
        for n, m in zip(crystal_batch.num_atoms.tolist(), crystal_batch.sym_mult.tolist())
    ])
    ''' test and compare this method on a large batch
        num_atoms = crystal_batch.num_atoms
        sym_mult = crystal_batch.sym_mult
        
        # total length = sum(num_atoms * sym_mult)
        lengths = num_atoms * sym_mult
        total = lengths.sum()
        
        # assign each position to its "unit cell"
        ucell_idx = torch.arange(total, device=crystal_batch.device) \
            .repeat_interleave(1)  # just a flat arange
        
        # expand each unit cell according to lengths
        ucell_idx = torch.repeat_interleave(torch.arange(len(lengths), device=total.device), lengths)
        
        # within each block, cycle through atom indices
        within = torch.arange(total, device=total.device) - torch.repeat_interleave(
            torch.cumsum(torch.cat([torch.zeros(1, device=total.device, dtype=num_atoms.dtype),
                                    lengths[:-1]]), 0),
            lengths
        )
        
        # reduce modulo num_atoms[i]
        molwise_nodes_within_unit_cell = within % torch.repeat_interleave(num_atoms, lengths)
        '''
    return atom_in_unit_cell_batch, molwise_nodes_within_unit_cell, supercell_batch


def _index_mols(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell, crystal_batch, ucells_per_cluster):
    """
    Finally, Z'>1-functional, asymmetric unit indexing (for mol_ind/aux_ind)
    """
    if crystal_batch.z_prime.amax() > 1:
        assert False, "Don't try cluster construction for Z'>1 crystals - the mol indexing is hideous and not up-to-date"
        atomwise_mol_ind = _zp_gr1_mol_indexing(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell,
                                                crystal_batch,
                                                ucells_per_cluster)
    else:  # much simpler to do it this way
        mols_per_ucell = crystal_batch.sym_mult
        mols_per_cluster = ucells_per_cluster * mols_per_ucell
        tot_num_mols = torch.sum(mols_per_cluster)

        global_mol_index = torch.arange(tot_num_mols, device=crystal_batch.device)
        mol_offset = torch.cat([torch.zeros(1, device=crystal_batch.device, dtype=torch.long),
                                torch.cumsum(mols_per_cluster, dim=0)[:-1]])
        local_mol_index = global_mol_index - mol_offset.repeat_interleave(mols_per_cluster)

        atoms_per_mol_in_cluster = crystal_batch.num_atoms.repeat_interleave(mols_per_cluster)
        atomwise_mol_ind = local_mol_index.repeat_interleave(
            atoms_per_mol_in_cluster
        )

    return atomwise_mol_ind


def _zp_gr1_mol_indexing(atom_in_unit_cell_batch, atoms_per_cluster, atoms_per_ucell, crystal_batch,
                         ucells_per_cluster):
    """
    This appears technically functional, but is an avoidable monstrosity
    :param atom_in_unit_cell_batch:
    :param atoms_per_cluster:
    :param atoms_per_ucell:
    :param crystal_batch:
    :param ucells_per_cluster:
    :return:
    """
    if not getattr(_zp_gr1_mol_indexing, '_warned', False):
        warnings.warn(
            "WARNING: You have done cluster building directly on a crystal batch with some Z prime greater than 1."
            "This calls a complicated and avoidable indexing function. We recommend instead splitting to separate Zp=1 "
            "graphs then recombining.", UserWarning, stacklevel=2)
        _zp_gr1_mol_indexing._warned = True

    mols_per_ucell = atoms_per_ucell // crystal_batch.num_atoms * crystal_batch.z_prime
    ucell_mol_offset = torch.cumsum(torch.cat([torch.zeros(1, device=mols_per_ucell.device, dtype=mols_per_ucell.dtype),
                                               mols_per_ucell[:-1]]), 0)
    within_ucell_mol_ind = (torch.arange(mols_per_ucell.sum(), device=mols_per_ucell.device, dtype=torch.long)
                            - ucell_mol_offset.repeat_interleave(mols_per_ucell))
    atoms_per_mol = crystal_batch.num_atoms // crystal_batch.z_prime
    molwise_ucell_sym_mult = crystal_batch.sym_mult.repeat_interleave(mols_per_ucell)
    # indexes the asymmetric units within each unit cell
    within_ucell_aunit_ind = within_ucell_mol_ind % molwise_ucell_sym_mult
    # ditto, atomwise
    within_ucell_aunit_atom_ind = within_ucell_aunit_ind.repeat_interleave(
        atoms_per_mol.repeat_interleave(mols_per_ucell))
    # we then need to expand this to the cluster batch
    cluster_aunit_inner_ind = within_ucell_aunit_atom_ind[atom_in_unit_cell_batch]
    # and add offsets to account for the new tiles in the cluster
    tot_num_ucells = ucells_per_cluster.sum()
    aunits_per_cluster = crystal_batch.sym_mult * ucells_per_cluster
    ucell_num_aunits = crystal_batch.sym_mult.repeat_interleave(ucells_per_cluster)
    aunits_per_cluster_ucell = crystal_batch.sym_mult.repeat_interleave(ucells_per_cluster)
    aunit_cum = torch.cumsum(
        torch.cat([torch.zeros(1, device=ucells_per_cluster.device, dtype=ucells_per_cluster.dtype),
                   ucell_num_aunits[:-1]]), 0)
    cluster_aunit_offset = torch.cumsum(
        torch.cat([torch.zeros(1, device=ucells_per_cluster.device, dtype=ucells_per_cluster.dtype),
                   aunits_per_cluster[:-1]]), 0)
    cluster_nodewise_offset = cluster_aunit_offset.repeat_interleave(atoms_per_cluster)
    aunit2ucell = torch.arange(tot_num_ucells, device=crystal_batch.device).repeat_interleave(
        aunits_per_cluster_ucell
    )
    aunit_offset = aunit_cum[aunit2ucell]
    atoms_per_aunit = crystal_batch.num_atoms.repeat_interleave(aunits_per_cluster)
    nodewise_offset = aunit_offset.repeat_interleave(atoms_per_aunit)
    mol_ind = cluster_aunit_inner_ind + nodewise_offset - cluster_nodewise_offset
    return mol_ind


def _instantiate_cluster(atom_in_unit_cell_batch,
                         atoms_per_cluster,
                         atomwise_translation,
                         crystal_batch,
                         cluster_node2aunit_node,
                         supercell_batch,
                         mol_ind):
    cluster_batch = crystal_batch.clone()
    # init cluster with right number of duplicated unit cells
    cluster_batch.pos = crystal_batch.unit_cell_pos[atom_in_unit_cell_batch]
    # build cluster by translating all atoms manually
    cluster_batch.pos += atomwise_translation
    # pattern out atom properties
    cluster_batch.z = crystal_batch.z[cluster_node2aunit_node]
    cluster_batch.x = crystal_batch.x[cluster_node2aunit_node]
    # rebuild batch indices
    cluster_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=crystal_batch.device),
                                   torch.cumsum(atoms_per_cluster, dim=0)]).long()
    cluster_batch.batch = supercell_batch

    # asymmetric unit indices
    cluster_batch.mol_ind = mol_ind
    cluster_batch.aux_ind = torch.ones_like(mol_ind)
    cluster_batch.aux_ind[mol_ind == 0] = 0

    return cluster_batch


def _molwise_indexing(cluster_batch,
                      crystal_batch,
                      mols_per_subunit,
                      tot_num_subunits):
    """
    Indexing for downstream analysis
    aux_ind labels 0: asymmetric unit, 1: convolutional window, 2: outside convolutional window
    mol_ind indexes the molecule (within cluster) for each atom
    """
    # which subunit each entry belongs to
    mol2subunit = torch.arange(tot_num_subunits, device=mols_per_subunit.device).repeat_interleave(mols_per_subunit)

    # index *within* the cluster
    offsets = torch.cumsum(torch.cat([torch.zeros(1, device=mols_per_subunit.device, dtype=mols_per_subunit.dtype),
                                      mols_per_subunit[:-1]]), 0)
    mol_within_subunit = torch.arange(mols_per_subunit.sum(), device=mols_per_subunit.device) - offsets[mol2subunit]

    atoms_per_mol_by_crystal = crystal_batch.num_atoms // crystal_batch.z_prime
    atoms_per_mol_by_subunit = atoms_per_mol_by_crystal.repeat_interleave(crystal_batch.z_prime)
    atoms_per_mol = atoms_per_mol_by_subunit.repeat_interleave(mols_per_subunit)

    cluster_batch.mol_ind = mol_within_subunit.repeat_interleave(
        atoms_per_mol
    )

    cluster_batch.aux_ind = torch.zeros_like(cluster_batch.z)
    cluster_batch.aux_ind[cluster_batch.mol_ind != 0] = 1  # MK this may have to be adjusted for Z'>1

    return cluster_batch


def get_cart_translations(cc_centroids,
                          T_fc,
                          mol_radii,
                          cutoff,
                          supercell_size: int = 9,
                          ):
    # precompute some stuff
    # get the set of all possible cartesian translations
    cell_vectors = T_fc.permute(0, 2, 1)
    # cell diagonals
    cell_diag = cell_vectors.sum(1).norm(dim=-1)
    # bounding box cutoff
    cutoff_distance = mol_radii * 2 + cutoff + 0.1 + cell_diag  # convolutional radius plus cell diagonal plus z'>1 buffer

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
    cells_per_cluster = good_translations_bool.sum(1)

    return good_translations, cells_per_cluster


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
    # TODO would be nice if this could be parallelized. Would be nice indeed
    """
    canonical_conformer_coords_list = []
    mol_position_list = []
    well_defined_asym_unit_list = []
    for i, (T_cf, sg_ind) in enumerate(zip(crystal_batch.T_cf, crystal_batch.sg_ind)):
        unit_cell_coords = crystal_batch.unit_cell_pos[crystal_batch.unit_cell_batch == i]
        unit_cell_atom_types = crystal_batch.z[crystal_batch.batch == i].repeat(crystal_batch.sym_mult[i])
        # filter for heavy only
        num_heavy_atoms = sum(crystal_batch.z[crystal_batch.batch == i] > 1)
        heavy_ucell_coords = unit_cell_coords[unit_cell_atom_types > 1]
        heavy_ucell_coords = heavy_ucell_coords.reshape(crystal_batch.sym_mult[i], num_heavy_atoms, 3)
        (canonical_conformer_index,
         centroids_fractional,
         well_defined_asym_unit) = (
            identify_canonical_asymmetric_unit(
                T_cf,
                asym_unit_dict,
                sg_ind,
                heavy_ucell_coords))

        unit_cell_coords = unit_cell_coords.reshape(crystal_batch.sym_mult[i], crystal_batch.num_atoms[i], 3)
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
        mol_batch.num_atoms,
        heavy_atoms_only=True,
        atom_types=mol_batch.z
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
    we can obtain uniqueness constraining the axis to a half-sphere.
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


def align_mol_batch_to_standard_axes(mol_batch, handedness=None, return_rot=False):
    """
    align principal inertial axes of molecules in a crystaldata object to the xyz or xy(-z) axes
    only works for geometric principal axes (all atoms mass = 1)
    """
    Ip, Ipm, I = batch_molecule_principal_axes_torch(
        mol_batch.pos,
        mol_batch.batch,
        mol_batch.num_graphs,
        mol_batch.num_atoms,
        heavy_atoms_only=True,
        atom_types=mol_batch.z
    )

    eye = torch.tile(torch.eye(3,
                               device=mol_batch.pos.device,
                               dtype=torch.float32),
                     (mol_batch.num_graphs, 1, 1)
                     )  # set as right-handed by default

    if handedness is not None:  # otherwise, custom
        # alignment vector is (x,y,z) by default, or (-x,y,z) for left-handed structures
        if torch.is_tensor(handedness):
            eye[:, 0, 0] = handedness[:, :1].flatten()  # flatten - in case it's higher dimensional [n,1]
        else:
            eye[:, 0, 0] = handedness[:, :1]

    # rotation is p_align = R @ p_orig,
    # R = p_align \ p_orig

    std_rotations = extract_rotmat(eye, torch.linalg.inv(Ip))
    mol_batch.recenter_molecules()  # ensures molecules are centered
    mol_batch.pos = apply_rotation_to_batch(mol_batch.pos,
                                            std_rotations,
                                            mol_batch.batch)
    # test
    # Ip, Ipm, I = batch_molecule_principal_axes_torch(
    #     mol_batch.pos,
    #     mol_batch.batch,
    #     mol_batch.num_graphs,
    #     mol_batch.num_atoms
    # )
    # assert torch.all(torch.isclose(Ip, eye, atol=1e-4))
    if return_rot:
        return mol_batch, std_rotations
    else:
        return mol_batch


def aunit2ucell(mol_batch):
    """
    :param mol_batch:
    :return:
    """
    atoms_per_mol = mol_batch.num_atoms
    sym_mult = mol_batch.sym_mult
    atoms_per_unit_cell = mol_batch.num_atoms * sym_mult
    tot_num_mols = torch.sum(sym_mult)

    aunit_coords_c = mol_batch.pos
    centroids_f = mol_batch.aunit_centroid

    # repeat each molecule Z times
    mol2ucell = torch.arange(len(sym_mult), device=sym_mult.device).repeat_interleave(sym_mult)

    ucell_centroids_f = mol_batch.assign_aunit_centroid(mol_batch.aunit_centroid[mol2ucell])
    fc_transform_list = mol_batch.T_fc
    cf_transform_list = mol_batch.T_cf

    # same, atomwise
    ucell_node2mol_node = block_repeat_interleave(mol_batch.num_atoms, sym_mult)

    unit_cell_coords_c = aunit_coords_c[ucell_node2mol_node]
    unit_cell_batch = mol_batch.batch[ucell_node2mol_node]

    # assign which molecule each atom is in
    ucell_node2ucell_mol = torch.arange(tot_num_mols, device=mol_batch.device).repeat_interleave(
        atoms_per_mol.repeat_interleave(sym_mult))

    # MK todo this tensor(np.concatenate) can't be very efficient
    molwise_sym_ops = torch.tensor(np.concatenate(mol_batch.symmetry_operators, axis=0),
                                   device=centroids_f.device, dtype=torch.float32)

    # add 4th dimension as a dummy for affine transforms, should have shape (num_aunits, 4, 4)
    unit_cell_affine_centroids_f = torch.cat(
        (
            ucell_centroids_f,
            torch.ones(ucell_centroids_f.shape[:-1] + (1,), dtype=torch.float32)
            .to(aunit_coords_c.device)),
        dim=-1)

    # get all molecule centroids via symmetry ops
    ucell_centroids_f = torch.einsum('nij,nj->ni', (molwise_sym_ops, unit_cell_affine_centroids_f))[..., :-1]

    # force centroids within unit cell
    centroids_f_in_cell = ucell_centroids_f - torch.floor(ucell_centroids_f)

    # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame
    centered_coords_c = center_batch(unit_cell_coords_c,
                                     unit_cell_batch,
                                     mol_batch.num_graphs,
                                     atoms_per_unit_cell,
                                     center_on_heavy_atoms=True,
                                     atom_types=mol_batch.z[ucell_node2mol_node])

    centered_coords_f = fractional_transform(centered_coords_c,
                                             cf_transform_list[unit_cell_batch])
    rot_coords_f = torch.einsum('nj,nij->ni', (centered_coords_f, molwise_sym_ops[ucell_node2ucell_mol, :3, :3]))

    # translate rotated aunits to their correct position
    unit_cell_pos = torch.einsum('nij,nj->ni',
                                 (fc_transform_list[unit_cell_batch],
                                  rot_coords_f + centroids_f_in_cell[ucell_node2ucell_mol, :]))

    return unit_cell_pos, unit_cell_batch, ucell_node2ucell_mol


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
                        std_orientation: bool = True,
                        mol_handedness: OptTensor = None,
                        ) -> Tensor:
    """
    pose asymmetric unit via fractional translations and rotations
    always take the zp=1 component
    """
    rotations_list = rotvec2rotmat(mol_batch.aunit_orientation[:, :3])

    if std_orientation:  # align canonical conformers principal axes to cartesian axes - not usually done here, but allowed
        assert mol_handedness is not None, "Must provide explicit handedness when aligning mol to canonical axes"
        mol_batch = align_mol_batch_to_standard_axes(mol_batch, handedness=mol_handedness[:, :1])
    else:
        mol_batch.recenter_molecules()

    # apply rotations and fractional translations
    pos = (apply_rotation_to_batch(mol_batch.pos,
                                   rotations_list,
                                   mol_batch.batch)
           + fractional_transform(mol_batch.aunit_centroid[:, :3], mol_batch.T_fc)[mol_batch.batch])

    # # ensure embeddings rotate along with the aunit positions
    # this is a bit risky - leave it for now
    # if hasattr(mol_batch, 'embedding'):
    #     if mol_batch.embedding.shape[1] == 3:  # if there is a vector embedding, we should also rotate it
    #         mol_batch.embedding = torch.einsum('nij, njk -> njk', rotations_list, mol_batch.embedding)

    return pos


def canonicalize_aunit_order(batch):
    # canonical aunit ordering according to distance to the origin, with x, y, z, tiebreak
    # TODO implement tiebreak
    per_aunit_centroids = batch.aunit_centroid.reshape(batch.num_graphs, batch.max_z_prime, 3)
    # mask out elements with lower z prime than max
    idx = torch.arange(batch.max_z_prime, device=batch.device)[None, ...]
    mask = (idx >= (batch.z_prime[..., None]))[..., None].expand(-1, -1, 3)
    per_aunit_centroids[mask] = 1  # this will put lower Z' options always at the end
    origin_dists = per_aunit_centroids.norm(dim=2)
    canonical_order = origin_dists.argsort(dim=1)
    canonical_order_exp = canonical_order.unsqueeze(-1).expand(-1, -1, 3)  # [n, k, 3]
    per_aunit_centroids[mask] = 0.5  # this is the placeholder value for 'not being used'
    sorted_centroids = torch.gather(per_aunit_centroids, dim=1, index=canonical_order_exp)

    aunit_centroid = sorted_centroids.reshape(batch.num_graphs, batch.max_z_prime * 3)

    # also reorder the orientations accordingly
    per_aunit_orient = batch.aunit_orientation.reshape(batch.num_graphs, batch.max_z_prime, 3)
    per_aunit_orient[mask] = torch.pi / 2  # this places all unused samples in the same state
    canonical_order_exp = canonical_order.unsqueeze(-1).expand(-1, -1, 3)  # [n, k, 3]
    sorted_orient = torch.gather(per_aunit_orient, dim=1, index=canonical_order_exp)
    aunit_orientation = sorted_orient.reshape(batch.num_graphs, batch.max_z_prime * 3)

    aunit_handedness = torch.gather(batch.aunit_handedness, dim=1, index=canonical_order)
    return aunit_centroid, aunit_orientation, aunit_handedness


def protonate_mol(atom_types, coords):
    """
    atom_types : list/array of atomic numbers
    coords     : (N,3) float array in Å
    returns: (new_atom_types, new_coords) with added hydrogens
    """

    # 1. Build an RDKit mol without sanitization
    mol = Chem.RWMol()
    for Z in atom_types:
        mol.AddAtom(Chem.Atom(int(Z)))

    # 2. Add a conformer with your coords
    conf = Chem.Conformer(len(atom_types))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol = mol.GetMol()
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    mol.AddConformer(conf, assignId=True)

    # 3. Add hydrogens (fast, valence-based)
    molH = Chem.AddHs(mol, addCoords=True)

    # If your input had 3D coordinates, this preserves heavy atom positions
    # and places hydrogens using RDKit's built-in valence geometry rules.
    # No re-embedding needed for most molecules.

    # 4. Extract arrays back out
    confH = molH.GetConformer()
    n = molH.GetNumAtoms()

    new_coords = np.array([list(confH.GetAtomPosition(i)) for i in range(n)], dtype=float)
    new_atom_types = np.array([molH.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)], dtype=int)

    return new_atom_types, new_coords

    """
    
    m1 = Atoms(positions=coords, numbers=atom_types)
    m2 = Atoms(positions=new_coords, numbers=new_atom_types)
    view([m1, m2])
    
    """
