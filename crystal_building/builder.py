import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from crystal_building.utils import \
    (update_supercell_data, ref_to_supercell, clean_cell_output, align_crystaldata_to_principal_axes, unit_cell_analysis, set_sym_ops)
from common.geometry_calculations import coor_trans_matrix
from constants.asymmetric_units import asym_unit_dict


def write_sg_to_all_crystals(override_sg, dataDims, supercell_data, symmetries_dict, sym_ops_list):
    # overrite point group one-hot
    # overwrite space group one-hot
    # overwrite crystal system one-hot
    # overwrite z value

    sg_num = list(symmetries_dict['space_groups'].values()).index(override_sg) + 1  # indexing from 0
    sg_ind = symmetries_dict['sg_feature_ind_dict'][symmetries_dict['space_groups'][sg_num]]
    crysys_ind = symmetries_dict['crysys_ind_dict'][symmetries_dict['lattice_type'][sg_num]]
    z_value_ind = max(list(symmetries_dict['crysys_ind_dict'].values())) + 1  # todo hardcode

    # todo replace datadims call by crystal generation features
    supercell_data.x[:, -dataDims['num crystal generation features']] = 0  # set all crystal features to 0
    supercell_data.x[:, sg_ind] = 1  # set all molecules to the given space group
    supercell_data.x[:, crysys_ind] = 1  # set all molecules to the given crystal system
    # todo replace sym_ops_list arg by Z value
    supercell_data.Z = len(sym_ops_list[0]) * torch.ones_like(supercell_data.Z)
    supercell_data.x[:, z_value_ind] = supercell_data.Z[0] * torch.ones_like(supercell_data.x[:, 0])
    supercell_data.sg_ind = sg_num * torch.ones_like(supercell_data.sg_ind)

    return supercell_data


def update_crystal_symmetry_elements(mol_data, generate_sgs, dataDims, symmetries_dict, randomize_sgs=False):
    """
    update the symmetry information in molecule-wise crystaldata objects
    """
    # identify the SG numbers we want to generate
    if type(generate_sgs[0]) == str:
        generate_sg_inds = [list(symmetries_dict['space_groups'].values()).index(SG) + 1 for SG in generate_sgs]  # indexing from 0
    else:
        generate_sg_inds = generate_sgs

    # randomly assign SGs to samples
    if randomize_sgs:
        sample_sg_inds = np.random.choice(generate_sg_inds, size=mol_data.num_graphs, replace=True)
    else:
        sample_sg_inds = generate_sg_inds

    # update sym ops
    mol_data.symmetry_operators = [torch.Tensor(symmetries_dict['sym_ops'][sg_ind]).to(mol_data.x.device) for sg_ind in sample_sg_inds]

    # compute and update Z values
    sample_Z_values = [len(mol_data.symmetry_operators[ii]) for ii in range(mol_data.num_graphs)]
    mol_data.Z = torch.tensor(sample_Z_values, dtype=mol_data.Z.dtype, device=mol_data.Z.device)  # * torch.ones_like(mol_data.Z)
    mol_data.sg_ind = torch.tensor(sample_sg_inds, dtype=mol_data.sg_ind.dtype, device=mol_data.sg_ind.device)

    mol_data.x[:, -dataDims['num crystal generation features']] = 0  # set all crystal features to 0
    # update sym ops, sg ind, sg one_hot, crystal system one_hot, Z value
    for ii, sg_ind in enumerate(sample_sg_inds):
        mol_inds = torch.arange(mol_data.ptr[ii], mol_data.ptr[ii + 1])
        mol_data.x[mol_inds, symmetries_dict['crysys_ind_dict'][symmetries_dict['lattice_type'][sg_ind]]] = 1  # one-hot for crystal system
        mol_data.x[mol_inds, symmetries_dict['sg_feature_ind_dict'][symmetries_dict['space_groups'][sg_ind]]] = 1  # one-hot for space group
        mol_data.x[mol_inds, symmetries_dict['crystal_z_value_ind']] = mol_data.Z[ii].float()  # set Z-value

    return mol_data


class SupercellBuilder:
    def __init__(self, symmetries_dict, dataDims, supercell_size=5, device='cuda'):
        """
        class for converting single molecules -> unit cells -> supercells/clusters
        """  # todo write tests to confirm correct reconstruction from cell params

        self.sym_ops = symmetries_dict['sym_ops']  # list of symmetry operations
        self.symmetries_dict = symmetries_dict  # other symmetry information
        self.dataDims = dataDims  # information about the dataset
        self.device = device
        self.numpy_asym_unit_dict = asym_unit_dict.copy()
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:  # todo make sure we aren't distorting this for other classes
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

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
        self.sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))].to(device)

    def build_supercells(self, data, cell_sample: torch.tensor, supercell_size: int=5, graph_convolution_cutoff: float=6, target_handedness=None,
                         skip_cell_cleaning=False, standardized_sample=True, align_molecules=True,
                         rescale_asymmetric_unit=True, pare_to_convolution_cluster=True):
        """
        convert cell parameters to unit cell in a fast, differentiable, invertible way
        convert reference cell to "supercell" (in fact, it's truncated to an appropriate cluster size)
        """
        supercell_data = data.clone()

        supercell_data, cell_sample, target_handedness = \
            self.move_cell_data_to_device(supercell_data, cell_sample, target_handedness)

        sym_ops_list, supercell_data = set_sym_ops(supercell_data)  # assign correct symmetry options

        # destandardize and constrain cell params to physical values
        cell_lengths, cell_angles, mol_position, mol_rotation = self.process_cell_params(
            supercell_data, cell_sample, skip_cell_cleaning, standardized_sample, rescale_asymmetric_unit=rescale_asymmetric_unit)

        # get transformation matrices
        T_fc_list, T_cf_list, generated_cell_volumes = coor_trans_matrix(cell_lengths, cell_angles)

        supercell_data.T_fc = T_fc_list
        supercell_data.cell_params = torch.cat((cell_lengths, cell_angles, mol_position, mol_rotation), dim=1)  # note: destandardized

        if align_molecules:  # align canonical conformers principal axes to cartesian axes - not usually done here, but allowed
            supercell_data = align_crystaldata_to_principal_axes(supercell_data, handedness=target_handedness)

        # get molecule information
        coords_list = []
        atomic_number_list = []
        for i in range(supercell_data.num_graphs):
            atomic_number_list.append(supercell_data.x[supercell_data.batch == i])
            coords_list.append(supercell_data.pos[supercell_data.batch == i])

        # convert rotvecs to rotation matrices
        rotations_list = self.rotvec_to_rmat(mol_rotation)

        # center, apply rotation, apply translation (to canonical conformer)
        canonical_conformer_coords_list = []
        for i, (rotation, coords, T_fc, new_frac_pos) in enumerate(zip(rotations_list, coords_list, T_fc_list, mol_position)):
            canonical_conformer_coords_list.append(
                torch.inner(rotation, coords - coords.mean(0)).T
                + torch.inner(T_fc, new_frac_pos)
            )

        # apply symmetry ops to build unit cell
        unit_cell_coords_list = self.build_unit_cell(supercell_data.clone(), canonical_conformer_coords_list, T_fc_list, T_cf_list, sym_ops_list)

        # reanalyze the constructed unit cell to get the canonical orientation & confirm correct construction
        # todo rebuild the below function in torch with new parameterization
        cell_analysis = [unit_cell_analysis(
            unit_cell_coords_list[ii].cpu().detach().numpy(),
            supercell_data.sg_ind[ii].cpu().detach().numpy(),
            self.numpy_asym_unit_dict,
            np.linalg.inv(supercell_data.T_fc[ii].cpu().detach().numpy()),
            enforce_right_handedness=False
        ) for ii in range(len(unit_cell_coords_list))]

        supercell_data.cell_params[:, 9:12] = torch.tensor([cell_analysis[ii][1] for ii in range(supercell_data.num_graphs)])  # overwrite to canonical parameters
        supercell_data.asym_unit_handedness = torch.tensor([cell_analysis[ii][2] for ii in range(supercell_data.num_graphs)])

        cell_vector_list = T_fc_list.permute(0, 2, 1)  # cell_vectors(T_fc_list)  # I think this just IS the T_fc matrix  # TODO confirm we want to transpose T_fc to get the cell vectors in the correct basis

        # get minimal supercell cluster for convolving about a given canonical conformer
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            ref_to_supercell(
                unit_cell_coords_list, cell_vector_list, T_fc_list, atomic_number_list, supercell_data.Z,
                supercell_scale=supercell_size, cutoff=graph_convolution_cutoff,
                sorted_fractional_translations=self.sorted_fractional_translations,
                pare_to_convolution_cluster=pare_to_convolution_cluster)

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list, unit_cell_coords_list)

        return supercell_data, generated_cell_volumes, None

    def unit_cell_to_supercell(self, supercell_data, supercell_size=5, graph_convolution_cutoff=6, pare_to_convolution_cluster=True):
        """
        build a supercell cluster using a pre-built unit cell
        will not check for physicality or apply any symmetry options - merely pattern the unit cell
        and keep molecules within convolution radius of the canonical conformer
        automatically pare NxNxN supercell to minimal set of molecules in the convolution radius of the canonical conformer
        """
        supercell_data = supercell_data.clone().to(self.device)

        T_fc_list, T_cf_list, generated_cell_volumes = coor_trans_matrix(
            cell_lengths=supercell_data.cell_params[:, 0:3], cell_angles=supercell_data.cell_params[:, 3:6])

        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_i = supercell_data.x[supercell_data.batch == i]
            atoms_list.append(atoms_i)

        cell_vector_list = T_fc_list.permute(0, 2, 1)  # cell_vectors(T_fc_list)  # todo as above, confirm this transposal
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            ref_to_supercell(supercell_data.ref_cell_pos, cell_vector_list,
                             T_fc_list, atoms_list, supercell_data.Z,
                             supercell_scale=supercell_size, cutoff=graph_convolution_cutoff,
                             sorted_fractional_translations=self.sorted_fractional_translations,
                             pare_to_convolution_cluster=pare_to_convolution_cluster)

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list, supercell_data.ref_cell_pos)

        return supercell_data.to(self.device)

    def process_cell_params(self, supercell_data, cell_sample, skip_cell_cleaning=False, standardized_sample=True, rescale_asymmetric_unit=True):
        if skip_cell_cleaning:  # don't clean up
            if standardized_sample:
                destandardized_cell_sample = (cell_sample * torch.tensor(self.dataDims['lattice stds'], device=self.device, dtype=cell_sample.dtype)) + torch.tensor(
                    self.dataDims['lattice means'], device=self.device, dtype=cell_sample.dtype)  # destandardize
                cell_lengths, cell_angles, mol_position, mol_rotation = destandardized_cell_sample.split(3, 1)
            else:
                cell_lengths, cell_angles, mol_position, mol_rotation = cell_sample.split(3, 1)
        else:
            cell_lengths, cell_angles, mol_position, mol_rotation = cell_sample.split(3, 1)
            lattices = [self.symmetries_dict['lattice_type'][int(supercell_data.sg_ind[n])] for n in range(supercell_data.num_graphs)]
            cell_lengths, cell_angles, mol_position, mol_rotation, _, _, _ = clean_cell_output(
                cell_lengths, cell_angles, mol_position, mol_rotation, lattices, self.dataDims,
                enforce_crystal_system=True, return_transforms=True, standardized_sample=standardized_sample)

        if rescale_asymmetric_unit:  # todo assert only our prepared sgs will be allowed
            mol_position = self.scale_asymmetric_unit(mol_position, supercell_data.sg_ind)

        return cell_lengths, cell_angles, mol_position, mol_rotation

    def get_canonical_conformer(self, supercell_data, mol_position, sym_ops_list, debug=False):
        """
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

    def rotvec_to_rmat(self, mol_rotation):
        """  # todo replace by axis-angle rotation in spherical units
        get applied rotation
        rotvec -> quat -> rotation matrix
        """
        theta = torch.linalg.norm(mol_rotation, dim=1)
        unit_vector = mol_rotation / theta[:, None]
        q = torch.cat([torch.cos(theta / 2)[:, None], unit_vector * torch.sin(theta / 2)[:, None]], dim=1)

        applied_rotation_list = torch.stack((torch.stack((1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3], 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]), dim=1),
                                             torch.stack((2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]), dim=1),
                                             torch.stack((2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2], 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2), dim=1)),
                                            dim=1)

        return applied_rotation_list

    def build_unit_cell(self, supercell_data, final_coords_list, T_fc_list, T_cf_list, sym_ops_list):
        """
        use cell symmetry to pattern canonical conformer into full unit cell
        batch crystals with same Z value together for added speed in large batches
        """
        reference_cell_list_i = []

        unique_z_values = torch.unique(supercell_data.Z)
        z_inds = [torch.where(supercell_data.Z == z)[0] for z in unique_z_values]

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

    def scale_asymmetric_unit(self, mol_position, sg_ind):
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
            scaled_mol_position[i, :] = mol_position[i, :] * self.asym_unit_dict[str(int(ind))]

        return scaled_mol_position

    def move_cell_data_to_device(self, supercell_data, cell_sample, target_handedness):
        supercell_data = supercell_data.to(self.device)

        if cell_sample is not None:
            cell_sample = cell_sample.to(self.device)

        if target_handedness is not None:
            target_handedness = torch.tensor(target_handedness, device=self.device, dtype=torch.float32)  # target_handedness.to(self.device)

        return supercell_data, cell_sample, target_handedness
