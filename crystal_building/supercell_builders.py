from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from crystal_building.crystal_builder_tools import \
    (update_supercell_data, compute_lattice_vector_overlap,
     coor_trans_matrix,
     ref_to_supercell, clean_cell_output, invert_coords,
     compute_Ip_handedness, align_crystaldata_to_principal_axes, asym_unit_dict,
     euler_XYZ_rotation_matrix)


def override_sg_info(override_sg, dataDims, supercell_data, symmetries_dict, sym_ops_list):
    # overrite point group one-hot
    # overwrite space group one-hot
    # overwrite crystal system one-hot
    # overwrite z value

    sg_num = list(symmetries_dict['space_groups'].values()).index(override_sg) + 1  # indexing from 0
    sg_ind = symmetries_dict['sg_feature_ind_dict'][symmetries_dict['space_groups'][sg_num]]
    crysys_ind = symmetries_dict['crysys_ind_dict'][symmetries_dict['lattice_type'][sg_num]]
    z_value_ind = max(list(symmetries_dict['crysys_ind_dict'].values())) + 1  # todo hardcode

    #
    supercell_data.x[:, -dataDims['num crystal generation features']] = 0  # set all crystal features to 0
    supercell_data.x[:, sg_ind] = 1  # set all molecules to the given space group
    supercell_data.x[:, crysys_ind] = 1  # set all molecules to the given crystal system
    supercell_data.Z = len(sym_ops_list[0]) * torch.ones_like(supercell_data.Z)
    supercell_data.x[:, z_value_ind] = supercell_data.Z[0] * torch.ones_like(supercell_data.x[:, 0])
    supercell_data.sg_ind = sg_num * torch.ones_like(supercell_data.sg_ind)

    return supercell_data


class SupercellBuilder():
    def __init__(self, sym_ops, symmetries_dict, normed_lattice_vectors, atom_weights, dataDims, supercell_scale = 5, device = 'cuda'):
        self.sym_ops = sym_ops
        self.atom_weights = atom_weights
        self.symmetries_dict = symmetries_dict
        self.dataDims = dataDims
        self.normed_lattice_vectors = normed_lattice_vectors
        self.device = device
        # confirm sym ops we are using agree with these settings
        # todo confirm these make right crystals
        # todo add support for all 230 space groups
        # todo add extra tools for non-parallelipped asymmetric units
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

        # init fractional translations for supercell construction
        n_cells = (2 * supercell_scale + 1) ** 3
        fractional_translations = torch.zeros((n_cells, 3))  # initialize the translations in fractional coords
        i = 0
        for xx in range(-supercell_scale, supercell_scale + 1):
            for yy in range(-supercell_scale, supercell_scale + 1):
                for zz in range(-supercell_scale, supercell_scale + 1):
                    fractional_translations[i] = torch.tensor((xx, yy, zz))
                    i += 1

        self.sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))].to(device)


    def build_supercells(self, data, cell_sample, supercell_size, graph_convolution_cutoff, target_handedness=None,
                         override_sg=None, skip_cell_cleaning=False, standardized_sample=True, align_molecules = True):
        '''
        convert cell parameters to unit cell in a fast, differentiable, invertible way
        convert reference cell to supercell with appropriate cluster size
        '''
        supercell_data = data.clone()

        supercell_data, cell_sample, target_handedness = \
            self.move_cell_data_to_device(supercell_data, cell_sample, target_handedness)
        sym_ops_list, supercell_data = self.set_sym_ops(override_sg, supercell_data)

        cell_lengths, cell_angles, mol_position, mol_rotation = \
            self.process_cell_params(supercell_data, cell_sample, skip_cell_cleaning, standardized_sample)

        T_fc_list, T_cf_list, generated_cell_volumes = coor_trans_matrix(cell_lengths, cell_angles)

        supercell_data.T_fc = T_fc_list
        supercell_data.cell_params = torch.cat((cell_lengths, cell_angles, mol_position, mol_rotation), dim=1)  # lengths are destandardized here

        if align_molecules: # align canonical conformers principal axes to cartesian axes
            supercell_data = align_crystaldata_to_principal_axes(supercell_data, handedness=target_handedness)

        coords_list = []
        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_list.append(supercell_data.x[supercell_data.batch == i])
            coords_list.append(supercell_data.pos[supercell_data.batch == i])

        rotations_list = self.rotvec_to_rmat(mol_rotation)
        final_coords_list = []
        for i, (rotation, coords, T_fc, new_frac_pos) in enumerate(zip(rotations_list, coords_list, T_fc_list, mol_position)):
            final_coords_list.append(torch.inner(rotation, coords - coords.mean(0)).T + torch.inner(T_fc, new_frac_pos))

        reference_cell_list = self.build_unit_cell(supercell_data.clone(), final_coords_list, T_fc_list, T_cf_list, sym_ops_list)

        cell_vector_list = T_fc_list.permute(0, 2, 1)  # cell_vectors(T_fc_list)  # I think this just IS the T_fc matrix
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, supercell_data.Z,
                             supercell_scale=supercell_size, cutoff=graph_convolution_cutoff,
                             sorted_fractional_translations=self.sorted_fractional_translations)

        overlaps_list = None  # expensive and not currently used # compute_lattice_vector_overlap(final_coords_list, T_cf_list, normed_lattice_vectors=self.normed_lattice_vectors.to(supercell_data.x.device))

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list)

        return supercell_data, generated_cell_volumes, overlaps_list

    def real_cell_to_supercell(self, supercell_data, config, return_overlaps=False):
        supercell_data = supercell_data.clone().to(self.device)

        T_fc_list, T_cf_list, generated_cell_volumes = coor_trans_matrix(
            cell_lengths=supercell_data.cell_params[:, 0:3], cell_angles=supercell_data.cell_params[:, 3:6])

        # masses_list = []
        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_i = supercell_data.x[supercell_data.batch == i]
            atoms_list.append(atoms_i)
            # atomic_numbers = atoms_i[:, 0]

        cell_vector_list = T_fc_list.permute(0, 2, 1)  # cell_vectors(T_fc_list)
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            ref_to_supercell(supercell_data.ref_cell_pos, cell_vector_list, T_fc_list,
                             atoms_list, supercell_data.Z, supercell_scale=config.supercell_size,
                             cutoff=config.discriminator.graph_convolution_cutoff,
                             sorted_fractional_translations=self.sorted_fractional_translations)

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list)

        # if return_overlaps:  # todo finish this
        #     overlaps_list = compute_lattice_vector_overlap(masses_list, final_coords_list, T_cf_list, self.normed_lattice_vectors=self.self.normed_lattice_vectors)
        #     return supercell_data.to(config.device), overlaps_list

        return supercell_data.to(config.device)

    def process_cell_params(self, supercell_data, cell_sample, skip_cell_cleaning=False, standardized_sample=True):
        if skip_cell_cleaning:  # don't clean up
            if standardized_sample:
                destandardized_cell_sample = (cell_sample * torch.Tensor(self.dataDims['lattice stds'])) + torch.Tensor(self.dataDims['lattice means'])  # destandardize
                cell_lengths, cell_angles, mol_position, mol_rotation = destandardized_cell_sample.split(3, 1)
            else:
                cell_lengths, cell_angles, mol_position, mol_rotation = cell_sample.split(3, 1)
        else:
            cell_lengths, cell_angles, mol_position, mol_rotation = cell_sample.split(3, 1)
            lattices = [self.symmetries_dict['lattice_type'][int(supercell_data.sg_ind[n])] for n in range(supercell_data.num_graphs)]
            cell_lengths, cell_angles, mol_position, mol_rotation, _, _, _ = clean_cell_output(
                cell_lengths, cell_angles, mol_position, mol_rotation, lattices, self.dataDims,
                enforce_crystal_system=True, return_transforms=True, standardized_sample=standardized_sample)

        # TODO convert from asymmetric unit to full cell fractional coordinates

        if True: # todo assert only our prepared sgs will be allowed
            mol_position = self.scale_asymmetric_unit(mol_position, supercell_data.sg_ind)

        return cell_lengths, cell_angles, mol_position, mol_rotation

    def get_canonical_conformer(self, supercell_data, mol_position, sym_ops_list, debug=False):
        '''
         identify canonical conformer
         '''
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

            if debug:
                assert F.l1_loss(canonical_fractional_centroids_list[inds], mol_position[inds], reduction='mean') < 0.001

        if debug:
            assert F.l1_loss(canonical_fractional_centroids_list, mol_position, reduction='mean') < 0.001

        return canonical_fractional_centroids_list

    def rotvec_to_rmat(self, mol_rotation):
        '''
        get applied rotation
        rotvec -> quat -> rotation matrix
        '''
        theta = torch.linalg.norm(mol_rotation, dim=1)
        unit_vector = mol_rotation / theta[:, None]
        q = torch.cat([torch.cos(theta / 2)[:, None], unit_vector * torch.sin(theta / 2)[:, None]], dim=1)

        applied_rotation_list = torch.stack((torch.stack((1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3], 2 * q[:, 1] * q[:, 3] + 2 * q[:, 0] * q[:, 2]), dim=1),
                                             torch.stack((2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2, 2 * q[:, 2] * q[:, 3] - 2 * q[:, 0] * q[:, 1]), dim=1),
                                             torch.stack((2 * q[:, 1] * q[:, 3] - 2 * q[:, 0] * q[:, 2], 2 * q[:, 2] * q[:, 3] + 2 * q[:, 0] * q[:, 1], 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2), dim=1)),
                                            dim=1)

        # fully random rotations
        #applied_rotation_list = torch.tensor(Rotation.random(num=len(mol_rotation)).as_matrix(), device=mol_rotation.device,dtype=mol_rotation.dtype)

        #euler angles -> rotation matrix
        # applied_rotation_list = torch.stack([
        #     euler_XYZ_rotation_matrix(mol_rotation[i].cpu()) for i in range(len(mol_rotation))
        # ]).to(mol_rotation.device)

        return applied_rotation_list

    def build_unit_cell(self, supercell_data, final_coords_list, T_fc_list, T_cf_list, sym_ops_list):
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

                # keep centroids within unit cell
                centroids_f_z_in_cell = centroids_f_z - torch.floor(centroids_f_z)

                # subtract centroids and apply point symmetry to the molecule coordinates in cartesian frame
                # rot_coords_c = torch.einsum('nmj,nij->nmi', (padded_coords_c - centroids_c[:, None, :], z_sym_ops[:, zv, :3, :3]))
                # add final centroid
                # ref_cells[zv, :, :, :] = rot_coords_c + torch.einsum('nij,nj->ni', (T_fc_list[inds], centroids_f_z_in_cell))[:, None, :]

                # subtract centroids and apply point symmetry to the molecule coordinates in fractional frame - the cartesian approach doesn't work for all point gruops
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
        '''
        input fractional coordinates are scaled on 0-1
        rescale these for the specific ranges according to each space group
        for now, hardcoded to do P-1 only
        Parameters
        ----------
        mol_position
        sg_ind

        Returns
        -------
        '''

        # assert all(sg_ind == 2)
        '''
        P-1 asymmetric unit bounds
        x[0,.5], yz[0,1]
        '''
        scaled_mol_position = mol_position.clone()
        for i, ind in enumerate(sg_ind):
            scaled_mol_position[i, :] = mol_position[i, :] * self.asym_unit_dict[str(int(ind))]

        return scaled_mol_position

    def set_sym_ops(self, override_sg, supercell_data):
        if override_sg is not None:
            override_sg_ind = list(self.symmetries_dict['space_groups'].values()).index(override_sg) + 1  # indexing from 0
            sym_ops_list = [torch.Tensor(self.symmetries_dict['sym_ops'][override_sg_ind]).to(supercell_data.x.device) for i in range(supercell_data.num_graphs)]
            supercell_data = override_sg_info(override_sg, self.dataDims, supercell_data, self.symmetries_dict, sym_ops_list)  # todo update the way we handle this
        else:
            sym_ops_list = [torch.Tensor(supercell_data.symmetry_operators[n]).to(supercell_data.x.device) for n in range(len(supercell_data.symmetry_operators))]

        return sym_ops_list, supercell_data

    def move_cell_data_to_device(self, supercell_data, cell_sample, target_handedness):
        supercell_data = supercell_data.to(self.device)

        if cell_sample is not None:
            cell_sample = cell_sample.to(self.device)

        if target_handedness is not None:
            target_handedness = target_handedness.to(self.device)

        return supercell_data, cell_sample, target_handedness
