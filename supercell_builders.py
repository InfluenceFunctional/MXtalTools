import numpy as np
from ase.calculators import lj
from ase import Atoms
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from crystal_builder_tools import \
    (cell_analysis, update_supercell_data, compute_lattice_vector_overlap,
     coor_trans_matrix,
     ref_to_supercell, clean_cell_output, compute_principal_axes_list, invert_coords,
     compute_Ip_handedness)


def override_sg_info(override_sg, dataDims, supercell_data, symmetries_dict, sym_ops_list):
    # overrite point group one-hot
    # overwrite space group one-hot
    # overwrite crystal system one-hot
    # overwrite z value

    sg_num = list(symmetries_dict['space_groups'].values()).index(override_sg) + 1  # indexing from 0
    sg_ind = symmetries_dict['sg_ind_dict'][symmetries_dict['space_groups'][sg_num]]
    pg_ind = symmetries_dict['pg_ind_dict'][symmetries_dict['point_groups'][sg_num]]
    crysys_ind = symmetries_dict['crysys_ind_dict'][symmetries_dict['lattice_type'][sg_num]]
    z_value_ind = max(list(symmetries_dict['crysys_ind_dict'].values())) + 1  # todo hardcode

    #

    supercell_data.x[:, -dataDims['num crystal generation features']] = 0  # set all crystal features to 0
    supercell_data.x[:, pg_ind] = 1  # set all molecules to the given pg
    supercell_data.x[:, sg_ind] = 1  # set all molecules to the given pg
    supercell_data.x[:, crysys_ind] = 1  # set all molecules to the given pg
    supercell_data.Z = len(sym_ops_list[0]) * torch.ones_like(supercell_data.Z)
    supercell_data.x[:, z_value_ind] = supercell_data.Z[0] * torch.ones_like(supercell_data.x[:, 0])
    supercell_data.sg_ind = sg_num * torch.ones_like(supercell_data.sg_ind)

    return supercell_data


class SupercellBuilder():
    def __init__(self, sym_ops, symmetries_dict, normed_lattice_vectors, atom_weights, dataDims, new_generation=False):
        self.sym_ops = sym_ops
        self.atom_weights = atom_weights
        self.symmetries_dict = symmetries_dict
        self.dataDims = dataDims
        self.normed_lattice_vectors = normed_lattice_vectors
        self.new_generation = new_generation
        # confirm sym ops we are using agree with these settings
        # todo confirm these make right crystals
        self.asym_unit_dict = { # https://www.lpl.arizona.edu/PMRG/sites/lpl.arizona.edu.PMRG/files/ITC-Vol.A%20%282005%29%28ISBN%200792365909%29.pdf
            '1':[1,1,1], # P1
            '2':[.5,1,1], # P-1
            '3': [1, 1, 0.5],  # P2
            '4': [1, 1, 0.5],  # P21
            '5': [0.5, 0.5, 1],  # C2
            '6': [1, 0.5, 1],  # Pm
            '7': [1, 0.5, 1],  # Pc
            '8': [1, 0.25, 1],  # Cm
            '9': [1, 0.25, 1],  # Cc
            '10': [.5, 0.5, 1],  # P2/m
            '11': [1, 0.25, 1],  # P21/m
            '12': [.5, 0.25, 1],  # C2/m
            '13': [.5, 1, 0.5],  # P2/c
            '14': [1, 0.25, 1],  # P21/c
            '15': [0.5, 0.5, 0.5],  # C2/c
            '16': [0.5, 0.5, 1],  # P222
            '17': [0.5, 0.5, 1],  # P2221
            '18': [0.5, 0.5, 1],  # P21212
            '19': [0.5, 0.5, 1],  # P212121
            '20': [0.5, 0.5, 0.5],  # C2221
            '21': [0.25, 0.5, 1],  # C222
        }
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key])

    def build_supercells(self, supercell_data, cell_sample, supercell_size, graph_convolution_cutoff, target_handedness=None,
                         do_on_cpu=True, override_sg=None, skip_cell_cleaning=False, ref_data=None, debug=False,
                         standardized_sample=True):
        '''
        convert cell parameters to reference cell in a fast, differentiable, invertible way
        convert reference cell to NxN supercell
        all using differentiable torch functions
        '''
        '''
        get data on correct device
        '''
        orig_device = supercell_data.x.device
        if do_on_cpu:
            supercell_data = supercell_data.cpu()
            if cell_sample is not None:
                cell_sample = cell_sample.cpu()
            if ref_data is not None:
                ref_data = ref_data.cpu()

        if target_handedness is not None: # todo update this when we finalize generator
            target_handedness = target_handedness.to(supercell_data.x.device)

        '''
        if searching a specific space group, override the relevant information here
        '''
        if override_sg is not None:
            sym_ops_list = [torch.Tensor(self.symmetries_dict['sym_ops'][override_sg]).to(supercell_data.x.device) for i in range(supercell_data.num_graphs)]
            supercell_data = override_sg_info(override_sg, self.dataDims, supercell_data, self.symmetries_dict)  # todo update the way we handle this
        else:
            sym_ops_list = [torch.Tensor(supercell_data.symmetry_operators[n]).to(supercell_data.x.device) for n in range(len(supercell_data.symmetry_operators))]

        '''
        if copying a reference, extract its parameters
        '''
        # TODO update analysis with new/finalized asymmetric unit method for all space groups
        if ref_data is not None:  # extract parameters directly from the ref_data
            cell_sample_i, target_handedness, ref_final_coords = \
                cell_analysis(ref_data.clone(), self.atom_weights,debug=debug,
                              return_final_coords=True, return_sym_ops=False)
            sym_ops_list = [torch.Tensor(ref_data.symmetry_operators[n]).to(supercell_data.x.device) for n in range(len(ref_data.symmetry_operators))]
            target_handedness = target_handedness.to(supercell_data.x.device)

            if standardized_sample:
                cell_sample = (cell_sample_i - torch.Tensor(self.dataDims['lattice means'])) / torch.Tensor(self.dataDims['lattice stds'])  # standardize
            else:
                cell_sample = cell_sample_i  # leave it

        '''
        process the cell parameters
        & denormalize the cell length
        '''
        cell_lengths, cell_angles, mol_position, mol_rotation = \
            self.process_cell_params(supercell_data, cell_sample, skip_cell_cleaning, standardized_sample)

        # todo - for fractional rotations, get this inside the above function, without breaking autograd. Good Luck
        T_fc_list, T_cf_list, generated_cell_volumes = coor_trans_matrix(cell_lengths, cell_angles)

        '''
        update cell params
        '''
        supercell_data.T_fc = T_fc_list
        supercell_data.cell_params = torch.cat((cell_lengths, cell_angles, mol_position, mol_rotation), dim=1)  # lengths are destandardized here

        '''
        collect molecule info
        '''
        coords_list = []
        # masses_list = []
        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_list.append(supercell_data.x[supercell_data.batch == i])
            coords_list.append(supercell_data.pos[supercell_data.batch == i])
            # masses_list.append(torch.tensor([self.atom_weights[int(number)] for number in atomic_numbers]).to(supercell_data.x.device)) # don't need these


        if False: # mol is pre-standardized and right-handed in the generator now
            '''
            determine standardization rotation
            '''
            # we want all rotations from right-handed orientations to right-handed orientations
            Ip_axes_list = compute_principal_axes_list(coords_list)
            mol_handedness = compute_Ip_handedness(Ip_axes_list)  # torch.sign(torch.mul(Ip_axes_list[:, 0], torch.cross(Ip_axes_list[:, 1], Ip_axes_list[:, 2], dim=1)).sum(1))
            if target_handedness is not None:  # if we are supplied with a certain handedness, enforce it # todo pointless for generator which pre-enforces right-handedness
                disagreements = torch.where(mol_handedness != target_handedness)[0]
                for n in range(len(disagreements)):  # invert molecules which disagree
                    coords_list[disagreements[n]] = invert_coords(coords_list[disagreements[n]])

            if debug:
                Ip_axes_list = compute_principal_axes_list(coords_list)
                mol_handedness = compute_Ip_handedness(Ip_axes_list)  # torch.sign(torch.mul(Ip_axes_list[:, 0], torch.cross(Ip_axes_list[:, 1], Ip_axes_list[:, 2], dim=1)).sum(1))
                assert torch.sum(mol_handedness == target_handedness) == len(mol_handedness)

            normed_alignment_target_list = torch.stack([torch.eye(3) for n in range(len(coords_list))]).to(coords_list[0].device)
            if target_handedness is not None:  # accomodate for left-handed targets
                normed_alignment_target_list[:, 0, 0] = target_handedness

            # rotation matrix between target and current Ip axes
            std_rotations_list = torch.matmul(normed_alignment_target_list.permute(0, 2, 1), torch.linalg.inv(Ip_axes_list).permute(0, 2, 1))  # RMAT = X_new * X_old^(-1)

        if self.new_generation:
            # in new generation mode, we don't have to identify canonical conformer as it's set automatically
            canonical_fractional_centroids_list = mol_position
        else:
            canonical_fractional_centroids_list = self.get_canonical_conformer(supercell_data, mol_position, sym_ops_list, debug=(ref_data is not None) and debug)

            if (ref_data is not None) and debug: # todo this probably has to be rewritten if we want to use ref data
                # apply the rotation and check it
                std_coords_list = []
                for i, (rotation, coords, T_fc, new_frac_pos) in enumerate(zip(std_rotations_list, coords_list, T_fc_list, canonical_fractional_centroids_list)):
                    std_coords_list.append(
                        torch.inner(rotation, coords - coords.mean(0)).T + torch.inner(T_fc, new_frac_pos)
                    )
                Ip_axes_list = compute_principal_axes_list(std_coords_list)
                assert F.l1_loss(Ip_axes_list, normed_alignment_target_list, reduction='mean') < 0.1

        '''
        get applied rotation
        '''
        applied_rotation_list = self.rotvec_to_rmat(mol_rotation)

        '''
        do standardization & applied rotations & apply translation
        '''
        final_coords_list = []
        #rotations_list = torch.matmul(applied_rotation_list, std_rotations_list)  # list of rotations to apply - order is important since these do not commute
        rotations_list = applied_rotation_list  # only now applying the rotation matrix - no standardization

        for i, (rotation, coords, T_fc, new_frac_pos) in enumerate(zip(rotations_list, coords_list, T_fc_list, canonical_fractional_centroids_list)):
            final_coords_list.append(
                torch.inner(rotation, coords - coords.mean(0)).T + torch.inner(T_fc, new_frac_pos)
            )

        if (ref_data is not None) and debug:
            # test if the rotation went through correctly
            Ip_axes_list = compute_principal_axes_list(final_coords_list)
            target_Ip_axes_list = compute_principal_axes_list(ref_final_coords)

            assert F.l1_loss(Ip_axes_list, target_Ip_axes_list, reduction='mean') < 0.01  # sometimes the rotations aren't perfect, what can I say

        '''
        apply point symmetry in Z-batches for speed
        '''
        reference_cell_list = self.build_unit_cell(supercell_data.clone(), final_coords_list, T_fc_list, T_cf_list, sym_ops_list, debug=(ref_data is not None) and debug)

        '''
        generate supercells
        '''
        cell_vector_list = T_fc_list.permute(0, 2, 1)  # cell_vectors(T_fc_list)  # I think this just IS the T_fc matrix
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            ref_to_supercell(reference_cell_list, cell_vector_list, T_fc_list, atoms_list, supercell_data.Z,
                             supercell_scale=supercell_size, cutoff=graph_convolution_cutoff)

        overlaps_list = None  # expensive and not currently used # compute_lattice_vector_overlap(final_coords_list, T_cf_list, normed_lattice_vectors=self.normed_lattice_vectors.to(supercell_data.x.device))

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list)

        #
        # if return_energy:  # expensive
        #     mols = [Atoms(positions=reference_cell_list[n].cpu().detach().numpy().reshape(reference_cell_list[n].shape[1] * reference_cell_list[n].shape[0], 3),
        #                   symbols=atoms_list[n][:, 0].repeat(supercell_data.Z[n]).cpu().detach().numpy(),
        #                   cell=supercell_data.T_fc[n].T.cpu().detach().numpy()
        #                   ) for n in range(len(reference_cell_list))]
        #
        #     pot_en = np.zeros(len(mols))
        #     for i, mol in enumerate(mols):
        #         mol.calc = lj.LennardJones()
        #         mol.set_pbc([True, True, True])
        #         pot_en[i] = mol.get_potential_energy() / len(mol)
        #
        #     return supercell_data.to(orig_device), generated_cell_volumes.to(orig_device), overlaps_list, pot_en
        #
        # else:
        return supercell_data.to(orig_device), generated_cell_volumes.to(orig_device), overlaps_list

    def real_cell_to_supercell(self, supercell_data, config, do_on_cpu=True, return_overlaps=False):
        '''
        should be faster than the old way
        pretty quick on cpu
        '''

        if do_on_cpu:
            supercell_data = supercell_data.cpu()

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
                             cutoff=config.discriminator.graph_convolution_cutoff)

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list)

        # if return_overlaps:  # todo finish this
        #     overlaps_list = compute_lattice_vector_overlap(masses_list, final_coords_list, T_cf_list, self.normed_lattice_vectors=self.self.normed_lattice_vectors)
        #     return supercell_data.to(config.device), overlaps_list
        # else:
        # if return_energy:
        #     mols = [Atoms(positions=supercell_data.ref_cell_pos[n].reshape(supercell_data.ref_cell_pos[n].shape[1] * supercell_data.ref_cell_pos[n].shape[0], 3),
        #                   symbols=atoms_list[n][:, 0].repeat(supercell_data.Z[n]).cpu().detach().numpy(),
        #                   cell=supercell_data.T_fc[n].T.cpu().detach().numpy()
        #                   ) for n in range(supercell_data.num_graphs)]
        #
        #     pot_en = np.zeros(len(mols))
        #     for i, mol in enumerate(mols):
        #         mol.calc = lj.LennardJones()
        #         mol.set_pbc([True, True, True])
        #         pot_en[i] = mol.get_potential_energy() / len(mol)
        #
        #     return supercell_data.to(config.device), pot_en
        # else:
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

        if self.new_generation:
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

        return applied_rotation_list

    def build_unit_cell(self, supercell_data, final_coords_list, T_fc_list, T_cf_list, sym_ops_list, debug=False):
        reference_cell_list_i = []

        unique_z_values = torch.unique(supercell_data.Z)
        z_inds = [torch.where(supercell_data.Z == z)[0] for z in unique_z_values]

        for i, (inds, z_value) in enumerate(zip(z_inds, unique_z_values)):
            # padding allows for parallel transforms below
            lens = torch.tensor([len(final_coords_list[ii]) for ii in inds])
            padded_coords_c = rnn.pad_sequence(final_coords_list, batch_first=True)[inds]
            centroids_c = torch.stack([final_coords_list[inds[ii]].mean(0) for ii in range(len(inds))])
            centroids_f = torch.einsum('nij,nj->ni', (T_cf_list[inds], centroids_c))

            if debug:
                assert torch.sum(centroids_f >= 1) == 0
                assert torch.sum(centroids_f < 0) == 0
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

        if debug:
            for i in range(supercell_data.num_graphs):  # some of the reference cells place centroids outside the box, in error. Not a problem most places, but weso we fix them here
                ref_centroids_f = torch.inner(T_cf_list[i], torch.Tensor(supercell_data.ref_cell_pos[i].mean(1))).T
                if (ref_centroids_f.max() >= 1) or (ref_centroids_f.min() < 0):
                    frac_vec = torch.floor(ref_centroids_f)
                    cart_vec = torch.inner(T_fc_list[i], frac_vec).T
                    for j in range(len(cart_vec)):
                        supercell_data.ref_cell_pos[i][j] -= cart_vec[j].cpu().detach().numpy()

                distmat = torch.cdist(torch.Tensor(supercell_data.ref_cell_pos[i].reshape(supercell_data.Z[i] * supercell_data.ref_cell_pos[i].shape[1], 3)),
                                      reference_cell_list[i].reshape(supercell_data.Z[i] * supercell_data.ref_cell_pos[i].shape[1], 3), p=2)

                assert (torch.sum(distmat < 0.05) / distmat.shape[0]) == 1
                # sg analysis
                # from pymatgen.symmetry import analyzer
                # from pymatgen.analysis import structure_matcher
                # from pymatgen.core import (structure, lattice)
                #
                # struc_lattice = lattice.Lattice(supercell_data.T_fc[i].T.type(dtype=torch.float16))
                # pymat_struc1 = structure.IStructure(species=supercell_data.x[supercell_data.batch == i, 0].repeat(supercell_data.Z[i]),
                #                                     coords=supercell_data.ref_cell_pos[i].reshape(int(supercell_data.Z[i] * len(supercell_data.pos[supercell_data.batch == i])), 3),
                #                                     lattice=struc_lattice, coords_are_cartesian=True)
                # sg_analyzer1 = analyzer.SpacegroupAnalyzer(pymat_struc1)
                #
                # struc_lattice = lattice.Lattice(supercell_data.T_fc[0].T.type(dtype=torch.float16))
                # pymat_struc2 = structure.IStructure(species=supercell_data.x[supercell_data.batch == i, 0].repeat(supercell_data.Z[i]),
                #                                     coords=reference_cell_list[i].reshape(int(supercell_data.Z[i] * len(supercell_data.pos[supercell_data.batch == i])), 3),
                #                                     lattice=struc_lattice, coords_are_cartesian=True)
                # sg_analyzer2 = analyzer.SpacegroupAnalyzer(pymat_struc2)

                # look at the offending structure
                # from ase.visualize import view
                # mols = [
                #     Atoms(positions=reference_cell_list[i].reshape(supercell_data.Z[i] * supercell_data.ref_cell_pos[i].shape[1], 3),
                #           symbols=supercell_data.x[supercell_data.batch == i, 0].repeat(int(supercell_data.Z[i])).cpu().detach(),
                #           cell=supercell_data.T_fc[i].T.cpu().detach()),
                #     Atoms(positions=supercell_data.ref_cell_pos[i].reshape(supercell_data.Z[i] * supercell_data.ref_cell_pos[i].shape[1], 3),
                #           symbols=supercell_data.x[supercell_data.batch == i, 0].repeat(int(supercell_data.Z[i])).cpu().detach(),
                #           cell=supercell_data.T_fc[i].T.cpu().detach())]
                # view(mols)

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

        #assert all(sg_ind == 2)
        '''
        P-1 asymmetric unit bounds
        x[0,.5], yz[0,1]
        '''
        scaled_mol_position = mol_position.clone()
        for i, ind in enumerate(sg_ind):
            scaled_mol_position[i, :] = mol_position[i, :] * self.asym_unit_dict[str(int(ind))]

        return scaled_mol_position
