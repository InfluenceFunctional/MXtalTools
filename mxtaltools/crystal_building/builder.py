import torch
from torch_geometric.loader.dataloader import Collater

from mxtaltools.common.utils import init_sym_info
from mxtaltools.crystal_building.utils import \
    (update_supercell_data, unit_cell_to_convolution_cluster,
     align_crystaldata_to_principal_axes,
     batch_asymmetric_unit_pose_analysis_torch, set_sym_ops,
     rotvec2rotmat, build_unit_cell, generate_sorted_fractional_translations)
from mxtaltools.common.geometry_calculations import compute_fractional_transform_torch, sph2rotvec
from mxtaltools.constants.asymmetric_units import asym_unit_dict


class SupercellBuilder:
    def __init__(self, supercell_size=5, device='cuda', rotation_basis='spherical'):
        """
        class for converting single molecules -> unit cells -> supercells/clusters
        """
        self.symmetries_dict = init_sym_info()  # other symmetry information
        self.sym_ops = self.symmetries_dict['sym_ops']  # list of symmetry operations
        self.device = device
        self.numpy_asym_unit_dict = asym_unit_dict.copy()
        self.asym_unit_dict = asym_unit_dict.copy()
        self.rotation_basis = rotation_basis
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(device)

        self.sorted_fractional_translations = generate_sorted_fractional_translations(supercell_size).to(device)
        self.collater = Collater(None, None)

    def build_integer_zp_supercells(self,
                                    molecule_data,
                                    cell_parameters: torch.tensor,
                                    supercell_size: int = 5,
                                    graph_convolution_cutoff: float = 6,
                                    z_primes_list: list = None,
                                    target_handedness=None,
                                    align_to_standardized_orientation=False,
                                    pare_to_convolution_cluster=True,
                                    skip_refeaturization=False):

        molecule_data = molecule_data.clone()
        num_crystals = molecule_data.num_graphs
        # infer maximum z prime value
        max_z_prime_explicit = max(z_primes_list)  # the maximum value proposed
        max_z_prime_params = (cell_parameters.shape[1] - 6) // 6  # the maximal value, from the given parameterization
        tot_molecules = sum(z_primes_list)

        molwise_data, molwise_parameters = self.expand_integer_zp_molecule_data(cell_parameters, molecule_data,
                                                                                tot_molecules, z_primes_list)

        # generate unit cells for each effective Z'=1 crystal
        (T_cf_list, T_fc_list, atomic_number_list, canonical_conformer_coords_list,
         generated_cell_volumes, molwise_data, sym_ops_list) = (
            self.build_zp_1_asymmetric_unit(
                align_to_standardized_orientation, molwise_parameters, molwise_data, target_handedness))

        # apply symmetry ops to build unit cell
        unit_cell_coords_list = build_unit_cell(
            molwise_data.mult, canonical_conformer_coords_list, T_fc_list,
            T_cf_list, sym_ops_list)

        if not skip_refeaturization:  # if the mol position is outside the asym unit, the below params will not correspond to the inputs
            self.refeaturize_generated_cell(molwise_data, unit_cell_coords_list)

        (combined_T_fc_list, combined_atomic_number_list,
         combined_unit_cell_coords_list, mol_size_list,
         mol_ind_list) = (
            self.contract_zp_unit_cells(
                T_fc_list, atomic_number_list, num_crystals, unit_cell_coords_list, z_primes_list))

        # get minimal supercell cluster for convolving about a given canonical conformer
        molecule_data.T_fc = T_fc_list
        cell_vector_list = T_fc_list.permute(0, 2, 1)  # transpose T_fc to cell vectors
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            unit_cell_to_convolution_cluster(
                combined_unit_cell_coords_list, cell_vector_list,
                combined_T_fc_list, combined_atomic_number_list, molecule_data.mult,
                supercell_scale=supercell_size,
                cutoff=graph_convolution_cutoff,
                sorted_fractional_translations=self.sorted_fractional_translations,
                pare_to_convolution_cluster=pare_to_convolution_cluster)

        supercell_data = update_supercell_data(molecule_data, supercell_atoms_list,
                                               supercell_list, ref_mol_inds_list,
                                               combined_unit_cell_coords_list)
        supercell_data.mol_size = mol_size_list
        supercell_data.mol_ind = torch.cat([  # if supercells are always indexed in increments of whole asymmetric units, this is valid
            mol_ind_list[ind].repeat(n_copies[ind]) for ind in range(supercell_data.num_graphs)
        ])

        # visualize samples
        # from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
        # import ase.io as io
        # for ind in range(supercell_data.num_graphs):
        #     mol = ase_mol_from_crystaldata(supercell_data, ind,
        #                                    highlight_canonical_conformer=False,
        #                                    exclusion_level='unit cell')
        #
        #     io.write(f'/home/mkilgour/gflownet-dev/sample_{ind}.cif', mol)

        return supercell_data, generated_cell_volumes

    def contract_zp_unit_cells(self, T_fc_list, atomic_number_list,
                               num_crystals, unit_cell_coords_list,
                               z_primes_list):
        # contract zp > 1 unit cells to single asymmetric unit
        # generate a batch of effective zp=1 molecules
        combined_unit_cell_coords_list = []
        combined_atomic_number_list = []
        mol_ind_list = []  # which atoms are assigned to which molecules in the asymmetric unit
        combined_T_fc_list = torch.zeros((num_crystals, 3, 3), dtype=torch.float32, device=self.device)
        mol_sizes = torch.zeros(num_crystals, dtype=torch.long, device=self.device)

        mol_ind = 0
        for ind, zp in enumerate(z_primes_list):
            combined_unit_cell = torch.cat([
                unit_cell_coords_list[mol_ind + ind2] for ind2 in range(zp)
            ], dim=1)

            combined_atoms = torch.cat([
                atomic_number_list[mol_ind + ind2] for ind2 in range(zp)
            ], dim=0)

            mol_inds = torch.cat([
                torch.ones(len(atomic_number_list[mol_ind + ind2]), dtype=torch.long, device=self.device) * ind2 for ind2 in range(zp)
            ], dim=0)

            combined_unit_cell_coords_list.append(combined_unit_cell)
            combined_atomic_number_list.append(combined_atoms)
            mol_ind_list.append(mol_inds)
            combined_T_fc_list[ind, :, :] = T_fc_list[mol_ind, :, :]
            mol_sizes[ind] = len(combined_unit_cell[0])  # NOTE this is really the asymmetric unit size

            mol_ind += zp

        assert mol_ind == sum(z_primes_list), "Indexing error between molecules and z-primes in unit cell contraction"

        return combined_T_fc_list, combined_atomic_number_list, combined_unit_cell_coords_list, mol_sizes, mol_ind_list

    def expand_integer_zp_molecule_data(self, cell_parameters, molecule_data, tot_molecules, z_primes_list):
        # generate a batch of effective zp=1 molecules
        molwise_data = []
        molwise_parameters = torch.zeros((tot_molecules, 12), dtype=torch.float32, device=self.device)
        molwise_sg_inds = torch.zeros((tot_molecules), dtype=torch.long, device=self.device)
        molwise_sym_ops = []
        mol_ind = 0
        for ind, zp in enumerate(z_primes_list):
            molwise_data.extend([molecule_data[ind] for z in range(zp)])
            for ind2 in range(zp):  # assume zp elements are ordered as [6 box params, 6 mol params1, 6 mol params2,...]
                molwise_parameters[mol_ind] = torch.cat(
                    [cell_parameters[ind, :6],
                     cell_parameters[ind, 6 + ind2 * 6: 12 + ind2 * 6]],
                    dim=0)
                molwise_sg_inds[mol_ind] = molecule_data.sg_ind[ind]
                molwise_sym_ops.append(molecule_data.symmetry_operators[ind])
                mol_ind += 1
        molwise_data = self.collater(molwise_data)  #
        molwise_data.cell_params = molwise_parameters
        molwise_data.sg_ind = molwise_sg_inds
        molwise_data.symmetry_operators = molwise_sym_ops
        return molwise_data, molwise_parameters

    def build_zp1_supercells(self,
                             molecule_data,
                             cell_parameters: torch.tensor,
                             supercell_size: int = 5,
                             graph_convolution_cutoff: float = 6,
                             target_handedness=None,
                             align_to_standardized_orientation=False,
                             pare_to_convolution_cluster=True,
                             skip_refeaturization=False):

        """
        convert cell parameters to unit cell in a fast, differentiable, invertible way
        convert reference cell to "supercell" (in fact, it's truncated to an appropriate cluster size)
        """

        T_cf_list, T_fc_list, atomic_number_list, canonical_conformer_coords_list, generated_cell_volumes, supercell_data, sym_ops_list = self.build_zp_1_asymmetric_unit(
            align_to_standardized_orientation, cell_parameters, molecule_data, target_handedness)

        # apply symmetry ops to build unit cell
        unit_cell_coords_list = build_unit_cell(supercell_data.mult, canonical_conformer_coords_list, T_fc_list,
                                                T_cf_list, sym_ops_list)

        #assert torch.sum(torch.isnan(torch.cat([elem.flatten() for elem in unit_cell_coords_list]))) == 0, f"{cell_parameters}, {coords_list}, {canonical_conformer_coords_list}"

        if not skip_refeaturization:  # if the mol position is outside the asym unit, the below params will not correspond to the inputs
            self.refeaturize_generated_cell(supercell_data, unit_cell_coords_list)

        # get minimal supercell cluster for convolving about a given canonical conformer
        cell_vector_list = T_fc_list.permute(0, 2, 1)
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            unit_cell_to_convolution_cluster(
                unit_cell_coords_list, cell_vector_list, T_fc_list, atomic_number_list, supercell_data.mult,
                supercell_scale=supercell_size,
                cutoff=graph_convolution_cutoff,
                sorted_fractional_translations=self.sorted_fractional_translations,
                pare_to_convolution_cluster=pare_to_convolution_cluster)

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list,
                                               unit_cell_coords_list)

        return supercell_data, generated_cell_volumes

    def build_zp_1_asymmetric_unit(self, align_to_standardized_orientation, cell_parameters, molecule_data,
                                   target_handedness):
        supercell_data = molecule_data.clone()
        supercell_data, cell_parameters, target_handedness = \
            self.move_cell_data_to_device(supercell_data, cell_parameters, target_handedness)

        # assumes cell params arrive appropriately pre-cleaned
        cell_lengths, cell_angles, mol_position, mol_rotation_i = (
            cell_parameters[:, :3], cell_parameters[:, 3:6], cell_parameters[:, 6:9], cell_parameters[:, 9:])

        if self.rotation_basis == 'spherical':
            mol_rotation = sph2rotvec(mol_rotation_i)
        else:
            mol_rotation = mol_rotation_i

        T_cf_list, T_fc_list, generated_cell_volumes, supercell_data, sym_ops_list = (
            self.get_symmetry_functions(cell_angles, cell_lengths, mol_position, mol_rotation, supercell_data))

        rotations_list = rotvec2rotmat(mol_rotation)

        if align_to_standardized_orientation:  # align canonical conformers principal axes to cartesian axes - not usually done here, but allowed
            supercell_data = align_crystaldata_to_principal_axes(supercell_data, handedness=target_handedness)

        # get molecule information
        atomic_number_list, coords_list = [], []
        for i in range(supercell_data.num_graphs):
            atomic_number_list.append(supercell_data.x[supercell_data.batch == i])
            coords_list.append(supercell_data.pos[supercell_data.batch == i])
        # center, apply rotation, apply translation (to canonical conformer)
        canonical_conformer_coords_list = []
        for i, (rotation, coords, T_fc, new_frac_pos) in enumerate(
                zip(rotations_list, coords_list, T_fc_list, mol_position)):
            canonical_conformer_coords_list.append(
                torch.inner(rotation, coords - coords.mean(0)).T + torch.inner(T_fc, new_frac_pos)
            )
        return T_cf_list, T_fc_list, atomic_number_list, canonical_conformer_coords_list, generated_cell_volumes, supercell_data, sym_ops_list

    def refeaturize_generated_cell(self, supercell_data, unit_cell_coords_list):
        # reanalyze the constructed unit cell to get the canonical orientation & confirm correct construction
        (mol_positions, mol_orientations,
         mol_handedness, asym_unit_is_well_defined,
         canonical_conformer_rebuild_list) = \
            batch_asymmetric_unit_pose_analysis_torch(
                unit_cell_coords_list,
                supercell_data.sg_ind,
                self.asym_unit_dict,
                supercell_data.T_fc,
                rotation_basis=self.rotation_basis,
                enforce_right_handedness=False,
                return_asym_unit_coords=True)
        # if not all(asym_unit_is_well_defined):  # todo solve this
        #     print("Warning: Some built crystals have ill defined asymmetric units")
        supercell_data.cell_params[:, 9:12] = mol_orientations  # overwrite to canonical parameters
        supercell_data.asym_unit_handedness = mol_handedness
        # if align_to_standardized_orientation:  # typically issue of handedness of the asymmetric unit
        #     if (torch.amax(torch.sum(torch.abs(mol_orientations - mol_rotation_i), dim=1)) > 1e-2 or
        #             torch.amax(torch.sum(torch.abs(mol_positions - mol_position), dim=1)) > 1e-2):  # in the spherical basis
        #         print("Warning: Rebuilt standardized crystal was not identical.")

    def get_symmetry_functions(self, cell_angles, cell_lengths, mol_position, mol_rotation, supercell_data):
        # get transformation matrices
        T_fc_list, T_cf_list, generated_cell_volumes = compute_fractional_transform_torch(cell_lengths, cell_angles)
        supercell_data.T_fc = T_fc_list
        supercell_data.cell_params = torch.cat((cell_lengths, cell_angles, mol_position, mol_rotation), dim=1)
        sym_ops_list, supercell_data = set_sym_ops(supercell_data)  # assign correct symmetry options
        return T_cf_list, T_fc_list, generated_cell_volumes, supercell_data, sym_ops_list

    def prebuilt_unit_cell_to_supercell(self, supercell_data,
                                        supercell_size=5,
                                        graph_convolution_cutoff=6,
                                        pare_to_convolution_cluster=True):
        """
        build a supercell cluster using a pre-built unit cell
        will not check for physicality or apply any symmetry options - merely pattern the unit cell
        and keep molecules within convolution radius of the canonical conformer
        automatically pare NxNxN supercell to minimal set of molecules in the convolution radius of the canonical conformer
        """
        supercell_data = supercell_data.clone().to(self.device)

        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_i = supercell_data.x[supercell_data.batch == i]
            atoms_list.append(atoms_i)

        cell_vector_list = supercell_data.T_fc.permute(0, 2, 1)  # confirmed this is the right way to do this
        supercell_list, supercell_atoms_list, ref_mol_inds_list, n_copies = \
            unit_cell_to_convolution_cluster(supercell_data.ref_cell_pos, cell_vector_list,
                                             supercell_data.T_fc, atoms_list, supercell_data.mult,
                                             supercell_scale=supercell_size, cutoff=graph_convolution_cutoff,
                                             sorted_fractional_translations=self.sorted_fractional_translations,
                                             pare_to_convolution_cluster=pare_to_convolution_cluster)

        supercell_data = update_supercell_data(supercell_data, supercell_atoms_list, supercell_list, ref_mol_inds_list,
                                               supercell_data.ref_cell_pos)

        return supercell_data.to(self.device)

    def move_cell_data_to_device(self, supercell_data, cell_sample, target_handedness):
        supercell_data = supercell_data.to(self.device)

        if cell_sample is not None:
            cell_sample = cell_sample.to(self.device)

        if target_handedness is not None:
            target_handedness = torch.tensor(target_handedness, device=self.device, dtype=torch.float32)

        return supercell_data, cell_sample, target_handedness
