import torch
from crystal_building.utils import \
    (update_supercell_data, ref_to_supercell, clean_cell_output,
     align_crystaldata_to_principal_axes,
     batch_asymmetric_unit_pose_analysis_torch, set_sym_ops,
     rotvec2rotmat, build_unit_cell, scale_asymmetric_unit)
from common.geometry_calculations import compute_fractional_transform, sph2rotvec, rotvec2sph
from constants.asymmetric_units import asym_unit_dict


class SupercellBuilder:
    def __init__(self, symmetries_dict, dataDims, supercell_size=5, device='cuda', rotation_basis='spherical'):
        """
        class for converting single molecules -> unit cells -> supercells/clusters
        """

        self.sym_ops = symmetries_dict['sym_ops']  # list of symmetry operations
        self.symmetries_dict = symmetries_dict  # other symmetry information
        self.dataDims = dataDims  # information about the dataset
        self.device = device
        self.numpy_asym_unit_dict = asym_unit_dict.copy()
        self.asym_unit_dict = asym_unit_dict.copy()
        self.rotation_basis = rotation_basis
        for key in self.asym_unit_dict:
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

    def build_supercells(self, molecule_data, cell_sample: torch.tensor, supercell_size: int = 5,
                         graph_convolution_cutoff: float = 6, target_handedness=None,
                         align_molecules=True, pare_to_convolution_cluster=True):
        """
        convert cell parameters to unit cell in a fast, differentiable, invertible way
        convert reference cell to "supercell" (in fact, it's truncated to an appropriate cluster size)
        """
        supercell_data = molecule_data.clone()

        supercell_data, cell_sample, target_handedness = \
            self.move_cell_data_to_device(supercell_data, cell_sample, target_handedness)

        sym_ops_list, supercell_data = set_sym_ops(supercell_data)  # assign correct symmetry options

        cell_lengths, cell_angles, mol_position, mol_rotation_i = (
            cell_sample[:, :3], cell_sample[:, 3:6], cell_sample[:, 6:9], cell_sample[:, 9:])

        if self.rotation_basis == 'spherical':
            mol_rotation = sph2rotvec(mol_rotation_i)
        else:
            mol_rotation = mol_rotation_i

        # get transformation matrices
        T_fc_list, T_cf_list, generated_cell_volumes = compute_fractional_transform(cell_lengths, cell_angles)

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
        rotations_list = rotvec2rotmat(mol_rotation)

        # center, apply rotation, apply translation (to canonical conformer)
        canonical_conformer_coords_list = []
        for i, (rotation, coords, T_fc, new_frac_pos) in enumerate(zip(rotations_list, coords_list, T_fc_list, mol_position)):
            canonical_conformer_coords_list.append(
                torch.inner(rotation, coords - coords.mean(0)).T
                + torch.inner(T_fc, new_frac_pos)
            )

        # apply symmetry ops to build unit cell
        unit_cell_coords_list = build_unit_cell(supercell_data.Z, canonical_conformer_coords_list, T_fc_list, T_cf_list, sym_ops_list)

        # reanalyze the constructed unit cell to get the canonical orientation & confirm correct construction
        mol_positions, mol_orientations, mol_handedness = \
            batch_asymmetric_unit_pose_analysis_torch(
                unit_cell_coords_list,
                supercell_data.sg_ind,
                self.asym_unit_dict,
                supercell_data.T_fc,
                rotation_basis=self.rotation_basis,
                enforce_right_handedness=False)

        supercell_data.cell_params[:, 9:12] = mol_orientations  # overwrite to canonical parameters
        supercell_data.asym_unit_handedness = mol_handedness

        cell_vector_list = T_fc_list.permute(0, 2, 1)

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

        T_fc_list, T_cf_list, generated_cell_volumes = compute_fractional_transform(
            cell_lengths=supercell_data.cell_params[:, 0:3], cell_angles=supercell_data.cell_params[:, 3:6])

        atoms_list = []
        for i in range(supercell_data.num_graphs):
            atoms_i = supercell_data.x[supercell_data.batch == i]
            atoms_list.append(atoms_i)

        cell_vector_list = T_fc_list.permute(0, 2, 1)  # confirmed this is the right way to do this
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
                cell_lengths, cell_angles, mol_position, mol_rotation_i = destandardized_cell_sample.split(3, 1)
            else:
                cell_lengths, cell_angles, mol_position, mol_rotation_i = cell_sample.split(3, 1)

            if self.rotation_basis == 'spherical':
                mol_rotation = sph2rotvec(mol_rotation_i)
            else:
                mol_rotation = mol_rotation_i
        else:
            cell_lengths, cell_angles, mol_position, mol_rotation = cell_sample.split(3, 1)
            lattices = [self.symmetries_dict['lattice_type'][int(supercell_data.sg_ind[n])] for n in range(supercell_data.num_graphs)]
            cell_lengths, cell_angles, mol_position, mol_rotation, _, _, _ = \
                clean_cell_output(
                    cell_lengths, cell_angles, mol_position, mol_rotation, lattices,
                    self.dataDims['lattice means'], self.dataDims['lattice stds'],
                    enforce_crystal_system=True, return_transforms=True,
                    standardized_sample=standardized_sample, rotation_basis=self.rotation_basis)

        if rescale_asymmetric_unit:
            mol_position = scale_asymmetric_unit(self.asym_unit_dict, mol_position, supercell_data.sg_ind)

        return cell_lengths, cell_angles, mol_position, mol_rotation

    '''
    K = torch.stack((
                torch.stack((torch.zeros_like(unit_vector[:,0]),-unit_vector[:,2], unit_vector[:,1]),dim=1),
                  torch.stack((unit_vector[:,2], torch.zeros_like(unit_vector[:,0]), -unit_vector[:,0]),dim=1),
                  torch.stack((-unit_vector[:,1],unit_vector[:,0],torch.zeros_like(unit_vector[:,0])),dim=1)
                  ),dim=1)
      
    R = torch.eye(3)[None,:,:].tile(34,1,1) +torch.sin(theta[:,None,None])*K + (1-torch.cos(theta[:,None,None])) * (K@K)
    '''

    def move_cell_data_to_device(self, supercell_data, cell_sample, target_handedness):
        supercell_data = supercell_data.to(self.device)

        if cell_sample is not None:
            cell_sample = cell_sample.to(self.device)

        if target_handedness is not None:
            target_handedness = torch.tensor(target_handedness, device=self.device, dtype=torch.float32)  # target_handedness.to(self.device)

        return supercell_data, cell_sample, target_handedness
