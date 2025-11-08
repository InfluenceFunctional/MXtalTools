from typing import Optional
import torch
from scipy.spatial.transform import Rotation

from mxtaltools.common.geometry_utils import batch_compute_mol_radius, compute_mol_radius, batch_compute_mol_mass, \
    compute_mol_mass, batch_compute_molecule_volume, center_mol_batch, apply_rotation_to_batch, rotvec2rotmat, \
    rotmat2rotvec, batch_molecule_principal_axes_torch
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes, canonicalize_rotvec
from mxtaltools.dataset_utils.mol_building import smiles2conformer
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import compute_molecule_uma_on_mxt_batch
from mxtaltools.models.functions.radial_graph import build_radial_graph


# noinspection PyAttributeOutsideInit
class MolDataMethods:
    def mol_analysis(self,
                     force_reanalysis: Optional[bool] = False
                     ):
        # only piece of analysis
        if self.radius is None or force_reanalysis:
            self.radius = self.radius_calculation()
        if self.mass is None or force_reanalysis:
            self.mass = self.mass_calculation()
        if self.mol_volume is None or force_reanalysis:
            self.mol_volume = self.volume_calculation()

    @classmethod
    def from_smiles(cls,
                    smiles: str,
                    protonate: bool = True,
                    minimize: bool = False,
                    scramble_dihedrals: bool = False,
                    allow_methyl_rotations: bool = False,
                    compute_partial_charges: bool = True,
                    pare_to_size: Optional[int] = None,
                    max_pare_iters: int = 10,
                    do_mol_analysis: Optional[bool] = False,
                    ):
        conf_out = smiles2conformer(allow_methyl_rotations,
                                    compute_partial_charges,
                                    max_pare_iters,
                                    minimize,
                                    pare_to_size,
                                    protonate,
                                    scramble_dihedrals,
                                    smiles)
        if conf_out is not None:
            charges, pos, z = conf_out
            return cls(
                z=torch.LongTensor(z),
                pos=torch.tensor(pos, dtype=torch.float32),
                x=torch.tensor(charges, dtype=torch.float32),
                smiles=smiles,
                identifier=smiles,
                y=None,
                graph_x=None,
                do_mol_analysis=do_mol_analysis,
            )
        else:
            return None

    def construct_intra_radial_graph(self, cutoff: float = 6):
        self.edges_dict = build_radial_graph(self.pos,
                                             self.batch,
                                             self.ptr,
                                             cutoff,
                                             max_num_neighbors=10000
                                             )
        self.edge_index = self.edges_dict['edge_index']

    def radius_calculation(self):
        if self.is_batch:
            return batch_compute_mol_radius(self.pos, self.batch, self.num_graphs, self.num_atoms)
        else:
            return compute_mol_radius(self.pos)

    def mass_calculation(self):
        masses_tensor = torch.tensor(list(ATOM_WEIGHTS.values()), device=self.z.device, dtype=torch.float32)
        if self.is_batch:
            return batch_compute_mol_mass(self.z, self.batch, masses_tensor, self.num_graphs)
        else:
            return compute_mol_mass(self.z, masses_tensor)

    def volume_calculation(self):
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()),
                                        device=self.z.device,
                                        dtype=torch.float32)
        if self.is_batch:
            return batch_compute_molecule_volume(
                self.z,
                self.pos,
                self.batch,
                self.num_graphs,
                vdw_radii_tensor)
        else:
            return batch_compute_molecule_volume(
                self.z,
                self.pos,
                torch.zeros_like(self.z),
                1,
                vdw_radii_tensor)[0]

    def recenter_molecules(self):
        if self.is_batch:
            self.pos = center_mol_batch(self.pos, self.batch, self.num_graphs, self.num_atoms)
        else:
            self.pos -= self.pos.mean(0)

    def noise_positions(self, magnitude: float):
        self.pos += torch.randn_like(self.pos) * magnitude

    def scale_positions(self, affine_scale: float):
        graph_scale_factor = torch.randn(self.num_graphs, device=self.pos.device, dtype=torch.float32)
        graph_scale_factor = (graph_scale_factor / 10 + affine_scale).clip(min=0.9, max=1.25)
        atomwise_scaling = graph_scale_factor.repeat_interleave(self.num_atoms)
        self.pos *= atomwise_scaling[:, None]

    def visualize(self,
                  sample_inds: Optional[list] = None,
                  **vis_kwargs):
        from mxtaltools.common.ase_interface import data_batch_to_ase_mols_list
        return data_batch_to_ase_mols_list(
            self,
            specific_inds=sample_inds,
            show_mols=True,
            **vis_kwargs)

    def orient_molecule(self,
                        mode: str,
                        include_inversion: bool = True,
                        target_handedness: Optional[torch.LongTensor] = None,
                        correct_orientation: bool = False,
                        override_random_rotations: Optional[torch.Tensor] = None,
                        ):
        if self.is_batch:
            # always center the molecules
            self.recenter_molecules()
            if mode == 'standardized' or mode=='std' or mode=='standard':
                mol_batch, std_rotation = align_mol_batch_to_standard_axes(self,
                                                             handedness=target_handedness,
                                                                           return_rot=True)
                self.pos = mol_batch.pos
                applied_rotation = std_rotation#.permute(0, 2, 1)  # different basis, unfortunately
            elif mode == 'random':  # technically given the below we can do any applied rotation
                if override_random_rotations is not None:
                    random_rotations = override_random_rotations
                else:
                    random_rotations = torch.tensor(
                        Rotation.random(num=self.num_graphs).as_matrix(),
                        device=self.device, dtype=torch.float32)

                    if include_inversion:
                        assert not correct_orientation, "Cannot reconstruct rotations through inversion."
                        invert_inds = torch.randint(2, (self.num_graphs,), device=random_rotations.device) * 2 - 1
                        random_rotations *= invert_inds[:, None, None]

                self.pos = apply_rotation_to_batch(
                    self.pos,
                    random_rotations,
                    self.batch)

                applied_rotation = random_rotations

            else:
                assert False, "Rotation must be standard or random"

            if correct_orientation and hasattr(self, 'aunit_orientation'):
                assert self.max_z_prime == 1, "Mol orientation should be done on unzipped batches of independent Z'=1 crystals, not mixed Z'>1"
                # adjust aunit_orientation such that the final orientation is unchanged
                # new combined rotation should undo the random rotation, then apply the aunit_orientation
                invert_random_rotation = torch.linalg.inv(applied_rotation)
                new_orientation_matrix = rotvec2rotmat(self.aunit_orientation) @ invert_random_rotation
                new_rotvec = rotmat2rotvec(new_orientation_matrix)
                new_rotvec_fixed = canonicalize_rotvec(new_rotvec)
                # correction is successful when this is small
                # err = (rotvec2rotmat(new_rotvec_fixed) @ applied_rotation - rotvec2rotmat(
                #     self.aunit_orientation)).abs().sum(dim=(1, 2))
                # print(err.mean())
                self.aunit_orientation = new_rotvec_fixed

        else:
            assert False, "molecule orientation not implemented for single samples"

    def rotate_embedding(self, rotations):
        """
        NOTE this assumes our standard rotations & representations,
        i.e., the j dimension are the cartesian axes, the k dimension are feature channels
        """
        return torch.einsum('nij, njk -> nik', rotations, self.embedding)

    def deprotonate(self):
        """danger - this breaks several batching methods and should be used carefully
        """
        heavy_atom_inds = torch.argwhere(self.z != 1).flatten()  # protons are atom type 1
        if len(heavy_atom_inds) > 0:
            self.z = self.z[heavy_atom_inds]
            self.pos = self.pos[heavy_atom_inds]
            if hasattr(self, 'batch') and self.batch is not None:
                self.batch = self.batch[heavy_atom_inds]
                a, b = torch.unique(self.batch, return_counts=True)
                self.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
                self.num_atoms = torch.diff(self.ptr).long()
            else:
                self.num_atoms = len(self.z)
            self.num_nodes = len(self.z)

    def compute_Ip(self):
        assert self.is_batch

        principal_axes, principal_moments, _ = batch_molecule_principal_axes_torch(
            self.pos,
            self.batch,
            self.num_graphs,
            self.num_atoms,
        )

        return principal_axes, principal_moments

    def compute_molecule_uma(self,
                            predictor,
                            ):
        if self.is_batch:
            return compute_molecule_uma_on_mxt_batch(self,
                                            predictor)
        else:
            return compute_molecule_uma_on_mxt_batch(collate_data_list(self),
                                            predictor)
