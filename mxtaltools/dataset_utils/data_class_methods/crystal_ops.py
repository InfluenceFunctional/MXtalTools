from typing import Optional, Union, Iterable

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from mxtaltools.common.geometry_utils import enforce_crystal_system, batch_cell_vol_torch, \
    batch_compute_fractional_transform
from mxtaltools.common.utils import softplus_shift
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.space_group_info import SYM_OPS, LATTICE_TYPE
from mxtaltools.crystal_building.crystal_latent_transforms import CompositeTransform, AunitTransform, NiggliTransform, \
    StdNormalTransform, enforce_niggli_plane
from mxtaltools.crystal_building.random_crystal_sampling import sample_aunit_lengths, sample_cell_angles, \
    sample_aunit_orientations, sample_aunit_centroids
from mxtaltools.crystal_building.utils import parameterize_crystal_batch, canonicalize_rotvec
from mxtaltools.crystal_search.crystal_opt_utils import gradient_descent_optimization
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import get_mol_embedding_for_proxy, enforce_1d_bound


# noinspection PyAttributeOutsideInit
class MolCrystalOps:

    def assign_aunit_centroid(self, values, eps=1e-4):
        """never allow this to touch exactly 1"""
        return values.clip(min=0, max=1 - eps)

    def set_symmetry_attrs(self,
                           nonstandard_symmetry: bool,
                           sg_ind: Union[int, torch.tensor],
                           symmetry_operators: Optional[list] = None,
                           z_prime: Optional[Union[int, torch.tensor]] = None,
                           ):
        if not torch.is_tensor(sg_ind):
            self.sg_ind = torch.tensor(sg_ind, dtype=torch.long, device=self.device)
        else:
            self.sg_ind = sg_ind.long().to(self.device)

        if nonstandard_symmetry:  # set as np stack for correct collation behavior (we don't want batches to stack)
            if symmetry_operators is not None:
                assert isinstance(symmetry_operators, np.ndarray)
                self.symmetry_operators = symmetry_operators
            else:
                raise RuntimeError("symmetry_operators must be given for nonstandard symmetry operations")
            self.nonstandard_symmetry = True
        else:  # standard symmetry
            if symmetry_operators is not None:
                assert isinstance(symmetry_operators, np.ndarray)
                self.symmetry_operators = symmetry_operators
            else:
                self.symmetry_operators = np.stack(
                    SYM_OPS[int(sg_ind)])  # if saved as a tensor, we get collation issues
            self.nonstandard_symmetry = False

        # AKA "Z value"
        self.sym_mult = torch.ones(1, dtype=torch.long, device=self.device) * len(self.symmetry_operators)

        if z_prime is not None:
            if torch.is_tensor(z_prime):
                self.z_prime = z_prime.long().to(self.device)
            else:
                self.z_prime = torch.tensor(z_prime, dtype=torch.long, device=self.device)
        else:  # fall back / assume Z'=1 if not stated
            self.z_prime = torch.ones(1, dtype=torch.long, device=self.device)

    def set_mol_attrs(self, molecule):
        # copy out the data
        if isinstance(molecule, list):  # Z'>1 comes with Z' molecules
            mol_dicts = [mol.to_dict() for mol in molecule]

            for key in mol_dicts[0].keys():
                vals = [d[key] for d in mol_dicts]

                if key == 'identifier':
                    if isinstance(vals[0], str):
                        setattr(self, key, vals[0])
                    elif isinstance(vals[0][0], str):
                        setattr(self, key, vals[0][0])
                    else:
                        setattr(self, key, 'null')

                elif isinstance(vals[0], str):
                    setattr(self, key, "|".join(vals))

                elif isinstance(vals[0], (int, float)):
                    setattr(self, key, sum(vals))

                elif torch.is_tensor(vals[0]):
                    if vals[0].ndim == 0:  # scalar tensor → graphwise
                        setattr(self, key, torch.stack(vals).sum())
                    else:  # nodewise tensor → concat
                        setattr(self, key, torch.cat(vals, dim=0))




            #setattr(self, 'aunit_batch',
            #        torch.arange(len(molecule)).repeat_interleave(torch.tensor([m['num_atoms'] for m in molecule])))

        else:
            mol_dict = molecule.to_dict()
            for key, value in mol_dict.items():
                setattr(self, key, value)

            #setattr(self, 'aunit_batch', torch.zeros(self.num_atoms))

    def box_analysis(self):
        self.T_fc, self.T_cf, self.cell_volume = (
            batch_compute_fractional_transform(self.cell_lengths,
                                               self.cell_angles))
        try:
            self.T_cf = torch.linalg.inv(self.T_fc)
        except:
            aa = 1
        self.packing_coeff = self.mol_volume * self.sym_mult / self.cell_volume
        self.density = self.mass * self.sym_mult / self.cell_volume * 1.66054  # conversion from D/A^3 to g/cm^3

    def sample_random_cell_lengths(self, target_packing_coeff: Optional = None):
        """
        NOTE depends on cell angles, so must do those first if resampling both
        """
        if self.is_batch:
            aunit_lengths = sample_aunit_lengths(self.num_graphs,
                                                 self.cell_angles,
                                                 self.mol_volume,
                                                 target_packing_coeff=target_packing_coeff)
        else:
            self_batch = collate_data_list([self])
            aunit_lengths = sample_aunit_lengths(1,
                                                 self_batch.cell_angles,
                                                 self_batch.mol_volume,
                                                 target_packing_coeff=target_packing_coeff)

        self.cell_lengths = self.scale_aunit_lengths_to_unit_cell(aunit_lengths)

    def sample_random_cell_angles(self):
        if self.is_batch:
            self.cell_angles = sample_cell_angles(self.num_graphs).to(self.device)
        else:
            self.cell_angles = sample_cell_angles(1).to(self.device)

    def sample_random_aunit_orientations(self):
        if self.is_batch:
            self.aunit_orientation = sample_aunit_orientations(self.num_graphs, z_prime=self.max_z_prime).to(self.device)
        else:
            self.aunit_orientation = sample_aunit_orientations(1,  z_prime=self.max_z_prime).to(self.device)

    def sample_random_aunit_centroids(self):
        if self.is_batch:
            aunit_centroid = self.assign_aunit_centroid(sample_aunit_centroids(self.num_graphs,
                                                                               z_prime=self.max_z_prime
                                                                               ).to(self.device))
        else:
            aunit_centroid = self.assign_aunit_centroid(sample_aunit_centroids(1,
                                                                               z_prime=self.max_z_prime
                                                                               ).to(self.device))

        self.aunit_centroid = self.scale_centroid_to_unit_cell(aunit_centroid)

    def sample_random_aunit_handedness(self):
        # randomize the handedness parameter for the asymmetric unit, including for Zp>12
        self.aunit_handedness = torch.randint(0, 2, (self.num_graphs, self.max_z_prime)) * 2 - 1

    def sample_random_crystal_parameters(self,
                                         target_packing_coeff: Optional = None,
                                         seed: Optional[int] = None,
                                         cleaning_mode: Optional[str] = 'hard'
                                         ):
        if seed is not None:
            torch.manual_seed(seed)
        self.sample_random_cell_angles()  # must do this one before cell lengths !!!
        self.sample_random_cell_lengths(target_packing_coeff)
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        if torch.any(self.z_prime > 1):
            self.sample_random_aunit_handedness()
        self.clean_cell_parameters(mode=cleaning_mode)
        self.box_analysis()

    def sample_random_reduced_crystal_parameters(self,
                                                 target_packing_coeff: Optional = None,
                                                 seed: Optional[int] = None,
                                                 ):
        """
        Sample random crystal parameters that are automatically Niggli-reduced
        """
        if seed is not None:
            torch.manual_seed(seed)
        assert self.is_batch, "Random crystal parameters not setup for single crystals"
        self.sample_reduced_box_vectors(target_packing_coeff=target_packing_coeff)
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        if torch.any(self.z_prime > 1):
            self.sample_random_aunit_handedness()
        # enforce agreement with crystal system
        # other cell parameters are valid by explicit construction
        # todo add crystal system conditions to sampling workflow, rather than cleaning up here
        self.cell_lengths, self.cell_angles = enforce_crystal_system(
            self.cell_lengths,
            self.cell_angles,
            self.sg_ind
        )
        self.box_analysis()

    def latent_to_cell_params(self, std_normal: torch.tensor):
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()

        if not hasattr(self, 'latent_transform'):
            self.init_latent_transform()

        self.set_cell_parameters(
            self.latent_transform.inverse(std_normal.clip(min=-6, max=6), self.sg_ind, self.radius)
        )

        self.cell_lengths, self.cell_angles = enforce_crystal_system(
            self.cell_lengths,
            self.cell_angles,
            self.sg_ind
        )
        self.box_analysis()

    def init_latent_transform(self,
                              c_log_mean=0.3,
                              c_log_std=0.25):
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()

        self.latent_transform = CompositeTransform([
            AunitTransform(asym_unit_dict=self.asym_unit_dict),
            NiggliTransform(),
            StdNormalTransform(c_log_mean=c_log_mean,
                               c_log_std=c_log_std),
            # SquashingTransform(min_val=-6,
            #                    max_val=6,
            #                    threshold=5.99,
            #                    ),
        ])

    def latent_params(self):
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()

        if not hasattr(self, 'latent_transform'):
            self.init_latent_transform()

        std_cell_params = self.latent_transform.forward(self.zp1_cell_parameters(), self.sg_ind, self.radius).clip(min=-6,
                                                                                                                   max=6)
        assert torch.isfinite(std_cell_params).all()
        return std_cell_params

    def sample_reduced_box_vectors(self, target_packing_coeff: Optional[float] = None):

        if not hasattr(self, 'latent_transform'):
            self.init_latent_transform()

        # use the latent transform for lengths and angles sampling, as its more stable and better tested
        # we can still use the old samplers for aunit params
        temp_params = self.latent_transform.inverse(torch.randn((self.num_graphs, 12), device=self.device),
                                                    self.sg_ind,
                                                    self.radius)
        cell_lengths = temp_params[:, :3]
        cell_angles = temp_params[:, 3:6]

        if target_packing_coeff is not None:
            vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
            cp1 = self.mol_volume * self.sym_mult / vol1
            correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
            cell_lengths *= correction_ratio[:, None]

        cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')

        self.cell_lengths, self.cell_angles = cell_lengths, cell_angles

    def sample_reasonable_random_parameters(self,
                                            tolerance: float = 1,
                                            target_packing_coeff: Optional = None,
                                            max_attempts: int = 100,
                                            seed: Optional[int] = None,
                                            sample_niggli: Optional[bool] = True):
        """
        Sample random crystal parameters
        build the resulting crystals and check their vdW overlaps
        If they are sufficiently small, retain that sample
        Repeat until convergence
        """
        good_params_tensor = torch.zeros_like(self.full_cell_parameters())
        found_params_flags = torch.zeros(self.num_graphs, dtype=torch.bool)
        best_ljs = torch.ones(self.num_graphs, dtype=torch.float32, device=self.device) * 1e7
        converged = False
        ind = 0
        while not converged and ind < max_attempts:
            if sample_niggli:
                self.sample_random_reduced_crystal_parameters(target_packing_coeff=target_packing_coeff,
                                                              seed=seed)
            else:
                self.sample_random_crystal_parameters(target_packing_coeff, seed=seed)

            _, _, scaled_lj = self.build_and_analyze(cutoff=3)
            improved_inds = torch.argwhere(scaled_lj < best_ljs)
            best_ljs[improved_inds] = scaled_lj[improved_inds]
            good_inds = torch.argwhere(scaled_lj < tolerance)
            good_params_tensor[improved_inds] = self.zp1_cell_parameters()[improved_inds]
            found_params_flags[good_inds] = True
            ind += 1
            if torch.all(found_params_flags):
                converged = True

        self.set_cell_parameters(good_params_tensor)
        #
        # if not converged:
        #     print(f"Failed to converge {torch.sum(~found_params_flags)} out of {self.num_graphs} crystals")

    def denorm_by_radius(self, arr):
        """
        arr *= mol_radius
        assumes arr shape [n, m]
        """
        if self.is_batch:
            return arr * self.radius[:, None]
        else:
            return arr * self.radius[None]

    def norm_by_radius(self, arr):
        """
        arr /= mol_radius
        assumes arr shape [n, m]
        """
        if self.is_batch:
            return arr / self.radius[:, None]
        else:
            return arr / self.radius[None]

    def build_asym_unit_dict(self,
                             force_all_sgs: Optional[bool] = False
                             ):  # todo add capability to only do this for the necessary keys
        """  # todo unify this throughout this file
        NOTE this function is extremely slow
        Best used during batch operations
        """
        asym_unit_dict = ASYM_UNITS.copy()
        if force_all_sgs:
            sgs_to_tensorize = asym_unit_dict.keys()
        else:
            relevant_sgs = self.sg_ind.unique()
            sgs_to_tensorize = [str(int(sg)) for sg in relevant_sgs]
        for key in sgs_to_tensorize:
            asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key]).to(self.device)
        return asym_unit_dict

    def scale_lengths_to_aunit(self):
        """
        scale unit cell lengths to the asymmetric unit, according to the canonical
        asymmetric unit shape within the unit cell.
        """
        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()

            return self.cell_lengths * torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return self.cell_lengths * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_aunit_lengths_to_unit_cell(self, aunit_lengths):
        """
        input asymmetric unit lengths
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
        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            return aunit_lengths / torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return aunit_lengths / torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_centroid_to_aunit(self):
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

        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            scales = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])

            return self.aunit_centroid / scales.repeat(1, self.max_z_prime)
        else:
            return self.aunit_centroid / torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_centroid_to_unit_cell(self, normed_centroid):
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
        if self.is_batch:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()

            scales = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
            return self.assign_aunit_centroid(normed_centroid * scales.repeat(1, self.max_z_prime))
        else:
            return self.assign_aunit_centroid(
                normed_centroid * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device).repeat(self.max_z_prime))

    def noise_cell_parameters(self, noise_level: float):
        # standardize
        std_cell_params = self.zp1_std_cell_parameters()
        # noise
        new_std_cell_params = std_cell_params + torch.randn_like(std_cell_params) * noise_level
        # destandardize
        destandardized_cell_params = self.destandardize_zp1_cell_parameters(new_std_cell_params)
        # assign
        self.set_cell_parameters(destandardized_cell_params, skip_box_analysis=True)
        # clean
        self.clean_cell_parameters(mode='hard')
        # refresh box
        self.box_analysis()

    def zp1_std_cell_parameters(self):
        """
        Return standardized zp=1 cell parameters
        write custom logic for zp>1 crystals
        :return:
        """
        standardized_aunit_lengths = self.standardize_cell_lengths()
        standardized_cell_angles = self.standardize_cell_angles()
        standardized_aunit_centroid = self.standardize_aunit_position()
        standardized_orientation = self.standardize_aunit_orientation()

        return torch.cat([standardized_aunit_lengths,
                          standardized_cell_angles,
                          standardized_aunit_centroid[:, :3],
                          standardized_orientation[:, :3]], dim=1)

    def destandardize_zp1_cell_parameters(self, std_cell_params):
        (std_aunit_lengths,
         std_cell_angles,
         std_aunit_positions,
         std_aunit_orientations) = std_cell_params.split(3, dim=1)

        cell_lengths = self.destandardize_cell_lengths(std_aunit_lengths)
        cell_angles = self.destandardize_cell_angles(std_cell_angles)
        aunit_positions = self.destandardize_aunit_position(std_aunit_positions)
        aunit_orientations = self.destandardize_aunit_orientation(std_aunit_orientations)

        return torch.cat([
            cell_lengths, cell_angles, aunit_positions, aunit_orientations
        ], dim=1)

    def crystal_system(self):
        if self.is_batch:
            return [LATTICE_TYPE[int(ind)] for ind in self.sg_ind]
        else:
            return LATTICE_TYPE[int(self.sg_ind)]

    def validate_cell_params(self, check_crystal_system: bool = True):
        """
        checking crystal system slightly slower, can optionally skip
        """
        self.validate_cell_params_ranges()
        if check_crystal_system:
            self.validate_crystal_system()
        return True

    def validate_cell_params_ranges(self, eps=1e-4):
        # assert valid ranges
        assert torch.all(self.aunit_centroid >= 0), "Aunit centroids must be greater than 0"
        assert torch.all(self.aunit_centroid <= 1 - eps), "Aunit centroids must be less than 0.999"
        assert torch.all(self.cell_lengths > 0), "Cell lengths must be positive"
        assert torch.all(self.cell_angles > 0), "Cell angles must be greater than 0"
        assert torch.all(self.cell_angles < torch.pi), "Cell angles must be less than pi"
        assert torch.all(torch.linalg.norm(self.aunit_orientation,
                                           dim=-1) <= 2 * torch.pi), "Cell orientation rotvec must have length <=2pi"
        assert torch.all(
            torch.linalg.norm(self.aunit_orientation, dim=-1) >= 0), "Cell orientation rotvec must have length >= 0"
        assert torch.all(self.aunit_orientation[:, -1] >= 0), "Cell orientation rotvec z component must be positive"

    def validate_crystal_system(self, atol=1e-3):  # todo cleanup/parallelize
        lattices = self.crystal_system()
        # enforce agreement with crystal system
        if self.is_batch:
            right_angle = torch.tensor(torch.pi / 2, dtype=torch.float32, device=self.device)
            oblate_angle = torch.tensor(2 * torch.pi / 3, dtype=torch.float32, device=self.device)
            for ind in range(self.num_graphs):
                lattice = lattices[ind]
                cell_lengths = self.cell_lengths[ind]
                cell_angles = self.cell_angles[ind]
                if lattice.lower() == 'triclinic':
                    pass
                elif lattice.lower() == 'monoclinic':  # fix alpha and gamma
                    assert torch.all(cell_angles[0] == torch.pi / 2), "Error in monoclinic alpha angle"
                    assert torch.all(cell_angles[2] == torch.pi / 2), "Error in monoclinic gamma angle"
                elif lattice.lower() == 'orthorhombic':  # fix all angles
                    assert torch.all(cell_angles == torch.ones(3) * torch.pi / 2), "Error in orthorhombic cell angles"
                elif lattice.lower() == 'tetragonal':  # fix all angles and a & b vectors
                    assert torch.all(cell_angles == torch.ones(3) * torch.pi / 2), "Error in tetragonal cell angles"
                    assert torch.all(cell_lengths[0] == cell_lengths[1]), "Error in tetragonal cell lengths"
                elif lattice.lower() == 'hexagonal':  # for rhombohedral, all angles and lengths equal, but not 90.
                    # for truly hexagonal, alpha=90, gamma is 120, a=b!=c
                    # todo check rhobohedral business
                    assert torch.allclose(cell_lengths[0], cell_lengths[1], atol=atol), "hexagonal a ≠ b"
                    assert not torch.allclose(cell_lengths[2], cell_lengths[0], atol=atol), "hexagonal requires c ≠ a"
                    assert torch.allclose(cell_angles[0], right_angle, atol=atol), "hexagonal alpha ≠ 90°"
                    assert torch.allclose(cell_angles[1], right_angle, atol=atol), "hexagonal beta ≠ 90°"
                    assert torch.allclose(cell_angles[2], oblate_angle, atol=atol), "hexagonal gamma ≠ 120°"
                    pass
                elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
                    assert torch.all(cell_lengths == cell_lengths.mean()), "Error in cubic cell lengths"
                    assert torch.all(cell_angles == torch.pi / 2), "Error in cubic cell angles"
                else:
                    raise RuntimeError(f"{lattice} + ' is not a valid crystal lattice!')")
        else:
            assert False, "Lattice checks not impelemented for single crystals"

        return True

    def reparameterize_unit_cell(self):
        (aunit_centroid, aunit_orientation,
         handedness_list, is_well_defined, pos) = (
            parameterize_crystal_batch(self,
                                       ASYM_UNITS,
                                       enforce_right_handedness=False,
                                       return_aunit=True))

        return aunit_centroid, aunit_orientation, handedness_list.long(), is_well_defined, torch.cat(pos)

    def compute_niggli_overlap(self, **kwargs):
        a, b, c, al, be, ga = self.zp1_cell_parameters()[:, :6].split(1, dim=1)
        ab = a * b
        ac = a * c
        bc = b * c

        al_cos = torch.cos(al)
        be_cos = torch.cos(be)
        ga_cos = torch.cos(ga)

        return (ab * ga_cos + ac * be_cos + bc * al_cos).flatten()

    def clean_cell_parameters(self, mode: str = 'hard',
                              canonicalize_orientations: bool = True,
                              angle_pad: float = 0.9,
                              length_pad: float = 3.0,
                              constrain_z: bool = False,
                              enforce_niggli: Optional[bool] = False):
        """
        force cell parameters into physical ranges
        """

        # cell lengths have to be positive nonzero
        if mode == 'hard':
            self.cell_lengths = self.cell_lengths.clip(min=length_pad)
        elif mode == 'soft':
            self.cell_lengths = softplus_shift(
                self.cell_lengths - length_pad) + length_pad  # enforces a minimum value of length_pad

        # range from (0,pi) with padding to prevent too-skinny cells
        self.cell_angles = enforce_1d_bound(self.cell_angles,
                                            x_span=torch.pi / 2 * angle_pad,
                                            x_center=torch.pi / 2,
                                            mode=mode)

        if enforce_niggli:
            self.enforce_niggli_box(mode)

        # positions must be on 0->1-eps in the asymmetric unit
        aunit_scaled_pos = self.scale_centroid_to_aunit()
        cleaned_aunit_scaled_pos = enforce_1d_bound(aunit_scaled_pos, x_span=0.5, x_center=0.5, mode=mode)
        self.aunit_centroid = self.assign_aunit_centroid(self.scale_centroid_to_unit_cell(cleaned_aunit_scaled_pos))

        # enforce range on vector norm
        if constrain_z:  # enforce a priori only positive Z possible
            assert not canonicalize_orientations, "Either enforce z positive, or post-canonicalize negative z's"
            # forces Z within the range 0-2pi
            new_orientation = []
            for zp in range(self.max_z_prime):
                fixed_z = enforce_1d_bound(
                    self.aunit_orientation[:, 2 + 3 * zp],
                    x_span=torch.pi,
                    x_center=torch.pi,
                    mode=mode
                )
                new_orientation.append(self.aunit_orientation[:, 3 * zp:3*zp+2])
                new_orientation.append(fixed_z[:, None])
            self.aunit_orientation = torch.cat(new_orientation, dim=1)

        new_orientation = []
        for zp in range(self.max_z_prime):
            norm = torch.linalg.norm(self.aunit_orientation[:, zp*3:3+zp*3], dim=1)
            new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode=mode)  # MUST be nonzero
            new_orientation.append(self.aunit_orientation[:, zp*3:3+zp*3] / norm[:, None] * new_norm[:, None])
        self.aunit_orientation = torch.cat(new_orientation, dim=1)


        # enforce agreement with crystal system
        self.cell_lengths, self.cell_angles = enforce_crystal_system(self.cell_lengths,
                                                                     self.cell_angles,
                                                                     self.sg_ind,
                                                                     )

        # enforce z component in the upper half-plane
        if canonicalize_orientations:  # converts the vector to its duplicate in the upper half-plane
            new_orientation = []
            for zp in range(self.max_z_prime):
                new_orientation.append(self.aunit_orientation[:, zp*3:3+zp*3])
            self.aunit_orientation = torch.cat(new_orientation, dim=1)

        # update cell vectors
        self.box_analysis()


    def enforce_niggli_box(self, mode, apply_plane_shift: bool = False):
        # enforce the scaling factor b/c and a/b in the range [0, 1]
        a, b, c = self.cell_lengths.split(1, 1)
        b_scale = enforce_1d_bound((b / c), 0.5, 0.5, mode=mode)
        a_scale = enforce_1d_bound((a / b), 0.5, 0.5, mode=mode)
        self.cell_lengths = torch.cat([
            a_scale * b_scale * c,
            b_scale * c,
            c],
            dim=1)

        # enforce the scaling factor cos(alpha) / (b/2c) in the range [0, 1]
        al_cos, be_cos, ga_cos = self.cell_angles.cos().split(1, 1)
        a, b, c = self.cell_lengths.split(1, 1)
        al_cos_max = (b / 2 / c)
        be_cos_max = (a / 2 / c)
        ga_cos_max = (a / 2 / b)

        # now improved - including obtuse cells
        al_cos_scale = enforce_1d_bound(al_cos / al_cos_max, 1.0, 0.0, mode=mode)
        be_cos_scale = enforce_1d_bound(be_cos / be_cos_max, 1.0, 0.0, mode=mode)
        ga_cos_scale = enforce_1d_bound(ga_cos / ga_cos_max, 1.0, 0.0, mode=mode)

        # limit it here due to instability in arccos
        al = torch.arccos(al_cos_max * al_cos_scale.clip(min=-0.99, max=0.99))
        be = torch.arccos(be_cos_max * be_cos_scale.clip(min=-0.99, max=0.99))
        ga = torch.arccos(ga_cos_max * ga_cos_scale.clip(min=-0.99, max=0.99))

        self.cell_angles = torch.cat([al, be, ga], dim=1)
        # here, enforce positivity of niggli overlap
        if apply_plane_shift:
            self.cell_angles = enforce_niggli_plane(self.cell_lengths,
                                                    self.cell_angles,
                                                    mode='shift')

    def niggli_angle_limits(self):
        a, b, c = self.cell_lengths.split(1, 1)
        al_cos_max = (b / 2 / c)
        be_cos_max = (a / 2 / c)
        ga_cos_max = (a / 2 / b)
        return torch.cat([al_cos_max, be_cos_max, ga_cos_max], dim=1)

    def aunit_volume(self):
        return self.cell_volume / self.sym_mult

    def zp1_cell_parameters(self):
        """
        return the zp=1 total cell parameter tensor
        for zp>1 crystals use custom logic
        Returns
        -------

        """
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.aunit_centroid[:, :3],
                          self.aunit_orientation[:, :3]], dim=1)

    def full_cell_parameters(self):
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.aunit_centroid,
                          self.aunit_orientation], dim=1)

    def standardize_cell_lengths(self):
        """
        Standardize the cell lengths tensor
        1. convert to asymmetric unit basis
        2. norm by mol radius
        3. standardize by stats
        mean = [1.0563, 0.7978, 1.7570]
        std = [0.6001, 0.5115, 0.7147]
        """
        aunit_lengths = self.scale_lengths_to_aunit()
        normed_aunit_lengths = self.norm_by_radius(aunit_lengths)
        # flatten stats across length dimensions
        mean = torch.tensor([1.0563, 0.7978, 1.7570], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.6001, 0.5115, 0.7147], dtype=torch.float32, device=self.device)
        mean = mean.mean() * torch.ones_like(mean)
        std = std.mean() * torch.ones_like(std)
        return (normed_aunit_lengths - mean[None, :]) / std[None, :]

    def destandardize_cell_lengths(self, std_aunit_lengths):
        """
        Destandardize the cell lengths tensor
        1. destandardize by stats
        2. denorm by mol radius
        3. convert to unit cell basis
        mean = [1.0563, 0.7978, 1.7570]
        std = [0.6001, 0.5115, 0.7147]
        """
        # flatten stats across length dimensions
        mean = torch.tensor([1.0563, 0.7978, 1.7570], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.6001, 0.5115, 0.7147], dtype=torch.float32, device=self.device)
        mean = mean.mean() * torch.ones_like(mean)
        std = std.mean() * torch.ones_like(std)
        normed_aunit_lengths = std_aunit_lengths * std[None, :] + mean[None, :]
        aunit_lengths = self.denorm_by_radius(normed_aunit_lengths)
        return self.scale_aunit_lengths_to_unit_cell(aunit_lengths)

    def standardize_cell_angles(self):
        """
        Using triclinic CSD stats
        mean = [1.5656, 1.5728, 1.5579]
        std = [0.2339, 0.2075, 0.2598]
        """
        mean = torch.tensor([1.5656, 1.5728, 1.5579], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.2339, 0.2075, 0.2598], dtype=torch.float32, device=self.device)
        return (self.cell_angles - mean[None, :]) / std[None, :]

    def destandardize_cell_angles(self, std_cell_angles):
        """
        Using triclinic CSD stats
        mean = [1.5656, 1.5728, 1.5579]
        std = [0.2339, 0.2075, 0.2598]
        """
        mean = torch.tensor([1.5656, 1.5728, 1.5579], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.2339, 0.2075, 0.2598], dtype=torch.float32, device=self.device)
        return std_cell_angles * std[None, :] + mean[None, :]

    def standardize_aunit_position(self):
        """
        uniform distribution on 0-1
        mean = 0.5
        std = 0.289
        """
        aunit_mean = 0.5
        aunit_std = 0.289
        return (self.scale_centroid_to_aunit() - aunit_mean) / aunit_std

    def destandardize_aunit_position(self, std_aunit_position):
        """
        uniform distribution on 0-1
        mean = 0.5
        std = 0.289
        """
        aunit_mean = 0.5
        aunit_std = 0.289
        return self.scale_centroid_to_unit_cell(std_aunit_position * aunit_std + aunit_mean)

    def standardize_aunit_orientation(self):
        """
        num_graphs = 100000
        random_vectors = torch.randn(size=(num_graphs, 3))

        # set norms uniformly between 0-2pi
        norms = random_vectors.norm(dim=1)
        applied_norms = (torch.rand(num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
        random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

        random_vectors = canonicalize_rotvec(random_vectors)
        std = random_vectors.std(0)
        mean = random_vectors.mean(0)

        functional only on zp=1 crystals. write custom logic for zp>1
        """
        orientation_stds = torch.tensor([[2.08, 2.08, 1.38]], dtype=torch.float32, device=self.device)
        orientation_means = torch.tensor([[0, 0, torch.pi / 2]], dtype=torch.float32, device=self.device)
        return (self.aunit_orientation[:, :3] - orientation_means) / orientation_stds

    def destandardize_aunit_orientation(self, std_aunit_orientation):
        # destandardize aunit orientation
        """
        num_graphs = 100000
        random_vectors = torch.randn(size=(num_graphs, 3))

        # set norms uniformly between 0-2pi
        norms = random_vectors.norm(dim=1)
        applied_norms = (torch.rand(num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
        random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

        random_vectors = canonicalize_rotvec(random_vectors)
        std = random_vectors.std(0)
        mean = random_vectors.mean(0)

        """
        orientation_stds = torch.tensor([[2.08, 2.08, 1.38]], dtype=torch.float32, device=self.device)
        orientation_means = torch.tensor([[0, 0, torch.pi / 2]], dtype=torch.float32, device=self.device)
        return std_aunit_orientation * orientation_stds + orientation_means

    def _build_lattice_feats(self):
        lattice_features = ['a length', 'b length', 'c length',
                            'alpha', 'beta', 'gamma']
        for zp in range(self.max_z_prime):
            lattice_features.extend([
                f'aunit{zp} x', f'aunit{zp} y', f'aunit{zp} z',
            ])
            lattice_features.extend([
                f'orientation{zp} 1', f'orientation{zp} 2', f'orientation{zp} 2'
            ])
        return lattice_features

    def _set_cell_ranges(self, space, fig):
        if space == 'latent':
            custom_ranges = {
                0: [-6.5, 6.5],  # for cell_a
                1: [-6.5, 6.5],  # for cell_b
                2: [-6.5, 6.5],  # for cell_c
                3: [-6.5, 6.5],  # for cell_alpha
                4: [-6.5, 6.5],  # for cell_beta
                5: [-6.5, 6.5],  # for cell_gamma
            }
            for ind in range(self.max_z_prime * 6):
                custom_ranges.update({ind: [-6.5, 6.5]})

            for i in range(6+self.max_z_prime*6):
                row = i // 3 + 1
                col = i % 3 + 1
                fig.update_xaxes(range=custom_ranges[i], row=row, col=col)

        elif space == 'stdandard':
            custom_ranges = {
                0: [-6.5, 6.5],  # for cell_a
                1: [-6.5, 6.5],  # for cell_b
                2: [-6.5, 6.5],  # for cell_c
                3: [-6.5, 6.5],  # for cell_alpha
                4: [-6.5, 6.5],  # for cell_beta
                5: [-6.5, 6.5],  # for cell_gamma
            }
            for ind in range(self.max_z_prime * 6):
                custom_ranges.update({ind: [-6.5, 6.5]})

            for i in range(6+self.max_z_prime*6):
                row = i // 3 + 1
                col = i % 3 + 1
                fig.update_xaxes(range=custom_ranges[i], row=row, col=col)


        elif space == 'real':
            pass

        return fig

    def _collect_sample_dists(self, samples, ref_dist, quantiles):
        num_dists = 1
        dist_names = ['Samples']
        dists = [samples]
        if ref_dist is not None:
            assert ref_dist.shape[1] == samples.shape[1]
            num_dists += 1
            dist_names += ['Reference']
            if torch.is_tensor(ref_dist):
                dists.append(ref_dist.cpu().detach().numpy())
            else:
                dists.append(ref_dist)

        if quantiles is not None:
            energies = self.lj_pot.cpu().detach().numpy()
            for q in quantiles:
                num_dists += 1
                dist_names += [f'{q} Quantile']
                dists.append(samples[energies < np.quantile(energies, q)])

        return num_dists, dist_names, dists

    def plot_batch_cell_params(self, space='real',
                               renderer: Optional[str] = None,
                               quantiles: Optional[Iterable[float]] = None,
                               ref_dist: Optional[torch.Tensor] = None,
                               show: bool = True,
                               return_fig: bool = False,):
        if not self.is_batch:
            print("Cell statistics only works for a batch of crystal data objects")
            return None

        lattice_features = self._build_lattice_feats()
        samples = self._get_samples(space)
        num_dists, dist_names, dists = self._collect_sample_dists(samples, ref_dist, quantiles)

        # 1d Histograms
        fig = make_subplots(rows=2 + 2 * self.max_z_prime,
                            cols=3,
                            subplot_titles=lattice_features)
        colors = self._get_color_set(num_dists)
        for i in range(len(lattice_features)):
            for j in range(num_dists):
                self._add_violin(
                    fig, dists[j], dist_names[j], colors[j], i
                )

        self._set_cell_ranges(space, fig)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
        fig.update_traces(opacity=0.5)
        if show:
            fig.show(renderer=renderer)
        if return_fig:
            return fig
        else:
            return None


    def _get_color_set(self, n):
        from plotly.colors import qualitative
        import plotly.colors as pc

        if n <= len(qualitative.Plotly):
            return qualitative.Plotly[:n]
        elif n <= len(qualitative.Dark24):
            return qualitative.Dark24[:n]
        else:
            return pc.n_colors('rgb(0,0,255)', 'rgb(255,0,0)', n, colortype='rgb')

    def _add_violin(self, fig, samples, name, color, column_index):
        row = column_index // 3 + 1
        col = column_index % 3 + 1
        fig.add_trace(go.Violin(
            x=samples[:, column_index],
            y=[0 for _ in range(len(samples))],
            side='positive',
            orientation='h',
            width=4,
            showlegend=True if column_index == 0 else False,
            name=name, legendgroup=name,
            meanline_visible=True,
            bandwidth=float(np.ptp(samples[:, column_index]) / 100),
            points=False,
            line_color=color,
        ),
            row=row, col=col
        )

    def _get_samples(self, space):
        if space == 'real':
            samples = self.full_cell_parameters().detach().cpu().numpy()
        elif space == 'latent':  # todo implement full latent method
            samples = self.latent_params().detach().cpu().numpy()
        elif space == 'standard':  # todo implement full std method
            samples = self.zp1_std_cell_parameters().detach().cpu().numpy()
        return samples

    def set_cell_parameters(self,
                            cell_parameters,
                            skip_box_analysis: bool = False):
        assert cell_parameters.shape[1] == 6 + self.max_z_prime * 6, (f"For crystal batch with max z_prime={self.max_z_prime},"
                                                                      f"require {6 + self.max_z_prime * 6} cell params.")

        self.cell_lengths = cell_parameters[:, :3]
        self.cell_angles = cell_parameters[:, 3:6]
        self.aunit_centroid = cell_parameters[:, 6:self.max_z_prime * 3 + 6]
        self.aunit_orientation = cell_parameters[:, self.max_z_prime * 3 + 6:]

        if not skip_box_analysis:
            self.box_analysis()

    def do_embedding(self,
                     embedding_type: str,
                     encoder: Optional = None):
        if self.is_batch:
            embedding = get_mol_embedding_for_proxy(self.clone(),
                                                    embedding_type,
                                                    encoder
                                                    )
            scaled_params = self.zp1_std_cell_parameters()
            return torch.cat([embedding, scaled_params], dim=1)
        else:
            assert False, "Crystal embedding not implemented for single samples"

    def optimize_crystal_parameters(self,
                                    mol_orientation: Optional[str] = 'standardized',
                                    return_record: Optional[bool] = False,
                                    **opt_kwargs):
        if self.is_batch:
            batch_to_optim = self.clone().to(self.device)
        else:
            batch_to_optim = collate_data_list([self]).clone().to(self.device)

        batch_to_optim.orient_molecule(mode=mol_orientation,
                                       target_handedness=self.aunit_handedness)
        opt_samples, opt_record = gradient_descent_optimization(
            batch_to_optim.full_cell_parameters(),
            batch_to_optim,
            **opt_kwargs
        )

        if return_record:
            return opt_samples, opt_record
        else:
            return opt_samples

    def _pad_tensor(self, val, shape, pad_val):
        out = torch.ones(shape, dtype=val.dtype) * pad_val
        out[:, :val.shape[-1]] = val
        return out


    def reset_sg_info(self, sg_ind):
        if isinstance(sg_ind, int):
            sg_ind_list = torch.ones_like(self.sg_ind) * sg_ind
        elif torch.is_tensor(sg_ind):
            sg_ind_list = sg_ind * 1
        else:
            raise TypeError("sg_ind must be a tensor or an integer")

        self.sg_ind = sg_ind_list

        # reset symmetries will always be standard
        self.nonstandard_symmetry[:] = False
        self.sym_mult = torch.tensor(
            [len(sym_ops) for sym_ops in self.symmetry_operators],
            dtype=torch.long, device=self.device
        )
