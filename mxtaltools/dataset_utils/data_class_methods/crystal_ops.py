from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from ase.geometry import cellpar_to_cell, cell_to_cellpar
from plotly.subplots import make_subplots

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.ase_interface import ase_write_cif
from mxtaltools.common.geometry_utils import enforce_crystal_system, batch_compute_fractional_transform, \
    cart2sph_rotvec, sph2cart_rotvec, crystal_parameter_distmat, sph_rotvec2lat, lat2sph_rotvec, fractional_transform, \
    rotvec2rotmat, rotmat2rotvec
from mxtaltools.common.utils import softplus_shift, log_rescale_positive, get_point_density
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.space_group_info import SYM_OPS, LATTICE_TYPE, NORMALIZER_OPS
from mxtaltools.crystal_building.random_crystal_sampling import sample_aunit_lengths, sample_cell_angles, \
    sample_aunit_orientations, sample_aunit_centroids
from mxtaltools.crystal_building.utils import parameterize_crystal_batch, canonicalize_aunit_order, canonicalize_rotvec
from mxtaltools.crystal_search.crystal_opt_utils import gradient_descent_optimization
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import get_mol_embedding_for_proxy, enforce_1d_bound
from mxtaltools.reporting.utils import lightweight_one_sided_violin


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
        elif molecule.is_batch:
            mol_dict = molecule.to_dict()
            n_graphs, n_nodes = molecule.num_graphs, molecule.num_nodes
            nodes_per_graph = molecule.num_atoms
            slice_dict = torch.arange(0, n_graphs + 1, 1, device=molecule.device)
            inc_dict = torch.zeros(n_graphs, dtype=torch.long, device=molecule.device)
            for key, value in mol_dict.items():
                if isinstance(value, dict) or isinstance(value, torch.nn.Module):
                    continue

                if not torch.is_tensor(value) and not isinstance(value, str) and not (
                        isinstance(value, list) and len(value) > 0 and isinstance(value[0], str)
                ):
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        value = np.asarray(value)

                    value = torch.as_tensor(
                        value,
                        device=self.device,
                        dtype=torch.long if isinstance(value, int) else torch.float32,
                    )

                if len(value.size()) == 0:  # batch variables
                    setattr(self, key, value)
                elif len(value) == n_graphs:
                    self.add_graph_attr(value, key, slice_dict=slice_dict, inc_dict=inc_dict)
                elif len(value) == n_nodes:
                    if key != 'batch':
                        self.add_node_attr(value, key, num_nodes_per_graph=nodes_per_graph)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)  # batch and ptr come through here for data batch objects
        elif not self.is_batch:
            mol_dict = molecule.to_dict()
            for key, value in mol_dict.items():
                setattr(self, key, value)

    def box_analysis(self):
        self.T_fc, self.T_cf, self.cell_volume = (
            batch_compute_fractional_transform(self.cell_lengths,
                                               self.cell_angles))

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
            self.aunit_orientation = sample_aunit_orientations(
                self.num_graphs, z_prime=self.max_z_prime).to(self.device)
        else:
            self.aunit_orientation = sample_aunit_orientations(
                1, z_prime=self.max_z_prime).to(self.device)

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
        if seed is not None:  # todo this seeding method is wrong
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
        Sample random crystal parameters that are automatically standardized
        Repeat random box params until valid
        """
        if seed is not None:
            torch.manual_seed(seed)
        assert self.is_batch, "Random crystal parameters not setup for single crystals"
        valid = torch.zeros(self.num_graphs, dtype=bool, device=self.device)
        good_params = torch.zeros((self.num_graphs, 6), dtype=torch.float32, device=self.device)
        while not valid.all():
            self.sample_random_cell_angles()  # must do this one before cell lengths !!!
            self.sample_random_cell_lengths(target_packing_coeff)
            self.cell_lengths, self.cell_angles = enforce_crystal_system(
                self.cell_lengths,
                self.cell_angles,
                self.sg_ind
            )
            penalty = self.compute_cell_reduction_penalty()
            valid_sample = penalty == 0
            good_params[valid_sample] = torch.cat([self.cell_lengths[valid_sample], self.cell_angles[valid_sample]],
                                                  dim=1)
            valid[valid_sample] = True

        self.cell_lengths = good_params[:, :3]
        self.cell_angles = good_params[:, 3:]
        self.cell_lengths, self.cell_angles = enforce_crystal_system(
            self.cell_lengths,
            self.cell_angles,
            self.sg_ind
        )
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        if torch.any(self.z_prime > 1):
            self.sample_random_aunit_handedness()
        # enforce agreement with crystal system
        # other cell parameters are valid by explicit construction
        # todo add crystal system conditions to sampling workflow, rather than cleaning up here

        self.box_analysis()

    def latent_to_cell_params(self,
                              latents: torch.tensor,
                              skip_box_analysis: bool = False,
                              skip_enforce_crystal_system: bool = False
                              ):
        """
        Transform from latent space to physical crystal parameters
        :param override_mode:
        :param latents:
        :return:
        """

        min_vals = -torch.ones(latents.shape[-1], dtype=torch.float32, device=self.device)
        min_vals[:3] = -0.99  # DO NOT allow micro cells to be instantiated
        # also do not allow rotation length = 0 to be instantiated
        for ind in range(self.max_z_prime):
            min_vals[5 + 6 * (1 + ind)] = -0.99

        max_vals = torch.ones(latents.shape[-1], dtype=torch.float32, device=self.device)
        max_vals[
            0:2] = 1 - 1e-4  # don't let it explicitly touch 1 or it can make an effective orthorhombic cell, and really pisses off ASE
        self.set_cell_parameters(self.inv_latent_transform(latents.clamp(min=min_vals, max=max_vals)))

        if not skip_enforce_crystal_system:
            self.cell_lengths, self.cell_angles = enforce_crystal_system(
                self.cell_lengths,
                self.cell_angles,
                self.sg_ind
            )
        if not skip_box_analysis:
            self.box_analysis()

    def latent_params(self):
        """
        Transform cell parameters from physical space to latent.
        :return:
        """
        self.canonicalize_zp_aunits()  # latent space is always in the canonical ordering
        return self.latent_transform(cell_params=self.full_cell_parameters()).clip(min=-1, max=1)

    def latent_transform(self, cell_params):
        if not hasattr(self, 'asym_unit_lut'):
            self.build_asym_unit_tensor()

        sg_inds = self.sg_ind
        # 'radius' for Z'>1 stuctures is Z'*radius for downstream reasons. Also it's not intensive so we need a scaling
        radius = self.radius / (self.z_prime ** (2 / 3))
        auvs = self.asym_unit_lut[sg_inds]
        #auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds]).to(self.device)

        'convert to latent basis'
        # get aunit lengths
        box_params, aunit_params = torch.tensor_split(cell_params, [6], dim=1)
        cell_lengths = box_params[:, :3]
        cell_angles = box_params[:, 3:]
        aunit_lengths = cell_lengths * auvs
        normed_aunit_lengths = aunit_lengths / (2 * radius[:, None])

        # get aunit-wise fractional centroids
        cell_centroids, cell_orientations = aunit_params.split(3 * self.max_z_prime, dim=1)
        # this is the parallel way to do it over multiple Z'
        aunit_centroids = (
                cell_centroids
                .reshape(-1, self.max_z_prime, 3)  # [n, max_z', 3]
                / auvs.unsqueeze(1)  # [n, 1, 3] → broadcast over Z′
        ).reshape(-1, 3 * self.max_z_prime)  # back to [n, 3 * max_z']

        # get spherical rotvec
        sph_rotvec = cart2sph_rotvec(cell_orientations.reshape(len(cell_orientations) * self.max_z_prime, 3)).reshape(
            len(cell_orientations), self.max_z_prime * 3)

        'rescale everything to [-1,1]'
        # halfpi = torch.pi / 2
        au_range = torch.tensor([[0.075, 0.075, 0.1], [3, 3, 4]], dtype=torch.float32,
                                device=self.device)  # range to scale aunit lengths
        log_au_range = torch.log(au_range)
        ang_range = [0.2 * torch.pi, 0.8 * torch.pi]  # range of allowed cell angles

        # lengths & angles - rescale from [min, max] to [-1,1]
        lat_lengths = ((normed_aunit_lengths.log() - log_au_range[0]) / (log_au_range[1] - log_au_range[0]) - 0.5) * 2
        lat_angles = ((cell_angles - ang_range[0]) / (ang_range[1] - ang_range[0]) - 0.5) * 2

        # aunit centroid [0,1] to [-1,1]
        lat_centroids = (aunit_centroids - 0.5) * 2

        lat_orientations = sph_rotvec2lat(sph_rotvec, self.max_z_prime)

        latents = torch.cat([lat_lengths, lat_angles, lat_centroids, lat_orientations], dim=1)
        return latents

    def inv_latent_transform(self, latents):
        if not hasattr(self, 'asym_unit_lut'):
            self.build_asym_unit_tensor()

        sg_inds = self.sg_ind
        radius = self.radius / (self.z_prime ** (
                2 / 3))  # 'radius' for Z'>1 stuctures is Z'*radius for downstream reasons. Also it's not intensive so we need a scaling        auvs = torch.stack([self.asym_unit_dict[str(int(ind))] for ind in sg_inds]).to(self.device)

        auvs = self.asym_unit_lut[sg_inds].to(self.device)

        lat_lengths, lat_angles, lat_centroids, lat_orientations = torch.split(latents, [3, 3, 3 * self.max_z_prime,
                                                                                         3 * self.max_z_prime], dim=1)

        '''reverse bounding transform'''
        sph_rotvec = lat2sph_rotvec(lat_orientations, self.max_z_prime)

        aunit_orientations = sph2cart_rotvec(sph_rotvec.reshape(len(sph_rotvec) * self.max_z_prime, 3)).reshape(
            len(sph_rotvec), self.max_z_prime * 3)

        aunit_centroids = lat_centroids / 2 + 0.5

        cell_centroids = (
                aunit_centroids
                .reshape(-1, self.max_z_prime, 3)  # [n, max_z', 3]
                * auvs.unsqueeze(1)  # [n, 1, 3] → broadcast over Z′
        ).reshape(-1, 3 * self.max_z_prime)  # back to [n, 3 * max_z']

        au_range = torch.tensor([[0.075, 0.075, 0.1], [3, 3, 4]], dtype=torch.float32,
                                device=self.device)  # range to scale aunit lengths
        log_au_range = torch.log(au_range)
        ang_range = [0.2 * torch.pi, 0.8 * torch.pi]  # range of allowed cell angles

        log_span = log_au_range[1] - log_au_range[0]
        logL = (lat_lengths / 2 + 0.5) * log_span + log_au_range[0]
        normed_aunit_lengths = logL.exp()

        # angles
        ang_span = ang_range[1] - ang_range[0]
        cell_angles = (lat_angles / 2 + 0.5) * ang_span + ang_range[0]

        aunit_lengths = normed_aunit_lengths * (2 * radius[:, None])
        cell_lengths = aunit_lengths / auvs

        return torch.cat([cell_lengths, cell_angles, cell_centroids, aunit_orientations], dim=1)

    def latent_distmat(self):
        return crystal_parameter_distmat(self.latent_params()).fill_diagonal_(0)

    def sample_reduced_box_vectors(self, target_packing_coeff: Optional[float] = None):
        """
        Sample box vectors in niggil-reduced basis from a random prior.
        Optionally target a particular packing coefficient.
        :param target_packing_coeff:
        :return:
        """
        assert False, "Reduced box sampling is deprecated - must be re-implemented with our new reduction methods"

        # if not hasattr(self, 'latent_transform'):
        #     self.init_latent_transform()
        #
        # # use the latent transform for lengths and angles sampling, as its more stable and better tested
        # # we can still use the old samplers for aunit params
        # # latent space is defined on [-1,1]
        # rands = (torch.rand((self.num_graphs, 6 + 6 * self.max_z_prime), device=self.device) - 0.5) * 2
        # temp_params = self.latent_transform.inverse(rands,
        #                                             self.sg_ind,
        #                                             self.radius)
        # cell_lengths = temp_params[:, :3]
        # cell_angles = temp_params[:, 3:6]
        #
        # if target_packing_coeff is not None:
        #     vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
        #     cp1 = self.mol_volume * self.sym_mult / vol1
        #     correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
        #     cell_lengths *= correction_ratio[:, None]
        #
        # cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')
        #
        # self.cell_lengths, self.cell_angles = cell_lengths, cell_angles

    def sample_reasonable_random_parameters(self,
                                            tolerance: float = 1,
                                            target_packing_coeff: Optional = None,
                                            max_attempts: int = 100,
                                            seed: Optional[int] = None,
                                            ):
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

    def build_asym_unit_tensor(self):
        max_sg = 230

        asym_unit_by_sg = torch.zeros(
            max_sg + 1, 3,
            dtype=torch.float32,
            device=self.device,
        )

        for k, v in ASYM_UNITS.items():
            asym_unit_by_sg[int(k)] = torch.as_tensor(
                v,
                dtype=torch.float32,
                device=self.device,
            )

        self.asym_unit_lut = asym_unit_by_sg

    def build_sym_mult_tensor(self):
        sym_mult_by_sg = torch.zeros(231, dtype=torch.long, device=self.device)

        for sg, sym_ops in SYM_OPS.items():
            sym_mult_by_sg[int(sg)] = len(sym_ops)

        self.sym_mult_lut = sym_mult_by_sg

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
                normed_centroid * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device).repeat(
                    self.max_z_prime))

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

    def noise_latent_parameters(self, noise_level: float):
        latents = self.latent_params()
        noised_params = latents + torch.randn_like(latents) * noise_level
        self.latent_to_cell_params(noised_params)

    def log_noise_latent_parameters(self, log_min: float, log_max: float, eps = 1e-6):
        latents = self.latent_params()
        rand_dir = torch.randn_like(latents)
        rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
        u = torch.rand(len(latents), device=self.device)
        rand_magnitude = 10 ** (log_min + (log_max - log_min) * u)
        noised_latents = (latents + rand_dir * rand_magnitude[:, None]).clip(min=-1 + eps, max=1 - eps)
        self.latent_to_cell_params(noised_latents)
        self.clean_cell_parameters(mode='hard')

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

    def reparameterize_unit_cell(self,enforce_right_handedness: bool = False):
        if self.max_z_prime > 1:
            zp1_batch = self.split_to_zp1_batch()
            (aunit_centroid, aunit_orientation,
             handedness_list, is_well_defined, pos) = (
                parameterize_crystal_batch(zp1_batch,
                                           ASYM_UNITS,
                                           enforce_right_handedness=enforce_right_handedness,
                                           return_aunit=True))

            # have to pad out these to max_z_prime
            n = self.num_graphs
            mzp = self.max_z_prime
            device = aunit_centroid.device
            zp = self.z_prime.long().to(device)
            crystal_idx = torch.arange(n, device=device).repeat_interleave(zp)
            slot_idx = torch.arange(zp.sum(), device=device) - (torch.cumsum(zp, 0) - zp)[crystal_idx]

            aunit_centroid_p = aunit_centroid.new_zeros(n, mzp, 3)
            aunit_orientation_p = aunit_orientation.new_zeros(n, mzp, 3)
            handedness_p = handedness_list.new_zeros(n, mzp)
            iwd_p = torch.ones(n, mzp, dtype=torch.bool, device=device)

            aunit_centroid_p[crystal_idx, slot_idx] = aunit_centroid
            aunit_orientation_p[crystal_idx, slot_idx] = aunit_orientation
            handedness_p[crystal_idx, slot_idx] = handedness_list
            iwd_p[crystal_idx, slot_idx] = torch.as_tensor(is_well_defined, device=device)

            aunit_centroid = aunit_centroid_p.reshape(n, 3 * mzp)
            aunit_orientation = aunit_orientation_p.reshape(n, 3 * mzp)
            handedness_list = handedness_p
            is_well_defined = iwd_p.all(dim=1)
            combo_pos = []
            counter = 0
            for ind in range(self.num_graphs):
                pos_list = []
                for i in range(self.z_prime[ind].item()):
                    pos_list.extend(pos[counter + i])
                combo_pos.append(torch.stack(pos_list))
                counter += self.z_prime[ind]
            pos = combo_pos

        else:
            (aunit_centroid, aunit_orientation,
             handedness_list, is_well_defined, pos) = (
                parameterize_crystal_batch(self,
                                           ASYM_UNITS,
                                           enforce_right_handedness=enforce_right_handedness,
                                           return_aunit=True))

        return aunit_centroid, aunit_orientation, handedness_list.long(), is_well_defined, torch.cat(pos)

    def _transform_aunit_params(self, centroid_frac, orientation, handedness, R_frac, t_frac, wrap: bool = True):
        """
        Apply one fractional affine operation (R_frac, t_frac) -- a SYM_OPS entry
        or a normalizer coset representative, the math doesn't care which -- to
        this batch's (aunit_centroid, aunit_orientation, aunit_handedness),
        returning the new params describing the same operation applied to every
        atom. Used by get_fd_params to walk the Euclidean-normalizer cosets.
        """
        centroid_frac = centroid_frac[:, :3]
        orientation = orientation[:, :3]
        n = centroid_frac.shape[0]
        device, dtype = centroid_frac.device, centroid_frac.dtype

        if R_frac.ndim == 2:
            R_frac = R_frac[None].expand(n, -1, -1)
        if t_frac.ndim == 1:
            t_frac = t_frac[None].expand(n, -1)
        R_frac = R_frac.to(device=device, dtype=dtype)
        t_frac = t_frac.to(device=device, dtype=dtype)

        # centroid: fractional space is native to SYM_OPS/normalizer generators alike
        new_centroid = torch.einsum('nij,nj->ni', R_frac, centroid_frac) + t_frac
        if wrap:
            new_centroid = new_centroid - torch.floor(new_centroid)

        # orientation/handedness: need the CARTESIAN rotation part. This conjugation
        # is a no-op determinant-wise (det(R_cart) == det(R_frac) always, since
        # T_cf = inv(T_fc)), but it DOES matter for R_orient_new itself whenever
        # R_frac isn't central in O(3) -- i.e. for anything except pure inversion.
        R_cart = self.T_fc @ R_frac @ self.T_cf
        det = torch.linalg.det(R_cart)  # (N,)

        h_old = handedness.reshape(-1).to(dtype)
        h_new = det * h_old

        R_orient_old = rotvec2rotmat(orientation)
        diag_det = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        diag_det[:, 0, 0] = det
        diag_det[:, 1, 1] = 1.
        diag_det[:, 2, 2] = 1.
        R_orient_new = R_cart @ R_orient_old @ diag_det
        orientation_new = rotmat2rotvec(R_orient_new)

        return new_centroid, orientation_new, h_new

    def get_fd_params(self, is_chiral: Optional[torch.Tensor] = None, normalizer_table: dict = NORMALIZER_OPS):
        """
        Reduce this batch's aunit_{centroid,orientation,handedness} to a canonical
        representative under the Euclidean normalizer N_E(G). Assumes one sg_ind
        for the whole batch, matching the rest of the pipeline's calling convention.

        Algorithm: candidate 0 is the batch's current aunit params (assumed
        already folded into G's box -- true if it came from
        reparameterize_unit_cell or init_batch's sampling path). For each
        additional normalizer coset rep g_i: build the candidate via
        _transform_aunit_params, apply the chirality gate (drop g_i if
        det(W_i) < 0 and the molecule is chiral -- an improper normalizer element
        maps a chiral molecule to its enantiomer, a different crystal, not a
        duplicate description), fold the candidate's centroid back into G's own
        box by actually rebuilding it and calling the existing
        pose_aunit -> build_unit_cell -> reparameterize_unit_cell pipeline (reuses
        identify_canonical_asymmetric_unit exactly as-is, rather than re-deriving
        G's fold-back symbolically), then canonicalize across all box-landing
        candidates by the same "distance from origin" tie-break
        identify_canonical_asymmetric_unit / canonicalize_aunit_order already use.

        is_chiral : (N,) bool, True where the molecule is not superimposable on
            its mirror image. Not computed here -- pull from wherever your
            featurization already flags this (RDKit CIP/stereocenter perception
            on the SMILES, most likely), don't want to silently default this to
            one value or the other.

        returns: C[best, idx], O[best, idx] -- the canonicalized aunit centroid
            and orientation, (N, 3) each.
        """
        sg_ind = int(self.sg_ind[0])
        assert torch.all(self.sg_ind == sg_ind), \
            "get_fd_params assumes one space group per batch, as elsewhere in the pipeline"

        generators = normalizer_table.get(str(sg_ind), [])
        if len(generators) == 0:
            raise NotImplementedError(
                f"No normalizer coset representatives on file for space group {sg_ind}. "
                f"Pull them from https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-norm"
                f"?gnum={sg_ind}&norgens=en (blocks automated fetches -- needs a human) "
                f"or ITA Vol. A1 Table 15.2, then add an entry to NORMALIZER_OPS."
            )

        n = self.num_graphs
        device = self.device
        dtype = self.aunit_centroid.dtype

        cand_centroids = [self.aunit_centroid[:, :3].clone()]
        cand_orients = [self.aunit_orientation[:, :3].clone()]
        cand_valid = [torch.ones(n, dtype=torch.bool, device=device)]

        for (W, w) in generators:
            W_t = torch.as_tensor(W, dtype=dtype, device=device)
            w_t = torch.as_tensor(w, dtype=dtype, device=device)
            det_W = torch.linalg.det(W_t)

            c, o, h = self._transform_aunit_params(
                self.aunit_centroid, self.aunit_orientation, self.aunit_handedness, W_t, w_t,
            )

            # fold this candidate back into G's box, reusing the existing pipeline
            # end to end rather than re-deriving G's own fold-back symbolically
            trial = self.clone()
            trial.aunit_centroid = c
            trial.aunit_orientation = o
            trial.aunit_handedness = h[:, None]
            trial.pose_aunit()
            trial.build_unit_cell()
            c_folded, o_folded, h_folded, well_defined, _ = trial.reparameterize_unit_cell()

            if is_chiral is None:
                chirality_ok = (~True) | (det_W > 0)
            else:
                chirality_ok = (~is_chiral) | (det_W > 0)
            valid = chirality_ok & torch.as_tensor(well_defined, device=device, dtype=torch.bool)

            cand_centroids.append(c_folded[:, :3])
            cand_orients.append(o_folded[:, :3])
            cand_valid.append(valid)

        C = torch.stack(cand_centroids, dim=0)  # (K, n, 3)
        O = torch.stack(cand_orients, dim=0)    # (K, n, 3)
        V = torch.stack(cand_valid, dim=0)      # (K, n)

        # Tie-break: dimension-by-dimension (x, then y, then z, then orientation),
        # matching identify_canonical_asymmetric_unit's own convention -- NOT a
        # joint norm. This matters, not just style: found empirically (7/3) that a
        # joint ||centroid||+||orientation|| norm under-reduces whenever a
        # normalizer generator's G-refold couples two dimensions together (P21/c's
        # 21-screw-mediated fold ties x and z into a single joint flip, (x,z) ->
        # (-x,-z+1/2), rather than touching them independently). A joint norm is
        # symmetric under swapping such coupled dimensions, so it can't
        # consistently resolve which one absorbs the extra reduction -- it picks
        # whichever raw value is smaller, which varies structure to structure and
        # leaves BOTH dimensions spanning their full pre-reduction range in
        # aggregate across a dataset, rather than confining either one. Sequential
        # dimension-ordered comparison fixes this: the first dimension checked
        # gets to use its full available freedom (quartering, for the coupled
        # pair), and whatever's left over resolves the next. Verified numerically
        # against the actual sg-14 orbit structure: joint norm left x and z both
        # spanning the full [0, 0.5) each; x-first lexicographic confined x to
        # [0, 0.25] and z to [0, 0.5) -- exactly the missing extra factor of 2.
        idx = torch.arange(n, device=device)
        keys = torch.cat([C, O], dim=-1)  # (K, n, 6): centroid xyz, then orientation xyz
        keys = torch.where(V[..., None].bool(), keys, torch.full_like(keys, float('inf')))
        still_tied = torch.ones(len(cand_centroids), n, dtype=torch.bool, device=device)
        atol = 1e-6
        for d in range(keys.shape[-1]):
            col = torch.where(still_tied, keys[..., d], torch.full_like(keys[..., d], float('inf')))
            dim_min = col.min(dim=0, keepdim=True).values
            still_tied = still_tied & (col <= dim_min + atol)
        # genuine ties surviving every dimension are measure-zero for generic
        # (non-special-position) structures; take the first survivor for determinism
        best = still_tied.float().argmax(dim=0)  # (n,)

        return C[best, idx], O[best, idx]

    def fold_to_fd(self, is_chiral: Optional[torch.Tensor] = None):
        """
        In-place counterpart to get_fd_params: overwrites this batch's
        aunit_centroid and aunit_orientation with their canonical N_E(G)
        fundamental-domain representatives.
        """
        self.aunit_centroid, self.aunit_orientation = self.get_fd_params(is_chiral)

    def split_cell_params(self):
        cell_lengths, cell_angles, aunit_positions, aunit_orientations = torch.split(self.full_cell_parameters(),
                                                                                     [3, 3, 3 * self.max_z_prime,
                                                                                      3 * self.max_z_prime], dim=1)
        return cell_lengths, cell_angles, aunit_positions, aunit_orientations

    def clean_cell_parameters(self, mode: str = 'hard',
                              canonicalize_orientations: bool = True,
                              angle_pad: float = 0.6,
                              length_pad: float = 3.0,
                              constrain_z: bool = False,
                              ):
        """  # todo align with latent transforms - otherwise we can get severe roundtrip errors. Lengths and interior angles
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

        # if enforce_niggli:
        #     self.enforce_niggli_box(mode)

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
                new_orientation.append(self.aunit_orientation[:, 3 * zp:3 * zp + 2])
                new_orientation.append(fixed_z[:, None])
            self.aunit_orientation = torch.cat(new_orientation, dim=1)

        new_orientation = []
        for zp in range(self.max_z_prime):
            norm = torch.linalg.norm(self.aunit_orientation[:, zp * 3:3 + zp * 3], dim=1)
            new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode=mode)  # MUST be nonzero
            new_orientation.append(self.aunit_orientation[:, zp * 3:3 + zp * 3] / norm[:, None].clamp(min=1e-6) * new_norm[:, None])
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
                new_orientation.append(self.aunit_orientation[:, zp * 3:3 + zp * 3])
            self.aunit_orientation = torch.cat(new_orientation, dim=1)

        # update cell vectors
        self.box_analysis()

    def enforce_niggli_box(self, mode, apply_plane_shift: bool = False):

        assert False, "This needs to be reimplemented with our new reduction workflow"
        # # enforce the scaling factor b/c and a/b in the range [0, 1]
        # a, b, c = self.cell_lengths.split(1, 1)
        # b_scale = enforce_1d_bound((b / c), 0.5, 0.5, mode=mode)
        # a_scale = enforce_1d_bound((a / b), 0.5, 0.5, mode=mode)
        # self.cell_lengths = torch.cat([
        #     a_scale * b_scale * c,
        #     b_scale * c,
        #     c],
        #     dim=1)
        #
        # # enforce the scaling factor cos(alpha) / (b/2c) in the range [0, 1]
        # al_cos, be_cos, ga_cos = self.cell_angles.cos().split(1, 1)
        # a, b, c = self.cell_lengths.split(1, 1)
        # al_cos_max = (b / 2 / c)
        # be_cos_max = (a / 2 / c)
        # ga_cos_max = (a / 2 / b)
        #
        # # now improved - including obtuse cells
        # al_cos_scale = enforce_1d_bound(al_cos / al_cos_max, 1.0, 0.0, mode=mode)
        # be_cos_scale = enforce_1d_bound(be_cos / be_cos_max, 1.0, 0.0, mode=mode)
        # ga_cos_scale = enforce_1d_bound(ga_cos / ga_cos_max, 1.0, 0.0, mode=mode)
        #
        # # limit it here due to instability in arccos
        # al = torch.arccos(al_cos_max * al_cos_scale.clip(min=-0.99, max=0.99))
        # be = torch.arccos(be_cos_max * be_cos_scale.clip(min=-0.99, max=0.99))
        # ga = torch.arccos(ga_cos_max * ga_cos_scale.clip(min=-0.99, max=0.99))
        #
        # self.cell_angles = torch.cat([al, be, ga], dim=1)
        # # here, enforce positivity of niggli overlap
        # if apply_plane_shift:
        #     self.cell_angles = enforce_niggli_plane(self.cell_lengths,
        #                                             self.cell_angles,
        #                                             mode='shift')

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

    def _build_feature_labels(self, space):
        lattice_features = ['a', 'b', 'c',
                            r'$\alpha$', r'$\beta$', r'$\gamma$']
        if self.max_z_prime == 1:
            lattice_features.extend([
                f'u', f'v', f'w',
            ])
            if space == 'latent':
                lattice_features.extend([
                    f'θ', f'φ', f'r'
                ])
            else:
                lattice_features.extend([
                    f'x', f'y', f'z'
                ])

        else:
            for zp in range(self.max_z_prime):
                lattice_features.extend([
                    f'aunit{zp} u', f'aunit{zp} v', f'aunit{zp} w',
                ])
            for zp in range(self.max_z_prime):
                if space == 'latent':
                    lattice_features.extend([
                        f'θ{zp}', f'φ{zp}', f'r{zp}'
                    ])
                else:
                    lattice_features.extend([
                        f'x{zp}', f'y{zp}', f'z{zp}'
                    ])
        return lattice_features

    def _set_cell_ranges(self, space, samples):
        if space == 'latent':
            custom_ranges = {}
            for ind in range(6 + self.max_z_prime * 6):
                custom_ranges.update({ind: [-1.1, 1.1]})

        elif space == 'standard':
            custom_ranges = {
                0: [-1.1, 1.1],  # for cell_a
                1: [-1.1, 1.1],  # for cell_b
                2: [-1.1, 1.1],  # for cell_c
                3: [-1.1, 1.1],  # for cell_alpha
                4: [-1.1, 1.1],  # for cell_beta
                5: [-1.1, 1.1],  # for cell_gamma
            }
            for ind in range(self.max_z_prime * 6):
                custom_ranges.update({6 + ind: [-1.1, 1.1]})

        else:
            custom_ranges = {ind: [np.amin(samples[:, ind]), np.amax(samples[:, ind])]
                             for ind in range(3)}
            custom_ranges.update(
                {3: [np.pi / 10, 0.9 * np.pi],
                 4: [np.pi / 10, 0.9 * np.pi],
                 5: [np.pi / 10, 0.9 * np.pi]
                 }
            )
            for ind in range(self.max_z_prime):
                custom_ranges.update({
                    6 + ind * 3 + 0: [0, 1.1],
                    6 + ind * 3 + 1: [0, 1.1],
                    6 + ind * 3 + 2: [0, 1.1],
                })
            for ind in range(self.max_z_prime):
                custom_ranges.update({
                    6 + self.max_z_prime * 3 + ind * 3 + 0: [-2 * np.pi, 2 * np.pi],
                    6 + self.max_z_prime * 3 + ind * 3 + 1: [-2 * np.pi, 2 * np.pi],
                    6 + self.max_z_prime * 3 + ind * 3 + 2: [0, 2 * np.pi],
                })

        return custom_ranges

    def _collect_sample_dists(self, samples, ref_dist, quantiles, split_by_sg, split_by_zp, aux_dists,
                              override_energy=None):
        num_dists = 1
        dists = []
        dist_names = []
        if ref_dist is not None:  # this should be first
            assert ref_dist.shape[1] == samples.shape[1]
            num_dists += 1
            dist_names += ['Reference']
            if torch.is_tensor(ref_dist):
                dists.append(ref_dist.cpu().detach().numpy())
            else:
                dists.append(ref_dist)

        dist_names.append('Samples')
        dists.append(samples)

        if quantiles is not None:
            if override_energy is not None:
                energies = override_energy
            else:
                energies = self.lj.cpu().detach().numpy()
            for q in quantiles:
                num_dists += 1
                dist_names += [f'{q} Quantile']
                dists.append(samples[energies < np.quantile(energies, q)])

        if split_by_sg:
            for sg in torch.unique(self.sg_ind):
                good_inds = self.sg_ind == sg
                dist_names += [f'SG={int(sg)}']
                dists.append(samples[good_inds])
                num_dists += 1

        if split_by_zp:
            for zp in torch.unique(self.z_prime):
                good_inds = self.z_prime == zp
                dist_names += [f"Z'={int(zp)}"]
                dists.append(samples[good_inds])
                num_dists += 1

        if aux_dists is not None:
            for ind, aux_dist in enumerate(aux_dists):
                dist_names += [f'Aux {ind}']
                dists.append(aux_dist)
                num_dists += 1

        return num_dists, dist_names, dists

    def plot_batch_density_funnel(self,
                                  renderer: Optional[str] = None,
                                  show: bool = True,
                                  return_fig: bool = False,
                                  split_by_sg: bool = False,
                                  split_by_zp: bool = False,
                                  override_energy: Union[str, torch.Tensor] = None,
                                  color_flag: torch.Tensor = None,
                                  show_colorbar: bool = False,
                                  max_y_quantile: Optional[float] = None,
                                  overwrite_yaxis_title: Optional[str] = None,
                                  ):

        if override_energy is None:
            energy = (log_rescale_positive(self.lj)).cpu().detach()
        else:
            if isinstance(override_energy, torch.Tensor):
                energy = (log_rescale_positive(override_energy)).cpu().detach()
            elif isinstance(override_energy, str):
                energy = log_rescale_positive(self[override_energy]).cpu().detach()
            else:
                assert False, "override_energyu must be tensor or string"

        xy = np.vstack([self.packing_coeff.cpu().detach(), energy])
        try:
            z = get_point_density(xy, bins=25)
        except:
            z = np.ones(xy.shape[1])

        scatter_dict = {'energy': energy,
                        'packing_coefficient': self.packing_coeff.cpu().detach(),
                        }
        if color_flag is not None:
            color_tag = 'Custom Flag'
            scatter_dict.update({'Custom Flag': color_flag})
            cscale = px.colors.qualitative.Dark24
        else:
            cscale = px.colors.cyclical.IceFire
            if split_by_sg:
                if split_by_zp:
                    print("Cannot split by both z prime and space group!")
                color_tag = 'Space Group'
                scatter_dict.update({'Space Group': [str(int(sg)) for sg in self.sg_ind]})
            elif split_by_zp:
                color_tag = "Z'"
                scatter_dict.update({"Z'": [str(int(zp)) for zp in self.z_prime]})
            else:
                color_tag = 'Point Density'
                scatter_dict.update({'Point Density': z})

        opacity = max(0.25, 1 - self.num_graphs / 5e4)
        df = pd.DataFrame.from_dict(scatter_dict)

        fig = px.scatter(df,
                         x='packing_coefficient', y='energy',
                         color=color_tag,
                         color_continuous_scale=cscale,
                         marginal_x='violin', marginal_y='violin',
                         color_discrete_sequence=px.colors.qualitative.Set3 if color_tag == 'Space Group' else None,
                         opacity=opacity
                         )

        fig.update_layout(yaxis_title='Energy', xaxis_title='Packing Coeff')
        if max_y_quantile is not None:
            max_y = np.quantile(df['energy'], max_y_quantile)
        else:
            max_y = min(10, np.amax(df['energy']) + np.ptp(df['energy']) * 0.05)

        fig.update_layout(yaxis_range=[np.amin(df['energy']) - np.ptp(df['energy']) * 0.05, max_y],
                          xaxis_range=[max(0, np.amin(df['packing_coefficient']) * 0.95),
                                       min(1, np.amax(df['packing_coefficient']) * 1.05)],
                          )

        if overwrite_yaxis_title is not None:
            yaxis_title = overwrite_yaxis_title
        else:
            yaxis_title = r'Energy per Atom /Arb Units'
        fig.update_layout(
            xaxis_title=r"Packing Coefficient",
            yaxis_title=yaxis_title,
        )
        fig.update_traces(
            marker=dict(
                size=5,
                line=dict(width=0.3, color='rgba(0,0,0,0.3)'),
                opacity=opacity,
            )
        )
        if color_tag == 'Point Density' and show_colorbar:
            fig.update_layout(coloraxis_colorbar=dict(
                title="Point Density (Normed)",
                tickfont=dict(size=10),
                title_font=dict(size=11),
            ))
        fig.update_traces(selector=dict(type='violin'), spanmode='hard')
        fig.update_traces(selector=dict(type='violin'), line=dict(width=0.6, color='black'))
        fig.update_layout(
            font=dict(family="Helvetica", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=20, t=40, b=50),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.15)',
            gridwidth=0.8,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            row=1, col=1  # target the main scatter subplot
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.15)',
            gridwidth=0.8,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            row=1, col=1  # target the main scatter subplot
        )
        if not show_colorbar:
            fig.update_layout(coloraxis_showscale=False)
        if show:
            fig.show(renderer=renderer)

        if return_fig:
            return fig
        return None

    def plot_batch_cell_params(self, space='real',
                               renderer: Optional[str] = None,
                               quantiles: Optional[Iterable[float]] = None,
                               ref_dist: Optional[torch.Tensor] = None,
                               aux_dists: Optional[list] = None,
                               show: bool = True,
                               return_fig: bool = False,
                               split_by_sg: bool = False,
                               split_by_zp: bool = False,
                               n_kde: int = 200,
                               bw_factor: float = 0.05,
                               override_energy: Optional[torch.Tensor] = None, ):
        if not self.is_batch:
            print("Cell statistics only works for a batch of crystal data objects")
            return None

        lattice_features = self._build_feature_labels(space=space)
        samples = self._get_samples(space)
        num_dists, dist_names, dists = self._collect_sample_dists(samples, ref_dist, quantiles, split_by_sg,
                                                                  split_by_zp, aux_dists, override_energy)
        # delete or NaN unused higher Z' elements
        # 1d Histograms
        fig = make_subplots(rows=int(2 + 2 * self.max_z_prime),
                            cols=3,
                            subplot_titles=lattice_features)
        colors = self._get_color_set(num_dists)
        ranges = self._set_cell_ranges(space, samples)
        for i in range(len(lattice_features)):
            for j in range(num_dists):
                self._add_violin(
                    fig, dists[j][:, i], dist_names[j], colors[j], i, ranges[i],
                    n_kde, bw_factor
                )

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          violinmode='overlay',
                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=1.05,
                              xanchor="center",
                              x=0.5,
                              bgcolor='rgba(0,0,0,0)',

                          ),
                          margin=dict(l=40, r=20, t=50, b=50),
                          font=dict(family="Helvetica", size=12, color='black'),
                          )

        for i in range(6 + self.max_z_prime * 6):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.update_xaxes(range=ranges[i], row=row, col=col)
        fig.update_xaxes(
            showgrid=False, zeroline=False, ticks='outside',
            tickwidth=1, mirror=True
        )
        # fig.update_yaxes(
        #     showgrid=False, zeroline=False, ticks='outside',
        #     tickwidth=1, mirror=True
        # )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, ticks='',
            mirror=True
        )
        if len(dists) > 1:
            fig.update_traces(opacity=0.5)
        if show:
            fig.show(renderer=renderer)
        if return_fig:
            return fig
        else:
            return None

    def plot_batch_staircase(self, space='real',
                             renderer=None,
                             show=True,
                             return_fig=False,
                             mode='contour',
                             cmap='icefire',
                             nbins=25,
                             colorbar=False,
                             ref_dist=None,
                             ):

        labels = self._build_feature_labels(space=space)
        samples = self._get_samples(space)
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
        N, D = samples.shape

        # Create D×D subplots (upper triangle empty)
        fig = make_subplots(
            rows=D, cols=D,
            horizontal_spacing=0.01, vertical_spacing=0.01,
            shared_xaxes=True, shared_yaxes=True,
        )

        # Loop over lower triangle
        for i in range(D):
            for j in range(D):
                if j >= i:
                    continue  # keep lower triangle only

                x = samples[:, j]
                y = samples[:, i]

                if mode == 'contour':
                    trace = go.Histogram2dContour(
                        x=x, y=y,
                        ncontours=32,
                        colorscale=cmap,
                        showscale=colorbar and (i == D - 1 and j == 0),
                        contours=dict(coloring='fill', showlines=False, start=0, end=None, size=None),
                        line=dict(smoothing=0.85, width=0),
                        nbinsx=nbins,
                        nbinsy=nbins,
                    )

                elif mode == 'heatmap':
                    trace = go.Histogram2d(
                        x=x, y=y,
                        nbinsx=nbins, nbinsy=nbins,
                        colorscale=cmap,
                        showscale=colorbar and (i == D - 1 and j == 0),
                    )
                else:
                    raise ValueError("mode must be 'contour' or 'heatmap'")
                fig.add_trace(trace, row=i + 1, col=j + 1)
                if ref_dist is not None:
                    trace = go.Scatter(x=ref_dist[:, j], y=ref_dist[:, i], mode='markers',
                                       marker_color='white', marker_line_width=1.5, opacity=0.8,
                                       marker_line_color='#333333', marker_size=8, showlegend=False)
                    fig.add_trace(trace, row=i + 1, col=j + 1)

        for i in range(D):
            fig.update_xaxes(title_text=labels[i], row=D, col=i + 1)
            fig.update_yaxes(title_text=labels[i], row=i + 1, col=1)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            # height=1000,
            # width=1000,
            showlegend=False,
        )
        fig.update_layout(
            font=dict(family="Helvetica", size=12),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=30, r=30, t=20, b=30),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)
        fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)
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

    def _add_violin(self, fig, samples, name, color, column_index, ranges, n_kde, bw_factor):
        row = column_index // 3 + 1
        col = column_index % 3 + 1
        x_samp, y_samp = lightweight_one_sided_violin(samples,
                                                      n_kde,
                                                      bandwidth_factor=bw_factor,
                                                      data_min=ranges[0],
                                                      data_max=ranges[1])
        fig.add_scatter(
            x=x_samp,
            y=y_samp,
            mode='lines',
            fill='toself',  # 'tonexty' if i == 0 else 'tonexty',  # Fill to next y (which is 0)
            fillcolor=color,
            line=dict(color=color, width=1.2),
            name=name,
            legendgroup=name,
            showlegend=True if column_index == 0 else False,
            row=row, col=col)

    def _get_samples(self, space):
        if space == 'real':
            samples = self.full_cell_parameters().detach().cpu().numpy()
        elif space == 'latent':
            samples = self.latent_params().detach().cpu().numpy()
        elif space == 'standard':  # todo implement full std method
            samples = self.zp1_std_cell_parameters().detach().cpu().numpy()
        return samples

    def set_cell_parameters(self,
                            cell_parameters,
                            skip_box_analysis: bool = False):
        assert cell_parameters.shape[1] == 6 + self.max_z_prime * 6, (
            f"For crystal batch with max z_prime={self.max_z_prime},"
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
            sg_ind_tensor = torch.ones_like(self.sg_ind, device=self.device) * sg_ind
        elif isinstance(sg_ind, list):
            sg_ind_tensor = torch.tensor(sg_ind, dtype=torch.long, device=self.device)
        elif torch.is_tensor(sg_ind):
            sg_ind_tensor = sg_ind * 1
        else:
            raise TypeError("sg_ind must be a tensor or an integer")

        unique_sgs = torch.unique(sg_ind_tensor)
        num_unique_sgs = len(unique_sgs)

        if self.is_batch:
            self.add_graph_attr(sg_ind_tensor, 'sg_ind')
            # reset symmetries will always be standard
            self.add_graph_attr(torch.zeros(self.num_graphs, dtype=torch.bool, device=self.device),
                                'nonstandard_symmetry')
        else:
            setattr(self, 'sg_ind', sg_ind_tensor)
            setattr(self, 'nonstandard_symmetry', False)

        if self.is_batch:
            if num_unique_sgs == 1:
                self.symmetry_operators = [np.stack(SYM_OPS[int(unique_sgs.item())])] * self.num_graphs
            else:
                self.symmetry_operators = [np.stack(SYM_OPS[int(ind)]) for ind in self.sg_ind]

            if not hasattr(self, 'sym_mult_lut'):
                self.build_sym_mult_tensor()

            sym_mult = self.sym_mult_lut[sg_ind_tensor]
            self.add_graph_attr(sym_mult, 'sym_mult')

        else:
            self.symmetry_operators = np.stack(SYM_OPS[int(self.sg_ind)])
            setattr(self, 'sym_mult', torch.tensor(
                len(self.symmetry_operators), dtype=torch.long, device=self.device)[None])

    def canonicalize_zp_aunits(self):
        if self.is_batch:
            self.aunit_centroid, self.aunit_orientation, self.aunit_handedness = canonicalize_aunit_order(self)
        else:
            self.aunit_centroid, self.aunit_orientation, self.aunit_handedness = canonicalize_aunit_order(
                collate_data_list([self], max_z_prime=self.max_z_prime))

    def canonicalize_orientation(self):
        flat_rotvecs = self.aunit_orientation.reshape(self.num_graphs * self.max_z_prime, 3)
        fixed_flat_rotvecs = canonicalize_rotvec(flat_rotvecs)
        fixed_rotvecs = fixed_flat_rotvecs.reshape(self.num_graphs, self.max_z_prime * 3)
        self.aunit_orientation = fixed_rotvecs

    def compute_standard_cell(self, confirm_transform: bool = False,
                              enforce_right_handedness: bool = False):
        assert self.is_batch, "Cell standardization currently only implemented for batch objects"
        import spglib

        """set up unit cell"""
        self.mol2ucell()

        target_params = self.full_cell_parameters()
        std_params = []
        transforms = []
        new_positions = []

        """get standard cell from spglib"""
        for ind, p in enumerate(target_params):
            "box params"
            cellpar = p[:6].clone().numpy()
            cellpar[3:] *= (180 / np.pi)
            lattice = cellpar_to_cell(cellpar)

            "fractional positions"
            positions = fractional_transform(self.unit_cell_pos[self.unit_cell_batch == ind],
                                             self.T_cf[ind]).cpu().detach().numpy()

            numbers = self.z[self.batch == ind].repeat(self.sym_mult[ind]).cpu().detach().numpy()

            "spglib standardize"
            cell = (lattice, positions, numbers)
            lattice_std, positions_std, numbers_std = spglib.standardize_cell(
                cell, to_primitive=False, no_idealize=True)

            "get new lattice"
            cellpar_std = cell_to_cellpar(lattice_std)

            "manually apply transform to atom positions (avoid breaking mols)"
            T = lattice @ np.linalg.inv(lattice_std)
            positions_new = positions @ T
            transforms.append(T)
            new_positions.append(positions_new)
            std_params.append(cellpar_std)

        std_params = np.stack(std_params)

        """instantiate standardized batch object"""
        std_batch = self.clone()
        "new box"
        std_batch.cell_lengths = torch.tensor(std_params[:, :3], dtype=torch.float32, device=self.device)
        std_batch.cell_angles = torch.tensor(std_params[:, 3:] / 180 * np.pi, dtype=torch.float32, device=self.device)
        std_batch.box_analysis()
        "transformed cartesian positions (not properly wrapped)"
        #ucell_pos_frac = torch.as_tensor(np.stack(new_positions), dtype=torch.float32, device=self.device)
        ucell_pos_frac = torch.as_tensor(np.concatenate(new_positions), dtype=torch.float32, device=self.device)
        ucell_batch = torch.arange(self.num_graphs).repeat_interleave((self.num_atoms * self.sym_mult).long())
        #ucell_pos_cart = fractional_transform(ucell_pos_frac.flatten(0, 1), std_batch.T_fc[ucell_batch])
        ucell_pos_cart = fractional_transform(ucell_pos_frac, std_batch.T_fc[ucell_batch])
        std_batch.unit_cell_pos = ucell_pos_cart
        std_batch.unit_cell_batch = ucell_batch

        "parameterize unit cell (wrapping handled here) and update crystal params"
        (aunit_centroid, aunit_orientation,
         handedness_list, is_well_defined, pos) = std_batch.reparameterize_unit_cell(
            enforce_right_handedness=enforce_right_handedness
        )

        std_batch.symmetry_operators = [SYM_OPS[int(ind)] for ind in
                                        std_batch.sg_ind]  # critical - give standard sym ops!
        std_batch.nonstandard_symmetry.fill_(False)

        std_batch.aunit_centroid = aunit_centroid
        std_batch.aunit_orientation = aunit_orientation
        if handedness_list.ndim == 1:
            std_batch.aunit_handedness = handedness_list[:, None]
        else:
            std_batch.aunit_handedness = handedness_list

        std_batch.is_well_defined = torch.tensor(is_well_defined, dtype=torch.bool, device=self.device)

        std_batch.pos = pos
        std_batch.pose_aunit()

        if confirm_transform:
            # estimate magnitude of the transform
            # transform_dist = torch.cdist(torch.tensor(transforms.reshape(-1, 9), dtype=torch.float32), torch.eye(3).flatten()[None, :]).flatten()

            rdf1 = self.analyze(['rdf'])['rdf'][0]
            rdf2 = std_batch.analyze(['rdf'])['rdf'][0]
            diffs = compute_rdf_distance(rdf1, rdf2, torch.linspace(0, 10, rdf1.shape[-1],
                                                                    dtype=torch.float32, device=self.device))

            penalties = std_batch.compute_cell_reduction_penalty()

            succeeded = (diffs < 1e-3) & (penalties < 1e-3)
            if not succeeded.all():
                print((~succeeded).sum().item(), " samples failed standardization")

        if confirm_transform:
            return std_batch, succeeded
        else:
            return std_batch

    def write_cif(self, inds, path, mode):
        ase_write_cif(self,inds, path, mode)