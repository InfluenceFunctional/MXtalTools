from typing import Optional

import torch
from torch_scatter import scatter

from mxtaltools.analysis.crystal_rdf import crystal_rdf
from mxtaltools.analysis.vdw_analysis import get_intermolecular_dists_dict, lj_analysis, electrostatic_analysis, \
    buckingham_energy, silu_energy, vdW_analysis, qlj_analysis, elj_analysis
from mxtaltools.common.sym_utils import niggli_reduction_penalty, cell_reduction_penalty
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import compute_crystal_mace_on_mxt_batch
from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
from mxtaltools.models.functions.radial_graph import build_radial_graph


def single_compack_run(ind, test_path, ref_path):
    from ccdc.io import CrystalReader
    from ccdc.crystal import PackingSimilarity
    ref_crystal = CrystalReader(ref_path)[0]
    sample_crystal = CrystalReader(test_path)[0]
    similarity_engine = PackingSimilarity()
    # similarity_engine.settings.distance_tolerance = 0.4
    # similarity_engine.settings.angle_tolerance = 20
    # similarity_engine.settings.allow_molecular_differences = True
    similarity_engine.settings.packing_shell_size = 20
    try:
        result = similarity_engine.compare(ref_crystal, sample_crystal)
        print(f"Crystal {ind} RMSD = {result.rmsd:.3f} Å, {result.nmatched_molecules} mols matched")
        return result.rmsd, result.nmatched_molecules
    except AttributeError:
        print("Analysis failed")
        return 0, 0


# noinspection PyAttributeOutsideInit
COMPUTES_REQUIRE_CLUSTER = {'lj': True,
                            'qlj': True,
                            'elj': True,
                            'es': True,
                            'bh': True,
                            'silu': True,
                            'vdw': True,
                            'vdw_max': True,
                            'ellipsoid': True,
                            'niggli_overlap': False,
                            'reduction_en': False,
                            'rdf': True,
                            'ellipsoid_emb': True,
                            'uma_pot': True,
                            'uma_gas_pot': True,
                            'uma': True,
                            'mace_pot': True,
                            'mace_gas_pot': True,
                            'mace': True,
                            'latent_harmonic': False,
                            'latent_multiharmonic': False,
                            }


class MolCrystalAnalysis:

    def construct_intra_radial_graph(self, cutoff: float = 6):
        if self.aux_ind is not None:
            raise RuntimeError("Cannot build molecular graph when we have already built the cluster")
        self.edges_dict = build_radial_graph(self.pos,
                                             self.batch,
                                             self.ptr,
                                             cutoff,
                                             max_num_neighbors=10000
                                             )
        self.edge_index = self.edges_dict['edge_index']

    def construct_radial_graph(self,
                               cutoff: float = 6,
                               max_num_neighbors=1000):
        if self.is_batch:
            self.edges_dict = get_intermolecular_dists_dict(
                self,
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

        else:
            self.edges_dict = get_intermolecular_dists_dict(
                collate_data_list([self]),
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

    def _pre_compute_checks(self):
        if not hasattr(self, 'vdw_radii_tensor'):
            self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing energies!")

    def compute(self, requests: dict, **kwargs):
        """
        Analyze a crystal cluster according to requests in input dict
        :return:
        """
        if not hasattr(self, 'computes'):
            self._init_computes()
        return {key: self.computes[key.lower()](**kwargs) for key in requests}

    def _init_computes(self, override: bool = False):
        if not hasattr(self, 'computes') or override:
            self.computes = {'lj': self.compute_LJ_energy,
                             'qlj': self.compute_qLJ_energy,
                             'elj': self.compute_eLJ_energy,
                             'es': self.compute_ES_energy,
                             'bh': self.compute_buckingham_energy,
                             'silu': self.compute_silu_energy,
                             'vdw': self.compute_vdW_overlap,
                             'vdw_max': self.compute_max_vdW_overlap,
                             'ellipsoid': self.compute_ellipsoidal_overlap,
                             'niggli_overlap': niggli_reduction_penalty,
                             'reduction_en': self.compute_cell_reduction_penalty,
                             'rdf': self.compute_rdf,
                             'ellipsoid_emb': self.compute_ellipsoid_embedding,
                             'uma_pot': self.compute_crystal_uma,
                             'uma_gas_pot': self.compute_lattice_gas_phase_uma,
                             'uma': self.compute_lattice_uma,
                             'mace_pot': self.compute_crystal_mace,
                             'mace_gas_pot': self.compute_lattice_gas_phase_mace,
                             'mace': self.compute_lattice_mace,
                             'latent_harmonic': self.latent_harmonic_en,
                             'latent_multiharmonic': self.latent_multiharmonic_en,
                             }
            self.computes_requires_cluster = COMPUTES_REQUIRE_CLUSTER

    def compute_ellipsoid_embedding(self):
        edge_j_good, molwise_batch, norm_factor, normed_v1, normed_v2, v1, v2, x = self.get_ellipsoid_embedding(
            surface_padding=1
        )
        return scatter(x, molwise_batch, dim=0, dim_size=self.num_graphs, reduce='sum')

    def compute_LJ_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = lj_analysis(self.vdw_radii_tensor,
                              self.edges_dict,
                              self.num_graphs,
                              )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_qLJ_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = qlj_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_eLJ_energy(self, repulsion: Optional[float] = 1.0, **kwargs):
        self._pre_compute_checks()
        stiffness = repulsion * 2.5  # baseline value is 2.5
        if self.is_batch:
            molwise_lj_pot \
                = elj_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               stiffness=stiffness,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_vdW_overlap(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_vdw_overlap \
                = vdW_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               reduce='sum'
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_vdw_overlap

    def compute_max_vdW_overlap(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_vdw_overlap \
                = vdW_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               reduce='max'
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_vdw_overlap

    def compute_ES_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_estat_energy = electrostatic_analysis(
                self.edges_dict,
                self.num_graphs)

        else:
            raise NotImplementedError("ES energies not implemented for single crystals")

        return molwise_estat_energy

    def compute_buckingham_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_buckingham_energy = buckingham_energy(
                self.edges_dict,
                self.num_graphs,
                self.vdw_radii_tensor
            )

        else:
            raise NotImplementedError("BH energies not implemented for single crystals")

        return molwise_buckingham_energy

    def compute_silu_energy(self,
                            repulsion: float = 1.0,
                            **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_silu_energy = silu_energy(
                self.edges_dict,
                self.num_graphs,
                self.vdw_radii_tensor,
                repulsion=repulsion,
            )
        else:
            raise NotImplementedError("SiLU energies not implemented for single crystals")

        return molwise_silu_energy

    def build_and_analyze(self,
                          return_cluster: Optional[bool] = False,
                          noise: Optional[float] = None,
                          cutoff: float = 6,
                          supercell_size: int = 10,
                          ):
        """  # todo this should be deprecated in favour of analyze
        full procedure for building and analyzing a molecular crystal
        this is the OLD version of this function - best practice is to use the 'analyze' function below
        """

        cluster_batch = self.mol2cluster(cutoff, supercell_size)

        if noise is not None:
            cluster_batch.pos += torch.randn_like(cluster_batch.pos) * noise

        cluster_batch.construct_radial_graph(cutoff=cutoff)

        outputs = cluster_batch.compute(requests=['lj', 'es'])
        self.lj = outputs['lj']
        self.es_pot = outputs['es']
        self.scaled_lj = log_rescale_positive(self.lj)

        if return_cluster:
            return self.lj, self.es_pot, self.scaled_lj, cluster_batch
        else:
            return self.lj, self.es_pot, self.scaled_lj

    def analyze(self,
                computes: list,
                return_cluster: Optional[bool] = False,
                noise: Optional[float] = None,
                cutoff: float = 10,
                supercell_size: int = 10,
                std_orientation: Optional[bool] = True,
                assign_outputs: Optional[bool] = False,
                **kwargs
                ):
        """
        full procedure for building and analyzing a molecular crystal
        """
        if not hasattr(self, 'computes'):
            self._init_computes()
        requires_cluster = any(self.computes_requires_cluster.get(k, False) for k in computes)
        if requires_cluster:
            batch_to_analyze = self.mol2cluster(
                cutoff, supercell_size,
                std_orientation=std_orientation)
            if noise is not None:
                batch_to_analyze.pos += torch.randn_like(batch_to_analyze.pos) * noise

            batch_to_analyze.construct_radial_graph(cutoff=cutoff)
            batch_to_analyze._init_computes(override=True)
        else:
            batch_to_analyze = self

        results = batch_to_analyze.compute(requests=computes, **kwargs)

        if assign_outputs:
            for key, value in results.items():
                if key != 'rdf':
                    self.add_graph_attr(value, key)
                else:
                    self.add_graph_attr(value[0], key)
                    self.add_graph_attr(value[1].repeat(self.num_graphs,1), 'rdf_bins')

        if return_cluster:
            return results, batch_to_analyze
        else:
            return results

    def compute_crystal_uma(self,
                            predictor,
                            std_orientation: bool = True,
                            ):
        if self.is_batch:
            return compute_crystal_uma_on_mxt_batch(self.clone(),
                                                    std_orientation,
                                                    predictor)
        else:
            return compute_crystal_uma_on_mxt_batch(collate_data_list(self),
                                                    std_orientation,
                                                    predictor)

    def compute_lattice_uma(self,
                            predictor,
                            std_orienation: bool = True,
                            **kwargs):
        if not hasattr(self, 'uma_gas_pot'):
            self.add_graph_attr(
                self.compute_lattice_gas_phase_uma(predictor),
                'uma_gas_pot')

        if not hasattr(self, 'uma_pot'):
            self.add_graph_attr(
                self.compute_crystal_uma(predictor, std_orientation=std_orienation),
                'uma_pot')

        # lattice energy per molecule, in eV
        # multiply by 96.485 to get kJ/mol
        return ((self.uma_pot / (self.sym_mult * self.z_prime) - self.uma_gas_pot) * 96.485).float()

    def compute_lattice_gas_phase_uma(self,
                                      predictor,
                                      std_orientation: bool = True,
                                      **kwargs,
                                      ):

        if self.is_batch:
            diffuse_batch = self.clone()
        else:
            diffuse_batch = collate_data_list(self)

        if hasattr(self, 'aux_ind'):
            if self.aux_ind is not None:  # if this is a cluster, we have to rebuild the original aunit batch
                diffuse_batch.pos = self.pos[self.aux_ind == 0].detach().clone()
                diffuse_batch.batch = self.batch[self.aux_ind == 0].detach().clone()
                diffuse_batch.z = self.z[self.aux_ind == 0].detach().clone()

        if diffuse_batch.z_prime.amax() > 1:  # do it separately per conformer
            zp1_batch = diffuse_batch.split_to_zp1_batch()
            zp1_batch.reset_sg_info(sg_ind=1)  # big P1 cells
            # zp1_batch.cell_lengths *= 100
            zp1_batch.box_analysis()
            split_ens = compute_crystal_uma_on_mxt_batch(zp1_batch,
                                                         std_orientation,
                                                         predictor,
                                                         pbc=False,
                                                         force_rebuild=True)
            zp_inds = torch.arange(self.num_graphs, dtype=torch.long, device=self.device
                                   ).repeat_interleave(self.z_prime).to(split_ens.device)
            return scatter(split_ens, zp_inds, reduce='mean', dim=0, dim_size=self.num_graphs)

        else:
            # evaluate a diffuse P1 5ell
            diffuse_batch.reset_sg_info(sg_ind=1)  # big P1 cells
            # diffuse_batch.cell_lengths *= 100
            diffuse_batch.box_analysis()
            return compute_crystal_uma_on_mxt_batch(diffuse_batch,
                                                    std_orientation,
                                                    predictor,
                                                    pbc=False,
                                                    force_rebuild=True
                                                    # we don't want to inherit real unit cells so we force rebuild them here
                                                    )

    def compute_crystal_mace(self,
                             predictor,
                             std_orientation: bool = True,
                             ):
        if self.is_batch:
            return compute_crystal_mace_on_mxt_batch(self.clone(),
                                                     predictor,
                                                     std_orientation=std_orientation)
        else:
            return compute_crystal_mace_on_mxt_batch(collate_data_list(self),
                                                     predictor,
                                                     std_orientation=std_orientation)

    def compute_lattice_mace(self,
                             predictor,
                             std_orientation: bool = True,
                             **kwargs):
        if not hasattr(self, 'mace_gas_pot'):
            self.add_graph_attr(
                self.compute_lattice_gas_phase_mace(predictor),
                'mace_gas_pot')

        if not hasattr(self, 'mace_pot'):
            self.add_graph_attr(
                self.compute_crystal_mace(predictor, std_orientation=std_orientation),
                'mace_pot')

        return ((self.mace_pot / (self.sym_mult * self.z_prime) - self.mace_gas_pot) * 96.485).float()

    def compute_lattice_gas_phase_mace(self,
                                       predictor,
                                       std_orientation: bool = True,
                                       **kwargs,
                                       ):
        if self.is_batch:
            diffuse_batch = self.clone()
        else:
            diffuse_batch = collate_data_list(self)

        if hasattr(self, 'aux_ind'):
            if self.aux_ind is not None:
                mask = self.aux_ind == 0
                #
                # # --- diagnostic: every graph must retain at least one canonical-aunit atom ---
                # kept_per_graph = torch.bincount(self.batch[mask], minlength=self.num_graphs)
                # if (kept_per_graph == 0).any():
                #     bad_graphs = (kept_per_graph == 0).nonzero(as_tuple=True)[0]
                #     aux_unique_per_bad = [
                #         self.aux_ind[self.batch == g].unique(return_counts=True)
                #         for g in bad_graphs.tolist()
                #     ]
                #     raise AssertionError(
                #         f"aux_ind == 0 filter produced empty graphs: {bad_graphs.tolist()}\n"
                #         f"  kept_per_graph={kept_per_graph.tolist()}\n"
                #         f"  num_atoms={self.num_atoms.tolist()}\n"
                #         f"  aux_ind values per bad graph (val, count): "
                #         f"{[(v.tolist(), c.tolist()) for v, c in aux_unique_per_bad]}\n"
                #         f"  cell_lengths[bad]={self.cell_lengths[bad_graphs].tolist()}\n"
                #         f"  cell_angles[bad]={self.cell_angles[bad_graphs].tolist()}"
                #     )
                # # --- end diagnostic ---

                diffuse_batch.pos = self.pos[self.aux_ind == 0].detach().clone()
                diffuse_batch.batch = self.batch[self.aux_ind == 0].detach().clone()
                diffuse_batch.z = self.z[self.aux_ind == 0].detach().clone()

        if diffuse_batch.z_prime.amax() > 1:
            zp1_batch = diffuse_batch.split_to_zp1_batch()
            zp1_batch.reset_sg_info(sg_ind=1)
            # zp1_batch.cell_lengths *= 100
            zp1_batch.box_analysis()
            split_ens = compute_crystal_mace_on_mxt_batch(zp1_batch,
                                                          predictor,
                                                          std_orientation=std_orientation,
                                                          pbc=False,
                                                          force_rebuild=True)
            zp_inds = torch.arange(self.num_graphs, dtype=torch.long, device=self.device
                                   ).repeat_interleave(self.z_prime).to(split_ens.device)
            return scatter(split_ens, zp_inds, reduce='mean', dim=0, dim_size=self.num_graphs)

        else:
            diffuse_batch.reset_sg_info(sg_ind=1)
            # diffuse_batch.cell_lengths *= 100
            diffuse_batch.box_analysis()
            return compute_crystal_mace_on_mxt_batch(diffuse_batch,
                                                     predictor,
                                                     std_orientation=std_orientation,
                                                     pbc=False,
                                                     force_rebuild=True)

    def latent_harmonic_en(self, modes: torch.tensor = None, scale: float = 10, **kwargs):
        # a trivial energy function, for testing
        latents = self.latent_params()
        if modes is None:
            modes = torch.zeros((1, latents.shape[-1]), device=self.device)
        energy = 0.5 * ((latents - modes[0]) * scale).pow(2).sum(dim=1)
        # analytic Z = (2pi*T)^(d/2) * scale ^ (-d)
        """
        import numpy as np
        T=2.5
        Z = (2*np.pi*T)**(12/2) * scale ** (-12)
        print(np.log(Z))
        """

        return energy

    def latent_multiharmonic_en(
            self,
            n_modes: int = 24,
            sigma_range: tuple = (0.04, 0.12),
            depth_range: tuple = (0.0, 4.0),
            edge_sigmas: float = 4.0,
            distribution_temperature: float = 1.0,
            seed: int = 0,
            **kwargs,
    ):
        """
        Exactly sampleable multi-well toy energy.

        Defines a Gaussian mixture density

            p(x) = sum_k pi_k N(x; mu_k, sigma_k^2 I)

        and returns

            E(x) = -T log p(x)

        so that exp(-E(x) / T) is exactly the GMM density.

        sigma controls well width.
        depth controls relative basin probability via pi_k ∝ exp(depth_k).

        Most mass is kept inside [-1, 1]^d by placing centers away from edges.
        """
        import math
        import torch

        latents = self.latent_params()
        d = latents.shape[-1]
        T = float(distribution_temperature)

        key = (d, seed, n_modes, sigma_range, depth_range, edge_sigmas, T)
        cache = getattr(self, "_gmm_cache", {})

        if cache.get("key") != key:
            g = torch.Generator(device="cpu").manual_seed(seed)

            sigma = torch.empty(n_modes).uniform_(*sigma_range, generator=g)
            depth = torch.empty(n_modes).uniform_(*depth_range, generator=g)

            # Keep centers sufficiently far from the boundary.
            m = edge_sigmas * float(sigma.max())
            assert m < 1.0, f"margin {m:.3f} >= 1; shrink sigma_range or edge_sigmas"

            lo, hi = -1.0 + m, 1.0 - m
            mu = torch.empty(n_modes, d).uniform_(0, 1, generator=g) * (hi - lo) + lo

            log_pi = depth - torch.logsumexp(depth, dim=0)

            cache = {
                "key": key,
                "mu": mu,  # (K, d)
                "sigma": sigma,  # (K,)
                "log_pi": log_pi,  # (K,)
                "temperature": T,
            }
            self._gmm_cache = cache

        mu = cache["mu"].to(self.device)
        sigma = cache["sigma"].to(self.device)
        log_pi = cache["log_pi"].to(self.device)

        x = latents
        K, d = mu.shape

        inv_var = 1.0 / sigma.pow(2)

        sq = ((x[:, None, :] - mu[None]) ** 2).sum(-1) * inv_var[None]

        # Full normalized Gaussian log density.
        log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * torch.log(sigma)
        log_comp = log_pi[None] + log_norm[None] - 0.5 * sq

        log_p = torch.logsumexp(log_comp, dim=1)

        energy = -T * log_p

        # Z=1 by construction at target T
        # the change in Z if trained at other temperatures generally intractable

        return energy

    def sample_latent_multiharmonic(
            self,
            n_samples: int,
            distribution_temperature: float = 1.0,
            seed: int = 0,
            **kwargs,
    ) -> torch.Tensor:
        """
        Exact sampler for latent_gmm_en.

        Note: temperature is checked only for consistency. The actual GMM density
        is cached by latent_gmm_en at construction time.
        """
        import torch

        cache = getattr(self, "_gmm_cache", None)
        assert cache is not None, "Call latent_gmm_en first to populate cache."

        cached_T = float(cache["temperature"])
        T = float(distribution_temperature)
        assert abs(T - cached_T) < 1e-8, (
            f"Sampler temperature {T} does not match cached energy temperature {cached_T}."
        )

        mu = cache["mu"].to(self.device)
        sigma = cache["sigma"].to(self.device)
        log_pi = cache["log_pi"].to(self.device)

        K, d = mu.shape

        g = torch.Generator(device="cpu").manual_seed(seed)

        component_ids = torch.multinomial(
            log_pi.exp().cpu(),
            n_samples,
            replacement=True,
            generator=g,
        ).to(self.device)

        eps = torch.randn(n_samples, d, generator=g, device="cpu").to(self.device)

        samples = mu[component_ids] + sigma[component_ids, None] * eps

        return samples

    def compute_rdf(self,  # todo rebuild analyses with a template
                    rdf_cutoff: float = 10,
                    bins: int = 100,
                    rdf_mode: Optional[str] = None,
                    **kwargs,
                    ):
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing RDF!"
                               "Do batch.construct_radial_graph before calling.")

        assert self.is_batch, "RDF method not implemented for single crystals"
        rdf, bin_edges, rdf_pair_dict = crystal_rdf(self,
                                                    self.edges_dict,
                                                    rrange=(0, rdf_cutoff),
                                                    bins=bins,
                                                    mode=rdf_mode
                                                    )
        return rdf / self.z_prime[:, None, None], bin_edges, rdf_pair_dict

    def compute_cell_reduction_penalty(self, margin: float = 0.0, **kwargs):
        """Compute a penalty term for the given crystal system that pushes it towards
        our canonical / standardized / reduced cell geometry, following spglib"""
        # todo check behaviors / correctness for higher crystal systems
        # cell_parameters = self.full_cell_parameters()
        cell_lengths, cell_angles, aunit_positions, aunit_orientations = self.split_cell_params()

        'get cs masks'
        # crystal_systems  #
        sg = self.sg_ind
        E = cell_reduction_penalty(cell_angles, cell_lengths, sg, margin)

        return E

    def batch_compack(self, ref_path, inds_to_check, n_cpus: int = 8):  # todo refactor into analysis code
        import numpy as np
        import multiprocessing as mp

        self.mol2ucell()
        self.write_cif(inds_to_check, f'compack', 'unit cell')
        pool = mp.Pool(n_cpus)
        results = []
        for ind in range(len(inds_to_check)):
            results.append(
                pool.apply_async(
                    single_compack_run,
                    (ind, f'compack_{ind}.cif', ref_path)
                )
            )
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        matches = np.array([res[1] for res in results])
        rmsds = np.array([res[0] for res in results])

        return matches, rmsds
