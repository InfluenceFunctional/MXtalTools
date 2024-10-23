from typing import Tuple, Optional

from torch import optim
from torch_geometric.loader.dataloader import Collater
import torch

from torch_scatter import scatter
from tqdm import tqdm
import multiprocessing as mp
from math import ceil
import torch.nn.functional as F

from mxtaltools.common.utils import init_sym_info
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.builder import CrystalBuilder
from mxtaltools.crystal_building.utils import set_molecule_alignment, overwrite_symmetry_info
from mxtaltools.crystal_search.utils import cell_clustering, coarse_filter, get_topk_samples, rdf_clustering
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.functions.crystal_rdf import new_crystal_rdf
from mxtaltools.models.functions.vdw_overlap import vdw_analysis
from mxtaltools.models.utils import denormalize_generated_cell_params, get_intermolecular_dists_dict, enforce_1d_bound, \
    enforce_crystal_system


class Sampler:
    def __init__(self,
                 seed,
                 device,
                 machine,
                 prior,
                 generator,
                 autoencoder,
                 discriminator,
                 show_tqdm=False,
                 skip_rdf=False,
                 gd_score_func='vdW',  # 'vdW' or 'discriminator'
                 ):
        torch.manual_seed(seed)
        self.device = device
        self.machine = machine
        self.supercell_size = 5
        self.cutoff = 6
        self.prior = prior
        self.generator = generator
        self.discriminator = discriminator
        self.sym_info = init_sym_info()
        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        self.autoencoder = autoencoder
        self.collater = Collater(0, 0)
        self.supercell_builder = CrystalBuilder(device=self.device,
                                                rotation_basis='cartesian')
        self.show_tqdm = show_tqdm
        self.max_samples_per_sampling_job = 1000
        self.max_samples_per_optimization_job = 50
        self.max_raw_samples = 10000
        self.skip_rdf = skip_rdf
        self.gd_score_func = gd_score_func

    def crystal_search(self,
                       molecule_data: CrystalData,
                       sampling_type: str,
                       num_samples: int,
                       batch_size: int,
                       sg_to_sample: torch.LongTensor,
                       packing_coeff_range: list[float],
                       vdw_threshold: float,
                       rdf_cutoff: float,
                       cell_params_cutoff: float,
                       opt_eps: float,
                       chain_length: Optional[int] = None,
                       step_size: float = None,
                       parallel=False,
                       ):

        if parallel:
            self.show_tqdm = False
            if self.machine == 'local':
                num_cpus = mp.cpu_count() - 4
            else:
                num_cpus = mp.cpu_count() - 1

            packing_coeff, rdf, rr, samples, vdw = self.do_sampling_parallel(batch_size, chain_length, molecule_data,
                                                                             num_cpus, num_samples, sampling_type,
                                                                             sg_to_sample, step_size)
        else:
            packing_coeff, rdf, rr, samples, vdw = self.do_sampling(batch_size, chain_length, molecule_data,
                                                                    num_samples,
                                                                    sampling_type, sg_to_sample, step_size)

        # filtering
        init_len, packing_coeff, rdf, samples, vdw = coarse_filter(packing_coeff, packing_coeff_range, rdf,
                                                                   samples, vdw, vdw_threshold)

        # take topk samples (max that fits nicely on local RAM)
        packing_coeff, rdf, samples, vdw = get_topk_samples(packing_coeff, rdf, samples, vdw, self.max_raw_samples)

        # clustering
        packing_coeff, rdf, samples, vdw = cell_clustering(cell_params_cutoff, packing_coeff, rdf, samples, vdw,
                                                           self.prior.norm_factors[sg_to_sample[0]])
        #packing_coeff, rdf, samples, vdw = rdf_clustering(packing_coeff, rdf, rdf_cutoff, rr, samples, vdw)
        #
        # # optimization
        if parallel:
            packing_coeff, rdf, samples, vdw = self._local_optimization_parallel(batch_size, molecule_data, num_cpus,
                                                                                 packing_coeff, rdf, samples,
                                                                                 sg_to_sample,
                                                                                 vdw,
                                                                                 opt_eps)
        else:
            packing_coeff, rdf, samples, vdw = self._local_optimization(batch_size,
                                                                        molecule_data,
                                                                        packing_coeff,
                                                                        rdf,
                                                                        samples,
                                                                        vdw,
                                                                        sg_to_sample,
                                                                        opt_eps)
        print("Local optimization finished")
        # re-cluster
        packing_coeff, rdf, samples, vdw = cell_clustering(cell_params_cutoff, packing_coeff, rdf, samples, vdw,
                                                           self.prior.norm_factors[sg_to_sample[0]])
        #packing_coeff, rdf, samples, vdw, rdf_distmat = rdf_clustering(packing_coeff, rdf, rdf_cutoff, rr, samples, vdw, num_cpus)

        results = {'samples': samples.cpu().detach().numpy(),
                   'vdw': vdw.cpu().detach().numpy(),
                   'rdf': rdf.cpu().detach().numpy(),
                   'packing_coeff': packing_coeff.cpu().detach().numpy(),
                   'rr': rr.cpu().detach().numpy(),
                   'molecule': molecule_data.cpu().detach(),
                   #'rdf_distmat': rdf_distmat
                   }

        return results

    '''
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=packing_coeff, y=vdw, mode='markers')).show()
    '''

    def _rebuild_sampled_crystals(self, molecule_data, space_group_ind, samples):

        mol_batch = self.collater([molecule_data for _ in range(len(samples))])
        mol_batch = overwrite_symmetry_info(mol_batch,
                                            [space_group_ind for _ in range(len(samples))],
                                            self.sym_info,
                                            randomize_sgs=False)
        descaled_sample = denormalize_generated_cell_params(samples, mol_batch, self.supercell_builder.asym_unit_dict)

        supercell_data, generated_cell_volumes = self.supercell_builder.build_zp1_supercells(
            mol_batch,
            cell_parameters=descaled_sample,
            supercell_size=self.supercell_size,
            graph_convolution_cutoff=self.cutoff,
            skip_refeaturization=True,
            align_to_standardized_orientation=True,
        )

        reduced_volume = generated_cell_volumes / supercell_data.sym_mult
        packing_coeff = mol_batch.mol_volume / reduced_volume
        dist_dict = get_intermolecular_dists_dict(supercell_data, self.cutoff, 100)
        molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, eval_vdw_loss \
            = vdw_analysis(self.vdw_radii_tensor,
                           dist_dict,
                           mol_batch.num_graphs,
                           turnover_potential=10)

        per_mol_vdw_loss = (vdw_loss / mol_batch.num_atoms)
        return supercell_data, packing_coeff, per_mol_vdw_loss,

    def do_sampling(self, batch_size, chain_length, molecule_data, num_samples, sampling_type, sg_to_sample, step_size):
        # sampling

        if 'MCMC' in sampling_type:
            mol_batch = self.collater([molecule_data for _ in range(batch_size)])
            mol_batch, scalar_mol_embedding, vector_mol_embedding = self.embed_molecule_for_sampling(
                mol_batch,
                'random',
                sg_to_sample,
                skip_embedding=True
            )
            init_state = self._sample_from_prior(mol_batch)

        if sampling_type == 'prior':
            samples, vdw, packing_coeff, rdf, rr = self.sample_from_distribution(
                molecule_data.clone(),
                batch_size,
                sg_to_sample,
                num_samples,
                'random',
                'prior',
            )
        elif sampling_type == 'random_MCMC':

            samples, vdw, packing_coeff, rdf, rr = self.MC_sampling(
                molecule_data.clone(),
                batch_size,
                init_state,
                sg_to_sample,
                num_samples,
                chain_length,
                step_size,
                'random',
                'random',
            )
        elif sampling_type == 'generator_MCMC':

            samples, vdw, packing_coeff, rdf, rr = self.MC_sampling(
                molecule_data.clone(),
                batch_size,
                init_state,
                sg_to_sample,
                num_samples,
                chain_length,
                step_size,
                'generator',
                'random',
            )
        else:
            assert False, f"{sampling_type} is not implemented as a sampler!"
        return packing_coeff, rdf, rr, samples, vdw

    def _local_optimization_parallel(self,
                                     batch_size,
                                     molecule_data,
                                     num_cpus,
                                     packing_coeff,
                                     rdf,
                                     samples,
                                     sg_to_sample,
                                     vdw,
                                     opt_eps):
        # optimization
        pool = mp.Pool(num_cpus)
        num_samples = len(vdw)
        num_opt_jobs = ceil(num_samples / self.max_samples_per_optimization_job)
        samples_per_job = ceil(num_samples / num_opt_jobs)
        optimization_output = []
        for ind in range(num_opt_jobs):
            relevant_inds = torch.arange(ind * samples_per_job, min((ind + 1) * samples_per_job, num_samples))
            optimization_output.append(
                pool.apply_async(
                    self._local_optimization,
                    (min(batch_size, len(relevant_inds)),
                     molecule_data,
                     packing_coeff[relevant_inds],
                     rdf[relevant_inds],
                     samples[relevant_inds],
                     vdw[relevant_inds],
                     sg_to_sample[:len(relevant_inds)],
                     opt_eps)
                )
            )
        pool.close()
        pool.join()
        optimization_output = [elem.get() for elem in optimization_output]
        packing_coeff = torch.cat([out[0] for out in optimization_output])
        rdf = torch.cat([out[1] for out in optimization_output])
        samples = torch.cat([out[2] for out in optimization_output])
        vdw = torch.cat([out[3] for out in optimization_output])
        return packing_coeff, rdf, samples, vdw

    def do_sampling_parallel(self, batch_size, chain_length, molecule_data, num_cpus, num_samples, sampling_type,
                             sg_to_sample, step_size):
        pool = mp.Pool(num_cpus)
        num_jobs = ceil(num_samples / self.max_samples_per_sampling_job)
        samples_per_job = ceil(num_samples / num_jobs)
        sampling_output = []
        for ind in range(num_jobs):
            sampling_output.append(
                pool.apply_async(
                    self.do_sampling,
                    (batch_size, chain_length, molecule_data, samples_per_job,
                     sampling_type, sg_to_sample, step_size)
                )
            )
        pool.close()
        pool.join()
        sampling_output = [elem.get() for elem in sampling_output]
        packing_coeff = torch.cat([out[0] for out in sampling_output])
        rdf = torch.cat([out[1] for out in sampling_output])
        rr = sampling_output[0][2]
        samples = torch.cat([out[3] for out in sampling_output])
        vdw = torch.cat([out[4] for out in sampling_output])
        return packing_coeff, rdf, rr, samples, vdw

    # todo topk rebuilding
    # final clustering & beautification

    def _local_optimization(self,
                            batch_size,
                            molecule_data,
                            packing_coeff,
                            rdf,
                            samples,
                            vdw,
                            sg_to_sample,
                            opt_eps):

        num_batches = ceil(len(samples) / batch_size)

        for batch_ind in tqdm(range(num_batches), disable=not self.show_tqdm):
            mol_batch = self.collater([molecule_data for _ in range(batch_size)])

            mol_batch, scalar_mol_embedding, vector_mol_embedding = self.embed_molecule_for_sampling(
                mol_batch,
                'standardized',
                sg_to_sample,
                skip_embedding=True
            )

            current_inds = torch.arange(batch_ind * batch_size, min(len(samples), (batch_ind + 1) * batch_size))

            (packing_coeff[current_inds], rdf[current_inds], samples[current_inds], vdw[current_inds],
             optimization_record) = self._gradient_descent_optimization(
                samples[current_inds],
                self.collater(mol_batch[:len(current_inds)]).clone(),
                max_num_steps=1000,
                convergence_eps=opt_eps,
                lr=1e-3,
                optimizer_func=torch.optim.Rprop,
                score_func=self.gd_score_func,
            )

        return packing_coeff, rdf, samples, vdw

    def sample_from_distribution(self,
                                 sample_data: CrystalData,
                                 batch_size: int,
                                 sg_to_sample: torch.LongTensor,
                                 tot_samples: int,
                                 cc_orientation: str = 'standardized',
                                 sample_source: str = 'prior',
                                 ):

        num_batches = ceil(tot_samples / batch_size)
        sample_record, vdw_potential_record, packing_coeff_record, rdf_record = [], [], [], []

        for _ in tqdm(range(num_batches), disable=not self.show_tqdm):
            samples, vdw_potential, packing_coeff, rdf, rr = self._batch_sample_from_distribution(
                sample_data,
                batch_size,
                cc_orientation,
                sg_to_sample,
                sample_source)

            sample_record.append(samples)
            vdw_potential_record.append(vdw_potential)
            packing_coeff_record.append(packing_coeff)
            rdf_record.append(rdf)

        return (torch.cat(sample_record),
                torch.cat(vdw_potential_record),
                torch.cat(packing_coeff_record),
                torch.cat(rdf_record),
                rr.cpu()
                )

    def _batch_sample_from_distribution(self,
                                        sample_data: CrystalData,
                                        batch_size: int,
                                        cc_orientation: str,
                                        sg_to_sample: torch.LongTensor,
                                        sample_source: str,
                                        ):

        data = self.collater([sample_data for _ in range(batch_size)]).to(self.device)

        mol_data, scalar_mol_embedding, vector_mol_embedding = self.embed_molecule_for_sampling(
            data,
            cc_orientation,
            sg_to_sample,
            skip_embedding=sample_source == 'prior'
        )

        if sample_source == 'prior':
            raw_generated_sample = self._sample_from_prior(mol_data)

            dist_dict, packing_coeff, per_mol_vdw_loss, sample_to_compare, supercell_data = self._build_and_analyze_crystal(
                raw_generated_sample, mol_data)

            if not self.skip_rdf:
                rdf, rr, _ = new_crystal_rdf(supercell_data, dist_dict,
                                             rrange=[0, 6], bins=2000,
                                             mode='intermolecular', atomwise=True,
                                             raw_density=True, cpu_detach=False,
                                             atomic_numbers_override=mol_data.x.unique().long())
            else:
                rdf, rr = torch.empty(batch_size), torch.empty(batch_size)
        else:
            assert False, f"sample_source {sample_source} not supported"

        return (sample_to_compare.detach().cpu(),
                per_mol_vdw_loss.detach().cpu(),
                packing_coeff.detach().cpu(),
                rdf.detach().cpu(),
                rr.detach().cpu())

    def _build_and_analyze_crystal(self,
                                   raw_sample,
                                   mol_batch):
        descaled_sample = denormalize_generated_cell_params(
            raw_sample,
            mol_batch,
            self.supercell_builder.asym_unit_dict
        )

        supercell_data, generated_cell_volumes = (
            self.supercell_builder.build_zp1_supercells(
                molecule_data=mol_batch,
                cell_parameters=descaled_sample,
                supercell_size=self.supercell_size,
                graph_convolution_cutoff=self.cutoff,
                align_to_standardized_orientation=False,
                skip_refeaturization=False,
            ))

        reduced_volume = generated_cell_volumes / supercell_data.sym_mult
        packing_coeff = mol_batch.mol_volume / reduced_volume
        dist_dict = get_intermolecular_dists_dict(supercell_data, self.cutoff, 100)
        molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, eval_vdw_loss \
            = vdw_analysis(self.vdw_radii_tensor,
                           dist_dict,
                           mol_batch.num_graphs,
                           turnover_potential=10)

        'canonicalize orientation'
        sample_to_compare = raw_sample.clone()
        sample_to_compare[:, 9:] = supercell_data.cell_params[:, 9:]

        per_mol_vdw_loss = vdw_loss / mol_batch.num_atoms

        return dist_dict, packing_coeff, per_mol_vdw_loss, sample_to_compare, supercell_data

    def _sample_from_prior(self, mol_data) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_samples = self.prior(mol_data.num_graphs, mol_data.sg_ind).to(self.device)

        return raw_samples

    def MC_sampling(self,
                    sample_data: CrystalData,
                    batch_size: int,
                    init_state: torch.FloatTensor,
                    sg_to_sample: torch.LongTensor,
                    tot_samples: int,
                    chain_length: int,
                    step_size: torch.FloatTensor,
                    sampler: str,
                    cc_orientation: str = 'standardized',
                    ):

        num_batches = ceil(tot_samples / batch_size)
        sample_record, vdw_potential_record, packing_coeff_record, rdf_record = [], [], [], []

        for _ in tqdm(range(num_batches), disable=not self.show_tqdm):
            data_batch = self.collater([sample_data for _ in range(batch_size)]).to(self.device)
            mol_data, scalar_mol_embedding, vector_mol_embedding = self.embed_molecule_for_sampling(
                data_batch,
                cc_orientation,
                sg_to_sample,
                skip_embedding=sampler == 'random'
            )

            samples, vdw_potential, packing_coeff, rdf, rr = self._MC_chain_batch(
                mol_data,
                init_state,
                sampler,
                chain_length,
                step_size,
                scalar_mol_embedding, vector_mol_embedding,
            )

            sample_record.append(samples)
            vdw_potential_record.append(vdw_potential)
            packing_coeff_record.append(packing_coeff)
            rdf_record.append(rdf)

        return (torch.cat(sample_record),
                torch.cat(vdw_potential_record),
                torch.cat(packing_coeff_record),
                torch.cat(rdf_record),
                rr
                )

    def _MC_chain_batch(self,
                        mol_batch: CrystalData,
                        init_state: torch.FloatTensor,
                        sampler: str,
                        chain_length: int,
                        step_size: torch.FloatTensor,
                        scalar_mol_embedding: Optional[torch.FloatTensor],
                        vector_mol_embedding: Optional[torch.FloatTensor],
                        MC_temperature: float = 0.1
                        ):

        batch_size = mol_batch.num_graphs
        step_size_tensor = step_size * torch.ones(batch_size, device=self.device)
        scaling_factor = (self.prior.norm_factors[mol_batch.sg_ind, :] + 1e-4)
        absolute_reference = torch.eye(3, dtype=torch.float32, device=self.device
                                       ).reshape(1, 9).repeat(mol_batch.num_graphs, 1)

        alpha_randoms = torch.rand((chain_length, batch_size)).to(self.device)

        samples_to_build_list, vdw_potential_list, packing_coeff_list, rdf_list = [], [], [], []
        vdw_traj = []
        with torch.no_grad():
            for chain_iter in range(chain_length):
                if chain_iter == 0:
                    # always accept the first step
                    prev_MC_score = torch.ones(batch_size, device=self.device) * 10000
                    prior = init_state.clone()

                elif chain_iter > 0:
                    # Metropolis MC update
                    score_difference = MC_score - prev_MC_score
                    acceptance_ratio = torch.minimum(torch.ones_like(score_difference),
                                                     torch.exp(-score_difference / MC_temperature))
                    accept_flags = alpha_randoms[chain_iter] < acceptance_ratio
                    prior[accept_flags] = raw_generated_sample[accept_flags]
                    prev_MC_score[accept_flags] = MC_score[accept_flags].cpu().detach()

                vdw_traj.append(prev_MC_score.detach().cpu().clone())

                if sampler == 'generator':
                    raw_generated_sample, descaled_generated_sample = self._generator_MC_step(
                        absolute_reference, mol_batch, prior,
                        scalar_mol_embedding, scaling_factor,
                        step_size_tensor, vector_mol_embedding)

                elif sampler == 'random':
                    raw_generated_sample, descaled_generated_sample = self._random_MC_step(mol_batch, prior,
                                                                                           step_size_tensor, )

                else:
                    assert False, f"sample_source {sampler} not supported"

                dist_dict, packing_coeff, per_mol_vdw_loss, sample_to_compare, supercell_data = self._build_and_analyze_crystal(
                    raw_generated_sample, mol_batch)

                MC_score = per_mol_vdw_loss

                if not self.skip_rdf:
                    rdf, rr, _ = new_crystal_rdf(supercell_data, dist_dict,
                                                 rrange=[0, 6], bins=2000,
                                                 mode='intermolecular', atomwise=True,
                                                 raw_density=True, cpu_detach=False,
                                                 atomic_numbers_override=mol_batch.x.unique().long())
                else:
                    rdf, rr = torch.empty(batch_size), torch.empty(batch_size)

                samples_to_build_list.append(sample_to_compare.detach().cpu())
                vdw_potential_list.append(per_mol_vdw_loss.detach().cpu())
                packing_coeff_list.append(packing_coeff.detach().cpu())
                rdf_list.append(rdf.detach().cpu())

        return (torch.cat(samples_to_build_list),
                torch.cat(vdw_potential_list),
                torch.cat(packing_coeff_list),
                torch.cat(rdf_list),
                rr.detach())

    # # visualize MC trajectories
    # import plotly.graph_objects as go
    # fig = go.Figure()
    # vdw_record = torch.stack(vdw_potential_list)
    # for ind in range(vdw_record.shape[1]):
    #     fig.add_scatter(y=vdw_record[:, ind], name=ind, legendgroup=ind, showlegend=True, opacity=0.3)
    # vdw_traj_combo = torch.stack(vdw_traj)
    # for ind in range(vdw_record.shape[1]):
    #     fig.add_scatter(y=vdw_traj_combo[:, ind], name=ind, legendgroup=ind, showlegend=False, opacity=1)
    # fig.update_layout(yaxis_range=[vdw_traj_combo.amin(), vdw_record.amax()])
    # fig.show()

    def _generator_MC_step(self,
                           absolute_reference,
                           mol_data,
                           prior,
                           scalar_mol_embedding,
                           scaling_factor,
                           step_size_tensor,
                           vector_mol_embedding):

        conditioning_vector = torch.cat((
            scalar_mol_embedding,
            prior,
            absolute_reference,
            (step_size_tensor * torch.rand_like(step_size_tensor))[:, None],
            scaling_factor),
            dim=1)
        raw_samples = self.generator.forward(
            conditioning_vector,
            vector_mol_embedding,
            mol_data.sg_ind,
            prior,
        )
        samples_to_build = denormalize_generated_cell_params(
            raw_samples,
            mol_data,
            self.supercell_builder.asym_unit_dict
        )
        return raw_samples, samples_to_build

    def _random_MC_step(self, mol_data, prior, step_size_tensor):
        rand_vector = torch.randn_like(prior)
        random_direction = rand_vector / rand_vector.norm(dim=1)[:, None]
        raw_samples = prior + random_direction * step_size_tensor[:, None]
        raw_samples = self.generator.cleanup_sample(raw_samples, mol_data.sg_ind)

        samples_to_build = denormalize_generated_cell_params(
            raw_samples,
            mol_data,
            self.supercell_builder.asym_unit_dict
        )
        return raw_samples, samples_to_build

    def embed_molecule_for_sampling(self, data, orientation, generate_sgs, skip_embedding=False):
        # clone, center, and orient the molecules
        _, mol_data = self.preprocess_ae_inputs(
            data,
            orientation_override=orientation)

        mol_data = overwrite_symmetry_info(mol_data,
                                           generate_sgs,
                                           self.sym_info,
                                           randomize_sgs=False)

        if not skip_embedding:
            # embed molecules
            with torch.no_grad():
                vector_mol_embedding = self.autoencoder.encode(mol_data.clone())
                scalar_mol_embedding = self.autoencoder.scalarizer(vector_mol_embedding)

        else:
            scalar_mol_embedding, vector_mol_embedding = None, None

        return mol_data, scalar_mol_embedding, vector_mol_embedding

    def preprocess_ae_inputs(self, data,
                             orientation_override=None,
                             deprotonate=False):
        # random global roto-inversion
        if orientation_override is not None:
            data = set_molecule_alignment(data,
                                          mode=orientation_override,
                                          right_handed=True,  # right handed
                                          include_inversion=False)  # force all samples to come with same handedness

        # optionally, deprotonate
        input_data = self.fix_autoencoder_protonation(data,
                                                      deprotonate=deprotonate)

        # subtract mean OF THE INPUT from BOTH reference and input
        centroids = scatter(input_data.pos, input_data.batch, reduce='mean', dim=0)
        data.pos -= torch.repeat_interleave(centroids, data.num_atoms, dim=0, output_size=data.num_nodes)
        input_data.pos -= torch.repeat_interleave(centroids, input_data.num_atoms, dim=0,
                                                  output_size=input_data.num_nodes)

        return data, input_data

    def fix_autoencoder_protonation(self, data, deprotonate=False):
        if deprotonate:
            heavy_atom_inds = torch.argwhere(data.x != 1).flatten()  # protons are atom type 1
            input_cloud = data.detach().clone()
            input_cloud.x = input_cloud.x[heavy_atom_inds]
            input_cloud.pos = input_cloud.pos[heavy_atom_inds]
            input_cloud.batch = input_cloud.batch[heavy_atom_inds]
            a, b = torch.unique(input_cloud.batch, return_counts=True)
            input_cloud.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
            input_cloud.num_atoms = torch.diff(input_cloud.ptr).long()
        else:
            input_cloud = data.detach().clone()

        return input_cloud

    def _gradient_descent_optimization(self,
                                       init_sample: torch.Tensor,
                                       mol_batch: CrystalData,
                                       max_num_steps: int,
                                       convergence_eps: float,
                                       lr: float,
                                       optimizer_func,
                                       score_func: str,
                                       ):
        """
        do a local optimization
        """

        sample = torch.tensor(init_sample.clone(),
                              device=mol_batch.x.device,
                              requires_grad=True,
                              dtype=torch.float32)

        optimizer = optimizer_func([sample], lr=lr)

        max_lr_target_time = max_num_steps // 10
        max_lr = 1e-2
        grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)

        scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
        scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
        hit_max_lr = False

        n_samples = len(sample)

        vdw_record = torch.zeros((max_num_steps, n_samples))
        samples_record = torch.zeros((max_num_steps, n_samples, 12))
        loss_record = torch.zeros_like(vdw_record)
        lr_record = torch.zeros(max_num_steps)
        packing_record = torch.zeros_like(vdw_record)

        converged = False
        with torch.enable_grad():
            with tqdm(total=max_num_steps, disable=not self.show_tqdm) as pbar:
                s_ind = 0
                while s_ind < max_num_steps and not converged:
                    optimizer.zero_grad()

                    cleaned_sample = self.cleanup_sample(sample, mol_batch.sg_ind)
                    descaled_cleaned_sample = denormalize_generated_cell_params(
                        cleaned_sample, mol_batch, self.supercell_builder.asym_unit_dict
                    )

                    # todo functionalize the below - it seems we do it everywhere
                    supercell_data, generated_cell_volumes = (
                        self.supercell_builder.build_zp1_supercells(
                            molecule_data=mol_batch,
                            cell_parameters=descaled_cleaned_sample,
                            supercell_size=self.supercell_size,
                            graph_convolution_cutoff=self.cutoff,
                            align_to_standardized_orientation=True,
                            skip_refeaturization=False,
                        ))

                    reduced_volume = generated_cell_volumes / supercell_data.sym_mult
                    packing_coeff = mol_batch.mol_volume / reduced_volume
                    if score_func == 'discriminator':
                        output, extra_outputs = self.discriminator(
                            supercell_data.clone(), return_dists=True, return_latent=False)
                        dist_dict = extra_outputs['dists_dict']
                    elif score_func == 'vdW':
                        dist_dict = get_intermolecular_dists_dict(supercell_data, self.cutoff, 100)
                    else:
                        assert False, f"{score_func} is not an implemented score function for gradient descent optimization"

                    molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, eval_vdw_loss \
                        = vdw_analysis(self.vdw_radii_tensor,
                                       dist_dict,
                                       mol_batch.num_graphs,
                                       turnover_potential=10)
                    if score_func == 'discriminator':
                        loss = F.softplus(output[:, 2])
                    elif score_func == 'vdw':
                        loss = vdw_loss / mol_batch.num_atoms

                    loss.mean().backward()  # compute gradients
                    optimizer.step()  # apply grad

                    sample_to_compare = cleaned_sample.clone()
                    sample_to_compare[:, 9:] = supercell_data.cell_params[:, 9:]

                    vdw_record[s_ind] = (vdw_loss / mol_batch.num_atoms).detach()
                    samples_record[s_ind] = sample_to_compare.detach()
                    loss_record[s_ind] = loss.detach()
                    packing_record[s_ind] = packing_coeff.detach()

                    lr = optimizer.param_groups[0]['lr']
                    lr_record[s_ind] = lr
                    if lr >= max_lr:
                        hit_max_lr = True
                    if hit_max_lr:
                        if lr > 1e-5:
                            scheduler1.step()  # shrink
                    else:
                        scheduler2.step()  # grow

                    if (s_ind > 10) and (s_ind % 50 == 0):
                        converged = all(vdw_record[s_ind - 10:s_ind, :].std(0) < convergence_eps)

                    s_ind += 1
                    if s_ind % 100 == 0:
                        pbar.update(100)

        sampling_dict = {'std_cell_params': samples_record[:s_ind],
                         'vdw_loss': vdw_record[:s_ind],
                         'overall_loss': loss_record[:s_ind],
                         'packing_coeff': packing_record[:s_ind],
                         }
        # do RDF on final sample
        if not self.skip_rdf:
            rdf, rr, _ = new_crystal_rdf(supercell_data, dist_dict,
                                         rrange=[0, 6], bins=2000,
                                         mode='intermolecular', atomwise=True,
                                         raw_density=True, cpu_detach=False,
                                         atomic_numbers_override=mol_batch.x.unique().long())
        else:
            rdf, rr = torch.empty(mol_batch.num_graphs), torch.empty(mol_batch.num_graphs)

        return (packing_coeff.detach().cpu(), rdf.detach().cpu(),
                sample_to_compare.detach().cpu(), loss.detach().cpu(),
                sampling_dict)

    # from mxtaltools.common.ase_interface import crystaldata_batch_to_ase_mols_list
    #
    # crystaldata_batch_to_ase_mols_list(supercell_data, show_mols=True, exclusion_level='convolve with',
    #                                    inclusion_distance=6)
    #
    # import plotly.graph_objects as go
    # fig = go.Figure()
    # for ind in range(vdw_record.shape[1]):
    #     fig.add_scatter(y=vdw_record[:, ind])
    # fig.show()

    def cleanup_sample(self, raw_sample, sg_ind_list):
        # force outputs into physical ranges
        # cell lengths have to be positive nonzero
        cell_lengths = raw_sample[:, :3].clip(min=0.01)
        # range from (0,pi) with 20% padding to prevent too-skinny cells
        cell_angles = enforce_1d_bound(raw_sample[:, 3:6], x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
                                       mode='hard')
        # positions must be on 0-1
        mol_positions = enforce_1d_bound(raw_sample[:, 6:9], x_span=0.5, x_center=0.5, mode='hard')
        # for now, just enforce vector norm
        rotvec = raw_sample[:, 9:12]
        norm = torch.linalg.norm(rotvec, dim=1)
        new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode='hard')  # MUST be nonzero
        new_rotvec = rotvec / norm[:, None] * new_norm[:, None]
        # invert_inds = torch.argwhere(new_rotvec[:, 2] < 0)
        # new_rotvec[invert_inds] = -new_rotvec[invert_inds]  # z direction always positive
        # force cells to conform to crystal system
        cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list,
                                                           self.supercell_builder.symmetries_dict)
        sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)
        return sample