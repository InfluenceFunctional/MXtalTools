from typing import Tuple

from torch_geometric.loader.dataloader import Collater
import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.utils import init_sym_info
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.crystal_building.utils import set_molecule_alignment, overwrite_symmetry_info
from mxtaltools.models.functions.crystal_rdf import new_crystal_rdf
from mxtaltools.models.functions.vdw_overlap import vdw_analysis
from mxtaltools.models.utils import denormalize_generated_cell_params, compute_prior_loss, \
    get_intermolecular_dists_dict, crystal_filter_cluster


class Sampler:
    def __init__(self,
                 seed,
                 device,
                 prior,
                 generator,
                 autoencoder,
                 ):
        self.device = device
        self.supercell_size = 5
        self.cutoff = 6
        self.prior = prior
        self.generator = generator
        self.sym_info = init_sym_info()
        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        self.autoencoder = autoencoder
        self.collater = Collater(0, 0)
        self.supercell_builder = SupercellBuilder(device=self.device,
                                                  rotation_basis='cartesian')

    def sample_and_cluster(self, init_sample, tot_samples,
                           batch_size, cc_orientation,
                           sg_to_sample, sample_source, show_progress=True,
                           packing_coeff_cutoff=1.2,
                           vdw_cutoff=1,
                           cell_params_threshold=0.01,
                           rdf_dist_threshold=0.05,
                           variation_factor=None,
                           ):
        with torch.no_grad():
            samples, vdw_potential, packing_coeff, rdf, rr = self.do_sampling(
                batch_size, cc_orientation,
                init_sample, sample_source,
                sg_to_sample, tot_samples,
                show_tqdm=show_progress, variation_factor=variation_factor)

            vdw_potential, samples, rdf, packing_coeff, rdf_distmat = \
                crystal_filter_cluster(vdw_potential, rdf, rr, samples, packing_coeff,
                                       packing_coeff_cutoff, vdw_cutoff,
                                       cell_params_threshold, rdf_dist_threshold,
                                       )

        return {'vdw_potential': vdw_potential.numpy(), 'rdf': rdf.numpy(), 'samples': samples.numpy(),
                'packing_coeff': packing_coeff.numpy(), 'rdf_distmat': rdf_distmat}

    def do_sampling(self, batch_size, cc_orientation,
                    init_sample, sample_source,
                    sg_to_sample, tot_samples,
                    show_tqdm=True, variation_factor=None):
        num_batches = tot_samples // batch_size + 1
        sample_record, vdw_potential_record, packing_coeff_record, rdf_record = [], [], [], []
        for _ in tqdm(range(num_batches), disable=not show_tqdm):
            samples, vdw_potential, packing_coeff, rdf, rr = self.sample_iter(
                init_sample, batch_size, cc_orientation,
                sg_to_sample, sample_source, variation_factor)
            sample_record.append(samples.cpu())
            vdw_potential_record.append(vdw_potential.cpu())
            packing_coeff_record.append(packing_coeff.cpu())
            rdf_record.append(rdf.cpu())

        return (torch.cat(sample_record),
                torch.cat(vdw_potential_record),
                torch.cat(packing_coeff_record),
                torch.cat(rdf_record),
                rr.cpu()
                )

    def sample_iter(self, init_sample, batch_size, cc_orientation, sg_to_sample, sample_source, variation_factor=None):
        data = self.collater([init_sample for _ in range(batch_size)]).to(self.device)

        mol_data, scalar_mol_embedding, vector_mol_embedding = self.process_embed_for_generator(
            data,
            cc_orientation,
            sg_to_sample,
            skip_embedding=sample_source == 'csd_prior'
        )

        if sample_source == 'csd_prior':
            generated_sample, descaled_generated_sample = self.sample_from_prior(mol_data)
        elif sample_source == 'generator':
            generated_sample, descaled_generated_sample = self.sample_from_generator(
                mol_data, scalar_mol_embedding, vector_mol_embedding, variation_factor)
        else:
            assert False, f"sample_source {sample_source} not supported"

        supercell_data, generated_cell_volumes = (
            self.supercell_builder.build_zp1_supercells(
                molecule_data=mol_data,
                cell_parameters=descaled_generated_sample,
                supercell_size=self.supercell_size,
                graph_convolution_cutoff=self.cutoff,
                align_to_standardized_orientation=False,
                skip_refeaturization=False,
            ))

        (eval_vdw_loss, vdw_loss, vdw_potential,
         molwise_normed_overlap, prior_loss,
         packing_coeff, scaled_deviation,
         dist_dict) = self.analyze_generated_crystals(
            mol_data,
            generated_cell_volumes,
            generated_sample,
            supercell_data,
            return_dist_dict=True
        )
        sample_to_compare = generated_sample.clone()
        sample_to_compare[:, 9:] = supercell_data.cell_params[:, 9:]

        rdf, rr, _ = new_crystal_rdf(supercell_data, dist_dict,
                                     rrange=[0, 6], bins=2000,
                                     mode='intermolecular', atomwise=True,
                                     raw_density=True, cpu_detach=False,
                                     atomic_numbers_override=mol_data.x.unique().long())

        return sample_to_compare.detach(), (
                    vdw_potential / mol_data.num_atoms).detach(), packing_coeff.detach(), rdf.detach(), rr.detach()

    def sample_from_prior(self, mol_data) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_samples = self.prior(mol_data.num_graphs, mol_data.sg_ind).to(self.device)

        samples_to_build = denormalize_generated_cell_params(
            raw_samples,
            mol_data,
            self.supercell_builder.asym_unit_dict
        )

        return raw_samples, samples_to_build

    def sample_from_generator(self, mol_data,
                              scalar_mol_embedding,
                              vector_mol_embedding,
                              variation_factor) -> Tuple[torch.Tensor, torch.Tensor]:
        prior, descaled_prior = self.sample_from_prior(mol_data)

        variation_tensor = variation_factor * torch.ones(scalar_mol_embedding.shape[0], device=self.device)
        scaling_factor = (self.prior.norm_factors[mol_data.sg_ind, :] + 1e-4)

        absolute_reference = torch.eye(3, dtype=torch.float32, device=self.device
                                       ).reshape(1, 9).repeat(mol_data.num_graphs, 1)

        conditioning_vector = torch.cat((
            scalar_mol_embedding,
            prior,
            absolute_reference,
            variation_tensor[:, None],
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

    def process_embed_for_generator(self, data, orientation, generate_sgs, skip_embedding=False):
        # clone, center, and orient the molecules
        _, mol_data = self.preprocess_ae_inputs(
            data,
            orientation_override=orientation)

        mol_data = overwrite_symmetry_info(mol_data,
                                           generate_sgs,
                                           self.sym_info,
                                           randomize_sgs=True)

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
                                          right_handed=False,
                                          include_inversion=True)

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

    def analyze_generated_crystals(self,
                                   mol_data,
                                   generated_cell_volumes,
                                   generator_raw_samples,
                                   supercell_data,
                                   variation_factor=None,
                                   prior=None,
                                   return_dist_dict=False,
                                   vdw_turnover_potential=10):
        if prior is not None:
            prior_loss, scaled_deviation = compute_prior_loss(
                self.prior.norm_factors,
                mol_data.sg_ind,
                generator_raw_samples,
                prior,
                variation_factor)
        else:
            prior_loss = torch.zeros_like(generated_cell_volumes)
            scaled_deviation = torch.zeros_like(generated_cell_volumes)

        reduced_volume = generated_cell_volumes / supercell_data.sym_mult
        packing_coeff = mol_data.mol_volume / reduced_volume

        dist_dict = get_intermolecular_dists_dict(supercell_data, self.cutoff, 100)
        molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, eval_vdw_loss \
            = vdw_analysis(self.vdw_radii_tensor, dist_dict, mol_data.num_graphs, vdw_turnover_potential)

        if return_dist_dict:
            return (eval_vdw_loss, vdw_loss, vdw_potential,
                    molwise_normed_overlap, prior_loss,
                    packing_coeff, scaled_deviation, dist_dict)

        else:
            return (eval_vdw_loss, vdw_loss, vdw_potential,
                    molwise_normed_overlap, prior_loss,
                    packing_coeff, scaled_deviation)




'''old code'''

#
# def crystal_search(self, molecule_data, batch_size=None, data_contains_ground_truth=True):  # currently deprecated
#     """
#     execute a search for a single crystal target
#     if the target is known, compare it to our best guesses
#     """
#     self.source_directory = os.getcwd()
#     self.prep_new_working_directory()
#
#     with wandb.init(config=self.config,
#                     project=self.config.wandb.project_name,
#                     entity=self.config.wandb.username,
#                     tags=[self.config.logger.experiment_tag],
#                     settings=wandb.Settings(code_dir=".")):
#
#         wandb.run.name = self.config.machine + '_' + self.config.mode + '_' + self.working_directory  # overwrite procedurally generated run name with our run name
#
#         if batch_size is None:
#             batch_size = self.config.min_batch_size
#
#         num_discriminator_opt_steps = 100
#         num_mcmc_opt_steps = 100
#         max_iters = 10
#
#         self.init_gaussian_generator()
#         self.initialize_models_optimizers_schedulers()
#
#         self.models_dict['generator'].eval()
#         self.models_dict['regressor'].eval()
#         self.models_dict['discriminator'].eval()
#
#         '''instantiate batch'''
#         crystaldata_batch = self.collater([molecule_data for _ in range(batch_size)]).to(self.device)
#         refresh_inds = torch.arange(batch_size)
#         converged_samples_list = []
#         optimization_trajectories = []
#
#         for opt_iter in range(max_iters):
#             crystaldata_batch = self.refresh_crystal_batch(crystaldata_batch, refresh_inds=refresh_inds)
#
#             crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(
#                 crystaldata_batch,
#                 mode='mcmc',
#                 num_steps=num_mcmc_opt_steps,
#                 temperature=0.05,
#                 step_size=0.01)
#             optimization_trajectories.append(opt_traj)
#
#             crystaldata_batch, opt_traj = self.optimize_crystaldata_batch(
#                 crystaldata_batch,
#                 mode='discriminator',
#                 num_steps=num_discriminator_opt_steps)
#             optimization_trajectories.append(opt_traj)
#
#             crystaldata_batch, refresh_inds, converged_samples = self.prune_crystaldata_batch(crystaldata_batch,
#                                                                                               optimization_trajectories)
#
#             converged_samples_list.extend(converged_samples)
#
#         aa = 1
#         # do clustering
#
#         # compare to ground truth
#         # add convergence flags based on completeness of sampling
#
#         # '''compare samples to ground truth'''
#         # if data_contains_ground_truth:
#         #     ground_truth_analysis = self.analyze_real_crystal(molecule_data)
#         #

# def prune_crystaldata_batch(self, crystaldata_batch, optimization_trajectories):
#     """
#     Identify trajectories which have converged.
#     """
#
#     """
#     combined_traj_dict = {key: np.concatenate(
#         [traj[key] for traj in optimization_trajectories], axis=0)
#         for key in optimization_trajectories[1].keys()}
#
#     from plotly.subplots import make_subplots
#     import plotly.graph_objects as go
#
#     from plotly.subplots import make_subplots
#     import plotly.graph_objects as go
#     fig = make_subplots(cols=3, rows=1, subplot_titles=['score','vdw_score','packing_coeff'])
#     for i in range(crystaldata_batch.num_graphs):
#         for j, key in enumerate(['score','vdw_score','packing_coeff']):
#             col = j % 3 + 1
#             row = j // 3 + 1
#             fig.add_scattergl(y=combined_traj_dict[key][:, i], name=i, legendgroup=i, showlegend=True if j == 0 else False, row=row, col=col)
#     fig.show(renderer='browser')
#
#     """
#
#     refresh_inds = np.arange(crystaldata_batch.num_graphs)  # todo write a function that actually checks for this
#     converged_samples = [crystaldata_batch[i] for i in refresh_inds.tolist()]
#
#     return crystaldata_batch, refresh_inds, converged_samples

# def optimize_crystaldata_batch(self, batch, mode, num_steps, temperature=None, step_size=None):  # DEPRECATED todo redevelop
#     """
#     method which takes a batch of crystaldata objects
#     and optimzies them according to a score model either
#     with MCMC or gradient descent
#     """
#     if mode.lower() == 'mcmc':
#         sampling_dict = mcmc_sampling(
#             self.models_dict['discriminator'], batch,
#             self.supercell_builder,
#             num_steps, self.vdw_radii,
#             supercell_size=5, cutoff=6,
#             sampling_temperature=temperature,
#             lattice_means=self.dataDims['lattice_means'],
#             lattice_stds=self.dataDims['lattice_stds'],
#             step_size=step_size,
#         )
#     elif mode.lower() == 'discriminator':
#         sampling_dict = gradient_descent_sampling(
#             self.models_dict['discriminator'], batch,
#             self.supercell_builder,
#             num_steps, 1e-3,
#             torch.optim.Rprop, self.vdw_radii,
#             lattice_means=self.dataDims['lattice_means'],
#             lattice_stds=self.dataDims['lattice_stds'],
#             supercell_size=5, cutoff=6,
#         )
#     else:
#         assert False, f"{mode.lower()} is not a valid sampling mode!"
#
#     '''return best sample'''
#     best_inds = np.argmax(sampling_dict['score'], axis=0)
#     best_samples = sampling_dict['std_cell_params'][best_inds, np.arange(batch.num_graphs), :]
#     supercell_data, _ = \
#         self.supercell_builder.build_zp1_supercells(
#             batch, torch.tensor(best_samples, dtype=torch.float32, device=batch.x.device),
#             5, 6,
#             align_to_standardized_orientation=True,
#             target_handedness=batch.aunit_handedness)
#
#     output, proposed_dist_dict = self.models_dict['discriminator'](supercell_data.clone().cuda(), return_dists=True)
#
#     rebuilt_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()
#
#     cell_params_difference = np.amax(
#         np.sum(np.abs(supercell_data.cell_params.cpu().detach().numpy() - best_samples), axis=1))
#     rebuilt_scores_difference = np.amax(np.abs(rebuilt_sample_scores - sampling_dict['score'].max(0)))
#
#     if rebuilt_scores_difference > 1e-2 or cell_params_difference > 1e-2:
#         aa = 1
#         assert False, "Best cell rebuilding failed!"  # confirm we rebuilt the cells correctly
#
#     sampling_dict['best_samples'] = best_samples
#     sampling_dict['best_scores'] = sampling_dict['score'].max(0)
#     sampling_dict['best_vdws'] = np.diag(sampling_dict['vdw_score'][best_inds, :])
#
#     best_batch = batch.clone()
#     best_batch.cell_params = torch.tensor(best_samples, dtype=torch.float32, device=supercell_data.x.device)
#
#     return best_batch, sampling_dict
#
# def refresh_crystal_batch(self, crystaldata, refresh_inds, generator='gaussian', space_groups: torch.tensor = None):
#     # crystaldata = self.set_molecule_alignment(crystaldata, right_handed=False, mode_override=mol_orientation)
#
#     if space_groups is not None:
#         crystaldata.sg_ind = space_groups
#
#     if generator == 'gaussian':
#         samples = self.gaussian_generator.forward(crystaldata.num_graphs, crystaldata).to(self.config.device)
#         crystaldata.cell_params = samples[refresh_inds]
#         # todo add option for generator here
#
#     return crystaldata
