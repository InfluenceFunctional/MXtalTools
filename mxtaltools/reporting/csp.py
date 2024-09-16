import plotly.graph_objects as go
import numpy as np
import torch
from typing import Union
import plotly.express.colors as pc
from plotly.subplots import make_subplots


def single_property_distribution(y: Union[np.ndarray, torch.Tensor],
                                 xaxis_title: str,
                                 bandwidth=None,
                                 xaxis_range=None):
    if torch.is_tensor(y):
        y = y.cpu().detach().numpy()

    fig = go.Figure()
    if bandwidth is None:
        bandwidth = float(np.ptp(y) / 200)

    fig.add_trace(go.Violin(
        x=y, side='positive', orientation='h',
        width=1,
        name=f'Mean = {np.mean(y):.2f}',
        meanline_visible=True,
        bandwidth=bandwidth,
        points=False,
    ))
    fig.update_layout(xaxis_title=xaxis_title)

    if xaxis_range is not None:
        fig.update_layout(xaxis_range=xaxis_range)

    return fig


def stacked_property_distribution(y: Union[np.ndarray, torch.Tensor],
                                  xaxis_title: str,
                                  yaxis_title: str,
                                  bandwidth=None,
                                  xaxis_range=None
                                  ):
    if torch.is_tensor(y):
        y = y.cpu().detach().numpy()

    if y.ndim == 1:
        y = y[np.newaxis, :]
        num_stacks = 1
    else:
        num_stacks = y.shape[0]

    colors = pc.n_colors('rgb(255, 150, 50)', 'rgb(0, 25, 250)', max(2, num_stacks), colortype='rgb')

    fig = go.Figure()
    if bandwidth is None:
        bandwidth = float(np.ptp(y.flatten()) / 200)

    for ind in range(num_stacks):
        fig.add_trace(go.Violin(
            x=y[ind, :], y=ind, side='positive', orientation='h',
            width=1,
            name=yaxis_title + f'Mean = {np.mean(y[ind, :]):.2f}',
            showlegend=True,
            meanline_visible=True,
            bandwidth=bandwidth,
            points=False,
            line_color=colors[ind]
        ))
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    if xaxis_range is not None:
        fig.update_xaxes(range=xaxis_range)

    fig.update_traces(opacity=0.5)

    return fig


def stacked_property_distribution_lists(y: list,
                                        xaxis_title: str,
                                        yaxis_title: str,
                                        bandwidth=None,
                                        xaxis_range=None
                                        ):
    num_stacks = len(y)

    colors = pc.n_colors('rgb(255, 150, 50)', 'rgb(0, 25, 250)', max(2, num_stacks), colortype='rgb')

    fig = go.Figure()
    if bandwidth is None:
        bandwidth = float(np.ptp(np.concatenate(y)) / 200)

    for ind in range(num_stacks):
        fig.add_trace(go.Violin(
            x=np.array(y[ind]), y=ind * np.ones_like(y[ind]), side='positive', orientation='h',
            width=4,
            name=yaxis_title + f' {ind} Mean {xaxis_title} = {float(y[ind].mean()):.2f}',
            showlegend=True,
            meanline_visible=True,
            bandwidth=bandwidth,
            points=False,
            line_color=colors[ind]
        ))
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    if xaxis_range is not None:
        fig.update_xaxes(range=xaxis_range)

    fig.update_traces(opacity=0.5)

    return fig


def stacked_cell_distributions(samples: Union[np.ndarray, torch.Tensor],
                               xaxis_title: str,
                               yaxis_title: str,
                               ):
    lattice_features = ['cell_a', 'cell_b', 'cell_c', 'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y',
                        'aunit_z', 'aunit_theta', 'aunit_phi', 'aunit_r']
    n_crystal_features = 12

    if torch.is_tensor(samples):
        samples = samples.cpu().detach().numpy()

    if samples.ndim == 2:
        samples = samples[np.newaxis, ...]
        num_stacks = 1
    else:
        num_stacks = samples.shape[0]

    colors = pc.n_colors('rgb(255, 150, 50)', 'rgb(0, 25, 250)', max(num_stacks, 2), colortype='rgb')
    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        for t_ind in range(num_stacks):
            values = samples[t_ind, :, i].flatten()

            fig.add_trace(go.Violin(
                x=values, y=t_ind * np.ones_like(values), side='positive', orientation='h', width=6,
                name=f'{len(values)} samples at {t_ind}',
                legendgroup=f'{len(values)} samples at {t_ind}',
                showlegend=True if i == 0 else False,
                meanline_visible=True,
                bandwidth=float(np.ptp(values) / 100),
                points=False,
                line_color=colors[t_ind],
            ),
                row=row, col=col
            )

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
    fig.update_traces(opacity=0.5)

    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title)

    return fig


def prior_vs_rdf_dist(prior_to_compare, rdf_dists):
    params_dists = torch.cdist(prior_to_compare, prior_to_compare)
    # for ind in range(len(params_dists)):
    #     params_dists[ind,ind] = torch.nan
    fig = go.Figure(
        go.Scatter(x=params_dists.cpu().detach().flatten(), y=torch.log10(1 + rdf_dists.cpu().detach().flatten()),
                   mode='markers'))
    fig.update_layout(xaxis_title='params dist', yaxis_title='RDF dist')

    return fig


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
