import numpy as np
import tqdm

from mxtaltools.crystal_building.utils import clean_cell_params
from mxtaltools.models.utils import softmax_and_score
from mxtaltools.models.vdw_overlap import vdw_overlap
import torch.optim as optim
import torch


def gradient_descent_sampling(discriminator, crystal_batch, supercell_builder,
                              num_steps, lr, optimizer_func, vdw_radii,
                              lattice_means, lattice_stds,
                              supercell_size=5, cutoff=6):
    """
    for a given sample
    1) generate a score from a discriminator model
    2) backpropagate the score as a loss to the original cell parameters
    3) compute and apply the gradient on the parameters
    4) generate updated sample and repeat

    Parameters
    ----------
    discriminator
    d_optimizer
    init_samples
    single_mol_data
    supercell_builder
    num_steps
    supercell_size
    cutoff
    generate_sgs

    Returns
    -------
    sampling dict containing scores and samples

    """

    sample = torch.tensor(crystal_batch.cell_params.clone(), device=crystal_batch.x.device, requires_grad=True, dtype=torch.float32)
    optimizer = optimizer_func([sample], lr=lr)

    max_lr_target_time = num_steps // 10
    max_lr = 1e-2
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)

    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False

    n_samples = len(sample)

    scores_record = np.zeros((num_steps, n_samples))
    samples_record = np.zeros((num_steps, n_samples, 12))
    loss_record = np.zeros_like(scores_record)
    vdw_record = np.zeros_like(scores_record)
    lr_record = np.zeros(num_steps)
    packing_record = np.zeros_like(scores_record)

    discriminator.eval()
    with torch.enable_grad():
        for s_ind in tqdm.tqdm(range(num_steps), miniters=int(num_steps / 25)):
            optimizer.zero_grad()

            cleaned_sample = clean_cell_params(sample, crystal_batch.sg_ind,
                                               lattice_means, lattice_stds,
                                               supercell_builder.symmetries_dict, supercell_builder.asym_unit_dict,
                                               rescale_asymmetric_unit=True, destandardize=False, mode='hard' if s_ind == 0 else 'soft',
                                               fractional_basis='unit_cell'
                                               )

            supercell_data, cell_volumes = \
                supercell_builder.build_supercells(
                    crystal_batch, cleaned_sample,
                    supercell_size, cutoff,
                    align_to_standardized_orientation=True,
                    target_handedness=crystal_batch.asym_unit_handedness)

            output, dist_dict = discriminator(supercell_data.clone().cuda(), return_dists=True)

            score = softmax_and_score(output[:, :2])
            loss = -score

            vdw_record[s_ind] = vdw_overlap(vdw_radii,
                                            dist_dict=dist_dict['dists_dict'],
                                            num_graphs=crystal_batch.num_graphs,
                                            return_score_only=True,
                                            graph_sizes=supercell_data.mol_size).cpu().detach().numpy()

            scores_record[s_ind] = score.cpu().detach().numpy()
            samples_record[s_ind] = supercell_data.cell_params.cpu().detach().numpy()
            loss_record[s_ind] = loss.cpu().detach().numpy()
            packing_record[s_ind] = (supercell_data.mult * supercell_data.mol_volume / cell_volumes).cpu().detach().numpy()

            loss.mean().backward()  # compute gradients
            optimizer.step()  # apply grad

            lr = optimizer.param_groups[0]['lr']
            lr_record[s_ind] = lr
            if lr >= max_lr:
                hit_max_lr = True
            if hit_max_lr:
                if lr > 1e-5:
                    scheduler1.step()  # shrink
            else:
                scheduler2.step()  # grow

    sampling_dict = {'std_cell_params': samples_record, 'score': scores_record,
                     'vdw_score': vdw_record, 'space_group': supercell_data.sg_ind.cpu().detach().numpy(),
                     'packing_coeff': packing_record}

    return sampling_dict


"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = make_subplots(cols=2,rows=2,subplot_titles=['score','vdw','packing_coeff','lr'])
fig.add_scattergl(y=scores_record.mean(1), line_width=8, line_color='black', showlegend=False,row=1,col=1)
fig.add_scattergl(y=vdw_record.mean(1), line_width=8, line_color='black', showlegend=False,row=1,col=2)
fig.add_scattergl(y=packing_record.mean(1), line_width=8, line_color='black', showlegend=False,row=2,col=1)
fig.add_scattergl(y=np.log10(lr_record), line_width=8, line_color='black', showlegend=False,row=2,col=2)
for i in range(scores_record.shape[1]):
    fig.add_scattergl(y=scores_record[:,i], name=i, legendgroup=i, showlegend=True,row=1,col=1)
    fig.add_scattergl(y=vdw_record[:,i], name=i, legendgroup=i, showlegend=False,row=1,col=2)
    fig.add_scattergl(y=packing_record[:,i], name=i, legendgroup=i, showlegend=False,row=2,col=1)

fig.show(renderer='browser')


"""


def mcmc_sampling(discriminator, crystal_batch, supercell_builder,
                  num_steps, vdw_radii, supercell_size, cutoff,
                  sampling_temperature, lattice_means, lattice_stds, step_size,
                  ):

    samples_record = np.zeros((num_steps, crystal_batch.num_graphs, 12))
    scores_record = np.zeros((num_steps, crystal_batch.num_graphs))
    vdw_record = np.zeros_like(scores_record)
    packing_record = np.zeros_like(scores_record)

    alpha_randoms = np.random.uniform(0, 1, size=(num_steps, crystal_batch.num_graphs))
    propose_randoms = step_size * (np.random.randn(num_steps, crystal_batch.num_graphs, 12) * lattice_stds + lattice_means)

    with torch.no_grad():
        for s_ind in tqdm.tqdm(range(num_steps), miniters=int(num_steps / 25)):  # sample for a certain number of iterations
            if s_ind != 0:
                proposed_samples = torch.tensor(np.copy(samples_record[s_ind - 1]) + propose_randoms[s_ind],
                                                dtype=torch.float32, device=crystal_batch.x.device)

                cleaned_proposed_samples = clean_cell_params(proposed_samples, crystal_batch.sg_ind,
                                                             lattice_means, lattice_stds,
                                                             supercell_builder.symmetries_dict, supercell_builder.asym_unit_dict,
                                                             rescale_asymmetric_unit=True, destandardize=False, mode='soft',
                                                             fractional_basis='unit_cell'
                                                             )

                proposed_crystals, cell_volumes = \
                    supercell_builder.build_supercells(
                        crystal_batch, cleaned_proposed_samples,
                        supercell_size, cutoff,
                        align_to_standardized_orientation=True,
                        target_handedness=crystal_batch.asym_unit_handedness)

                output, proposed_dist_dict = discriminator(proposed_crystals.clone().cuda(), return_dists=True)

                proposed_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()

                proposed_sample_vdws = vdw_overlap(vdw_radii,
                                                   dist_dict=proposed_dist_dict['dists_dict'],
                                                   num_graphs=crystal_batch.num_graphs,
                                                   return_score_only=True,
                                                   graph_sizes=proposed_crystals.mol_size).cpu().detach().numpy()

                score_difference = scores_record[s_ind - 1] - proposed_sample_scores
                acceptance_ratio = np.minimum(
                    1,
                    np.exp(-score_difference / sampling_temperature)
                )
                accept_flags = alpha_randoms[s_ind] < acceptance_ratio

                packing_coeffs = (proposed_crystals.mult * proposed_crystals.mol_volume / cell_volumes).cpu().detach().numpy()

                samples_record[s_ind] = samples_record[s_ind - 1]
                samples_record[s_ind, accept_flags] = proposed_crystals.cell_params[accept_flags].cpu().detach().numpy()
                scores_record[s_ind] = scores_record[s_ind - 1]
                scores_record[s_ind, accept_flags] = proposed_sample_scores[accept_flags]
                vdw_record[s_ind] = vdw_record[s_ind - 1]
                vdw_record[s_ind, accept_flags] = proposed_sample_vdws[accept_flags]
                packing_record[s_ind] = packing_record[s_ind - 1]
                packing_record[s_ind, accept_flags] = packing_coeffs[accept_flags]
            else:
                proposed_samples = crystal_batch.cell_params

                proposed_samples = clean_cell_params(proposed_samples, crystal_batch.sg_ind,
                                                     lattice_means, lattice_stds,
                                                     supercell_builder.symmetries_dict, supercell_builder.asym_unit_dict,
                                                     rescale_asymmetric_unit=True, destandardize=False, mode='hard',
                                                     fractional_basis='unit_cell'
                                                     )

                proposed_crystals, cell_volumes = \
                    supercell_builder.build_supercells(
                        crystal_batch, proposed_samples,
                        supercell_size, cutoff,
                        align_to_standardized_orientation=True,
                        target_handedness=crystal_batch.asym_unit_handedness)

                output, proposed_dist_dict = discriminator(proposed_crystals.clone().cuda(), return_dists=True)

                proposed_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()

                proposed_sample_vdws = vdw_overlap(vdw_radii,
                                                   dist_dict=proposed_dist_dict['dists_dict'],
                                                   num_graphs=crystal_batch.num_graphs,
                                                   return_score_only=True,
                                                   graph_sizes=proposed_crystals.mol_size).cpu().detach().numpy()

                packing_coeffs = (proposed_crystals.mult * proposed_crystals.mol_volume / cell_volumes).cpu().detach().numpy()

                samples_record[s_ind] = proposed_crystals.cell_params.cpu().detach().numpy()
                scores_record[s_ind] = proposed_sample_scores
                vdw_record[s_ind] = proposed_sample_vdws
                packing_record[s_ind] = packing_coeffs

    sampling_dict = {'std_cell_params': samples_record, 'score': scores_record,
                     'vdw_score': vdw_record, 'space_group': crystal_batch.sg_ind.cpu().detach().numpy(),
                     'packing_coeff': packing_record}

    return sampling_dict


"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(cols=2,rows=2,subplot_titles=['score','vdw','packing_coeff','lr'])
fig.add_scattergl(y=scores_record.mean(1), line_width=8, line_color='black', showlegend=False,row=1,col=1)
fig.add_scattergl(y=vdw_record.mean(1), line_width=8, line_color='black', showlegend=False,row=1,col=2)
fig.add_scattergl(y=packing_record.mean(1), line_width=8, line_color='black', showlegend=False,row=2,col=1)
for i in range(scores_record.shape[1]):
    fig.add_scattergl(y=scores_record[:,i], name=i, legendgroup=i, showlegend=True,row=1,col=1)
    fig.add_scattergl(y=vdw_record[:,i], name=i, legendgroup=i, showlegend=False,row=1,col=2)
    fig.add_scattergl(y=packing_record[:,i], name=i, legendgroup=i, showlegend=False,row=2,col=1)

fig.show(renderer='browser')


"""
