import copy
import gc
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import optim
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.geometry_utils import enforce_crystal_system
from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import enforce_1d_bound, softmax_and_score


def get_annealing_factor(start_value, stop_value, total_time, step_iters):
    assert stop_value > 0, "Setting final value as zero breaks this module"
    return (stop_value / start_value) ** (1 / (total_time / step_iters))


class CrystalParams(nn.Module):
    def __init__(self, init_sample):
        super().__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(row.clone().detach()) for row in init_sample]
        )

    def stacked(self):
        return torch.stack(list(self.params), dim=0)

    def update_params(self, new_values, mask=None):
        """
        new_values: [batch_size, param_dim] tensor
        mask: optional boolean mask [batch_size] to only update accepted samples
        """
        with torch.no_grad():
            if mask is None:
                for i, val in enumerate(new_values):
                    self.params[i].copy_(val)
            else:
                # Only update the specific batch indices that were accepted
                indices = torch.where(mask)[0]
                for idx in indices:
                    self.params[idx].copy_(new_values[idx])


def gradient_descent_optimization(
        init_sample: torch.Tensor,
        init_crystal_batch,
        optimizer_func: Union[torch.optim.Optimizer, str],
        init_lr: Optional[float] = 1e-1,
        max_num_steps: Optional[int] = 500,
        show_tqdm: Optional[bool] = False,
        convergence_eps: float = 1e-3,
        grad_norm_clip: Optional[float] = 0.1,
        optim_target: Optional[str] = 'lj',
        score_model: Optional[torch.nn.Module] = None,
        target_packing_coeff: Optional[torch.Tensor] = None,
        do_box_restriction: Optional[bool] = False,
        cutoff: Optional[float] = None,
        compression_factor: Optional[float] = 0,
        enforce_reduced: Optional[bool] = False,
        supercell_size: Optional[int] = 10,
        anneal_lr: Optional[bool] = False,
        uma_predictor: Optional = None,
        mc_kT: float = 0.5,
        log_noise_range=[-2, -1]
):
    """
    do a local optimization via gradient descent on some score function
    """  # todo implement wrapping over periodic latent DoF

    if cutoff is None:
        # lennard jones need 10 angstroms to nicely converge
        cutoff = 10

    energy_computes = ['lj']
    min_num_steps = 50
    num_samples = init_crystal_batch.num_graphs

    if optim_target.lower() == 'silu':
        energy_computes.append('silu')
    elif optim_target.lower() == 'qlj':
        energy_computes.append('qlj')
    elif optim_target.lower() == 'elj':
        energy_computes.append('elj')
    elif optim_target.lower() == 'ellipsoid':
        energy_computes.append('ellipsoid')
    elif optim_target.lower() == 'reduce':
        energy_computes.append('reduction_en')
    elif optim_target.lower() == 'uma':
        energy_computes.append('uma')

    if enforce_reduced:  # use an energy based penalty to enforce sampling in the reduced subspace
        energy_computes.append('reduction_en')
        std_margin = 0.01
    else:
        std_margin = 0

    param_module = CrystalParams(init_sample)  # <-- init_sample shape [n, 6 + max_z_prime * 6]

    optimizer = init_opt(init_lr, optimizer_func, param_module)
    monte_carlo = optimizer_func.lower() == 'mcmc'
    if monte_carlo:
        # We need a starting energy for the very first comparison
        with torch.no_grad():
            T, current_energy, current_state = init_mc_state(compression_factor, cutoff, do_box_restriction,
                                                              energy_computes, enforce_reduced, init_crystal_batch,
                                                              mc_kT, optim_target, param_module, score_model,
                                                              std_margin, supercell_size, target_packing_coeff,
                                                              uma_predictor)

    if anneal_lr:
        lr_factor = get_annealing_factor(1, 0.01, 500, 1)
    else:
        lr_factor = 1
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                     lr_lambda=lambda epoch: lr_factor)

    params_record = torch.zeros((max_num_steps, init_crystal_batch.num_graphs, init_sample.shape[-1]),
                                dtype=torch.float32)

    if optim_target.lower() == 'ellipsoid':
        init_crystal_batch.load_ellipsoid_model()
        ellipsoid_model = copy.deepcopy(init_crystal_batch.ellipsoid_model)
        ellipsoid_model = ellipsoid_model.to(init_crystal_batch.device)
        ellipsoid_model.eval()  # unify initialization of this vs other score models, uma models, etc.

    records = None
    converged = torch.zeros(init_crystal_batch.num_graphs, dtype=torch.bool)
    # torch.autograd.set_detect_anomaly(True)  # for debugging
    try:
        with torch.set_grad_enabled(not monte_carlo):
            with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
                s_ind = 0
                while (s_ind < (max_num_steps - 1)) and not converged.all():
                    optimizer.zero_grad(set_to_none=True)
                    crystal_batch = init_crystal_batch.clone().detach()  # this is necessary to not retain lots of intermediate tensors
                    if monte_carlo:
                        crystal_batch.set_cell_parameters(param_module.stacked(), skip_box_analysis=False)
                        samples = crystal_batch.latent_params()
                        rand_dir = torch.randn_like(samples)
                        rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
                        # rand_magnitude = torch.randn(len(samples), device=samples.device).abs() * noise_level
                        u = torch.rand(len(samples), device=samples.device)
                        rand_magnitude = 10 ** (log_noise_range[0] + (log_noise_range[1] - log_noise_range[0]) * u)
                        noised_samples = (samples + rand_dir * rand_magnitude[:, None]).clip(min=-1, max=1)
                        crystal_batch.latent_to_cell_params(noised_samples,
                                                            skip_box_analysis=True,
                                                            skip_enforce_crystal_system=True)
                    else:
                        crystal_batch.set_cell_parameters(param_module.stacked(),
                                                          skip_box_analysis=True)  # we will do box analysis in the next step, after cleanup
                    crystal_batch.clean_cell_parameters(
                        mode='hard',
                        canonicalize_orientations=True,
                    )
                    outputs, cluster_batch = crystal_batch.analyze(
                        computes=energy_computes,
                        cutoff=cutoff,
                        supercell_size=supercell_size,
                        return_cluster=True,
                        repulsion=1,
                        surface_padding=0,
                        std_orientation=True,
                        margin=std_margin,
                        predictor=uma_predictor,
                    )

                    """
                    record some stats
                    """
                    params_record[
                        s_ind] = crystal_batch.full_cell_parameters().detach().cpu()  # must put this before the .backward()
                    if records is None:
                        records = {key: [] for key in outputs}
                        records['cp'] = []
                        records['loss'] = []
                    for key, value in outputs.items():
                        records[key].append(value.detach().cpu())
                    records['cp'].append(crystal_batch.packing_coeff.detach().cpu())

                    """
                    loss and backprop
                    """
                    if monte_carlo:
                        # 1. Compute energy of the proposal (noised_samples)
                        proposal_energy = compute_loss(cluster_batch, crystal_batch, optim_target, outputs, score_model)
                        proposal_energy = compute_auxiliary_loss(cluster_batch, compression_factor,
                                                                 do_box_restriction, proposal_energy,
                                                                 target_packing_coeff, enforce_reduced, outputs)

                        # 2. Calculate Metropolis Criterion
                        # For minimization: P(accept) = exp(-(E_prop - E_curr) / T)
                        delta_e = proposal_energy - current_energy
                        acceptance_prob = torch.exp(-delta_e / T).clamp(max=1.0)

                        # 3. Batch-wise Acceptance
                        accepted = torch.rand_like(acceptance_prob) < acceptance_prob

                        # 4. Update the "Current" State
                        current_energy[accepted] = proposal_energy[accepted]
                        current_state[accepted] = crystal_batch.full_cell_parameters()[accepted]

                        # 5. Sync the param_module so the next iteration starts from the accepted state
                        param_module.update_params(current_state)  # You'll need a setter in your module

                        # Record the energy for your stats
                        records['loss'].append(current_energy.detach().cpu())

                        # 6. Cooling (Simulated Annealing)
                        # T *= cooling_rate
                        del proposal_energy, cluster_batch, outputs, crystal_batch

                    else:
                        loss = compute_loss(cluster_batch, crystal_batch, optim_target, outputs, score_model)
                        loss = compute_auxiliary_loss(cluster_batch, compression_factor,
                                                      do_box_restriction, loss, target_packing_coeff, enforce_reduced,
                                                      outputs)
                        records['loss'].append(loss.detach().cpu())
                        loss_to_backprop = loss[~converged].mean()  # save some effort in backprop
                        loss_to_backprop.backward()
                        torch.nn.utils.clip_grad_norm_(param_module.parameters(), grad_norm_clip)  # gradient clipping
                        optimizer.step()  # apply grad

                        del loss_to_backprop, cluster_batch, outputs, crystal_batch, loss
                    if s_ind % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                    scheduler1.step()  # shrink

                    s_ind += 1
                    if s_ind % 50 == 0:
                        pbar.update(50)
                    if s_ind >= min(max_num_steps, max(50, min_num_steps)):
                        converged = check_convergence(params_record, s_ind, convergence_eps,
                                                      optimizer, init_lr)
    except (RuntimeError, ValueError) as e:
        if is_cuda_oom(e):
            if s_ind > 0:
                records = {k: torch.stack(v) for k, v in records.items()}
                best_sample_ind = torch.argmin(records['loss'],
                                               dim=0).flatten()  # pick the best sample from each trajectory
                best_samples = torch.stack([params_record[best_sample_ind[ind], ind] for ind in range(num_samples)])
                torch.save(best_samples, 'opt_intermediates.pt')
            raise e

    records = {k: torch.stack(v) for k, v in records.items()}
    """
    Pull out and re-analyze the best samples from each trajectory
    """
    best_sample_ind = torch.argmin(records['loss'], dim=0).flatten()  # pick the best sample from each trajectory
    best_samples = torch.stack([params_record[best_sample_ind[ind], ind] for ind in range(num_samples)])
    crystal_batch = init_crystal_batch.clone().detach()  # this is necessary to not retain lots of intermediate tensors
    crystal_batch.set_cell_parameters(best_samples.to(crystal_batch.device),
                                      skip_box_analysis=False)
    outputs, cluster_batch = crystal_batch.analyze(
        computes=energy_computes,
        cutoff=cutoff,
        supercell_size=supercell_size,
        return_cluster=True,
        repulsion=1,
        surface_padding=0,
        predictor=uma_predictor
    )
    crystal_batch.add_graph_attr(outputs['lj'].detach(), 'lj')
    crystal_batch.add_graph_attr(crystal_batch.packing_coeff.detach(), 'packing_coeff')
    samples_list = crystal_batch.batch_to_list()

    if enforce_reduced:
        penalty = crystal_batch.compute_cell_reduction_penalty()
        samples_list = [elem for i, elem in enumerate(samples_list) if penalty[i] < 1e-3]
    """
    # analyze trajectory information, if we want
    
    from mxtaltools.common.utils import log_rescale_positive
    timesteps = torch.arange(s_ind).repeat(init_crystal_batch.num_graphs, 1).T
    traj_fig(timesteps, log_rescale_positive(records['loss']), names=['time', 'loss'])
    traj_fig(timesteps, (records['cp']), names=['time', 'cp'])
    traj_fig(records['cp'], log_rescale_positive(records['lj']), names=['cp', 'lj'])
    
    crystal_batch.plot_batch_cell_params()
    crystal_batch.plot_batch_density_funnel()
    import plotly.graph_objects as go
    penalty = crystal_batch.compute_cell_reduction_penalty()
    go.Figure(go.Histogram(x=(penalty+1e-6).log10().cpu().detach().numpy())).show()
    """
    return samples_list, records


def init_mc_state(compression_factor, cutoff, do_box_restriction, energy_computes, enforce_reduced, init_crystal_batch,
                  mc_kT, optim_target, param_module, score_model, std_margin, supercell_size, target_packing_coeff,
                  uma_predictor):
    # Initialize param_module and get initial energy
    crystal_batch = init_crystal_batch.clone().detach()
    crystal_batch.set_cell_parameters(param_module.stacked())
    outputs, cluster_batch = crystal_batch.analyze(computes=energy_computes,
                                                   cutoff=cutoff,
                                                   supercell_size=supercell_size,
                                                   return_cluster=True,
                                                   repulsion=1,
                                                   surface_padding=0,
                                                   std_orientation=True,
                                                   margin=std_margin,
                                                   predictor=uma_predictor, )
    current_energy = compute_loss(cluster_batch, crystal_batch, optim_target, outputs, score_model)
    current_energy = compute_auxiliary_loss(cluster_batch, compression_factor,
                                            do_box_restriction, current_energy, target_packing_coeff,
                                            enforce_reduced,
                                            outputs)
    current_state = param_module.stacked().clone()
    # Temperature schedule (Initial T should be ~ typical delta energy)
    T = mc_kT
    return T, current_energy, current_state


def init_opt(init_lr, optimizer_func, param_module):
    if isinstance(optimizer_func, str):
        if optimizer_func.lower() == 'sgd':
            optimizer_func = optim.SGD
        elif optimizer_func.lower() == 'adam':
            optimizer_func = optim.Adam
        elif optimizer_func.lower() == 'rprop':
            optimizer_func = optim.Rprop
        elif optimizer_func.lower() == 'rmsprop':
            optimizer_func = optim.RMSprop
        elif optimizer_func.lower() == 'adamw':
            optimizer_func = optim.AdamW
        elif optimizer_func.lower() == 'adadelta':
            optimizer_func = optim.Adadelta
        elif optimizer_func.lower() == 'nadam':
            optimizer_func = optim.Nadam
        elif optimizer_func.lower() == 'adagrad':
            optimizer_func = optim.Adagrad
        elif optimizer_func.lower() == 'adadelta':
            optimizer_func = optim.Adadelta
        elif optimizer_func.lower() == 'asgd':
            optimizer_func = optim.ASGD
        elif optimizer_func.lower() == 'mcmc':
            optimizer_func = optim.SGD  # this is a dummy
        else:
            assert False, "Must pass an optimizer or an implemented string"
    optimizer = optimizer_func(param_module.parameters(), lr=init_lr)
    return optimizer


def traj_fig(x, y, names=[None, None], yrange=None, xrange=None):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=x[0, :], y=y[0, :], mode='markers', marker_size=20, marker_color='grey',
                    name='Initial State')
    fig.add_scatter(x=x[-1, :], y=y[-1, :], mode='markers', marker_size=20, marker_color='black',
                    name='Final State')
    for ind in range(x.shape[1]):
        fig.add_scatter(x=x[:, ind], y=y[:, ind], name=f"Run {ind}", opacity=0.5)
    fig.update_layout(xaxis_title=names[0], yaxis_title=names[1], xaxis_range=xrange, yaxis_range=yrange)
    fig.show()


"""

# params trajectories
from plotly.subplots import make_subplots
fig = make_subplots(rows=4, cols=3, subplot_titles = ['a','b','c','al','be','ga','x','y','z','u','v','w'])
for ind in range(12):
    row = ind // 3 + 1
    col = ind % 3 + 1
    for ind2 in range(params_record.shape[1]):
        fig.add_scatter(y=params_record[:s_ind, ind2, ind], showlegend=False, row=row, col=col)
fig.show()

"""


def ema_trajectory(traj: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Vectorized EMA along time (dim=0).
    traj: [T, N, D]
    """
    T = traj.size(0)
    w = (1 - alpha) ** torch.arange(T, device=traj.device, dtype=traj.dtype)  # [T]
    w = w.flip(0).view(T, 1, 1)  # decay weights

    # weighted cumulative sum
    numer = torch.cumsum(alpha * traj * w, dim=0)
    denom = torch.cumsum(w, dim=0)

    return numer / denom


def compute_loss(cluster_batch, crystal_batch, optim_target, outputs, score_model):
    if optim_target.lower() == 'lj':  # todo obviate this with analysis keys
        loss = outputs['lj']

    elif optim_target.lower() == 'silu':
        loss = outputs['silu']

    elif optim_target.lower() == 'qlj':
        loss = outputs['qlj']

    elif optim_target.lower() == 'elj':
        loss = outputs['elj']

    elif optim_target.lower() == 'inter_overlaps':  # force molecules apart by separating their centroids
        loss = inter_overlap_loss(cluster_batch, crystal_batch)

    elif optim_target.lower() == 'classification_score':
        loss = -softmax_and_score(score_model(cluster_batch, force_edges_rebuild=False)[:, :2])

    elif optim_target.lower() == 'rdf_score':
        loss = F.softplus(score_model(cluster_batch, force_edges_rebuild=False)[:, 2])

    elif optim_target.lower() == 'ellipsoid':
        loss = outputs['ellipsoid']

    elif optim_target.lower() == 'reduce':
        loss = outputs['reduction_en']

    elif optim_target.lower() == 'uma':
        loss = outputs['uma'] * 96.485

    return loss


def compute_auxiliary_loss(cluster_batch, compression_factor, do_box_restriction,
                           loss, target_packing_coeff, enforce_reduced, outputs):
    if target_packing_coeff is not None:
        cp_loss = (cluster_batch.packing_coeff - target_packing_coeff) ** 2
        loss = loss + cp_loss
    if do_box_restriction:
        # enforce box shape cannot become too long (3 mol radii) or narrow (3 angstroms) in any direction
        # repulsive from about range 3 #(80000/aunit_lengths**12 + 10*aunit_lengths - 31.25).sum(dim=1)  # forces boxes to be larger than 3 angstroms, but squeezes them otherwise
        aunit_lengths = cluster_batch.scale_lengths_to_aunit()
        box_loss = (80000 / aunit_lengths ** 12).sum(dim=1)
        loss = loss + box_loss
    if compression_factor != 0:
        aunit_lengths = cluster_batch.scale_lengths_to_aunit()
        loss = loss + aunit_lengths.sum(dim=1) * compression_factor

    if enforce_reduced:
        penalty = F.relu(outputs['reduction_en'])
        loss = loss + 10000 * penalty  # severely punish reduction violations

    return loss


def inter_overlap_loss(cluster_batch):
    # enforce molecules far enough away that they cannot possibly overlap
    # intermolecular centroid range repulsion
    assert cluster_batch.z_prime.amax() == 1, "molwise business not implemented for Z'>1"
    _, atoms_per_cluster = torch.unique(cluster_batch.batch, return_counts=True)
    mols_per_cluster = (atoms_per_cluster / cluster_batch.num_atoms).long()
    molwise_batch = torch.arange(cluster_batch.num_graphs, device=cluster_batch.device).repeat_interleave(
        mols_per_cluster,
        dim=0)
    flat_mol_inds = cluster_batch.mol_ind + torch.cat(
        [torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
         torch.cumsum(mols_per_cluster, dim=0)]).long()[:-1].repeat_interleave(
        torch.diff(cluster_batch.ptr), dim=0)
    mol_centroids = scatter(cluster_batch.pos, flat_mol_inds, reduce='mean', dim=0, dim_size=flat_mol_inds[-1] + 1)
    edge_i, edge_j = gnn.radius_graph(
        x=mol_centroids,
        batch=molwise_batch,
        max_num_neighbors=10,
        r=float(cluster_batch.radius.amax() * 2))
    inter_dists = torch.linalg.norm(
        mol_centroids[edge_i] - mol_centroids[edge_j], dim=1
    )
    scaled_inter_dists = inter_dists / cluster_batch.radius[molwise_batch[edge_i]]
    edgewise_losses = F.relu(-(scaled_inter_dists - 2))  # push molecules apart if they are within each others' radii
    loss = scatter(edgewise_losses,
                   molwise_batch[edge_i],
                   dim_size=cluster_batch.num_graphs,
                   dim=0,
                   reduce='sum')
    return loss


""" # viz sample distribution

import plotly.graph_objects as go
ljs = torch.tensor([elem.scaled_lj for elem in opt_samples[:len(samples_record[0])]])
ljs2 = torch.tensor([elem.scaled_lj for elem in opt_samples])
fig = go.Figure()
fig.add_histogram(x=ljs, nbinsx=100, histnorm='probability density')
fig.add_histogram(x=ljs2, nbinsx=100, histnorm='probability density')
fig.show()

"""

"""
import plotly.graph_objects as go
fig = go.Figure()
lj_pots = torch.stack(
    [torch.tensor([sample.scaled_lj for sample in sample_list]) for sample_list in samples_record])
es_pots = torch.stack([torch.tensor([sample.es_pot for sample in sample_list]) for sample_list in samples_record])
orig_batch = collate_data_list(opt_samples)
lj_pots = torch.cat([orig_batch.scaled_lj[None, :], lj_pots], dim=0)
es_pots = torch.cat([orig_batch.es_pot[None, :], es_pots], dim=0)
for ind in range(lj_pot.shape[-1]):
    fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
                    showlegend=True if ind == 0 else False)
    fig.add_scatter(y=es_pots[..., ind] * 10, marker_color='red', name='es', legendgroup='es',
                    showlegend=True if ind == 0 else False)
    fig.add_scatter(y=es_pots[..., ind] * 10 + lj_pots[..., ind], marker_color='green', name='combo', legendgroup='combo',
                    showlegend=True if ind == 0 else False)

fig.add_scatter(y=(lj_pots).mean(1), marker_color='black', name='lj', legendgroup='lg',
                showlegend=True if ind == 0 else False)
fig.add_scatter(y=(es_pots * 10).mean(1), marker_color='black', name='es', legendgroup='es',
                showlegend=True if ind == 0 else False)
fig.add_scatter(y=(es_pots * 10 + lj_pots).mean(1), marker_color='black', name='combo', legendgroup='combo',
                showlegend=True if ind == 0 else False)

fig.show()
"""


def _init_for_local_opt(lr, max_num_steps, optimizer_func, sample, num_atoms):
    optimizer = optimizer_func([sample], lr=lr)
    max_lr_target_time = max_num_steps // 10
    max_lr = lr * 100
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False
    num_samples = len(sample)
    vdw_record = torch.zeros((max_num_steps, num_samples))
    samples_record = torch.zeros((max_num_steps, num_samples, 12))
    raw_samples_record = torch.zeros_like(samples_record)
    handedness_record = torch.zeros((max_num_steps, num_samples))
    loss_record = torch.zeros_like(vdw_record)
    lr_record = torch.zeros(max_num_steps)
    packing_record = torch.zeros_like(vdw_record)
    aunit_poses = torch.zeros((len(vdw_record), num_atoms, 3))
    return (hit_max_lr, loss_record, lr_record, max_lr,
            optimizer, packing_record, samples_record, raw_samples_record, handedness_record,
            scheduler1, scheduler2, vdw_record, aunit_poses)

#
# def cleanup_sample(raw_sample, sg_ind_list, symmetries_dict):
#     # force outputs into physical ranges
#     # cell lengths have to be positive nonzero
#     cell_lengths = raw_sample[:, :3].clip(min=0.01)
#     # range from (0,pi) with 20% padding to prevent too-skinny cells
#     cell_angles = enforce_1d_bound(raw_sample[:, 3:6], x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
#                                    mode='hard')
#     # positions must be on 0-1
#     mol_positions = enforce_1d_bound(raw_sample[:, 6:9], x_span=0.5, x_center=0.5, mode='hard')
#     # for now, just enforce vector norm
#     rotvec = raw_sample[:, 9:12]
#     norm = torch.linalg.norm(rotvec, dim=1)
#     new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode='hard')  # MUST be nonzero
#     new_rotvec = rotvec / norm[:, None] * new_norm[:, None]
#     # invert_inds = torch.argwhere(new_rotvec[:, 2] < 0)
#     # new_rotvec[invert_inds] = -new_rotvec[invert_inds]  # z direction always positive
#     # force cells to conform to crystal system
#     cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list,
#                                                        symmetries_dict)
#     sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)
#     return sample
#

def check_convergence(params_record, s_ind, convergence_eps, optimizer, init_lr):
    smoothed = ema_trajectory(params_record[:s_ind])
    diffs = smoothed[s_ind - 50:s_ind, :, :].diff(dim=0).abs().mean((0, 2))
    converged = diffs < convergence_eps
    if (converged.float().mean() > 0.95):
        converged.fill_(True)

    return converged
