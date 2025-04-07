import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import optim
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.geometry_utils import enforce_crystal_system, batch_cell_vol_torch
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import enforce_1d_bound


def standalone_gradient_descent_optimization(
        init_sample: torch.Tensor,
        crystal_batch,
        max_num_steps: int,
        convergence_eps: float,
        lr: float,
        optimizer_func,
        show_tqdm: bool = False,
        es_scaling_factor: float = 10,
        grad_norm_clip: float = 1.0
):
    """
    do a local optimization via gradient descent on some score function
    """

    params_to_optim = init_sample.clone().detach().requires_grad_(True)

    optimizer = optimizer_func([params_to_optim], lr=lr)
    max_lr_target_time = max_num_steps // 10
    max_lr = lr * 100
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False
    loss_record = torch.zeros((max_num_steps, crystal_batch.num_graphs))
    lr_record = torch.zeros(max_num_steps)

    samples_record = []

    converged = False
    with (torch.enable_grad()):
        with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
            s_ind = 0
            while not converged:
                optimizer.zero_grad()
                crystal_batch.set_cell_parameters(params_to_optim)
                crystal_batch.clean_cell_parameters()
                lj_pot, es_pot, scaled_lj_pot, cluster_batch = crystal_batch.build_and_analyze(return_cluster=True,
                                                                                               cutoff=20)
                samples_list = crystal_batch.detach().cpu().to_data_list()

                for si, sample in enumerate(samples_list):
                    sample.lj_pot = lj_pot[si].detach()
                    sample.scaled_lj_pot = scaled_lj_pot[si].detach()
                    sample.es_pot = es_pot[si].detach()

                samples_record.append(samples_list)

                cell_vols = batch_cell_vol_torch(params_to_optim[:, :3], params_to_optim[:, 3:6])
                packing_coeffs = crystal_batch.mol_volume / (crystal_batch.sym_mult * cell_vols)
                cp_loss = F.relu(-(packing_coeffs - 0.65))**2  # encourages cells to close large voids
                # cp_loss = (F.softplus(params_to_optim[:, :3])/crystal_batch.radius[:,None]).sum()  # omnidirectional pressure
                # # intermolecular centroid long range attraction
                # _, atoms_per_cluster = torch.unique(cluster_batch.batch, return_counts=True)
                # mols_per_cluster = (atoms_per_cluster / cluster_batch.num_atoms).long()
                # molwise_batch = torch.arange(cluster_batch.num_graphs, device=cluster_batch.device).repeat_interleave(mols_per_cluster,
                #     dim=0)
                # flat_mol_inds = cluster_batch.mol_ind + torch.cat(
                #     [torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
                #      torch.cumsum(mols_per_cluster, dim=0)]).long()[:-1].repeat_interleave(
                #     torch.diff(cluster_batch.ptr), dim=0)
                # mol_centroids = scatter(cluster_batch.pos, flat_mol_inds, reduce='mean', dim=0, dim_size=flat_mol_inds[-1] + 1)
                # edge_i, edge_j = gnn.radius_graph(x=mol_centroids, batch=molwise_batch, max_num_neighbors=1000, r=1000)
                # inter_dists = torch.linalg.norm(
                #     mol_centroids[edge_i] - mol_centroids[edge_j], dim=1
                # )
                # scaled_inter_dists = inter_dists / crystal_batch.radius[molwise_batch[edge_i]]
                # inter_pot = 1/scaled_inter_dists**12 - 1/scaled_inter_dists
                # cp_loss = scatter(inter_pot, molwise_batch[edge_i], reduce='sum', dim_size=cluster_batch.num_graphs, dim=0)

                # enforce box shape cannot become too long in any direction
                normed_aunit_lengths = cluster_batch.norm_by_radius(cluster_batch.scale_lengths_to_aunit())
                box_loss = F.relu(normed_aunit_lengths - 3).sum(dim=1)**2

                loss = lj_pot + cp_loss + 10*box_loss #+ es_pot.clip(min=-5) * es_scaling_factor + 0.1 * cp_loss
                loss.mean().backward()  # compute gradients
                torch.nn.utils.clip_grad_norm_(params_to_optim, grad_norm_clip)  # gradient clipping
                optimizer.step()  # apply grad
                lr = optimizer.param_groups[0]['lr']
                lr_record[s_ind] = lr
                loss_record[s_ind] = loss.detach().cpu()
                if lr >= max_lr:
                    hit_max_lr = True
                if hit_max_lr:
                    if lr > 1e-7:
                        scheduler1.step()  # shrink
                else:
                    scheduler2.step()  # grow

                s_ind += 1
                if s_ind % 100 == 0:
                    pbar.update(100)

                if s_ind > 250:
                    flag1 = torch.all((loss_record[s_ind - 10:s_ind, :] / loss_record[s_ind - 10:s_ind, :].mean()).std(
                        0) < convergence_eps)  # loss is converged
                    flag2 = s_ind > (max_num_steps - 1)  # run out of time
                    if flag1 or flag2:
                        converged = True

    return samples_record


"""
import plotly.graph_objects as go
fig = go.Figure()
lj_pots = torch.stack(
    [torch.tensor([sample.scaled_lj_pot for sample in sample_list]) for sample_list in samples_record])
es_pots = torch.stack([torch.tensor([sample.es_pot for sample in sample_list]) for sample_list in samples_record])
for ind in range(lj_pot.shape[-1]):
    fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
                    showlegend=True if ind == 0 else False)
    fig.add_scatter(y=es_pots[..., ind] * es_scaling_factor, marker_color='red', name='es', legendgroup='es',
                    showlegend=True if ind == 0 else False)
    fig.add_scatter(y=es_pots[..., ind] * es_scaling_factor + lj_pots[..., ind], marker_color='green', name='combo', legendgroup='combo',
                    showlegend=True if ind == 0 else False)

fig.show()


import plotly.graph_objects as go
fig = go.Figure()
lj_pots = torch.stack(
    [torch.tensor([sample.packing_coeff for sample in sample_list]) for sample_list in samples_record])
for ind in range(lj_pot.shape[-1]):
    fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
                    showlegend=True if ind == 0 else False)
fig.update_layout(yaxis_range=[0,1])
fig.show()

lj_pot, es_pot, scaled_lj_pot, cluster_batch = crystal_batch.build_and_analyze(return_cluster=True)
cluster_batch.visualize(mode='convolve with')

"""


def standalone_opt_random_crystals(
        crystal_batch,
        init_state,
        opt_eps,
        post_scramble_each: int = None,
):
    # recenter and align to xyz axes
    crystal_batch.orient_molecule(mode='standardized')

    # print("doing opt")
    standalone_gradient_descent_optimization(
        init_state,
        crystal_batch.clone(),
        max_num_steps=1000,
        convergence_eps=opt_eps,
        lr=1e-5,  # initial LR
        optimizer_func=torch.optim.Rprop,
    )

    # extract optimized samples
    opt_samples = samples_record[-1]

    # filter unbound states
    opt_samples = [sample for sample in opt_samples if sample.lj_pot < 0]

    # sample noisily about optimized minima
    nearby_samples = sample_about_crystal(opt_samples,
                                          noise_level=0.05,  # empirically gets us an LJ std about 3
                                          num_samples=post_scramble_each)

    for ind in range(post_scramble_each):
        opt_samples.extend(nearby_samples[ind])

    opt_samples = [sample for sample in opt_samples if sample.lj_pot < 0]  # filter bound states

    return opt_samples


""" # viz sample distribution

import plotly.graph_objects as go
ljs = torch.tensor([elem.scaled_lj_pot for elem in opt_samples[:len(samples_record[0])]])
ljs2 = torch.tensor([elem.scaled_lj_pot for elem in opt_samples])
fig = go.Figure()
fig.add_histogram(x=ljs, nbinsx=100, histnorm='probability density')
fig.add_histogram(x=ljs2, nbinsx=100, histnorm='probability density')
fig.show()

"""


def sample_about_crystal(opt_samples: list,
                         noise_level: float,
                         num_samples: int,
                         ):
    samples_record = []
    for ind in range(num_samples):
        crystal_batch = collate_data_list(opt_samples)
        crystal_batch.noise_cell_parameters(noise_level)
        lj_pot, es_pot, scaled_lj_pot = crystal_batch.build_and_analyze()
        samples_list = crystal_batch.detach().cpu().to_data_list()

        for si, sample in enumerate(samples_list):
            sample.lj_pot = lj_pot[si]
            sample.scaled_lj_pot = scaled_lj_pot[si]
            sample.es_pot = es_pot[si]

        samples_record.append(samples_list)

    return samples_record


"""
import plotly.graph_objects as go
fig = go.Figure()
lj_pots = torch.stack(
    [torch.tensor([sample.scaled_lj_pot for sample in sample_list]) for sample_list in samples_record])
es_pots = torch.stack([torch.tensor([sample.es_pot for sample in sample_list]) for sample_list in samples_record])
orig_batch = collate_data_list(opt_samples)
lj_pots = torch.cat([orig_batch.scaled_lj_pot[None, :], lj_pots], dim=0)
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


def scramble_resample(post_scramble_each, samples_record):  # deprecated
    resampled_record = []
    for sample_list in samples_record[::post_scramble_each]:
        crystal_batch = collate_data_list(sample_list)

        # random directions on the sphere, getting naturally the correct distribution of theta, phi
        random_vectors = torch.randn(size=(crystal_batch.num_graphs, 3))

        # set norms uniformly between 0-2pi
        norms = random_vectors.norm(dim=1)
        applied_norms = (torch.rand(crystal_batch.num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
        random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

        crystal_batch.aunit_orientation = random_vectors
        lj_pot, es_pot, scaled_lj_pot = crystal_batch.build_and_analyze()
        samples_list = crystal_batch.detach().cpu().to_data_list()

        for si, sample in enumerate(samples_list):
            sample.lj_pot = lj_pot[si]
            sample.scaled_lj_pot = scaled_lj_pot[si]
            sample.es_pot = es_pot[si]

        resampled_record.append(samples_list)
    samples_record.extend(resampled_record)
    return samples_record


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


def cleanup_sample(raw_sample, sg_ind_list, symmetries_dict):
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
                                                       symmetries_dict)
    sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)
    return sample
