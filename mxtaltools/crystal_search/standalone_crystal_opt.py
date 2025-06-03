import copy
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import optim
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.geometry_utils import enforce_crystal_system
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import enforce_1d_bound, softmax_and_score


def standalone_gradient_descent_optimization(
        init_sample: torch.Tensor,
        init_crystal_batch,
        optimizer_func: Optional = torch.optim.Rprop,
        convergence_eps: Optional[float] = 1e-3,
        lr: Optional[float] = 1e-4,
        max_num_steps: Optional[int] = 500,
        show_tqdm: Optional[bool] = False,
        grad_norm_clip: Optional[float] = 0.1,
        optim_target: Optional[str] = 'LJ',
        score_model: Optional[torch.nn.Module] = None,
        target_packing_coeff: Optional[torch.Tensor] = None,
        do_box_restriction: Optional[bool] = False,
        cutoff: Optional[float] = 10,
        compression_factor: Optional[float] = 0,
        enforce_niggli: Optional[bool] = False,
):
    """
    do a local optimization via gradient descent on some score function
    """

    params_to_optim = init_sample.clone().detach().requires_grad_(True)

    optimizer = optimizer_func([params_to_optim], lr=lr)
    max_lr_target_time = max_num_steps // 10
    max_lr = lr * 10
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.985)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False
    loss_record = torch.zeros((max_num_steps, init_crystal_batch.num_graphs))
    lr_record = torch.zeros(max_num_steps)
    params_record = torch.zeros((max_num_steps, init_crystal_batch.num_graphs, 12), dtype=torch.float32)

    optimization_trajectory = []

    if optim_target.lower() == 'ellipsoid':
        init_crystal_batch.load_ellipsoid_model()
        ellipsoid_model = copy.deepcopy(init_crystal_batch.ellipsoid_model)
        ellipsoid_model = ellipsoid_model.to(init_crystal_batch.device)
        ellipsoid_model.eval()

    converged = False
    # torch.autograd.set_detect_anomaly(True)  # for debugging
    with (torch.enable_grad()):
        with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
            s_ind = 0
            while not converged:
                optimizer.zero_grad()
                crystal_batch = init_crystal_batch.clone().detach()
                crystal_batch.set_cell_parameters(params_to_optim)
                crystal_batch.clean_cell_parameters(
                    enforce_niggli=enforce_niggli,
                    mode='hard',
                )
                lj_pot, es_pot, scaled_lj_pot, cluster_batch = crystal_batch.build_and_analyze(return_cluster=True,
                                                                                               cutoff=cutoff)

                params_record[s_ind] = cluster_batch.standardize_cell_parameters().detach().cpu()

                if optim_target.lower() == 'lj':
                    loss = lj_pot

                elif optim_target.lower() == 'silu':
                    silu_energy = cluster_batch.compute_silu_energy()
                    loss = silu_energy

                elif optim_target.lower() == 'inter_overlaps':  # force molecules apart by separating their centroids
                    loss = inter_overlap_loss(cluster_batch, crystal_batch)

                elif optim_target.lower() == 'classification_score':
                    score = softmax_and_score(score_model(cluster_batch, force_edges_rebuild=False)[:, :2])
                    loss = -score

                elif optim_target.lower() == 'rdf_score':
                    dist_pred = F.softplus(score_model(cluster_batch, force_edges_rebuild=False)[:, 2])
                    loss = dist_pred

                elif optim_target.lower() == 'ellipsoid':
                    overlap = cluster_batch.ellipsoid_overlap = cluster_batch.compute_ellipsoidal_overlap(
                        semi_axis_scale=1, model=ellipsoid_model).clip(min=0)
                    loss = overlap**2

                "auxiliary losses"
                if target_packing_coeff is not None:
                    cp_loss = (cluster_batch.packing_coeff - target_packing_coeff) ** 2
                    loss = loss + cp_loss

                if do_box_restriction:
                    # enforce box shape cannot become too long (3 mol radii) or narrow (3 angstroms) in any direction
                    aunit_lengths = cluster_batch.scale_lengths_to_aunit()
                    box_loss = (80000 / aunit_lengths ** 12).sum(
                        dim=1)  # repulsive from about range 3 #(80000/aunit_lengths**12 + 10*aunit_lengths - 31.25).sum(dim=1)  # forces boxes to be larger than 3 angstroms, but squeezes them otherwise
                    loss = loss + box_loss

                if compression_factor != 0:
                    aunit_lengths = cluster_batch.scale_lengths_to_aunit()
                    loss = loss + aunit_lengths.sum(dim=1) * compression_factor

                loss.mean().backward()  # compute gradients
                torch.nn.utils.clip_grad_norm_(params_to_optim, grad_norm_clip)  # gradient clipping
                optimizer.step()  # apply grad
                lr = optimizer.param_groups[0]['lr']

                """record keeping"""
                samples_list = crystal_batch.clone().cpu().detach().to_data_list()
                lj_pot = lj_pot.cpu().detach()
                scaled_lj_pot = scaled_lj_pot.cpu().detach()
                es_pot = es_pot.cpu().detach()
                packing_coeffs = crystal_batch.packing_coeff.cpu().detach()
                if optim_target.lower() == 'silu':
                    silu_energy = silu_energy.cpu().detach()

                for si, sample in enumerate(samples_list):
                    sample.lj_pot = lj_pot[si]
                    sample.scaled_lj_pot = scaled_lj_pot[si]
                    sample.es_pot = es_pot[si]
                    sample.packing_coeff = packing_coeffs[si]
                    if optim_target.lower() == 'silu':
                        sample.silu_pot = silu_energy[si]

                optimization_trajectory.append(samples_list)
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
                if s_ind % 25 == 0:
                    pbar.update(25)

                if s_ind >= min(max_num_steps, 50):
                    diffs = params_record[s_ind - 10:s_ind, :, :].diff(dim=0).abs()
                    flag1 = torch.all(diffs < convergence_eps)
                    flag2 = s_ind > (max_num_steps - 1)  # run out of time
                    if flag1 or flag2:
                        converged = True

    return optimization_trajectory


"""
import plotly.graph_objects as go

lj_pots = torch.stack(
    [torch.tensor([sample.scaled_lj_pot for sample in sample_list]) for sample_list in optimization_trajectory])
coeffs = torch.stack(
    [torch.tensor([sample.packing_coeff for sample in sample_list]) for sample_list in optimization_trajectory])

fig = go.Figure()
fig.add_scatter(x=coeffs[0,:], y=lj_pots[0, :], mode='markers', marker_size=20, marker_color='grey', name='Initial State')
fig.add_scatter(x=coeffs[-1,:], y=lj_pots[-1, :], mode='markers', marker_size=20, marker_color='black', name='Final State')
for ind in range(coeffs.shape[1]):
    fig.add_scatter(x=coeffs[:, ind], y=lj_pots[:, ind], name=f"Run {ind}")
fig.update_layout(xaxis_title='Packing Coeff', yaxis_title='Scaled LJ')
fig.show()

params = torch.stack([torch.cat([sample.cell_parameters() for sample in sample_list]) for sample_list in optimization_trajectory])
from plotly.subplots import make_subplots
fig = make_subplots(rows=4, cols=3, subplot_titles = ['a','b','c','al','be','ga','x','y','z','u','v','w'])
for ind in range(12):
    row = ind // 3 + 1
    col = ind % 3 + 1
    for ind2 in range(params.shape[1]):
        fig.add_scatter(y=params[:, ind2, ind], showlegend=False, row=row, col=col)
fig.show()

# import plotly.graph_objects as go
# fig = go.Figure()
# for ind in range(lj_pots.shape[-1]):
#     fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
#                     showlegend=True if ind == 0 else False)
# fig.show()
# 
# 
# import plotly.graph_objects as go
# fig = go.Figure()
# 
# for ind in range(lj_pots.shape[-1]):
#     fig.add_scatter(y=coeffs[..., ind], marker_color='blue', name='packing coeff', legendgroup='lg',
#                     showlegend=True if ind == 0 else False)
# fig.update_layout(yaxis_range=[0,1])
# fig.show()

lj_pot, es_pot, scaled_lj_pot, cluster_batch = crystal_batch.build_and_analyze(return_cluster=True)
cluster_batch.visualize(mode='convolve with')

"""


def inter_overlap_loss(cluster_batch, crystal_batch):
    # enforce molecules far enough away that they cannot possibly overlap
    # intermolecular centroid range repulsion
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
    scaled_inter_dists = inter_dists / crystal_batch.radius[molwise_batch[edge_i]]
    edgewise_losses = F.relu(-(scaled_inter_dists - 2))  # push molecules apart if they are within each others' radii
    loss = scatter(edgewise_losses,
                   molwise_batch[edge_i],
                   dim_size=cluster_batch.num_graphs,
                   dim=0,
                   reduce='sum')
    return loss


""" # viz sample distribution

import plotly.graph_objects as go
ljs = torch.tensor([elem.scaled_lj_pot for elem in opt_samples[:len(samples_record[0])]])
ljs2 = torch.tensor([elem.scaled_lj_pot for elem in opt_samples])
fig = go.Figure()
fig.add_histogram(x=ljs, nbinsx=100, histnorm='probability density')
fig.add_histogram(x=ljs2, nbinsx=100, histnorm='probability density')
fig.show()

"""


def sample_about_crystal(opt_samples: Union[list],
                         noise_level: float,
                         num_samples: int,
                         cutoff: Optional[float] = 10,
                         enforce_niggli: Optional[bool] = False,
                         do_silu_pot: Optional[bool] = False
                         ):
    samples_record = []
    for ind in range(num_samples):
        if isinstance(opt_samples, list):
            crystal_batch = collate_data_list(opt_samples)
        else:
            crystal_batch = opt_samples.clone()

        crystal_batch.noise_cell_parameters(noise_level)
        crystal_batch.clean_cell_parameters(
            enforce_niggli=enforce_niggli,
            mode='hard',
        )
        lj_pot, es_pot, scaled_lj_pot, cluster_batch = crystal_batch.build_and_analyze(
            cutoff=cutoff, return_cluster=True)
        samples_list = crystal_batch.detach().cpu().to_data_list()

        if do_silu_pot:
            silu_energy = cluster_batch.compute_silu_energy()

        for si, sample in enumerate(samples_list):
            sample.lj_pot = lj_pot[si]
            sample.scaled_lj_pot = scaled_lj_pot[si]
            sample.es_pot = es_pot[si]
            if do_silu_pot:
                sample.silu_pot = silu_energy[si]

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
