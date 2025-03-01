from typing import Optional

import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from mxtaltools.analysis.crystals_analysis import get_intermolecular_dists_dict
from mxtaltools.analysis.vdw_analysis import electrostatic_analysis, vdw_analysis, scale_molwise_vdw_pot
from mxtaltools.common.geometry_utils import enforce_crystal_system
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import enforce_1d_bound


def standalone_add_scrambled_molecule_samples(aunit_poses, dist_dict, handedness_record, hit_max_lr, loss, loss_record,
                                              lr_record, max_lr, mol_batch, optimizer, packing_coeff, packing_record,
                                              post_scramble_each, raw_samples_record, s_ind, sample, sample_to_compare,
                                              samples_record, scheduler1, scheduler2, score_func, standardize_pose,
                                              store_aunit, supercell_batch, vdw_record, supercell_size, cutoff,
                                              supercell_builder):
    s_ind -= 1
    inds_to_scramble = torch.arange(0, s_ind, max(1, s_ind // post_scramble_each))
    scrambled_samples_record = torch.zeros_like(samples_record)[:len(inds_to_scramble)]
    scrambled_packing_record = torch.zeros_like(packing_record)[:len(inds_to_scramble)]
    scrambled_loss_record = torch.zeros_like(packing_record)[:len(inds_to_scramble)]
    scrambled_vdw_record = torch.zeros_like(packing_record)[:len(inds_to_scramble)]
    scrambled_aunit_poses = torch.zeros_like(aunit_poses)[:len(inds_to_scramble)]
    scrambled_handedness_record = torch.zeros_like(handedness_record)[
                                  :len(inds_to_scramble)]
    for s_ind2, scramble_ind in enumerate(inds_to_scramble):
        sample = raw_samples_record[scramble_ind] + torch.cat([
            torch.zeros_like(sample[:, :9]), torch.randn_like(sample[:, -3:])
        ], dim=1)  # scramble molecule orientation
        sample.requires_grad_(True)
        descaled_cleaned_sample, dist_dict, loss, packing_coeff, supercell_batch, vdw_potential = standalone_gd_opt_step(
            hit_max_lr, lr_record, max_lr, mol_batch, optimizer, s_ind, sample, scheduler1,
            scheduler2,
            score_func, standardize_pose, supercell_size, cutoff, supercell_builder)

        sample_to_compare = descaled_cleaned_sample.clone()
        sample_to_compare[:, 9:] = supercell_batch.cell_params[:, 9:]

        scrambled_vdw_record[s_ind2] = vdw_potential.detach()
        scrambled_samples_record[s_ind2] = sample_to_compare.detach()
        scrambled_loss_record[s_ind2] = loss.detach()
        scrambled_packing_record[s_ind2] = packing_coeff.detach()
        scrambled_handedness_record[s_ind2] = supercell_batch.aunit_handedness
        if store_aunit:
            scrambled_aunit_poses[s_ind2] = supercell_batch.pos[
                supercell_batch.aux_ind == 0].detach()
    s_ind += len(inds_to_scramble)
    samples_record = torch.cat([samples_record, scrambled_samples_record], dim=0)
    vdw_record = torch.cat([vdw_record, scrambled_vdw_record], dim=0)
    loss_record = torch.cat([loss_record, scrambled_loss_record], dim=0)
    packing_record = torch.cat([packing_record, scrambled_packing_record], dim=0)
    aunit_poses = torch.cat([aunit_poses, scrambled_aunit_poses], dim=0)
    return aunit_poses, loss, loss_record, packing_coeff, packing_record, s_ind, sample, sample_to_compare, samples_record, supercell_batch, vdw_record


def standalone_score_crystal_batch(mol_batch, score_func, supercell_data, vdw_radii_tensor, cutoff: float = 6, discriminator: Optional = None):
    if score_func == 'discriminator':
        output, extra_outputs = discriminator(
            supercell_data.clone(), return_dists=True, return_latent=False)
        dist_dict = extra_outputs['dists_dict']
    elif score_func.lower() == 'vdw':
        dist_dict = get_intermolecular_dists_dict(supercell_data,
                                                  cutoff, 100)
    else:
        assert False, f"{score_func} is not an implemented score function for gradient descent optimization"
    molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, lj_pot \
        = vdw_analysis(vdw_radii_tensor,
                       dist_dict,
                       mol_batch.num_graphs)
    estat_energy = electrostatic_analysis(dist_dict, supercell_data.num_graphs)
    vdw_potential += estat_energy
    vdw_loss += estat_energy
    if score_func == 'discriminator':
        loss = F.softplus(output[:, 2])
    elif score_func.lower() == 'vdw':
        loss = vdw_loss
    return dist_dict, loss, vdw_potential


def standalone_gradient_descent_optimization(
        init_sample: torch.Tensor,
        crystal_batch,
        max_num_steps: int,
        convergence_eps: float,
        lr: float,
        optimizer_func,
        show_tqdm: bool = False,
        quantile_to_optim: float = 0.75
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
                lj_pot, es_pot, scaled_lj_pot = crystal_batch.build_and_analyze()
                samples_list = crystal_batch.detach().cpu().to_data_list()

                for si, sample in enumerate(samples_list):
                    sample.lj_pot = lj_pot[si]
                    sample.scaled_lj_pot = scaled_lj_pot[si]
                    sample.es_pot = es_pot[si]

                samples_record.append(samples_list)

                loss = scaled_lj_pot + es_pot * 10
                loss.mean().backward()  # compute gradients
                optimizer.step()  # apply grad
                lr = optimizer.param_groups[0]['lr']
                lr_record[s_ind] = lr
                loss_record[s_ind] = loss.detach().cpu()
                if lr >= max_lr:
                    hit_max_lr = True
                if hit_max_lr:
                    if lr > 1e-5:
                        scheduler1.step()  # shrink
                else:
                    scheduler2.step()  # grow

                s_ind += 1
                if s_ind % 100 == 0:
                    pbar.update(100)

                if s_ind > 10:
                    flag1 = torch.quantile(
                        loss_record[s_ind - 10:s_ind, :].std(0), quantile_to_optim
                    ) < convergence_eps  # loss is converged
                    flag2 = s_ind > (max_num_steps - 1)  # run out of time
                    if flag1 or flag2:
                        converged = True

    return samples_record

    # import plotly.graph_objects as go
    # fig = go.Figure()
    # lj_pots = torch.stack(
    #     [torch.tensor([sample.scaled_lj_pot for sample in sample_list]) for sample_list in samples_record])
    # es_pots = torch.stack([torch.tensor([sample.es_pot for sample in sample_list]) for sample_list in samples_record])
    # for ind in range(lj_pot.shape[-1]):
    #     fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
    #                     showlegend=True if ind == 0 else False)
    #     fig.add_scatter(y=es_pots[..., ind], marker_color='red', name='es', legendgroup='es',
    #                     showlegend=True if ind == 0 else False)
    #
    # fig.show()


def standalone_gd_opt_step(hit_max_lr, lr_record, max_lr, mol_batch, optimizer, s_ind, sample, scheduler1,
                           scheduler2,
                           score_func, standardize_pose, cutoff, supercell_size):
    optimizer.zero_grad()
    # cleaned_sample = cleanup_sample(sample, mol_batch.sg_ind, supercell_builder.symmetries_dict)
    # descaled_cleaned_sample = denormalize_generated_cell_params(cleaned_sample, mol_batch, supercell_builder.ASYM_UNITS)

    # loss.mean().backward()  # compute gradients
    # optimizer.step()  # apply grad
    # lr = optimizer.param_groups[0]['lr']
    # lr_record[s_ind] = lr
    # if lr >= max_lr:
    #     hit_max_lr = True
    # if hit_max_lr:
    #     if lr > 1e-5:
    #         scheduler1.step()  # shrink
    # else:
    #     scheduler2.step()  # grow
    # return descaled_cleaned_sample, dist_dict, loss, packing_coeff, supercell_batch, vdw_potential


def standalone_opt_random_crystals(
        crystal_batch,
        init_state,
        opt_eps,
        post_scramble_each: int = None,
        ):
    # recenter and align to xyz axes
    crystal_batch.orient_molecule(mode='standardized')

    # print("doing opt")
    samples_record = standalone_gradient_descent_optimization(
        init_state,
        crystal_batch.clone(),
        max_num_steps=1000,
        convergence_eps=opt_eps,
        lr=1e-6,
        optimizer_func=torch.optim.Rprop,
    )
    # do resampling
    if post_scramble_each is not None:
        samples_record = scramble_resample(post_scramble_each, samples_record)

    # return flattened list
    return [sample for samples_list in samples_record for sample in samples_list]


def scramble_resample(post_scramble_each, samples_record):
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
