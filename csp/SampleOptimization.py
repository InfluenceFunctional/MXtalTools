import numpy as np
import tqdm
from models.utils import softmax_and_score
from models.vdw_overlap import vdw_overlap
import torch.optim as optim
import torch


def gradient_descent_sampling(discriminator, crystal_batch, supercell_builder,
                              n_iter, lr, optimizer_func, vdw_radii,
                              supercell_size=5, cutoff=6):
    ''' DEPRECATED
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
    n_iter
    supercell_size
    cutoff
    generate_sgs

    Returns
    -------
    sampling dict containing scores and samples

    '''

    sample = crystal_batch.cell_params.clone()
    optimizer = optimizer_func([sample], lr=lr)

    max_lr_target_time = n_iter // 10
    max_lr = 1e-2
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)

    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False

    n_samples = len(sample)

    scores_record = np.zeros((n_iter, n_samples))
    samples_record = np.zeros((n_iter, n_samples, 12))
    loss_record = np.zeros_like(scores_record)
    vdw_record = np.zeros_like(scores_record)

    for i in tqdm.tqdm(range(n_iter), miniters=int(n_iter / 25)):

        optimizer.zero_grad()
        supercell_data, _, _ = \
            supercell_builder.build_supercells(
                crystal_batch, sample,
                supercell_size, cutoff,
                align_to_standardized_orientation=False,
                target_handedness=crystal_batch.asym_unit_handedness)

        output, dist_dict = discriminator(supercell_data.clone().cuda(), return_dists=True)

        score = softmax_and_score(output[:, :2])
        # loss = 1 - F.softmax(output, dim=-1)[:, 1]
        loss = -score
        loss.mean().backward()  # compute gradients
        optimizer.step()  # apply grad

        vdw_record[i] = vdw_overlap(vdw_radii,
                                    dists=dist_dict['dists_dict']['intermolecular_dist'],
                                    atomic_numbers=dist_dict['dists_dict']['intermolecular_dist_atoms'],
                                    batch_numbers=dist_dict['dists_dict']['intermolecular_dist_batch'],
                                    num_graphs=crystal_batch.num_graphs).cpu().detach().numpy()
        scores_record[i] = score.cpu().detach().numpy()
        samples_record[i] = supercell_data.cell_params.cpu().detach().numpy()
        loss_record[i] = loss.cpu().detach().numpy()

        lr = optimizer.param_groups[0]['lr']
        if lr >= max_lr:
            hit_max_lr = True
        if hit_max_lr:
            if lr > 1e-5:
                scheduler2.step()  # shrink
        else:
            scheduler1.step()  # grow

    sampling_dict = {'std_cell_params': samples_record, 'score': scores_record,
                     'vdw_score': vdw_record, 'space_group': supercell_data.sg_ind}

    return crystal_batch, sampling_dict


def mcmc_sampling(model, crystal_batch, supercell_builder, num_steps, vdw_radii, supercell_size, cutoff, sampling_temperature, lattice_means, lattice_stds, step_size):
    samples_record = np.zeros((num_steps, crystal_batch.num_graphs, 12))
    scores_record = np.zeros((num_steps, crystal_batch.num_graphs))
    vdw_record = np.zeros_like(scores_record)
    space_group_record = np.zeros_like(scores_record)

    alpha_randoms = np.random.uniform(0, 1, size=(num_steps, crystal_batch.num_graphs))
    propose_randoms = step_size * (np.random.randn(num_steps, crystal_batch.num_graphs, 12) * lattice_stds + lattice_means)

    with torch.no_grad():
        for s_ind in tqdm.tqdm(range(num_steps)):  # sample for a certain number of iterations
            if s_ind != 0:
                proposed_samples = torch.tensor(np.copy(samples_record[s_ind - 1]) + propose_randoms[s_ind],
                                                dtype=torch.float32, device=crystal_batch.x.device)

                proposed_crystals, _ = \
                    supercell_builder.build_supercells(
                        crystal_batch, proposed_samples,
                        supercell_size, cutoff,
                        align_to_standardized_orientation=True,
                        target_handedness=crystal_batch.asym_unit_handedness)

                output, proposed_dist_dict = model(proposed_crystals.clone().cuda(), return_dists=True)

                proposed_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()

                proposed_sample_vdws = vdw_overlap(vdw_radii,
                                                   crystaldata=crystal_batch,
                                                   dists=proposed_dist_dict['dists_dict']['intermolecular_dist'],
                                                   atomic_numbers=proposed_dist_dict['dists_dict']['intermolecular_dist_atoms'],
                                                   batch_numbers=proposed_dist_dict['dists_dict']['intermolecular_dist_batch'],
                                                   num_graphs=crystal_batch.num_graphs,
                                                   return_score_only=True).cpu().detach().numpy()

                score_difference = scores_record[s_ind - 1] - proposed_sample_scores
                acceptance_ratio = np.minimum(
                    1,
                    np.exp(-score_difference / sampling_temperature)
                )
                accept_flags = alpha_randoms[s_ind] < acceptance_ratio

                samples_record[s_ind] = samples_record[s_ind - 1]
                samples_record[s_ind, accept_flags] = proposed_samples[accept_flags].cpu().detach().numpy()
                scores_record[s_ind] = scores_record[s_ind - 1]
                scores_record[s_ind, accept_flags] = proposed_sample_scores[accept_flags]
                vdw_record[s_ind] = vdw_record[s_ind - 1]
                vdw_record[s_ind, accept_flags] = proposed_sample_vdws[accept_flags]
                space_group_record[s_ind] = crystal_batch.sg_ind.cpu().detach().numpy()
            else:
                proposed_samples = crystal_batch.cell_params
                proposed_crystals, _ = \
                    supercell_builder.build_supercells(
                        crystal_batch, proposed_samples,
                        supercell_size, cutoff,
                        align_to_standardized_orientation=True,
                        target_handedness=crystal_batch.asym_unit_handedness)

                output, proposed_dist_dict = model(proposed_crystals.clone().cuda(), return_dists=True)

                proposed_sample_scores = softmax_and_score(output[:, :2]).cpu().detach().numpy()

                proposed_sample_vdws = vdw_overlap(vdw_radii,
                                                   crystaldata=crystal_batch,
                                                   dists=proposed_dist_dict['dists_dict']['intermolecular_dist'],
                                                   atomic_numbers=proposed_dist_dict['dists_dict']['intermolecular_dist_atoms'],
                                                   batch_numbers=proposed_dist_dict['dists_dict']['intermolecular_dist_batch'],
                                                   num_graphs=crystal_batch.num_graphs,
                                                   return_score_only=True).cpu().detach().numpy()

                samples_record[s_ind] = proposed_samples.cpu().detach().numpy()
                scores_record[s_ind] = proposed_sample_scores
                vdw_record[s_ind] = proposed_sample_vdws
                space_group_record[s_ind] = crystal_batch.sg_ind.cpu().detach().numpy()

    sampling_dict = {'std_cell_params': samples_record, 'score': scores_record,
                     'vdw_score': vdw_record, 'space_group': space_group_record}

    '''return final sample'''
    supercell_data, _, _ = \
        supercell_builder.build_supercells(
            crystal_batch, torch.Tensor(samples_record[-1], dtype=torch.float32, device=crystal_batch.x.device),
            supercell_size, cutoff,
            align_to_standardized_orientation=True,
            target_handedness=crystal_batch.asym_unit_handedness)

    return supercell_data, sampling_dict
