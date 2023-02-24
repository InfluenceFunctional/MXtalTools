import numpy as np
import tqdm
from utils import softmax_and_score
from models.vdw_overlap import vdw_overlap
import torch.nn.functional as F
import torch.optim as optim
import torch


def gradient_descent_sampling(discriminator, init_samples, single_mol_data, supercell_builder, n_iter, lr, optimizer_func,
                              return_vdw=False, vdw_radii=None, supercell_size=5, cutoff=6, generate_sgs='P-1',
                              align_molecules = False):
    '''
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

    sample = torch.tensor(init_samples, device='cuda', requires_grad=True, dtype=torch.float32)
    optimizer = optimizer_func([sample], lr=lr)

    max_lr_target_time = n_iter // 10
    max_lr = 1e-2
    grow_lambda = (max_lr/lr)**(1/max_lr_target_time)

    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = 0

    n_samples = len(sample)

    scores_record = np.zeros((n_iter, n_samples))
    samples_record = np.zeros((n_iter, n_samples, 12))
    loss_record = np.zeros_like(scores_record)
    vdw_record = np.zeros_like(scores_record)


    for i in tqdm.tqdm(range(n_iter), miniters=int(n_iter / 25)):

        optimizer.zero_grad()
        supercell_data, _, _ = \
            supercell_builder.build_supercells(single_mol_data, sample,
                                               supercell_size=supercell_size,
                                               graph_convolution_cutoff=cutoff,
                                               override_sg=generate_sgs,
                                               align_molecules=align_molecules,
                                               skip_cell_cleaning = False, # first iter are pre-cleaned cells
                                               rescale_asymmetric_unit= True, # first iter are pre-cleaned cells
                                               )

        output, dist_dict = discriminator(supercell_data.clone().cuda(), return_dists=True)

        score = softmax_and_score(output)
        #loss = 1 - F.softmax(output, dim=-1)[:, 1]
        loss = -score
        loss.mean().backward()  # compute gradients
        optimizer.step()  # apply grad

        if return_vdw:
            vdw_record[i] = vdw_overlap(vdw_radii, dists=dist_dict['dists dict']['intermolecular dist'],
                                        atomic_numbers=dist_dict['dists dict']['intermolecular dist atoms'],
                                        batch_numbers=dist_dict['dists dict']['intermolecular dist batch'],
                                        num_graphs=single_mol_data.num_graphs).cpu().detach().numpy()

        scores_record[i] = score.cpu().detach().numpy()
        samples_record[i] = sample.cpu().detach().numpy()
        loss_record[i] = loss.cpu().detach().numpy()

        lr = optimizer.param_groups[0]['lr']
        if  lr >= max_lr:
            hit_max_lr = True
        if hit_max_lr:
            if lr > 1e-5:
                scheduler2.step() # shrink
        else:
            scheduler1.step() # grow

    sampling_dict = {'samples': samples_record, 'scores': scores_record,
                     'vdw': vdw_record, 'loss': loss_record}

    return sampling_dict
