import numpy as np
import tqdm
from utils import softmax_and_score
from models.vdw_overlap import vdw_overlap
import torch.nn.functional as F



def gradient_descent_sampling(discriminator, d_optimizer, init_samples, single_mol_data, supercell_builder, n_iter,
                              return_vdw = False, vdw_radii = None, supercell_size=5, cutoff=6, generate_sgs=None, lr_scale = 1):
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
    if not init_samples.requires_grad:
        init_samples.requires_grad = True
    sample = init_samples.clone().cuda()

    sample.retain_grad()
    n_samples = len(sample)

    scores_record = np.zeros((n_iter, n_samples))
    samples_record = np.zeros((n_iter, n_samples, 12))
    vdw_record = np.zeros_like(scores_record)

    for i in tqdm.tqdm(range(n_iter)):
        if i > 0:
            sample = new_sample
            new_sample.retain_grad()

        supercell_data, generated_cell_volumes, overlaps_list = \
            supercell_builder.build_supercells(single_mol_data, sample,
                                               supercell_size, cutoff, override_sg=generate_sgs)

        loss = 1-F.softmax(discriminator(supercell_data),dim=-1)[:,1] #-softmax_and_score(discriminator(supercell_data))
        scores_record[i, :] = 1-loss.cpu().detach().numpy()
        samples_record[i] = sample.cpu().detach().numpy()
        if return_vdw:
            vdw_record[i] = vdw_overlap(vdw_radii, crystaldata = supercell_data).cpu().detach().numpy()

        d_optimizer.zero_grad()  # reset gradients from previous passes
        loss.mean().backward()  # back-propagation

        new_sample = sample.detach() + sample.grad.detach() * lr_scale
        new_sample.requires_grad = True

    sampling_dict = {'samples': samples_record, 'scores': scores_record, 'vdw':vdw_record}

    return sampling_dict