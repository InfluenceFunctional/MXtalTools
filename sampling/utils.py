import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import constants.asymmetric_units
from crystal_building.builder import write_sg_to_all_crystals
from models.utils import undo_1d_bound, softmax_and_score


def de_clean_samples(supercell_builder, samples, sg_inds):
    means = supercell_builder.dataDims['lattice means']
    stds = supercell_builder.dataDims['lattice stds']

    # soft clipping to ensure correct range with finite gradients
    cell_lengths = torch.Tensor(samples[:, :3] * stds[0:3] + means[0:3])
    cell_angles = torch.Tensor(samples[:, 3:6] * stds[3:6] + means[3:6])
    mol_position = torch.Tensor(samples[:, 6:9] * stds[6:9] + means[6:9])
    mol_rotation = torch.Tensor(samples[:, 9:12] * stds[9:12] + means[9:12])

    # descale asymmetric unit
    descaled_mol_position = mol_position.clone()
    for i, ind in enumerate(sg_inds):
        descaled_mol_position[i, :] = mol_position[i, :] / constants.asymmetric_units.asym_unit_dict[
            str(int(ind))].cpu()

    # undo cleaning
    unclean_cell_lengths = np.log(np.exp(cell_lengths) - np.exp(1) ** (1 / 10))
    unclean_cell_angles = undo_1d_bound(cell_angles, x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2, mode='soft')
    unclean_mol_position = undo_1d_bound(descaled_mol_position, 0.5, 0.5, mode='soft')
    norms = torch.linalg.norm(mol_rotation, dim=1)
    unclean_norms = undo_1d_bound(norms, torch.pi, torch.pi, mode='soft')
    unclean_mol_rotation = mol_rotation / norms[:, None] * unclean_norms[:, None]

    # restandardize samples
    unclean_cell_lengths = (unclean_cell_lengths.detach().numpy() - means[0:3]) / stds[0:3]
    unclean_cell_angles = (unclean_cell_angles.detach().numpy() - means[3:6]) / stds[3:6]
    unclean_mol_position = (unclean_mol_position.detach().numpy() - means[6:9]) / stds[6:9]
    unclean_mol_rotation = (unclean_mol_rotation.detach().numpy() - means[9:12]) / stds[9:12]

    unclean_best_samples = np.concatenate(
        (unclean_cell_lengths, unclean_cell_angles, unclean_mol_position, unclean_mol_rotation), axis=1)
    return unclean_best_samples


def sample_clustering(supercell_builder, config, sampling_dict, collater, extra_test_loader, discriminator):
    # DEPRECATED

    # first level filter - remove subsequent duplicates
    n_runs = sampling_dict['canonical samples'].shape[1]
    n_steps = sampling_dict['canonical samples'].shape[2]
    filtered_samples = [[sampling_dict['canonical samples'][:, ii, 0]] for ii in range(n_runs)]
    filtered_samples_inds = [[0] for ii in range(n_runs)]
    for i in range(1, n_steps):
        for j in range(n_runs):
            if not all(
                    sampling_dict['canonical samples'][:, j, i] == sampling_dict['canonical samples'][:, j, i - 1]):
                filtered_samples[j].append(sampling_dict['canonical samples'][:, j, i])
                filtered_samples_inds[j].append(i)
    filtered_samples = [torch.tensor(filtered_samples[ii], requires_grad=False, dtype=torch.float32) for ii in
                        range(n_runs)]
    filtered_samples_inds = [np.asarray(filtered_samples_inds[ii]) for ii in range(n_runs)]
    filtered_samples_scores = [np.asarray(sampling_dict['scores'][ii, filtered_samples_inds[ii]]) for ii in
                               range(n_runs)]

    all_filtered_samples = np.concatenate(filtered_samples)
    all_filtered_samples_scores = np.concatenate(filtered_samples_scores)
    dists = torch.cdist(torch.Tensor(all_filtered_samples), torch.Tensor(all_filtered_samples)).detach().numpy()

    model = AgglomerativeClustering(distance_threshold=1, linkage="average", affinity='euclidean', n_clusters=None)
    model = model.fit(all_filtered_samples)
    n_clusters = model.n_clusters_
    classes = model.labels_

    '''
    visualize classwise distances
    '''
    class_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if j >= i:
                class_distances[i, j] = np.mean(dists[classes == i][:, classes == j])

    # #plot the top three levels of the dendrogram
    # plt.clf()
    # plt.subplot(1,2,1)
    # plot_dendrogram(model, truncate_mode="level", p=3)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
    # plt.subplot(1,2,2)
    # plt.imshow(class_distances)

    '''
    pick out best samples in each class with reasonably good scoress
    '''
    best_samples = np.zeros((n_clusters, 12))
    best_samples_scores = np.zeros((n_clusters))
    for i in range(n_clusters):
        best_ind = np.argmax(all_filtered_samples_scores[classes == i])
        best_samples_scores[i] = all_filtered_samples_scores[classes == i][best_ind]
        best_samples[i] = all_filtered_samples[classes == i][best_ind]

    sort_inds = np.argsort(best_samples_scores)
    best_samples = best_samples[sort_inds]
    best_samples_scores = best_samples_scores[sort_inds]

    n_samples_to_build = min(100, len(best_samples))
    best_samples_to_build = best_samples[:n_samples_to_build]
    single_mol_data_0 = extra_test_loader.dataset[0]
    big_single_mol_data = collater([single_mol_data_0 for n in range(n_samples_to_build)]).cuda()
    override_sg_ind = list(supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
    sym_ops_list = [torch.Tensor(supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(
        big_single_mol_data.x.device) for i in range(big_single_mol_data.num_graphs)]
    big_single_mol_data = write_sg_to_all_crystals('P-1', supercell_builder.dataDims, big_single_mol_data,
                                                   supercell_builder.symmetries_dict, sym_ops_list)

    best_cells, _, _ = supercell_builder.build_supercells(big_single_mol_data,
                                                          torch.tensor(best_samples_to_build, device='cuda',
                                                                            dtype=torch.float32),
                                                          supercell_size=config.supercell_size,
                                                          graph_convolution_cutoff=config.discriminator.graph_convolution_cutoff,
                                                          align_molecules=True,
                                                          skip_cell_cleaning=True,
                                                          rescale_asymmetric_unit=False,
                                                          standardized_sample=True, )

    assert np.mean(np.abs(best_cells.cell_params.cpu().detach().numpy() - (
            best_samples_to_build * supercell_builder.dataDims['lattice stds'] +
            supercell_builder.dataDims['lattice means']))) < 1e-4
    ss = softmax_and_score(discriminator(best_cells.clone().cuda())).cpu().detach().numpy()

    # mols = [ase_mol_from_crystaldata(best_cells, ii, highlight_aux=True, exclusion_level='distance', inclusion_distance=5) for ii in range(best_cells.num_graphs)]
    # view(mols)

    return best_samples, best_samples_scores, best_cells.cpu().detach()