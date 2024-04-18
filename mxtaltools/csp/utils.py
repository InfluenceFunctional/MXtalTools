import ase.io
import numpy as np
import torch
import tqdm
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering

import constants.asymmetric_units
from mxtaltools.common.geometry_calculations import cell_vol_torch
from mxtaltools.common.utils import compute_rdf_distance
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.crystal_building.utils import update_crystal_symmetry_elements, DEPRECATED_write_sg_to_all_crystals
from mxtaltools.models.crystal_rdf import crystal_rdf
from mxtaltools.models.utils import softmax_and_score, undo_1d_bound
from mxtaltools.models.vdw_overlap import vdw_overlap


def compute_csp_sample_distances(config, real_samples_dict, generated_samples_dict, num_crystals, num_samples, rr):
    """compute various distances"""
    """rdf distances"""
    # for i in range(num_crystals):
    #     for j in tqdm.tqdm(range(num_samples)):
    #         for k in range(j, num_samples):
    #             intra_sample_rdf_distance[i, j, k] = compute_rdf_distance(generated_samples_dict['RDF'][i][j], generated_samples_dict['RDF'][i][k], rr)
    # intra_sample_rdf_distance = intra_sample_rdf_distance + np.moveaxis(intra_sample_rdf_distance, (1, 2), (2, 1))  # add lower diagonal (distance matrix is symmetric)

    intra_sample_rdf_distance = np.zeros((num_crystals, num_samples, num_samples))
    rdfs = torch.Tensor(generated_samples_dict['RDF'][0]).to(config.device)
    rrc = torch.Tensor(rr).to(config.device)
    for i in range(num_crystals):  # much faster in parallel and on cuda
        for j in tqdm.tqdm(range(num_samples)):
            intra_sample_rdf_distance[i, j] = compute_rdf_distance(rdfs[j], rdfs, rrc, num_samples).cpu().detach().numpy()

    real_sample_rdf_distance = np.zeros((num_crystals, num_samples))
    for i in range(num_crystals):
        for j in range(num_samples):
            real_sample_rdf_distance[i, j] = compute_rdf_distance(real_samples_dict['RDF'][i], generated_samples_dict['RDF'][i][j], rr)

    """cell parameter and discriminator latent distances"""
    intra_sample_cell_distance = np.zeros((num_crystals, num_samples, num_samples))
    intra_sample_latent_distance = np.zeros((num_crystals, num_samples, num_samples))
    std_sample_cell_params = (generated_samples_dict['cell params'] - config.dataDims['lattice_means']) / config.dataDims['lattice_stds']

    for i in range(num_crystals):  # dot product - it's normed
        x1 = torch.Tensor(std_sample_cell_params[i])
        x2 = torch.Tensor(generated_samples_dict['discriminator latent'][i])
        intra_sample_cell_distance[i] = (torch.cdist(x1, x1) / torch.outer(torch.linalg.norm(x1, dim=-1), torch.linalg.norm(x1, dim=-1))).numpy()
        intra_sample_latent_distance[i] = (torch.cdist(x2, x2) / torch.outer(torch.linalg.norm(x2, dim=-1), torch.linalg.norm(x2, dim=-1))).numpy()

    real_sample_cell_distance = np.zeros((num_crystals, num_samples))
    real_sample_latent_distance = np.zeros((num_crystals, num_samples))
    std_real_cell_params = (real_samples_dict['cell params'] - config.dataDims['lattice_means']) / config.dataDims['lattice_stds']
    for i in range(num_crystals):
        x1 = torch.Tensor(std_sample_cell_params[i])
        x2 = torch.Tensor(generated_samples_dict['discriminator latent'][i])
        y1 = torch.Tensor(std_real_cell_params[i])[None, :]
        y2 = torch.Tensor(real_samples_dict['discriminator latent'][i])[None, :]
        real_sample_cell_distance[i] = (torch.cdist(y1, x1) / torch.outer(torch.linalg.norm(y1, dim=-1), torch.linalg.norm(x1, dim=-1))).numpy()
        real_sample_latent_distance[i] = (torch.cdist(y2, x2) / torch.outer(torch.linalg.norm(y2, dim=-1), torch.linalg.norm(x2, dim=-1))).numpy()

    real_dists_dict = {
        'real_sample_rdf_distance': real_sample_rdf_distance,
        'real_sample_cell_distance': real_sample_cell_distance,
        'real_sample_latent_distance': real_sample_latent_distance
    }
    intra_dists_dict = {
        'intra_sample_rdf_distance': intra_sample_rdf_distance,
        'intra_sample_cell_distance': intra_sample_cell_distance,
        'intra_sample_latent_distance': intra_sample_latent_distance
    }

    return real_dists_dict, intra_dists_dict


def mini_csp_rdf_and_distance_analysis(scores_dict, real_data, sampling_dict, real_samples_dict, discriminator, config, supercell_builder, min_k=10):
    reconstructed_best_scores, best_supercells_list, best_rdfs, best_scores_dict, best_samples_latents, rr, topk_inds = \
        rebuild_topk_crystals(list(scores_dict.keys()), scores_dict, real_samples_dict,
                              sampling_dict, real_data, discriminator, config, supercell_builder, min_k=min_k)

    return reconstructed_best_scores, best_scores_dict, best_supercells_list, topk_inds, best_rdfs, rr


def log_best_mini_csp_samples(num_crystals, num_samples,
                              reconstructed_best_scores, best_rdfs, best_scores_dict, best_supercells_list, rr,
                              config, wandb, generated_samples_dict, real_samples_dict, real_data, mol_volume_ind, sym_info, vdw_radii):
    """
    extract the best guesses for each crystal and reconstruct and analyze them
    compare best samples to the experimental crystal
    """

    real_sample_rdf_distance, intra_sample_rdf_distance, \
        real_sample_cell_distance, intra_sample_cell_distance, \
        real_sample_latent_distance, intra_sample_latent_distance = (
        compute_csp_sample_distances(reconstructed_best_scores, real_data, best_rdfs, rr, real_samples_dict, config, generated_samples_dict, num_samples))

    '''dist vs score plot'''
    plot_mini_csp_dist_vs_score(real_sample_rdf_distance, real_sample_cell_distance, real_sample_latent_distance, reconstructed_best_scores, real_samples_dict, best_scores_dict, generated_samples_dict, wandb)

    '''visualize best samples'''
    best_supercells = best_supercells_list[-1]  # last sample was the best
    save_3d_structure_examples(wandb, best_supercells)

    """
    Overlaps & Crystal-wise summary plot
    """
    sample_wise_overlaps_and_summary_plot(config, wandb, num_crystals, best_supercells, sym_info, best_scores_dict, vdw_radii, mol_volume_ind)

    """
    CSP Funnels
    """
    sample_density_funnel_plot(config, wandb, best_supercells, generated_samples_dict, real_samples_dict)
    sample_rdf_funnel_plot(config, wandb, best_supercells, reconstructed_best_scores, real_samples_dict, real_sample_rdf_distance)

    return (real_sample_rdf_distance, intra_sample_rdf_distance, real_sample_cell_distance, intra_sample_cell_distance,
            real_sample_latent_distance, intra_sample_latent_distance, reconstructed_best_scores, best_scores_dict, best_supercells_list, best_rdfs, rr)


def log_csp_summary_stats(wandb, generated_samples_dict, sym_info):
    unique_space_group_inds = np.unique(generated_samples_dict['space group'].flatten())
    n_space_groups = len(unique_space_group_inds)
    space_groups = np.asarray([sym_info['space_groups'][sg] for sg in generated_samples_dict['space group'].flatten()])
    unique_space_groups = np.asarray([sym_info['space_groups'][sg] for sg in unique_space_group_inds])

    '''
    overall and SG-wise mean scores
    fraction below certain cutoffs
    '''
    score_dict = {}
    for label in ['score', 'vdw overlap', 'density']:
        all_scores = generated_samples_dict[label]

        for k in range(n_space_groups):
            sg_wise_score = all_scores.flatten()[space_groups == unique_space_groups[k]]
            score_dict[f"Mini-CSP {unique_space_groups[k]} average {label}"] = np.average(sg_wise_score)

            if label == 'vdw overlap':
                good_fraction = np.average(sg_wise_score < 0.05)
                decent_fraction = np.average(sg_wise_score < 0.1)
                score_dict[f"Mini-CSP {unique_space_groups[k]} {label} fraction below 0.05"] = good_fraction
                score_dict[f"Mini-CSP {unique_space_groups[k]} {label} fraction below 0.1"] = decent_fraction
            elif label == 'score':
                good_fraction = np.average(sg_wise_score > 0)
                score_dict[f"Mini-CSP {unique_space_groups[k]} {label} fraction above 50%"] = good_fraction

        all_scores = generated_samples_dict[label]
        score_dict[f"Mini-CSP overall average {label}"] = np.average(all_scores)
        if label == 'vdw overlap':
            good_fraction = np.average(all_scores < 0.05)
            decent_fraction = np.average(all_scores < 0.1)
            score_dict[f"Mini-CSP {label} fraction below 0.05"] = good_fraction
            score_dict[f"Mini-CSP {label} fraction below 0.1"] = decent_fraction
        elif label == 'score':
            good_fraction = np.average(all_scores > 0)  # score function 'middle' is at 0
            score_dict[f"Mini-CSP  {label} fraction above 50%"] = good_fraction

    wandb.log(score_dict)

    return None


def sample_wise_overlaps_and_summary_plot(config, wandb, num_crystals, best_supercells, sym_info, best_scores_dict, vdw_radii, mol_volume_ind):
    num_samples = min(num_crystals, 25)
    _, vdw_score, _, _, normed_vdw_overlaps = \
        vdw_overlap(vdw_radii,
                    crystaldata=best_supercells,
                    loss_func=None)

    volumes_list = []
    for i in range(best_supercells.num_graphs):
        volumes_list.append(
            cell_vol_torch(best_supercells.cell_params[i, 0:3], best_supercells.cell_params[i, 3:6]))
    volumes = torch.stack(volumes_list)
    generated_packing_coeffs = (best_supercells.mult * best_supercells.tracking[:,
                                                    mol_volume_ind] / volumes).cpu().detach().numpy()
    target_packing = (best_supercells.y * config.dataDims['target_std'] + config.dataDims[
        'target_mean']).cpu().detach().numpy()

    fig = go.Figure()
    for i in range(min(best_supercells.num_graphs, num_samples)):
        pens = normed_vdw_overlaps[i].cpu().detach()
        fig.add_trace(go.Violin(x=pens[pens != 0], side='positive', orientation='h',
                                bandwidth=0.01, width=1, showlegend=False, opacity=1,
                                name=f'{best_supercells.identifier[i]} : ' + f'SG={sym_info["space_groups"][int(best_supercells.sg_ind[i])]} <br /> ' +
                                     f'c_t={target_packing[i]:.2f} c_p={generated_packing_coeffs[i]:.2f} <br /> ' +
                                     f'tot_norm_ov={-vdw_score[i]:.2f} <br />' +
                                     f'Score={best_scores_dict["score"][i, -1]:.2f}'
                                ),
                      )

    fig.update_layout(width=800, height=800, font=dict(size=12), xaxis_range=[0, 2])
    fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Nonzero vdW overlaps', yaxis_title='packing_prediction')

    wandb.log({'Generated Sample Analysis': fig})

    return None


def sample_density_funnel_plot(config, wandb, num_crystals, identifiers, sampling_dict, real_samples_dict):
    num_reporting_samples = min(25, num_crystals)
    n_rows = int(np.ceil(np.sqrt(num_reporting_samples)))
    n_cols = int(n_rows)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(identifiers)[:num_reporting_samples],
                        x_title='Packing Coefficient', y_title='Model Score')
    for ii in range(num_reporting_samples):
        row = ii // n_cols + 1
        col = ii % n_cols + 1
        x = sampling_dict['density'][ii]
        y = sampling_dict['score'][ii]
        z = sampling_dict['vdw overlap'][ii]
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', marker=dict(color=z, colorbar=dict(title="vdW Overlap"), cmin=0, cmax=np.amax(sampling_dict['vdw overlap']), opacity=0.5, colorscale="viridis"), opacity=1),
                      row=row, col=col)

        fig.add_trace(go.Scattergl(x=[real_samples_dict['density'][ii]], y=[real_samples_dict['score'][ii]],
                                   mode='markers', marker=dict(color=[real_samples_dict['vdw overlap'][ii]], colorscale='viridis', size=20,
                                                               colorbar=dict(title="vdW Overlap"), cmin=0, cmax=np.amax(sampling_dict['vdw overlap'])),
                                   showlegend=False),
                      row=row, col=col)

        fig.update_xaxes(range=[min(real_samples_dict['density'][ii], np.amin(sampling_dict['density'][ii])),
                                min(1, max(real_samples_dict['density'][ii], np.amax(sampling_dict['density'][ii])))],
                         row=row, col=col)

    fig.update_yaxes(autorange="reversed")

    if config.logger.log_figures:
        wandb.log({'Density Funnel': fig})
    if (config.machine == 'local') and False:
        fig.show(renderer='browser')

    return None


def sample_rdf_funnel_plot(config, wandb, num_crystals, identifiers, reconstructed_best_scores, real_samples_dict, rdf_real_dists):
    num_reporting_samples = min(25, num_crystals)
    n_rows = int(np.ceil(np.sqrt(num_reporting_samples)))
    n_cols = int(n_rows)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(identifiers)[:num_reporting_samples],
                        x_title='log10 RDF Distance', y_title='Diff from Exp. Score')
    for ii in range(num_reporting_samples):
        row = ii // n_cols + 1
        col = ii % n_cols + 1
        x = rdf_real_dists[ii]
        y = reconstructed_best_scores[ii]
        fig.add_trace(go.Scattergl(x=np.log10(x), y=real_samples_dict['score'][ii] - y, showlegend=False,
                                   mode='markers', opacity=1),
                      row=row, col=col)

    # fig.update_layout(xaxis_title='RDF Distance', yaxis_title='Model Score')
    fig.update_xaxes(range=[-2, np.log10(np.amax(rdf_real_dists[:num_reporting_samples]) + 0.1)])
    # fig.update_yaxes(autorange="reversed")

    if config.logger.log_figures:
        wandb.log({'RDF Funnel': fig})
    if (config.machine == 'local') and False:
        fig.show(renderer='browser')

    return None


def save_3d_structure_examples(wandb, generated_supercell_examples):
    num_samples = min(25, generated_supercell_examples.num_graphs)
    identifiers = [generated_supercell_examples.identifier[i] for i in range(num_samples)]
    sgs = [str(int(generated_supercell_examples.sg_ind[i])) for i in range(num_samples)]

    crystals = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                         index=i, exclusion_level='distance', inclusion_distance=4)
                for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(crystals)):
        ase.io.write(f'supercell_{identifiers[i]}.cif', crystals[i])
        wandb.log({'Generated Clusters': wandb.Molecule(open(f"supercell_{identifiers[i]}.cif"),
                                                        caption=identifiers[i] + ' ' + sgs[i])})

    unit_cells = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                           index=i, exclusion_level='unit cell')
                  for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(crystals)):
        ase.io.write(f'unit_cell_{identifiers[i]}.cif', unit_cells[i])
        wandb.log({'Generated Unit Cells': wandb.Molecule(open(f"unit_cell_{identifiers[i]}.cif"),
                                                          caption=identifiers[i] + ' ' + sgs[i])})

    unit_cells_inside = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                                  index=i, exclusion_level='inside cell')
                         for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(crystals)):
        ase.io.write(f'unit_cell_inside_{identifiers[i]}.cif', unit_cells_inside[i])
        wandb.log({'Generated Unit Cells 2': wandb.Molecule(open(f"unit_cell_inside_{identifiers[i]}.cif"),
                                                            caption=identifiers[i] + ' ' + sgs[i])})

    mols = [ase_mol_from_crystaldata(generated_supercell_examples,
                                     index=i, exclusion_level='conformer')
            for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(mols)):
        ase.io.write(f'conformer_{identifiers[i]}.cif', mols[i])
        wandb.log({'Single Conformers': wandb.Molecule(open(f"conformer_{identifiers[i]}.cif"), caption=identifiers[i])})

    return None


def rebuild_topk_crystals(scores_list, scores_dict, real_samples_dict, sampling_dict, real_data, discriminator, config, supercell_builder, min_k=10):
    # identify the best samples (later, use clustering to filter down to a diverse set)

    num_crystals, num_samples = scores_dict['score'].shape

    topk_size = min(min_k, sampling_dict['score'].shape[1])
    sort_inds = sampling_dict['score'].argsort(axis=-1)[:, -topk_size:]  #
    best_scores_dict = {key: np.asarray([sampling_dict[key][ii, sort_inds[ii]] for ii in range(num_crystals)]) for key in scores_list}
    best_samples = np.asarray([sampling_dict['cell params'][ii, sort_inds[ii], :] for ii in range(num_crystals)])
    best_samples_space_groups = np.asarray([sampling_dict['space group'][ii, sort_inds[ii]] for ii in range(num_crystals)])
    best_samples_handedness = np.asarray([sampling_dict['handedness'][ii, sort_inds[ii]] for ii in range(num_crystals)])
    best_samples_latents = np.zeros_like(np.asarray([sampling_dict['discriminator latent'][ii, sort_inds[ii]] for ii in range(num_crystals)]))

    # reconstruct the best samples from the cell params
    best_supercells_list = []
    best_supercell_scores = []
    best_supercell_rdfs = []

    rdf_bins = 100
    rdf_range = [0, 10]

    discriminator.eval()
    with torch.no_grad():
        for n in tqdm.tqdm(range(topk_size)):
            real_data_i = real_data.clone()

            real_data_i = update_crystal_symmetry_elements(real_data_i, best_samples_space_groups[:, n], supercell_builder.symmetries_dict, randomize_sgs=False)

            fake_supercell_data = supercell_builder.build_supercells(
                real_data_i,
                torch.tensor(best_samples[:, n, :], device=real_data_i.x.device, dtype=torch.float32),
                config.supercell_size,
                config.discriminator.graph_convolution_cutoff,
                align_to_standardized_orientation=True,  # recorded cell params in standardized basis
                target_handedness=best_samples_handedness[:, n])

            output, extra_outputs = discriminator(fake_supercell_data.clone(), return_dists=True, return_latent=True)  # reshape output from flat filters to channels * filters per channel
            best_supercell_scores.append(softmax_and_score(output).cpu().detach().numpy())

            rdf, rr, dist_dict = crystal_rdf(fake_supercell_data, rrange=rdf_range, bins=rdf_bins, raw_density=True, atomwise=True, mode='intermolecular', cpu_detach=True)
            best_supercell_rdfs.append(rdf)

            best_supercells_list.append(fake_supercell_data.cpu().detach())

            best_samples_latents[:, n, :] = extra_outputs['final_activation']

    reconstructed_best_scores = np.asarray(best_supercell_scores).T
    # todo add even more robustness around these
    print(f'cell reconstruction mean score difference = {np.mean(np.abs(best_scores_dict["score"] - reconstructed_best_scores)):.4f}')  # should be ~0
    print(f'cell reconstruction median score difference = {np.median(np.abs(best_scores_dict["score"] - reconstructed_best_scores)):.4f}')  # should be ~0
    print(f'cell reconstruction 95% quantile score difference = {np.quantile(np.abs(best_scores_dict["score"] - reconstructed_best_scores), .95):.4f}')  # should be ~0

    best_rdfs = [np.stack([best_supercell_rdfs[ii][jj] for ii in range(topk_size)]) for jj in range(real_data.num_graphs)]

    return reconstructed_best_scores, best_supercells_list, best_rdfs, best_scores_dict, best_samples_latents, rr, sort_inds


def plot_mini_csp_dist_vs_score(real_sample_rdf_distance, real_sample_cell_distance, real_sample_latent_distance, generated_samples_dict, real_samples_dict, wandb):
    fig = make_subplots(rows=1, cols=3)

    y = (generated_samples_dict['score'] - real_samples_dict['score'][:, None]).flatten()
    z = generated_samples_dict['vdw overlap'].flatten()
    fig.add_trace(go.Scattergl(x=np.log10(real_sample_rdf_distance.flatten()), y=y, opacity=0.75, mode='markers', marker=dict(color=z, colorscale='viridis',
                                                                                                                      colorbar=dict(title="vdW Overlap"), cmin=0, cmax=np.amax(generated_samples_dict['vdw overlap'].flatten()))),
                  row=1, col=1)

    y = (generated_samples_dict['score'] - real_samples_dict['score'][:, None]).flatten()
    z = generated_samples_dict['vdw overlap'].flatten()

    fig.add_trace(go.Scattergl(x=np.log10(real_sample_cell_distance.flatten()), y=y, opacity=0.75, mode='markers', marker=dict(color=z, colorscale='viridis',
                                                                                                                       colorbar=dict(title="vdW Overlap"), cmin=0, cmax=np.amax(z))),
                  row=1, col=2)

    fig.add_trace(go.Scattergl(x=np.log10(real_sample_latent_distance.flatten()), y=y, opacity=0.75, mode='markers', marker=dict(color=z, colorscale='viridis',
                                                                                                                         colorbar=dict(title="vdW Overlap"), cmin=0, cmax=np.amax(z))),
                  row=1, col=3)

    fig.update_layout(yaxis_title='Sample vs. exp score diff', showlegend=False)
    fig.update_xaxes(title_text='log10 rdf distance to exp', row=1, col=1)
    fig.update_xaxes(title_text='log10 cell params distance to exp', row=1, col=2)
    fig.update_xaxes(title_text='log10 latent distance to exp', row=1, col=3)

    wandb.log({"Sample RDF vs. Score": fig})


def de_clean_samples(supercell_builder, samples, sg_inds):
    means = supercell_builder.dataDims['lattice_means']
    stds = supercell_builder.dataDims['lattice_stds']

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
    big_single_mol_data = DEPRECATED_write_sg_to_all_crystals('P-1', supercell_builder.dataDims, big_single_mol_data,
                                                              supercell_builder.symmetries_dict, sym_ops_list)

    best_cells, _ = supercell_builder.build_supercells(big_single_mol_data,
                                                          torch.tensor(best_samples_to_build, device='cuda',
                                                                            dtype=torch.float32),
                                                          supercell_size=config.supercell_size,
                                                          graph_convolution_cutoff=config.discriminator.graph_convolution_cutoff,
                                                          align_to_standardized_orientation=True,
                                                          skip_cell_cleaning=True,
                                                          rescale_asymmetric_unit=False,
                                                          standardized_sample=True, )

    assert np.mean(np.abs(best_cells.cell_params.cpu().detach().numpy() - (
            best_samples_to_build * supercell_builder.dataDims['lattice_stds'] +
            supercell_builder.dataDims['lattice_means']))) < 1e-4
    ss = softmax_and_score(discriminator(best_cells.clone().cuda())).cpu().detach().numpy()

    # mols = [ase_mol_from_crystaldata(best_cells, ii, highlight_aux=True, exclusion_level='distance', inclusion_distance=5) for ii in range(best_cells.num_graphs)]
    # view(mols)

    return best_samples, best_samples_scores, best_cells.cpu().detach()
