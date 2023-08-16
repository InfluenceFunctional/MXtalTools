import ase.io
import numpy as np
import torch
import tqdm
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from common.geometry_calculations import cell_vol_torch
from common.utils import ase_mol_from_crystaldata, compute_rdf_distance
from crystal_building.utils import update_crystal_symmetry_elements
from models.crystal_rdf import crystal_rdf
from models.utils import softmax_and_score
from models.vdw_overlap import vdw_overlap


def log_best_mini_csp_samples(config, wandb, discriminator, sampling_dict, real_samples_dict, real_data, supercell_builder, mol_volume_ind, sym_info, vdw_radii):
    """
    extract the best guesses for each crystal and reconstruct and analyze them
    compare best samples to the experimental crystal
    """
    scores_list = ['score', 'vdw overlap', 'h bond score', 'density']
    if 'distortion_size' in sampling_dict.keys():
        scores_list += ['distortion_size']
    scores_dict = {key: sampling_dict[key] for key in scores_list}
    num_crystals, num_samples = scores_dict['score'].shape

    sample_rdf_dists, rdf_real_dists, reconstructed_best_scores, best_supercells_list, best_rdfs, best_scores_dict = \
        topk_mini_csp_analysis(scores_list, scores_dict, real_samples_dict,
                               sampling_dict, real_data, discriminator, config, supercell_builder, min_k=10)

    '''dist vs score plot'''
    plot_mini_csp_dist_vs_score(rdf_real_dists, reconstructed_best_scores, real_samples_dict, wandb)

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
    sample_csp_funnel_plot(config, wandb, best_supercells, sampling_dict, real_samples_dict)
    sample_wise_rdf_funnel_plot(config, wandb, best_supercells, reconstructed_best_scores, real_samples_dict, rdf_real_dists)

    return None


def sample_wise_overlaps_and_summary_plot(config, wandb, num_crystals, best_supercells, sym_info, best_scores_dict, vdw_radii, mol_volume_ind):
    num_samples = min(num_crystals, 25)
    vdw_loss, normed_vdw_loss, vdw_penalties = \
        vdw_overlap(vdw_radii, crystaldata=best_supercells, return_atomwise=True, return_normed=True,
                    graph_sizes=best_supercells.tracking[:,
                                config.dataDims['tracking features dict'].index('molecule num atoms')])
    vdw_loss /= best_supercells.tracking[:,
                config.dataDims['tracking features dict'].index('molecule num atoms')]

    volumes_list = []
    for i in range(best_supercells.num_graphs):
        volumes_list.append(
            cell_vol_torch(best_supercells.cell_params[i, 0:3], best_supercells.cell_params[i, 3:6]))
    volumes = torch.stack(volumes_list)
    generated_packing_coeffs = (best_supercells.Z * best_supercells.tracking[:,
                                                    mol_volume_ind] / volumes).cpu().detach().numpy()
    target_packing = (best_supercells.y * config.dataDims['target std'] + config.dataDims[
        'target mean']).cpu().detach().numpy()

    fig = go.Figure()
    for i in range(min(best_supercells.num_graphs, num_samples)):
        pens = vdw_penalties[i].cpu().detach()
        fig.add_trace(go.Violin(x=pens[pens != 0], side='positive', orientation='h',
                                bandwidth=0.01, width=1, showlegend=False, opacity=1,
                                name=f'{best_supercells.csd_identifier[i]} : ' + f'SG={sym_info["space_groups"][int(best_supercells.sg_ind[i])]} <br /> ' +
                                     f'c_t={target_packing[i]:.2f} c_p={generated_packing_coeffs[i]:.2f} <br /> ' +
                                     f'tot_norm_ov={normed_vdw_loss[i]:.2f} <br />' +
                                     f'Score={best_scores_dict["score"][i, -1]:.2f}'
                                ),
                      )

    # Can only run this section with RDKit installed, which doesn't always work
    # Commented out - not that important anyway.
    # molecule = rdkit.Chem.MolFromSmiles(supercell_examples[i].smiles)
    # try:
    #     rdkit.Chem.AllChem.Compute2DCoords(molecule)
    #     rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
    #     pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(500, 500))
    #     pil_image.save('mol_img.png', 'png')
    #     # Add trace
    #     img = Image.open("mol_img.png")
    #     # Add images
    #     fig.add_layout_image(
    #         dict(
    #             source=img,
    #             xref="x domain", yref="y domain",
    #             x=.1 + (.15 * (i % 2)), y=i / 10.5 + 0.05,
    #             sizex=.15, sizey=.15,
    #             xanchor="center",
    #             yanchor="middle",
    #             opacity=0.75
    #         )
    #     )
    # except:  # ValueError("molecule was rejected by rdkit"):
    #     pass

    fig.update_layout(width=800, height=800, font=dict(size=12), xaxis_range=[0, 4])
    fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Nonzero vdW overlaps', yaxis_title='packing prediction')

    wandb.log({'Generated Sample Analysis': fig})

    return None


def sample_csp_funnel_plot(config, wandb, best_supercells, sampling_dict, real_samples_dict):
    num_crystals = best_supercells.num_graphs
    num_reporting_samples = min(25, num_crystals)
    n_rows = int(np.ceil(np.sqrt(num_reporting_samples)))
    n_cols = int(n_rows)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(best_supercells.csd_identifier)[:num_reporting_samples],
                        x_title='Packing Coefficient', y_title='Model Score')
    for ii in range(num_reporting_samples):
        row = ii // n_cols + 1
        col = ii % n_cols + 1
        x = sampling_dict['density'][ii]
        y = sampling_dict['score'][ii]
        z = sampling_dict['vdw overlap'][ii]
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', marker=dict(color=z, colorbar=dict(title="vdW Overlap"), cmin=0, cmax=4, opacity=0.5, colorscale="rainbow"), opacity=1),
                      row=row, col=col)

        fig.add_trace(go.Scattergl(x=[real_samples_dict['density'][ii]], y=[real_samples_dict['score'][ii]],
                                   mode='markers', marker=dict(color=[real_samples_dict['vdw overlap'][ii]], colorscale='rainbow', size=10,
                                                               colorbar=dict(title="vdW Overlap"), cmin=0, cmax=4),
                                   showlegend=False),
                      row=row, col=col)

    fig.update_xaxes(range=[np.amin(sampling_dict['density']), min(1, np.amax(sampling_dict['density']))])
    fig.update_yaxes(autorange="reversed")

    if config.wandb.log_figures:
        wandb.log({'Density Funnel': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None


def sample_wise_rdf_funnel_plot(config, wandb, best_supercells, reconstructed_best_scores, real_samples_dict, rdf_real_dists):

    num_crystals = best_supercells.num_graphs
    num_reporting_samples = min(25, num_crystals)
    n_rows = int(np.ceil(np.sqrt(num_reporting_samples)))
    n_cols = int(n_rows)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(best_supercells.csd_identifier)[:num_reporting_samples],
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
    #fig.update_xaxes(range=[-0.1, np.amax(rdf_real_dists[:num_reporting_samples]) + 0.1])
    fig.update_yaxes(autorange="reversed")

    if config.wandb.log_figures:
        wandb.log({'RDF Funnel': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None


def save_3d_structure_examples(wandb, generated_supercell_examples):
    num_samples = min(25, generated_supercell_examples.num_graphs)
    identifiers = [generated_supercell_examples.csd_identifier[i] for i in range(num_samples)]
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


def topk_mini_csp_analysis(scores_list, scores_dict, real_samples_dict, sampling_dict, real_data, discriminator, config, supercell_builder, min_k=10):
    # identify the best samples (later, use clustering to filter down to a diverse set)

    num_crystals, num_samples = scores_dict['score'].shape

    topk_size = min(min_k, sampling_dict['score'].shape[1])
    sort_inds = sampling_dict['score'].argsort(axis=-1)[:, -topk_size:]  #
    best_scores_dict = {key: np.asarray([sampling_dict[key][ii, sort_inds[ii]] for ii in range(num_crystals)]) for key in scores_list}
    best_samples = np.asarray([sampling_dict['cell params'][ii, sort_inds[ii], :] for ii in range(num_crystals)])
    best_samples_space_groups = np.asarray([sampling_dict['space group'][ii, sort_inds[ii]] for ii in range(num_crystals)])
    best_samples_handedness = np.asarray([sampling_dict['handedness'][ii, sort_inds[ii]] for ii in range(num_crystals)])

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

            real_data_i = update_crystal_symmetry_elements(
                real_data_i, best_samples_space_groups[:, n],
                config.dataDims, supercell_builder.symmetries_dict, randomize_sgs=False)

            fake_supercell_data, _, _ = supercell_builder.build_supercells(
                real_data_i,
                torch.tensor(best_samples[:, n, :], device=real_data_i.x.device, dtype=torch.float32),
                config.supercell_size,
                config.discriminator.graph_convolution_cutoff,
                align_molecules=True,  # recorded cell params in standardized basis
                target_handedness=best_samples_handedness[:, n])

            output, extra_outputs = discriminator(fake_supercell_data.clone(), return_dists=True)  # reshape output from flat filters to channels * filters per channel
            best_supercell_scores.append(softmax_and_score(output).cpu().detach().numpy())

            rdf, rr, dist_dict = crystal_rdf(fake_supercell_data, rrange=rdf_range, bins=rdf_bins, raw_density=True, atomwise=True, mode='intermolecular', cpu_detach=True)
            best_supercell_rdfs.append(rdf)

            best_supercells_list.append(fake_supercell_data.cpu().detach())

    reconstructed_best_scores = np.asarray(best_supercell_scores).T
    # todo add even more robustness around these
    print(f'cell reconstruction mean score difference = {np.mean(np.abs(best_scores_dict["score"] - reconstructed_best_scores)):.4f}')  # should be ~0
    print(f'cell reconstruction median score difference = {np.median(np.abs(best_scores_dict["score"] - reconstructed_best_scores)):.4f}')  # should be ~0
    print(f'cell reconstruction 95% quantile score difference = {np.quantile(np.abs(best_scores_dict["score"] - reconstructed_best_scores), .95):.4f}')  # should be ~0

    best_rdfs = [np.stack([best_supercell_rdfs[ii][jj] for ii in range(topk_size)]) for jj in range(real_data.num_graphs)]

    rdf_dists = np.zeros((real_data.num_graphs, topk_size, topk_size))
    for i in range(real_data.num_graphs):
        for j in range(topk_size):
            for k in range(j, topk_size):
                rdf_dists[i, j, k] = compute_rdf_distance(best_rdfs[i][j], best_rdfs[i][k], rr)

    rdf_dists = rdf_dists + np.moveaxis(rdf_dists, (1, 2), (2, 1))  # add lower diagonal (distance matrix is symmetric)

    rdf_real_dists = np.zeros((real_data.num_graphs, topk_size))
    for i in range(real_data.num_graphs):
        for j in range(topk_size):
            rdf_real_dists[i, j] = compute_rdf_distance(real_samples_dict['RDF'][i], best_rdfs[i][j], rr)

    return rdf_dists, rdf_real_dists, reconstructed_best_scores, best_supercells_list, best_rdfs, best_scores_dict


def plot_mini_csp_dist_vs_score(rdf_real_dists, reconstructed_best_scores, real_samples_dict, wandb):
    x = rdf_real_dists.flatten()
    y = (reconstructed_best_scores - real_samples_dict['score'][:, None]).flatten()
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones_like(x)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=dict(color=z)))
    fig.update_layout(xaxis_title='RDF Distance From Target', yaxis_title='Sample vs. experimental score difference', showlegend=False)
    wandb.log({"Sample RDF vs. Score": fig})
