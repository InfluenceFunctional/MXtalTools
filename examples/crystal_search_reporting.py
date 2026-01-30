import os
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch
from ase.spacegroup import Spacegroup
from ccdc.crystal import PackingSimilarity
from ccdc.io import CrystalReader
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.nn import functional as F
from tqdm import tqdm

from mxtaltools.analysis.crystal_rdf import crystal_rdf, compute_rdf_distance
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.common.ase_interface import get_niggli_cell
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import softmax_and_score


def csp_reporting(optimized_samples,
                  original_crystal,
                  device,
                  score_model
                  ):
    """
    Visualize results of crystal search
    """
    original_crystal_batch = collate_data_list([original_crystal])
    """analyze original crystal"""
    ref_lj_pot, ref_es_pot, ref_scaled_lj_pot, original_cluster_batch = (
        original_crystal_batch.build_and_analyze(return_cluster=True))
    ref_model_output = score_model(original_cluster_batch.to(device), force_edges_rebuild=True).cpu()
    ref_model_score = softmax_and_score(ref_model_output[:, :2])
    ref_rdf_dist_pred = F.softplus(ref_model_output[:, 2])
    ref_packing_coeff = original_cluster_batch.packing_coeff

    noisy_crystal_batch = collate_data_list([original_crystal for _ in range(25)])
    noisy_crystal_batch.noise_cell_parameters(0.05)
    noisy_lj_pot, noisy_es_pot, noisy_scaled_lj_pot, noisy_cluster_batch = (
        noisy_crystal_batch.build_and_analyze(return_cluster=True))
    noisy_model_output = score_model(noisy_cluster_batch.to(device), force_edges_rebuild=True).cpu()
    noisy_model_score = softmax_and_score(noisy_model_output[:, :2])
    noisy_rdf_dist_pred = F.softplus(noisy_model_output[:, 2])
    noisy_packing_coeff = noisy_cluster_batch.packing_coeff

    """extract sampling results"""
    optimized_crystal_batch = collate_data_list(optimized_samples)
    packing_coeff = optimized_crystal_batch.packing_coeff
    model_score = softmax_and_score(optimized_crystal_batch.model_output[:, :2])
    rdf_dist_pred = F.softplus(optimized_crystal_batch.model_output[:, 2])
    scaled_lj_pot = optimized_crystal_batch.scaled_lj
    fake_rdfs = optimized_crystal_batch.rdf

    """
    Compute true distances
    """
    original_cluster_batch = original_crystal_batch.mol2cluster().to(device)
    real_rdf, rr, _ = crystal_rdf(original_cluster_batch.to(device),
                                  original_cluster_batch.edges_dict,
                                  rrange=[0, 6], bins=2000,
                                  mode='intermolecular',
                                  elementwise=True,
                                  raw_density=True,
                                  cpu_detach=False)

    rdf_dists = torch.zeros(len(rdf_dist_pred), device=original_cluster_batch.device, dtype=torch.float32)
    for i in range(len(fake_rdfs)):
        rdf_dists[i] = compute_rdf_distance(real_rdf[0], fake_rdfs[i].to(device), rr) / \
                       optimized_crystal_batch.num_atoms[i]
    rdf_dists = rdf_dists.cpu()

    """also to noisy crystals"""

    noisy_rdfs, rr, _ = crystal_rdf(noisy_cluster_batch.to(device),
                                    noisy_cluster_batch.edges_dict,
                                    rrange=[0, 6], bins=2000,
                                    mode='intermolecular',
                                    elementwise=True,
                                    raw_density=True,
                                    cpu_detach=False)

    noisy_rdf_dists = torch.zeros(len(noisy_rdfs), device=original_cluster_batch.device, dtype=torch.float32)
    for i in range(len(noisy_rdfs)):
        noisy_rdf_dists[i] = compute_rdf_distance(real_rdf[0], noisy_rdfs[i].to(device), rr) / \
                             optimized_crystal_batch.num_atoms[i]
    noisy_rdf_dists = noisy_rdf_dists.cpu()

    """
    reduced cells and dists
    """
    if not os.path.exists('opt_lengths.pt'):
        print("Getting reduced cells")
        reduced_opt_lengths = torch.zeros((optimized_crystal_batch.num_graphs, 3), dtype=torch.float32)
        for ind in tqdm(range(len(reduced_opt_lengths))):
            reduced_opt_lengths[ind] = torch.tensor(get_niggli_cell(optimized_crystal_batch, ind)[:3])
        torch.save(reduced_opt_lengths, 'opt_lengths.pt')
    else:
        reduced_opt_lengths = torch.load('opt_lengths.pt')

    if not os.path.exists('noisy_lengths.pt'):
        reduced_noisy_lengths = torch.zeros((noisy_crystal_batch.num_graphs, 3), dtype=torch.float32)
        for ind in tqdm(range(len(reduced_noisy_lengths))):
            reduced_noisy_lengths[ind] = torch.tensor(get_niggli_cell(noisy_crystal_batch, ind)[:3])
            torch.save(reduced_noisy_lengths, 'noisy_lengths.pt')
    else:
        reduced_noisy_lengths = torch.load('noisy_lengths.pt')

    reduced_orig_cell = torch.tensor(get_niggli_cell(original_crystal_batch, 0)[:3], dtype=torch.float32)[None, :]
    c_dists = torch.cdist(reduced_orig_cell,
                          reduced_opt_lengths)[0]
    noisy_c_dists = torch.cdist(reduced_orig_cell,
                                reduced_noisy_lengths)[0]

    # metric tensor distance
    # ref_metric_tensor = (original_crystal_batch.T_fc[0] @ original_crystal_batch.T_fc[0]).cpu()
    # opt_metten_dist = torch.zeros_like(c_dists)
    # for ind in range(len(opt_metten_dist)):
    #     metric_tensor = (optimized_crystal_batch.T_fc[ind] @ optimized_crystal_batch.T_fc[ind]).cpu()
    #     opt_metten_dist[ind] = torch.sqrt(torch.sum(torch.pow(ref_metric_tensor - metric_tensor, 2)))

    """
    COMPACK analysis
    """
    #num_samples = 1000
    #best_sample_inds = torch.argsort(model_score, descending=True)[:num_samples]
    best_sample_inds = torch.argwhere((packing_coeff > 0.6) * (packing_coeff < 0.8) * (rdf_dist_pred < 0.01)).squeeze()
    #rmsds = list(np.load('rmsds_final.npy'))
    #matches = list(np.load('matches_final.npy'))
    #
    # density_funnel(model_score, packing_coeff, rdf_dists, ref_model_score, ref_packing_coeff)
    # distance_fig(c_dists, model_score, noisy_c_dists, noisy_model_score, noisy_rdf_dists, rdf_dists, packing_coeff)

    matches, rmsds = batch_compack(best_sample_inds, optimized_samples, original_cluster_batch)

    all_matched = np.argwhere(matches == 20).flatten()
    matched_rmsds = rmsds[all_matched]
    compack_fig(matches, rmsds, rdfs=rdf_dists[best_sample_inds])
    print(all_matched)
    print(rmsds[all_matched])
    aa = 1


def compack_fig(matches, rmsds, write_fig):
    fontsize = 26
    fig = go.Figure()
    fig.add_scatter(x=matches[matches > 4], y=rmsds[matches > 4],
                    mode='markers', marker_size=12,
                    marker_color='black',
                    # marker_color=rdf_dists[best_sample_inds[matches > 4]].clip(
                    #    max=torch.quantile(rdf_dists, 0.99)).log10().cpu().detach(),  # 'blue',
                    # marker_colorbar=dict(title=dict(text="log RDF EMD")),
                    # marker_colorscale='bluered', opacity=1, marker_line_width=1.5,
                    marker_line_color='white',
                    name='Optimized Samples',
                    # marker_coloraxis='coloraxis1'
                    )
    fig.update_layout(xaxis1_title='Matched Cluster Size', yaxis1_title='Matched Cluster RMSD',
                      # xaxis1_range=[0, 1]
                      )
    fig.update_annotations(font=dict(size=fontsize))
    fig.update_layout(font_size=fontsize)
    fig.update_layout(
        coloraxis1=dict(
            colorscale='bluered',
            colorbar=dict(
                title='log RDF EMD',
                #titlefont_size=18,
                #tickfont_size=18,
                # x=1
            )
        ),
    )
    fig.update_layout(legend_orientation='h')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,  # Move the legend above the plot
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100)  # Increase top margin to make space
    )
    fig.show()
    if write_fig:
        fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\mxt_code\compack_fig.png', width=900, height=900)


def batch_compack(best_sample_inds, optimized_samples, reference_cluster_batch): # todo refactor into analysis code
    # generate the crystals in ccdc format
    best_crystals_batch = collate_data_list([optimized_samples[ind] for ind in best_sample_inds])
    best_cluster_batch = best_crystals_batch.mol2cluster().to('cpu')
    _ = cluster_batch_to_ccdc_crystals(best_cluster_batch, np.arange(best_cluster_batch.num_graphs))
    mol = ase_mol_from_crystaldata(reference_cluster_batch, index=0, mode='unit cell')
    mol.info['spacegroup'] = Spacegroup(int(best_cluster_batch.sg_ind[0]), setting=1)
    mol.write('DAFMUV.cif')

    print(f"Running COMPACK on {len(best_sample_inds)} crystals")
    pool = mp.Pool(8)
    results = []
    for ind in range(len(best_sample_inds)):
        results.append(
            pool.apply_async(
                single_compack_run,
                (ind,)
            )
        )
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    matches = np.array([res[1] for res in results])
    rmsds = np.array([res[0] for res in results])


    return matches, rmsds


def single_compack_run(ind):
    ref_crystal = CrystalReader('DAFMUV.cif')[0]
    sample_crystal = CrystalReader(f'temp_{ind}.cif')[0]
    similarity_engine = PackingSimilarity()
    similarity_engine.settings.distance_tolerance = 0.4
    similarity_engine.settings.angle_tolerance = 40
    similarity_engine.settings.allow_molecular_differences = True
    similarity_engine.settings.packing_shell_size = 20
    try:
        result = similarity_engine.compare(ref_crystal, sample_crystal)
        print(f"Crystal {ind} RMSD = {result.rmsdmat:.3f} Å, {result.nmatched_molecules} mols matched")
        return result.rmsdmat, result.nmatched_molecules
    except AttributeError:
        print("Analysis failed")
        return 0, 0


def cluster_batch_to_ccdc_crystals(cluster_batch, inds):
    crystals = []
    for ind in inds:
        mol = ase_mol_from_crystaldata(cluster_batch, index=ind, mode='unit cell')
        mol.info['spacegroup'] = Spacegroup(int(cluster_batch.sg_ind[ind]), setting=1)
        mol.write(f'temp_{ind}.cif')
        crystal = CrystalReader(f'temp_{ind}.cif')[0]
        crystals.append(crystal)
    return crystals


def distance_fig(c_dists, model_score, noisy_c_dists, noisy_model_score, noisy_rdf_dists, rdf_dists, packing_coeff):
    fontsize = 26
    good_inds = torch.argwhere((packing_coeff > 0.55) * (model_score > 0) * (packing_coeff < 0.9)).squeeze()
    good_noisy_inds = torch.argwhere(noisy_model_score > 0).squeeze()
    fig = make_subplots(rows=1, cols=2, subplot_titles=['(a)', '(b)'])
    fig.add_scatter(x=rdf_dists[good_inds].log10().cpu().detach(), y=c_dists[good_inds].cpu().detach(),
                    mode='markers', marker_size=12,
                    marker_color=model_score[good_inds].cpu().detach(),  # 'blue',
                    marker_coloraxis='coloraxis',
                    opacity=0.6, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples', row=1, col=1)
    fig.add_scatter(x=noisy_rdf_dists[good_noisy_inds].log10().cpu().detach(),
                    y=noisy_c_dists[good_noisy_inds].cpu().detach(),
                    mode='markers', marker_size=12,
                    marker_color=noisy_model_score[good_noisy_inds],  # 'black',
                    marker_line_width=2,
                    marker_line_color='black',
                    marker_coloraxis='coloraxis',
                    #marker_colorscale='turbo_r',
                    opacity=0.75,
                    name='Noised Experimental Samples', row=1, col=1)
    fig.add_scatter(x=rdf_dists[good_inds].log10().cpu().detach(), y=c_dists[good_inds].cpu().detach(),
                    mode='markers', marker_size=12,
                    marker_color=model_score[good_inds].cpu().detach(),  # 'blue',
                    marker_colorbar=dict(title=dict(text="Model Score")),
                    #marker_colorscale='turbo_r',
                    opacity=0.6, marker_line_width=1,
                    marker_coloraxis='coloraxis',
                    marker_line_color='white',
                    name='Optimized Samples', row=1, col=2, showlegend=False)
    fig.add_scatter(x=noisy_rdf_dists[good_noisy_inds].log10().cpu().detach(),
                    y=noisy_c_dists[good_noisy_inds].cpu().detach(),
                    mode='markers', marker_size=12,
                    marker_color=noisy_model_score[good_noisy_inds],  # 'black',
                    marker_line_width=2,
                    marker_line_color='black',
                    marker_coloraxis='coloraxis',
                    opacity=0.75,
                    name='Noised Experimental Samples', row=1, col=2, showlegend=False)
    # fig.add_scatter(x=torch.zeros(1), y=torch.zeros(1), mode='markers', marker_color='yellow',
    #                 marker_size=25, marker_line_color='black', marker_line_width=2,
    #                 name='Experimental Sample')
    fig.update_layout(yaxis1_range=[0, 40], xaxis1_range=[-1.75, -0.75])  # 1.5])
    fig.update_layout(yaxis2_range=[0, 5], xaxis2_range=[-1.75, -.75])
    fig.update_layout(legend_orientation='h')
    fig.update_layout(xaxis1_title='log RDF EMD', yaxis1_title='Lattice Distance (Angstrom)')
    fig.update_layout(xaxis2_title='log RDF EMD', yaxis2_title='Lattice Distance (Angstrom)')
    fig.update_layout(
        coloraxis=dict(
            colorscale='turbo_r',
            colorbar=dict(
                title='Model Score',
                titlefont_size=18,
                tickfont_size=18,
                x=1.025  # default position
            )
        ))
    fig.update_annotations(font=dict(size=fontsize))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_layout(font_size=fontsize)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,  # Move the legend above the plot
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100)  # Increase top margin to make space
    )
    fig.show()
    fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\mxt_code\distance_fig.png', width=1920, height=800)


def density_funnel(model_score, packing_coeff, rdf_dists, ref_model_score, ref_packing_coeff, yaxis_title,
                   write_fig: bool=True):
    fontsize = 26
    good_inds = torch.argwhere((packing_coeff > 0.55) * (packing_coeff < 0.9)).squeeze()
    fig = go.Figure()
    fig.add_scatter(x=packing_coeff[good_inds].cpu(), y=model_score[good_inds].cpu(),
                    mode='markers', marker_size=12,
                    showlegend=True,
                    marker_color=rdf_dists[good_inds].clip(max=torch.quantile(rdf_dists, 0.99)).log10().cpu().detach(),
                    # 'blue',
                    marker_colorbar=dict(title=dict(text="log RDF EMD")),
                    marker_colorscale='bluered', opacity=0.6, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples',
                    marker_coloraxis='coloraxis2',
                    #row=1, col=2
                    )

    fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_model_score.cpu(), mode='markers', marker_color='yellow',
                    marker_size=25, marker_line_color='black', marker_line_width=2,
                    name='Experimental Sample')
    fig.update_layout(xaxis1_title='Packing Coefficient', yaxis1_title=yaxis_title,
                      )
    fig.update_annotations(font=dict(size=fontsize))
    fig.update_layout(font_size=fontsize)
    fig.update_layout(

        coloraxis2=dict(
            colorscale='bluered',
            colorbar=dict(
                title='log RDF EMD',
                #titlefont_size=18,
                #tickfont_size=18,
                x=1  # shift it to the right so it doesn't overlap
            )
        )
    )
    fig.update_layout(legend_orientation='h')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,  # Move the legend above the plot
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100)  # Increase top margin to make space
    )

    fig.show()
    if write_fig:
        fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\mxt_code\density_funnel.png', width=1000, height=800)


if __name__ == '__main__':
    device = 'cuda'
    mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
    score_checkpoint = r"../checkpoints/crystal_score.pt"
    density_checkpoint = r"../checkpoints/cp_regressor.pt"
    csp_dir_path = Path(r'C:\Users\mikem\crystals\CSP_runs\searches\mxt_paper_search')
    chunk_paths = os.listdir(csp_dir_path)
    chunk_paths = [csp_dir_path.joinpath(elem) for elem in chunk_paths if '.pt' in elem]
    score_model = load_crystal_score_model(score_checkpoint, device).to(device)
    score_model.eval()

    "load and batch batch_size copies of the same molecule"
    # BIGDOK, CETLAQ, DAFMUV, HEVTIL, DAFMUV
    example_crystals = torch.load(mini_dataset_path)
    elem_index = [elem.identifier for elem in example_crystals].index('DAFMUV')  # .index('ACRLAC06')  #.index('FEDGOK01')
    original_crystal = example_crystals[elem_index]
    optimized_samples = []
    for elem in chunk_paths:
        optimized_samples.extend(torch.load(elem))

    csp_reporting(optimized_samples, original_crystal, device, score_model)
