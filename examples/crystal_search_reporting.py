import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import plotly.graph_objects as go

from mxtaltools.analysis.crystal_rdf import new_crystal_rdf, compute_rdf_distance
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
    scaled_lj_pot = optimized_crystal_batch.scaled_lj_pot
    fake_rdfs = optimized_crystal_batch.rdf

    """
    Compute true distances
    """
    _, _, _, original_cluster_batch = original_crystal_batch.to(device).build_and_analyze(
        return_cluster=True, cutoff=6)
    real_rdf, rr, _ = new_crystal_rdf(original_cluster_batch.to(device),
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

    c_dists = torch.cdist(original_crystal_batch.standardize_cell_parameters().cpu()[:, :6],
                          optimized_crystal_batch.standardize_cell_parameters().cpu()[:, :6])[0]

    """also to noisy crystals"""

    noisy_rdfs, rr, _ = new_crystal_rdf(noisy_cluster_batch.to(device),
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
    noisy_df_dists = noisy_rdf_dists.cpu()

    noisy_c_dists = torch.cdist(original_crystal_batch.standardize_cell_parameters().cpu()[:, :6],
                                noisy_crystal_batch.standardize_cell_parameters().cpu()[:, :6])[0]
    """
    Density/score plot
    """

    fig = go.Figure()
    fig.add_scatter(x=packing_coeff.cpu(), y=model_score.cpu(),
                    mode='markers', marker_size=8,
                    marker_color=rdf_dists.clip(max=torch.quantile(rdf_dists, 0.99)).log10().cpu().detach(),  #'blue',
                    marker_colorbar=dict(title=dict(text="log RDF EMD")),
                    marker_colorscale='bluered', opacity=0.75, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples')
    #
    # fig.add_scatter(x=noisy_packing_coeff.cpu(), y=noisy_model_score.cpu(),
    #                 mode='markers', marker_size=12,
    #                 opacity=0.75,
    #                 marker_color='green',
    #                 name='Noised Ground Truth')

    fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_model_score.cpu(), mode='markers', marker_color='yellow',
                    marker_size=25, marker_line_color='black', marker_line_width=2,
                    name='Experimental Sample')
    fig.update_layout(xaxis_title='Packing Coefficient', yaxis_title='Model Score',
                      xaxis_range=[0, 1]
                      )
    fig.update_annotations(font=dict(size=18))
    fig.update_layout(legend_orientation='h')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.show()
    fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\mxt_code\density_funnel.png', width=800, height=600)

    """
    Distances plot
    """

    fig = go.Figure()
    fig.add_scatter(x=rdf_dists.cpu().detach(), y=c_dists.cpu().detach(),
                    mode='markers', marker_size=8,
                    marker_color=model_score.cpu().detach(),  # 'blue',
                    marker_colorbar=dict(title=dict(text="Model Score")),
                    marker_colorscale='bluered_r', opacity=0.75, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples')
    fig.add_scatter(x=noisy_rdf_dists.cpu().detach(), y=noisy_c_dists.cpu().detach(),
                    mode='markers', marker_size=12,
                    marker_color='green',
                    opacity=0.75,
                    name='Noised Experimental Sample')
    # fig.add_scatter(x=torch.zeros(1), y=torch.zeros(1), mode='markers', marker_color='yellow',
    #                 marker_size=25, marker_line_color='black', marker_line_width=2,
    #                 name='Experimental Sample')
    fig.update_layout(yaxis_range=[-np.inf, 1], xaxis_range=[np.log10(noisy_rdf_dists.cpu().detach().amin()), 0])
    fig.update_layout(legend_orientation='h')
    fig.update_layout(xaxis_title='RDF EMD', yaxis_title='Lattice Distance')
    fig.update_annotations(font=dict(size=18))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True, type="log")
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True, type="log")
    fig.show()
    fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\mxt_code\distance_fig.png', width=800, height=600)
    #
    # """
    # 3D cell fig
    # """
    # quant = 50 / len(optimized_samples)
    # good_inds = model_score >= torch.quantile(model_score, 1 - quant)
    # fig = go.Figure()
    # fig.add_scatter3d(
    #     x=optimized_crystal_batch.cell_lengths[good_inds, 0].cpu().detach(),
    #     y=optimized_crystal_batch.cell_lengths[good_inds, 1].cpu().detach(),
    #     z=optimized_crystal_batch.cell_lengths[good_inds, 2].cpu().detach(),
    #     marker_color=rdf_dists[good_inds].cpu().detach(),
    #     mode='markers',
    #     opacity=.5,
    #     marker_size=8,
    #     marker_colorscale='bluered_r',
    #     marker_colorbar=dict(title=dict(text="Model Score")),
    #
    # )
    # fig.add_scatter3d(
    #     x=original_crystal_batch.cell_lengths[:, 0].cpu().detach(),
    #     y=original_crystal_batch.cell_lengths[:, 1].cpu().detach(),
    #     z=original_crystal_batch.cell_lengths[:, 2].cpu().detach(),
    #     mode='markers',
    #     opacity=1,
    #     marker_color='yellow',
    #     marker_size=16,
    # )
    # # fig.update_layout(
    # #     scene=dict(
    # #         xaxis=dict(range=[0, 50]),
    # #         yaxis=dict(range=[0, 50]),
    # #         zaxis=dict(range=[0, 50])
    # #     )
    # # )
    # fig.show()

    """
    Visualize best clusters
    """
    best_samples = torch.argsort(model_score, descending=True)
    best_crystals_batch = collate_data_list([optimized_samples[ind] for ind in best_samples[:5]])
    _, _, _, best_cluster_batch = best_crystals_batch.to('cpu').build_and_analyze(return_cluster=True)
    ase_mols = best_cluster_batch.visualize(mode='unit cell')
    from ase.io import write
    for ind, mol in enumerate(ase_mols):
        write(f'samples{ind}.png', mol)

    original_cluster_batch.visualize([0], mode='unit cell')

    from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
    from ccdc.io import CrystalReader
    from ase.spacegroup import Spacegroup
    ref_crystal = CrystalReader('temp.cif')[0]

    from ccdc.crystal import PackingSimilarity
    similarity_engine = PackingSimilarity()

    def cluster_batch_to_ccdc_crystals(cluster_batch, inds):
        crystals = []
        for ind in inds:
            mol = ase_mol_from_crystaldata(cluster_batch, index=ind, mode='unit cell')
            mol.info['spacegroup'] = Spacegroup(int(cluster_batch.sg_ind[ind]), setting=1)
            mol.write('temp.cif')
            crystal = CrystalReader('temp.cif')[0]
            crystals.append(crystal)
        return crystals

    best_crystals = cluster_batch_to_ccdc_crystals(best_cluster_batch, np.arange(best_cluster_batch.num_graphs))
    noised_crystals = cluster_batch_to_ccdc_crystals(noisy_cluster_batch, [0, 1, 2, 3, 4, 5])
    mol = ase_mol_from_crystaldata(original_cluster_batch, index=0, mode='unit cell')
    mol.info['spacegroup'] = Spacegroup(int(best_cluster_batch.sg_ind[ind]), setting=1)
    mol.write('temp.cif')

    similarity_engine.settings.distance_tolerance = 0.4
    similarity_engine.settings.angle_tolerance = 40
    similarity_engine.settings.allow_molecular_differences = True
    similarity_engine.settings.packing_shell_size = 20

    for crystal in best_crystals:
        result = similarity_engine.compare(ref_crystal, crystal)
        print(f"RMSD = {result.rmsd:.3f} Å, {result.nmatched_molecules} mols matched")

    for crystal in noised_crystals:
        result = similarity_engine.compare(ref_crystal, crystal)
        print(f"RMSD = {result.rmsd:.3f} Å, {result.nmatched_molecules} mols matched")


if __name__ == '__main__':
    device = 'cuda'
    mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
    score_checkpoint = r"../checkpoints/crystal_score.pt"
    density_checkpoint = r"../checkpoints/cp_regressor.pt"
    csp_dir_path = Path(r'C:\Users\mikem\crystals\CSP_runs\searches\mxt_paper_search')
    chunk_paths = os.listdir(csp_dir_path)
    chunk_paths = [csp_dir_path.joinpath(elem) for elem in chunk_paths]
    score_model = load_crystal_score_model(score_checkpoint, device).to(device)
    score_model.eval()

    "load and batch batch_size copies of the same molecule"
    # BIGDOK, CETLAQ, DAFMUV, HEVTIL, DAFMUV
    example_crystals = torch.load(mini_dataset_path)
    elem_index = [elem.identifier for elem in example_crystals].index(
        'DAFMUV')  # .index('ACRLAC06')  #.index('FEDGOK01')
    original_crystal = example_crystals[elem_index]
    optimized_samples = []
    for elem in chunk_paths:
        optimized_samples.extend(torch.load(elem))

    csp_reporting(optimized_samples, original_crystal, device, score_model)
