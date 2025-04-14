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

    """analyze original crystal"""
    ref_lj_pot, ref_es_pot, ref_scaled_lj_pot, original_cluster_batch = (
        original_crystal.build_and_analyze(return_cluster=True))
    ref_model_output = score_model(original_cluster_batch.to(device), force_edges_rebuild=True).cpu()
    ref_model_score = softmax_and_score(ref_model_output[:, :2])
    ref_rdf_dist_pred = F.softplus(ref_model_output[:, 2])
    ref_packing_coeff = original_cluster_batch.packing_coeff

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
    _, _, _, original_cluster_batch = original_crystal.to(device).build_and_analyze(
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

    c_dists = torch.cdist(original_crystal.standardize_cell_parameters().cpu()[:, :6],
                          optimized_crystal_batch.standardize_cell_parameters().cpu()[:, :6])[0]
    """
    Density/score plot
    """

    fig = go.Figure()
    fig.add_scatter(x=packing_coeff.cpu(), y=model_score.cpu(),
                    mode='markers', marker_size=12,
                    marker_color=rdf_dists.log10().cpu().detach(),#'blue',
                    marker_colorbar=dict(title=dict(text="log RDF EMD")),
                    marker_colorscale='bluered', opacity=0.75, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples')
    fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_model_score.cpu(), mode='markers', marker_color='yellow',
                    marker_size=25, marker_line_color='black', marker_line_width=2,
                    name='Experimental Sample')
    fig.update_layout(xaxis_title='Packing Coefficient', yaxis_title='Model Score',
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
                    mode='markers', marker_size=12,
                    marker_color=model_score.cpu().detach(),  # 'blue',
                    marker_colorbar=dict(title=dict(text="Model Score")),
                    marker_colorscale='bluered_r', opacity=0.75, marker_line_width=1,
                    marker_line_color='white',
                    name='Optimized Samples')
    # fig.add_scatter(x=torch.zeros(1), y=torch.zeros(1), mode='markers', marker_color='yellow',
    #                 marker_size=25, marker_line_color='black', marker_line_width=2,
    #                 name='Experimental Sample')
    fig.update_layout(yaxis_range=[-np.inf, 1], xaxis_range=[-np.inf, 0])
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

    """
    Visualize best clusters
    """
    best_samples = torch.argsort(c_dists, descending=False)
    best_crystals_batch = collate_data_list([optimized_samples[ind] for ind in best_samples[:5]])
    _, _, _, best_cluster_batch = best_crystals_batch.to('cpu').build_and_analyze(return_cluster=True)
    best_cluster_batch.visualize(mode='unit cell')
    original_cluster_batch.visualize([0], mode='unit cell')


if __name__ == '__main__':
    device = 'cuda'
    mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
    score_checkpoint = r"../checkpoints/crystal_score.pt"
    density_checkpoint = r"../checkpoints/cp_regressor.pt"
    csp_path = r'D:\crystal_datasets\optimized_samples_42.pt'
    score_model = load_crystal_score_model(score_checkpoint, device).to(device)
    score_model.eval()

    "load and batch batch_size copies of the same molecule"
    # BIGDOK, CETLAQ, DAFMUV, HEVTIL, DAFMUV
    example_crystals = torch.load(mini_dataset_path)
    elem_index = [elem.identifier for elem in example_crystals].index(
        'DAFMUV')  # .index('ACRLAC06')  #.index('FEDGOK01')
    original_crystal = collate_data_list([example_crystals[elem_index]]).to(device)
    optimized_samples = torch.load(csp_path)
    csp_reporting(optimized_samples, original_crystal, device, score_model)
