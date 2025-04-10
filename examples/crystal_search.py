import numpy as np
import torch
import torch.nn.functional as F

from mxtaltools.analysis.crystal_rdf import new_crystal_rdf, compute_rdf_distance
from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import softmax_and_score

if __name__ == '__main__':
    device = 'cuda'
    mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
    checkpoint = r"../checkpoints/crystal_score.pt"
    batch_size = 100
    num_samples = 10000
    num_batches = num_samples // batch_size
    sym_info = init_sym_info()

    "load and batch batch_size copies of the same molecule"
    example_crystals = torch.load(mini_dataset_path)
    elem_index = [elem.identifier for elem in example_crystals].index('ACRLAC06')  #.index('FEDGOK01')
    original_crystal = collate_data_list([example_crystals[elem_index]]).to(device)

    """load crystal score model"""
    model = load_crystal_score_model(checkpoint, device).to(device)
    model.eval()

    optimized_samples = []
    for batch_ind in range(num_batches):
        crystal_batch = collate_data_list([example_crystals[elem_index] for _ in range(batch_size)]).to(device)

        """
        generate a batch of random crystals for this molecule, 
        force overlapping molecules apart,
        do a rigid-body optimization of the crystal parameters
        """
        crystal_batch.sample_random_crystal_parameters(target_packing_coeff=0.65)
        # converged = False
        # while not converged:
        #     lj, _, _ = crystal_batch.build_and_analyze()
        #     bad_inds = torch.argwhere(lj > 0)
        #     cell_params = crystal_batch.cell_parameters()
        #     cell_params[bad_inds, :3] *= 1.1  # grow the box
        #     cell_params[bad_inds, 6:9] += torch.randn_like(cell_params[bad_inds, 6:9] * 0.05)  # noise the molecule a bit
        #     crystal_batch.set_cell_parameters(cell_params)
        #     crystal_batch.clean_cell_parameters(mode='hard')
        #     if torch.all(lj <= 0):
        #         converged = True
        _, crystal_batch = (
            crystal_batch.optimize_crystal_parameters(
                opt_func='inter_overlaps',
                init_lr=1e-3,
                show_tqdm=False
            ))

        optimization_trajectory, _ = (
            crystal_batch.optimize_crystal_parameters(
                opt_func='LJ',
                show_tqdm=True,
                opt_eps=1e-3,
            ))

        optimized_crystal_batch = collate_data_list(optimization_trajectory[-1])
        _, _, _, optimized_cluster_batch = (
            optimized_crystal_batch.build_and_analyze(return_cluster=True))
        model_output = model(optimized_cluster_batch.to(device)).cpu()
        model_score = softmax_and_score(model_output[:, :2])

        fake_rdf, _, _ = new_crystal_rdf(optimized_cluster_batch,
                                         optimized_cluster_batch.edges_dict,
                                         rrange=[0, 6], bins=2000,
                                         mode='intermolecular', elementwise=True, raw_density=True,
                                         cpu_detach=False)

        for ind, sample in enumerate(optimization_trajectory[-1]):
            sample.model_output = model_output[ind][None, :].clone().detach()
            sample.rdf = fake_rdf[ind][None, :].clone().detach()

        optimized_samples.extend(optimization_trajectory[-1])

    """filter unrealistic samples"""
    optimized_samples = [sample for sample in optimized_samples if sample.lj_pot < 10]
    optimized_samples = [sample for sample in optimized_samples if sample.packing_coeff < 1]
    optimized_samples = [sample for sample in optimized_samples if sample.packing_coeff > 0.5]

    """extract sampling results"""
    optimized_crystal_batch = collate_data_list(optimized_samples)
    packing_coeff = optimized_crystal_batch.packing_coeff
    model_score = softmax_and_score(optimized_crystal_batch.model_output[:, :2])
    rdf_dist_pred = F.softplus(optimized_crystal_batch.model_output[:, 2])
    lj_pot = optimized_crystal_batch.lj_pot
    fake_rdfs = optimized_crystal_batch.rdf
    """
    Do cell analysis
    """
    ref_lj_pot, ref_es_pot, ref_scaled_lj_pot, original_cluster_batch = (
        original_crystal.build_and_analyze(return_cluster=True))
    ref_model_output = model(original_cluster_batch.to(device)).cpu()
    ref_model_score = softmax_and_score(ref_model_output[:, :2])
    ref_rdf_dist_pred = F.softplus(ref_model_output[:, 2])
    ref_packing_coeff = original_cluster_batch.packing_coeff

    """
    Compute true distances
    """
    original_crystal_batch = collate_data_list([example_crystals[elem_index]])
    _, _, _, original_cluster_batch = original_crystal_batch.to(device).build_and_analyze(return_cluster=True)
    real_rdf, rr, _ = new_crystal_rdf(original_cluster_batch,
                                      original_cluster_batch.edges_dict,
                                      rrange=[0, 6], bins=2000,
                                      mode='intermolecular', elementwise=True, raw_density=True,
                                      cpu_detach=False)

    rdf_dists = torch.zeros(len(rdf_dist_pred), device=original_cluster_batch.device, dtype=torch.float32)
    for i in range(len(fake_rdfs)):
        rdf_dists[i] = compute_rdf_distance(real_rdf[0], fake_rdfs[i], rr) / optimized_crystal_batch.num_atoms[i]

    """
    Density/score plot
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_scatter(x=packing_coeff.cpu(), y=model_score.cpu(), mode='markers', marker_color='blue',
                    name='Optimized Samples')
    fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_model_score.cpu(), mode='markers', marker_color='black',
                    marker_size=20,
                    name='Experimental Sample')
    fig.update_layout(xaxis_title='Packing Coefficient', yaxis_title='Model Score',
                      xaxis_range=[0, 1])
    fig.update_annotations(font=dict(size=18))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.show()

    fig = go.Figure()
    fig.add_scatter(x=packing_coeff.cpu(), y=lj_pot.clip(max=150).cpu(), mode='markers', marker_color='blue',
                    name='Optimized Samples')
    fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_lj_pot.cpu(), mode='markers', marker_color='black', marker_size=20,
                    name='Experimental Sample')
    fig.update_layout(xaxis_title='Packing Coefficient', yaxis_title='LJ Energy',
                      xaxis_range=[0, 1])
    fig.update_annotations(font=dict(size=18))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.show()

    """
    RDF Plot
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_scatter(y=rdf_dist_pred.cpu().detach(), x=rdf_dists.cpu().detach(), mode='markers', marker_color='blue',
                    name='Optimized Samples')
    fig.add_scatter(y=ref_rdf_dist_pred.cpu().detach(), x=torch.zeros(1), mode='markers', marker_color='black',
                    marker_size=20,
                    name='Experimental Sample')
    fig.update_layout(yaxis_title='Predicted RDF Distance', xaxis_title='True RDF Distance',
                      yaxis_range=[0, min(0.1, torch.amax(rdf_dist_pred.cpu().detach()))],
                      xaxis_range=[0, min(0.1, torch.amax(rdf_dists.cpu().detach()))])
    fig.update_annotations(font=dict(size=18))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.show()

    """
    Visualize best clusters
    """
    best_samples = torch.argsort(model_score, descending=True)
    best_crystals_batch = collate_data_list([optimized_samples[ind] for ind in best_samples[:5]])
    _, _, _, best_cluster_batch = best_crystals_batch.to('cpu').build_and_analyze(return_cluster=True)
    best_cluster_batch.visualize(mode='unit cell')
    original_cluster_batch.visualize([0], mode='unit cell')

    torch.save(optimized_samples, 'optimized_samples.pt')

    print("Analysis finished!")
