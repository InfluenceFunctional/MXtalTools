import numpy as np
import torch
import torch.nn.functional as F
import wandb

from mxtaltools.analysis.crystal_rdf import new_crystal_rdf, compute_rdf_distance
from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.common.training_utils import load_crystal_score_model, load_molecule_scalar_regressor, enable_dropout
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import softmax_and_score

if __name__ == '__main__':
    with (wandb.init(
                     project="MXtalTools",
                     entity='mkilgour')):
        device = 'cuda'
        mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
        score_checkpoint = r"../checkpoints/crystal_score.pt"
        density_checkpoint = r"../checkpoints/cp_regressor.pt"
        visualize=True

        batch_size = 25
        num_samples = 500
        num_batches = num_samples // batch_size
        sym_info = init_sym_info()

        "load and batch batch_size copies of the same molecule"
        # BIGDOK, CETLAQ, DAFMUV, HEVTIL, DAFMUV
        example_crystals = torch.load(mini_dataset_path)
        elem_index = [elem.identifier for elem in example_crystals].index('DAFMUV') #.index('ACRLAC06')  #.index('FEDGOK01')
        original_crystal = collate_data_list([example_crystals[elem_index]]).to(device)

        """load crystal score model and density prediction model"""
        score_model = load_crystal_score_model(score_checkpoint, device).to(device)
        score_model.eval()
        density_model = load_molecule_scalar_regressor(density_checkpoint, device)
        density_model.eval()

        num_density_predictions = 50
        with torch.no_grad():
            """predict crystal packing coefficient - single-point"""
            target_packing_coeff = density_model(original_crystal).flatten() * density_model.target_std + density_model.target_mean
            aunit_volume_pred = original_crystal.mol_volume / target_packing_coeff  # A^3
            density_pred = original_crystal.mass / aunit_volume_pred * 1.6654  # g/cm^3

            """get prediction with uncertainty via resampling with dropout"""
            predictions = []
            model = enable_dropout(density_model)
            for _ in range(num_density_predictions):
                predictions.append(model(original_crystal).flatten() * model.target_std + model.target_mean)

            predictions = torch.stack(predictions)
            packing_coeff_mean = predictions.mean(0)
            packing_coeff_std = predictions.std(0)

        print(f"True cp={float(original_crystal.packing_coeff):.2f} "
              f"predicted cp = {float(target_packing_coeff):.2f} "
              f"error {float(torch.abs(original_crystal.packing_coeff - target_packing_coeff)/torch.abs(original_crystal.packing_coeff)):.3f}")

        """analyze original crystal"""
        ref_lj_pot, ref_es_pot, ref_scaled_lj_pot, original_cluster_batch = (
            original_crystal.build_and_analyze(return_cluster=True))
        ref_model_output = score_model(original_cluster_batch.to(device), force_edges_rebuild=True).cpu()
        ref_model_score = softmax_and_score(ref_model_output[:, :2])
        ref_rdf_dist_pred = F.softplus(ref_model_output[:, 2])
        ref_packing_coeff = original_cluster_batch.packing_coeff

        optimized_samples = []
        for batch_ind in range(num_batches):
            """
            generate a batch of random crystals for this molecule, 
            force overlapping molecules apart,
            do a rigid-body optimization of the crystal parameters
            analyze resulting crystals
            """

            print(f'Starting batch {batch_ind}')
            crystal_batch = collate_data_list([example_crystals[elem_index] for _ in range(batch_size)]).to(device)

            crystal_batch.sample_reasonable_random_parameters(
                target_packing_coeff=target_packing_coeff * 0.75,
                tolerance=3,
                max_attempts=500
            )

            crystal_batch.to(device)
            opt1_trajectory = (
                crystal_batch.optimize_crystal_parameters(
                    optim_target='LJ',
                    show_tqdm=True,
                    convergence_eps=1e-6,
                    #score_model=score_model,
                    target_packing_coeff=target_packing_coeff,
                    do_box_restriction=True,
                    cutoff=10,
                ))
            crystal_batch = collate_data_list(opt1_trajectory[-1]).to(device)
            opt2_trajectory = (
                crystal_batch.optimize_crystal_parameters(
                    optim_target='rdf_score',
                    show_tqdm=True,
                    convergence_eps=1e-6,
                    score_model=score_model,
                    do_box_restriction=False,
                    cutoff=6,
                ))

            """visualize optimization trajectory"""
            opt1_trajectory.extend(opt2_trajectory)

            if False:
                import plotly.graph_objects as go

                lj_pots = torch.stack(
                    [torch.tensor([sample.scaled_lj_pot for sample in sample_list]) for sample_list in opt1_trajectory])
                coeffs = torch.stack(
                    [torch.tensor([sample.packing_coeff for sample in sample_list]) for sample_list in opt1_trajectory])

                fig = go.Figure()
                fig.add_scatter(x=coeffs[0, :], y=lj_pots[0, :], mode='markers', marker_size=20, marker_color='grey',
                                name='Initial State')
                fig.add_scatter(x=coeffs[-1, :], y=lj_pots[-1, :], mode='markers', marker_size=20, marker_color='black',
                                name='Final State')
                for ind in range(coeffs.shape[1]):
                    fig.add_scatter(x=coeffs[:, ind], y=lj_pots[:, ind], name=f"Run {ind}")
                fig.update_layout(xaxis_title='Packing Coeff', yaxis_title='Scaled LJ')
                fig.show()

                fig = go.Figure()
                for ind in range(lj_pots.shape[-1]):
                    fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
                                    showlegend=True if ind == 0 else False)
                fig.show()

                fig = go.Figure()
                for ind in range(lj_pots.shape[-1]):
                    fig.add_scatter(y=coeffs[..., ind], marker_color='blue', name='packing coeff', legendgroup='lg',
                                    showlegend=True if ind == 0 else False)
                fig.update_layout(yaxis_range=[0, 1])
                fig.show()

            """analyze optimized samples"""
            optimized_crystal_batch = collate_data_list(opt2_trajectory[-1]).to(device)
            p1, p2, p3, optimized_cluster_batch = (
                optimized_crystal_batch.build_and_analyze(return_cluster=True,
                                                          cutoff=10))
            with torch.no_grad():
                model_output = score_model(optimized_cluster_batch.to(device), force_edges_rebuild=True).cpu()
                model_score = softmax_and_score(model_output[:, :2])

                fake_rdf, _, _ = new_crystal_rdf(optimized_cluster_batch,
                                                 optimized_cluster_batch.edges_dict,
                                                 rrange=[0, 6], bins=2000,
                                                 mode='intermolecular', elementwise=True, raw_density=True,
                                                 cpu_detach=False)

                for ind, sample in enumerate(opt2_trajectory[-1]):
                    sample.model_output = model_output[ind][None, :].clone().detach().cpu()
                    sample.rdf = fake_rdf[ind][None, :].clone().detach().cpu()
                    sample.lj_pot = p1.clone().deatch().cpu()
                    sample.scaled_lj_pot = p3.clone().deatch().cpu()
                    sample.es_pot = p2.clone().deatch().cpu()

            optimized_samples.extend(opt2_trajectory[-1])
            torch.save(optimized_samples, f'optimized_samples_{elem_index}.pt')

        """filter unrealistic samples"""
        #optimized_samples = [sample for sample in optimized_samples if sample.lj_pot < 10]
        #optimized_samples = [sample for sample in optimized_samples if sample.packing_coeff < 1]
        #optimized_samples = [sample for sample in optimized_samples if sample.packing_coeff > 0.5]

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
        original_crystal_batch = collate_data_list([example_crystals[elem_index]])
        _, _, _, original_cluster_batch = original_crystal_batch.to(device).build_and_analyze(return_cluster=True,
                                                                                              cutoff=6)
        real_rdf, rr, _ = new_crystal_rdf(original_cluster_batch.to(device),
                                          original_cluster_batch.edges_dict,
                                          rrange=[0, 6], bins=2000,
                                          mode='intermolecular',
                                          elementwise=True,
                                          raw_density=True,
                                          cpu_detach=False)

        rdf_dists = torch.zeros(len(rdf_dist_pred), device=original_cluster_batch.device, dtype=torch.float32)
        for i in range(len(fake_rdfs)):
            rdf_dists[i] = compute_rdf_distance(real_rdf[0], fake_rdfs[i].to(device), rr) / optimized_crystal_batch.num_atoms[i]
        rdf_dists = rdf_dists.cpu()
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
        fig.add_scatter(x=packing_coeff.cpu(), y=scaled_lj_pot.cpu(), mode='markers', marker_color='blue',
                        name='Optimized Samples')
        fig.add_scatter(x=ref_packing_coeff.cpu(), y=ref_scaled_lj_pot.cpu(), mode='markers', marker_color='black', marker_size=20,
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

        fig = go.Figure()
        fig.add_scatter(x=model_score.cpu(), y=scaled_lj_pot.cpu(), mode='markers', marker_color='blue',
                        name='Optimized Samples')
        fig.add_scatter(x=ref_model_score.cpu(), y=ref_scaled_lj_pot.cpu(), mode='markers', marker_color='black', marker_size=20,
                        name='Experimental Sample')
        fig.update_layout(xaxis_title='Model Score', yaxis_title='LJ Energy',
                          )
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
                          yaxis_range=[0, min(0.5, torch.amax(rdf_dist_pred.cpu().detach()))],
                          xaxis_range=[0, min(0.5, torch.amax(rdf_dists.cpu().detach()))])
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

        torch.save(optimized_samples, f'optimized_samples_{elem_index}.pt')

        print("Analysis finished!")
