import argparse
import os
import sys

import torch
import wandb

sys.path.insert(0, os.path.abspath("../"))

from datetime import datetime
from mxtaltools.analysis.crystal_rdf import crystal_rdf
from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.common.training_utils import load_crystal_score_model, load_molecule_scalar_regressor, enable_dropout
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import softmax_and_score

if __name__ == '__main__':
    with (wandb.init(
            project="MXtalTools",
            entity='mkilgour')):

        # Create the parser
        parser = argparse.ArgumentParser(description='Process an integer.')

        # Add an argument for the integer
        parser.add_argument('--seed', type=int, default=0, help='An integer passed from the command line')

        # Parse the arguments
        args = parser.parse_args()
        seed = args.seed

        wandb.run.name = 'crystal_search_' + datetime.today().strftime("%d-%m-%H-%M-%S")
        device = 'cuda'
        mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
        score_checkpoint = r"../checkpoints/crystal_score.pt"
        density_checkpoint = r"../checkpoints/cp_regressor.pt"
        visualize = True

        batch_size = 10
        num_samples = 10
        num_batches = num_samples // batch_size
        sym_info = init_sym_info()

        "load and batch batch_size copies of the same molecule"
        # BIGDOK, CETLAQ, DAFMUV, HEVTIL, DAFMUV
        example_crystals = torch.load(mini_dataset_path)
        elem_index = [elem.identifier for elem in example_crystals].index(
            'DAFMUV')  #.index('ACRLAC06')  #.index('FEDGOK01')
        original_crystal = collate_data_list([example_crystals[elem_index]]).to(device)

        """load crystal score model and density prediction model"""
        score_model = load_crystal_score_model(score_checkpoint, device).to(device)
        score_model.eval()
        density_model = load_molecule_scalar_regressor(density_checkpoint, device)
        density_model.eval()

        """
        Density prediction
        """
        num_density_predictions = 50
        with torch.no_grad():
            """predict crystal packing coefficient - single-point"""
            target_packing_coeff = density_model(
                original_crystal).flatten() * density_model.target_std + density_model.target_mean
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

        print(f"True cp={float(original_crystal.packing_coeff):.3f} "
              f"predicted cp = {float(target_packing_coeff):.3f} "
              f"error {float(torch.abs(original_crystal.packing_coeff - target_packing_coeff) / torch.abs(original_crystal.packing_coeff)):.3f}")

        """
        Crystal optimization
        """
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
                max_attempts=500,
                seed=seed,
            )
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
                    [torch.tensor([sample.scaled_lj for sample in sample_list]) for sample_list in opt1_trajectory])
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
            optimized_crystal_batch = collate_data_list(opt1_trajectory[-1]).to(device)
            p1, p2, p3, optimized_cluster_batch = (
                optimized_crystal_batch.build_and_analyze(return_cluster=True,
                                                          cutoff=10))
            with torch.no_grad():
                model_output = score_model(optimized_cluster_batch.to(device), force_edges_rebuild=True).cpu()
                model_score = softmax_and_score(model_output[:, :2])

                fake_rdf, _, _ = crystal_rdf(optimized_cluster_batch,
                                             optimized_cluster_batch.edges_dict,
                                             rrange=[0, 6], bins=2000,
                                             mode='intermolecular', elementwise=True, raw_density=True,
                                             cpu_detach=False)

                for ind, sample in enumerate(opt1_trajectory[-1]):
                    sample.model_output = model_output[ind][None, :].clone().cpu()
                    sample.rdf = fake_rdf[ind][None, :].clone().cpu()
                    sample.lj = p1.clone().cpu()
                    sample.scaled_lj = p3.clone().cpu()
                    sample.es_pot = p2.clone().cpu()

            optimized_samples.extend(opt1_trajectory[-1])
            chunks = os.listdir()
            chunks = [elem for elem in chunks if f'optimized_samples_{elem_index}' in elem]
            chunk_ind = len(chunks)
            torch.save(optimized_samples, f'optimized_samples_{elem_index}_{chunk_ind}.pt')
