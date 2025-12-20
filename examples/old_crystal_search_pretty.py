# import torch
#
# from mxtaltools.analysis.crystal_rdf import crystal_rdf
# from mxtaltools.common.sym_utils import init_sym_info
# from mxtaltools.common.training_utils import load_crystal_score_model, load_molecule_scalar_regressor
# from mxtaltools.dataset_utils.utils import collate_data_list
# from mxtaltools.models.utils import softmax_and_score
#
# # Parse the arguments
# seed = 1234
#
# device = 'cuda'
# mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
# score_checkpoint = r"../checkpoints/crystal_score.pt"
# density_checkpoint = r"../checkpoints/cp_regressor.pt"
#
# batch_size = 10
# num_samples = 10
# num_batches = num_samples // batch_size
# sym_info = init_sym_info()
#
# dafmuv_crystal = torch.load(dafmuv_path)
# dafmuv_crystal.to(device)
#
# """load crystal score model and density prediction model"""
# score_model = load_crystal_score_model(score_checkpoint, device)
# score_model.eval()
#
# density_model = load_molecule_scalar_regressor(density_checkpoint, device)
# density_model.eval()
#
# """
# Density prediction
# """
# num_density_predictions = 50
# with torch.no_grad():
#     """predict crystal packing coefficient - single-point"""
#     target_packing_coeff = density_model(
#         dafmuv_crystal).flatten() * density_model.target_std + density_model.target_mean
#     aunit_volume_pred = dafmuv_crystal.mol_volume / target_packing_coeff  # A^3
#     density_pred = dafmuv_crystal.mass / aunit_volume_pred * 1.6654  # g/cm^3
#
# """
# Crystal optimization
# """
# optimized_samples = []
# for batch_ind in range(num_batches):
#     """
#     generate a batch of random crystals for this molecule,
#     force overlapping molecules apart,
#     do a rigid-body optimization of the crystal parameters
#     analyze resulting crystals
#     """
#
#     print(f'Starting batch {batch_ind}')
#     crystal_batch = collate_data_list([dafmuv_crystal[0] for _ in range(batch_size)]).to(device)
#
#     crystal_batch.sample_reasonable_random_parameters(
#         target_packing_coeff=target_packing_coeff * 0.75,
#         tolerance=3,
#         max_attempts=500,
#         seed=seed,
#     )
#     first_optimization_trajectory = (
#         crystal_batch.optimize_crystal_parameters(
#             optim_target='LJ',
#             show_tqdm=True,
#             convergence_eps=1e-6,
#             target_packing_coeff=target_packing_coeff,
#             do_box_restriction=True,
#             cutoff=10,
#         ))
#     crystal_batch = collate_data_list(first_optimization_trajectory[-1]).to(device)
#     second_optimization_trajectory = (
#         crystal_batch.optimize_crystal_parameters(
#             optim_target='rdf_score',
#             show_tqdm=True,
#             convergence_eps=1e-6,
#             score_model=score_model,
#             do_box_restriction=False,
#             cutoff=6,
#         ))
#
#     """analyze optimized samples"""
#     optimized_crystal_batch = collate_data_list(second_optimization_trajectory[-1]).to(device)
#     p1, p2, p3, optimized_cluster_batch = (
#         optimized_crystal_batch.build_and_analyze(return_cluster=True,
#                                                   cutoff=10))
#     with torch.no_grad():
#         model_output = score_model(optimized_cluster_batch.to(device), force_edges_rebuild=True).cpu()
#         model_score = softmax_and_score(model_output[:, :2])
#
#         sample_rdf, _, _ = crystal_rdf(optimized_cluster_batch,
#                                        optimized_cluster_batch.edges_dict,
#                                        rrange=[0, 6], bins=2000,
#                                        mode='intermolecular', elementwise=True, raw_density=True,
#                                        cpu_detach=False)
#
#         for ind, sample in enumerate(second_optimization_trajectory[-1]):
#             sample.model_output = model_output[ind][None, :].clone().cpu()
#             sample.rdf = sample_rdf[ind][None, :].clone().cpu()
#             sample.lj = p1.clone().cpu()
#             sample.scaled_lj = p3.clone().cpu()
#             sample.es_pot = p2.clone().cpu()
#
#     optimized_samples.extend(second_optimization_trajectory[-1])
#     torch.save(optimized_samples, f'optimized_samples.pt')
