import torch

from mxtaltools.dataset_utils.utils import collate_data_list


"""
test round-trip transforms

cell -> latent -> cell -> latent
cell -> std -> cell -> std

"""

def tensor_max_discrepancy(t1, t2):
    return (t1-t2).abs().amax()

def test_latent_roundtrip():
    # from a dataset
    crystal_batch = collate_data_list(torch.load(r'../mini_datasets/mini_reduced_CSD_dataset.pt', weights_only=False))

    cell_params = crystal_batch.zp1_cell_parameters()
    latent_params = crystal_batch.latent_params()

    crystal_batch.latent_to_cell_params(latent_params)
    cell_params2 = crystal_batch.zp1_cell_parameters()
    latent_params2 = crystal_batch.latent_params()

    assert tensor_max_discrepancy(cell_params, cell_params2) < 1e-1
    assert tensor_max_discrepancy(latent_params, latent_params2) < 1e-1

    # from a random batch
    crystal_batch.sample_random_reduced_crystal_parameters()

    cell_params = crystal_batch.zp1_cell_parameters()
    latent_params = crystal_batch.latent_params()

    crystal_batch.latent_to_cell_params(latent_params)
    cell_params2 = crystal_batch.zp1_cell_parameters()
    latent_params2 = crystal_batch.latent_params()

    assert tensor_max_discrepancy(cell_params, cell_params2) < 1e-1
    assert tensor_max_discrepancy(latent_params, latent_params2) < 1e-1

def test_std_roundtrip():
    # from a dataset
    crystal_batch = collate_data_list(torch.load(r'../mini_datasets/mini_reduced_CSD_dataset.pt', weights_only=False))

    cell_params = crystal_batch.zp1_cell_parameters()
    std_params = crystal_batch.zp1_std_cell_parameters()

    cell_params2 = crystal_batch.destandardize_zp1_cell_parameters(std_params)
    crystal_batch.set_cell_parameters(cell_params2)
    std_params2 = crystal_batch.zp1_std_cell_parameters()

    assert tensor_max_discrepancy(cell_params, cell_params2) < 1e-1
    assert tensor_max_discrepancy(std_params, std_params2) < 1e-1


    # from a random batch
    crystal_batch.sample_random_reduced_crystal_parameters()

    cell_params = crystal_batch.zp1_cell_parameters()
    std_params = crystal_batch.zp1_std_cell_parameters()

    cell_params2 = crystal_batch.destandardize_zp1_cell_parameters(std_params)
    crystal_batch.set_cell_parameters(cell_params2)
    std_params2 = crystal_batch.zp1_std_cell_parameters()

    assert tensor_max_discrepancy(cell_params, cell_params2) < 1e-1
    assert tensor_max_discrepancy(std_params, std_params2) < 1e-1