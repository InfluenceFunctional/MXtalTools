import numpy as np
import torch
import torch.nn.functional as F

from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import softmax_and_score

if __name__ == '__main__':
    device = 'cpu'
    mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
    checkpoint = r"../models/crystal_score.pt"
    space_groups_to_sample = ["P1", "P-1", "P21/c", "C2/c", "P212121"]
    sym_info = init_sym_info()

    "load and batch example crystals"
    example_crystals = torch.load(mini_dataset_path)
    crystal_batch = collate_data_list(example_crystals[:10])

    """
    A core function of our code is crystal parameterization and construction, 
    and so we show a simple example of building crystals, starting from the same 
    molecules as before, but with random space groups and lattice parameters.
    """
    # initialize prior distribution
    crystal_batch2 = crystal_batch.detach().clone()
    # pick space groups to sample
    sgs_to_build = np.random.choice(space_groups_to_sample,
                                    replace=True,
                                    size=crystal_batch.num_graphs)
    sg_rand_inds = torch.tensor(
        [list(sym_info['space_groups'].values()).index(SG) + 1 for SG in sgs_to_build],
        dtype=torch.long,
        device=device)  # indexing from 0
    # assign SG info to crystals - critical to do this before resampling cell parameters
    crystal_batch2.reset_sg_info(sg_rand_inds)
    # sample random cell parameters
    crystal_batch2.sample_random_crystal_parameters()
    """load crystal score model"""
    model = load_crystal_score_model(checkpoint, device)

    """
    And proceed to analyzing both sets of crystals.
    We present here a very basic analysis, computing a very basic 
    Lennard-Jones-type and short-range electrostatic potential. 
    We also show the outpudts of the crystal scoring model, 
    (1) it's classification confidence between "real" CSD samples 
    and "fake" samples, not from the CSD, and 
    (2) the predicted distance in RDF space from the given crystal 
    to the "correct" crystal for the given molecule.
    """
    lj_pot, es_pot, scaled_lj_pot, cluster_batch = (
        crystal_batch.build_and_analyze(return_cluster=True))
    model_output = model(cluster_batch)
    model_score = softmax_and_score(model_output[:, :2])
    rdf_dist_pred = F.softplus(model_output[:, 2])
    packing_coeff = crystal_batch.packing_coeff
    cluster_batch.visualize([1, 2, 3, 4], mode='convolve with')

    lj_pot2, es_pot2, scaled_lj_pot2, cluster_batch2 = (
        crystal_batch2.build_and_analyze(return_cluster=True))
    model_output2 = model(cluster_batch2)
    model_score2 = softmax_and_score(model_output2[:, :2])
    rdf_dist_pred2 = F.softplus(model_output2[:, 2])
    packing_coeff2 = crystal_batch2.packing_coeff
    cluster_batch2.visualize([1, 2, 3, 4], mode='convolve with')

    print("Finished Crystal Analysis!")