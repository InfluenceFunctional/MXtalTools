import numpy as np
import torch
import torch.nn.functional as F

from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.task_models.generator_models import CSDPrior
from mxtaltools.models.utils import softmax_and_score

if __name__ == '__main__':
    device = 'cpu'
    mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
    checkpoint = r"../models/crystal_discriminator.pt"
    space_groups_to_sample = ["P1", "P-1", "P21/c", "C2/c", "P212121"]
    sym_info = init_sym_info()

    "load example crystals"
    example_crystals = torch.load(mini_dataset_path)
    crystal_batch = collate_data_list(example_crystals[:10])

    "synthesize random crystals from the same molecules"
    # initialize prior distribution
    crystal_batch2 = crystal_batch.detach().clone()
    prior = CSDPrior(sym_info=sym_info, device=device)
    # pick space groups to sample
    sgs_to_build = np.random.choice(space_groups_to_sample, replace=True, size=crystal_batch.num_graphs)
    sg_rand_inds = torch.tensor([list(sym_info['space_groups'].values()).index(SG) + 1 for SG in sgs_to_build],
                                dtype=torch.long, device=device)  # indexing from 0
    crystal_batch2.reset_sg_info(sg_rand_inds)
    # sample cell parameters
    normed_cell_params = prior(len(crystal_batch), sg_rand_inds).to(crystal_batch.device)
    # assign new parameters to crystal
    normed_cell_lengths, crystal_batch2.cell_angles, crystal_batch2.aunit_centroid, crystal_batch2.aunit_orientation = normed_cell_params.split(
        3, dim=1)
    crystal_batch2.cell_lengths = crystal_batch2.denorm_cell_lengths(normed_cell_lengths)
    crystal_batch2.box_analysis()

    "load crystal score model"
    model = load_crystal_score_model(
        checkpoint,
        device
    )

    "do analysis"
    lj_pot, es_pot, scaled_lj_pot, crystal_cluster = crystal_batch.build_and_analyze(return_cluster=True)
    model_output = model(crystal_cluster)
    model_score = softmax_and_score(model_output[:, :2])
    rdf_dist_pred = F.softplus(model_output[:, 2])

    lj_pot2, es_pot2, scaled_lj_pot2, crystal_cluster2 = crystal_batch2.build_and_analyze(return_cluster=True)
    model_output2 = model(crystal_cluster2)
    model_score2 = softmax_and_score(model_output2[:, :2])
    rdf_dist_pred2 = F.softplus(model_output2[:, 2])

    print("Crystal analysis finished!")
