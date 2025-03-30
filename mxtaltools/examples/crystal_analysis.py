import torch

from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.utils import collate_data_list

if __name__ == '__main__':
    device = 'cpu'
    mini_dataset_path = '../../mini_datasets/mini_CSD_dataset.pt'
    checkpoint = r"../../models/crystal_discri"

    "load example crystals"
    example_crystals = torch.load(mini_dataset_path)
    crystal_batch = collate_data_list(example_crystals[:10])

    "synthesize random crystals"

    "load crystal score model"
    model = load_crystal_score_model(
        checkpoint,
        device
    )

    "do analysis"
    lj_pot, es_pot, scaled_lj_pot, crystal_cluster = crystal_batch.build_and_analyze(return_cluster=True)
    model_score = model(crystal_cluster)
