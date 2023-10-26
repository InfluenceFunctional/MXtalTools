import os

import warnings
import torch.optim as optim
import wandb


from classify_lammps_trajs.utils import generate_dataset_from_dumps, class_names, collect_to_traj_dataloaders, init_classifier, train_classifier

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore ovito error

num_convs = 2
num_epochs = 1000
dataset_size = 100
conv_cutoff = 6
batch_size = 5
train_model = True
model_checkpoint = None

dataset_name = 'nicotinamide_trajectories_dataset_full'
dataset_path = f'C:/Users/mikem/crystals/clusters/cluster_structures/bulk_trajs1/{dataset_name}.pkl'

dumps_dirs = [r'C:\Users\mikem\crystals\clusters\cluster_structures\bulk_trajs1\T_100',
              r'C:\Users\mikem\crystals\clusters\cluster_structures\bulk_trajs1\T_350',
              r'C:\Users\mikem\crystals\clusters\cluster_structures\hot_trajs/melted_trajs_T_950']

device = 'cuda'

if __name__ == "__main__":
    if not os.path.exists(dataset_path):
        generate_dataset_from_dumps(dumps_dirs, dataset_path)

    train_loader, _ = collect_to_traj_dataloaders(dataset_path, dataset_size=dataset_size, batch_size=1, temperatures=[100, 950], test_fraction=0.01)
    test_loader, _ = collect_to_traj_dataloaders(dataset_path, dataset_size=dataset_size, batch_size=1, temperatures=[350], test_fraction=0.01)

    classifier = init_classifier(conv_cutoff, num_convs)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    # todo add checkpoint loading
    if model_checkpoint is not None:
        assert False

    classifier.to(device)
    if train_model:
        train_classifier(classifier, optimizer,
                         train_loader, test_loader,
                         num_epochs, wandb,
                         class_names, device,
                         batch_size
                         )



