import os

import warnings
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import plotly.graph_objects as go

from classify_lammps_trajs.utils import generate_dataset_from_dumps, num2atomicnum, type2num, convert_box_to_cell_params, collect_to_traj_dataloaders, init_classifier

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error

num_convs = 2
num_epochs = 1000
dataset_size = 1000
conv_cutoff = 6

class_names = ['V', 'VII', 'VIII', 'I', 'II', 'III', 'IV', 'IX', 'VI', 'Disordered']

device = 'cuda'

if __name__ == "__main__":
    dataset_path = r'D:\crystal_datasets\bulk_crystal_trajs\nicotinamide_trajectory_dataset.pkl'
    if not os.path.exists(dataset_path):
        generate_dataset_from_dumps()

    train_loader, test_loader = collect_to_traj_dataloaders(dataset_path, dataset_size=dataset_size, batch_size=1)

    classifier = init_classifier(conv_cutoff, num_convs)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    classifier.train(True)
    classifier.to(device)
    with wandb.init(project='cluster_classifier', entity='mkilgour'):

        for epoch in range(num_epochs):
            print(f"starting epoch {epoch}")
            classifier.train(True)
            train_loss = []
            train_true_labels = []
            train_probs = []
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)

                output = classifier(sample)
                loss = F.cross_entropy(output, sample.y)

                train_probs.append(F.softmax(output, dim=1).cpu().detach().numpy())
                train_true_labels.append(sample.y.cpu().detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())

            with torch.no_grad():
                classifier.eval()
                test_loss = []
                test_probs = []
                test_true_labels = []
                for step, data in enumerate(tqdm(test_loader)):
                    sample = data.to(device)

                    output = classifier(sample)  # fix mini-batch behavior
                    loss = F.cross_entropy(output, sample.y)

                    test_loss.append(loss.cpu().detach().numpy())
                    test_probs.append(F.softmax(output, dim=1).cpu().detach().numpy())
                    test_true_labels.append(sample.y.cpu().detach().numpy())

            print(f"Log Train Loss {np.log10(np.mean(np.array(train_loss))):.4f}")
            print(f"Log Test Loss {np.log10(np.mean(np.array(test_loss))):.4f}")

            if epoch % 10 == 0:
                train_true_labels = np.concatenate(train_true_labels)
                train_probs = np.concatenate(train_probs)
                test_true_labels = np.concatenate(test_true_labels)
                test_probs = np.concatenate(test_probs)

                if len(np.unique(train_true_labels)) == 10:
                    predicted_class = np.argmax(train_probs, axis=1)

                    train_score = roc_auc_score(train_true_labels, train_probs, multi_class='ovo')
                    train_f1_score = f1_score(train_true_labels, predicted_class, average='micro')
                    train_cmat = confusion_matrix(train_true_labels, predicted_class, normalize='true')
                    fig = go.Figure(go.Heatmap(z=train_cmat, x=class_names, y=class_names))
                    fig.update_layout(xaxis=dict(title="Predicted Forms"),
                                      yaxis=dict(title="True Forms")
                                      )

                    wandb.log({"Train ROC_AUC": train_score,
                               "Train F1 Score": train_f1_score,
                               "Train Confusion Matrix": fig})

                if len(np.unique(test_true_labels)) == 10:
                    predicted_class = np.argmax(test_probs, axis=1)

                    test_score = roc_auc_score(test_true_labels, test_probs, multi_class='ovo')
                    test_f1_score = f1_score(test_true_labels, predicted_class, average='micro')
                    test_cmat = confusion_matrix(test_true_labels, predicted_class, normalize='true')
                    fig = go.Figure(go.Heatmap(z=test_cmat, x=class_names, y=class_names))
                    fig.update_layout(xaxis=dict(title="Predicted Forms"),
                                      yaxis=dict(title="True Forms")
                                      )

                    wandb.log({"Test ROC_AUC": test_score,
                               "Test F1 Score": test_f1_score,
                               "Test Confusion Matrix": fig})

            wandb.log({
                'train_loss': np.asarray(train_loss).mean(),
                'test_loss': np.asarray(test_loss).mean()
            })
