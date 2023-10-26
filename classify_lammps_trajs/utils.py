import glob
import os

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.utils import delete_from_dataframe
from dataset_management.CrystalData import CrystalData
from dataset_management.utils import get_dataloaders
from models.base_models import molecule_graph_model

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import plotly.graph_objects as go

class_names = ['V', 'VII', 'VIII', 'I', 'II', 'III', 'IV', 'IX', 'VI', 'Disordered']


def process_dump(path):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()

    timestep = None
    n_atoms = None
    frame_outputs = {}
    for ind, line in enumerate(lines):
        if "ITEM: TIMESTEP" in line:
            timestep = int(lines[ind + 1])
        elif "ITEM: NUMBER OF ATOMS" in line:
            n_atoms = int(lines[ind + 1])
        elif "ITEM: BOX BOUNDS" in line:
            cell_params = np.stack([
                np.asarray(lines[ind + 1].split()).astype(float),
                np.asarray(lines[ind + 2].split()).astype(float),
                np.asarray(lines[ind + 3].split()).astype(float)
            ]
            )
        elif "ITEM: ATOMS" in line:  # atoms header
            headers = line.split()[2:-3]
            atom_data = np.zeros((n_atoms, len(headers)))
            for ind2 in range(n_atoms):
                newline = lines[1 + ind + ind2].split()
                # newline[2] = type2num[newline[2]]
                newline[2] = num2atomicnum[int(newline[2])]

                atom_data[ind2] = np.asarray(newline[:-3]).astype(float)  # cut off velocity elements

            frame_data = pd.DataFrame(atom_data, columns=headers)
            frame_data.attrs['cell_params'] = cell_params  # add attribute directly to dataframe
            frame_outputs[timestep] = frame_data
        else:
            pass

    return frame_outputs


def generate_dataset_from_dumps(dumps_dirs, dataset_path):
    sample_df = pd.DataFrame()
    for dumps_dir in dumps_dirs:
        os.chdir(dumps_dir)
        dump_files = glob.glob(r'*/*.dump', recursive=True)

        for path in tqdm(dump_files):
            print(f"Processing dump {path}")
            temperature = int(dumps_dir.split('_')[-1])
            form = int(path.split('/')[0])
            trajectory_dict = process_dump(path)

            for ts, (times, vals) in enumerate(tqdm(trajectory_dict.items())):
                new_dict = {'atom_type': [vals['element'].astype(int)],
                            'mol_ind': [vals['mol']],
                            'coordinates': [np.concatenate((
                                np.asarray(vals['x'])[:, None],
                                np.asarray(vals['y'])[:, None],
                                np.asarray(vals['z'])[:, None]), axis=-1)],
                            'temperature': temperature,
                            'form': form,
                            'time_step': times,
                            'cell_params': vals.attrs['cell_params'],
                            }

                new_df = pd.DataFrame()
                for key in new_dict.keys():
                    new_df[key] = [new_dict[key]]

                sample_df = pd.concat([sample_df, new_df])

    sample_df.to_pickle(dataset_path)


type2num = {
    'Ca1': 1,
    'Ca2': 2,
    'Ca': 3,
    'C': 4,
    'Nb': 5,
    'N': 6,
    'O': 7,
    'Hn': 8,
    'H4': 9,
    'Ha': 10,
}
num2atomicnum = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 7,
    6: 7,
    7: 8,
    8: 1,
    9: 1,
    10: 1,
}


def convert_box_to_cell_params(box_params):
    """
    ITEM: BOX BOUNDS xy xz yz
    xlo_bound xhi_bound xy
    ylo_bound yhi_bound xz
    zlo_bound zhi_bound yz

    a = xhi-xlo, 0, 0
    b = xy, yhi-ylo, 0
    c = xz, yz, zhi-zlo

    xlo = xlo_bound - MIN(0, xy, xz, xy+xz)
    xhi = xhi_bound - MAX(0, xy, xz, xy+xz)
    ylo = ylo_bound - MIN(0, yz)
    yhi = yhi_bound - MAX(0, yz)
    zlo = zlo_bound
    zhi = zhi_bound
    """
    xlo_bound = box_params[:, 0, 0]
    ylo_bound = box_params[:, 1, 0]
    zlo_bound = box_params[:, 2, 0]
    xhi_bound = box_params[:, 0, 1]
    yhi_bound = box_params[:, 1, 1]
    zhi_bound = box_params[:, 2, 1]
    xy = box_params[:, 0, 2]
    xz = box_params[:, 1, 2]
    yz = box_params[:, 2, 2]

    xlo = xlo_bound - np.stack((np.zeros_like(xy), xy, xz, xy + xz)).T.min(1)
    xhi = xhi_bound - np.stack((np.zeros_like(xy), xy, xz, xy + xz)).T.max(1)
    ylo = ylo_bound - np.stack((np.zeros_like(yz), yz)).T.min(1)
    yhi = yhi_bound - np.stack((np.zeros_like(yz), yz)).T.max(1)
    zlo = zlo_bound
    zhi = zhi_bound

    a = np.asarray([xhi - xlo, np.zeros_like(xhi), np.zeros_like(xhi)]).T
    b = np.asarray([xy, yhi - ylo, np.zeros_like(xy)]).T
    c = np.asarray([xz, yz, zhi - zlo]).T

    return a, b, c


def collect_to_traj_dataloaders(dataset_path, dataset_size, batch_size, test_fraction=0.2, filter_early=True, temperatures: list = None):
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.reset_index().drop(columns='index')  # reindexing is crucial here

    if temperatures is not None:
        good_inds = []
        for temperature in temperatures:
            good_inds.append(np.argwhere(np.asarray(dataset['temperature']) == temperature)[:, 0])

        good_inds = np.unique(np.concatenate(good_inds))
        bad_inds = np.asarray([ind for ind in np.arange(len(dataset)) if ind not in good_inds])
        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    if filter_early:
        bad_inds = np.argwhere(np.asarray(dataset['time_step']) <= int(1e5))[:, 0]  # filter first 100k steps for equilibration
        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    forms = np.unique(dataset['form'])
    forms2tgt = {form: i for i, form in enumerate(forms)}
    targets = np.asarray(
        [forms2tgt[form] for form in dataset['form']]
    )

    # set high temperature samples to 'melted' class
    for i in range(len(dataset)):
        if dataset.iloc[i]['temperature'] == 950:
            targets[i] = 9

    a, b, c = convert_box_to_cell_params(np.stack(dataset['cell_params']))

    T_fc_list = np.zeros((len(a), 3, 3))
    for i in range(len(T_fc_list)):
        T_fc_list[i] = np.stack((a[i], b[i], c[i]))

    print('Generating training datapoints')
    datapoints = []
    rand_inds = np.random.choice(len(dataset), size=min(dataset_size, len(dataset)), replace=False)
    for i in tqdm(rand_inds):
        ref_coords = torch.Tensor(dataset.loc[i]['coordinates'][0])
        atoms = dataset.loc[i]['atom_type'][0]

        atomic_numbers = torch.tensor([num2atomicnum[atom] for atom in atoms], dtype=torch.long)
        num_molecules = (len(ref_coords)) // 15

        mol_ind = torch.tensor(dataset.loc[i]['mol_ind'][0], dtype=torch.long)
        assert num_molecules == torch.amax(mol_ind)

        # we cannot trust the default indexing, so manually reindex according to the recorded molecule index
        cluster_coords = torch.stack([ref_coords[mol_ind == ind] for ind in torch.unique(mol_ind)])
        cluster_atoms = torch.stack([atomic_numbers[mol_ind == ind] for ind in torch.unique(mol_ind)])
        cluster_targets = torch.tensor(targets[i].repeat(num_molecules), dtype=torch.long)

        # pare off molecules which are fragmented
        mol_centroids = cluster_coords.mean(1)
        intramolecular_centroid_dists = torch.linalg.norm(mol_centroids[:, None, :] - cluster_coords, dim=2)
        mol_radii = intramolecular_centroid_dists.amax(1)

        good_mols = torch.argwhere(mol_radii < 4)[:, 0]
        cluster_coords = cluster_coords[good_mols]
        cluster_atoms = cluster_atoms[good_mols]
        cluster_targets = cluster_targets[good_mols]

        # identify surface mols
        centroids = cluster_coords.mean(1)
        dist = torch.cdist(centroids, centroids)
        coordination_cutoff = 4 + 6
        coordination_number = (dist < coordination_cutoff).sum(1)
        surface_mols_ind = torch.argwhere(coordination_number < 20)[:, 0]  # 4 is normal - this is quite permissive
        cluster_targets[surface_mols_ind] = len(forms2tgt)  # label surface molecules

        # reindex mol_inds
        cluster_mol_ind = torch.arange(len(good_mols)).repeat(15, 1).T

        datapoints.append(
            CrystalData(
                x=cluster_atoms.reshape(len(good_mols) * 15),
                mol_ind=cluster_mol_ind.reshape(len(good_mols) * 15),
                pos=cluster_coords.reshape(len(good_mols) * 15, 3),
                y=cluster_targets,
                tracking=np.asarray(dataset.loc[i, ['temperature', 'time_step']]),
                T_fc=torch.Tensor(T_fc_list[i]),
                asym_unit_handedness=torch.ones(1),
                symmetry_operators=torch.ones(1),
            )
        )
    del dataset
    return get_dataloaders(datapoints, machine='local', batch_size=batch_size, test_fraction=test_fraction)


def init_classifier(conv_cutoff, num_convs, embedding_depth, dropout, graph_norm, fc_norm, num_fcs, message_depth):
    return molecule_graph_model(
        num_atom_feats=1,
        output_dimension=9 + 1,
        graph_aggregator='molwise',
        concat_pos_to_atom_features=False,
        concat_mol_to_atom_features=False,
        concat_crystal_to_atom_features=False,
        activation='gelu',
        num_mol_feats=0,
        num_fc_layers=num_fcs,
        fc_depth=embedding_depth,
        fc_dropout_probability=dropout,
        fc_norm_mode=fc_norm,
        graph_node_norm=graph_norm,
        graph_node_dropout=dropout,
        graph_message_norm=None,
        graph_message_dropout=0,
        num_radial=32,
        num_attention_heads=4,
        graph_message_depth=message_depth,
        graph_node_dims=embedding_depth,
        num_graph_convolutions=num_convs,
        graph_embedding_depth=embedding_depth,
        nodewise_fc_layers=1,
        radial_function='bessel',
        max_num_neighbors=100,
        convolution_cutoff=conv_cutoff,
        atom_type_embedding_dims=5,
        seed=0,
        periodic_structure=False,
        outside_convolution_type='none',
        graph_convolution_type='TransformerConv',
    )


# coords = np.random.normal(size=(10, 3))
# atom_types = np.random.randint(5, 8, size=len(coords)).astype(np.int32)
# mol_flags = np.ones(len(coords)) * 3
#
# write_ovito_xyz(coords, atom_types, mol_flags)


def train_classifier(classifier, optimizer,
                     train_loader, test_loader,
                     num_epochs, wandb,
                     class_names, device,
                     batch_size, reporting_frequency):
    with wandb.init(project='cluster_classifier', entity='mkilgour'):
        test_record = []
        for epoch in range(num_epochs):
            print(f"starting epoch {epoch}")
            wandb.log({'epoch': epoch})

            train_loss = []
            train_true_labels = []
            train_probs = []

            classifier.train(True)
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)

                output = classifier(sample)
                loss = F.cross_entropy(output, sample.y)

                train_probs.append(F.softmax(output, dim=1).cpu().detach().numpy())
                train_true_labels.append(sample.y.cpu().detach().numpy())

                loss.backward()
                if step % batch_size == 0:  # use gradient accumulation for synthetically larger batch sizes
                    optimizer.step()
                    optimizer.zero_grad()

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

            test_record.append(np.mean(test_loss))
            if test_record[-1] == np.amin(test_record):
                torch.save({'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           r'C:\Users\mikem\crystals\clusters\cluster_structures\bulk_trajs1/best_classifier_checkpoint')

            print(f"Log Train Loss {np.log10(np.mean(np.array(train_loss))):.4f}")
            print(f"Log Test Loss {np.log10(np.mean(np.array(test_loss))):.4f}")

            if epoch % reporting_frequency == 0:

                train_true_labels = np.concatenate(train_true_labels)
                train_probs = np.concatenate(train_probs)
                test_true_labels = np.concatenate(test_true_labels)
                test_probs = np.concatenate(test_probs)

                classifier_reporting(train_true_labels, train_probs, class_names, wandb, 'Train')
                classifier_reporting(test_true_labels, test_probs, class_names, wandb, 'Test')

            wandb.log({
                'train_loss': np.asarray(train_loss).mean(),
                'test_loss': np.asarray(test_loss).mean()
            })


def classifier_reporting(true_labels, probs, class_names, wandb, epoch_type):
    # todo add raw numbers to confusion matrix
    # todo add classwise accuracy
    if len(np.unique(true_labels)) == 10:  # only if we have all classes represented
        predicted_class = np.argmax(probs, axis=1)

        train_score = roc_auc_score(true_labels, probs, multi_class='ovo')
        train_f1_score = f1_score(true_labels, predicted_class, average='micro')
        train_cmat = confusion_matrix(true_labels, predicted_class, normalize='true')
        fig = go.Figure(go.Heatmap(z=train_cmat, x=class_names, y=class_names))
        fig.update_layout(xaxis=dict(title="Predicted Forms"),
                          yaxis=dict(title="True Forms")
                          )

        wandb.log({f"{epoch_type} ROC_AUC": train_score,
                   f"{epoch_type} F1 Score": train_f1_score,
                   f"{epoch_type} Confusion Matrix": fig})


def reload_model(model, optimizer, path, reload_optimizer=False):
    """
    load model and state dict from path
    includes fix for potential dataparallel issue
    """
    checkpoint = torch.load(path)
    if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer