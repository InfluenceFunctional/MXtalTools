import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from bulk_molecule_classification.classifier_constants import num2atomicnum
from common.utils import delete_from_dataframe, softmax_np
from dataset_management.CrystalData import CrystalData
from dataset_management.utils import get_dataloaders
from models.base_models import molecule_graph_model
from common.geometry_calculations import coor_trans_matrix
from bulk_molecule_classification.classifier_constants import defect_names, nic_ordered_class_names

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import plotly.graph_objects as go


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
    if box_params[0].shape == (3, 3):  # non-orthogonal box
        xy = box_params[:, 0, 2]
        xz = box_params[:, 1, 2]
        yz = box_params[:, 2, 2]
    else:
        xy = np.zeros_like(xhi_bound)
        xz = np.zeros_like(xy)
        yz = np.zeros_like(xy)

    xlo = xlo_bound - np.stack((np.zeros_like(xy), xy, xz, xy + xz)).T.min(1)
    xhi = xhi_bound - np.stack((np.zeros_like(xy), xy, xz, xy + xz)).T.max(1)
    ylo = ylo_bound - np.stack((np.zeros_like(yz), yz)).T.min(1)
    yhi = yhi_bound - np.stack((np.zeros_like(yz), yz)).T.max(1)
    zlo = zlo_bound
    zhi = zhi_bound

    av = np.asarray([xhi - xlo, np.zeros_like(xhi), np.zeros_like(xhi)]).T
    bv = np.asarray([xy, yhi - ylo, np.zeros_like(xy)]).T
    cv = np.asarray([xz, yz, zhi - zlo]).T

    T_fc_list = np.zeros((len(av), 3, 3))
    for i in range(len(T_fc_list)):  # warning dubious
        T_fc_list[i] = np.stack((av[i], bv[i], cv[i])).T

    a = xhi - xlo
    b = np.sqrt((yhi - ylo) ** 2 + xy ** 2)
    c = np.sqrt((zhi - zlo) ** 2 + xz ** 2 + yz ** 2)
    alpha = np.arccos((xy * xz + (yhi - ylo) * yz) / (b * c))
    beta = np.arccos(xz / c)
    gamma = np.arccos(xy / b)
    lengths = np.stack([a, b, c]).T
    angles = np.stack([alpha, beta, gamma]).T

    T_fc_list2 = []  # double check the answer
    for i in range(len(lengths)):
        T_fc_list2.append(coor_trans_matrix('f_to_c', lengths[i], angles[i]))
    T_fc_list2 = np.stack(T_fc_list2)

    assert np.sum(np.abs(T_fc_list - T_fc_list2)) < 1e-3

    return T_fc_list


def collect_to_traj_dataloaders(mol_num_atoms, dataset_path, dataset_size, batch_size,
                                conv_cutoff, test_fraction=0.2, filter_early=True,
                                temperatures: list = None, shuffle=True,
                                melt_only=False, no_melt=False, early_only=False):
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
        bad_inds = np.argwhere(np.asarray(dataset['time_step']) <= int(1e4))[:, 0]  # filter first 10ps steps for equilibration
        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    if early_only:
        bad_inds = np.argwhere(np.asarray(dataset['time_step']) >= int(1e6))[:, 0]  # keep only 1 ns maximum
        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    if True:
        if 'gap_rate' in dataset.columns:
            bad_inds = np.argwhere(np.asarray(dataset['gap_rate']) > 0)[:, 0]  # cannot process gaps right now
            dataset = delete_from_dataframe(dataset, bad_inds)
            dataset = dataset.reset_index().drop(columns='index')

    # forms = np.sort(np.unique(dataset['form']))
    # forms2tgt = {form: i for i, form in enumerate(forms)}
    targets = np.asarray(dataset['form']) - 1  # no longer need to reindex, as we have this now managed through the constants
    # this will throw an error later if the combined dataset is missing any forms, but it shouldn't be missing any forms in general
    # so that's fine

    # set high temperature samples to 'melted' class
    if 'urea' in dataset_path:
        melt_class_num = 6
    else:
        melt_class_num = 9  # nicotinamide

    if melt_only:
        bad_inds = np.argwhere(targets != melt_class_num)[:, 0]
        good_inds = np.argwhere(targets == melt_class_num)[:, 0]
        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
        targets = targets[good_inds]

    if no_melt:
        bad_inds = np.argwhere(targets == melt_class_num)[:, 0]
        good_inds = np.argwhere(targets != melt_class_num)[:, 0]

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
        targets = targets[good_inds]

    # T_fc_list = convert_box_to_cell_params(np.stack(dataset['cell_params']))  # we don't use this anywhere

    print('Generating training datapoints')
    datapoints = []
    time_inds = np.arange(len(dataset))[::max(len(dataset) // dataset_size, 1)]
    for i in tqdm(time_inds):
        ref_coords = torch.Tensor(dataset.loc[i]['coordinates'][0])
        atoms = dataset.loc[i]['atom_type'][0]

        atomic_numbers = torch.tensor([num2atomicnum[atom] for atom in atoms], dtype=torch.long)
        num_molecules = (len(ref_coords)) // mol_num_atoms

        mol_ind = torch.tensor(dataset.loc[i]['mol_ind'][0], dtype=torch.long)
        assert num_molecules == len(torch.unique(mol_ind))

        # we cannot trust the default indexing, so manually reindex according to the recorded molecule index
        cluster_coords = torch.stack([ref_coords[mol_ind == ind] for ind in torch.unique(mol_ind)])
        cluster_atoms = torch.stack([atomic_numbers[mol_ind == ind] for ind in torch.unique(mol_ind)])
        cluster_targets = torch.tensor(targets[i].repeat(num_molecules), dtype=torch.long)

        # pare off molecules which are fragmented
        mol_centroids = cluster_coords.mean(1)
        intramolecular_centroid_dists = torch.linalg.norm(mol_centroids[:, None, :] - cluster_coords, dim=2)
        mol_radii = intramolecular_centroid_dists.amax(1)

        max_mol_radius = torch.quantile(mol_radii, 0.05) * 1.25  # 25% leniency on the 5% quantile

        good_mols = torch.argwhere(mol_radii < max_mol_radius)[:, 0]
        cluster_coords = cluster_coords[good_mols]
        cluster_atoms = cluster_atoms[good_mols]
        cluster_targets = cluster_targets[good_mols]

        # identify surface mols
        coord_shell_num = 20

        true_max_mol_radius = torch.amax(mol_radii[good_mols])
        centroids = cluster_coords.mean(1)
        dist = torch.cdist(centroids, centroids)
        coordination_cutoff = true_max_mol_radius + conv_cutoff
        coordination_number = (dist < coordination_cutoff).sum(1)
        surface_mols_ind = torch.argwhere(coordination_number < coord_shell_num)[:, 0]

        defect_type = torch.zeros_like(cluster_targets)
        defect_type[surface_mols_ind] = 1  # defect type 1 is surfaces

        # cluster_targets[surface_mols_ind] = len(forms2tgt)  # label surface molecules as 'disordered'
        cluster_mol_ind = torch.arange(len(good_mols)).repeat(mol_num_atoms, 1).T

        datapoints.append(
            CrystalData(
                x=cluster_atoms.reshape(len(good_mols) * mol_num_atoms),
                mol_ind=cluster_mol_ind.reshape(len(good_mols) * mol_num_atoms),
                pos=cluster_coords.reshape(len(good_mols) * mol_num_atoms, 3),
                y=cluster_targets,
                defect=defect_type,
                centroid_pos=(centroids - centroids.mean(0)).cpu().detach().numpy(),
                coord_number=coordination_number.cpu().detach().numpy(),
                tracking=np.asarray(dataset.loc[i, ['temperature', 'time_step']]),
                asym_unit_handedness=torch.ones(1),
                symmetry_operators=torch.ones(1),
            )
        )
    del dataset
    return get_dataloaders(datapoints, machine='local', batch_size=batch_size, test_fraction=test_fraction, shuffle=shuffle)


def init_classifier(conv_cutoff, num_convs, embedding_depth, dropout, graph_norm, fc_norm, num_fcs, message_depth, num_forms, num_topologies, seed):
    return molecule_graph_model(
        num_atom_feats=1,
        output_dimension=num_forms + num_topologies,
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
        seed=seed,
        periodic_structure=False,
        outside_convolution_type='none',
        graph_convolution_type='TransformerConv',
    )


def classifier_reporting(true_labels, true_defects, probs, class_names, ordered_class_names, wandb, epoch_type):
    if len(np.unique(true_labels)) == len(class_names):  # only if we have all classes represented
        type_probs = softmax_np(probs[:, :len(class_names)])
        predicted_class = np.argmax(type_probs, axis=1)

        defect_probs = softmax_np(probs[:, len(class_names):])
        predicted_defect = np.argmax(defect_probs, axis=-1)

        train_score = roc_auc_score(true_labels, type_probs, multi_class='ovo')
        train_f1_score = f1_score(true_labels, predicted_class, average='micro')
        train_cmat = confusion_matrix(true_labels, predicted_class, normalize='true')
        fig = go.Figure(go.Heatmap(z=train_cmat, x=ordered_class_names, y=ordered_class_names))
        fig.update_layout(xaxis=dict(title="Predicted Forms"),
                          yaxis=dict(title="True Forms")
                          )

        wandb.log({f"{epoch_type} ROC_AUC": train_score,
                   f"{epoch_type} F1 Score": train_f1_score,
                   f"{epoch_type} 1-ROC_AUC": 1 - train_score,
                   f"{epoch_type} 1-F1 Score": 1 - train_f1_score,
                   f"{epoch_type} Confusion Matrix": fig})

        train_score = roc_auc_score(true_defects, defect_probs[:, 1], multi_class='ovo')
        train_f1_score = f1_score(true_defects, predicted_defect, average='micro')
        train_cmat = confusion_matrix(true_defects, predicted_defect, normalize='true')
        fig = go.Figure(go.Heatmap(z=train_cmat, x=defect_names, y=defect_names))
        fig.update_layout(xaxis=dict(title="Predicted Defect"),
                          yaxis=dict(title="True Defect")
                          )

        wandb.log({f"{epoch_type} Defect ROC_AUC": train_score,
                   f"{epoch_type} Defect F1 Score": train_f1_score,
                   f"{epoch_type} 1-Defect ROC_AUC": 1 - train_score,
                   f"{epoch_type} 1-Defect F1 Score": 1 - train_f1_score,
                   f"{epoch_type} Defect Confusion Matrix": fig})


def reload_model(model, device, optimizer, path, reload_optimizer=False):
    """
    load model and state dict from path
    includes fix for potential dataparallel issue
    """
    checkpoint = torch.load(path, map_location=device)
    if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def record_step_results(results_dict, output, sample, data, latents_dict, step, config, index_offset=0):
    if results_dict is None:
        results_dict = {'Temperature': [],
                        'Time_Step': [],
                        'Loss': [],
                        'Type_Prediction': [],
                        'Defect_Prediction': [],
                        'Targets': [],
                        'Defects': [],
                        'Latents': [],
                        'Sample_Index': [],
                        'Coordinates': [],
                        'Atom_Types': [],
                        'Molecule_Index': [],
                        'Molecule_Centroids': [],
                        'Coordination_Numbers': []}

    results_dict['Loss'].append(get_loss(output, sample, config['num_forms']).cpu().detach().numpy())
    results_dict['Type_Prediction'].append(F.softmax(output[:, :config['num_forms']], dim=1).cpu().detach().numpy())
    results_dict['Defect_Prediction'].append(F.softmax(output[:, config['num_forms']:], dim=1).cpu().detach().numpy())
    results_dict['Targets'].append(sample.y.cpu().detach().numpy())
    results_dict['Defects'].append(sample.defect.cpu().detach().numpy())
    results_dict['Latents'].append(latents_dict['final_activation'])
    results_dict['Temperature'].append(np.ones(len(sample.y)) * data.tracking[0][0])
    results_dict['Time_Step'].append(np.ones(len(sample.y)) * data.tracking[0][1])
    results_dict['Sample_Index'].append(np.ones(len(sample.y)) * step + index_offset)
    results_dict['Coordinates'].append(sample.pos.cpu().detach().numpy())
    results_dict['Atom_Types'].append(sample.x.cpu().detach().numpy())
    results_dict['Molecule_Index'].append(sample.mol_ind.cpu().detach().numpy())
    results_dict['Molecule_Centroids'].append(sample.centroid_pos[0])
    results_dict['Coordination_Numbers'].append(sample.coord_number[0])

    return results_dict


def process_trajectory_results_dict(results_dict, loader, mol_num_atoms):
    num_atoms = len(results_dict['Atomwise_Sample_Index'])
    num_mols = len(results_dict['Sample_Index'])
    molwise_results_dict = {}
    for key in results_dict.keys():
        if len(results_dict[key]) == num_atoms:
            index = results_dict['Atomwise_Sample_Index']
            molwise_results_dict[key] = [results_dict[key][index == ind] for ind in range(len(loader))]

        elif len(results_dict[key]) == num_mols:
            index = results_dict['Sample_Index']
            molwise_results_dict[key] = [results_dict[key][index == ind].repeat(mol_num_atoms) for ind in range(len(loader))]
            molwise_results_dict['Molecule_' + key] = [results_dict[key][index == ind] for ind in range(len(loader))]
        else:
            print(f"{key} is omitted from results dict")
            pass

    time_inds = [time[0] for time in molwise_results_dict['Time_Step']]
    sort_inds = np.argsort(np.asarray(time_inds))

    sorted_molwise_results_dict = {}
    for key in molwise_results_dict.keys():
        sorted_molwise_results_dict[key] = [molwise_results_dict[key][ind] for ind in sort_inds]

    return sorted_molwise_results_dict, np.asarray(time_inds)[sort_inds]


def get_loss(output, sample, num_forms):
    return F.cross_entropy(output[:, :num_forms], sample.y) + F.cross_entropy(output[:, num_forms:], sample.defect)
