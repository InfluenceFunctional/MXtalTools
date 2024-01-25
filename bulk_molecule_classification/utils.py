import numpy as np
import torch
import torch.nn.functional as F

from bulk_molecule_classification.classifier_constants import num2atomicnum
from common.utils import delete_from_dataframe, softmax_np
from models.base_models import molecule_graph_model
from bulk_molecule_classification.mol_classifier import MoleculeClassifier
from common.geometry_calculations import coor_trans_matrix
from bulk_molecule_classification.classifier_constants import defect_names

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

    try:
        box_params = np.stack(box_params)
    except ValueError:  # pad zeros to orthorhombic boxes
        box_params = box_params.tolist()
        for ind in range(len(box_params)):
            if box_params[ind].shape[-1] == 2:
                box_params[ind] = np.concatenate([box_params[ind], np.zeros(3)[:, None]], axis=-1)
        box_params = np.stack(box_params)

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


def reindex_mols(dataset, i, mol_num_atoms):
    ref_coords = torch.Tensor(dataset.loc[i]['coordinates'][0])
    atoms = dataset.loc[i]['atom_type'][0]
    atomic_numbers = torch.tensor([num2atomicnum[atom] for atom in atoms], dtype=torch.long)
    num_molecules = (len(ref_coords)) // mol_num_atoms
    mol_ind = torch.tensor(dataset.loc[i]['mol_ind'][0], dtype=torch.long)
    assert num_molecules == len(torch.unique(mol_ind))
    return atomic_numbers, mol_ind, num_molecules, ref_coords


def filter_mols(dataset, dataset_path, early_only, filter_early, melt_only, no_melt, temperatures,
                periodic_only, aperiodic_only, max_box_volume, min_box_volume):
    if temperatures is not None:
        good_inds = []
        for temperature in temperatures:
            good_inds.append(np.argwhere(np.asarray(dataset['temperature']) == temperature)[:, 0])

        good_inds = np.unique(np.concatenate(good_inds))
        bad_inds = np.asarray([ind for ind in np.arange(len(dataset)) if ind not in good_inds])
        print(f"Temperature filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
    if filter_early:
        bad_inds = np.argwhere(np.asarray(dataset['time_step']) <= int(1e4))[:, 0]  # filter first 10ps steps for equilibration
        print(f"Early filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
    if early_only:
        bad_inds = np.argwhere(np.asarray(dataset['time_step']) >= int(1e6))[:, 0]  # keep only 1 ns maximum
        print(f"Early only filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    if max_box_volume is not None:
        T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))
        approx_box_volume = (T_fc_list[:, 0, 0] * T_fc_list[:, 1, 1] * T_fc_list[:, 2, 2])
        bad_inds = np.argwhere(approx_box_volume > max_box_volume)[0, :]
        print(f"Max box filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    if min_box_volume is not None:
        T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))
        approx_box_volume = (T_fc_list[:, 0, 0] * T_fc_list[:, 1, 1] * T_fc_list[:, 2, 2])
        bad_inds = np.argwhere(approx_box_volume < min_box_volume)[0, :]
        print(f"Min box filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')

    if periodic_only or aperiodic_only:
        num_atoms = np.asarray([len(thing[0]) for thing in dataset['atom_type']])
        T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))
        density = num_atoms / (T_fc_list[:, 0, 0] * T_fc_list[:, 1, 1] * T_fc_list[:, 2, 2])

    if periodic_only:
        bad_inds = np.argwhere(density <= 0.025).flatten()
        print(f"Periodic only filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
    if aperiodic_only:
        bad_inds = np.argwhere(density > 0.025).flatten()
        print(f"Aperiodic only filter killed {len(bad_inds)} out of {len(dataset)} samples")

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
    if True:
        if 'gap_rate' in dataset.columns:
            bad_inds = np.argwhere(np.asarray(dataset['gap_rate']) > 0)[:, 0]  # cannot process gaps right now
            print(f"No Gaps filter killed {len(bad_inds)} out of {len(dataset)} samples")

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
        print(f"Melt only filter killed {len(bad_inds)} out of {len(dataset)} samples")

        good_inds = np.argwhere(targets == melt_class_num)[:, 0]
        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
        targets = targets[good_inds]
    if no_melt:
        bad_inds = np.argwhere(targets == melt_class_num)[:, 0]
        print(f"No Melt filter killed {len(bad_inds)} out of {len(dataset)} samples")

        good_inds = np.argwhere(targets != melt_class_num)[:, 0]

        dataset = delete_from_dataframe(dataset, bad_inds)
        dataset = dataset.reset_index().drop(columns='index')
        targets = targets[good_inds]
    return dataset, targets


def identify_surface_molecules(cluster_coords, cluster_targets, conv_cutoff, good_mols, mol_num_atoms, mol_radii):
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
    return centroids, cluster_mol_ind, coordination_number, defect_type


def pare_fragmented_molecules(cluster_atoms, cluster_coords, cluster_targets, pare_fragmented):
    mol_centroids = cluster_coords.mean(1)
    intramolecular_centroid_dists = torch.linalg.norm(mol_centroids[:, None, :] - cluster_coords, dim=2)
    mol_radii = intramolecular_centroid_dists.amax(1)
    max_mol_radius = torch.quantile(mol_radii, 0.05) * 1.25  # 25% leniency on the 5% quantile
    if pare_fragmented:
        good_mols = torch.argwhere(mol_radii < max_mol_radius)[:, 0]
    else:
        good_mols = torch.arange(len(mol_radii))  # keep everything if periodic
    cluster_coords = cluster_coords[good_mols]
    cluster_atoms = cluster_atoms[good_mols]
    cluster_targets = cluster_targets[good_mols]
    return cluster_atoms, cluster_coords, cluster_targets, good_mols, mol_radii


def compute_mol_radii(cluster_coords, pare_fragmented):
    mol_centroids = cluster_coords.mean(1)
    intramolecular_centroid_dists = torch.linalg.norm(mol_centroids[:, None, :] - cluster_coords, dim=2)
    mol_radii = intramolecular_centroid_dists.amax(1)
    if pare_fragmented:
        max_mol_radius = torch.quantile(mol_radii, 0.05) * 1.25  # 25% leniency on the 5% quantile
        good_mols = torch.argwhere(mol_radii < max_mol_radius)[:, 0]
    else:
        good_mols = torch.arange(len(mol_radii))
    return good_mols, mol_radii


def reindex_molecules(atomic_numbers, i, mol_ind, num_molecules, ref_coords, targets):
    cluster_coords, cluster_atoms = [], []
    for ind in torch.unique(mol_ind):
        inds = mol_ind == ind
        cluster_coords.append(ref_coords[inds])
        cluster_atoms.append(atomic_numbers[inds])

    cluster_coords = torch.stack(cluster_coords)
    cluster_atoms = torch.stack(cluster_atoms)

    cluster_targets = torch.tensor(targets[i].repeat(num_molecules), dtype=torch.long)
    return cluster_atoms, cluster_coords, cluster_targets


def force_molecules_into_box(T_fc_list, cluster_coords, i, periodic):
    """
    will have no effect on fragmented molecules or
    molecules otherwise wrapped
    """
    # recenter about zero
    if periodic:  # don't do this for clusters or other floating objects
        cluster_coords -= cluster_coords.amin((0, 1))[None, None, :]
    mol_centroids = cluster_coords.mean(1)
    frac_mol_centroids = mol_centroids @ torch.linalg.inv(T_fc_list[i].T)
    adjustment_fractional_vector = -torch.floor(frac_mol_centroids)
    adjustment_cart_vector = adjustment_fractional_vector @ T_fc_list[i].T
    cluster_coords += adjustment_cart_vector[:, None, :]
    return cluster_coords


def pare_cluster_radius(cluster_atoms, cluster_coords, cluster_targets, max_cluster_radius):
    mol_centroid_dists = torch.linalg.norm(cluster_coords.mean(1) - cluster_coords.mean((0, 1)), dim=1)
    good_mols = torch.argwhere(mol_centroid_dists < max_cluster_radius)[:, 0]  # 60 angstrom sphere at maximum
    cluster_coords = cluster_coords[good_mols]
    cluster_atoms = cluster_atoms[good_mols]
    cluster_targets = cluster_targets[good_mols]
    return cluster_atoms, cluster_coords, cluster_targets


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


def new_init_classifier(conv_cutoff, num_convs, embedding_depth, dropout, graph_norm, fc_norm, num_fcs, message_depth, num_forms, num_topologies, seed):
    return MoleculeClassifier(
        input_node_depth=1,
        node_embedding_depth=embedding_depth,
        nodewise_fc_layers=1,
        message_depth=message_depth,
        num_blocks=num_convs,
        nodewise_norm=graph_norm,
        nodewise_dropout=dropout,
        num_fcs=num_fcs,
        fc_norm=fc_norm,
        output_dimension=num_forms + num_topologies,
        activation='gelu',
        num_radial=32,
        graph_embedding_depth=embedding_depth,
        radial_embedding='bessel',
        max_num_neighbors=100,
        cutoff=conv_cutoff,
        embedding_hidden_dimension=5,
        seed=seed,
        convolution_type='TransformerConv',
    )


def classifier_reporting(true_labels, true_defects, probs, class_names, ordered_class_names, wandb, epoch_type):
    present_classes = np.unique(true_labels)
    present_class_names = [ordered_class_names[ind] for ind in present_classes]

    type_probs = softmax_np(probs[:, present_classes])
    predicted_class = np.argmax(type_probs, axis=1)

    present_defects = np.unique(true_defects)
    present_defect_names = [defect_names[ind] for ind in present_defects]
    defect_probs = softmax_np(probs[:, len(present_classes):])
    predicted_defect = np.argmax(defect_probs, axis=-1)

    train_score = roc_auc_score(true_labels, type_probs, multi_class='ovo')
    train_f1_score = f1_score(true_labels, predicted_class, average='micro')
    train_cmat = confusion_matrix(true_labels, predicted_class, normalize='true')
    fig = go.Figure(go.Heatmap(z=train_cmat, x=present_class_names, y=present_class_names))
    fig.update_layout(xaxis=dict(title="Predicted Forms"),
                      yaxis=dict(title="True Forms")
                      )

    wandb.log({f"{epoch_type} ROC_AUC": train_score,
               f"{epoch_type} F1 Score": train_f1_score,
               f"{epoch_type} 1-ROC_AUC": 1 - train_score,
               f"{epoch_type} 1-F1 Score": 1 - train_f1_score,
               f"{epoch_type} Confusion Matrix": fig})

    if len(present_defects) > 1:
        train_score = roc_auc_score(true_defects, defect_probs[:, 1], multi_class='ovo')
        train_f1_score = f1_score(true_defects, predicted_defect, average='micro')
        train_cmat = confusion_matrix(true_defects, predicted_defect, normalize='true')
        fig = go.Figure(go.Heatmap(z=train_cmat, x=present_defect_names, y=present_defect_names))
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


def record_step_results(results_dict, output, sample, data, latents, embeddings, step, config, index_offset=0):
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
                        'Coordination_Numbers': [],
                        'Embeddings': []}

    results_dict['Loss'].append(get_loss(output, sample, config['num_forms']).cpu().detach().numpy())
    results_dict['Type_Prediction'].append(F.softmax(output[:, :config['num_forms']], dim=1).cpu().detach().numpy())
    results_dict['Defect_Prediction'].append(F.softmax(output[:, config['num_forms']:], dim=1).cpu().detach().numpy())
    results_dict['Targets'].append(sample.y.cpu().detach().numpy())
    results_dict['Defects'].append(sample.defect.cpu().detach().numpy())
    results_dict['Latents'].append(latents.cpu().detach().numpy())  # ['final_activation'])
    results_dict['Embeddings'].append(embeddings.cpu().detach().numpy())
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
    keys = list(results_dict.keys())
    for key in keys:
        if len(results_dict[key]) == num_atoms:
            index = results_dict['Atomwise_Sample_Index']
            molwise_results_dict[key] = [results_dict[key][index == ind] for ind in range(len(loader))]

        elif len(results_dict[key]) == num_mols:
            index = results_dict['Sample_Index']
            molwise_results_dict['Molecule_' + key] = [results_dict[key][index == ind] for ind in range(len(loader))]
            # molwise_results_dict[key] = [results_dict[key][index == ind].repeat(mol_num_atoms) for ind in range(len(loader))]
        else:
            print(f"{key} is omitted from results dict")
            pass

        if 'Index' not in key:
            del results_dict[key]

    time_inds = [time[0] for time in molwise_results_dict['Molecule_Time_Step']]
    sort_inds = np.argsort(np.asarray(time_inds))

    sorted_molwise_results_dict = {}
    for key in molwise_results_dict.keys():
        sorted_molwise_results_dict[key] = [molwise_results_dict[key][ind] for ind in sort_inds]

    centroid_dists = []
    for ind in range(len(sorted_molwise_results_dict['Coordinates'])):
        coords = sorted_molwise_results_dict['Coordinates'][ind]
        centroids = coords.reshape(coords.shape[0] // mol_num_atoms, mol_num_atoms, 3).mean(1)
        centroid_dists.append(np.linalg.norm(centroids - centroids.mean(0), axis=1))

    sorted_molwise_results_dict['Centroid Radii'] = centroid_dists

    return sorted_molwise_results_dict, np.asarray(time_inds)[sort_inds]


def get_loss(output, sample, num_forms):
    return F.cross_entropy(output[:, :num_forms], sample.y) + F.cross_entropy(output[:, num_forms:], sample.defect)
