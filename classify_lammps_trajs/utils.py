import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from classify_lammps_trajs.NICOAM_constants import num2atomicnum, identifier2form
from classify_lammps_trajs.dump_data_processing import generate_dataset_from_dumps
from classify_lammps_trajs.traj_analysis_figs import embedding_fig, form_accuracy_fig, classifier_trajectory_analysis_fig, defect_accuracy_fig
from common.utils import delete_from_dataframe, softmax_np
from crystal_building.utils import fractional_transform
from dataset_management.CrystalData import CrystalData
from dataset_management.utils import get_dataloaders
from models.base_models import molecule_graph_model
from common.geometry_calculations import coor_trans_matrix
from classify_lammps_trajs.NICOAM_constants import defect_names

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


def collect_to_traj_dataloaders(dataset_path, dataset_size, batch_size, test_fraction=0.2, filter_early=True, temperatures: list = None, shuffle=True):
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

    # if True:
    #     bad_inds = np.argwhere(np.asarray(dataset['gap_rate']) > 0)[:, 0]  # cannot process gaps right now
    #     dataset = delete_from_dataframe(dataset, bad_inds)
    #     dataset = dataset.reset_index().drop(columns='index')

    forms = np.unique(dataset['form'])
    forms2tgt = {form: i for i, form in enumerate(forms)}
    targets = np.asarray(
        [forms2tgt[form] for form in dataset['form']]
    )

    # set high temperature samples to 'melted' class
    for i in range(len(dataset)):
        if dataset.iloc[i]['temperature'] == 950:
            targets[i] = 9

    T_fc_list = convert_box_to_cell_params(np.stack(dataset['cell_params']))  # we don't use this anywhere

    print('Generating training datapoints')
    datapoints = []
    rand_inds = np.random.choice(len(dataset), size=min(dataset_size, len(dataset)), replace=False)  # todo build indexing capability for defects
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

        defect_type = torch.zeros_like(cluster_targets)
        defect_type[surface_mols_ind] = 1  # defect type 1 is surfaces

        #cluster_targets[surface_mols_ind] = len(forms2tgt)  # label surface molecules as 'disordered'
        cluster_mol_ind = torch.arange(len(good_mols)).repeat(15, 1).T

        #
        # if dataset.loc[i]['gap_rate'] > 0:
        #     # find gaps and any neighboring molecules
        #     aa = 1
        #     from scipy.stats import gaussian_kde
        #     transform = gaussian_kde(cluster_coords.reshape(np.prod(cluster_coords.shape[:2]), 3).T, bw_method=0.1)
        #     refrac = fractional_transform(cluster_coords.reshape(np.prod(cluster_coords.shape[:2]), 3), torch.linalg.inv(torch.Tensor(T_fc_list[i])))
        #     frac_lin_list = [torch.linspace(0.25, 0.75, 25) for ind in range(3)]  # look only well within the structure
        #     frac_grid = torch.cartesian_prod(*frac_lin_list)
        #     cart_grid = fractional_transform(frac_grid, torch.Tensor(T_fc_list[i]))
        #
        #     density = transform(cart_grid.T)
        #     point_density = transform(centroids.T)
        #     x, y, z = cart_grid.T
        #
        #     normed_density = density / density.sum()
        #     void = 1 / normed_density
        #
        #     fig = go.Figure()
        #     fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker_color=void, marker_opacity=0.1)
        #     # fig.add_scatter3d(x=x, y=y, z=z, mode='markers', marker_color=density, marker_opacity=0.1)
        #     # fig.add_scatter3d(x=centroids[:,0], y=centroids[:,1], z=centroids[:,2], mode='markers', marker_color=point_density, marker_opacity=0.5)
        #     fig.show()
        #

        datapoints.append(
            CrystalData(
                x=cluster_atoms.reshape(len(good_mols) * 15),
                mol_ind=cluster_mol_ind.reshape(len(good_mols) * 15),
                pos=cluster_coords.reshape(len(good_mols) * 15, 3),
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


def init_classifier(conv_cutoff, num_convs, embedding_depth, dropout, graph_norm, fc_norm, num_fcs, message_depth):
    return molecule_graph_model(
        num_atom_feats=1,
        output_dimension=9 + 1 + 2,  # 9 forms + disordered: bulk or surface
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


def train_classifier(config, classifier, optimizer,
                     train_loader, test_loader,
                     num_epochs, wandb,
                     class_names, device,
                     batch_size, reporting_frequency,
                     runs_path, run_name):
    with wandb.init(project='cluster_classifier', entity='mkilgour'):
        wandb.run.name = run_name
        wandb.log({'config': config})
        test_record = []
        time_since_best = 0
        for epoch in range(num_epochs):
            print(f"starting epoch {epoch}")
            wandb.log({'epoch': epoch})

            train_loss = []
            train_true_labels = []
            train_probs = []
            train_true_defects = []
            test_loss = []
            test_probs = []
            test_true_labels = []
            test_true_defects = []
            classifier.train(True)
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)

                output = classifier(sample)
                loss = get_loss(output, sample)
                train_probs.append(output.cpu().detach().numpy())
                train_true_labels.append(sample.y.cpu().detach().numpy())
                train_true_defects.append(sample.defect.cpu().detach().numpy())

                loss.backward()
                if step % batch_size == 0:  # use gradient accumulation for synthetically larger batch sizes
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss.append(loss.cpu().detach().numpy())

            with torch.no_grad():
                classifier.eval()

                for step, data in enumerate(tqdm(test_loader)):
                    sample = data.to(device)

                    output = classifier(sample)  # fix mini-batch behavior
                    loss = get_loss(output, sample)

                    test_loss.append(loss.cpu().detach().numpy())
                    test_probs.append(output.cpu().detach().numpy())
                    test_true_labels.append(sample.y.cpu().detach().numpy())
                    test_true_defects.append(sample.defect.cpu().detach().numpy())


            test_record.append(np.mean(test_loss))
            if test_record[-1] == np.amin(test_record):
                torch.save({'model_state_dict': classifier.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           runs_path + run_name + '_best_classifier_checkpoint')

                time_since_best = 0
            else:
                time_since_best += 1

            if time_since_best > config['convergence_history']:
                break  # stop training if we are not improving test loss

            print(f"Log Train Loss {np.log10(np.mean(np.array(train_loss))):.4f}")
            print(f"Log Test Loss {np.log10(np.mean(np.array(test_loss))):.4f}")

            if epoch % reporting_frequency == 0:
                train_true_labels = np.concatenate(train_true_labels)
                train_true_defects = np.concatenate(train_true_defects)
                train_probs = np.concatenate(train_probs)
                test_true_labels = np.concatenate(test_true_labels)
                test_true_defects = np.concatenate(test_true_defects)
                test_probs = np.concatenate(test_probs)

                classifier_reporting(train_true_labels, train_true_defects, train_probs, class_names, wandb, 'Train')
                classifier_reporting(test_true_labels, test_true_defects, test_probs, class_names, wandb, 'Test')

            wandb.log({
                'train_loss': np.asarray(train_loss).mean(),
                'test_loss': np.asarray(test_loss).mean()
            })


def classifier_reporting(true_labels, true_defects, probs, class_names, wandb, epoch_type):
    if len(np.unique(true_labels)) == 10:  # only if we have all classes represented
        type_probs = softmax_np(probs[:, :10])
        predicted_class = np.argmax(type_probs, axis=1)

        defect_probs = softmax_np(probs[:, 10:])
        predicted_defect = np.argmax(defect_probs, axis=-1)

        train_score = roc_auc_score(true_labels, type_probs, multi_class='ovo')
        train_f1_score = f1_score(true_labels, predicted_class, average='micro')
        train_cmat = confusion_matrix(true_labels, predicted_class, normalize='true')
        fig = go.Figure(go.Heatmap(z=train_cmat, x=class_names, y=class_names))
        fig.update_layout(xaxis=dict(title="Predicted Forms"),
                          yaxis=dict(title="True Forms")
                          )

        wandb.log({f"{epoch_type} ROC_AUC": train_score,
                   f"{epoch_type} F1 Score": train_f1_score,
                   f"{epoch_type} Confusion Matrix": fig})

        train_score = roc_auc_score(true_defects, defect_probs[:,1], multi_class='ovo')
        train_f1_score = f1_score(true_defects, predicted_defect, average='micro')
        train_cmat = confusion_matrix(true_defects, predicted_defect, normalize='true')
        fig = go.Figure(go.Heatmap(z=train_cmat, x=defect_names, y=defect_names))
        fig.update_layout(xaxis=dict(title="Predicted Defect"),
                          yaxis=dict(title="True Defect")
                          )

        wandb.log({f"{epoch_type} Defect ROC_AUC": train_score,
                   f"{epoch_type} Defect F1 Score": train_f1_score,
                   f"{epoch_type} Defect Confusion Matrix": fig})


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


def record_step_results(results_dict, output, sample, data, latents_dict, step, index_offset=0):
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

    results_dict['Loss'].append(get_loss(output, sample).cpu().detach().numpy())
    results_dict['Type_Prediction'].append(F.softmax(output[:, :10], dim=1).cpu().detach().numpy())
    results_dict['Defect_Prediction'].append(F.softmax(output[:, 10:], dim=1).cpu().detach().numpy())
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


def classifier_evaluation(config, classifier, optimizer,
                          train_loader, test_loader,
                          num_epochs, wandb,
                          class_names, device,
                          batch_size, reporting_frequency,
                          runs_path, run_name):
    with wandb.init(project='cluster_classifier', entity='mkilgour'):
        wandb.run.name = run_name + '_evaluation'
        wandb.log({'config': config})

        results_dict = None
        classifier.train(False)
        with torch.no_grad():
            classifier.eval()
            for step, data in enumerate(tqdm(train_loader)):
                sample = data.to(device)
                output, latents_dict = classifier(sample, return_latent=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents_dict, step)

            for step, data in enumerate(tqdm(test_loader)):
                sample = data.to(device)
                output, latents_dict = classifier(sample, return_latent=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents_dict, step, index_offset=len(train_loader))

        for key in results_dict.keys():
            try:
                results_dict[key] = np.stack(results_dict[key])
            except:
                results_dict[key] = np.concatenate(results_dict[key], axis=0)

        results_dict['Atomwise_Sample_Index'] = results_dict['Sample_Index'].repeat(15)
        num_samples = len(results_dict['Targets'])

        fig_dict = {}
        '''embed train, test, melt'''
        fig_dict['Embedding_Analysis'] = embedding_fig(results_dict, num_samples)

        '''accuracy metrics'''
        fig_dict['Form_Confusion_Matrices'], accuracy_scores = form_accuracy_fig(results_dict)
        fig_dict['Defect_Confusion_Matrices'], accuracy_scores = defect_accuracy_fig(results_dict)
        fig_dict['All_Confusion_Matrices'], accuracy_scores = all_accuracy_fig(results_dict)

        '''model games'''

        '''training curves'''

        os.chdir(config['runs_path'])
        # [fig_dict[key].write_image(f'Figure_{i}') for i, key in enumerate(fig_dict.keys())]
        wandb.log(fig_dict)


def trajectory_analysis(config, classifier, run_name, wandb, device, dumps_dir):

    # if not os.path.exists('traj_analysis_outputs.npy'):
    #     traj_analysis = {}
    # else:
    #     traj_analysis = np.load('traj_analysis_outputs.npy', allow_pickle=True).item()

    from classify_lammps_trajs.ovito_utils import write_ovito_xyz
    dataset_name = ''.join(dumps_dir.split('/')[-3:])
    datasets_path = config['datasets_path']
    dataset_path = f'{datasets_path}{dataset_name}.pkl'

    if True:  # dataset_path not in traj_analysis.keys():

        if not os.path.exists(dataset_path):
            made_dataset = generate_dataset_from_dumps([dumps_dir], dataset_path)  # todo add more expressive dataset name

            if not made_dataset:
                print(f'{dumps_dir} does not contain valid dump to analyze')
                return False

        os.chdir(config['runs_path'])

        _, loader = collect_to_traj_dataloaders(
            dataset_path, int(1e7), batch_size=1, temperatures=None, test_fraction=1, shuffle=False, filter_early=False)

        results_dict = None
        classifier.train(False)
        with torch.no_grad():
            classifier.eval()
            for step, data in enumerate(tqdm(loader)):
                sample = data.to(device)
                output, latents_dict = classifier(sample, return_latent=True)
                results_dict = record_step_results(results_dict, output, sample, data, latents_dict, step)

        for key in results_dict.keys():
            try:
                results_dict[key] = np.concatenate(results_dict[key], axis=0)
            except:
                results_dict[key] = np.stack(results_dict[key])

        results_dict['Atomwise_Sample_Index'] = results_dict['Sample_Index'].repeat(15)
        results_dict['Type_Prediction_Choice'] = np.argmax(results_dict['Type_Prediction'], axis=-1)  # argmax sample
        results_dict['Type_Prediction_Confidence'] = np.amax(results_dict['Type_Prediction'], axis=-1)  # argmax sample

        sorted_molwise_results_dict, time_steps = process_trajectory_results_dict(results_dict, loader)

        write_ovito_xyz(sorted_molwise_results_dict['Coordinates'],
                        sorted_molwise_results_dict['Atom_Types'],
                        sorted_molwise_results_dict['Type_Prediction_Choice'], filename=dataset_name + '_prediction')  # write a trajectory  # todo give more descriptive title

        write_ovito_xyz(sorted_molwise_results_dict['Coordinates'],
                        sorted_molwise_results_dict['Atom_Types'],
                        sorted_molwise_results_dict['Type_Prediction_Confidence'], filename=dataset_name + '_confidence')  # write a trajectory  # todo give more descriptive title

        fig, fig2 = classifier_trajectory_analysis_fig(sorted_molwise_results_dict, time_steps)  # todo add confidence to prediction

        run_config = np.load(dumps_dir + 'run_config.npy', allow_pickle=True).item()
        fig.update_layout(
            title=f"Form {identifier2form[run_config['structure_identifier']]}, "
                  f"Cluster Radius {run_config['max_sphere_radius']}A, "
                  f"Temperature {run_config['temperature']}K")
        fig2.update_layout(
            title=f"Form {identifier2form[run_config['structure_identifier']]}, "
                  f"Cluster Radius {run_config['max_sphere_radius']}A, "
                  f"Temperature {run_config['temperature']}K")


        # if not os.path.exists('traj_analysis_outputs.npy'):
        #     traj_analysis = {}
        # else:
        #     traj_analysis = np.load('traj_analysis_outputs.npy',allow_pickle=True).item()

        # traj_analysis[dataset_path] = sorted_molwise_results_dict
        # np.save('traj_analysis_outputs', traj_analysis)

        wandb.log({f"{dataset_name} Trajectory Analysis": fig,
                   f"{dataset_name} Confidence Analysis": fig2})


def process_trajectory_results_dict(results_dict, loader):
    num_atoms = len(results_dict['Atomwise_Sample_Index'])
    num_mols = len(results_dict['Sample_Index'])
    molwise_results_dict = {}
    for key in results_dict.keys():
        if len(results_dict[key]) == num_atoms:
            index = results_dict['Atomwise_Sample_Index']
            molwise_results_dict[key] = [results_dict[key][index == ind] for ind in range(len(loader))]

        elif len(results_dict[key]) == num_mols:
            index = results_dict['Sample_Index']
            molwise_results_dict[key] = [results_dict[key][index == ind].repeat(15) for ind in range(len(loader))]
            molwise_results_dict['Molecule_' + key] = [results_dict[key][index == ind] for ind in range(len(loader))]
        else:
            pass

    time_inds = [time[0] for time in molwise_results_dict['Time_Step']]
    sort_inds = np.argsort(np.asarray(time_inds))

    sorted_molwise_results_dict = {}
    for key in molwise_results_dict.keys():
        sorted_molwise_results_dict[key] = [molwise_results_dict[key][ind] for ind in sort_inds]

    return sorted_molwise_results_dict, np.asarray(time_inds)[sort_inds]


def get_loss(output, sample):
    return F.cross_entropy(output[:, :10], sample.y) + F.cross_entropy(output[:, 10:], sample.defect)
