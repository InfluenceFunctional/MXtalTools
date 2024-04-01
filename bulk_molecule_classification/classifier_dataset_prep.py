import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common.mol_classifier_utils import filter_mols, convert_box_to_cell_params, reindex_mols, reindex_molecules, force_molecules_into_box, pare_cluster_radius, \
    identify_surface_molecules, pare_fragmented_molecules
from dataset_management.CrystalData import CrystalData
from dataset_management.utils import get_dataloaders


def collect_to_traj_dataloaders(mol_num_atoms, dataset_path, dataset_size, batch_size,
                                conv_cutoff, test_fraction=0.2, filter_early=True,
                                temperatures: list = None, shuffle=True,
                                melt_only=False, no_melt=False, early_only=False,
                                run_config=None, pare_to_cluster=False,
                                periodic_only=False, aperiodic_only=False,
                                max_cluster_radius=None, max_box_volume=None,
                                min_box_volume=None, single_trajectory=False):
    dataset = pd.read_pickle(dataset_path)
    dataset = dataset.reset_index().drop(columns='index')  # reindexing is crucial here

    dataset, targets = filter_mols(dataset, dataset_path, early_only, filter_early,
                                   melt_only, no_melt, temperatures, periodic_only, aperiodic_only,
                                   max_box_volume, min_box_volume)
    assert len(dataset) > 0, "Full dataset was filtered!"
    # recompute from filtered
    T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))

    print('Generating training datapoints')
    datapoints = []
    if single_trajectory:
        t_jump = max(len(dataset) // dataset_size, 1)
        time_inds = np.arange(len(dataset))[::t_jump]
    else:
        time_inds = np.random.choice(len(dataset), dataset_size, replace=False)
    for i in tqdm(time_inds):
        atomic_numbers, mol_ind, num_molecules, ref_coords = reindex_mols(dataset, i, mol_num_atoms)

        density = len(atomic_numbers) / torch.prod(T_fc_list[i].diag())
        if density > 0.025:
            periodic = True
        else:
            periodic = False

        if pare_to_cluster:
            periodic = False

        # we cannot trust the default indexing, so manually reindex according to the recorded molecule index
        cluster_atoms, cluster_coords, cluster_targets = reindex_molecules(atomic_numbers, i, mol_ind, num_molecules, ref_coords, targets)

        # force any periodic images back into the box
        cluster_coords = force_molecules_into_box(T_fc_list, cluster_coords, i, periodic)

        # pare the cluster down to a manageable overall size
        if run_config is not None and pare_to_cluster:
            if max_cluster_radius is None:
                max_cluster_radius = run_config['max_sphere_radius'] + 2 * conv_cutoff  # minimal cluster plus buffer

            cluster_atoms, cluster_coords, cluster_targets = pare_cluster_radius(cluster_atoms, cluster_coords, cluster_targets, max_cluster_radius=max_cluster_radius)

        # pare off molecules which are fragmented
        cluster_atoms, cluster_coords, cluster_targets, good_mols, mol_radii = pare_fragmented_molecules(
            cluster_atoms, cluster_coords, cluster_targets, pare_fragmented=not periodic
        )

        # identify surface mols
        centroids, cluster_mol_ind, coordination_number, defect_type = identify_surface_molecules(
            cluster_coords, cluster_targets, conv_cutoff, good_mols, mol_num_atoms, mol_radii)
        if periodic:
            defect_type = torch.zeros_like(defect_type)  # no surfaces in periodic structures

        datapoints.append(
            CrystalData(
                x=cluster_atoms.reshape(len(good_mols) * mol_num_atoms),
                mol_ind=cluster_mol_ind.reshape(len(good_mols) * mol_num_atoms),
                pos=cluster_coords.reshape(len(good_mols) * mol_num_atoms, 3),
                y=cluster_targets,
                T_fc=T_fc_list[i],
                defect=defect_type,
                centroid_pos=(centroids - centroids.mean(0)).cpu().detach().numpy(),
                coord_number=coordination_number.cpu().detach().numpy(),
                tracking=np.asarray(dataset.loc[i, ['temperature', 'time_step']]),
                asym_unit_handedness=torch.ones(1),
                symmetry_operators=torch.ones(1),
                periodic=periodic,
            )
        )
    del dataset
    return get_dataloaders(datapoints, machine='local', batch_size=batch_size, test_fraction=test_fraction, shuffle=shuffle)


def collate_training_dataloaders(config, dataset_path, mode='cold'):

    if 'nic' in dataset_path.lower():
        melt_frac = 1 / 10
        box_transition_size = 31  # approximate cutoff from histograms between 20x20x20 and 40x40x40 boxes

    elif 'urea' in dataset_path.lower():
        melt_frac = 1 / 7
        box_transition_size = 24  # the distribution is extremely bimodal with a big split, this is a lower end

    bulk_frac = 0.5
    surface_frac = 0.5
    if mode == 'cold':
        # small-scale periodic samples
        _, train_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                      dataset_path, int(config['dataset_size'] * bulk_frac),
                                                      conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                      temperatures=[config['training_temps'][0]], test_fraction=1,
                                                      no_melt=True, periodic_only=True, max_box_volume=box_transition_size ** 3)
        _, test_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                     dataset_path, int(config['dataset_size'] * 0.2 * bulk_frac),
                                                     conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                     temperatures=[config['training_temps'][1]], test_fraction=1,
                                                     no_melt=True, periodic_only=True, max_box_volume=box_transition_size ** 3)
        _, hot_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                    dataset_path, int(config['dataset_size'] * melt_frac * bulk_frac),
                                                    conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                    filter_early=False,
                                                    temperatures=[config['training_temps'][-1]], test_fraction=1,
                                                    melt_only=True, periodic_only=True, max_box_volume=box_transition_size ** 3)

        # carve spheres out of larger bulk samples
        _, s_train_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                        dataset_path, int(config['dataset_size'] * surface_frac),
                                                        conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                        temperatures=[config['training_temps'][0]], test_fraction=1,
                                                        no_melt=True, periodic_only=True, pare_to_cluster=True,
                                                        min_box_volume=box_transition_size ** 3,
                                                        max_cluster_radius=15)
        _, s_test_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                       dataset_path, int(config['dataset_size'] * 0.2 * surface_frac),
                                                       conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                       temperatures=[config['training_temps'][1]], test_fraction=1,
                                                       no_melt=True, periodic_only=True, pare_to_cluster=True,
                                                       min_box_volume=box_transition_size ** 3,
                                                       max_cluster_radius=15)
        _, s_hot_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                      dataset_path, int(config['dataset_size'] * melt_frac * surface_frac),
                                                      conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                      temperatures=[config['training_temps'][-1]], test_fraction=1,
                                                      melt_only=True, periodic_only=True, pare_to_cluster=True,
                                                      filter_early=True,
                                                      min_box_volume=box_transition_size ** 3,
                                                      max_cluster_radius=15)

        # split the hot trajs equally
        hot_length = len(hot_loader)
        train_loader.dataset.extend(hot_loader.dataset[:hot_length // 2])
        test_loader.dataset.extend(hot_loader.dataset[hot_length // 2:])

        # append the surface trajs
        train_loader.dataset.extend(s_train_loader.dataset)
        test_loader.dataset.extend(s_test_loader.dataset)
        hot_length = len(s_hot_loader)
        train_loader.dataset.extend(s_hot_loader.dataset[:hot_length // 2])
        test_loader.dataset.extend(s_hot_loader.dataset[hot_length // 2:])

        del hot_loader, s_test_loader, s_train_loader

    elif mode == 'hot':
        # small-scale periodic samples
        train_loader, test_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                                dataset_path, int(config['dataset_size'] * bulk_frac),
                                                                conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                                temperatures=config['training_temps'], test_fraction=.2,
                                                                no_melt=True, periodic_only=True, max_box_volume=box_transition_size ** 3)
        _, hot_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                    dataset_path, int(config['dataset_size'] * melt_frac * bulk_frac),
                                                    conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                    filter_early=False,
                                                    temperatures=[config['training_temps'][-1]], test_fraction=1,
                                                    melt_only=True, periodic_only=True, max_box_volume=box_transition_size ** 3)

        # carve spheres out of larger bulk samples
        s_train_loader, s_test_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                                    dataset_path, int(config['dataset_size'] * surface_frac),
                                                                    conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                                    temperatures=config['training_temps'], test_fraction=.2,
                                                                    no_melt=True, periodic_only=True, pare_to_cluster=True,
                                                                    min_box_volume=box_transition_size ** 3,
                                                                    max_cluster_radius=15)
        _, s_hot_loader = collect_to_traj_dataloaders(config['mol_num_atoms'],
                                                      dataset_path, int(config['dataset_size'] * melt_frac * surface_frac),
                                                      conv_cutoff=config['conv_cutoff'], batch_size=1,
                                                      temperatures=[config['training_temps'][-1]], test_fraction=1,
                                                      melt_only=True, periodic_only=True, pare_to_cluster=True,
                                                      filter_early=True,
                                                      min_box_volume=box_transition_size ** 3,
                                                      max_cluster_radius=15)

        # split the hot trajs equally
        hot_length = len(hot_loader)
        train_loader.dataset.extend(hot_loader.dataset[:hot_length // 2])
        test_loader.dataset.extend(hot_loader.dataset[hot_length // 2:])

        # append the surface trajs
        train_loader.dataset.extend(s_train_loader.dataset)
        test_loader.dataset.extend(s_test_loader.dataset)
        hot_length = len(s_hot_loader)
        train_loader.dataset.extend(s_hot_loader.dataset[:hot_length // 2])
        test_loader.dataset.extend(s_hot_loader.dataset[hot_length // 2:])

        del hot_loader, s_test_loader, s_train_loader

    traint = torch.Tensor([thing.y[0] for thing in train_loader.dataset])
    testt = torch.Tensor([thing.y[0] for thing in test_loader.dataset])
    print(torch.unique(traint, return_counts=True))
    print(torch.unique(testt, return_counts=True))

    return train_loader, test_loader
