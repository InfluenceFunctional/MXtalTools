import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bulk_molecule_classification import filter_mols, convert_box_to_cell_params, reindex_mols, reindex_molecules, force_molecules_into_box, pare_cluster_radius, \
    identify_surface_molecules, pare_fragmented_molecules
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.dataset_management.dataloader_utils import get_dataloaders


def collect_to_traj_dataloaders(mol_num_atoms, dataset_path, dataset_size, batch_size,
                                conv_cutoff, test_fraction=0.2, filter_early=True,
                                temperatures: list = None, shuffle=True,
                                melt_only=False, no_melt=False, early_only=False,
                                run_config=None, pare_to_cluster=False,
                                periodic_only=False, aperiodic_only=False,
                                max_cluster_radius=None, max_box_volume=None,
                                min_box_volume=None):
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
    t_jump = max(len(dataset) // dataset_size, 1)
    time_inds = np.arange(len(dataset))[::t_jump]
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
        cluster_coords = force_molecules_into_box(T_fc_list, cluster_coords, i)

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
