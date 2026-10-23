"""Helpers for molecular-cluster classification: LAMMPS box conversion to cell vectors, molecule-wise regrouping and filtering of cluster coordinates, checkpoint reloading, the classifier loss, and result recording and metric logging."""
import numpy as np
import torch
import torch.nn.functional as F

from mxtaltools.common.utils import softmax_np
from mxtaltools.common.geometry_utils import coor_trans_matrix_np
from mxtaltools.constants.classifier_constants import defect_names

from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import plotly.graph_objects as go

import pandas as pd

"""
utility functions for the molecular crystal local environment classification module
"""


def delete_pandas_dataframe_rows(df: pd.DataFrame, inds):
    """
    Drop rows from a DataFrame in place and reset its index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify; it is mutated in place.
    inds
        Index labels of the rows to drop, passed to ``df.drop(index=...)``.

    Returns
    -------
    pd.DataFrame
        The same ``df`` object, with the rows removed and the old index
        replaced by a fresh range index.
    """
    df.drop(index=inds, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


#

def convert_box_to_cell_params(box_params):
    """
    Convert LAMMPS box bounds to fractional-to-Cartesian matrices, with a consistency check.

    Calls :func:`convert_box_to_cell_vectors`, rebuilds each matrix from the
    resulting cell lengths and angles with ``coor_trans_matrix_np('f_to_c', ...)``,
    and asserts that the two sets of matrices agree.

    Parameters
    ----------
    box_params : sequence of np.ndarray
        Per-sample LAMMPS box bounds, each [3, 2] or [3, 3]; see
        :func:`convert_box_to_cell_vectors`.

    Returns
    -------
    T_fc_list : np.ndarray
        [num_samples, 3, 3] matrices whose columns are the cell vectors a, b, c,
        as returned by :func:`convert_box_to_cell_vectors` (not the rebuilt
        matrices).

    Raises
    ------
    AssertionError
        If the sum of absolute differences between the two sets of matrices,
        over all samples and elements, is not below 1e-3.
    """
    T_fc_list, angles, lengths = convert_box_to_cell_vectors(box_params)

    T_fc_list2 = []  # double check the answer
    for i in range(len(lengths)):
        T_fc_list2.append(coor_trans_matrix_np('f_to_c', lengths[i], angles[i]))
    T_fc_list2 = np.stack(T_fc_list2)

    assert np.sum(np.abs(T_fc_list - T_fc_list2)) < 1e-3

    return T_fc_list


def convert_box_to_cell_vectors(box_params):
    """
    Convert LAMMPS periodic box bounds to cell vectors, angles and lengths.

    LAMMPS periodic box style::

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

    Parameters
    ----------
    box_params : sequence of np.ndarray
        Per-sample box bounds with rows x, y, z and columns lo_bound, hi_bound
        and, if present, the tilt factor (xy, xz, yz in row order). Each entry
        is [3, 2] (tilts taken as zero) or [3, 3]. If the entries cannot be
        combined with ``np.stack``, ``box_params.tolist()`` is called (so the
        container must provide ``tolist``, e.g. a pandas Series or object
        array) and every [3, 2] entry is padded with a zero tilt column.

    Returns
    -------
    T_fc_list : np.ndarray
        [num_samples, 3, 3] float64 matrices whose columns are a, b, c.
    angles : np.ndarray
        [num_samples, 3] cell angles (alpha, beta, gamma) in radians.
    lengths : np.ndarray
        [num_samples, 3] cell lengths (a, b, c), in the units of the box
        bounds.
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
    return T_fc_list, angles, lengths


def reindex_mols(dataset, i, mol_num_atoms):
    """
    Read one sample's atoms, coordinates and molecule assignment from a DataFrame row.

    Parameters
    ----------
    dataset : pd.DataFrame
        Table with ``'coordinates'``, ``'atom_type'`` and ``'mol_ind'``
        columns; element ``[0]`` of each cell is used.
    i
        Row label passed to ``dataset.loc``.
    mol_num_atoms : int
        Atoms per molecule; ``num_molecules`` is
        ``len(coordinates) // mol_num_atoms``.

    Returns
    -------
    atomic_numbers : torch.Tensor
        [num_atoms] long atom types.
    mol_ind : torch.Tensor
        [num_atoms] long molecule index of each atom.
    num_molecules : int
        Number of atoms divided (integer division) by ``mol_num_atoms``.
    ref_coords : torch.Tensor
        Atom coordinates, one row per atom, in torch's default float dtype.

    Raises
    ------
    AssertionError
        If ``num_molecules`` differs from the number of distinct values in
        ``mol_ind``.
    """
    ref_coords = torch.Tensor(dataset.loc[i]['coordinates'][0])
    atoms = dataset.loc[i]['atom_type'][0]
    atomic_numbers = torch.tensor(atoms, dtype=torch.long)
    num_molecules = (len(ref_coords)) // mol_num_atoms  # todo this must be adaptive e.g., inclusive of defects
    mol_ind = torch.tensor(dataset.loc[i]['mol_ind'][0], dtype=torch.long)
    assert num_molecules == len(torch.unique(mol_ind))
    return atomic_numbers, mol_ind, num_molecules, ref_coords

#
# def filter_mols(dataset, dataset_path, early_only, filter_early, melt_only, no_melt, temperatures,
#                 periodic_only, aperiodic_only, max_box_volume, min_box_volume):
#     if temperatures is not None:
#         good_inds = []
#         for temperature in temperatures:
#             good_inds.append(np.argwhere(np.asarray(dataset['temperature']) == temperature)[:, 0])
#
#         good_inds = np.unique(np.concatenate(good_inds))
#         bad_inds = np.asarray([ind for ind in np.arange(len(dataset)) if ind not in good_inds])
#         print(f"Temperature filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#     if filter_early:
#         bad_inds = np.argwhere(np.asarray(dataset['time_step']) <= int(1e4))[:,
#                    0]  # filter first 10ps steps for equilibration
#         print(f"Early filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#     if early_only:
#         bad_inds = np.argwhere(np.asarray(dataset['time_step']) >= int(1e6))[:, 0]  # keep only 1 ns maximum
#         print(f"Early only filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#
#     if max_box_volume is not None:
#         T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))
#         approx_box_volume = (T_fc_list[:, 0, 0] * T_fc_list[:, 1, 1] * T_fc_list[:, 2, 2])
#         bad_inds = np.argwhere(approx_box_volume > max_box_volume)[0, :]
#         print(f"Max box filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#
#     if min_box_volume is not None:
#         T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))
#         approx_box_volume = (T_fc_list[:, 0, 0] * T_fc_list[:, 1, 1] * T_fc_list[:, 2, 2])
#         bad_inds = np.argwhere(approx_box_volume < min_box_volume)[0, :]
#         print(f"Min box filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#
#     if periodic_only or aperiodic_only:
#         num_atoms = np.asarray([len(thing[0]) for thing in dataset['atom_type']])
#         T_fc_list = torch.Tensor(convert_box_to_cell_params(dataset['cell_params']))
#         density = num_atoms / (T_fc_list[:, 0, 0] * T_fc_list[:, 1, 1] * T_fc_list[:, 2, 2])
#
#     if periodic_only:
#         bad_inds = np.argwhere(density <= 0.025).flatten()
#         print(f"Periodic only filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#     if aperiodic_only:
#         bad_inds = np.argwhere(density > 0.025).flatten()
#         print(f"Aperiodic only filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#     if True:
#         if 'gap_rate' in dataset.columns:
#             bad_inds = np.argwhere(np.asarray(dataset['gap_rate']) > 0)[:, 0]  # cannot process gaps right now
#             print(f"No Gaps filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#             dataset = delete_from_dataframe(dataset, bad_inds)
#             dataset = dataset.reset_index().drop(columns='index')
#     # forms = np.sort(np.unique(dataset['form']))
#     # forms2tgt = {form: i for i, form in enumerate(forms)}
#     targets = np.asarray(
#         dataset['form']) - 1  # no longer need to reindex, as we have this now managed through the constants
#     # this will throw an error later if the combined dataset is missing any forms, but it shouldn't be missing any forms in general
#     # so that's fine
#     # set high temperature samples to 'melted' class
#     if 'urea' in dataset_path:
#         melt_class_num = 6
#     else:
#         melt_class_num = 9  # nicotinamide
#     if melt_only:
#         bad_inds = np.argwhere(targets != melt_class_num)[:, 0]
#         print(f"Melt only filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         good_inds = np.argwhere(targets == melt_class_num)[:, 0]
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#         targets = targets[good_inds]
#     if no_melt:
#         bad_inds = np.argwhere(targets == melt_class_num)[:, 0]
#         print(f"No Melt filter killed {len(bad_inds)} out of {len(dataset)} samples")
#
#         good_inds = np.argwhere(targets != melt_class_num)[:, 0]
#
#         dataset = delete_from_dataframe(dataset, bad_inds)
#         dataset = dataset.reset_index().drop(columns='index')
#         targets = targets[good_inds]
#     return dataset, targets


def identify_surface_molecules(cluster_coords, cluster_targets, conv_cutoff, good_mols, mol_num_atoms, mol_radii):
    """
    Label molecules with few neighbouring centroids as surface molecules.

    A molecule's coordination number is the number of molecule centroids,
    itself included, strictly closer than
    ``max(mol_radii[good_mols]) + conv_cutoff`` to its centroid. Molecules with
    a coordination number below 20 get defect type 1 (surface); all others get
    0 (bulk).

    Parameters
    ----------
    cluster_coords : torch.Tensor
        [num_molecules, mol_num_atoms, 3] molecule-wise coordinates.
    cluster_targets : torch.Tensor
        [num_molecules] per-molecule targets; used only as the shape and dtype
        template for ``defect_type``.
    conv_cutoff : float
        Added to the largest retained molecular radius to form the
        coordination cutoff, in the units of ``cluster_coords``.
    good_mols : torch.Tensor
        Indices into ``mol_radii`` of the retained molecules; its length also
        sets the first dimension of ``cluster_mol_ind``.
    mol_num_atoms : int
        Atoms per molecule.
    mol_radii : torch.Tensor
        Per-molecule radii, indexed by ``good_mols``.

    Returns
    -------
    centroids : torch.Tensor
        [num_molecules, 3] mean coordinate of each molecule.
    cluster_mol_ind : torch.Tensor
        [len(good_mols), mol_num_atoms] long; row ``k`` is filled with ``k``.
    coordination_number : torch.Tensor
        [num_molecules] number of centroids within the cutoff, including the
        molecule itself.
    defect_type : torch.Tensor
        Same shape and dtype as ``cluster_targets``; 1 for surface molecules,
        0 otherwise.
    """
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
    """
    Identify molecules which are fragmented, or split across a periodic boundary, and optionally delete them.

    Fragmented molecules are identified as having larger molecular radii than
    normal: a molecule's radius is the largest distance of any of its atoms
    from its centroid, and when ``pare_fragmented`` is True, molecules whose
    radius is not below 1.25 times the 5th-percentile radius are dropped. When
    ``pare_fragmented`` is False, all molecules are kept.

    Parameters
    ----------
    cluster_atoms : torch.Tensor
        [num_molecules, mol_num_atoms] per-molecule atom types.
    cluster_coords : torch.Tensor
        [num_molecules, mol_num_atoms, 3] molecule-wise coordinates.
    cluster_targets : torch.Tensor
        [num_molecules] per-molecule targets.
    pare_fragmented : bool
        Whether to apply the radius filter.

    Returns
    -------
    cluster_atoms : torch.Tensor
        ``cluster_atoms`` restricted to the kept molecules.
    cluster_coords : torch.Tensor
        ``cluster_coords`` restricted to the kept molecules.
    cluster_targets : torch.Tensor
        ``cluster_targets`` restricted to the kept molecules.
    good_mols : torch.Tensor
        Indices of the kept molecules.
    mol_radii : torch.Tensor
        [num_molecules] radii of all input molecules, before filtering.
    """
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
    """
    Compute per-molecule radii and the indices kept by the fragment filter.

    Uses the same radius and threshold as :func:`pare_fragmented_molecules`,
    without indexing any inputs.

    Parameters
    ----------
    cluster_coords : torch.Tensor
        [num_molecules, mol_num_atoms, 3] molecule-wise coordinates.
    pare_fragmented : bool
        If True, keep only molecules whose radius is below 1.25 times the
        5th-percentile radius; otherwise keep all.

    Returns
    -------
    good_mols : torch.Tensor
        Indices of the kept molecules.
    mol_radii : torch.Tensor
        [num_molecules] largest atom-to-centroid distance of each molecule.
    """
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
    """
    Group one sample's atoms into molecule-wise tensors.

    Molecules are ordered by ascending ``mol_ind`` value. All molecules must
    have the same number of atoms, since the per-molecule slices are combined
    with ``torch.stack``.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        [num_atoms] atom types.
    i : int
        Sample index into ``targets``.
    mol_ind : torch.Tensor
        [num_atoms] molecule index of each atom.
    num_molecules : int
        Length of ``cluster_targets``; not checked against the number of
        distinct ``mol_ind`` values.
    ref_coords : torch.Tensor
        Atom coordinates, one row per atom.
    targets : array-like
        Per-sample targets; ``targets[i].repeat(num_molecules)`` gives the
        per-molecule targets.

    Returns
    -------
    cluster_atoms : torch.Tensor
        [num_unique_mol_ind, mol_num_atoms] atom types.
    cluster_coords : torch.Tensor
        [num_unique_mol_ind, mol_num_atoms, 3] coordinates.
    cluster_targets : torch.Tensor
        [num_molecules] long, every entry equal to ``targets[i]``.
    """
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
    Translate each molecule by a lattice vector so its centroid lies in the unit cell.

    If ``periodic`` is True, all coordinates are first shifted so that the
    minimum over all atoms along each Cartesian axis is zero. Each molecule's
    centroid is then converted to fractional coordinates with
    ``T_fc_list[i]``, and the whole molecule is moved by the lattice
    translation ``-floor(frac_centroid)``, which puts its fractional centroid
    in [0, 1). Will fail on fragmented molecules or molecules otherwise
    wrapped: such molecules are moved rigidly like any other, so they are not
    reassembled, and no error is raised.

    Parameters
    ----------
    T_fc_list : torch.Tensor
        [num_samples, 3, 3] fractional-to-Cartesian matrices with the cell
        vectors as columns.
    cluster_coords : torch.Tensor
        [num_molecules, mol_num_atoms, 3] Cartesian coordinates; modified in
        place.
    i : int
        Index of the matrix to use in ``T_fc_list``.
    periodic : bool
        Whether to apply the initial shift of the coordinate minimum to zero.

    Returns
    -------
    torch.Tensor
        ``cluster_coords``, the same tensor, after the in-place translations.
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
    """
    Keep molecules whose centroid lies within ``max_cluster_radius`` of the cluster centre.

    The cluster centre is the mean over all atoms of ``cluster_coords``.

    Parameters
    ----------
    cluster_atoms : torch.Tensor
        [num_molecules, mol_num_atoms] atom types.
    cluster_coords : torch.Tensor
        [num_molecules, mol_num_atoms, 3] coordinates.
    cluster_targets : torch.Tensor
        [num_molecules] per-molecule targets.
    max_cluster_radius : float
        Strict upper bound on the centroid-to-centre distance, in the units of
        ``cluster_coords``.

    Returns
    -------
    cluster_atoms : torch.Tensor
        ``cluster_atoms`` restricted to the kept molecules.
    cluster_coords : torch.Tensor
        ``cluster_coords`` restricted to the kept molecules.
    cluster_targets : torch.Tensor
        ``cluster_targets`` restricted to the kept molecules.
    """
    mol_centroid_dists = torch.linalg.norm(cluster_coords.mean(1) - cluster_coords.mean((0, 1)), dim=1)
    good_mols = torch.argwhere(mol_centroid_dists < max_cluster_radius)[:, 0]  # 60 angstrom sphere at maximum
    cluster_coords = cluster_coords[good_mols]
    cluster_atoms = cluster_atoms[good_mols]
    cluster_targets = cluster_targets[good_mols]
    return cluster_atoms, cluster_coords, cluster_targets


def classifier_reporting(true_labels, true_defects, probs, class_names, ordered_class_names, wandb, epoch_type):
    """
    Log form and defect classification metrics and confusion matrices to wandb.

    Form scores are ``softmax_np`` over the columns of ``probs`` whose indices
    are the labels present in ``true_labels`` (sorted); the predicted form is
    the argmax over those columns. Defect scores are ``softmax_np`` over the
    columns of ``probs`` from ``len(present_classes)`` onward. Logs
    ``"<epoch_type> ROC_AUC"`` (one-vs-one), ``"<epoch_type> F1 Score"``
    (micro-averaged), ``"<epoch_type> 1-ROC_AUC"``, ``"<epoch_type> 1-F1 Score"``
    and ``"<epoch_type> Confusion Matrix"`` (plotly heatmap of the confusion
    matrix normalized over true labels). If more than one defect label is
    present, also logs the same five keys with ``Defect`` inserted
    (``"<epoch_type> Defect ROC_AUC"``, ...), with the defect ROC AUC computed
    from the column-1 defect score.

    Parameters
    ----------
    true_labels : np.ndarray
        [num_samples] integer form labels.
    true_defects : np.ndarray
        [num_samples] integer defect labels indexing ``defect_names``
        (``'Bulk'``, ``'Surface'``).
    probs : np.ndarray
        [num_samples, num_outputs] scores passed through ``softmax_np``; form
        columns first, then defect columns.
    class_names
        Unused.
    ordered_class_names : sequence of str
        Form names indexed by label; used for the heatmap axes.
    wandb
        Object whose ``log`` method receives the metrics (e.g. the ``wandb``
        module).
    epoch_type : str
        Prefix of the logged keys.

    Notes
    -----
    The predicted form is a column position within the selected form columns
    and is compared directly with ``true_labels``; the two coincide with label
    values only when the present labels are ``0 .. k-1``. The defect columns
    start at the number of form labels present in ``true_labels``, not at a
    fixed number of form outputs, so they include form columns whenever fewer
    form labels are present than ``probs`` has form columns.
    """
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
    Load model and optimizer state dicts from a checkpoint file.

    Includes a fix for the ``nn.DataParallel`` key prefix: the checkpoint is
    read with ``torch.load(path, map_location=device, weights_only=False)``,
    and if the first key of ``checkpoint['model_state_dict']`` starts with
    ``'module'``, the first 7 characters are removed from every key before
    loading.

    Parameters
    ----------
    model : torch.nn.Module
        Receives ``checkpoint['model_state_dict']`` via ``load_state_dict``;
        modified in place.
    device
        ``map_location`` for ``torch.load``.
    optimizer : torch.optim.Optimizer or None
        Receives ``checkpoint['optimizer_state_dict']`` when not None and
        ``reload_optimizer`` is True.
    path : str or path-like
        Checkpoint file.
    reload_optimizer : bool, optional
        Whether to restore the optimizer state. Default False.

    Returns
    -------
    model : torch.nn.Module
        The same ``model`` object.
    optimizer : torch.optim.Optimizer or None
        The same ``optimizer`` object.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if list(checkpoint['model_state_dict'])[0][
       0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def record_step_results(results_dict, output, sample, data, latents, embeddings, step, config, index_offset=0):
    """
    Append one step's classifier outputs and sample data to a results dictionary.

    Tensor values are detached and converted to CPU numpy arrays before
    appending. ``'Loss'`` is :func:`get_loss`; ``'Type_Prediction'`` and
    ``'Defect_Prediction'`` are softmaxes over the first
    ``config['num_forms']`` and the remaining columns of ``output``;
    ``'Temperature'``, ``'Time_Step'`` and ``'Sample_Index'`` are constant
    arrays of length ``len(sample.y)`` holding ``data.tracking[0][0]``,
    ``data.tracking[0][1]`` and ``step + index_offset``;
    ``'Molecule_Centroids'`` and ``'Coordination_Numbers'`` are
    ``sample.centroid_pos[0]`` and ``sample.coord_number[0]`` without
    conversion.

    Parameters
    ----------
    results_dict : dict or None
        Dictionary of lists to append to, modified in place. If None, a new
        one is created with the keys ``'Temperature'``, ``'Time_Step'``,
        ``'Loss'``, ``'Type_Prediction'``, ``'Defect_Prediction'``,
        ``'Targets'``, ``'Defects'``, ``'Latents'``, ``'Sample_Index'``,
        ``'Coordinates'``, ``'Atom_Types'``, ``'Molecule_Index'``,
        ``'Molecule_Centroids'``, ``'Coordination_Numbers'`` and
        ``'Embeddings'``.
    output : torch.Tensor
        [len(sample.y), num_outputs] classifier logits: form columns first,
        then defect columns.
    sample
        Data object with ``y`` (form labels), ``defect``, ``pos``, ``z``,
        ``mol_ind``, ``centroid_pos`` and ``coord_number``.
    data
        Object whose ``tracking[0][0]`` and ``tracking[0][1]`` are recorded as
        the temperature and time step.
    latents : torch.Tensor
        Recorded under ``'Latents'``.
    embeddings : torch.Tensor
        Recorded under ``'Embeddings'``.
    step : int
        Sample index recorded, plus ``index_offset``, under
        ``'Sample_Index'``.
    config : dict
        Must provide ``config['num_forms']``, the number of form columns in
        ``output``.
    index_offset : int, optional
        Added to ``step``. Default 0.

    Returns
    -------
    dict
        ``results_dict`` with one entry appended to each list.
    """
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
    results_dict['Atom_Types'].append(sample.z.cpu().detach().numpy())
    results_dict['Molecule_Index'].append(sample.mol_ind.cpu().detach().numpy())
    results_dict['Molecule_Centroids'].append(sample.centroid_pos[0])
    results_dict['Coordination_Numbers'].append(sample.coord_number[0])

    return results_dict


def process_trajectory_results_dict(results_dict, loader, mol_num_atoms):
    """
    Split flat results into per-sample lists sorted by time step.

    Each value in ``results_dict`` is classified by length. If it has
    ``len(results_dict['Atomwise_Sample_Index'])`` entries it is split by that
    index and kept under its own key; otherwise, if it has
    ``len(results_dict['Sample_Index'])`` entries, it is split by that index
    and stored as ``'Molecule_' + key``; any other value is omitted with a
    printed message. Splits are taken for sample indices
    ``0 .. len(loader) - 1``, and the samples are then sorted by the first
    entry of their ``'Molecule_Time_Step'``.

    Parameters
    ----------
    results_dict : dict
        Flat arrays already concatenated across steps, including
        ``'Atomwise_Sample_Index'``, ``'Sample_Index'``, a molecule-wise
        ``'Time_Step'`` and an atom-wise ``'Coordinates'`` of shape
        [num_atoms, 3]. Modified in place: every key without ``'Index'`` in its
        name is deleted.
    loader
        Only ``len(loader)`` is used, as the number of samples.
    mol_num_atoms : int
        Atoms per molecule, used to reshape each sample's coordinates into
        molecules.

    Returns
    -------
    sorted_molwise_results_dict : dict
        Per-sample lists in time order, plus ``'Centroid Radii'``: for each
        sample, the distance of every molecule centroid from the mean
        centroid.
    time_steps : np.ndarray
        [len(loader)] time steps in sorted order.
    """
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
    """
    Return the sum of the mean form and defect cross-entropy losses.

    Parameters
    ----------
    output : torch.Tensor
        [N, num_outputs] logits: the first ``num_forms`` columns are form
        logits, the rest defect logits.
    sample
        Object with ``y`` (form labels) and ``defect`` (defect labels), each
        [N] long.
    num_forms : int
        Number of form columns in ``output``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return F.cross_entropy(output[:, :num_forms], sample.y) + F.cross_entropy(output[:, num_forms:], sample.defect)
