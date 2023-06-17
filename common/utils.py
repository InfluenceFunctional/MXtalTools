"""import statement"""
import numpy as np

import torch
from ase import Atoms
from scipy.cluster.hierarchy import dendrogram
import pandas as pd

'''
general utilities
'''

def plot_dendrogram(model, **kwargs):
    """
    make a dendrogram plot from a scipy clustering model
    """
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def initialize_dict_of_lists(keys: list):
    return {key: [] for key in keys}


def chunkify(lst: list, n: int):
    """
    break up a list into n chunks of equal size (up to last chunk)
    """
    return [lst[i::n] for i in range(n)]


def delete_from_dataframe(df: pd.DataFrame, inds):
    """
    hacky way to delete rows "inds" from datafram df
    """  # todo we're doing something wrong here
    df = df.drop(index=inds)
    if 'level_0' in df.columns:  # delete unwanted samples
        df = df.drop(columns='level_0')
    df = df.reset_index()

    return df


def torch_ptp(tensor: torch.tensor):
    return torch.max(tensor) - torch.min(tensor)


def standardize(data: np.ndarray, return_standardization: bool = False, known_mean: float = None, known_std: float = None):
    """
    standardize an input 1D array by subtracting mean and dividing by standard deviation
    optionally use precomputed mean and standard deviation (useful to compare data between datasets)
    optionally return standardization parameters
    """
    data = data.astype('float32')
    if known_mean is not None:
        mean = known_mean
    else:
        mean = np.mean(data)

    if known_std is not None:
        std = known_std
    else:
        std = np.std(data)

    if std == 0:
        std = 0.01  # if std == 0, data-mean will be all zeros. Doesn't matter what this value is.

    std_data = (data - mean) / std

    if return_standardization:
        return std_data, mean, std
    else:
        return std_data


def normalize(x: np.ndarray):
    """
    normalize an input by its span
    subtract min_x so that range is fixed on [0,1]
    """
    normed_x = (x - np.amin(x)) / np.ptp(x)
    return normed_x


def np_softmax(x: np.ndarray, temperature: float = 1):
    """
    softmax function implemented in numpy
    """
    if x.ndim == 1:
        x = x[None, :]
    x = x.astype(float)
    probabilities = np.exp(x / temperature) / np.sum(np.exp(x / temperature), axis=1)[:, None]
    return probabilities


def compute_rdf_distance(rdf1, rdf2, rr):
    """
    compute a distance metric between two radial distribution functions with shapes
    [num_sub_rdfs, num_bins] where sub_rdfs are e.g., particular interatomic RDFS within a certain sample
    rr is the bin edges used for both rdfs

    option for input to be torch tensors or numpy arrays, but has to be the same either way
    computation is done in torch
    range rr can be independently either np.array or torch.tensor
    will return same format as given
    """
    return_numpy = False
    if not torch.is_tensor(rdf1):
        torch_rdf1 = torch.Tensor(rdf1)
        torch_rdf2 = torch.Tensor(rdf2)
        return_numpy = True
    else:
        torch_rdf1 = rdf1
        torch_rdf2 = rdf2

    if not torch.is_tensor(rr):
        torch_range = torch.Tensor(rr)
    else:
        torch_range = rr

    # TODO come up with a clever way to norm these
    # emd_norm = (torch_rdf1.sum(-1) + torch_rdf2.sum(-1)) / 2  # sub-rdf-wise symmetrical norm sub-rdf-wise

    normed_rdf1 = torch_rdf1  # / emd_norm[:, None]
    normed_rdf2 = torch_rdf2  # / emd_norm[:, None]

    emd = earth_movers_distance_torch(normed_rdf1, normed_rdf2)

    range_normed_emd = emd * (torch_range[1] - torch_range[0])  # rescale the distance from units of bins to the real physical range

    if return_numpy:
        distance = range_normed_emd.mean().cpu().detach().numpy()
    else:
        distance = range_normed_emd.mean()

    assert np.sum(np.isnan(distance)) == 0

    return distance


def earth_movers_distance_torch(x: torch.tensor, y: torch.tensor):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    """
    return torch.sum(torch.abs(torch.cumsum(x, dim=-1) - torch.cumsum(y, dim=-1)), dim=-1)


def earth_movers_distance_np(d1: np.ndarray, d2: np.ndarray):
    """
    earth mover's distance between two PDFs
    not normalized or aggregated
    """
    return np.sum(np.abs(np.cumsum(d1, axis=-1) - np.cumsum(d2, axis=-1)), axis=-1)


def histogram_overlap_np(d1: np.ndarray, d2: np.ndarray):
    """
    compute the symmetric overlap of two histograms
    """
    return np.sum(np.minimum(d1, d2)) / np.average((d1.sum(), d2.sum()))


def update_stats_dict(dictionary: dict, keys, values, mode='append'):
    """
    update dict of running statistics in batches of key:list pairs or one at a time
    """
    if isinstance(keys, list):
        for key, value in zip(keys, values):
            # if isinstance(value, list):
            #     value = np.stack(value)

            if key not in dictionary.keys():
                dictionary[key] = []

            if mode == 'append':
                dictionary[key].append(value)
            elif mode == 'extend':
                dictionary[key].extend(value)
    else:
        key, value = keys, values
        if key not in dictionary.keys():
            dictionary[key] = []
        #
        # if isinstance(value, list):
        #     value = np.stack(value)

        if mode == 'append':
            dictionary[key].append(value)
        elif mode == 'extend':
            dictionary[key].extend(value)

    return dictionary


def compute_rdf_distance_old(target_rdf: np.ndarray, sample_rdf: np.ndarray):
    '''
    earth mover's distance
    assuming dimension [sample, element-pair, radius]
    normed against target rdf (sample is not strictly a PDF in this case)
    averaged over nnz elements - only works for single type of molecule per call
    OLD way of doing this
    '''

    nonzero_element_pairs = np.sum(np.sum(target_rdf, axis=1) > 0)
    target_CDF = np.cumsum(target_rdf, axis=-1)
    sample_CDF = np.cumsum(sample_rdf, axis=-1)
    norm = target_CDF[:, -1]
    target_CDF = np.nan_to_num(target_CDF / norm[:, None])
    sample_CDF = np.nan_to_num(sample_CDF / norm[None, :, None])
    emd = np.sum(np.abs(target_CDF - sample_CDF), axis=(1, 2))
    return emd / nonzero_element_pairs  # manual normalization elementwise


def crystals_to_ase_mols(crystaldata, max_ind=np.inf, highlight_aux=False, exclusion_level='distance', inclusion_distance=4):
    return [ase_mol_from_crystaldata(crystaldata, ii, highlight_canonical_conformer=highlight_aux, exclusion_level=exclusion_level, inclusion_distance=inclusion_distance)
            for ii in range(min(max_ind, crystaldata.num_graphs))]


def ase_mol_from_crystaldata(data, index=None, highlight_canonical_conformer=False, exclusion_level=None, inclusion_distance=4):
    """
    generate an ASE Atoms object from a crystaldata object, up to certain exclusions
    optionally highlight atoms in the asymmetric unit

    """
    data = data.clone().cpu().detach()
    if data.batch is not None:  # more than one crystal in the datafile
        atom_inds = torch.where(data.batch == index)[0]
    else:
        atom_inds = torch.arange(len(data.x))

    if exclusion_level == 'conformer':  # only the canonical conformer itself
        inside_inds = torch.where(data.aux_ind == 0)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'unit cell':
        # assume that by construction the first Z molecules are the ones in the unit cell
        mol_size = data.mol_size[index]
        num_molecules = int((data.ptr[index + 1] - data.ptr[index]) / mol_size)

        molecule_centroids = torch.stack([torch.mean(data.pos[data.ptr[index] + int(mol_size * multiplier):data.ptr[index] + int(mol_size * multiplier + 1)], dim=0)
                                          for multiplier in range(num_molecules)])

        fractional_centroids = torch.inner(torch.linalg.inv(data.T_fc[index]), molecule_centroids).T

        inside_centroids = torch.prod((fractional_centroids < 1) * (fractional_centroids > 0), dim=-1)
        #assert inside_centroids.sum() == data.Z[index]  # must be exactly Z molecules in the unit cell
        inside_centroids_inds = torch.where(inside_centroids)[0]

        inside_inds = torch.cat(
            [torch.arange(mol_size) + mol_size * inside_centroids_inds[ind]
             for ind in range(len(inside_centroids_inds))]
        ).long()
        inside_inds += data.ptr[index]
        atom_inds = inside_inds
        coords = data.pos[inside_inds].cpu().detach().numpy()


    elif exclusion_level == 'inside cell':
        fractional_coords = torch.inner(torch.linalg.inv(data.T_fc[index]), data.pos[data.batch == index]).T
        inside_coords = torch.prod((fractional_coords < 1) * (fractional_coords > 0), dim=-1)
        inside_inds = torch.where(inside_coords)[0]
        inside_inds += data.ptr[index]
        atom_inds = inside_inds
        coords = data.pos[inside_inds].cpu().detach().numpy()

    elif exclusion_level == 'convolve with':  # atoms potentially in the convolutional field
        inside_inds = torch.where(data.aux_ind < 2)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'distance':  # atoms within a certain distance of the conformer radius
        crystal_coords = data.pos[atom_inds]
        crystal_inds = data.aux_ind[atom_inds]

        canonical_conformer_inds = torch.where(crystal_inds == 0)[0]
        mol_centroid = crystal_coords[canonical_conformer_inds].mean(0)
        mol_radius = torch.max(torch.cdist(mol_centroid[None], crystal_coords[canonical_conformer_inds], p=2))
        in_range_inds = torch.where((torch.cdist(mol_centroid[None], crystal_coords, p=2) < (mol_radius + inclusion_distance))[0])[0]
        atom_inds = atom_inds[in_range_inds]
        coords = crystal_coords[in_range_inds].cpu().detach().numpy()
    else:
        coords = data.pos[atom_inds].cpu().detach().numpy()

    if highlight_canonical_conformer:  # highlight the atom aux index
        numbers = data.aux_ind[atom_inds].cpu().detach().numpy() + 6
    else:
        numbers = data.x[atom_inds, 0].cpu().detach().numpy()

    if index is not None:
        cell = data.T_fc[index].T.cpu().detach().numpy()
    else:
        cell = data.T_fc[0].T.cpu().detach().numpy()

    mol = Atoms(symbols=numbers, positions=coords, cell=cell)

    return mol
