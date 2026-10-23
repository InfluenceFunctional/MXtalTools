"""Radial distribution functions of crystal cluster batches and distances between them.

`crystal_rdf` histograms the intermolecular edge lengths of a crystal cluster batch into one or
more channels (all pairs, element pairs, within-molecule atom-index pairs, or bonded-environment
pairs) and divides by the volume of each radial shell. `compute_rdf_distance` compares RDFs by the
mean, over active channels, of the 1-D earth mover's distance between unit-normalised channels;
`compute_rdf_distmat`, `compute_rdf_distmat_parallel` and `rdf_radial_graph` build pairwise
matrices and neighbour lists from that distance.
"""

import itertools
import math
from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.utils import repeat_interleave, torch_ptp, is_cuda_oom


def crystal_rdf(crystal_batch,
                precomputed_distances_dict=None,
                rrange: Tuple[float, float] = (0, 10),
                bins: int = 100,
                mode: Optional[str] = None,
                atomic_numbers_override: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Radial density function over the intermolecular edges of a crystal cluster batch.

    Distances are the Euclidean lengths ``|pos[edges[0]] - pos[edges[1]]|`` of the edges in
    ``precomputed_distances_dict['edge_index_inter']``, each assigned to the graph of its first
    endpoint (``crystal_batch.batch[edges[0]]``). Only edges in that list contribute, so distances
    beyond the cutoff the radial graph was built with are absent whatever ``rrange`` is. Counts are
    divided by the volume of each spherical shell, ``4/3 * pi * (r_{k+1}**3 - r_k**3)`` for the bin
    edges ``r = linspace(rrange[0], rrange[1], bins + 1)``. No division by a number density is made,
    so the result is a radial density, not the dimensionless g(r). Finally every bin is multiplied by
    `smooth_cutoff` evaluated at its upper edge with cutoff ``rrange[-1]`` and onset
    ``rrange[-1] - 1``: bins whose upper edge is at or below the onset are unchanged and the last bin
    is zero.

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Batched crystal cluster; not modified. Every mode reads ``pos``, ``batch`` and
        ``num_graphs``. ``'elementwise'`` also reads ``z``; ``'atomwise'`` reads ``num_atoms[0]`` and
        the length of ``z``; ``'envwise'`` reads ``aux_ind``, ``z``, ``z_prime`` and ``mol_ind``
        through `get_1WL_env_labels`.
    precomputed_distances_dict : dict
        Edge dictionary of the batch's radial graph; required despite the None default.
        ``'edge_index_inter'`` [2, num_edges] is read in every mode; ``'envwise'`` also reads
        ``'edge_index'`` and ``'intramolecular_dist'``.
    rrange : tuple of float
        (r_min, r_max) of the histogram, in the length units of ``crystal_batch.pos``. Default
        (0, 10).
    bins : int
        Number of histogram bins. Default 100.
    mode : str, optional
        Channel partition. ``'elementwise'``: one channel per unordered pair of elements
        (`get_elementwise_dists`). ``'atomwise'``: one channel per unordered pair of within-molecule
        atom indices, the indices being ``arange(num_atoms[0])`` tiled over all atoms of the batch, so
        every molecule in the batch is assumed to have the same size and atom order (not checked).
        ``'envwise'``: one channel per unordered pair of bonded-environment labels from
        `get_1WL_env_labels`. These three modes use the Gaussian-smoothed `smooth_1d_histogram`. Any
        other value, including None, gives one all-pairs channel histogrammed by the hard-binned
        `batch_histogram_1d`.
    atomic_numbers_override : torch.LongTensor, optional
        Element list for ``'elementwise'`` mode; ignored in other modes.

    Returns
    -------
    smoothed_rdf : torch.Tensor
        [num_graphs, num_channels, bins] shell-volume-normalised histogram after the envelope;
        num_channels is 1 when ``mode`` is not one of the three channelled modes.
    bin_centers : torch.Tensor
        [bins] upper bin edges, ``r[1:]``, with ``requires_grad`` set.
    rdfs_dict : dict
        Map from channel index to channel name (for example ``'C_to_N'`` or ``'0_to_3'``); empty
        when ``mode`` is not one of the three channelled modes.
    """
    device = crystal_batch.pos.device
    num_graphs = crystal_batch.num_graphs
    edges = precomputed_distances_dict['edge_index_inter']
    bin_edges = torch.linspace(rrange[0], rrange[1], bins + 1, device=device)
    # track which edges go with which crystals
    edge_batch = crystal_batch.batch[edges[0]]

    # compute all the dists
    dists = torch.linalg.norm(crystal_batch.pos[edges[0]] - crystal_batch.pos[edges[1]], dim=1)
    envwise = mode == 'envwise'
    elementwise = mode == 'elementwise'
    atomwise = mode == 'atomwise'
    assert not (elementwise and atomwise) or (elementwise and envwise) or (atomwise and envwise)

    '''
    efficiently gather the relevant distances
    '''
    if elementwise:
        graph_indexed_pairs, rdfs_dict = get_elementwise_dists(crystal_batch.z,
                                                               edges,
                                                               device,
                                                               edge_batch,
                                                               atomic_numbers_override)

        num_pairs = len(rdfs_dict.keys())
        hist, rdf_density = smooth_1d_histogram(bins, device, dists, graph_indexed_pairs, num_graphs, num_pairs, rrange)
        # volume of the shell at radius r+dr
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[:-1] ** 3)
        rdf = hist / shell_volumes[None, None, :] / rdf_density[..., None]  # un-smoothed radial density

    elif atomwise:  # todo this is only implemented for an identical atom indexing (assumes batch is repetition of same molecule)
        # todo add flag that this won't work on mixed batches
        # the labels tile graph 0's WHOLE asymmetric unit (num_atoms[0]), not its units: at Z'>1 the channel meaning
        # changes (audit #24, scheduled by no stage); no unit is read here, so the unit map is not consulted (a row
        # whose units cannot be read is refused earlier, by the split)
        mol_size = crystal_batch.num_atoms[0].item()
        tot_num_repeats = len(crystal_batch.z) // mol_size
        atom_inds = torch.arange(mol_size, device=device).repeat(tot_num_repeats)
        graph_indexed_pairs, rdfs_dict = get_atomwise_dists(atom_inds,
                                                            edges,
                                                            edge_batch,
                                                            crystal_batch.num_atoms)

        num_pairs = len(rdfs_dict.keys())
        hist, rdf_density = smooth_1d_histogram(bins, device, dists, graph_indexed_pairs, num_graphs, num_pairs, rrange)
        # volume of the shell at radius r+dr
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[:-1] ** 3)
        rdf = hist / shell_volumes[None, None, :] / rdf_density[..., None]  # un-smoothed radial density

    elif envwise:  # distinguish 2WL intramolecular environments. Will combine symmetry classes.
        # todo add flag that this won't work on mixed batches
        labels = get_1WL_env_labels(crystal_batch, precomputed_distances_dict)
        graph_indexed_pairs, rdfs_dict = get_atomwise_dists(labels,
                                                            edges,
                                                            edge_batch,
                                                            crystal_batch.num_atoms)

        num_pairs = len(rdfs_dict.keys())
        hist, rdf_density = smooth_1d_histogram(bins, device, dists, graph_indexed_pairs, num_graphs, num_pairs, rrange)
        # volume of the shell at radius r+dr
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[:-1] ** 3)
        rdf = hist / shell_volumes[None, None, :] / rdf_density[..., None]  # un-smoothed radial density

    else:
        dists_per_hist = [torch.sum(edge_batch == n) for n in range(num_graphs)]
        sorted_dists = torch.cat([dists[edge_batch == n] for n in range(num_graphs)])
        rdfs_dict = {}
        batch = repeat_interleave(dists_per_hist, device='cpu').to(device)
        hist, bin_edges = batch_histogram_1d(sorted_dists, batch, num_graphs, rrange=rrange, nbins=bins)
        rdf_density = torch.ones(num_graphs, device=device, dtype=torch.float32)
        shell_volumes = (4 / 3) * torch.pi * ((bin_edges[:-1] + torch.diff(bin_edges)) ** 3 - bin_edges[
            :-1] ** 3)  # volume of the shell at radius r+dr
        rdf = hist / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density
        rdf = rdf[:, None, :]  # rdf format is [n, channels, bins]

    bin_centers = (bin_edges[:-1] + torch.diff(bin_edges)).requires_grad_()  # todo move this to discriminator code

    envelope_func = smooth_cutoff(bin_edges[1:], rrange[-1], rrange[-1] - 1)  # smooth out the last 1 angstrom
    smoothed_rdf = rdf * envelope_func.to(rdf.device)
    return smoothed_rdf, bin_centers, rdfs_dict


def get_1WL_env_labels(crystal_batch, precomputed_distances_dict):
    """Label every atom of a crystal cluster batch by its bonded environment in the asymmetric unit.

    Bonds are the edges of ``precomputed_distances_dict['edge_index']`` whose
    ``'intramolecular_dist'`` is below 1.8 (in the units of the stored distances). Every bond endpoint
    must be an asymmetric-unit atom (``aux_ind == 0``). On the asymmetric-unit atoms of the whole
    batch jointly, labels start from the distinct (atomic number, degree) pairs, degree counting the
    bond edges whose first index is the atom, and are refined by 1-WL colour refinement: each round
    relabels an atom by its current label together with the sorted labels of its bonded neighbours.
    Refinement stops when the number of distinct labels stops changing, or after 32 rounds. Labels
    are the consecutive integers from 0 and are shared by all graphs of the batch.

    The labels are then copied onto the cluster. For graph g with Z' = ``z_prime[g]``, the
    asymmetric-unit atoms are split into its units in storage order, as the unit map of the rows the
    cluster was built from says (``crystal_building.utils.unit_layout``: Z' equal blocks of
    ``num_atoms // z_prime`` atoms at this stage); the cluster atoms of graph
    g whose ``mol_ind`` is at least the first ``mol_ind`` of block z and below that of block z + 1
    (for the last block, up to and including the graph's largest ``mol_ind``) receive block z's
    labels, tiled. This assumes
    those atoms are stored as whole molecule copies in the asymmetric-unit atom order; atom-count
    divisibility is asserted, the order is not.

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Batched crystal cluster; reads ``aux_ind``, ``z``, ``batch``, ``num_graphs``, ``z_prime``
        and ``mol_ind``. Not modified.
    precomputed_distances_dict : dict
        Provides ``'edge_index'`` [2, num_intra_edges] and ``'intramolecular_dist'``
        [num_intra_edges].

    Returns
    -------
    torch.LongTensor
        [num_nodes] environment label of every atom of ``crystal_batch``.

    Raises
    ------
    ValueError
        If a crystal's units cannot be read (``unit_layout``: e.g. ``num_atoms`` not divisible by its
        Z'), or a graph's asymmetric-unit (``aux_ind == 0``) atom count is not its units' total
        (explicit raises; before 2026-10-22 the divisibility was a bare assert).
    AssertionError
        If a bond touches an atom outside the asymmetric unit, a Z'-molecule's cluster atom count is
        not divisible by its atoms per molecule, or any atom is left unlabelled.
    """
    intra_edges = precomputed_distances_dict['edge_index']
    intra_dists = precomputed_distances_dict['intramolecular_dist']

    def bond_edges_from_distances(edge_index, intramolecular_dist,
                                  bond_cutoff=1.8):
        """Filter a distance-cutoff edge list down to covalent bonds."""
        mask = intramolecular_dist < bond_cutoff
        return edge_index[:, mask]

    edge_index = bond_edges_from_distances(intra_edges, intra_dists)
    max_iter = 32
    asym_mask = crystal_batch.aux_ind == 0
    z = crystal_batch.z[asym_mask]
    device = z.device
    N = z.shape[0]
    # global → local-asym remap; -1 for non-asym atoms
    global_to_local = torch.full(
        (crystal_batch.z.shape[0],), -1, dtype=torch.long, device=device
    )
    global_to_local[asym_mask] = torch.arange(N, device=device)
    src = global_to_local[edge_index[0]]
    dst = global_to_local[edge_index[1]]
    assert (src >= 0).all() and (dst >= 0).all(), \
        "bond edges touch a non-asym atom — check aux_ind / edge_index consistency"
    degree = torch.zeros(N, dtype=torch.long, device=device).scatter_add_(
        0, src, torch.ones_like(src, dtype=torch.long)
    )
    max_deg = int(degree.max().item())
    # Initial invariant: (Z, degree). Adding degree just speeds convergence.
    init = torch.stack([z.long(), degree], dim=1)
    _, labels = torch.unique(init, dim=0, return_inverse=True)
    num_classes = int(labels.max().item()) + 1
    offset = torch.zeros(N, dtype=torch.long, device=device)
    offset[1:] = degree.cumsum(0)[:-1]
    for iter in range(max_iter):
        K = int(labels.max().item()) + 1
        msg = labels[dst]
        # Lex-sort edges by (src, neighbor-label) → within each src, msgs are sorted
        order = (src * K + msg).argsort()
        src_s, msg_s = src[order], msg[order]
        within = torch.arange(src_s.numel(), device=device) - offset[src_s]

        # [N, max_deg] padded sorted-neighbor-label matrix
        pad = torch.full((N, max_deg), -1, dtype=labels.dtype, device=device)
        pad[src_s, within] = msg_s

        sig = torch.cat([labels[:, None], pad], dim=1)
        _, new_labels = torch.unique(sig, dim=0, return_inverse=True)
        new_num = int(new_labels.max().item()) + 1

        labels = new_labels
        if new_num == num_classes:
            break
        num_classes = new_num

    # doesn't work on mixed / Z'>1 batches
    # labels_full = torch.empty(crystal_batch.z.shape[0], dtype=torch.long, device=device)
    # for g in range(crystal_batch.num_graphs):
    #     node_mask = crystal_batch.batch == g  # all atoms in graph g
    #     asym_g_mask = node_mask & asym_mask  # asym atoms in graph g
    #     labels_g = labels[global_to_local[asym_g_mask]]  # asym-unit labels for g
    #     n_copies_g = int(node_mask.sum().item()) // labels_g.shape[0]
    #     labels_full[node_mask] = labels_g.repeat(n_copies_g)
    labels_full = torch.full((crystal_batch.z.shape[0],), -1, dtype=torch.long, device=device)
    node_idx = torch.arange(crystal_batch.z.shape[0], device=device)
    # the units of each crystal, from the unit map of the rows the cluster was built from (the one reader,
    # crystal_building.utils.unit_layout; a cluster carries no unit_ind, so the layout is the rows' Z' equal
    # contiguous blocks of num_atoms // z_prime atoms, which its aux_ind == 0 atoms hold in unit order). This reads
    # the ABSENCE path even when the rows carried a map: right only because every build refuses a map that is not
    # those blocks before a cluster exists. The stage that lifts equal / contiguous units must carry the rows' unit
    # atom counts (or unit_ptr) onto the cluster and read them here (stage-2 review, 2026-10-23)
    from mxtaltools.crystal_building.utils import unit_layout
    per_unit = unit_layout(crystal_batch, 'envwise RDF labels').atoms_per_unit.tolist()

    for g in range(crystal_batch.num_graphs):
        node_mask = crystal_batch.batch == g
        asym_g_mask = node_mask & asym_mask
        labels_g = labels[global_to_local[asym_g_mask]]

        z_prime_g = int(crystal_batch.z_prime[g].item())
        n_asym = labels_g.shape[0]
        atoms_per_mol = per_unit[g]
        if n_asym != atoms_per_mol * z_prime_g:
            raise ValueError(f"envwise RDF labels: graph {g} holds {n_asym} asymmetric-unit (aux_ind == 0) atoms, "
                             f"but its {z_prime_g} unit(s) hold {atoms_per_mol} atoms each")

        # asym mol_inds, one per Z'-mol, in asym-block order
        asym_block_ids = crystal_batch.mol_ind[asym_g_mask][::atoms_per_mol]

        cluster_mi = crystal_batch.mol_ind[node_mask]
        global_idx_g = node_idx[node_mask]

        for z in range(z_prime_g):
            lower = asym_block_ids[z]
            upper = asym_block_ids[z + 1] if z + 1 < z_prime_g else cluster_mi.max() + 1
            sub_mask = (cluster_mi >= lower) & (cluster_mi < upper)

            n_sub = int(sub_mask.sum().item())
            assert n_sub % atoms_per_mol == 0, (
                f"graph {g} Z'-mol {z}: cluster atoms {n_sub} not divisible by {atoms_per_mol}"
            )
            n_copies_z = n_sub // atoms_per_mol
            labels_mol = labels_g[z * atoms_per_mol:(z + 1) * atoms_per_mol]
            labels_full[global_idx_g[sub_mask]] = labels_mol.repeat(n_copies_z)

    assert (labels_full >= 0).all(), "some atoms were not labelled"

    return labels_full

def smooth_cutoff(d, rc, ron):
    """Raised-cosine envelope, 1 up to ``ron`` and falling to 0 at ``rc``.

    Returns ``0.5 * (1 + cos(pi * x)) * (d <= rc)`` with ``x = clamp((d - ron) / (rc - ron), 0, 1)``.

    Parameters
    ----------
    d : torch.Tensor
        Distances, any shape.
    rc : float
        Cutoff; the envelope is 0 at and beyond it.
    ron : float
        Onset of damping, below ``rc``.

    Returns
    -------
    torch.Tensor
        Envelope with the shape of ``d``.
    """
    # d: distances
    # rc: cutoff
    # ron: onset of damping

    x = (d - ron) / (rc - ron)
    x = torch.clamp(x, 0.0, 1.0)

    envelope = 0.5 * (1.0 + torch.cos(torch.pi * x))
    envelope = envelope * (d <= rc)

    return envelope


def smooth_1d_histogram(bins, device, dist, graph_indexed_pairs, num_graphs, num_pairs, rrange, sigma=None):
    """Gaussian-smoothed histograms of distances, one per (graph, channel).

    Each distance whose ``graph_indexed_pairs`` entry is non-negative adds
    ``delta_r * N(c_k; d, sigma)`` to every bin k of histogram ``graph_indexed_pairs``, where N is
    the normal density, ``delta_r = (rmax - rmin) / bins`` and the kernel centres are
    ``c = linspace(rmin, rmax, bins)``. Distances outside ``rrange`` are neither clipped nor dropped;
    they contribute through the Gaussian tails. Entries with a negative index are dropped.

    Parameters
    ----------
    bins : int
        Number of bins per histogram.
    device : torch.device or str
        Device of the returned tensors.
    dist : torch.Tensor
        [num_edges] distances.
    graph_indexed_pairs : torch.LongTensor
        [num_edges] flat histogram index ``graph * num_pairs + channel`` of each distance, or
        negative to drop it.
    num_graphs : int
        Number of graphs.
    num_pairs : int
        Number of channels per graph.
    rrange : tuple of float
        (rmin, rmax).
    sigma : float, optional
        Gaussian width; defaults to ``(rmax - rmin) / bins``.

    Returns
    -------
    hist : torch.Tensor
        [num_graphs, num_pairs, bins] smoothed counts.
    rdf_density : torch.Tensor
        [num_graphs, num_pairs] ones.

    Notes
    -----
    The kernel centres are spaced by ``(rmax - rmin) / (bins - 1)`` and run from ``rmin`` to
    ``rmax`` inclusive, whereas ``delta_r`` and the default ``sigma`` use ``(rmax - rmin) / bins``;
    the centres do not coincide with the midpoints of ``linspace(rmin, rmax, bins + 1)``.
    """
    rmin, rmax = rrange
    bin_centers = torch.linspace(rmin, rmax, bins, device=device)

    if sigma is None:
        sigma = (rmax - rmin) / bins

    valid = graph_indexed_pairs >= 0
    gip = graph_indexed_pairs[valid]
    d = dist[valid]

    delta_r = (rmax - rmin) / bins

    weights = torch.exp(
        -0.5 * ((d[:, None] - bin_centers[None, :]) / sigma) ** 2
    )

    weights = weights / (math.sqrt(2 * math.pi) * sigma)

    weights = weights * delta_r

    # flatten (hist, bin) → single index
    flat_idx = gip[:, None] * bins + torch.arange(bins, device=device)[None, :]

    flat_hist = torch.zeros(
        num_graphs * num_pairs * bins,
        device=device
    )

    flat_hist.scatter_add_(
        0,
        flat_idx.reshape(-1),
        weights.reshape(-1),
    )

    hist = flat_hist.view(num_graphs, num_pairs, bins)

    rdf_density = torch.ones((num_graphs, num_pairs), device=device)
    return hist, rdf_density


def new_1d_histogram(bins, device, dists, graph_indexed_pairs, num_graphs, num_pairs, rrange):
    """Hard-binned histograms of distances, one per (graph, channel).

    Each distance whose ``graph_indexed_pairs`` entry is non-negative and with
    ``rmin <= d < rmax`` adds 1 to bin ``floor((d - rmin) / delta_r)``, where
    ``delta_r = (rmax - rmin) / bins``; all other distances are dropped.

    Parameters
    ----------
    bins : int
        Number of bins per histogram.
    device : torch.device or str
        Device of the returned tensors.
    dists : torch.Tensor
        [num_edges] distances.
    graph_indexed_pairs : torch.LongTensor
        [num_edges] flat histogram index ``graph * num_pairs + channel`` of each distance, or
        negative to drop it.
    num_graphs : int
        Number of graphs.
    num_pairs : int
        Number of channels per graph.
    rrange : tuple of float
        (rmin, rmax).

    Returns
    -------
    hist : torch.Tensor
        [num_graphs, num_pairs, bins] counts.
    rdf_density : torch.Tensor
        [num_graphs, num_pairs] float32 ones.
    """
    valid = graph_indexed_pairs >= 0
    gip = graph_indexed_pairs[valid]
    d = dists[valid]
    # compute bin id explicitly
    rmin, rmax = rrange
    delta_r = (rmax - rmin) / bins
    # mask too-distant bins
    t = (d - rmin) / delta_r
    valid_bins = (t >= 0) & (t < bins)

    t = t[valid_bins]
    gip = gip[valid_bins]

    bin_id = t.floor().long()  # flatten (hist, bin) → single index
    scatter_idx = gip * bins + bin_id
    flat_hist = torch.zeros(num_graphs * num_pairs * bins, device=device)
    flat_hist.scatter_add_(
        0,
        scatter_idx,
        torch.ones_like(scatter_idx, dtype=torch.float32)
    )
    hist = flat_hist.view(num_graphs, num_pairs, bins)
    rdf_density = torch.ones((num_graphs, num_pairs), device=device, dtype=torch.float32)
    return hist, rdf_density


def get_elementwise_dists(atom_types: torch.LongTensor,
                          edges: torch.LongTensor,
                          device: Union[torch.device, str],
                          edge_in_crystal_number: torch.LongTensor,
                          atomic_numbers_override: Optional[torch.LongTensor] = None
                          ):
    """Assign each edge to the unordered element-pair channel of its graph.

    Parameters
    ----------
    atom_types : torch.LongTensor
        [num_nodes] atomic numbers, used directly as lookup-table indices.
    edges : torch.LongTensor
        [2, num_edges] node indices.
    device : torch.device or str
        Device of the default element list and the lookup table.
    edge_in_crystal_number : torch.LongTensor
        [num_edges] graph index of each edge.
    atomic_numbers_override : torch.LongTensor, optional
        Atomic numbers defining the channels. Default: B, C, N, O, F, P, S, Cl, Br
        (5, 6, 7, 8, 9, 15, 16, 17, 35).

    Returns
    -------
    graph_indexed_pairs : torch.LongTensor
        [num_edges] ``graph * num_pairs + pair``, with ``num_pairs = E * (E + 1) / 2`` for E listed
        elements and ``pair`` the upper-triangular index of the two endpoints' list positions; -1
        where either endpoint's element is not listed.
    rdfs_dict : dict
        Map from pair index to ``'<el1>_to_<el2>'`` with el1 at or before el2 in the element list.
        Names are element symbols for the default list and ``str(Z)`` for an override.
    """
    if atomic_numbers_override is None:
        relevant_elements = torch.tensor([5, 6, 7, 8, 9, 15, 16, 17, 35], dtype=torch.long, device=device)
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
    else:
        relevant_elements = atomic_numbers_override
        element_symbols = {int(i): str(int(i)) for i in relevant_elements}

    num_relevant_elements = len(relevant_elements)
    num_pairs = int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2)

    # NEW WAY
    # get atom types

    Z1 = atom_types[edges[0]]
    Z2 = atom_types[edges[1]]

    # reindex
    maxZ = max(relevant_elements.max() + 1, atom_types.amax() + 1)
    lut = -torch.ones(maxZ, dtype=torch.long, device=device)
    lut[relevant_elements] = torch.arange(len(relevant_elements), device=device)

    ii = lut[Z1]
    jj = lut[Z2]
    i = torch.minimum(ii, jj)
    j = torch.maximum(ii, jj)
    pair_ind = i * num_relevant_elements - i * (i - 1) // 2 + (j - i)
    graph_indexed_pairs = edge_in_crystal_number * num_pairs + pair_ind
    graph_indexed_pairs[pair_ind < 0] = -1

    rdfs_dict = {}
    ind = 0
    for i, element1 in enumerate(relevant_elements):
        for j, element2 in enumerate(relevant_elements):
            if j >= i:
                rdfs_dict[ind] = element_symbols[int(element1)] + '_to_' + element_symbols[int(element2)]
                ind += 1

    return graph_indexed_pairs, rdfs_dict


def get_atomwise_dists_old(atom_types: torch.LongTensor,
                           edges: torch.LongTensor,
                           dists: torch.Tensor,
                           device: Union[torch.device, str],
                           num_graphs: int,
                           edge_in_crystal_number: torch.LongTensor,
                           mol_num_atoms: torch.LongTensor,
                           ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Group edge distances by (graph, within-molecule atom-index pair) with dense boolean masks.

    With A = ``mol_num_atoms[0]``, pair (a, b), a <= b, collects the edges whose first endpoint has
    index a and whose second endpoint has index b; edges whose first index exceeds the second are
    not collected.

    Parameters
    ----------
    atom_types : torch.LongTensor
        [num_nodes] within-molecule atom index of each node.
    edges : torch.LongTensor
        [2, num_edges] node indices.
    dists : torch.Tensor
        [num_edges] edge lengths.
    device : torch.device or str
        Device of the masks.
    num_graphs : int
        Number of graphs.
    edge_in_crystal_number : torch.LongTensor
        [num_edges] graph index of each edge.
    mol_num_atoms : torch.LongTensor
        [num_graphs] atoms per molecule; all entries must equal the first (asserted).

    Returns
    -------
    dists_per_hist : torch.Tensor
        [num_graphs * num_pairs] number of distances in each (graph, pair) group, graph-major, with
        ``num_pairs = A * (A + 1) / 2``.
    sorted_dists : torch.Tensor
        The distances concatenated in that group order.
    rdfs_dict : dict
        Map from pair index to a name built from the string form of 0-d tensors, for example
        ``'tensor(0)_to_tensor(1)'`` on CPU.
    """
    assert torch.all(mol_num_atoms == mol_num_atoms[0]), "atomwise rdf not set up for variable molecule sizes"
    all_atoms = torch.arange(mol_num_atoms[0], device=atom_types.device)
    num_atoms = len(all_atoms)

    num_pairs = int((len(all_atoms) ** 2 + len(all_atoms)) / 2)

    atoms = [atom_types[edges[0]].long(), atom_types[edges[1]].long()]
    rdfs_dict = {}
    # prebuild pair lists
    elem_pair_list = torch.zeros((num_pairs, len(atoms[0])), dtype=torch.bool, device=device)
    elem1 = atoms[0].repeat(num_atoms, 1) == torch.Tensor(all_atoms)[:, None].to(device)
    elem2 = atoms[1].repeat(num_atoms, 1) == torch.Tensor(all_atoms)[:, None].to(device)
    ind = 0
    for i, element1 in enumerate(all_atoms):
        for j, element2 in enumerate(all_atoms):
            if j >= i:
                rdfs_dict[ind] = str(element1) + '_to_' + str(element2)
                elem_pair_list[ind] = elem1[i] * elem2[j]
                ind += 1

    bool_list = torch.zeros((num_pairs * num_graphs, len(dists)), dtype=torch.bool, device=device)
    for i in range(num_graphs):
        bool_list[i * num_pairs:(i + 1) * num_pairs, :] = (edge_in_crystal_number == i
                                                           ).repeat(num_pairs, 1) * elem_pair_list

    sorted_dists = dists.repeat(num_graphs * num_pairs, 1)[bool_list]
    # test assert bool_list.sum() == len(sorted_dists)
    dists_per_hist = bool_list.sum(1)

    return dists_per_hist, sorted_dists, rdfs_dict


def get_atomwise_dists(atom_inds,
                       edges: torch.LongTensor,
                       edge_in_crystal_number: torch.LongTensor,
                       mol_num_atoms: torch.LongTensor,
                       ) -> Tuple[torch.Tensor, dict]:
    """Assign each edge to the unordered label-pair channel of its graph.

    Parameters
    ----------
    atom_inds : torch.Tensor
        [num_nodes] integer label of each node. With A the number of distinct labels, the labels
        are assumed to be exactly 0, ..., A - 1 (not checked).
    edges : torch.LongTensor
        [2, num_edges] node indices.
    edge_in_crystal_number : torch.LongTensor
        [num_edges] graph index of each edge.
    mol_num_atoms : torch.LongTensor
        Unused.

    Returns
    -------
    graph_indexed_pairs : torch.LongTensor
        [num_edges] ``graph * num_pairs + pair``, with ``num_pairs = A * (A + 1) / 2`` and ``pair``
        the upper-triangular index of (smaller label, larger label).
    rdfs_dict : dict
        Map from pair index to ``'<a>_to_<b>'`` for labels a <= b.
    """

    A = len(torch.unique(atom_inds))
    num_pairs = A * (A + 1) // 2

    # --- get atom indices per edge ---
    i = atom_inds[edges[0]].long()
    j = atom_inds[edges[1]].long()

    # enforce i ≤ j
    ii = torch.minimum(i, j)
    jj = torch.maximum(i, j)

    # triangular indexing
    pair_ind = ii * A - ii * (ii - 1) // 2 + (jj - ii)

    # combine with graph index
    graph_indexed_pairs = edge_in_crystal_number * num_pairs + pair_ind

    # --- build dictionary ---
    rdfs_dict = {}
    ind = 0
    for a in range(A):
        for b in range(a, A):
            rdfs_dict[ind] = f"{a}_to_{b}"
            ind += 1

    return graph_indexed_pairs, rdfs_dict


def get_rdf_inds(crystal_batch,
                 mode: str,
                 device: Union[torch.device, str]
                 ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Select the centre ('inside') and neighbour ('outside') node indices for an RDF.

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Reads ``aux_ind`` and ``pos``.
    mode : str
        ``'intermolecular'``, ``'all'`` or ``'intramolecular'``; consulted only when ``aux_ind`` is
        not None.
    device : torch.device or str
        Device ``out_inds`` is moved to in the ``'intermolecular'`` and ``'all'`` modes; when
        ``aux_ind`` is None, the device of both outputs. ``in_inds`` otherwise stays on the device
        of ``aux_ind``.

    Returns
    -------
    in_inds : torch.LongTensor
        Nodes with ``aux_ind == 0``, or all nodes when ``aux_ind`` is None.
    out_inds : torch.LongTensor
        Nodes with ``aux_ind == 1`` for ``'intermolecular'``, all nodes for ``'all'``, and
        ``in_inds`` itself for ``'intramolecular'``; all nodes when ``aux_ind`` is None.

    Raises
    ------
    AssertionError
        If ``aux_ind`` is not None and ``mode`` is none of the three values; a message is printed
        first.
    """
    #  whether to include intramolecular edges

    if crystal_batch.aux_ind is not None:
        in_inds = torch.where(crystal_batch.aux_ind == 0)[0]
        if mode == 'intermolecular':
            out_inds = torch.where(crystal_batch.aux_ind == 1)[0].to(device)
        elif mode == 'all':
            out_inds = torch.arange(len(crystal_batch.pos)).to(device)
        elif mode == 'intramolecular':
            out_inds = in_inds
        else:
            print(mode + ' is not a valid rdf mode!')
            assert False
    else:
        # if we have no inside/outside labels, just consider the whole thing one big molecule
        in_inds = torch.arange(len(crystal_batch.pos)).to(device)
        out_inds = in_inds

    return in_inds, out_inds


# inspired by https://github.com/pytorch/pytorch/issues/99719#issuecomment-1664135524
def batch_histogram_1d(data_tensor: torch.Tensor,
                       batch: torch.LongTensor,
                       num_hists: int,
                       rrange: list[int, int] = [0, 10],
                       nbins: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Histogram a flat tensor of values into one histogram per batch index.

    A value x of sample b is counted in bin
    ``round(clip(x, rrange[0], rrange[1] - epsilon) / rrange[1] * nbins)`` of histogram b, with
    ``epsilon = (rrange[1] - rrange[0]) / nbins`` and ``torch.round`` (half to even). For
    ``rrange[0] == 0`` bin k therefore collects values within half a bin width of ``k * epsilon``,
    that is, of the lower edge of returned bin k rather than its centre; values below the range
    are counted in the first bin, and values from about ``rrange[1] - 1.5 * epsilon`` upward,
    including those beyond ``rrange[1]``, in the last. The index divides by ``rrange[1]``, not by the
    range width, so for ``rrange[0] != 0`` the bins do not follow the returned edges.

    Parameters
    ----------
    data_tensor : torch.Tensor
        [num_values] values to histogram.
    batch : torch.LongTensor
        [num_values] histogram index of each value, in [0, num_hists).
    num_hists : int
        Number of histograms.
    rrange : list of float
        [min, max] histogram range. Default [0, 10].
    nbins : int
        Number of bins per histogram. Default 100.

    Returns
    -------
    hists : torch.Tensor
        [num_hists, nbins] counts.
    bin_edges : torch.Tensor
        [nbins + 1] ``linspace(rrange[0], rrange[1], nbins + 1)``.
    """
    epsilon = (rrange[1] - rrange[0]) / nbins  # important to bracket bins
    ones = torch.ones_like(data_tensor)
    scatter_inds = batch.long() * nbins + (torch.round(
        data_tensor.clip(min=rrange[0], max=rrange[1] - epsilon) / rrange[1] * nbins)).long()  # convert float to long

    flat_hist = scatter(ones, scatter_inds, dim=0, dim_size=num_hists * nbins)
    hists = flat_hist.reshape(num_hists, nbins)

    return hists, torch.linspace(*rrange, nbins + 1, device=data_tensor.device)

    # a,b = torch.unique(batch, return_counts = True)  # test
    # for i, num in enumerate(a):
    #     assert hists[int(num)].sum() == b[i], f'{i} {num}'


def parallel_compute_rdf_torch(dists_list: torch.tensor, raw_density=True, rrange=None, bins=None,
                               remove_radial_scaling=False):
    """Compute one radial distribution per tensor of distances.

    Each tensor is histogrammed with ``torch.histc`` over ``rrange`` (values outside the range are
    ignored), one tensor at a time, and the histograms are stacked.

    Parameters
    ----------
    dists_list : list of torch.Tensor
        n 1-D tensors of distances (or the rows of a 2-D tensor); lengths may differ.
    raw_density : bool
        If True, every density is 1. If False, each density is estimated from the smaller half of
        that tensor's sorted distances as ``count / (4/3 * pi * ptp**3)``, ptp being their maximum
        minus minimum; any exception during the estimate sets that density to 1.
    rrange : list of float, optional
        [min, max] histogram range; defaults to [0.5, 10].
    bins : int, optional
        Number of bins; defaults to 100.
    remove_radial_scaling : bool
        If True, the histograms are divided by the density only, not by the shell volumes.

    Returns
    -------
    rdf : torch.Tensor
        [n, bins] histograms divided by the shell volumes ``4/3 * pi * (r_{k+1}**3 - r_k**3)`` and
        by the density (by the density only when ``remove_radial_scaling``).
    bin_upper_edges : torch.Tensor
        [bins] upper bin edges ``r[1:]`` of ``r = linspace(rrange[0], rrange[1], bins + 1)``, with
        ``requires_grad`` set.
    """

    hist_range = [0.5, 10] if rrange is None else rrange
    hist_bins = 100 if bins is None else bins

    if not raw_density:  # estimate the density from the distances
        rdf_density = torch.zeros(len(dists_list)).to(dists_list[0].device)
        for i in range(len(dists_list)):
            dists = dists_list[i]
            try:
                sorted_dists = torch.sort(dists)[0][
                    :len(dists) // 2]  # we will use 1/2 the dist radius to avoid edge / cutoff effects
                volume = 4 / 3 * torch.pi * torch_ptp(
                    sorted_dists) ** 3  # volume of a sphere #np.ptp(sorted_dists[:, 0]) * np.ptp(sorted_dists[:, 1]) * np.ptp(sorted_dists[:, 2])
                # number of particles divided by the volume
                rdf_density[i] = len(sorted_dists) / volume
            except:
                rdf_density[i] = 1
    else:
        rdf_density = torch.ones(len(dists_list), device=dists_list[0].device, dtype=torch.float32)

    hh_list = torch.stack(
        [torch.histc(dists, min=hist_range[0], max=hist_range[1], bins=hist_bins) for dists in dists_list])
    rr = torch.linspace(hist_range[0], hist_range[1], hist_bins + 1, device=hh_list.device)
    if remove_radial_scaling:
        rdf = hh_list / rdf_density[:, None]  # un-smoothed radial density
    else:
        shell_volumes = (4 / 3) * torch.pi * (
                (rr[:-1] + torch.diff(rr)) ** 3 - rr[:-1] ** 3)  # volume of the shell at radius r+dr
        rdf = hh_list / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density

    return rdf, (rr[:-1] + torch.diff(rr)).requires_grad_()  # rdf and x-axis


def compute_rdf_distmat(rdf_record, bins, show_tqdm=True, chunk_size=250):
    """Symmetric matrix of `compute_rdf_distance` between every pair of RDFs.

    For each i >= 1, the distances from ``rdf_record[i]`` to ``rdf_record[:i]`` are computed in
    chunks of at most ``chunk_size`` RDFs and stored in row i; the strictly lower triangle is then
    added to its transpose. The diagonal is zero.

    Parameters
    ----------
    rdf_record : torch.Tensor
        [num_rdfs, num_channels, num_bins] RDFs.
    bins : torch.Tensor
        Uniformly spaced radial grid of the RDFs; only ``bins[1] - bins[0]`` is used, as the bin
        width.
    show_tqdm : bool
        Show a progress bar over rows.
    chunk_size : int
        Maximum number of RDFs compared with row i per call.

    Returns
    -------
    torch.Tensor
        [num_rdfs, num_rdfs] distances in the units of ``bins``, allocated by ``torch.zeros`` on the
        default device whatever the device of ``rdf_record``.
    """
    rdf_dists = torch.zeros(rdf_record.shape[0], rdf_record.shape[0])

    for i in tqdm(range(1, len(rdf_record)), disable=not show_tqdm):
        num_chunks = i // chunk_size + 1
        for j in range(num_chunks):
            start_ind = j * chunk_size
            stop_ind = min(i, (j + 1) * chunk_size)
            rdf_dists[i, start_ind:stop_ind] = compute_rdf_distance(
                rdf_record[i],
                rdf_record[start_ind:stop_ind],
                bins,
                n_parallel_rdf2=stop_ind - start_ind)
    rdf_dists = rdf_dists + rdf_dists.T  # symmetric distance matrix
    return rdf_dists


def compute_rdf_distmat_block(rdf_record: torch.Tensor,
                              rr,
                              i_range,
                              j_range) -> torch.Tensor:
    """One block of the strictly lower triangle of the RDF distance matrix.

    Parameters
    ----------
    rdf_record : torch.Tensor
        [num_rdfs, num_channels, num_bins] RDFs.
    rr : torch.Tensor
        Uniformly spaced radial grid; only ``rr[1] - rr[0]`` is used.
    i_range : list of int
        [start, stop) global row indices of the block.
    j_range : list of int
        [start, stop) global column indices of the block.

    Returns
    -------
    torch.Tensor
        [i_range[1] - i_range[0], j_range[1] - j_range[0]] float32 block holding
        `compute_rdf_distance` where the global row index exceeds the global column index, and 0
        elsewhere.

    Notes
    -----
    Row and column indices are truncated at the number of entries in the block,
    ``(i_range[1] - i_range[0]) * (j_range[1] - j_range[0])``, not at the number of RDFs; rows or
    columns at or above that product are left at 0.
    """
    device = rdf_record.device

    num_rdfs = (j_range[1] - j_range[0]) * (i_range[1] - i_range[0])
    i_inds = torch.arange(i_range[0], min(i_range[1], num_rdfs), device=device)
    j_inds = torch.arange(j_range[0], min(j_range[1], num_rdfs), device=device)

    mg = torch.meshgrid(i_inds, j_inds)
    ind1, ind2 = mg[0].flatten(), mg[1].flatten()
    good_bools = ind1 > ind2
    ind1_g = ind1[good_bools]
    ind2_g = ind2[good_bools]
    dists = compute_rdf_distance(
        rdf_record[ind1_g],
        rdf_record[ind2_g],
        rr,
    )
    distmat = torch.zeros((i_range[1] - i_range[0], j_range[1] - j_range[0]), dtype=torch.float32, device=device)
    distmat[ind1_g - i_range[1], ind2_g - j_range[1]] = dists

    return distmat


def compute_rdf_distmat_parallel(rdf_record, rr, num_cpus, chunk_size=250):
    """Symmetric matrix of ``log10(1 + d)`` for the RDF distances d, computed block-wise in a process pool.

    The index range is split into blocks of ``chunk_size``; every (row block, column block) pair is
    evaluated by `compute_rdf_distmat_block` in a ``multiprocessing.Pool`` of ``num_cpus``
    workers, the assembled strictly lower triangle is added to its transpose, and
    ``log10(1 + d)`` is returned. Unlike `compute_rdf_distmat`, the result is log-transformed.

    Parameters
    ----------
    rdf_record : torch.Tensor
        [num_rdfs, num_channels, num_bins] RDFs.
    rr : torch.Tensor
        Uniformly spaced radial grid; only ``rr[1] - rr[0]`` is used.
    num_cpus : int
        Number of worker processes.
    chunk_size : int
        Block edge length.

    Returns
    -------
    torch.Tensor
        [num_rdfs, num_rdfs] ``log10(1 + d)``, allocated by ``torch.zeros`` on the default device.
        Entries affected by the index truncation in `compute_rdf_distmat_block` are 0.
    """
    import multiprocessing as mp
    from math import ceil

    pool = mp.Pool(num_cpus)
    num_rdfs = len(rdf_record)
    num_chunks = ceil(len(rdf_record) / chunk_size)

    out = []
    for i in range(num_chunks):
        for j in range(num_chunks):
            out.append(
                pool.apply_async(
                    compute_rdf_distmat_block,
                    (rdf_record,
                     rr,
                     [i * chunk_size, min((i + 1) * chunk_size, num_rdfs)],
                     [j * chunk_size, min((j + 1) * chunk_size, num_rdfs)],
                     )
                )
            )
    pool.close()
    pool.join()
    out = [thing.get() for thing in out]

    rdf_dists = torch.zeros(rdf_record.shape[0], rdf_record.shape[0])
    ind = 0
    for i in range(num_chunks):
        for j in range(num_chunks):
            rdf_dists[i * chunk_size: min((i + 1) * chunk_size, num_rdfs),
            j * chunk_size: min((j + 1) * chunk_size, num_rdfs)] = out[ind]
            ind += 1

    rdf_dists = rdf_dists + rdf_dists.T  # symmetric distance matrix
    rdf_dists = torch.log10(1 + rdf_dists)
    return rdf_dists


def compute_rdf_distance(rdf1, rdf2, rr, n_parallel_rdf2: int = None, return_numpy: bool = False, channel_weights: torch.tensor=None):
    """Distance between multi-channel RDFs: mean over active channels of the 1-D earth mover's distance.

    Each channel of each RDF is divided by its sum over bins (plus 1e-10); the earth mover's
    distance ``sum_k |cumsum(p1)_k - cumsum(p2)_k|`` (`earth_movers_distance_torch`) is computed per
    channel and multiplied by the bin width ``rr[1] - rr[0]``. Channels are active when their
    unnormalised sum exceeds 1e-12 in either RDF. Without ``channel_weights`` the result is
    ``sum(emd * active) / max(n_active, 1)``, so two RDFs with no active channel are at distance 0;
    with them it is ``sum(emd * active * w) / max(sum(active * w), 1e-10)``. ``rdf1`` and ``rdf2``
    broadcast against each other over their leading dimensions.

    Parameters
    ----------
    rdf1 : torch.Tensor or np.ndarray
        [..., num_channels, num_bins] RDF(s), for example [num_channels, num_bins] for one
        structure. If it is not a tensor, ``rdf1`` and ``rdf2`` are converted with
        ``torch.Tensor`` (float32, CPU) and the result is returned as a NumPy array.
    rdf2 : torch.Tensor or np.ndarray
        [..., num_channels, num_bins] RDF(s) with the same channel count as ``rdf1`` (asserted);
        converted only when ``rdf1`` is not a tensor.
    rr : torch.Tensor or np.ndarray
        Uniformly spaced radial grid of both RDFs, such as the bin positions returned by
        `crystal_rdf`; converted with ``torch.Tensor`` if not a tensor. Only ``rr[1] - rr[0]`` is
        used, and the distance is in its units.
    n_parallel_rdf2 : int, optional
        If given, ``rdf1`` is first replaced by ``rdf1.tile(n_parallel_rdf2, 1, 1)``, giving
        [n_parallel_rdf2, num_channels, num_bins] for a single [num_channels, num_bins] ``rdf1``.
    return_numpy : bool
        Return a detached CPU NumPy array; forced to True when ``rdf1`` is not a tensor.
    channel_weights : torch.Tensor, optional
        [num_channels] weight of each channel, moved to the device of the distances.

    Returns
    -------
    torch.Tensor or np.ndarray
        Distances with the broadcast leading shape of the inputs: [n] for ``rdf1``
        [num_channels, num_bins] and ``rdf2`` [n, num_channels, num_bins], or 0-d for two single
        RDFs without ``channel_weights`` ([1] with them).

    Raises
    ------
    AssertionError
        If the channel counts differ or any distance is NaN.
    """

    if not torch.is_tensor(rdf1):
        torch_rdf1 = torch.Tensor(rdf1)
        torch_rdf2 = torch.Tensor(rdf2)
        return_numpy = True
    else:
        torch_rdf1 = rdf1
        torch_rdf2 = rdf2

    assert torch_rdf1.shape[-2] == torch_rdf2.shape[-2], "RDF distance must be done over equal channel sizes"

    if not torch.is_tensor(rr):
        torch_range = torch.Tensor(rr)
    else:
        torch_range = rr

    if n_parallel_rdf2 is not None: # todo consider deprecating
        torch_rdf1_f = torch_rdf1.tile(n_parallel_rdf2, 1, 1)
    else:
        torch_rdf1_f = torch_rdf1

    bin_width = torch_range[1] - torch_range[0]

    # RDF should measure how much mass needs to move, by how far
    # emd = earth_movers_distance_torch(torch_rdf1_f, torch_rdf2) * bin_width
    rdf1_normalized = torch_rdf1_f / (torch_rdf1_f.sum(dim=-1, keepdim=True) + 1e-10)
    rdf2_normalized = torch_rdf2 / (torch_rdf2.sum(dim=-1, keepdim=True) + 1e-10)

    # EMD between normalized distributions (now intensive)
    emd = earth_movers_distance_torch(rdf1_normalized, rdf2_normalized) * bin_width

    # take the raw average over nonzero element pairs
    eps = 1e-12
    active = (torch_rdf1_f.sum(dim=-1) > eps) | (torch_rdf2.sum(dim=-1) > eps)

    if channel_weights is not None:
        w = channel_weights.to(emd.device)[None, :]  # [1, n_channels]
        distance = (emd * active * w).sum(dim=-1) / (active * w).sum(dim=-1).clamp_min(1e-10)
    else:
        distance = (emd * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(1)

    # distance = emd.mean(-1)
    #
    # mass = (rdf1.sum(-1) + rdf2.sum(-1)) / 2
    # mass = mass / (1e-3 + mass.mean(dim=-1, keepdim=True))
    #
    # distance = (emd * mass).mean(-1)

    assert torch.sum(torch.isnan(distance)) == 0, "NaN EMD Distances Computed"
    if return_numpy:
        distance = distance.cpu().detach().numpy()

    return distance


def earth_movers_distance_torch(pdf1: torch.tensor, pdf2: torch.tensor):
    """1-D earth mover's distance between histograms along the last dimension.

    Returns ``sum(|cumsum(pdf1) - cumsum(pdf2)|)`` over the last dimension, without normalising the
    inputs and without a bin-width factor.

    Parameters
    ----------
    pdf1 : torch.Tensor
        [..., num_bins] histograms.
    pdf2 : torch.Tensor
        [..., num_bins] histograms, broadcastable with ``pdf1``.

    Returns
    -------
    emd : torch.Tensor
        [...] distances.
    """

    return torch.sum(torch.abs(torch.cumsum(pdf1, dim=-1) - torch.cumsum(pdf2, dim=-1)), dim=-1)


def earth_movers_distance_np(pdf1: np.ndarray, pdf2: np.ndarray):
    """NumPy version of `earth_movers_distance_torch`.

    Returns ``sum(|cumsum(pdf1) - cumsum(pdf2)|)`` over the last axis, without normalising the
    inputs and without a bin-width factor.

    Parameters
    ----------
    pdf1 : np.ndarray
        [..., num_bins] histograms.
    pdf2 : np.ndarray
        [..., num_bins] histograms, broadcastable with ``pdf1``.

    Returns
    -------
    emd : np.ndarray
        [...] distances.
    """
    return np.sum(np.abs(np.cumsum(pdf1, axis=-1) - np.cumsum(pdf2, axis=-1)), axis=-1)

def rdf_radial_graph(rdfs, device, d_cut, bins, chunk_size: int = 4):
    """Neighbour lists of RDFs within an RDF-distance cutoff.

    ``rdfs`` is moved to ``device`` and, for chunks of query rows, the distance to every RDF is
    computed as in `compute_rdf_distance` without channel weights (per-channel unit-sum
    normalisation, earth mover's distance times ``bins[1] - bins[0]``, mean over the channels
    active in either RDF). The neighbours of row i are the RDFs at distance strictly below
    ``d_cut``, excluding i itself. When an exception is classified as out-of-memory by
    `is_cuda_oom`, the chunk size is reduced by one (to at least 1), ``torch.cuda.synchronize`` and
    ``torch.cuda.empty_cache`` are called and the chunk is retried; other exceptions propagate. A tqdm progress bar is always shown.

    Parameters
    ----------
    rdfs : torch.Tensor
        [N, num_channels, num_bins] RDFs.
    device : torch.device or str
        Device on which the distances are computed.
    d_cut : float
        Neighbour cutoff, in the units of ``bins``.
    bins : torch.Tensor
        Uniformly spaced radial grid; only ``bins[1] - bins[0]`` is used.
    chunk_size : int
        Initial number of query rows per chunk.

    Returns
    -------
    neighbor_lists : list of torch.LongTensor
        N CPU tensors of neighbour indices, in row order.
    neighbor_dists : list of torch.Tensor
        N CPU tensors of the matching distances.

    Notes
    -----
    The retry has no attempt limit: an out-of-memory error that recurs at chunk size 1 is retried
    indefinitely. A chunk that fails after some of its rows were appended appends those rows again
    on retry.
    """
    rdfs_gpu = rdfs.to(device)
    rdf_norm = rdfs_gpu / (rdfs_gpu.sum(dim=-1, keepdim=True) + 1e-10)
    cdfs = torch.cumsum(rdf_norm, dim=-1)
    bin_width = bins[1] - bins[0]
    eps = 1e-12
    N = rdfs_gpu.shape[0]

    neighbor_lists = []
    neighbor_dists = []

    i = 0
    pbar = tqdm(total=N)

    while True:
        c = min(chunk_size, N-i)
        try:
            c = min(chunk_size, N - i)
            emd = torch.sum(torch.abs(cdfs[i:i + c, None] - cdfs[None, :]), dim=-1) * bin_width  # (c, N, 91)
            active = (rdfs_gpu[i:i + c, None].sum(-1) > eps) | (rdfs_gpu[None].sum(-1) > eps)  # (c, N, 91)
            d = (emd * active).sum(-1) / active.sum(-1).clamp_min(1)  # (c, N)

            for j in range(c):
                mask = d[j] < d_cut
                mask[i + j] = False
                neighbor_lists.append(mask.nonzero(as_tuple=False).squeeze(-1).cpu())
                neighbor_dists.append(d[j][mask].cpu())

            i += c
            pbar.update(c)
            if i >= N:
                break
        except Exception as e:
            if is_cuda_oom(e):
                print("cutting chunk size")
                chunk_size = max(1, chunk_size - 1)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            else:
                raise e

    return neighbor_lists, neighbor_dists