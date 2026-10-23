"""Intermolecular distance lists and pairwise energy and overlap functions on molecular clusters.

``get_intermolecular_dists_dict`` builds a distance dictionary from a cluster
batch. The per-graph energy and overlap functions read its intermolecular
entries, evaluate a pair function on every intermolecular edge and sum it per
graph with no factor of 1/2 (``vdW_analysis`` takes a configurable reduction
instead of the sum). The Lennard-Jones family (``lj_analysis``,
``qlj_analysis``, ``elj_analysis``) uses sigma = the sum of the two atoms'
van der Waals radii and a hard-coded epsilon of 1, so those energies are in
units of epsilon.

``vdw_radii`` arguments throughout are tensors of van der Waals radii indexed
by atomic number.
"""
from typing import Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mxtaltools.models.functions.radial_graph import radius, build_radial_graph


def old_compute_num_h_bonds(supercell_data, atom_acceptor_ind, atom_donor_ind, i):
    """Count close donor-acceptor atom pairs for one graph of a supercell batch.

    Non-directional count, up to a loose range (3.3 Angstrom): pairs of an
    acceptor atom in the canonical conformer (``aux_ind == 0``) and a donor
    atom outside it (``aux_ind == 1``) closer than 3.3. Pairs of a canonical
    donor and an outside acceptor are not counted.

    Parameters
    ----------
    supercell_data : batch object
        Reads ``ptr``, ``aux_ind``, ``x`` ([num_nodes, num_features]) and
        ``pos``.
    atom_acceptor_ind : int
        Column of ``supercell_data.x`` that equals 1 for acceptor atoms.
    atom_donor_ind : int
        Column of ``supercell_data.x`` that equals 1 for donor atoms.
    i : int
        Graph index; nodes ``ptr[i]:ptr[i + 1]`` are examined.

    Returns
    -------
    torch.Tensor
        0-dim tensor, the number of such pairs.
    """
    batch_inds = torch.arange(supercell_data.ptr[i], supercell_data.ptr[i + 1], device=supercell_data.x.device)

    # find the canonical conformers
    canonical_conformers_inds = torch.where(supercell_data.aux_ind[batch_inds] == 0)[0]
    outside_inds = torch.where(supercell_data.aux_ind[batch_inds] == 1)[0]

    # identify and count canonical conformer acceptors and intermolecular donors
    canonical_conformer_acceptors_inds = \
        torch.where(supercell_data.x[batch_inds[canonical_conformers_inds], atom_acceptor_ind] == 1)[0]
    outside_donors_inds = torch.where(supercell_data.x[batch_inds[outside_inds], atom_donor_ind] == 1)[0]

    donors_pos = supercell_data.pos[batch_inds[outside_inds[outside_donors_inds]]]
    acceptors_pos = supercell_data.pos[batch_inds[canonical_conformers_inds[canonical_conformer_acceptors_inds]]]

    return torch.sum(torch.cdist(donors_pos, acceptors_pos, p=2) < 3.3)


def electrostatic_analysis(dist_dict, num_graphs: int, cutoff: float = 0.92):
    """Per-graph screened Coulomb energy over intermolecular edges.

    Technically this is a Yukawa potential
    (https://en.wikipedia.org/wiki/Yukawa_potential): each edge contributes
    ``q_i * q_j / d * exp(-cutoff * d)``, with no Coulomb prefactor, and edge
    energies are summed per graph.

    Parameters
    ----------
    dist_dict : dict
        Reads ``'intermolecular_dist'`` [num_inter_edges],
        ``'intermolecular_dist_batch'`` [num_inter_edges] and
        ``'intermolecular_partial_charges'`` (two [num_inter_edges] tensors,
        as written by ``get_intermolecular_dists_dict``).
    num_graphs : int
        Output length.
    cutoff : float, optional
        Decay constant of the exponential screening factor, in inverse units
        of the distances; it is not a distance cutoff. Default 0.92.

    Returns
    -------
    torch.Tensor
        [num_graphs] summed edge energies.

    Raises
    ------
    KeyError
        If `dist_dict` has no ``'intermolecular_partial_charges'`` entry.
    """
    batch = dist_dict['intermolecular_dist_batch']
    dists = dist_dict['intermolecular_dist']
    charges = dist_dict['intermolecular_partial_charges']
    # default cutoff of 0.77 gets us 99% exponentially squashed by 6 angstroms distance
    estat_energy = ((charges[0] * charges[1]) / dists) * torch.exp(-(dists * cutoff))
    molwise_estat_energy = scatter(estat_energy, batch, reduce='sum', dim_size=num_graphs)

    return molwise_estat_energy


def buckingham_energy(dist_dict,
                      num_graphs,
                      vdw_radii,
                      A: float = -3.2619,
                      B: float = 1.0,
                      C: float = -0.2,
                      ):
    """Per-graph Buckingham-form energy over intermolecular edges, with distances normalised by vdW radius sums.

    With ``x = d / (r_vdw_i + r_vdw_j)``, each edge contributes
    ``A * exp(-B * x) - C / x**6``; non-finite values are replaced
    (NaN -> 0, +inf -> 1e20, -inf -> -1e-20) and the result is summed per
    graph. The default A, B, C are fitted to very roughly match a 12-6 LJ
    potential: with the default signs the exponential term is attractive and
    the ``x**-6`` term repulsive, and the pair function has its minimum,
    about -1, at ``x = 1``.

    Parameters
    ----------
    dist_dict : dict
        Reads ``'intermolecular_dist'``, ``'intermolecular_dist_atoms'``
        (two long tensors of atomic numbers) and
        ``'intermolecular_dist_batch'``, each over [num_inter_edges].
    num_graphs : int
        Output length.
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    A : float, optional
        Prefactor of the exponential term. Default -3.2619.
    B : float, optional
        Decay rate of the exponential term in ``x``. Default 1.0.
    C : float, optional
        Coefficient of the ``x**-6`` term. Default -0.2.

    Returns
    -------
    torch.Tensor
        [num_graphs] summed edge energies.
    """
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    batch = dist_dict['intermolecular_dist_batch']

    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]
    normed_dists = dists / radii_sums
    bh = torch.nan_to_num(A * torch.exp(-B * normed_dists) - C / normed_dists ** 6,
                          nan=0.0, posinf=1e20, neginf=-1e-20
                          )
    molwise_bh = scatter(bh, batch, reduce='sum', dim_size=num_graphs)

    return molwise_bh

    #
    # def fit_buckingham_to_lj(r0=1.0, epsilon=1.0, B=1.2):
    #     import numpy as np
    #     from scipy.optimize import fsolve
    #     def equations(vars):
    #         A, C = vars
    #         # Energy match at r0
    #         V = A * np.exp(-B * r0) - C / r0**6 + epsilon
    #         # Derivative match at r0 (minimum)
    #         dV = -A * B * np.exp(-B * r0) + 6 * C / r0**7
    #         return [V, dV]
    #     # Initial guess
    #     guess = [100.0, 1000.0]
    #     A, C = fsolve(equations, guess)
    #     return A, B, C
    #     """
    #     A, B, C = fit_buckingham_to_lj(r0=1.0, epsilon=1.0, B=1.2)
    #     xx = torch.linspace(0, 10, 1001)
    #     sigma = 3.0
    #     epsilon = 1.0
    #     lj = 4 * ((sigma/xx)**12 - (sigma/xx)**6)
    #     B = 1
    #     A, B, C = fit_buckingham_to_lj(sigma, epsilon, B)
    #     """


def old_new_hydrogen_bond_analysis(supercell_data, dist_dict, cutoff: float = 2.2):
    """Count N-H and O-H donor to N and O acceptor contacts per graph.

    Donor protons are the row-0 nodes of intramolecular edges
    (``dist_dict['edge_index']``) that join an H atom (row 0) to an N or O
    atom (row 1) with length <= 1.5. Acceptors are outside nodes
    (``aux_ind == 1``) whose flattened ``supercell_data.x`` value is 7 (N) or
    8 (O). Every (proton, acceptor) pair in the same graph within `cutoff`
    counts once, with at most 100 protons found per acceptor.

    Parameters
    ----------
    supercell_data : batch object
        Reads ``x`` (one value per node, compared with 7 and 8), ``pos``,
        ``batch``, ``aux_ind`` and ``num_graphs``.
    dist_dict : dict
        Distance dict from ``get_intermolecular_dists_dict``; reads
        ``'edge_index'``, ``'intramolecular_dist'`` and
        ``'intramolecular_dist_atoms'``.
    cutoff : float, optional
        Maximum proton-acceptor distance. Default 2.2.

    Returns
    -------
    tuple of torch.Tensor
        Four [num_graphs] float tensors: counts of N-H...N, N-H...O, O-H...N
        and O-H...O contacts.
    """
    # identify donors as H in O-H or N-H
    proton_intra_edges_bool = dist_dict['intramolecular_dist_atoms'][0] == 1
    nitrogen_intra_edges_bool = dist_dict['intramolecular_dist_atoms'][1] == 7
    oxygen_intra_edges_bool = dist_dict['intramolecular_dist_atoms'][1] == 8

    intra_NH_bool = proton_intra_edges_bool * nitrogen_intra_edges_bool
    intra_OH_bool = proton_intra_edges_bool * oxygen_intra_edges_bool

    proton_is_NH_donor = dist_dict['edge_index'][0][(dist_dict['intramolecular_dist'] <= 1.5) * intra_NH_bool]
    proton_is_OH_donor = dist_dict['edge_index'][0][(dist_dict['intramolecular_dist'] <= 1.5) * intra_OH_bool]

    # identify acceptors in intermolecular edges
    N_acceptors = supercell_data.x.flatten() == 7
    O_acceptors = supercell_data.x.flatten() == 8

    # get hydrogen bonds
    NH_donors_pos = supercell_data.pos[proton_is_NH_donor]
    NH_donors_batch = supercell_data.batch[proton_is_NH_donor]
    OH_donors_pos = supercell_data.pos[proton_is_OH_donor]
    OH_donors_batch = supercell_data.batch[proton_is_OH_donor]

    outside_inds = supercell_data.aux_ind == 1
    N_acceptors_pos = supercell_data.pos[N_acceptors * outside_inds]
    N_acceptors_batch = supercell_data.batch[N_acceptors * outside_inds]
    O_acceptors_pos = supercell_data.pos[O_acceptors * outside_inds]
    O_acceptors_batch = supercell_data.batch[O_acceptors * outside_inds]

    NH_N_edge_index = radius(
        NH_donors_pos,
        N_acceptors_pos,
        cutoff,
        NH_donors_batch,
        N_acceptors_batch,
        max_num_neighbors=100,
    )
    NH_O_edge_index = radius(
        NH_donors_pos,
        O_acceptors_pos,
        cutoff,
        NH_donors_batch,
        O_acceptors_batch,
        max_num_neighbors=100,
    )
    OH_N_edge_index = radius(
        OH_donors_pos,
        N_acceptors_pos,
        cutoff,
        OH_donors_batch,
        N_acceptors_batch,
        max_num_neighbors=100,
    )
    OH_O_edge_index = radius(
        OH_donors_pos,
        O_acceptors_pos,
        cutoff,
        OH_donors_batch,
        O_acceptors_batch,
        max_num_neighbors=100,
    )

    # collect all bonds
    num_NH_N_bonds = scatter(torch.ones(NH_N_edge_index.shape[1]),
                             NH_donors_batch[NH_N_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)
    num_NH_O_bonds = scatter(torch.ones(NH_O_edge_index.shape[1]),
                             NH_donors_batch[NH_O_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)
    num_OH_N_bonds = scatter(torch.ones(OH_N_edge_index.shape[1]),
                             OH_donors_batch[OH_N_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)
    num_OH_O_bonds = scatter(torch.ones(OH_O_edge_index.shape[1]),
                             OH_donors_batch[OH_O_edge_index[1, :]],
                             dim_size=supercell_data.num_graphs)

    return num_NH_N_bonds, num_NH_O_bonds, num_OH_N_bonds, num_OH_O_bonds


def get_intermolecular_dists_dict(cluster_batch,
                                  conv_cutoff: float,
                                  max_num_neighbors: int = 10000):
    """Build the edge and distance dictionary for a cluster batch.

    Calls ``build_radial_graph`` with the batch's `aux_ind` and `mol_ind` and
    adds distances, atomic numbers and bookkeeping entries. Distances are
    computed from ``cluster_batch.pos`` without detaching, so they carry
    gradients to the positions; the edge selection is not differentiable.
    `cluster_batch` is not modified.

    Parameters
    ----------
    cluster_batch : batch object
        Reads ``pos``, ``batch``, ``ptr``, ``aux_ind``, ``mol_ind``,
        ``num_graphs``, ``num_atoms``, ``z`` and ``x``.
    conv_cutoff : float
        Edge radius, in the units of ``pos``; also stored as ``'cutoff'``.
    max_num_neighbors : int, optional
        Neighbour cap per query node, forwarded to ``build_radial_graph``.
        Default 10000.

    Returns
    -------
    dict
        Every key returned by ``build_radial_graph`` with `aux_ind`
        (``'edge_index'``, ``'edge_index_inter'``, ``'inside_inds'``,
        ``'outside_inds'``, ``'inside_batch'``, ``'n_repeats'``), plus:

        - ``'cutoff'``: ``float(conv_cutoff)``, read by the LJ-family
          envelope.
        - ``'num_graphs'``: ``cluster_batch.num_graphs``.
        - ``'graph_size'``: ``cluster_batch.num_atoms`` as stored on the
          batch.
        - ``'outside_batch'``: ``cluster_batch.batch``, the graph index of
          every node (not only outside nodes).
        - ``'intramolecular_dist'``: [num_intra_edges] Euclidean lengths of
          the ``'edge_index'`` edges.
        - ``'intramolecular_dist_atoms'``: list of two [num_intra_edges] long
          tensors, the atomic numbers (``z``) of the row-0 and row-1 nodes.
        - ``'intermolecular_dist'``: [num_inter_edges] Euclidean lengths of
          the ``'edge_index_inter'`` edges.
        - ``'intermolecular_dist_batch'``: [num_inter_edges] graph index of
          each edge's row-0 (source) node.
        - ``'intermolecular_dist_atoms'``: list of two [num_inter_edges] long
          tensors, the atomic numbers of the source and target nodes.
        - ``'intermolecular_partial_charges'``: present only when
          ``cluster_batch.x`` is not None and ``len(x) == len(z)``; list of
          two [num_inter_edges] tensors taken at the source and target nodes
          from ``x[:, 0]`` when ``x`` is 2-D, otherwise from ``x``.

    Raises
    ------
    KeyError
        If ``cluster_batch.aux_ind`` is None, since ``build_radial_graph``
        then returns no ``'edge_index_inter'``.
    TypeError
        If ``cluster_batch.aux_ind`` is set and ``cluster_batch.mol_ind`` is
        None (raised in ``build_radial_graph``).
    """
    dist_dict = {}
    edges_dict = build_radial_graph(
        cluster_batch.pos,
        cluster_batch.batch,
        cluster_batch.ptr,
        conv_cutoff,
        max_num_neighbors,
        aux_ind=cluster_batch.aux_ind,
        mol_ind=cluster_batch.mol_ind,
    )
    dist_dict.update(edges_dict)
    dist_dict['cutoff'] = float(conv_cutoff)  # the radius the pair list was built to; read by lj_cutoff_envelope
    dist_dict['num_graphs'] = cluster_batch.num_graphs
    dist_dict['graph_size'] = cluster_batch.num_atoms
    dist_dict['outside_batch'] = cluster_batch.batch
    dist_dict['intramolecular_dist'] = (
        (cluster_batch.pos[edges_dict['edge_index'][0]] - cluster_batch.pos[
            edges_dict['edge_index'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intramolecular_dist_atoms'] = \
        [cluster_batch.z[edges_dict['edge_index'][0]].long(),
         cluster_batch.z[edges_dict['edge_index'][1]].long()]

    dist_dict['intermolecular_dist'] = (
        (cluster_batch.pos[edges_dict['edge_index_inter'][0]] - cluster_batch.pos[
            edges_dict['edge_index_inter'][1]]).pow(2).sum(
            dim=-1).sqrt())

    dist_dict['intermolecular_dist_batch'] = cluster_batch.batch[edges_dict['edge_index_inter'][0]]

    dist_dict['intermolecular_dist_atoms'] = \
        [cluster_batch.z[edges_dict['edge_index_inter'][0]].long(),
         cluster_batch.z[edges_dict['edge_index_inter'][1]].long()]

    # if we have partial charges in the batch (always first column in x)
    if cluster_batch.x is not None and len(cluster_batch.x) == len(cluster_batch.z):
        if cluster_batch.x.ndim == 2:
            dist_dict['intermolecular_partial_charges'] = \
                [cluster_batch.x[edges_dict['edge_index_inter'][0], 0],
                 cluster_batch.x[edges_dict['edge_index_inter'][1], 0]]

        else:
            dist_dict['intermolecular_partial_charges'] = \
                [cluster_batch.x[edges_dict['edge_index_inter'][0]],
                 cluster_batch.x[edges_dict['edge_index_inter'][1]]]

    return dist_dict


def silu_energy(dist_dict,
                num_graphs: int,
                vdw_radii,
                repulsion: float = 1.0
                ):
    """Per-graph SiLU-based pair energy over intermolecular edges: a shorter-range and softer LJ-type energy.

    With s the vdW radius sum of an edge, each edge contributes
    ``silu(-4 * repulsion * (d - repulsion * s)) / 0.28`` and edge energies
    are summed per graph. The pair function is positive for
    ``d < repulsion * s``, negative beyond it with a minimum of about -0.99 at
    ``d = repulsion * s + 1.28 / (4 * repulsion)``, and tends to 0 at large
    ``d``.

    Parameters
    ----------
    dist_dict : dict
        Reads ``'intermolecular_dist'``, ``'intermolecular_dist_atoms'`` and
        ``'intermolecular_dist_batch'``.
    num_graphs : int
        Output length.
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    repulsion : float, optional
        Scales both the zero crossing (``repulsion * s``) and the steepness.
        Default 1.0.

    Returns
    -------
    torch.Tensor
        [num_graphs] summed edge energies.
    """

    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]
    # replusion = 1 roughly equivalent to LJ functional form
    # repulsion < 1 broadens the potential well and shifts the minimum to the left
    edgewise_potentials = (F.silu(-4 * (dists - radii_sums * repulsion) * repulsion) / 0.28)
    molwise_silu_energy = scatter(edgewise_potentials, dist_dict['intermolecular_dist_batch'],
                                  reduce='sum', dim_size=num_graphs)
    return molwise_silu_energy


def quadratic_edgewise_lj_energy(vdw_radii: torch.Tensor,
                                 dist_dict: dict,
                                 ):
    """Edgewise LJ energy with a quadratic repulsive wall inside sigma.

    For an edge with distance d and vdW radius sum s the energy is
    ``24 * ((d - s)**2 / s**2 - (d - s) / s)`` when ``d < s`` and
    ``compute_lj_edgewise`` (12-6 LJ, sigma = s, epsilon = 1) otherwise. The
    wall matches the LJ value (0) and slope (-24 / s) at ``d = s`` and
    equals 48 at ``d = 0``.

    Parameters
    ----------
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    dist_dict : dict
        Reads ``'intermolecular_dist'`` and ``'intermolecular_dist_atoms'``.

    Returns
    -------
    torch.Tensor
        [num_inter_edges] pair energies, in units of epsilon.
    """
    lj = compute_lj_edgewise(dist_dict, vdw_radii)
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]

    # Quadratic wall below sigma
    rep = 24 * 1 * ((dists - radii_sums) ** 2 / radii_sums ** 2 - (dists - radii_sums) / radii_sums)
    return torch.where(dists < radii_sums, rep, lj)


def exponential_edgewise_lj_energy(vdw_radii: torch.Tensor,
                                   dist_dict: dict,
                                   k_factor: float = 2.5 # controls stiffness. 2.5 yields E[r=0]=98
                                   ):
    """Edgewise LJ energy with an exponential repulsive wall inside sigma (the eLJ pair function).

    For an edge with distance d and vdW radius sum s, with
    ``k = k_factor / s`` and ``A = 24 / k_factor``, the energy is
    ``A * (exp(-k * (d - s)) - 1)`` when ``d < s`` and
    ``compute_lj_edgewise`` (12-6 LJ, sigma = s, epsilon = 1) otherwise. The
    wall matches the LJ value (0) and slope (-24 / s) at ``d = s``; at
    ``d = 0`` it equals ``(24 / k_factor) * (exp(k_factor) - 1)`` for every
    s, about 107.35 for ``k_factor = 2.5``.

    Parameters
    ----------
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    dist_dict : dict
        Reads ``'intermolecular_dist'`` and ``'intermolecular_dist_atoms'``.
    k_factor : float, optional
        Stiffness of the wall; larger is stiffer. Default 2.5.

    Returns
    -------
    torch.Tensor
        [num_inter_edges] pair energies, in units of epsilon.
    """
    lj = compute_lj_edgewise(dist_dict, vdw_radii)
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]


    V0 = torch.zeros_like(dists)  # LJ(sigma) = +1 * epsilon
    F0 = -24.0 / radii_sums  # dV_LJ/dr at sigma

    # Build exponential parameters (smooth match)
    k = k_factor / radii_sums  # stiffness
    A = -F0 / k
    B = V0 - A

    # Exponential core
    rep = A * torch.exp(-k * (dists - radii_sums)) + B

    return torch.where(dists < radii_sums, rep, lj)


def lj_cutoff_envelope(dists: torch.Tensor, r_off: float, width: float) -> torch.Tensor:
    """Smooth switch that takes a pair energy to zero at the pair-list cutoff.

    S(r) = 1 for r <= r_off - width, 0 for r >= r_off, and the quintic smoothstep
    1 - x^3 (10 - 15 x + 6 x^2), x = (r - (r_off - width)) / width, in between. S is C2: its
    first and second derivatives vanish at both ends, so an enveloped energy and its forces
    are continuous as pairs enter and leave the list. Without it, the LJ family's energy steps
    by the pair energy at r_off each time a pair crosses the cutoff (about -0.006 epsilon per
    pair at 10 A for sigma ~ 3.4 A).

    Parameters
    ----------
    dists : torch.Tensor
        Pair distances, any shape.
    r_off : float
        Distance at which S reaches 0, in the units of `dists`.
    width : float
        Width of the switching window. Not validated here; the LJ-family
        analysis functions reject a width outside (0, r_off] before calling
        this.

    Returns
    -------
    torch.Tensor
        S evaluated elementwise, same shape as `dists`.
    """
    x = ((dists - (r_off - width)) / width).clamp(0.0, 1.0)
    # factored form of 1 - x^3 (10 - 15 x + 6 x^2): vanishes as (1 - x)^3 without cancellation in float32
    return (1.0 - x) ** 3 * (6.0 * x ** 2 + 3.0 * x + 1.0)


def _apply_lj_envelope(edgewise: torch.Tensor, dist_dict: dict, envelope: Optional[float]) -> torch.Tensor:
    """Multiply edgewise pair energies by lj_cutoff_envelope ending at the pair list's own
    cutoff, or return them untouched when envelope is None (the default)."""
    if envelope is None:
        return edgewise
    if isinstance(envelope, bool) or not isinstance(envelope, (int, float)):
        raise TypeError(f'lj envelope must be a width in Angstrom (int or float) or None, got {envelope!r}')
    width = float(envelope)
    if 'cutoff' not in dist_dict:
        raise ValueError('lj envelope requested but the distance dict carries no `cutoff`; build it with '
                         'get_intermolecular_dists_dict (construct_radial_graph), which records the radius')
    r_off = float(dist_dict['cutoff'])
    if not 0.0 < width <= r_off:
        raise ValueError(f'lj envelope width must be in (0, cutoff = {r_off}] Angstrom, got {width}')
    return edgewise * lj_cutoff_envelope(dist_dict['intermolecular_dist'], r_off, width)


def qlj_analysis(vdw_radii: torch.Tensor,
                 dist_dict: dict,
                 num_graphs: int,
                 envelope: Optional[float] = None,
                 ):
    """Per-graph LJ energy with a quadratic wall, summed over intermolecular edges.

    Pair energies come from ``quadratic_edgewise_lj_energy``, are optionally
    multiplied by ``lj_cutoff_envelope`` and are summed per graph over
    ``'intermolecular_dist_batch'``.

    Parameters
    ----------
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    dist_dict : dict
        Distance dict from ``get_intermolecular_dists_dict``; reads
        ``'intermolecular_dist'``, ``'intermolecular_dist_atoms'``,
        ``'intermolecular_dist_batch'``, and ``'cutoff'`` when `envelope` is
        set.
    num_graphs : int
        Output length.
    envelope : float, optional
        Width (Angstrom) of the smooth switch to zero at the pair-list cutoff
        ``dist_dict['cutoff']``; None (default) leaves pair energies
        unswitched.

    Returns
    -------
    torch.Tensor
        [num_graphs] summed pair energies, in units of epsilon.

    Raises
    ------
    TypeError
        If `envelope` is a bool or not an int or float.
    ValueError
        If `envelope` is set and `dist_dict` has no ``'cutoff'``, or the
        width is not in (0, cutoff].
    """
    batch = dist_dict['intermolecular_dist_batch']
    edgewise_lj_pot = _apply_lj_envelope(quadratic_edgewise_lj_energy(vdw_radii, dist_dict), dist_dict, envelope)
    molwise_lj_pot = scatter(edgewise_lj_pot, batch, reduce='sum', dim_size=num_graphs)

    return molwise_lj_pot

def elj_analysis(vdw_radii: torch.Tensor,
                 dist_dict: dict,
                 num_graphs: int,
                 stiffness: float = 2.5,
                 envelope: Optional[float] = None):
    """Per-graph eLJ energy: LJ with an exponential wall, summed over intermolecular edges.

    Pair energies come from ``exponential_edgewise_lj_energy`` with
    ``k_factor = stiffness``, are optionally multiplied by
    ``lj_cutoff_envelope`` and are summed per graph over
    ``'intermolecular_dist_batch'``.

    Parameters
    ----------
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    dist_dict : dict
        Distance dict from ``get_intermolecular_dists_dict``; reads
        ``'intermolecular_dist'``, ``'intermolecular_dist_atoms'``,
        ``'intermolecular_dist_batch'``, and ``'cutoff'`` when `envelope` is
        set.
    num_graphs : int
        Output length.
    stiffness : float, optional
        Wall stiffness, passed as ``k_factor``. Default 2.5.
    envelope : float, optional
        Width (Angstrom) of the smooth switch to zero at the pair-list cutoff
        ``dist_dict['cutoff']``; None (default) leaves pair energies
        unswitched.

    Returns
    -------
    torch.Tensor
        [num_graphs] summed pair energies, in units of epsilon.

    Raises
    ------
    TypeError
        If `envelope` is a bool or not an int or float.
    ValueError
        If `envelope` is set and `dist_dict` has no ``'cutoff'``, or the
        width is not in (0, cutoff].

    Notes
    -----
    No factor of 1/2 is applied. With a dict from
    ``get_intermolecular_dists_dict``, each inside atom's intermolecular
    neighbours within the cutoff (as defined by ``build_radial_graph``)
    appear once per inside atom, so the sum runs over (inside atom,
    neighbour atom) pairs and a pair of inside atoms in different molecules
    is counted twice.
    """
    batch = dist_dict['intermolecular_dist_batch']
    edgewise_lj_pot = _apply_lj_envelope(exponential_edgewise_lj_energy(vdw_radii, dist_dict, stiffness),
                                         dist_dict, envelope)
    molwise_lj_pot = scatter(edgewise_lj_pot, batch, reduce='sum', dim_size=num_graphs)

    return molwise_lj_pot


def lj_analysis(vdw_radii: torch.Tensor,
                dist_dict: dict,
                num_graphs: int,
                envelope: Optional[float] = None,
                ):
    """Per-graph 12-6 LJ energy, summed over intermolecular edges.

    Pair energies come from ``compute_lj_edgewise``, are optionally
    multiplied by ``lj_cutoff_envelope`` and are summed per graph over
    ``'intermolecular_dist_batch'``.

    Parameters
    ----------
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    dist_dict : dict
        Distance dict from ``get_intermolecular_dists_dict``; reads
        ``'intermolecular_dist'``, ``'intermolecular_dist_atoms'``,
        ``'intermolecular_dist_batch'``, and ``'cutoff'`` when `envelope` is
        set.
    num_graphs : int
        Output length.
    envelope : float, optional
        Width (Angstrom) of the smooth switch to zero at the pair-list cutoff
        ``dist_dict['cutoff']``; None (default) leaves pair energies
        unswitched.

    Returns
    -------
    torch.Tensor
        [num_graphs] summed pair energies, in units of epsilon.

    Raises
    ------
    TypeError
        If `envelope` is a bool or not an int or float.
    ValueError
        If `envelope` is set and `dist_dict` has no ``'cutoff'``, or the
        width is not in (0, cutoff].
    """
    batch = dist_dict['intermolecular_dist_batch']
    edgewise_lj_pot = _apply_lj_envelope(compute_lj_edgewise(dist_dict, vdw_radii), dist_dict, envelope)
    molwise_lj_pot = scatter(edgewise_lj_pot, batch, reduce='sum', dim_size=num_graphs)

    return molwise_lj_pot


def vdW_analysis(vdw_radii: torch.Tensor,
                 dist_dict: dict,
                 num_graphs: int,
                 reduce: str = 'sum'
                 ):
    """Per-graph reduction of vdW overlaps over intermolecular edges.

    Edge overlaps come from ``compute_vdW_overlap`` and are reduced per graph
    over ``'intermolecular_dist_batch'``.

    Parameters
    ----------
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.
    dist_dict : dict
        Reads ``'intermolecular_dist'``, ``'intermolecular_dist_atoms'`` and
        ``'intermolecular_dist_batch'``.
    num_graphs : int
        Output length.
    reduce : str, optional
        Reduction passed to ``torch_scatter.scatter``. Default 'sum'.

    Returns
    -------
    torch.Tensor
        [num_graphs] reduced overlaps, in the units of the distances.
    """
    batch = dist_dict['intermolecular_dist_batch']
    vdw_overlap = compute_vdW_overlap(dist_dict, vdw_radii)
    molwise_vdw_overlap = scatter(vdw_overlap, batch, reduce=reduce, dim_size=num_graphs)

    return molwise_vdw_overlap


def compute_lj_edgewise(dist_dict, vdw_radii):
    """Edgewise 12-6 Lennard-Jones energy with sigma = vdW radius sum and epsilon = 1.

    Each edge gets ``4 * ((s / d)**12 - (s / d)**6)``, with s the sum of the
    two atoms' radii. Non-finite results are replaced: NaN -> 0,
    +inf -> 1e20, -inf -> -1e-20. An edge with ``d = 0`` and ``s > 0``
    evaluates to inf - inf = NaN and so contributes 0.

    Parameters
    ----------
    dist_dict : dict
        Reads ``'intermolecular_dist'`` and ``'intermolecular_dist_atoms'``.
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.

    Returns
    -------
    torch.Tensor
        [num_inter_edges] pair energies, in units of epsilon.
    """
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']

    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]

    # uniform lennard jones potential
    sigma_r6 = torch.pow(radii_sums / dists, 6)
    sigma_r12 = torch.pow(sigma_r6, 2)
    lj_pot = torch.nan_to_num(
        4 * 1 * (sigma_r12 - sigma_r6),
        nan=0.0, posinf=1e20, neginf=-1e-20
    )
    return lj_pot


def compute_vdW_overlap(dist_dict, vdw_radii):
    """Edgewise vdW overlap ``relu(s - d)``, with s the sum of the two atoms' radii.

    Parameters
    ----------
    dist_dict : dict
        Reads ``'intermolecular_dist'`` and ``'intermolecular_dist_atoms'``.
    vdw_radii : torch.Tensor
        Van der Waals radius per atomic number.

    Returns
    -------
    torch.Tensor
        [num_inter_edges] overlaps, positive where ``d < s`` and 0 elsewhere.
    """
    dists = dist_dict['intermolecular_dist']
    elements = dist_dict['intermolecular_dist_atoms']
    # only punish positives (meaning overlaps)
    atom_radii = [vdw_radii[elements[0]], vdw_radii[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]

    overlap = F.relu(radii_sums - dists)
    return overlap
