"""RDKit-based construction of single-molecule conformers from SMILES: embedding, optional
skeleton paring and random torsions about rotatable bonds, clash relief by short MMFF
minimization, and Gasteiger partial charges."""
import os
import pickle

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, AllChem as AllChem
from rdkit.Chem.rdchem import BondType as BT
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def smiles2conformer(allow_methyl_rotations,
                     compute_partial_charges,
                     max_pare_iters,
                     minimize,
                     pare_to_size,
                     protonate,
                     scramble_dihedrals,
                     smiles):
    """
    Build one 3D conformer from a SMILES string with RDKit.

    Steps: :func:`embed_mol`; if ``pare_to_size`` is not None, :func:`get_rotatable_edges` and
    :func:`pare_molecule_skeleton` (with ``deprotonate=not protonate``), which re-embeds a pared
    molecule; if ``scramble_dihedrals``, :func:`scramble_dihedral_angles` followed by
    :func:`minimal_minimization`; if ``minimize``, a full ``AllChem.MMFFOptimizeMolecule``;
    finally partial charges.

    Parameters
    ----------
    allow_methyl_rotations : bool
        Passed to :func:`scramble_dihedral_angles` as ``allow_simple_hydrogen_rotations``.
    compute_partial_charges : bool
        If True, charges come from :func:`get_partial_charges`; otherwise they are zeros.
    max_pare_iters : int
        Maximum number of fragment removals in :func:`pare_molecule_skeleton`.
    minimize : bool
        If True, ``AllChem.MMFFOptimizeMolecule`` is run on the final molecule and the
        positions are re-read.
    pare_to_size : int or None
        Maximum number of heavy atoms (atomic number > 1); None disables paring.
    protonate : bool
        If True, explicit hydrogens are added; otherwise hydrogens are removed.
    scramble_dihedrals : bool
        Enables the torsion-scrambling branch; see Notes.
    smiles : str
        Input SMILES.

    Returns
    -------
    tuple or None
        ``(charges, pos, z)``, or None if SMILES parsing, embedding, paring or clash relief
        fails. ``charges`` is a float64 numpy array [num_atoms] when computed, otherwise a
        float32 torch tensor of zeros [num_atoms]; ``pos`` is a numpy array [num_atoms, 3] of
        RDKit conformer coordinates (Angstrom); ``z`` is an integer numpy array [num_atoms] of
        atomic numbers.

    Notes
    -----
    The ``scramble_dihedrals`` branch unpacks the three values returned by
    :func:`scramble_dihedral_angles` into two names, so it raises ``ValueError`` when reached.
    """
    # todo add catch errors when adding or removing Hs
    mol = embed_mol(smiles, protonate)
    if not mol:  # failed embed
        return None
    try:
        conf = mol.GetConformer()
    except ValueError:
        return None
    if pare_to_size is not None:
        # get rotatable bond information
        pos, z, edge_index, adjacency_matrix, nx_graph = (
            extract_mol_info(
                mol, conf,
                do_adjacency_analysis=True))

        mask_edges, mask_rotate = (
            get_rotatable_edges(nx_graph, edge_index))

        mol, conf, flag = pare_molecule_skeleton(conf, mask_rotate,
                                                 max_pare_iters, mol,
                                                 pare_to_size, pos, z,
                                                 deprotonate=not protonate)

        if flag:  # paring or embedding failed
            return None
    if scramble_dihedrals:
        pos, z, edge_index, adjacency_matrix, nx_graph = (
            extract_mol_info(
                mol, conf,
                do_adjacency_analysis=True))

        mask_edges, mask_rotate = (
            get_rotatable_edges(nx_graph, edge_index))

        pos, mol, = (
            scramble_dihedral_angles(
                allow_methyl_rotations, conf,
                edge_index, mask_rotate, mask_edges,
                mol, pos, z))

        # eliminate severe clashes
        pos, converged, num_min_iters = (
            minimal_minimization(
                mol, conf, pos, adjacency_matrix))

        if not converged or len(pos) == 0:
            return None
    else:
        pos, z = extract_mol_info(
            mol, conf,
            do_adjacency_analysis=False)
    if minimize:  # full minimization
        AllChem.MMFFOptimizeMolecule(mol)
        pos = conf.GetPositions()
    if compute_partial_charges:
        charges = get_partial_charges(mol)
    else:
        charges = torch.zeros(len(z), dtype=torch.float32)
    return charges, pos, z


def pare_molecule_skeleton(conf, mask_rotate, max_pare_iters, mol, pare_to_size, pos, z,
                           deprotonate: bool = False):
    """
    Remove rotatable-bond fragments until at most ``pare_to_size`` heavy atoms remain, then re-embed.

    While the number of heavy atoms (``z > 1``) exceeds ``pare_to_size``, rows of
    ``mask_rotate`` remain, and fewer than ``max_pare_iters`` removals have been made, one row
    (fragment) is drawn with probability proportional to ``exp(-number of heavy atoms in the
    fragment)`` using the global numpy RNG, and its atoms are removed. The removed atoms are
    then deleted from a copy of ``mol``; hydrogens are removed and re-added (and removed again
    if ``deprotonate``), and the molecule is re-embedded with ``AllChem.EmbedMolecule``.

    Parameters
    ----------
    conf : rdkit.Chem.Conformer
        Returned unchanged only if the re-embedded molecule has no conformer.
    mask_rotate : numpy.ndarray
        Bool [num_rotatable_bonds, num_atoms] from :func:`get_rotatable_edges`.
    max_pare_iters : int
        Maximum number of fragment removals.
    mol : rdkit.Chem.Mol
        Source molecule; not modified.
    pare_to_size : int
        Target maximum number of heavy atoms.
    pos : numpy.ndarray
        [num_atoms, 3]; only its length affects the result.
    z : numpy.ndarray
        [num_atoms] atomic numbers.
    deprotonate : bool, optional
        If True, hydrogens are removed after being re-added. Default False.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The pared, re-embedded molecule.
    conf : rdkit.Chem.Conformer
        Its conformer, or the input ``conf`` if it has none.
    flag : bool
        True if the heavy-atom target was not reached or the re-embedded molecule has no
        conformer.
    """
    # use rotatable bonds as fragmentation sites to pare the molecule down to an acceptable size
    mol_num_atoms = len(pos)
    atoms_kept = np.arange(mol_num_atoms)
    iter = 0
    flag = False
    while np.sum(z > 1) > pare_to_size and len(mask_rotate) > 0 and iter < max_pare_iters:
        # how many heavy atoms in each fragment
        fragment_size = np.sum(mask_rotate[:, z > 1], axis=1)
        # sample which fragment to pare, weighted to smaller sizes
        fragment_to_pare = \
            np.random.choice(len(fragment_size), 1,
                             p=np.exp(-fragment_size) / np.sum(np.exp(-fragment_size))
                             )[0]
        # effect the paring
        atoms_to_pare = mask_rotate[fragment_to_pare, :]
        pos, z = pos[~atoms_to_pare], z[~atoms_to_pare]
        mask_rotate = np.delete(mask_rotate, fragment_to_pare, axis=0)
        mask_rotate = mask_rotate[:, ~atoms_to_pare]
        atoms_kept = atoms_kept[~atoms_to_pare]
        iter += 1

    if np.sum(z > 1) > pare_to_size:
        flag = True

    atoms_pared = [ind for ind in np.arange(mol_num_atoms) if ind not in atoms_kept]

    # update rdkit molecule object
    m2 = Chem.RWMol(mol)
    m2.BeginBatchEdit()
    for ind in atoms_pared:
        m2.RemoveAtom(int(ind))
    m2.CommitBatchEdit()

    # re-embed molecule and extract info
    mol = Chem.AddHs(Chem.RemoveHs(m2))
    if deprotonate:
        mol = Chem.RemoveHs(mol)
    AllChem.EmbedMolecule(mol)
    if mol is None:
        flag = True
    try:
        conf = mol.GetConformer()
    except ValueError:
        flag = True

    return mol, conf, flag


def get_rotatable_edges(G, edge_index):
    """
    Find rotatable bonds and, for each, the atoms on the side that a torsion moves.

    ``edge_index`` must list each bond as consecutive directed pairs ``(u, v), (v, u)``; only
    ``u`` matching the second atom of the following edge is asserted. For each bond, the bond is removed from an undirected copy of ``G``; if this
    disconnects the graph and the smaller component has more than one atom, the bond is
    rotatable and that component's atoms are assigned to whichever of its two directed edges
    has its second atom in the component.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed molecular graph with node ids ``0`` to ``num_atoms - 1`` and one edge per
        column of ``edge_index``, as returned by :func:`extract_mol_info`.
    edge_index : torch.Tensor
        [2, num_edges] directed edges on CPU.

    Returns
    -------
    mask_edges : numpy.ndarray
        Bool [num_edges]; True for the directed edge that carries a rotating fragment.
    mask_rotate : numpy.ndarray
        Bool [num_rotatable_bonds, num_atoms]; row k marks the atoms moved about the k-th True
        entry of ``mask_edges``.
    """
    to_rotate = []
    edges = edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i + 1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


def apply_torsions(pos,
                   edge_index,
                   mask_rotate,
                   torsion_updates,
                   as_numpy=False):
    """
    Rotate groups of atoms about bond axes by given angles.

    Edges are processed in order. For each edge ``k`` with a nonzero ``torsion_updates[k]``,
    with ``(u, v) = edge_index[k]`` (asserted: ``mask_rotate[k, u]`` False and
    ``mask_rotate[k, v]`` True), the atoms in ``mask_rotate[k]`` are rotated about the axis
    through ``pos[v]`` along ``pos[u] - pos[v]`` by ``torsion_updates[k]`` radians, using a
    ``scipy`` rotation vector.

    Parameters
    ----------
    pos : numpy.ndarray or torch.Tensor
        [num_atoms, 3] positions. A numpy array is modified in place; a tensor is converted
        with ``.cpu().numpy()``, which shares memory with a CPU tensor.
    edge_index : torch.Tensor
        [num_rotatable_bonds, 2] rotatable edges, one per row.
    mask_rotate : numpy.ndarray
        Bool [num_rotatable_bonds, num_atoms] atoms moved about each edge.
    torsion_updates : sequence of float
        [num_rotatable_bonds] rotation angles in radians.
    as_numpy : bool, optional
        If True, the numpy array is returned; otherwise a float32 torch tensor. Default False.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        [num_atoms, 3] updated positions.
    """
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec)  # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


def minimal_minimization(mol, conf, pos, adjacency_matrix,
                         iters_per_cycle: int = 10,
                         max_num_iters: int = 10,
                         cutoff: float = 1.2):
    """
    Run short MMFF minimizations until no non-bonded atom pair is within ``cutoff``.

    Non-bonded pairs are those with ``adjacency_matrix`` False; self-pairs are included with
    their distance set to 2. While the smallest non-bonded distance is not greater than
    ``cutoff`` and fewer than ``max_num_iters`` cycles have run,
    ``AllChem.MMFFOptimizeMolecule(mol, maxIters=iters_per_cycle)`` is called and the
    positions are re-read from ``conf``.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Minimized in place.
    conf : rdkit.Chem.Conformer
        Conformer of ``mol`` from which positions are re-read.
    pos : numpy.ndarray
        [num_atoms, 3] initial positions (Angstrom).
    adjacency_matrix : numpy.ndarray
        Bool [num_atoms, num_atoms] bond adjacency.
    iters_per_cycle : int, optional
        MMFF iterations per cycle. Default 10.
    max_num_iters : int, optional
        Maximum number of cycles. Default 10.
    cutoff : float, optional
        Minimum allowed non-bonded distance (Angstrom). Default 1.2.

    Returns
    -------
    pos : numpy.ndarray
        [num_atoms, 3] final positions.
    converged : bool
        Whether the smallest non-bonded distance exceeds ``cutoff``.
    num_min_iters : int
        Number of minimization cycles run.
    """
    num_min_iters = 0
    d1 = cdist(pos, pos) + np.eye(len(pos)) * 2
    nonbonded_dists = d1.flatten()[~adjacency_matrix.flatten()]
    converged = np.amin(nonbonded_dists) > cutoff
    while not converged and num_min_iters < max_num_iters:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=iters_per_cycle)
        pos = conf.GetPositions()
        d1 = cdist(pos, pos) + np.eye(len(pos)) * 2
        nonbonded_dists = d1.flatten()[~adjacency_matrix.flatten()]
        converged = np.amin(nonbonded_dists) > cutoff
        num_min_iters += 1

    return pos, converged, num_min_iters


def extract_mol_info(mol,
                     conf,
                     do_adjacency_analysis: bool = True):
    """
    Read coordinates, atomic numbers and, optionally, bond connectivity from an RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule providing atoms and bonds.
    conf : rdkit.Chem.Conformer
        Conformer of ``mol`` providing the coordinates.
    do_adjacency_analysis : bool, optional
        If True, the bond graph is also built and returned. Default True.

    Returns
    -------
    pos : numpy.ndarray
        [num_atoms, 3] float64 coordinates from ``conf.GetPositions()`` (Angstrom).
    z : numpy.ndarray
        [num_atoms] integer atomic numbers.
    edge_index : torch.Tensor
        Only if ``do_adjacency_analysis``. int64 [2, 2 * num_bonds]; each bond appears as
        consecutive columns ``(start, end), (end, start)``.
    adjacency_matrix : numpy.ndarray
        Only if ``do_adjacency_analysis``. Symmetric bool [num_atoms, num_atoms].
    G : networkx.DiGraph
        Only if ``do_adjacency_analysis``. Directed graph of ``edge_index`` from
        ``torch_geometric.utils.to_networkx``, without node or edge attributes.

    Raises
    ------
    KeyError
        If ``do_adjacency_analysis`` and a bond is not single, double, triple or aromatic
        (the module-level ``bonds`` map).
    """
    pos = conf.GetPositions()
    z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    if do_adjacency_analysis:
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        adjacency_matrix = np.zeros((len(z), len(z)), dtype=bool)
        adjacency_matrix[edge_index[0, :], edge_index[1, :]] = True
        G = to_networkx(Data(x=torch.LongTensor(z), pos=torch.Tensor(pos), edge_index=edge_index))

        return pos, z, edge_index, adjacency_matrix, G
    else:
        return pos, z


def embed_mol(smile, protonate):
    """
    Parse a SMILES string and embed one 3D conformer with RDKit.

    Parameters
    ----------
    smile : str
        Input SMILES.
    protonate : bool
        If True, explicit hydrogens are added before embedding; otherwise hydrogens are
        removed.

    Returns
    -------
    rdkit.Chem.Mol or bool
        The molecule after ``AllChem.EmbedMolecule``, or False if the SMILES does not parse.
        The embedding return code is not checked; a failed embedding leaves the molecule
        without a conformer.
    """
    mol = Chem.MolFromSmiles(smile)
    if not mol:
        return False
    if protonate:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)
    AllChem.EmbedMolecule(mol)
    return mol


def generate_random_conformers_from_smiles(smile: str,
                                           protonate: bool = True,
                                           max_rotamers_per_samples: int = 10,
                                           allow_simple_hydrogen_rotations: bool = False,
                                           do_partial_charges: bool = False,
                                           ):
    """
    Embed a SMILES string and scramble its torsions; see Notes for the current behavior.

    Parameters
    ----------
    smile : str
        Input SMILES.
    protonate : bool, optional
        Passed to :func:`embed_mol`. Default True.
    max_rotamers_per_samples : int, optional
        Not used. Default 10.
    allow_simple_hydrogen_rotations : bool, optional
        Passed to :func:`scramble_dihedral_angles` in its ``edge_index`` position (see Notes).
        Default False.
    do_partial_charges : bool, optional
        If True, charges from :func:`get_partial_charges`; otherwise zeros like ``z``.
        Default False.

    Returns
    -------
    tuple
        ``(False, False, False, False)`` if parsing or embedding fails. The success path
        would return the five values ``(pos, z, mask_rotate, mask_edges, charges)``.

    Notes
    -----
    The call to :func:`scramble_dihedral_angles` passes arguments in an order that does not
    match that function's signature and unpacks four names from its three return values, so
    the function raises for any SMILES that embeds successfully.
    """
    mol = embed_mol(smile, protonate)
    if not mol:  # embedding failed
        return False, False, False, False
    try:
        conf = mol.GetConformer()
    except ValueError:
        return False, False, False, False
    pos, z, edge_index, adjacency_matrix, G = extract_mol_info(mol, conf)

    converged, pos, mask_edges, mask_rotate = (
        scramble_dihedral_angles(
            G, adjacency_matrix, allow_simple_hydrogen_rotations, conf,
            edge_index, mol,pos, z))

    if not converged:
        return False, False, False, False
    if len(pos) == 0:
        return False, False, False, False

    if do_partial_charges:
        charges = get_partial_charges(mol)
    else:
        charges = np.zeros_like(z)
    return pos, z, mask_rotate, mask_edges, charges


def scramble_dihedral_angles(allow_simple_hydrogen_rotations,
                             conf,
                             edge_index,
                             mask_rotate,
                             mask_edges,
                             mol,
                             pos,
                             z):
    """
    Apply uniform random torsions about rotatable bonds and write the result to the conformer.

    When ``allow_simple_hydrogen_rotations`` is False, rotations whose moved atoms, as sorted
    atomic numbers, are [1, 1, 1, 6], [1, 1, 6], [1, 6], [1, 1, 7], [1, 7] or [1, 8] (methyl,
    amino or hydroxyl groups) are classed as trivial; otherwise every rotation is classed as
    non-trivial. Torsions are applied only if at least one rotation is trivial, i.e. only when
    ``allow_simple_hydrogen_rotations`` is False and such a rotation exists; then every
    rotatable bond, trivial ones included, receives an angle drawn uniformly from
    [-pi, pi) with the global numpy RNG, via :func:`apply_torsions`, and the new positions are
    set on ``conf``.

    Parameters
    ----------
    allow_simple_hydrogen_rotations : bool
        See above.
    conf : rdkit.Chem.Conformer
        Conformer of ``mol``; its atom positions are overwritten when torsions are applied.
    edge_index : torch.Tensor
        [2, num_edges] directed edges.
    mask_rotate : numpy.ndarray
        Bool [num_rotatable_bonds, num_atoms] from :func:`get_rotatable_edges`.
    mask_edges : numpy.ndarray
        Bool [num_edges] from :func:`get_rotatable_edges`.
    mol : rdkit.Chem.Mol
        Molecule whose atom count sets the positions written to ``conf``.
    pos : numpy.ndarray
        [num_atoms, 3]; modified in place when torsions are applied.
    z : numpy.ndarray
        [num_atoms] atomic numbers.

    Returns
    -------
    pos : numpy.ndarray
        [num_atoms, 3] positions (the input array, updated if torsions were applied).
    mol : rdkit.Chem.Mol
        The input molecule.
    conf : rdkit.Chem.Conformer
        The input conformer.
    """
    "restrict trivial rotations"
    if not allow_simple_hydrogen_rotations:
        nontrivial_rotations = []
        for rot_ind, rotation_mask in enumerate(mask_rotate):
            atoms_to_rotate = sorted(z[rotation_mask])
            if not any([
                atoms_to_rotate == [1, 1, 1, 6],
                atoms_to_rotate == [1, 1, 6],
                atoms_to_rotate == [1, 6],

                atoms_to_rotate == [1, 1, 7],
                atoms_to_rotate == [1, 7],

                atoms_to_rotate == [1, 8]]
            ):  # rotating only a methyl, amino, or alcohol
                nontrivial_rotations.append(rot_ind)
        #
        # mask_rotate = mask_rotate[nontrivial_rotations]
        # mask_edge_inds = np.argwhere(mask_edges).flatten()
        # mask_edges *= False
        # mask_edges[mask_edge_inds[nontrivial_rotations]] = True
    else:
        nontrivial_rotations = np.arange(len(mask_rotate))

    "apply torsions"
    if len(mask_rotate) > len(nontrivial_rotations):  # rotate bonds
        torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=len(mask_rotate))
        pos = apply_torsions(pos, edge_index.T[mask_edges], mask_rotate, torsion_updates, as_numpy=True)

        # apply also to the conformer for downstream RDKit compatibility
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i])

    return pos, mol, conf


def generate_random_conformers_from_smiles_list(smiles, dump_path, chunk_ind):
    """
    Build and pickle a dataset of random conformers for a list of SMILES; see Notes for the current behavior.

    Each SMILES is passed to :func:`generate_random_conformers_from_smiles` (``protonate=True``,
    ``max_rotamers_per_samples=10``, ``allow_simple_hydrogen_rotations=False``) and results
    are collected into a dict with keys ``'coordinates'``, ``'atom_types'`` and ``'smiles'``,
    which is pickled to ``os.path.join(dump_path, f'{chunk_ind}.pkl')``.

    Parameters
    ----------
    smiles : iterable of str
        Input SMILES.
    dump_path : str
        Output directory.
    chunk_ind : int or str
        Output file stem.

    Notes
    -----
    For each SMILES, :func:`generate_random_conformers_from_smiles` either raises or returns
    four values that are unpacked here into two names, so any SMILES raises; only an empty
    ``smiles`` writes a file (with empty lists).
    """
    dataset = {
        'coordinates': [],
        'atom_types': [],
        'smiles': []
    }

    for smile in smiles:
        "generate rd mol"
        coords, types = generate_random_conformers_from_smiles(smile,
                                                               protonate=True,
                                                               max_rotamers_per_samples=10,
                                                               allow_simple_hydrogen_rotations=False)
        if not coords:
            continue

        dataset['coordinates'].extend(coords)
        dataset['atom_types'].extend(types)
        dataset['smiles'].extend([smile for _ in range(len(coords))])

    with open(os.path.join(dump_path, f'{chunk_ind}.pkl'), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_partial_charges(rd_mol):
    """
    Compute Gasteiger partial charges with RDKit.

    ``AllChem.ComputeGasteigerCharges(rd_mol)`` sets the ``'_GasteigerCharge'`` property on
    each atom of ``rd_mol`` (in place), which is then read back.

    Parameters
    ----------
    rd_mol : rdkit.Chem.Mol
        Molecule to charge.

    Returns
    -------
    numpy.ndarray
        float64 [num_atoms] charges, one per atom in ``rd_mol.GetAtoms()`` order (hydrogens
        have entries only if they are explicit atoms).
    """
    AllChem.ComputeGasteigerCharges(rd_mol)
    return np.array([float(atom.GetProp('_GasteigerCharge')) for atom in rd_mol.GetAtoms()])
