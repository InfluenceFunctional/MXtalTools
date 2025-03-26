import os
import pickle

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from mxtaltools.dataset_utils.construction.featurization_utils import get_partial_charges

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def smiles2conformer(allow_methyl_rotations,
                     compute_partial_charges,
                     max_pare_iters,
                     minimize,
                     pare_to_size,
                     protonate,
                     scramble_dihedrals,
                     smiles):
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
    """Adjust conformational DoF"""
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
