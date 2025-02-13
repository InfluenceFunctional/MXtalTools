"""
functionality for
1. embed SMILES as 3D conformer
2. modulate over rotatable bonds
3. modulate bond lengths
"""
import multiprocessing as mp
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
from torch_geometric.utils import to_networkx

from mxtaltools.common.utils import chunkify
from mxtaltools.dataset_utils.CrystalData import CrystalData
from mxtaltools.dataset_utils.construction.featurization_utils import get_partial_charges

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


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


def modify_conformer(pos,
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
                         cutoff: float = 1):
    num_min_iters = 0
    d1 = cdist(pos, pos) + np.eye(len(pos)) * 2
    nonbonded_dists = d1.flatten()[~adjacency_matrix.flatten()]
    converged = np.amin(nonbonded_dists) > cutoff
    while not converged and num_min_iters < max_num_iters:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=iters_per_cycle)
        pos = conf.GetPositions()
        d1 = cdist(pos, pos) + np.eye(len(pos)) * 2
        nonbonded_dists = d1.flatten()[~adjacency_matrix.flatten()]
        converged = np.amin(nonbonded_dists) < cutoff
        num_min_iters += 1

    return pos, converged


def extract_mol_info(mol, conf):
    pos = conf.GetPositions()
    z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    adjacency_matrix = np.zeros((len(z), len(z)), dtype=bool)
    adjacency_matrix[edge_index[0, :], edge_index[1, :]] = True
    G = to_networkx(CrystalData(x=torch.LongTensor(z), pos=torch.Tensor(pos), edge_index=edge_index))

    return mol, conf, pos, z, edge_index, adjacency_matrix, G


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
    mol, conf, pos, z, edge_index, adjacency_matrix, G = extract_mol_info(mol, conf)

    coords, types = [], []
    "Adjust conformational DoF"
    mask_edges, mask_rotate = get_rotatable_edges(G, edge_index)

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

    "apply rotations"
    if len(mask_rotate) > len(nontrivial_rotations):  # rotate bonds
        for c_ind in range(max_rotamers_per_samples):
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=len(mask_rotate))
            pos = modify_conformer(pos, edge_index.T[mask_edges], mask_rotate, torsion_updates, as_numpy=True)

            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, pos[i])

            pos, converged = minimal_minimization(mol, conf, pos, adjacency_matrix)
            if not converged:
                break

            coords.append(1 * pos)
            types.append(z)
    else:
        pos, converged = minimal_minimization(mol, conf, pos, adjacency_matrix)
        if not converged:
            return False, False, False, False
        coords.append(pos)
        types.append(z)

    if len(coords) == 0:
        return False, False, False, False

    if do_partial_charges:
        charges = get_partial_charges(mol)
    else:
        charges = np.zeros_like(types)
    return coords, types, mask_rotate, mask_edges, charges


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


if __name__ == '__main__':
    smiles = ['CC1(C)C2COC=NC12', 'CC1(O)CC(=N)NC1=O', 'CCOCC(O)C1CN1', 'OCC12CC(N1)C2C#N', 'C1NC11C2NC2C2NC12',
              'CCC1=C(C)C(C)=CO1', 'CN=C1C=NOC(F)=C1', 'NC12CC(O)C1C2C#N', 'COC(C)C1=NC=CN1', 'O=C1OCCOC2CC12',
              'N#CCOC1CC=CC1', '[NH3+]CC1=C[N-]N=NC1=O', 'CC1=CCC(O)C(O)C1', 'CN1C=CC(C=O)=C1N', 'CCC(C)(CC)CCO',
              'CCCCNC(=N)C#N', 'CC1C=C2CC3C1C23C', 'CC1C2CC(O)(CO)C12', 'CNC(=O)CC1CCC1', 'CNC1=C(N)OC(N)=N1',
              'CC1(CO)CC(O)C1', 'CC(CCC#C)CC#N', 'CCC1(C)CCCCC1', 'C1NC11CC2NC2C=C1', 'CC1CC2CC3C(O1)C23',
              'O=CC1NC23CN(C2)C13', 'CC1=NOC=C1CO', 'CC1(CC1)C1OCC1=O', 'OCCC1CC(O)CO1', 'CC12CC(O)(C1)C2C#C',
              'OCC1C2C3OC2C13', 'CC(C)C12CC(C1)C2=O', 'NC1=NC2COC(C2)O1', 'OCC#CC#CC#C', 'CCC1OCCC=C1C',
              'O=C(C#C)N1C=CC=C1', 'COC=NCC(C)=O', 'CN1N=C(N)N=C1C', 'CCC1C2CC(CO)C12', 'COC1C2OC(C)C12C',
              'N=C1COCC(=O)O1',
              'CN=CN(C)C(N)=O', 'COC(C)C1COC1', 'COCC(O)C(=O)C#C', 'CC1=COC=C([O-])C1=[NH2+]', 'C1C2C3C4C5C(C2N35)N14',
              'CC1OC1(C)CC1CC1', 'CN1CC1(C#C)C1CN1', 'COCC1NC(C)C1=O', 'O=COC(C#N)C1CC1', 'COCC1CCC2OC12',
              'CC12CC1C(C)(C2)C#N', 'OCC12CCC3C1OC23', 'NC1C(O)C2NC12C#N', 'CCC(C)C1=NCCO1', 'NC1=NCCC1(N)C#N',
              'O=C1C=CC2C=CCN12', 'N=C1OC=NC=C1', 'C1OCC2COCC1N2', 'O=C1CCC1OCC#N', 'COC1=CON=CC1=O',
              'N#CC12NCC3C1CC23',
              'C1C2CCC3CC2C1O3', 'CN1N=C(C)C(O)=C1N', 'CC1CCC2=C1C=NO2', 'OCCC1NC1CO', 'O=C1CC2(CCO2)C=C1',
              'OC1CCC2OC(=N)C12', 'CCOCC1CC2NC12', 'CC12CC3OC(C1O)C23', 'CC(C)C1COC(C)O1', 'COC12CC1CC2C=O',
              'CC([NH3+])(CC#C)C([O-])=O', 'N#CC#CC1=NNN=C1', 'CC1=CC2C(C#C)N2C1', 'CC(=O)C1=CC=NO1', 'CC1=C(N)N=NN1',
              'CC1N2CC1(CC#C)C2', 'COC1(CCCC1)C#C', 'CC1OC(C=O)C1O', 'O=C1C2C3OC1C=CC23', 'NC1=C(OC=O)C=NN1',
              'CC1C(CC#C)OC1=N', 'OC(C#C)C1C2COC12', 'C#CCC1CCCOC1', 'CC1CC2(C)CC(C2)O1', 'CN1N=NNC(=O)C1=N',
              'CC12CC3C(C1O)N23', 'COC1=C(C)N=CN1', 'O=CCC12CC1C1CN21', 'CCCC1CC(C)(C)O1', 'OC1=NOC=C1C#N',
              'COCC(=O)C(C)C',
              'CC12CC1OCOC2', 'CC12CC3OC1(CO)C23', 'O=C1NC=NCCO1', 'CC1(C#N)N2CC1(O)C2', 'C1CN(C1)C1=COC=C1',
              'CC(C#C)C1=CC=CO1', 'CC12CC(O)(C1)C1NC21']

    pool = mp.Pool(mp.cpu_count() - 1)

    chunks = chunkify(smiles, int(np.ceil(len(smiles) / 1000)))

    for chunk_ind, chunk in enumerate(chunks):
        pool.apply_async(generate_random_conformers_from_smiles_list, args=(chunk, dump_path, chunk_ind))

    pool.close()
    pool.join()

''' check results
from ase import Atoms
from ase.visualize import view

mols = []

for ind in range(30):
    mols.append(Atoms(positions=dataset['coordinates'][ind], numbers=dataset['atom_types'][ind]))
view(mols)

'''
