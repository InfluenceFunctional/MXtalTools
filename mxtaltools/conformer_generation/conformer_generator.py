"""
functionality for
1. embed SMILES as 3D conformer
2. modulate over rotatable bonds
3. modulate bond lengths
"""

import copy
from rdkit.Chem.rdchem import BondType as BT

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation as R
import networkx as nx
from torch_geometric.utils import to_networkx
from mxtaltools.dataset_management.CrystalData import CrystalData

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


def modify_conformer(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]
        # if not mask_rotate[idx_edge, u] or mask_rotate[idx_edge, v]:
        #     mask_rotate[idx_edge] = ~mask_rotate[idx_edge]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards. NOTE: DIFFERENT FROM THE PAPER!
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec)  # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos


smiles = ['CC1(C)C2COC=NC12', 'CC1(O)CC(=N)NC1=O', 'CCOCC(O)C1CN1', 'OCC12CC(N1)C2C#N', 'C1NC11C2NC2C2NC12',
          'CCC1=C(C)C(C)=CO1', 'CN=C1C=NOC(F)=C1', 'NC12CC(O)C1C2C#N', 'COC(C)C1=NC=CN1', 'O=C1OCCOC2CC12',
          'N#CCOC1CC=CC1', '[NH3+]CC1=C[N-]N=NC1=O', 'CC1=CCC(O)C(O)C1', 'CN1C=CC(C=O)=C1N', 'CCC(C)(CC)CCO',
          'CCCCNC(=N)C#N', 'CC1C=C2CC3C1C23C', 'CC1C2CC(O)(CO)C12', 'CNC(=O)CC1CCC1', 'CNC1=C(N)OC(N)=N1',
          'CC1(CO)CC(O)C1', 'CC(CCC#C)CC#N', 'CCC1(C)CCCCC1', 'C1NC11CC2NC2C=C1', 'CC1CC2CC3C(O1)C23',
          'O=CC1NC23CN(C2)C13', 'CC1=NOC=C1CO', 'CC1(CC1)C1OCC1=O', 'OCCC1CC(O)CO1', 'CC12CC(O)(C1)C2C#C',
          'OCC1C2C3OC2C13', 'CC(C)C12CC(C1)C2=O', 'NC1=NC2COC(C2)O1', 'OCC#CC#CC#C', 'CCC1OCCC=C1C',
          'O=C(C#C)N1C=CC=C1', 'COC=NCC(C)=O', 'CN1N=C(N)N=C1C', 'CCC1C2CC(CO)C12', 'COC1C2OC(C)C12C', 'N=C1COCC(=O)O1',
          'CN=CN(C)C(N)=O', 'COC(C)C1COC1', 'COCC(O)C(=O)C#C', 'CC1=COC=C([O-])C1=[NH2+]', 'C1C2C3C4C5C(C2N35)N14',
          'CC1OC1(C)CC1CC1', 'CN1CC1(C#C)C1CN1', 'COCC1NC(C)C1=O', 'O=COC(C#N)C1CC1', 'COCC1CCC2OC12',
          'CC12CC1C(C)(C2)C#N', 'OCC12CCC3C1OC23', 'NC1C(O)C2NC12C#N', 'CCC(C)C1=NCCO1', 'NC1=NCCC1(N)C#N',
          'O=C1C=CC2C=CCN12', 'N=C1OC=NC=C1', 'C1OCC2COCC1N2', 'O=C1CCC1OCC#N', 'COC1=CON=CC1=O', 'N#CC12NCC3C1CC23',
          'C1C2CCC3CC2C1O3', 'CN1N=C(C)C(O)=C1N', 'CC1CCC2=C1C=NO2', 'OCCC1NC1CO', 'O=C1CC2(CCO2)C=C1',
          'OC1CCC2OC(=N)C12', 'CCOCC1CC2NC12', 'CC12CC3OC(C1O)C23', 'CC(C)C1COC(C)O1', 'COC12CC1CC2C=O',
          'CC([NH3+])(CC#C)C([O-])=O', 'N#CC#CC1=NNN=C1', 'CC1=CC2C(C#C)N2C1', 'CC(=O)C1=CC=NO1', 'CC1=C(N)N=NN1',
          'CC1N2CC1(CC#C)C2', 'COC1(CCCC1)C#C', 'CC1OC(C=O)C1O', 'O=C1C2C3OC1C=CC23', 'NC1=C(OC=O)C=NN1',
          'CC1C(CC#C)OC1=N', 'OC(C#C)C1C2COC12', 'C#CCC1CCCOC1', 'CC1CC2(C)CC(C2)O1', 'CN1N=NNC(=O)C1=N',
          'CC12CC3C(C1O)N23', 'COC1=C(C)N=CN1', 'O=CCC12CC1C1CN21', 'CCCC1CC(C)(C)O1', 'OC1=NOC=C1C#N', 'COCC(=O)C(C)C',
          'CC12CC1OCOC2', 'CC12CC3OC1(CO)C23', 'O=C1NC=NCCO1', 'CC1(C#N)N2CC1(O)C2', 'C1CN(C1)C1=COC=C1',
          'CC(C#C)C1=CC=CO1', 'CC12CC(O)(C1)C1NC21']

if __name__ == '__main__':
    protonate = False
    numConfs = 1
    for smile in smiles:
        "generate rd mol"
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            continue
        if protonate:
            mol = Chem.AddHs(mol)
        else:
            mol = Chem.RemoveHs(mol)

        "3D embedding"
        AllChem.EmbedMultipleConfs(mol, numConfs=numConfs)

        conf = mol.GetConformer()
        pos = conf.GetPositions()
        z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        row, col, edge_type = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)

        "Adjust conformational DoF"

        G = to_networkx(CrystalData(x=torch.LongTensor(z), pos=torch.Tensor(pos), edge_index=edge_index))
        mask_edges, mask_rotate = get_rotatable_edges(G, edge_index)
        if mask_edges.sum() > 0:
            mols = []
            for conf in range(5):
                torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=mask_edges.sum())
                pos = modify_conformer(pos, edge_index.T[mask_edges], mask_rotate, torsion_updates, as_numpy=True)

                from ase import Atoms
                from ase.visualize import view

                mols.append(Atoms(numbers=z, positions=pos))
            view(mols)

    aa = 0
zzzzz