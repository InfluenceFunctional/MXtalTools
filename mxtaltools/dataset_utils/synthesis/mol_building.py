import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from mxtaltools.conformer_generation.conformer_generator import embed_mol, extract_mol_info, get_rotatable_edges, \
    scramble_dihedral_angles, minimal_minimization
from mxtaltools.dataset_utils.construction.featurization_utils import get_partial_charges


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
