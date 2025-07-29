"""
file for extracting smiles from our pure QM9 dataset
and combining it with smiles from the zinc dataset

Output should be list of valid, unique SMILES, and separate evaluation set of QM9 molecules
"""
import gzip
import os
from pathlib import Path
from random import shuffle, seed
from tqdm import tqdm
import multiprocessing as mp

import rdkit.Chem as Chem
import torch


def get_qm9_like_csd():
    qm9_like_path = 'D:/crystal_datasets/qm9_like_csd_crystals.pt'
    if not os.path.exists(qm9_like_path):
        pp = "D:/crystal_datasets/CSD_dataset.pt"
        dataset = torch.load(pp)

        qm9_like = []
        for ind, elem in enumerate(dataset):
            if torch.sum(elem.z > 1) < 10:
                if torch.sum(elem.z > 1) > 6:
                    if set(elem.z.tolist()).issubset([1, 6, 7, 8, 9]):
                        qm9_like.append(elem)

        torch.save(qm9_like, qm9_like_path)
    else:
        qm9_like = torch.load(qm9_like_path)

    return qm9_like


def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


if __name__ == '__main__':
    eval_path = Path(r"D:\crystal_datasets\csd_free_qm9.pt")
    qm9_path = Path(r"D:\crystal_datasets\qm9_dataset.pt")

    qm9_like_mols = get_qm9_like_csd()
    dataset = torch.load(qm9_path)

    qm9_set = set(filter(None, [canonicalize(data.smiles) for data in dataset]))
    csd_set = set(filter(None, [canonicalize(data.smiles) for data in qm9_like_mols]))
    overlap = qm9_set & csd_set
    print(f"{len(overlap)} molecules in overlap.")

    good_qm9 = []
    for elem in dataset:
        if elem not in overlap:
            good_qm9.append(elem)
    torch.save(good_qm9, Path(r"D:\crystal_datasets\csd_free_qm9_dataset.pt"))
    torch.save(good_qm9[:10000], Path(r"D:\crystal_datasets\test_csd_free_qm9_dataset.pt"))

    aa = 1
