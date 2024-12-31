"""
file for extracting smiles from my QM9 dataset
and combining it with smiles from the zinc dataset
"""
import gzip
import os
from pathlib import Path
from random import shuffle, seed
from tqdm import tqdm
import multiprocessing as mp

import rdkit.Chem as Chem
import torch


def process_zinc_dir(filename, eval_smiles_list):
    new_smiles_list = []
    with gzip.open(filename, 'r') as f:
        for line in f:
            smiles = line.split()[0]
            try:
                mol = Chem.MolFromSmiles(smiles)
                canon_smiles = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=False,
                    canonical=True,
                    allHsExplicit=False,
                )
                if mol.GetNumHeavyAtoms() <= 9:  # if it might conceivably overlap
                    if canon_smiles not in eval_smiles_list:
                        new_smiles_list.append(canon_smiles)
                else:  # if not, add it for free
                    new_smiles_list.append(canon_smiles)

            except:
                pass
    print(f"Processed {filename}")
    return new_smiles_list


if __name__ == '__main__':
    eval_path = Path(r"D:\crystal_datasets\eval_qm9_dataset.pt")
    qm9_path = Path(r"D:\crystal_datasets\qm9_dataset.pt")
    zinc_path = Path(r"D:\crystal_datasets\zinc22\archive")
    output_file_name = Path(r"D:\crystal_datasets\zinc22\H_all\qm9_zinc_dataset.txt")

    seed(0)
    dataset = torch.load(qm9_path)

    '''pull out evaluation qm9 data'''
    shuffle(dataset)
    eval_dataset = dataset[:10000]
    torch.save(eval_dataset, eval_path)

    print('''take the rest of the qm9 smiles and canonicalize them''')
    dataset = dataset[len(dataset) // 5:]
    smiles_list = []
    for data in tqdm(dataset):
        smiles = data.smiles
        try:
            canon_smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles),
                isomericSmiles=False,
                canonical=True,
                allHsExplicit=False,
            )
            smiles_list.append(canon_smiles)
        except:
            pass

    print('''canonicalize the eval smiles too''')
    eval_smiles_list = []
    for data in tqdm(eval_dataset):
        smiles = data.smiles
        try:
            canon_smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles),
                isomericSmiles=False,
                canonical=True,
                allHsExplicit=False,
            )
            eval_smiles_list.append(canon_smiles)
        except:
            pass

    print('''get all the zinc smiles''')
    os.chdir(zinc_path)
    h_dirs = os.listdir(zinc_path)
    h_dirs = [elem for elem in h_dirs if elem[0] == 'H']
    smiles_paths = []
    for dir in h_dirs:
        files = os.listdir(dir)
        smiles_paths.extend(
            [os.path.join(Path(dir), Path(file)) for file in files]
        )

    '''canonicalize zinc, if they are absent from the eval'''
    outs = []
    pool = mp.Pool(6)
    for filename in tqdm(smiles_paths):
        if 'QM9' not in filename:
            outs.append(
                pool.apply_async(process_zinc_dir, (filename, eval_smiles_list))
            )

    pool.close()
    pool.join()
    outs = [out.get() for out in outs]
    for out in outs:
        smiles_list.extend(out)

    '''save unique zinc+qm9 smiles'''
    smiles_list = list(set(smiles_list))
    shuffle(smiles_list)

    with open(output_file_name, 'w') as f:
        for smile in smiles_list:
            f.write(smile + '\n')

    aa = 1
