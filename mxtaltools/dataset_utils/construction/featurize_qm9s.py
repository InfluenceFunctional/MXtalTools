import os
import warnings
from pathlib import Path

import torch
import tqdm
from rdkit import Chem as Chem

from mxtaltools.common.utils import chunkify
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII, ATOMIC_SYMBOLS, \
    ELECTRONEGATIVITY, PERIOD, GROUP
from mxtaltools.dataset_utils.data_classes import MolData

HDonorSmarts = Chem.MolFromSmarts(
    '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
HAcceptorSmarts = Chem.MolFromSmarts(
    '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
    '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
    '$([nH0,o,s;+0])]')

vdw_radii_dict = VDW_RADII
element_symbols_dict = ATOMIC_SYMBOLS
electronegativity_dict = ELECTRONEGATIVITY
period_dict = PERIOD
group_dict = GROUP
atomic_weight_dict = ATOM_WEIGHTS

warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error

"""
Convert qm9s Data objects to chunks of MolData objects - just a format change
"""

if __name__ == '__main__':
    chunks_path = r'D:\crystal_datasets\QM9s_chunks'

    if not os.path.exists(chunks_path):
        os.mkdir(chunks_path)

    qm9s_list = torch.load(r'D:\crystal_datasets\QM9s\qm9s.pt', weights_only=False)

    chunks = chunkify(qm9s_list, 100)
    for ind, chunk in enumerate(tqdm.tqdm(chunks)):
        data_list = []
        for elem in chunk:
            data = MolData(
                z=elem.z,
                pos=elem.pos,
                smiles=elem.smile,
                identifier=elem.smile,
                x=torch.zeros(len(elem.z), dtype=torch.float32),  # we won't be packing these, so no need fo featurize
                # there's a utility in the featurization utils if you want to do this
                skip_mol_analysis=False,
            )
            for key in elem.keys():  # copy over all attributes
                setattr(data, key, elem[key])

            data_list.append(data)

        torch.save(data_list,  Path(chunks_path).joinpath(f"chunk_{ind}.pkl"))



