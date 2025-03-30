from pathlib import Path

import torch

from mxtaltools.common.training_utils import load_molecule_autoencoder
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

test_smiles = [
    "CCC#CCC#N",
    "CCC(O)(C=O)C#N",
    "CC(O)CC=O",
    "OCC1=C(O)C=CN1",
    "COCC1(O)CC1O",
    "CNC(=O)CCCO",
    "CCC(=O)C(C)C",
    "COCC(=O)N(C)C",
    "NC1=C(NC=C1)C#C",
    "COC1(C)COC1C"
]

if __name__ == '__main__':
    """configs"""
    device = 'cpu'
    checkpoint = Path(r"../models/autoencoder.pt")

    """load some molecules"""
    base_molData = MolData()
    num_mols = len(test_smiles)
    mols = [base_molData.from_smiles(test_smiles[ind],
                                     compute_partial_charges=True,
                                     minimize=True,
                                     protonate=True,
                                     ) for ind in range(num_mols)]
    mols = [mol for mol in mols if mol is not None]  # sometimes the embedding fails
    mol_batch = collate_data_list(mols).to(device)
    mol_batch.recenter_molecules()

    with torch.no_grad():
        """load model"""
        model = load_molecule_autoencoder(
            checkpoint,
            device
        )

        """get encoding & decoding"""
        vector_encoding = model.encode(mol_batch.clone())
        scalar_encoding = model.scalarizer(vector_encoding)

        """check the quality of the embedding for this molecule"""
        reconstruction_loss, rmsd, matched_molecule = model.check_embedding_quality(mol_batch, visualize=True)
