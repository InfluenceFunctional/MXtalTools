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
    checkpoint = Path(r"../checkpoints/autoencoder.pt")

    """
    First we load some molecules, and ensure they are each centered on the origin.
    Here we are having RDKit build and minimize molecules generated 
    from some SMILES codes. This is automated in our MolData class.
    One can also directly input the atom types and coordinates.
    """
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

    """load pre-trained model"""
    model = load_molecule_autoencoder(
        checkpoint,
        device
    )

    with torch.no_grad():
        """get vector and scalar embeddings"""
        vector_encoding = model.encode(mol_batch.clone())
        scalar_encoding = model.scalarizer(vector_encoding)

        """
        Check the quality of the embedding for this batch of
        molecules, and visualize the reconstruction
        """
        reconstruction_loss, rmsd, matched_molecule = (
            model.check_embedding_quality(mol_batch, visualize=True))
