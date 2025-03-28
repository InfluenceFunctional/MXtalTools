from pathlib import Path

import torch

from mxtaltools.common.training_utils import load_molecule_scalar_regressor, enable_dropout
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

test_smiles = [
    r"CC(F)(F)C1CC1NCCS(C)(=O)=O",
    r"CNC(=S)N1CCC(C)(C)C1",
    r"O=C1CC(CNC(=O)C2CC2Br)C1",
    r"C=C=CCOC(=O)C(CC)(CC)CC",
    r"CC1(CNC(=O)C2CCNC2)N=N1",
    r"O=C(OCCn1cncn1)C(Br)CBr",
    r"CSCC(C)(C)C(=O)NOCC(F)F",
    r"C#CCC(C)C(=O)N1CC=C(Br)CC1",
    r"C#CCC1(NCCO)CCC1",
    r"NOCCCNc1cncc(Cl)n1",
    r"Clc1ccc2c(Br)ccnc2c1",
    r"CC(C)CC(C)(C)C(=O)NCC(F)F",
    r"CNC(=O)COC1CN(C(=O)NC)C1",
    r"O=C(OCc1nncs1)c1cccs1",
    r"CCSCCCNC(=O)CO",
    r"C=CC1CC1(C)C(=O)NNCC(F)F",
    r"FC1(F)CCC(SCCC2CCC2)C1",
    r"CCC(C)(C)NS(=O)(=O)CI",
    r"CCC(C)CN(CC)C(=O)C(C)(Br)Br",
    r"O=C(CN1CC=CC1)N1CC(=CBr)C1",
    r"CC(F)CCN(C)Cc1sccc1Cl",
    r"CCCCN(C)N(C)CCOCC"
]

if __name__ == '__main__':
    """configs"""
    device = 'cpu'
    checkpoint = Path(r"C:\Users\mikem\PycharmProjects\Python_Codes\MXtalTools\models\cp_regressor.pt")
    num_samples = 50

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

    """load model"""
    model = load_molecule_scalar_regressor(
        checkpoint,
        device
    )

    """predict crystal packing coefficient - single-point"""
    packing_coeff_pred = model(mol_batch).flatten() * model.target_std + model.target_mean
    aunit_volume_pred = mol_batch.mol_volume / packing_coeff_pred  # A^3
    density_pred = mol_batch.mass / aunit_volume_pred * 1.6654  # g/cm^3

    """get prediction with uncertainty via resampling with dropout"""
    predictions = []
    model = enable_dropout(model)
    for _ in range(num_samples):
        predictions.append(model(mol_batch).flatten() * model.target_std + model.target_mean)

    predictions = torch.stack(predictions)
    packing_coeff_mean = predictions.mean(0)
    packing_coeff_std = predictions.std(0)
