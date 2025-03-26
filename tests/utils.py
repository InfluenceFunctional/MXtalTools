import os
import torch

from mxtaltools.common.config_processing import process_main_config
# from mxtaltools.modeller import Modeller

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
#
# def train_model(config_path, test_user_path):
#     source_dir = os.getcwd()
#     os.chdir(source_dir)
#     user_path = test_user_path
#     config = process_main_config(user_yaml_path=user_path,
#                                  main_yaml_path=source_dir + config_path)
#     modeller = Modeller(config)
#     modeller.fit_models()


def check_tensor_similarity(t1, t2, eps=1e-4):
    """
    check that two tensors are sufficiently similar
    """
    diff = torch.mean(torch.abs(t1 - t2) / torch.abs(t2))
    assert diff < eps, f"Difference is too large {diff:.5f}"


def is_module_equivariant(v, rotation_matrix, module, batch=None, x=None, **kwargs):  # todo rewrite with all kwargs

    rotated_vector_batch = torch.einsum('ij, njk -> nik', rotation_matrix, v)
    if x is None:
        if batch is None:
            output = module(v,  **kwargs)
            output_from_rotated = module(rotated_vector_batch, **kwargs)
        else:
            output = module(v, batch=batch, **kwargs)
            output_from_rotated = module(rotated_vector_batch, batch=batch, **kwargs)
    else:
        if batch is None:
            output = module(x=x, v=v,  **kwargs)
            output_from_rotated = module(x=x, v=rotated_vector_batch,  **kwargs)
        else:
            output = module(x=x, v=v, batch=batch,  **kwargs)
            output_from_rotated = module(x=x, v=rotated_vector_batch, batch=batch,  **kwargs)

    if isinstance(output, tuple):
        output = output[1]
        output_from_rotated = output_from_rotated[1]

    rotated_output = torch.einsum('ij, njk -> nik', rotation_matrix, output)

    return rotated_output, output_from_rotated
