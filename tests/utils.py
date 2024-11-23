import os
import torch

from mxtaltools.common.config_processing import process_main_config
from mxtaltools.modeller import Modeller


def train_model(config_path):
    source_dir = os.getcwd()
    os.chdir(source_dir)
    user_path = r'/configs/Users/mikem.yaml'
    config = process_main_config(user_yaml_path=user_path, main_yaml_path=source_dir + config_path)
    modeller = Modeller(config)
    modeller.fit_models()


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
