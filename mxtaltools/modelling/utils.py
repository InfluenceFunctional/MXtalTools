import os
from distutils.dir_util import copy_tree
from shutil import copy

import numpy as np

from mxtaltools.common.training_utils import get_n_config


def get_model_sizes(models_dict: dict):
    num_params_dict = {model_name + "_num_params": get_n_config(model) for model_name, model in
                       models_dict.items()}
    [print(
        f'{model_name} {num_params_dict[model_name] / 1e6:.3f} million or {int(num_params_dict[model_name])} parameters')
        for model_name in num_params_dict.keys()]
    return num_params_dict

