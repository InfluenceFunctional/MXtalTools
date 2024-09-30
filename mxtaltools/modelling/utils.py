import os
from distutils.dir_util import copy_tree
from shutil import copy

import numpy as np

from mxtaltools.models.utils import get_n_config


def get_model_sizes(models_dict: dict):
    num_params_dict = {model_name + "_num_params": get_n_config(model) for model_name, model in
                       models_dict.items()}
    [print(
        f'{model_name} {num_params_dict[model_name] / 1e6:.3f} million or {int(num_params_dict[model_name])} parameters')
        for model_name in num_params_dict.keys()]
    return num_params_dict


def copy_source_to_workdir(working_directory, yaml_path, config):
    os.mkdir(working_directory + '/source')
    copy_tree("mxtaltools/common", working_directory + "/source/common")
    copy_tree("mxtaltools/crystal_building", working_directory + "/source/crystal_building")
    copy_tree("mxtaltools/dataset_management", working_directory + "/source/dataset_management")
    copy_tree("mxtaltools/models", working_directory + "/source/models")
    copy_tree("mxtaltools/reporting", working_directory + "/source/reporting")
    copy("mxtaltools/modeller.py", working_directory + "/source")
    copy("main.py", working_directory + "/source")
    np.save(working_directory + '/run_config', config)
    os.chdir(working_directory)  # move to working dir
    copy(yaml_path, os.getcwd())  # copy full config for reference
