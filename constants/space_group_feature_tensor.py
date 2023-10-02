import torch
from constants.space_group_info import LATTICE_TYPE, SYM_OPS

"""
generate a feature tensor encoding the symmetry information of a given sample
SG one-hot : crystal one-hot : symmetry multiplicity
"""
feature_length = 230 + 6 + 1
SG_FEATURE_TENSOR = torch.zeros((231, feature_length))

unique_lattices = list(set(LATTICE_TYPE.values()))

for i in range(1, 231):  # first row stays empty - indexing with true SG index
    SG_FEATURE_TENSOR[i, i - 1] = 1  # SG one-hot
    lattice_ind = unique_lattices.index(LATTICE_TYPE[i])
    SG_FEATURE_TENSOR[i, 230 + lattice_ind] = 1
    SG_FEATURE_TENSOR[i, -1] = len(SYM_OPS[i])
