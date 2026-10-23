"""Per-space-group feature vectors, built at import from SYM_OPS and LATTICE_TYPE.

Attributes
----------
feature_length : int
    237 (230 + 6 + 1), the width of each feature row.
unique_lattices : list of str
    The six distinct LATTICE_TYPE values ('triclinic', 'monoclinic',
    'orthorhombic', 'tetragonal', 'hexagonal', 'cubic') in the iteration
    order of a Python ``set`` built from them. That order follows string
    hashing, which is randomised per interpreter process unless
    ``PYTHONHASHSEED`` is fixed, so the list order, and with it the column
    each crystal system occupies in SG_FEATURE_TENSOR, can differ between
    processes.
SG_FEATURE_TENSOR : torch.Tensor
    [231, 237] tensor created with torch's default device and floating dtype
    at import time (CPU and float32 unless changed beforehand), indexed
    directly by space-group number. Row 0 is all zeros. For space
    group i in 1-230, column i - 1 is 1 (space-group one-hot), column
    230 + unique_lattices.index(LATTICE_TYPE[i]) is 1 (crystal-system
    one-hot over columns 230-235), and column 236 holds len(SYM_OPS[i]), the
    unnormalised number of general-position operators in the conventional
    cell (1 to 192). All other entries are 0.
"""
import torch
from mxtaltools.constants.space_group_info import LATTICE_TYPE, SYM_OPS

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
