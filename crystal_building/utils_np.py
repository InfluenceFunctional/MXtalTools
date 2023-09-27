import numpy as np


def fractional_transform_np(coords, T_mat):
    if coords.ndim == 2:
        return np.einsum('nj,ij->ni', coords, T_mat)
    elif coords.ndim == 3:
        return np.einsum('nmj,ij->nmi', coords, T_mat)


def find_coord_in_box_np(coords, box, epsilon=0):
    # which of the given coords is inside the specified box, with option for a little leeway
    return np.where((coords[:, 0] <= (box[0] + epsilon)) * (coords[:, 1] <= (box[1] + epsilon) * (coords[:, 2] <= (box[2] + epsilon))))[0]
