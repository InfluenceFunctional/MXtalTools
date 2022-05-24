import numba as nb
import numpy as np

'''
1) find rotation matrix between 2 vectors, nice!
2) rotate a point cloud (r) for a given rotation matrix, useful!
3) rotate a point cloud (v) for a given axis (k) and angle of rotation

'''


@nb.jit(nopython=True)
def rotation_matrix_from_vectors(vec1, vec2):
    ''' Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    '''
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    Imat = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rotation_matrix = Imat + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


@nb.jit(nopython=True)
def euler_rotation(Rmat, r):
    n_atoms = len(r)

    v = np.zeros((n_atoms, 3), dtype=np.float_)
    for at in range(n_atoms):
        v[at, 0] = Rmat[0, 0] * r[at, 0] + Rmat[0, 1] * r[at, 1] + Rmat[0, 2] * r[at, 2]
        v[at, 1] = Rmat[1, 0] * r[at, 0] + Rmat[1, 1] * r[at, 1] + Rmat[1, 2] * r[at, 2]
        v[at, 2] = Rmat[2, 0] * r[at, 0] + Rmat[2, 1] * r[at, 1] + Rmat[2, 2] * r[at, 2]

    return v


@nb.jit(nopython=True)
def rodriguez_rotation(k, v, angle):
    angle *= np.pi / 180.0

    n_atoms = len(v)

    r = np.zeros((n_atoms, 3), dtype=np.float_)
    for at in range(n_atoms):
        k_dot_v = k[0] * v[at, 0] + k[1] * v[at, 1] + k[2] * v[at, 2]

        r[at, 0] = v[at, 0] * np.cos(angle) + (k[1] * v[at, 2] - k[2] * v[at, 1]) * np.sin(angle) + k[0] * k_dot_v * (1.0 - np.cos(angle))
        r[at, 1] = v[at, 1] * np.cos(angle) + (k[2] * v[at, 0] - k[0] * v[at, 2]) * np.sin(angle) + k[1] * k_dot_v * (1.0 - np.cos(angle))
        r[at, 2] = v[at, 2] * np.cos(angle) + (k[0] * v[at, 1] - k[1] * v[at, 0]) * np.sin(angle) + k[2] * k_dot_v * (1.0 - np.cos(angle))

    return r
