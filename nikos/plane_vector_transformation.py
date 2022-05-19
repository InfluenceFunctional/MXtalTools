import numba as nb
import numpy as np

'''
something like, moving around vectors?
'''

@nb.jit(nopython=True)
def plane_vector_transformation(opt, p, vec, ang):
    if opt == 'f_to_c':
        a = vec[0]
        b = vec[1]
        c = vec[2]

        alpha = ang[0] * np.pi / 180.0
        beta = ang[1] * np.pi / 180.0
        gamma = ang[2] * np.pi / 180.0

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)

        sin_gamma = np.sin(gamma)

        omega = a * b * c * np.sqrt(1.0 - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 + 2.0 * cos_alpha * cos_beta * cos_gamma)
        f = (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        k1 = p[0] * omega / a
        k2 = p[1] * omega / b / sin_gamma - p[0] * omega * cos_gamma / a / sin_gamma
        k3 = p[2] * a * b * sin_gamma - p[1] * a * c * f + p[0] * b * c * (f * cos_gamma - cos_beta * sin_gamma)

        k = np.array([k1, k2, k3], dtype=np.float_)

        norm = np.sqrt(np.dot(k, k))

        k /= norm
    elif opt == 'c_to_f':
        a = vec[0]
        b = vec[1]
        c = vec[2]

        alpha = ang[0] * np.pi / 180.0
        beta = ang[1] * np.pi / 180.0
        gamma = ang[2] * np.pi / 180.0

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)

        sin_gamma = np.sin(gamma)

        omega = a * b * c * np.sqrt(1.0 - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 + 2.0 * cos_alpha * cos_beta * cos_gamma)
        f = (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        k1 = p[0] * a / omega
        k2 = p[0] * b * cos_gamma / omega + p[1] * b * sin_gamma / omega
        k3 = p[0] * c * cos_beta / omega + p[1] * f * c / omega + p[2] / a / b / sin_gamma

        k = np.array([k1, k2, k3], dtype=np.float_)

        norm = np.sqrt(np.dot(k, k))

        k /= norm

    return k
