"""
Validates the orientation/handedness composition rule against ALL FOUR of the real
P21/c (space group 14) symmetry operations from mxtaltools.constants.space_group_info,
across many random (conformer, orientation, handedness, cell) trials.

This is a from-scratch numpy reimplementation of the exact formulas in
mxtaltools/common/geometry_utils.py and mxtaltools/crystal_building/utils.py
(rotvec2rotmat, rotmat2rotvec, extract_rotmat, compute_Ip_handedness, the
furthest-atom-direction sign convention in batch_molecule_principal_axes_torch /
correct_Ip_directions, align_mol_batch_to_standard_axes, get_aunit_positions,
extract_aunit_orientation) -- see validate_composition.py for the single-case,
heavily-commented walkthrough. This file just widens that same check.

Claim under test: given a Cartesian operation (R_cart, t_cart) applied to ALL
atoms of an already-posed molecule,
    handedness_new = det(R_cart) * handedness_old
    R_orient_new   = R_cart @ R_orient_old @ diag(det(R_cart), 1, 1)
    centroid_new   = R_cart @ centroid_old + t_cart      (unwrapped, see note below)
where R_orient = rotvec2rotmat(aunit_orientation), correctly reproduces literally
transforming every atom, for BOTH proper and improper (and specifically
non-central improper, i.e. not just pure inversion) operations.
"""
import numpy as np

# ---- exact SYM_OPS[14] from mxtaltools.constants.space_group_info, fetched live ----
SYM_OPS_14 = [
    np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
    np.array([[-1., 0., 0., 0.], [0., 1., 0., 0.5], [0., 0., -1., 0.5], [0., 0., 0., 1.]]),
    np.array([[-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]]),
    np.array([[1., 0., 0., 0.], [0., -1., 0., 0.5], [0., 0., 1., 0.5], [0., 0., 0., 1.]]),
]
OP_NAMES = ["identity", "21 screw (proper)", "inversion (improper, central)", "c-glide (improper, non-central)"]


def rotvec2rotmat(v):
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def rotmat2rotvec(R):
    tr = np.trace(R)
    theta = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / np.linalg.norm(axis) * theta


def compute_Ip_handedness(Ip):
    return np.sign(np.dot(Ip[0], np.cross(Ip[1], Ip[2])))


def principal_axes(coords):
    c = coords - coords.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    Ip = v[:, np.argsort(-w)].T
    direction = c[np.argmax(np.linalg.norm(c, axis=1))]
    direction /= np.linalg.norm(direction)
    signs = np.sign(Ip @ direction)
    signs[signs == 0] = 1.0
    return Ip * signs[:, None]


def align_to_standard(ref_coords, handedness):
    Ip_ref = principal_axes(ref_coords)
    eye = np.eye(3)
    eye[0, 0] = handedness
    A = eye @ Ip_ref
    return (A @ (ref_coords - ref_coords.mean(0)).T).T


def get_world_pos(ref_coords, rotvec, handedness, centroid_cart):
    aligned = align_to_standard(ref_coords, handedness)
    return (rotvec2rotmat(rotvec) @ aligned.T).T + centroid_cart


def extract_params(world_pos):
    Ip_world = principal_axes(world_pos)
    h = compute_Ip_handedness(Ip_world)
    eye = np.eye(3)
    eye[0, 0] = h
    R = (eye @ Ip_world).T
    return rotmat2rotvec(R), h


def random_monoclinic_T_fc(rng):
    a, b, c = rng.uniform(4, 12, size=3)
    beta = np.radians(rng.uniform(95, 125))
    return np.array([[a, 0, c * np.cos(beta)], [0, b, 0], [0, 0, c * np.sin(beta)]])


rng = np.random.default_rng(42)
n_trials = 200
worst_err = 0.0
n_checked = 0

for trial in range(n_trials):
    ref_coords = rng.normal(size=(rng.integers(6, 14), 3))
    rotvec_old = rng.normal(size=3)
    rotvec_old = rotvec_old / np.linalg.norm(rotvec_old) * rng.uniform(0.05, np.pi - 0.05)
    handedness_old = float(rng.choice([-1., 1.]))
    T_fc = random_monoclinic_T_fc(rng)
    T_cf = np.linalg.inv(T_fc)
    centroid_old_frac = rng.uniform(0.05, 0.95, size=3)
    centroid_old_cart = T_fc @ centroid_old_frac

    world_old = get_world_pos(ref_coords, rotvec_old, handedness_old, centroid_old_cart)

    for op_idx, op in enumerate(SYM_OPS_14):
        R_frac, t_frac = op[:3, :3], op[:3, 3]
        R_cart = T_fc @ R_frac @ T_cf
        t_cart = T_fc @ t_frac
        det_R = np.linalg.det(R_cart)

        # ground truth: transform every atom directly
        world_new_gt = (R_cart @ world_old.T).T + t_cart

        # formula
        handedness_new = det_R * handedness_old
        R_orient_new = R_cart @ rotvec2rotmat(rotvec_old) @ np.diag([det_R, 1., 1.])
        rotvec_new = rotmat2rotvec(R_orient_new)
        centroid_new_cart = R_cart @ centroid_old_cart + t_cart  # unwrapped, see docstring

        world_new_pred = get_world_pos(ref_coords, rotvec_new, handedness_new, centroid_new_cart)

        err = np.abs(world_new_pred - world_new_gt).max()
        worst_err = max(worst_err, err)
        n_checked += 1
        assert err < 1e-7, f"trial {trial}, op '{OP_NAMES[op_idx]}': err={err}"

print(f"checked {n_checked} (trial x symop) combinations across all 4 SYM_OPS[14], "
      f"{n_trials} random (conformer, orientation, handedness, monoclinic cell) draws each.")
print(f"worst-case max atomic-position error: {worst_err:.3e}")
print("PASSED for identity, the proper 21-screw, and both improper ops "
      "(central inversion AND non-central c-glide).")
