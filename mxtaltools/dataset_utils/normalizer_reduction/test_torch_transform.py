"""
Exercises the ACTUAL torch code in normalizer_reduction.transform_aunit_params (not the
numpy stand-in) against literal atom-by-atom transformation, using rotvec2rotmat /
rotmat2rotvec copied verbatim from mxtaltools/common/geometry_utils.py (these two
have no non-torch deps, unlike the rest of that module which pulls in torch_scatter /
torch_geometric -- copied rather than pip-installing the whole stack just to test
two self-contained functions).
"""
import torch
import sys
sys.path.insert(0, '/home/claude')

# ---- verbatim from mxtaltools/common/geometry_utils.py:1649-1745 ----
def rotvec2rotmat(mol_rotation: torch.tensor, basis='cartesian'):
    r = torch.linalg.norm(mol_rotation, dim=1)
    unit_vector = mol_rotation / r[:, None]
    K = torch.stack((
        torch.stack((torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 2], unit_vector[:, 1]), dim=1),
        torch.stack((unit_vector[:, 2], torch.zeros_like(unit_vector[:, 0]), -unit_vector[:, 0]), dim=1),
        torch.stack((-unit_vector[:, 1], unit_vector[:, 0], torch.zeros_like(unit_vector[:, 0])), dim=1)
    ), dim=1)
    identity_batch = torch.eye(3, device=r.device, dtype=torch.float32)[None, :, :].tile(len(r), 1, 1)
    applied_rotation_list = identity_batch + torch.sin(r[:, None, None]) * K + (1 - torch.cos(r[:, None, None])) * (K @ K)
    return applied_rotation_list


def rotmat2rotvec(rotation_matrix_list, warn_on_bad_determinant=True):
    direction_vector_list = torch.stack([
        rotation_matrix_list[:, 2, 1] - rotation_matrix_list[:, 1, 2],
        rotation_matrix_list[:, 0, 2] - rotation_matrix_list[:, 2, 0],
        rotation_matrix_list[:, 1, 0] - rotation_matrix_list[:, 0, 1]],
    ).T
    trace = torch.einsum('nii->n', rotation_matrix_list)
    r_arg = (trace - 1) / 2
    r = torch.arccos(r_arg)
    bad_inds = torch.any(torch.stack([r_arg.abs() >= 1,
                                      torch.isnan(r),
                                      direction_vector_list.sum(1) == 0,
                                      torch.isnan(direction_vector_list).sum(dim=1) > 0]
                                     ).T, dim=1)
    direction_vector_list[bad_inds, :] = torch.ones_like(direction_vector_list[bad_inds, :])
    r[bad_inds] = torch.pi
    rotvecs = direction_vector_list / (direction_vector_list.norm(dim=1, keepdim=True).clamp(min=1e-8)) * r[:, None]
    return rotvecs
# ---- end verbatim ----


def transform_aunit_params(centroid_frac, orientation, handedness, T_fc, T_cf, R_frac, t_frac, wrap=True):
    """copied from normalizer_reduction.py, with the mxtaltools import swapped for the local copies above"""
    centroid_frac = centroid_frac[:, :3]
    orientation = orientation[:, :3]
    n = centroid_frac.shape[0]
    device, dtype = centroid_frac.device, centroid_frac.dtype

    if R_frac.ndim == 2:
        R_frac = R_frac[None].expand(n, -1, -1)
    if t_frac.ndim == 1:
        t_frac = t_frac[None].expand(n, -1)
    R_frac = R_frac.to(device=device, dtype=dtype)
    t_frac = t_frac.to(device=device, dtype=dtype)

    new_centroid = torch.einsum('nij,nj->ni', R_frac, centroid_frac) + t_frac
    if wrap:
        new_centroid = new_centroid - torch.floor(new_centroid)

    R_cart = T_fc @ R_frac @ T_cf
    det = torch.linalg.det(R_cart)

    h_old = handedness.reshape(-1).to(dtype)
    h_new = det * h_old

    R_orient_old = rotvec2rotmat(orientation)
    diag_det = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    diag_det[:, 0, 0] = det
    diag_det[:, 1, 1] = 1.
    diag_det[:, 2, 2] = 1.
    R_orient_new = R_cart @ R_orient_old @ diag_det
    orientation_new = rotmat2rotvec(R_orient_new)

    return new_centroid, orientation_new, h_new


# also need align_mol_batch_to_standard_axes / get_aunit_positions equivalents to
# check round-trip against literal positions -- copy the same logic, torch this time
def compute_Ip_handedness(Ip):
    if Ip.ndim == 2:
        return torch.sign(torch.mul(Ip[0], torch.cross(Ip[1], Ip[2], dim=0)).sum()).float()
    return torch.sign(torch.mul(Ip[:, 0], torch.cross(Ip[:, 1], Ip[:, 2], dim=1)).sum(1))


def principal_axes(coords):
    c = coords - coords.mean(0)
    cov = c.T @ c
    w, v = torch.linalg.eigh(cov)
    order = torch.argsort(-w)
    Ip = v[:, order].T
    dists = torch.linalg.norm(c, dim=1)
    direction = c[torch.argmax(dists)]
    direction = direction / torch.linalg.norm(direction)
    signs = torch.sign(Ip @ direction)
    signs[signs == 0] = 1.0
    return Ip * signs[:, None]


def align_to_standard(ref_coords, handedness):
    Ip_ref = principal_axes(ref_coords)
    eye = torch.eye(3, dtype=ref_coords.dtype)
    eye[0, 0] = handedness
    A = eye @ Ip_ref
    return (A @ (ref_coords - ref_coords.mean(0)).T).T


def get_world_pos(ref_coords, rotvec, handedness, centroid_cart):
    aligned = align_to_standard(ref_coords, handedness)
    R = rotvec2rotmat(rotvec[None])[0]
    return (R @ aligned.T).T + centroid_cart


torch.manual_seed(7)
SYM_OPS_14 = [
    torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
    torch.tensor([[-1., 0., 0., 0.], [0., 1., 0., 0.5], [0., 0., -1., 0.5], [0., 0., 0., 1.]]),
    torch.tensor([[-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]]),
    torch.tensor([[1., 0., 0., 0.], [0., -1., 0., 0.5], [0., 0., 1., 0.5], [0., 0., 0., 1.]]),
]
NAMES = ["identity", "21 screw", "inversion", "c-glide"]

worst = 0.0
n_checked = 0
for trial in range(100):
    ref_coords = torch.randn(10, 3, dtype=torch.float64)
    rotvec_old = torch.randn(3, dtype=torch.float64)
    rotvec_old = rotvec_old / rotvec_old.norm() * (0.05 + torch.rand(1).item() * (torch.pi - 0.1))
    handedness_old = torch.tensor(-1.0 if trial % 2 == 0 else 1.0)
    a, b, c = (4 + 8 * torch.rand(3)).tolist()
    beta = torch.deg2rad(torch.tensor(95 + 30 * torch.rand(1).item()))
    T_fc = torch.tensor([[a, 0, c * torch.cos(beta).item()],
                         [0, b, 0],
                         [0, 0, c * torch.sin(beta).item()]], dtype=torch.float64)
    T_cf = torch.linalg.inv(T_fc)
    centroid_old_frac = 0.05 + 0.9 * torch.rand(3, dtype=torch.float64)
    centroid_old_cart = T_fc @ centroid_old_frac

    world_old = get_world_pos(ref_coords, rotvec_old, handedness_old, centroid_old_cart)

    for op_idx, op in enumerate(SYM_OPS_14):
        op = op.to(torch.float64)
        R_frac, t_frac = op[:3, :3], op[:3, 3]
        R_cart = T_fc @ R_frac @ T_cf
        t_cart = T_fc @ t_frac

        world_new_gt = (R_cart @ world_old.T).T + t_cart

        c_new, o_new, h_new = transform_aunit_params(
            centroid_old_frac[None], rotvec_old[None], handedness_old[None],
            T_fc[None], T_cf[None], R_frac, t_frac, wrap=False,
        )
        centroid_new_cart = T_fc @ c_new[0]
        world_new_pred = get_world_pos(ref_coords, o_new[0], h_new[0], centroid_new_cart)

        err = (world_new_pred - world_new_gt).abs().max().item()
        worst = max(worst, err)
        n_checked += 1
        assert err < 1e-8, f"trial {trial} op {NAMES[op_idx]}: err {err}"

print(f"checked {n_checked} (trial x symop) combos through the ACTUAL torch "
      f"transform_aunit_params code path (normalizer_reduction.py, copy-pasted here "
      f"verbatim). worst err: {worst:.3e}")
print("PASSED.")
