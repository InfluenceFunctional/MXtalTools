import torch
from torch.nn import functional as F

from mxtaltools.constants.space_group_info import SYM_OPS, POINT_GROUPS, LATTICE_TYPE, SPACE_GROUPS, LATTICE_TO_CODE

def make_lattice_code_lookup(lattice_type):
    """
    Build lookup tensor for crystallographic space-group indices.

    sg_ind is expected to be in [1, 230].
    Index 0 is unused and marked invalid.
    """

    lattice_code = torch.full((231,), -1, dtype=torch.long)

    if isinstance(lattice_type, dict):
        items = lattice_type.items()
    else:
        # If lattice_type is list-like with entries for SG 1..230.
        # Supports either length 231 with index 0 unused,
        # or length 230 with SG 1 at list index 0.
        if len(lattice_type) == 231:
            items = enumerate(lattice_type)
        elif len(lattice_type) == 230:
            items = ((i + 1, lattice) for i, lattice in enumerate(lattice_type))
        else:
            raise ValueError(
                f"Expected lattice_type length 230 or 231, got {len(lattice_type)}"
            )

    for sg_ind, lattice in items:
        sg_ind = int(sg_ind)

        if sg_ind == 0:
            continue

        if not (1 <= sg_ind <= 230):
            raise ValueError(f"Invalid space-group index {sg_ind}; expected [1, 230]")

        lattice = lattice.lower()

        if lattice not in LATTICE_TO_CODE:
            raise ValueError(f"{lattice!r} is not a valid crystal lattice")

        lattice_code[sg_ind] = LATTICE_TO_CODE[lattice]

    if (lattice_code[1:] < 0).any():
        missing = torch.where(lattice_code[1:] < 0)[0].add(1).tolist()
        raise ValueError(f"Missing lattice types for space groups: {missing}")

    return lattice_code
def init_sym_info():
    """
    Initialize dict containing symmetry info for crystals with standard settings and general positions.

    Returns
    -------
    sym_info : dict
    """
    sym_ops = SYM_OPS
    point_groups = POINT_GROUPS
    lattice_type = LATTICE_TYPE
    space_groups = SPACE_GROUPS

    lattice_code = make_lattice_code_lookup(lattice_type)

    sym_info = {
        "sym_ops": sym_ops,
        "point_groups": point_groups,
        "lattice_type": lattice_type,
        "lattice_code": lattice_code,
        "space_groups": space_groups,
    }

    return sym_info

def bounding_penalty(x, lower, upper, margin: float = 0.0):
    return (torch.relu(x - (upper - margin)) ** 2) + (torch.relu((lower + margin) - x) ** 2)


def tri_reduction_penalty(cell_lengths, cell_angles, margin):
    """triclinic cells reduction ruels"""
    eps = 1e-6
    bc_error = F.relu(cell_lengths[:, 1] / cell_lengths[:, 2] - (1 - margin)) ** 2  # c>b
    ab_error = F.relu(cell_lengths[:, 0] / cell_lengths[:, 1] - (1 - margin)) ** 2  # # b>a

    a, b, c = cell_lengths.unbind(dim=1)
    al, be, ga = cell_angles.unbind(dim=1)
    al_max_cos = b / 2 / c
    be_max_cos = a / 2 / c
    ga_max_cos = a / 2 / b

    alpha_error = bounding_penalty(al.cos() / al_max_cos.clamp(min=eps), -1, 1, margin=margin)
    beta_error = bounding_penalty(be.cos() / be_max_cos.clamp(min=eps), -1, 1, margin=margin)
    gamma_error = bounding_penalty(ga.cos() / ga_max_cos.clamp(min=eps), -1, 1, margin=margin)

    return bc_error + ab_error + alpha_error + beta_error + gamma_error


def mono_reduction_penalty(cell_lengths, cell_angles, margin):


    # enforces our reduction scheme
    a, b, c = cell_lengths.unbind(dim=1)
    al, be, ga = cell_angles.unbind(dim=1)
    be_min_cos = (- a / (c)).clamp(min=-1)
    be_max_cos = 0
    beta_error = bounding_penalty(be.cos(), be_min_cos, be_max_cos, margin=margin)

    # enforces the crystal system
    alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
    gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

    return beta_error + alpha_error + gamma_error


def ortho_reduction_penalty(cell_lengths, cell_angles, margin):
    # crystal system enforcement
    alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
    beta_error = (cell_angles[:, 1] - torch.pi / 2) ** 2
    gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

    # cell reduction enforcement
    # bc_error = F.relu(cell_lengths[:, 1] / cell_lengths[:, 2] - (1 - margin)) ** 2  # c>b
    # ab_error = F.relu(cell_lengths[:, 0] / cell_lengths[:, 1] - (1 - margin)) ** 2  # # b>a

    return alpha_error + beta_error + gamma_error  # + ab_error + bc_error


def tetra_reduction_penalty(cell_lengths, cell_angles, margin):
    a, b, c = cell_lengths.unbind(dim=-1)

    # reduction term
    # abc_error = F.relu(b / c - (1 - margin)) ** 2 + F.relu(a / c - (1 - margin)) ** 2

    # crystal system terms

    # enforce a=b
    ab_error = (a - b) ** 2
    # enforce right angles
    alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
    beta_error = (cell_angles[:, 1] - torch.pi / 2) ** 2
    gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

    return ab_error + alpha_error + beta_error + gamma_error  # + abc_error


def trig_reduction_penalty(cell_lengths, cell_angles, margin):
    a, b, c = cell_lengths.unbind(dim=-1)
    al, be, ga = cell_angles.unbind(dim=-1)

    # crystal system enforcement
    # a = b
    ab_error = (a - b) ** 2

    # alpha = beta = 90°
    alpha_error = (al - torch.pi / 2) ** 2
    beta_error = (be - torch.pi / 2) ** 2

    # gamma = 120°
    gamma_error = (ga - 2 * torch.pi / 3) ** 2

    return ab_error + alpha_error + beta_error + gamma_error


def hex_reduction_penalty(cell_lengths, cell_angles, margin):
    a, b, c = cell_lengths.unbind(dim=-1)
    al, be, ga = cell_angles.unbind(dim=-1)

    # crystal system enforcement
    ab_error = (a - b) ** 2
    alpha_error = (al - torch.pi / 2) ** 2
    beta_error = (be - torch.pi / 2) ** 2
    gamma_error = (ga - 2 * torch.pi / 3) ** 2

    return ab_error + alpha_error + beta_error + gamma_error


def rhombo_reduction_penalty(cell_lengths, cell_angles, margin):
    a, b, c = cell_lengths.unbind(dim=1)
    al, be, ga = cell_angles.unbind(dim=1)

    # crystal system enforcement
    ab_error = (a - b) ** 2
    bc_error = (b - c) ** 2
    angle_eq_error = (al - be) ** 2 + (be - ga) ** 2

    return ab_error + bc_error + angle_eq_error


def cube_reduction_penalty(cell_lengths, cell_angles, margin):
    a, b, c = cell_lengths.unbind(dim=-1)
    al, be, ga = cell_angles.unbind(dim=-1)
    # no reduction terms

    # crystal system enforcement
    # lengths equal
    ab_error = (a - b) ** 2
    bc_error = (b - c) ** 2

    # right angles
    alpha_error = (al - torch.pi / 2) ** 2
    beta_error = (be - torch.pi / 2) ** 2
    gamma_error = (ga - torch.pi / 2) ** 2

    return ab_error + bc_error + alpha_error + beta_error + gamma_error


def niggli_reduction_penalty(cell_lengths, cell_angles, **kwargs):
    #a, b, c, al, be, ga = self.zp1_cell_parameters()[:, :6].split(1, dim=1)
    a,b,c = cell_lengths.split(1, dim=1)
    al, be, ga = cell_angles.split(1, dim=1)
    ab = a * b
    ac = a * c
    bc = b * c

    al_cos = torch.cos(al)
    be_cos = torch.cos(be)
    ga_cos = torch.cos(ga)

    return (ab * ga_cos + ac * be_cos + bc * al_cos).flatten()


def cell_reduction_penalty(cell_angles, cell_lengths, sg, margin: float = 0.1):
    masks = {'triclinic': (sg == 1) | (sg == 2),
             'monoclinic': (sg >= 3) & (sg <= 15),
             'orthorhombic': (sg >= 16) & (sg <= 74),
             'tetragonal': (sg >= 75) & (sg <= 142),
             'trigonal': (sg >= 143) & (sg <= 167),
             'hexagonal': (sg >= 168) & (sg <= 194),
             'cubic': (sg >= 195) & (sg <= 230),
             }
    reduction_penalties = {
        'triclinic': tri_reduction_penalty,
        'monoclinic': mono_reduction_penalty,
        'orthorhombic': ortho_reduction_penalty,
        'tetragonal': tetra_reduction_penalty,
        'trigonal': trig_reduction_penalty,
        'hexagonal': hex_reduction_penalty,
        'cubic': cube_reduction_penalty,
    }
    E = torch.zeros(len(cell_lengths), dtype=torch.float32, device=cell_lengths.device)
    for cs, mask in masks.items():
        if mask.sum() > 0:
            E[mask] = reduction_penalties[cs](cell_lengths[mask], cell_angles[mask], margin)
            if cs == 'triclinic':  # this is actually used/required! Two separate reduction terms
                E[mask] = E[mask] + F.relu(niggli_reduction_penalty(cell_lengths, cell_angles)[
                                               mask] - margin) ** 2  # penalize positive overlaps
    return E
