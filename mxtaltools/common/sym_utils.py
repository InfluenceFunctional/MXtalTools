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


# monoclinic setting class per sg (SYM_OPS setting: b-unique, cell choice 1): the group of ac-plane basis changes
# a' = p a + q c, c' = r a + s c (det +-1, b' = det b) that keep the SYM_OPS operator list up to an origin shift
MONO_GLIDE, MONO_CENTRED, MONO_FREE = 0, 1, 2
MONO_CLASS = torch.full((231,), -1, dtype=torch.long)
MONO_CLASS[[7, 13, 14]] = MONO_GLIDE  # c-glide, primitive: r even
MONO_CLASS[[5, 8, 9, 12, 15]] = MONO_CENTRED  # C-centred: q even
MONO_CLASS[[3, 4, 6, 10, 11]] = MONO_FREE  # primitive, no glide: any basis change


def mono_reduction_penalty(cell_lengths, cell_angles, sg, margin):
    """Monoclinic (sg 3-15) reduction walls: one fundamental domain per setting class, so each lattice has exactly one
    zero-penalty cell among the cells sharing its SYM_OPS operator list; that cell is spglib's standard cell
    (checked empirically, spglib 2.7.0).
      class    sg                 basis changes   walls
      GLIDE    7, 13, 14          r even          W1 cos(beta) <= 0;  W2  c|cos(beta)| <= a;  W3  a|cos(beta)| <= c/2
      CENTRED  5, 8, 9, 12, 15    q even          W1;                 W2' a|cos(beta)| <= c;  W3' c|cos(beta)| <= a/2
      FREE     3, 4, 6, 10, 11    any             W1;                 a <= c;                 W3' c|cos(beta)| <= a/2
    plus alpha = gamma = 90 deg. Max beta in the zero set: 135 deg (GLIDE, CENTRED), 120 deg (FREE).
    margin = 0: the zero set is the closed domain (2 cells with identical a, c, beta on a wall).
    margin > 0: cos(beta) in [lower + margin, -margin] and FREE a/c <= 1 - margin; a lattice whose domain cell lies in
    that band has no zero-penalty cell.
    One cell per lattice holds for E == 0 only: E < 1e-3 admits several cells per lattice, and in float32 a cell within
    ~1e-7 (relative) of a wall can have 0 or 2 zero-penalty images.
    The FREE ordering term is unbounded (up to ~3.5e3 in the sg 10 latent box, where the cos(beta) terms stay below 0.66)
    and overflows float32 above a/c ~ 1.84e19.
    a and c are clamped at 1e-6: below that dE/da = 0, and the penalty can be lower than the previous walls
    (cos(beta) in [-a/c, 0] for every sg)."""
    eps = 1e-6
    a, b, c = cell_lengths.unbind(dim=1)
    al, be, ga = cell_angles.unbind(dim=1)
    cls = MONO_CLASS.to(sg.device)[sg.long()]  # any integer or float sg dtype, as the other systems accept
    is_glide, is_centred, is_free = cls == MONO_GLIDE, cls == MONO_CENTRED, cls == MONO_FREE

    a_ = a.clamp(min=eps)
    c_ = c.clamp(min=eps)
    # |cos(beta)| limit per class; sg 0-230 outside 3-15 gives NaN (negative sg wraps, sg > 230 raises)
    lim = torch.where(is_glide, torch.minimum(a_ / c_, c_ / (2 * a_)),
                      torch.where(is_centred, torch.minimum(c_ / a_, a_ / (2 * c_)),
                                  torch.where(is_free, a_ / (2 * c_), torch.full_like(a_, float('nan')))))
    be_min_cos = (-lim).clamp(min=-1, max=0)
    beta_error = bounding_penalty(be.cos(), be_min_cos, 0, margin=margin)

    # FREE only: a <= c, the ordering spglib picks
    order_error = torch.where(is_free, F.relu(a_ / c_ - (1 - margin)) ** 2, torch.zeros_like(a_))

    # enforces the crystal system
    alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
    gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

    return beta_error + order_error + alpha_error + gamma_error


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
    reduction_penalties = {  # monoclinic is dispatched below: its walls also need sg
        'triclinic': tri_reduction_penalty,
        'orthorhombic': ortho_reduction_penalty,
        'tetragonal': tetra_reduction_penalty,
        'trigonal': trig_reduction_penalty,
        'hexagonal': hex_reduction_penalty,
        'cubic': cube_reduction_penalty,
    }
    E = torch.zeros(len(cell_lengths), dtype=torch.float32, device=cell_lengths.device)
    for cs, mask in masks.items():
        if mask.sum() > 0:
            if cs == 'monoclinic':  # walls depend on the setting class of each sg
                E[mask] = mono_reduction_penalty(cell_lengths[mask], cell_angles[mask], sg[mask], margin)
            else:
                E[mask] = reduction_penalties[cs](cell_lengths[mask], cell_angles[mask], margin)
            if cs == 'triclinic':  # this is actually used/required! Two separate reduction terms
                E[mask] = E[mask] + F.relu(niggli_reduction_penalty(cell_lengths, cell_angles)[
                                               mask] - margin) ** 2  # penalize positive overlaps
    return E
