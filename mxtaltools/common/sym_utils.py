from mxtaltools.constants.space_group_info import SYM_OPS, POINT_GROUPS, LATTICE_TYPE, SPACE_GROUPS


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
    sym_info = {  # collect space group info into single dict
        'sym_ops': sym_ops,
        'point_groups': point_groups,
        'lattice_type': lattice_type,
        'space_groups': space_groups}

    return sym_info
