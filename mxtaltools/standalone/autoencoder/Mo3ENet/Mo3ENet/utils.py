from argparse import Namespace
import torch
from pathlib import Path
import yaml
from torch_scatter import scatter

from typing import Optional

import torch
import torch_cluster
from torch_scatter import scatter_softmax
import torch.nn.functional as F
import plotly.graph_objects as go


def get_node_weights(mol_batch, decoded_mol_batch, decoding, num_decoder_nodes, node_weight_temperature):
    # per-atom weights of each graph
    molwise_weight_per_swarm_point = mol_batch.num_atoms / num_decoder_nodes

    # cast to num_decoder_nodes
    weight_per_swarm_point = molwise_weight_per_swarm_point.repeat_interleave(num_decoder_nodes)

    # softmax over decoding weight dimension, adjusted by temperature
    nodewise_weights = scatter_softmax(decoding[:, -1] / node_weight_temperature,
                                       decoded_mol_batch.batch,
                                       dim=0,
                                       dim_size=decoded_mol_batch.num_nodes)

    # reweigh against the number of atoms
    nodewise_weights_tensor = nodewise_weights * mol_batch.num_atoms.repeat_interleave(
        num_decoder_nodes)

    return weight_per_swarm_point, nodewise_weights, nodewise_weights_tensor


def init_decoded_data(data, decoding, device, num_nodes):
    decoded_data = data.detach().clone()
    decoded_data.pos = decoding[:, :3]
    decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(num_nodes).to(device)
    return decoded_data


def collate_decoded_data(data, decoding, num_decoder_nodes, node_weight_temperature, device):
    # generate input reconstructed as a data type
    decoded_mol_batch = init_decoded_data(data,
                                          decoding,
                                          device,
                                          num_decoder_nodes
                                          )
    # compute the distributional weight of each node
    nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = \
        get_node_weights(data, decoded_mol_batch, decoding,
                         num_decoder_nodes,
                         node_weight_temperature)
    decoded_mol_batch.aux_ind = nodewise_weights_tensor
    # input node weights are always 1 - corresponding each to an atom
    data.aux_ind = torch.ones(data.num_nodes, dtype=torch.float32, device=device)
    # get probability distribution over type dimensions
    decoded_mol_batch.x = F.softmax(decoding[:, 3:-1], dim=1)
    return decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor


def radius(x: torch.Tensor, y: torch.Tensor, r: float,
           batch_x: Optional[torch.Tensor] = None,
           batch_y: Optional[torch.Tensor] = None,
           max_num_neighbors: int = 32,
           num_workers: int = 1) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)

    .. code-block:: python

        import torch
        from torch_cluster import radius

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    ptr_x: Optional[torch.Tensor] = None
    if batch_x is not None:
        assert x.size(0) == batch_x.numel()
        batch_size = int(batch_x.max()) + 1

        deg = x.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))

        ptr_x = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_x[1:])

    ptr_y: Optional[torch.Tensor] = None
    if batch_y is not None:
        assert y.size(0) == batch_y.numel()
        batch_size = int(batch_y.max()) + 1

        deg = y.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch_y, torch.ones_like(batch_y))

        ptr_y = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_y[1:])

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers)


# @torch.jit.script
def asymmetric_radius_graph(x: torch.Tensor,
                            r: float,
                            inside_inds: torch.Tensor,
                            convolve_inds: torch.Tensor,
                            batch: torch.Tensor,
                            loop: bool = False,
                            max_num_neighbors: int = 32, flow: str = 'source_to_target',
                            num_workers: int = 1) -> torch.Tensor:
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element.
            If the number of actual neighbors is greater than
            :obj:`max_num_neighbors`, returned neighbors are picked randomly.
            (default: :obj:`32`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        inside_inds (Tensor): original indices for the nodes in the y subgraph

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import radius_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """
    if convolve_inds is None:  # indexes of items within x to convolve against y
        convolve_inds = torch.arange(len(x))

    assert flow in ['source_to_target', 'target_to_source']
    if batch is not None:
        edge_index = radius(x[convolve_inds], x[inside_inds], r, batch[convolve_inds], batch[inside_inds],
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)
    else:
        edge_index = radius(x[convolve_inds], x[inside_inds], r, None, None,
                            max_num_neighbors if loop else max_num_neighbors + 1,
                            num_workers)

    target, source = edge_index[0], edge_index[1]

    # edge_index[1] = inside_inds[edge_index[1, :]] # reindex
    target = inside_inds[target]  # contains correct indexes
    source = convolve_inds[source]

    if flow == 'source_to_target':
        row, col = source, target
    else:
        row, col = target, source

    if not loop:  # now properly deletes self-loops
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def load_yaml(path):
    """
    Safely load yaml file as dict.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


def batch_molecule_vdW_volume(atom_types_in, pos, batch, num_graphs, vdw_radii_tensor):
    if atom_types_in.ndim > 1:
        atom_types = atom_types_in[:, 0]
    else:
        atom_types = atom_types_in.clone()

    atom_volumes = 4 / 3 * torch.pi * vdw_radii_tensor[atom_types] ** 3
    raw_vdw_volumes = scatter(atom_volumes, batch, dim=0, dim_size=num_graphs, reduce='sum')
    bonds_i, bonds_j = radius(pos, pos,
                              r=2 * vdw_radii_tensor.max(),
                              batch_x=batch,
                              batch_y=batch,
                              max_num_neighbors=6)
    mask = ~(bonds_i >= bonds_j)  # eliminate duplicates
    bonds_i, bonds_j = bonds_i[mask], bonds_j[mask]
    bond_lengths = torch.linalg.norm(pos[bonds_i] - pos[bonds_j], dim=1)
    radii_i, radii_j = vdw_radii_tensor[atom_types[bonds_i]], vdw_radii_tensor[atom_types[bonds_j]]
    # https://mathworld.wolfram.com/Sphere-SphereIntersection.html
    sphere_overlaps = (torch.pi * (radii_i + radii_j - bond_lengths) ** 2 *
                       (bond_lengths ** 2 + 2 * bond_lengths * radii_j - 3 * radii_j ** 2
                        + 2 * bond_lengths * radii_i + 6 * radii_j * radii_i - 3 * radii_i ** 2) / (12 * bond_lengths))
    sphere_overlaps[bond_lengths > (radii_i + radii_j)] = 0
    molwise_sphere_overlaps = scatter(sphere_overlaps, batch[bonds_i], dim=0, dim_size=num_graphs,
                                      reduce='sum')
    corrected_mol_volume = raw_vdw_volumes - molwise_sphere_overlaps
    return corrected_mol_volume


VDW_RADII = {
    0: 0.0, 1: 1.2, 2: 1.4, 3: 2.2, 4: 1.9, 5: 1.8, 6: 1.7, 7: 1.6, 8: 1.55, 9: 1.5, 10: 1.54, 11: 2.4, 12: 2.2,
    13: 2.1, 14: 2.1, 15: 1.95, 16: 1.8, 17: 1.8, 18: 1.88, 19: 2.8, 20: 2.4, 21: 2.3, 22: 2.15, 23: 2.05, 24: 2.05,
    25: 2.05,
    26: 2.05, 27: 2.0, 28: 2.0, 29: 2.0, 30: 2.1, 31: 2.1, 32: 2.1, 33: 2.05, 34: 1.9, 35: 1.9, 36: 2.02, 37: 2.9,
    38: 2.55, 39: 2.4, 40: 2.3, 41: 2.15, 42: 2.1, 43: 2.05, 44: 2.05, 45: 2.0, 46: 2.05, 47: 2.1, 48: 2.2, 49: 2.2,
    50: 2.25, 51: 2.2, 52: 2.1, 53: 2.1, 54: 2.16, 55: 3.0, 56: 2.7, 57: 2.5, 58: 2.48, 59: 2.47, 60: 2.45, 61: 2.43,
    62: 2.42, 63: 2.4, 64: 2.38, 65: 2.37, 66: 2.35, 67: 2.33, 68: 2.32, 69: 2.3, 70: 2.28, 71: 2.27, 72: 2.25, 73: 2.2,
    74: 2.1, 75: 2.05, 76: 2.0, 77: 2.0, 78: 2.05, 79: 2.1, 80: 2.05, 81: 2.2, 82: 2.3, 83: 2.3, 84: 2.0, 85: 2.0,
    86: 2.0, 87: 2.0, 88: 2.0, 89: 2.0, 90: 2.4, 91: 2.0, 92: 2.3, 93: 2.0, 94: 2.0, 95: 2.0, 96: 2.0, 97: 2.0, 98: 2.0,
    99: 2.0
}

ATOM_WEIGHTS = {0: 1.0, 1: 1.008, 2: 4.002602, 3: 6.94, 4: 9.0121831, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999,
                9: 18.998403163, 10: 20.1797,
                11: 22.98976928, 12: 24.305, 13: 26.9815385, 14: 28.085, 15: 30.973761998, 16: 32.06, 17: 35.45,
                18: 39.948, 19: 39.0983, 20: 40.078,
                21: 44.955908, 22: 47.867, 23: 50.9415, 24: 51.9961, 25: 54.938044, 26: 55.845, 27: 58.933194,
                28: 58.6934, 29: 63.546, 30: 65.38,
                31: 69.723, 32: 72.63, 33: 74.921595, 34: 78.971, 35: 79.904, 36: 83.798, 37: 85.4678, 38: 87.62,
                39: 88.90584, 40: 91.224,
                41: 92.90637, 42: 95.95, 43: 97.90721, 44: 101.07, 45: 102.9055, 46: 106.42, 47: 107.8682, 48: 112.414,
                49: 114.818, 50: 118.71,
                51: 121.76, 52: 127.6, 53: 126.90447, 54: 131.293, 55: 132.90545196, 56: 137.327, 57: 138.90547,
                58: 140.116, 59: 140.90766, 60: 144.242,
                61: 144.91276, 62: 150.36, 63: 151.964, 64: 157.25, 65: 158.92535, 66: 162.5, 67: 164.93033,
                68: 167.259, 69: 168.93422, 70: 173.054,
                71: 174.9668, 72: 178.49, 73: 180.94788, 74: 183.84, 75: 186.207, 76: 190.23, 77: 192.217, 78: 195.084,
                79: 196.966569, 80: 200.592,
                81: 204.38, 82: 207.2, 83: 208.9804, 84: 208.98243, 85: 209.98715, 86: 222.01758, 87: 223.01974,
                88: 226.02541, 89: 227.02775, 90: 232.0377,
                91: 231.03588, 92: 238.02891, 93: 237.04817, 94: 244.06421, 95: 243.06138, 96: 247.07035, 97: 247.07031,
                98: 251.07959, 99: 252.083}

"from Mendeleev - slow to load there"
ELECTRONEGATIVITY = {0: -1, 1: 2.2, 2: -1, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: -1,
                     11: 0.93, 12: 1.31, 13: 1.61, 14: 1.9, 15: 2.19, 16: 2.58, 17: 3.16, 18: -1, 19: 0.82, 20: 1.0,
                     21: 1.36, 22: 1.54, 23: 1.63,
                     24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.9, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18,
                     34: 2.55, 35: 2.96, 36: -1, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 43: 2.1,
                     44: 2.2, 45: 2.28,
                     46: 2.2, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 54: 2.6, 55: 0.79,
                     56: 0.89, 57: 1.1, 58: 1.12, 59: 1.13, 60: 1.14, 61: -1, 62: 1.17, 63: -1, 64: 1.2, 65: -1,
                     66: 1.22, 67: 1.23,
                     68: 1.24, 69: 1.25, 70: -1, 71: 1.0, 72: 1.3, 73: 1.5, 74: 1.7, 75: 1.9, 76: 2.2, 77: 2.2, 78: 2.2,
                     79: 2.4, 80: 1.9, 81: 1.8, 82: 1.8, 83: 1.9, 84: 2.0, 85: 2.2, 86: -1, 87: 0.7, 88: 0.9, 89: 1.1,
                     90: 1.3, 91: 1.5,
                     92: 1.7, 93: 1.3, 94: 1.3, 95: -1, 96: -1, 97: -1, 98: -1, 99: -1}

PERIOD = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 3, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3,
          17: 3, 18: 3, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4, 29: 4, 30: 4, 31: 4,
          32: 4, 33: 4, 34: 4,
          35: 4, 36: 4, 37: 5, 38: 5, 39: 5, 40: 5, 41: 5, 42: 5, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5, 49: 5,
          50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 6, 56: 6, 57: 6, 58: 6, 59: 6, 60: 6, 61: 6, 62: 6, 63: 6, 64: 6,
          65: 6, 66: 6, 67: 6,
          68: 6, 69: 6, 70: 6, 71: 6, 72: 6, 73: 6, 74: 6, 75: 6, 76: 6, 77: 6, 78: 6, 79: 6, 80: 6, 81: 6, 82: 6,
          83: 6, 84: 6, 85: 6, 86: 6, 87: 7, 88: 7, 89: 7, 90: 7, 91: 7, 92: 7, 93: 7, 94: 7, 95: 7, 96: 7, 97: 7,
          98: 7, 99: 7,
          }

GROUP = {0: 0, 1: 1, 2: 18, 3: 1, 4: 2, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 11: 1, 12: 2, 13: 13, 14: 14, 15: 15,
         16: 16, 17: 17, 18: 18, 19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 6, 25: 7, 26: 8, 27: 9, 28: 10, 29: 11, 30: 12,
         31: 13,
         32: 14, 33: 15, 34: 16, 35: 17, 36: 18, 37: 1, 38: 2, 39: 3, 40: 4, 41: 5, 42: 6, 43: 7, 44: 8, 45: 9, 46: 10,
         47: 11, 48: 12, 49: 13, 50: 14, 51: 15, 52: 16, 53: 17, 54: 18, 55: 1, 56: 2, 57: 3, 58: 19, 59: 19, 60: 19,
         61: 19,
         62: 19, 63: 19, 64: 19, 65: 19, 66: 19, 67: 19, 68: 19, 69: 19, 70: 19, 71: 19, 72: 4, 73: 5, 74: 6, 75: 7,
         76: 8, 77: 9, 78: 10, 79: 11, 80: 12, 81: 13, 82: 14, 83: 15, 84: 16, 85: 17, 86: 18, 87: 1, 88: 2, 89: 3,
         90: 19, 91: 19,
         92: 19, 93: 19, 94: 19, 95: 19, 96: 19, 97: 19, 98: 19, 99: 19}

ATOMIC_SYMBOLS = {0: 'null', 1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                  11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
                  21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr',
                  25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
                  35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru',
                  45: 'Rh', 46: 'Pd', 47: 'Ag',
                  48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
                  58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho',
                  68: 'Er', 69: 'Tm', 70: 'Yb',
                  71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
                  81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
                  91: 'Pa', 92: 'U', 93: 'Np',
                  94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es'}

ATOMIC_NUMBERS = {value: key for key, value in ATOMIC_SYMBOLS.items()}


def reload_model(model, device, optimizer, path, reload_optimizer=False):
    """
    load model and state dict from path
    includes fix for potential dataparallel issue
    """
    checkpoint = torch.load(path, map_location=device)
    if list(checkpoint['model_state_dict'])[0][
       0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        if reload_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def swarm_vs_tgt_fig(data, decoded_data, max_point_types, graph_ind=0):
    cmax = 1
    fig = go.Figure()  # scatter all the true & predicted points, colorweighted by atom type
    colors = ['rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)',
              'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)',
              'rgb(165, 170, 153)'] * 10
    colorscales = [[[0, 'rgba(0, 0, 0, 0)'], [1, color]] for color in colors]
    points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
    points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()
    for j in range(max_point_types):
        ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()

        pred_type_weights = (decoded_data.aux_ind[decoded_data.batch == graph_ind] * decoded_data.x[
            decoded_data.batch == graph_ind, j]).cpu().detach().numpy()

        fig.add_trace(go.Scatter3d(x=points_true[ref_type_inds][:, 0], y=points_true[ref_type_inds][:, 1],
                                   z=points_true[ref_type_inds][:, 2],
                                   mode='markers', marker_color=colors[j], marker_size=7, marker_line_width=5,
                                   marker_line_color='black',
                                   showlegend=True if (j == 0 and graph_ind == 0) else False,
                                   name=f'True type', legendgroup=f'True type'
                                   ))

        fig.add_trace(go.Scatter3d(x=points_pred[:, 0], y=points_pred[:, 1], z=points_pred[:, 2],
                                   mode='markers',
                                   marker=dict(size=10, color=pred_type_weights, colorscale=colorscales[j], cmax=cmax,
                                               cmin=0), opacity=1, marker_line_color='white',
                                   showlegend=True,
                                   name=f'Predicted type {j}'
                                   ))
    return fig


def compute_gaussian_overlap(ref_types,
                             mol_batch,
                             decoded_data,
                             sigma,
                             nodewise_weights,
                             dist_to_self=False,
                             isolate_dimensions: list = None,
                             type_distance_scaling=0.1,
                             return_dists=False
                             ):
    """
    same as previous version
    except atom type differences are treated as high dimensional distances
    """
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)

    if dist_to_self:
        pred_points = ref_points
    else:
        pred_types = decoded_data.x * type_distance_scaling  # nodes are already weighted at 1
        pred_points = torch.cat((decoded_data.pos, pred_types), dim=1)  # assume input x has already been normalized

    if isolate_dimensions is not None:  # only compute distances over certain dimensions
        ref_points = ref_points[:, isolate_dimensions[0]:isolate_dimensions[1]]
        pred_points = pred_points[:, isolate_dimensions[0]:isolate_dimensions[1]]

    edges = radius(ref_points, pred_points,
                   # r=2 * ref_points[:, :3].norm(dim=1).amax(),  # max range encompasses largest molecule in the batch
                   # alternatively any point which will have even a small overlap - should be faster by ignoring unimportant edges, where the gradient will anyway be vanishing
                   r=4 * sigma,
                   max_num_neighbors=10000,
                   batch_x=mol_batch.batch,
                   batch_y=decoded_data.batch)  # this step is slower than before
    dists = torch.linalg.norm(ref_points[edges[1]] - pred_points[edges[0]], dim=1)
    overlap = torch.exp(-torch.pow(dists / sigma, 2))
    scaled_overlap = overlap * nodewise_weights[edges[0]]  # reweight appropriately
    nodewise_overlap = scatter(scaled_overlap,
                               edges[1],
                               reduce='sum',
                               dim_size=mol_batch.num_nodes)

    if not return_dists:
        return nodewise_overlap
    else:
        return nodewise_overlap, edges, dists


def ae_reconstruction_loss(mol_batch,
                           decoded_mol_batch,
                           graph_normed_nodewise_weights,
                           nodewise_weights_tensor,
                           num_atom_types,
                           type_distance_scaling,
                           autoencoder_sigma,
                           ):
    true_node_one_hot = F.one_hot(mol_batch.x.flatten().long(), num_classes=num_atom_types).float()

    decoder_likelihoods, input2output_edges, input2output_dists = (
        compute_gaussian_overlap(true_node_one_hot, mol_batch, decoded_mol_batch, autoencoder_sigma,
                                 nodewise_weights=decoded_mol_batch.aux_ind,
                                 type_distance_scaling=type_distance_scaling,
                                 return_dists=True
                                 ))

    # if sigma is too large, these can be > 1, so we map to the overlap of the true density with itself
    self_likelihoods = compute_gaussian_overlap(true_node_one_hot, mol_batch, mol_batch, autoencoder_sigma,
                                                nodewise_weights=mol_batch.aux_ind, dist_to_self=True,
                                                type_distance_scaling=type_distance_scaling)

    # typewise agreement for whole graph
    per_graph_true_types = scatter(
        true_node_one_hot, mol_batch.batch[:, None], dim=0, reduce='mean')
    per_graph_pred_types = scatter(
        decoded_mol_batch.x * graph_normed_nodewise_weights[:, None], decoded_mol_batch.batch[:, None], dim=0,
        reduce='sum')

    nodewise_type_loss = (
            F.binary_cross_entropy(per_graph_pred_types.clip(min=1e-6, max=1 - 1e-6), per_graph_true_types) -
            F.binary_cross_entropy(per_graph_true_types, per_graph_true_types))

    nodewise_reconstruction_loss = F.smooth_l1_loss(decoder_likelihoods, self_likelihoods, reduction='none')
    graph_reconstruction_loss = scatter(nodewise_reconstruction_loss, mol_batch.batch, reduce='mean')

    # new losses -
    # 1 penalize output components for distance to nearest atom
    nearest_node_dist = scatter(input2output_dists,
                                input2output_edges[0],
                                reduce='min',
                                dim_size=decoded_mol_batch.num_nodes
                                )
    nearest_node_loss = scatter(nearest_node_dist, decoded_mol_batch.batch, reduce='mean',
                                dim_size=mol_batch.num_graphs)
    # 1a also identify reciprocal distance from each atom to nearest component
    nearest_component_dist = scatter(input2output_dists,
                                     input2output_edges[1],
                                     reduce='min',
                                     dim_size=mol_batch.num_nodes
                                     )
    nearest_component_loss = scatter(nearest_component_dist,
                                     mol_batch.batch,
                                     reduce='mean',
                                     dim_size=mol_batch.num_graphs)
    # 2 penalize area near an atom for not being a part of an exactly atom-size clump
    collect_bools = input2output_dists < 0.5
    inds_within_cutoff = input2output_edges[0][collect_bools]
    inside_edge_nodes = input2output_edges[1][collect_bools]
    collected_particle_weights = nodewise_weights_tensor[inds_within_cutoff]
    pred_particle_weights = scatter(collected_particle_weights,
                                    inside_edge_nodes,
                                    reduce='sum',
                                    dim_size=mol_batch.num_nodes,
                                    )

    nodewise_clumping_loss = F.smooth_l1_loss(pred_particle_weights, torch.ones_like(pred_particle_weights),
                                              reduction='none')
    graph_clumping_loss = scatter(nodewise_clumping_loss, mol_batch.batch, reduce='mean')

    return (nodewise_reconstruction_loss, nodewise_type_loss,
            graph_reconstruction_loss, self_likelihoods,
            nearest_node_loss, graph_clumping_loss,
            nearest_component_dist, nearest_component_loss)


def batch_rmsd(mol_batch,
               decoded_mol_batch,
               true_node_one_hot,
               intrapoint_cutoff: float = 0.5,
               probability_threshold: float = 0.25,
               type_distance_scaling: float = 2):
    ref_types = true_node_one_hot.float()
    ref_points = torch.cat((mol_batch.pos, ref_types * type_distance_scaling), dim=1)
    pred_types = decoded_mol_batch.x * type_distance_scaling  # nodes are already weighted at 1
    pred_points = torch.cat((decoded_mol_batch.pos, pred_types), dim=1)  # assume input x has already been normalized
    nodewise_weights = decoded_mol_batch.aux_ind

    edges = radius(ref_points,
                   pred_points,
                   r=intrapoint_cutoff,
                   max_num_neighbors=10000,
                   batch_x=mol_batch.batch,
                   batch_y=decoded_mol_batch.batch)  # this step is slower than before
    dists = torch.linalg.norm(ref_points[edges[1]] - pred_points[edges[0]], dim=1)

    collect_bools = dists < intrapoint_cutoff
    inds_within_cutoff = edges[0][collect_bools]
    inside_edge_nodes = edges[1][collect_bools]
    collected_particles = pred_points[inds_within_cutoff]
    collected_particle_weights = nodewise_weights[inds_within_cutoff]
    # # confirm each output is mapped to a single input
    # a, b = torch.unique(edges[0][collect_bools], return_counts=True)
    # assert b.max() == 1
    pred_particle_weights = scatter(collected_particle_weights,
                                    inside_edge_nodes,
                                    reduce='sum',
                                    dim_size=mol_batch.num_nodes,
                                    )
    # filter here for where we do not match the scaffold (no nearby nodes, or insufficient probability mass)
    missing_particle_bools = (1 - pred_particle_weights).abs() >= probability_threshold
    complete_graph_bools = scatter((~missing_particle_bools).long(),
                                   mol_batch.batch,
                                   reduce='mul',
                                   dim_size=mol_batch.num_graphs,
                                   dim=0
                                   ).bool()
    pred_particle_points = scatter(collected_particles * collected_particle_weights[:, None],
                                   inside_edge_nodes,
                                   reduce='sum',
                                   dim=0,
                                   dim_size=mol_batch.num_nodes,
                                   )
    pred_dists = torch.linalg.norm(ref_points - pred_particle_points, dim=1)
    rmsd = scatter(pred_dists, mol_batch.batch, reduce='mean', dim_size=mol_batch.num_graphs)
    rmsd[~complete_graph_bools] = torch.nan
    pred_particle_points[missing_particle_bools] *= torch.nan

    return rmsd, pred_dists, complete_graph_bools, ~missing_particle_bools, pred_particle_points, pred_particle_weights
