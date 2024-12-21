import copy
from collections.abc import Mapping, Sequence
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional)

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.deprecation import deprecated
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import subgraph
from torch_sparse import SparseTensor

from mxtaltools.constants.space_group_info import SYM_OPS


###############################################################################


def cell_parameters_to_box_vectors(opt: str,
                                   cell_lengths: torch.tensor,
                                   cell_angles: torch.tensor,
                                   return_vol: bool = False):
    """
    Initially borrowed from Nikos
    Quickly convert from cell lengths and angles to fractional transform matrices fractional->cartesian or cartesian->fractional
    """
    ''' Calculate cos and sin of cell angles '''
    cos_a = torch.cos(cell_angles)
    sin_a = torch.sin(cell_angles)

    ''' Calculate volume of the unit cell '''
    val = 1.0 - cos_a[0] ** 2 - cos_a[1] ** 2 - cos_a[2] ** 2 + 2.0 * cos_a[0] * cos_a[1] * cos_a[2]
    vol = torch.sign(val) * cell_lengths[0] * cell_lengths[1] * cell_lengths[2] * torch.sqrt(
        torch.abs(val))  # technically a signed quanitity

    ''' Setting the transformation matrix '''
    m = torch.zeros((3, 3))
    if opt == 'c_to_f':
        ''' Converting from cartesian to fractional '''
        m[0, 0] = 1.0 / cell_lengths[0]
        m[0, 1] = -cos_a[2] / cell_lengths[0] / sin_a[2]
        m[0, 2] = cell_lengths[1] * cell_lengths[2] * (cos_a[0] * cos_a[2] - cos_a[1]) / vol / sin_a[2]
        m[1, 1] = 1.0 / cell_lengths[1] / sin_a[2]
        m[1, 2] = cell_lengths[0] * cell_lengths[2] * (cos_a[1] * cos_a[2] - cos_a[0]) / vol / sin_a[2]
        m[2, 2] = cell_lengths[0] * cell_lengths[1] * sin_a[2] / vol
    elif opt == 'f_to_c':
        ''' Converting from fractional to cartesian '''
        m[0, 0] = cell_lengths[0]
        m[0, 1] = cell_lengths[1] * cos_a[2]
        m[0, 2] = cell_lengths[2] * cos_a[1]
        m[1, 1] = cell_lengths[1] * sin_a[2]
        m[1, 2] = cell_lengths[2] * (cos_a[0] - cos_a[1] * cos_a[2]) / sin_a[2]
        m[2, 2] = vol / cell_lengths[0] / cell_lengths[1] / sin_a[2]

    # todo create m in a single-step
    if return_vol:
        return m, torch.abs(vol)
    else:
        return m


# noinspection PyPropertyAccess
class CrystalData(BaseData):
    r"""A data object describing a homogeneous graph.  # todo update docstring
    The data object can hold node-level, link-level and graph-level attributes.
    In general, :class:`~torch_geometric.data.Data` tries to mimic the
    behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    x: atom-wise node features for asymmetric unit
    pos: node positions typically cartesian space, for asymmetric unit
    mol_x: molecule-wise features

    y: graph-wise target features - e.g., regression or classification targets

    tracking: array of graph-wise features for tracking & analysis
    smiles: SMILES strings of molecule in crystal
    csd_identifier: unique identifier string for this crystal structure
    mol_volume: volume of a single molecule # TODO deprecate this is unreliably computed

    sg_ind: space group index 1-230. Always assume general Wyckhoff positions
    # todo add z_prime
    T_fc: fractional-cartesian transform matrix, transpose of box vectors
    unit_cell_pos: node positions for the full unit cell in format [Z, n_atoms, 3]
    cell_params: 12D cell parameters # TOTO unify formatting between intrinsic and extrinsic methods  # todo issue for Z'>1
    aunit_handendess: handedness of the molecule in the asymmetric unit, according to our principal inertial handedness convention  # todo issue for Z'>1
    mult: Z/Z', the symmetry multiplicity of the crystal. A feature of the space group.
    symmetry_operators: list of symmetry operations to generate the crystal from asymmetric unit. Only necessary when using nonstandard SG settings, like data from the CSD.

    aux_ind: auxiliary index for determining nodes which are canonical vs symmetry images
    mol_ind: auxiliary index identify which of the Z' molecules each atom is a part of

    """

    def __init__(self,
                 x: OptTensor = None,
                 graph_x: OptTensor = None,
                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 y: OptTensor = None,
                 pos: OptTensor = None,
                 aux_ind: OptTensor = None,
                 mol_ind: OptTensor = None,
                 cell_lengths: OptTensor = torch.ones(3),
                 cell_angles: OptTensor = torch.ones(3) * torch.pi / 2,
                 z_prime: int = 0,
                 sg_ind: int = 1,  # default to P1
                 pose_parameters: list = [],
                 smiles: str = None,
                 identifier: str = None,
                 nonstandard_symmetry: bool = False,
                 unit_cell_pos: torch.tensor = None,
                 symmetry_operators: list = None,
                 aunit_handedness: list = None,
                 is_well_defined: bool = True,
                 require_crystal_features: bool = True,
                 vdw_pot: float = None,
                 vdw_loss: float = None,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # fix node & graph attributes
        if x is not None:
            self.x = x
            self.num_atoms = len(x)
        if graph_x is not None:
            self.graph_x = graph_x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if pos is not None:
            self.pos = pos
            if self.radius is None:
                centroid = pos.mean(dim=0)
                self.radius = torch.amax(torch.linalg.norm(pos - centroid, dim=1))
        if vdw_pot is not None:
            self.vdw_pot = vdw_pot
        if vdw_loss is not None:
            self.vdw_loss = vdw_loss

        # fix identifiers
        if smiles is not None:
            self.smiles = smiles
        if identifier is not None:
            self.identifier = identifier

        if require_crystal_features:
            # fix helper indices
            if aux_ind is not None:
                self.aux_ind = aux_ind
                assert len(aux_ind) == len(x)
            if mol_ind is not None:
                self.mol_ind = mol_ind
                assert len(mol_ind) == len(x)

            # fix symmetries
            self.sg_ind = sg_ind
            if nonstandard_symmetry:  # set as list for correct collation behavior
                self.symmetry_operators = symmetry_operators
                self.nonstandard_symmetry = True
            else:  # standard symmetry
                self.symmetry_operators = np.stack(SYM_OPS[sg_ind])
                self.nonstandard_symmetry = False

            self.sym_mult = torch.ones(1, dtype=torch.long) * len(self.symmetry_operators)
            self.z_prime = z_prime

            # fix box
            self.cell_lengths = cell_lengths[None, ...]
            self.cell_angles = cell_angles[None, ...]
            if all(cell_lengths.flatten() == torch.ones(3)):  # if we are going with the default box
                self.T_fc, self.T_cf = torch.eye(3), torch.eye(3)
                self.cell_volume = 1
            else:
                self.T_fc, self.cell_volume = (
                    cell_parameters_to_box_vectors('f_to_c', cell_lengths, cell_angles, return_vol=True))
                self.T_fc = self.T_fc[None, ...]
                self.T_cf = torch.linalg.inv(self.T_fc[0])[None, ...]
            self.reduced_volume = self.cell_volume / self.sym_mult

            # fix molecule poses
            assert len(pose_parameters) == z_prime
            if z_prime > 0:
                for zp in range(4):
                    setattr(self, f'pose_params{zp}', torch.ones(6)[None, ...])  # fill with dummy values
                for zp in range(z_prime):  # fill with any real values
                    setattr(self, f'pose_params{zp}', pose_parameters[zp][None, ...])

            # record prebuilt unit cell coordinates
            if unit_cell_pos is not None:
                self.unit_cell_pos = unit_cell_pos.numpy()  # if it's saved as a tensor, we get problems in collation
                assert unit_cell_pos.shape == (self.sym_mult, len(x), 3)

            if aunit_handedness is not None:
                self.aunit_handedness = aunit_handedness

            if is_well_defined is not None:
                self.is_well_defined = is_well_defined

    def cell_parameters(self):
        """
        return the zp=1 total cell parameter tensor
        Returns
        -------

        """
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.pose_params1], dim=1)

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        has_dict = any([isinstance(v, Mapping) for v in self._store.values()])

        if not has_dict:
            info = [size_repr(k, v) for k, v in self._store.items()]
            info = ', '.join(info)
            return f'{cls}({info})'
        else:
            info = [size_repr(k, v, indent=2) for k, v in self._store.items()]
            info = ',\n'.join(info)
            return f'{cls}(\n{info}\n)'

    def stores_as(self, data: 'Data'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or 'face' in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def debug(self):
        pass

    def is_node_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes a
        node-level attribute."""
        return self._store.is_node_attr(key)

    def is_edge_attr(self, key: str) -> bool:
        r"""Returns :obj:`True` if the object at key :obj:`key` denotes an
        edge-level attribute."""
        return self._store.is_edge_attr(key)

    def subgraph(self, subset: Tensor):
        r"""Returns the induced subgraph given by the node indices
        :obj:`subset`.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """

        out = subgraph(subset, self.edge_index, relabel_nodes=True,
                       num_nodes=self.num_nodes, return_edge_mask=True)
        edge_index, _, edge_mask = out

        if subset.dtype == torch.bool:
            num_nodes = int(subset.sum())
        else:
            num_nodes = subset.size(0)

        data = copy.copy(self)

        for key, value in data:
            if key == 'edge_index':
                data.edge_index = edge_index
            elif key == 'num_nodes':
                data.num_nodes = num_nodes
            elif isinstance(value, Tensor):
                if self.is_node_attr(key):
                    data[key] = value[subset]
                elif self.is_edge_attr(key):
                    data[key] = value[edge_mask]

        return data

    def to_heterogeneous(self, node_type: Optional[Tensor] = None,
                         edge_type: Optional[Tensor] = None,
                         node_type_names: Optional[List[NodeType]] = None,
                         edge_type_names: Optional[List[EdgeType]] = None):
        r"""Converts a :class:`~torch_geometric.data.Data` object to a
        heterogeneous :class:`~torch_geometric.data.HeteroData` object.
        For this, node and edge attributes are splitted according to the
        node-level and edge-level vectors :obj:`node_type` and
        :obj:`edge_type`, respectively.
        :obj:`node_type_names` and :obj:`edge_type_names` can be used to give
        meaningful node and edge type names, respectively.
        That is, the node_type :obj:`0` is given by :obj:`node_type_names[0]`.
        If the :class:`~torch_geometric.data.Data` object was constructed via
        :meth:`~torch_geometric.data.HeteroData.to_homogeneous`, the object can
        be reconstructed without any need to pass in additional arguments.

        Args:
            node_type (Tensor, optional): A node-level vector denoting the type
                of each node. (default: :obj:`None`)
            edge_type (Tensor, optional): An edge-level vector denoting the
                type of each edge. (default: :obj:`None`)
            node_type_names (List[str], optional): The names of node types.
                (default: :obj:`None`)
            edge_type_names (List[Tuple[str, str, str]], optional): The names
                of edge types. (default: :obj:`None`)
        """
        from torch_geometric.data import HeteroData

        if node_type is None:
            node_type = self._store.get('node_type', None)
        if node_type is None:
            node_type = torch.zeros(self.num_nodes, dtype=torch.long)

        if node_type_names is None:
            store = self._store
            node_type_names = store.__dict__.get('_node_type_names', None)
        if node_type_names is None:
            node_type_names = [str(i) for i in node_type.unique().tolist()]

        if edge_type is None:
            edge_type = self._store.get('edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(self.num_edges, dtype=torch.long)

        if edge_type_names is None:
            store = self._store
            edge_type_names = store.__dict__.get('_edge_type_names', None)
        if edge_type_names is None:
            edge_type_names = []
            edge_index = self.edge_index
            for i in edge_type.unique().tolist():
                src, dst = edge_index[:, edge_type == i]
                src_types = node_type[src].unique().tolist()
                dst_types = node_type[dst].unique().tolist()
                if len(src_types) != 1 and len(dst_types) != 1:
                    raise ValueError(
                        "Could not construct a 'HeteroData' object from the "
                        "'Data' object because single edge types span over "
                        "multiple node types")
                edge_type_names.append((node_type_names[src_types[0]], str(i),
                                        node_type_names[dst_types[0]]))

        # We iterate over node types to find the local node indices belonging
        # to each node type. Furthermore, we create a global `index_map` vector
        # that maps global node indices to local ones in the final
        # heterogeneous graph:
        node_ids, index_map = {}, torch.empty_like(node_type)
        for i, key in enumerate(node_type_names):
            node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
            index_map[node_ids[i]] = torch.arange(len(node_ids[i]))

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, key in enumerate(edge_type_names):
            edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        data = HeteroData()

        for i, key in enumerate(node_type_names):
            for attr, value in self.items():
                if attr == 'node_type' or attr == 'edge_type':
                    continue
                elif isinstance(value, Tensor) and self.is_node_attr(attr):
                    data[key][attr] = value[node_ids[i]]

            if len(data[key]) == 0:
                data[key].num_nodes = node_ids[i].size(0)

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in self.items():
                if attr == 'node_type' or attr == 'edge_type':
                    continue
                elif attr == 'edge_index':
                    edge_index = value[:, edge_ids[i]]
                    edge_index[0] = index_map[edge_index[0]]
                    edge_index[1] = index_map[edge_index[1]]
                    data[key].edge_index = edge_index
                elif isinstance(value, Tensor) and self.is_edge_attr(attr):
                    data[key][attr] = value[edge_ids[i]]

        # Add global attributes.
        keys = set(data.keys) | {'node_type', 'edge_type', 'num_nodes'}
        for attr, value in self.items():
            if attr in keys:
                continue
            if len(data.node_stores) == 1:
                data.node_stores[0][attr] = value
            else:
                data[attr] = value

        return data

    ###########################################################################

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]):
        r"""Creates a :class:`~torch_geometric.data.Data` object from a Python
        dictionary."""
        return cls(**mapping)

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the graph."""
        return self._store.num_node_features

    @property
    def device(self):
        return self.x.device

    @property
    def num_features(self) -> int:
        r"""Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`."""
        return self._store.num_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

    def __iter__(self) -> Iterable:
        r"""Iterates over all attributes in the data, yielding their attribute
        names and values."""
        for key, value in self._store.items():
            yield key, value

    def __call__(self, *args: List[str]) -> Iterable:
        r"""Iterates over all attributes :obj:`*args` in the data, yielding
        their attribute names and values.
        If :obj:`*args` is not given, will iterate over all attributes."""
        for key, value in self._store.items(*args):
            yield key, value

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def graph_x(self) -> Any:
        return self['graph_x'] if 'graph_x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def edge_attr(self) -> Any:
        return self['edge_attr'] if 'edge_attr' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def vdw_pot(self) -> Any:
        return self['vdw_pot'] if 'vdw_pot' in self._store else None
    @property
    def vdw_loss(self) -> Any:
        return self['vdw_loss'] if 'vdw_loss' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def radius(self) -> Any:
        return self['radius'] if 'radius' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    @property
    def aux_ind(self) -> Any:
        return self['aux_ind'] if 'aux_ind' in self._store else None

    @property
    def mol_ind(self) -> Any:
        return self['mol_ind'] if 'mol_ind' in self._store else None

    @property
    def sg_ind(self) -> Any:
        return self['sg_ind'] if 'sg_ind' in self._store else None

    @property
    def z_prime(self) -> Any:
        return self['z_prime'] if 'z_prime' in self._store else None

    @property
    def symmetry_operators(self) -> Any:
        return self['symmetry_operators'] if 'symmetry_operators' in self._store else None

    @property
    def cell_lengths(self) -> Any:
        return self['cell_lengths'] if 'cell_lengths' in self._store else None

    @property
    def cell_angles(self) -> Any:
        return self['cell_angles'] if 'cell_angles' in self._store else None

    @property
    def T_fc(self) -> Any:
        return self['T_fc'] if 'T_fc' in self._store else None

    @property
    def identifier(self) -> Any:
        return self['identifier'] if 'identifier' in self._store else None

    @property
    def cell_volume(self) -> Any:
        return self['cell_volume'] if 'cell_volume' in self._store else None

    @property
    def reduced_volume(self) -> Any:
        return self['reduced_volume'] if 'reduced_volume' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None

    @property
    def identifier(self) -> Any:
        return self['identifier'] if 'identifier' in self._store else None

    @property
    def unit_cell_pos(self) -> Any:
        return self['unit_cell_pos'] if 'unit_cell_pos' in self._store else None

    @property
    def aunit_handedness(self) -> Any:
        return self['aunit_handedness'] if 'aunit_handedness' in self._store else None

    @property
    def is_well_defined(self) -> Any:
        return self['is_well_defined'] if 'is_well_defined' in self._store else None

    @property
    def nonstandard_symmetry(self) -> Any:
        return self['nonstandard_symmetry'] if 'nonstandard_symmetry' in self._store else None

    # Deprecated functions ####################################################

    @property
    @deprecated(details="use 'data.face.size(-1)' instead")
    def num_faces(self) -> Optional[int]:
        r"""Returns the number of faces in the mesh."""
        if 'face' in self._store and isinstance(self.face, Tensor):
            return self.face.size(self.__cat_dim__('face', self.face))
        return None


###############################################################################


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = ' ' * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, SparseTensor):
        out = str(value.sizes())[:-1] + f', nnz={value.nnz()}]'
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = '{}'
    elif (isinstance(value, Mapping) and len(value) == 1
          and not isinstance(list(value.values())[0], Mapping)):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, BaseStorage):
        return f'{pad}\033[1m{key}\033[0m={out}'
    else:
        return f'{pad}{key}={out}'
