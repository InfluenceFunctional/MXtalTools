import copy
from collections.abc import Mapping, Sequence
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.deprecation import deprecated
from torch_geometric.typing import EdgeType, NodeType, OptTensor
from torch_geometric.utils import subgraph
from torch_geometric.data.data import BaseData


###############################################################################


class CrystalData(BaseData):
    r"""A data object describing a homogeneous graph.
    The data object can hold node-level, link-level and graph-level attributes.
    In general, :class:`~torch_geometric.data.Data` tries to mimic the
    behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    .. code-block:: python

        from torch_geometric.data import Data

        data = Data(x=x, edge_index=edge_index, ...)

        # Add additional arguments to `data`:
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)

        # Analyzing the graph structure:
        data.num_nodes
        >>> 23

        data.is_directed()
        >>> False

        # PyTorch tensor functionality:
        data = data.pin_memory()
        data = data.to('cuda:0', non_blocking=True)

    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None,
                 edge_attr: OptTensor = None, y: OptTensor = None,
                 pos: OptTensor = None, mult: OptTensor = None,
                 mol_x: OptTensor = None,
                 tracking: OptTensor = None, sg_ind: OptTensor = None,
                 smiles: OptTensor = None, ref_cell_pos: OptTensor = None,
                 cell_params: OptTensor = None, T_fc: OptTensor = None,
                 aux_ind: OptTensor = None, mol_size: OptTensor = None,
                 csd_identifier: OptTensor = None, mol_volume: OptTensor = None,
                 asym_unit_handedness: OptTensor = None, symmetry_operators: OptTensor = None,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if mol_x is not None:
            self.mol_x = mol_x
        if pos is not None:
            self.pos = pos
        if mult is not None:
            self.mult = mult
        if sg_ind is not None:
            self.sg_ind = sg_ind
        if tracking is not None:
            self.tracking = tracking
        if smiles is not None:
            self.smiles = smiles
        if ref_cell_pos is not None:
            self.ref_cell_pos = ref_cell_pos
        if cell_params is not None:
            self.cell_params = cell_params
        if T_fc is not None:
            self.T_fc = T_fc
        if aux_ind is not None:
            self.aux_ind = aux_ind
        if mol_size is not None:
            self.mol_size = mol_size
        if csd_identifier is not None:
            self.csd_identifier = csd_identifier
        if mol_volume is not None:
            self.mol_volume = mol_volume
        if asym_unit_handedness is not None:
            self.asym_unit_handedness = asym_unit_handedness
        if symmetry_operators is not None:
            self.symmetry_operators = symmetry_operators


        for key, value in kwargs.items():
            setattr(self, key, value)

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
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    '''
    custom crystal features
    '''
    @property
    def mol_x(self) -> Any:
        return self['mol_x'] if 'mol_x' in self._store else None

    @property
    def mult(self) -> Any:
        return self['mult'] if 'mult' in self._store else None

    @property
    def sg_ind(self) -> Any:
        return self['sg_ind'] if 'sg_ind' in self._store else None

    @property
    def tracking(self) -> Any:
        return self['tracking'] if 'tracking' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None

    @property
    def ref_cell_pos(self) -> Any:
        return self['ref_cell_pos'] if 'ref_cell_pos' in self._store else None

    @property
    def cell_params(self) -> Any:
        return self['cell_params'] if 'cell_params' in self._store else None

    @property
    def T_fc(self) -> Any:
        return self['T_fc'] if 'T_fc' in self._store else None

    @property
    def aux_ind(self) -> Any:
        return self['aux_ind'] if 'aux_ind' in self._store else None

    @property
    def mol_size(self) -> Any:
        return self['mol_size'] if 'mol_size' in self._store else None

    @property
    def csd_identifier(self) -> Any:
        return self['csd_identifier'] if 'csd_identifier' in self._store else None

    @property
    def mol_volume(self) -> Any:
        return self['mol_volume'] if 'mol_volume' in self._store else None

    @property
    def asym_unit_handedness(self) -> Any:
        return self['asym_unit_handedness'] if 'asym_unit_handedness' in self._store else None

    @property
    def symmetry_operators(self) -> Any:
        return self['symmetry_operators'] if 'symmetry_operators' in self._store else None


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
