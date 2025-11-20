import copy
from collections.abc import Mapping, Sequence
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional, Union)

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor

from mxtaltools.constants.space_group_info import SPACE_GROUP_NAMES
from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import MolCrystalAnalysis
from mxtaltools.dataset_utils.data_class_methods.crystal_building import MolCrystalBuilding
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import MolCrystalOps
from mxtaltools.dataset_utils.data_class_methods.ellipsoid_ops import MolCrystalEllipsoidOps
from mxtaltools.dataset_utils.data_class_methods.mol_methods import MolDataMethods
from mxtaltools.dataset_utils.utils import collate_data_list


###############################################################################

# noinspection PyPropertyAccess


class MXtalBase(BaseData):
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
        return self.z.device

    @property
    def is_batch(self):
        return 'Batch' in self.__class__.__name__

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

    @staticmethod
    def to_batch(data_list):
        return collate_data_list(data_list)

    def add_graph_attr(self, values: torch.Tensor, name: str, slice_dict = None, inc_dict=None):
        """
        Attach a per-graph attribute to this Batch so it survives to_data_list().

        Args:
            values: tensor of shape [num_graphs, ...]
            name:   attribute name to assign
        """
        if not self.is_batch:
            raise TypeError("add_graph_attr only works on Batch objects")

        num_graphs = self.num_graphs
        assert values.size(0) == num_graphs, \
            f"{name} must have shape [num_graphs, ...], got {values.shape}"

        # Store attribute on the batch
        setattr(self, name, values)

        # Tell PyG how to split it back into Data objects
        if slice_dict is not None:
            self._slice_dict[name] = slice_dict
        else:
            self._slice_dict[name] = torch.arange(0, num_graphs + 1, 1, device=self.device)

        if inc_dict is not None:
            self._inc_dict[name] = inc_dict
        else:
            self._inc_dict[name] = torch.zeros(num_graphs, dtype=torch.long, device=self.device)

        return self

    def add_node_attr(self, values: torch.Tensor, name: str, num_nodes_per_graph):
        """
        Attach a per-node attribute to this Batch so it survives to_data_list().
        """
        if not self.is_batch:
            raise TypeError("add_node_attr only works on Batch objects")

        # concat all node tensors already; now define slices
        setattr(self, name, values)

        self._slice_dict[name] = torch.cumsum(
            torch.tensor([0] + list(num_nodes_per_graph)), dim=0
        ).to(torch.long).to(values.device)

        self._inc_dict[name] = torch.zeros(len(num_nodes_per_graph),
                                           dtype=torch.long, device=values.device)


    def batch_to_list(self):
        """
        Needed some custom logic to wrap around PyG default to_data_list
        :return:
        """
        assert self.is_batch
        zp = int(self.max_z_prime)
        is_well_defined = self.is_well_defined.clone()
        batch_to_listify = self.clone()
        del batch_to_listify.max_z_prime, batch_to_listify.is_well_defined

        samples_list = batch_to_listify.to_data_list()
        for ind, elem in enumerate(samples_list):
            elem.max_z_prime = zp
            elem.is_well_defined = is_well_defined[ind].unsqueeze(0)
        del batch_to_listify
        return samples_list


# noinspection PyPropertyAccess


class MolData(MXtalBase, MolDataMethods):
    r"""
    A graph representing a single molecule
    """

    def __init__(self,
                 z: Optional[torch.Tensor] = None,
                 pos: Optional[torch.Tensor] = None,
                 x: OptTensor = None,
                 graph_x: OptTensor = None,
                 y: OptTensor = None,
                 edge_index: OptTensor = None,
                 smiles: str = None,
                 identifier: str = None,
                 mol_volume: Optional[float] = None,
                 mass: Optional[float] = None,
                 radius: Optional[float] = None,
                 do_mol_analysis: Optional[bool] = False,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        # set attributes
        kwargs.update(
            z=z,
            pos=pos,
            x=x,
            y=y,
            graph_x=graph_x,
            edge_index=edge_index,
            smiles=smiles,
            identifier=identifier,
            mol_volume=mol_volume,
            mass=mass,
            radius=radius
        )
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        # only piece of analysis for init
        if self.z is not None and not hasattr(self, 'num_atoms'):
            self.num_atoms = torch.tensor(len(z), dtype=torch.long)

        if do_mol_analysis:
            self.mol_analysis()

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def z(self) -> Any:
        return self['z'] if 'z' in self._store else None

    @property
    def graph_x(self) -> Any:
        return self['graph_x'] if 'graph_x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def radius(self) -> Any:
        return self['radius'] if 'radius' in self._store else None

    @property
    def mass(self) -> Any:
        return self['mass'] if 'mass' in self._store else None

    @property
    def mol_volume(self) -> Any:
        return self['mol_volume'] if 'mol_volume' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    @property
    def ptr(self) -> Any:
        return self['ptr'] if 'ptr' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None


# noinspection PyPropertyAccess
class MolCrystalData(  # order crystal ops first to overwrite any mol ops
    MolCrystalOps,
    MolCrystalBuilding,
    MolCrystalAnalysis,
    MolCrystalEllipsoidOps,
    MolData,
):
    r"""
    A data object representing a molecular crystal with Z prime = 1 (exactly one molecule in asymmetric unit)
    """

    def __init__(self,
                 molecule: Optional[Union[list,MolData]] = None,
                 sg_ind: Optional[Union[int, torch.Tensor]] = None,
                 sg_name: Optional[str] = None,
                 z_prime: Optional[torch.Tensor] = None,
                 cell_lengths: Optional[torch.Tensor] = None,
                 cell_angles: Optional[torch.Tensor] = None,
                 aunit_centroid: Optional[torch.Tensor] = None,
                 aunit_orientation: Optional[torch.Tensor] = None,
                 aunit_handedness: Optional[Union[torch.Tensor, int]] = None,
                 identifier: Optional[str] = None,
                 nonstandard_symmetry: Optional[bool] = False,
                 symmetry_operators: Optional[list] = None,
                 is_well_defined: Optional[bool] = True,
                 aux_ind: Optional[torch.Tensor] = None,
                 mol_ind: Optional[torch.Tensor] = None,
                 do_box_analysis: Optional[bool] = False,
                 max_z_prime: Optional[int] = 1,
                 **kwargs):
        """
        Initialize a molecular crystal from an existing molecule
        Does natively enforce physicality or symmetry correctness.
        Do clean_cell_parameters and/or box_analysis for further processing.
        """
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        if molecule is not None:
            self.set_mol_attrs(molecule)

        kwargs.update(
            identifier=identifier,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
            aux_ind=aux_ind,
            mol_ind=mol_ind,
            is_well_defined=is_well_defined,
            max_z_prime=max_z_prime,
        )

        if aunit_centroid is not None:
            self.aunit_centroid = self._pad_tensor(
                aunit_centroid, (1, 3*max_z_prime), 0)
        if aunit_orientation is not None:
            self.aunit_orientation = self._pad_tensor(
                aunit_orientation, (1, 3*max_z_prime), 1)
        if aunit_handedness is not None:
            self.aunit_handedness = self._pad_tensor(
                aunit_handedness, (1, max_z_prime), 1)

        # set attrs
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        # functions / analysis
        # will raise error if sg_ind is None, but then, it can't be a real crystal
        if sg_ind is not None or sg_name is not None:
            if sg_name is not None:
                valid_sgs = list(SPACE_GROUP_NAMES.keys())
                if sg_name in valid_sgs:
                    sg_ind = SPACE_GROUP_NAMES[sg_name]
                else:
                    raise ValueError(f"{sg_name} is not in the set of canonical space group names!"
                                     f"{valid_sgs}")
            self.set_symmetry_attrs(nonstandard_symmetry=nonstandard_symmetry,
                                    sg_ind=sg_ind,
                                    symmetry_operators=symmetry_operators,
                                    z_prime=z_prime)

        # cell parameters need a leading dimension [n, 3]
        # will raise error if self.cell_lengths is None, but then, it can't be a real crystal
        if self.cell_lengths is not None:
            if self.cell_lengths.ndim == 1:
                self.cell_lengths = self.cell_lengths[None, ...]
            if self.cell_angles.ndim == 1:
                self.cell_angles = self.cell_angles[None, ...]
            if self.aunit_centroid.ndim == 1:
                self.aunit_centroid = self.aunit_centroid[None, ...]
            if self.aunit_orientation.ndim == 1:
                self.aunit_orientation = self.aunit_orientation[None, ...]
            if self.aunit_handedness.ndim == 1:
                self.aunit_handedness = self.aunit_handedness[None, ...]

        if do_box_analysis:
            self.box_analysis()

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def z(self) -> Any:
        return self['z'] if 'z' in self._store else None

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
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def radius(self) -> Any:
        return self['radius'] if 'radius' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    @property
    def sg_ind(self) -> Any:
        return self['sg_ind'] if 'sg_ind' in self._store else None

    @property
    def symmetry_operators(self) -> Any:
        return self['symmetry_operators'] if 'symmetry_operators' in self._store else None

    @property
    def cell_lengths(self) -> Any:
        return self['cell_lengths'] if 'cell_lengths' in self._store else None

    @property
    def z_prime(self) -> Any:
        return self['z_prime'] if 'z_prime' in self._store else None

    @property
    def cell_angles(self) -> Any:
        return self['cell_angles'] if 'cell_angles' in self._store else None


    @property
    def T_fc(self) -> Any:
        return self['T_fc'] if 'T_fc' in self._store else None

    @property
    def T_cf(self) -> Any:
        return self['T_cf'] if 'T_cf' in self._store else None

    @property
    def identifier(self) -> Any:
        return self['identifier'] if 'identifier' in self._store else None

    @property
    def mol_ind(self) -> Any:
        return self['mol_ind'] if 'mol_ind' in self._store else None

    @property
    def aux_ind(self) -> Any:
        return self['aux_ind'] if 'aux_ind' in self._store else None

    @property
    def cell_volume(self) -> Any:
        return self['cell_volume'] if 'cell_volume' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None

    @property
    def packing_coeff(self) -> Any:
        return self['packing_coeff'] if 'packing_coeff' in self._store else None

    @property
    def density(self) -> Any:
        return self['density'] if 'density' in self._store else None

    @property
    def unit_cell_pos(self) -> Any:
        return self['unit_cell_pos'] if 'unit_cell_pos' in self._store else None

    @property
    def unit_cell_batch(self) -> Any:
        return self['unit_cell_batch'] if 'unit_cell_pos' in self._store else None

    @property
    def unit_cell_mol_ind(self) -> Any:
        return self['unit_cell_mol_ind'] if 'unit_cell_pos' in self._store else None

    @property
    def aunit_handedness(self) -> Any:
        return self['aunit_handedness'] if 'aunit_handedness' in self._store else None

    @property
    def is_well_defined(self) -> Any:
        return self['is_well_defined'] if 'is_well_defined' in self._store else None

    @property
    def nonstandard_symmetry(self) -> Any:
        return self['nonstandard_symmetry'] if 'nonstandard_symmetry' in self._store else None

    @property
    def cocrystal(self) -> Any:
        return self['cocrystal'] if 'cocrystal' in self._store else None

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
