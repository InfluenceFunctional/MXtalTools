import copy
from collections.abc import Mapping, Sequence
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional)

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.nn import radius_graph
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor

from mxtaltools.common.geometry_utils import cell_parameters_to_box_vectors, batch_molecule_vdW_volume, \
    compute_mol_radius, batch_compute_mol_radius, batch_compute_mol_mass, compute_mol_mass, get_batch_centroids
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.constants.space_group_info import SYM_OPS, LATTICE_TYPE
from mxtaltools.crystal_building.utils import get_aunit_positions, new_aunit2unit_cell, parameterize_crystal_batch


###############################################################################


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


# noinspection PyPropertyAccess
class MolData(MXtalBase):
    r"""
    A graph representing a single molecule
    """

    def __init__(self,
                 z: Optional[torch.LongTensor] = None,
                 pos: Optional[torch.FloatTensor] = None,
                 x: OptTensor = None,
                 graph_x: OptTensor = None,

                 y: OptTensor = None,

                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 smiles: str = None,
                 identifier: str = None,
                 skip_mol_analysis: bool = True,

                 mol_volume: Optional[float] = None,
                 mass: Optional[float] = None,
                 radius: Optional[float] = None,

                 construct_radial_graph: bool = False,

                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # fix node & graph attributes
        if z is not None:
            self.z = z
            self.num_atoms = len(z)
        if x is not None:
            self.x = x
            assert len(self.x) == len(self.z), "Number of atom types and number of node features are not equal"

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

        # fix identifiers
        if smiles is not None:
            self.smiles = smiles
        if identifier is not None:
            self.identifier = identifier

        if not skip_mol_analysis:  # expensive - use only if lazy # todo add log/warning for this
            if radius is None:
                self.radius = self.radius_calculation()
            if mass is None:
                self.mass = self.mass_calculation()
            if mol_volume is None:
                self.mol_volume = self.volume_calculation()

        if construct_radial_graph:
            self.edge_index = self.construct_radial_graph()

    def construct_radial_graph(self, cutoff: float = 7):
        if 'Batch' in self.__class__.__name__:
            return radius_graph(self.pos,
                                batch=self.batch,
                                r=cutoff,
                                max_num_neighbors=100,
                                flow='source_to_target')
        else:  # if we do this before batching, these will be all wrong
            return radius_graph(self.pos,
                                r=cutoff,
                                max_num_neighbors=100,
                                flow='source_to_target')  # note - requires batch be monotonically increasing

    def radius_calculation(self):
        if 'Batch' in self.__class__.__name__:
            return batch_compute_mol_radius(self.pos, self.batch, self.num_graphs, self.num_atoms)
        else:
            return compute_mol_radius(self.pos)

    def mass_calculation(self):
        masses_tensor = torch.FloatTensor(list(ATOM_WEIGHTS.values()), device=self.z.device)
        if 'Batch' in self.__class__.__name__:
            return batch_compute_mol_mass(self.z, self.batch, masses_tensor, self.num_graphs)
        else:
            return compute_mol_mass(self.z, masses_tensor)

    def volume_calculation(self):
        vdw_radii_tensor = torch.FloatTensor(list(VDW_RADII.values()), device=self.z.device)
        if 'Batch' in self.__class__.__name__:
            return batch_molecule_vdW_volume(
                self.z,
                self.pos,
                self.batch,
                self.num_graphs,
                vdw_radii_tensor)
        else:
            return batch_molecule_vdW_volume(
                self.z,
                self.pos,
                torch.zeros_like(self.z),
                1,
                vdw_radii_tensor)[0]

    def recenter_molecules(self):
        if 'Batch' in self.__class__.__name__:
            centroids = get_batch_centroids(self.pos, self.batch, self.num_graphs)
            self.pos -= centroids.repeat_interleave(self.num_atoms, 0)
        else:
            self.pos -= self.pos.mean(0)

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
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None


# noinspection PyPropertyAccess
class MolCrystalData(MXtalBase):
    r"""
    A data object representing a molecular crystal with Z prime = 1 (exactly one molecule in asymmetric unit)
    """

    def __init__(self,
                 molecule: Optional[MolData] = None,
                 sg_ind: Optional[int] = None,
                 cell_lengths: Optional[torch.FloatTensor] = None,
                 cell_angles: Optional[torch.FloatTensor] = None,
                 aunit_centroid: Optional[torch.FloatTensor] = None,
                 aunit_orientation: Optional[torch.FloatTensor] = None,
                 aunit_handedness: Optional[torch.BoolTensor] = None,
                 identifier: Optional[str] = None,
                 unit_cell_pos: Optional[np.ndarray] = None,
                 nonstandard_symmetry: Optional[bool] = False,
                 symmetry_operators: Optional[list] = None,
                 is_well_defined: Optional[bool] = True,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)
        # initialize crystal from an existing molecule
        if molecule is not None:
            mol_dict = molecule.to_dict()
            for key, value in mol_dict.items():
                setattr(self, key, value)

        # then overwrite any relevant features
        for key, value in kwargs.items():
            setattr(self, key, value)

        # and add crystal features
        if identifier is not None:
            self.identifier = identifier
        if sg_ind is not None:
            self.sg_ind = sg_ind
            if nonstandard_symmetry:  # set as np stack for correct collation behavior (we don't want batches to stack)
                if symmetry_operators is not None:
                    self.symmetry_operators = symmetry_operators
                else:
                    assert False, "symmetry_operators must be given for nonstandard symmetry operations"
                self.nonstandard_symmetry = True
            else:  # standard symmetry
                self.symmetry_operators = np.stack(SYM_OPS[sg_ind])  # if saved as a tensor, we get collation issues
                self.nonstandard_symmetry = False

            self.sym_mult = torch.ones(1, dtype=torch.long, device=self.device) * len(self.symmetry_operators)
            self.is_well_defined = is_well_defined

            # record prebuilt unit cell coordinates
            if unit_cell_pos is not None:
                self.unit_cell_pos = unit_cell_pos.cpu().detach().numpy()  # if it's saved as a tensor, we get problems in collation
                assert unit_cell_pos.shape == (self.sym_mult, self.num_nodes, 3)
            else:  # make a placeholder
                self.unit_cell_pos = np.zeros((self.sym_mult, self.num_nodes, 3))

        # cell parameters
        if cell_lengths is not None:
            self.cell_lengths = cell_lengths[None, ...]
            self.cell_angles = cell_angles[None, ...]
            self.aunit_centroid = aunit_centroid[None, ...]
            self.aunit_orientation = aunit_orientation[None, ...]
            self.aunit_handedness = aunit_handedness

            if self.T_fc is None:  # better to do this in batches and feed it as kwargs # todo add a log/warning for this
                assert (
                        self.T_cf is None and self.cell_volume is None), "T_fc, T_cf, and cell volume must all be provided all together or not at all"

                self.T_fc, self.cell_volume = (
                    cell_parameters_to_box_vectors('f_to_c', cell_lengths, cell_angles, return_vol=True))

                if self.T_fc.ndim == 2:
                    self.T_fc = self.T_fc[None, ...]

                self.T_fc = self.T_fc
                self.T_cf = torch.linalg.inv(self.T_fc)
            else:
                assert (
                        self.T_cf is not None and self.cell_volume is not None), "T_fc, T_cf, and cell volume must all be provided all together or not at all"

            self.packing_coeff = self.mol_volume * self.sym_mult / self.cell_volume

    def denorm_cell_lengths(self, normed_cell_length):
        """
        abc = (Z*Vol_m)^(1/3) * abc_normed
        """
        if 'Batch' in self.__class__.__name__:
            return torch.pow(self.sym_mult * self.mol_volume, 1 / 3)[:, None] * normed_cell_length
        else:
            return torch.pow(self.sym_mult * self.mol_volume, 1 / 3)[None] * normed_cell_length

    def norm_cell_lengths(self):
        """
        abc_normed = abc / (Z*Vol_m)^(1/3)
        """
        if 'Batch' in self.__class__.__name__:
            return self.cell_lengths / torch.pow(self.sym_mult * self.mol_volume, 1 / 3)[:, None]
        else:
            return self.cell_lengths / torch.pow(self.sym_mult * self.mol_volume, 1 / 3)[None]

    def scale_centroid_to_aunit(self):
        """
        input fractional coordinates are scaled on 0-max
        rescale these for the specific ranges according to each space group
        only space groups in asym_unit_dict will work - not all have been manually encoded
        this approach will not work for asymmetric units which are not neat parallelpipeds
        Parameters
        ----------
        asym_unit_dict
        mol_position
        sg_inds

        Returns
        -------
        """
        asym_unit_dict = ASYM_UNITS.copy()
        for key in asym_unit_dict:
            asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(self.device)

        if 'Batch' in self.__class__.__name__:
            return self.aunit_centroid / torch.stack([asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return self.aunit_centroid / asym_unit_dict[str(int(self.sg_ind))]

    def scale_centroid_to_unit_cell(self, normed_centroid):
        """
        input fractional coordinates are scaled on 0-1
        rescale these for the specific ranges according to each space group
        only space groups in asym_unit_dict will work - not all have been manually encoded
        this approach will not work for asymmetric units which are not neat parallelpipeds
        Parameters
        ----------
        asym_unit_dict
        mol_position
        sg_inds

        Returns
        -------
        """
        asym_unit_dict = ASYM_UNITS.copy()
        for key in asym_unit_dict:
            asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key]).to(self.device)

        if 'Batch' in self.__class__.__name__:
            return normed_centroid * torch.stack([asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return normed_centroid * asym_unit_dict[str(int(self.sg_ind))]

    def compute_normed_cell_parameters(self):
        return torch.cat(
            [self.norm_cell_lengths(), self.cell_angles,
             self.scale_centroid_to_aunit(), self.aunit_orientation],
            dim=-1
        )

    def compute_denormed_cell_parameters(self, normed_cell_parameters):
        return torch.cat(
            [self.denorm_cell_lengths(normed_cell_parameters[..., :3]),
             self.cell_angles,
             self.scale_centroid_to_unit_cell(normed_cell_parameters[..., 6:9]),
             self.aunit_orientation],
            dim=-1
        )

    def standardize_box_parameters(self,
                                   cell_means: OptTensor = None,
                                   cell_stds: OptTensor = None):
        if cell_means is None:
            means = torch.tensor(  # todo replace by call to constants
                [1.0411, 1.1640, 1.4564, 1.5619, 1.5691, 1.5509],  # use triclinic
                dtype=torch.float32)

        else:
            means = cell_means
        if cell_stds is None:
            stds = torch.tensor(
                [0.3846, 0.4280, 0.4864, 0.2363, 0.2046, 0.2624],
                dtype=torch.float32)
        else:
            stds = cell_stds
        return (torch.cat([self.cell_lengths, self.cell_angles], dim=-1) - means) / stds

    def destandardize_box_parameters(self,
                                     cell_means: OptTensor = None,
                                     cell_stds: OptTensor = None):
        if cell_means is None:
            means = torch.tensor(
                [1.0411, 1.1640, 1.4564, 1.5619, 1.5691, 1.5509],
                dtype=torch.float32)

        else:
            means = cell_means
        if cell_stds is None:
            stds = torch.tensor(
                [0.3846, 0.4280, 0.4864, 0.2363, 0.2046, 0.2624],
                dtype=torch.float32)
        else:
            stds = cell_stds
        return torch.cat([self.cell_lengths, self.cell_angles], dim=-1) * stds + means

    def pose_aunit(self):
        if 'Batch' in self.__class__.__name__:
            self.pos = get_aunit_positions(
                self,
                align_to_standardized_orientation=True,
                mol_handedness=self.aunit_handedness,
            )
        else:  # TODO
            assert False, "aunit posing not yet implemented for single crystals"

    def build_unit_cell(self):
        if 'Batch' in self.__class__.__name__:
            self.unit_cell_pos = new_aunit2unit_cell(self)  # keep in numpy format
        else:  # TODO
            assert False, "unit cell construction not yet implemented for single crystals"

    def crystal_system(self):
        if 'Batch' in self.__class__.__name__:
            return [LATTICE_TYPE[int(ind)] for ind in self.sg_ind]
        else:
            return LATTICE_TYPE[int(self.sg_ind)]

    def validate_cell_params(self, check_crystal_system: bool = True):
        """
        checking crystal system slightly slower, can optionally skip
        """
        self.validate_cell_params_ranges()
        if check_crystal_system:
            self.validate_crystal_system()
        return True

    def validate_cell_params_ranges(self):
        # assert valid ranges
        assert torch.all(self.aunit_centroid >= 0), "Aunit centroids must be greater than 0"
        assert torch.all(self.aunit_centroid <= 1), "Aunit centroids must be less than 1"
        assert torch.all(self.cell_lengths > 0), "Cell lengths must be positive"
        assert torch.all(self.cell_angles > 0), "Cell angles must be greater than 0"
        assert torch.all(self.cell_angles < torch.pi), "Cell angles must be less than pi"
        assert torch.all(torch.linalg.norm(self.aunit_orientation,
                                           dim=-1) <= 2 * torch.pi), "Cell orientation rotvec must have length <=2pi"
        assert torch.all(
            torch.linalg.norm(self.aunit_orientation, dim=-1) >= 0), "Cell orientation rotvec must have length >= 0"
        assert torch.all(self.aunit_orientation[:, -1] >= 0), "Cell orientation rotvec z component must be positive"

    def validate_crystal_system(self):
        lattices = self.crystal_system()
        # enforce agreement with crystal system
        if "Batch" in self.__class__.__name__:
            for ind in range(self.num_graphs):
                lattice = lattices[ind]
                cell_lengths = self.cell_lengths[ind]
                cell_angles = self.cell_angles[ind]
                if lattice.lower() == 'triclinic':
                    pass
                elif lattice.lower() == 'monoclinic':  # fix alpha and gamma
                    assert cell_angles[0] == torch.pi/2, "Error in monoclinic alpha angle"
                    assert cell_angles[2] == torch.pi / 2, "Error in monoclinic gamma angle"
                elif lattice.lower() == 'orthorhombic':  # fix all angles
                    assert cell_angles == torch.ones(3) * torch.pi / 2, "Error in orthorhombic cell angles"
                elif lattice.lower() == 'tetragonal':  # fix all angles and a & b vectors
                    assert cell_angles == torch.ones(3) * torch.pi / 2, "Error in tetragonal cell angles"
                    assert cell_lengths[0] == cell_lengths[1], "Error in tetragonal cell lengths"
                elif lattice.lower() == 'hexagonal':  # for rhombohedral, all angles and lengths equal, but not 90.
                    # for truly hexagonal, alpha=90, gamma is 120, a=b!=c
                    # todo implement 3&6 fold lattices
                    print('hexagonal lattice checks are not implemented!')
                    pass
                elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
                    assert torch.all(cell_lengths == cell_lengths.mean()), "Error in cubic cell lengths"
                    assert torch.all(cell_angles == torch.pi /2), "Error in cubic cell angles"
                else:
                    assert False, f"{lattice} + ' is not a valid crystal lattice!')"

        return True

    def reparameterize_unit_cell(self):
        mol_position_list, rotvec_list, handedness_list, well_defined_asym_unit_list, canonical_conformer_coords_list =(
            parameterize_crystal_batch(self,
                                       ASYM_UNITS,
                                       enforce_right_handedness=False,
                                       return_aunit=True))

        d1 = (self.aunit_centroid - mol_position_list).abs().sum(1)
        d2 = (rotvec_list - self.aunit_orientation).abs().sum(1)
        print(torch.vstack((d1, d2, torch.Tensor(well_defined_asym_unit_list))).T)
        aa = 1

    def cell_parameters(self):
        """
        return the zp=1 total cell parameter tensor
        Returns
        -------

        """
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.aunit_centroid,
                          self.aunit_orientation], dim=1)

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
    def cell_volume(self) -> Any:
        return self['cell_volume'] if 'cell_volume' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None

    @property
    def packing_coeff(self) -> Any:
        return self['packing_coeff'] if 'packing_coeff' in self._store else None

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


class MolClusterData(MolCrystalData):
    r"""
    A data object representing a molecular cluster, based on a preexisting molecular crystal lattice
    """

    def __init__(self,
                 crystal: Optional[MolCrystalData] = None,
                 aux_ind: Optional[torch.LongTensor] = None,
                 mol_ind: Optional[torch.LongTensor] = None,
                 cluster_z: Optional[torch.LongTensor] = None,
                 cluster_pos: Optional[torch.FloatTensor] = None,
                 cluster_x: OptTensor = None,
                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)
        # initialize cluster from an existing crystal
        if crystal is not None:
            crystal_dict = crystal.to_dict()
            for key, value in crystal_dict.items():
                setattr(self, key, value)

        # then overwrite any relevant features
        for key, value in kwargs.items():
            setattr(self, key, value)

        # overwrite molecule features as cluster features
        if cluster_z is not None:
            for key in ['pos', 'z', 'edge_index', 'edge_attr', 'x', 'mol_ind', 'aux_ind']:
                if hasattr(self, key):
                    delattr(self, key)

            setattr(self, 'pos', cluster_pos)
            setattr(self, 'z', cluster_z)
            setattr(self, 'aux_ind', aux_ind)
            setattr(self, 'mol_ind', mol_ind)

            assert len(self.mol_ind) == len(self.aux_ind) == self.num_nodes
            assert self.num_nodes % self.num_atoms == 0, "Cluster must comprise N whole molecules"

        if edge_index is not None:
            setattr(self, 'edge_index', edge_index)
        if edge_attr is not None:
            setattr(self, 'edge_attr', edge_attr)
        if cluster_x is not None:
            setattr(self, 'x', cluster_x)

    def cell_parameters(self):
        """
        return the zp=1 total cell parameter tensor
        Returns
        -------

        """
        return torch.cat([self.cell_lengths,
                          self.cell_angles,
                          self.aunit_centroid,
                          self.aunit_orientation], dim=1)

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
    def aux_ind(self) -> Any:
        return self['aux_ind'] if 'aux_ind' in self._store else None

    @property
    def mol_ind(self) -> Any:
        return self['mol_ind'] if 'mol_ind' in self._store else None

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
    def cell_angles(self) -> Any:
        return self['cell_angles'] if 'cell_angles' in self._store else None

    @property
    def T_fc(self) -> Any:
        return self['T_fc'] if 'T_fc' in self._store else None

    def T_cf(self) -> Any:
        return self['T_cf'] if 'T_cf' in self._store else None

    @property
    def identifier(self) -> Any:
        return self['identifier'] if 'identifier' in self._store else None

    @property
    def cell_volume(self) -> Any:
        return self['cell_volume'] if 'cell_volume' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None

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
