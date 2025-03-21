import copy
from collections.abc import Mapping, Sequence
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional)

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor

from mxtaltools.analysis.crystals_analysis import get_intermolecular_dists_dict
from mxtaltools.analysis.vdw_analysis import vdw_analysis, electrostatic_analysis
from mxtaltools.common.geometry_utils import batch_compute_molecule_volume, \
    compute_mol_radius, batch_compute_mol_radius, batch_compute_mol_mass, compute_mol_mass, center_mol_batch, \
    apply_rotation_to_batch, enforce_crystal_system, batch_compute_fractional_transform
from mxtaltools.common.utils import softplus_shift
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.constants.space_group_info import SYM_OPS, LATTICE_TYPE
from mxtaltools.crystal_building.utils import get_aunit_positions, new_aunit2unit_cell, parameterize_crystal_batch, \
    unit_cell_to_supercell_cluster, align_mol_batch_to_standard_axes, canonicalize_rotvec
from mxtaltools.dataset_utils.synthesis.mol_building import smiles2conformer
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.modules.components import construct_radial_graph
from mxtaltools.models.utils import enforce_1d_bound, get_mol_embedding_for_proxy


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


# noinspection PyPropertyAccess


class MolData(MXtalBase):
    r"""
    A graph representing a single molecule
    """

    def __init__(self,
                 z: Optional[torch.LongTensor] = None,
                 pos: Optional[torch.Tensor] = None,
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
            self.construct_radial_graph()

    @classmethod
    def from_smiles(cls,
                    smiles: str,
                    protonate: bool = True,
                    minimize: bool = False,
                    scramble_dihedrals: bool = False,
                    allow_methyl_rotations: bool = False,
                    skip_mol_analysis: bool = False,
                    compute_partial_charges: bool = True,
                    pare_to_size: Optional[int] = None,
                    max_pare_iters: int = 10,
                    ):
        assert "Batch" not in cls.__name__, "sample generation from smiles only implemented for single samples"
        conf_out = smiles2conformer(allow_methyl_rotations,
                                    compute_partial_charges,
                                    max_pare_iters,
                                    minimize,
                                    pare_to_size,
                                    protonate,
                                    scramble_dihedrals,
                                    smiles)
        if conf_out is not None:
            charges, pos, z = conf_out
            return cls(
                z=torch.LongTensor(z),
                pos=torch.tensor(pos, dtype=torch.float32),
                x=torch.tensor(charges, dtype=torch.float32),
                smiles=smiles,
                identifier=smiles,
                skip_mol_analysis=skip_mol_analysis,
                y=None,
                graph_x=None,
            )
        else:
            return None

    def construct_radial_graph(self, cutoff: float = 6):
        self.edges_dict = construct_radial_graph(self.pos,
                                                 self.batch,
                                                 self.ptr,
                                                 cutoff,
                                                 max_num_neighbors=10000
                                                 )
        self.edge_index = self.edges_dict['edge_index']

    def radius_calculation(self):
        if 'Batch' in self.__class__.__name__:
            return batch_compute_mol_radius(self.pos, self.batch, self.num_graphs, self.num_atoms)
        else:
            return compute_mol_radius(self.pos)

    def mass_calculation(self):
        masses_tensor = torch.tensor(list(ATOM_WEIGHTS.values()), device=self.z.device, dtype=torch.float32)
        if 'Batch' in self.__class__.__name__:
            return batch_compute_mol_mass(self.z, self.batch, masses_tensor, self.num_graphs)
        else:
            return compute_mol_mass(self.z, masses_tensor)

    def volume_calculation(self):
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()),
                                        device=self.z.device,
                                        dtype=torch.float32)
        if 'Batch' in self.__class__.__name__:
            return batch_compute_molecule_volume(
                self.z,
                self.pos,
                self.batch,
                self.num_graphs,
                vdw_radii_tensor)
        else:
            return batch_compute_molecule_volume(
                self.z,
                self.pos,
                torch.zeros_like(self.z),
                1,
                vdw_radii_tensor)[0]

    def recenter_molecules(self):
        if 'Batch' in self.__class__.__name__:
            self.pos = center_mol_batch(self.pos, self.batch, self.num_graphs, self.num_atoms)
        else:
            self.pos -= self.pos.mean(0)

    def noise_positions(self, magnitude: float):
        self.pos += torch.randn_like(self.pos) * magnitude

    def scale_positions(self, affine_scale: float):
        graph_scale_factor = torch.randn(self.num_graphs, device=self.pos.device, dtype=torch.float32)
        graph_scale_factor = (graph_scale_factor / 10 + affine_scale).clip(min=0.9, max=1.25)
        atomwise_scaling = graph_scale_factor.repeat_interleave(self.num_atoms)
        self.pos *= atomwise_scaling[:, None]

    def visualize(self,
                  sample_inds: Optional[list] = None,
                  **vis_kwargs):
        from mxtaltools.common.ase_interface import data_batch_to_ase_mols_list
        data_batch_to_ase_mols_list(
            self,
            specific_inds=sample_inds,
            show_mols=True,
            **vis_kwargs)

    def orient_molecule(self, mode: str,
                        include_inversion: bool = True,
                        target_handedness: Optional[torch.LongTensor] = None,
                        ):
        if 'Batch' in self.__class__.__name__:
            # always center the molecules
            self.recenter_molecules()
            if mode == 'standardized':
                mol_batch = align_mol_batch_to_standard_axes(self, handedness=target_handedness)
                self.pos = mol_batch.pos
            elif mode == 'random':
                random_rotations = torch.tensor(
                    Rotation.random(num=self.num_graphs).as_matrix(),
                    device=self.device, dtype=torch.float32)
                if include_inversion:
                    invert_inds = torch.randint(2, (self.num_graphs,)) * 2 - 1
                    random_rotations *= invert_inds[:, None, None]

                self.pos = apply_rotation_to_batch(
                    self.pos,
                    random_rotations,
                    self.batch)
            else:
                pass

        else:
            assert False, "molecule orientation not implemented for single samples"

    def deprotonate(self):
        """danger - this breaks several batching methods and should be used carefully
        """
        heavy_atom_inds = torch.argwhere(self.z != 1).flatten()  # protons are atom type 1
        self.z = self.z[heavy_atom_inds]
        self.pos = self.pos[heavy_atom_inds]
        self.batch = self.batch[heavy_atom_inds]
        a, b = torch.unique(self.batch, return_counts=True)
        self.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
        self.num_atoms = torch.diff(self.ptr).long()
        self.num_nodes = len(self.z)

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
    def ptr(self) -> Any:
        return self['ptr'] if 'ptr' in self._store else None

    @property
    def smiles(self) -> Any:
        return self['smiles'] if 'smiles' in self._store else None


# noinspection PyPropertyAccess
class MolCrystalData(MolData):
    r"""
    A data object representing a molecular crystal with Z prime = 1 (exactly one molecule in asymmetric unit)
    """

    def __init__(self,
                 molecule: Optional[MolData] = None,
                 sg_ind: Optional[int] = None,
                 cell_lengths: Optional[torch.Tensor] = None,
                 cell_angles: Optional[torch.Tensor] = None,
                 aunit_centroid: Optional[torch.Tensor] = None,
                 aunit_orientation: Optional[torch.Tensor] = None,
                 aunit_handedness: Optional[int] = None,
                 identifier: Optional[str] = None,
                 unit_cell_pos: Optional[np.ndarray] = None,
                 nonstandard_symmetry: Optional[bool] = False,
                 symmetry_operators: Optional[list] = None,
                 is_well_defined: Optional[bool] = True,
                 aux_ind: Optional[torch.LongTensor] = None,
                 mol_ind: Optional[torch.LongTensor] = None,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)
        # initialize crystal from an existing molecule
        if molecule is not None:
            # copy out the data
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
                if torch.is_tensor(unit_cell_pos):
                    self.unit_cell_pos = unit_cell_pos.cpu().detach().numpy()  # if it's saved as a tensor, we get problems in collation
                else:
                    self.unit_cell_pos = unit_cell_pos

                assert unit_cell_pos.shape == (self.sym_mult, self.num_nodes, 3)
            else:  # make a placeholder
                self.unit_cell_pos = np.zeros((self.sym_mult, self.num_nodes, 3))

        # cell parameters
        if cell_lengths is not None:
            self.cell_lengths = cell_lengths[None, ...]
            self.cell_angles = cell_angles[None, ...]

            if any([
                self.T_fc is None,
                self.T_cf is None,
                self.cell_volume is None,
                self.packing_coeff is None,
                self.density is None
            ]):  # better to do this in batches and feed it as kwargs # todo add a log/warning for this
                self.box_analysis()
            else:
                assert (self.T_cf is not None and self.cell_volume is not None), \
                    "T_fc, T_cf, and cell volume must all be provided all together or not at all"

        if aunit_centroid is not None:
            self.aunit_centroid = aunit_centroid[None, ...]
        if aunit_orientation is not None:
            self.aunit_orientation = aunit_orientation[None, ...]
        if aunit_handedness is not None:
            self.aunit_handedness = aunit_handedness

        if mol_ind is not None:
            self.mol_ind = mol_ind
        if aux_ind is not None:
            self.aux_ind = aux_ind

    def box_analysis(self):
        self.T_fc, self.T_cf, self.cell_volume = (
            batch_compute_fractional_transform(self.cell_lengths,
                                               self.cell_angles))
        self.T_cf = torch.linalg.inv(self.T_fc)
        self.packing_coeff = self.mol_volume * self.sym_mult / self.cell_volume
        self.density = self.mass * self.sym_mult / self.cell_volume * 1.66054  # conversion from D/A^3 to g/cm^3

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
            asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key]).to(self.device)

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

    def standardize_normed_cell_vectors(self,
                                        normed_lengths: torch.Tensor,
                                        cell_angles: torch.Tensor,
                                        cell_means: OptTensor = None,
                                        cell_stds: OptTensor = None):
        if cell_means is None:
            means = torch.tensor(  # todo replace by call to constants file
                [1.0411, 1.1640, 1.4564, 1.5619, 1.5691, 1.5509],  # use triclinic statistics
                dtype=torch.float32, device=self.device)

        else:
            means = cell_means

        if cell_stds is None:
            stds = torch.tensor(
                [0.3846, 0.4280, 0.4864, 0.2363, 0.2046, 0.2624],
                dtype=torch.float32, device=self.device)
        else:
            stds = cell_stds

        if self.is_batch:
            stds = stds.unsqueeze(0)
            means = means.unsqueeze(0)

        return (torch.cat([normed_lengths, cell_angles], dim=-1) - means) / stds

    def destandardize_normed_cell_vectors(self,
                                          normed_lengths: torch.Tensor,
                                          cell_angles: torch.Tensor,
                                          cell_means: OptTensor = None,
                                          cell_stds: OptTensor = None):
        if cell_means is None:  # these norms assume incoming cell lengths are pre-normed
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
        return torch.cat([normed_lengths, cell_angles], dim=-1) * stds + means

    def noise_cell_parameters(self, noise_level: float):
        # standardize
        std_cell_params = self.standardized_cell_parameters()
        # noise
        new_std_cell_params = std_cell_params + torch.randn_like(std_cell_params) * noise_level
        # destandardize
        destandardized_cell_params = self.destandardize_cell_parameters(new_std_cell_params)
        # assign
        self.set_cell_parameters(destandardized_cell_params)
        # clean
        self.clean_cell_parameters(mode='hard')
        # refresh box
        self.box_analysis()

    def destandardize_cell_parameters(self, std_cell_params):
        (std_cell_lengths,
         std_cell_angles,
         std_aunit_positions,
         std_aunit_orientations) = std_cell_params.split(3, dim=1)

        normed_cell_lengths, cell_angles = self.destandardize_normed_cell_vectors(
            std_cell_lengths,
            std_cell_angles
        ).split(3, dim=1)
        aunit_positions = self.destandardize_aunit_position(std_aunit_positions)
        aunit_orientations = self.destandardize_aunit_orientation(std_aunit_orientations)

        cell_lengths = self.denorm_cell_lengths(normed_cell_lengths)
        return torch.cat([
            cell_lengths, cell_angles, aunit_positions, aunit_orientations
        ], dim=1)

    def pose_aunit(self):
        if 'Batch' in self.__class__.__name__:
            self.pos = get_aunit_positions(
                self,
                align_to_standardized_orientation=True,
                mol_handedness=self.aunit_handedness,
            )
        else:
            self.pos = get_aunit_positions(
                collate_data_list([self]),
                align_to_standardized_orientation=True,
                mol_handedness=self.aunit_handedness,
            )
            #assert False, "aunit posing not yet implemented for single crystals"

    def build_unit_cell(self):
        if 'Batch' in self.__class__.__name__:
            self.unit_cell_pos = new_aunit2unit_cell(self)  # keep in numpy format
        else:
            self.unit_cell_pos = new_aunit2unit_cell(collate_data_list([self]))
            #assert False, "unit cell construction not yet implemented for single crystals"

    def build_cluster(self, supercell_size: int = 10):
        if 'Batch' in self.__class__.__name__:
            return unit_cell_to_supercell_cluster(self, supercell_size)
        else:
            crystal_batch = collate_data_list([self])
            crystal_batch.build_unit_cell()
            return unit_cell_to_supercell_cluster(crystal_batch, supercell_size)
            #False, "unit cell construction not yet implemented for single crystals"

    def de_cluster(self):
        # delete cluster information and reset this object as a molecule
        if self.aux_ind is not None:
            aunit_bools = self.aux_ind == 0
            self.pos = self.pos[aunit_bools]
            if self.x is not None:
                self.x = self.x[aunit_bools]
            self.z = self.z[aunit_bools]
            self.batch = torch.arange(self.num_graphs, device=self.device
                                      ).repeat_interleave(self.num_atoms)
            self.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device),
                                  torch.cumsum(self.num_atoms, dim=0)]).long()

            self.aux_ind = None
            self.mol_ind = None
            self.edges_dict = None
        else:
            print("can't de-cluster - this is already not a cluster")

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
                    assert torch.all(cell_angles[0] == torch.pi / 2), "Error in monoclinic alpha angle"
                    assert torch.all(cell_angles[2] == torch.pi / 2), "Error in monoclinic gamma angle"
                elif lattice.lower() == 'orthorhombic':  # fix all angles
                    assert torch.all(cell_angles == torch.ones(3) * torch.pi / 2), "Error in orthorhombic cell angles"
                elif lattice.lower() == 'tetragonal':  # fix all angles and a & b vectors
                    assert torch.all(cell_angles == torch.ones(3) * torch.pi / 2), "Error in tetragonal cell angles"
                    assert torch.all(cell_lengths[0] == cell_lengths[1]), "Error in tetragonal cell lengths"
                elif lattice.lower() == 'hexagonal':  # for rhombohedral, all angles and lengths equal, but not 90.
                    # for truly hexagonal, alpha=90, gamma is 120, a=b!=c
                    # todo implement 3&6 fold lattices
                    print('hexagonal lattice checks are not implemented!')
                    pass
                elif lattice.lower() == 'cubic':  # all angles 90 all lengths equal
                    assert torch.all(cell_lengths == cell_lengths.mean()), "Error in cubic cell lengths"
                    assert torch.all(cell_angles == torch.pi / 2), "Error in cubic cell angles"
                else:
                    assert False, f"{lattice} + ' is not a valid crystal lattice!')"
        else:
            assert False, "Lattice checks not impelemented for single crystals"

        return True

    def reparameterize_unit_cell(self):
        (aunit_centroid, aunit_orientation,
         handedness_list, is_well_defined, pos) = (
            parameterize_crystal_batch(self,
                                       ASYM_UNITS,
                                       enforce_right_handedness=False,
                                       return_aunit=True))

        return aunit_centroid, aunit_orientation, handedness_list.long(), is_well_defined, torch.cat(pos)

    def construct_intra_radial_graph(self, cutoff: float = 6):
        if self.aux_ind is not None:
            assert False, "Cannot build molecular graph when we have already built the supercell"
        self.edges_dict = construct_radial_graph(self.pos,
                                                 self.batch,
                                                 self.ptr,
                                                 cutoff,
                                                 max_num_neighbors=10000
                                                 )
        self.edge_index = self.edges_dict['edge_index']

    def construct_radial_graph(self, cutoff: float = 6):
        if "Batch" in self.__class__.__name__:
            self.edges_dict = get_intermolecular_dists_dict(
                self,
                cutoff
            )

        else:
            self.edges_dict = get_intermolecular_dists_dict(
                collate_data_list([self]),
                cutoff
            )
            #assert False, "Radial graph construction not implemented for single crystals"

    def compute_LJ_energy(self):
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if "Batch" in self.__class__.__name__:
            (molwise_overlap, molwise_normed_overlap,
             molwise_lj_pot, molwise_scaled_lj_pot, edgewise_lj_pot) \
                = vdw_analysis(vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               self.num_atoms,
                               )
        else:
            assert False, "LJ energies not implemented for single crystals"

        return molwise_lj_pot, molwise_scaled_lj_pot

    def compute_ES_energy(self):
        if "Batch" in self.__class__.__name__:
            molwise_estat_energy = electrostatic_analysis(
                self.edges_dict,
                self.num_graphs)

        else:
            assert False, "ES energies not implemented for single crystals"

        return molwise_estat_energy

    def clean_cell_parameters(self, mode: str = 'hard',
                              canonicalize_orientations: bool = True,
                              angle_pad: float = 0.5):
        # force outputs into physical ranges

        # cell lengths have to be positive nonzero
        if mode == 'hard':
            self.cell_lengths = self.cell_lengths.clip(min=0.01)
        elif mode == 'soft':
            self.cell_angles = softplus_shift(self.cell_angles)

        # range from (0,pi) with 50% padding to prevent too-skinny cells
        self.cell_angles = enforce_1d_bound(self.cell_angles,
                                            x_span=torch.pi / 2 * angle_pad,
                                            x_center=torch.pi / 2,
                                            mode=mode)

        # positions must be on 0-1 in the asymmetric unit
        aunit_scaled_pos = self.scale_centroid_to_aunit()
        cleaned_aunit_scaled_pos = enforce_1d_bound(aunit_scaled_pos, x_span=0.5, x_center=0.5, mode=mode)
        self.aunit_centroid = self.scale_centroid_to_unit_cell(cleaned_aunit_scaled_pos)

        # enforce range on vector norm  # todo decide how/where to enforce canonical z direction
        norm = torch.linalg.norm(self.aunit_orientation, dim=1)
        new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode=mode)  # MUST be nonzero
        self.aunit_orientation = self.aunit_orientation / norm[:, None] * new_norm[:, None]

        # enforce agreement with crystal system
        self.cell_lengths, self.cell_angles = enforce_crystal_system(self.cell_lengths,
                                                                     self.cell_angles,
                                                                     self.sg_ind,
                                                                     )

        # enforce z component in the upper half-plane
        if canonicalize_orientations:
            self.aunit_orientation = canonicalize_rotvec(self.aunit_orientation)

        # update cell vectors
        self.box_analysis()

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

    def standardized_cell_parameters(self):
        # todo more sophisticated standardization here
        # norm cell lengths by aunit volume
        normed_lengths = self.norm_cell_lengths()
        standardized_aunit_centroid = self.standardize_aunit_position()
        standardized_orientation = self.standardize_aunit_orientation()
        standardized_box_params = self.standardize_normed_cell_vectors(
            normed_lengths, self.cell_angles
        )
        return torch.cat([standardized_box_params,
                          standardized_aunit_centroid,
                          standardized_orientation], dim=1)

    def standardize_aunit_position(self):
        """
        uniform distribution on 0-1
        mean = 0.5
        std = 0.289
        """
        aunit_mean = 0.5
        aunit_std = 0.289
        return (self.scale_centroid_to_aunit() - aunit_mean) / aunit_std

    def destandardize_aunit_position(self, std_aunit_position):
        """
        uniform distribution on 0-1
        mean = 0.5
        std = 0.289
        """
        aunit_mean = 0.5
        aunit_std = 0.289
        return self.scale_centroid_to_unit_cell(std_aunit_position * aunit_std + aunit_mean)

    def standardize_aunit_orientation(self):
        # standardize aunit orientation
        """
        num_graphs = 100000
        random_vectors = torch.randn(size=(num_graphs, 3))

        # set norms uniformly between 0-2pi
        norms = random_vectors.norm(dim=1)
        applied_norms = (torch.rand(num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
        random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

        random_vectors = canonicalize_rotvec(random_vectors)
        std = random_vectors.std(0)
        mean = random_vectors.mean(0)

        """
        orientation_stds = torch.tensor([[2.08, 2.08, 1.38]], dtype=torch.float32, device=self.device)
        orientation_means = torch.tensor([[0, 0, torch.pi / 2]], dtype=torch.float32, device=self.device)
        return (self.aunit_orientation - orientation_means) / orientation_stds

    def destandardize_aunit_orientation(self, std_aunit_orientation):
        # destandardize aunit orientation
        """
        num_graphs = 100000
        random_vectors = torch.randn(size=(num_graphs, 3))

        # set norms uniformly between 0-2pi
        norms = random_vectors.norm(dim=1)
        applied_norms = (torch.rand(num_graphs) * 2 * torch.pi).clip(min=0.05)  # cannot be exactly zero
        random_vectors = random_vectors / norms[:, None] * applied_norms[:, None]

        random_vectors = canonicalize_rotvec(random_vectors)
        std = random_vectors.std(0)
        mean = random_vectors.mean(0)

        """
        orientation_stds = torch.tensor([[2.08, 2.08, 1.38]], dtype=torch.float32, device=self.device)
        orientation_means = torch.tensor([[0, 0, torch.pi / 2]], dtype=torch.float32, device=self.device)
        return std_aunit_orientation * orientation_stds + orientation_means

    def set_cell_parameters(self, cell_parameters):
        (self.cell_lengths, self.cell_angles,
         self.aunit_centroid, self.aunit_orientation) = (
            cell_parameters.split(3, dim=1))

    def build_and_analyze(self):
        """
        full procedure for building and analyzing a molecular crystal
        """
        cluster_batch = self.mol2cluster()
        cluster_batch.construct_radial_graph()
        self.lj_pot, self.scaled_lj_pot = cluster_batch.compute_LJ_energy()
        self.es_pot = cluster_batch.compute_ES_energy()
        return self.lj_pot, self.es_pot, self.scaled_lj_pot

    def mol2cluster(self):
        self.pose_aunit()
        self.build_unit_cell()
        return self.build_cluster()

    def do_embedding(self,
                     embedding_type: str,
                     encoder: Optional = None):
        if self.is_batch:
            embedding = get_mol_embedding_for_proxy(self.clone(),
                                                    embedding_type,
                                                    encoder
                                                    )
            scaled_params = self.standardized_cell_parameters()
            #return torch.cat([embedding, scaled_params], dim=1)
            return torch.cat([embedding, scaled_params[..., :-3]], dim=1)

        else:
            assert False, "Crystal embedding not implemented for single samples"

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
