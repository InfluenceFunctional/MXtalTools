import copy
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (Any, Dict, Iterable, List, NamedTuple, Optional, Union)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import (BaseStorage, EdgeStorage,
                                          GlobalStorage, NodeStorage)
from torch_geometric.typing import OptTensor
from torch_scatter import scatter
from torch_sparse import SparseTensor

from mxtaltools.analysis.vdw_analysis import vdw_analysis, electrostatic_analysis, get_intermolecular_dists_dict, \
    buckingham_energy, silu_energy
from mxtaltools.common.ellipsoid_ops import compute_ellipsoid_volume, \
    compute_cosine_similarity_matrix
from mxtaltools.common.geometry_utils import batch_compute_molecule_volume, \
    compute_mol_radius, batch_compute_mol_radius, batch_compute_mol_mass, compute_mol_mass, center_mol_batch, \
    apply_rotation_to_batch, enforce_crystal_system, batch_compute_fractional_transform, batch_cell_vol_torch, \
    rotvec2rotmat, rotmat2rotvec
from mxtaltools.common.utils import softplus_shift
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from mxtaltools.constants.space_group_info import SYM_OPS, LATTICE_TYPE
from mxtaltools.crystal_building.crystal_latent_transforms import CompositeTransform, AunitTransform, NiggliTransform, \
    StdNormalTransform, enforce_niggli_plane
from mxtaltools.crystal_building.random_crystal_sampling import sample_aunit_lengths, sample_cell_angles, \
    sample_aunit_orientations, sample_aunit_centroids
from mxtaltools.crystal_building.utils import get_aunit_positions, parameterize_crystal_batch, \
    align_mol_batch_to_standard_axes, canonicalize_rotvec, new_aunit2unit_cell, \
    new_unit_cell_to_supercell_cluster
from mxtaltools.crystal_search.standalone_crystal_opt import standalone_gradient_descent_optimization
from mxtaltools.dataset_utils.mol_building import smiles2conformer
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.functions.radial_graph import build_radial_graph, asymmetric_radius_graph
from mxtaltools.models.modules.components import ResidualMLP
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


class MolData(MXtalBase):  # todo add method for batch_molecule_compute_principal_axes
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
        self.edges_dict = build_radial_graph(self.pos,
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
        return data_batch_to_ase_mols_list(
            self,
            specific_inds=sample_inds,
            show_mols=True,
            **vis_kwargs)

    def orient_molecule(self, mode: str,
                        include_inversion: bool = True,
                        target_handedness: Optional[torch.LongTensor] = None,
                        correct_orientation: bool = False,
                        ):
        if 'Batch' in self.__class__.__name__:
            # always center the molecules
            if correct_orientation: assert mode == 'random', "Orientation correction only implemented for random rotations"
            self.recenter_molecules()
            if mode == 'standardized':
                mol_batch = align_mol_batch_to_standard_axes(self, handedness=target_handedness)
                self.pos = mol_batch.pos
            elif mode == 'random':
                random_rotations = torch.tensor(
                    Rotation.random(num=self.num_graphs).as_matrix(),
                    device=self.device, dtype=torch.float32)
                if include_inversion:
                    assert not correct_orientation, "Cannot reconstruct rotations through inversion."
                    invert_inds = torch.randint(2, (self.num_graphs,)) * 2 - 1
                    random_rotations *= invert_inds[:, None, None]

                self.pos = apply_rotation_to_batch(
                    self.pos,
                    random_rotations,
                    self.batch)
                if correct_orientation:
                    # adjust aunit_orientation such that the final orientation is unchanged
                    # new combined rotation should undo the random rotation, then apply the aunit_orientation
                    invert_random_rotation = torch.linalg.inv(random_rotations)
                    new_orientation_matrix = rotvec2rotmat(self.aunit_orientation) @ invert_random_rotation
                    new_rotvec = rotmat2rotvec(new_orientation_matrix)
                    new_rotvec_fixed = canonicalize_rotvec(new_rotvec)
                    self.aunit_orientation = new_rotvec_fixed
                    # correction is successful when this is small
                    # (rotvec2rotmat(new_rotvec_fixed) @ random_rotations - rotvec2rotmat(self.aunit_orientation)).abs().sum()
            else:
                pass

        else:
            assert False, "molecule orientation not implemented for single samples"

    def deprotonate(self):
        """danger - this breaks several batching methods and should be used carefully
        """
        heavy_atom_inds = torch.argwhere(self.z != 1).flatten()  # protons are atom type 1
        if len(heavy_atom_inds) > 0:
            self.z = self.z[heavy_atom_inds]
            self.pos = self.pos[heavy_atom_inds]
            if hasattr(self, 'batch') and self.batch is not None:
                self.batch = self.batch[heavy_atom_inds]
                a, b = torch.unique(self.batch, return_counts=True)
                self.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
                self.num_atoms = torch.diff(self.ptr).long()
            else:
                self.num_atoms = len(self.z)
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
                 sg_ind: Optional[Union[int, torch.Tensor]] = None,
                 cell_lengths: Optional[torch.Tensor] = None,
                 cell_angles: Optional[torch.Tensor] = None,
                 aunit_centroid: Optional[torch.Tensor] = None,
                 aunit_orientation: Optional[torch.Tensor] = None,
                 aunit_handedness: Optional[Union[torch.Tensor, int]] = None,
                 identifier: Optional[str] = None,
                 unit_cell_pos: Optional[np.ndarray] = None,
                 nonstandard_symmetry: Optional[bool] = False,
                 symmetry_operators: Optional[list] = None,
                 is_well_defined: Optional[bool] = True,
                 aux_ind: Optional[torch.LongTensor] = None,
                 mol_ind: Optional[torch.LongTensor] = None,
                 skip_box_analysis: Optional[bool] = False,
                 dummy_box_analysis: Optional[bool] = False,
                 **kwargs):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)
        # initialize crystal from an existing molecule
        if molecule is not None:
            self.set_mol_attrs(molecule)

        # then overwrite any relevant features
        for key, value in kwargs.items():
            setattr(self, key, value)

        # and add crystal features
        if identifier is not None:
            self.identifier = identifier
        if sg_ind is not None:
            self.set_sg_attrs(is_well_defined, nonstandard_symmetry, sg_ind, symmetry_operators)

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
            # todo add a check here in case the leading dim is already there
            self.cell_lengths = cell_lengths[None, ...]
            self.cell_angles = cell_angles[None, ...]

            if not skip_box_analysis:
                if not any([  # if we have the box analysis output, just don't do it
                    self.T_fc is None,
                    self.T_cf is None,
                    self.cell_volume is None,
                    self.packing_coeff is None,
                    self.density is None
                ]):
                    pass
                elif dummy_box_analysis:
                    self.T_fc = torch.eye(3, device=self.device)
                    self.T_cf = torch.eye(3, device=self.device)
                    self.cell_volume = torch.ones(1, device=self.device)
                    self.packing_coeff = torch.ones(1, device=self.device)
                    self.density = torch.ones(1, device=self.device)
                else:
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
            self.aunit_centroid = self.assign_aunit_centroid(aunit_centroid[None, ...])
        if aunit_orientation is not None:
            self.aunit_orientation = aunit_orientation[None, ...]
        if aunit_handedness is not None:
            self.aunit_handedness = aunit_handedness

        if mol_ind is not None:
            self.mol_ind = mol_ind
        if aux_ind is not None:
            self.aux_ind = aux_ind

    def assign_aunit_centroid(self, values, eps=1e-4):
        """never allow this to touch exactly 1"""
        return values.clip(min=0, max=1-eps)

    def set_sg_attrs(self, is_well_defined, nonstandard_symmetry, sg_ind, symmetry_operators):
        if not torch.is_tensor(sg_ind):
            self.sg_ind = torch.tensor(sg_ind, dtype=torch.long, device=self.device)
        else:
            self.sg_ind = sg_ind.long().to(self.device)
        if nonstandard_symmetry:  # set as np stack for correct collation behavior (we don't want batches to stack)
            if symmetry_operators is not None:
                self.symmetry_operators = symmetry_operators
            else:
                assert False, "symmetry_operators must be given for nonstandard symmetry operations"
            self.nonstandard_symmetry = True
        else:  # standard symmetry
            if symmetry_operators is not None:
                self.symmetry_operators = symmetry_operators
            else:
                self.symmetry_operators = np.stack(SYM_OPS[int(sg_ind)])  # if saved as a tensor, we get collation issues
            self.nonstandard_symmetry = False
        self.sym_mult = torch.ones(1, dtype=torch.long, device=self.device) * len(self.symmetry_operators)
        self.is_well_defined = is_well_defined

    def set_mol_attrs(self, molecule):
        # copy out the data
        mol_dict = molecule.to_dict()
        for key, value in mol_dict.items():
            setattr(self, key, value)

    def box_analysis(self):
        self.T_fc, self.T_cf, self.cell_volume = (
            batch_compute_fractional_transform(self.cell_lengths,
                                               self.cell_angles))
        self.T_cf = torch.linalg.inv(self.T_fc)
        self.packing_coeff = self.mol_volume * self.sym_mult / self.cell_volume
        self.density = self.mass * self.sym_mult / self.cell_volume * 1.66054  # conversion from D/A^3 to g/cm^3

    def sample_random_cell_lengths(self, target_packing_coeff: Optional = None):
        """
        NOTE depends on cell angles, so must do those first if resampling both
        """
        if 'Batch' in self.__class__.__name__:
            aunit_lengths = sample_aunit_lengths(self.num_graphs,
                                                 self.cell_angles,
                                                 self.mol_volume,
                                                 target_packing_coeff=target_packing_coeff)
        else:
            self_batch = collate_data_list([self])
            aunit_lengths = sample_aunit_lengths(1,
                                                 self_batch.cell_angles,
                                                 self_batch.mol_volume,
                                                 target_packing_coeff=target_packing_coeff)

        self.cell_lengths = self.scale_aunit_lengths_to_unit_cell(aunit_lengths)

    def sample_random_cell_angles(self):
        if 'Batch' in self.__class__.__name__:
            self.cell_angles = sample_cell_angles(self.num_graphs).to(self.device)
        else:
            self.cell_angles = sample_cell_angles(1).to(self.device)

    def sample_random_aunit_orientations(self):
        if 'Batch' in self.__class__.__name__:
            self.aunit_orientation = sample_aunit_orientations(self.num_graphs).to(self.device)
        else:
            self.aunit_orientation = sample_aunit_orientations(1).to(self.device)

    def sample_random_aunit_centroids(self):
        if 'Batch' in self.__class__.__name__:
            aunit_centroid = self.assign_aunit_centroid(sample_aunit_centroids(self.num_graphs).to(self.device))
        else:
            aunit_centroid = self.assign_aunit_centroid(sample_aunit_centroids(1).to(self.device))

        self.aunit_centroid = self.scale_centroid_to_unit_cell(aunit_centroid)

    def sample_random_crystal_parameters(self,
                                         target_packing_coeff: Optional = None,
                                         seed: Optional[int] = None,
                                         cleaning_mode: Optional[str] = 'hard'
                                         ):
        if seed is not None:
            torch.manual_seed(seed)
        self.sample_random_cell_angles()  # must do this one before cell lengths !!!
        self.sample_random_cell_lengths(target_packing_coeff)
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        self.clean_cell_parameters(mode=cleaning_mode)

    def sample_random_reduced_crystal_parameters(self,
                                                 target_packing_coeff: Optional = None,
                                                 seed: Optional[int] = None,
                                                 ):
        """
        Sample random crystal parameters that are automatically Niggli-reduced
        """
        if seed is not None:
            torch.manual_seed(seed)
        assert self.is_batch, "Random crystal parameters not setup for single crystals"
        self.sample_reduced_box_vectors(target_packing_coeff=target_packing_coeff)
        self.sample_random_aunit_orientations()
        self.sample_random_aunit_centroids()
        # enforce agreement with crystal system
        # other cell parameters are valid by explicit construction
        # todo add crystal system conditions to sampling workflow, rather than cleaning up here
        self.cell_lengths, self.cell_angles = enforce_crystal_system(
            self.cell_lengths,
            self.cell_angles,
            self.sg_ind
        )
        self.box_analysis()

    def gen_basis_to_cell_params(self, std_normal: torch.tensor):
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()

        if not hasattr(self, 'latent_transform'):
            self.init_latent_transform()

        self.set_cell_parameters(
            self.latent_transform.inverse(std_normal.clip(min=-6, max=6), self.sg_ind, self.radius)
        )

        self.cell_lengths, self.cell_angles = enforce_crystal_system(
            self.cell_lengths,
            self.cell_angles,
            self.sg_ind
        )
        self.box_analysis()

    def init_latent_transform(self,
                              c_log_mean=0.3,
                              c_log_std=0.25):
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()

        self.latent_transform = CompositeTransform([
            AunitTransform(asym_unit_dict=self.asym_unit_dict),
            NiggliTransform(),
            StdNormalTransform(c_log_mean=c_log_mean,
                               c_log_std=c_log_std),
            # SquashingTransform(min_val=-6,
            #                    max_val=6,
            #                    threshold=5.99,
            #                    ),
        ])

    def cell_params_to_gen_basis(self):
        if not hasattr(self, 'asym_unit_dict'):
            self.asym_unit_dict = self.build_asym_unit_dict()

        if not hasattr(self, 'latent_transform'):
            self.init_latent_transform()

        std_cell_params = self.latent_transform.forward(self.cell_parameters(), self.sg_ind, self.radius).clip(min=-6,
                                                                                                               max=6)
        assert torch.isfinite(std_cell_params).all()
        return std_cell_params

    def sample_reduced_box_vectors(self, target_packing_coeff: Optional[float] = None):

        if not hasattr(self, 'latent_transform'):
            self.init_latent_transform()

        # use the latent transform for lengths and angles sampling, as its more stable and better tested
        # we can still use the old samplers for aunit params
        temp_params = self.latent_transform.inverse(torch.randn((self.num_graphs, 12), device=self.device),
                                                    self.sg_ind,
                                                    self.radius)
        cell_lengths = temp_params[:, :3]
        cell_angles = temp_params[:, 3:6]

        if target_packing_coeff is not None:
            vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
            cp1 = self.mol_volume * self.sym_mult / vol1
            correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
            cell_lengths *= correction_ratio[:, None]

        cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')

        self.cell_lengths, self.cell_angles = cell_lengths, cell_angles

        # if 'Batch' in self.__class__.__name__:
        #     self.cell_lengths, self.cell_angles = sample_reduced_box_vectors(self.num_graphs,
        #                                                                      self.radius,
        #                                                                      self.mol_volume,
        #                                                                      self.sg_ind,
        #                                                                      self.sym_mult,
        #                                                                      self.asym_unit_dict,
        #                                                                      target_packing_coeff=target_packing_coeff
        #                                                                      )
        # else:
        #     self.cell_lengths, self.cell_angles = sample_reduced_box_vectors(1,
        #                                                                      self.radius,
        #                                                                      self.mol_volume,
        #                                                                      [self.sg_ind],
        #                                                                      self.sym_mult,
        #                                                                      self.asym_unit_dict,
        #                                                                      target_packing_coeff=target_packing_coeff
        #                                                                      )

    def sample_reasonable_random_parameters(self,
                                            tolerance: float = 1,
                                            target_packing_coeff: Optional = None,
                                            max_attempts: int = 100,
                                            seed: Optional[int] = None,
                                            sample_niggli: Optional[bool] = True):
        """
        Sample random crystal parameters
        build the resulting crystals and check their vdW overlaps
        If they are sufficiently small, retain that sample
        Repeat until convergence
        """
        good_params_tensor = torch.zeros(self.num_graphs, 12, device=self.device)
        found_params_flags = torch.zeros(self.num_graphs, dtype=torch.bool)
        best_ljs = torch.ones(self.num_graphs, dtype=torch.float32, device=self.device) * 1e7
        converged = False
        iter = 0
        while not converged and iter < max_attempts:
            if sample_niggli:
                self.sample_random_reduced_crystal_parameters(target_packing_coeff=target_packing_coeff)
            else:
                self.sample_random_crystal_parameters(target_packing_coeff, seed=seed)

            _, _, scaled_lj = self.build_and_analyze(cutoff=3)
            improved_inds = torch.argwhere(scaled_lj < best_ljs)
            best_ljs[improved_inds] = scaled_lj[improved_inds]
            good_inds = torch.argwhere(scaled_lj < tolerance)
            good_params_tensor[improved_inds] = self.cell_parameters()[improved_inds]
            found_params_flags[good_inds] = True
            iter += 1
            if torch.all(found_params_flags):
                converged = True

        self.set_cell_parameters(good_params_tensor)
        #
        # if not converged:
        #     print(f"Failed to converge {torch.sum(~found_params_flags)} out of {self.num_graphs} crystals")

    def denorm_by_radius(self, arr):
        """
        arr *= mol_radius
        assumes arr shape [n, m]
        """
        if 'Batch' in self.__class__.__name__:
            return arr * self.radius[:, None]
        else:
            return arr * self.radius[None]

    def norm_by_radius(self, arr):
        """
        arr /= mol_radius
        assumes arr shape [n, m]
        """
        if 'Batch' in self.__class__.__name__:
            return arr / self.radius[:, None]
        else:
            return arr / self.radius[None]

    def build_asym_unit_dict(self,
                             force_all_sgs: Optional[bool] = False
                             ):  # todo add capability to only do this for the necessary keys
        """  # todo unify this throughout this file
        NOTE this function is extremely slow
        Best used during batch operations
        """
        asym_unit_dict = ASYM_UNITS.copy()
        if force_all_sgs:
            sgs_to_tensorize = asym_unit_dict.keys()
        else:
            relevant_sgs = self.sg_ind.unique()
            sgs_to_tensorize = [str(int(sg)) for sg in relevant_sgs]
        for key in sgs_to_tensorize:
            asym_unit_dict[key] = torch.Tensor(asym_unit_dict[key]).to(self.device)
        return asym_unit_dict

    def scale_lengths_to_aunit(self):
        """
        scale unit cell lengths to the asymmetric unit, according to the canonical
        asymmetric unit shape within the unit cell.
        """
        if 'Batch' in self.__class__.__name__:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()

            return self.cell_lengths * torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return self.cell_lengths * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

    def scale_aunit_lengths_to_unit_cell(self, aunit_lengths):
        """
        input asymmetric unit lengths
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
        if 'Batch' in self.__class__.__name__:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            return aunit_lengths / torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return aunit_lengths / torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

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

        if 'Batch' in self.__class__.__name__:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            return self.aunit_centroid / torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind])
        else:
            return self.aunit_centroid / torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device)

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
        if 'Batch' in self.__class__.__name__:
            if not hasattr(self, 'asym_unit_dict'):
                self.asym_unit_dict = self.build_asym_unit_dict()
            return self.assign_aunit_centroid(normed_centroid * torch.stack([self.asym_unit_dict[str(int(ind))] for ind in self.sg_ind]))
        else:
            return self.assign_aunit_centroid(normed_centroid * torch.Tensor(ASYM_UNITS[str(int(self.sg_ind))]).to(self.device))

    def noise_cell_parameters(self, noise_level: float):
        # standardize
        std_cell_params = self.standardize_cell_parameters()
        # noise
        new_std_cell_params = std_cell_params + torch.randn_like(std_cell_params) * noise_level
        # destandardize
        destandardized_cell_params = self.destandardize_cell_parameters(new_std_cell_params)
        # assign
        self.set_cell_parameters(destandardized_cell_params, skip_box_analysis=True)
        # clean
        self.clean_cell_parameters(mode='hard')
        # refresh box
        self.box_analysis()

    def standardize_cell_parameters(self):
        standardized_aunit_lengths = self.standardize_cell_lengths()
        standardized_cell_angles = self.standardize_cell_angles()
        standardized_aunit_centroid = self.standardize_aunit_position()
        standardized_orientation = self.standardize_aunit_orientation()

        return torch.cat([standardized_aunit_lengths,
                          standardized_cell_angles,
                          standardized_aunit_centroid,
                          standardized_orientation], dim=1)

    def destandardize_cell_parameters(self, std_cell_params):
        (std_aunit_lengths,
         std_cell_angles,
         std_aunit_positions,
         std_aunit_orientations) = std_cell_params.split(3, dim=1)

        cell_lengths = self.destandardize_cell_lengths(std_aunit_lengths)
        cell_angles = self.destandardize_cell_angles(std_cell_angles)
        aunit_positions = self.destandardize_aunit_position(std_aunit_positions)
        aunit_orientations = self.destandardize_aunit_orientation(std_aunit_orientations)

        return torch.cat([
            cell_lengths, cell_angles, aunit_positions, aunit_orientations
        ], dim=1)

    def pose_aunit(self, align_to_standardized_orientation: Optional[bool] = True):
        if 'Batch' in self.__class__.__name__:
            self.pos = get_aunit_positions(
                self,
                align_to_standardized_orientation=align_to_standardized_orientation,
                mol_handedness=self.aunit_handedness,
            )
        else:
            self.pos = get_aunit_positions(
                collate_data_list([self]),
                align_to_standardized_orientation=align_to_standardized_orientation,
                mol_handedness=self.aunit_handedness,
            )

    def build_unit_cell(self):
        if 'Batch' in self.__class__.__name__:
            self.unit_cell_pos, self.unit_cell_batch, self.unit_cell_mol_ind = new_aunit2unit_cell(self)
            #self.unit_cell_pos = aunit2unit_cell(self)

            # for comparison to the new method
            # self.visualize([0], mode='unit cell')
            #
            # ref_cell_list = aunit2unit_cell(self)
            # e2 = []
            # for elem in ref_cell_list:
            #     for thing in elem:
            #         e2.append(thing)
            # pp2 = torch.cat(e2)
            # self.unit_cell_pos = pp2
            # self.visualize([0], mode='unit cell')

        else:
            self.unit_cell_pos, self.unit_cell_batch, self.unit_cell_mol_ind  = new_aunit2unit_cell(collate_data_list([self]))

    def build_cluster(self, cutoff: float = 6, supercell_size: int = 10):
        if 'Batch' in self.__class__.__name__:
            return new_unit_cell_to_supercell_cluster(self, cutoff=cutoff, supercell_size=supercell_size)
            #return unit_cell_to_supercell_cluster(self, cutoff=cutoff, supercell_size=supercell_size)
        else:
            crystal_batch = collate_data_list([self])
            crystal_batch.build_unit_cell()
            return new_unit_cell_to_supercell_cluster(crystal_batch, supercell_size)
            #return unit_cell_to_supercell_cluster(crystal_batch, supercell_size)
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

    def validate_cell_params_ranges(self, eps=1e-4):
        # assert valid ranges
        assert torch.all(self.aunit_centroid >= 0), "Aunit centroids must be greater than 0"
        assert torch.all(self.aunit_centroid <= 1-eps), "Aunit centroids must be less than 0.999"
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
        self.edges_dict = build_radial_graph(self.pos,
                                             self.batch,
                                             self.ptr,
                                             cutoff,
                                             max_num_neighbors=10000
                                             )
        self.edge_index = self.edges_dict['edge_index']

    def construct_radial_graph(self,
                               cutoff: float = 6,
                               max_num_neighbors=10000):
        if "Batch" in self.__class__.__name__:
            self.edges_dict = get_intermolecular_dists_dict(
                self,
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

        else:
            self.edges_dict = get_intermolecular_dists_dict(
                collate_data_list([self]),
                cutoff,
                max_num_neighbors=max_num_neighbors
            )
            #assert False, "Radial graph construction not implemented for single crystals"

    def compute_niggli_overlap(self):
        a, b, c, al, be, ga = self.cell_parameters()[:, :6].split(1, dim=1)
        ab = a * b
        ac = a * c
        bc = b * c

        al_cos = torch.cos(al)
        be_cos = torch.cos(be)
        ga_cos = torch.cos(ga)

        return (ab * ga_cos + ac * be_cos + bc * al_cos).flatten()

    def compute_LJ_energy(self, return_overlaps: Optional[bool] = False):
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if "Batch" in self.__class__.__name__:
            (molwise_overlap, molwise_normed_overlap,
             molwise_lj_pot, molwise_scaled_lj_pot, edgewise_lj_pot) \
                = vdw_analysis(vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               )
        else:
            assert False, "LJ energies not implemented for single crystals"
        if return_overlaps:
            return molwise_lj_pot, molwise_scaled_lj_pot, edgewise_lj_pot, molwise_overlap, molwise_normed_overlap
        else:
            return molwise_lj_pot, molwise_scaled_lj_pot

    def compute_ES_energy(self):
        if "Batch" in self.__class__.__name__:
            molwise_estat_energy = electrostatic_analysis(
                self.edges_dict,
                self.num_graphs)

        else:
            assert False, "ES energies not implemented for single crystals"

        return molwise_estat_energy

    def compute_buckingham_energy(self):
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if "Batch" in self.__class__.__name__:
            molwise_buckingham_energy = buckingham_energy(
                self.edges_dict,
                self.num_graphs,
                vdw_radii_tensor
            )

        else:
            assert False, "BH energies not implemented for single crystals"

        return molwise_buckingham_energy

    def compute_silu_energy(self,
                            repulsion: Optional[float] = None):
        vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if "Batch" in self.__class__.__name__:
            molwise_silu_energy = silu_energy(
                self.edges_dict,
                self.num_graphs,
                vdw_radii_tensor,
                repulsion=repulsion,
            )
        else:
            assert False, "SiLU energies not implemented for single crystals"

        return molwise_silu_energy

    def compute_ellipsoidal_overlap(self,
                                    surface_padding: float = 1,
                                    return_details: Optional[bool] = False,
                                    model: Optional = None):
        """
        Compute an energy function given the overlaps of molecules in an ellipsoid representation
        Using our pretrained ellipsoid overlap function

        Only works on cluster batches (rather than crystal batches)

        1) embed all molecules as ellipsoids
        2) get inter-ellipsoidal edges
        3) parameterize ellipsoid pairs
        4) format batch to ellipsoid model
        5) evaluate overlaps
        6) convert overlaps to energies
        """
        if model is None:
            if not hasattr(self, "ellipsoid_model"):
                self.load_ellipsoid_model()
        else:
            ellipsoid_model = model

        mols_per_cluster = torch.tensor(self.edges_dict['n_repeats'], device=self.device)
        tot_num_mols = torch.sum(mols_per_cluster)
        tot_mol_index = torch.arange(tot_num_mols, device=self.device).repeat_interleave(self.num_atoms.repeat_interleave(mols_per_cluster))
        molwise_batch = torch.arange(self.num_graphs, device=self.device).repeat_interleave(mols_per_cluster, dim=0)

        edge_i_good, edge_j_good, mol_centroids = self.get_intermolecular_ellipsoid_edges(molwise_batch,
                                                                                          surface_padding,
                                                                                          tot_mol_index, tot_num_mols)

        atoms_per_necessary_mol, mol_id_map, molwise_batch_subset, num_necessary_mols, subset_pos, tot_mol_index_subset = self.reindex_ellipsoid_mols(
            edge_i_good, edge_j_good, molwise_batch, tot_mol_index, tot_num_mols)

        """get ellipsoids"""
        add_noise = 0.01
        cov_eps = 0.01

        eigvals_sorted, eigvecs_sorted = self.compute_ellipsoid_eigvecs(add_noise, atoms_per_necessary_mol, cov_eps,
                                                                        molwise_batch_subset, num_necessary_mols, subset_pos,
                                                                        tot_mol_index_subset)

        """
        Set default as the ellipsoid tip being at the surface of the molecule (assume largest radius)
        Then, plus or minus angstroms to expose or cover surface atoms
        """
        eps = 1e-3

        longest_length = self.radius[molwise_batch_subset]
        padding_scaling_factor = (longest_length + surface_padding) / longest_length
        min_axis_length = torch.amax(
            torch.stack([1.5 * padding_scaling_factor, 0.1 * torch.ones_like(padding_scaling_factor)]),
            dim=0)  # need a finite thickness for flat molecules
        sqrt_eigenvalues = torch.sqrt(eigvals_sorted.clamp(min=0) + eps)
        normed_eigs = sqrt_eigenvalues / sqrt_eigenvalues.amax(1, keepdim=True)  # normalize to relative lengths
        # semi axis scale now controls how much of the surface is revealed - for negative values, surface atoms will poke out
        # if the surface padding is set too small, the ellipsoid will just retreat into a tiny sphere
        semi_axis_lengths = (normed_eigs * longest_length[:, None] + surface_padding).clip(min=min_axis_length[:, None])

        Ip = eigvecs_sorted


        """ featurize ellipsoids """
        norm_factor, normed_v1, normed_v2, v1, v2, x = self.featurize_ellipsoids(Ip, edge_i_good, edge_j_good, eps,
                                                                             mol_centroids, mol_id_map,
                                                                             semi_axis_lengths)

        # pass to the model
        if hasattr(self, 'ellipsoid_model') and model is None:
            output = self.ellipsoid_model(x)
        else:
            output = ellipsoid_model(x)

        # process results
        v1_pred, v2_pred, normed_overlap_pred = (output[:, 0] * norm_factor ** 3,
                                                 output[:, 1] * norm_factor ** 3,
                                                 output[:, 2].clip(min=0)
                                                 )
        reduced_volume = (normed_v1 * normed_v2) / (normed_v1 + normed_v2)
        denormed_overlap_pred = normed_overlap_pred * reduced_volume  # model works in the reduced basis
        overlap_pred = denormed_overlap_pred * norm_factor ** 3  # inunits of cubic angstroms
        v_pred_error = (v1_pred - v1).abs() / v1 + (v2_pred - v2).abs() / v2

        # sum of overlaps cubic angstroms per molecule
        molwise_ellipsoid_overlap = scatter(overlap_pred, molwise_batch[edge_j_good], dim=0, dim_size=self.num_graphs,
                                            reduce='sum')

        normed_ellipsoid_overlap = scatter(normed_overlap_pred, molwise_batch[edge_j_good], dim=0,
                                           dim_size=self.num_graphs,
                                           reduce='sum')

        if not return_details:
            return molwise_ellipsoid_overlap
        else:
            return molwise_ellipsoid_overlap, v1_pred, v2_pred, v1, v2, norm_factor, normed_ellipsoid_overlap

    def featurize_ellipsoids(self, Ip, edge_i_good, edge_j_good, eps, mol_centroids, mol_id_map, semi_axis_lengths):
        # featurize pairs
        r = mol_centroids[edge_j_good] - mol_centroids[edge_i_good]
        edge_i_local = mol_id_map[edge_i_good]
        edge_j_local = mol_id_map[edge_j_good]
        assert torch.isfinite(Ip).all(), "Non-finite principal axes!"
        e1 = Ip[edge_i_local] * semi_axis_lengths[edge_i_local, :, None]
        e2 = Ip[edge_j_local] * semi_axis_lengths[edge_j_local, :, None]
        v1 = compute_ellipsoid_volume(e1)
        v2 = compute_ellipsoid_volume(e2)
        assert torch.isfinite(v1).all()
        assert (v1 > 0).all()
        assert torch.isfinite(semi_axis_lengths).all()
        assert (semi_axis_lengths > 0).all()
        # normalize
        max_e1 = e1.norm(dim=-1).amax(1)
        max_e2 = e2.norm(dim=-1).amax(1)
        max_val = torch.stack([max_e1, max_e2]).T.amax(1) + eps
        normed_e1 = e1 / max_val[:, None, None]
        normed_e2 = e2 / max_val[:, None, None]
        normed_r = r / max_val[:, None]
        normed_v1 = v1 / max_val ** 3
        normed_v2 = v2 / max_val ** 3
        # standardize directions
        dot1 = torch.einsum('nij,ni->nj', normed_e1, normed_r)
        sign_flip1 = (dot1 < 0).float() * -2 + 1  # flips points against r
        std_normed_e1 = normed_e1 * sign_flip1.unsqueeze(-1)
        dot2 = torch.einsum('nij,ni->nj', normed_e2, -normed_r)
        sign_flip2 = (dot2 < 0).float() * -2 + 1  # flips points same way as r
        std_normed_e2 = normed_e2 * sign_flip2.unsqueeze(-1)
        # parameterize
        r_hat = F.normalize(normed_r + eps, dim=-1)
        r1_local = torch.einsum('nij,nj->ni', std_normed_e1, r_hat)  # r in frame of ellipsoid 1
        r2_local = torch.einsum('nij,nj->ni', std_normed_e2, -r_hat)  # r in frame of ellipsoid 2
        unit_std_normed_e1 = std_normed_e1 / std_normed_e1.norm(dim=-1, keepdim=True)
        unit_std_normed_e2 = std_normed_e2 / std_normed_e2.norm(dim=-1, keepdim=True)
        # relative rotation matrix
        # R_rel = torch.einsum('nik, njk -> nij', unit_std_normed_e1, unit_std_normed_e2)
        cmat = compute_cosine_similarity_matrix(unit_std_normed_e1, unit_std_normed_e2)
        x = torch.cat([
            normed_r.norm(dim=-1, keepdim=True),
            cmat.reshape(len(e1), 9),
            std_normed_e1.norm(dim=-1),
            std_normed_e2.norm(dim=-1),
            r1_local,
            r2_local,
        ], dim=1)
        return max_val, normed_v1, normed_v2, v1, v2, x

    def compute_ellipsoid_eigvecs(self, add_noise, atoms_per_necessary_mol, cov_eps, molwise_batch_subset, num_necessary_mols,
                                  subset_pos, tot_mol_index_subset):
        # get principal axes
        centered_mol_pos = center_mol_batch(subset_pos,
                                            tot_mol_index_subset,
                                            num_graphs=len(molwise_batch_subset),
                                            nodes_per_graph=atoms_per_necessary_mol)
        if centered_mol_pos.requires_grad or add_noise:
            coords_to_compute = centered_mol_pos + torch.randn_like(centered_mol_pos) * cov_eps
        else:
            coords_to_compute = centered_mol_pos
        # we'll get the eigenvalues of the covariance matrix to approximate the molecule spatial extent
        # Compute outer products: [N, 3, 3]
        outer = coords_to_compute[:, :, None] * coords_to_compute[:, None, :]
        # Accumulate covariance sums per molecule
        cov_sums = torch.zeros((num_necessary_mols, 3, 3), device=self.device)
        cov_sums = cov_sums.index_add(0, tot_mol_index_subset, outer)
        # Normalize by number of atoms per molecule
        atoms_per_necessary_mol = atoms_per_necessary_mol.to(dtype=torch.float32)  # in case it's int
        covariances = cov_sums / atoms_per_necessary_mol[:, None, None]
        # explicitly symmetrize
        covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
        covariances = covariances + torch.eye(3, device=covariances.device).expand(len(covariances), -1, -1) * 1e-3
        eigvals, eigvecs_c = self.safe_batched_eigh(covariances)
        eigvecs = eigvecs_c.permute(0, 2, 1)  # switch to row-wise eigenvectors
        sort_inds = torch.argsort(eigvals, dim=1, descending=True)  # we want eigenvectors sorted a>b>c row-wise
        eigvals_sorted = torch.gather(eigvals, dim=1, index=sort_inds)
        eigvecs_sorted = torch.gather(eigvecs, dim=1, index=sort_inds.unsqueeze(2).expand(-1, -1, 3))
        return eigvals_sorted, eigvecs_sorted

    def reindex_ellipsoid_mols(self, edge_i_good, edge_j_good, molwise_batch, tot_mol_index, tot_num_mols):
        necessary_mol_inds = torch.unique(torch.cat([edge_i_good, edge_j_good]))
        num_necessary_mols = len(necessary_mol_inds)
        mol_id_map = torch.full((tot_num_mols,), -1, device=self.device)
        mol_id_map[necessary_mol_inds] = torch.arange(len(necessary_mol_inds), device=self.device)
        # atom_mask = torch.isin(tot_mol_index, necessary_mol_inds)
        atom_mask = mol_id_map[tot_mol_index] != -1
        tot_mol_index_subset = mol_id_map[tot_mol_index[atom_mask]]
        molwise_batch_subset = molwise_batch[necessary_mol_inds]
        crystal_with_necessary_mol_ind, mols_per_necessary_cluster = torch.unique(molwise_batch_subset,
                                                                                  return_counts=True)
        full_mols_per_necessary_cluster = torch.zeros(self.num_graphs, device=self.device, dtype=torch.long)
        full_mols_per_necessary_cluster[crystal_with_necessary_mol_ind] = mols_per_necessary_cluster
        # mols_per_necessary_cluster = scatter(torch.ones_like(molwise_batch_subset), molwise_batch_subset, reduce='sum', dim=0)
        # mols_per_necessary_cluster = mols_per_necessary_cluster[mols_per_necessary_cluster>0]
        atoms_per_necessary_mol = scatter(torch.ones_like(tot_mol_index_subset), tot_mol_index_subset, dim=0,
                                          reduce='sum')  # self.num_atoms[molwise_batch_subset].repeat_interleave(mols_per_necessary_cluster, dim=0)
        subset_pos = self.pos[atom_mask]
        return atoms_per_necessary_mol, mol_id_map, molwise_batch_subset, num_necessary_mols, subset_pos, tot_mol_index_subset

    def get_intermolecular_ellipsoid_edges(self, molwise_batch, surface_padding, tot_mol_index, tot_num_mols):
        """get edges"""
        max_ellipsoid_radius = self.radius + surface_padding
        mol_centroids = scatter(self.pos, tot_mol_index, dim=0, dim_size=tot_num_mols, reduce='mean')
        mol_aux_inds = scatter(self.aux_ind, tot_mol_index, dim=0, dim_size=tot_num_mols, reduce='max')
        # get edges
        edge_i, edge_j = asymmetric_radius_graph(
            x=mol_centroids,
            batch=molwise_batch,
            inside_inds=torch.argwhere(mol_aux_inds == 0).flatten(),
            convolve_inds=torch.argwhere(mol_aux_inds >= 1).flatten(),
            # take 1 and 2 here, or we might have indexing issues
            max_num_neighbors=50,
            r=max_ellipsoid_radius.amax() * 2)
        # filter edges longer than 2x mol_radius for each sample
        dists = torch.linalg.norm(mol_centroids[edge_i] - mol_centroids[edge_j], dim=1)
        good_inds = dists < (2 * max_ellipsoid_radius[molwise_batch[edge_i]])
        edge_i_good = edge_i[good_inds]
        edge_j_good = edge_j[good_inds]
        return edge_i_good, edge_j_good, mol_centroids

    """
    # Check predictions
    from mxtaltools.common.ellipsoid_ops import compute_ellipsoid_overlap
    
    n_samples = 10
    overlaps = torch.zeros(n_samples, device=self.device)
    for ind in range(n_samples):
        overlaps[ind], _ = compute_ellipsoid_overlap(e1[ind], e2[ind], v1[ind], v2[ind], r[ind], num_probes=10000,
                                                     show_tqdm=True)
    
    import plotly.graph_objects as go
    
    
    def simple_parity(x, y):
        go.Figure(
            go.Scatter(x=x, y=y.cpu().detach(), mode='markers', showlegend=True,
                       name=f'R={torch.corrcoef(torch.stack([x, y]))[0, 1].cpu().detach().numpy():.4f}')).show(renderer='browser')
    
    
    simple_parity(overlaps.cpu().detach(), overlap_pred[:n_samples].cpu().detach())
    simple_parity(torch.cat([v1, v2]).cpu().detach(), torch.cat([v1_pred, v2_pred]).cpu().detach())


    """

    """  # to visualize ellipsoid fit
    import numpy as np
    import plotly.graph_objects as go

    
    def plot_ellipsoid_and_points(center, eigvals, eigvecs, points, n=50):
    
        # Create unit sphere
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack((x, y, z), axis=-1)  # shape (n, n, 3)
    
        # Scale by sqrt of eigenvalues (semi-axes)
        radii = np.sqrt(eigvals)
        ellipsoid = sphere * radii
    
        # Rotate by eigenvectors
        ellipsoid = ellipsoid @ eigvecs.T  # (n, n, 3)
    
        # Translate to center
        ellipsoid += center
    
        # Extract coordinates for surface plot
        x_e = ellipsoid[..., 0]
        y_e = ellipsoid[..., 1]
        z_e = ellipsoid[..., 2]
    
        # Create ellipsoid surface
        ellipsoid_surface = go.Surface(
            x=x_e, y=y_e, z=z_e,
            opacity=0.5,
            colorscale='Blues',
            showscale=False,
        )
    
        # Create scatter plot of points
        scatter_points = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Atoms'
        )
    
        # Center marker (optional)
        center_marker = go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers',
            marker=dict(size=5, color='black'),
            name='Center'
        )
    
        fig = go.Figure(data=[ellipsoid_surface, scatter_points, center_marker])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()
    
    eigvecs = Ip[0].cpu().detach().numpy()
    eigvals = (semi_axis_lengths[0].cpu().detach().numpy())**2
    points = self.pos[tot_mol_index == 0].cpu().detach().numpy()
    center = np.mean(points, axis=0)
    
    plot_ellipsoid_and_points(center, eigvals, eigvecs, points)
    
    """

    def safe_batched_eigh(self, covs, chunk=10000):
        out_vals, out_vecs = [], []
        for i in range(0, covs.shape[0], chunk):
            cchunk = covs[i:i + chunk]
            try:
                ev, evec = torch.linalg.eigh(cchunk)
            except torch.cuda.OutOfMemoryError:
                raise
            except RuntimeError as e:
                if "CUSOLVER_STATUS_INVALID_VALUE" in str(e):
                    print("Invalid matrix to eigh! Switching to CPU.")
                    ev, evec = torch.linalg.eigh(cchunk.cpu())
                    ev, evec = ev.to(covs.device), evec.to(covs.device)
                else:
                    raise e
            out_vals.append(ev)
            out_vecs.append(evec)
        return torch.cat(out_vals, dim=0).float(), torch.cat(out_vecs, dim=0).float()

    def load_ellipsoid_model(self):
        self.ellipsoid_model = ResidualMLP(
            22, 512, 3, 8, None, 0
        )
        self_path = Path(os.path.realpath(__file__)).parent.resolve()
        checkpoint = torch.load(self_path.joinpath(Path('ellipsoid_overlap_model.pt')), weights_only=True)
        self.ellipsoid_model.load_state_dict(checkpoint)
        self.ellipsoid_model.to(self.device).eval()

    def clean_cell_parameters(self, mode: str = 'hard',
                              canonicalize_orientations: bool = True,
                              angle_pad: float = 0.9,
                              length_pad: float = 3.0,
                              constrain_z: bool = False,
                              enforce_niggli: Optional[bool] = False):
        """
        force cell parameters into physical ranges
        """

        # cell lengths have to be positive nonzero
        if mode == 'hard':
            self.cell_lengths = self.cell_lengths.clip(min=length_pad)
        elif mode == 'soft':
            self.cell_lengths = softplus_shift(
                self.cell_lengths - length_pad) + length_pad  # enforces a minimum value of length_pad

        # range from (0,pi) with padding to prevent too-skinny cells
        self.cell_angles = enforce_1d_bound(self.cell_angles,
                                            x_span=torch.pi / 2 * angle_pad,
                                            x_center=torch.pi / 2,
                                            mode=mode)

        if enforce_niggli:  # TODO functionalize this
            # enforce the scaling factor b/c and a/b in the range [0, 1]
            a, b, c = self.cell_lengths.split(1, 1)
            b_scale = enforce_1d_bound((b / c), 0.5, 0.5, mode=mode)
            a_scale = enforce_1d_bound((a / b), 0.5, 0.5, mode=mode)
            self.cell_lengths = torch.cat([
                a_scale * b_scale * c,
                b_scale * c,
                c],
                dim=1)

            # enforce the scaling factor cos(alpha) / (b/2c) in the range [0, 1]
            al_cos, be_cos, ga_cos = self.cell_angles.cos().split(1, 1)
            a, b, c = self.cell_lengths.split(1, 1)
            al_cos_max = (b / 2 / c)
            be_cos_max = (a / 2 / c)
            ga_cos_max = (a / 2 / b)
            # now improved - including obtuse cells
            al_cos_scale = enforce_1d_bound(al_cos / al_cos_max, 1.0, 0.0, mode=mode)
            be_cos_scale = enforce_1d_bound(be_cos / be_cos_max, 1.0, 0.0, mode=mode)
            ga_cos_scale = enforce_1d_bound(ga_cos / ga_cos_max, 1.0, 0.0, mode=mode)
            # limit it here due to instability in arccos
            al = torch.arccos(al_cos_max * al_cos_scale.clip(min=-0.99, max=0.99))
            be = torch.arccos(be_cos_max * be_cos_scale.clip(min=-0.99, max=0.99))
            ga = torch.arccos(ga_cos_max * ga_cos_scale.clip(min=-0.99, max=0.99))

            cell_angles = torch.cat([al, be, ga], dim=1)
            self.cell_angles = enforce_niggli_plane(self.cell_lengths,
                                                    cell_angles,
                                                    mode='shift')

        # positions must be on 0-1 in the asymmetric unit
        aunit_scaled_pos = self.scale_centroid_to_aunit()
        cleaned_aunit_scaled_pos = enforce_1d_bound(aunit_scaled_pos, x_span=0.5, x_center=0.5, mode=mode)
        self.aunit_centroid = self.scale_centroid_to_unit_cell(cleaned_aunit_scaled_pos)

        # enforce range on vector norm
        if constrain_z:  # enforce a priori only positive Z possible
            assert not canonicalize_orientations, "Either enforce z positive, or post-canonicalize negative z's"
            # forces Z within the range 0-2pi
            fixed_z = enforce_1d_bound(
                self.aunit_orientation[:, -1],
                x_span=torch.pi,
                x_center=torch.pi,
                mode=mode
            )
            self.aunit_orientation = torch.cat([self.aunit_orientation[:, :2], fixed_z[:, None]], dim=1)
        norm = torch.linalg.norm(self.aunit_orientation, dim=1)
        new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode=mode)  # MUST be nonzero
        self.aunit_orientation = self.aunit_orientation / norm[:, None] * new_norm[:, None]

        # enforce agreement with crystal system
        self.cell_lengths, self.cell_angles = enforce_crystal_system(self.cell_lengths,
                                                                     self.cell_angles,
                                                                     self.sg_ind,
                                                                     )

        # enforce z component in the upper half-plane
        if canonicalize_orientations:  # converts the vector to its duplicate in the upper half-plane
            self.aunit_orientation = canonicalize_rotvec(self.aunit_orientation)

        # update cell vectors
        self.box_analysis()

    def niggli_angle_limits(self):
        a, b, c = self.cell_lengths.split(1, 1)
        al_cos_max = (b / 2 / c)
        be_cos_max = (a / 2 / c)
        ga_cos_max = (a / 2 / b)
        return torch.cat([al_cos_max, be_cos_max, ga_cos_max], dim=1)

    def aunit_volume(self):
        return self.cell_volume / self.sym_mult

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

    def standardize_cell_lengths(self):
        """
        Standardize the cell lengths tensor
        1. convert to asymmetric unit basis
        2. norm by mol radius
        3. standardize by stats
        mean = [1.0563, 0.7978, 1.7570]
        std = [0.6001, 0.5115, 0.7147]
        """
        aunit_lengths = self.scale_lengths_to_aunit()
        normed_aunit_lengths = self.norm_by_radius(aunit_lengths)
        # flatten stats across length dimensions
        mean = torch.tensor([1.0563, 0.7978, 1.7570], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.6001, 0.5115, 0.7147], dtype=torch.float32, device=self.device)
        mean = mean.mean() * torch.ones_like(mean)
        std = std.mean() * torch.ones_like(std)
        return (normed_aunit_lengths - mean[None, :]) / std[None, :]

    def destandardize_cell_lengths(self, std_aunit_lengths):
        """
        Destandardize the cell lengths tensor
        1. destandardize by stats
        2. denorm by mol radius
        3. convert to unit cell basis
        mean = [1.0563, 0.7978, 1.7570]
        std = [0.6001, 0.5115, 0.7147]
        """
        # flatten stats across length dimensions
        mean = torch.tensor([1.0563, 0.7978, 1.7570], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.6001, 0.5115, 0.7147], dtype=torch.float32, device=self.device)
        mean = mean.mean() * torch.ones_like(mean)
        std = std.mean() * torch.ones_like(std)
        normed_aunit_lengths = std_aunit_lengths * std[None, :] + mean[None, :]
        aunit_lengths = self.denorm_by_radius(normed_aunit_lengths)
        return self.scale_aunit_lengths_to_unit_cell(aunit_lengths)

    def standardize_cell_angles(self):
        """
        Using triclinic CSD stats
        mean = [1.5656, 1.5728, 1.5579]
        std = [0.2339, 0.2075, 0.2598]
        """
        mean = torch.tensor([1.5656, 1.5728, 1.5579], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.2339, 0.2075, 0.2598], dtype=torch.float32, device=self.device)
        return (self.cell_angles - mean[None, :]) / std[None, :]

    def destandardize_cell_angles(self, std_cell_angles):
        """
        Using triclinic CSD stats
        mean = [1.5656, 1.5728, 1.5579]
        std = [0.2339, 0.2075, 0.2598]
        """
        mean = torch.tensor([1.5656, 1.5728, 1.5579], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.2339, 0.2075, 0.2598], dtype=torch.float32, device=self.device)
        return std_cell_angles * std[None, :] + mean[None, :]

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

    def plot_batch_cell_params(self, space='real'):
        if not self.is_batch:
            print("Cell statistics only works for a batch of samples")
            return None
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        lattice_features = ['cell_a', 'cell_b', 'cell_c',
                            'cell_alpha', 'cell_beta', 'cell_gamma',
                            'aunit_x', 'aunit_y', 'aunit_z',
                            'orientation_1', 'orientation_2', 'orientation_2']
        if space == 'real':
            samples = self.cell_parameters()
        elif space == 'latent':
            samples = self.cell_params_to_gen_basis()
        elif space == 'standard':
            samples = self.standardize_cell_parameters()

        # 1d Histograms
        colors = 'red'
        fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
        for i in range(12):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(go.Violin(
                x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
                showlegend=False,
                meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
                line_color=colors,
            ),
                row=row, col=col
            )

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
        fig.update_traces(opacity=0.5)
        fig.show(renderer='browser')

    def set_cell_parameters(self,
                            cell_parameters,
                            skip_box_analysis: bool = False):
        (self.cell_lengths, self.cell_angles,
         self.aunit_centroid, self.aunit_orientation) = (
            cell_parameters.split(3, dim=1))
        if not skip_box_analysis:
            self.box_analysis()

    def build_and_analyze(self,
                          return_cluster: Optional[bool] = False,
                          noise: Optional[float] = None,
                          cutoff: float = 10,
                          supercell_size: int = 10,
                          ):
        """
        full procedure for building and analyzing a molecular crystal
        """
        cluster_batch = self.mol2cluster(cutoff, supercell_size)

        if noise is not None:
            cluster_batch.pos += torch.randn_like(cluster_batch.pos) * noise

        cluster_batch.construct_radial_graph(cutoff=cutoff)
        self.lj_pot, self.scaled_lj_pot = cluster_batch.compute_LJ_energy()
        self.es_pot = cluster_batch.compute_ES_energy()

        if return_cluster:
            return self.lj_pot, self.es_pot, self.scaled_lj_pot, cluster_batch
        else:
            return self.lj_pot, self.es_pot, self.scaled_lj_pot

    def mol2cluster(self, cutoff: float = 6,
                    supercell_size: int = 10,
                    align_to_standardized_orientation: Optional[bool] = True):
        self.pose_aunit(align_to_standardized_orientation=align_to_standardized_orientation)
        self.build_unit_cell()
        return self.build_cluster(cutoff, supercell_size)

    def reset_sg_info(self, sg_ind):
        if isinstance(sg_ind, int):
            sg_ind_list = torch.ones_like(self.sg_ind) * sg_ind
        elif torch.is_tensor(sg_ind):
            sg_ind_list = sg_ind * 1
        else:
            assert False, "sg_ind must be a tensor or an integer"

        self.sg_ind = sg_ind_list
        self.symmetry_operators = [np.stack(SYM_OPS[int(SG)]) for SG in
                                   sg_ind_list]  # if saved as a tensor, we get collation issues
        self.sym_mult = torch.tensor(
            [len(sym_ops) for sym_ops in self.symmetry_operators],
            dtype=torch.long, device=self.device
        )

    def do_embedding(self,
                     embedding_type: str,
                     encoder: Optional = None):
        if self.is_batch:
            embedding = get_mol_embedding_for_proxy(self.clone(),
                                                    embedding_type,
                                                    encoder
                                                    )
            scaled_params = self.standardize_cell_parameters()
            return torch.cat([embedding, scaled_params], dim=1)
        else:
            assert False, "Crystal embedding not implemented for single samples"

    def optimize_crystal_parameters(self,
                                    mol_orientation: Optional[str] = 'standardized',
                                    **opt_kwargs):
        if self.is_batch:
            batch_to_optim = self.clone().to(self.device)
        else:
            batch_to_optim = collate_data_list([self]).clone().to(self.device)

        batch_to_optim.orient_molecule(mode=mol_orientation)
        optimization_record = standalone_gradient_descent_optimization(
            batch_to_optim.cell_parameters(),
            batch_to_optim,
            **opt_kwargs
        )
        return optimization_record

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
