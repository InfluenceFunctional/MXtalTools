from constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from constants.space_group_info import SYM_OPS, POINT_GROUPS, LATTICE_TYPE, SPACE_GROUPS
from crystal_building.builder import SupercellBuilder
from crystal_building.utils import update_crystal_symmetry_elements
from models.discriminator_models import crystal_discriminator
import sys

from models.utils import softmax_and_score
import numpy as np
from argparse import Namespace


class MCryNetD():
    """
    standalone score model for molecular crystals
    """

    def __init__(self, device, rescaling_func='score'):

        config = np.load('configs/standalone.npy', allow_pickle=True).items()  # load special dict

        self.device = device
        self.supercell_size = 5
        self.graph_convolution_cutoff = 6

        self.dataDims = config['dataDims']

        self.atom_weights = ATOM_WEIGHTS
        self.vdw_radii = VDW_RADII
        self.sym_ops = SYM_OPS
        self.point_groups = POINT_GROUPS
        self.lattice_type = LATTICE_TYPE
        self.space_groups = SPACE_GROUPS

        self.sym_info = {  # collect space group info into single dict
            'sym_ops': self.sym_ops,
            'point_groups': self.point_groups,
            'lattice_type': self.lattice_type,
            'space_groups': self.space_groups}

        self.model = crystal_discriminator(Namespace(**config), Namespace(**config['discriminator']), self.dataDims)
        self.supercell_builder = SupercellBuilder(self.sym_info, self.dataDims, device=device, rotation_basis="spherical")

        if rescaling_func == 'score':
            self.rescaling_func = softmax_and_score
        else:
            print(f"{self.rescaling_func} not implemented in standalone discriminator")
            sys.exit()

    def forward(self, cell_params, mol_data):
        """
        build crystal given cell params and molecule
        """
        space_groups = cell_params[:, 0]
        cell_params_i = cell_params[:, 1:]

        # overwrite the appropriate symmetry operations in the mol data for the new space groups
        mol_data = update_crystal_symmetry_elements(
            mol_data,
            space_groups,
            self.dataDims,
            self.sym_info, randomize_sgs=False)

        supercell_data, _, _ = self.supercell_builder.build_supercells(
            mol_data, cell_params_i,
            self.supercell_size,
            self.graph_convolution_cutoff,
            align_molecules=False,
            target_handedness=mol_data.asym_unit_handedness,
        )

        output, extra_outputs = self.model(
            mol_data.clone(),
            return_dists=True,
            return_latent=False)  # reshape output from flat filters to channels * filters per channel

        return self.rescaling_func(output)
