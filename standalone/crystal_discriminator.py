from crystal_building.builder import SupercellBuilder
from models.discriminator_models import crystal_discriminator
import sys

from models.utils import softmax_and_score


class MCryNetD():
    """
    standalone score model for molecular crystals
    """
    def __init__(self, device, rescaling_func = 'score'):

        self.device = device
        self.supercell_size = 5
        self.graph_convolution_cutoff = 6

        self.model = crystal_discriminator(config, dataDims)
        self.supercell_builder = SupercellBuilder(sym_info, dataDims, device=device, rotation_basis="spherical")

        if rescaling_func == 'score':
            self.rescaling_func = softmax_and_score
        else:
            print(f"{self.rescaling_func} not implemented in standalone discriminator")
            sys.exit()

    def forward(self, cell_params, mol_data):
        """
        build crystal given cell params and molecule
        """
        supercell_data, _, _ = self.supercell_builder.build_supercells(
            mol_data, cell_params, self.supercell_size, self.graph_convolution_cutoff,
            align_molecules=False,
            target_handedness=mol_data.asym_unit_handedness,
        )

        output, extra_outputs = self.model(
            mol_data.clone(), return_dists=True, return_latent=False)  # reshape output from flat filters to channels * filters per channel

        return self.rescaling_func(output)