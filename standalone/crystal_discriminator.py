from constants.atom_properties import ATOM_WEIGHTS, VDW_RADII
from constants.space_group_info import SYM_OPS, POINT_GROUPS, LATTICE_TYPE, SPACE_GROUPS
from crystal_building.builder import SupercellBuilder
from crystal_building.utils import update_crystal_symmetry_elements
from models.discriminator_models import crystal_discriminator
import sys

from models.utils import softmax_and_score, reload_model
from crystal_modeller import generator_density_matching_loss
import numpy as np
import torch
from argparse import Namespace
from pathlib import Path

from models.vdw_overlap import vdw_overlap


class StandaloneDiscriminator():
    """
    standalone score model for molecular crystals
    """

    def __init__(self, device, rescaling_func='score', temperature = 0.05):

        self.device = device
        self.supercell_size = 5
        self.graph_convolution_cutoff = 6
        self.temperature = temperature  # smaller means higher focus on best sample
        self.inv_loss_fraction = 0 # smaller means worse samples get punished more

        std_dataDims_path = str(Path(__file__).parent.parent.resolve()) + r'/old_dataset_management/standard_dataDims.npy'
        self.dataDims = np.load(std_dataDims_path, allow_pickle=True).item()

        self.tracking_mol_volume_ind = self.dataDims['tracking_features'].index('molecule volume')

        self.atom_weights = ATOM_WEIGHTS
        self.vdw_radii = VDW_RADII
        self.sym_ops = SYM_OPS
        self.point_groups = POINT_GROUPS
        self.lattice_type = LATTICE_TYPE
        self.space_groups = SPACE_GROUPS
        self.sg_feature_ind_dict = {thing[14:]: ind + self.dataDims['num atomwise features'] for ind, thing in
                                    enumerate(self.dataDims[.mult]) if 'sg is' in thing}
        self.crysys_ind_dict = {thing[18:]: ind + self.dataDims['num atomwise features'] for ind, thing in
                                enumerate(self.dataDims['molecule_features']) if 'crystal system is' in thing}

        self.sym_info = {'sym_ops': self.sym_ops,
                         'point_groups': self.point_groups,
                         'lattice_type': self.lattice_type,
                         'space_groups': self.space_groups,
                         'sg_feature_ind_dict': self.sg_feature_ind_dict,
                         'crysys_ind_dict': self.crysys_ind_dict,
                         'crystal_z_value_ind': self.dataDims['num atomwise features'] + self.dataDims['molecule_features'].index('crystal z value')}

        # discriminator_path = r'/home/mkilgour/models/best_discriminator_10413'

        # self.model = crystal_discriminator(Namespace(**config), Namespace(**config['discriminator']), self.dataDims)
        #
        # discriminator, discriminator_optimizer = reload_model(self.model,
        #                                                       optimizer=None,
        #                                                       path=discriminator_path,
        #                                                       reload_optimizer=False)

        # if rescaling_func == 'score':
        #     self.rescaling_func = softmax_and_score
        # else:
        #     print(f"{self.rescaling_func} not implemented in standalone discriminator")
        #     sys.exit()

        self.supercell_builder = SupercellBuilder(self.sym_info, self.dataDims, device=device, rotation_basis="spherical")

    @torch.no_grad()
    def __call__(self, cell_params, mol_data, return_analysis=False):
        """
        build crystal given cell params and molecule
        """
        mol_data = mol_data.clone().to(cell_params.device)

        space_groups = cell_params[:, 0].clone()
        cell_params_i = cell_params[:, 1:].clone()

        # convert angles from degrees to radians
        cell_params_i[:,3:6] = cell_params_i[:, 3:6] / 180 * torch.pi
        cell_params_i[:,9:12] = cell_params_i[:, 9:12] / 180 * torch.pi

        # denormalize the cell lengths against the molecule size and Z value
        cell_params_i[:, 0:3] = cell_params_i[:, 0:3] * (mol_data.mult ** (1/3))[:, None] * (mol_data.mol_volume ** (1/3))[:, None]

        # overwrite the appropriate symmetry operations in the mol data for the new space groups
        mol_data = update_crystal_symmetry_elements(
            mol_data,
            space_groups,
            self.dataDims,
            self.sym_info, randomize_sgs=False)

        supercell_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
            mol_data, cell_params_i,
            self.supercell_size,
            self.graph_convolution_cutoff,
            align_molecules=False,
            target_handedness=mol_data.asym_unit_handedness,
        )

        # output, extra_outputs = self.model(
        #     mol_data.clone(),
        #     return_dists=True,
        #     return_latent=False)  # reshape output from flat filters to channels * filters per channel

        # return self.rescaling_func(output)

        # for now, train on heuristic losses (simpler)
        vdw_loss, vdw_score, _, _ = vdw_overlap(self.vdw_radii, crystaldata=supercell_data, loss_func=None)

        packing_loss, packing_prediction, packing_target, packing_csd = \
            generator_density_matching_loss(
                mol_data.y,
                self.dataDims['target_mean'], self.dataDims['target_std'],
                self.tracking_mol_volume_ind, self.dataDims['tracking_features'].index('crystal packing coefficient'),
                supercell_data, cell_params_i,
                precomputed_volumes=generated_cell_volumes, loss_func='l1')

        loss = packing_loss
        score = torch.exp(-loss / self.temperature)*(1-self.inv_loss_fraction) + self.inv_loss_fraction/(loss + 1)  # combined function decays slower than -exp

        if return_analysis:
            analysis_dict = {
                'reward': score,
                'packing_loss':packing_loss,
                'vdw_loss':vdw_loss,
                'vdw_score':vdw_score,
                'canonical_cell_params':supercell_data.cell_params,
                'generated_cell_params':cell_params_i,
                'space_groups':space_groups,
                'packing_prediction':packing_prediction,
                'packing_target':packing_target,
                'csd_packing_target':packing_csd,
            }

            for key in analysis_dict.keys():
                if torch.is_tensor(analysis_dict[key]):
                    analysis_dict[key] = analysis_dict[key].cpu().detach().numpy()

            return score.float(), analysis_dict
        else:
            return score.float()
