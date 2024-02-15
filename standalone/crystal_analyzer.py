"""
standalone code for molecular crystal analyzer
requirements numpy, scipy, torch, torch_geometric, torch_scatter, torch_cluster, pyyaml
"""
from pathlib import Path

import yaml

import numpy as np
import torch
import torch.nn.functional as F

from bulk_molecule_classification.utils import reload_model
from common.config_processing import dict2namespace
from constants.atom_properties import VDW_RADII
from crystal_building.builder import SupercellBuilder
from models.discriminator_models import crystal_discriminator
from models.regression_models import molecule_regressor
from models.utils import compute_packing_coefficient, softmax_and_score
from models.vdw_overlap import vdw_overlap

config_path = '/standalone/crystal_analyzer.yaml'
discriminator_checkpoint_path = 'abc'  # todo get checkpoint
density_checkpoint_path = 'abc'  # todo get checkpoint
dataDims_path = 'abc'


def load_yaml(path):
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


class CrystalAnalyzer(torch.nn.Module):
    def __init__(self,
                 device):
        super(CrystalAnalyzer, self).__init__()

        self.device = device
        self.config = dict2namespace(load_yaml(config_path))
        self.dataDims = np.load(dataDims_path, allow_pickle=True).item()

        self.model = crystal_discriminator(seed=12345, config=self.config.discriminator.model, dataDims=self.dataDims)
        for param in self.model.parameters():  # freeze encoder
            param.requires_grad = False
        self.model = reload_model(self.model, device=self.device, optimizer=None, path=discriminator_checkpoint_path)
        self.model.eval()

        self.density_model = molecule_regressor(seed=12345, config=self.config.regressor.model, dataDims=self.dataDims)
        for param in self.density_model.parameters():  # freeze encoder
            param.requires_grad = False
        self.density_model = reload_model(self.density_model, device=self.device, optimizer=None, path=density_checkpoint_path)
        self.density_model.eval()

        self.packing_mean = self.dataDims['target_mean']
        self.packing_std = self.dataDims['target_std']

        self.supercell_builder = SupercellBuilder(device=self.config.device, rotation_basis='spherical')
        self.vdw_radii = VDW_RADII

    def forward(self, data, proposed_cell_params: torch.tensor, proposed_sgs: torch.tensor, score_type='heuristic', return_stats=False):
        with torch.no_grad():
            if score_type in ['classifier', 'rdf_distance', 'heuristic']:
                proposed_crystaldata = self.build_crystal(data, proposed_cell_params, proposed_sgs)

                discriminator_output, pair_dist_dict = self.adversarial_score(proposed_crystaldata)
                classification_score = softmax_and_score(discriminator_output[:, :2])
                predicted_distance = discriminator_output[:, -1]
                predicted_packing_coeff = compute_packing_coefficient(cell_params=proposed_crystaldata.cell_params,
                                                                      mol_volumes=proposed_crystaldata.mol_volume,
                                                                      crystal_multiplicity=proposed_crystaldata.mult)
                vdw_score = vdw_overlap(self.vdw_radii, crystaldata=proposed_crystaldata, return_score_only=True).cpu().detach().numpy()
                model_packing_coeff = self.estimate_density(data)
                packing_loss = F.smooth_l1_loss(predicted_packing_coeff, model_packing_coeff, reduction='none')

                heuristic_score = vdw_score * torch.exp(-packing_loss * 5)

                if score_type == 'classifier':
                    output = classification_score
                elif score_type == 'rdf_distance':
                    output = torch.exp(-predicted_distance) - 1
                elif score_type == 'heuristic':
                    output = heuristic_score

                if return_stats:
                    stats_dict = {
                        'classification_score': classification_score.cpu().detach().numpy(),
                        'predicted_distance': predicted_distance.cpu().detach().numpy(),
                        'heuristic_score': heuristic_score.cpu().detach().numpy(),
                        'predicted_packing_coeff': predicted_packing_coeff.cpu().detach().numpy(),
                        'model_packing_coeff': model_packing_coeff.cpu().detach().numpy(),
                        'vdw_score': vdw_score.cpu().detach().numpy(),
                    }
                    return output, stats_dict
                else:
                    return output

            elif score_type == 'density':
                data = self.preprocess(data, proposed_sgs)
                predicted_packing_coeff = compute_packing_coefficient(cell_params=proposed_cell_params,
                                                                      mol_volumes=data.mol_volume,
                                                                      crystal_multiplicity=data.mult)

                model_packing_coeff = self.estimate_density(data)
                packing_loss = F.smooth_l1_loss(predicted_packing_coeff, model_packing_coeff, reduction='none')
                output = torch.exp(-5 * packing_loss)

                if return_stats:
                    stats_dict = {
                        'predicted_packing_coeff': predicted_packing_coeff.cpu().detach().numpy(),
                        'model_packing_coeff': model_packing_coeff.cpu().detach().numpy(),
                    }
                    return output, stats_dict
                else:
                    return output

    def estimate_density(self, data):
        model_packing_coeff = self.density_model(data) * self.packing_std + self.packing_mean
        return model_packing_coeff

    def build_crystal(self, data, proposed_cell_params, proposed_sgs):
        data = self.prep_molecule_data(data, proposed_cell_params, proposed_sgs)
        # todo add parameter safety assertions
        proposed_crystaldata, proposed_cell_volumes = self.supercell_builder.build_supercells(
            data, proposed_cell_params, self.config.supercell_size,
            self.config.discriminator.model.convolution_cutoff,
            align_to_standardized_orientation=False,
            target_handedness=data.asym_unit_handedness,
            skip_refeaturization=True,
        )
        return proposed_crystaldata

    def prep_molecule_data(self, data, proposed_cell_params, proposed_sgs):
        data.symmetry_operators = [self.supercell_builder.symmetries_dict['sym_ops'][ind] for ind in proposed_sgs]
        data.sg_ind = proposed_sgs
        data.cell_params = proposed_cell_params
        data.mult = torch.tensor([
            len(sym_op) for sym_op in data.symmetry_operators
        ], device=data.x.device, dtype=torch.long)
        return data

    def adversarial_score(self, data):
        """
        get the score from the discriminator on data
        """
        output, extra_outputs = self.model(data.clone(), return_dists=True, return_latent=False)
        return output, extra_outputs['dists_dict']
