"""
standalone code for molecule autoencoder
trained on QM9
C N O F up to 9 heavy atoms - note NO PROTONS in this version

requirements numpy, scipy, torch, torch_geometric, torch_scatter, torch_cluster, pyyaml, tqdm

"""
from argparse import Namespace
from pathlib import Path

import yaml
import torch.nn.functional as F
from torch_scatter import scatter_softmax

from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.crystal_building.utils import random_crystaldata_alignment
from mxtaltools.models.autoencoder_models import PointAutoencoder
import numpy as np
import torch

from mxtaltools.models.utils import compute_full_evaluation_overlap, reload_model

config_path = '../standalone/qm9_encoder.yaml'
# checkpoint_path = 'C:/Users/mikem/crystals/CSP_runs/models/cluster/best_autoencoder_autoencoder_tests_qm9_test21_43_17-02-20-55-35'  # without protons
checkpoint_path = 'C:/Users/mikem/crystals/CSP_runs/models/cluster/best_autoencoder_autoencoder_tests_qm9_test21_39_17-02-09-19-00'  # with protons


def load_yaml(path):
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


class Qm9Autoencoder(torch.nn.Module):
    def __init__(self,
                 device,
                 num_atom_types=4,
                 max_molecule_radius=7.40425,
                 min_num_atoms=2,
                 max_num_atoms=9):
        super(Qm9Autoencoder, self).__init__()

        self.device = device
        self.config = dict2namespace(load_yaml(config_path))
        self.config.autoencoder.molecule_radius_normalization = max_molecule_radius
        self.config.autoencoder.min_num_atoms = min_num_atoms
        self.config.autoencoder.max_num_atoms = max_num_atoms
        self.num_atom_types = num_atom_types

        checkpoint = torch.load(checkpoint_path)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config.autoencoder.model = model_config.model

        self.model = PointAutoencoder(seed=12345, config=self.config.autoencoder.model, num_atom_types=num_atom_types)
        for param in self.model.parameters():  # freeze encoder
            param.requires_grad = False
        self.model, _ = reload_model(self.model, optimizer=None, path=checkpoint_path)
        self.model.eval()

        if num_atom_types == 5:
            allowed_types = np.asarray([1, 6, 7, 8, 9])
        else:
            allowed_types = np.asarray([6, 7, 8, 9])

        type_translation_index = np.zeros(allowed_types.max()) - 1
        for ind, atype in enumerate(allowed_types):
            type_translation_index[atype - 1] = ind
        self.autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

    def forward(self, data, noise=None):
        with torch.no_grad():
            if self.model.variational:
                encoding = self.model.encode(data, z=torch.zeros((data.num_graphs, 3,  # uniform prior for comparison/inference
                                                                  self.config.autoencoder.model.bottleneck_dim),
                                                                 dtype=torch.float32,
                                                                 device=self.config.device))
            else:
                encoding = self.model.encode(data)

            if noise is not None:
                encoding += torch.randn_like(encoding) * noise

            return encoding

    def get_encoding_decoding(self, data, noise=None):
        encoding = self.forward(data.clone(), noise)
        decoding = self.model.decode(encoding)
        return encoding, decoding

    def evaluate_encoding(self, data, return_encodings=False, noise=None):
        with torch.no_grad():
            encoding, decoding = self.get_encoding_decoding(data, noise)

            decoded_data = self.generate_decoded_data(data, decoding)
            data.aux_ind = torch.ones(data.num_nodes, dtype=torch.float32, device=self.device)

            nodewise_weights_tensor = decoded_data.aux_ind

            true_nodes = F.one_hot(data.x[:, 0].long(), num_classes=self.num_atom_types).float()
            full_overlap, self_overlap = compute_full_evaluation_overlap(data, decoded_data, nodewise_weights_tensor, true_nodes,
                                                                         evaluation_sigma=0.05, type_distance_scaling=0.5)

            fidelity = 1 - torch.abs(1 - full_overlap / self_overlap).cpu().detach().numpy()  # higher is better

            if return_encodings:
                return fidelity, encoding, decoding
            else:
                return fidelity

    def generate_decoded_data(self, data, decoding):
        decoded_data = data.clone()
        decoded_data.pos = decoding[:, :3]
        decoded_data.batch = torch.arange(data.num_graphs).repeat_interleave(
            self.config.autoencoder.model.num_decoder_points).to(self.device)
        nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = self.get_node_weights(
            data, decoded_data, decoding)
        decoded_data.x = F.softmax(decoding[:, 3:-1], dim=1)
        decoded_data.aux_ind = nodewise_weights_tensor
        return decoded_data

    def get_node_weights(self, data, decoded_data, decoding):
        graph_weights = data.mol_size / self.config.autoencoder.model.num_decoder_points
        nodewise_graph_weights = graph_weights.repeat_interleave(self.config.autoencoder.model.num_decoder_points)
        nodewise_weights = scatter_softmax(decoding[:, -1] / self.config.autoencoder.node_weight_temperature, decoded_data.batch, dim=0)
        nodewise_weights_tensor = nodewise_weights * data.mol_size.repeat_interleave(self.config.autoencoder.model.num_decoder_points)

        return nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor

    @staticmethod
    def preprocess_molecule_data(data, noise_level: float = 0, randomize_orientation=False):
        """
        optionally noise atom positions or roto-invert the input
        """
        if noise_level > 0:
            data.pos += torch.randn_like(data.pos) * noise_level

        if randomize_orientation:
            data = random_crystaldata_alignment(data, include_inversion=True)
        else:
            pass

        return data
