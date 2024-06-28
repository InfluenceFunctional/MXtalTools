"""
standalone code for molecular crystal analyzer
requirements numpy, scipy, torch, torch_geometric, torch_scatter, torch_cluster, pyyaml, tqdm

# todo visualizer, autoencoder functions, cif i/o
"""
import pathlib
from argparse import Namespace
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader.dataloader import Collater
from torch_scatter import scatter

from mxtaltools.common.config_processing import process_main_config, dict2namespace
from mxtaltools.common.geometry_calculations import cell_vol_torch
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.functions.vdw_overlap import vdw_overlap
from mxtaltools.models.task_models.autoencoder_models import Mo3ENet
from mxtaltools.models.task_models.crystal_models import MolecularCrystalModel
from mxtaltools.models.task_models.regression_models import MoleculeScalarRegressor
from mxtaltools.models.utils import softmax_and_score, reload_model, prep_ae_io_for_analysis, ae_reconstruction_loss

module_path = str(pathlib.Path(__file__).parent.resolve())

discriminator_checkpoint_path = module_path + '/models/crystal_score_model.pt'
volume_checkpoint_path = module_path + '/models/volume_model.pt'
autoencoder_checkpoint_path = module_path + '/models/autoencoder_model.pt'  # checkpoint with protons in I/O


def parse_to_torch(array: Union[torch.Tensor, np.ndarray, list],
                   device: Union[torch.device, str],
                   dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(array):
        return torch.tensor(array.clone().detach(), dtype=dtype, device=device)
    elif isinstance(array, np.ndarray):
        return torch.Tensor(array, dtype=dtype, device=device)
    elif isinstance(array, list):
        return torch.Tensor(array, dtype=dtype, device=device)


def load_yaml(path):
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


# noinspection PyAttributeOutsideInit
class CrystalAnalyzer(torch.nn.Module):
    def __init__(self,
                 device,
                 machine='local',
                 supercell_size=5):
        super(CrystalAnalyzer, self).__init__()

        self.device = device
        self.machine = machine

        self.config = process_main_config(None,
                                          user_yaml_path='../../configs/users/mkilgour.yaml',
                                          main_yaml_path='../../configs/standalone/crystal_analyzer.yaml',
                                          machine=machine)

        self.supercell_size = supercell_size
        self.load_models()

        self.auvol_mean = self.r_dataDims['target_mean']
        self.auvol_std = self.r_dataDims['target_std']
        self.vdw_radii = torch.tensor(list(VDW_RADII.values()))

        self.collater = Collater(None, None)
        self.supercell_builder = SupercellBuilder(device=self.device, rotation_basis='spherical')

    def load_models(self):
        """"""
        # update configs from checkpoints
        self.load_crystal_score_model()

        self.load_volume_prediction_model()

        self.load_autoencoder_model()

    def load_autoencoder_model(self):
        """Equivariant molecular point cloud autoencoder model"""
        checkpoint = torch.load(autoencoder_checkpoint_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config.autoencoder.optimizer = model_config.optimizer
        self.config.autoencoder.model = model_config.model
        self.a_dataDims = checkpoint['dataDims']

        allowed_types = np.array(self.a_dataDims['allowed_atom_types'])
        type_translation_index = np.zeros(allowed_types.max() + 1) - 1
        for ind, atype in enumerate(allowed_types):
            type_translation_index[atype] = ind
        self.autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

        self.autoencoder_model = Mo3ENet(seed=12345,
                                         config=self.config.autoencoder.model,
                                         num_atom_types=int(torch.sum(self.autoencoder_type_index != -1)),
                                         atom_embedding_vector=self.autoencoder_type_index,
                                         radial_normalization=1,  # dummy - will be overwritten
                                         infer_protons=False,  # dummy - will be overwritten
                                         protons_in_input=False,  # dummy - will be overwritten
                                         )
        for param in self.autoencoder_model.parameters():  # freeze volume model
            param.requires_grad = False
        self.autoencoder_model, _ = reload_model(self.autoencoder_model, device=self.device, optimizer=None,
                                                 path=autoencoder_checkpoint_path)
        self.autoencoder_model.eval()
        self.autoencoder_model.to(self.device)

    def load_volume_prediction_model(self):
        'asymmetric unit volume prediction model'
        checkpoint = torch.load(volume_checkpoint_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config.regressor.optimizer = model_config.optimizer
        self.config.regressor.model = model_config.model
        self.r_dataDims = checkpoint['dataDims']
        self.volume_model = MoleculeScalarRegressor(seed=12345,
                                                    config=self.config.regressor.model,
                                                    atom_features=self.r_dataDims['atom_features'],
                                                    molecule_features=self.r_dataDims['molecule_features'],
                                                    )
        for param in self.volume_model.parameters():  # freeze volume model
            param.requires_grad = False
        self.volume_model, _ = reload_model(self.volume_model, device=self.device, optimizer=None,
                                            path=volume_checkpoint_path)
        self.volume_model.eval()
        self.volume_model.to(self.device)

    def load_crystal_score_model(self):
        """crystal scoring model"""
        checkpoint = torch.load(discriminator_checkpoint_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config.discriminator.optimizer = model_config.optimizer
        self.config.discriminator.model = model_config.model
        self.d_dataDims = checkpoint['dataDims']
        self.model = MolecularCrystalModel(seed=12345,
                                           config=self.config.discriminator.model,
                                           output_dim=3,
                                           atom_features=self.d_dataDims['atom_features'],
                                           molecule_features=self.d_dataDims['molecule_features'],
                                           )
        for param in self.model.parameters():  # freeze score model
            param.requires_grad = False
        self.model, _ = reload_model(self.model, device=self.device, optimizer=None, path=discriminator_checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def score_crystal(self,
                      atomic_numbers: Union[torch.Tensor, np.ndarray, list],
                      atom_coordinates: Union[torch.Tensor, np.ndarray, list],
                      cell_lengths: Union[torch.Tensor, np.ndarray, list],
                      cell_angles: Union[torch.Tensor, np.ndarray, list],
                      space_group_number: int,
                      pose_parameters: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
                      use_existing_pose: bool = False
                      ):
        """
        score a single proposed molecular crystal

        Parameters
        ----------
        use_existing_pose
        atomic_numbers
        atom_coordinates
        cell_lengths
        cell_angles
        space_group_number
        pose_parameters

        Returns
        -------

        """

        atomic_numbers = parse_to_torch(atomic_numbers, dtype=torch.long, device=self.device)
        atom_coordinates = parse_to_torch(atom_coordinates, dtype=torch.float32, device=self.device)

        cell_lengths = parse_to_torch(cell_lengths, dtype=torch.long, device=self.device)
        cell_angles = parse_to_torch(cell_angles, dtype=torch.float32, device=self.device)

        assert cell_lengths.min() > 0
        assert cell_angles.min() >= 0
        assert cell_angles.max() <= torch.pi

        if pose_parameters is not None:
            pose_parameters = parse_to_torch(pose_parameters, dtype=torch.float32, device=self.device)
            assert 0 <= pose_parameters[:3].min()
            assert 1 >= pose_parameters[:3].max()

        crystal_batch = self.prep_crystal_data_batch([atomic_numbers],
                                                     [atom_coordinates],
                                                     [cell_lengths],
                                                     [cell_angles],
                                                     [pose_parameters],
                                                     [space_group_number],
                                                     use_existing_pose)

        discriminator_output, pair_dist_dict = self.adversarial_score(crystal_batch.clone())
        classification_score = softmax_and_score(discriminator_output[:, :2])
        predicted_distance = F.softplus(discriminator_output[:, -1])
        vdw_loss, vdw_score, _, _, _ = vdw_overlap(self.vdw_radii,
                                                   crystaldata=crystal_batch,
                                                   return_score_only=False,
                                                   loss_func='inv')
        sample_auv = self.compute_aunit_volume(cell_lengths, cell_angles, crystal_batch.sym_mult)

        return classification_score, predicted_distance, vdw_score, sample_auv

    def prep_crystal_data_batch(self,
                                atom_types_list,
                                coords_list,
                                cell_lengths,
                                cell_angles,
                                pose_parameters,
                                space_group_number,
                                use_existing_pose):
        for ind, (pos, z) in enumerate(zip(coords_list, atom_types_list)):
            good_inds = torch.argwhere(z.flatten() != 1).flatten()
            coords_list[ind] = pos[good_inds]
            atom_types_list[ind] = z[good_inds]

        data_batch = [
            CrystalData(
                x=atom_types_list[ind],
                pos=coords_list[ind],
                mol_size=torch.ones(1) * len(atom_types_list[ind]),
                cell_angles=cell_angles[ind].flatten(),
                cell_lengths=cell_lengths[ind].flatten(),
                sg_ind=int(space_group_number[ind]),
                pose_parameters=[pose_parameters[ind].flatten()],
                z_prime=1
            )
            for ind in range(len(coords_list))
        ]
        data_batch = self.collater(data_batch)
        data_batch.to(self.device)

        if pose_parameters[0] is not None and not use_existing_pose:
            crystal_batch, proposed_cell_volumes = self.supercell_builder.build_zp1_supercells(
                data_batch, data_batch.cell_parameters(), self.supercell_size,
                self.config.discriminator.model.graph.cutoff,
                align_to_standardized_orientation=True,
                target_handedness=data_batch.aunit_handedness,
                skip_refeaturization=True,
            )
        else:  # skip posing the molecule
            crystal_batch, proposed_cell_volumes = self.supercell_builder.build_zp1_supercells(
                data_batch, data_batch.cell_parameters(), self.supercell_size,
                self.config.discriminator.model.graph.cutoff,
                align_to_standardized_orientation=True,
                target_handedness=data_batch.aunit_handedness,
                skip_refeaturization=True,
                skip_molecule_posing=True,
            )

        return crystal_batch

    def predict_aunit_volume(self,
                             atomic_numbers: Union[torch.Tensor, np.ndarray, list],
                             atom_coordinates: Union[torch.Tensor, np.ndarray, list],
                             dropout_repeats: int = 10
                             ) -> Tuple[torch.Tensor, torch.Tensor]:

        atomic_numbers = parse_to_torch(atomic_numbers, dtype=torch.long, device=self.device)
        atom_coordinates = parse_to_torch(atom_coordinates, dtype=torch.float32, device=self.device)

        if dropout_repeats > 1:
            for module in self.volume_model.modules():
                if 'Dropout' in type(module).__name__:
                    module.train()

        molecules_batch = self.prep_molecule_data_batch([atomic_numbers for _ in range(dropout_repeats)],
                                                        [atom_coordinates for _ in range(dropout_repeats)])

        reduced_auv = self.volume_model(molecules_batch.clone()) * self.auvol_std + self.auvol_mean

        # asymmetric unit volume = reduced_auv * SUM_atoms[atom vdw volume]
        auv = reduced_auv.flatten() * scatter(
            4 / 3 * torch.pi * self.vdw_radii[molecules_batch.x.flatten().long()] ** 3,
            index=molecules_batch.batch,
            reduce='sum')

        return auv, reduced_auv  # todo work on uncertainty

    def encode_molecule(self,
                        atomic_numbers: Union[torch.Tensor, np.ndarray, list],
                        atom_coordinates: Union[torch.Tensor, np.ndarray, list],
                        check_reconstruction: bool = False,
                        ):
        atomic_numbers = parse_to_torch(atomic_numbers, dtype=torch.long, device=self.device)
        atom_coordinates = parse_to_torch(atom_coordinates, dtype=torch.float32, device=self.device)

        atom_coordinates -= atom_coordinates.mean(0)

        molecules_batch = self.prep_molecule_data_batch([atomic_numbers],
                                                        [atom_coordinates])

        decoding, encoding = self.autoencoder_model(molecules_batch.clone(), return_encoding=True)
        scalar_encoding = self.models_dict['autoencoder'].scalarizer(encoding)

        if check_reconstruction:
            decoded_data, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
                prep_ae_io_for_analysis(molecules_batch, decoding,
                                        self.models_dict['autoencoder'],
                                        self.config.autoencoder.node_weight_temperature,
                                        self.device))

            nodewise_reconstruction_loss, nodewise_type_loss, reconstruction_loss, self_likelihoods = (
                ae_reconstruction_loss(molecules_batch, decoded_data, nodewise_weights,
                                       self.dataDims['num_atom_types'],
                                       self.config.autoencoder.type_distance_scaling,
                                       self.config.autoencoder_sigma))

            return encoding, scalar_encoding, reconstruction_loss
        else:
            return encoding, scalar_encoding

    def prep_molecule_data_batch(self, atom_types_list, coords_list):
        # pre-filter hydrogen atoms
        for ind, (pos, z) in enumerate(zip(coords_list, atom_types_list)):
            good_inds = torch.argwhere(z.flatten() != 1).flatten()
            coords_list[ind] = pos[good_inds]
            atom_types_list[ind] = z[good_inds]

        data_batch = [
            CrystalData(
                x=atom_types_list[ind],
                pos=coords_list[ind],
                mol_size=torch.ones(1) * len(atom_types_list[ind]),
            )
            for ind in range(len(coords_list))
        ]
        data_batch = self.collater(data_batch)
        data_batch.to(self.device)

        return data_batch

    def compute_aunit_volume(self,
                             cell_lengths: torch.Tensor,
                             cell_angles: torch.Tensor,
                             multiplicity: torch.Tensor):
        volumes_list = []
        for i in range(len(cell_lengths)):
            volumes_list.append(cell_vol_torch(cell_lengths[i], cell_angles[i]))

        volumes_list = torch.tensor(volumes_list, dtype=torch.float32, device=self.device)
        return volumes_list / multiplicity

    def adversarial_score(self, data):
        """
        get the score from the discriminator on data
        """
        output, extra_outputs = self.model(data.clone(), return_dists=True, return_latent=False)
        return output, extra_outputs['dists_dict']


if __name__ == '__main__':
    # test this class
    analyzer = CrystalAnalyzer(device='cpu',
                               machine='local',
                               supercell_size=5)


    'Asymmetric unit volume prediction'
    from mxtaltools.dataset_management.data_manager import DataManager

    miner = DataManager(device='cpu',
                        datasets_path=r"D:\crystal_datasets/",
                        dataset_type='crystal',
                        config=dict2namespace(load_yaml('../..//configs/dataset/skinny_discriminator.yaml')))

    miner.load_dataset_for_modelling(dataset_name='test_CSD_dataset.pt',
                                     filter_conditions=[
                                         ['crystal_z_prime', 'in', [1]],
                                         ['crystal_symmetry_operations_are_nonstandard', 'in', [False]],
                                         ['max_atomic_number', 'range', [1, 100]],
                                         ['molecule_num_atoms', 'range', [3, 100]],
                                         ['molecule_radius', 'range', [1, 5]],
                                         ['asymmetric_unit_is_well_defined', 'in', [True]],
                                         ['reduced_volume_fraction', 'range', [0.75, 1.15]],
                                     ])

    predictions = []
    for ind in range(10):
        auv, red_auv = analyzer.predict_aunit_volume(atomic_numbers=miner.dataset[ind].x,
                                                     atom_coordinates=miner.dataset[ind].pos)
        predictions.append(red_auv.cpu().detach().mean())

    reference_red_auvs = torch.tensor(
        [elem.y * analyzer.auvol_std + analyzer.auvol_mean for elem in miner.dataset[0:10]])
    volume_error = F.l1_loss(torch.stack(predictions).flatten(), reference_red_auvs)

    'Stability analysis - from parameters'
    for ind in range(10):
        crystal_analysis = analyzer.score_crystal(atomic_numbers=miner.dataset[ind].x,
                                                  atom_coordinates=miner.dataset[ind].pos,
                                                  cell_lengths=miner.dataset[ind].cell_lengths,
                                                  cell_angles=miner.dataset[ind].cell_angles,
                                                  space_group_number=miner.dataset[ind].sg_ind,
                                                  pose_parameters=miner.dataset[ind].pose_params1,
                                                  use_existing_pose=True,
                                                  )  # todo fix handedness issue in parameterization
        print(crystal_analysis)
