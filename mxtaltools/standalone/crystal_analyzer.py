"""
standalone code for molecular crystal analyzer
requirements numpy, scipy, torch, torch_geometric, torch_scatter, torch_cluster, pyyaml, tqdm
"""
from pathlib import Path

import yaml

import torch
from torch_geometric.loader.dataloader import Collater
from torch_geometric.typing import OptTensor

import numpy as np
from torch_scatter import scatter
from argparse import Namespace
import torch.nn.functional as F

from mxtaltools.common.config_processing import process_main_config, dict2namespace
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.common.geometry_calculations import cell_vol_torch
from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.task_models.crystal_models import MolCrystal
from mxtaltools.models.task_models.regression_models import MoleculeRegressor
from mxtaltools.models.utils import softmax_and_score, reload_model
from mxtaltools.models.functions.vdw_overlap import vdw_overlap
from mxtaltools.constants.atom_properties import VDW_RADII

import pathlib

module_path = str(pathlib.Path(__file__).parent.resolve())

discriminator_checkpoint_path = module_path + '/models/crystal_score_model.pt'
volume_checkpoint_path = module_path + '/models/volume_model.pt'


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
        'crystal scoring model'
        checkpoint = torch.load(discriminator_checkpoint_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config.discriminator.optimizer = model_config.optimizer
        self.config.discriminator.model = model_config.model
        self.d_dataDims = checkpoint['dataDims']
        self.model = MolCrystal(seed=12345, config=self.config.discriminator.model,
                                atom_features=self.d_dataDims['atom_features'],
                                molecule_features=self.d_dataDims['molecule_features'],
                                )
        for param in self.model.parameters():  # freeze score model
            param.requires_grad = False
        self.model, _ = reload_model(self.model, device=self.device, optimizer=None, path=discriminator_checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        'asymmetric unit volume prediction model'
        checkpoint = torch.load(volume_checkpoint_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config.regressor.optimizer = model_config.optimizer
        self.config.regressor.model = model_config.model
        self.r_dataDims = checkpoint['dataDims']

        self.volume_model = MoleculeRegressor(seed=12345, config=self.config.regressor.model,
                                              atom_features=self.r_dataDims['atom_features'],
                                              molecule_features=self.r_dataDims['molecule_features'],
                                              )
        for param in self.volume_model.parameters():  # freeze volume model
            param.requires_grad = False
        self.volume_model, _ = reload_model(self.volume_model, device=self.device, optimizer=None,
                                            path=volume_checkpoint_path)
        self.volume_model.eval()
        self.volume_model.to(self.device)

    def __call__(self,
                 data_list: list[CrystalData] = None,
                 coords_list: list[torch.tensor] = None,
                 atom_types_list: list[torch.tensor] = None,
                 proposed_cell_params: OptTensor = None,
                 proposed_sgs: OptTensor = None,
                 proposed_zps: OptTensor = None,
                 proposed_handedness: OptTensor = None,
                 analysis_type: str = 'heuristic',
                 return_stats: bool = False,
                 n_top_k: int = 1):
        """
        Analysis function for molecular crystal properties

        Parameters
        ----------
        coords_list : list[torch.tensor]
            List of tensors of molecule coordinates [x, y, z]
        atom_types_list : list[torch.tensor]
            List of tensors of molecule atom types
        proposed_cell_params : OptTensor
            Cell parameters for proposed crystal
        proposed_sgs : OptTensor
            Space group indices for proposed crystals
        proposed_zps : OptTensor
            Z' values for proposed crystals
        analysis_type : str
            Type of analysis to do
        return_stats : bool
            Whether to return detailed extra analysis of generated crystals
        n_top_k : int
            Number of samples to explicitly return

        Returns
        -------

        """
        with torch.no_grad():
            if data_list is None:
                molecules_batch = self.prep_molecule_data_batch(atom_types_list, coords_list)
            else:
                molecules_batch = self.collater(data_list)
                assert torch.sum(molecules_batch.x == 1) == 0, 'Must pre-clean hydrogens from data_list arguments'

            if analysis_type in ['classifier', 'rdf_distance', 'heuristic']:
                return self.crystal_analysis(analysis_type,
                                             molecules_batch,
                                             n_top_k,
                                             proposed_cell_params,
                                             proposed_sgs,
                                             proposed_zps,
                                             return_stats,
                                             proposed_handedness)
            elif analysis_type == 'volume':
                return self.estimate_aunit_volume(molecules_batch)

            else:
                assert False, f"{analysis_type} is not an implemented crystal analysis function"

    def crystal_analysis(self,
                         analysis_type,
                         data_batch,
                         n_top_k,
                         proposed_cell_params,
                         proposed_sgs,
                         proposed_zps,
                         return_stats,
                         proposed_handedness=None):
        proposed_crystaldata = self.build_crystal(data_batch,
                                                  proposed_cell_params,
                                                  proposed_sgs.long().tolist(),
                                                  proposed_zps.long().tolist(),
                                                  proposed_handedness)
        discriminator_output, pair_dist_dict = self.adversarial_score(proposed_crystaldata)
        classification_score = softmax_and_score(discriminator_output[:, :2])
        predicted_distance = discriminator_output[:, -1]
        vdw_loss, vdw_score, _, _, _ = vdw_overlap(self.vdw_radii,
                                                   crystaldata=proposed_crystaldata,
                                                   return_score_only=False,
                                                   lfoss_func='inv')
        sample_auv = self.compute_aunit_volume(proposed_cell_params, proposed_crystaldata.sym_mult)
        target_auv = self.estimate_aunit_volume(data_batch)[:, 0]
        # packing_loss = (F.smooth_l1_loss(target_auv, sample_auv, reduction='none')/target_auv)
        # something finicky with packing loss prediction right now - substitute for maximal density
        atom_volumes = 4 / 3 * self.vdw_radii[data_batch.x[:, 0].long()] ** 3
        sum_of_spheres_volume = scatter(atom_volumes, data_batch.batch,
                                        reduce='sum') / 2  # crudely add spherical volumes and make a lower bound
        packing_loss = F.relu(sample_auv - sum_of_spheres_volume) / torch.diff(
            data_batch.ptr) / 50  # loss coefficient of 1/10 relative to vdW
        heuristic_score = - vdw_loss - packing_loss
        if analysis_type == 'classifier':
            output = classification_score
        elif analysis_type == 'rdf_distance':
            output = -predicted_distance
        elif analysis_type == 'heuristic':
            output = heuristic_score
        if return_stats:
            sort_inds = torch.argsort(heuristic_score)[
                        -n_top_k:].cpu().detach().numpy()  # save top k samples (k smallest distances)
            mols = [ase_mol_from_crystaldata(proposed_crystaldata,
                                             index=ind,
                                             exclusion_level='distance',
                                             inclusion_distance=6) for ind in sort_inds]
            # import ase.io
            # [ase.io.write(f'/home/mk8347/gflownet-dev/sample_{i}.cif', mols[i]) for i in range(len(mols))]

            stats_dict = {
                'log_vdw_loss': np.log10(vdw_loss.cpu().detach().numpy()),
                'log_packing_loss': -np.log10(packing_loss.cpu().detach().numpy()),
                'vdw_score': vdw_score.cpu().detach().numpy(),
                'vdw_loss': vdw_loss.cpu().detach().numpy(),
                'packing_loss': packing_loss.cpu().detach().numpy(),
                'classification_score': classification_score.cpu().detach().numpy(),
                'predicted_distance': (10 ** (predicted_distance) - 1).cpu().detach().numpy(),
                'log_predicted_distance': np.log10((10 ** (predicted_distance) - 1).cpu().detach().numpy()),
                'heuristic_score': heuristic_score.cpu().detach().numpy(),
                'log_heuristic_loss': np.log10(-heuristic_score.cpu().detach().numpy()),
                'sample_auv': sample_auv.cpu().detach().numpy(),
                'target_auv': target_auv.cpu().detach().numpy(),
                'topk_samples': mols,
            }

            return output, stats_dict
        else:
            return output

    def prep_molecule_data_batch(self, atom_types_list, coords_list):
        # pre-filter hydrogen atoms
        for ind, (pos, z) in enumerate(zip(coords_list, atom_types_list)):
            good_inds = torch.argwhere(z != 1).flatten()
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

    def compute_aunit_volume(self, cell_params, multiplicity):
        volumes_list = []
        for i in range(len(cell_params)):
            volumes_list.append(cell_vol_torch(cell_params[i, 0:3], cell_params[i, 3:6]))

        volumes_list = torch.tensor(volumes_list, dtype=torch.float32, device=self.device)
        return volumes_list / multiplicity

    def estimate_aunit_volume(self, data):
        reduced_auv = self.volume_model(data.clone()) * self.auvol_std + self.auvol_mean
        '''
        reduced volume fraction = asymmetric unit volume / sum of vdw volumes
        AUV = RVF * sum(vdW)
        '''
        return reduced_auv[:, 0] * scatter(4 / 3 * torch.pi * self.vdw_radii[data.x.flatten().long()] ** 3, data.batch,
                                           reduce='sum'), reduced_auv

    def build_crystal(self, data, proposed_cell_params, proposed_sgs, proposed_zps, proposed_handedness=None):
        data = self.prep_molecule_data(data, proposed_cell_params, proposed_sgs, fixed_handedness=proposed_handedness)
        # todo add parameter safety assertions
        proposed_crystaldata, proposed_cell_volumes = self.supercell_builder.build_integer_zp_supercells(
            data, proposed_cell_params, self.supercell_size,
            self.config.discriminator.model.graph.cutoff,
            z_primes_list=proposed_zps,
            align_to_standardized_orientation=False,
            target_handedness=data.aunit_handedness,
            skip_refeaturization=True,
        )
        return proposed_crystaldata

    def prep_molecule_data(self, data, proposed_cell_params, proposed_sgs, fixed_handedness=None):
        data.symmetry_operators = [self.supercell_builder.symmetries_dict['sym_ops'][ind] for ind in proposed_sgs]
        data.sg_ind = proposed_sgs
        data.cell_params = proposed_cell_params
        data.sym_mult = torch.tensor([
            len(sym_op) for sym_op in data.symmetry_operators
        ], device=data.x.device, dtype=torch.long)
        if fixed_handedness is not None:
            data.aunit_handedness = fixed_handedness
        return data

    def adversarial_score(self, data):
        """
        get the score from the discriminator on data
        """
        output, extra_outputs = self.model(data.clone(), return_dists=True, return_latent=False)
        return output, extra_outputs['dists_dict']


analyzer = CrystalAnalyzer(device='cpu')

if __name__ == '__main__':
    # test this class
    analyzer = CrystalAnalyzer(device='cpu',
                               machine='local',
                               supercell_size=5)
    #
    # # try caffeine
    # coords = torch.tensor([
    #     [0.4700, 2.5688, 0.0006, ],
    #     [- 3.1271, - 0.4436, - 0.0003, ],
    #     [- 0.9686, - 1.3125, 0.0000, ],
    #     [2.2182, 0.1412, - 0.0003, ],
    #     [- 1.3477, 1.0797, - 0.0001, ],
    #     [1.4119, - 1.9372, 0.0002, ],
    #     [0.8579, 0.2592, - 0.0008, ],
    #     [0.3897, - 1.0264, - 0.0004, ],
    #     [0.0307, 1.4220, - 0.0006, ],
    #     [- 1.9061, - 0.2495, - 0.0004, ],
    #     [2.5032, - 1.1998, 0.0003, ],
    #     [- 1.4276, - 2.6960, 0.0008, ],
    #     [3.1926, 1.2061, 0.0003, ],
    #     [- 2.2969, 2.1881, 0.0007, ],
    #     [3.5163, - 1.5787, 0.0008, ],
    #     [- 1.0451, - 3.1973, - 0.8937, ],
    #     [- 2.5186, - 2.7596, 0.0011, ],
    #     [- 1.0447, - 3.1963, 0.8957, ],
    #     [4.1992, 0.7801, 0.0002, ],
    #     [3.0468, 1.8092, - 0.8992, ],
    #     [3.0466, 1.8083, 0.9004, ],
    #     [- 1.8087, 3.1651, - 0.0003, ],
    #     [- 2.9322, 2.1027, 0.8881, ],
    #     [- 2.9346, 2.1021, - 0.8849],
    # ])
    # types = torch.tensor([8, 8, 7, 7, 7, 7,
    #                       6, 6, 6, 6, 6, 6, 6, 6,
    #                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    #
    # auv = analyzer(coords_list=[coords],
    #                atom_types_list=[types],
    #                analysis_type='volume')
    #
    # avogadro = 6.022*10**23
    # molar_volume = auv * avogadro / 1e24  # A^3 / molecule ==> cm^3 / mol
    # mw = 194  # g/mol  # g/mol
    # density = mw / molar_volume  # g / cm^3

    '''
    Testing
    '''

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

    auv, reduced_auv = analyzer(data_list=miner.dataset[0:10],
                                analysis_type='volume')

    reference_auvs = torch.tensor([elem.y * analyzer.auvol_std + analyzer.auvol_mean for elem in miner.dataset[0:10]])
    volume_error = F.mse_loss(reduced_auv.flatten(), reference_auvs)

    'Stability analysis - from parameters'
    crystal_analysis = analyzer(atom_types_list=[elem.x for elem in miner.dataset[0:10]],
                                coords_list=[elem.pos for elem in miner.dataset[0:10]],
                                proposed_cell_params=torch.stack([torch.cat([
                                    elem.cell_lengths, elem.cell_angles, elem.pose_params0], dim=1)[0] for elem in
                                                                  miner.dataset[0:10]]),
                                proposed_sgs=torch.tensor([elem.sg_ind for elem in miner.dataset[0:10]],
                                                          dtype=torch.long),
                                proposed_zps=torch.ones(10),
                                analysis_type='heuristic')

    'Stability analysis - without pose parameters'

    aa = 1
