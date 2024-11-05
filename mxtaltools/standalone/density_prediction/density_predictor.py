"""
standalone code for molecular crystal density prediction model
requirements numpy, scipy, torch, torch_geometric, torch_scatter, torch_cluster, pyyaml, tqdm

"""
import pathlib
from argparse import Namespace
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader.dataloader import Collater

from mxtaltools.standalone.density_prediction.utils import (
    dict2namespace, load_yaml, batch_molecule_vdW_volume,
    VDW_RADII, ATOM_WEIGHTS, reload_model
)
from mxtaltools.standalone.density_prediction.CrystalData import CrystalData
from mxtaltools.standalone.density_prediction.models import MoleculeScalarRegressor

module_path = str(pathlib.Path(__file__).parent.resolve())
density_model_path = module_path + '/models/density_model.pt'


# noinspection PyAttributeOutsideInit
class DensityPredictor(torch.nn.Module):
    def __init__(self,
                 device,
                 machine='local',
                 ):
        super(DensityPredictor, self).__init__()

        self.device = device
        self.machine = machine

        self.load_models()
        self.vdw_radii = torch.tensor(list(VDW_RADII.values())).to(self.device)
        self.atomic_weights = torch.tensor(list(ATOM_WEIGHTS.values())).to(self.device)
        self.collater = Collater(None, None)

    @torch.no_grad()
    def predict(self,
                atomic_numbers: Union[torch.Tensor, np.ndarray, list],
                atom_coordinates: Union[torch.Tensor, np.ndarray, list],
                num_samples: int = 5,
                ):
        """
        Predict the density for a molecular crystal or batch of molecular crystals, given molecular information.
        Quanitify uncertainty using dropout over a num_samples.

        Returns
        -------

        """

        mol_batch = self.prep_molecule_batch(atomic_numbers,
                                             atom_coordinates,
                                             )

        all_samples = torch.zeros((num_samples, mol_batch.num_graphs), dtype=torch.float32, device=self.device)
        for d_ind in range(num_samples):
            all_samples[d_ind] = self.gnn(mol_batch).flatten() * self.gnn.target_std + self.gnn.target_mean

        packing_coeffs = all_samples.mean(0)
        packing_coeff_stds = all_samples.std(0)

        return (packing_coeffs.cpu().detach().numpy(),
                packing_coeff_stds.cpu().detach().numpy(),
                mol_batch.mol_volume.cpu().detach().numpy())

    def prep_molecule_batch(self,
                            atom_types_list,
                            coords_list,
                            ):
        for ind, (pos, z) in enumerate(zip(coords_list, atom_types_list)):  # filtering hydrogens
            good_inds = torch.argwhere(z.flatten() != 1).flatten()
            coords_list[ind] = pos[good_inds]
            atom_types_list[ind] = z[good_inds]

        mol_batch = [
            CrystalData(
                x=atom_types_list[ind],
                pos=coords_list[ind],
                mol_size=torch.ones(1) * len(atom_types_list[ind]),
            )
            for ind in range(len(coords_list))
        ]
        mol_batch = self.collater(mol_batch).to(self.device)
        mol_batch.mol_volume = batch_molecule_vdW_volume(
            mol_batch.x,
            mol_batch.pos,
            mol_batch.batch,
            mol_batch.num_graphs,
            self.vdw_radii
        )

        return mol_batch

    #
    def load_models(self):
        self.load_density_model()

    def load_density_model(self):
        """asymmetric unit volume prediction model"""
        checkpoint = torch.load(density_model_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config = Namespace(**{'regressor': Namespace(**{'optimizer': {}, 'model': {}})})

        self.config.regressor.optimizer = model_config.optimizer
        self.config.regressor.model = model_config.model
        self.gnn = MoleculeScalarRegressor(seed=12345,
                                           config=self.config.regressor.model,
                                           atom_features=['atomic_number', 'vdw_radii', 'atom_weight',
                                                          'electronegativity', 'group', 'period'],
                                           molecule_features=['num_atoms', 'radius', 'mol_volume'],
                                           )
        for param in self.gnn.parameters():  # freeze volume model
            param.requires_grad = False
        self.gnn, _ = reload_model(self.gnn, device=self.device, optimizer=None,
                                   path=density_model_path)
        self.gnn.eval()
        for m in self.gnn.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        self.gnn.to(self.device)


if __name__ == '__main__':
    # test this class
    analyzer = DensityPredictor(device='cpu',
                                machine='local',
                                )

    # load up some data
    from mxtaltools.dataset_management.data_manager import DataManager

    miner = DataManager(device='cuda',
                        datasets_path=r"D:\crystal_datasets/",
                        dataset_type='crystal',
                        config=dict2namespace(load_yaml('../../..///configs/dataset/skinny_regression.yaml')))

    miner.load_dataset_for_modelling(dataset_name='test_CSD_dataset.pt',
                                     filter_conditions=[
                                         ['crystal_z_prime', 'in', [1]],
                                         ['crystal_symmetry_operations_are_nonstandard', 'in', [False]],
                                         ['max_atomic_number', 'range', [1, 100]],
                                         ['molecule_num_atoms', 'range', [3, 100]],
                                         ['molecule_radius', 'range', [1, 5]],
                                         ['asymmetric_unit_is_well_defined', 'in', [True]],
                                         ['crystal_packing_coefficient', 'range', [0.5, 0.9]],
                                     ])

    predictions = []
    stds = []
    batch_size = 50
    for ind in range(10):
        packing_coeff, packing_std, mol_volume = analyzer.predict(
            atomic_numbers=[miner.dataset[ii].x for ii in range(ind*batch_size,(ind+1)*batch_size)],
            atom_coordinates=[miner.dataset[ii].pos for ii in range(ind*batch_size,(ind+1)*batch_size)]
        )
        predictions.extend(packing_coeff)
        stds.extend(packing_std)

    predictions = torch.Tensor(predictions)
    stds = torch.Tensor(stds)
    reference_coeffs = torch.tensor(
        [elem.packing_coeff for elem in miner.dataset[0:10*batch_size]])
    volume_error = F.l1_loss(predictions, reference_coeffs, reduction='none')
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=reference_coeffs, y=predictions,mode='markers')).show(renderer='browser')

    aa = 1
