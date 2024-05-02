"""
script for standalone loading of QM9 dataset

requires numpy, torch, pandas, torch_geometric, torch_scatter, torch_cluster, scipy, tqdm
"""
from mxtaltools.common.config_processing import load_yaml, dict2namespace
from mxtaltools.dataset_management.data_manager import DataManager
from mxtaltools.dataset_management.dataloader_utils import get_dataloaders

datasets_path = 'D:/crystal_datasets/'
dataset_name = '/qm9_molecules_dataset.pkl'
misc_dataset_name = '/misc_data_for_qm9_molecules_dataset.npy'
dataset_yaml_path = '../standalone/qm9_loader.yaml'


class QM9Loader:
    def __init__(self, device):
        self.device = device
        self.data_manager = DataManager(device=self.device,
                                        datasets_path=datasets_path,
                                        dataset_type = 'molecule')

        self.dataset_config = dict2namespace(load_yaml(dataset_yaml_path))

    def load_dataset(self, max_dataset_length=None):
        self.data_manager.load_dataset_for_modelling(
            dataset_name=dataset_name,
            filter_conditions=self.dataset_config.filter_conditions,
            filter_polymorphs=self.dataset_config.filter_polymorphs,
            filter_duplicate_molecules=self.dataset_config.filter_duplicate_molecules,
            filter_protons=self.dataset_config.filter_protons,
            override_length=max_dataset_length,
        )
        self.dataDims = self.data_manager.dataDims
        self.t_i_d = {feat: index for index, feat in enumerate(self.dataDims['tracking_features'])}  # tracking feature index dictionary

    def get_dataloaders(self, batch_size=10, test_fraction=0.2):
        train_loader, test_loader = get_dataloaders(self.data_manager,
                                                    machine='cluster',
                                                    batch_size=batch_size,
                                                    test_fraction=test_fraction)

        return train_loader, test_loader

    def load_dataset_for_modelling(self, batch_size, test_fraction):
        self.load_dataset()
        train_loader, test_loader = self.get_dataloaders(batch_size, test_fraction)
        del self.data_manager

        return train_loader, test_loader
