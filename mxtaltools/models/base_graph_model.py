import torch


class BaseGraphModel(torch.nn.Module):
    def __init__(self):
        super(BaseGraphModel, self).__init__()

    def get_data_stats(self, dataDims: dict = None,
                       num_atom_features: int = None,
                       num_molecule_features: int = None,
                       node_standardization_tensor: torch.tensor = None,
                       graph_standardization_tensor: torch.tensor = None):
        if dataDims is not None:
            self.n_atom_feats = dataDims['num_atom_features']
            self.n_mol_feats = dataDims['num_molecule_features']

            # todo make sure that these reload with model state
            self.register_buffer('node_standardization_tensor',
                                 torch.tensor(dataDims['node_standardization_vector'], dtype=torch.float32))
            if self.n_mol_feats != 0:
                self.register_buffer('graph_standardization_tensor',
                                     torch.tensor(dataDims['graph_standardization_vector'], dtype=torch.float32))

        else:
            self.n_atom_feats = num_atom_features
            self.n_mol_feats = num_molecule_features

            if node_standardization_tensor is not None:
                self.register_buffer('node_standardization_tensor', node_standardization_tensor)
            if graph_standardization_tensor:
                if self.n_mol_feats != 0:
                    self.register_buffer('graph_standardization_tensor', graph_standardization_tensor)

    def standardize(self, data):
        data.x = (data.x - self.node_standardization_tensor[:, 0]) / self.node_standardization_tensor[:, 1]
        if self.n_mol_feats > 0:
            data.mol_x = (data.mol_x - self.graph_standardization_tensor[:, 0]) / self.graph_standardization_tensor[:,1]

        return data

    def forward(self, data, return_dists=False, return_latent=False, skip_standardization=False):
        """standardize on the fly from model-attached statistics"""
        if not skip_standardization:
            data = self.standardize(data)

        return self.model(data, return_dists=return_dists, return_latent=return_latent)
