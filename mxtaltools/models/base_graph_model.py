import torch
from torch_geometric.typing import OptTensor

from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD


class BaseGraphModel(torch.nn.Module):
    def __init__(self):
        super(BaseGraphModel, self).__init__()
        self.atom_feats = None
        self.mol_feats = None
        self.n_mol_feats = None
        self.n_atom_feats = None

    def get_data_stats(self,
                       atom_features: list,
                       molecule_features: list,
                       node_standardization_tensor: OptTensor = None,
                       graph_standardization_tensor: OptTensor = None
                       ):

        if node_standardization_tensor is None:
            node_standardization_tensor = torch.ones((len(atom_features), 2), dtype=torch.float32)
            node_standardization_tensor[:, 0] = 0
        if graph_standardization_tensor is None:
            graph_standardization_tensor = torch.ones((len(molecule_features), 2), dtype=torch.float32)
            graph_standardization_tensor[:, 0] = 0

        self.n_atom_feats = len(atom_features)
        self.n_mol_feats = len(molecule_features)
        self.atom_feats = atom_features
        self.mol_feats = molecule_features

        # generate atom property embeddings
        atom_embeddings_list = [torch.arange(len(VDW_RADII))]  # start with raw atomic number
        if 'vdw_radii' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(VDW_RADII.values())))
        if 'atom_weight' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(ATOM_WEIGHTS.values())))
        if 'electronegativity' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(ELECTRONEGATIVITY.values())))
        if 'group' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(GROUP.values())))
        if 'period' in self.atom_feats:
            atom_embeddings_list.append(torch.tensor(list(PERIOD.values())))

        assert len(atom_embeddings_list) == self.n_atom_feats

        self.register_buffer('atom_properties_tensor', torch.stack(atom_embeddings_list).T)

        if not torch.is_tensor(node_standardization_tensor):
            node_standardization_tensor = torch.tensor(node_standardization_tensor, dtype=torch.float32)
        if not torch.is_tensor(graph_standardization_tensor):
            graph_standardization_tensor = torch.tensor(graph_standardization_tensor, dtype=torch.float32)

        # store atom standardizations
        self.register_buffer('node_standardization_tensor', node_standardization_tensor)
        if self.n_mol_feats != 0:
            self.register_buffer('graph_standardization_tensor', graph_standardization_tensor)

    def featurize_input_graph(self, data):
        if data.x.ndim > 1:
            data.x = data.x[:, 0]
        data.x = self.atom_properties_tensor[data.x.long()]
        if self.n_mol_feats > 0:
            mol_x_list = []
            if 'num_atoms' in self.mol_feats:
                mol_x_list.append(data.num_atoms)
            if 'radius' in self.mol_feats:
                mol_x_list.append(data.radius)
            data.mol_x = torch.stack(mol_x_list).T

        return data

    def standardize(self, data):
        data.x = (data.x - self.node_standardization_tensor[:, 0]) / self.node_standardization_tensor[:, 1]
        if self.n_mol_feats > 0:
            data.mol_x = (
                    (data.mol_x - self.graph_standardization_tensor[:, 0]) / self.graph_standardization_tensor[:, 1])

        return data

    def forward(self, data, return_dists=False, return_latent=False):
        # featurize atom properties on the fly
        data = self.featurize_input_graph(data)

        # standardize on the fly from model-attached statistics
        data = self.standardize(data)

        return self.model(data, return_dists=return_dists, return_latent=return_latent)
