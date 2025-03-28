import torch
from torch_geometric.typing import OptTensor

from mxtaltools.constants.atom_properties import VDW_RADII, ATOM_WEIGHTS, ELECTRONEGATIVITY, GROUP, PERIOD


class BaseGraphModel(torch.nn.Module):
    def __init__(self):
        super(BaseGraphModel, self).__init__()
        self.atom_feats = 0
        self.mol_feats = 0
        self.n_mol_feats = 0
        self.n_atom_feats = 0

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

    def featurize_input_graph(self,
                              data
                              ):

        data.x = self.atom_properties_tensor[data.z.long()]

        if self.n_mol_feats > 0:
            mol_x_list = []
            if 'num_atoms' in self.mol_feats:
                mol_x_list.append(data.num_atoms)
            if 'radius' in self.mol_feats:
                mol_x_list.append(data.radius)
            if 'mol_volume' in self.mol_feats:
                mol_x_list.append(data.mol_volume)
            data.mol_x = torch.stack(mol_x_list).T

        return data

    def standardize(self,
                    data
                    ):

        data.x = (data.x - self.node_standardization_tensor[:, 0]) / self.node_standardization_tensor[:, 1]

        if self.n_mol_feats > 0:
            data.mol_x = (
                    (data.mol_x - self.graph_standardization_tensor[:, 0]) / self.graph_standardization_tensor[:, 1])

        return data

    def forward(self,
                data_batch,
                return_dists: bool = False,
                return_latent: bool = False,
                force_edges_rebuild: bool = False,
                ):
        # featurize atom properties on the fly
        data_batch = self.featurize_input_graph(data_batch)

        # standardize on the fly from model-attached statistics
        data_batch = self.standardize(data_batch)

        # get radial graph
        if data_batch.edge_index is None or force_edges_rebuild:
            if 'crystal' in data_batch.__class__.__name__.lower():
                data_batch.construct_intra_radial_graph(float(self.model.convolution_cutoff))
            else:
                data_batch.construct_radial_graph(float(self.model.convolution_cutoff))

        return self.model(data_batch.x,
                          data_batch.pos,
                          data_batch.batch,
                          data_batch.ptr,
                          data_batch.mol_x,
                          data_batch.num_graphs,
                          edge_index=data_batch.edge_index,
                          return_dists=return_dists,
                          return_latent=return_latent)

    def compile_self(self, dynamic=True, fullgraph=False):
        self.model = torch.compile(self.model, dynamic=dynamic, fullgraph=fullgraph)
