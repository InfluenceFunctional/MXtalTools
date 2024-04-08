import torch.nn as nn

from mxtaltools.models.base_models import MoleculeGraphModel


class PolymorphClassifier(nn.Module):
    def __init__(self, seed, config, dataDims: dict):  # todo convert this to our standard molecule graph model
        super(PolymorphClassifier, self).__init__()
        self.model = MoleculeGraphModel(
            num_atom_feats=dataDims['num_atom_features'],
            num_mol_feats=0,
            output_dimension=dataDims['num_polymorphs'] + dataDims['num_topologies'],
            seed=seed,
            graph_aggregator='molwise',
            concat_pos_to_atom_features=False,
            concat_mol_to_atom_features=False,
            concat_crystal_to_atom_features=False,
            activation=config.activation,
            num_fc_layers=config.num_fc_layers,
            fc_depth=config.graph_embedding_depth,
            fc_norm_mode=config.fc_norm_mode,
            fc_dropout_probability=config.fc_dropout_probability,
            graph_node_norm=config.graph_node_norm,
            graph_node_dropout=config.graph_node_dropout,
            graph_message_dropout=0,
            num_attention_heads=config.num_attention_heads,
            graph_message_depth=config.graph_message_depth,
            graph_node_dims=config.graph_embedding_depth,
            num_graph_convolutions=config.num_graph_convolutions,
            graph_embedding_depth=config.graph_embedding_depth,
            nodewise_fc_layers=config.nodewise_fc_layers,
            num_radial=config.num_radial,
            radial_function=config.radial_function,
            max_num_neighbors=config.max_num_neighbors,
            convolution_cutoff=config.convolution_cutoff,
            atom_type_embedding_dims=config.atom_type_embedding_dims,
            outside_convolution_type='none'
        )

    def forward(self, data, return_dists=False, return_latent=False, return_embedding=False):
        return self.model(data, return_dists=return_dists, return_latent=return_latent, return_embedding=return_embedding)


    #
    #     self.max_num_neighbors = max_num_neighbors
    #     self.cutoff = cutoff
    #
    #     if radial_embedding == 'bessel':
    #         self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
    #     elif radial_embedding == 'gaussian':
    #         self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)
    #
    #     self.atom_embedding = EmbeddingBlock(node_embedding_depth,
    #                                          num_embedding_types,
    #                                          input_node_depth,
    #                                          embedding_hidden_dimension)
    #
    #     self.interaction_blocks = torch.nn.ModuleList([
    #         GC_Block(message_depth,
    #                  node_embedding_depth,
    #                  num_radial,
    #                  heads=attention_heads,
    #                  )
    #         for _ in range(num_blocks)
    #     ])
    #
    #     self.fc_blocks = torch.nn.ModuleList([
    #         MLP(
    #             layers=nodewise_fc_layers,
    #             filters=node_embedding_depth,
    #             input_dim=node_embedding_depth,
    #             output_dim=node_embedding_depth,
    #             activation=activation,
    #             norm=nodewise_norm,
    #             dropout=nodewise_dropout,
    #         )
    #         for _ in range(num_blocks)
    #     ])
    #
    #     self.global_pool = GlobalAggregation('molwise', graph_embedding_depth)
    #
    #     self.gnn_mlp = MLP(layers=num_fcs,
    #                        filters=node_embedding_depth,
    #                        norm=fc_norm,
    #                        dropout=fc_dropout,
    #                        input_dim=graph_embedding_depth,
    #                        output_dim=output_dimension,
    #                        conditioning_dim=0,
    #                        seed=seed
    #                        )
    #
    # def get_geom_embedding(self, edge_index, pos):
    #     """
    #     compute elements for radial & spherical embeddings
    #     """
    #     i, j = edge_index  # i->j source-to-target
    #     dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    #
    #     return dist, self.rbf(dist)
    #
    # def forward(self, data, return_latent=False, return_embedding=False):
    #
    #     x = self.atom_embedding(data.x)  # embed atomic numbers & compute initial atom-wise feature vector
    #     batch = data.batch
    #
    #     if data.periodic[0]:  # get radial embeddings periodically using minimum image convention
    #
    #     else:  # just get the radial graph the normal way
    #         edges_dict = construct_radial_graph(data.pos, data.batch, data.ptr, self.cutoff, self.max_num_neighbors)
    #         edge_index = edges_dict['edge_index']
    #         dist, rbf = self.get_geom_embedding(edge_index, data.pos)
    #
    #     for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
    #         x = convolution(x, rbf, edge_index, batch)  # graph convolution
    #         x = fc(x, batch=batch)  # feature-wise 1D convolution
    #
    #     x = self.global_pool(x, batch, cluster=data.mol_ind, output_dim=data.num_graphs)
    #
    #     if not return_embedding:
    #         return self.gnn_mlp(x, return_latent=return_latent)
    #     else:
    #         return self.gnn_mlp(x, return_latent=return_latent), x
    #
