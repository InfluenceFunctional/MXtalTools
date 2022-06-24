import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MikesGraphNet import MikesGraphNet
import torch_geometric.nn as gnn
from models.torch_models import general_MLP


class crystal_discriminator(nn.Module):
    def __init__(self, config, dataDims):
        super(crystal_discriminator, self).__init__()
        # initialize constants and layers
        self.activation = config.discriminator.activation
        self.num_fc_layers = config.discriminator.num_fc_layers
        self.fc_depth = config.discriminator.fc_depth
        self.output_classes = 1 if config.gan_loss == 'wasserstein' else 2
        self.graph_filters = config.discriminator.graph_filters
        self.n_mol_feats = dataDims['n mol features']
        self.n_atom_feats = dataDims['atom features']
        self.n_atom_feats -= self.n_mol_feats
        self.pool_type = config.discriminator.pooling
        self.fc_norm_mode = config.discriminator.fc_norm_mode
        self.embedding_dim = config.discriminator.atom_embedding_size
        self.crystal_mode = True

        torch.manual_seed(config.seeds.model)

        self.graph_net = MikesGraphNet(
            crystal_mode=True,
            graph_convolution_filters=config.discriminator.graph_filters,
            graph_convolution=config.discriminator.graph_convolution,
            out_channels=config.discriminator.fc_depth,
            hidden_channels=config.discriminator.atom_embedding_size,
            num_blocks=config.discriminator.graph_convolution_layers,
            num_radial=config.discriminator.num_radial,
            num_spherical=config.discriminator.num_spherical,
            max_num_neighbors=config.discriminator.max_num_neighbors,
            cutoff=config.discriminator.graph_convolution_cutoff,
            activation='gelu',
            embedding_hidden_dimension=config.discriminator.atom_embedding_size,
            num_atom_features=self.n_atom_feats,
            norm=config.discriminator.graph_norm,
            dropout=config.discriminator.fc_dropout_probability,
            spherical_embedding=config.discriminator.add_spherical_basis,
            radial_embedding=config.discriminator.radial_function,
            atom_embedding_dims=dataDims['atom embedding dict sizes'],
            attention_heads=config.discriminator.num_attention_heads
        )

        # initialize global pooling operation
        if config.discriminator.pooling == 'mean':
            self.global_pool = gnn.global_mean_pool
        elif config.discriminator.pooling == 'sum':
            self.global_pool = gnn.global_add_pool
        elif config.discriminator.pooling == 'attention':
            self.global_pool = gnn.GlobalAttention(nn.Sequential(nn.Linear(config.discriminator.fc_depth, config.discriminator.fc_depth),
                                                                 nn.GELU(), nn.Linear(config.discriminator.fc_depth, 1)))

        # molecule features FC layer
        if self.n_mol_feats != 0:
            self.mol_fc = nn.Linear(self.n_mol_feats, self.n_mol_feats)

        self.gnn_mlp = general_MLP(layers=config.discriminator.num_fc_layers,
                                   filters=config.discriminator.fc_depth,
                                   norm=config.discriminator.fc_norm_mode,
                                   dropout=config.discriminator.fc_dropout_probability,
                                   input_dim=self.fc_depth,
                                   output_dim=self.fc_depth,
                                   conditioning_dim=self.n_mol_feats,
                                   seed=config.seeds.model
                                   )

        self.output_fc = nn.Linear(self.fc_depth, self.output_classes, bias=False)

    def forward(self, data):
        x = data.x
        pos = data.pos

        x = self.graph_net(torch.cat((x[:, :self.n_atom_feats], x[:, -1:]), dim=1), pos, data.batch)  # extra dim for crystal indexing

        keep_cell_inds = torch.where(data.x[:, -1] == 1)[0]
        data.batch = data.batch[keep_cell_inds]  # keep only atoms inside the reference cell
        data.x = data.x[keep_cell_inds]  # also apply to molecular_data

        x = self.global_pool(x, data.batch)  # aggregate atoms to molecule

        mol_inputs = data.x[:, -self.n_mol_feats:]
        mol_feats = self.mol_fc(gnn.global_max_pool(mol_inputs, data.batch).float())  # not actually pooling here, as the values are all the same for each molecule

        x = self.gnn_mlp(x, conditions=mol_feats)

        return self.output_fc(x)

