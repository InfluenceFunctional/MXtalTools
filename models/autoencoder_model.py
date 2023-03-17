import torch
from torch import nn as nn
from torch_geometric.nn import global_max_pool
from models.model_components import general_MLP
from models.torch_models import molecule_graph_model, PointCloudDecoder
from utils import get_strides


class molecule_autoencoder(nn.Module):
    def __init__(self, config, dataDims):
        super(molecule_autoencoder, self).__init__()

        self.device = config.device
        torch.manual_seed(config.seeds.model)

        '''
        conditioning model
        '''
        self.crystal_features_to_ignore = config.dataDims['num crystal generation features']
        conv_embedding_dim = config.conditioner.decoder_embedding_dim
        self.conditioner = molecule_graph_model(
            dataDims=dataDims,
            atom_embedding_dims=config.conditioner.init_atom_embedding_dim, #len(config.conditioner_classes) + 1,
            seed=config.seeds.model,
            num_atom_feats=dataDims['num atom features'] + 3 - self.crystal_features_to_ignore,  # we will add directly the normed coordinates to the node features
            num_mol_feats=dataDims['num mol features'] - self.crystal_features_to_ignore,
            output_dimension=config.conditioner.output_dim,  # starting size for decoder model
            activation=config.conditioner.activation,
            num_fc_layers=config.conditioner.num_fc_layers,
            fc_depth=config.conditioner.fc_depth,
            fc_dropout_probability=config.conditioner.fc_dropout_probability,
            fc_norm_mode=config.conditioner.fc_norm_mode,
            graph_filters=config.conditioner.graph_filters,
            graph_convolutional_layers=config.conditioner.graph_convolution_layers,
            concat_mol_to_atom_features=config.conditioner.concat_mol_features,
            pooling=config.conditioner.pooling,
            graph_norm=config.conditioner.graph_norm,
            num_spherical=config.conditioner.num_spherical,
            num_radial=config.conditioner.num_radial,
            graph_convolution=config.conditioner.graph_convolution,
            num_attention_heads=config.conditioner.num_attention_heads,
            add_spherical_basis=config.conditioner.add_spherical_basis,
            add_torsional_basis=config.conditioner.add_torsional_basis,
            graph_embedding_size=config.conditioner.atom_embedding_size,
            radial_function=config.conditioner.radial_function,
            max_num_neighbors=config.conditioner.max_num_neighbors,
            convolution_cutoff=config.conditioner.graph_convolution_cutoff,
            positional_embedding=config.conditioner.positional_embedding,
            max_molecule_size=config.max_molecule_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
            skip_mlp=False
        )

        '''
        generator model
        common atom types
        '''
        # stride 4 adds 3N - 1
        # stride 3 adds 2N
        # stride 2 adds N+1
        # stride 1 adds 2
        self.rescale_output_dims = nn.Linear(config.conditioner.output_dim,
                                             config.conditioner.decoder_embedding_dim * config.conditioner.init_decoder_size**3)

        n_target_bins = int((config.max_molecule_radius) * 2 / config.conditioner.decoder_resolution)
        strides, final_image_size = get_strides(n_target_bins, init_size = config.conditioner.init_decoder_size)  # automatically find the right number of strides within 4-5 steps (minimizes overall stack depth)

        self.decoder = PointCloudDecoder(input_filters=conv_embedding_dim,
                                         n_classes=len(config.conditioner_classes) + 1,
                                         strides=strides,
                                         init_image_size = config.conditioner.init_decoder_size)

        self.mlp = general_MLP(input_dim=config.conditioner.decoder_embedding_dim * config.conditioner.init_decoder_size**3,
                               layers=2,
                               output_dim=1,
                               filters=256,
                               norm='layer',
                               dropout=0.1,
                               activation='gelu')

    def forward(self, data):
        #normed_coords = data.pos / self.conditioner.max_molecule_size  # norm coords by maximum molecule radius
        #data.x = torch.cat((data.x[:, :-self.crystal_features_to_ignore], normed_coords), dim=-1)  # concatenate position to input features

        # #normed_pos = data.pos / self.conditioner.max_molecule_size
        # extrema = global_max_pool(torch.linalg.norm(data.pos, dim=-1), batch=data.batch)
        #
        # extrema_list = torch.cat([torch.ones(data.ptr[ii+1]-data.ptr[ii],device=extrema.device,dtype=extrema.dtype) * extrema[ii]
        #                           for ii in range(data.num_graphs)])[:,None]

        normed_pos = data.pos / self.conditioner.max_molecule_size
        data.x = torch.cat((data.x[:, :-self.crystal_features_to_ignore], normed_pos), dim=-1)  # concatenate position to input features


        conditions_encoding = self.rescale_output_dims(self.conditioner(data))
        return self.decoder(conditions_encoding), self.mlp(conditions_encoding)  # return decoder and regression target