import torch

from models.base_models import molecule_graph_model


class MoleculeConditioner():
    def __init__(self, device):
        """
        define conditioning model taking in a molecule in a given pose
        and encoding it to a vector to pass to the generator/policy model
        """

        self.device = device
        self.symmetries_dict = sym_info
        self.lattice_means = torch.tensor(dataDims['lattice means'], dtype=torch.float32, device=device)
        self.lattice_stds = torch.tensor(dataDims['lattice stds'], dtype=torch.float32, device=device)
        self.norm_lattice_lengths = False

        # initialize asymmetric unit dict
        self.asym_unit_dict = asym_unit_dict.copy()
        for key in self.asym_unit_dict:
            self.asym_unit_dict[key] = torch.Tensor(self.asym_unit_dict[key]).to(self.device)

        '''conditioning model'''
        self.num_crystal_features = config.dataDims['num crystal generation features']
        torch.manual_seed(config.seeds.model)

        self.atom_input_feats = dataDims['num atom features'] + 3 - self.num_crystal_features
        self.num_mol_feats = dataDims['num mol features'] - self.num_crystal_features

        self.conditioner = molecule_graph_model(
            dataDims=dataDims,
            atom_embedding_dims=config.generator.conditioner.init_atom_embedding_dim,
            seed=config.seeds.model,
            num_atom_feats=self.atom_input_feats,  # we will add directly the normed coordinates to the node features
            num_mol_feats=self.num_mol_feats,
            output_dimension=config.generator.conditioner.output_dim,  # starting size for decoder model
            activation=config.generator.conditioner.activation,
            num_fc_layers=config.generator.conditioner.num_fc_layers,
            fc_depth=config.generator.conditioner.fc_depth,
            fc_dropout_probability=config.generator.conditioner.fc_dropout_probability,
            fc_norm_mode=config.generator.conditioner.fc_norm_mode,
            graph_filters=config.generator.conditioner.graph_filters,
            graph_convolutional_layers=config.generator.conditioner.graph_convolution_layers,
            concat_mol_to_atom_features=config.generator.conditioner.concat_mol_features,
            pooling=config.generator.conditioner.pooling,
            graph_norm=config.generator.conditioner.graph_norm,
            num_spherical=config.generator.conditioner.num_spherical,
            num_radial=config.generator.conditioner.num_radial,
            graph_convolution=config.generator.conditioner.graph_convolution,
            num_attention_heads=config.generator.conditioner.num_attention_heads,
            add_spherical_basis=config.generator.conditioner.add_spherical_basis,
            add_torsional_basis=config.generator.conditioner.add_torsional_basis,
            graph_embedding_size=config.generator.conditioner.atom_embedding_size,
            radial_function=config.generator.conditioner.radial_function,
            max_num_neighbors=config.generator.conditioner.max_num_neighbors,
            convolution_cutoff=config.generator.conditioner.graph_convolution_cutoff,
            positional_embedding=config.generator.conditioner.positional_embedding,
            max_molecule_size=config.max_molecule_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
        )

    def forward(self, conditions):
        """
        add atom positions to node features, norm by the maximum allowable molecule size
        remove crystal features
        return conditioning vector
        """

        normed_coords = conditions.pos / self.conditioner.max_molecule_size  # norm coords by maximum molecule radius
        crystal_information = conditions.x[:, -self.num_crystal_features:]
        conditions.x = torch.cat((conditions.x[:, :-self.num_crystal_features], normed_coords), dim=-1)  # concatenate to input features, leaving out crystal info from conditioner

        return self.conditioner(conditions)
