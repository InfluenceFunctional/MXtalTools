from argparse import Namespace
from pathlib import Path

import yaml


def update_args2config(args2config, arg, config=None):
    if config is not None:
        args2config.update({arg: config})
    else:
        args2config.update({arg: [arg]})


def add_args(parser):
    # high level
    args2config = {}
    parser.add_argument('--user', type=str, default='mkilgour', required=False)
    parser.add_argument('--yaml_config', type=str, default=None, required=False)
    parser.add_argument('--run_num', type=int, default=0)
    add_bool_arg(parser, 'explicit_run_enumeration', default=False)  # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
    add_bool_arg(parser, 'test_mode', default=True)
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--machine', type=str, default='local')  # 'local' (local windows venv) or 'cluster' (linux env)
    parser.add_argument("--device", default="cuda", type=str)  # 'cuda' or 'cpu'
    parser.add_argument("--mode", default="gan", type=str)  # 'gan' or 'regression'
    add_bool_arg(parser, 'skip_saving_and_loading', default=True)
    parser.add_argument("--discriminator_path", default=None, type=str)
    parser.add_argument("--generator_path", default=None, type=str)
    parser.add_argument("--regressor_path", default=None, type=str)
    add_bool_arg(parser, 'extra_test_evaluation', default=False)
    parser.add_argument("--extra_test_set_paths", default=None, type=list)

    add_bool_arg(parser, "save_checkpoints", default=False)  # will revert to True on cluster machine

    update_args2config(args2config, 'user')
    update_args2config(args2config, 'yaml_config')
    update_args2config(args2config, 'run_num')
    update_args2config(args2config, 'explicit_run_enumeration')
    update_args2config(args2config, 'test_mode')
    update_args2config(args2config, 'model_seed', ['seeds', 'model'])
    update_args2config(args2config, 'dataset_seed', ['seeds', 'dataset'])
    update_args2config(args2config, 'machine')
    update_args2config(args2config, 'device')
    update_args2config(args2config, 'mode')
    update_args2config(args2config, 'skip_saving_and_loading')
    update_args2config(args2config, 'discriminator_path')
    update_args2config(args2config, 'generator_path')
    update_args2config(args2config, 'regressor_path')
    update_args2config(args2config, 'extra_test_evaluation')
    update_args2config(args2config, 'extra_test_set_paths')
    update_args2config(args2config, 'save_checkpoints')

    # wandb / logging
    parser.add_argument('--logger_experiment_tag', type=str, default='')
    parser.add_argument('--wandb_username', type=str, default='')
    parser.add_argument('--wandb_project_name', type=str, default='')
    parser.add_argument('--logger_sample_reporting_frequency', type=int, default=1)
    parser.add_argument('--logger_mini_csp_frequency', type=int, default=100)
    add_bool_arg(parser, 'logger_log_figures', default=True)

    update_args2config(args2config, 'logger_experiment_tag', ['logger', 'experiment_tag'])
    update_args2config(args2config, 'wandb_username', ['wandb', 'username'])
    update_args2config(args2config, 'wandb_project_name', ['wandb', 'project_name'])
    update_args2config(args2config, 'logger_sample_reporting_frequency', ['logger', 'sample_reporting_frequency'])
    update_args2config(args2config, 'logger_mini_csp_frequency', ['logger', 'mini_csp_frequency'])
    update_args2config(args2config, 'logger_log_figures', ['logger', 'log_figures'])

    # dataset settings
    parser.add_argument('--target', type=str,
                        default='packing')  # 'packing' only # todo deprecate
    parser.add_argument('--dataset_length', type=int, default=int(1e3))  # maximum number of items in the dataset before filtration
    parser.add_argument('--feature_richness', type=str, default="minimal")  # atom & molecule feature richness

    # dataset composition
    parser.add_argument('--rotation_basis', type=str, default="spherical")  # spherical or cartesian
    parser.add_argument('--include_sgs', type=list, default=None)  # ['P21/c'] spacegroups to explicitly include in modelling - new!
    parser.add_argument('--include_pgs', type=str, default=None)  # ['222', '-1'] point groups to pull from dataset
    parser.add_argument('--generate_sgs', type=list, default=None)  # ['222', '-1'] space groups to generate
    parser.add_argument('--supercell_size', type=int, default=1)  # point groups to generate
    parser.add_argument('--max_crystal_temperature', type=float, default=int(1e3))
    parser.add_argument('--min_crystal_temperature', type=int, default=0)
    parser.add_argument('--max_num_atoms', type=int, default=int(1e3))
    parser.add_argument('--min_num_atoms', type=int, default=0)
    parser.add_argument('--max_molecule_radius', type=float, default=10)
    parser.add_argument('--min_packing_coefficient', type=float, default=0.55)
    add_bool_arg(parser, 'include_organic', default=True)
    add_bool_arg(parser, 'include_organometallic', default=True)
    parser.add_argument('--max_atomic_number', type=int, default=87)
    add_bool_arg(parser, 'exclude_disordered_crystals', default=True)
    add_bool_arg(parser, 'exclude_polymorphs', default=True)
    add_bool_arg(parser, 'exclude_nonstandard_settings', default=True)
    add_bool_arg(parser, 'exclude_missing_r_factor', default=True)
    parser.add_argument('--exclude_crystal_systems', type=list, default=None)  # ['hexagonal']
    add_bool_arg(parser, 'exclude_blind_test_targets', default=True)
    parser.add_argument('--target_identifiers', type=list, default=None)  # list of identifier strings e.g., ["ABEBUF", "NICOAM01"]
    parser.add_argument('--single_molecule_dataset_identifier', type=str, default=None)  # identifier string e.g., "NICOAM03"

    update_args2config(args2config, 'rotation_basis')
    update_args2config(args2config, 'target')
    update_args2config(args2config, 'dataset_length')
    update_args2config(args2config, 'feature_richness')
    update_args2config(args2config, 'include_sgs')
    update_args2config(args2config, 'include_pgs')
    update_args2config(args2config, 'generate_sgs')
    update_args2config(args2config, 'supercell_size')
    update_args2config(args2config, 'max_crystal_temperature')
    update_args2config(args2config, 'min_crystal_temperature')
    update_args2config(args2config, 'max_num_atoms')
    update_args2config(args2config, 'min_num_atoms')
    update_args2config(args2config, 'max_molecule_radius')
    update_args2config(args2config, 'min_packing_coefficient')
    update_args2config(args2config, 'include_organic')
    update_args2config(args2config, 'include_organometallic')
    update_args2config(args2config, 'max_atomic_number')
    update_args2config(args2config, 'exclude_disordered_crystals')
    update_args2config(args2config, 'exclude_polymorphs')
    update_args2config(args2config, 'exclude_nonstandard_settings')
    update_args2config(args2config, 'exclude_missing_r_factor')
    update_args2config(args2config, 'exclude_crystal_systems')
    update_args2config(args2config, 'exclude_blind_test_targets')
    update_args2config(args2config, 'target_identifiers')
    update_args2config(args2config, 'single_molecule_dataset_identifier')

    #  training settings
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--min_batch_size', type=int, default=50)
    parser.add_argument('--max_batch_size', type=int, default=10000)
    parser.add_argument('--batch_growth_increment', type=int, default=0.05)
    add_bool_arg(parser, 'grow_batch_size', default=True)  # whether to densely connect dimenet outputs
    parser.add_argument('--gradient_norm_clip', type=float, default=1)
    add_bool_arg(parser, 'anomaly_detection', default=False)

    update_args2config(args2config, 'max_epochs')
    update_args2config(args2config, 'history')
    update_args2config(args2config, 'min_batch_size')
    update_args2config(args2config, 'max_batch_size')
    update_args2config(args2config, 'batch_growth_increment')
    update_args2config(args2config, 'grow_batch_size')
    update_args2config(args2config, 'gradient_norm_clip')
    update_args2config(args2config, 'anomaly_detection')

    # optimizer settings
    parser.add_argument('--discriminator_optimizer_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--discriminator_optimizer_init_lr', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--discriminator_optimizer_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--discriminator_optimizer_min_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--discriminator_optimizer_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--discriminator_optimizer_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--discriminator_optimizer_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--discriminator_optimizer_convergence_eps', type=float, default=1e-5)
    parser.add_argument('--discriminator_optimizer_training_period', type=int, default=5)  # period between discriminator training
    add_bool_arg(parser, 'discriminator_optimizer_lr_schedule', default=False)
    parser.add_argument('--discriminator_optimizer_lr_growth_lambda', type=float, default=0.1)
    parser.add_argument('--discriminator_optimizer_lr_shrink_lambda', type=float, default=0.95)

    parser.add_argument('--generator_optimizer_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--generator_optimizer_init_lr', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--generator_optimizer_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--generator_optimizer_min_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--generator_optimizer_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--generator_optimizer_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--generator_optimizer_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--generator_optimizer_convergence_eps', type=float, default=1e-5)
    add_bool_arg(parser, 'generator_optimizer_lr_schedule', default=False)
    parser.add_argument('--generator_optimizer_lr_growth_lambda', type=float, default=0.1)
    parser.add_argument('--generator_optimizer_lr_shrink_lambda', type=float, default=0.95)

    parser.add_argument('--regressor_optimizer_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--regressor_optimizer_init_lr', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--regressor_optimizer_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--regressor_optimizer_min_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--regressor_optimizer_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--regressor_optimizer_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--regressor_optimizer_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--regressor_optimizer_convergence_eps', type=float, default=1e-5)
    add_bool_arg(parser, 'regressor_optimizer_lr_schedule', default=False)
    parser.add_argument('--regressor_optimizer_lr_growth_lambda', type=float, default=0.1)
    parser.add_argument('--regressor_optimizer_lr_shrink_lambda', type=float, default=0.95)

    parser.add_argument('--generator_positional_noise', type=float, default=0)
    parser.add_argument('--regressor_positional_noise', type=float, default=0)
    parser.add_argument('--discriminator_positional_noise', type=float, default=0)

    update_args2config(args2config, 'regressor_positional_noise')
    update_args2config(args2config, 'generator_positional_noise')
    update_args2config(args2config, 'discriminator_positional_noise')

    update_args2config(args2config, 'discriminator_optimizer_optimizer', ['discriminator_optimizer', 'optimizer'])
    update_args2config(args2config, 'discriminator_optimizer_init_lr', ['discriminator_optimizer', 'init_lr'])
    update_args2config(args2config, 'discriminator_optimizer_max_lr', ['discriminator_optimizer', 'max_lr'])
    update_args2config(args2config, 'discriminator_optimizer_min_lr', ['discriminator_optimizer', 'min_lr'])
    update_args2config(args2config, 'discriminator_optimizer_beta1', ['discriminator_optimizer', 'beta1'])
    update_args2config(args2config, 'discriminator_optimizer_beta2', ['discriminator_optimizer', 'beta2'])
    update_args2config(args2config, 'discriminator_optimizer_weight_decay', ['discriminator_optimizer', 'weight_decay'])
    update_args2config(args2config, 'discriminator_optimizer_convergence_eps', ['discriminator_optimizer', 'convergence_eps'])
    update_args2config(args2config, 'discriminator_optimizer_training_period', ['discriminator_optimizer', 'training_period'])
    update_args2config(args2config, 'discriminator_optimizer_lr_schedule', ['discriminator_optimizer', 'lr_schedule'])
    update_args2config(args2config, 'discriminator_optimizer_lr_growth_lambda', ['discriminator_optimizer', 'lr_growth_lambda'])
    update_args2config(args2config, 'discriminator_optimizer_lr_shrink_lambda', ['discriminator_optimizer', 'lr_shrink_lambda'])

    update_args2config(args2config, 'generator_optimizer_optimizer', ['generator_optimizer', 'optimizer'])
    update_args2config(args2config, 'generator_optimizer_init_lr', ['generator_optimizer', 'init_lr'])
    update_args2config(args2config, 'generator_optimizer_max_lr', ['generator_optimizer', 'max_lr'])
    update_args2config(args2config, 'generator_optimizer_min_lr', ['generator_optimizer', 'min_lr'])
    update_args2config(args2config, 'generator_optimizer_beta1', ['generator_optimizer', 'beta1'])
    update_args2config(args2config, 'generator_optimizer_beta2', ['generator_optimizer', 'beta2'])
    update_args2config(args2config, 'generator_optimizer_weight_decay', ['generator_optimizer', 'weight_decay'])
    update_args2config(args2config, 'generator_optimizer_convergence_eps', ['generator_optimizer', 'convergence_eps'])
    update_args2config(args2config, 'generator_optimizer_lr_schedule', ['generator_optimizer', 'lr_schedule'])
    update_args2config(args2config, 'generator_optimizer_lr_growth_lambda', ['generator_optimizer', 'lr_growth_lambda'])
    update_args2config(args2config, 'generator_optimizer_lr_shrink_lambda', ['generator_optimizer', 'lr_shrink_lambda'])

    update_args2config(args2config, 'regressor_optimizer_optimizer', ['regressor_optimizer', 'optimizer'])
    update_args2config(args2config, 'regressor_optimizer_init_lr', ['regressor_optimizer', 'init_lr'])
    update_args2config(args2config, 'regressor_optimizer_max_lr', ['regressor_optimizer', 'max_lr'])
    update_args2config(args2config, 'regressor_optimizer_min_lr', ['regressor_optimizer', 'min_lr'])
    update_args2config(args2config, 'regressor_optimizer_beta1', ['regressor_optimizer', 'beta1'])
    update_args2config(args2config, 'regressor_optimizer_beta2', ['regressor_optimizer', 'beta2'])
    update_args2config(args2config, 'regressor_optimizer_weight_decay', ['regressor_optimizer', 'weight_decay'])
    update_args2config(args2config, 'regressor_optimizer_convergence_eps', ['regressor_optimizer', 'convergence_eps'])
    update_args2config(args2config, 'regressor_optimizer_lr_schedule', ['regressor_optimizer', 'lr_schedule'])
    update_args2config(args2config, 'regressor_optimizer_lr_growth_lambda', ['regressor_optimizer', 'lr_growth_lambda'])
    update_args2config(args2config, 'regressor_optimizer_lr_shrink_lambda', ['regressor_optimizer', 'lr_shrink_lambda'])

    # generator model settings
    parser.add_argument('--regressor_positional_embedding', type=str, default='sph')  # sph or pos
    parser.add_argument('--regressor_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--regressor_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--regressor_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'GATv2' 'full message passing'
    parser.add_argument('--regressor_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--regressor_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--regressor_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--regressor_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--regressor_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--regressor_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--regressor_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'regressor_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    add_bool_arg(parser, 'regressor_add_torsional_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    parser.add_argument('--regressor_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'

    parser.add_argument('--regressor_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--regressor_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--regressor_activation', type=str, default='gelu')
    parser.add_argument('--regressor_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--regressor_fc_norm_mode', type=str, default='layer')

    parser.add_argument('--generator_conditioner_concat_mol_features', type=bool, default=True)
    parser.add_argument('--generator_conditioner_init_atom_embedding_dim', type=int, default=5)  # int
    parser.add_argument('--generator_conditioner_positional_embedding', type=str, default='sph')  # sph or pos
    parser.add_argument('--generator_conditioner_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--generator_conditioner_output_dim', type=int, default=128)  # embedding dimension for atoms
    parser.add_argument('--generator_conditioner_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--generator_conditioner_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'GATv2' 'full message passing'
    parser.add_argument('--generator_conditioner_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--generator_conditioner_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--generator_conditioner_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--generator_conditioner_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--generator_conditioner_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--generator_conditioner_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--generator_conditioner_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'generator_conditioner_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    add_bool_arg(parser, 'generator_conditioner_add_torsional_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    parser.add_argument('--generator_conditioner_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'

    parser.add_argument('--generator_conditioner_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--generator_conditioner_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--generator_conditioner_activation', type=str, default='gelu')
    parser.add_argument('--generator_conditioner_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--generator_conditioner_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'

    parser.add_argument('--generator_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--generator_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--generator_activation', type=str, default='gelu')
    parser.add_argument('--generator_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--generator_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'

    parser.add_argument('--generator_prior', type=str, default='multivariate normal')  # type of prior distribution
    parser.add_argument('--generator_prior_dimension', type=int, default=12)  # type of prior distribution

    update_args2config(args2config, 'regressor_positional_embedding', ['regressor', 'positional_embedding'])
    update_args2config(args2config, 'regressor_atom_embedding_size', ['regressor', 'atom_embedding_size'])
    update_args2config(args2config, 'regressor_graph_filters', ['regressor', 'graph_filters'])
    update_args2config(args2config, 'regressor_graph_convolution', ['regressor', 'graph_convolution'])
    update_args2config(args2config, 'regressor_graph_convolutions_layers', ['regressor', 'graph_convolutions_layers'])
    update_args2config(args2config, 'regressor_graph_norm', ['regressor', 'graph_norm'])
    update_args2config(args2config, 'regressor_num_spherical', ['regressor', 'num_spherical'])
    update_args2config(args2config, 'regressor_num_radial', ['regressor', 'num_radial'])
    update_args2config(args2config, 'regressor_graph_convolution_cutoff', ['regressor', 'graph_convolution_cutoff'])
    update_args2config(args2config, 'regressor_max_num_neighbors', ['regressor', 'max_num_neighbors'])
    update_args2config(args2config, 'regressor_radial_function', ['regressor', 'radial_function'])
    update_args2config(args2config, 'regressor_add_spherical_basis', ['regressor', 'add_spherical_basis'])
    update_args2config(args2config, 'regressor_add_torsional_basis', ['regressor', 'add_torsional_basis'])
    update_args2config(args2config, 'regressor_pooling', ['regressor', 'pooling'])
    update_args2config(args2config, 'regressor_num_fc_layers', ['regressor', 'num_fc_layers'])
    update_args2config(args2config, 'regressor_fc_depth', ['regressor', 'fc_depth'])
    update_args2config(args2config, 'regressor_activation', ['regressor', 'activation'])
    update_args2config(args2config, 'regressor_fc_dropout_probability', ['regressor', 'fc_dropout_probability'])
    update_args2config(args2config, 'regressor_fc_norm_mode', ['regressor', 'fc_norm_mode'])

    update_args2config(args2config, 'generator_conditioner_skinny_atomwise_features', ['generator', 'conditioner', 'skinny_atomwise_features'])
    update_args2config(args2config, 'generator_conditioner_concat_mol_features', ['generator', 'conditioner', 'concat_mol_features'])
    update_args2config(args2config, 'generator_conditioner_init_decoder_size', ['generator', 'conditioner', 'init_decoder_size'])
    update_args2config(args2config, 'generator_conditioner_init_atom_embedding_dim', ['generator', 'conditioner', 'init_atom_embedding_dim'])
    update_args2config(args2config, 'generator_conditioner_output_dim', ['generator', 'conditioner', 'output_dim'])
    update_args2config(args2config, 'generator_conditioner_positional_embedding', ['generator', 'conditioner', 'positional_embedding'])
    update_args2config(args2config, 'generator_conditioner_positional_embedding', ['generator', 'conditioner', 'positional_embedding'])
    update_args2config(args2config, 'generator_conditioner_atom_embedding_size', ['generator', 'conditioner', 'atom_embedding_size'])
    update_args2config(args2config, 'generator_conditioner_graph_filters', ['generator', 'conditioner', 'graph_filters'])
    update_args2config(args2config, 'generator_conditioner_graph_convolution', ['generator', 'conditioner', 'graph_convolution'])
    update_args2config(args2config, 'generator_conditioner_graph_convolutions_layers', ['generator', 'conditioner', 'graph_convolutions_layers'])
    update_args2config(args2config, 'generator_conditioner_graph_norm', ['generator', 'conditioner', 'graph_norm'])
    update_args2config(args2config, 'generator_conditioner_num_spherical', ['generator', 'conditioner', 'num_spherical'])
    update_args2config(args2config, 'generator_conditioner_num_radial', ['generator', 'conditioner', 'num_radial'])
    update_args2config(args2config, 'generator_conditioner_graph_convolution_cutoff', ['generator', 'conditioner', 'graph_convolution_cutoff'])
    update_args2config(args2config, 'generator_conditioner_max_num_neighbors', ['generator', 'conditioner', 'max_num_neighbors'])
    update_args2config(args2config, 'generator_conditioner_radial_function', ['generator', 'conditioner', 'radial_function'])
    update_args2config(args2config, 'generator_conditioner_add_spherical_basis', ['generator', 'conditioner', 'add_spherical_basis'])
    update_args2config(args2config, 'generator_conditioner_add_torsional_basis', ['generator', 'conditioner', 'add_torsional_basis'])
    update_args2config(args2config, 'generator_conditioner_pooling', ['generator', 'conditioner', 'pooling'])
    update_args2config(args2config, 'generator_conditioner_num_fc_layers', ['generator', 'conditioner', 'num_fc_layers'])
    update_args2config(args2config, 'generator_conditioner_fc_depth', ['generator', 'conditioner', 'fc_depth'])
    update_args2config(args2config, 'generator_conditioner_activation', ['generator', 'conditioner', 'activation'])
    update_args2config(args2config, 'generator_conditioner_fc_dropout_probability', ['generator', 'conditioner', 'fc_dropout_probability'])
    update_args2config(args2config, 'generator_conditioner_fc_norm_mode', ['generator', 'conditioner', 'fc_norm_mode'])

    update_args2config(args2config, 'generator_num_fc_layers', ['generator', 'num_fc_layers'])
    update_args2config(args2config, 'generator_fc_depth', ['generator', 'fc_depth'])
    update_args2config(args2config, 'generator_activation', ['generator', 'activation'])
    update_args2config(args2config, 'generator_fc_dropout_probability', ['generator', 'fc_dropout_probability'])
    update_args2config(args2config, 'generator_fc_norm_mode', ['generator', 'fc_norm_mode'])
    update_args2config(args2config, 'generator_prior', ['generator', 'prior'])
    update_args2config(args2config, 'generator_prior_dimension', ['generator', 'prior_dimension'])

    # crystal cell graph Net
    parser.add_argument('--discriminator_crystal_convolution_type', type=int, default=1)  # 1 - counts inter and intramolecular the same, 2 - separates intermolecular
    parser.add_argument('--discriminator_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--discriminator_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--discriminator_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'GATv2' 'full message passing'
    parser.add_argument('--discriminator_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--discriminator_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--discriminator_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--discriminator_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--discriminator_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--discriminator_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--discriminator_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'discriminator_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    add_bool_arg(parser, 'discriminator_add_torsional_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet

    parser.add_argument('--discriminator_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--discriminator_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--discriminator_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'
    parser.add_argument('--discriminator_activation', type=str, default='gelu')
    parser.add_argument('--discriminator_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--discriminator_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'

    update_args2config(args2config, 'discriminator_crystal_convolution_type', ['discriminator', 'crystal_convolution_type'])
    update_args2config(args2config, 'discriminator_atom_embedding_size', ['discriminator', 'atom_embedding_size'])
    update_args2config(args2config, 'discriminator_graph_filters', ['discriminator', 'graph_filters'])
    update_args2config(args2config, 'discriminator_graph_convolution', ['discriminator', 'graph_convolution'])
    update_args2config(args2config, 'discriminator_graph_convolutions_layers', ['discriminator', 'graph_convolutions_layers'])
    update_args2config(args2config, 'discriminator_graph_norm', ['discriminator', 'graph_norm'])
    update_args2config(args2config, 'discriminator_num_spherical', ['discriminator', 'num_spherical'])
    update_args2config(args2config, 'discriminator_num_radial', ['discriminator', 'num_radial'])
    update_args2config(args2config, 'discriminator_graph_convolution_cutoff', ['discriminator', 'graph_convolution_cutoff'])
    update_args2config(args2config, 'discriminator_max_num_neighbors', ['discriminator', 'max_num_neighbors'])
    update_args2config(args2config, 'discriminator_radial_function', ['discriminator', 'radial_function'])
    update_args2config(args2config, 'discriminator_add_spherical_basis', ['discriminator', 'add_spherical_basis'])
    update_args2config(args2config, 'discriminator_add_torsional_basis', ['discriminator', 'add_torsional_basis'])
    update_args2config(args2config, 'discriminator_num_fc_layers', ['discriminator', 'num_fc_layers'])
    update_args2config(args2config, 'discriminator_fc_depth', ['discriminator', 'fc_depth'])
    update_args2config(args2config, 'discriminator_pooling', ['discriminator', 'pooling'])
    update_args2config(args2config, 'discriminator_activation', ['discriminator', 'activation'])
    update_args2config(args2config, 'discriminator_fc_dropout_probability', ['discriminator', 'fc_dropout_probability'])
    update_args2config(args2config, 'discriminator_fc_norm_mode', ['discriminator', 'fc_norm_mode'])

    # cell generator
    parser.add_argument('--canonical_conformer_orientation', type=str, default='standardized')  # standardized or random
    add_bool_arg(parser, 'train_generator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_generator_vdw', default=False)  #
    parser.add_argument('--packing_target_noise', type=float, default=0)  # noise added to density target in standardized basis
    parser.add_argument('--vdw_loss_func', type=str, default=None)  # None, 'log', 'mse'
    parser.add_argument('--density_loss_func', type=str, default='l1')  # 'l1' 'mse'
    parser.add_argument('--generator_adversarial_loss_func', type=str, default='l1')  # 'softmax' 'hot softmax' 'minimax' 'score'
    add_bool_arg(parser, 'train_generator_h_bond', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_discriminator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_discriminator_on_randn', default=False)  # train generator on cells generated from appropriately fit multivariate gaussians
    add_bool_arg(parser, 'train_discriminator_on_distorted', default=False)  # train generator on distorted CSD data
    parser.add_argument('--sample_distortion_magnitude', type=float, default=0)  # amount of noise to add to cell params for distorted cell training
    parser.add_argument('--generator_similarity_penalty', type=float, default=0)  # coefficient weighting penalty for self-similarity in generator batches
    parser.add_argument('--extra_test_period', type=int, default=10)  # how often to report stats on the extra test data
    add_bool_arg(parser, 'sample_after_training', default=False)  # run sampler after model converges
    parser.add_argument('--sample_ind', type=int, default=0)  # which sample from test dataset to sample
    parser.add_argument('--sample_steps', type=int, default=1000)  #

    update_args2config(args2config, 'canonical_conformer_orientation')
    update_args2config(args2config, 'packing_target_noise')
    update_args2config(args2config, 'train_generator_adversarially')
    update_args2config(args2config, 'train_generator_vdw')
    update_args2config(args2config, 'vdw_loss_func')
    update_args2config(args2config, 'density_loss_func')
    update_args2config(args2config, 'generator_adversarial_loss_func')
    update_args2config(args2config, 'train_generator_h_bond')
    update_args2config(args2config, 'train_discriminator_adversarially')
    update_args2config(args2config, 'train_discriminator_on_randn')
    update_args2config(args2config, 'train_discriminator_on_distorted')
    update_args2config(args2config, 'sample_distortion_magnitude')
    update_args2config(args2config, 'generator_similarity_penalty')
    update_args2config(args2config, 'extra_test_period')
    update_args2config(args2config, 'sample_after_training')
    update_args2config(args2config, 'sample_ind')
    update_args2config(args2config, 'sample_steps')

    return parser, args2config


def get_config(args, override_args, args2config):
    """
    Combines YAML configuration file, command line arguments and default arguments into
    a single configuration dictionary.

    - Values in YAML file override default values
    - Command line arguments override values in YAML file

    Returns
    -------
        Namespace
    """

    def _update_config(arg, val, config, override=False):
        config_aux = config
        for k in args2config[arg]:
            if k not in config_aux:
                if k is args2config[arg][-1]:
                    config_aux.update({k: val})
                else:
                    config_aux.update({k: {}})
                    config_aux = config_aux[k]
            else:
                if k is args2config[arg][-1] and override:
                    config_aux[k] = val
                else:
                    config_aux = config_aux[k]

    '''get user-specific configs'''
    user_path = f'configs/users/{args.user}.yaml'
    yaml_path = Path(user_path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        user_config = yaml.safe_load(f)

    if args.yaml_config is None:  # default to user's dev config
        args.yaml_config = user_config['paths']['dev_yaml_path']

    # Read YAML config
    if args.yaml_config:
        yaml_path = Path(args.yaml_config)
        assert yaml_path.exists()
        assert yaml_path.suffix in {".yaml", ".yml"}
        with yaml_path.open("r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    # Add args to config: add if not provided; override if in command line
    override_args = [
        arg.strip("--").split("=")[0] for arg in override_args if "--" in arg
    ]
    override_args_extra = []
    for k1 in override_args:
        if k1 in args2config:
            v1 = args2config[k1]
            for k2, v2 in args2config.items():
                if v2 == v1 and k2 != k1:
                    override_args_extra.append(k2)
    override_args = override_args + override_args_extra
    for k, v in vars(args).items():
        if k in override_args:
            _update_config(k, v, config, override=True)
        else:
            _update_config(k, v, config, override=False)

    config = dict2namespace(config)
    # update user paths
    if config.test_mode:
        dataset_name = 'test_dataset'
    else:
        dataset_name = 'full_dataset'

    if config.machine == 'local':
        config.workdir = user_config['paths']['local_workdir_path']
        config.dataset_path = user_config['paths']['local_dataset_dir_path'] + dataset_name
        config.checkpoint_dir_path = user_config['paths']['local_checkpoint_dir_path']

    elif config.machine == 'cluster':
        config.workdir = user_config['paths']['cluster_workdir_path']
        config.dataset_path = user_config['paths']['cluster_dataset_dir_path'] + dataset_name
        config.checkpoint_dir_path = user_config['paths']['cluster_checkpoint_dir_path']
        config.save_checkpoints = True  # always save checkpoints on cluster

    config.wandb.username = user_config['wandb']['username']
    config.wandb.project_name = user_config['wandb']['project_name']

    return config


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def add_arg_list(parser, arg_list):
    for entry in arg_list:
        if entry['type'] == 'bool':
            add_bool_arg(parser, entry['name'], entry['default'])
        else:
            parser.add_argument('--' + entry['name'], type=entry['type'], default=entry['default'])

    return parser


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def load_yaml(path, append_config_dir=True):
    if append_config_dir:
        path = "configs/" + path
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict
