'''import statements'''
import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # annoying w&b error
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=PerformanceWarning) # annoying pandas error

from utils import add_bool_arg, get_config
from crystal_modeller import Modeller


def update_args2config(args2config, arg, config=None):
    if config is not None:
        args2config.update({arg: config})
    else:
        args2config.update({arg: [arg]})


def add_args(parser):
    # high level
    args2config = {}
    parser.add_argument('--yaml_config', type=str, default='configs/dev.yaml', required=False)
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
    parser.add_argument("--conditioner_path", default=None, type=str)
    parser.add_argument("--regressor_path", default=None, type=str)
    add_bool_arg(parser, 'extra_test_evaluation', default=False)
    parser.add_argument("--extra_test_set_paths", default=None, type=list)
    add_bool_arg(parser,"save_checkpoints", default=False) # will revert to True on cluster machine

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
    update_args2config(args2config, 'conditioner_path')
    update_args2config(args2config, 'regressor_path')
    update_args2config(args2config, 'extra_test_evaluation')
    update_args2config(args2config, 'extra_test_set_paths')
    update_args2config(args2config, 'save_checkpoints')


    # wandb
    parser.add_argument('--wandb_experiment_tag', type=str, default='MCryGAN_dev')
    parser.add_argument('--wandb_username', type=str, default='mkilgour')
    parser.add_argument('--wandb_project_name', type=str, default='MCryGAN')
    parser.add_argument('--wandb_sample_reporting_frequency', type=int, default=1)
    add_bool_arg(parser, 'wandb_log_figures', default=True)

    update_args2config(args2config, 'wandb_experiment_tag', ['wandb', 'experiment_tag'])
    update_args2config(args2config, 'wandb_username', ['wandb', 'username'])
    update_args2config(args2config, 'wandb_project_name', ['wandb', 'project_name'])
    update_args2config(args2config, 'wandb_sample_reporting_frequency', ['wandb', 'sample_reporting_frequency'])
    update_args2config(args2config, 'wandb_log_figures', ['wandb', 'log_figures'])

    # dataset settings
    # todo update target - mostly not used
    parser.add_argument('--target', type=str,
                        default='molecule spherical defect')  # 'rings', 'groups', 'screw', 'inversion','rotoinversion','mirror','rotation','glide', 'crystal system', 'lattice centering', 'spherical', 'planar'(not in Jan17 dataset)
    parser.add_argument("--dataset_path", type=str, default='C:/Users\mikem\Desktop\CSP_runs\datasets/full_dataset')
    parser.add_argument('--dataset_length', type=int, default=int(1e3))  # maximum number of items in the dataset before filtration
    parser.add_argument('--feature_richness', type=str, default='minimal')  # atom & molecule feature richness

    # dataset composition
    parser.add_argument('--include_sgs', type=str, default=None)  # ['P21/c'] spacegroups to explicitly include in modelling - new!
    parser.add_argument('--include_pgs', type=str, default=None)  # ['222', '-1'] point groups to pull from dataset
    parser.add_argument('--generate_sgs', type=str, default=None)  # ['222', '-1'] point groups to generate
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

    update_args2config(args2config, 'target')
    update_args2config(args2config, 'dataset_path')
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

    #  training settings
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--min_batch_size', type=int, default=50)
    parser.add_argument('--max_batch_size', type=int, default=10000)
    parser.add_argument('--batch_growth_increment', type=int, default=0.05)
    add_bool_arg(parser, 'auto_batch_sizing', default=True)  # whether to densely connect dimenet outputs
    add_bool_arg(parser, 'grow_batch_size', default=True)  # whether to densely connect dimenet outputs
    parser.add_argument('--auto_batch_reduction', type=float, default=0.2)  # leeway factor to reduce batch size at end of auto-sizing run
    parser.add_argument('--gradient_norm_clip', type=float, default=1)
    add_bool_arg(parser, 'anomaly_detection', default=False)
    add_bool_arg(parser, 'accumulate_gradients', default=False)  # whether to densely connect dimenet outputs
    parser.add_argument('--accumulate_batch_size', type=int, default=100)

    update_args2config(args2config, 'max_epochs')
    update_args2config(args2config, 'history')
    update_args2config(args2config, 'min_batch_size')
    update_args2config(args2config, 'max_batch_size')
    update_args2config(args2config, 'batch_growth_increment')
    update_args2config(args2config, 'auto_batch_sizing')
    update_args2config(args2config, 'grow_batch_size')
    update_args2config(args2config, 'auto_batch_reduction')
    update_args2config(args2config, 'gradient_norm_clip')
    update_args2config(args2config, 'anomaly_detection')
    update_args2config(args2config, 'accumulate_gradients')
    update_args2config(args2config, 'accumulate_batch_size')


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

    parser.add_argument('--conditioner_optimizer_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--conditioner_optimizer_init_lr', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--conditioner_optimizer_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--conditioner_optimizer_min_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--conditioner_optimizer_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--conditioner_optimizer_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--conditioner_optimizer_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--conditioner_optimizer_convergence_eps', type=float, default=1e-5)
    add_bool_arg(parser, 'conditioner_optimizer_lr_schedule', default=False)
    parser.add_argument('--conditioner_optimizer_lr_growth_lambda', type=float, default=0.1)
    parser.add_argument('--conditioner_optimizer_lr_shrink_lambda', type=float, default=0.95)

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
    parser.add_argument('--conditioner_positional_noise', type=float, default=0)
    parser.add_argument('--regressor_positional_noise', type=float, default=0)
    parser.add_argument('--discriminator_positional_noise', type=float, default=0)

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

    update_args2config(args2config, 'conditioner_optimizer_optimizer', ['conditioner_optimizer', 'optimizer'])
    update_args2config(args2config, 'conditioner_optimizer_init_lr', ['conditioner_optimizer', 'init_lr'])
    update_args2config(args2config, 'conditioner_optimizer_max_lr', ['conditioner_optimizer', 'max_lr'])
    update_args2config(args2config, 'conditioner_optimizer_min_lr', ['conditioner_optimizer', 'min_lr'])
    update_args2config(args2config, 'conditioner_optimizer_beta1', ['conditioner_optimizer', 'beta1'])
    update_args2config(args2config, 'conditioner_optimizer_beta2', ['conditioner_optimizer', 'beta2'])
    update_args2config(args2config, 'conditioner_optimizer_weight_decay', ['conditioner_optimizer', 'weight_decay'])
    update_args2config(args2config, 'conditioner_optimizer_convergence_eps', ['conditioner_optimizer', 'convergence_eps'])
    update_args2config(args2config, 'conditioner_optimizer_lr_schedule', ['conditioner_optimizer', 'lr_schedule'])
    update_args2config(args2config, 'conditioner_optimizer_lr_growth_lambda', ['conditioner_optimizer', 'lr_growth_lambda'])
    update_args2config(args2config, 'conditioner_optimizer_lr_shrink_lambda', ['conditioner_optimizer', 'lr_shrink_lambda'])

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

    update_args2config(args2config, 'conditioner_positional_noise', ['conditioner', 'positional_noise'])
    update_args2config(args2config, 'regressor_positional_noise', ['regressor', 'positional_noise'])
    update_args2config(args2config, 'generator_positional_noise', ['generator', 'positional_noise'])
    update_args2config(args2config, 'discriminator_positional_noise', ['discriminator', 'positional_noise'])

    # generator model settings
    parser.add_argument('--regressor_positional_embedding', type=str, default='sph')  # sph or pos
    parser.add_argument('--regressor_graph_model', type=str, default='mike')  # 'dime', or 'schnet', or 'mike' or None
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

    parser.add_argument('--conditioner_init_decoder_size', type=int, default=3)  # int
    parser.add_argument('--conditioner_init_atom_embedding_dim', type=int, default=5)  # int
    parser.add_argument('--conditioner_positional_embedding', type=str, default='sph')  # sph or pos
    parser.add_argument('--conditioner_graph_model', type=str, default='mike')  # 'dime', or 'schnet', or 'mike' or None
    parser.add_argument('--conditioner_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--conditioner_output_dim', type=int, default=128)  # embedding dimension for atoms
    parser.add_argument('--conditioner_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--conditioner_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'GATv2' 'full message passing'
    parser.add_argument('--conditioner_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--conditioner_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--conditioner_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--conditioner_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--conditioner_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--conditioner_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--conditioner_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'conditioner_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    add_bool_arg(parser, 'conditioner_add_torsional_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    parser.add_argument('--conditioner_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'

    parser.add_argument('--conditioner_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--conditioner_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--conditioner_activation', type=str, default='gelu')
    parser.add_argument('--conditioner_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--conditioner_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'
    parser.add_argument('--conditioner_decoder_resolution', type=float, default=0.5)  #\
    parser.add_argument('--conditioner_decoder_classes', type=str, default='minimal') # 'minimal' or 'full'
    parser.add_argument('--conditioner_decoder_embedding_dim', type=int, default=64)  # number of neurons per graph convolution

    parser.add_argument('--generator_canonical_conformer_orientation', type=str, default='standardized')  # standardized or random
    parser.add_argument('--generator_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--generator_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--generator_activation', type=str, default='gelu')
    parser.add_argument('--generator_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--generator_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'

    parser.add_argument('--generator_prior', type=str, default='multivariate normal')  # type of prior distribution
    parser.add_argument('--generator_prior_dimension', type=int, default=12)  # type of prior distribution

    update_args2config(args2config, 'regressor_positional_embedding', ['regressor', 'positional_embedding'])
    update_args2config(args2config, 'regressor_graph_model', ['regressor', 'graph_model'])
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

    update_args2config(args2config, 'conditioner_init_decoder_size', ['conditioner', 'init_decoder_size'])
    update_args2config(args2config, 'conditioner_init_atom_embedding_dim', ['conditioner', 'init_atom_embedding_dim'])
    update_args2config(args2config, 'conditioner_output_dim', ['conditioner', 'output_dim'])
    update_args2config(args2config, 'conditioner_positional_embedding', ['conditioner', 'positional_embedding'])
    update_args2config(args2config, 'conditioner_positional_embedding', ['conditioner', 'positional_embedding'])
    update_args2config(args2config, 'conditioner_graph_model', ['conditioner', 'graph_model'])
    update_args2config(args2config, 'conditioner_atom_embedding_size', ['conditioner', 'atom_embedding_size'])
    update_args2config(args2config, 'conditioner_graph_filters', ['conditioner', 'graph_filters'])
    update_args2config(args2config, 'conditioner_graph_convolution', ['conditioner', 'graph_convolution'])
    update_args2config(args2config, 'conditioner_graph_convolutions_layers', ['conditioner', 'graph_convolutions_layers'])
    update_args2config(args2config, 'conditioner_graph_norm', ['conditioner', 'graph_norm'])
    update_args2config(args2config, 'conditioner_num_spherical', ['conditioner', 'num_spherical'])
    update_args2config(args2config, 'conditioner_num_radial', ['conditioner', 'num_radial'])
    update_args2config(args2config, 'conditioner_graph_convolution_cutoff', ['conditioner', 'graph_convolution_cutoff'])
    update_args2config(args2config, 'conditioner_max_num_neighbors', ['conditioner', 'max_num_neighbors'])
    update_args2config(args2config, 'conditioner_radial_function', ['conditioner', 'radial_function'])
    update_args2config(args2config, 'conditioner_add_spherical_basis', ['conditioner', 'add_spherical_basis'])
    update_args2config(args2config, 'conditioner_add_torsional_basis', ['conditioner', 'add_torsional_basis'])
    update_args2config(args2config, 'conditioner_pooling', ['conditioner', 'pooling'])
    update_args2config(args2config, 'conditioner_num_fc_layers', ['conditioner', 'num_fc_layers'])
    update_args2config(args2config, 'conditioner_fc_depth', ['conditioner', 'fc_depth'])
    update_args2config(args2config, 'conditioner_activation', ['conditioner', 'activation'])
    update_args2config(args2config, 'conditioner_fc_dropout_probability', ['conditioner', 'fc_dropout_probability'])
    update_args2config(args2config, 'conditioner_fc_norm_mode', ['conditioner', 'fc_norm_mode'])
    update_args2config(args2config, 'conditioner_decoder_resolution', ['conditioner', 'decoder_resolution'])
    update_args2config(args2config, 'conditioner_decoder_classes', ['conditioner', 'decoder_classes'])
    update_args2config(args2config, 'conditioner_decoder_embedding_dim', ['conditioner', 'decoder_embedding_dim'])

    update_args2config(args2config, 'generator_canonical_conformer_orientation', ['generator', 'canonical_conformer_orientation'])
    update_args2config(args2config, 'generator_num_fc_layers', ['generator', 'num_fc_layers'])
    update_args2config(args2config, 'generator_fc_depth', ['generator', 'fc_depth'])
    update_args2config(args2config, 'generator_activation', ['generator', 'activation'])
    update_args2config(args2config, 'generator_fc_dropout_probability', ['generator', 'fc_dropout_probability'])
    update_args2config(args2config, 'generator_fc_norm_mode', ['generator', 'fc_norm_mode'])
    update_args2config(args2config, 'generator_prior', ['generator', 'prior'])
    update_args2config(args2config, 'generator_prior_dimension', ['generator', 'prior_dimension'])


    # crystal cell graph Net
    parser.add_argument('--discriminator_graph_model', type=str, default='mike')  # 'dime', or 'schnet', or 'mike' or None
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

    update_args2config(args2config, 'discriminator_graph_model', ['discriminator', 'graph_model'])
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
    add_bool_arg(parser, 'freeze_generator_conditioner', default=False)  #
    add_bool_arg(parser, 'train_generator_combo', default=False)  # train on a packing + vdw combined score
    add_bool_arg(parser, 'train_generator_packing', default=False)  # boost packing density
    add_bool_arg(parser, 'train_generator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_generator_vdw', default=False)  #
    parser.add_argument('--vdw_loss_rescaling', type=str, default=None)  # None, 'log', 'mse'
    parser.add_argument('--packing_loss_rescaling', type=str, default=None)  # None, 'log', 'mse'
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
    parser.add_argument('--sample_move_size', type=float, default=0.05)  #

    update_args2config(args2config, 'freeze_generator_conditioner')
    update_args2config(args2config, 'train_generator_combo')
    update_args2config(args2config, 'train_generator_packing')
    update_args2config(args2config, 'train_generator_adversarially')
    update_args2config(args2config, 'train_generator_vdw')
    update_args2config(args2config, 'packing_loss_rescaling')
    update_args2config(args2config, 'vdw_loss_rescaling')
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
    update_args2config(args2config, 'sample_move_size')

    return parser, args2config


def process_config(config):
    if config.machine == 'local':
        config.workdir = 'C:/Users\mikem\Desktop/CSP_runs'
    elif config.machine == 'cluster':
        config.workdir = '/scratch/mk8347/csd_runs/'
        config.dataset_path = '/scratch/mk8347/csd_runs/datasets/full_dataset'
        config.save_checkpoints = True

    config.seeds.model = config.seeds.model % 10
    config.seeds.dataset = config.seeds.dataset % 10

    if config.test_mode:
        # config.max_batch_size = min((config.max_batch_size, 50))
        # config.auto_batch_sizing = False
        # config.anomaly_detection = True
        if config.machine == 'cluster':
            config.dataset_path = '/scratch/mk8347/csd_runs/datasets/test_dataset'
        else:
            config.dataset_path = 'C:/Users\mikem\Desktop\CSP_runs\datasets/test_dataset'

    return config


# ====================================
if __name__ == '__main__':
    '''
    parse arguments and generate config namespace
    '''
    # get command line input
    parser = argparse.ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    config = process_config(config)
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))

    '''
    run the code
    '''
    predictor = Modeller(config)
    if config.mode == 'figures':
        predictor.make_nice_figures()
    elif config.mode == 'sampling':
        predictor.model_sampling()
    else:
        predictor.train()
