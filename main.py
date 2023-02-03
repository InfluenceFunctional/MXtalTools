'''import statements'''
import wandb
import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # annoying w&b error
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=PerformanceWarning) # annoying pandas error

from utils import load_yaml, add_bool_arg, get_config
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
    parser.add_argument("--d_model_path", default=None, type=str)
    parser.add_argument("--g_model_path", default=None, type=str)
    add_bool_arg(parser, 'extra_test_evaluation', default=False)
    parser.add_argument("--extra_test_set_paths", default=None, type=list)

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
    update_args2config(args2config, 'd_model_path')
    update_args2config(args2config, 'g_model_path')
    update_args2config(args2config, 'extra_test_evaluation')
    update_args2config(args2config, 'extra_test_set_paths')

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
    parser.add_argument('--auto_batch_reduction', type=float, default=0.2)  # leeway factor to reduce batch size at end of auto-sizing run
    parser.add_argument('--gradient_norm_clip', type=float, default=1)
    add_bool_arg(parser, 'anomaly_detection', default=False)
    add_bool_arg(parser, 'accumulate_gradients', default=False)  # whether to densely connect dimenet outputs
    parser.add_argument('--accumulate_batch_size', type=int, default=100)
    parser.add_argument('--lr_growth_lambda', type=float, default=0.1)
    parser.add_argument('--lr_shrink_lambda', type=float, default=0.95)
    update_args2config(args2config, 'max_epochs')
    update_args2config(args2config, 'history')
    update_args2config(args2config, 'min_batch_size')
    update_args2config(args2config, 'max_batch_size')
    update_args2config(args2config, 'batch_growth_increment')
    update_args2config(args2config, 'auto_batch_sizing')
    update_args2config(args2config, 'auto_batch_reduction')
    update_args2config(args2config, 'gradient_norm_clip')
    update_args2config(args2config, 'anomaly_detection')
    update_args2config(args2config, 'accumulate_gradients')
    update_args2config(args2config, 'accumulate_batch_size')
    update_args2config(args2config, 'lr_growth_lambda')
    update_args2config(args2config, 'lr_shrink_lambda')

    # optimizer settings
    parser.add_argument('--discriminator_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--discriminator_learning_rate', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--discriminator_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--discriminator_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--discriminator_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--discriminator_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--discriminator_convergence_eps', type=float, default=1e-5)
    parser.add_argument('--discriminator_training_period', type=int, default=5)  # period between discriminator training
    add_bool_arg(parser, 'discriminator_lr_schedule', default=False)
    parser.add_argument('--discriminator_positional_noise', type=float, default=0)

    parser.add_argument('--generator_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--generator_learning_rate', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--generator_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--generator_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--generator_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--generator_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--generator_convergence_eps', type=float, default=1e-5)
    add_bool_arg(parser, 'generator_lr_schedule', default=False)
    parser.add_argument('--generator_positional_noise', type=float, default=0)

    update_args2config(args2config, 'discriminator_optimizer', ['discriminator', 'optimizer'])
    update_args2config(args2config, 'discriminator_learning_rate', ['discriminator', 'learning_rate'])
    update_args2config(args2config, 'discriminator_max_lr', ['discriminator', 'max_lr'])
    update_args2config(args2config, 'discriminator_beta1', ['discriminator', 'beta1'])
    update_args2config(args2config, 'discriminator_beta2', ['discriminator', 'beta2'])
    update_args2config(args2config, 'discriminator_weight_decay', ['discriminator', 'weight_decay'])
    update_args2config(args2config, 'discriminator_convergence_eps', ['discriminator', 'convergence_eps'])
    update_args2config(args2config, 'discriminator_training_period', ['discriminator', 'training_period'])
    update_args2config(args2config, 'discriminator_lr_schedule', ['discriminator', 'lr_schedule'])
    update_args2config(args2config, 'discriminator_positional_noise', ['discriminator', 'positional_noise'])

    update_args2config(args2config, 'generator_optimizer', ['generator', 'optimizer'])
    update_args2config(args2config, 'generator_learning_rate', ['generator', 'learning_rate'])
    update_args2config(args2config, 'generator_max_lr', ['generator', 'max_lr'])
    update_args2config(args2config, 'generator_beta1', ['generator', 'beta1'])
    update_args2config(args2config, 'generator_beta2', ['generator', 'beta2'])
    update_args2config(args2config, 'generator_weight_decay', ['generator', 'weight_decay'])
    update_args2config(args2config, 'generator_convergence_eps', ['generator', 'convergence_eps'])
    update_args2config(args2config, 'generator_lr_schedule', ['generator', 'lr_schedule'])
    update_args2config(args2config, 'generator_positional_noise', ['generator', 'positional_noise'])

    # generator model settings
    parser.add_argument('--generator_canonical_conformer_orientation', type=str, default='standardized')  # standardized or random
    parser.add_argument('--generator_graph_model', type=str, default='mike')  # 'dime', or 'schnet', or 'mike' or None
    parser.add_argument('--generator_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--generator_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--generator_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'GATv2' 'full message passing'
    parser.add_argument('--generator_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--generator_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--generator_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--generator_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--generator_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--generator_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--generator_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'generator_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    add_bool_arg(parser, 'generator_add_torsional_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet
    parser.add_argument('--generator_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'

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
    parser.add_argument('--generator_conditioning_mode', type=str, default='graph model')  # how to derive molecular conditioning - graph model or just selected features

    parser.add_argument('--generator_prior', type=str, default='multivariate normal')  # type of prior distribution
    parser.add_argument('--generator_prior_dimension', type=int, default=12)  # type of prior distribution
    add_bool_arg(parser, 'generator_conditional_modelling', default=True)  # whether to use molecular features as conditions for normalizing flow model

    update_args2config(args2config, 'generator_canonical_conformer_orientation', ['generator', 'canonical_conformer_orientation'])
    update_args2config(args2config, 'generator_graph_model', ['generator', 'graph_model'])
    update_args2config(args2config, 'generator_atom_embedding_size', ['generator', 'atom_embedding_size'])
    update_args2config(args2config, 'generator_graph_filters', ['generator', 'graph_filters'])
    update_args2config(args2config, 'generator_graph_convolution', ['generator', 'graph_convolution'])
    update_args2config(args2config, 'generator_graph_convolutions_layers', ['generator', 'graph_convolutions_layers'])
    update_args2config(args2config, 'generator_graph_norm', ['generator', 'graph_norm'])
    update_args2config(args2config, 'generator_num_spherical', ['generator', 'num_spherical'])
    update_args2config(args2config, 'generator_num_radial', ['generator', 'num_radial'])
    update_args2config(args2config, 'generator_graph_convolution_cutoff', ['generator', 'graph_convolution_cutoff'])
    update_args2config(args2config, 'generator_max_num_neighbors', ['generator', 'max_num_neighbors'])
    update_args2config(args2config, 'generator_radial_function', ['generator', 'radial_function'])
    update_args2config(args2config, 'generator_add_spherical_basis', ['generator', 'add_spherical_basis'])
    update_args2config(args2config, 'generator_add_torsional_basis', ['generator', 'add_torsional_basis'])
    update_args2config(args2config, 'generator_pooling', ['generator', 'pooling'])
    update_args2config(args2config, 'generator_conditioner_num_fc_layers', ['generator', 'conditioner_num_fc_layers'])
    update_args2config(args2config, 'generator_conditioner_fc_depth', ['generator', 'conditioner_fc_depth'])
    update_args2config(args2config, 'generator_conditioner_activation', ['generator', 'conditioner_activation'])
    update_args2config(args2config, 'generator_conditioner_fc_dropout_probability', ['generator', 'conditioner_fc_dropout_probability'])
    update_args2config(args2config, 'generator_conditioner_fc_norm_mode', ['generator', 'conditioner_fc_norm_mode'])
    update_args2config(args2config, 'generator_num_fc_layers', ['generator', 'num_fc_layers'])
    update_args2config(args2config, 'generator_fc_depth', ['generator', 'fc_depth'])
    update_args2config(args2config, 'generator_activation', ['generator', 'activation'])
    update_args2config(args2config, 'generator_fc_dropout_probability', ['generator', 'fc_dropout_probability'])
    update_args2config(args2config, 'generator_fc_norm_mode', ['generator', 'fc_norm_mode'])
    update_args2config(args2config, 'generator_prior', ['generator', 'prior'])
    update_args2config(args2config, 'generator_prior_dimension', ['generator', 'prior_dimension'])
    update_args2config(args2config, 'generator_conditional_modelling', ['generator', 'conditional_modelling'])
    update_args2config(args2config, 'generator_conditioning_mode', ['generator', 'conditioning_mode'])

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
    parser.add_argument('--gan_loss', type=str, default='standard')  # stnandard only
    add_bool_arg(parser, 'new_generation', default=True)  # new way of defining the asymmetric unit
    add_bool_arg(parser, 'train_generator_combo', default=False)  # train on a packing + vdw combined score
    add_bool_arg(parser, 'train_generator_packing', default=False)  # boost packing density
    add_bool_arg(parser, 'train_generator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_generator_vdw', default=False)  # train generator on adversarially
    parser.add_argument('--vdw_loss_rescaling', type=str, default=None)  # None, 'log', 'mse'
    parser.add_argument('--packing_loss_rescaling', type=str, default=None)  # None, 'log', 'mse'
    add_bool_arg(parser, 'train_generator_h_bond', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_discriminator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_discriminator_on_randn', default=False)  # train generator on cells generated from appropriately fit multivariate gaussians
    add_bool_arg(parser, 'train_discriminator_on_noise', default=False)  # train generator on distorted CSD data
    parser.add_argument('--generator_noise_level', type=float, default=0)  # amount of noise to add to cell params for distorted cell training
    parser.add_argument('--generator_similarity_penalty', type=float, default=0)  # coefficient weighting penalty for self-similarity in generator batches
    parser.add_argument('--extra_test_period', type=int, default=10)  # how often to report stats on the extra test data
    add_bool_arg(parser, 'sample_after_training', default=False)  # run sampler after model converges
    parser.add_argument('--sample_ind', type=int, default=0)  # which sample from test dataset to sample
    parser.add_argument('--sample_steps', type=int, default=1000)  #
    parser.add_argument('--sample_move_size', type=float, default=0.05)  #

    update_args2config(args2config, 'gan_loss')
    update_args2config(args2config, 'new_generation')
    update_args2config(args2config, 'train_generator_combo')
    update_args2config(args2config, 'train_generator_packing')
    update_args2config(args2config, 'train_generator_adversarially')
    update_args2config(args2config, 'train_generator_vdw')
    update_args2config(args2config, 'vdw_loss_rescaling')
    update_args2config(args2config, 'packing_loss_rescaling')
    update_args2config(args2config, 'train_generator_h_bond')
    update_args2config(args2config, 'train_discriminator_adversarially')
    update_args2config(args2config, 'train_discriminator_on_randn')
    update_args2config(args2config, 'train_discriminator_on_noise')
    update_args2config(args2config, 'generator_noise_level')
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
