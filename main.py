'''import statements'''
import wandb
import argparse
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # annoying w&b error
from utils import load_yaml, add_bool_arg, get_config
from CSG_predictor import Predictor

'''
Predict crystal features given atom and molecule-level information
'''


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
    add_bool_arg(parser, 'skip_run_init', default=False)
    parser.add_argument("--mode", default="cell gan", type=str)  # 'single molecule classification' 'joint modelling' 'single molecule regresion' 'cell classification', 'cell gan'
    add_bool_arg(parser, 'skip_saving_and_loading', default=True)

    update_args2config(args2config, 'yaml_config')
    update_args2config(args2config, 'run_num')
    update_args2config(args2config, 'explicit_run_enumeration')
    update_args2config(args2config, 'test_mode')
    update_args2config(args2config, 'model_seed', ['seeds', 'model'])
    update_args2config(args2config, 'dataset_seed', ['seeds', 'dataset'])
    update_args2config(args2config, 'machine')
    update_args2config(args2config, 'device')
    update_args2config(args2config, 'skip_run_init')
    update_args2config(args2config, 'mode')
    update_args2config(args2config, 'skip_saving_and_loading')

    # wandb login
    parser.add_argument('--wandb_experiment_tag', type=str, default='MCryGAN_dev')
    parser.add_argument('--wandb_username', type=str, default='mkilgour')
    parser.add_argument('--wandb_project_name', type=str, default='MCryGAN')
    # wandb reporting
    parser.add_argument('--wandb_sample_reporting_frequency', type=int, default=1)
    add_bool_arg(parser, 'wandb_log_figures', default=True)
    # wandb sweeps
    parser.add_argument('--sweep_config_file', type=str, default='sweep_1.yaml')
    add_bool_arg(parser, 'sweep', default=True)  # whether to do a single run or use w&b to run a Bayesian sweep
    parser.add_argument('--sweep_id', type=str, default=None)  # put something here to continue a prior sweep, else None fo fresh sweeps
    parser.add_argument('--sweep_num_runs', type=int, default=100)

    update_args2config(args2config, 'wandb_experiment_tag', ['wandb', 'experiment_tag'])
    update_args2config(args2config, 'wandb_username', ['wandb', 'username'])
    update_args2config(args2config, 'wandb_project_name', ['wandb', 'project_name'])
    update_args2config(args2config, 'wandb_sample_reporting_frequency', ['wandb', 'sample_reporting_frequency'])
    update_args2config(args2config, 'wandb_log_figures', ['wandb', 'log_figures'])
    update_args2config(args2config, 'sweep_config_file', ['wandb', 'sweep_config_file'])
    update_args2config(args2config, 'sweep', ['wandb', 'sweep'])
    update_args2config(args2config, 'sweep_id', ['wandb', 'sweep_id'])
    update_args2config(args2config, 'sweep_num_runs', ['wandb', 'sweep_num_runs'])

    # dataset settings
    parser.add_argument('--target', type=str,
                        default='molecule spherical defect')  # 'rings', 'groups', 'screw', 'inversion','rotoinversion','mirror','rotation','glide', 'crystal system', 'lattice centering', 'spherical', 'planar'(not in Jan17 dataset)
    parser.add_argument("--dataset_path", type=str, default='C:/Users\mikem\Desktop\CSP_runs\datasets/full_dataset')
    parser.add_argument('--dataset_length', type=int, default=int(1e3))  # maximum number of items in the dataset before filtration

    # dataset composition
    parser.add_argument('--include_sgs', type=str, default=['P21/c'])  # spacegroups to explicitly include in modelling - new!
    parser.add_argument('--include_pgs', type=str, default=['222', '-1'])  # point groups to pull from dataset
    parser.add_argument('--generate_sgs', type=str, default=['222', '-1'])  # point groups to generate
    parser.add_argument('--supercell_size', type=int, default=1)  # point groups to generate
    parser.add_argument('--max_crystal_temperature', type=float, default=int(1e3))
    parser.add_argument('--min_crystal_temperature', type=int, default=0)
    parser.add_argument('--max_num_atoms', type=float, default=int(1e3))
    parser.add_argument('--min_num_atoms', type=int, default=0)
    parser.add_argument('--min_packing_coefficient', type=float, default=0.55)
    add_bool_arg(parser, 'include_organic', default=True)
    add_bool_arg(parser, 'include_organometallic', default=True)
    parser.add_argument('--max_atomic_number', type=int, default=87)
    add_bool_arg(parser, 'exclude_disordered_crystals', default=True)
    add_bool_arg(parser, 'exclude_polymorphs', default=True)
    add_bool_arg(parser, 'exclude_nonstandard_settings', default=True)
    add_bool_arg(parser, 'exclude_missing_r_factor', default=True)

    update_args2config(args2config, 'target')
    update_args2config(args2config, 'dataset_path')
    update_args2config(args2config, 'dataset_length')
    update_args2config(args2config, 'include_sgs')
    update_args2config(args2config, 'include_pgs')
    update_args2config(args2config, 'generate_sgs')
    update_args2config(args2config, 'supercell_size')
    update_args2config(args2config, 'max_crystal_temperature')
    update_args2config(args2config, 'min_crystal_temperature')
    update_args2config(args2config, 'max_num_atoms')
    update_args2config(args2config, 'min_num_atoms')
    update_args2config(args2config, 'min_packing_coefficient')
    update_args2config(args2config, 'include_organic')
    update_args2config(args2config, 'include_organometallic')
    update_args2config(args2config, 'max_atomic_number')
    update_args2config(args2config, 'exclude_disordered_crystals')
    update_args2config(args2config, 'exclude_polymorphs')
    update_args2config(args2config, 'exclude_nonstandard_settings')
    update_args2config(args2config, 'exclude_missing_r_factor')

    #  training settings
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--min_batch_size', type=int, default=50)
    parser.add_argument('--max_batch_size', type=int, default=10000)
    add_bool_arg(parser, 'auto_batch_sizing', default=True)  # whether to densely connect dimenet outputs
    parser.add_argument('--auto_batch_reduction', type=float, default=0.2)  # leeway factor to reduce batch size at end of auto-sizing run
    parser.add_argument('--gradient_norm_clip', type=float, default=1)
    add_bool_arg(parser, 'anomaly_detection', default=False)

    update_args2config(args2config, 'max_epochs')
    update_args2config(args2config, 'history')
    update_args2config(args2config, 'min_batch_size')
    update_args2config(args2config, 'max_batch_size')
    update_args2config(args2config, 'auto_batch_sizing')
    update_args2config(args2config, 'auto_batch_reduction')
    update_args2config(args2config, 'gradient_norm_clip')
    update_args2config(args2config, 'anomaly_detection')

    # optimizer settings
    parser.add_argument('--discriminator_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--discriminator_learning_rate', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--discriminator_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--discriminator_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--discriminator_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--discriminator_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--discriminator_convergence_eps', type=float, default=1e-5)
    add_bool_arg(parser, 'discriminator_lr_schedule', default=False)

    parser.add_argument('--generator_optimizer', type=str, default='adamw')  # adam, adamw, sgd
    parser.add_argument('--generator_learning_rate', type=float, default=1e-5)  # base learning rate
    parser.add_argument('--generator_max_lr', type=float, default=1e-3)  # for warmup schedules
    parser.add_argument('--generator_beta1', type=float, default=0.9)  # adam and adamw opt
    parser.add_argument('--generator_beta2', type=float, default=0.999)  # adam and adamw opt
    parser.add_argument('--generator_weight_decay', type=float, default=0.01)  # for opt
    parser.add_argument('--generator_convergence_eps', type=float, default=1e-5)
    add_bool_arg(parser, 'generator_lr_schedule', default=False)

    update_args2config(args2config, 'discriminator_optimizer', ['discriminator', 'optimizer'])
    update_args2config(args2config, 'discriminator_learning_rate', ['discriminator', 'learning_rate'])
    update_args2config(args2config, 'discriminator_max_lr', ['discriminator', 'max_lr'])
    update_args2config(args2config, 'discriminator_beta1', ['discriminator', 'beta1'])
    update_args2config(args2config, 'discriminator_beta2', ['discriminator', 'beta2'])
    update_args2config(args2config, 'discriminator_weight_decay', ['discriminator', 'weight_decay'])
    update_args2config(args2config, 'discriminator_convergence_eps', ['discriminator', 'convergence_eps'])
    update_args2config(args2config, 'discriminator_lr_schedule', ['discriminator', 'lr_schedule'])

    update_args2config(args2config, 'generator_optimizer', ['generator', 'optimizer'])
    update_args2config(args2config, 'generator_learning_rate', ['generator', 'learning_rate'])
    update_args2config(args2config, 'generator_max_lr', ['generator', 'max_lr'])
    update_args2config(args2config, 'generator_beta1', ['generator', 'beta1'])
    update_args2config(args2config, 'generator_beta2', ['generator', 'beta2'])
    update_args2config(args2config, 'generator_weight_decay', ['generator', 'weight_decay'])
    update_args2config(args2config, 'generator_convergence_eps', ['generator', 'convergence_eps'])
    update_args2config(args2config, 'generator_lr_schedule', ['generator', 'lr_schedule'])

    # generator model settings
    parser.add_argument('--generator_model_type', type=str, default='mlp')  # random, 'csd cell', 'model'
    parser.add_argument('--generator_graph_model', type=str, default='mike')  # 'dime', or 'schnet', or 'mike' or None
    parser.add_argument('--generator_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--generator_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--generator_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'self attention' 'full message passing'
    parser.add_argument('--generator_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--generator_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--generator_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--generator_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--generator_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--generator_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--generator_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'generator_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet

    parser.add_argument('--generator_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--generator_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--generator_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'
    parser.add_argument('--generator_activation', type=str, default='gelu')
    parser.add_argument('--generator_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--generator_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'

    # flow model
    parser.add_argument('--generator_num_flow_layers', type=int, default=3)  # number of flow layers
    parser.add_argument('--generator_flow_depth', type=int, default=16)  # number of filters per flow layer
    parser.add_argument('--generator_flow_basis_fns', type=int, default=8)  # number of basis functions for spline NF model
    parser.add_argument('--generator_prior', type=str, default='multivariate normal')  # type of prior distribution
    parser.add_argument('--generator_flow_type', type=str, default='nsf_cl')  # type of flow model 'nsf-cl' is legit
    parser.add_argument('--generator_num_samples', type=int, default=10000)  # number of samples to generate for analysis
    add_bool_arg(parser, 'generator_conditional_modelling', default=True)  # whether to use molecular features as conditions for normalizing flow model
    parser.add_argument('--generator_conditioning_mode', type=str, default='graph model')  # how to derive molecular conditioning - graph model or just selected features

    update_args2config(args2config, 'generator_model_type', ['generator', 'model_type'])
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
    update_args2config(args2config, 'generator_num_fc_layers', ['generator', 'num_fc_layers'])
    update_args2config(args2config, 'generator_fc_depth', ['generator', 'fc_depth'])
    update_args2config(args2config, 'generator_pooling', ['generator', 'pooling'])
    update_args2config(args2config, 'generator_activation', ['generator', 'activation'])
    update_args2config(args2config, 'generator_fc_dropout_probability', ['generator', 'fc_dropout_probability'])
    update_args2config(args2config, 'generator_fc_norm_mode', ['generator', 'fc_norm_mode'])
    update_args2config(args2config, 'generator_num_flow_layers', ['generator', 'num_flow_layers'])
    update_args2config(args2config, 'generator_flow_depth', ['generator', 'flow_depth'])
    update_args2config(args2config, 'generator_flow_basis_fns', ['generator', 'flow_basis_fns'])
    update_args2config(args2config, 'generator_prior', ['generator', 'prior'])
    update_args2config(args2config, 'generator_flow_type', ['generator', 'flow_type'])
    update_args2config(args2config, 'generator_num_samples', ['generator', 'num_samples'])
    update_args2config(args2config, 'generator_conditional_modelling', ['generator', 'conditional_modelling'])
    update_args2config(args2config, 'generator_conditioning_mode', ['generator', 'conditioning_mode'])

    # crystal cell graph Net
    parser.add_argument('--discriminator_graph_model', type=str, default='mike')  # 'dime', or 'schnet', or 'mike' or None
    parser.add_argument('--discriminator_atom_embedding_size', type=int, default=32)  # embedding dimension for atoms
    parser.add_argument('--discriminator_graph_filters', type=int, default=28)  # number of neurons per graph convolution
    parser.add_argument('--discriminator_graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'self attention' 'full message passing'
    parser.add_argument('--discriminator_graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
    parser.add_argument('--discriminator_graph_norm', type=str, default='layer')  # None, 'layer', 'graph'
    parser.add_argument('--discriminator_num_spherical', type=int, default=6)  # dime angular basis functions, default is 6
    parser.add_argument('--discriminator_num_radial', type=int, default=12)  # dime radial basis functions, default is 12
    parser.add_argument('--discriminator_graph_convolution_cutoff', type=int, default=5)  # dime default is 5.0 A, schnet default is 10
    parser.add_argument('--discriminator_max_num_neighbors', type=int, default=32)  # dime default is 32
    parser.add_argument('--discriminator_radial_function', type=str, default='bessel')  # 'bessel' or 'gaussian' - only applies to mikenet
    add_bool_arg(parser, 'discriminator_add_spherical_basis', default=False)  # include spherical information in message aggregation - only applies to mikenet

    parser.add_argument('--discriminator_num_fc_layers', type=int, default=1)  # number of layers in NN models
    parser.add_argument('--discriminator_fc_depth', type=int, default=27)  # number of neurons per NN layer
    parser.add_argument('--discriminator_pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'
    parser.add_argument('--discriminator_activation', type=str, default='gelu')
    parser.add_argument('--discriminator_fc_dropout_probability', type=float, default=0)  # dropout probability, [0,1)
    parser.add_argument('--discriminator_fc_norm_mode', type=str, default='layer')  # None, 'batch', 'instance', 'layer'

    update_args2config(args2config, 'discriminator_graph_model', ['discriminator', 'graph_model'])
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
    update_args2config(args2config, 'discriminator_num_fc_layers', ['discriminator', 'num_fc_layers'])
    update_args2config(args2config, 'discriminator_fc_depth', ['discriminator', 'fc_depth'])
    update_args2config(args2config, 'discriminator_pooling', ['discriminator', 'pooling'])
    update_args2config(args2config, 'discriminator_activation', ['discriminator', 'activation'])
    update_args2config(args2config, 'discriminator_fc_dropout_probability', ['discriminator', 'fc_dropout_probability'])
    update_args2config(args2config, 'discriminator_fc_norm_mode', ['discriminator', 'fc_norm_mode'])

    # cell generator
    parser.add_argument('--gan_loss', type=str, default='wasserstein')  # 'wasserstein, 'standard'
    add_bool_arg(parser, 'train_generator_density', default=True)  # train on cell volume
    add_bool_arg(parser, 'train_generator_as_flow', default=False)  # train normalizing flow generator via flow loss
    add_bool_arg(parser, 'train_generator_on_randn', default=False)  # train model to match appropriate multivariate gaussian
    add_bool_arg(parser, 'train_generator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_generator_range_cutoff', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_generator_pure_packing', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_discriminator_adversarially', default=False)  # train generator on adversarially
    add_bool_arg(parser, 'train_discriminator_on_randn', default=False)  # train generator on cells generated from appropriately fit multivariate gaussians
    add_bool_arg(parser, 'train_discriminator_on_noise', default=False)  # train generator on extremely unit cells
    parser.add_argument('--cut_max_prob_training_after', type=int, default=10)  # stop applying flow losses after xx epochs

    update_args2config(args2config, 'gan_loss')
    update_args2config(args2config, 'train_generator_density')
    update_args2config(args2config, 'train_generator_as_flow')
    update_args2config(args2config, 'train_generator_on_randn')
    update_args2config(args2config, 'train_generator_adversarially')
    update_args2config(args2config, 'train_generator_range_cutoff')
    update_args2config(args2config, 'train_generator_pure_packing')
    update_args2config(args2config, 'train_discriminator_adversarially')
    update_args2config(args2config, 'train_discriminator_on_randn')
    update_args2config(args2config, 'train_discriminator_on_noise')
    update_args2config(args2config, 'cut_max_prob_training_after')

    return parser, args2config


def process_config(config):
    if config.machine == 'local':
        config.workdir = 'C:/Users\mikem\Desktop/CSP_runs'  # Working directory
    elif config.machine == 'cluster':
        config.workdir = '/scratch/mk8347/csd_runs/'
        config.dataset_path = '/scratch/mk8347/csd_runs/datasets/full_dataset'

    config.seeds.model = config.seeds.model % 10
    config.seeds.dataset = config.seeds.dataset % 10

    if config.test_mode:
        config.max_batch_size = 50
        #config.auto_batch_sizing = False
        config.num_samples = 1000
        #config.anomaly_detection = True
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
    #
    # config = parser.parse_args()
    # if config.config_file is not None:  # load up config from file
    #     yaml_config = load_yaml(config.config_file)
    #     for key in yaml_config.keys():  # overwrite config from yaml
    #         vars(config)[key] = yaml_config[key]

    # have to load before we go to the workdir
    if config.wandb.sweep:
        sweep_config = load_yaml(config.wandb.sweep_config_file)

    '''
    run the code
    '''

    predictor = Predictor(config)

    if config.wandb.sweep:  # todo sweep won't work in new no-save-and-load method, since we delete dataset at first instance
        for sweep_run in range(config.wandb.sweep_num_runs):
            wandb.login()
            if config.wandb.sweep_id is not None:  # continue a prior sweep
                sweep_id = config.wandb.sweep_id
            else:
                sweep_id = wandb.sweep(sweep_config, project=config.wandb.project_name)
                config.wandb.sweep_id = sweep_id
            wandb.agent(sweep_id, predictor.train, project=config.wandb.project_name, count=1)
    else:
        predictor.train()
