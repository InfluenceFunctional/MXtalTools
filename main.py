'''import statements'''
import wandb
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # annoying numpy error
warnings.filterwarnings("ignore", category=UserWarning) # annoying w&b error
from utils import load_yaml, add_bool_arg
from CSG_predictor import Predictor


'''
Predict crystal features given atom and molecule-level information
'''

# get command line input
parser = argparse.ArgumentParser()

# high level
parser.add_argument('--config_file', type=str, default='dev.yaml', required=False)
parser.add_argument('--run_num', type=int, default=0)
add_bool_arg(parser, 'explicit_run_enumeration', default=False)  # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
add_bool_arg(parser, 'test_mode', default=True)
parser.add_argument('--model_seed', type=int, default=0)
parser.add_argument('--dataset_seed', type=int, default=0)
parser.add_argument('--machine', type=str, default='local')  # 'local' (local windows venv) or 'cluster' (linux env)
parser.add_argument("--device", default="cuda", type=str)  # 'cuda' or 'cpu'
add_bool_arg(parser, 'skip_run_init', default=False)
parser.add_argument("--mode", default="single molecule classification", type=str)  # 'single molecule classification' 'joint modelling' 'single molecule regresion' 'cell classification'
parser.add_argument("--dataset_path", type=str, default = 'C:/Users\mikem\Desktop\CSP_runs\datasets/full_dataset')

# wandb login
parser.add_argument('--experiment_tag', type = str, default = 'MCryGAN_dev')
parser.add_argument('--wandb_username', type=str, default = 'mkilgour')
parser.add_argument('--project_name', type=str, default = 'MCryGAN')
# wandb reporting
parser.add_argument('--sample_reporting_frequency', type = int, default = 1)
add_bool_arg(parser, 'log_figures', default=True)
# wandb sweeps
parser.add_argument('--sweep_config_file', type=str, default='sweep_1.yaml')
add_bool_arg(parser, 'sweep', default=True) # whether to do a single run or use w&b to run a Bayesian sweep
parser.add_argument('--sweep_id', type=str, default = None) # put something here to continue a prior sweep, else None fo fresh sweeps
parser.add_argument('--sweep_num_runs', type=int, default=100)

# dataset settings
parser.add_argument('--target', type=str, default='molecule spherical defect') # 'rings', 'groups', 'screw', 'inversion','rotoinversion','mirror','rotation','glide', 'crystal system', 'lattice centering', 'spherical', 'planar'(not in Jan17 dataset)
parser.add_argument('--dataset_length', type=int, default=int(1e3))  # maximum number of items in the dataset before filtration
# data in modelling
parser.add_argument('--amount_of_features',type=str,default='minimum') #minimum, maximum
add_bool_arg(parser, '--concat_mol_to_atom_features',default=False)
#dataset composition
parser.add_argument('--include_sgs', type = str, default = ['P21/c']) # spacegroups to explicitly include in modelling - new!
parser.add_argument('--max_crystal_temperature', type=float, default=int(1e3))
parser.add_argument('--min_crystal_temperature', type=int, default=0)
parser.add_argument('--max_num_atoms', type=float, default=int(1e3))
parser.add_argument('--min_num_atoms', type=int, default=0)
parser.add_argument('--min_packing_coefficient', type=float, default = 0.55)
add_bool_arg(parser, '--include_organic', default = True)
add_bool_arg(parser, '--include_organometallic', default = True)
parser.add_argument('--max_atomic_number',type=int,default=87)
add_bool_arg(parser,'exclude_disordered_crystals',default=True)
add_bool_arg(parser,'exclude_polymorphs',default=True)
add_bool_arg(parser,'exclude_nonstandard_settings',default=True)
add_bool_arg(parser,'exclude_missing_r_factor',default=True)

#  training settings
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--history', type = int, default = 5)
parser.add_argument('--optimizer', type = str, default = 'adamw') # adam, adamw, sgd
parser.add_argument('--learning_rate', type = float, default = 1e-5) # base learning rate
parser.add_argument('--max_lr', type = float, default = 1e-3) # for warmup schedules
parser.add_argument('--beta1', type = float, default = 0.9) # adam and adamw opt
parser.add_argument('--beta2', type = float, default = 0.999) # adam and adamw opt
parser.add_argument('--weight_decay', type = float, default = 0.01) # for opt
parser.add_argument('--convergence_eps', type=float, default=1e-5)
add_bool_arg(parser, 'lr_schedule', default = False)
parser.add_argument('--gradient_norm_clip', type = float, default = 1)
add_bool_arg(parser, 'anomaly_detection', default=False)
# batching
parser.add_argument('--initial_batch_size', type=int, default=10000)
add_bool_arg(parser, 'auto_batch_sizing', default=True) # whether to densely connect dimenet outputs
parser.add_argument('--auto_batch_reduction', type = float, default = 0.2) # leeway factor to reduce batch size at end of auto-sizing run

#  single-molecule graph Net
parser.add_argument('--graph_model', type=str, default='mike') #'dime', or 'schnet', or 'mike' or None
parser.add_argument('--atom_embedding_size', type=int, default = 32) # embedding dimension for atoms
parser.add_argument('--graph_filters', type=int, default=28)  # number of neurons per graph convolution
parser.add_argument('--graph_convolution', type=str, default='full message passing')  # type of graph convolution for mikenet only 'self attention' 'full message passing'
parser.add_argument('--graph_convolutions_layers', type=int, default=0)  # number of graph convolution blocks
parser.add_argument('--graph_norm',type = str, default = 'layer') # None, 'layer', 'graph'
parser.add_argument('--num_spherical', type = int, default = 6) # dime angular basis functions, default is 6
parser.add_argument('--num_radial', type = int, default = 12) # dime radial basis functions, default is 12
parser.add_argument('--graph_convolution_cutoff', type = int, default = 5) # dime default is 5.0 A, schnet default is 10
parser.add_argument('--max_num_neighbors', type = int, default = 32) # dime default is 32
parser.add_argument('--radial_function', type=str, default = 'bessel') # 'bessel' or 'gaussian' - only applies to mikenet
add_bool_arg(parser, 'add_spherical_basis', default = False) # include spherical information in message aggregation - only applies to mikenet

parser.add_argument('--num_fc_layers', type=int, default=1)  # number of layers in NN models
parser.add_argument('--fc_depth', type=int, default=27)  # number of neurons per NN layer
parser.add_argument('--pooling', type=str, default='attention')  # 'mean', 'attention', 'set2set', 'combo'
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--fc_dropout_probability', type=float, default = 0) # dropout probability, [0,1)
parser.add_argument('--fc_norm_mode', type=str, default='layer') # None, 'batch', 'instance', 'layer'

# flow model
parser.add_argument('--num_flow_layers', type=int, default=3) # number of flow layers
parser.add_argument('--flow_depth', type=int, default=16) # number of filters per flow layer
parser.add_argument('--flow_basis_fns', type=int, default=8) # number of basis functions for spline NF model
parser.add_argument('--flow_prior', type=str, default='multivariate normal') # type of prior distribution
parser.add_argument('--flow_type', type=str, default='nsf_ar') # type of flow model
parser.add_argument('--num_samples', type=int, default=10000) # number of samples to generate for analysis
add_bool_arg(parser,'conditional_modelling', default=True) # whether to use molecular features as conditions for normalizing flow model
parser.add_argument('--conditioning_mode', type=str, default='graph model') # how to derive molecular conditioning - graph model or just selected features


config = parser.parse_args()
if config.config_file is not None: # load up config from file
    yaml_config = load_yaml(config.config_file)
    for key in yaml_config.keys(): # overwrite config from yaml
        vars(config)[key] = yaml_config[key]

# have to load before we go to the workdir
if config.sweep:
    sweep_config = load_yaml(config.sweep_config_file)

if config.machine == 'local':
    config.workdir ='C:/Users\mikem\Desktop/CSP_runs'  # Working directory
elif config.machine == 'cluster':
    config.workdir = '/scratch/mk8347/csd_runs/'

config.model_seed = config.model_seed % 10
config.dataset_seed = config.dataset_seed % 10


if config.test_mode:
    if config.mode == 'joint modelling':
        config.initial_batch_size = 100
    else:
        config.initial_batch_size = 10
    config.auto_batch_sizing = False
    config.num_samples = 1000
    config.dataset_path = 'C:/Users\mikem\Desktop\CSP_runs\datasets/test_dataset'
    config.anomaly_detection = True


# =====================================
if __name__ == '__main__':

    predictor = Predictor(config)

    if config.sweep:
        for sweep_run in range(config.sweep_num_runs):
            wandb.login()
            if config.sweep_id is not None:  # continue a prior sweep
                sweep_id = config.sweep_id
            else:
                sweep_id = wandb.sweep(sweep_config, project=config.project_name)
                config.sweep_id = sweep_id
            wandb.agent(sweep_id, predictor.train, project=config.project_name, count=1)
    else:
        predictor.train()