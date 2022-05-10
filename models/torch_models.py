'''Import statements'''
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric.nn as gnn
from DimeNetCustom import CustomDimeNet
from CustomSchNet import CustomSchNet
from MikesGraphNet import MikesGraphNet, FCBlock
import sys
from nflib.flows import *
from nflib.nets import *
from nflib.spline_flows import *
from torch.distributions import MultivariateNormal
import itertools
from sklearn.decomposition import PCA



class FlowModel(nn.Module):
    def __init__(self, config, dataDims):
        super(FlowModel,self).__init__()
        torch.manual_seed(config.model_seed)
        # https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib1.ipynb
        # nice review https://arxiv.org/pdf/1912.02762.pdf
        self.flow_dimension = dataDims['n crystal features']

        if config.conditional_modelling:
            self.n_conditional_features = dataDims['n conditional features']
            if config.conditioning_mode == 'graph model':
                self.conditioner = CSP_model(config, dataDims, return_latent=True)
                self.n_conditional_features = config.fc_depth # will concatenate the graph model latent representation to the selected molecule features
            elif config.conditioning_mode == 'molecule features':
                self.conditioner = None
            else:
                print(config.conditioning_mode + ' is not an implemented conditioner!')
                sys.exit()
        else:
            self.n_conditional_features = 0

        # normalizing flow is a combination of a prior and some flows
        if config.device.lower() == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # prior
        if config.flow_prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(dataDims['n crystal features']), torch.eye(dataDims['n crystal features']))
        else:
            print(config.flow_prior + ' is not an implemented prior!!')
            sys.exit()

        # flows
        if config.flow_type == 'nsf':
            nfs_flow = NSF_CL if True else NSF_AR
            flows = [nfs_flow(dim=dataDims['n crystal features'],
                              K=config.flow_basis_fns,
                              B=3,
                              hidden_dim=config.flow_depth,
                              conditioning_dim=self.n_conditional_features
                              ) for _ in range(config.num_flow_layers)]
            convs = [Invertible1x1Conv(dim=dataDims['n crystal features']) for _ in flows]
            norms = [ActNorm(dim=dataDims['n crystal features']) for _ in flows]
            self.flow = NormalizingFlow(list(itertools.chain(*zip(norms,convs, flows))),self.n_conditional_features)
        elif config.flow_type.lower() == 'glow':
            flows = [AffineHalfFlow(dim=dataDims['n crystal features'], nh=config.flow_depth, parity=i % 2)
                     for i in range(config.num_flow_layers)]
            convs = [Invertible1x1Conv(dim=dataDims['n crystal features']) for _ in flows]
            norms = [ActNorm(dim=dataDims['n crystal features']) for _ in flows]
            self.flow = NormalizingFlow(list(itertools.chain(*zip(norms,convs,flows))),self.n_conditional_features)
        elif config.flow_type.lower() == 'made':
            flows = [MAF(dim=dataDims['n crystal features'], nh=config.flow_depth, parity=i % 2)
                     for i in range(config.num_flow_layers)]
            norms = [ActNorm(dim=dataDims['n crystal features']) for _ in flows]
            self.flow = NormalizingFlow(list(itertools.chain(*zip(norms,flows))),self.n_conditional_features)
        else:
            print(config.flow_type + ' is not an implemented flow!!')
            sys.exit()


    def destandardize_samples(self,x,dataDims, do_rounding=True):
        y=x.copy()
        for i in range(dataDims['n crystal features']):
            if y.ndim == 2:
                vec = x[:,i]
            elif y.ndim == 3:
                vec = x[:,:,i]
            else:
                print("Array has wrong number of dims!")
                sys.exit()

            vec = vec * dataDims['stds'][i] + dataDims['means'][i]

            if do_rounding:
                if dataDims['dtypes'][i] == 'bool':
                    vec = np.round(np.clip(vec,a_min=0,a_max=1))
                elif dataDims['dtypes'][i] == 'int32':
                    vec = np.round(vec)

            if y.ndim == 2:
                y[:,i] = vec
            elif y.ndim == 3:
                y[:,:,i] = vec

        return y

    def standardize_samples(self,x,dataDims):
        y=x.copy()
        for i in range(dataDims['n crystal features']):
            if y.ndim == 2:
                y[:,i] = (y[:,i] - dataDims['means'][i]) / dataDims['stds'][i]
            elif y.ndim == 3:
                y[:,:,i] = (y[:,:,i] -  dataDims['means'][i])/ dataDims['stds'][i]
        return y

    def fit_pca(self,data,print_variance = False):
        pca = PCA(n_components = data.shape[1])
        pca.fit(data)
        if print_variance:
            print("Pca Explained Variance:")
            print(pca.explained_variance_ratio_)
        return pca

    def pca_sampling(self,pca,n_samples):
        pc_prior = np.zeros((n_samples, pca.n_components))
        for i in range(pca.n_components):
            pc_prior[:, i] = np.random.randn(len(pc_prior)) * np.sqrt(pca.explained_variance_[i])
        return pca.inverse_transform(pc_prior)

    def forward(self, x):
        if self.n_conditional_features > 0:
            x = self.get_conditions(x)
        zs, log_det = self.flow.forward(x.float())
        prior_logprob = self.prior.log_prob(zs[-1].cpu()).view(x.size(0), -1).sum(1)
        return zs[-1], prior_logprob.to(log_det.device), log_det

    def backward(self, z):
        if self.n_conditional_features > 0:
            z = self.get_conditions(z)
        xs, log_det = self.flow.backward(z.float())
        return xs[-1], log_det

    def sample(self, num_samples, conditions = None):
        if conditions is not None:
            if self.conditioner is not None:
                conditions = self.conditioner(conditions)

        z = self.prior.sample((num_samples,)).to(self.device)
        if conditions is not None:
            z = torch.cat((z,conditions.to(z.device)),dim=1)
        xs, _ = self.flow.backward(z.float())
        return xs[-1]


    def score(self, x):
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det)


    def get_conditions(self,x):
        if self.conditioner is not None:
            conditions =  self.conditioner(x)
            return torch.cat((x.y[0],conditions),dim=1)
        else:
            return x

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, n_conditional_features = 0):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.conditioning_dims = n_conditional_features

    def forward(self, x):
        if self.conditioning_dims > 0:
            conditions = x[:,-self.conditioning_dims:]
            x = x[:,:-self.conditioning_dims]

        log_det = torch.zeros(len(x)).to(x.device)
        zs = [x]
        for i,flow in enumerate(self.flows):
            if ('nsf' in flow._get_name().lower()) and (self.conditioning_dims > 0): # conditioning only implemented for spline flow
                x, ld = flow.forward(torch.cat((x,conditions),dim=1))
            else:
                x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        if self.conditioning_dims > 0:
            conditions = z[:,-self.conditioning_dims:]
            z = z[:,:-self.conditioning_dims]

        log_det = torch.zeros(len(z)).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            if ('nsf' in flow._get_name().lower()) and (self.conditioning_dims > 0):
                z, ld = flow.backward(torch.cat((z,conditions),dim=1))
            else:
                z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class CSP_model(nn.Module):
    def __init__(self,config, dataDims, return_latent=False):
        super(CSP_model,self).__init__()
        # initialize constants and layers
        self.return_latent = return_latent
        self.activation = config.activation
        self.num_fc_layers = config.num_fc_layers
        self.fc_depth = config.fc_depth
        self.graph_model = config.graph_model
        self.classes = dataDims['output classes']
        self.graph_filters = config.graph_filters
        self.n_mol_feats = dataDims['mol features']
        self.n_atom_feats = dataDims['atom features']
        if not config.concat_mol_to_atom_features: # if we are not adding molwise feats to atoms, subtract the dimension
            self.n_atom_feats -= self.n_mol_feats
        self.pool_type = config.pooling
        self.fc_norm_mode = config.fc_norm_mode
        self.embedding_dim = config.atom_embedding_size

        torch.manual_seed(config.model_seed)

        if config.graph_model is not None:
            if config.graph_model == 'dime':
                self.graph_net = CustomDimeNet(
                    graph_filters = config.graph_filters,
                    hidden_channels= config.fc_depth,
                    out_channels=self.classes,
                    num_blocks=config.graph_convolution_layers,
                    num_spherical=config.num_spherical,
                    num_radial=config.num_radial,
                    cutoff=config.graph_convolution_cutoff,
                    max_num_neighbors=config.max_num_neighbors,
                    num_output_layers=1,
                    dense = False,
                    num_atom_features = self.n_atom_feats,
                    atom_embedding_dims = dataDims['atom embedding dict sizes'],
                    embedding_hidden_dimension = config.atom_embedding_size,
                    norm = config.graph_norm,
                    dropout = config.fc_dropout_probability
                )
            elif config.graph_model == 'schnet':
                self.graph_net = CustomSchNet(
                    hidden_channels = config.fc_depth,
                    num_filters = config.graph_filters,
                    num_interactions = config.graph_convolution_layers,
                    num_gaussians = config.num_radial,
                    cutoff = config.graph_convolution_cutoff,
                    max_num_neighbors = config.max_num_neighbors,
                    readout = 'mean',
                    num_atom_features = self.n_atom_feats,
                    embedding_hidden_dimension = config.atom_embedding_size,
                    atom_embedding_dims = dataDims['atom embedding dict sizes'],
                    norm = config.graph_norm,
                    dropout=config.fc_dropout_probability
                )
            elif config.graph_model == 'mike': # mike's net
                self.graph_net = MikesGraphNet(
                    graph_convolution_filters = config.graph_filters,
                    graph_convolution=config.graph_convolution,
                    out_channels = config.fc_depth,
                    hidden_channels = config.fc_depth,
                    num_blocks = config.graph_convolution_layers,
                    num_radial = config.num_radial,
                    num_spherical = config.num_spherical,
                    max_num_neighbors = config.max_num_neighbors,
                    cutoff = config.graph_convolution_cutoff,
                    activation = 'gelu',
                    embedding_hidden_dimension = config.atom_embedding_size,
                    num_atom_features=self.n_atom_feats,
                    norm = config.graph_norm,
                    dropout = config.fc_dropout_probability,
                    spherical_embedding=config.add_spherical_basis,
                    radial_embedding = config.radial_function,
                    atom_embedding_dims=dataDims['atom embedding dict sizes'],
                    attention_heads=config.num_attention_heads
                )
            else:
                print(config.graph_model + ' is not a valid graph model!!')
                sys.exit()
        else:
            self.graph_net = nn.Identity()
            self.graph_filters = 0 # no accounting for dime inputs or outputs
            self.pools = nn.Identity()

        # initialize global pooling operation
        if self.graph_model is not None:
            if config.pooling == 'mean':
                self.global_pool = gnn.global_mean_pool
            elif config.pooling == 'sum':
                self.global_pool = gnn.global_add_pool
            elif config.pooling == 'attention':
                self.global_pool = gnn.GlobalAttention(nn.Sequential(nn.Linear(config.fc_depth,config.fc_depth),nn.GELU(),nn.Linear(config.fc_depth,1)))
            elif config.pooling == 'set2set': # deprecated
                self.global_pool = nn.Sequential(gnn.Set2Set(config.fc_depth, processing_steps = 5, num_layers = 1), nn.Linear(config.fc_depth * 2, config.fc_depth))

        # molecule features FC layer
        if self.n_mol_feats != 0:
            self.mol_fc = nn.Linear(self.n_mol_feats, self.n_mol_feats)

        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(config.fc_depth, config.fc_depth)
            for _ in range(config.num_fc_layers)
        ])

        self.fc_norms = torch.nn.ModuleList([
            Normalization(config.fc_norm_mode, config.fc_depth)
            for _ in range(config.num_fc_layers)
        ])

        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=config.fc_dropout_probability)
            for _ in range(config.num_fc_layers)
        ])

        self.fc_activations = torch.nn.ModuleList([
            Activation('gelu', config.fc_depth)
            for _ in range(config.num_fc_layers)
        ])

        # adjust first layer
        if config.graph_model is None:
            self.fc_layers[0] = nn.Linear(self.n_mol_feats, config.fc_depth)
            self.fc_norms[0] = Normalization(config.fc_norm_mode, self.n_mol_feats)
        else:
            self.fc_layers[0] = nn.Linear(config.fc_depth + self.n_mol_feats, config.fc_depth)
            self.fc_norms[0] = Normalization(config.fc_norm_mode, config.fc_depth + self.n_mol_feats)

        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_norms = nn.ModuleList(self.fc_norms)

        self.output_heads = torch.nn.ModuleList([
            nn.Sequential(nn.Linear(config.fc_depth, config.fc_depth), nn.ReLU(), nn.Linear(config.fc_depth, self.classes[i], bias=False))
            for i in range(dataDims['prediction tasks'])
        ])

    def forward(self,data):
        if self.graph_model is not None:
            x = data.x
            pos = data.pos
            x = self.graph_net(x[:,:self.n_atom_feats],pos,data.batch) # get atoms encoding
            x = self.global_pool(x, data.batch) # aggregate atoms to molecule

        mol_inputs = data.x[:,-self.n_mol_feats:]
        mol_feats = self.mol_fc(gnn.global_max_pool(mol_inputs,data.batch).float()) # not actually pooling here, as the values are all the same for each molecule

        if self.graph_model is not None:
            x = torch.cat((x,mol_feats),dim=1)
        else:
            x = mol_feats

        for norm, linear, activation, dropout in zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts):
            x = dropout(activation(linear(norm(x))))

        if self.return_latent: # immediately return the latent space prediction
            return x
        else:
            if len(self.output_heads) == 1:
                return self.output_heads[0](x) # each task has its own head
            else:
                return [self.output_heads[i](x) for i in range(len(self.output_heads))]


class kernelActivation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis)) # positive and negative values for Dirichlet Kernel
        gamma = 1/(6*(self.dict[-1]-self.dict[-2])**2) # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma',torch.ones(1) * gamma) #

        #self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1,1), groups=int(channels), bias=False)

        #nn.init.normal(self.linear.weight.data, std=0.1)


    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x)==2:
            x = x.reshape(2,self.channels,1)

        return torch.exp(-self.gamma*(x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1) # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]) # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
        super().__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self, input):
        return self.norm(input)
