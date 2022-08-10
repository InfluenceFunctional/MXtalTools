import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from nflib.flows import Invertible1x1Conv
from nflib.spline_flows import NSF_CL
from torch.distributions import MultivariateNormal, Uniform
import itertools
from models.torch_models import molecule_graph_model, Normalization, ActNorm, general_MLP, independent_gaussian_model


class crystal_generator(nn.Module):
    def __init__(self, config, dataDims, model_override = None):
        super(crystal_generator, self).__init__()

        self.device = config.device
        self.generator_model_type = config.generator.model_type

        if config.generator.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(dataDims['n crystal features']), torch.eye(dataDims['n crystal features']))
        elif config.generator.prior.lower() == 'uniform':
            self.prior = Uniform(low=0, high=1)
        else:
            print(config.generator.prior + ' is not an implemented prior!!')
            sys.exit()
        '''
        conditioning model
        '''
        if config.generator.conditioning_mode == 'graph model':  # molecular graph model
            self.conditioner = molecule_graph_model(dataDims,
                                                    seed=config.seeds.model,
                                                    output_dimension=config.generator.fc_depth,
                                                    activation=config.generator.conditioner_activation,
                                                    num_fc_layers=config.generator.conditioner_num_fc_layers,
                                                    fc_depth=config.generator.conditioner_fc_depth,
                                                    fc_dropout_probability=config.generator.conditioner_fc_dropout_probability,
                                                    fc_norm_mode=config.generator.conditioner_fc_norm_mode,
                                                    graph_model=config.generator.graph_model,
                                                    graph_filters=config.generator.graph_filters,
                                                    graph_convolutional_layers=config.generator.graph_convolution_layers,
                                                    concat_mol_to_atom_features=True,
                                                    pooling=config.generator.pooling,
                                                    graph_norm=config.generator.graph_norm,
                                                    num_spherical=config.generator.num_spherical,
                                                    num_radial=config.generator.num_radial,
                                                    graph_convolution=config.generator.graph_convolution,
                                                    num_attention_heads=config.generator.num_attention_heads,
                                                    add_spherical_basis=config.generator.add_spherical_basis,
                                                    atom_embedding_size=config.generator.atom_embedding_size,
                                                    radial_function=config.generator.radial_function,
                                                    max_num_neighbors=config.generator.max_num_neighbors,
                                                    convolution_cutoff=config.generator.graph_convolution_cutoff,
                                                    )
        elif config.generator.conditioning_mode == 'molecule features':
            self.conditioner = general_MLP(layers=config.generator.conditioner_num_fc_layers,
                                           filters=config.generator.conditioner_fc_depth,
                                           norm=config.generator.conditioner_fc_norm_mode,
                                           dropout=config.generator.conditioner_fc_dropout_probability,
                                           input_dim=dataDims['n conditional features'],
                                           output_dim=config.generator.fc_depth,
                                           conditioning_dim=0,
                                           seed=config.seeds.model
                                           )
        '''
        generator model
        '''
        if self.generator_model_type.lower() == 'mlp':  # simple MLP
            self.model = general_MLP(layers=config.generator.num_fc_layers,
                                     filters=config.generator.fc_depth,
                                     norm=config.generator.fc_norm_mode,
                                     dropout=config.generator.fc_dropout_probability,
                                     input_dim=dataDims['n crystal features'],
                                     output_dim=dataDims['n crystal features'],
                                     conditioning_dim=config.generator.fc_depth,
                                     seed=config.seeds.model
                                     )
        elif self.generator_model_type.lower() == 'nf':  # conditioned normalizing flow
            self.model = crystal_nf(config, dataDims, self.prior)
        elif self.generator_model_type.lower() == 'fit normal':
            assert config.generator.prior.lower() == 'multivariate normal'
            self.model = independent_gaussian_model(config, dataDims, dataDims['means'], dataDims['stds'])
        else:
            print(self.generator_model_type + ' is not an implemented generator model!')
            sys.exit()

    def sample_latent(self, n_samples):
        #return torch.ones((n_samples,12)).to(self.device) # when we don't actually want any noise (test purposes)
        return self.prior.sample((n_samples,)).to(self.device)

    def forward(self, n_samples, z=None, conditions=None, return_latent = False, return_condition = False, return_prior = False):
        if z is None:  # sample random numbers from simple prior
            z = self.sample_latent(n_samples)
            #z = torch.zeros_like(z0)

        if conditions is not None:
            conditions_encoding = self.conditioner(conditions)
        else:
            conditions_encoding = None

        # run through model
        if any((return_condition, return_prior, return_latent)):
            output = [self.model(z, conditions=conditions_encoding, return_latent = return_latent)]
            if return_prior:
                output.append(z)
            if return_condition:
                output.append(conditions_encoding)
            return output

        else:
            if not 'nf' in self.generator_model_type: # todo implement latent return in NF model
                return self.model(z, conditions=conditions_encoding, return_latent = return_latent)
            else:
                x, _ = self.model.backward(z, conditions=conditions_encoding)  # normalizing flow runs backwards from z->x
                return x

    def nf_forward(self, x, conditions=None):
        if conditions is not None:
            conditions_encoding = self.conditioner(conditions)
        else:
            conditions_encoding = None

        return self.model.forward(x, conditions=conditions_encoding)


class crystal_nf(nn.Module):
    def __init__(self, config, dataDims, prior):
        super(crystal_nf, self).__init__()
        torch.manual_seed(config.seeds.model)
        # https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib1.ipynb
        # nice review https://arxiv.org/pdf/1912.02762.pdf
        self.flow_dimension = dataDims['n crystal features']
        self.prior = prior

        if config.generator.conditional_modelling:
            self.n_conditional_features = dataDims['n conditional features']
            if config.generator.conditioning_mode == 'graph model':
                self.n_conditional_features = config.generator.fc_depth  # will concatenate the graph model latent representation to the selected molecule features
        else:
            self.n_conditional_features = 0

        # normalizing flow is a combination of a prior and some flows
        if config.device.lower() == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # flows
        nsf_flow = NSF_CL
        flows = [nsf_flow(dim=dataDims['n crystal features'],
                          K=config.generator.flow_basis_fns,
                          B=3,
                          hidden_dim=config.generator.flow_depth,
                          conditioning_dim=self.n_conditional_features
                          ) for _ in range(config.generator.num_flow_layers)]
        convs = [Invertible1x1Conv(dim=dataDims['n crystal features']) for _ in flows]
        norms = [ActNorm(dim=dataDims['n crystal features']) for _ in flows]
        self.flow = NormalizingFlow2(list(itertools.chain(*zip(norms, convs, flows))), self.n_conditional_features)

    def forward(self, x, conditions=None):
        zs, log_det = self.flow.forward(x.float(), conditions=conditions)

        prior_logprob = self.prior.log_prob(zs.cpu()).view(x.size(0), -1).sum(1)

        return zs, prior_logprob.to(log_det.device), log_det

    def backward(self, z, conditions=None):
        xs, log_det = self.flow.backward(z.float(), conditions=conditions)

        return xs, log_det

    def sample(self, num_samples, conditions=None):
        z = self.prior.sample((num_samples,)).to(self.device)
        prior_logprob = self.prior.log_prob(z.cpu())

        xs, log_det = self.flow.backward(z.float(), conditions=conditions)

        return xs, z, prior_logprob, log_det

    def score(self, x):
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det)


class NormalizingFlow2(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, n_conditional_features=0):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.conditioning_dims = n_conditional_features

    def forward(self, x, conditions=None):
        log_det = torch.zeros(len(x)).to(x.device)

        for i, flow in enumerate(self.flows):
            if ('nsf' in flow._get_name().lower()) and (self.conditioning_dims > 0):  # conditioning only implemented for spline flow
                x, ld = flow.forward(torch.cat((x, conditions), dim=1))
            else:
                x, ld = flow.forward(x)

            log_det += ld

        return x, log_det

    def backward(self, z, conditions=None):
        log_det = torch.zeros(len(z)).to(z.device)

        zz = []
        for i, flow in enumerate(self.flows[::-1]):
            if ('nsf' in flow._get_name().lower()) and (self.conditioning_dims > 0):
                z, ld = flow.backward(torch.cat((z, conditions), dim=1))
            else:
                z, ld = flow.backward(z)
            log_det += ld
            zz.append(z)

        return z, log_det
