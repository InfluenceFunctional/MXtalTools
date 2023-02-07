
class FlowModel(nn.Module):
    def __init__(self, config, dataDims):
        super(FlowModel, self).__init__()
        torch.manual_seed(config.seeds.model)
        # https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib1.ipynb
        # nice review https://arxiv.org/pdf/1912.02762.pdf
        self.flow_dimension = dataDims['num lattice features']

        if config.generator.conditional_modelling:
            self.n_conditional_features = dataDims['num conditional features']
            if config.generator.conditioning_mode == 'graph model':
                self.conditioner = molecule_graph_model(config, dataDims, return_latent=True)
                self.n_conditional_features = config.fc_depth  # will concatenate the graph model latent representation to the selected molecule features
            elif config.generator.conditioning_mode == 'molecule features':
                self.conditioner = None
            else:
                print(config.generator.conditioning_mode + ' is not an implemented conditioner!')
                sys.exit()
        else:
            self.n_conditional_features = 0

        # normalizing flow is a combination of a prior and some flows
        if config.device.lower() == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # prior
        if config.generator.prior == 'multivariate normal':
            self.prior = MultivariateNormal(torch.zeros(dataDims['num lattice features']), torch.eye(dataDims['num lattice features']))
        else:
            print(config.generator.prior + ' is not an implemented prior!!')
            sys.exit()

        # flows
        if config.flow_type == 'nsf':
            nsf_flow = NSF_CL if True else NSF_AR
            flows = [nsf_flow(dim=dataDims['num lattice features'],
                              K=config.flow_basis_fns,
                              B=3,
                              hidden_dim=config.flow_depth,
                              conditioning_dim=self.n_conditional_features
                              ) for _ in range(config.num_flow_layers)]
            convs = [Invertible1x1Conv(dim=dataDims['num lattice features']) for _ in flows]
            norms = [ActNorm(dim=dataDims['num lattice features']) for _ in flows]
            self.flow = NormalizingFlow(list(itertools.chain(*zip(norms, convs, flows))), self.n_conditional_features)
        elif config.flow_type.lower() == 'glow':
            flows = [AffineHalfFlow(dim=dataDims['num lattice features'], nh=config.flow_depth, parity=i % 2)
                     for i in range(config.num_flow_layers)]
            convs = [Invertible1x1Conv(dim=dataDims['num lattice features']) for _ in flows]
            norms = [ActNorm(dim=dataDims['num lattice features']) for _ in flows]
            self.flow = NormalizingFlow(list(itertools.chain(*zip(norms, convs, flows))), self.n_conditional_features)
        elif config.flow_type.lower() == 'made':
            flows = [MAF(dim=dataDims['num lattice features'], nh=config.flow_depth, parity=i % 2)
                     for i in range(config.num_flow_layers)]
            norms = [ActNorm(dim=dataDims['num lattice features']) for _ in flows]
            self.flow = NormalizingFlow(list(itertools.chain(*zip(norms, flows))), self.n_conditional_features)
        else:
            print(config.flow_type + ' is not an implemented flow!!')
            sys.exit()

    def destandardize_samples(self, x, dataDims, do_rounding=True):
        y = x.copy()
        for i in range(dataDims['num lattice features']):
            if y.ndim == 2:
                vec = x[:, i]
            elif y.ndim == 3:
                vec = x[:, :, i]
            else:
                print("Array has wrong number of dims!")
                sys.exit()

            vec = vec * dataDims['lattice stds'][i] + dataDims['lattice means'][i]

            if do_rounding:
                if dataDims['lattice dtypes'][i] == 'bool':
                    vec = np.round(np.clip(vec, a_min=0, a_max=1))
                elif dataDims['lattice dtypes'][i] == 'int32':
                    vec = np.round(vec)

            if y.ndim == 2:
                y[:, i] = vec
            elif y.ndim == 3:
                y[:, :, i] = vec

        return y

    def standardize_samples(self, x, dataDims):
        y = x.copy()
        for i in range(dataDims['num lattice features']):
            if y.ndim == 2:
                y[:, i] = (y[:, i] - dataDims['lattice means'][i]) / dataDims['lattice stds'][i]
            elif y.ndim == 3:
                y[:, :, i] = (y[:, :, i] - dataDims['lattice means'][i]) / dataDims['lattice stds'][i]
        return y

    def fit_pca(self, data, print_variance=False):
        pca = PCA(n_components=data.shape[1])
        pca.fit(data)
        if print_variance:
            print("Pca Explained Variance:")
            print(np.asarray(pca.explained_variance_ratio_).astype('float16'))
        return pca

    def pca_sampling(self, pca, n_samples):
        pc_prior = np.zeros((n_samples, pca.n_components))
        for i in range(pca.n_components):
            pc_prior[:, i] = np.random.randn(len(pc_prior)) * np.sqrt(pca.explained_variance_[i])
        return pca.inverse_transform(pc_prior)

    def forward(self, x):  # todo fix y
        if self.n_conditional_features > 0:
            conditions = self.get_conditions(x)
            if self.conditioner is not None:
                x.y[0] = torch.cat((x.y[0], conditions), dim=1)

        zs, log_det = self.flow.forward(x.y[0].float())
        prior_logprob = self.prior.log_prob(zs[-1].cpu()).view(x.y[0].size(0), -1).sum(1)
        return zs[-1], prior_logprob.to(log_det.device), log_det

    def backward(self, z):  # todo fix y
        if self.n_conditional_features > 0:
            conditions = self.get_conditions(z)
            if self.conditioner is not None:
                z.y[0] = torch.cat((z.y[0], conditions), dim=1)

        xs, log_det = self.flow.backward(z.y[0].float())
        return xs[-1], log_det

    def sample(self, num_samples, conditions=None):
        z = self.prior.sample((num_samples,)).to(self.device)

        if conditions is not None:
            conditions = self.get_conditions(conditions)
            z = torch.cat((z, conditions.to(z.device)), dim=1)

        xs, _ = self.flow.backward(z.float())
        return xs[-1]

    def score(self, x):
        _, prior_logprob, log_det = self.forward(x)
        return (prior_logprob + log_det)

    def get_conditions(self, x):  # todo fix y
        if self.conditioner is not None:
            return self.conditioner(x)
        else:
            return x.y[0][:, -self.n_conditional_features:]


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows, n_conditional_features=0):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.conditioning_dims = n_conditional_features

    def forward(self, x):
        if self.conditioning_dims > 0:
            conditions = x[:, -self.conditioning_dims:]
            x = x[:, :-self.conditioning_dims]

        log_det = torch.zeros(len(x)).to(x.device)
        zs = [x]
        for i, flow in enumerate(self.flows):
            if ('nsf' in flow._get_name().lower()) and (self.conditioning_dims > 0):  # conditioning only implemented for spline flow
                x, ld = flow.forward(torch.cat((x, conditions), dim=1))
            else:
                x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        if self.conditioning_dims > 0:
            conditions = z[:, -self.conditioning_dims:]
            z = z[:, :-self.conditioning_dims]

        log_det = torch.zeros(len(z)).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            if ('nsf' in flow._get_name().lower()) and (self.conditioning_dims > 0):
                z, ld = flow.backward(torch.cat((z, conditions), dim=1))
            else:
                z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

