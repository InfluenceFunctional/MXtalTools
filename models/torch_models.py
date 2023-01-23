'''Import statements'''
from models.MikesGraphNet import MikesGraphNet
import sys

from models.global_aggregation import global_aggregation
from models.model_components import general_MLP
from nflib.spline_flows import *
from torch.distributions import MultivariateNormal
from ase import Atoms
from models.asymmetric_radius_graph import asymmetric_radius_graph
from utils import parallel_compute_rdf_torch


class molecule_graph_model(nn.Module):
    def __init__(self, dataDims, seed,
                 num_atom_feats,
                 num_mol_feats,
                 output_dimension,
                 activation,
                 num_fc_layers,
                 fc_depth,
                 fc_dropout_probability,
                 fc_norm_mode,
                 graph_model,
                 graph_filters,
                 graph_convolutional_layers,
                 concat_mol_to_atom_features,
                 pooling,
                 graph_norm,
                 num_spherical,
                 num_radial,
                 graph_convolution,
                 num_attention_heads,
                 add_spherical_basis,
                 add_torsional_basis,
                 atom_embedding_size,
                 radial_function,
                 max_num_neighbors,
                 convolution_cutoff,
                 return_latent=False, crystal_mode=False, crystal_convolution_type=None, device='cuda'):
        super(molecule_graph_model, self).__init__()
        # initialize constants and layers
        self.device = device
        self.return_latent = return_latent
        self.activation = activation
        self.num_fc_layers = num_fc_layers
        self.fc_depth = fc_depth
        self.fc_dropout_probability = fc_dropout_probability
        self.fc_norm_mode = fc_norm_mode
        self.graph_model = graph_model
        self.graph_convolution = graph_convolution
        self.output_classes = output_dimension
        self.graph_convolution_layers = graph_convolutional_layers
        self.graph_filters = graph_filters
        self.graph_norm = graph_norm
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_attention_heads = num_attention_heads
        self.add_spherical_basis = add_spherical_basis
        self.add_torsional_basis = add_torsional_basis
        self.n_mol_feats = num_mol_feats  # dataDims['num mol features']
        self.n_atom_feats = num_atom_feats  # dataDims['num atom features']
        self.radial_function = radial_function
        self.max_num_neighbors = max_num_neighbors
        self.graph_convolution_cutoff = convolution_cutoff
        if not concat_mol_to_atom_features:  # if we are not adding molwise feats to atoms, subtract the dimension
            self.n_atom_feats -= self.n_mol_feats
        self.pooling = pooling
        self.fc_norm_mode = fc_norm_mode
        self.embedding_dim = atom_embedding_size
        self.crystal_mode = crystal_mode
        self.crystal_convolution_type = crystal_convolution_type

        torch.manual_seed(seed)

        if self.graph_model is not None:
            if self.graph_model == 'mike':  # mike's net - others currently deprecated. For future, implement new convolutions in the mikenet class
                self.graph_net = MikesGraphNet(
                    crystal_mode=crystal_mode,
                    crystal_convolution_type=self.crystal_convolution_type,
                    graph_convolution_filters=self.graph_filters,
                    graph_convolution=self.graph_convolution,
                    out_channels=self.fc_depth,
                    hidden_channels=self.embedding_dim,
                    num_blocks=self.graph_convolution_layers,
                    num_radial=self.num_radial,
                    num_spherical=self.num_spherical,
                    max_num_neighbors=self.max_num_neighbors,
                    cutoff=self.graph_convolution_cutoff,
                    activation='gelu',
                    embedding_hidden_dimension=self.embedding_dim,
                    num_atom_features=self.n_atom_feats,
                    norm=self.graph_norm,
                    dropout=self.fc_dropout_probability,
                    spherical_embedding=self.add_spherical_basis,
                    torsional_embedding=self.add_torsional_basis,
                    radial_embedding=self.radial_function,
                    atom_embedding_dims=dataDims['atom embedding dict sizes'],
                    attention_heads=self.num_attention_heads,
                )
            else:
                print(self.graph_model + ' is not a valid graph model!!')
                sys.exit()
        else:
            self.graph_net = nn.Identity()
            self.graph_filters = 0  # no accounting for dime inputs or outputs
            self.pools = nn.Identity()

        # initialize global pooling operation
        if self.graph_model is not None:
            self.global_pool = global_aggregation(self.pooling, self.fc_depth)

        # molecule features FC layer
        if self.n_mol_feats != 0:
            self.mol_fc = nn.Linear(self.n_mol_feats, self.n_mol_feats)

        # FC model to post-process graph fingerprint
        self.gnn_mlp = general_MLP(layers=self.num_fc_layers,
                                   filters=self.fc_depth,
                                   norm=self.fc_norm_mode,
                                   dropout=self.fc_dropout_probability,
                                   input_dim=self.fc_depth,
                                   output_dim=self.fc_depth,
                                   conditioning_dim=self.n_mol_feats,
                                   seed=seed
                                   )

        self.output_fc = nn.Linear(self.fc_depth, self.output_classes, bias=False)

    def forward(self, data, return_latent=False, return_dists=False):
        extra_outputs = {}
        if self.graph_model is not None:
            x, dists_dict = self.graph_net(data.x[:, :self.n_atom_feats], data.pos, data.batch, ptr=data.ptr, ref_mol_inds=data.aux_ind, return_dists=return_dists)  # get atoms encoding
            if self.crystal_mode:  # model only outputs ref mol atoms - many fewer
                x = self.global_pool(x, data.pos, data.batch[torch.where(data.aux_ind == 0)[0]])
            else:
                x = self.global_pool(x, data.pos, data.batch)  # aggregate atoms to molecule

        mol_feats = self.mol_fc(data.x[data.ptr[:-1], -self.n_mol_feats:])  # molecule features are repeated, only need one per molecule (hence data.ptr)

        if self.graph_model is not None:
            x = self.gnn_mlp(x, conditions=mol_feats)  # mix graph fingerprint with molecule-scale features
        else:
            x = self.gnn_mlp(mol_feats)

        output = self.output_fc(x)

        if return_dists:
            if self.graph_model is not None:
                extra_outputs['dists dict'] = dists_dict
            else:
                extra_outputs['dists dict'] = None
        if return_latent:
            extra_outputs['latent'] = output.cpu().detach().numpy()

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output


class independent_gaussian_model(nn.Module):
    def __init__(self, input_dim, means, stds, normed_length_means, normed_length_stds, cov_mat=None):
        super(independent_gaussian_model, self).__init__()

        self.input_dim = input_dim
        fixed_norms = torch.Tensor(means)
        fixed_norms[:3] = torch.Tensor(normed_length_means)
        fixed_stds = torch.Tensor(stds)
        fixed_stds[:3] = torch.Tensor(normed_length_stds)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('stds', torch.Tensor(stds))
        self.register_buffer('fixed_norms', torch.Tensor(fixed_norms))
        self.register_buffer('fixed_stds', torch.Tensor(fixed_stds))

        if cov_mat is not None:
            pass
        else:
            cov_mat = torch.diag(torch.Tensor(fixed_stds).pow(2))

        fixed_means = means.copy()
        fixed_means[:3] = normed_length_means
        self.prior = MultivariateNormal(fixed_norms, torch.Tensor(cov_mat))  # apply standardization
        self.dummy_params = nn.Parameter(torch.ones(100))

    def forward(self, data, num_samples):
        '''
        sample comes out in non-standardized basis, but with normalized cell lengths
        so, denormalize cell length (multiply by Z^(1/3) and vol^(1/3)
        then standardize
        '''
        # conditions are unused - dummy
        # denormalize sample before standardizing
        samples = self.prior.sample((num_samples,)).to(data.x.device)
        samples[:, :3] = samples[:, :3] * (data.Z[:, None] ** (1 / 3)) * (data.mol_volume[:, None] ** (1 / 3))
        return (samples - self.means.to(samples.device)) / self.stds.to(samples.device)  # we want samples in standardized basis

    def backward(self, samples):
        return samples * self.stds + self.means

    def score(self, samples):
        return self.prior.log_prob(samples)


def crystal_rdf(crystaldata, rrange=[0, 10], bins=100, intermolecular=False, elementwise=False, raw_density=False, atomwise=False):
    '''
    compute the RDF for all the supercells in a CrystalData object
    without respect for atom type
    '''
    if crystaldata.aux_ind is not None:
        in_inds = torch.where(crystaldata.aux_ind == 0)[0]
        if intermolecular:
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(crystaldata.pos.device)
        else:
            out_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
    else:
        in_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
        out_inds = in_inds

    edges = asymmetric_radius_graph(crystaldata.pos,
                                    batch=crystaldata.batch,
                                    inside_inds=in_inds,
                                    convolve_inds=out_inds,
                                    r=max(rrange), max_num_neighbors=500, flow='source_to_target')

    crystal_number = crystaldata.batch[edges[0]]

    dists = (crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()

    assert not (elementwise and atomwise)

    if elementwise:
        if raw_density:
            density = torch.ones(crystaldata.num_graphs).to(dists.device)
        else:
            density = None
        relevant_elements = [5, 6, 7, 8, 9, 15, 16, 17, 35]
        element_symbols = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
        elements = [crystaldata.x[edges[0], 0], crystaldata.x[edges[1], 0]]
        rdfs_dict = {}
        rdfs_array = torch.zeros((crystaldata.num_graphs, int((len(relevant_elements) ** 2 + len(relevant_elements)) / 2), bins))
        ind = 0
        for i, element1 in enumerate(relevant_elements):
            for j, element2 in enumerate(relevant_elements):
                if j >= i:
                    rdfs_dict[ind] = element_symbols[element1] + ' to ' + element_symbols[element2]
                    rdfs_array[:, ind], rr = parallel_compute_rdf_torch([dists[(crystal_number == n) * (elements[0] == element1) * (elements[1] == element2)] for n in range(crystaldata.num_graphs)],
                                                                        rrange=rrange, bins=bins, density=density)
                    ind += 1
        return rdfs_array, rr, rdfs_dict
    elif atomwise:  # todo finish & test
        # generate atomwise indices
        rdfs_array_list = []
        rdfs_dict_list = []
        all_atom_inds = []
        for i in range(crystaldata.num_graphs):
            all_atom_inds.append(torch.arange(crystaldata.mol_size[i]).tile(int((crystaldata.batch == i).sum() // crystaldata.mol_size[i])))
        atom_inds = torch.cat(all_atom_inds)
        atoms = [atom_inds[edges[0]].to(crystaldata.x.device), atom_inds[edges[1]].to(crystaldata.x.device)]
        ind = 0
        for n in range(crystaldata.num_graphs):
            atom = []
            for i in range(int(crystaldata.mol_size[n])):
                for j in range(int(crystaldata.mol_size[n])):
                    if j >= i:
                        atom.append([i, j])
            atom = torch.Tensor(atom)
            rdfs_dict_list.append(atom)

            if raw_density:
                density = torch.ones(len(atom)).to(dists.device)
            else:
                density = None

            rdfs_array, rr = parallel_compute_rdf_torch([dists[(crystal_number == n) * (atoms[0] == atom[m, 0]) * (atoms[1] == atom[m, 1])]
                                                         for m in range(len(atom))],
                                                        rrange=rrange, bins=bins, density=density)
            ind += 1

            rdfs_array_list.append(rdfs_array)

        return rdfs_array_list, rr, rdfs_dict_list

    else:
        if raw_density:
            density = torch.ones(crystaldata.num_graphs).to(dists.device)
        else:
            density = None
        return parallel_compute_rdf_torch([dists[crystal_number == n] for n in range(crystaldata.num_graphs)], rrange=rrange, bins=bins, density=density)


def vdw_overlap(vdw_radii, dists=None, batch_numbers=None, atomic_numbers=None, num_graphs=None, crystaldata=None, return_atomwise=False):
    if crystaldata is not None:  # extract distances from the crystal
        if crystaldata.aux_ind is not None:
            in_inds = torch.where(crystaldata.aux_ind == 0)[0]
            # default to always intermolecular distances
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(crystaldata.pos.device)

        else:  # if we lack the info, just do it intramolecular
            in_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
            out_inds = in_inds

        '''
        compute all distances
        '''
        edges = asymmetric_radius_graph(crystaldata.pos,
                                        batch=crystaldata.batch,
                                        inside_inds=in_inds,
                                        convolve_inds=out_inds,
                                        r=6, max_num_neighbors=500, flow='source_to_target')  # max vdW range as six

        crystal_number = crystaldata.batch[edges[0]]

        dists = (crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()
        elements = [crystaldata.x[edges[0], 0].long().to(dists.device), crystaldata.x[edges[1], 0].long().to(dists.device)]
        num_graphs = crystaldata.num_graphs
    elif dists is not None:  # precomputed intermolecular crystal distances
        crystal_number = batch_numbers
        elements = atomic_numbers
        num_graphs = num_graphs

    else:
        assert False  # must do one or the other

    '''
    compute vdW radii respectfulness
    '''
    vdw_radii_vector = torch.Tensor(list(vdw_radii.values())).to(dists.device)
    atom_radii = [vdw_radii_vector[elements[0]], vdw_radii_vector[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]

    # penalties = torch.clip(torch.exp(-overlaps / radii_sums) - 1, min=0) / 1.71828  # strictly normed vdW loss
    penalties = F.relu(-(dists - radii_sums))  # only punish negatives (meaning overlaps)
    assert torch.sum(torch.isnan(penalties)) == 0

    scores = torch.nan_to_num(
        torch.stack(
            # [torch.mean(torch.topk(penalties[crystal_number == ii], 5)[0]) for ii in range(num_graphs)]
            [torch.max(penalties[crystal_number == ii]) if (len(penalties[crystal_number == ii]) > 0) else torch.zeros(1)[0].to(penalties.device) for ii in range(num_graphs)]
        )
    )
    mean_scores = torch.nan_to_num(
        torch.stack(
            [torch.mean(penalties[crystal_number == ii]) for ii in range(num_graphs)]
        )
    )
    tot_scores = (scores + mean_scores) / 2 # combine mean score with max score
    assert len(tot_scores) == num_graphs
    if return_atomwise:
        return tot_scores, [penalties[crystal_number == ii] for ii in range(num_graphs)]
    else:
        return tot_scores


def ase_mol_from_crystaldata(data, index=None, highlight_aux=False, exclusion_level=None, inclusion_distance=True):
    '''
    generate an ASE Atoms object from a crystaldata object

    from ase.visualize import view
    mols = [ase_mol_from_crystaldata(supercell_data,i,exclusion_level = 'convolve with', highlight_aux=True) for i in range(10)]
    view(mols)
    '''

    if data.batch is not None:  # more than one crystal in the datafile
        atom_inds = torch.where(data.batch == index)[0]
    else:
        atom_inds = torch.arange(len(data.x))

    if exclusion_level == 'ref only':
        inside_inds = torch.where(data.aux_ind == 0)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'convolve with':
        inside_inds = torch.where(data.aux_ind < 2)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'distance':
        crystal_coords = data.pos[atom_inds]
        crystal_inds = data.aux_ind[atom_inds]

        canonical_conformer_inds = torch.where(crystal_inds == 0)[0]
        mol_centroid = crystal_coords[canonical_conformer_inds].mean(0)
        mol_radius = torch.max(torch.cdist(mol_centroid[None], crystal_coords[canonical_conformer_inds], p=2))
        in_range_inds = torch.where((torch.cdist(mol_centroid[None], crystal_coords, p=2) < (mol_radius + inclusion_distance))[0])[0]
        atom_inds = atom_inds[in_range_inds]
        coords = crystal_coords[in_range_inds].cpu().detach().numpy()
    else:
        coords = data.pos[atom_inds].cpu().detach().numpy()

    if highlight_aux:  # highlight the atom aux index
        numbers = data.aux_ind[atom_inds].cpu().detach().numpy() + 6
    else:
        numbers = data.x[atom_inds, 0].cpu().detach().numpy()

    if index is not None:
        cell = data.T_fc[index].T.cpu().detach().numpy()
    else:
        cell = data.T_fc[0].T.cpu().detach().numpy()

    mol = Atoms(symbols=numbers, positions=coords, cell=cell)

    return mol

