from math import sqrt, pi as PI
import sys
import sympy as sym
import tqdm
from torch_geometric.nn.models.dimenet_utils import (bessel_basis,
                                                     associated_legendre_polynomials,
                                                     sph_harm_prefactor)
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_sparse import SparseTensor
import torch_geometric.nn as gnn
from models.asymmetric_radius_graph import asymmetric_radius_graph
from models.model_components import general_MLP


def real_sph_harm(k, zero_m_only=True, spherical_coordinates=True):
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, k):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(k, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(1, len(S_m)):  # todo mk range from 1
                S_m[i] = S_m[i].subs(x,
                                     sym.sin(theta) * sym.cos(phi)).subs(
                    y,
                    sym.sin(theta) * sym.sin(phi))
            for i in range(1, len(C_m)):  # todo mk range from 1
                C_m[i] = C_m[i].subs(x,
                                     sym.sin(theta) * sym.cos(phi)).subs(
                    y,
                    sym.sin(theta) * sym.sin(phi))

    Y_func_l_m = [['0'] * (2 * j + 1) for j in range(k)]
    for i in range(k):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, k):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


class MikesGraphNet(torch.nn.Module):
    def __init__(self, hidden_channels: int,
                 graph_convolution_filters: int,
                 graph_convolution: str,
                 out_channels: int,
                 num_blocks: int,
                 num_spherical: int,
                 num_radial: int,
                 atom_embedding_dims,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 embedding_hidden_dimension=5,
                 norm=None,
                 dropout=0,
                 radial_embedding='bessel',
                 spherical_embedding=True,
                 torsional_embedding=True,
                 num_atom_features=1,
                 attention_heads=1,
                 crystal_mode=False,
                 crystal_convolution_type=1,
                 ):
        super(MikesGraphNet, self).__init__()

        self.num_blocks = num_blocks
        self.spherical_embedding = spherical_embedding
        self.torsional_embedding = torsional_embedding
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.crystal_mode = crystal_mode
        self.convolution_mode = graph_convolution
        self.crystal_convolution_type = crystal_convolution_type

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)
        if spherical_embedding:
            self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        if torsional_embedding:
            self.tbf = TorsionalEmbedding(num_spherical, num_radial, cutoff, bessel_forms=self.sbf.bessel_forms)

        self.atom_embeddings = EmbeddingBlock(hidden_channels, atom_embedding_dims, num_atom_features, embedding_hidden_dimension, activation)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(graph_convolution_filters,
                    hidden_channels,
                    radial_dim=num_radial,
                    spherical_dim=num_spherical,
                    spherical=spherical_embedding,
                    torsional=torsional_embedding,
                    convolution_mode=graph_convolution,
                    norm=norm,
                    dropout=dropout,
                    heads=attention_heads,
                    )
            for _ in range(num_blocks)
        ])

        self.fc_blocks = torch.nn.ModuleList([
            FCBlock(hidden_channels, norm, dropout, activation)
            for _ in range(num_blocks)
        ])

        self.global_blocks = torch.nn.ModuleList([
            nn.Identity()
            for _ in range(num_blocks)
        ])
        # self.global_blocks = torch.nn.ModuleList([
        #     GlobalBlock(hidden_channels, graph_convolution_filters, norm, dropout, activation)
        #     for _ in range(num_blocks)
        # ])

        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def get_geom_embedding(self, edge_index, pos, num_nodes):
        '''
        compute elements for radial & spherical embeddings
        '''
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        sbf, tbf, idx_kj, idx_ji = None, None, None, None

        if torch.sum(dist == 0) > 0:
            zeros_at = torch.where(dist == 0) # add a little jitter, we absolutely cannot have zero distances
            pos[i[zeros_at]] += (torch.ones_like(pos[i[zeros_at]]) / 5)
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        assert torch.sum(dist == 0) == 0

        if self.spherical_embedding:
            i, j = edge_index

            # value = torch.arange(j.size(0), device=j.device)
            adj_t = SparseTensor(row=i, col=j, value=torch.arange(j.size(0), device=j.device),
                                 sparse_sizes=(num_nodes, num_nodes))
            adj_t_row = adj_t[j]
            num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

            # Node indices (k->j->i) for triplets.
            idx_i = i.repeat_interleave(num_triplets)
            idx_j = j.repeat_interleave(num_triplets)
            idx_k = adj_t_row.storage.col()
            mask = idx_i != idx_k  # Remove i == k triplets.
            idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

            # Edge indices (k-j, j->i) for triplets.
            idx_kj = adj_t_row.storage.value()[mask]
            idx_ji = adj_t_row.storage.row()[mask]

            # Calculate angles. 0 to pi
            pos_ji = pos[idx_i] - pos[idx_j]
            pos_jk = pos[idx_k] - pos[idx_j]
            # a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
            # b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
            angle = torch.atan2(torch.cross(pos_ji, pos_jk).norm(dim=-1), (pos_ji * pos_jk).sum(dim=-1))

            sbf = self.sbf(dist, angle, idx_kj)

            if self.torsional_embedding:
                idx_batch = torch.arange(len(idx_i), device=angle.device)
                idx_k_n = adj_t[idx_j].storage.col()
                repeat = num_triplets
                num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
                idx_i_t = idx_i.repeat_interleave(num_triplets_t)
                idx_j_t = idx_j.repeat_interleave(num_triplets_t)
                idx_k_t = idx_k.repeat_interleave(num_triplets_t)
                idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
                mask = idx_i_t != idx_k_n
                idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

                pos_j0 = pos[idx_k_t] - pos[idx_j_t]
                pos_ji = pos[idx_i_t] - pos[idx_j_t]
                pos_jk = pos[idx_k_n] - pos[idx_j_t]
                # dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
                plane1 = torch.cross(pos_ji, pos_j0)
                plane2 = torch.cross(pos_ji, pos_jk)
                a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
                b = torch.cross(plane1, plane2).norm(dim=-1)  # sin_angle * |plane1| * |plane2|
                torsion1 = torch.atan2(b, a)  # -pi to pi
                torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
                torsion = scatter(torsion1, idx_batch_t, reduce='min')
                tbf = self.tbf(dist, angle, torsion, idx_kj)
            else:
                tbf = None

        return dist, self.rbf(dist), sbf, tbf, idx_kj, idx_ji

    def forward(self, z, pos, batch, ptr, ref_mol_inds=None, return_dists=False, n_repeats=None):
        """"""
        if self.crystal_mode:
            inside_inds = torch.where(ref_mol_inds == 0)[0]
            outside_inds = torch.where(ref_mol_inds == 1)[0]  # atoms which are not in the asymmetric unit but which we will convolve - pre-excluding many from outside the cutoff
            inside_batch = batch[inside_inds]  # get the feature vectors we want to repeat
            n_repeats = [int(torch.sum(batch == ii) / torch.sum(inside_batch == ii)) for ii in range(len(ptr) - 1)]
            # intramolecular edges
            edge_index = asymmetric_radius_graph(pos, batch=batch, r=self.cutoff,  # intramolecular interactions - stack over range 3 convolutions
                                                 max_num_neighbors=self.max_num_neighbors, flow='source_to_target',
                                                 inside_inds=inside_inds, convolve_inds=inside_inds)
            # intermolecular edges
            edge_index_inter = asymmetric_radius_graph(pos, batch=batch, r=self.cutoff,  # extra radius for intermolecular graph convolution
                                                       max_num_neighbors=self.max_num_neighbors, flow='source_to_target',
                                                       inside_inds=inside_inds, convolve_inds=outside_inds)

            if self.crystal_convolution_type == 1:
                edge_index = torch.cat((edge_index, edge_index_inter), dim=1)

        else:
            edge_index = gnn.radius_graph(pos, r=self.cutoff, batch=batch,
                                          max_num_neighbors=self.max_num_neighbors, flow='source_to_target')

        dist, rbf, sbf, tbf, idx_kj, idx_ji = self.get_geom_embedding(edge_index, pos, num_nodes=len(z))

        '''
        import networkx as nx
        import matplotlib.pyplot as plt
        intra_edges = (edge_index[:, edge_index[0, :] < ptr[1]].cpu().detach().numpy().T)
        inter_edges = (edge_index_inter[:, edge_index_inter[0, :] < ptr[1]].cpu().detach().numpy().T)
        plt.clf()
        G = nx.Graph()
        G = G.to_directed()
        G.add_weighted_edges_from(np.concatenate((intra_edges, np.ones(len(intra_edges))[:, None] * 2), axis=1))
        G.add_weighted_edges_from(np.concatenate((inter_edges, np.ones(len(inter_edges))[:, None] * 0.25), axis=1))
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        node_weights = np.concatenate((np.ones(9)*2, np.ones(len(G.nodes)-9)))
        nx.draw_kamada_kawai(G, arrows=True, node_size=node_weights * 100, edge_color=weights, linewidths = 1, width=weights, 
        edge_cmap=plt.cm.RdYlGn, node_color = node_weights, cmap=plt.cm.RdYlGn)
        '''

        # graph model starts here
        x = self.atom_embeddings(z)  # embed atomic numbers & compute initial atom-wise feature vector
        for n, (convolution, fc, global_agg) in enumerate(zip(self.interaction_blocks, self.fc_blocks, self.global_blocks)):
            if self.crystal_mode:
                if n < (self.num_blocks - 1):  # to do this molecule-wise, we need to multiply n_repeats by Z for each crystal
                    x = x + convolution(x, rbf, dist, edge_index, sbf=sbf, tbf=tbf, idx_kj=idx_kj, idx_ji=idx_ji)  # graph convolution
                    x[inside_inds] = x[inside_inds] + fc(x[inside_inds])  # feature-wise 1D convolution on only relevant atoms
                    for ii in range(len(ptr) - 1):  # for each crystal
                        x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)  # copy the first unit cell to all periodic images

                else:  # on the final convolutional block, do not broadcast the reference cell, and include intermolecular interactions
                    dist_inter, rbf_inter, sbf_inter, tbf_inter, idx_kj_inter, idx_ji_inter = \
                        self.get_geom_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos, num_nodes=len(z))  # compute for tracking
                    if self.crystal_convolution_type == 2:
                        x = convolution(x, rbf_inter, dist_inter, torch.cat((edge_index, edge_index_inter), dim=1),
                                        sbf=sbf_inter, tbf=tbf_inter, idx_kj=idx_kj_inter, idx_ji=idx_ji_inter)  # return only the results of the intermolecular convolution, omitting intermolecular features
                    elif self.crystal_convolution_type == 1:
                        x = x + convolution(x, rbf, dist, edge_index, sbf=sbf, idx_kj=idx_kj, idx_ji=idx_ji)  # standard graph convolution

                    x = x[inside_inds] + fc(x[inside_inds])  # feature-wise 1D convolution on only relevant atoms, and return only those atoms

            else:
                x = x + convolution(x, rbf, dist, edge_index, sbf=sbf, idx_kj=idx_kj, idx_ji=idx_ji)  # graph convolution
                x = x + fc(x)  # feature-wise 1D convolution

        if return_dists:  # return dists, batch #, and inside/outside identity
            dist_output = {}
            dist_output['intramolecular dist'] = dist
            dist_output['intramolecular dist batch'] = batch[edge_index[0]]
            if self.crystal_mode:
                dist_output['intermolecular dist'] = dist_inter
                dist_output['intermolecular dist batch'] = batch[edge_index_inter[0]]

        out = self.output_layer(x)
        assert torch.sum(torch.isnan(out)) == 0

        return out, dist_output if return_dists else None


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in tqdm.tqdm(range(num_spherical)):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

        self.bessel_forms = bessel_forms  # saving this because it takes forever to compute

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class TorsionalEmbedding(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5, bessel_forms=None):
        super(TorsionalEmbedding, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical  #
        self.num_radial = num_radial
        self.cutoff = cutoff
        # self.envelope = Envelope(envelope_exponent)
        if bessel_forms is None:
            bessel_forms = bessel_basis(num_spherical, num_radial)
        #        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical, zero_m_only=False)
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        phi = sym.symbols('phi')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(self.num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta, phi], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(lambda x, y: torch.zeros_like(x) + torch.zeros_like(y) + sph1(0, 0))  # torch.zeros_like(x) + torch.zeros_like(y)
            else:
                for k in range(-i, i + 1):
                    sph = sym.lambdify([theta, phi], sph_harm_forms[i][k + i], modules)
                    self.sph_funcs.append(sph)
            for j in range(self.num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, phi, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        cbf = torch.stack([f(angle, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, 1, n, k) * cbf.view(-1, n, n, 1)).view(-1, n * n * k)
        return out


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        # self.freq = torch.arange(1,self.freq.numel() + 1).mul(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class GaussianEmbedding(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianEmbedding, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, embedding_size, num_atom_features, embedding_dimension, activation='gelu'):
        super(EmbeddingBlock, self).__init__()
        self.num_embeddings = 1
        for key in embedding_size.keys():
            self.embeddings = nn.Embedding(int(embedding_size[key]), embedding_dimension)
        self.linear = nn.Linear(embedding_dimension + num_atom_features - self.num_embeddings, hidden_channels)
        self.activation = Activation(activation, hidden_channels)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(x[:, 0].long())
        concat_vec = torch.cat([embedding, x[:, self.num_embeddings:]], dim=-1)

        return self.activation(self.linear(concat_vec))


class GCBlock(torch.nn.Module):
    def __init__(self, graph_convolution_filters, hidden_channels, radial_dim, convolution_mode, spherical_dim=None, spherical=False, torsional=False, norm=None, dropout=0, heads=1):
        super(GCBlock, self).__init__()
        self.norm = Normalization(norm, graph_convolution_filters)
        self.node_to_message = nn.Linear(hidden_channels, graph_convolution_filters)
        self.message_to_node = nn.Linear(graph_convolution_filters, hidden_channels, bias=False)  # don't want to send spurious messages, though it probably doesn't matter anyway
        self.radial_to_message = nn.Linear(radial_dim, graph_convolution_filters)
        self.convolution_mode = convolution_mode

        if spherical:  # need more linear layers to aggregate angular information to radial
            assert spherical_dim is not None, "Spherical information must have a dimension != 0 for spherical message aggregation"
            self.spherical_to_message = nn.Linear(radial_dim * spherical_dim, graph_convolution_filters)
            self.radial_spherical_aggregation = nn.Linear(graph_convolution_filters * 2, graph_convolution_filters)  # torch.add  # could also do dot
        if torsional:
            assert spherical
            self.torsional_to_message = nn.Linear(spherical_dim * spherical_dim * radial_dim, graph_convolution_filters)
            self.radial_spherical_torsional_aggregation = nn.Linear(graph_convolution_filters * 3, graph_convolution_filters)  # torch.add  # could also do dot

        if convolution_mode == 'self attention':
            self.GConv = gnn.GATv2Conv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=graph_convolution_filters,
            )
        elif convolution_mode == 'full message passing':
            self.GConv = MPConv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters,
                edge_dim=graph_convolution_filters,
                dropout = dropout,
                norm = norm,
            )
        elif convolution_mode.lower() == 'schnet':  #
            assert not spherical, 'schnet currently only works with pure radial bases'
            self.GConv = CFConv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters,
                num_filters=graph_convolution_filters,
                cutoff=5
                , )

    def forward(self, x, rbf, dists, edge_index, sbf=None, tbf=None, idx_kj=None, idx_ji=None):
        # convert local information into edge weights
        if tbf is not None:
            # aggregate spherical and torsional messages to radial
            edge_attr = self.radial_spherical_torsional_aggregation(
                torch.cat((self.radial_to_message(rbf)[idx_kj], self.spherical_to_message(sbf), self.torsional_to_message(tbf)), dim=1))  # combine radial and spherical info in triplet space
            # torch.sum(torch.stack((self.radial_to_message(rbf)[idx_kj], self.spherical_to_message(sbf), self.torsional_to_message(tbf))),dim=0)
            edge_attr = scatter(edge_attr, idx_ji, dim=0)  # collect triplets back down to pair space

        elif sbf is not None:
            # aggregate spherical messages to radial
            # rbf = self.radial_to_message(rbf)
            # sbf = self.spherical_message(sbf)
            edge_attr = self.radial_spherical_aggregation(torch.cat((self.radial_to_message(rbf)[idx_kj], self.spherical_to_message(sbf)), dim=1))  # combine radial and spherical info in triplet space
            edge_attr = scatter(edge_attr, idx_ji, dim=0)  # collect triplets back down to pair space

        else:  # no angular information
            edge_attr = self.radial_to_message(rbf)

        # convolve # todo only update nodes which will actually pass messages on this round
        x = self.norm(self.node_to_message(x))
        if self.convolution_mode.lower() == 'schnet':
            x = self.GConv(x, edge_index, dists, edge_attr)
        else:
            x = self.GConv(x, edge_index, edge_attr)

        return self.message_to_node(x)


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(gnn.MessagePassing):
    '''
    ~~the graph convolution used in the popular SchNet~~
    '''

    def __init__(self, in_channels, out_channels, num_filters, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)  # de-weight distant nodes
        # W = self.nn(edge_attr) * C.view(-1, 1)
        W = edge_attr * C.view(-1, 1)  # in my method, edge_attr are pre-featurized

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)

        return x

    def message(self, x_j, W):
        return x_j * W


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, dropout = 0, norm = None, activation='leaky relu'):
        super(MPConv, self).__init__()

        self.MLP = general_MLP(layers = 4,
                               filters = out_channels,
                               input_dim = in_channels * 2 + edge_dim,
                               dropout = dropout,
                               norm = norm,
                               output_dim = out_channels,
                               activation=activation,
                               )
        #self.linear1 = nn.Linear(in_channels * 2 + edge_dim, out_channels)
        #self.linear2 = nn.Linear(out_channels, out_channels)
        #self.norm = nn.LayerNorm(out_channels)
        #self.activation = Activation(activation, filters=None)

    def forward(self, x, edge_index, edge_attr):
        # i, j = edge_index
        # m = self.linear2(self.activation(self.norm(self.linear1(torch.cat((x[i], x[j], edge_attr), dim=-1)))))
        # return scatter(m, j, dim=0, dim_size=len(x))  # send directional messages from i to j, enforcing the size of the output dimension

        #m = self.linear2(self.activation(self.norm(self.linear1(torch.cat((x[edge_index[0]], x[edge_index[1]], edge_attr), dim=-1)))))
        # m = self.linear1(torch.cat((x[i], x[j], edge_attr), dim=-1))

        m = self.MLP(torch.cat((x[edge_index[0]], x[edge_index[1]], edge_attr), dim=-1))

        return scatter(m, edge_index[1], dim=0, dim_size=len(x))  # send directional messages from i to j, enforcing the size of the output dimension


class FCBlock(torch.nn.Module):
    '''
    fully-connected block, following the original transformer architecture with norm first
    '''

    def __init__(self, hidden_channels, norm, dropout, activation):
        super(FCBlock, self).__init__()
        self.norm = Normalization(norm, hidden_channels)
        self.activation = Activation(activation, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))


class GlobalBlock(torch.nn.Module):
    '''
    fully-connected block, following the original transformer architecture with norm first
    '''

    def __init__(self, hidden_channels, graph_convolution_filters, norm, dropout, activation):
        super(GlobalBlock, self).__init__()
        self.global_pool = gnn.GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(graph_convolution_filters, graph_convolution_filters), nn.GELU(), nn.Linear(graph_convolution_filters, 1)),
        )
        self.norm = Normalization(norm, hidden_channels)
        self.activation = Activation(activation, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(hidden_channels, graph_convolution_filters)
        self.linear2 = nn.Linear(graph_convolution_filters + graph_convolution_filters, hidden_channels)

    def forward(self, x, batch):
        x = self.dropout(self.activation(self.linear1(self.norm(x))))  # scale to graph conv dimension
        global_embedding = self.global_pool(x, batch)  # attention aggregation

        return self.linear2(torch.cat((x, global_embedding[batch]), dim=-1))  # expand global to local basis, concatenate and forward


class kernelActivation(nn.Module):  # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[-2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1, 1), groups=int(channels), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x


def triplets(edge_index, num_nodes):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


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


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func.lower() == 'relu':
            self.activation = F.relu
        elif activation_func.lower() == 'gelu':
            self.activation = F.gelu
        elif activation_func.lower() == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)
        elif activation_func.lower() == 'leaky relu':
            self.activation = F.leaky_relu

    def forward(self, input):
        return self.activation(input)
