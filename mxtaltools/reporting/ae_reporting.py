import numpy as np
import torch
import wandb
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from torch.nn import functional as F
from torch_scatter import scatter_softmax

from mxtaltools.models.functions.asymmetric_radius_graph import radius
from mxtaltools.models.autoencoder_utils import compute_gaussian_overlap, compute_type_evaluation_overlap, \
    compute_coord_evaluation_overlap, compute_full_evaluation_overlap, get_node_weights, decoding2mol_batch, \
    batch_rmsd


def autoencoder_evaluation_overlaps(data, decoded_data, config, dataDims):
    # if len(epoch_stats_dict['sample']) > 1:
    #     print("more than one batch of AE samples were saved but only the first is being analyzed")

    # compute overlaps with evaluation settings
    true_nodes = F.one_hot(data.x.long(), num_classes=dataDims['num_atom_types']).float()
    full_overlap, self_overlap = compute_full_evaluation_overlap(data,
                                                                 decoded_data,
                                                                 true_nodes,
                                                                 sigma=config.autoencoder.evaluation_sigma,
                                                                 distance_scaling=config.autoencoder.type_distance_scaling
                                                                 )
    coord_overlap, self_coord_overlap = compute_coord_evaluation_overlap(config, data, decoded_data,
                                                                         true_nodes)
    self_type_overlap, type_overlap = compute_type_evaluation_overlap(config, data, dataDims['num_atom_types'],
                                                                      decoded_data, true_nodes)

    return coord_overlap, full_overlap, self_coord_overlap, self_overlap, self_type_overlap, type_overlap


def gaussian_3d_overlap_plot(mol_batch, decoded_mol_batch, atom_types, graph_ind):
    # 3D swarm fig
    fig = swarm_vs_tgt_fig(mol_batch, decoded_mol_batch, atom_types, graph_ind=graph_ind)

    return fig


def decoder_agglomerative_clustering(points_pred, sample_weights, intrapoint_cutoff):
    ag = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='complete',
                                 distance_threshold=intrapoint_cutoff).fit_predict(points_pred[:, :3])
    n_clusters = len(np.unique(ag))
    pred_particle_weights = np.zeros(n_clusters)
    pred_particles = np.zeros((n_clusters, points_pred.shape[1]))
    for ind in range(n_clusters):
        collect_inds = ag == ind
        collected_particles = points_pred[collect_inds]
        collected_particle_weights = sample_weights[collect_inds]
        pred_particle_weights[ind] = collected_particle_weights.sum()
        pred_particles[ind] = np.sum(collected_particle_weights[:, None] * collected_particles, axis=0)

    '''aggregate any 'floaters' or low-probability particles into their nearest neighbors'''
    single_node_weight = np.amax(
        pred_particle_weights)  # the largest particle weight as set as the norm against which to measure other particles
    # if there is a double particle, this will fail - but then the decoding is bad anyway
    weak_particles = np.argwhere(pred_particle_weights < single_node_weight / 2).flatten()
    ind = 0
    while len(weak_particles >= 1):
        particle = pred_particles[weak_particles[ind], :3]
        dmat = cdist(particle[None, :], pred_particles[:, :3])
        dmat[dmat == 0] = 100
        nearest_neighbor = np.argmin(dmat)
        pred_particles[nearest_neighbor] = pred_particles[nearest_neighbor] * pred_particle_weights[nearest_neighbor] + \
                                           pred_particles[weak_particles[ind]] * pred_particle_weights[
                                               weak_particles[ind]]
        pred_particle_weights[nearest_neighbor] = pred_particle_weights[nearest_neighbor] + pred_particle_weights[
            weak_particles[ind]]
        pred_particles = np.delete(pred_particles, weak_particles[ind], axis=0)
        pred_particle_weights = np.delete(pred_particle_weights, weak_particles[ind], axis=0)
        weak_particles = np.argwhere(pred_particle_weights < 0.5).flatten()

    #
    # if len(pred_particles) == int(torch.sum(data.batch == graph_ind)):
    #     matched_dists = cdist(pred_particles[:, :3], coords_true)
    #     matched_inds = np.argmin(matched_dists, axis=1)[:, None]
    #     if len(np.unique(matched_inds)) < len(pred_particles):
    #         rmsd = np.Inf
    #         max_dist = np.Inf
    #     else:
    #         rmsd = np.mean(np.amin(matched_dists, axis=1))
    #         max_dist = np.amax(np.amin(matched_dists, axis=1))
    #
    # else:
    #     rmsd = np.Inf
    #     max_dist = np.Inf

    return pred_particles, pred_particle_weights  # , rmsd, max_dist # todo add flags around unequal weights


def scaffolded_decoder_clustering(graph_ind, data, decoded_data, num_classes,
                                  return_fig=True):  # todo parallelize over samples
    assert False, "This has been obviated in gaussian_3d_overlap_plots"
    (pred_particles, pred_particle_weights, points_true) = (
        decoder_scaffolded_clustering(data, decoded_data, graph_ind, num_classes))

    matched_particles, max_dist, rmsd, match_successful = compute_point_cloud_rmsd(points_true, pred_particle_weights,
                                                                                   pred_particles)

    if return_fig:
        fig2 = swarm_cluster_fig(data,
                                 graph_ind,
                                 matched_particles,
                                 pred_particle_weights,
                                 points_true)

        return rmsd, max_dist, pred_particle_weights.mean(), match_successful, fig2
    else:
        return rmsd, max_dist, pred_particle_weights.mean(), match_successful


def decoder_scaffolded_clustering(data, decoded_data, graph_ind, num_classes):
    """"""
    '''extract true and predicted points'''
    coords_true, coords_pred, points_true, points_pred, sample_weights = (
        extract_true_and_predicted_points(data, decoded_data, graph_ind, num_classes,
                                          to_numpy=True))

    '''get minimum true bond lengths'''
    intra_distmat = cdist(points_true, points_true) + np.eye(len(points_true)) * 10
    intrapoint_cutoff = np.amin(intra_distmat) / 2

    '''assign output density mass to each input particle (scaffolded clustering)'''
    distmat = cdist(points_true, points_pred)
    pred_particle_weights = np.zeros(len(distmat))
    pred_particles = np.zeros((len(distmat), points_pred.shape[1]))
    for ind in range(len(distmat)):  # for each particle in the scaffold
        # collect decoder points within the cartesian cutoff range
        collect_inds = np.argwhere(distmat[ind] < intrapoint_cutoff)[:, 0]
        # get the relevant 3+n_class dim points
        collected_particles = points_pred[collect_inds]
        # get the relevant swarm weights
        collected_particle_weights = sample_weights[collect_inds]
        # sum the probability mass within the cutoff of this scaffold particle
        pred_particle_weights[ind] = collected_particle_weights.sum()
        # aggregate the nearby swarm into a single synthetic particle
        pred_particles[ind] = np.sum(
            collected_particle_weights[:, None] * collected_particles,
            axis=0
        )

    return pred_particles, pred_particle_weights, points_true


def compute_point_cloud_rmsd(points_true, pred_particle_weights, pred_particles, weight_threshold=0.5):
    """get distances to true and predicted particles"""

    # confirm we have the same number of predicted and scaffold particles, and that each predicted particle
    # is closer to its scaffold point than any other particle
    dists = cdist(pred_particles, points_true)
    matched_particle_inds = np.argmin(dists, axis=0)
    all_targets_matched = len(np.unique(matched_particle_inds)) == len(points_true)
    matched_particles = pred_particles[matched_particle_inds]

    matched_dists = np.linalg.norm(points_true[:, :3] - matched_particles[:, :3], axis=1)
    if all_targets_matched and np.amax(
            np.abs(1 - pred_particle_weights)) < weight_threshold:  # +/- X% of 100% probability mass on site
        match_successful = True
    else:
        match_successful = False
    rmsd = matched_dists.mean()
    max_dist = matched_dists.max()
    return matched_particles, max_dist, rmsd, match_successful


def extract_true_and_predicted_points(mol_batch,
                                      decoded_mol_batch,
                                      graph_ind,
                                      num_classes,
                                      to_numpy=False):
    if mol_batch.x.ndim > 1:
        mol_batch.x = mol_batch.x[:, 0]
    coords_true = mol_batch.pos[mol_batch.batch == graph_ind]
    coords_pred = decoded_mol_batch.pos[decoded_mol_batch.batch == graph_ind]
    points_true = torch.cat(
        [coords_true, F.one_hot(mol_batch.x[mol_batch.batch == graph_ind].long(), num_classes=num_classes)], dim=1)
    points_pred = torch.cat([coords_pred, decoded_mol_batch.x[decoded_mol_batch.batch == graph_ind]], dim=1)
    sample_weights = decoded_mol_batch.aux_ind[decoded_mol_batch.batch == graph_ind]

    if to_numpy:
        return (coords_true.cpu().detach().numpy(), coords_pred.cpu().detach().numpy(),
                points_true.cpu().detach().numpy(), points_pred.cpu().detach().numpy(),
                sample_weights.cpu().detach().numpy())
    else:
        return coords_true, coords_pred, points_true, points_pred, sample_weights


def swarm_cluster_fig(atom_types,
                      graph_ind,
                      pred_particles,
                      pred_particle_weights,
                      points_true):
    fig2 = go.Figure()
    colors = (
        'rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)',
        'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)',
        'rgb(165, 170, 153)')
    for j in range(len(points_true)):
        vec = np.stack([pred_particles[j, :3], points_true[j, :3]])
        fig2.add_trace(
            go.Scatter3d(x=vec[:, 0],
                         y=vec[:, 1],
                         z=vec[:, 2],
                         mode='lines',
                         line_color='black',
                         showlegend=False))

    for j in range(points_true.shape[-1] - 3):
        ref_type_inds = torch.argwhere(atom_types == j)[:, 0].cpu().detach().numpy()
        pred_type_inds = np.argwhere(pred_particles[:, 3:].argmax(1) == j)[:, 0]
        fig2.add_trace(go.Scatter3d(x=points_true[ref_type_inds][:, 0], y=points_true[ref_type_inds][:, 1],
                                    z=points_true[ref_type_inds][:, 2],
                                    mode='markers', marker_color=colors[j], marker_size=12, marker_line_width=8,
                                    marker_line_color='black',
                                    opacity=0.6,
                                    showlegend=True if (j == 0 and graph_ind == 0) else False,
                                    name=f'True type', legendgroup=f'True type'
                                    ))
        fig2.add_trace(go.Scatter3d(x=pred_particles[pred_type_inds][:, 0], y=pred_particles[pred_type_inds][:, 1],
                                    z=pred_particles[pred_type_inds][:, 2],
                                    mode='markers', marker_color=colors[j], marker_size=6, marker_line_width=2,
                                    marker_line_color='white', showlegend=True,
                                    marker_symbol='diamond',
                                    name=f'Predicted type {j}'))
    fig2.update_layout(title=f"Overlapping Mass {pred_particle_weights.mean():.3f}")
    return fig2


def swarm_vs_tgt_fig(data, decoded_data, atom_types = [1,6,7,8,9], graph_ind=0):
    cmax = 1
    fig = go.Figure()  # scatter all the true & predicted points, colorweighted by atom type
    colors = ['rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)',
              'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)',
              'rgb(165, 170, 153)'] * 10
    colorscales = [[[0, 'rgba(0, 0, 0, 0)'], [1, color]] for color in colors]
    points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
    points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()
    pred_weights = decoded_data.aux_ind[decoded_data.batch == graph_ind]
    for j in range(len(atom_types)):
        ref_type_inds = torch.argwhere(data.z[data.batch == graph_ind] == atom_types[j])[:, 0].cpu().detach().numpy()

        pred_type_weights = (pred_weights * decoded_data.x[decoded_data.batch == graph_ind, j]).cpu().detach().numpy()

        fig.add_trace(go.Scatter3d(x=points_true[ref_type_inds][:, 0],
                                   y=points_true[ref_type_inds][:, 1],
                                   z=points_true[ref_type_inds][:, 2],
                                   mode='markers', marker_color=colors[j], marker_size=10, marker_line_width=7,
                                   marker_line_color='black',
                                   showlegend=True if j == 0 else False,
                                   name=f'True type', legendgroup=f'True type'
                                   ))

        fig.add_trace(go.Scatter3d(x=points_pred[:, 0],
                                   y=points_pred[:, 1],
                                   z=points_pred[:, 2],
                                   mode='markers',
                                   marker=dict(size=10, color=pred_type_weights, colorscale=colorscales[j], cmax=cmax,
                                               cmin=0),
                                   opacity=1, marker_line_color='white',
                                   showlegend=True,
                                   name=f'Predicted type {j}'
                                   ))

    '''add bonds'''
    distmat = cdist(points_true, points_true) + np.eye(len(points_true)) * 100
    from mxtaltools.constants.atom_properties import VDW_RADII
    atom_radii = np.array([VDW_RADII[elem] for elem in data.z[data.batch == graph_ind].cpu().numpy().tolist()])
    max_bond_lengths = (atom_radii[:, None] + atom_radii[None, :]) / 1.65
    edges = np.stack(np.where(distmat < max_bond_lengths))
    for e_ind, edge in enumerate(edges.T):
        points = np.vstack([points_true[edge[0]], points_true[edge[1]]])
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='lines',
            marker=dict(size=0, color='black'),
            line=dict(color='black', width=4),
            showlegend=False,
            name='True type',
            legendgroup='True type',
        ))

    min_val = np.amin(points_true) - 1
    max_val = np.amax(points_true) + 1
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[min_val, max_val]),
            yaxis=dict(range=[min_val, max_val]),
            zaxis=dict(range=[min_val, max_val]),
            aspectmode='manual',  # Necessary for fixed aspect ratios
            aspectratio=dict(x=1, y=1, z=1)
        )
    )

    return fig


def autoencoder_embedding_map(stats_dict, max_num_samples=1000000):
    """

    """
    ''' # spherical coordinates analysis
    def to_spherical(x, y, z):
        """Converts a cartesian coordinate (x, y, z) into a spherical one (radius, theta, phi)."""
        theta = np.arctan2(np.sqrt(x * x + y * y), z)
        phi = np.arctan2(y, x)
        return theta, phi
    
    
    def direction_coefficient(v):
        """
        norm vectors
        take inner product
        sum the gaussian-weighted dot product components
        """
        norms = np.linalg.norm(v, axis=1)
        nv = v / (norms[:, None, :] + 1e-3)
        dp = np.einsum('nik,nil->nkl', nv, nv)
    
        return np.exp(-(1 - dp) ** 2).mean(-1)
    
    theta, phi = to_spherical(vector_encodings[:, 0, :],vector_encodings[:, 1, :],vector_encodings[:, 2, :],)
    fig = make_subplots(rows=4, cols=4)
    for ind in range(16):
        fig.add_histogram2d(x=theta[ind], y=phi[ind],nbinsx=50, nbinsy=50, row=ind // 4 + 1, col = ind % 4 + 1)
    fig.show(renderer='browser')
    
    coeffs = []
    for ind in range(100):
        coeffs.append(direction_coefficient(vector_encodings[ind * 100: (ind+1) * 100]))
    coeffs = np.concatenate(coeffs)
    
    fig = make_subplots(rows=4, cols=4)
    for ind in range(16):
        fig.add_histogram2d(x=coeffs[ind], y=np.log10(np.linalg.norm(vector_encodings[ind], axis=0)) ,nbinsx=50, nbinsy=50, row=ind // 4 + 1, col = ind % 4 + 1)
    fig.show(renderer='browser')
    '''

    """embedding analysis"""
    scalar_encodings = stats_dict['scalar_encoding'][:max_num_samples]

    import umap

    reducer = umap.UMAP(n_components=2,
                        metric='cosine',
                        n_neighbors=100,
                        min_dist=0.01)

    embedding = reducer.fit_transform((scalar_encodings - scalar_encodings.mean()) / scalar_encodings.std())
    if 'principal_inertial_moments' in stats_dict.keys():
        stats_dict['molecule_principal_moment1'] = stats_dict['principal_inertial_moments'][:, 0]
        stats_dict['molecule_principal_moment2'] = stats_dict['principal_inertial_moments'][:, 2]
        stats_dict['molecule_principal_moment3'] = stats_dict['principal_inertial_moments'][:, 2]
        stats_dict['molecule_Ip_ratio1'] = stats_dict['principal_inertial_moments'][:, 0] / stats_dict[
                                                                                                'principal_inertial_moments'][
                                                                                            :, 1]
        stats_dict['molecule_Ip_ratio2'] = stats_dict['principal_inertial_moments'][:, 0] / stats_dict[
                                                                                                'principal_inertial_moments'][
                                                                                            :, 2]
        stats_dict['molecule_Ip_ratio3'] = stats_dict['principal_inertial_moments'][:, 1] / stats_dict[
                                                                                                'principal_inertial_moments'][
                                                                                            :, 2]
        mol_keys = [
            'molecule_num_atoms', 'molecule_volume', 'molecule_mass',
            'molecule_C_fraction', 'molecule_N_fraction', 'molecule_O_fraction',
            'molecule_principal_moment1', 'molecule_principal_moment2', 'molecule_principal_moment3',
            'molecule_Ip_ratio1', 'molecule_Ip_ratio2', 'molecule_Ip_ratio3'
        ]
        n_rows = 5
    else:
        mol_keys = [
            'molecule_num_atoms', 'molecule_volume', 'molecule_mass',
            'molecule_C_fraction', 'molecule_N_fraction', 'molecule_O_fraction',
            'evaluation_overlap', 'evaluation_coord_overlap', 'evaluation_type_overlap',
        ]
        n_rows = 3

    latents_keys = mol_keys + ['evaluation_overlap', 'evaluation_coord_overlap', 'evaluation_type_overlap']

    fig = make_subplots(cols=3, rows=n_rows, subplot_titles=latents_keys, horizontal_spacing=0.05,
                        vertical_spacing=0.05)
    for ind, mol_key in enumerate(latents_keys):
        try:
            mol_feat = np.concatenate(stats_dict[mol_key])[:max_num_samples]
        except:
            mol_feat = stats_dict[mol_key][:max_num_samples]

        if 'overlap' in mol_key:
            mol_feat = np.log10(np.abs(1 - mol_feat))

        normed_mol_feat = mol_feat - mol_feat.min()
        normed_mol_feat /= normed_mol_feat.max()
        row = ind // 3 + 1
        col = ind % 3 + 1
        fig.add_trace(go.Scattergl(x=embedding[:, 0], y=embedding[:, 1],
                                   mode='markers',
                                   marker_color=normed_mol_feat,
                                   opacity=.1,
                                   marker_colorbar=dict(title=mol_key),
                                   marker_cmin=0, marker_cmax=1, marker_showscale=False,
                                   showlegend=False,
                                   ),
                      row=row, col=col)
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                          xaxis_showticklabels=False, yaxis_showticklabels=False,
                          plot_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(linecolor='black', mirror=True)  # , gridcolor='grey', zerolinecolor='grey')
        fig.update_xaxes(linecolor='black', mirror=True)  # , gridcolor='grey', zerolinecolor='grey')
        fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))

    fig.show(renderer='browser')

    fig.write_image('embedding.png', width=1080, height=1080)
    wandb.log({'Latent Embedding Analysis': wandb.Image('embedding.png')})

    """overlap correlates"""
    score = np.concatenate(stats_dict['evaluation_overlap'])
    correlates_dict = {}
    for mol_key in mol_keys:
        try:
            mol_feat = np.concatenate(stats_dict[mol_key])[:max_num_samples]
        except:
            mol_feat = stats_dict[mol_key][:max_num_samples]
        coeff = np.corrcoef(score, mol_feat, rowvar=False)[0, 1]
        if np.abs(coeff) > 0.05:
            correlates_dict[mol_key] = coeff

    sort_inds = np.argsort(np.asarray([(correlates_dict[key]) for key in correlates_dict.keys()]))
    keys_list = list(correlates_dict.keys())
    sorted_correlates_dict = {keys_list[ind]: correlates_dict[keys_list[ind]] for ind in sort_inds}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(sorted_correlates_dict.keys()),
        x=[corr for corr in sorted_correlates_dict.values()],
        textposition='auto',
        orientation='h',
        text=[corr for corr in sorted_correlates_dict.values()],
    ))

    fig.update_layout(barmode='relative')
    fig.update_traces(texttemplate='%{text:.2f}')
    fig.update_yaxes(title_font=dict(size=24), tickfont=dict(size=24))
    fig.show(renderer='browser')

    """score distribution"""
    fig = go.Figure()
    fig.add_histogram(x=np.log10(1 - np.abs(score)),
                      nbinsx=500,
                      marker_color='rgba(0,0,100,1)')
    fig.update_layout(xaxis_title='log10(1-overlap)')
    fig.show(renderer='browser')

    return None
