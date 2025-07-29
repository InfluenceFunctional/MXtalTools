from typing import Optional

import ase
import ase.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import wandb
from _plotly_utils.colors import n_colors
from plotly.subplots import make_subplots
from scipy.stats import linregress
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch import scatter
from torch_scatter import scatter

from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.common.geometry_utils import cell_vol_np
from mxtaltools.common.utils import get_point_density, get_plotly_fig_size_mb
from mxtaltools.constants.mol_classifier_constants import polymorph2form
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.autoencoder_utils import batch_rmsd
from mxtaltools.reporting.ae_reporting import autoencoder_evaluation_overlaps, gaussian_3d_overlap_plot
from mxtaltools.reporting.crystal_search_visualizations import stacked_property_distribution_lists
from mxtaltools.reporting.old_figures import discriminator_BT_reporting
from mxtaltools.reporting.utils import lightweight_one_sided_violin, plotly_setup, process_discriminator_outputs


def cell_params_hist(stats_dict, sample_sources_list):
    n_crystal_features = 12
    samples_dict = {name: stats_dict[name] for name in sample_sources_list}

    for key in samples_dict.keys():
        if isinstance(samples_dict[key], list):
            samples_dict[key] = np.stack(samples_dict[key])

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    # 1d Histograms
    colors = n_colors('rgb(255,0,0)', 'rgb(0, 0, 255)', len(samples_dict.keys()) + 1, colortype='rgb')
    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        for k_ind, key in enumerate(samples_dict.keys()):
            samples = samples_dict[key]
            fig.add_trace(go.Violin(
                x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
                name=key, legendgroup=key, showlegend=True if i == 0 else False,
                meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
                line_color=colors[k_ind],
            ),
                row=row, col=col
            )

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
    fig.update_traces(opacity=0.5)
    return fig


def niggli_hist(stats_dict, sample_sources_list):
    n_crystal_features = 6
    samples_dict = {name: stats_dict[name] for name in sample_sources_list}

    for key in samples_dict.keys():
        if isinstance(samples_dict[key], list):
            lattice_feats = np.stack(samples_dict[key])[:, :6]
            niggli_feats = np.stack([
                lattice_feats[:, 2] / stats_dict['mol_radius'] / 2,
                lattice_feats[:, 1] / lattice_feats[:, 2],
                lattice_feats[:, 0] / lattice_feats[:, 1],
                np.cos(lattice_feats[:, 3]) / (lattice_feats[:, 1] / 2 / lattice_feats[:, 2]),
                np.cos(lattice_feats[:, 4]) / (lattice_feats[:, 0] / 2 / lattice_feats[:, 2]),
                np.cos(lattice_feats[:, 5]) / (lattice_feats[:, 0] / 2 / lattice_feats[:, 1]),
            ]).T
            samples_dict[key] = niggli_feats

    lattice_features = ['normed_c', 'b/c', 'a/b', 'alpha scale', 'beta scale', 'gamma scale']
    # 1d Histograms
    colors = n_colors('rgb(255,0,0)', 'rgb(0, 0, 255)', len(samples_dict.keys()) + 1, colortype='rgb')
    fig = make_subplots(rows=2, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        for k_ind, key in enumerate(samples_dict.keys()):
            samples = samples_dict[key]
            fig.add_trace(go.Violin(
                x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
                name=key, legendgroup=key, showlegend=True if i == 0 else False,
                meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
                line_color=colors[k_ind],
            ),
                row=row, col=col
            )

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
    fig.update_traces(opacity=0.5)
    return fig


def simple_cell_hist(sample_batch, reference_dist=None, n_kde_points=200, bw_ratio=50, mode='cell'):
    if mode == 'cell':
        samples = sample_batch.cell_parameters().cpu().detach().numpy()
        custom_ranges = {
            0: [0, float(np.max(samples[:, 0]) * 1.1)],
            1: [0, float(np.max(samples[:, 1]) * 1.1)],
            2: [0, float(np.max(samples[:, 2]) * 1.1)],
            3: [1, np.pi / 2],  # for cell_alpha
            4: [1, np.pi / 2],  # for cell_beta
            5: [1, np.pi / 2],  # for cell_gamma
            6: [0, 1],  # for aunit_x
            7: [0, 1],  # for aunit_y
            8: [0, 1],  # for aunit_z
            9: [-2 * np.pi, 2 * np.pi],  # orientation_1
            10: [-2 * np.pi, 2 * np.pi],  # orientation_2
            11: [0, 2 * np.pi],  # orientation_3
        }
    elif mode == 'latent':
        samples = sample_batch.cell_params_to_gen_basis().cpu().detach().numpy()
        custom_ranges = {i: [-6.5, 6.5] for i in range(12)}
    else:
        assert False

    if hasattr(sample_batch, 'lj_pot'):
        if sample_batch.lj_pot is not None:
            energies = sample_batch.lj_pot
            good_inds = torch.argwhere(energies <= torch.quantile(energies, 0.1))
            good_samples = samples[good_inds.flatten()]
        else:
            good_samples = None
    else:
        good_samples = None

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']

    n_crystal_features = 12
    colors = ['red', 'blue', 'green']
    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)

    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        bw = 1.0 / bw_ratio  # float(np.ptp(samples[:, i]) / bw_ratio)
        # Reference distribution
        if reference_dist is not None:
            x_ref, y_ref = lightweight_one_sided_violin(reference_dist[:, i],
                                                        n_kde_points,
                                                        bandwidth_factor=bw,
                                                        data_min=custom_ranges[i][0],
                                                        data_max=custom_ranges[i][1])
            if len(x_ref) > 0:
                fig.add_trace(go.Scatter(
                    x=x_ref,
                    y=y_ref,
                    mode='lines',
                    fill='toself',#'tonexty' if i == 0 else 'tonexty',  # Fill to next y (which is 0)
                    fillcolor='rgba(0, 0, 255, 0.3)',  # Semi-transparent blue
                    line=dict(color=colors[1], width=1),
                    name='Reference',
                    legendgroup='Reference',
                    showlegend=True if i == 0 else False,
                ), row=row, col=col)

                # # Add zero line for reference to fill against
                # fig.add_trace(go.Scatter(
                #     x=x_ref,
                #     y=np.zeros_like(x_ref),
                #     mode='lines',
                #     line=dict(color='rgba(0,0,0,0)', width=0),
                #     showlegend=False,
                #     hoverinfo='skip'
                # ), row=row, col=col)

        # Main samples
        x_samp, y_samp = lightweight_one_sided_violin(samples[:, i],
                                                      n_kde_points,
                                                      bandwidth_factor=bw,
                                                      data_min=custom_ranges[i][0],
                                                      data_max=custom_ranges[i][1])
        if len(x_samp) > 0:
            fig.add_trace(go.Scatter(
                x=x_samp,
                y=y_samp,
                mode='lines',
                fill='toself',
                fillcolor=f'rgba(255, 0, 0, 0.3)',  # Semi-transparent red
                line=dict(color=colors[0], width=1),
                name='Samples',
                legendgroup='Samples',
                showlegend=True if i == 0 else False,
            ), row=row, col=col)

            # # Zero line for samples
            # fig.add_trace(go.Scatter(
            #     x=x_samp,
            #     y=np.zeros_like(x_samp),
            #     mode='lines',
            #     line=dict(color='rgba(0,0,0,0)', width=0),
            #     showlegend=False,
            #     hoverinfo='skip'
            # ), row=row, col=col)

        # Top 10% samples
        if good_samples is not None:
            x_good, y_good = lightweight_one_sided_violin(good_samples[:, i],
                                                          n_kde_points,
                                                          bandwidth_factor=bw,
                                                          data_min=custom_ranges[i][0],
                                                          data_max=custom_ranges[i][1])
            if len(x_good) > 0:
                fig.add_trace(go.Scatter(
                    x=x_good,
                    y=y_good,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(0, 128, 0, 0.3)',  # Semi-transparent green
                    line=dict(color=colors[2], width=1),
                    name='Top 10%',
                    legendgroup='Top 10%',
                    showlegend=True if i == 0 else False,
                ), row=row, col=col)
                #
                # # Zero line for good samples
                # fig.add_trace(go.Scatter(
                #     x=x_good,
                #     y=np.zeros_like(x_good),
                #     mode='lines',
                #     line=dict(color='rgba(0,0,0,0)', width=0),
                #     showlegend=False,
                #     hoverinfo='skip'
                # ), row=row, col=col)

    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.update_xaxes(range=custom_ranges[i], row=row, col=col)
        fig.update_yaxes(range=[0, None], row=row, col=col)  # Only show positive y

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def simple_latent_hist(sample_batch, samples=None, reference_dist=None):
    if samples is None:
        samples = sample_batch.cell_params_to_gen_basis().cpu().detach().numpy()

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    # 1d Histograms
    n_crystal_features = 12
    bw = 50
    colors = ['red', 'blue']
    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        if reference_dist is not None:
            fig.add_trace(go.Violin(
                x=reference_dist[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
                meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / bw), points=False,
                name='Reference', legendgroup='Reference',
                showlegend=True if ((i == 0) and reference_dist is not None) else False,
                line_color=colors[1],
            ),
                row=row, col=col
            )
        fig.add_trace(go.Violin(
            x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
            meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / bw), points=False,
            name='Samples', legendgroup='Samples',
            showlegend=True if ((i == 0) and reference_dist is not None) else False,
            line_color=colors[0],
        ),
            row=row, col=col
        )

    custom_ranges = {
        0: [-6.5, 6.5],  # for cell_a
        1: [-6.5, 6.5],  # for cell_b
        2: [-6.5, 6.5],  # for cell_c
        3: [-6.5, 6.5],  # for cell_alpha
        4: [-6.5, 6.5],  # for cell_beta
        5: [-6.5, 6.5],  # for cell_gamma
        6: [-6.5, 6.5],  # for aunit_x
        7: [-6.5, 6.5],  # for aunit_y
        8: [-6.5, 6.5],  # for aunit_z
        9: [-6.5, 6.5],  # orientation_1
        10: [-6.5, 6.5],  # orientation_2
        11: [-6.5, 6.5],  # orientation_3
    }

    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.update_xaxes(range=custom_ranges[i], row=row, col=col)

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5
        )
    )
    #fig.update_traces(opacity=0.5)
    return fig


def iter_wise_hist(stats_dict, target_key, log=False):
    energy = stats_dict[target_key]
    batch = stats_dict['sample_iter']
    vdw_list = [energy[batch == int(ind)] for ind in range(int(np.max(batch)) + 1)]
    fig = stacked_property_distribution_lists(y=vdw_list,
                                              xaxis_title=target_key,
                                              yaxis_title='Sampling Iter',
                                              log=log,
                                              )
    return fig


def discriminator_scores_plots(scores_dict, vdw_penalty_dict, packing_coeff_dict, layout):
    fig_dict = {}
    plot_color_dict = {'CSD': ('rgb(250,150,50)'),
                       'Generator': ('rgb(100,50,0)'),
                       'Gaussian': ('rgb(0,50,0)'),
                       'Distorted': ('rgb(0,100,100)'),
                       'extra_test': ('rgb(100, 25, 100)')}

    sample_types = list(scores_dict.keys())

    all_vdws = np.concatenate([vdw_penalty_dict[stype] for stype in sample_types])
    all_scores = np.concatenate([scores_dict[stype] for stype in sample_types])
    all_coeffs = np.concatenate([packing_coeff_dict[stype] for stype in sample_types])

    sample_source = np.concatenate([[stype for _ in range(len(scores_dict[stype]))] for stype in sample_types])
    scores_range = np.ptp(all_scores)

    'score vs vs vdw'
    bandwidth1, fig, viridis = score_vs_vdw_plot(all_scores, all_vdws, layout,
                                                 plot_color_dict, sample_types, scores_dict,
                                                 scores_range, vdw_penalty_dict)
    fig_dict['Discriminator vs vdw scores'] = fig

    'score vs packing'
    fig_dict['Discriminator vs Reduced Volume'] = score_vs_volume_plot(all_coeffs, all_scores, bandwidth1, layout,
                                                                       packing_coeff_dict,
                                                                       plot_color_dict, sample_types, scores_dict,
                                                                       viridis)

    fig_dict['Discriminator Scores Analysis'] = combined_scores_plot(all_coeffs, all_scores, all_vdws, layout,
                                                                     sample_source)

    return fig_dict


def score_vs_vdw_plot(all_scores, all_vdws, layout, plot_color_dict, sample_types, scores_dict, scores_range,
                      vdw_penalty_dict):
    bandwidth1 = scores_range / 200
    bandwidth2 = np.ptp(all_vdws) / 200
    viridis = px.colors.sequential.Viridis
    scores_labels = sample_types
    fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)'),
                        specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.14)
    for i, label in enumerate(scores_labels):
        legend_label = label
        fig.add_trace(go.Violin(x=scores_dict[label], name=legend_label, line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4,
                                meanline_visible=True, bandwidth=bandwidth1, points=False),
                      row=1, col=1)
        fig.add_trace(go.Violin(x=vdw_penalty_dict[label], name=legend_label,
                                line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4, meanline_visible=True,
                                bandwidth=bandwidth2, points=False),
                      row=1, col=2)
    # fig.update_xaxes(col=2, row=1, range=[-3, 0])
    rrange = np.logspace(3, 0, len(viridis))
    cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
    cscale[0][0] = 0
    fig.add_trace(go.Histogram2d(x=np.clip(all_scores, a_min=np.quantile(all_scores, 0.001), a_max=np.amax(all_scores)),
                                 y=np.clip(all_vdws, a_min=np.amin(all_vdws), a_max=np.quantile(all_vdws, 0.999)),
                                 showscale=False,
                                 nbinsy=50, nbinsx=200,
                                 colorscale=cscale,
                                 colorbar=dict(
                                     tick0=0,
                                     tickmode='array',
                                     tickvals=[0, 1000, 10000]
                                 )),
                  row=2, col=1)
    fig.update_layout(showlegend=False, yaxis_showgrid=True)
    fig.update_xaxes(title_text='Model Score', row=1, col=1)
    fig.update_xaxes(title_text='vdW Score', row=1, col=2)
    fig.update_xaxes(title_text='Model Score', row=2, col=1)
    fig.update_yaxes(title_text='vdW Score', row=2, col=1)
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.layout.annotations[0].update(x=0.025)
    fig.layout.annotations[1].update(x=0.575)
    fig.layout.margin = layout.margin
    return bandwidth1, fig, viridis


def score_vs_volume_plot(all_coeffs, all_scores, bandwidth1, layout, packing_coeff_dict, plot_color_dict, sample_types,
                         scores_dict, viridis):
    bandwidth2 = 0.01
    scores_labels = sample_types
    fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)'),
                        specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.14)
    for i, label in enumerate(scores_labels):
        legend_label = label
        fig.add_trace(go.Violin(x=scores_dict[label], name=legend_label, line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4,
                                meanline_visible=True, bandwidth=bandwidth1, points=False),
                      row=1, col=1)
        fig.add_trace(go.Violin(x=packing_coeff_dict[label], name=legend_label,
                                line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4, meanline_visible=True,
                                bandwidth=bandwidth2, points=False),
                      row=1, col=2)
    rrange = np.logspace(3, 0, len(viridis))
    cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
    cscale[0][0] = 0
    fig.add_trace(go.Histogram2d(x=np.clip(all_scores, a_min=np.quantile(all_scores, 0.001), a_max=np.amax(all_scores)),
                                 y=np.clip(all_coeffs, a_min=0, a_max=1.1),
                                 showscale=False,
                                 nbinsy=50, nbinsx=200,
                                 colorscale=cscale,
                                 colorbar=dict(
                                     tick0=0,
                                     tickmode='array',
                                     tickvals=[0, 1000, 10000]
                                 )),
                  row=2, col=1)
    fig.update_layout(showlegend=False, yaxis_showgrid=True)
    fig.update_xaxes(title_text='Model Score', row=1, col=1)
    fig.update_xaxes(title_text='Packing Coefficient', row=1, col=2)
    fig.update_xaxes(title_text='Model Score', row=2, col=1)
    fig.update_yaxes(title_text='Packing Coefficient', row=2, col=1)
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.layout.annotations[0].update(x=0.025)
    fig.layout.annotations[1].update(x=0.575)
    fig.layout.margin = layout.margin
    return fig


def combined_scores_plot(all_coeffs, all_scores, all_vdws, layout, sample_source):
    """
    # All in one  # todo replace 'packing coefficient' throughout
    """
    scatter_dict = {'LJ Potential': all_vdws, 'Classification Score': all_scores,
                    'Packing Coefficient': all_coeffs, 'Sample Source': sample_source}
    opacity = max(0.1, 1 - len(all_vdws) / 5e4)

    df = pd.DataFrame.from_dict(scatter_dict)
    fig = px.scatter(df,
                     x='LJ Potential', y='Packing Coefficient',
                     color='Classification Score', symbol='Sample Source',
                     marginal_x='histogram', marginal_y='histogram',
                     range_color=(np.amin(all_scores), np.amax(all_scores)),
                     opacity=opacity,
                     )
    fig.layout.margin = layout.margin
    fig.update_layout(  #xaxis_range=[-vdw_cutoff, 0.1],
        yaxis_range=[np.quantile(all_coeffs, 0.001), np.quantile(all_coeffs, 0.999)])
    fig.update_layout(xaxis_title='Scaled LJ Potential', yaxis_title='Packing Coefficient')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=1))
    return fig


def score_correlates_fig(scores_dict, dataDims, tracking_features, layout):
    # todo replace standard correlates fig with this one - much nicer
    # correlate losses with molecular features
    tracking_features = np.asarray(tracking_features)
    g_loss_correlations = np.zeros(dataDims['num_tracking_features'])
    features = []
    ind = 0
    for i in range(dataDims['num_tracking_features']):  # not that interesting
        if ('space_group' not in dataDims['tracking_features'][i]) and \
                ('system' not in dataDims['tracking_features'][i]) and \
                ('density' not in dataDims['tracking_features'][i]) and \
                ('asymmetric_unit' not in dataDims['tracking_features'][i]):
            if (np.average(tracking_features[:, i] != 0) > 0.05) and \
                    (dataDims['tracking_features'][i] != 'crystal_z_prime') and \
                    (dataDims['tracking_features'][
                         i] != 'molecule_is_asymmetric_top'):  # if we have at least 1# relevance
                corr = np.corrcoef(scores_dict['CSD'], tracking_features[:, i], rowvar=False)[0, 1]
                if np.abs(corr) > 0.05:
                    features.append(dataDims['tracking_features'][i])
                    g_loss_correlations[ind] = corr
                    ind += 1

    g_loss_correlations = g_loss_correlations[:ind]

    g_sort_inds = np.argsort(g_loss_correlations)
    g_loss_correlations = g_loss_correlations[g_sort_inds]
    features_sorted = [features[i] for i in g_sort_inds]
    # features_sorted_cleaned_i = [feat.replace('molecule', 'mol') for feat in features_sorted]
    # features_sorted_cleaned_ii = [feat.replace('crystal', 'crys') for feat in features_sorted_cleaned_i]
    features_sorted_cleaned = [feat.replace('molecule_atom_heavier_than', 'atomic # >') for feat in features_sorted]

    functional_group_dict = {
        'NH0': 'tert amine',
        'para_hydroxylation': 'para-hydroxylation',
        'Ar_N': 'aromatic N',
        'aryl_methyl': 'aryl methyl',
        'Al_OH_noTert': 'non-tert al-hydroxyl',
        'C_O': 'carbonyl O',
        'Al_OH': 'al-hydroxyl',
    }
    ff = []
    for feat in features_sorted_cleaned:
        for func in functional_group_dict.keys():
            if func in feat:
                feat = feat.replace(func, functional_group_dict[func])
        ff.append(feat)
    features_sorted_cleaned = ff

    g_loss_dict = {feat: corr for feat, corr in zip(features_sorted, g_loss_correlations)}

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=(
        'a) Molecule & Crystal Features', 'b) Atom Fractions', 'c) Functional Groups Count'), x_title='R Value')

    crystal_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' not in key]
    atom_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' in key]
    mol_keys = [key for key in features_sorted_cleaned if 'count' in key and 'fraction' not in key]

    fig.add_trace(go.Bar(
        y=crystal_keys,
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in crystal_keys],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in crystal_keys]).astype(
            'float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(100,0,0,1)')
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=[feat.replace('molecule_', '') for feat in features_sorted_cleaned if feat in atom_keys],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in atom_keys],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in atom_keys]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(0,0,100,1)')
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=[feat.replace('molecule_', '').replace('_count', '') for feat in features_sorted_cleaned if feat in mol_keys],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in mol_keys],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in mol_keys]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(0,100,0,1)')
    ), row=1, col=3)

    fig.update_yaxes(tickfont=dict(size=14), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=14), row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=13), row=1, col=3)

    fig.layout.annotations[0].update(x=0.12)
    fig.layout.annotations[1].update(x=0.45)
    fig.layout.annotations[2].update(x=0.88)

    fig.layout.margin = layout.margin
    fig.layout.margin.b = 50
    fig.update_xaxes(range=[np.amin(list(g_loss_dict.values())), np.amax(list(g_loss_dict.values()))])
    fig.update_layout(width=1200, height=400)
    fig.update_layout(showlegend=False)

    return fig


def plot_generator_loss_correlates(dataDims, wandb, epoch_stats_dict, generator_losses, layout):
    correlates_dict = {}
    generator_losses['all'] = np.vstack([generator_losses[key] for key in generator_losses.keys()]).T.sum(1)
    loss_labels = list(generator_losses.keys())

    tracking_features = np.asarray(epoch_stats_dict['tracking_features'])

    for i in range(dataDims['num_tracking_features']):  # not that interesting
        if np.average(tracking_features[:, i] != 0) > 0.05:
            corr_dict = {
                loss_label: np.corrcoef(generator_losses[loss_label], tracking_features[:, i], rowvar=False)[0, 1]
                for loss_label in loss_labels}
            correlates_dict[dataDims['tracking_features'][i]] = corr_dict

    sort_inds = np.argsort(np.asarray([(correlates_dict[key]['all']) for key in correlates_dict.keys()]))
    keys_list = list(correlates_dict.keys())
    sorted_correlates_dict = {keys_list[ind]: correlates_dict[keys_list[ind]] for ind in sort_inds}

    fig = go.Figure()
    for label in loss_labels:
        fig.add_trace(go.Bar(name=label,
                             y=list(sorted_correlates_dict.keys()),
                             x=[corr[label] for corr in sorted_correlates_dict.values()],
                             textposition='auto',
                             orientation='h',
                             text=[corr[label] for corr in sorted_correlates_dict.values()],
                             ))
    fig.update_layout(barmode='relative')
    fig.update_traces(texttemplate='%{text:.2f}')
    fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=10))

    fig.layout.margin = layout.margin

    wandb.log(data={'Generator Loss Correlates': fig}, commit=False)


def plot_discriminator_score_correlates(dataDims, epoch_stats_dict, layout):
    correlates_dict = {}
    real_scores = epoch_stats_dict['discriminator_real_score']
    tracking_features = np.asarray(epoch_stats_dict['tracking_features'])

    for i in range(dataDims['num_tracking_features']):  # not that interesting
        if (np.average(tracking_features[:, i] != 0) > 0.05):
            corr = np.corrcoef(real_scores, tracking_features[:, i], rowvar=False)[0, 1]
            if np.abs(corr) > 0.05:
                correlates_dict[dataDims['tracking_features'][i]] = corr

    sort_inds = np.argsort(np.asarray([(correlates_dict[key]) for key in correlates_dict.keys()]))
    keys_list = list(correlates_dict.keys())
    sorted_correlates_dict = {keys_list[ind]: correlates_dict[keys_list[ind]] for ind in sort_inds}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(sorted_correlates_dict.keys()),
        x=[corr for corr in sorted_correlates_dict.values()],
        textposition='auto',
        orientation='h',
    ))
    fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=10))

    fig.layout.margin = layout.margin

    return fig


def log_cubic_defect(samples):
    cleaned_samples = samples
    cubic_distortion = np.abs(1 - np.nan_to_num(np.stack(
        [cell_vol_np(cleaned_samples[i, 0:3], cleaned_samples[i, 3:6]) / np.prod(cleaned_samples[i, 0:3], axis=-1) for
         i in range(len(cleaned_samples))])))
    cubic_distortion_hist = np.histogram(cubic_distortion, bins=256, range=(0, 1))

    return np.average(cubic_distortion), cubic_distortion_hist


def cell_generation_analysis(config, dataDims, epoch_stats_dict):
    """
    do analysis and plotting for cell generator
    """
    layout = plotly_setup(config)
    fig_dict = {}
    if isinstance(epoch_stats_dict['cell_parameters'], list):
        cell_parameters = np.stack(epoch_stats_dict['cell_parameters'])
    else:
        cell_parameters = epoch_stats_dict['cell_parameters']

    mean_cubic_distortion, cubic_distortion_hist = log_cubic_defect(cell_parameters)
    fig_dict['Generator Samples'] = generated_cell_scatter_fig(epoch_stats_dict, layout)
    samples_list = log_crystal_samples(epoch_stats_dict)

    wandb.log(data={'Avg generated cubic distortion': mean_cubic_distortion}, commit=False)
    wandb.log(data={"Generated cubic distortions": wandb.Histogram(np_histogram=cubic_distortion_hist, num_bins=256)},
              commit=False)
    wandb.log(data={"Generated cell parameter variation": cell_parameters.std(0).mean()}, commit=False)
    [wandb.log({f'crystal_sample_{ind}': samples_list[ind]}, commit=False) for ind in range(len(samples_list))]
    wandb.log(fig_dict, commit=False)

    return None


def vdw_vs_variation_dist(epoch_stats_dict):
    fig = go.Figure()
    variation_factor = epoch_stats_dict['generator_variation_factor']
    vdw_loss = epoch_stats_dict['generator_per_mol_vdw_loss']
    fig.add_histogram2d(x=variation_factor, y=vdw_loss, nbinsx=64, nbinsy=64)
    fig.update_layout(xaxis_title="Variation Factor",
                      yaxis_title="vdW Loss"
                      )
    wandb.log({"vdW vs Variation Loss": fig}, commit=False)


def variation_vs_prior_dist(epoch_stats_dict):
    fig = go.Figure()
    variation_factor = epoch_stats_dict['generator_variation_factor']
    prior_loss = epoch_stats_dict['generator_prior_loss']
    fig.add_histogram2d(x=variation_factor, y=np.log10(prior_loss), nbinsx=64, nbinsy=64)
    fig.update_layout(xaxis_title="Variation Factor",
                      yaxis_title="Prior Loss"
                      )
    wandb.log({"Variation vs Prior Loss": fig}, commit=False)


def variation_vs_deviation_dist(epoch_stats_dict: dict):
    fig = go.Figure()
    variation_factor = epoch_stats_dict['generator_variation_factor']
    scaled_deviations = np.stack(epoch_stats_dict['generator_scaled_deviation'])
    fig.add_histogram2d(x=variation_factor, y=np.linalg.norm(scaled_deviations, axis=1), nbinsx=64, nbinsy=64)
    fig.update_layout(xaxis_title="Variation Factor",
                      yaxis_title="Scaled Deviations"
                      )
    wandb.log({"Variation vs Deviation": fig}, commit=False)


def vdw_vs_prior_dist(epoch_stats_dict: dict):
    fig = go.Figure()
    prior_loss = epoch_stats_dict['generator_prior_loss']
    vdw_loss = epoch_stats_dict['generator_per_mol_vdw_loss']
    fig.add_histogram2d(x=np.log10(prior_loss), y=vdw_loss, nbinsx=64, nbinsy=64)
    fig.update_layout(xaxis_title="Prior Loss",
                      yaxis_title="vdW Loss"
                      )
    wandb.log({"vdW vs Prior Loss": fig}, commit=False)


def log_crystal_samples(epoch_stats_dict: Optional[dict] = None, sample_batch: Optional = None,
                        return_filenames: bool = False):
    if epoch_stats_dict is not None:
        sample_crystals = epoch_stats_dict['generator_samples']
    else:
        sample_crystals = sample_batch

    topk_samples = torch.arange(6)  #torch.argsort(sample_crystals.loss)[:6]

    mols = [ase_mol_from_crystaldata(collate_data_list(sample_crystals),
                                     index=int(topk_samples[ind]),
                                     mode='distance',
                                     cutoff=8) for ind in range(min(6, len(topk_samples)))]
    filenames = []
    for i in range(len(mols)):
        cp = float(sample_crystals[i].packing_coeff)
        lj_pot = float(sample_crystals[i].lj_pot)
        filename = f'cp={cp:.2f}_LJ={lj_pot:.1g}.cif'
        filenames.append(filename)
        ase.io.write(filename, mols[i])

    samples_list = []
    for ind, file in enumerate(filenames):
        try:
            samples_list.append(wandb.Molecule(open(file), caption=file))
        except FileNotFoundError:
            pass

    if return_filenames:
        return samples_list, filenames
    else:
        return samples_list


def generated_cell_scatter_fig(epoch_stats_dict, layout):
    scaled_vdw = epoch_stats_dict['per_mol_scaled_LJ_energy']
    vdw_overlap = epoch_stats_dict['per_mol_normed_overlap']
    packing_coeff = epoch_stats_dict['packing_coefficient']
    xy = np.vstack([packing_coeff, scaled_vdw])
    try:
        z = get_point_density(xy, bins=25)
    except:
        z = np.ones(len(xy))
    scatter_dict = {'vdw_overlap': vdw_overlap,
                    'packing_coefficient': packing_coeff,
                    'energy': scaled_vdw,
                    'point_density': z,
                    }
    opacity = max(0.25, 1 - len(scatter_dict['energy']) / 5e4)
    df = pd.DataFrame.from_dict(scatter_dict)
    fig = px.scatter(scatter_dict,
                     x='packing_coefficient', y='energy',
                     color='point_density',
                     marginal_x='violin', marginal_y='violin',
                     opacity=opacity
                     )

    fig.update_layout(yaxis_title='Energy', xaxis_title='Packing Coeff')
    fig.update_layout(yaxis_range=[np.amin(df['energy']) - np.ptp(df['energy']) * 0.1,
                                   np.amax(df['energy']) + np.ptp(df['energy']) * 0.1],
                      xaxis_range=[max(0, np.amin(df['packing_coefficient']) * 0.9),
                                   min(2, np.amax(df['packing_coefficient']) * 1.1)],
                      )

    if len(df['vdw_overlap']) > 1000:
        fig.write_image('fig.png', width=512, height=512)  # save the image rather than the fig, for size reasons
        return wandb.Image('fig.png')
    else:
        return fig


def simple_cell_scatter_fig(sample_batch, cluster_inds=None, aux_array=None, aux_scalar_name: str = ''):
    xy = np.vstack([sample_batch.packing_coeff.cpu().detach(), sample_batch.silu_pot.cpu().detach()])
    try:
        z = get_point_density(xy, bins=25)
    except:
        z = np.ones(xy.shape[1])

    # if hasattr(sample_batch, 'gfn_energy'):
    #     energy = sample_batch.gfn_energy
    # else:
    energy = sample_batch.lj_pot.cpu().detach()
    energy[energy > 0] = np.log(energy[energy > 0])
    scatter_dict = {'energy': energy,
                    'packing_coefficient': sample_batch.packing_coeff.cpu().detach(),
                    'point_density': z
                    }
    if aux_array is not None:
        scatter_dict[aux_scalar_name] = aux_array
        color_tag = aux_scalar_name
    else:
        color_tag = 'point_density'
    opacity = max(0.25, 1 - sample_batch.num_graphs / 5e4)
    df = pd.DataFrame.from_dict(scatter_dict)

    fig = px.scatter(scatter_dict,
                     x='packing_coefficient', y='energy',
                     color=color_tag,
                     marginal_x='violin', marginal_y='violin',
                     opacity=opacity
                     )

    fig.update_layout(yaxis_title='Energy', xaxis_title='Packing Coeff')
    fig.update_layout(yaxis_range=[np.amin(df['energy']) - 10,
                                   min(500, np.amax(df['energy']) + np.ptp(df['energy']) * 0.1)],
                      xaxis_range=[max(0, np.amin(df['packing_coefficient']) * 0.9),
                                   min(2, np.amax(df['packing_coefficient']) * 1.1)],
                      )

    if cluster_inds is not None:
        energy_np = energy.cpu().numpy()
        packing_np = sample_batch.packing_coeff.cpu().numpy()
        anchor_x = []
        anchor_y = []

        for c in np.unique(cluster_inds):
            mask = cluster_inds == c
            if np.any(mask):
                min_idx = np.argmin(energy_np[mask])
                true_idx = np.where(mask)[0][min_idx]
                anchor_x.append(packing_np[true_idx])
                anchor_y.append(energy_np[true_idx])

        fig.add_scatter(
            x=anchor_x,
            y=anchor_y,
            mode='markers',
            marker=dict(
                line=dict(
                    color='grey',
                    width=4),
                color='black',
                size=14,
            ),
            name='Cluster Minima',
            showlegend=False
        )

    return fig


def log_regression_accuracy(config, dataDims, epoch_stats_dict):
    target_key = config.dataset.regression_target

    try:
        raw_target = np.asarray(epoch_stats_dict['regressor_target'])
        raw_prediction = np.asarray(epoch_stats_dict['regressor_prediction'])
    except:
        raw_target = np.concatenate(epoch_stats_dict['regressor_target'])
        raw_prediction = np.concatenate(epoch_stats_dict['regressor_prediction'])

    'single-variable regression analysis'
    prediction = raw_prediction.flatten()
    target = raw_target.flatten()
    losses = ['abs_error', 'abs_normed_error', 'squared_error']
    loss_dict = {}
    fig_dict = {}
    for loss in losses:
        if loss == 'abs_error':
            loss_i = np.abs(target - prediction)
        elif loss == 'abs_normed_error':
            loss_i = np.abs((target - prediction) / np.abs(target))
        elif loss == 'squared_error':
            loss_i = (target - prediction) ** 2
        else:
            assert False, "Loss not implemented"

        loss_dict[loss + '_mean'] = np.mean(loss_i)
        loss_dict[loss + '_std'] = np.std(loss_i)

        linreg_result = linregress(target, prediction)
        loss_dict['regression_R_value'] = linreg_result.rvalue
        loss_dict['regression_slope'] = linreg_result.slope

    # predictions vs target trace
    xline = np.linspace(max(min(target), min(prediction)),
                        min(max(target), max(prediction)), 2)

    xy = np.vstack([target, prediction])
    try:
        z = get_point_density(xy)
    except:
        z = np.ones_like(target)

    fig = make_subplots(cols=2, rows=1)

    num_points = len(prediction)
    opacity = max(0.1, np.exp(-num_points / 10000))
    fig.add_trace(go.Scattergl(x=target, y=prediction, mode='markers', marker=dict(color=z), opacity=opacity,
                               showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                  row=1, col=1)

    fig.add_trace(go.Histogram(x=prediction - target,
                               histnorm='probability density',
                               nbinsx=100,
                               name="Error Distribution",
                               showlegend=False,
                               marker_color='rgba(0,0,100,1)'),
                  row=1, col=2)  #

    #
    target_name = dataDims['regression_target']
    fig.update_yaxes(title_text=f'Predicted {target_name}', row=1, col=1, tickformat=".2g")

    fig.update_xaxes(title_text=f'True {target_name}', row=1, col=1, tickformat=".2g")

    fig.update_xaxes(title_text=f'{target_name} Error', row=1, col=2, tickformat=".2g")

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig.write_image('fig.png', width=1024, height=512)  # save the image rather than the fig, for size reasons
    fig_dict['Regression Results'] = wandb.Image('fig.png')
    wandb.log(fig_dict, commit=False)
    wandb.log(loss_dict, commit=False)

    return None


def detailed_reporting(config, dataDims, train_epoch_stats_dict, test_epoch_stats_dict,
                       extra_test_dict=None):
    """
    Do analysis and upload results to w&b
    """
    if test_epoch_stats_dict is not None:
        if 'generated_cell_parameters' in test_epoch_stats_dict.keys() or 'cell_parameters' in test_epoch_stats_dict.keys():
            if config.logger.log_figures:
                if config.mode == 'generator':
                    fig_dict = {}
                    fig_dict['Lattice Features Distribution'] = cell_params_hist(test_epoch_stats_dict,
                                                                                 ['prior', 'cell_parameters'])
                    fig_dict['Niggli Features Distribution'] = niggli_hist(test_epoch_stats_dict,
                                                                           ['prior', 'cell_parameters'])
                    fig_dict['Iterwise vdW'] = iter_wise_hist(test_epoch_stats_dict, 'per_mol_scaled_LJ_energy')
                    fig_dict['Iterwise Packing Coeff'] = iter_wise_hist(test_epoch_stats_dict, 'packing_coefficient')
                    fig_dict['Iterwise Ellipsoid Energy'] = iter_wise_hist(test_epoch_stats_dict, 'ellipsoid_energy')
                    for key in fig_dict.keys():
                        fig = fig_dict[key]
                        if get_plotly_fig_size_mb(fig) > 0.1:  # bigger than .1 MB
                            fig.write_image(key + 'fig.png', width=1024,
                                            height=512)  # save the image rather than the fig, for size reasons
                            fig_dict[key] = wandb.Image(key + 'fig.png')
                    wandb.log(fig_dict, commit=False)

                elif config.mode == 'discriminator':
                    fig = cell_params_hist(test_epoch_stats_dict,
                                           ['real_cell_parameters', 'generated_cell_parameters'])
                    wandb.log(data={'Lattice Features Distribution': fig}, commit=False)

                elif config.mode == 'proxy_discriminator':
                    fig = cell_params_hist(wandb, test_epoch_stats_dict,
                                           ['generated_cell_parameters'])
                    wandb.log(data={'Lattice Features Distribution': fig}, commit=False)

        if config.mode == 'generator':
            cell_generation_analysis(config, dataDims, test_epoch_stats_dict)

        if config.mode == 'discriminator':
            discriminator_analysis(config, dataDims, test_epoch_stats_dict, extra_test_dict)

        elif config.mode == 'proxy_discriminator':
            proxy_discriminator_analysis(config, dataDims, test_epoch_stats_dict, extra_test_dict)

        elif config.mode == 'regression' or config.mode == 'embedding_regression' or config.mode == 'crystal_regression':
            log_regression_accuracy(config, dataDims, test_epoch_stats_dict)

        elif config.mode == 'autoencoder':
            if len(train_epoch_stats_dict) > 0:
                log_autoencoder_analysis(config, dataDims, train_epoch_stats_dict,
                                         'train')
            if len(test_epoch_stats_dict) > 0:
                log_autoencoder_analysis(config, dataDims, test_epoch_stats_dict,
                                         'test')

        elif config.mode == 'polymorph_classification':
            if train_epoch_stats_dict != {}:
                classifier_reporting(true_labels=np.concatenate(train_epoch_stats_dict['true_labels']),
                                     probs=np.stack(train_epoch_stats_dict['probs']),
                                     ordered_class_names=polymorph2form['acridine'],
                                     wandb=wandb,
                                     epoch_type='train')
            if test_epoch_stats_dict != {}:
                classifier_reporting(true_labels=np.concatenate(test_epoch_stats_dict['true_labels']),
                                     probs=np.stack(test_epoch_stats_dict['probs']),
                                     ordered_class_names=polymorph2form['acridine'],
                                     wandb=wandb,
                                     epoch_type='test')

        # todo rewrite this - it's extremely slow and doesn't always work
        # combined_stats_dict = train_epoch_stats_dict.copy()
        # for key in combined_stats_dict.keys():
        #     if isinstance(train_epoch_stats_dict[key], list) and isinstance(train_epoch_stats_dict[key][0], np.ndarray):
        #         if isinstance(test_epoch_stats_dict[key], np.ndarray):
        #             combined_stats_dict[key] = np.concatenate(train_epoch_stats_dict[key] + [test_epoch_stats_dict[key]])
        #         else:
        #             combined_stats_dict[key] = np.concatenate([train_epoch_stats_dict[key] + test_epoch_stats_dict[key]])
        #     elif isinstance(train_epoch_stats_dict[key], np.ndarray):
        #         combined_stats_dict[key] = np.concatenate([train_epoch_stats_dict[key], test_epoch_stats_dict[key]])
        #     else:
        #         pass

        # autoencoder_embedding_map(combined_stats_dict)

    if extra_test_dict is not None and len(extra_test_dict) > 0 and 'blind_test' in config.extra_test_set_name:
        discriminator_BT_reporting(dataDims, wandb, test_epoch_stats_dict, extra_test_dict)

    return None


def classifier_reporting(true_labels, probs, ordered_class_names, wandb, epoch_type):
    present_classes = np.unique(true_labels)
    present_class_names = [ordered_class_names[ind] for ind in present_classes]
    #if len(present_classes) > 1:

    type_probs = F.softmax(torch.Tensor(probs[:, list(present_classes.astype(int))]), dim=1).detach().numpy()
    predicted_class = np.argmax(type_probs, axis=1)

    if len(present_classes) == 2:
        type_probs = type_probs[:, 1]

    if len(present_classes) > 1:
        train_score = roc_auc_score(true_labels, type_probs, multi_class='ovo')
    else:
        train_score = 1
    train_f1_score = f1_score(true_labels, predicted_class, average='micro')
    train_cmat = confusion_matrix(true_labels, predicted_class, normalize='true')
    fig = go.Figure(go.Heatmap(z=train_cmat, x=present_class_names, y=present_class_names))
    fig.update_layout(xaxis=dict(title="Predicted Forms"),
                      yaxis=dict(title="True Forms")
                      )

    wandb.log({f"{epoch_type} ROC_AUC": train_score,
               f"{epoch_type} F1 Score": train_f1_score,
               f"{epoch_type} 1-ROC_AUC": 1 - train_score,
               f"{epoch_type} 1-F1 Score": 1 - train_f1_score,
               f"{epoch_type} Confusion Matrix": fig}, commit=False)


def log_autoencoder_analysis(config, dataDims, epoch_stats_dict, epoch_type):
    """
    analyze performance of autoencoder model
    Parameters
    ----------
    config
    dataDims
    epoch_stats_dict
    epoch_type

    Returns
    -------

    """
    allowed_types = np.array(dataDims['allowed_atom_types'])
    type_translation_index = np.zeros(allowed_types.max() + 1) - 1
    for ind, atype in enumerate(allowed_types):
        type_translation_index[atype] = ind

    # get samples
    mol_batch = collate_data_list(epoch_stats_dict['sample'])
    type_index_tensor = torch.tensor(type_translation_index, dtype=torch.long, device=mol_batch.z.device)
    decoded_mol_batch = collate_data_list(epoch_stats_dict['decoded_sample'])

    mol_batch.x = type_index_tensor[mol_batch.z]
    # compute various distribution overlaps
    (coord_overlap, full_overlap,
     self_coord_overlap, self_overlap,
     self_type_overlap, type_overlap,
     ) = (
        autoencoder_evaluation_overlaps(mol_batch, decoded_mol_batch, config, dataDims))

    rmsd, nodewise_dists, matched_graphs, matched_nodes, _, pred_particle_weights = batch_rmsd(
        mol_batch,
        decoded_mol_batch,
        F.one_hot(mol_batch.x, decoded_mol_batch.z.shape[1]).float(),
    )

    probability_mass_per_graph = scatter(pred_particle_weights, mol_batch.batch,
                                         reduce='sum', dim_size=mol_batch.num_graphs)
    probability_mass_overlap = probability_mass_per_graph / mol_batch.num_atoms

    overall_overlap = scatter(full_overlap / self_overlap, mol_batch.batch, reduce='mean').cpu().detach().numpy()
    evaluation_overlap_loss = scatter(F.smooth_l1_loss(self_overlap, full_overlap, reduction='none'), mol_batch.batch,
                                      reduce='mean')
    eval_stats = {epoch_type + "_evaluation_positions_wise_overlap":
                      scatter(coord_overlap / self_coord_overlap, mol_batch.batch,
                              reduce='mean').mean().cpu().detach().numpy(),
                  epoch_type + "_evaluation_typewise_overlap":
                      scatter(type_overlap / self_type_overlap, mol_batch.batch,
                              reduce='mean').mean().cpu().detach().numpy(),
                  epoch_type + "_evaluation_overall_overlap": overall_overlap.mean(),
                  epoch_type + "_evaluation_matching_clouds_fraction": (np.sum(1 - overall_overlap) < 0.01).mean(),
                  epoch_type + "_evaluation_overlap_loss": evaluation_overlap_loss.mean().cpu().detach().numpy(),
                  epoch_type + "_rmsd": rmsd[matched_graphs].cpu().detach().numpy(),
                  epoch_type + "_matched_graph_fraction": (
                          torch.sum(matched_graphs) / len(matched_graphs)).cpu().detach().numpy(),
                  epoch_type + "_matched_node_fraction": (
                          torch.sum(matched_nodes) / len(matched_nodes)).cpu().detach().numpy(),
                  epoch_type + "_probability_mass_overlap": probability_mass_overlap.mean().cpu().detach().numpy(),
                  }
    wandb.log(data=eval_stats,
              commit=False)

    if config.logger.log_figures:
        mol_batch = collate_data_list(epoch_stats_dict['sample'])

        atom_types = dataDims['allowed_atom_types']

        worst_graph_ind = overall_overlap.argmin()
        fig = (
            gaussian_3d_overlap_plot(mol_batch, decoded_mol_batch,
                                     atom_types,
                                     graph_ind=worst_graph_ind
                                     ))
        wandb.log(data={
            epoch_type + "_worst_pointwise_sample_distribution": fig,
        }, commit=False)

        best_graph_ind = overall_overlap.argmax()
        fig = (
            gaussian_3d_overlap_plot(mol_batch, decoded_mol_batch,
                                     atom_types,
                                     graph_ind=best_graph_ind
                                     ))
        wandb.log(data={
            epoch_type + "_best_pointwise_sample_distribution": fig,
        }, commit=False)

    return None


def discriminator_analysis(config, dataDims, epoch_stats_dict, extra_test_dict=None):
    """
    do analysis and plotting for cell discriminator

    -: scores distribution and vdw penalty by sample source
    -: loss correlates
    """
    fig_dict = {}
    layout = plotly_setup(config)
    scores_dict, vdw_penalty_dict, reduced_volume_dict, pred_distance_dict, true_distance_dict \
        = process_discriminator_outputs(dataDims, epoch_stats_dict, extra_test_dict)

    fig_dict.update(discriminator_scores_plots(scores_dict, vdw_penalty_dict, reduced_volume_dict, layout))
    fig_dict['Distance Results'], dist_rvalue, dist_slope = discriminator_distances_plots(
        pred_distance_dict, true_distance_dict, epoch_stats_dict)

    fig_dict['Score vs. Distance'] = score_vs_distance_plot(pred_distance_dict, scores_dict)

    for key, fig in fig_dict.items():
        fig.write_image(key + 'fig.png', width=1024, height=512)  # save the image rather than the fig, for size reasons
        fig_dict[key] = wandb.Image(key + 'fig.png')

    wandb.log(data=fig_dict, commit=False)
    wandb.log(data={"distance_R_value": dist_rvalue,
                    "distance_slope": dist_slope}, commit=False)

    return None


def proxy_discriminator_analysis(config, dataDims, epoch_stats_dict, extra_test_dict=None):
    """
    do analysis and plotting for cell discriminator

    -: scores distribution and vdw penalty by sample source
    -: loss correlates
    """
    fig_dict = {}
    layout = plotly_setup(config)
    tgt_value = epoch_stats_dict['vdw_score']
    pred_value = epoch_stats_dict['vdw_prediction']

    linreg_result = linregress(tgt_value, pred_value)

    # predictions vs target trace
    xline = np.linspace(max(min(tgt_value), min(pred_value)),
                        min(max(tgt_value), max(pred_value)), 2)

    opacity = 0.35
    xy = np.vstack([tgt_value, pred_value])
    try:
        z = get_point_density(xy, bins=25)
    except:
        z = np.ones(len(xy))

    try:
        scatter_dict = {'true_energy': tgt_value, 'predicted_energy': pred_value, 'point_density': z}
        df = pd.DataFrame.from_dict(scatter_dict)
        fig = px.scatter(df,
                         x='true_energy', y='predicted_energy',
                         color='point_density',
                         marginal_x='histogram', marginal_y='histogram',
                         opacity=opacity
                         )
        fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=True, name='Diagonal', marker_color='rgba(0,0,0,1)'),
                      )
        # fig.add_trace(go.Histogram2d(x=df['true_distance'], y=df['predicted_distance'], nbinsx=100, nbinsy=100, colorbar_dtick="log", showlegend=False))

        fig.update_layout(xaxis_title='Target Energy', yaxis_title='Predicted Energy')

        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig_dict['Proxy Discriminator Parity Plot'] = fig

        # --------- residuals ----------
        opacity = 0.35
        xy = np.vstack([tgt_value, tgt_value - pred_value])
        try:
            z = get_point_density(xy, bins=25)
        except:
            z = np.ones(len(xy))

        fig = go.Figure()
        fig.add_scatter(x=tgt_value, y=tgt_value - pred_value,
                        mode='markers', marker_color=z, opacity=opacity)
        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_layout(xaxis_title='Target Distance', yaxis_title='Error')

        fig_dict['Proxy Residuals'] = fig

        # for key, fig in fig_dict.items():
        #     fig.write_image(key + 'fig.png', width=480,
        #                     height=480)  # save the image rather than the fig, for size reasons
        #     fig_dict[key] = wandb.Image(key + 'fig.png')

        wandb.log(data=fig_dict, commit=False)
        wandb.log(data={"proxy_discrim_R_value": linreg_result.rvalue,
                        "proxy_discrim_slope": linreg_result.slope}, commit=False)

    except:  # sometimes it fails, but I never want it to crash
        pass

    return None


def score_vs_distance_plot(pred_distance_dict, scores_dict):
    sample_types = list(scores_dict.keys())

    fig = make_subplots(cols=2, rows=1)
    x = np.concatenate([scores_dict[stype] for stype in sample_types])
    y = np.concatenate([pred_distance_dict[stype] for stype in sample_types])
    xy = np.vstack([x, y])
    try:
        z = get_point_density(xy, bins=25)
    except:
        z = np.ones(len(xy))

    fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', opacity=0.2, marker_color=z, showlegend=False), row=1, col=1)

    x = np.concatenate([scores_dict[stype] for stype in sample_types])
    y = np.concatenate([pred_distance_dict[stype] for stype in sample_types])

    xy = np.vstack([x, np.log10(y)])

    try:
        z = get_point_density(xy, bins=25)
    except:
        z = np.ones(len(x))

    fig.add_trace(go.Scattergl(x=x, y=y + 1e-16, mode='markers', opacity=0.2, marker_color=z, showlegend=False),
                  row=1, col=2)

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_xaxes(title_text='Model Score')
    fig.update_yaxes(title_text='Distance', row=1, col=1)
    fig.update_yaxes(title_text="Distance", type="log", row=1, col=2)

    return fig


def discriminator_distances_plots(pred_distance_dict, true_distance_dict, epoch_stats_dict):
    if epoch_stats_dict['discriminator_fake_predicted_distance'] is not None:
        tgt_value = np.concatenate(list(true_distance_dict.values())) + 1e-4
        pred_value = np.concatenate(list(pred_distance_dict.values()))

        # filter out 'true' samples
        good_inds = tgt_value != 0
        tgt_value_0 = tgt_value[good_inds]
        pred_value_0 = pred_value[good_inds]

        linreg_result = linregress(tgt_value_0, pred_value_0)

        # predictions vs target trace
        xline = np.linspace(max(min(tgt_value), min(pred_value)),
                            min(max(tgt_value), max(pred_value)), 2)

        opacity = 0.35

        sample_sources = pred_distance_dict.keys()
        sample_source = np.concatenate(
            [[stype for _ in range(len(pred_distance_dict[stype]))] for stype in sample_sources])
        scatter_dict = {'true_distance': tgt_value, 'predicted_distance': pred_value, 'sample_source': sample_source}
        df = pd.DataFrame.from_dict(scatter_dict)
        fig = px.scatter(df,
                         x='true_distance', y='predicted_distance',
                         symbol='sample_source', color='sample_source',
                         marginal_x='histogram', marginal_y='histogram',
                         range_color=(0, np.amax(pred_value)),
                         opacity=opacity,
                         log_x=True, log_y=True
                         )
        fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=True, name='Diagonal', marker_color='rgba(0,0,0,1)'),
                      )
        # fig.add_trace(go.Histogram2d(x=df['true_distance'], y=df['predicted_distance'], nbinsx=100, nbinsy=100, colorbar_dtick="log", showlegend=False))

        fig.update_layout(xaxis_title='Target Distance', yaxis_title='Predicted Distance')

        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    return fig, linreg_result.rvalue, linreg_result.slope
