import numpy as np
import torch
import wandb
from _plotly_utils.colors import n_colors, sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import linregress

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch import scatter
from torch_scatter import scatter
import torch.nn.functional as F

from mxtaltools.common.utils import get_point_density, softmax_np

from mxtaltools.common.geometry_calculations import cell_vol
from mxtaltools.constants.mol_classifier_constants import polymorph2form
from mxtaltools.reporting.ae_reporting import autoencoder_decoder_sample_validation, gaussian_3d_overlap_plots

blind_test_targets = [  # 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII']

target_identifiers = {
    'XVI': 'OBEQUJ',
    'XVII': 'OBEQOD',
    'XVIII': 'OBEQET',
    'XIX': 'XATJOT',
    'XX': 'OBEQIX',
    'XXI': 'KONTIQ',
    'XXII': 'NACJAF',
    'XXIII': 'XAFPAY',
    'XXIII_1': 'XAFPAY01',
    'XXIII_2': 'XAFPAY02',
    'XXXIII_3': 'XAFPAY03',
    'XXXIII_4': 'XAFPAY04',
    'XXIV': 'XAFQON',
    'XXVI': 'XAFQIH',
    'XXXI_1': '2199671_p10167_1_0',
    'XXXI_2': '2199673_1_0',
    # 'XXXI_3': '2199672_1_0',
}


def cell_params_analysis(config, dataDims, wandb, train_loader, epoch_stats_dict):
    n_crystal_features = 12
    # slightly expensive to do this every time
    dataset_cell_distribution = np.asarray(
        [train_loader.dataset[ii].cell_params[0].cpu().detach().numpy() for ii in range(len(train_loader.dataset))])

    cleaned_samples = epoch_stats_dict['final_generated_cell_parameters']
    if 'raw_generated_cell_parameters' in epoch_stats_dict.keys():
        raw_samples = epoch_stats_dict['raw_generated_cell_parameters']
    else:
        raw_samples = None

    if isinstance(cleaned_samples, list):
        cleaned_samples = np.stack(cleaned_samples)

    if isinstance(raw_samples, list):
        raw_samples = np.stack(raw_samples)

    overlaps_1d = {}
    sample_means = {}
    sample_stds = {}
    for i, key in enumerate(dataDims['lattice_features']):
        mini, maxi = np.amin(dataset_cell_distribution[:, i]), np.amax(dataset_cell_distribution[:, i])
        h1, r1 = np.histogram(dataset_cell_distribution[:, i], bins=100, range=(mini, maxi))
        h1 = h1 / len(dataset_cell_distribution[:, i])

        h2, r2 = np.histogram(cleaned_samples[:, i], bins=r1)
        h2 = h2 / len(cleaned_samples[:, i])

        overlaps_1d[f'{key}_1D_Overlap'] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

        sample_means[f'{key}_mean'] = np.mean(cleaned_samples[:, i])
        sample_stds[f'{key}_std'] = np.std(cleaned_samples[:, i])

    average_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys()])
    overlaps_1d['average_1D_overlap'] = average_overlap
    overlap_results = {}
    overlap_results.update(overlaps_1d)
    overlap_results.update(sample_means)
    overlap_results.update(sample_stds)
    wandb.log(overlap_results)

    if config.logger.log_figures:
        fig_dict = {}  # consider replacing by Joy plot

        # bar graph of 1d overlaps
        fig = go.Figure(go.Bar(
            y=list(overlaps_1d.keys()),
            x=[overlaps_1d[key] for key in overlaps_1d],
            orientation='h',
            marker=dict(color='red')
        ))
        fig_dict['1D_overlaps'] = fig

        # 1d Histograms
        fig = make_subplots(rows=4, cols=3, subplot_titles=dataDims['lattice_features'])
        for i in range(n_crystal_features):
            row = i // 3 + 1
            col = i % 3 + 1

            fig.add_trace(go.Histogram(
                x=dataset_cell_distribution[:, i],
                histnorm='probability density',
                nbinsx=100,
                legendgroup="Dataset Samples",
                name="Dataset Samples",
                showlegend=True if i == 0 else False,
                marker_color='#1f77b4',
            ), row=row, col=col)

            fig.add_trace(go.Histogram(
                x=cleaned_samples[:, i],
                histnorm='probability density',
                nbinsx=100,
                legendgroup="Generated Samples",
                name="Generated Samples",
                showlegend=True if i == 0 else False,
                marker_color='#ff7f0e',
            ), row=row, col=col)
            if raw_samples is not None:
                fig.add_trace(go.Histogram(
                    x=raw_samples[:, i],
                    histnorm='probability density',
                    nbinsx=100,
                    legendgroup="Raw Generated Samples",
                    name="Raw Generated Samples",
                    showlegend=True if i == 0 else False,
                    marker_color='#ec0000',
                ), row=row, col=col)
        fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(opacity=0.5)

        fig_dict['lattice_features_distribution'] = fig

        wandb.log(fig_dict)


def plotly_setup(config):
    if config.machine == 'local':
        import plotly.io as pio
        pio.renderers.default = 'browser'

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
        )
    )
    return layout


def cell_density_plot(config, wandb, epoch_stats_dict, layout):
    if epoch_stats_dict['generator_packing_prediction'] is not None and \
            epoch_stats_dict['generator_packing_target'] is not None:

        x = epoch_stats_dict['generator_packing_target']  # generator_losses['generator_per_mol_vdw_loss']
        y = epoch_stats_dict['generator_packing_prediction']  # generator_losses['generator packing loss']

        xy = np.vstack([x, y])
        try:
            z = get_point_density(xy)
        except:
            z = np.ones_like(x)

        xline = np.asarray([np.amin(x), np.amax(x)])
        linreg_result = linregress(x, y)
        yline = xline * linreg_result.slope + linreg_result.intercept

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', marker=dict(color=z), opacity=1))

        fig.add_trace(
            go.Scattergl(x=xline, y=yline, name=f' R={linreg_result.rvalue:.3f}, m={linreg_result.slope:.3f}'))

        fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)', showlegend=False))

        fig.layout.margin = layout.margin
        fig.update_layout(xaxis_title='Asymmetric Unit Volume Target', yaxis_title='Asymmetric Unit Volume Prediction')

        # #fig.write_image('../paper1_figs_new_architecture/scores_vs_emd.png', scale=4)
        if config.logger.log_figures:
            wandb.log({'Cell Packing': fig})
        if (config.machine == 'local') and False:
            fig.show(renderer='browser')


def process_discriminator_outputs(dataDims, epoch_stats_dict, extra_test_dict=None):
    scores_dict = {}
    vdw_penalty_dict = {}
    tracking_features_dict = {}
    packing_coeff_dict = {}
    pred_distance_dict = {}
    true_distance_dict = {}

    generator_inds = np.where(epoch_stats_dict['generator_sample_source'] == 0)[0]
    randn_inds = np.where(epoch_stats_dict['generator_sample_source'] == 1)[0]
    distorted_inds = np.where(epoch_stats_dict['generator_sample_source'] == 2)[0]

    scores_dict['CSD'] = epoch_stats_dict['discriminator_real_score']
    scores_dict['Gaussian'] = epoch_stats_dict['discriminator_fake_score'][randn_inds]
    scores_dict['Generator'] = epoch_stats_dict['discriminator_fake_score'][generator_inds]
    scores_dict['Distorted'] = epoch_stats_dict['discriminator_fake_score'][distorted_inds]

    tracking_features_dict['CSD'] = {feat: vec for feat, vec in zip(dataDims['tracking_features'],
                                                                    epoch_stats_dict['tracking_features'].T)}
    tracking_features_dict['Distorted'] = {feat: vec for feat, vec in
                                           zip(dataDims['tracking_features'],
                                               epoch_stats_dict['tracking_features'][distorted_inds].T)}
    tracking_features_dict['Gaussian'] = {feat: vec for feat, vec in
                                          zip(dataDims['tracking_features'],
                                              epoch_stats_dict['tracking_features'][randn_inds].T)}
    tracking_features_dict['Generator'] = {feat: vec for feat, vec in
                                           zip(dataDims['tracking_features'],
                                               epoch_stats_dict['tracking_features'][generator_inds].T)}

    vdw_penalty_dict['CSD'] = epoch_stats_dict['real_vdw_penalty']
    vdw_penalty_dict['Gaussian'] = epoch_stats_dict['fake_vdw_penalty'][randn_inds]
    vdw_penalty_dict['Generator'] = epoch_stats_dict['fake_vdw_penalty'][generator_inds]
    vdw_penalty_dict['Distorted'] = epoch_stats_dict['fake_vdw_penalty'][distorted_inds]

    packing_coeff_dict['CSD'] = epoch_stats_dict['real_packing_coefficients']
    packing_coeff_dict['Gaussian'] = epoch_stats_dict['generated_packing_coefficients'][randn_inds]
    packing_coeff_dict['Generator'] = epoch_stats_dict['generated_packing_coefficients'][generator_inds]
    packing_coeff_dict['Distorted'] = epoch_stats_dict['generated_packing_coefficients'][distorted_inds]

    pred_distance_dict['CSD'] = epoch_stats_dict['discriminator_real_predicted_distance']
    pred_distance_dict['Gaussian'] = epoch_stats_dict['discriminator_fake_predicted_distance'][randn_inds]
    pred_distance_dict['Generator'] = epoch_stats_dict['discriminator_fake_predicted_distance'][generator_inds]
    pred_distance_dict['Distorted'] = epoch_stats_dict['discriminator_fake_predicted_distance'][distorted_inds]

    true_distance_dict['CSD'] = epoch_stats_dict['discriminator_real_true_distance']
    true_distance_dict['Gaussian'] = epoch_stats_dict['discriminator_fake_true_distance'][randn_inds]
    true_distance_dict['Generator'] = epoch_stats_dict['discriminator_fake_true_distance'][generator_inds]
    true_distance_dict['Distorted'] = epoch_stats_dict['discriminator_fake_true_distance'][distorted_inds]

    if len(extra_test_dict) > 0:
        scores_dict['extra_test'] = extra_test_dict['discriminator_real_score']
        tracking_features_dict['extra_test'] = {feat: vec for feat, vec in zip(dataDims['tracking_features'],
                                                                               extra_test_dict['tracking_features'].T)}
        vdw_penalty_dict['extra_test'] = extra_test_dict['real_vdw_penalty']
        packing_coeff_dict['extra_test'] = extra_test_dict['real_packing_coefficients']
        pred_distance_dict['extra_test'] = extra_test_dict['discriminator_real_predicted_distance']
        true_distance_dict['extra_test'] = extra_test_dict['discriminator_real_true_distance']

    return scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict, pred_distance_dict, true_distance_dict


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

    bandwidth1, fig, vdw_cutoff, viridis = score_vs_vdw_plot(all_scores, all_vdws, layout,
                                                             plot_color_dict, sample_types, scores_dict,
                                                             scores_range, vdw_penalty_dict)
    fig_dict['Discriminator vs vdw scores'] = fig

    '''
    vs coeff
    '''
    fig = score_vs_packing_plot(all_coeffs, all_scores, bandwidth1, layout, packing_coeff_dict,
                                plot_color_dict, sample_types, scores_dict, viridis)
    fig_dict['Discriminator vs packing coefficient'] = fig

    fig = combined_scores_plot(all_coeffs, all_scores, all_vdws, layout, sample_source, vdw_cutoff)
    fig_dict['Discriminator Scores Analysis'] = fig

    return fig_dict


def score_vs_vdw_plot(all_scores, all_vdws, layout, plot_color_dict, sample_types, scores_dict, scores_range,
                      vdw_penalty_dict):
    bandwidth1 = scores_range / 200
    vdw_cutoff = min(np.quantile(all_vdws, 0.975), 3)
    bandwidth2 = vdw_cutoff / 200
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
        fig.add_trace(go.Violin(x=-np.minimum(vdw_penalty_dict[label], vdw_cutoff), name=legend_label,
                                line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4, meanline_visible=True,
                                bandwidth=bandwidth2, points=False),
                      row=1, col=2)
    # fig.update_xaxes(col=2, row=1, range=[-3, 0])
    rrange = np.logspace(3, 0, len(viridis))
    cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
    cscale[0][0] = 0
    fig.add_trace(go.Histogram2d(x=np.clip(all_scores, a_min=np.quantile(all_scores, 0.05), a_max=np.amax(all_scores)),
                                 y=-np.minimum(all_vdws, vdw_cutoff),
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
    return bandwidth1, fig, vdw_cutoff, viridis


def score_vs_packing_plot(all_coeffs, all_scores, bandwidth1, layout, packing_coeff_dict, plot_color_dict, sample_types,
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
        fig.add_trace(go.Violin(x=np.clip(packing_coeff_dict[label], a_min=0, a_max=1), name=legend_label,
                                line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4, meanline_visible=True,
                                bandwidth=bandwidth2, points=False),
                      row=1, col=2)
    rrange = np.logspace(3, 0, len(viridis))
    cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
    cscale[0][0] = 0
    fig.add_trace(go.Histogram2d(x=np.clip(all_scores, a_min=np.quantile(all_scores, 0.05), a_max=np.amax(all_scores)),
                                 y=np.clip(all_coeffs, a_min=0, a_max=1),
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


def combined_scores_plot(all_coeffs, all_scores, all_vdws, layout, sample_source, vdw_cutoff):
    """
    # All in one
    """
    scatter_dict = {'vdw_score': -all_vdws.clip(max=vdw_cutoff), 'model_score': all_scores,
                    'packing_coefficient': all_coeffs.clip(min=0, max=1), 'sample_source': sample_source}
    df = pd.DataFrame.from_dict(scatter_dict)
    fig = px.scatter(df,
                     x='vdw_score', y='packing_coefficient',
                     color='model_score', symbol='sample_source',
                     marginal_x='histogram', marginal_y='histogram',
                     range_color=(np.amin(all_scores), np.amax(all_scores)),
                     opacity=0.1
                     )
    fig.layout.margin = layout.margin
    fig.update_layout(xaxis_range=[-vdw_cutoff, 0.1], yaxis_range=[0, 1.1])
    fig.update_layout(xaxis_title='vdw score', yaxis_title='packing coefficient')
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=1))
    return fig


def functional_group_analysis_fig(scores_dict, tracking_features, layout, dataDims):
    tracking_features_names = dataDims['tracking_features']
    # get the indices for each functional group
    functional_group_inds = {}
    fraction_dict = {}
    for ii, key in enumerate(tracking_features_names):
        if 'molecule' in key and 'fraction' in key:
            if np.average(tracking_features[:, ii] > 0) > 0.01:
                fraction_dict[key] = np.average(tracking_features[:, ii] > 0)
                functional_group_inds[key] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]
        elif 'molecule' in key and 'count' in key:
            if np.average(tracking_features[:, ii] > 0) > 0.01:
                fraction_dict[key] = np.average(tracking_features[:, ii] > 0)
                functional_group_inds[key] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]

    sort_order = np.argsort(list(fraction_dict.values()))[-1::-1]
    sorted_functional_group_keys = [list(functional_group_inds.keys())[i] for i in sort_order]

    if len(sorted_functional_group_keys) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=[f'{key}_{fraction_dict[key]:.2f}' for key in sorted_functional_group_keys],
                                   y=[np.average(scores_dict['CSD'][functional_group_inds[key]]) for key in
                                      sorted_functional_group_keys],
                                   error_y=dict(type='data',
                                                array=[np.std(scores_dict['CSD'][functional_group_inds[key]]) for key in
                                                       sorted_functional_group_keys],
                                                visible=True
                                                ),
                                   showlegend=False,
                                   mode='markers'))

        fig.update_layout(yaxis_title='Mean Score and Standard Deviation')
        fig.update_layout(width=1600, height=600)
        fig.update_layout(font=dict(size=12))
        fig.layout.margin = layout.margin

        return fig
    else:
        return None


def group_wise_analysis_fig(identifiers_list, crystals_for_targets, scores_dict, normed_scores_dict, layout):
    target_identifiers = {}
    rankings = {}
    group = {}
    list_num = {}
    for label in ['XXII', 'XXIII', 'XXVI']:
        target_identifiers[label] = [identifiers_list[crystals_for_targets[label][n]] for n in
                                     range(len(crystals_for_targets[label]))]
        rankings[label] = []
        group[label] = []
        list_num[label] = []
        for ident in target_identifiers[label]:
            if 'edited' in ident:
                ident = ident[7:]

            long_ident = ident.split('_')
            list_num[label].append(int(ident[len(label) + 1]))
            rankings[label].append(int(long_ident[-1]) + 1)
            rankings[label].append(int(long_ident[-1]) + 1)
            group[label].append(long_ident[1])

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        ['Brandenburg XXII', 'Brandenburg XXIII', 'Brandenburg XXVI']),  # , 'Facelli XXII']),
                        x_title='Model Score')

    quantiles = [np.quantile(normed_scores_dict['CSD'], 0.01), np.quantile(normed_scores_dict['CSD'], 0.05),
                 np.quantile(normed_scores_dict['CSD'], 0.1)]

    for ii, label in enumerate(['XXII', 'XXIII', 'XXVI']):
        good_inds = np.where(np.asarray(group[label]) == 'Brandenburg')[0]
        submissions_list_num = np.asarray(list_num[label])[good_inds]
        list1_inds = np.where(submissions_list_num == 1)[0]
        list2_inds = np.where(submissions_list_num == 2)[0]

        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 1 Score",
                                   showlegend=False,
                                   marker_color='#0c4dae'),
                      row=1, col=ii + 1)  # row=(ii) // 2 + 1, col=(ii) % 2 + 1)

        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 2 Score",
                                   showlegend=False,
                                   marker_color='#d60000'),
                      row=1, col=ii + 1)  # row=(ii) // 2 + 1, col=(ii) % 2 + 1)

    # label = 'XXII'
    # good_inds = np.where(np.asarray(group[label]) == 'Facelli')[0]
    # submissions_list_num = np.asarray(list_num[label])[good_inds]
    # list1_inds = np.where(submissions_list_num == 1)[0]
    # list2_inds = np.where(submissions_list_num == 2)[0]
    #
    # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
    #                            histnorm='probability density',
    #                            nbinsx=50,
    #                            name="Submission 1 Score",
    #                            showlegend=False,
    #                            marker_color='#0c4dae'), row=2, col=2)
    # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
    #                            histnorm='probability density',
    #                            nbinsx=50,
    #                            name="Submission 2 Score",
    #                            showlegend=False,
    #                            marker_color='#d60000'), row=2, col=2)

    fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=1, col=1)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=1, col=2)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=1, col=3)
    # fig.add_vline(x=quantiles[1], line_dash='dash', line_color='black', row=2, col=2)

    fig.update_layout(width=1000, height=300)
    fig.layout.margin = layout.margin
    fig.layout.margin.b = 50
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

    wandb.log({'Generator Loss Correlates': fig})


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


def get_BT_identifiers_inds(extra_test_dict):
    # determine which samples go with which targets
    crystals_for_targets = {key: [] for key in blind_test_targets}
    for i in range(len(extra_test_dict['identifiers'])):
        item = extra_test_dict['identifiers'][i]
        for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
            if blind_test_targets[-1 - j] in item:
                crystals_for_targets[blind_test_targets[-1 - j]].append(i)
                break

    # determine which samples ARE the targets (mixed in the dataloader)
    target_identifiers_inds = {key: [] for key in blind_test_targets}
    for i, item in enumerate(extra_test_dict['identifiers']):
        for key in target_identifiers.keys():
            if item == target_identifiers[key]:
                target_identifiers_inds[key] = i

    return crystals_for_targets, target_identifiers_inds


def get_BT_scores(crystals_for_targets, target_identifiers_inds, extra_test_dict, tracking_features_dict, scores_dict,
                  vdw_penalty_dict, distance_dict, dataDims, test_epoch_stats_dict):
    bt_score_correlates = {}
    for target in crystals_for_targets.keys():  # run the analysis for each target
        if target_identifiers_inds[target] != []:  # record target data

            target_index = target_identifiers_inds[target]
            scores = extra_test_dict['discriminator_real_score'][target_index]
            scores_dict[target + '_exp'] = scores[None]

            tracking_features_dict[target + '_exp'] = {feat: vec for feat, vec in zip(dataDims['tracking_features'],
                                                                                      extra_test_dict[
                                                                                          'tracking_features'][
                                                                                          target_index][None, :].T)}

            vdw_penalty_dict[target + '_exp'] = extra_test_dict['real_vdw_penalty'][target_index][None]

            distance_dict[target + '_exp'] = extra_test_dict['discriminator_real_predicted_distance'][target_index][
                None]

            wandb.log({f'Average_{target}_exp_score': np.average(scores)})

        if crystals_for_targets[target] != []:  # record sample data
            target_indices = crystals_for_targets[target]
            scores = extra_test_dict['discriminator_real_score'][target_indices]
            scores_dict[target] = scores
            tracking_features_dict[target] = {feat: vec for feat, vec in zip(dataDims['tracking_features'],
                                                                             extra_test_dict['tracking_features'][
                                                                                 target_indices].T)}

            vdw_penalty_dict[target] = extra_test_dict['real_vdw_penalty'][target_indices]

            distance_dict[target] = extra_test_dict['discriminator_real_predicted_distance'][target_indices]

            wandb.log({f'Average_{target}_score': np.average(scores)})
            wandb.log({f'Average_{target}_std': np.std(scores)})

            # correlate losses with molecular features
            tracking_features = np.asarray(extra_test_dict['tracking_features'])
            loss_correlations = np.zeros(dataDims['num_tracking_features'])
            features = []
            for j in range(tracking_features.shape[-1]):  # not that interesting
                features.append(dataDims['tracking_features'][j])
                loss_correlations[j] = np.corrcoef(scores, tracking_features[target_indices, j], rowvar=False)[0, 1]

            bt_score_correlates[target] = loss_correlations

    # compute loss correlates
    loss_correlations = np.zeros(dataDims['num_tracking_features'])
    features = []
    for j in range(dataDims['num_tracking_features']):  # not that interesting
        features.append(dataDims['tracking_features'][j])
        loss_correlations[j] = \
            np.corrcoef(scores_dict['CSD'], test_epoch_stats_dict['tracking_features'][:, j], rowvar=False)[0, 1]
    bt_score_correlates['CSD'] = loss_correlations

    return scores_dict, vdw_penalty_dict, distance_dict, bt_score_correlates


def process_BT_evaluation_outputs(dataDims, wandb, extra_test_dict, test_epoch_stats_dict):
    crystals_for_targets, target_identifiers_inds = get_BT_identifiers_inds(extra_test_dict)

    '''
    record all the stats for the usual test dataset
    '''
    (scores_dict, vdw_penalty_dict,
     tracking_features_dict, packing_coeff_dict,
     pred_distance_dict, true_distance_dict) \
        = process_discriminator_outputs(dataDims, test_epoch_stats_dict, extra_test_dict)

    '''
    build property dicts for the submissions and BT targets
    '''
    scores_dict, vdw_penalty_dict, pred_distance_dict, bt_score_correlates = (
        get_BT_scores(
            crystals_for_targets, target_identifiers_inds, extra_test_dict,
            tracking_features_dict, scores_dict, vdw_penalty_dict,
            pred_distance_dict, dataDims, test_epoch_stats_dict))

    # collect all BT targets & submissions into single dicts
    BT_target_scores = np.asarray([scores_dict[key] for key in scores_dict.keys() if 'exp' in key])
    BT_submission_scores = np.concatenate(
        [scores_dict[key] for key in scores_dict.keys() if key in crystals_for_targets.keys()])
    BT_target_distances = np.asarray([pred_distance_dict[key] for key in pred_distance_dict.keys() if 'exp' in key])
    BT_submission_distances = np.concatenate(
        [pred_distance_dict[key] for key in pred_distance_dict.keys() if key in crystals_for_targets.keys()])

    wandb.log({'Average BT submission score': np.average(BT_submission_scores)})
    wandb.log({'Average BT target score': np.average(BT_target_scores)})
    wandb.log({'BT submission score std': np.std(BT_target_scores)})
    wandb.log({'BT target score std': np.std(BT_target_scores)})

    wandb.log({'Average BT submission distance': np.average(BT_submission_distances)})
    wandb.log({'Average BT target distance': np.average(BT_target_distances)})
    wandb.log({'BT submission distance std': np.std(BT_target_distances)})
    wandb.log({'BT target distance std': np.std(BT_target_distances)})

    return bt_score_correlates, scores_dict, pred_distance_dict, \
        crystals_for_targets, blind_test_targets, target_identifiers, target_identifiers_inds, \
        BT_target_scores, BT_submission_scores, \
        vdw_penalty_dict, tracking_features_dict, \
        BT_target_distances, BT_submission_distances


def BT_separation_tables(layout, scores_dict, BT_submission_scores, crystals_for_targets, normed_scores_dict,
                         normed_BT_submission_scores):
    vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    quantiles = np.quantile(scores_dict['CSD'], vals)
    submissions_fraction_below_csd_quantile = {value: np.average(BT_submission_scores < cutoff) for value, cutoff in
                                               zip(vals, quantiles)}

    normed_quantiles = np.quantile(normed_scores_dict['CSD'], vals)
    normed_submissions_fraction_below_csd_quantile = {value: np.average(normed_BT_submission_scores < cutoff) for
                                                      value, cutoff in zip(vals, normed_quantiles)}

    submissions_fraction_below_target = {key: np.average(scores_dict[key] < scores_dict[key + '_exp']) for key in
                                         crystals_for_targets.keys() if key in scores_dict.keys()}
    submissions_average_below_target = np.average(list(submissions_fraction_below_target.values()))

    fig1 = go.Figure(data=go.Table(
        header=dict(values=['CSD Test Quantile', 'Fraction of Submissions']),
        cells=dict(values=[list(submissions_fraction_below_csd_quantile.keys()),
                           list(submissions_fraction_below_csd_quantile.values()),
                           ], format=[".3", ".3"])))
    fig1.update_layout(width=200)
    fig1.layout.margin = layout.margin
    fig1.write_image('scores_separation_table.png', scale=4)

    fig2 = go.Figure(data=go.Table(
        header=dict(values=['CSD Test Quantile(normed)', 'Fraction of Submissions (normed)']),
        cells=dict(values=[list(normed_submissions_fraction_below_csd_quantile.keys()),
                           list(normed_submissions_fraction_below_csd_quantile.values()),
                           ], format=[".3", ".3"])))
    fig2.update_layout(width=200)
    fig2.layout.margin = layout.margin
    fig2.write_image('normed_scores_separation_table.png', scale=4)

    wandb.log({"Scores Separation": submissions_fraction_below_csd_quantile})
    wandb.log({"Normed Scores Separation": normed_submissions_fraction_below_csd_quantile})

    return fig1, fig2


def make_and_plot_BT_figs(crystals_for_targets, target_identifiers_inds, identifiers_list,
                          scores_dict, BT_target_scores, BT_submission_scores,
                          tracking_features_dict, layout, tracking_features, dataDims, score_name):
    """generate and log the various BT analyses"""

    '''score distributions'''
    fig, normed_scores_dict, normed_BT_submission_scores, normed_BT_target_scores = (
        blind_test_scores_distributions_fig(
            crystals_for_targets, target_identifiers_inds,
            scores_dict, BT_target_scores, BT_submission_scores,
            tracking_features_dict, layout))
    fig.write_image(f'bt_submissions_{score_name}_distribution.png', scale=4)
    wandb.log({f"BT Submissions {score_name} Distribution": fig})

    '''score separations'''
    fig1, fig2 = BT_separation_tables(layout, scores_dict, BT_submission_scores,
                                      crystals_for_targets, normed_scores_dict, normed_BT_submission_scores)
    wandb.log({f"{score_name} Separation Table": fig1,
               f"Normed {score_name} Separation Table": fig2})

    '''functional group analysis'''
    fig = functional_group_analysis_fig(scores_dict, tracking_features, layout, dataDims)
    if fig is not None:
        fig.write_image(f'functional_group_{score_name}.png', scale=2)
        wandb.log({f"Functional Group {score_name}": fig})

    fig = group_wise_analysis_fig(identifiers_list, crystals_for_targets, scores_dict, normed_scores_dict, layout)
    fig.write_image(f'interesting_groups_{score_name}.png', scale=4)
    wandb.log({f"Interesting Groups {score_name}": fig})

    '''
    S2.  score correlates
    '''
    fig = make_correlates_plot(tracking_features, scores_dict['CSD'], dataDims)
    fig.write_image(f'{score_name}_correlates.png', scale=4)
    wandb.log({f"{score_name} Correlates": fig})

    '''
    S1. All group-wise analysis  # todo rebuild
    '''
    #
    # for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
    #     names = np.unique(list(group[label]))
    #     uniques = len(names)
    #     rows = int(np.floor(np.sqrt(uniques)))
    #     cols = int(np.ceil(np.sqrt(uniques)) + 1)
    #     fig = make_subplots(rows=rows, cols=cols,
    #                         subplot_titles=(names), x_title='Group Ranking', y_title='Model Score', vertical_spacing=0.1)
    #
    #     for j, group_name in enumerate(np.unique(group[label])):
    #         good_inds = np.where(np.asarray(group[label]) == group_name)[0]
    #         submissions_list_num = np.asarray(list_num[label])[good_inds]
    #         list1_inds = np.where(submissions_list_num == 1)[0]
    #         list2_inds = np.where(submissions_list_num == 2)[0]
    #
    #         xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list1_inds]])])
    #         linreg_result = linregress(np.asarray(rankings[label])[good_inds[list1_inds]], np.asarray(scores_dict[label])[good_inds[list1_inds]])
    #         yline = xline * linreg_result.slope + linreg_result.intercept
    #
    #         fig.add_trace(go.Scattergl(x=np.asarray(rankings[label])[good_inds], y=np.asarray(scores_dict[label])[good_inds], showlegend=False,
    #                                    mode='markers', opacity=0.5, marker=dict(size=6, color=submissions_list_num, colorscale='portland', cmax=2, cmin=1, showscale=False)),
    #                       row=j // cols + 1, col=j % cols + 1)
    #
    #         fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{group_name} R={linreg_result.rvalue:.3f}', line=dict(color='#0c4dae')), row=j // cols + 1, col=j % cols + 1)
    #
    #         if len(list2_inds) > 0:
    #             xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list2_inds]])])
    #             linreg_result2 = linregress(np.asarray(rankings[label])[good_inds[list2_inds]], np.asarray(scores_dict[label])[good_inds[list2_inds]])
    #             yline2 = xline * linreg_result2.slope + linreg_result2.intercept
    #             fig.add_trace(go.Scattergl(x=xline, y=yline2, name=f'{group_name} R={linreg_result2.rvalue:.3f}', line=dict(color='#d60000')), row=j // cols + 1, col=j % cols + 1)
    #
    #     fig.update_layout(title=label)
    #
    #     fig.update_layout(width=1200, height=600)
    #     fig.layout.margin = layout.margin
    #     fig.layout.margin.t = 50
    #     fig.layout.margin.b = 55
    #     fig.layout.margin.l = 60
    #     fig.write_image(f'groupwise_analysis_{i}.png', scale=4)


def discriminator_BT_reporting(dataDims, wandb, test_epoch_stats_dict, extra_test_dict):
    tracking_features = test_epoch_stats_dict['tracking_features']
    identifiers_list = extra_test_dict['identifiers']

    (bt_score_correlates, scores_dict, pred_distance_dict,
     crystals_for_targets, blind_test_targets,
     target_identifiers, target_identifiers_inds,
     BT_target_scores, BT_submission_scores,
     vdw_penalty_dict, tracking_features_dict,
     BT_target_distances, BT_submission_distances) = \
        process_BT_evaluation_outputs(
            dataDims, wandb, extra_test_dict, test_epoch_stats_dict)

    layout = go.Layout(  # todo increase upper margin - figure titles are getting cutoff
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=80,  # top margin
        )
    )

    make_and_plot_BT_figs(crystals_for_targets, target_identifiers_inds, identifiers_list,
                          scores_dict, BT_target_scores, BT_submission_scores,
                          tracking_features_dict, layout, tracking_features, dataDims, score_name='score')

    dist2score = lambda x: -np.log10(10 ** x - 1)
    distance_score_dict = {key: dist2score(value) for key, value in pred_distance_dict.items()}
    BT_target_dist_scores = dist2score(BT_target_scores)
    BT_submission_dist_scores = dist2score(BT_submission_distances)

    make_and_plot_BT_figs(crystals_for_targets, target_identifiers_inds, identifiers_list,
                          distance_score_dict, BT_target_dist_scores, BT_submission_dist_scores,
                          tracking_features_dict, layout, tracking_features, dataDims, score_name='distance')

    return None


def blind_test_scores_distributions_fig(crystals_for_targets, target_identifiers_inds, scores_dict, BT_target_scores,
                                        BT_submission_scores, tracking_features_dict, layout):
    lens = [len(val) for val in crystals_for_targets.values()]
    targets_list = list(target_identifiers_inds.values())
    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)',
                      max(np.count_nonzero(lens), sum([1 for ll in targets_list if ll != []])), colortype='rgb')

    plot_color_dict = {'Train Real': 'rgb(250,50,50)',
                       'CSD': 'rgb(250,150,50)',
                       'Gaussian': 'rgb(0,50,0)',
                       'Distorted': 'rgb(0,100,100)'}
    ind = 0
    for target in crystals_for_targets.keys():
        if crystals_for_targets[target] != []:
            plot_color_dict[target] = colors[ind]
            plot_color_dict[target + '_exp'] = colors[ind]
            ind += 1

    # plot 1
    scores_range = np.ptp(scores_dict['CSD'])
    bandwidth = scores_range / 200

    fig = make_subplots(cols=2, rows=2, horizontal_spacing=0.15, subplot_titles=('a)', 'b)', 'c)'),
                        specs=[[{"rowspan": 2}, {}], [None, {}]], vertical_spacing=0.12)
    fig.layout.annotations[0].update(x=0.025)
    fig.layout.annotations[1].update(x=0.525)
    fig.layout.annotations[2].update(x=0.525)
    scores_labels = {'CSD': 'CSD Test', 'Gaussian': 'Gaussian', 'Distorted': 'Distorted'}

    for i, label in enumerate(scores_dict.keys()):
        if label in plot_color_dict.keys():

            if label in scores_labels.keys():
                name_label = scores_labels[label]
            else:
                name_label = label
            if 'exp' in label:
                fig.add_trace(
                    go.Violin(x=scores_dict[label], name=name_label, line_color=plot_color_dict[label], side='positive',
                              orientation='h', width=6),
                    row=1, col=1)
            else:
                fig.add_trace(
                    go.Violin(x=scores_dict[label], name=name_label, line_color=plot_color_dict[label], side='positive',
                              orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False),
                    row=1, col=1)

    good_scores = np.concatenate(
        [score for i, (key, score) in enumerate(scores_dict.items()) if key not in ['Gaussian', 'Distorted']])
    fig.update_xaxes(range=[np.amin(good_scores), np.amax(good_scores)], row=1, col=1)

    # plot2 inset
    plot_color_dict = {}
    plot_color_dict['CSD'] = ('rgb(200,0,50)')  # test
    plot_color_dict['BT Targets'] = ('rgb(50,0,50)')
    plot_color_dict['BT Submissions'] = ('rgb(50,150,250)')

    scores_range = np.ptp(scores_dict['CSD'])
    bandwidth = scores_range / 200

    # test data
    fig.add_trace(go.Violin(x=scores_dict['CSD'], name='CSD Test',
                            line_color=plot_color_dict['CSD'], side='positive', orientation='h', width=1.5,
                            meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

    # BT distribution
    fig.add_trace(go.Violin(x=BT_target_scores, name='BT Targets',
                            line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5,
                            meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=1, col=2)
    # Submissions
    fig.add_trace(go.Violin(x=BT_submission_scores, name='BT Submissions',
                            line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5,
                            meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

    quantiles = [np.quantile(scores_dict['CSD'], 0.01), np.quantile(scores_dict['CSD'], 0.05),
                 np.quantile(scores_dict['CSD'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['CSD'], row=1, col=2)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['CSD'], row=1, col=2)
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['CSD'], row=1, col=2)

    normed_scores_dict = scores_dict.copy()
    for key in normed_scores_dict.keys():
        normed_scores_dict[key] = normed_scores_dict[key] / tracking_features_dict[key]['molecule_num_atoms']

    normed_BT_target_scores = np.concatenate(
        [normed_scores_dict[key] for key in normed_scores_dict.keys() if 'exp' in key])
    normed_BT_submission_scores = np.concatenate(
        [normed_scores_dict[key] for key in normed_scores_dict.keys() if key in crystals_for_targets.keys()])
    scores_range = np.ptp(normed_scores_dict['CSD'])
    bandwidth = scores_range / 200

    # test data
    fig.add_trace(go.Violin(x=normed_scores_dict['CSD'], name='CSD Test',
                            line_color=plot_color_dict['CSD'], side='positive', orientation='h', width=1.5,
                            meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

    # BT distribution
    fig.add_trace(go.Violin(x=normed_BT_target_scores, name='BT Targets',
                            line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5,
                            meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=2, col=2)
    # Submissions
    fig.add_trace(go.Violin(x=normed_BT_submission_scores, name='BT Submissions',
                            line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5,
                            meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

    quantiles = [np.quantile(normed_scores_dict['CSD'], 0.01), np.quantile(normed_scores_dict['CSD'], 0.05),
                 np.quantile(normed_scores_dict['CSD'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['CSD'], row=2, col=2)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['CSD'], row=2, col=2)
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['CSD'], row=2, col=2)

    fig.update_layout(showlegend=False, yaxis_showgrid=True, width=1000, height=500)
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_xaxes(title_text='Model Score', row=1, col=2)
    fig.update_xaxes(title_text='Model Score', row=1, col=1)
    fig.update_xaxes(title_text='Model Score / molecule # atoms', row=2, col=2)

    fig.layout.margin = layout.margin
    return fig, normed_scores_dict, normed_BT_submission_scores, normed_BT_target_scores


def log_cubic_defect(samples):
    cleaned_samples = samples
    cubic_distortion = np.abs(1 - np.nan_to_num(np.stack(
        [cell_vol(cleaned_samples[i, 0:3], cleaned_samples[i, 3:6]) / np.prod(cleaned_samples[i, 0:3], axis=-1) for
         i in range(len(cleaned_samples))])))
    wandb.log({'Avg generated cubic distortion': np.average(cubic_distortion)})
    hist = np.histogram(cubic_distortion, bins=256, range=(0, 1))
    wandb.log({"Generated cubic distortions": wandb.Histogram(np_histogram=hist, num_bins=256)})


def process_generator_losses(config, epoch_stats_dict):
    generator_loss_keys = ['generator_packing_prediction', 'generator_packing_target', 'generator_per_mol_vdw_loss',
                           'generator_adversarial_loss', 'generator h bond loss']
    generator_losses = {}
    for key in generator_loss_keys:
        if key in epoch_stats_dict.keys():
            if epoch_stats_dict[key] is not None:
                if key == 'generator_adversarial_loss':
                    if config.generator.train_adversarially:
                        generator_losses[key[10:]] = epoch_stats_dict[key]
                    else:
                        pass
                else:
                    generator_losses[key[10:]] = epoch_stats_dict[key]

                if key == 'generator_packing_target':
                    generator_losses['packing normed mae'] = np.abs(
                        generator_losses['packing_prediction'] - generator_losses['packing_target']) / \
                                                             generator_losses['packing_target']
                    del generator_losses['packing_prediction'], generator_losses['packing_target']
            else:
                generator_losses[key[10:]] = None

    return generator_losses, {key: np.average(value) for i, (key, value) in enumerate(generator_losses.items()) if
                              value is not None}


def cell_generation_analysis(config, dataDims, epoch_stats_dict):
    """
    do analysis and plotting for cell generator
    """
    layout = plotly_setup(config)
    if isinstance(epoch_stats_dict['final_generated_cell_parameters'], list):
        cell_parameters = np.stack(epoch_stats_dict['final_generated_cell_parameters'])
    else:
        cell_parameters = epoch_stats_dict['final_generated_cell_parameters']
    log_cubic_defect(cell_parameters)
    wandb.log({"Generated cell parameter variation": cell_parameters.std(0).mean()})
    generator_losses, average_losses_dict = process_generator_losses(config, epoch_stats_dict)
    wandb.log(average_losses_dict)

    cell_density_plot(config, wandb, epoch_stats_dict, layout)
    plot_generator_loss_correlates(dataDims, wandb, epoch_stats_dict, generator_losses, layout)
    cell_scatter(epoch_stats_dict, wandb, layout,
                 num_atoms_index=dataDims['tracking_features'].index('molecule_num_atoms'),
                 extra_category='generated_space_group_numbers')

    return None


def cell_scatter(epoch_stats_dict, wandb, layout, num_atoms_index, extra_category=None):
    model_scores = epoch_stats_dict['generator_adversarial_score']
    scatter_dict = {'vdw_score': epoch_stats_dict['generator_per_mol_vdw_score'],
                    'model_score': model_scores,
                    'volume_per_atom': epoch_stats_dict['generator_packing_prediction']
                                       / epoch_stats_dict['tracking_features'][:, num_atoms_index]}
    if extra_category is not None:
        scatter_dict[extra_category] = epoch_stats_dict[extra_category]

    vdw_cutoff = max(-1.5, np.amin(scatter_dict['vdw_score']))
    opacity = max(0.1, 1 - len(model_scores) / 5e4)
    df = pd.DataFrame.from_dict(scatter_dict)
    if extra_category is not None:
        fig = px.scatter(df,
                         x='vdw_score', y='volume_per_atom',
                         color='model_score', symbol=extra_category,
                         marginal_x='histogram', marginal_y='histogram',
                         range_color=(np.amin(model_scores), np.amax(model_scores)),
                         opacity=opacity
                         )
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=1))

    else:
        fig = px.scatter(df,
                         x='vdw_score', y='volume_per_atom',
                         color='model_score',
                         marginal_x='histogram', marginal_y='histogram',
                         opacity=opacity
                         )
    fig.layout.margin = layout.margin
    fig.update_layout(xaxis_title='vdw score', yaxis_title='Reduced Volume')
    fig.update_layout(xaxis_range=[vdw_cutoff, 0.1],
                      yaxis_range=[scatter_dict['volume_per_atom'].min(), scatter_dict['volume_per_atom'].max()])
    wandb.log({'Generator Samples': fig})


def log_regression_accuracy(config, dataDims, epoch_stats_dict):
    target_key = config.dataset.regression_target

    target = np.asarray(epoch_stats_dict['regressor_target'])
    prediction = np.asarray(epoch_stats_dict['regressor_prediction'])

    tgt_value = target
    pred_value = prediction
    losses = ['abs_error', 'abs_normed_error', 'squared_error']
    loss_dict = {}
    fig_dict = {}
    fig = make_subplots(cols=2, rows=1)
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

        linreg_result = linregress(tgt_value, pred_value)
        loss_dict['regression_R_value'] = linreg_result.rvalue
        loss_dict['regression_slope'] = linreg_result.slope

    # predictions vs target trace
    xline = np.linspace(max(min(tgt_value), min(pred_value)),
                        min(max(tgt_value), max(pred_value)), 2)

    xy = np.vstack([tgt_value, pred_value])
    try:
        z = get_point_density(xy)
    except:
        z = np.ones_like(tgt_value)

    num_points = len(pred_value)
    opacity = np.exp(-num_points / 10000)
    fig.add_trace(go.Scattergl(x=tgt_value, y=pred_value, mode='markers', marker=dict(color=z), opacity=opacity,
                               showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                  row=1, col=1)

    fig.add_trace(go.Histogram(x=pred_value - tgt_value,
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

    fig_dict['Regression Results'] = fig

    wandb.log(loss_dict)
    wandb.log(fig_dict)

    return None


def make_correlates_plot(tracking_features, values, dataDims):
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
                corr = np.corrcoef(values, tracking_features[:, i], rowvar=False)[0, 1]
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

    loss_correlates_dict = {feat: corr for feat, corr in zip(features_sorted, g_loss_correlations)}

    if len(loss_correlates_dict) != 0:

        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=(
            'a) Molecule & Crystal Features', 'b) Atom Fractions', 'c) Functional Groups Count'), x_title='R Value')

        crystal_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' not in key]
        atom_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' in key]
        mol_keys = [key for key in features_sorted_cleaned if 'count' in key and 'fraction' not in key]

        fig.add_trace(go.Bar(
            y=crystal_keys,
            x=[g for i, (feat, g) in enumerate(loss_correlates_dict.items()) if feat in crystal_keys],
            orientation='h',
            text=np.asarray(
                [g for i, (feat, g) in enumerate(loss_correlates_dict.items()) if feat in crystal_keys]).astype(
                'float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(100,0,0,1)')
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            y=[feat.replace('molecule_', '') for feat in features_sorted_cleaned if feat in atom_keys],
            x=[g for i, (feat, g) in enumerate(loss_correlates_dict.items()) if feat in atom_keys],
            orientation='h',
            text=np.asarray(
                [g for i, (feat, g) in enumerate(loss_correlates_dict.items()) if feat in atom_keys]).astype('float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(0,0,100,1)')
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            y=[feat.replace('molecule_', '').replace('_count', '') for feat in features_sorted_cleaned if
               feat in mol_keys],
            x=[g for i, (feat, g) in enumerate(loss_correlates_dict.items()) if feat in mol_keys],
            orientation='h',
            text=np.asarray([g for i, (feat, g) in enumerate(loss_correlates_dict.items()) if feat in mol_keys]).astype(
                'float16'),
            textposition='auto',
            texttemplate='%{text:.2}',
            marker=dict(color='rgba(0,100,0,1)')
        ), row=1, col=3)

        fig.update_yaxes(tickfont=dict(size=14), row=1, col=1)
        fig.update_yaxes(tickfont=dict(size=14), row=1, col=2)
        fig.update_yaxes(tickfont=dict(size=14), row=1, col=3)

        fig.layout.annotations[0].update(x=0.12)
        fig.layout.annotations[1].update(x=0.45)
        fig.layout.annotations[2].update(x=0.88)

        fig.update_xaxes(
            range=[np.amin(list(loss_correlates_dict.values())), np.amax(list(loss_correlates_dict.values()))])
        # fig.update_layout(width=1200, height=400)
        fig.update_layout(showlegend=False)

    else:
        fig = go.Figure()  # empty figure

    return fig


def detailed_reporting(config, dataDims, test_loader, train_epoch_stats_dict, test_epoch_stats_dict,
                       extra_test_dict=None):
    """
    Do analysis and upload results to w&b
    """
    # rec = np.load(r'C:\Users\mikem\crystals\CSP_runs\_experiments_dev_12-11-13-36-50/multi_discriminator_stats_dicts.npy', allow_pickle=True)
    # test_epoch_stats_dict = rec[1]
    # extra_test_dict = rec[2]
    # dataDims = rec[0]
    if test_epoch_stats_dict is not None:
        if config.mode == 'gan' or config.mode == 'discriminator':
            if 'final_generated_cell_parameters' in test_epoch_stats_dict.keys():
                cell_params_analysis(config, dataDims, wandb, test_loader, test_epoch_stats_dict)

            if config.generator.train_vdw or config.generator.train_adversarially:
                cell_generation_analysis(config, dataDims, test_epoch_stats_dict)

            if config.discriminator.train_on_distorted or config.discriminator.train_on_randn or config.discriminator.train_adversarially:
                discriminator_analysis(config, dataDims, test_epoch_stats_dict, extra_test_dict)

        elif config.mode == 'regression' or config.mode == 'embedding_regression':
            log_regression_accuracy(config, dataDims, test_epoch_stats_dict)

        elif config.mode == 'autoencoder':
            log_autoencoder_analysis(config, dataDims, train_epoch_stats_dict,
                                     'train', config.autoencoder.molecule_radius_normalization)
            log_autoencoder_analysis(config, dataDims, test_epoch_stats_dict,
                                     'test', config.autoencoder.molecule_radius_normalization)

        elif config.mode == 'polymorph_classification':
            classifier_reporting(true_labels=train_epoch_stats_dict['true_labels'],
                                 probs=np.stack(train_epoch_stats_dict['probs']),
                                 ordered_class_names=polymorph2form['acridine'],
                                 wandb=wandb,
                                 epoch_type='train')
            classifier_reporting(true_labels=np.stack(test_epoch_stats_dict['true_labels']),
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

    type_probs = softmax_np(probs[:, list(present_classes.astype(int))])
    predicted_class = np.argmax(type_probs, axis=1)

    train_score = roc_auc_score(true_labels, type_probs, multi_class='ovo')
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
               f"{epoch_type} Confusion Matrix": fig})


def log_autoencoder_analysis(config, dataDims, epoch_stats_dict, epoch_type, molecule_radius_normalization):
    """
    analyze performance of autoencoder model
    Parameters
    ----------
    config
    dataDims
    epoch_stats_dict
    epoch_type
    molecule_radius_normalization

    Returns
    -------

    """
    allowed_types = np.array(dataDims['allowed_atom_types'])
    type_translation_index = np.zeros(allowed_types.max() + 1) - 1
    for ind, atype in enumerate(allowed_types):
        type_translation_index[atype] = ind
    autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

    # get samples
    data = epoch_stats_dict['sample'][0]
    data.x = autoencoder_type_index[data.x[:, 0].long()]
    decoded_data = epoch_stats_dict['decoded_sample'][0]

    # compute various distribution overlaps
    (coord_overlap, full_overlap,
     self_coord_overlap, self_overlap, self_type_overlap, type_overlap) = (
        autoencoder_decoder_sample_validation(data, decoded_data, config, dataDims, epoch_stats_dict))
    overall_overlap = scatter(full_overlap / self_overlap, data.batch, reduce='mean').cpu().detach().numpy()
    evaluation_overlap_loss = scatter(F.smooth_l1_loss(self_overlap, full_overlap, reduction='none'), data.batch,
                                      reduce='mean')

    wandb.log({epoch_type + "_evaluation_positions_wise_overlap":
                   scatter(coord_overlap / self_coord_overlap,data.batch,reduce='mean').mean().cpu().detach().numpy(),
               epoch_type + "_evaluation_typewise_overlap":
                   scatter(type_overlap / self_type_overlap, data.batch,reduce='mean').mean().cpu().detach().numpy(),
               epoch_type + "_evaluation_overall_overlap": overall_overlap.mean(),
               epoch_type + "_evaluation_matching_clouds_fraction": (np.sum(1 - overall_overlap) < 0.01).mean(),
               epoch_type + "_evaluation_overlap_loss": evaluation_overlap_loss.mean().cpu().detach().numpy(),
               })

    if config.logger.log_figures:
        fig, fig2, rmsd, max_dist, tot_overlap = (
            gaussian_3d_overlap_plots(data, decoded_data,
                                      dataDims['num_atom_types'],
                                      molecule_radius_normalization))
        wandb.log({
            epoch_type + "_pointwise_sample_distribution": fig,
            epoch_type + "_cluster_sample_distribution": fig2,
            epoch_type + "_sample_RMSD": rmsd,
            epoch_type + "_max_dist": max_dist,
            epoch_type + "_probability_mass_overlap": tot_overlap,
        })

    return None


def proxy_discriminator_analysis(epoch_stats_dict):
    tgt_value = np.concatenate(
        (epoch_stats_dict['discriminator_real_score'], epoch_stats_dict['discriminator_fake_score']))
    pred_value = np.concatenate((epoch_stats_dict['proxy_real_score'], epoch_stats_dict['proxy_fake_score']))

    num_real_samples = len(epoch_stats_dict['discriminator_real_score'])
    generator_inds = np.where(epoch_stats_dict['generator_sample_source'] == 0)[0] + num_real_samples
    randn_inds = np.where(epoch_stats_dict['generator_sample_source'] == 1)[0] + num_real_samples
    distorted_inds = np.where(epoch_stats_dict['generator_sample_source'] == 2)[0] + num_real_samples
    csd_inds = np.arange(num_real_samples)

    linreg_result = linregress(tgt_value, pred_value)

    # predictions vs target trace
    xline = np.linspace(max(min(tgt_value), min(pred_value)),
                        min(max(tgt_value), max(pred_value)), 2)

    xy = np.vstack([tgt_value, pred_value])
    try:
        z = get_point_density(xy)
    except:
        z = np.ones_like(tgt_value)

    fig = go.Figure()
    opacity = max(0.1, 1 - len(tgt_value) / 5e4)
    for inds, name, symbol in zip([csd_inds, randn_inds, distorted_inds, generator_inds],
                                  ['CSD', 'Gaussian', 'Distorted', 'Generated'],
                                  ['circle', 'square', 'diamond', 'cross']):
        fig.add_trace(go.Scattergl(x=tgt_value[inds], y=pred_value[inds], mode='markers', marker=dict(color=z[inds]),
                                   opacity=opacity, showlegend=True, name=name, marker_symbol=symbol),
                      )

    fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                  )

    fig.update_layout(xaxis_title='Discriminator Score', yaxis_title='Proxy Score')

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    wandb.log({"Proxy Results": fig,
               "proxy_R_value": linreg_result.rvalue,
               "proxy_slope": linreg_result.slope})


def discriminator_analysis(config, dataDims, epoch_stats_dict, extra_test_dict=None):
    """
    do analysis and plotting for cell discriminator

    -: scores distribution and vdw penalty by sample source
    -: loss correlates
    """
    fig_dict = {}
    layout = plotly_setup(config)
    scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict, pred_distance_dict, true_distance_dict \
        = process_discriminator_outputs(dataDims, epoch_stats_dict, extra_test_dict)

    fig_dict.update(discriminator_scores_plots(scores_dict, vdw_penalty_dict, packing_coeff_dict, layout))
    fig_dict['Discriminator Score Correlates'] = plot_discriminator_score_correlates(dataDims, epoch_stats_dict, layout)
    fig_dict['Distance Results'], dist_rvalue, dist_slope = discriminator_distances_plots(
        pred_distance_dict, true_distance_dict, epoch_stats_dict)

    fig_dict['Score vs. Distance'] = score_vs_distance_plot(pred_distance_dict, scores_dict)

    # img_dict = {key: wandb.Image(fig) for key, fig in fig_dict.items()}
    wandb.log(fig_dict)
    wandb.log({"distance_R_value": dist_rvalue,
               "distance_slope": dist_slope})

    return None


def score_vs_distance_plot(pred_distance_dict, scores_dict):
    sample_types = list(scores_dict.keys())

    fig = make_subplots(cols=2, rows=1)
    x = np.concatenate([scores_dict[stype] for stype in sample_types])
    y = np.concatenate([pred_distance_dict[stype] for stype in sample_types])
    xy = np.vstack([x, y])
    try:
        z = get_point_density(xy, bins=200)
    except:
        z = np.ones(len(xy))

    fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', opacity=0.2, marker_color=z, showlegend=False), row=1, col=1)

    x = np.concatenate([scores_dict[stype] for stype in sample_types])
    y = np.concatenate([pred_distance_dict[stype] for stype in sample_types])

    xy = np.vstack([x, np.log10(y)])

    try:
        z = get_point_density(xy, bins=200)
    except:
        z = np.ones(len(x))

    fig.add_trace(go.Scattergl(x=x, y=y + 1e-16, mode='markers', opacity=0.2, marker_color=z, showlegend=False), row=1,
                  col=2)

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_xaxes(title_text='Model Score')
    fig.update_yaxes(title_text='Distance', row=1, col=1)
    fig.update_yaxes(title_text="Distance", type="log", row=1, col=2)

    return fig


def discriminator_distances_plots(pred_distance_dict, true_distance_dict, epoch_stats_dict):
    if epoch_stats_dict['discriminator_fake_predicted_distance'] is not None:
        tgt_value = np.concatenate(list(true_distance_dict.values()))
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
                         opacity=opacity
                         )
        fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=True, name='Diagonal', marker_color='rgba(0,0,0,1)'),
                      )
        # fig.add_trace(go.Histogram2d(x=df['true_distance'], y=df['predicted_distance'], nbinsx=100, nbinsy=100, colorbar_dtick="log", showlegend=False))

        fig.update_layout(xaxis_title='Target Distance', yaxis_title='Predicted Distance')

        fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
        fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    return fig, linreg_result.rvalue, linreg_result.slope


def log_mini_csp_scores_distributions(config, wandb, generated_samples_dict, real_samples_dict, real_data, sym_info):
    """
    report on key metrics from mini-csp
    """
    scores_labels = ['score', 'vdw overlap', 'density']  # , 'h bond score']
    fig = make_subplots(rows=1, cols=len(scores_labels),
                        vertical_spacing=0.075, horizontal_spacing=0.075)

    colors = sample_colorscale('viridis', 1 + len(np.unique(generated_samples_dict['space group'])))
    real_color = 'rgb(250,0,250)'
    opacity = 0.65

    for i, label in enumerate(scores_labels):
        row = 1  # i // 2 + 1
        col = i + 1  # i % 2 + 1
        for j in range(min(15, real_data.num_graphs)):
            bandwidth1 = np.ptp(generated_samples_dict[label][j]) / 50
            real_score = real_samples_dict[label][j]

            unique_space_group_inds = np.unique(generated_samples_dict['space group'][j])
            n_space_groups = len(unique_space_group_inds)
            space_groups = np.asarray([sym_info['space_groups'][sg] for sg in generated_samples_dict['space group'][j]])
            unique_space_groups = np.asarray([sym_info['space_groups'][sg] for sg in unique_space_group_inds])

            all_sample_score = generated_samples_dict[label][j]
            for k in range(n_space_groups):
                sample_score = all_sample_score[space_groups == unique_space_groups[k]]

                fig.add_trace(
                    go.Violin(x=sample_score, y=[str(real_data.identifier[j]) for _ in range(len(sample_score))],
                              side='positive', orientation='h', width=2, line_color=colors[k],
                              meanline_visible=True, bandwidth=bandwidth1, opacity=opacity,
                              name=unique_space_groups[k], legendgroup=unique_space_groups[k], showlegend=False),
                    row=row, col=col)

            fig.add_trace(go.Violin(x=[real_score], y=[str(real_data.identifier[j])], line_color=real_color,
                                    side='positive', orientation='h', width=2, meanline_visible=True,
                                    name="Experiment", showlegend=True if (i == 0 and j == 0) else False),
                          row=row, col=col)

            fig.update_xaxes(title_text=label, row=1, col=col)

        unique_space_group_inds = np.unique(generated_samples_dict['space group'].flatten())
        n_space_groups = len(unique_space_group_inds)
        space_groups = np.asarray(
            [sym_info['space_groups'][sg] for sg in generated_samples_dict['space group'].flatten()])
        unique_space_groups = np.asarray([sym_info['space_groups'][sg] for sg in unique_space_group_inds])

        if real_data.num_graphs > 1:
            for k in range(n_space_groups):
                all_sample_score = generated_samples_dict[label].flatten()[space_groups == unique_space_groups[k]]

                fig.add_trace(go.Violin(x=all_sample_score, y=['all samples' for _ in range(len(all_sample_score))],
                                        side='positive', orientation='h', width=2, line_color=colors[k],
                                        meanline_visible=True,
                                        bandwidth=np.ptp(generated_samples_dict[label].flatten()) / 100,
                                        opacity=opacity,
                                        name=unique_space_groups[k], legendgroup=unique_space_groups[k],
                                        showlegend=True if i == 0 else False),
                              row=row, col=col)

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=100,  # top margin
        )
    )
    fig.update_xaxes(row=1, col=scores_labels.index('vdw overlap') + 1,
                     range=[0, np.minimum(1, generated_samples_dict['vdw overlap'].flatten().max())])

    fig.update_layout(yaxis_showgrid=True)  # legend_traceorder='reversed',

    fig.layout.margin = layout.margin

    if config.logger.log_figures:
        wandb.log({'Mini-CSP Scores': fig})
    if (config.machine == 'local') and False:
        fig.show(renderer='browser')

    return None


def log_csp_cell_params(config, wandb, generated_samples_dict, real_samples_dict, crystal_name, crystal_ind):
    fig = make_subplots(rows=4, cols=3, subplot_titles=config.dataDims['lattice_features'])
    for i in range(12):
        bandwidth = np.ptp(generated_samples_dict['cell params'][crystal_ind, :, i]) / 100
        col = i % 3 + 1
        row = i // 3 + 1
        fig.add_trace(go.Violin(
            x=[real_samples_dict['cell params'][crystal_ind, i]],
            bandwidth=bandwidth,
            name="Samples",
            showlegend=False,
            line_color='darkorchid',
            side='positive',
            orientation='h',
            width=2,
        ), row=row, col=col)
        for cc, cutoff in enumerate([0, 0.5, 0.95]):
            colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', 3, colortype='rgb')
            good_inds = np.argwhere(
                generated_samples_dict['score'][crystal_ind] > np.quantile(generated_samples_dict['score'][crystal_ind],
                                                                           cutoff))[:, 0]
            fig.add_trace(go.Violin(
                x=generated_samples_dict['cell params'][crystal_ind, :, i][good_inds],
                bandwidth=bandwidth,
                name="Samples",
                showlegend=False,
                line_color=colors[cc],
                side='positive',
                orientation='h',
                width=2,
            ), row=row, col=col)

    fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_traces(opacity=0.5)
    fig.update_layout(title=crystal_name)

    wandb.log({"Mini-CSP Cell Parameters": fig})
    return None

#
# TODO deprecated - rewrite for toy / demo purposes
# def oneD_gaussian_overlap_plot(cmax, data, decoded_data, max_point_types, max_xval, min_xval, sigma):
#     fig = make_subplots(rows=max_point_types, cols=min(4, data.num_graphs))
#     x = np.linspace(min(-1, min_xval), max(1, max_xval), 1001)
#     for j in range(max_point_types):
#         for graph_ind in range(min(4, data.num_graphs)):
#             row = j + 1
#             col = graph_ind + 1
#
#             points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
#             points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()
#
#             ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()
#             pred_type_weights = decoded_data.x[decoded_data.batch == graph_ind, j].cpu().detach().numpy()[:, None]
#
#             fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / sigma), axis=0),
#                               line_color='blue', showlegend=True if (j == 0 and graph_ind == 0) else False,
#                               name=f'True type {j}', legendgroup=f'Predicted type {j}', row=row, col=col)
#
#             fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / sigma), axis=0),
#                               line_color='red', showlegend=True if j == 0 and graph_ind == 0 else False,
#                               name=f'Predicted type {j}', legendgroup=f'Predicted type {j}', row=row, col=col)
#
#             # fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / 0.00001), axis=0), line_color='blue', showlegend=False, name='True', row=row, col=col)
#             # fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / 0.00001), axis=0), line_color='red', showlegend=False, name='Predicted', row=row, col=col)
#     fig.update_yaxes(range=[0, cmax])
#     return fig
#
#
# def twoD_gaussian_overlap_plot(cmax, data, decoded_data, max_point_types, max_xval, min_xval, sigma):
#     fig = make_subplots(rows=max_point_types, cols=min(4, data.num_graphs))
#     num_gridpoints = 25
#     x = np.linspace(min(-1, min_xval), max(1, max_xval), num_gridpoints)
#     y = np.copy(x)
#     xx, yy = np.meshgrid(x, y)
#     grid_array = np.stack((xx.flatten(), yy.flatten())).T
#     for j in range(max_point_types):
#         for graph_ind in range(min(4, data.num_graphs)):
#             row = j + 1
#             col = graph_ind + 1
#
#             points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
#             points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()
#
#             ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()
#             pred_type_weights = decoded_data.x[decoded_data.batch == graph_ind, j].cpu().detach().numpy()[:, None]
#
#             pred_dist = np.sum(pred_type_weights.mean() * np.exp(-(cdist(grid_array, points_pred) ** 2 / sigma)), axis=-1).reshape(num_gridpoints, num_gridpoints)
#
#             fig.add_trace(go.Contour(x=x, y=y, z=pred_dist,
#                                      showlegend=True if (j == 0 and graph_ind == 0) else False,
#                                      name=f'Predicted type', legendgroup=f'Predicted type',
#                                      coloraxis="coloraxis",
#                                      contours=dict(start=0, end=cmax, size=cmax / 50)
#                                      ), row=row, col=col)
#
#             fig.add_trace(go.Scattergl(x=points_true[ref_type_inds][:, 0], y=points_true[ref_type_inds][:, 1],
#                                        mode='markers', marker_color='white', marker_size=10, marker_line_width=2, marker_line_color='green',
#                                        showlegend=True if (j == 0 and graph_ind == 0) else False,
#                                        name=f'True type', legendgroup=f'True type'
#                                        ), row=row, col=col)
#     fig.update_coloraxes(cmin=0, cmax=cmax, autocolorscale=False, colorscale='viridis')
#     return fig


