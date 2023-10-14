import numpy as np
import wandb
from _plotly_utils.colors import n_colors, sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, linregress
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import torch

from common.geometry_calculations import cell_vol
from models.utils import norm_scores


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
            z = gaussian_kde(xy)(xy)
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
        fig.update_layout(xaxis_title='packing_target', yaxis_title='packing_prediction')

        # #fig.write_image('../paper1_figs_new_architecture/scores_vs_emd.png', scale=4)
        if config.logger.log_figures:
            wandb.log({'Cell Packing': fig})
        if (config.machine == 'local') and False:
            fig.show()


def process_discriminator_outputs(dataDims, epoch_stats_dict):
    scores_dict = {}
    vdw_penalty_dict = {}
    tracking_features_dict = {}
    packing_coeff_dict = {}

    generator_inds = np.where(epoch_stats_dict['generator_sample_source'] == 0)
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

    return scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict


def discriminator_scores_plots(wandb, scores_dict, vdw_penalty_dict, packing_coeff_dict, layout):
    plot_color_dict = {'CSD': ('rgb(250,150,50)'),
                       'Generator': ('rgb(100,50,0)'),
                       'Gaussian': ('rgb(0,50,0)'),
                       'Distorted': ('rgb(0,100,100)')}

    all_vdws = np.concatenate((vdw_penalty_dict['CSD'], vdw_penalty_dict['Gaussian'], vdw_penalty_dict['Distorted'],
                               vdw_penalty_dict['Generator']))
    all_scores = np.concatenate(
        (scores_dict['CSD'], scores_dict['Gaussian'], scores_dict['Distorted'], scores_dict['Generator']))

    sample_source = np.concatenate([
        ['CSD' for _ in range(len(scores_dict['CSD']))],
        ['Gaussian' for _ in range(len(scores_dict['Gaussian']))],
        ['Distorted' for _ in range(len(scores_dict['Distorted']))],
        ['Generator' for _ in range(len(scores_dict['Generator']))],
    ])
    scores_range = np.ptp(all_scores)
    bandwidth1 = scores_range / 200

    vdw_cutoff = min(np.quantile(all_vdws, 0.975), 3)
    bandwidth2 = vdw_cutoff / 200
    viridis = px.colors.sequential.Viridis

    scores_labels = ['CSD', 'Gaussian', 'Distorted', 'Generator']
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
    wandb.log({'Discriminator vs vdw scores': fig})

    '''
    vs coeff
    '''
    bandwidth2 = 0.01

    scores_labels = ['CSD', 'Gaussian', 'Distorted', 'Generator']
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

    all_coeffs = np.concatenate((packing_coeff_dict['CSD'], packing_coeff_dict['Gaussian'],
                                 packing_coeff_dict['Distorted'], packing_coeff_dict['Generator']))

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
    wandb.log({'Discriminator vs packing coefficient': fig})

    '''
    # All in one
    '''
    scatter_dict = {'vdw_score': -all_vdws.clip(max=vdw_cutoff), 'model_score': all_scores, 'packing_coefficient': all_coeffs.clip(min=0, max=1), 'sample_source': sample_source}
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
    wandb.log({'Discriminator Scores Analysis': fig})

    return None


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
                                   y=[np.average(scores_dict['CSD'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                   error_y=dict(type='data',
                                                array=[np.std(scores_dict['CSD'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
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
        target_identifiers[label] = [identifiers_list[crystals_for_targets[label][n]] for n in range(len(crystals_for_targets[label]))]
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

    quantiles = [np.quantile(normed_scores_dict['CSD'], 0.01), np.quantile(normed_scores_dict['CSD'], 0.05), np.quantile(normed_scores_dict['CSD'], 0.1)]

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
                    (dataDims['tracking_features'][i] != 'molecule_is_asymmetric_top'):  # if we have at least 1# relevance
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

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=('a) Molecule & Crystal Features', 'b) Atom Fractions', 'c) Functional Groups Count'), x_title='R Value')

    crystal_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' not in key]
    atom_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' in key]
    mol_keys = [key for key in features_sorted_cleaned if 'count' in key and 'fraction' not in key]

    fig.add_trace(go.Bar(
        y=crystal_keys,
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in crystal_keys],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in crystal_keys]).astype('float16'),
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


def plot_discriminator_score_correlates(dataDims, wandb, epoch_stats_dict, layout):
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

    wandb.log({'Discriminator Score Correlates': fig})


def process_BT_evaluation_outputs(dataDims, wandb, extra_test_dict, test_epoch_stats_dict, train_epoch_stats_dict, size_normed_score=False):
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

    '''
    record all the stats for the CSD data
    '''
    scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict \
        = process_discriminator_outputs(dataDims, test_epoch_stats_dict)

    '''
    build property dicts for the submissions and BT targets
    '''
    bt_score_correlates = {}
    for target in crystals_for_targets.keys():  # run the analysis for each target
        if target_identifiers_inds[target] != []:  # record target data

            target_index = target_identifiers_inds[target]
            scores = extra_test_dict['discriminator_real_score'][target_index]
            scores_dict[target + '_exp'] = scores[None]

            tracking_features_dict[target + '_exp'] = {feat: vec for feat, vec in zip(dataDims['tracking_features'], extra_test_dict['tracking_features'][target_index][None, :].T)}

            if size_normed_score:
                scores_dict[target + '_exp'] = norm_scores(scores_dict[target + '_exp'], extra_test_dict['tracking_features'][target_index][None, :], dataDims)

            vdw_penalty_dict[target + '_exp'] = extra_test_dict['real_vdw_penalty'][target_index][None]

            wandb.log({f'Average_{target}_exp_score': np.average(scores)})

        if crystals_for_targets[target] != []:  # record sample data
            target_indices = crystals_for_targets[target]
            scores = extra_test_dict['discriminator_real_score'][target_indices]
            scores_dict[target] = scores
            tracking_features_dict[target] = {feat: vec for feat, vec in zip(dataDims['tracking_features'], extra_test_dict['tracking_features'][target_indices].T)}

            if size_normed_score:
                scores_dict[target] = norm_scores(scores_dict[target], extra_test_dict['tracking_features'][target_indices], dataDims)

            vdw_penalty_dict[target] = extra_test_dict['real_vdw_penalty'][target_indices]

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
        loss_correlations[j] = np.corrcoef(scores_dict['CSD'], test_epoch_stats_dict['tracking_features'][:, j], rowvar=False)[0, 1]
    bt_score_correlates['CSD'] = loss_correlations

    # collect all BT targets & submissions into single dicts
    BT_target_scores = np.asarray([scores_dict[key] for key in scores_dict.keys() if 'exp' in key])
    BT_submission_scores = np.concatenate([scores_dict[key] for key in scores_dict.keys() if key in crystals_for_targets.keys()])
    # BT_scores_dists = {key: np.histogram(scores_dict[key], bins=200, range=[-15, 15])[0] / len(scores_dict[key]) for key in scores_dict.keys() if key in crystals_for_targets.keys()}
    # BT_balanced_dist = np.average(np.stack(list(BT_scores_dists.values())), axis=0)

    wandb.log({'Average BT submission score': np.average(BT_submission_scores)})
    wandb.log({'Average BT target score': np.average(BT_target_scores)})
    wandb.log({'BT submission score std': np.std(BT_target_scores)})
    wandb.log({'BT target score std': np.std(BT_target_scores)})

    return bt_score_correlates, scores_dict, \
        crystals_for_targets, blind_test_targets, target_identifiers, target_identifiers_inds, \
        BT_target_scores, BT_submission_scores, \
        vdw_penalty_dict, tracking_features_dict


def discriminator_BT_reporting(dataDims, wandb, test_epoch_stats_dict, extra_test_dict):
    # todo clean up the analysis here - it's a mess!
    tracking_features = test_epoch_stats_dict['tracking_features']
    identifiers_list = extra_test_dict['identifiers']

    (bt_score_correlates, scores_dict, crystals_for_targets,
     blind_test_targets, target_identifiers,
     target_identifiers_inds, BT_target_scores, BT_submission_scores,
     vdw_penalty_dict, tracking_features_dict) = \
        process_BT_evaluation_outputs(dataDims, wandb, extra_test_dict,
                                      test_epoch_stats_dict, None, size_normed_score=False)

    layout = go.Layout(  # todo increase upper margin - figure titles are getting cutoff
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=40,  # top margin
        )
    )

    fig, normed_scores_dict, normed_BT_submission_scores, normed_BT_target_scores = blind_test_scores_distributions_fig(
        crystals_for_targets, target_identifiers_inds,
        scores_dict, BT_target_scores, BT_submission_scores,
        tracking_features_dict, layout)
    fig.write_image('bt_submissions_distribution.png', scale=4)
    wandb.log({"BT Submissions Distribution": fig})

    '''
    7. Table of BT separation statistics
    '''
    vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    quantiles = np.quantile(scores_dict['CSD'], vals)
    submissions_fraction_below_csd_quantile = {value: np.average(BT_submission_scores < cutoff) for value, cutoff in zip(vals, quantiles)}

    normed_quantiles = np.quantile(normed_scores_dict['CSD'], vals)
    normed_submissions_fraction_below_csd_quantile = {value: np.average(normed_BT_submission_scores < cutoff) for value, cutoff in zip(vals, normed_quantiles)}

    submissions_fraction_below_target = {key: np.average(scores_dict[key] < scores_dict[key + '_exp']) for key in crystals_for_targets.keys() if key in scores_dict.keys()}
    submissions_average_below_target = np.average(list(submissions_fraction_below_target.values()))

    fig = go.Figure(data=go.Table(
        header=dict(values=['CSD Test Quantile', 'Fraction of Submissions']),
        cells=dict(values=[list(submissions_fraction_below_csd_quantile.keys()),
                           list(submissions_fraction_below_csd_quantile.values()),
                           ], format=[".3", ".3"])))
    fig.update_layout(width=200)
    fig.layout.margin = layout.margin
    fig.write_image('scores_separation_table.png', scale=4)
    wandb.log({"Nice Scores Separation Table": fig})

    fig = go.Figure(data=go.Table(
        header=dict(values=['CSD Test Quantile(normed)', 'Fraction of Submissions (normed)']),
        cells=dict(values=[list(normed_submissions_fraction_below_csd_quantile.keys()),
                           list(normed_submissions_fraction_below_csd_quantile.values()),
                           ], format=[".3", ".3"])))
    fig.update_layout(width=200)
    fig.layout.margin = layout.margin
    fig.write_image('normed_scores_separation_table.png', scale=4)
    wandb.log({"Nice Normed Scores Separation Table": fig})

    wandb.log({"Scores Separation": submissions_fraction_below_csd_quantile})
    wandb.log({"Normed Scores Separation": normed_submissions_fraction_below_csd_quantile})

    '''
    8. Functional group analysis
    '''

    fig = functional_group_analysis_fig(scores_dict, tracking_features, layout, dataDims)
    if fig is not None:
        fig.write_image('functional_group_scores.png', scale=2)
        wandb.log({"Functional Group Scores": fig})

    '''
    10. Interesting Group-wise analysis
    '''
    fig = group_wise_analysis_fig(identifiers_list, crystals_for_targets, scores_dict, normed_scores_dict, layout)
    fig.write_image('interesting_groups.png', scale=4)
    wandb.log({"Interesting Groups": fig})

    '''
    S1. All group-wise analysis
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

    '''
    S2.  score correlates
    '''
    fig = make_correlates_plot(tracking_features, scores_dict['CSD'], dataDims)
    fig.write_image('scores_correlates.png', scale=4)
    wandb.log({"Score Correlates": fig})

    return None


def blind_test_scores_distributions_fig(crystals_for_targets, target_identifiers_inds, scores_dict, BT_target_scores, BT_submission_scores, tracking_features_dict, layout):

    lens = [len(val) for val in crystals_for_targets.values()]
    targets_list = list(target_identifiers_inds.values())
    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(np.count_nonzero(lens), sum([1 for ll in targets_list if ll != []])), colortype='rgb')

    plot_color_dict = {'Train Real': ('rgb(250,50,50)'),
                       'CSD': ('rgb(250,150,50)'),
                       'Gaussian': ('rgb(0,50,0)'),
                       'Test NF': ('rgb(0,150,0)'),
                       'Distorted': ('rgb(0,100,100)')}
    ind = 0
    for target in crystals_for_targets.keys():
        if crystals_for_targets[target] != []:
            plot_color_dict[target] = colors[ind]
            plot_color_dict[target + '_exp'] = colors[ind]
            ind += 1

    # plot 1
    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
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
                fig.add_trace(go.Violin(x=scores_dict[label], name=name_label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6),
                              row=1, col=1)
            else:
                fig.add_trace(go.Violin(x=scores_dict[label], name=name_label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False),
                              row=1, col=1)

    # plot2 inset
    plot_color_dict = {}
    plot_color_dict['CSD'] = ('rgb(200,0,50)')  # test
    plot_color_dict['BT Targets'] = ('rgb(50,0,50)')
    plot_color_dict['BT Submissions'] = ('rgb(50,150,250)')

    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth = scores_range / 200

    # test data
    fig.add_trace(go.Violin(x=scores_dict['CSD'], name='CSD Test',
                            line_color=plot_color_dict['CSD'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

    # BT distribution
    fig.add_trace(go.Violin(x=BT_target_scores, name='BT Targets',
                            line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=1, col=2)
    # Submissions
    fig.add_trace(go.Violin(x=BT_submission_scores, name='BT Submissions',
                            line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

    quantiles = [np.quantile(scores_dict['CSD'], 0.01), np.quantile(scores_dict['CSD'], 0.05), np.quantile(scores_dict['CSD'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['CSD'], row=1, col=2)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['CSD'], row=1, col=2)
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['CSD'], row=1, col=2)

    normed_scores_dict = scores_dict.copy()
    for key in normed_scores_dict.keys():
        normed_scores_dict[key] = normed_scores_dict[key] / tracking_features_dict[key]['molecule_num_atoms']

    normed_BT_target_scores = np.concatenate([normed_scores_dict[key] for key in normed_scores_dict.keys() if 'exp' in key])
    normed_BT_submission_scores = np.concatenate([normed_scores_dict[key] for key in normed_scores_dict.keys() if key in crystals_for_targets.keys()])
    scores_range = np.ptp(np.concatenate(list(normed_scores_dict.values())))
    bandwidth = scores_range / 200
    # test data
    fig.add_trace(go.Violin(x=normed_scores_dict['CSD'], name='CSD Test',
                            line_color=plot_color_dict['CSD'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

    # BT distribution
    fig.add_trace(go.Violin(x=normed_BT_target_scores, name='BT Targets',
                            line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=2, col=2)
    # Submissions
    fig.add_trace(go.Violin(x=normed_BT_submission_scores, name='BT Submissions',
                            line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

    quantiles = [np.quantile(normed_scores_dict['CSD'], 0.01), np.quantile(normed_scores_dict['CSD'], 0.05), np.quantile(normed_scores_dict['CSD'], 0.1)]
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
    log_cubic_defect(epoch_stats_dict['final_generated_cell_parameters'])
    wandb.log({"Generated cell parameter variation": epoch_stats_dict['final_generated_cell_parameters'].std(0).mean()})
    generator_losses, average_losses_dict = process_generator_losses(config, epoch_stats_dict)
    wandb.log(average_losses_dict)

    cell_density_plot(config, wandb, epoch_stats_dict, layout)
    plot_generator_loss_correlates(dataDims, wandb, epoch_stats_dict, generator_losses, layout)
    cell_scatter(epoch_stats_dict, wandb, layout, extra_category='generated_space_group_numbers')

    return None


def cell_scatter(epoch_stats_dict, wandb, layout, extra_category=None):
    model_scores = epoch_stats_dict['generator_adversarial_score']
    scatter_dict = {'vdw_score': epoch_stats_dict['generator_per_mol_vdw_score'], 'model_score': model_scores,
                    'packing_coefficient': epoch_stats_dict['generator_packing_prediction'].clip(min=0, max=1)}
    if extra_category is not None:
        scatter_dict[extra_category] = epoch_stats_dict[extra_category]

    vdw_cutoff = max(-1.5, np.amin(scatter_dict['vdw_score']))
    opacity = max(0.1, 1 - len(model_scores) / 5e4)
    df = pd.DataFrame.from_dict(scatter_dict)
    if extra_category is not None:
        fig = px.scatter(df,
                         x='vdw_score', y='packing_coefficient',
                         color='model_score', symbol=extra_category,
                         marginal_x='histogram', marginal_y='histogram',
                         range_color=(np.amin(model_scores), np.amax(model_scores)),
                         opacity=opacity
                         )
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=1))

    else:
        fig = px.scatter(df,
                         x='vdw_score', y='packing_coefficient',
                         color='model_score',
                         marginal_x='histogram', marginal_y='histogram',
                         opacity=opacity
                         )
    fig.layout.margin = layout.margin
    fig.update_layout(xaxis_title='vdw score', yaxis_title='packing coefficient')
    fig.update_layout(xaxis_range=[vdw_cutoff, 0.1], yaxis_range=[0, 1.1])
    fig.update_layout(xaxis_title='vdw score', yaxis_title='packing coefficient')
    wandb.log({'Generator Samples': fig})


def log_regression_accuracy(config, dataDims, epoch_stats_dict):
    target_key = config.dataset.regression_target

    target = np.asarray(epoch_stats_dict['regressor_target'])
    prediction = np.asarray(epoch_stats_dict['regressor_prediction'])

    multiplicity = epoch_stats_dict['tracking_features'][:, dataDims['tracking_features'].index('crystal_symmetry_multiplicity')]
    mol_volume = epoch_stats_dict['tracking_features'][:, dataDims['tracking_features'].index('molecule_volume')]
    mol_mass = epoch_stats_dict['tracking_features'][:, dataDims['tracking_features'].index('molecule_mass')]
    target_density = epoch_stats_dict['tracking_features'][:, dataDims['tracking_features'].index('crystal_density')]
    target_volume = epoch_stats_dict['tracking_features'][:, dataDims['tracking_features'].index('crystal_cell_volume')]
    target_packing_coefficient = epoch_stats_dict['tracking_features'][:, dataDims['tracking_features'].index('crystal_packing_coefficient')]

    if target_key == 'crystal_reduced_volume':
        predicted_volume = prediction
        predicted_packing_coefficient = mol_volume * multiplicity / (prediction * multiplicity)
    elif target_key == 'crystal_packing_coefficient':
        predicted_volume = mol_volume * multiplicity / prediction
        predicted_packing_coefficient = prediction
    elif target_key == 'crystal_density':
        predicted_volume = prediction / (mol_mass * multiplicity) / 1.66
        predicted_packing_coefficient = prediction * mol_volume / mol_mass / 1.66
    else:
        assert False, f"Detailed reporting for {target_key} is not yet implemented"

    predicted_density = (mol_mass * multiplicity) / predicted_volume * 1.66
    losses = ['normed_error', 'abs_normed_error', 'squared_error']
    loss_dict = {}
    fig_dict = {}
    fig = make_subplots(cols=3, rows=2, subplot_titles=['asym_unit_volume', 'packing_coefficient', 'density', 'asym_unit_volume error', 'packing_coefficient error', 'density error'])
    for ind, (name, tgt_value, pred_value) in enumerate(zip(['asym_unit_volume', 'packing_coefficient', 'density'], [target_volume, target_packing_coefficient, target_density], [predicted_volume, predicted_packing_coefficient,
                                                                                                                                                                                  predicted_density])):
        for loss in losses:
            if loss == 'normed_error':
                loss_i = (tgt_value - pred_value) / np.abs(tgt_value)
            elif loss == 'abs_normed_error':
                loss_i = np.abs((tgt_value - pred_value) / np.abs(tgt_value))
            elif loss == 'squared_error':
                loss_i = (tgt_value - pred_value) ** 2
            else:
                assert False, "Loss not implemented"
            loss_dict[name + '_' + loss + '_mean'] = np.mean(loss_i)
            loss_dict[name + '_' + loss + '_std'] = np.std(loss_i)

        linreg_result = linregress(tgt_value, pred_value)
        loss_dict[name + '_regression_R_value'] = linreg_result.rvalue
        loss_dict[name + '_regression_slope'] = linreg_result.slope

        # predictions vs target trace
        xline = np.linspace(max(min(tgt_value), min(pred_value)),
                            min(max(tgt_value), max(pred_value)), 2)

        xy = np.vstack([tgt_value, pred_value])
        try:
            z = gaussian_kde(xy)(xy)
        except:
            z = np.ones_like(tgt_value)

        row = 1
        col = ind % 3 + 1
        fig.add_trace(go.Scattergl(x=tgt_value, y=pred_value, mode='markers', marker=dict(color=z), opacity=0.1, showlegend=False),
                      row=row, col=col)
        fig.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                      row=row, col=col)

        row = 2
        fig.add_trace(go.Histogram(x=pred_value - tgt_value,
                                   histnorm='probability density',
                                   nbinsx=100,
                                   name="Error Distribution",
                                   showlegend=False,
                                   marker_color='rgba(0,0,100,1)'),
                      row=row, col=col)
    #
    fig.update_yaxes(title_text='Predicted AUnit Volume (A<sup>3</sup>)', row=1, col=1, dtick=2500, range=[0, 15000], tickformat=".0f")
    fig.update_yaxes(title_text='Predicted Density (g/cm<sup>3</sup>)', row=1, col=3, dtick=0.5, range=[0.8, 4], tickformat=".1f")
    fig.update_yaxes(title_text='Predicted Packing Coefficient', row=1, col=2, dtick=0.05, range=[0.55, 0.8], tickformat=".2f")

    fig.update_xaxes(title_text='True AUnit Volume (A<sup>3</sup>)', row=1, col=1, dtick=2500, range=[0, 15000], tickformat=".0f")
    fig.update_xaxes(title_text='True Density (g/cm<sup>3</sup>)', row=1, col=3, dtick=0.5, range=[0.8, 4], tickformat=".1f")
    fig.update_xaxes(title_text='True Packing Coefficient', row=1, col=2, dtick=0.05, range=[0.55, 0.8], tickformat=".2f")

    fig.update_xaxes(title_text='Packing Coefficient Error', row=2, col=2, dtick=0.05, tickformat=".2f")
    fig.update_xaxes(title_text='Density Error (g/cm<sup>3</sup>)', row=2, col=3, dtick=0.1, tickformat=".1f")
    fig.update_xaxes(title_text='AUnit Volume (A<sup>3</sup>)', row=2, col=1, dtick=250, tickformat=".0f")

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    fig_dict['Regression Results'] = fig

    """
    correlate losses with molecular features
    """  # todo convert the below to a function
    fig = make_correlates_plot(np.asarray(epoch_stats_dict['tracking_features']),
                               np.abs(target_packing_coefficient - predicted_packing_coefficient) / target_packing_coefficient, dataDims)
    fig_dict['Regressor Loss Correlates'] = fig

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
                    (dataDims['tracking_features'][i] != 'molecule_is_asymmetric_top'):  # if we have at least 1# relevance
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

    g_loss_dict = {feat: corr for feat, corr in zip(features_sorted, g_loss_correlations)}

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=('a) Molecule & Crystal Features', 'b) Atom Fractions', 'c) Functional Groups Count'), x_title='R Value')

    crystal_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' not in key]
    atom_keys = [key for key in features_sorted_cleaned if 'count' not in key and 'fraction' in key]
    mol_keys = [key for key in features_sorted_cleaned if 'count' in key and 'fraction' not in key]

    fig.add_trace(go.Bar(
        y=crystal_keys,
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in crystal_keys],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if feat in crystal_keys]).astype('float16'),
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
    fig.update_yaxes(tickfont=dict(size=14), row=1, col=3)

    fig.layout.annotations[0].update(x=0.12)
    fig.layout.annotations[1].update(x=0.45)
    fig.layout.annotations[2].update(x=0.88)

    fig.update_xaxes(range=[np.amin(list(g_loss_dict.values())), np.amax(list(g_loss_dict.values()))])
    # fig.update_layout(width=1200, height=400)
    fig.update_layout(showlegend=False)

    return fig


def detailed_reporting(config, dataDims, test_loader, test_epoch_stats_dict, extra_test_dict=None):
    """
    Do analysis and upload results to w&b
    """
    if (test_epoch_stats_dict is not None) and config.mode == 'gan':
        if 'final_generated_cell_parameters' in test_epoch_stats_dict.keys():
            cell_params_analysis(config, dataDims, wandb, test_loader, test_epoch_stats_dict)

        if config.generator.train_vdw or config.generator.train_adversarially:
            cell_generation_analysis(config, dataDims, test_epoch_stats_dict)

        if config.discriminator.train_on_distorted or config.discriminator.train_on_randn or config.discriminator.train_adversarially:
            discriminator_analysis(config, dataDims, test_epoch_stats_dict)

    elif config.mode == 'regression':
        log_regression_accuracy(config, dataDims, test_epoch_stats_dict)

    if extra_test_dict is not None and len(extra_test_dict) > 0:
        discriminator_BT_reporting(dataDims, wandb, test_epoch_stats_dict, extra_test_dict)

    return None


def discriminator_analysis(config, dataDims, epoch_stats_dict):
    """
    do analysis and plotting for cell discriminator

    -: scores distribution and vdw penalty by sample source
    -: loss correlates
    """
    layout = plotly_setup(config)

    scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict \
        = process_discriminator_outputs(dataDims, epoch_stats_dict)

    discriminator_scores_plots(wandb, scores_dict, vdw_penalty_dict, packing_coeff_dict, layout)
    plot_discriminator_score_correlates(dataDims, wandb, epoch_stats_dict, layout)

    return None


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

                fig.add_trace(go.Violin(x=sample_score, y=[str(real_data.csd_identifier[j]) for _ in range(len(sample_score))],
                                        side='positive', orientation='h', width=2, line_color=colors[k],
                                        meanline_visible=True, bandwidth=bandwidth1, opacity=opacity,
                                        name=unique_space_groups[k], legendgroup=unique_space_groups[k], showlegend=False),
                              row=row, col=col)

            fig.add_trace(go.Violin(x=[real_score], y=[str(real_data.csd_identifier[j])], line_color=real_color,
                                    side='positive', orientation='h', width=2, meanline_visible=True,
                                    name="Experiment", showlegend=True if (i == 0 and j == 0) else False),
                          row=row, col=col)

            fig.update_xaxes(title_text=label, row=1, col=col)

        unique_space_group_inds = np.unique(generated_samples_dict['space group'].flatten())
        n_space_groups = len(unique_space_group_inds)
        space_groups = np.asarray([sym_info['space_groups'][sg] for sg in generated_samples_dict['space group'].flatten()])
        unique_space_groups = np.asarray([sym_info['space_groups'][sg] for sg in unique_space_group_inds])

        if real_data.num_graphs > 1:
            for k in range(n_space_groups):
                all_sample_score = generated_samples_dict[label].flatten()[space_groups == unique_space_groups[k]]

                fig.add_trace(go.Violin(x=all_sample_score, y=['all samples' for _ in range(len(all_sample_score))],
                                        side='positive', orientation='h', width=2, line_color=colors[k],
                                        meanline_visible=True, bandwidth=np.ptp(generated_samples_dict[label].flatten()) / 100, opacity=opacity,
                                        name=unique_space_groups[k], legendgroup=unique_space_groups[k], showlegend=True if i == 0 else False),
                              row=row, col=col)

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=100,  # top margin
        )
    )
    fig.update_xaxes(row=1, col=scores_labels.index('vdw overlap') + 1, range=[0, np.minimum(1, generated_samples_dict['vdw overlap'].flatten().max())])

    fig.update_layout(yaxis_showgrid=True)  # legend_traceorder='reversed',

    fig.layout.margin = layout.margin

    if config.logger.log_figures:
        wandb.log({'Mini-CSP Scores': fig})
    if (config.machine == 'local') and False:
        fig.show()

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
            good_inds = np.argwhere(generated_samples_dict['score'][crystal_ind] > np.quantile(generated_samples_dict['score'][crystal_ind], cutoff))[:, 0]
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
