import numpy as np
import umap
import wandb
from _plotly_utils.colors import n_colors, sample_colorscale
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

from mxtaltools.common.utils import get_point_density, softmax_np
from mxtaltools.reporting.utils import process_BT_evaluation_outputs



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


def cell_density_plot(config, wandb, epoch_stats_dict, layout):
    if epoch_stats_dict['generator_packing_prediction'] is not None and \
            epoch_stats_dict['generator_packing_target'] is not None:

        if config.logger.log_figures:

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
            fig.update_layout(xaxis_title='Asymmetric Unit Volume Target',
                              yaxis_title='Asymmetric Unit Volume Prediction')

            fig.write_image('fig.png', width=512, height=512)  # save the image rather than the fig, for size reasons
            wandb.log({'Cell Packing': wandb.Image('fig.png')}, commit=False)


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

    wandb.log(data={"Scores Separation": submissions_fraction_below_csd_quantile,
                    "Normed Scores Separation": normed_submissions_fraction_below_csd_quantile},
              commit=False)

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
    wandb.log(data={f"BT Submissions {score_name} Distribution": fig}, commit=False)

    '''score separations'''
    fig1, fig2 = BT_separation_tables(layout, scores_dict, BT_submission_scores,
                                      crystals_for_targets, normed_scores_dict, normed_BT_submission_scores)
    wandb.log(data={f"{score_name} Separation Table": fig1,
                    f"Normed {score_name} Separation Table": fig2},
              commit=False)

    '''functional group analysis'''
    fig = functional_group_analysis_fig(scores_dict, tracking_features, layout, dataDims)
    if fig is not None:
        fig.write_image(f'functional_group_{score_name}.png', scale=2)
        wandb.log(data={f"Functional Group {score_name}": fig},
                  commit=False)

    fig = group_wise_analysis_fig(identifiers_list, crystals_for_targets, scores_dict, normed_scores_dict, layout)
    fig.write_image(f'interesting_groups_{score_name}.png', scale=4)
    wandb.log(data={f"Interesting Groups {score_name}": fig},
              commit=False)

    '''
    S2.  score correlates
    '''
    fig = make_correlates_plot(tracking_features, scores_dict['CSD'], dataDims)
    fig.write_image(f'{score_name}_correlates.png', scale=4)
    wandb.log(data={f"{score_name} Correlates": fig}, commit=False)

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
    plot_color_dict = {'CSD': ('rgb(200,0,50)'), 'BT Targets': ('rgb(50,0,50)'), 'BT Submissions': ('rgb(50,150,250)')}

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
        wandb.log(data={'Mini-CSP Scores': fig}, commit=False)
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

    wandb.log(data={"Mini-CSP Cell Parameters": fig}, commit=False)
    return None


def polymorph_classification_trajectory_analysis(test_loader, stats_dict, traj_name):
    # analysis here comprises plotting relative polymorph compositions
    # and returning a trajectory with classification information
    datapoints = test_loader.dataset
    probabilities = softmax_np(np.stack(stats_dict['probs']))
    num_classes = probabilities.shape[1]
    num_atoms = datapoints[0].num_nodes

    class_prob_traj = np.zeros((len(datapoints), probabilities.shape[1]))
    class_selection_traj = np.zeros_like(class_prob_traj)
    time_traj = np.zeros(len(datapoints))
    coords_traj = np.zeros((len(datapoints), num_atoms, 3))
    atom_types_traj = np.zeros((len(datapoints), num_atoms))
    atomwise_prediction_traj = np.zeros_like(atom_types_traj)
    mol_counter = 0
    for ind in range(len(datapoints)):
        sample = datapoints[ind]
        num_mols = sample.mol_ind.max()
        sample_probabilities = probabilities[mol_counter:mol_counter + num_mols]
        sample_predictions = np.argmax(sample_probabilities, axis=1)
        coords_traj[ind] = sample.pos.cpu().detach().numpy()
        atom_types_traj[ind] = sample.x.flatten().cpu().detach().numpy()
        atomwise_prediction_traj[ind] = sample_predictions[sample.mol_ind - 1]

        class_prob_traj[ind] = sample_probabilities.mean(0)
        class_selection_traj[ind] = np.eye(num_classes)[sample_predictions].sum(0)
        time_traj[ind] = int(sample.time_step)

        mol_counter += num_mols

    sort_inds = np.argsort(time_traj)

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Molecules / Class', 'Classwise Probability'])
    for cind in range(num_classes):
        fig.add_scattergl(x=time_traj[sort_inds], y=class_selection_traj[sort_inds, cind],
                          name=f'Polymorph {cind}',
                          row=1, col=1)
        fig.add_scattergl(x=time_traj[sort_inds], y=class_prob_traj[sort_inds, cind],
                          showlegend=False,
                          row=1, col=2)
    fig.show(renderer='browser')

    from mxtaltools.common.ovito_utils import write_ovito_xyz

    write_ovito_xyz(coords_traj,
                    atom_types_traj,
                    atomwise_prediction_traj,
                    filename=traj_name[0].replace('\\', '/').replace('/', '_') + '_prediction')  # write a trajectory


def simple_embedding_fig(std_cell_params, aux_array=None,
                         reference_distribution=None,
                         max_umap_samples: int = 1000,
                         known_minima=None):
    reducer = umap.UMAP(n_components=2,
                        metric='euclidean',
                        n_neighbors=15,
                        min_dist=0.05,
                        )

    if reference_distribution is not None:
        if len(reference_distribution > max_umap_samples):
            inds = np.random.choice(len(reference_distribution), size=max_umap_samples, replace=False)
            reducer.fit(reference_distribution[inds])
        else:
            reducer.fit(reference_distribution)

        ref_embedding = reducer.transform(reference_distribution)
        embedding = reducer.transform(std_cell_params)
    else:
        embedding = reducer.fit_transform(std_cell_params)

    if known_minima is not None:
        known_embedding = reducer.transform(known_minima)

    if aux_array is not None:
        color_array = aux_array
    else:
        xy = np.vstack([embedding[:, 0], embedding[:, 1]])
        try:
            z = get_point_density(xy, bins=25)
        except:
            z = np.ones(len(xy))
        color_array = z

    fig = go.Figure()
    if reference_distribution is not None:
        fig.add_trace(go.Scattergl(x=ref_embedding[:, 0],
                                   y=ref_embedding[:, 1],
                                   mode='markers',
                                   opacity=.15,
                                   name='Reference Distribution',
                                   showlegend=True,
                                   marker=dict(
                                       size=5,
                                       color='black',
                                   )
                                   ))
    if known_minima is not None:
        fig.add_trace(go.Scattergl(x=known_embedding[:, 0],
                                   y=known_embedding[:, 1],
                                   mode='markers',
                                   opacity=1,
                                   name='Known Modes',
                                   showlegend=True,
                                   marker=dict(
                                       size=15,
                                       color='green',  # Fill color
                                       line=dict(
                                           color='black',  # Outline color
                                           width=4  # Outline thickness
                                       )
                                   )
                                   ))
    fig.add_trace(go.Scattergl(x=embedding[:, 0],
                               y=embedding[:, 1],
                               mode='markers',
                               opacity=0.85,
                               name='Policy Samples',
                               showlegend=True,
                               marker=dict(
                                   size=6,
                                   color=color_array.clip(max=100),
                                   colorscale="portland",
                                   cmax=100,
                                   colorbar=dict(title="Sample Energy")
                               )
                               ))
    fig.update_layout(legend_xanchor='center', legend_y=0.0, legend_orientation="h")
    fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    return fig
