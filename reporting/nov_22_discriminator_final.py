from plotly.colors import n_colors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.stats import linregress
from reporting.nov_22_discriminator import process_discriminator_evaluation_data

'''
figures which were actually used in the paper
'''

def nov_22_paper_discriminator_plots(config, wandb):
    test_epoch_stats_dict = np.load('C:/Users\mikem\crystals\CSP_runs/275_test_epoch_stats_dict.npy', allow_pickle=True).item()
    extra_test_dict = np.load('C:/Users\mikem\crystals\CSP_runs/275_extra_test_dict.npy', allow_pickle=True).item()

    tracking_features = test_epoch_stats_dict['tracking_features']
    identifiers_list = extra_test_dict['identifiers']
    score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, \
    scores_dict, all_identifiers, blind_test_targets, target_identifiers, \
    target_identifiers_inds, BT_target_scores, BT_submission_scores, \
    BT_scores_dists, BT_balanced_dist, vdw_penalty_dict, tracking_features_dict = \
        process_discriminator_evaluation_data(config,wandb,extra_test_dict,
                                              test_epoch_stats_dict,
                                              None, size_normed_score=False)

    del test_epoch_stats_dict
    del extra_test_dict

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
        )
    )

    '''
    4. true-false model scores distribution
    '''
    lens = [len(val) for val in all_identifiers.values()]
    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(np.count_nonzero(lens), np.count_nonzero(list(target_identifiers_inds.values()))), colortype='rgb')

    plot_color_dict = {}
    plot_color_dict['Test Real'] = ('rgb(250,150,50)')  # test
    plot_color_dict['Test Randn'] = ('rgb(0,50,0)')  # fake csd
    plot_color_dict['Test Distorted'] = ('rgb(0,100,100)')  # fake distortion
    ind = 0
    for target in all_identifiers.keys():
        if all_identifiers[target] != []:
            plot_color_dict[target] = colors[ind]
            plot_color_dict[target + ' exp'] = colors[ind]
            ind += 1

    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth1 = scores_range / 200

    bandwidth2 = 15 / 200
    viridis = px.colors.sequential.Viridis

    scores_labels = {'Test Real': 'CSD Test', 'Test Randn': 'Gaussian', 'Test Distorted': 'Distorted'}
    fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)'),
                        specs=[[{}, {}], [{"colspan": 2}, None]], vertical_spacing=0.14)

    for i, label in enumerate(scores_labels):
        legend_label = scores_labels[label]
        fig.add_trace(go.Violin(x=scores_dict[label], name=legend_label, line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4,
                                meanline_visible=True, bandwidth=bandwidth1, points=False),
                      row=1, col=1)
        fig.add_trace(go.Violin(x=-np.log(vdw_penalty_dict[label] + 1e-6), name=legend_label, line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth2, points=False),
                      row=1, col=2)

    all_vdws = np.concatenate((vdw_penalty_dict['Test Real'], vdw_penalty_dict['Test Randn'], vdw_penalty_dict['Test Distorted']))
    all_scores_i = np.concatenate((scores_dict['Test Real'], scores_dict['Test Randn'], scores_dict['Test Distorted']))

    rrange = np.logspace(3, 0, len(viridis))
    cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
    cscale[0][0] = 0
    # colorscale = [
    #     [0, viridis[0]],
    #     [1. / 1000000, viridis[2]],
    #     [1. / 10000, viridis[4]],
    #     [1. / 100, viridis[7]],
    #     [1., viridis[9]],

    fig.add_trace(go.Histogram2d(x=all_scores_i,
                                 y=-np.log(all_vdws + 1e-6),
                                 showscale=False,
                                 nbinsy=50, nbinsx=200,
                                 colorscale=cscale,
                                 colorbar=dict(
                                     tick0=0,
                                     tickmode='array',
                                     tickvals=[0, 1000, 10000]
                                 )),
                  row=2, col=1)

    fig.update_layout(showlegend=False, yaxis_showgrid=True, width=800, height=500)
    fig.update_xaxes(title_text='Model Score', row=1, col=1)
    fig.update_xaxes(title_text='vdw Score', row=1, col=2)
    fig.update_xaxes(title_text='Model Score', row=2, col=1)
    fig.update_yaxes(title_text='vdw Score', row=2, col=1)

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    fig.layout.annotations[0].update(x=0.025)
    fig.layout.annotations[1].update(x=0.575)

    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/real_vs_fake_scores.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    5. BT scores distributions w aggregate inset
    '''

    lens = [len(val) for val in all_identifiers.values()]
    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(np.count_nonzero(lens), np.count_nonzero(list(target_identifiers_inds.values()))), colortype='rgb')

    plot_color_dict = {}
    plot_color_dict['Train Real'] = ('rgb(250,50,50)')  # train
    plot_color_dict['Test Real'] = ('rgb(250,150,50)')  # test
    plot_color_dict['Test Randn'] = ('rgb(0,50,0)')  # fake csd
    plot_color_dict['Test NF'] = ('rgb(0,150,0)')  # fake nf
    plot_color_dict['Test Distorted'] = ('rgb(0,100,100)')  # fake distortion
    ind = 0
    for target in all_identifiers.keys():
        if all_identifiers[target] != []:
            plot_color_dict[target] = colors[ind]
            plot_color_dict[target + ' exp'] = colors[ind]
            ind += 1

    # plot 1
    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth = scores_range / 200

    fig = make_subplots(cols=2, rows=2, horizontal_spacing=0.15, subplot_titles=('a)', 'b)', 'c)'),
                        specs=[[{"rowspan": 2}, {}], [None, {}]], vertical_spacing=0.12)
    fig.layout.annotations[0].update(x=0.025)
    fig.layout.annotations[1].update(x=0.525)
    fig.layout.annotations[2].update(x=0.525)
    scores_labels = {'Test Real': 'CSD Test', 'Test Randn': 'Gaussian', 'Test Distorted': 'Distorted'}

    for i, label in enumerate(scores_dict.keys()):
        # if 'X' in label and 'exp' not in label:
        #     agrees_with_exp_sg = tracking_features_dict[label]['crystal spacegroup number'] == tracking_features_dict[label + ' exp']['crystal spacegroup number']
        #     fig.add_trace(go.Violin(x=scores_dict[label][agrees_with_exp_sg], name=label, line_color=plot_color_dict['Test NF'], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False),
        #                   row=1, col=1)
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
    plot_color_dict['Test Real'] = ('rgb(200,0,50)')  # test
    plot_color_dict['BT Targets'] = ('rgb(50,0,50)')
    plot_color_dict['BT Submissions'] = ('rgb(50,150,250)')

    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth = scores_range / 200

    # test data
    fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test',
                            line_color=plot_color_dict['Test Real'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

    # BT distribution
    fig.add_trace(go.Violin(x=BT_target_scores, name='BT Targets',
                            line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=1, col=2)
    # Submissions
    fig.add_trace(go.Violin(x=BT_submission_scores, name='BT Submissions',
                            line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=1, col=2)

    quantiles = [np.quantile(scores_dict['Test Real'], 0.01), np.quantile(scores_dict['Test Real'], 0.05), np.quantile(scores_dict['Test Real'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'], row=1, col=2)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'], row=1, col=2)
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'], row=1, col=2)

    normed_scores_dict = scores_dict.copy()
    for key in normed_scores_dict.keys():
        normed_scores_dict[key] = normed_scores_dict[key] / tracking_features_dict[key]['molecule num atoms']

    normed_BT_target_scores = np.concatenate([normed_scores_dict[key] for key in normed_scores_dict.keys() if 'exp' in key])
    normed_BT_submission_scores = np.concatenate([normed_scores_dict[key] for key in normed_scores_dict.keys() if key in all_identifiers.keys()])
    scores_range = np.ptp(np.concatenate(list(normed_scores_dict.values())))
    bandwidth = scores_range / 200
    # test data
    fig.add_trace(go.Violin(x=normed_scores_dict['Test Real'], name='CSD Test',
                            line_color=plot_color_dict['Test Real'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

    # BT distribution
    fig.add_trace(go.Violin(x=normed_BT_target_scores, name='BT Targets',
                            line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth / 100, points=False), row=2, col=2)
    # Submissions
    fig.add_trace(go.Violin(x=normed_BT_submission_scores, name='BT Submissions',
                            line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=1.5, meanline_visible=True, bandwidth=bandwidth, points=False), row=2, col=2)

    quantiles = [np.quantile(normed_scores_dict['Test Real'], 0.01), np.quantile(normed_scores_dict['Test Real'], 0.05), np.quantile(normed_scores_dict['Test Real'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'], row=2, col=2)
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'], row=2, col=2)
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'], row=2, col=2)


    fig.update_layout(showlegend=False, yaxis_showgrid=True, width=1000, height=500)
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_xaxes(title_text='Model Score', row=1, col=2)
    fig.update_xaxes(title_text='Model Score', row=1, col=1)
    fig.update_xaxes(title_text='Model Score / molecule # atoms', row=2, col=2)

    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/bt_submissions_distribution.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    7. Table of BT separation statistics
    '''
    vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    quantiles = np.quantile(scores_dict['Test Real'], vals)
    submissions_fraction_below_csd_quantile = {value: np.average(BT_submission_scores < cutoff) for value, cutoff in zip(vals, quantiles)}

    normed_quantiles = np.quantile(normed_scores_dict['Test Real'], vals)
    normed_submissions_fraction_below_csd_quantile = {value: np.average(normed_BT_submission_scores < cutoff) for value, cutoff in zip(vals, normed_quantiles)}

    submissions_fraction_below_target = {key: np.average(scores_dict[key] < scores_dict[key + ' exp']) for key in all_identifiers.keys() if key in scores_dict.keys()}
    submissions_average_below_target = np.average(list(submissions_fraction_below_target.values()))

    fig = go.Figure(data=go.Table(
        header=dict(values=['CSD Test Quantile', 'Fraction of Submissions']),
        cells=dict(values=[list(submissions_fraction_below_csd_quantile.keys()),
                           list(submissions_fraction_below_csd_quantile.values()),
                           ], format=[".3", ".3"])))
    fig.update_layout(width=200)
    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/scores_separation_table.png', scale=4)
    if config.machine == 'local':
        fig.show()

    fig = go.Figure(data=go.Table(
        header=dict(values=['CSD Test Quantile', 'Fraction of Submissions']),
        cells=dict(values=[list(normed_submissions_fraction_below_csd_quantile.keys()),
                           list(normed_submissions_fraction_below_csd_quantile.values()),
                           ], format=[".3", ".3"])))
    fig.update_layout(width=200)
    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/normed_scores_separation_table.png')
    fig.update_layout(title=dict(text="Normed Scores Fractions"))
    if config.machine == 'local':
        fig.show()
    wandb.log({"Nice Normed Scores Separation Table": fig})

    wandb.log({"Scores Separation": submissions_fraction_below_csd_quantile})
    wandb.log({"Normed Scores Separation": normed_submissions_fraction_below_csd_quantile})

    '''
    8. Functional group analysis
    '''
    tracking_features_names = config.dataDims['tracking_features']
    # get the indices for each functional group
    functional_group_inds = {}
    fraction_dict = {}
    for ii, key in enumerate(tracking_features_names):
        if ('molecule' in key and 'fraction' in key):
            if np.average(tracking_features[:, ii] > 0) > 0.01:
                fraction_dict[key.split()[1]] = np.average(tracking_features[:, ii] > 0)
                functional_group_inds[key.split()[1]] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]
        elif 'molecule has' in key:
            if np.average(tracking_features[:, ii] > 0) > 0.01:
                fraction_dict[key.split()[2]] = np.average(tracking_features[:, ii] > 0)
                functional_group_inds[key.split()[2]] = np.argwhere(tracking_features[:, ii] > 0)[:, 0]

    sort_order = np.argsort(list(fraction_dict.values()))[-1::-1]
    sorted_functional_group_keys = [list(functional_group_inds.keys())[i] for i in sort_order]
    #
    # colors = n_colors('rgb(100,10,5)', 'rgb(5,110,200)', len(list(functional_group_inds.keys())), colortype='rgb')
    # plot_color_dict = {}
    # for ind, target in enumerate(sorted_functional_group_keys):
    #     plot_color_dict[target] = colors[ind]
    #
    #
    # scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    # bandwidth = scores_range / 200
    #
    # fig = go.Figure()
    # fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test',
    #                         line_color='#0c4dae', side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
    #
    # for ii, label in enumerate(sorted_functional_group_keys):
    #     fraction = fraction_dict[label]
    #     if fraction > 0.01:
    #         fig.add_trace(go.Violin(x=scores_dict['Test Real'][functional_group_inds[label]], name=f'Fraction containing {label}={fraction:.2f}',
    #                                 line_color=plot_color_dict[label], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
    #
    # fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
    # fig.update_layout(xaxis_title='Model Score')
    # fig.update_layout(showlegend=False)
    #
    # fig.layout.margin = layout.margin
    # fig.write_image('../paper1_figs/scores_separation_table.png')
    # if config.machine == 'local':
    #     fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[f'{key} {fraction_dict[key]:.2f}' for key in sorted_functional_group_keys],
                               y=[np.average(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                               error_y=dict(type='data',
                                            array=[np.std(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                            visible=True
                                            ),
                               showlegend=False,
                               mode='markers'))

    fig.update_layout(yaxis_title='Mean Score and Standard Deviation')
    fig.update_layout(width=1600, height=600)
    fig.update_layout(font=dict(size=12))
    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/functional_group_scores.png', scale=2)
    if config.machine == 'local':
        fig.show()
    aa = 1
    '''
    9. Score vs. EMD on BT submissions
    '''

    all_scores = np.concatenate([(scores_dict[key]) for key in scores_dict.keys() if key in blind_test_targets])  # if key in normed_energy_dict.keys()])
    full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
    inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])

    clip = np.quantile(full_rdf, 0.99) * 1.9
    # full_rdf = np.clip(full_rdf, a_min=0, a_max=clip)

    fig = make_subplots(rows=2, cols=4,
                        vertical_spacing=0.075,
                        subplot_titles=(list(rdf_full_distance_dict.keys())),
                        x_title='Distance from Target',
                        y_title='Model Score')  # + ['All'])

    from scipy.stats import gaussian_kde
    for i, label in enumerate(rdf_full_distance_dict.keys()):
        row = i // 4 + 1
        col = i % 4 + 1
        dist = rdf_full_distance_dict[label]
        dist = np.clip(dist, a_min=0, a_max=clip)
        xline = np.asarray([np.amin(dist), np.amax(dist)])
        linreg_result = linregress(dist, scores_dict[label])
        yline = xline * linreg_result.slope + linreg_result.intercept

        xy = np.vstack([dist, scores_dict[label]])
        z = gaussian_kde(xy)(xy)

        fig.add_trace(go.Scattergl(x=dist, y=scores_dict[label], showlegend=False,
                                   mode='markers', marker=dict(color=z), opacity=0.1),
                      row=row, col=col)
        fig.add_trace(go.Scattergl(x=np.zeros(1), y=scores_dict[label + ' exp'], showlegend=False, mode='markers',
                                   marker=dict(color='Black', size=10, line=dict(color='White', width=2))), row=row, col=col)

        fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=row, col=col)
        fig.update_xaxes(range=[-5, clip], row=row, col=col)
    #
    # xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
    # linreg_result = linregress(full_rdf, all_scores)
    # yline = xline * linreg_result.slope + linreg_result.intercept
    # fig.add_trace(go.Scattergl(x=full_rdf, y=all_scores, showlegend=False,
    #                            mode='markers', ),
    #               row=2, col=4)
    # fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'All Targets R={linreg_result.rvalue:.3f}'), row=2, col=4)
    # fig.update_xaxes(range=[-5, clip], row=2, col=4)

    fig.update_layout(width=1000, height=500)
    fig.layout.margin = layout.margin
    fig.layout.margin.b = 60
    fig.layout.margin.l = 90
    fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    10. Interesting Group-wise analysis
    '''

    target_identifiers = {}
    rankings = {}
    group = {}
    list_num = {}
    for label in ['XXII', 'XXIII', 'XXVI']:
        target_identifiers[label] = [identifiers_list[all_identifiers[label][n]] for n in range(len(all_identifiers[label]))]
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

    quantiles = [np.quantile(normed_scores_dict['Test Real'], 0.01), np.quantile(normed_scores_dict['Test Real'], 0.05), np.quantile(normed_scores_dict['Test Real'], 0.1)]

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
    fig.write_image('../paper1_figs/interesting_groups.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    S1. All group-wise analysis
    '''

    for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
        names = np.unique(list(group[label]))
        uniques = len(names)
        rows = int(np.floor(np.sqrt(uniques)))
        cols = int(np.ceil(np.sqrt(uniques)) + 1)
        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=(names), x_title='Group Ranking', y_title='Model Score', vertical_spacing=0.1)

        for j, group_name in enumerate(np.unique(group[label])):
            good_inds = np.where(np.asarray(group[label]) == group_name)[0]
            submissions_list_num = np.asarray(list_num[label])[good_inds]
            list1_inds = np.where(submissions_list_num == 1)[0]
            list2_inds = np.where(submissions_list_num == 2)[0]

            xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list1_inds]])])
            linreg_result = linregress(np.asarray(rankings[label])[good_inds[list1_inds]], np.asarray(scores_dict[label])[good_inds[list1_inds]])
            yline = xline * linreg_result.slope + linreg_result.intercept

            fig.add_trace(go.Scattergl(x=np.asarray(rankings[label])[good_inds], y=np.asarray(scores_dict[label])[good_inds], showlegend=False,
                                       mode='markers', opacity=0.5, marker=dict(size=6, color=submissions_list_num, colorscale='portland', cmax=2, cmin=1, showscale=False)),
                          row=j // cols + 1, col=j % cols + 1)

            fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{group_name} R={linreg_result.rvalue:.3f}', line=dict(color='#0c4dae')), row=j // cols + 1, col=j % cols + 1)

            if len(list2_inds) > 0:
                xline = np.asarray([0, max(np.asarray(rankings[label])[good_inds[list2_inds]])])
                linreg_result2 = linregress(np.asarray(rankings[label])[good_inds[list2_inds]], np.asarray(scores_dict[label])[good_inds[list2_inds]])
                yline2 = xline * linreg_result2.slope + linreg_result2.intercept
                fig.add_trace(go.Scattergl(x=xline, y=yline2, name=f'{group_name} R={linreg_result2.rvalue:.3f}', line=dict(color='#d60000')), row=j // cols + 1, col=j % cols + 1)

        fig.update_layout(title=label)

        fig.update_layout(width=1200, height=600)
        fig.layout.margin = layout.margin
        fig.layout.margin.t = 50
        fig.layout.margin.b = 55
        fig.layout.margin.l = 60
        fig.write_image(f'../paper1_figs/groupwise_analysis_{i}.png', scale=4)
        if config.machine == 'local':
            fig.show()

    '''
    S2.  score correlates
    '''

    # correlate losses with molecular features
    tracking_features = np.asarray(tracking_features)
    g_loss_correlations = np.zeros(config.dataDims['num_tracking_features'])
    features = []
    ind = 0
    for i in range(config.dataDims['num_tracking_features']):  # not that interesting
        if ('spacegroup' not in config.dataDims['tracking_features'][i]) and \
                ('system' not in config.dataDims['tracking_features'][i]) and \
                ('density' not in config.dataDims['tracking_features'][i]):
            if (np.average(tracking_features[:, i] != 0) > 0.05) and \
                    (config.dataDims['tracking_features'][i] != 'crystal z prime') and \
                    (config.dataDims['tracking_features'][i] != 'molecule point group is C1'):  # if we have at least 1# relevance
                corr = np.corrcoef(scores_dict['Test Real'], tracking_features[:, i], rowvar=False)[0, 1]
                if np.abs(corr) > 0.05:
                    features.append(config.dataDims['tracking_features'][i])
                    g_loss_correlations[ind] = corr
                    ind += 1

    g_loss_correlations = g_loss_correlations[:ind]

    g_sort_inds = np.argsort(g_loss_correlations)
    g_loss_correlations = g_loss_correlations[g_sort_inds]
    features_sorted = [features[i] for i in g_sort_inds]
    features_sorted_cleaned_i = [feat.replace('molecule', 'mol') for feat in features_sorted]
    features_sorted_cleaned_ii = [feat.replace('crystal', 'crys') for feat in features_sorted_cleaned_i]
    features_sorted_cleaned = [feat.replace('mol atom heavier than', 'atomic # >') for feat in features_sorted_cleaned_ii]

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

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=('a) Molecule & Crystal Features', 'b) Atom Fractions', 'c) Contains Functional Groups'), x_title='R Value')

    fig.add_trace(go.Bar(
        y=[feat for feat in features_sorted_cleaned if 'has' not in feat and 'fraction' not in feat],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(100,0,0,1)')
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=[feat.replace('mol ', '').replace('fraction', '') for feat in features_sorted_cleaned if 'has' not in feat and 'fraction' in feat],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(0,0,100,1)')
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        y=[feat.replace('mol has ', '') for feat in features_sorted_cleaned if 'has' in feat and 'fraction' not in feat],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat]).astype('float16'),
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
    fig.write_image('../paper1_figs/scores_correlates.png', scale=4)
    if config.machine == 'local':
        fig.show()

    fig = go.Figure()
    label = 'Test Real'
    fig.add_trace(go.Violin(x=scores_dict[label], name='Real',
                            side='positive', orientation='h', width=4,
                            meanline_visible=True, bandwidth=bandwidth1, points=False),
                  )
    # label = 'BT Submissions'
    # fig.add_trace(go.Violin(x=BT_submission_scores, name='BT 5&6 Subs.',
    #                         side='positive', orientation='h', width=4,
    #                         meanline_visible=True, bandwidth=bandwidth1, points=False),
    #               )

    fig.add_trace(go.Violin(x=np.concatenate((scores_dict['Test Randn'], scores_dict['Test Distorted'])), name='Fake',
                            side='positive', orientation='h', width=4,
                            meanline_visible=True, bandwidth=bandwidth1, points=False),
                  )


    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=20), tickfont=dict(size=20))
    fig.update_xaxes(title_text=r'$\text{Score}$')
    fig.update_layout(showlegend=False, yaxis_showgrid=True, xaxis_showgrid=False, width=350, height=350)
    fig.layout.margin = layout.margin
    fig.layout.margin.b = 60

    fig.write_image('../paper1_figs/ToC_discriminator.png', scale=4)
    if config.machine == 'local':
        fig.show()

    aa = 1
    return None
