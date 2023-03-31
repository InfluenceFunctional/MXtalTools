from plotly.colors import n_colors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from common.utils import normalize, earth_movers_distance_np, histogram_overlap
from models.utils import softmax_and_score, norm_scores
from common.rdf_calculation import compute_rdf_distance_old
from scipy.stats import linregress


def blind_test_analysis(config, wandb, train_epoch_stats_dict, test_epoch_stats_dict, extra_test_dict):
    '''
    analyze and plot discriminator training results on blind test 5 & 6 data
    '''
    identifiers_list = extra_test_dict['identifiers']
    score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, \
    scores_dict, all_identifiers, blind_test_targets, target_identifiers, \
    target_identifiers_inds, BT_target_scores, BT_submission_scores, \
    BT_scores_dists, BT_balanced_dist, vdw_penalty_dict, tracking_features_dict = \
        process_discriminator_evaluation_data(config, wandb, extra_test_dict, test_epoch_stats_dict, train_epoch_stats_dict)

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
        )
    )

    violin_scores_plot(config, wandb, layout, all_identifiers, scores_dict, target_identifiers_inds)
    violin_vdw_plot(config, wandb, layout, all_identifiers, vdw_penalty_dict, target_identifiers_inds)
    violin_scores_plot2(config, wandb, layout, all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist)
    functional_group_violin_plot(config, wandb, layout, scores_dict, tracking_features_names=config.dataDims['tracking features dict'], tracking_features=test_epoch_stats_dict['tracking features'])
    scores_distributions_plot(config, wandb, layout, all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist)
    score_correlations_plot(config, wandb, layout, test_epoch_stats_dict['tracking features'], scores_dict)
    distance_vs_score_plot(config, wandb, layout, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets)
    targetwise_distance_vs_score_plot(config, wandb, layout, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets)
    group, rankings, list_num = target_ranking_analysis(config, wandb, layout, identifiers_list, scores_dict, all_identifiers)
    groupwise_target_ranking_analysis(config, wandb, layout, group, rankings, list_num, scores_dict)

    return None


def process_discriminator_evaluation_data(config, wandb, extra_test_dict, test_epoch_stats_dict, train_epoch_stats_dict, size_normed_score=False):
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
    all_identifiers = {key: [] for key in blind_test_targets}
    for i in range(len(extra_test_dict['identifiers'])):
        item = extra_test_dict['identifiers'][i]
        for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
            if blind_test_targets[-1 - j] in item:
                all_identifiers[blind_test_targets[-1 - j]].append(i)
                break

    # determine which samples ARE the targets (mixed in the dataloader)
    target_identifiers_inds = {key: [] for key in blind_test_targets}
    for i in range(len(extra_test_dict['identifiers'])):
        item = extra_test_dict['identifiers'][i]
        for key in target_identifiers.keys():
            if item == target_identifiers[key]:
                target_identifiers_inds[key] = i

    '''
    record all the stats for the CSD data
    '''
    scores_dict = {}
    vdw_penalty_dict = {}
    tracking_features_dict = {}
    # nf_inds = np.where(test_epoch_stats_dict['generator sample source'] == 0)
    randn_inds = np.where(test_epoch_stats_dict['generator sample source'] == 1)[0]
    distorted_inds = np.where(test_epoch_stats_dict['generator sample source'] == 2)[0]

    if True: # config.gan_loss == 'standard':
        scores_dict['Test Real'] = softmax_and_score(test_epoch_stats_dict['discriminator real score'],old_method = True, correct_discontinuity = True)
        scores_dict['Test Randn'] = softmax_and_score(test_epoch_stats_dict['discriminator fake score'][randn_inds],old_method = True, correct_discontinuity = True)
        # scores_dict['Test NF'] = np_softmax(test_epoch_stats_dict['discriminator fake score'][nf_inds])[:, 1]
        scores_dict['Test Distorted'] = softmax_and_score(test_epoch_stats_dict['discriminator fake score'][distorted_inds],old_method = True, correct_discontinuity = True)

        tracking_features_dict['Test Real'] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'].T)}
        tracking_features_dict['Test Distorted'] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'][distorted_inds].T)}
        tracking_features_dict['Test Randn'] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'][randn_inds].T)}

        if size_normed_score:
            scores_dict['Test Real'] = norm_scores(scores_dict['Test Real'], test_epoch_stats_dict['tracking features'], config.dataDims)
            scores_dict['Test Randn'] = norm_scores(scores_dict['Test Randn'], test_epoch_stats_dict['tracking features'][randn_inds], config.dataDims)
            scores_dict['Test Distorted'] = norm_scores(scores_dict['Test Distorted'], test_epoch_stats_dict['tracking features'][distorted_inds], config.dataDims)

        if train_epoch_stats_dict is not None:
            scores_dict['Train Real'] = softmax_and_score(train_epoch_stats_dict['discriminator real score'],old_method = True, correct_discontinuity = True)
            tracking_features_dict['Train Real'] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'], train_epoch_stats_dict['tracking features'].T)}

            if size_normed_score:
                scores_dict['Train Real'] = norm_scores(scores_dict['Train Real'], train_epoch_stats_dict['tracking features'], config.dataDims)

            vdw_penalty_dict['Train Real'] = train_epoch_stats_dict['real vdW penalty']
            wandb.log({'Average Train score': np.average(scores_dict['Train Real'])})
            wandb.log({'Train score std': np.std(scores_dict['Train Real'])})

        vdw_penalty_dict['Test Real'] = test_epoch_stats_dict['real vdW penalty']
        vdw_penalty_dict['Test Randn'] = test_epoch_stats_dict['fake vdW penalty'][randn_inds]
        vdw_penalty_dict['Test Distorted'] = test_epoch_stats_dict['fake vdW penalty'][distorted_inds]

        wandb.log({'Average Test score': np.average(scores_dict['Test Real'])})
        wandb.log({'Average Randn Fake score': np.average(scores_dict['Test Randn'])})
        # wandb.log({'Average NF Fake score': np.average(scores_dict['Test NF'])})
        wandb.log({'Average Distorted Fake score': np.average(scores_dict['Test Distorted'])})

        wandb.log({'Test Real std': np.std(scores_dict['Test Real'])})
        wandb.log({'Distorted Fake score std': np.std(scores_dict['Test Distorted'])})
        wandb.log({'Randn score std': np.std(scores_dict['Test Randn'])})

    else:
        print("Analysis only setup for cross entropy loss")
        assert False

    '''
    build property dicts for the submissions and BT targets
    '''
    score_correlations_dict = {}
    rdf_full_distance_dict = {}
    rdf_inter_distance_dict = {}

    for target in all_identifiers.keys():  # run the analysis for each target
        if target_identifiers_inds[target] != []:  # record target data

            target_index = target_identifiers_inds[target]
            raw_scores = extra_test_dict['scores'][target_index]
            scores = softmax_and_score(raw_scores[None,:],old_method = True, correct_discontinuity = True)
            scores_dict[target + ' exp'] = scores

            tracking_features_dict[target + ' exp'] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'], extra_test_dict['tracking features'][target_index][None, :].T)}

            if size_normed_score:
                scores_dict[target + ' exp'] = norm_scores(scores_dict[target + ' exp'], extra_test_dict['tracking features'][target_index][None, :], config.dataDims)

            vdw_penalty_dict[target + ' exp'] = extra_test_dict['vdW penalty'][target_index][None]

            wandb.log({f'Average {target} exp score': np.average(scores)})

            target_full_rdf = extra_test_dict['full rdf'][target_index]
            target_inter_rdf = extra_test_dict['intermolecular rdf'][target_index]

        if all_identifiers[target] != []:  # record sample data
            target_indices = all_identifiers[target]
            raw_scores = extra_test_dict['scores'][target_indices]
            scores = softmax_and_score(raw_scores, old_method=True, correct_discontinuity=True)
            scores_dict[target] = scores
            tracking_features_dict[target] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'], extra_test_dict['tracking features'][target_indices].T)}

            if size_normed_score:
                scores_dict[target] = norm_scores(scores_dict[target], extra_test_dict['tracking features'][target_indices], config.dataDims)

            # energy_dict[target] = extra_test_dict['atomistic energy'][target_indices]
            vdw_penalty_dict[target] = extra_test_dict['vdW penalty'][target_indices]

            wandb.log({f'Average {target} score': np.average(scores)})
            wandb.log({f'Average {target} std': np.std(scores)})

            submission_full_rdf = extra_test_dict['full rdf'][target_indices]
            submission_inter_rdf = extra_test_dict['intermolecular rdf'][target_indices]

            rdf_full_distance_dict[target] = compute_rdf_distance_old(target_full_rdf, submission_full_rdf)
            rdf_inter_distance_dict[target] = compute_rdf_distance_old(target_inter_rdf, submission_inter_rdf)

            # correlate losses with molecular features
            tracking_features = np.asarray(extra_test_dict['tracking features'])
            loss_correlations = np.zeros(config.dataDims['num tracking features'])
            features = []
            for j in range(tracking_features.shape[-1]):  # not that interesting
                features.append(config.dataDims['tracking features dict'][j])
                loss_correlations[j] = np.corrcoef(scores, tracking_features[target_indices, j], rowvar=False)[0, 1]

            score_correlations_dict[target] = loss_correlations

    # compute loss correlates
    loss_correlations = np.zeros(config.dataDims['num tracking features'])
    features = []
    for j in range(config.dataDims['num tracking features']):  # not that interesting
        features.append(config.dataDims['tracking features dict'][j])
        loss_correlations[j] = np.corrcoef(scores_dict['Test Real'], test_epoch_stats_dict['tracking features'][:, j], rowvar=False)[0, 1]
    score_correlations_dict['Test Real'] = loss_correlations

    # collect all BT targets & submissions into single dicts
    BT_target_scores = np.concatenate([scores_dict[key] for key in scores_dict.keys() if 'exp' in key])
    BT_submission_scores = np.concatenate([scores_dict[key] for key in scores_dict.keys() if key in all_identifiers.keys()])
    BT_scores_dists = {key: np.histogram(scores_dict[key], bins=200, range=[-15, 15])[0] / len(scores_dict[key]) for key in scores_dict.keys() if key in all_identifiers.keys()}
    BT_balanced_dist = np.average(np.stack(list(BT_scores_dists.values())), axis=0)

    wandb.log({'Average BT submission score': np.average(BT_submission_scores)})
    wandb.log({'Average BT target score': np.average(BT_target_scores)})
    wandb.log({'BT submission score std': np.std(BT_target_scores)})
    wandb.log({'BT target score std': np.std(BT_target_scores)})

    return score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, \
           all_identifiers, blind_test_targets, target_identifiers, target_identifiers_inds, \
           BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist, \
           vdw_penalty_dict, tracking_features_dict

def violin_scores_plot(config, wandb, layout, all_identifiers, scores_dict, target_identifiers_inds):
    '''
    prep violin figure colors
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

    '''
    violin scores plot
    '''
    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth = scores_range / 200

    fig = go.Figure()
    for i, label in enumerate(scores_dict.keys()):
        if 'exp' in label:
            fig.add_trace(go.Violin(x=scores_dict[label], name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6))
        else:
            fig.add_trace(go.Violin(x=scores_dict[label], name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False))

        quantiles = [np.quantile(scores_dict['Test Real'], 0.01), np.quantile(scores_dict['Test Real'], 0.05), np.quantile(scores_dict['Test Real'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'])
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'])
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'])

    fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Model Score')
    fig.update_layout(font=dict(size=18))
    fig.layout.margin = layout.margin
    wandb.log({'Discriminator Test Scores': fig})

def violin_vdw_plot(config, wandb, layout, all_identifiers, vdw_penalty_dict, target_identifiers_inds):
    '''
    prep violin figure colors
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

    '''
    violin scores plot
    '''
    scores_range = np.ptp(np.log10(1e-6 + np.concatenate(list(vdw_penalty_dict.values()))))
    bandwidth = scores_range / 200

    fig = go.Figure()
    for i, label in enumerate(vdw_penalty_dict.keys()):
        if 'exp' in label:
            fig.add_trace(go.Violin(x=np.log10(vdw_penalty_dict[label] + 1e-6), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=6))
        else:
            fig.add_trace(go.Violin(x=np.log10(vdw_penalty_dict[label] + 1e-6), name=label, line_color=plot_color_dict[label], side='positive', orientation='h', width=4, meanline_visible=True, bandwidth=bandwidth, points=False))

    fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Model Score')
    fig.update_layout(font=dict(size=18))
    fig.layout.margin = layout.margin
    fig.update_layout(xaxis_title='log vdW penalty')
    wandb.log({'vdW Penalty': fig})

def violin_scores_plot2(config, wandb, layout, all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist):

    plot_color_dict = {}
    plot_color_dict['Test Real'] = ('rgb(200,0,50)')  # test
    plot_color_dict['BT Targets'] = ('rgb(50,0,50)')
    plot_color_dict['BT Submissions'] = ('rgb(50,150,250)')

    '''
    violin scores plot
    '''
    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth = scores_range / 200

    fig = go.Figure()
    # 99 and 95 quantiles
    quantiles = [np.quantile(scores_dict['Test Real'], 0.01), np.quantile(scores_dict['Test Real'], 0.05), np.quantile(scores_dict['Test Real'], 0.1)]
    fig.add_vline(x=quantiles[0], line_dash='dash', line_color=plot_color_dict['Test Real'])
    fig.add_vline(x=quantiles[1], line_dash='dash', line_color=plot_color_dict['Test Real'])
    fig.add_vline(x=quantiles[2], line_dash='dash', line_color=plot_color_dict['Test Real'])
    # test data
    fig.add_trace(go.Violin(x=scores_dict['Test Real'], name='CSD Test', line_color=plot_color_dict['Test Real'], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))

    # BT distribution
    fig.add_trace(go.Violin(x=BT_target_scores, name='BT Targets', line_color=plot_color_dict['BT Targets'], side='positive', orientation='h', width=1, meanline_visible=True, bandwidth=bandwidth / 100, points=False))
    # Submissions
    fig.add_trace(go.Violin(x=BT_submission_scores, name='BT Submissions', line_color=plot_color_dict['BT Submissions'], side='positive', orientation='h', width=2, meanline_visible=True, bandwidth=bandwidth, points=False))
    quantiles = [np.quantile(BT_submission_scores, 0.01), np.quantile(BT_submission_scores, 0.05), np.quantile(BT_submission_scores, 0.1)]
    fig.add_shape(type="line",
                  x0=quantiles[0], y0=2, x1=quantiles[0], y1=3,
                  line=dict(color=plot_color_dict['BT Submissions'], dash='dash'))
    fig.add_shape(type="line",
                  x0=quantiles[1], y0=2, x1=quantiles[1], y1=3,
                  line=dict(color=plot_color_dict['BT Submissions'], dash='dash'))
    fig.add_shape(type="line",
                  x0=quantiles[2], y0=2, x1=quantiles[2], y1=3,
                  line=dict(color=plot_color_dict['BT Submissions'], dash='dash'))
    fig.update_layout(legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Model Score')
    # fig.show()
    fig.update_layout(title='Scores and 0.01, 0.05, 0.1 quantiles')
    fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Model Score')
    fig.update_layout(font=dict(size=18))
    fig.layout.margin = layout.margin
    fig.layout.margin.t = 50
    wandb.log({'Scores Distribution': fig})

    return None

def functional_group_violin_plot(config, wandb, layout, scores_dict, tracking_features_names, tracking_features):
    '''
    plot scores distributions for different functional groups
    '''

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

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=[f'{key} {fraction_dict[key]:.2f}' for key in sorted_functional_group_keys],
                               y=[np.average(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                               error_y=dict(type='data',
                                            array=[np.std(scores_dict['Test Real'][functional_group_inds[key]]) for key in sorted_functional_group_keys],
                                            visible=True
                                            ),
                               showlegend=False,
                               mode='markers'))

    fig.update_layout(xaxis_title='Molecule Containing Functional Groups & Elements')
    fig.update_layout(yaxis_title='Mean Score and Standard Deviation')
    fig.layout.margin = layout.margin

    wandb.log({'Functional Group Scores Statistics': fig})

    # '''
    # violin scores plot
    # '''
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
    # fig.show()

    # wandb.log({'Functional Group Scores Distributions': fig})

    return None

def scores_distributions_plot(config, wandb, layout, all_identifiers, scores_dict, BT_target_scores, BT_submission_scores, BT_scores_dists, BT_balanced_dist):
    '''
    compute fraction of submissions below each quantile of the CSD data
    compute fraction of submissions above & below each experimental structrue
    '''
    csd_scores = scores_dict['Test Real']
    hists = {}
    scores_hist, rr = np.histogram(csd_scores, bins=200, range=[-15, 15], density=True)
    hists['Test Real'] = scores_hist / scores_hist.sum()

    submissions_hist, _ = np.histogram(BT_submission_scores, bins=200, range=[-15, 15], density=True)
    hists['BT submissions'] = submissions_hist / submissions_hist.sum()

    distorted_hist, rr = np.histogram(scores_dict['Test Distorted'], bins=200, range=[-15, 15], density=True)
    hists['Test Distorted'] = distorted_hist / distorted_hist.sum()

    randn_hist, rr = np.histogram(scores_dict['Test Randn'], bins=200, range=[-15, 15], density=True)
    hists['Test Randn'] = randn_hist / randn_hist.sum()

    emds = {}
    overlaps = {}
    for i, label1 in enumerate(hists.keys()):
        for j, label2 in enumerate(hists.keys()):
            if i > j:
                emds[f'{label1} <-> {label2} emd'] = earth_movers_distance_np(hists[label1], hists[label2])
                overlaps[f'{label1} <-> {label2} overlap'] = histogram_overlap(hists[label1], hists[label2])

    wandb.log(emds)
    wandb.log(overlaps)

    vals = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
    quantiles = np.quantile(csd_scores, vals)
    submissions_fraction_below_csd_quantile = {value: np.average(BT_submission_scores < cutoff) for value, cutoff in zip(vals, quantiles)}
    targets_fraction_below_csd_quantile = {value: np.average(BT_target_scores < cutoff) for value, cutoff in zip(vals, quantiles)}

    submissions_fraction_below_target = {key: np.average(scores_dict[key] < scores_dict[key + ' exp']) for key in all_identifiers.keys() if key in scores_dict.keys()}
    submissions_average_below_target = np.average(list(submissions_fraction_below_target.values()))

    distributions_dict = {
        'Submissions below CSD quantile': submissions_fraction_below_csd_quantile,
        'Targets below CSD quantile': targets_fraction_below_csd_quantile,
        'Submissions below target': submissions_fraction_below_target,
        'Submissions below target mean': submissions_average_below_target,
    }
    wandb.log(distributions_dict)

    return None

def score_correlations_plot(config, wandb, layout, tracking_features, scores_dict):

    # correlate losses with molecular features
    tracking_features = np.asarray(tracking_features)
    g_loss_correlations = np.zeros(config.dataDims['num tracking features'])
    features = []
    ind = 0
    for i in range(config.dataDims['num tracking features']):  # not that interesting
        if 'spacegroup' not in config.dataDims['tracking features dict'][i]:
            if (np.average(tracking_features[:, i] != 0) > 0.01) and \
                    (config.dataDims['tracking features dict'][i] != 'crystal z prime') and \
                    (config.dataDims['tracking features dict'][i] != 'molecule point group is C1'):  # if we have at least 1# relevance
                corr = np.corrcoef(scores_dict['Test Real'], tracking_features[:, i], rowvar=False)[0, 1]
                if np.abs(corr) > 0.05:
                    features.append(config.dataDims['tracking features dict'][i])
                    g_loss_correlations[ind] = corr
                    ind += 1

    g_loss_correlations = g_loss_correlations[:ind]

    g_sort_inds = np.argsort(g_loss_correlations)
    g_loss_correlations = g_loss_correlations[g_sort_inds]
    features_sorted = [features[i] for i in g_sort_inds]
    g_loss_dict = {feat: corr for feat, corr in zip(features_sorted, g_loss_correlations)}

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14, subplot_titles=('a)', 'b)', 'c)'), x_title='R Value')

    fig.add_trace(go.Bar(
        y=[feat for feat in features_sorted if 'has' not in feat and 'fraction' not in feat],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' not in feat]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(100,0,0,1)')
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=[feat for feat in features_sorted if 'has' in feat and 'fraction' not in feat],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' in feat and 'fraction' not in feat]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(0,100,0,1)')
    ), row=1, col=3)
    fig.add_trace(go.Bar(
        y=[feat for feat in features_sorted if 'has' not in feat and 'fraction' in feat],
        x=[g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat],
        orientation='h',
        text=np.asarray([g for i, (feat, g) in enumerate(g_loss_dict.items()) if 'has' not in feat and 'fraction' in feat]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}',
        marker=dict(color='rgba(0,0,100,1)')
    ), row=1, col=2)

    fig.update_yaxes(tickfont=dict(size=14), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=14), row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=10), row=1, col=3)
    fig.update_xaxes(range=[np.amin(list(g_loss_dict.values())), np.amax(list(g_loss_dict.values()))])

    fig.layout.annotations[0].update(x=0.02)
    fig.layout.annotations[1].update(x=0.358)
    fig.layout.annotations[2].update(x=0.75)

    fig.layout.margin = layout.margin
    fig.layout.margin.b = 50

    wandb.log({'Test loss correlates': fig})

def distance_vs_score_plot(config, wandb, layout, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets):

    '''
    rdf distance vs score
    '''
    fig = make_subplots(rows=1, cols=2)
    full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
    inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])
    normed_score = np.concatenate([normalize(scores_dict[key]) for key in scores_dict.keys() if key in blind_test_targets])  # if key in normed_energy_dict.keys()])

    clip = np.quantile(full_rdf, 0.99) * 2
    full_rdf = np.clip(full_rdf, a_min=0, a_max=clip)
    inter_rdf = np.clip(inter_rdf, a_min=0, a_max=clip)

    xline = np.asarray([np.amin(full_rdf), np.amax(full_rdf)])
    linreg_result = linregress(full_rdf, normed_score)
    yline = xline * linreg_result.slope + linreg_result.intercept
    fig.add_trace(go.Scattergl(x=full_rdf, y=normed_score, showlegend=False,
                               mode='markers'),
                  row=1, col=1)

    fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Full RDF R={linreg_result.rvalue:.3f}'), row=1, col=1)

    xline = np.asarray([np.amin(inter_rdf), np.amax(inter_rdf)])
    linreg_result = linregress(inter_rdf, normed_score)
    yline = xline * linreg_result.slope + linreg_result.intercept
    fig.add_trace(go.Scattergl(x=inter_rdf, y=normed_score, showlegend=False,
                               mode='markers'),
                  row=1, col=2)
    fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'Intermolecular RDF R={linreg_result.rvalue:.3f}'), row=1, col=2)

    fig.update_layout(title='All BT Targets')
    fig.update_yaxes(title_text='Target-wise Normed Model score', row=1, col=1)
    fig.update_xaxes(title_text='Full RDF Distance', row=1, col=1)
    fig.update_xaxes(title_text='Intermolecular RDF Distance', row=1, col=2)
    wandb.log({f'Distance vs. Score': fig})

def targetwise_distance_vs_score_plot(config, wandb, layout, rdf_full_distance_dict, rdf_inter_distance_dict, scores_dict, blind_test_targets):
    all_scores = np.concatenate([(scores_dict[key]) for key in scores_dict.keys() if key in blind_test_targets])  # if key in normed_energy_dict.keys()])
    full_rdf = np.concatenate([val for val in rdf_full_distance_dict.values()])
    inter_rdf = np.concatenate([val for val in rdf_inter_distance_dict.values()])

    clip = np.quantile(full_rdf, 0.99) * 1.9
    # full_rdf = np.clip(full_rdf, a_min=0, a_max=clip)

    fig = make_subplots(rows=2, cols=4,
                        vertical_spacing=0.075,
                        subplot_titles=(list(rdf_full_distance_dict.keys())))  # + ['All'])

    for i, label in enumerate(rdf_full_distance_dict.keys()):
        row = i // 4 + 1
        col = i % 4 + 1
        dist = rdf_full_distance_dict[label]
        dist = np.clip(dist, a_min=0, a_max=clip)
        xline = np.asarray([np.amin(dist), np.amax(dist)])
        linreg_result = linregress(dist, scores_dict[label])
        yline = xline * linreg_result.slope + linreg_result.intercept

        fig.add_trace(go.Scattergl(x=dist, y=scores_dict[label], showlegend=False,
                                   mode='markers', marker_color='rgba(100,0,0,1)', opacity=0.05),
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

    fig.update_yaxes(title_text='Model Score', row=1, col=1)
    fig.update_yaxes(title_text='Model Score', row=2, col=1)
    fig.layout.margin = layout.margin

    wandb.log({f'Targetwise Distance vs. Score': fig})

def target_ranking_analysis(config, wandb, layout, identifiers_list, scores_dict, all_identifiers):
    '''
    within-submission score vs rankings
    file formats are different between BT 5 and BT6
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
            group[label].append(long_ident[1])

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=(['XXII', 'XXIII', 'XXVI']))

    for i, label in enumerate(['XXII', 'XXIII', 'XXVI']):
        xline = np.asarray([0, 100])
        linreg_result = linregress(rankings[label], scores_dict[label])
        yline = xline * linreg_result.slope + linreg_result.intercept

        fig.add_trace(go.Scattergl(x=rankings[label], y=scores_dict[label], showlegend=False,
                                   mode='markers', marker=dict(size=6, color=list_num[label], colorscale='portland', showscale=False)),
                      row=1, col=i + 1)

        fig.add_trace(go.Scattergl(x=xline, y=yline, name=f'{label} R={linreg_result.rvalue:.3f}'), row=1, col=i + 1)
    fig.update_xaxes(title_text='Submission Rank', row=1, col=1)
    fig.update_yaxes(title_text='Model score', row=1, col=1)
    fig.update_xaxes(title_text='Submission Rank', row=1, col=2)
    fig.update_xaxes(title_text='Submission Rank', row=1, col=3)

    # fig.show()
    wandb.log({'Target Score Rankings': fig})
    return group, rankings, list_num

def groupwise_target_ranking_analysis(config, wandb, layout, group, rankings, list_num, scores_dict):

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
                fig.add_trace(go.Scattergl(x=xline, y=yline2, name=f'{group_name} #2 R={linreg_result2.rvalue:.3f}', line=dict(color='#d60000')), row=j // cols + 1, col=j % cols + 1)

        fig.update_layout(title=label)

        fig.update_layout(width=1000, height=500)
        fig.layout.margin = layout.margin
        fig.layout.margin.t = 50
        fig.layout.margin.b = 55
        fig.layout.margin.l = 60
        wandb.log({f"{label} Groupwise Analysis": fig})

    # specifically interesting groups & targets
    # brandenberg XXVI
    # Brandenberg XXII
    # Facelli XXII
    # Price XXII
    # Goto XXII
    # Brandenberg XXIII

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.075, subplot_titles=(
        ['Brandenburg XXII', 'Brandenburg XXIII', 'Brandenburg XXVI', 'Facelli XXII']),
                        x_title='Model Score')

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
                      row=(ii) // 2 + 1, col=(ii) % 2 + 1)
        fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   name="Submission 2 Score",
                                   showlegend=False,
                                   marker_color='#d60000'),
                      row=(ii) // 2 + 1, col=(ii) % 2 + 1)

    label = 'XXII'
    good_inds = np.where(np.asarray(group[label]) == 'Facelli')[0]
    submissions_list_num = np.asarray(list_num[label])[good_inds]
    list1_inds = np.where(submissions_list_num == 1)[0]
    list2_inds = np.where(submissions_list_num == 2)[0]

    fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
                               histnorm='probability density',
                               nbinsx=50,
                               name="Submission 1 Score",
                               showlegend=False,
                               marker_color='#0c4dae'), row=2, col=2)
    fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
                               histnorm='probability density',
                               nbinsx=50,
                               name="Submission 2 Score",
                               showlegend=False,
                               marker_color='#d60000'), row=2, col=2)

    fig.layout.margin = layout.margin
    fig.layout.margin.b = 50

    wandb.log({'Group Submissions Analysis': fig})

    #
    # label = 'XXII'
    # good_inds = np.where(np.asarray(group[label]) == 'Price')[0]
    # submissions_list_num = np.asarray(list_num[label])[good_inds]
    # list1_inds = np.where(submissions_list_num == 1)[0]
    # list2_inds = np.where(submissions_list_num == 2)[0]
    #
    # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
    #                            histnorm='probability density',
    #                            nbinsx=50,
    #                            name="Submission 1 Score",
    #                            showlegend=False,
    #                            marker_color='#0c4dae'),
    #               row=2, col=2)
    # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
    #                            histnorm='probability density',
    #                            nbinsx=50,
    #                            name="Submission 2 Score",
    #                            showlegend=False,
    #                            marker_color='#d60000'),
    #               row=2, col=2)
    #
    # label = 'XXII'
    # good_inds = np.where(np.asarray(group[label]) == 'Goto')[0]
    # submissions_list_num = np.asarray(list_num[label])[good_inds]
    # list1_inds = np.where(submissions_list_num == 1)[0]
    # list2_inds = np.where(submissions_list_num == 2)[0]
    #
    # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list1_inds]],
    #                            histnorm='probability density',
    #                            nbinsx=50,
    #                            name="Submission 1 Score",
    #                            showlegend=False,
    #                            marker_color='#0c4dae'),
    #               row=2, col=3)
    # fig.add_trace(go.Histogram(x=np.asarray(scores_dict[label])[good_inds[list2_inds]],
    #                            histnorm='probability density',
    #                            nbinsx=50,
    #                            name="Submission 2 Score",
    #                            showlegend=False,
    #                            marker_color='#d60000'),
    #               row=2, col=3)

    return None