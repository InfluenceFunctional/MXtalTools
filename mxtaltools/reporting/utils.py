import numpy as np
import wandb
from plotly import graph_objects as go
from scipy.stats import gaussian_kde


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

def lightweight_one_sided_violin(data, n_points=100, bandwidth_factor=1.0, data_min=None, data_max=None):
    """Create lightweight one-sided violin data using KDE with minimal points"""

    if len(data) == 0:
        return np.array([]), np.array([])

    # Create KDE
    kde = gaussian_kde(data, bw_method=bandwidth_factor)

    # Create evaluation points
    if data_min is not None:
        data_min = max(data_min, data.min())
    else:
        data_min = data.min()
    if data_max is not None:
        data_max = min(data_max, data.max())
    else:
        data_max = data.max()

    data_range = data_max - data_min
    x_vals = np.linspace(data_min - 0.1 * data_range,
                         data_max + 0.1 * data_range,
                         n_points)

    # Evaluate KDE
    y_vals = kde(x_vals)

    # Normalize for consistent width (optional)
    y_vals = y_vals / y_vals.max() * 2  # Scale to max width of 2

    return x_vals, y_vals


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

            wandb.log(data={f'Average_{target}_exp_score': np.average(scores)}, commit=False)

        if crystals_for_targets[target] != []:  # record sample data
            target_indices = crystals_for_targets[target]
            scores = extra_test_dict['discriminator_real_score'][target_indices]
            scores_dict[target] = scores
            tracking_features_dict[target] = {feat: vec for feat, vec in zip(dataDims['tracking_features'],
                                                                             extra_test_dict['tracking_features'][
                                                                                 target_indices].T)}

            vdw_penalty_dict[target] = extra_test_dict['real_vdw_penalty'][target_indices]

            distance_dict[target] = extra_test_dict['discriminator_real_predicted_distance'][target_indices]

            wandb.log(data={f'Average_{target}_score': np.average(scores)}, commit=False)
            wandb.log(data={f'Average_{target}_std': np.std(scores)}, commit=False)

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
    (scores_dict, vdw_penalty_dict,  # todo check/rewrite - got rid of tracking features fed through samples
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

    wandb.log(data={'Average BT submission score': np.average(BT_submission_scores),
                    'Average BT target score': np.average(BT_target_scores),
                    'BT submission score std': np.std(BT_target_scores),
                    'BT target score std': np.std(BT_target_scores),
                    'Average BT submission distance': np.average(BT_submission_distances),
                    'Average BT target distance': np.average(BT_target_distances),
                    'BT submission distance std': np.std(BT_target_distances),
                    'BT target distance std': np.std(BT_target_distances)},
              commit=False)

    return bt_score_correlates, scores_dict, pred_distance_dict, \
        crystals_for_targets, blind_test_targets, target_identifiers, target_identifiers_inds, \
        BT_target_scores, BT_submission_scores, \
        vdw_penalty_dict, tracking_features_dict, \
        BT_target_distances, BT_submission_distances


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
                    generator_losses['packing_normed_mae'] = np.abs(
                        generator_losses['packing_prediction'] - generator_losses['packing_target']) / \
                                                             generator_losses['packing_target']
                    del generator_losses['packing_prediction'], generator_losses['packing_target']
            else:
                generator_losses[key[10:]] = None

    return generator_losses, {key: np.average(value) for i, (key, value) in enumerate(generator_losses.items()) if
                              value is not None}


def process_discriminator_outputs(dataDims, epoch_stats_dict, extra_test_dict=None):
    scores_dict = {}
    vdw_penalty_dict = {}
    #tracking_features_dict = {}
    packing_coeff_dict = {}
    pred_distance_dict = {}
    true_distance_dict = {}

    generator_inds = np.where(epoch_stats_dict['generator_sample_source'] == 2)[0]
    randn_inds = np.where(epoch_stats_dict['generator_sample_source'] == 1)[0]
    distorted_inds = np.where(epoch_stats_dict['generator_sample_source'] == 0)[0]

    scores_dict['CSD'] = epoch_stats_dict['discriminator_real_score']
    scores_dict['Gaussian'] = epoch_stats_dict['discriminator_fake_score'][randn_inds]
    scores_dict['Generator'] = epoch_stats_dict['discriminator_fake_score'][generator_inds]
    scores_dict['Distorted'] = epoch_stats_dict['discriminator_fake_score'][distorted_inds]
    #
    # tracking_features_dict['CSD'] = {feat: vec for feat, vec in zip(dataDims['tracking_features'],
    #                                                                 epoch_stats_dict['tracking_features'].T)}
    # tracking_features_dict['Distorted'] = {feat: vec for feat, vec in
    #                                        zip(dataDims['tracking_features'],
    #                                            epoch_stats_dict['tracking_features'][distorted_inds].T)}
    # tracking_features_dict['Gaussian'] = {feat: vec for feat, vec in
    #                                       zip(dataDims['tracking_features'],
    #                                           epoch_stats_dict['tracking_features'][randn_inds].T)}
    # tracking_features_dict['Generator'] = {feat: vec for feat, vec in
    #                                        zip(dataDims['tracking_features'],
    #                                            epoch_stats_dict['tracking_features'][generator_inds].T)}

    vdw_penalty_dict['CSD'] = epoch_stats_dict['real_vdw_penalty']
    vdw_penalty_dict['Gaussian'] = epoch_stats_dict['fake_vdw_penalty'][randn_inds]
    vdw_penalty_dict['Generator'] = epoch_stats_dict['fake_vdw_penalty'][generator_inds]
    vdw_penalty_dict['Distorted'] = epoch_stats_dict['fake_vdw_penalty'][distorted_inds]

    packing_coeff_dict['CSD'] = epoch_stats_dict['real_packing_coeff']
    packing_coeff_dict['Gaussian'] = epoch_stats_dict['fake_packing_coeff'][randn_inds]
    packing_coeff_dict['Generator'] = epoch_stats_dict['fake_packing_coeff'][generator_inds]
    packing_coeff_dict['Distorted'] = epoch_stats_dict['fake_packing_coeff'][distorted_inds]

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
        #tracking_features_dict['extra_test'] = {feat: vec for feat, vec in zip(dataDims['tracking_features'], extra_test_dict['tracking_features'].T)}
        vdw_penalty_dict['extra_test'] = extra_test_dict['real_vdw_penalty']
        packing_coeff_dict['extra_test'] = extra_test_dict['real_packing_coeff']
        pred_distance_dict['extra_test'] = extra_test_dict['discriminator_real_predicted_distance']
        true_distance_dict['extra_test'] = extra_test_dict['discriminator_real_true_distance']

    return scores_dict, vdw_penalty_dict, packing_coeff_dict, pred_distance_dict, true_distance_dict
