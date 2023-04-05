import numpy as np
import torch
from _plotly_utils.colors import n_colors
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, linregress
import plotly.graph_objects as go
import plotly.express as px

from common.utils import update_stats_dict, np_softmax
from crystal_building.builder import update_sg_to_all_crystals
from models.utils import softmax_and_score


def cell_params_analysis(config, wandb, train_loader, test_epoch_stats_dict):
    n_crystal_features = config.dataDims['num lattice features']
    generated_samples = test_epoch_stats_dict['generated cell parameters']
    if generated_samples.ndim == 3:
        generated_samples = generated_samples[0]
    means = config.dataDims['lattice means']
    stds = config.dataDims['lattice stds']

    # slightly expensive to do this every time
    dataset_cell_distribution = np.asarray(
        [train_loader.dataset[ii].cell_params[0].cpu().detach().numpy() for ii in range(len(train_loader.dataset))])

    # raw outputs
    renormalized_samples = np.zeros_like(generated_samples)
    for i in range(generated_samples.shape[1]):
        renormalized_samples[:, i] = generated_samples[:, i] * stds[i] + means[i]

    cleaned_samples = test_epoch_stats_dict['final generated cell parameters']

    overlaps_1d = {}
    sample_means = {}
    sample_stds = {}
    for i, key in enumerate(config.dataDims['lattice features']):
        mini, maxi = np.amin(dataset_cell_distribution[:, i]), np.amax(dataset_cell_distribution[:, i])
        h1, r1 = np.histogram(dataset_cell_distribution[:, i], bins=100, range=(mini, maxi))
        h1 = h1 / len(dataset_cell_distribution[:, i])

        h2, r2 = np.histogram(cleaned_samples[:, i], bins=r1)
        h2 = h2 / len(cleaned_samples[:, i])

        overlaps_1d[f'{key} 1D Overlap'] = np.min(np.concatenate((h1[None], h2[None]), axis=0), axis=0).sum()

        sample_means[f'{key} mean'] = np.mean(cleaned_samples[:, i])
        sample_stds[f'{key} std'] = np.std(cleaned_samples[:, i])

    average_overlap = np.average([overlaps_1d[key] for key in overlaps_1d.keys()])
    overlaps_1d['average 1D overlap'] = average_overlap
    wandb.log(overlaps_1d.copy())
    wandb.log(sample_means)
    wandb.log(sample_stds)

    if config.wandb.log_figures:
        fig_dict = {}  # consider replacing by Joy plot

        # bar graph of 1d overlaps
        fig = go.Figure(go.Bar(
            y=list(overlaps_1d.keys()),
            x=[overlaps_1d[key] for key in overlaps_1d],
            orientation='h',
            marker=dict(color='red')
        ))
        fig_dict['1D overlaps'] = fig

        # 1d Histograms
        for i in range(n_crystal_features):
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=dataset_cell_distribution[:, i],
                histnorm='probability density',
                nbinsx=100,
                name="Dataset samples",
                showlegend=True,
            ))

            fig.add_trace(go.Histogram(
                x=renormalized_samples[:, i],
                histnorm='probability density',
                nbinsx=100,
                name="Samples",
                showlegend=True,
            ))
            fig.add_trace(go.Histogram(
                x=cleaned_samples[:, i],
                histnorm='probability density',
                nbinsx=100,
                name="Cleaned Samples",
                showlegend=True,
            ))
            fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.update_traces(opacity=0.5)

            fig_dict[config.dataDims['lattice features'][i] + ' distribution'] = fig

        wandb.log(fig_dict)


def log_supercell_examples(supercell_examples, i, rand_batch_ind, epoch_stats_dict):
    if (supercell_examples is not None) and (i == rand_batch_ind):  # for a random batch in the epoch
        epoch_stats_dict['generated supercell examples'] = supercell_examples.cpu().detach()
        # if supercell_examples.num_graphs > 100:  # todo find a way to take only the few that we need - maybe using the Collater
        #     print('WARNING. Saving over 100 supercells for analysis')

    epoch_stats_dict = update_stats_dict(epoch_stats_dict, 'final generated cell parameters',
                                         supercell_examples.cell_params.cpu().detach().numpy(), mode='extend')

    del supercell_examples
    return epoch_stats_dict


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
    if epoch_stats_dict['generator packing prediction'] is not None and \
            epoch_stats_dict['generator packing target'] is not None:

        x = np.concatenate(
            epoch_stats_dict['generator packing target'])  # generator_losses['generator per mol vdw loss']
        y = np.concatenate(
            epoch_stats_dict['generator packing prediction'])  # generator_losses['generator packing loss']

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
        fig.update_layout(xaxis_title='packing target', yaxis_title='packing prediction')

        # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
        if config.wandb.log_figures:
            wandb.log({'Cell Packing': fig})
        if (config.machine == 'local') and False:
            fig.show()


def all_losses_plot(config, wandb, epoch_stats_dict, generator_losses, layout):
    num_samples = min(10, epoch_stats_dict['generated supercell examples'].num_graphs)
    supercell_identifiers = [epoch_stats_dict['generated supercell examples'].csd_identifier[i] for i in
                             range(num_samples)]
    supercell_inds = [np.argwhere(epoch_stats_dict['identifiers'] == ident)[0, 0] for ident in
                      supercell_identifiers]

    generator_losses_i = {key: value[supercell_inds] for i, (key, value) in
                          enumerate(generator_losses.items())}  # limit to 10 samples
    generator_losses_i['identifier'] = supercell_identifiers
    losses = list(generator_losses_i.keys())
    fig = px.bar(generator_losses_i, x="identifier", y=losses)

    fig.layout.margin = layout.margin
    fig.update_layout(xaxis_title='Sample', yaxis_title='Per-Sample Losses')

    # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
    if config.wandb.log_figures:
        wandb.log({'Cell Generation Losses': fig})
    if (config.machine == 'local') and False:
        fig.show()

    x = generator_losses['packing normed mae']
    y = generator_losses['per mol vdw loss']
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones_like(x)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                               mode='markers', marker=dict(color=z), opacity=1))
    fig.layout.margin = layout.margin
    fig.update_layout(xaxis_title='Packing Loss', yaxis_title='vdW Loss')
    fig.update_layout(yaxis_range=[0, 10], xaxis_range=[0, 2])

    # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
    if config.wandb.log_figures:
        wandb.log({'Loss Balance': fig})
    if (config.machine == 'local') and False:
        fig.show()


def report_conditioner_training(config, wandb, epoch_stats_dict):
    reconstruction_losses = np.concatenate(epoch_stats_dict['reconstruction loss']).flatten()
    pack_true = np.concatenate(epoch_stats_dict['conditioner packing target']).flatten() * config.dataDims[
        'target std'] + config.dataDims['target mean']
    pack_pred = np.concatenate(epoch_stats_dict['conditioner packing prediction']).flatten() * config.dataDims[
        'target std'] + config.dataDims['target mean']
    packing_mae = np.abs(pack_true - pack_pred) / pack_true

    wandb.log({
        'reconstruction loss': reconstruction_losses.mean(),
        'packing MAE': packing_mae.mean(),
    })
    layout = plotly_setup(config)
    x = packing_mae
    y = reconstruction_losses
    xy = np.vstack([x, y])
    #
    # try:
    #     z = gaussian_kde(xy)(xy)
    # except:
    #     z = np.ones_like(x)
    #
    # fig = go.Figure()
    # fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
    #                            mode='markers', marker=dict(color=z), opacity=0.5))
    # fig.layout.margin = layout.margin
    # fig.update_layout(xaxis_title='Packing Loss', yaxis_title='Reconstruction Loss')
    #
    # # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
    # if self.config.wandb.log_figures:
    #     wandb.log({'Conditioner Loss Balance': fig})
    # if (self.config.machine == 'local') and False:
    #     fig.show()

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    for img_i in range(2):
        sample_guess = epoch_stats_dict['prediction_sample'][0, img_i].argmax(0)
        sample_density = np_softmax(epoch_stats_dict['prediction_sample'][0, img_i][None, ...])[0, 1:].sum(0)
        sample_true = epoch_stats_dict['target_sample'][0, img_i]

        X, Y, Z = (sample_guess + 1).nonzero()
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=sample_density.flatten(),
            isomin=0.00001,
            isomax=1,
            opacity=0.05,  # needs to be small to see through all surfaces
            surface_count=50,  # needs to be a large number for good volume rendering
            colorscale='Jet',
            cmin=0,
            showlegend=True,
            # caps=dict(x_show=False, y_show=False, z_show=False),
        ), row=1, col=img_i + 1)

        x, y, z = sample_true.nonzero()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            showlegend=True,
            marker=dict(
                size=10,
                color=sample_true[x, y, z],
                colorscale='Jet',
                cmin=0, cmax=6,
                opacity=0.5
            )), row=1, col=img_i + 1)
        fig.update_layout(showlegend=True)

    if config.wandb.log_figures:
        wandb.log({'Conditioner Reconstruction Samples': fig})
    if (config.machine == 'local') and False:
        fig.show()

    class_names = ['empty'] + [str(key) for key in config.conditioner_classes.keys()]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='True', x=class_names, y=epoch_stats_dict['conditioner particle true'].mean(0)))
    fig.add_trace(
        go.Bar(name='Predictions', x=class_names, y=epoch_stats_dict['conditioner particle prediction'].mean(0)))
    fig.update_layout(barmode='group')
    fig.update_yaxes(type='log')
    fig.layout.margin = layout.margin

    if config.wandb.log_figures:
        wandb.log({'Conditioner Classwise Density': fig})
    if (config.machine == 'local') and False:
        fig.show()

    x = pack_true  # generator_losses['generator per mol vdw loss']
    y = pack_pred  # generator_losses['generator packing loss']

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
    fig.update_layout(xaxis_title='packing target', yaxis_title='packing prediction')

    # fig.write_image('../paper1_figs/scores_vs_emd.png', scale=4)
    if config.wandb.log_figures:
        wandb.log({'Cell Packing': fig})
    if (config.machine == 'local') and False:
        fig.show()

    '''
    classwise bce loss
    '''
    classwise_losses = np.mean(epoch_stats_dict['conditioner classwise bce'], axis=0)
    class_names = ['empty'] + [str(key) for key in config.conditioner_classes.keys()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=class_names, y=classwise_losses))
    fig.layout.margin = layout.margin

    if config.wandb.log_figures:
        wandb.log({'Conditioner Classwise Losses': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None


def process_discriminator_outputs(config, epoch_stats_dict):
    scores_dict = {}
    vdw_penalty_dict = {}
    tracking_features_dict = {}
    packing_coeff_dict = {}

    generator_inds = np.where(epoch_stats_dict['generator sample source'] == 0)
    randn_inds = np.where(epoch_stats_dict['generator sample source'] == 1)[0]
    distorted_inds = np.where(epoch_stats_dict['generator sample source'] == 2)[0]

    scores_dict['CSD'] = softmax_and_score(epoch_stats_dict['discriminator real score'])
    scores_dict['Gaussian'] = softmax_and_score(epoch_stats_dict['discriminator fake score'][randn_inds])
    scores_dict['Generator'] = softmax_and_score(epoch_stats_dict['discriminator fake score'][generator_inds])
    scores_dict['Distorted'] = softmax_and_score(epoch_stats_dict['discriminator fake score'][distorted_inds])

    tracking_features_dict['CSD'] = {feat: vec for feat, vec in zip(config.dataDims['tracking features dict'],
                                                                    epoch_stats_dict['tracking features'].T)}
    tracking_features_dict['Distorted'] = {feat: vec for feat, vec in
                                           zip(config.dataDims['tracking features dict'],
                                               epoch_stats_dict['tracking features'][distorted_inds].T)}
    tracking_features_dict['Gaussian'] = {feat: vec for feat, vec in
                                          zip(config.dataDims['tracking features dict'],
                                              epoch_stats_dict['tracking features'][randn_inds].T)}
    tracking_features_dict['Generator'] = {feat: vec for feat, vec in
                                           zip(config.dataDims['tracking features dict'],
                                               epoch_stats_dict['tracking features'][generator_inds].T)}

    vdw_penalty_dict['CSD'] = epoch_stats_dict['real vdw penalty']
    vdw_penalty_dict['Gaussian'] = epoch_stats_dict['fake vdw penalty'][randn_inds]
    vdw_penalty_dict['Generator'] = epoch_stats_dict['fake vdw penalty'][generator_inds]
    vdw_penalty_dict['Distorted'] = epoch_stats_dict['fake vdw penalty'][distorted_inds]

    packing_coeff_dict['CSD'] = epoch_stats_dict['real packing coefficients']
    packing_coeff_dict['Gaussian'] = epoch_stats_dict['generated packing coefficients'][randn_inds]
    packing_coeff_dict['Generator'] = epoch_stats_dict['generated packing coefficients'][generator_inds]
    packing_coeff_dict['Distorted'] = epoch_stats_dict['generated packing coefficients'][distorted_inds]

    return scores_dict, vdw_penalty_dict, tracking_features_dict, packing_coeff_dict


def discriminator_scores_plot(wandb, scores_dict, vdw_penalty_dict, packing_coeff_dict, layout):
    plot_color_dict = {}
    plot_color_dict['CSD'] = ('rgb(250,150,50)')  # test
    plot_color_dict['Generator'] = ('rgb(100,50,0)')  # test
    plot_color_dict['Gaussian'] = ('rgb(0,50,0)')  # fake csd
    plot_color_dict['Distorted'] = ('rgb(0,100,100)')  # fake distortion

    scores_range = np.ptp(np.concatenate(list(scores_dict.values())))
    bandwidth1 = scores_range / 200

    bandwidth2 = 15 / 200
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
        fig.add_trace(go.Violin(x=-np.log10(vdw_penalty_dict[label] + 1e-3), name=legend_label,
                                line_color=plot_color_dict[label],
                                side='positive', orientation='h', width=4, meanline_visible=True,
                                bandwidth=bandwidth2, points=False),
                      row=1, col=2)

    all_vdws = np.concatenate((vdw_penalty_dict['CSD'], vdw_penalty_dict['Gaussian'], vdw_penalty_dict['Distorted'],
                               vdw_penalty_dict['Generator']))
    all_scores_i = np.concatenate(
        (scores_dict['CSD'], scores_dict['Gaussian'], scores_dict['Distorted'], scores_dict['Generator']))

    rrange = np.logspace(3, 0, len(viridis))
    cscale = [[1 / rrange[i], viridis[i]] for i in range(len(rrange))]
    cscale[0][0] = 0

    fig.add_trace(go.Histogram2d(x=all_scores_i,
                                 y=-np.log10(all_vdws + 1e-3),
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
    fig.update_xaxes(title_text='vdw Score', row=1, col=2)
    fig.update_xaxes(title_text='Model Score', row=2, col=1)
    fig.update_yaxes(title_text='vdw Score', row=2, col=1)

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

    fig.add_trace(go.Histogram2d(x=all_scores_i,
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
    All in one
    '''
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=-np.log10(all_vdws + 1e-3),
        y=np.clip(all_coeffs, a_min=0, a_max=1),
        mode='markers',
        marker=dict(color=all_scores_i, opacity=.75,
                    colorbar=dict(title="Score"),
                    colorscale="inferno",
                    )
    ))
    fig.layout.margin = layout.margin

    fig.update_layout(xaxis_title='vdw score', yaxis_title='packing coefficient')
    wandb.log({'Discriminator Scores Analysis': fig})

    return None


def plot_generator_loss_correlates(config, wandb, epoch_stats_dict, generator_losses, layout):
    correlates_dict = {}
    generator_losses['all'] = np.vstack([generator_losses[key] for key in generator_losses.keys()]).T.sum(1)
    loss_labels = list(generator_losses.keys())

    tracking_features = np.asarray(epoch_stats_dict['tracking features'])

    for i in range(config.dataDims['num tracking features']):  # not that interesting
        if (np.average(tracking_features[:, i] != 0) > 0.05):
            corr_dict = {
                loss_label: np.corrcoef(generator_losses[loss_label], tracking_features[:, i], rowvar=False)[0, 1]
                for loss_label in loss_labels}
            correlates_dict[config.dataDims['tracking features dict'][i]] = corr_dict

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


def plot_discriminator_score_correlates(config, wandb, epoch_stats_dict, layout):
    correlates_dict = {}
    real_scores = softmax_and_score(epoch_stats_dict['discriminator real score'])
    tracking_features = np.asarray(epoch_stats_dict['tracking features'])

    for i in range(config.dataDims['num tracking features']):  # not that interesting
        if (np.average(tracking_features[:, i] != 0) > 0.05):
            corr = np.corrcoef(real_scores, tracking_features[:, i], rowvar=False)[0, 1]
            if np.abs(corr) > 0.05:
                correlates_dict[config.dataDims['tracking features dict'][i]] = corr

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


def mini_csp_reporting(config, wandb, sampling_dict, real_samples_dict, real_data):
    '''
    report on key metrics
    '''
    scores_labels = ['score', 'vdw overlap', 'density', 'h bond score']
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.075,
                        horizontal_spacing=0.075)

    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', min(15, real_data.num_graphs), colortype='rgb')

    for i, label in enumerate(scores_labels):  # todo record sample stats in canonical frame for rebuilding
        legend_label = label
        for j in range(min(15, real_data.num_graphs)):
            bandwidth1 = np.ptp(sampling_dict[label][j]) / 50
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(go.Violin(x=[real_samples_dict[label][j]], name=str(j) + '_', line_color=colors[j],
                                    side='positive', orientation='h', width=6, meanline_visible=True),
                          row=row, col=col)

            fig.add_trace(go.Violin(x=sampling_dict[label][j], name=j,
                                    side='positive', orientation='h', width=4, line_color=colors[j],
                                    meanline_visible=True, bandwidth=bandwidth1),
                          row=row, col=col)

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=100,  # top margin
        )
    )
    fig.update_xaxes(title_text='Score', row=1, col=1)
    fig.update_xaxes(title_text='vdW overlap', row=1, col=2)
    fig.update_xaxes(title_text='Packing Coeff.', row=2, col=1)
    fig.update_xaxes(title_text='H-bond score', row=2, col=2)

    fig.update_layout(showlegend=False)
    fig.layout.margin = layout.margin
    if config.wandb.log_figures:
        wandb.log({'Mini-CSP Scores': fig})
    if (config.machine == 'local') and False:
        fig.show()
    '''
    cluster via RDF
    '''
    #
    # real_fake_dists = np.zeros((num_molecules,n_sampling_iters))
    # for i in range(num_molecules):
    #     for j in range(n_sampling_iters):
    #         real_fake_dists[i,j] = earth_movers_distance_np(real_samples_dict['RDF'][i], sampling_dict['RDF'][i,j]) / RDF_shape[1]
    #
    # fake_fake_dists = np.zeros((num_molecules, n_sampling_iters, n_sampling_iters))
    #
    # for i in range(num_molecules):
    #     for j in range(n_sampling_iters):
    #         for k in range(n_sampling_iters):
    #             fake_fake_dists[i, j, k] = earth_movers_distance_np(sampling_dict['RDF'][i,j], sampling_dict['RDF'][i, k]) / RDF_shape[1]
    #
    # fake_fake_dists /= fake_fake_dists.mean((1,2))[:,None,None]
    #
    # clusters = []
    # classes = []
    # for i in range(num_molecules):
    #     model = AgglomerativeClustering(distance_threshold=1, linkage="average", affinity='precomputed', n_clusters=None)
    #     model = model.fit(fake_fake_dists[i])
    #     clusters.append(model.n_clusters_)
    #     classes.append(model.labels_)

    '''
    did we find the real crystal?
    '''

    return None


def sampling_telemetry_plot(config, wandb, sampling_dict):
    n_runs = len(sampling_dict['scores'])
    num_iters = sampling_dict['scores'].shape[1]

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=40,  # top margin
        )
    )

    '''
    full telemetry
    '''
    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', n_runs, colortype='rgb')
    fig = make_subplots(cols=2, rows=1, subplot_titles=['Model Score', 'vdw Score'])
    for i in range(n_runs):
        x = np.arange(num_iters)
        y = sampling_dict['scores'][i]
        opacity = np.clip(
            1 - np.abs(np.amax(y) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']), a_min=0.1,
            a_max=1)
        fig.add_trace(go.Scattergl(x=x, y=y, opacity=opacity, line_color=colors[i], name=f'score_{i}'),
                      col=1, row=1)
        fig.add_trace(go.Scattergl(x=sampling_dict['resampled state record'][i],
                                   y=y[sampling_dict['resampled state record'][i]],
                                   mode='markers', line_color=colors[i], marker=dict(size=10), opacity=1,
                                   showlegend=False),
                      col=1, row=1)
    for i in range(n_runs):
        y = -sampling_dict['vdw penalties'][i]
        opacity = np.clip(
            1 - np.abs(np.amax(y) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']), a_min=0.1,
            a_max=1)
        fig.add_trace(go.Scattergl(x=x, y=y, opacity=opacity, line_color=colors[i], name=f'vdw_{i}'),
                      col=2, row=1)
        fig.add_trace(go.Scattergl(x=sampling_dict['resampled state record'][i],
                                   y=y[sampling_dict['resampled state record'][i]],
                                   mode='markers', line_color=colors[i], marker=dict(size=10), opacity=1,
                                   showlegend=False),
                      col=2, row=1)
    # for i in range(n_samples):
    #     opacity = np.clip(1 - np.abs(np.amax(sampling_dict['scores'][i]) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']),
    #                       a_min=0.1, a_max=1)
    #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=sampling_dict['acceptance ratio'][i], opacity=opacity, line_color=colors[i], name=f'run_{i}'),
    #                   col=1, row=2)
    # for i in range(n_samples):
    #     opacity = np.clip(1 - np.abs(np.amax(sampling_dict['scores'][i]) - np.amax(sampling_dict['scores'])) / np.amax(sampling_dict['scores']),
    #                       a_min=0.1, a_max=1)
    #     fig.add_trace(go.Scattergl(x=np.arange(num_iters), y=np.log10(sampling_dict['step size'][i]), opacity=opacity, line_color=colors[i], name=f'run_{i}'),
    #                   col=2, row=2)
    fig.update_layout(showlegend=True)
    # fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.layout.margin = layout.margin
    # fig.write_image('../paper1_figs/sampling_telemetry.png')
    # wandb.log({'Sampling Telemetry': fig})
    if config.machine == 'local':
        import plotly.io as pio
        pio.renderers.default = 'browser'
        fig.show()


def cell_params_tracking_plot(wandb, supercell_builder, layout, config, sampling_dict, collater, extra_test_loader):
    # DEPRECATED
    num_iters = sampling_dict['scores'].shape[1]
    n_runs = sampling_dict['canonical samples'].shape[1]

    all_samples = torch.tensor(sampling_dict['canonical samples'].reshape(12, n_runs * num_iters).T)
    single_mol_data_0 = extra_test_loader.dataset[0]
    big_single_mol_data = collater([single_mol_data_0 for n in range(len(all_samples))]).cuda()
    override_sg_ind = list(supercell_builder.symmetries_dict['space_groups'].values()).index('P-1') + 1
    sym_ops_list = [torch.Tensor(supercell_builder.symmetries_dict['sym_ops'][override_sg_ind]).to(
        big_single_mol_data.x.device) for i in range(big_single_mol_data.num_graphs)]
    big_single_mol_data = update_sg_to_all_crystals('P-1', supercell_builder.dataDims, big_single_mol_data,
                                                    supercell_builder.symmetries_dict, sym_ops_list)
    processed_cell_params = torch.cat(supercell_builder.process_cell_params(big_single_mol_data, all_samples.cuda(),
                                              rescale_asymmetric_unit=False, skip_cell_cleaning=True),dim=-1).T
    del big_single_mol_data

    processed_cell_params = processed_cell_params.reshape(12, n_runs, num_iters).cpu().detach().numpy()

    fig = make_subplots(rows=4, cols=3, subplot_titles=[
        'a', 'b', 'c', 'alpha', 'beta', 'gamma',
        'x', 'y', 'z', 'phi', 'psi', 'theta'
    ])
    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', min(10, n_runs), colortype='rgb')
    for i in range(12):
        row = i // 3 + 1
        col = i % 3 + 1
        x = np.arange(num_iters * n_runs)
        for j in range(min(10, n_runs)):
            y = processed_cell_params[i, j]
            opacity = 0.75  # np.clip(1 - np.abs(np.ptp(y) - np.ptp(processed_cell_params[i])) / np.ptp(processed_cell_params[i]), a_min=0.1, a_max=1)
            fig.add_trace(go.Scattergl(x=x, y=y, line_color=colors[j], opacity=opacity),
                          row=row, col=col)
            fig.add_trace(go.Scattergl(x=sampling_dict['resampled state record'][j],
                                       y=y[sampling_dict['resampled state record'][j]],
                                       mode='markers', line_color=colors[j], marker=dict(size=7), opacity=opacity,
                                       showlegend=False),
                          row=row, col=col)

    fig.update_layout(showlegend=False)
    fig.layout.margin = layout.margin
    # fig.write_image('../paper1_figs/sampling_telemetry.png')
    # wandb.log({'Sampling Telemetry': fig})
    if config.machine == 'local':
        import plotly.io as pio
        pio.renderers.default = 'browser'
        fig.show()
