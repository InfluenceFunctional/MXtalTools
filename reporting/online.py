import ase.io
import numpy as np
import torch
from _plotly_utils.colors import n_colors
from plotly import graph_objects
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, linregress
import plotly.graph_objects as go
import plotly.express as px
from torch_geometric.loader.dataloader import Collater

from common.geometry_calculations import cell_vol_torch
from common.utils import update_stats_dict, np_softmax, earth_movers_distance_torch, earth_movers_distance_np, compute_rdf_distance
from crystal_building.builder import update_sg_to_all_crystals, update_crystal_symmetry_elements
from models.crystal_rdf import crystal_rdf
from models.utils import softmax_and_score, norm_scores, ase_mol_from_crystaldata
from models.vdw_overlap import vdw_overlap
from reporting.nov_22_discriminator import process_discriminator_evaluation_data


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

        # fig.write_image('../paper1_figs_new_architecture/scores_vs_emd.png', scale=4)
        if config.wandb.log_figures:
            wandb.log({'Cell Packing': fig})
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
    # # fig.write_image('../paper1_figs_new_architecture/scores_vs_emd.png', scale=4)
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

    # fig.write_image('../paper1_figs_new_architecture/scores_vs_emd.png', scale=4)
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


def log_mini_csp_scores_distributions(config, wandb, sampling_dict, real_samples_dict, real_data):
    """
    report on key metrics from mini-csp
    """
    scores_labels = ['score', 'vdw overlap', 'density', 'h bond score']
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.075,
                        horizontal_spacing=0.075)

    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', min(15, real_data.num_graphs) + 1, colortype='rgb')

    for i, label in enumerate(scores_labels):
        row = i // 2 + 1
        col = i % 2 + 1
        for j in range(min(15, real_data.num_graphs)):
            bandwidth1 = np.ptp(sampling_dict[label][j]) / 50

            sample_score = sampling_dict[label][j]
            real_score = real_samples_dict[label][j]
            if label == 'score':
                sample_score = sample_score - real_score
                real_score = real_score - real_score

            fig.add_trace(go.Violin(x=[real_score], name=str(real_data.csd_identifier[j]), line_color=colors[j],
                                    side='positive', orientation='h', width=2, meanline_visible=True),
                          row=row, col=col)

            fig.add_trace(go.Violin(x=sample_score, name=str(real_data.csd_identifier[j]),
                                    side='positive', orientation='h', width=2, line_color=colors[j],
                                    meanline_visible=True, bandwidth=bandwidth1),
                          row=row, col=col)

        all_sample_score = sampling_dict[label].flatten()
        if label == 'score':
            all_sample_score = sampling_dict[label].flatten() - real_samples_dict[label].mean()

        fig.add_trace(go.Violin(x=all_sample_score, name='all samples',
                                side='positive', orientation='h', width=2, line_color=colors[-1],
                                meanline_visible=True, bandwidth=np.ptp(sampling_dict[label].flatten()) / 100),
                      row=row, col=col)

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=100,  # top margin
        )
    )
    fig.update_xaxes(title_text='Score - CSD Score', row=1, col=1)
    fig.update_xaxes(title_text='vdW overlap', row=1, col=2)
    fig.update_xaxes(title_text='Packing Coeff.', row=2, col=1)
    fig.update_xaxes(title_text='H-bond score', row=2, col=2)
    fig.update_layout(showlegend=False, yaxis_showgrid=True)  # legend_traceorder='reversed',

    fig.layout.margin = layout.margin

    if config.wandb.log_figures:
        wandb.log({'Mini-CSP Scores': fig})
    if (config.machine == 'local') and False:
        fig.show()

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
    # fig.write_image('../paper1_figs_new_architecture/sampling_telemetry.png')
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
                                                                            rescale_asymmetric_unit=False, skip_cell_cleaning=True), dim=-1).T
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
    # fig.write_image('../paper1_figs_new_architecture/sampling_telemetry.png')
    # wandb.log({'Sampling Telemetry': fig})
    if config.machine == 'local':
        import plotly.io as pio
        pio.renderers.default = 'browser'
        fig.show()


def log_best_mini_csp_samples(config, wandb, discriminator, sampling_dict, real_samples_dict, real_data, supercell_builder, mol_volume_ind, sym_info, vdw_radii):
    """
    extract the best guesses for each crystal and reconstruct and analyze them
    compare best samples to the experimental crystal
    """

    # identify the best samples (later, use clustering to filter down to a diverse set)
    scores_list = ['score', 'vdw overlap', 'h bond score', 'density']
    scores_dict = {key: sampling_dict[key] for key in scores_list}

    num_crystals, num_samples = scores_dict['score'].shape

    topk_size = min(10,sampling_dict['score'].shape[1])
    sort_inds = sampling_dict['score'].argsort(axis=-1)[:, -topk_size:]  #
    best_scores_dict = {key: np.asarray([sampling_dict[key][ii, sort_inds[ii]] for ii in range(num_crystals)]) for key in scores_list}
    best_samples = np.asarray([sampling_dict['cell params'][ii, sort_inds[ii], :] for ii in range(num_crystals)])
    best_samples_space_groups = np.asarray([sampling_dict['space group'][ii, sort_inds[ii]] for ii in range(num_crystals)])
    best_samples_handedness = np.asarray([sampling_dict['handedness'][ii, sort_inds[ii]] for ii in range(num_crystals)])

    # reconstruct the best samples from the cell params
    best_supercells_list = []
    best_supercell_scores = []
    best_supercell_rdfs = []

    rdf_bins = 100
    rdf_range = [0, 10]

    discriminator.eval()
    with torch.no_grad():
        for n in range(topk_size):
            real_data_i = real_data.clone()

            real_data_i = update_crystal_symmetry_elements(
                real_data_i, best_samples_space_groups[:, n],
                config.dataDims, supercell_builder.symmetries_dict, randomize_sgs=False)

            fake_supercell_data, _, _ = supercell_builder.build_supercells(
                real_data_i,
                torch.tensor(best_samples[:, n, :], device=real_data_i.x.device, dtype=torch.float32),
                config.supercell_size,
                config.discriminator.graph_convolution_cutoff,
                align_molecules=True, skip_cell_cleaning=True, standardized_sample=False,
                target_handedness=best_samples_handedness[:, n],
                rescale_asymmetric_unit=False)

            output, extra_outputs = discriminator(fake_supercell_data.clone(), return_dists=True)  # reshape output from flat filters to channels * filters per channel
            best_supercell_scores.append(softmax_and_score(output).cpu().detach().numpy())

            rdf, rr, dist_dict = crystal_rdf(fake_supercell_data, rrange=rdf_range, bins=rdf_bins, raw_density=True, atomwise=True, mode='intermolecular', cpu_detach=True)
            best_supercell_rdfs.append(rdf)

            best_supercells_list.append(fake_supercell_data.cpu().detach())

    reconstructed_best_scores = np.asarray(best_supercell_scores).T
    print(f'cell reconstruction mean score difference = {np.mean(np.abs(best_scores_dict["score"] - reconstructed_best_scores)):.4f}')  # should be ~0
    print(f'cell reconstruction median score difference = {np.median(np.abs(best_scores_dict["score"] - reconstructed_best_scores)):.4f}')  # should be ~0
    print(f'cell reconstruction 95% quantile score difference = {np.quantile(np.abs(best_scores_dict["score"] - reconstructed_best_scores), .95):.4f}')  # should be ~0

    best_rdfs = [np.stack([best_supercell_rdfs[ii][jj] for ii in range(topk_size)]) for jj in range(real_data.num_graphs)]

    rdf_dists = np.zeros((real_data.num_graphs, topk_size, topk_size))
    for i in range(real_data.num_graphs):
        for j in range(topk_size):
            for k in range(j, topk_size):
                rdf_dists[i, j, k] = compute_rdf_distance(best_rdfs[i][j], best_rdfs[i][k], rr)

    rdf_dists = rdf_dists + np.moveaxis(rdf_dists, (1, 2), (2, 1))  # add lower diagonal (distance matrix is symmetric)

    rdf_real_dists = np.zeros((real_data.num_graphs, topk_size))
    for i in range(real_data.num_graphs):
        for j in range(topk_size):
            rdf_real_dists[i, j] = compute_rdf_distance(real_samples_dict['RDF'][i], best_rdfs[i][j], rr)

    x = rdf_real_dists.flatten()
    y = (reconstructed_best_scores - real_samples_dict['score'][:, None]).flatten()
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)
    except:
        z = np.ones_like(x)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=dict(color=z)))

    fig.update_layout(xaxis_title='RDF Distance From Target', yaxis_title='Sample vs. experimental score difference', showlegend=False)
    wandb.log({"Sample RDF vs. Score": fig})

    best_supercells = best_supercells_list[-1]  # last sample was the best
    save_3d_structure_examples(wandb, best_supercells)

    """
    Overlaps & Crystal-wise summary plot
    """
    sample_wise_overlaps_and_summary_plot(config, wandb, num_crystals, best_supercells, sym_info, best_scores_dict, vdw_radii, mol_volume_ind)

    """
    CSP Funnels
    """
    sample_csp_funnel_plot(config, wandb, best_supercells, sampling_dict, real_samples_dict)

    sample_wise_rdf_funnel_plot(config,wandb,best_supercells, reconstructed_best_scores, real_samples_dict, rdf_real_dists)

    ## debug check to make sure we are recreating the same cells
    # fig = go.Figure()
    # colors = n_colors('rgb(250,40,100)', 'rgb(5,250,200)', real_data.num_graphs, colortype='rgb')
    # for ii in range(num_crystals):
    #     # fig.add_trace(go.Scatter(x=best_scores_dict['score'][ii], y=reconstructed_best_scores[ii], mode='markers', line_color=colors[ii]))
    #     fig.add_trace(go.Histogram(x=np.abs(best_scores_dict['score'][ii] - reconstructed_best_scores[ii]),
    #                                nbinsx=10,
    #                                marker_color=colors[ii]))
    # fig.update_layout(title=str(config.discriminator.graph_norm) + ' ' + str(config.discriminator.fc_norm_mode) + ' ' + str(config.discriminator.fc_dropout_probability))
    # fig.show()
    # print(np.mean(np.abs(best_scores_dict['score'] - reconstructed_best_scores))) # should be ~0

    return None


def sample_wise_overlaps_and_summary_plot(config, wandb, num_crystals, best_supercells, sym_info, best_scores_dict, vdw_radii, mol_volume_ind):
    num_samples = min(num_crystals, 25)
    vdw_loss, normed_vdw_loss, vdw_penalties = \
        vdw_overlap(vdw_radii, crystaldata=best_supercells, return_atomwise=True, return_normed=True,
                    graph_sizes=best_supercells.tracking[:,
                                config.dataDims['tracking features dict'].index('molecule num atoms')])
    vdw_loss /= best_supercells.tracking[:,
                config.dataDims['tracking features dict'].index('molecule num atoms')]

    volumes_list = []
    for i in range(best_supercells.num_graphs):
        volumes_list.append(
            cell_vol_torch(best_supercells.cell_params[i, 0:3], best_supercells.cell_params[i, 3:6]))
    volumes = torch.stack(volumes_list)
    generated_packing_coeffs = (best_supercells.Z * best_supercells.tracking[:,
                                                    mol_volume_ind] / volumes).cpu().detach().numpy()
    target_packing = (best_supercells.y * config.dataDims['target std'] + config.dataDims[
        'target mean']).cpu().detach().numpy()

    fig = go.Figure()
    for i in range(min(best_supercells.num_graphs, num_samples)):
        pens = vdw_penalties[i].cpu().detach()
        fig.add_trace(go.Violin(x=pens[pens != 0], side='positive', orientation='h',
                                bandwidth=0.01, width=1, showlegend=False, opacity=1,
                                name=f'{best_supercells.csd_identifier[i]} : ' + f'SG={sym_info["space_groups"][int(best_supercells.sg_ind[i])]} <br /> ' +
                                     f'c_t={target_packing[i]:.2f} c_p={generated_packing_coeffs[i]:.2f} <br /> ' +
                                     f'tot_norm_ov={normed_vdw_loss[i]:.2f} <br />' +
                                     f'Score={best_scores_dict["score"][i, -1]:.2f}'
                                ),
                      )

    # Can only run this section with RDKit installed, which doesn't always work
    # Commented out - not that important anyway.
    # molecule = rdkit.Chem.MolFromSmiles(supercell_examples[i].smiles)
    # try:
    #     rdkit.Chem.AllChem.Compute2DCoords(molecule)
    #     rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(molecule, molecule)
    #     pil_image = rdkit.Chem.Draw.MolToImage(molecule, size=(500, 500))
    #     pil_image.save('mol_img.png', 'png')
    #     # Add trace
    #     img = Image.open("mol_img.png")
    #     # Add images
    #     fig.add_layout_image(
    #         dict(
    #             source=img,
    #             xref="x domain", yref="y domain",
    #             x=.1 + (.15 * (i % 2)), y=i / 10.5 + 0.05,
    #             sizex=.15, sizey=.15,
    #             xanchor="center",
    #             yanchor="middle",
    #             opacity=0.75
    #         )
    #     )
    # except:  # ValueError("molecule was rejected by rdkit"):
    #     pass

    fig.update_layout(width=800, height=800, font=dict(size=12), xaxis_range=[0, 4])
    fig.update_layout(showlegend=False, legend_traceorder='reversed', yaxis_showgrid=True)
    fig.update_layout(xaxis_title='Nonzero vdW overlaps', yaxis_title='packing prediction')

    wandb.log({'Generated Sample Analysis': fig})

    return None


def sample_csp_funnel_plot(config, wandb, best_supercells, sampling_dict, real_samples_dict):
    num_crystals = best_supercells.num_graphs
    num_reporting_samples = min(25, num_crystals)
    n_rows = int(np.ceil(np.sqrt(num_reporting_samples)))
    n_cols = int(n_rows)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(best_supercells.csd_identifier)[:num_reporting_samples],
                        x_title='Packing Coefficient', y_title='Model Score')
    for ii in range(num_reporting_samples):
        row = ii // n_cols + 1
        col = ii % n_cols + 1
        x = sampling_dict['density'][ii]
        y = sampling_dict['score'][ii]
        z = sampling_dict['vdw overlap'][ii]
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', marker=dict(color=z, colorbar=dict(title="vdW Overlap"), cmin=0, cmax=4, opacity=0.5, colorscale="rainbow"), opacity=1),
                      row=row, col=col)

        fig.add_trace(go.Scattergl(x=[real_samples_dict['density'][ii]], y=[real_samples_dict['score'][ii]],
                                   mode='markers', marker=dict(color=[real_samples_dict['vdw overlap'][ii]], colorscale='rainbow', size=10,
                                                               colorbar=dict(title="vdW Overlap"), cmin=0, cmax=4),
                                   showlegend=False),
                      row=row, col=col)

    # fig.update_layout(xaxis_range=[0, 1])
    fig.update_yaxes(autorange="reversed")

    if config.wandb.log_figures:
        wandb.log({'Density Funnel': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None



def sample_wise_rdf_funnel_plot(config,wandb,best_supercells, reconstructed_best_scores, real_samples_dict, rdf_real_dists):

    num_crystals = best_supercells.num_graphs
    num_reporting_samples = min(25, num_crystals)
    n_rows = int(np.ceil(np.sqrt(num_reporting_samples)))
    n_cols = int(n_rows)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(best_supercells.csd_identifier)[:num_reporting_samples],
                        x_title='RDF Distance', y_title='Model Score')
    for ii in range(num_reporting_samples):
        row = ii // n_cols + 1
        col = ii % n_cols + 1
        x = rdf_real_dists[ii]
        y = reconstructed_best_scores[ii]
        fig.add_trace(go.Scattergl(x=x, y=y, showlegend=False,
                                   mode='markers', opacity=1),
                      row=row, col=col)

        fig.add_trace(go.Scattergl(x=[0], y=[real_samples_dict['score'][ii]],
                                   mode='markers', marker=dict(size=10),
                                   showlegend=False),
                      row=row, col=col)

    #fig.update_layout(xaxis_title='RDF Distance', yaxis_title='Model Score')
    # fig.update_layout(xaxis_range=[0, 1])
    fig.update_yaxes(autorange="reversed")

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
        )
    )
    fig.layout.margin = layout.margin

    if config.wandb.log_figures:
        wandb.log({'RDF Funnel': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None


def save_3d_structure_examples(wandb, generated_supercell_examples):
    num_samples = min(25, generated_supercell_examples.num_graphs)
    identifiers = [generated_supercell_examples.csd_identifier[i] for i in range(num_samples)]
    sgs = [str(int(generated_supercell_examples.sg_ind[i])) for i in range(num_samples)]

    crystals = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                         index=i, exclusion_level='distance', inclusion_distance=4)
                for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(crystals)):
        ase.io.write(f'supercell_{identifiers[i]}.cif', crystals[i])
        wandb.log({'Generated Clusters': wandb.Molecule(open(f"supercell_{identifiers[i]}.cif"),
                                                          caption=identifiers[i] + ' ' + sgs[i])})

    unit_cells = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                           index=i, exclusion_level='unit cell')
                  for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(crystals)):
        ase.io.write(f'unit_cell_{identifiers[i]}.cif', unit_cells[i])
        wandb.log({'Generated Unit Cells': wandb.Molecule(open(f"unit_cell_{identifiers[i]}.cif"),
                                                          caption=identifiers[i] + ' ' + sgs[i])})

    unit_cells_inside = [ase_mol_from_crystaldata(generated_supercell_examples, highlight_canonical_conformer=False,
                                                  index=i, exclusion_level='inside cell')
                         for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(crystals)):
        ase.io.write(f'unit_cell_inside_{identifiers[i]}.cif', unit_cells_inside[i])
        wandb.log({'Generated Unit Cells 2': wandb.Molecule(open(f"unit_cell_inside_{identifiers[i]}.cif"),
                                                          caption=identifiers[i] + ' ' + sgs[i])})

    mols = [ase_mol_from_crystaldata(generated_supercell_examples,
                                     index=i, exclusion_level='conformer')
            for i in range(min(num_samples, generated_supercell_examples.num_graphs))]

    for i in range(len(mols)):
        ase.io.write(f'conformer_{identifiers[i]}.cif', mols[i])
        wandb.log({'Single Conformers': wandb.Molecule(open(f"conformer_{identifiers[i]}.cif"), caption=identifiers[i])})

    return None


def new_process_discriminator_evaluation_data(dataDims, wandb, extra_test_dict, test_epoch_stats_dict, train_epoch_stats_dict, size_normed_score=False):
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

    '''
    extract all the various types of scores
    '''
    scores_dict['Test Real'] = softmax_and_score(test_epoch_stats_dict['discriminator real score'], old_method=True, correct_discontinuity=True)
    scores_dict['Test Randn'] = softmax_and_score(test_epoch_stats_dict['discriminator fake score'][randn_inds], old_method=True, correct_discontinuity=True)
    # scores_dict['Test NF'] = np_softmax(test_epoch_stats_dict['discriminator fake score'][nf_inds])[:, 1]
    scores_dict['Test Distorted'] = softmax_and_score(test_epoch_stats_dict['discriminator fake score'][distorted_inds], old_method=True, correct_discontinuity=True)

    tracking_features_dict['Test Real'] = {feat: vec for feat, vec in zip(dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'].T)}
    tracking_features_dict['Test Distorted'] = {feat: vec for feat, vec in zip(dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'][distorted_inds].T)}
    tracking_features_dict['Test Randn'] = {feat: vec for feat, vec in zip(dataDims['tracking features dict'], test_epoch_stats_dict['tracking features'][randn_inds].T)}

    if size_normed_score:
        scores_dict['Test Real'] = norm_scores(scores_dict['Test Real'], test_epoch_stats_dict['tracking features'], dataDims)
        scores_dict['Test Randn'] = norm_scores(scores_dict['Test Randn'], test_epoch_stats_dict['tracking features'][randn_inds], dataDims)
        scores_dict['Test Distorted'] = norm_scores(scores_dict['Test Distorted'], test_epoch_stats_dict['tracking features'][distorted_inds], dataDims)

    if train_epoch_stats_dict is not None:
        scores_dict['Train Real'] = softmax_and_score(train_epoch_stats_dict['discriminator real score'], old_method=True, correct_discontinuity=True)
        tracking_features_dict['Train Real'] = {feat: vec for feat, vec in zip(dataDims['tracking features dict'], train_epoch_stats_dict['tracking features'].T)}

        if size_normed_score:
            scores_dict['Train Real'] = norm_scores(scores_dict['Train Real'], train_epoch_stats_dict['tracking features'], dataDims)

        vdw_penalty_dict['Train Real'] = train_epoch_stats_dict['real vdW penalty']
        wandb.log({'Average Train score': np.average(scores_dict['Train Real'])})
        wandb.log({'Train score std': np.std(scores_dict['Train Real'])})

    vdw_penalty_dict['Test Real'] = test_epoch_stats_dict['real vdw penalty']
    vdw_penalty_dict['Test Randn'] = test_epoch_stats_dict['fake vdw penalty'][randn_inds]
    vdw_penalty_dict['Test Distorted'] = test_epoch_stats_dict['fake vdw penalty'][distorted_inds]

    wandb.log({'Average Test score': np.average(scores_dict['Test Real'])})
    wandb.log({'Average Randn Fake score': np.average(scores_dict['Test Randn'])})
    # wandb.log({'Average NF Fake score': np.average(scores_dict['Test NF'])})
    wandb.log({'Average Distorted Fake score': np.average(scores_dict['Test Distorted'])})

    wandb.log({'Test Real std': np.std(scores_dict['Test Real'])})
    wandb.log({'Distorted Fake score std': np.std(scores_dict['Test Distorted'])})
    wandb.log({'Randn score std': np.std(scores_dict['Test Randn'])})

    '''
    build property dicts for the submissions and BT targets
    '''
    score_correlations_dict = {}
    rdf_full_distance_dict = {}
    rdf_inter_distance_dict = {}

    for target in all_identifiers.keys():  # run the analysis for each target
        if target_identifiers_inds[target] != []:  # record target data

            target_index = target_identifiers_inds[target]
            raw_scores = extra_test_dict['discriminator real score'][target_index]
            scores = softmax_and_score(raw_scores[None, :], old_method=True, correct_discontinuity=True)
            scores_dict[target + ' exp'] = scores

            tracking_features_dict[target + ' exp'] = {feat: vec for feat, vec in zip(dataDims['tracking features dict'], extra_test_dict['tracking features'][target_index][None, :].T)}

            if size_normed_score:
                scores_dict[target + ' exp'] = norm_scores(scores_dict[target + ' exp'], extra_test_dict['tracking features'][target_index][None, :], dataDims)

            vdw_penalty_dict[target + ' exp'] = extra_test_dict['real vdw penalty'][target_index][None]

            wandb.log({f'Average {target} exp score': np.average(scores)})

        if all_identifiers[target] != []:  # record sample data
            target_indices = all_identifiers[target]
            raw_scores = extra_test_dict['discriminator real score'][target_indices]
            scores = softmax_and_score(raw_scores, old_method=True, correct_discontinuity=True)
            scores_dict[target] = scores
            tracking_features_dict[target] = {feat: vec for feat, vec in zip(dataDims['tracking features dict'], extra_test_dict['tracking features'][target_indices].T)}

            if size_normed_score:
                scores_dict[target] = norm_scores(scores_dict[target], extra_test_dict['tracking features'][target_indices], dataDims)

            vdw_penalty_dict[target] = extra_test_dict['real vdw penalty'][target_indices]

            wandb.log({f'Average {target} score': np.average(scores)})
            wandb.log({f'Average {target} std': np.std(scores)})

            # correlate losses with molecular features
            tracking_features = np.asarray(extra_test_dict['tracking features'])
            loss_correlations = np.zeros(dataDims['num tracking features'])
            features = []
            for j in range(tracking_features.shape[-1]):  # not that interesting
                features.append(dataDims['tracking features dict'][j])
                loss_correlations[j] = np.corrcoef(scores, tracking_features[target_indices, j], rowvar=False)[0, 1]

            score_correlations_dict[target] = loss_correlations

    # compute loss correlates
    loss_correlations = np.zeros(dataDims['num tracking features'])
    features = []
    for j in range(dataDims['num tracking features']):  # not that interesting
        features.append(dataDims['tracking features dict'][j])
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


def discriminator_BT_reporting(config, wandb, test_epoch_stats_dict, extra_test_dict):
    # test_epoch_stats_dict = np.load('C:/Users\mikem\crystals\CSP_runs/275_test_epoch_stats_dict.npy', allow_pickle=True).item()
    # extra_test_dict = np.load('C:/Users\mikem\crystals\CSP_runs/275_extra_test_dict.npy', allow_pickle=True).item()

    tracking_features = test_epoch_stats_dict['tracking features']
    identifiers_list = extra_test_dict['identifiers']
    dataDims = test_epoch_stats_dict['data dims']
    score_correlations_dict, rdf_full_distance_dict, rdf_inter_distance_dict, \
        scores_dict, all_identifiers, blind_test_targets, target_identifiers, \
        target_identifiers_inds, BT_target_scores, BT_submission_scores, \
        BT_scores_dists, BT_balanced_dist, vdw_penalty_dict, tracking_features_dict = \
        new_process_discriminator_evaluation_data(dataDims, wandb, extra_test_dict,
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
    fig.write_image('../paper1_figs_new_architecture/real_vs_fake_scores.png', scale=4)
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
    fig.write_image('../paper1_figs_new_architecture/bt_submissions_distribution.png', scale=4)
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
    fig.write_image('../paper1_figs_new_architecture/scores_separation_table.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    8. Functional group analysis
    '''
    tracking_features_names = dataDims['tracking features dict']
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
    # fig.write_image('../paper1_figs_new_architecture/scores_separation_table.png')
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
    fig.write_image('../paper1_figs_new_architecture/functional_group_scores.png', scale=2)
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
    fig.write_image('../paper1_figs_new_architecture/interesting_groups.png', scale=4)
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
        fig.write_image(f'../paper1_figs_new_architecture/groupwise_analysis_{i}.png', scale=4)
        if config.machine == 'local':
            fig.show()

    '''
    S2.  score correlates
    '''

    # correlate losses with molecular features
    tracking_features = np.asarray(tracking_features)
    g_loss_correlations = np.zeros(dataDims['num tracking features'])
    features = []
    ind = 0
    for i in range(dataDims['num tracking features']):  # not that interesting
        if ('spacegroup' not in dataDims['tracking features dict'][i]) and \
                ('system' not in dataDims['tracking features dict'][i]) and \
                ('density' not in dataDims['tracking features dict'][i]):
            if (np.average(tracking_features[:, i] != 0) > 0.05) and \
                    (dataDims['tracking features dict'][i] != 'crystal z prime') and \
                    (dataDims['tracking features dict'][i] != 'molecule point group is C1'):  # if we have at least 1# relevance
                corr = np.corrcoef(scores_dict['Test Real'], tracking_features[:, i], rowvar=False)[0, 1]
                if np.abs(corr) > 0.05:
                    features.append(dataDims['tracking features dict'][i])
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
    fig.write_image('../paper1_figs_new_architecture/scores_correlates.png', scale=4)
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

    fig.write_image('../paper1_figs_new_architecture/ToC_discriminator.png', scale=4)
    if config.machine == 'local':
        fig.show()

    aa = 0
    return None
