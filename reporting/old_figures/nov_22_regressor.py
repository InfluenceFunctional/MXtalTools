from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import linregress
from scipy.stats import gaussian_kde


def nov_22_paper_regression_plots(config):
    '''
    Calculations
    '''
    #test_epoch_stats_dict = np.load('C:/Users\mikem\Desktop\CSP_runs/951_test_epoch_stats_dict.npy', allow_pickle=True).item()
    test_epoch_stats_dict = np.load('C:/Users\mikem\crystals\CSP_runs/951_test_epoch_stats_dict.npy', allow_pickle=True).item()

    target_mean = config.dataDims['target_mean']
    target_std = config.dataDims['target_std']

    target = np.asarray(test_epoch_stats_dict['generator density target'])
    prediction = np.asarray(test_epoch_stats_dict['generator density prediction'])
    orig_target = target * target_std + target_mean
    orig_prediction = prediction * target_std + target_mean

    volume_ind = config.dataDims['tracking_features'].index('molecule volume')
    mass_ind = config.dataDims['tracking_features'].index('molecule mass')
    molwise_density = test_epoch_stats_dict['tracking_features'][:, mass_ind] / test_epoch_stats_dict['tracking_features'][:, volume_ind]
    target_density = molwise_density * orig_target * 1.66  # conversion from amu/A^3 to g/mL
    predicted_density = molwise_density * orig_prediction * 1.66

    losses = ['normed error', 'abs normed error', 'squared error']
    loss_dict = {}
    losses_dict = {}
    for loss in losses:
        if loss == 'normed error':
            loss_i = (orig_target - orig_prediction) / np.abs(orig_target)
        elif loss == 'abs normed error':
            loss_i = np.abs((orig_target - orig_prediction) / np.abs(orig_target))
        elif loss == 'squared error':
            loss_i = (orig_target - orig_prediction) ** 2
        losses_dict[loss] = loss_i  # huge unnecessary upload
        loss_dict[loss + ' mean'] = np.mean(loss_i)
        loss_dict[loss + ' std'] = np.std(loss_i)
        print(loss + ' mean: {:.3f} std: {:.3f}'.format(loss_dict[loss + ' mean'], loss_dict[loss + ' std']))

    linreg_result = linregress(orig_target, orig_prediction)
    loss_dict['Regression R'] = linreg_result.rvalue
    loss_dict['Regression slope'] = linreg_result.slope

    losses = ['density normed error', 'density abs normed error', 'density squared error']
    for loss in losses:
        if loss == 'density normed error':
            loss_i = (target_density - predicted_density) / np.abs(target_density)
        elif loss == 'density abs normed error':
            loss_i = np.abs((target_density - predicted_density) / np.abs(target_density))
        elif loss == 'density squared error':
            loss_i = (target_density - predicted_density) ** 2
        losses_dict[loss] = loss_i  # huge unnecessary upload
        loss_dict[loss + ' mean'] = np.mean(loss_i)
        loss_dict[loss + ' std'] = np.std(loss_i)
        print(loss + ' mean: {:.3f} std: {:.3f}'.format(loss_dict[loss + ' mean'], loss_dict[loss + ' std']))

    linreg_result = linregress(target_density, predicted_density)
    loss_dict['Density Regression R'] = linreg_result.rvalue
    loss_dict['Density Regression slope'] = linreg_result.slope

    '''
    summary table
    '''
    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
        )
    )

    fig = go.Figure(data=go.Table(
        header=dict(values=['Metric', '$C_{pack}$', 'density']),
        cells=dict(values=[['MAE', '$\sigma$', 'R', 'Slope'],
                           [loss_dict['abs normed error mean'], loss_dict['abs normed error std'], loss_dict['Regression R'], loss_dict['Regression slope']],
                           [loss_dict['density abs normed error mean'], loss_dict['density abs normed error std'], loss_dict['Density Regression R'], loss_dict['Density Regression slope']],
                           ], format=["", ".3", ".3"])))
    fig.update_layout(width=300)
    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/regression_topline.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    4-panel error distribution
    '''
    xy = np.vstack([orig_target, orig_prediction])
    z = gaussian_kde(xy)(xy)
    xy2 = np.vstack([target_density, predicted_density])
    z2 = gaussian_kde(xy)(xy)

    fig = make_subplots(rows=2, cols=2, subplot_titles=('a)', 'b)', 'c)', 'd)'), vertical_spacing=0.12)

    xline = [0, 10]  # np.linspace(max(min(orig_target), min(orig_prediction)), min(max(orig_target), max(orig_prediction)), 2)
    fig.add_trace(go.Scattergl(x=orig_target, y=orig_prediction, mode='markers', marker=dict(color=z), opacity=0.1),
                  row=1, col=1)
    fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)'), row=1, col=1)
    fig.update_layout(xaxis_title='targets', yaxis_title='predictions')

    fig.add_trace(go.Histogram(x=orig_target - orig_prediction,
                               histnorm='probability density',
                               nbinsx=500,
                               name="Error Distribution",
                               marker_color='rgba(0,0,100,1)'), row=2, col=1)

    xline = [0, 10]  # np.linspace(max(min(target_density), min(predicted_density)), min(max(target_density), max(predicted_density)), 2)
    fig.add_trace(go.Scattergl(x=target_density, y=predicted_density, mode='markers', marker=dict(color=z2), opacity=0.1),
                  row=1, col=2)
    fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)'), row=1, col=2)
    fig.update_layout(xaxis_title='targets', yaxis_title='predictions')

    fig.add_trace(go.Histogram(x=target_density - predicted_density,
                               histnorm='probability density',
                               nbinsx=500,
                               name="Error Distribution",
                               marker_color='rgba(0,0,100,1)', ), row=2, col=2)
    fig.update_layout(showlegend=False)

    fig.update_yaxes(title_text='Predicted Packing Coefficient', row=1, col=1, dtick=0.05, range=[0.55, 0.8], tickformat=".2f")
    fig.update_yaxes(title_text='Predicted Density (g/cm<sup>3</sup>)', row=1, col=2, dtick=0.5, range=[0.8, 4], tickformat=".1f")
    fig.update_xaxes(title_text='True Packing Coefficient', row=1, col=1, dtick=0.05, range=[0.55, 0.8], tickformat=".2f")
    fig.update_xaxes(title_text='True Density (g/cm<sup>3</sup>)', row=1, col=2, dtick=0.5, range=[0.8, 4], tickformat=".1f")

    fig.update_xaxes(title_text='Packing Coefficient Error', row=2, col=1, dtick=0.05, tickformat=".2f")
    fig.update_xaxes(title_text='Density Error (g/cm<sup>3</sup>)', row=2, col=2, dtick=0.1, tickformat=".1f")

    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))

    fig.layout.annotations[0].update(x=0.025)
    fig.layout.annotations[2].update(x=0.025)
    fig.layout.annotations[1].update(x=0.575)
    fig.layout.annotations[3].update(x=0.575)

    fig.update_layout(height=600, width=800, font_family="Arial")
    fig.layout.margin = layout.margin
    fig.layout.margin.b += 20

    fig.write_image('../paper1_figs/regression_distributions.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    Error correlates
    '''
    # correlate losses with molecular features
    tracking_features = np.asarray(test_epoch_stats_dict['tracking_features'])
    g_loss_correlations = np.zeros(config.dataDims['num_tracking_features'])
    features = []
    ind = 0
    for i in range(config.dataDims['num_tracking_features']):  # not that interesting
        if (np.average(tracking_features[:, i] != 0) > 0.05) and \
                (config.dataDims['tracking_features'][i] != 'crystal z prime') and \
                (config.dataDims['tracking_features'][i] != 'molecule point group is C1') and \
                (config.dataDims['tracking_features'][i] != 'crystal calculated density'):  # if we have at least 1# relevance

            coeff = np.corrcoef(np.abs((orig_target - orig_prediction) / np.abs(orig_target)), tracking_features[:, i], rowvar=False)[0, 1]
            if np.abs(coeff) > 0.05:
                features.append(config.dataDims['tracking_features'][i])
                g_loss_correlations[ind] = coeff
                ind += 1
    g_loss_correlations = g_loss_correlations[:ind]

    g_sort_inds = np.argsort(g_loss_correlations)
    g_loss_correlations = g_loss_correlations[g_sort_inds]
    features_sorted = [features[i] for i in g_sort_inds]
    features_sorted_cleaned_i = [feat.replace('molecule', 'mol') for feat in features_sorted]
    features_sorted_cleaned = [feat.replace('crystal', 'crys') for feat in features_sorted_cleaned_i]

    fig = go.Figure(go.Bar(
        y=features_sorted_cleaned,
        x=[corr for corr in g_loss_correlations],
        orientation='h',
        text=np.asarray([corr for corr in g_loss_correlations]).astype('float16'),
        textposition='auto',
        texttemplate='%{text:.2}'
    ))
    fig.layout.margin = layout.margin
    fig.update_layout(width=500, height=600, font=dict(size=12))
    fig.update_layout(xaxis_title='R Value')

    fig.write_image('../paper1_figs/regression_correlates.png', scale=4)
    if config.machine == 'local':
        fig.show()

    '''
    need to isolate the samples which have bad predictions - 
    low density structures which are thought by the model to be higher density
    '''
    # bad_inds = np.argwhere(np.abs(orig_prediction - orig_target) > 0.05)[:, 0]
    # plt.clf()
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.title(config.dataDims['tracking_features'][i])
    #     plt.hist(tracking_features[bad_inds,i],density=True,bins=25,alpha=0.5)
    #     plt.hist(tracking_features[:, i], density=True, bins=25,alpha=0.5)
    # plt.tight_layout()

    fig = go.Figure()
    xline = [0, 10]  # np.linspace(max(min(target_density), min(predicted_density)), min(max(target_density), max(predicted_density)), 2)
    fig.add_trace(go.Scattergl(x=target_density, y=predicted_density, mode='markers', marker=dict(color=z2), opacity=0.1),
                  )
    fig.add_trace(go.Scattergl(x=xline, y=xline, marker_color='rgba(0,0,0,1)'))
    fig.update_layout(xaxis_title='targets', yaxis_title='predictions')

    fig.update_yaxes(title_text=r'$\text{Predicted Density }$', dtick=0.5, range=[0.8, 4], tickformat=".1f")
    fig.update_xaxes(title_text=r'$\text{True Density }$', dtick=0.5, range=[0.8, 4], tickformat=".1f")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=18), tickfont=dict(size=14))

    fig.update_layout(height=400, width=400)
    fig.write_image('../paper1_figs/TOC_regression.png', scale=4)

    return None

