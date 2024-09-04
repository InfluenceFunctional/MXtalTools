import plotly.graph_objects as go
import numpy as np
import torch
from typing import Union
import plotly.express.colors as pc
from plotly.subplots import make_subplots


def single_property_distribution(y: Union[np.ndarray, torch.Tensor],
                                 xaxis_title: str,
                                 bandwidth=None,
                                 xaxis_range=None):
    if torch.is_tensor(y):
        y = y.cpu().detach().numpy()

    fig = go.Figure()
    if bandwidth is None:
        bandwidth = float(np.ptp(y) / 200)

    fig.add_trace(go.Violin(
        x=y, side='positive', orientation='h',
        width=1,
        name=f'Mean = {np.mean(y):.2f}',
        meanline_visible=True,
        bandwidth=bandwidth,
        points=False,
    ))
    fig.update_layout(xaxis_title=xaxis_title)

    if xaxis_range is not None:
        fig.update_layout(xaxis_range=xaxis_range)

    return fig


def stacked_property_distribution(y: Union[np.ndarray, torch.Tensor],
                                  xaxis_title: str,
                                  yaxis_title: str,
                                  bandwidth=None,
                                  xaxis_range=None
                                  ):
    if torch.is_tensor(y):
        y = y.cpu().detach().numpy()

    if y.ndim == 1:
        y = y[np.newaxis, :]
        num_stacks = 1
    else:
        num_stacks = y.shape[0]

    colors = pc.n_colors('rgb(255, 150, 50)', 'rgb(0, 25, 250)', num_stacks)

    fig = go.Figure()
    if bandwidth is None:
        bandwidth = float(np.ptp(y.flatten()) / 200)

    for ind in range(num_stacks):
        fig.add_trace(go.Violin(
            x=y[ind, :], y=ind, side='positive', orientation='h',
            width=1,
            name=yaxis_title + f'Mean = {np.mean(y[ind, :]):.2f}',
            showlegend=True,
            meanline_visible=True,
            bandwidth=bandwidth,
            points=False,
            line_color=colors[ind]
        ))
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    if xaxis_range is not None:
        fig.update_xaxes(range=xaxis_range)

    fig.update_traces(opacity=0.5)

    return fig


def stacked_cell_distributions(samples: Union[np.ndarray, torch.Tensor],
                               xaxis_title: str,
                               yaxis_title: str,
                               ):
    lattice_features = ['cell_a', 'cell_b', 'cell_c', 'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y',
                        'aunit_z', 'aunit_theta', 'aunit_phi', 'aunit_r']
    n_crystal_features = 12

    if torch.is_tensor(samples):
        samples = samples.cpu().detach().numpy()

    num_stacks = samples.shape[0]

    colors = pc.n_colors('rgb(255, 150, 50)', 'rgb(0, 25, 250)', num_stacks)
    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        row = i // 3 + 1
        col = i % 3 + 1
        for t_ind in range(num_stacks):
            values = samples[t_ind, :, i].flatten()

            fig.add_trace(go.Violin(
                x=values, y=t_ind * np.ones_like(values), side='positive', orientation='h', width=6,
                name=f'{len(values)} samples at {t_ind}',
                legendgroup=f'{len(values)} samples at {t_ind}',
                showlegend=True if i == 0 else False,
                meanline_visible=True,
                bandwidth=float(np.ptp(values) / 100),
                points=False,
                line_color=colors[t_ind],
            ),
                row=row, col=col
            )

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
    fig.update_traces(opacity=0.5)

    fig.update_xaxes(title=xaxis_title)
    fig.update_yaxes(title=yaxis_title)

    fig.show()


def prior_vs_rdf_dist(prior_to_compare, rdf_dists):
    params_dists = torch.cdist(prior_to_compare, prior_to_compare)
    # for ind in range(len(params_dists)):
    #     params_dists[ind,ind] = torch.nan
    fig = go.Figure(
        go.Scatter(x=params_dists.cpu().detach().flatten(), y=torch.log10(1 + rdf_dists.cpu().detach().flatten()),
                   mode='markers'))
    fig.update_layout(xaxis_title='params dist', yaxis_title='RDF dist')

    return fig
