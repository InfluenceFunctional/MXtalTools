import numpy as np
from _plotly_utils.colors import n_colors
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def log_mini_csp_scores_distributions(config, wandb, sampling_dict, real_samples_dict, real_data):
    """
    report on key metrics from mini-csp
    """
    scores_labels = ['score', 'vdw overlap', 'density']  # , 'h bond score']
    fig = make_subplots(rows=1, cols=len(scores_labels), vertical_spacing=0.075,
                        horizontal_spacing=0.075)

    colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', min(15, real_data.num_graphs) + 1, colortype='rgb')

    for i, label in enumerate(scores_labels):
        row = 1  # i // 2 + 1
        col = i + 1  # i % 2 + 1
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
            fig.update_xaxes(title_text=label, row=1, col=col)

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

    fig.update_layout(showlegend=False, yaxis_showgrid=True)  # legend_traceorder='reversed',

    fig.layout.margin = layout.margin

    if config.wandb.log_figures:
        wandb.log({'Mini-CSP Scores': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None
