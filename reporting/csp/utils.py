import numpy as np
from _plotly_utils.colors import n_colors
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale


def log_mini_csp_scores_distributions(config, wandb, sampling_dict, real_samples_dict, real_data, sym_info):
    """
    report on key metrics from mini-csp
    """
    scores_labels = ['score', 'vdw overlap', 'density']  # , 'h bond score']
    fig = make_subplots(rows=1, cols=len(scores_labels),
                        vertical_spacing=0.075, horizontal_spacing=0.075)

    colors = sample_colorscale('viridis',len(np.unique(sampling_dict['space group'])))
    #n_colors('rgb(250,50,5)', 'rgb(5,120,200)', len(np.unique(sampling_dict['space group'])), colortype='rgb')
    real_color = ('rgb(250,0,250)')
    opacity = 0.65

    for i, label in enumerate(scores_labels):
        row = 1  # i // 2 + 1
        col = i + 1  # i % 2 + 1
        for j in range(min(15, real_data.num_graphs)):
            bandwidth1 = np.ptp(sampling_dict[label][j]) / 50
            real_score = real_samples_dict[label][j]

            unique_space_group_inds = np.unique(sampling_dict['space group'][j])
            n_space_groups = len(unique_space_group_inds)
            space_groups = np.asarray([sym_info['space_groups'][sg] for sg in sampling_dict['space group'][j]])
            unique_space_groups = np.asarray([sym_info['space_groups'][sg] for sg in unique_space_group_inds])

            all_sample_score = sampling_dict[label][j]
            for k in range(n_space_groups):
                sample_score = all_sample_score[space_groups == unique_space_groups[k]]
                if label == 'score':
                    sample_score = sample_score - real_score
                    real_score = real_score - real_score

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

        unique_space_group_inds = np.unique(sampling_dict['space group'].flatten())
        n_space_groups = len(unique_space_group_inds)
        space_groups = np.asarray([sym_info['space_groups'][sg] for sg in sampling_dict['space group'].flatten()])
        unique_space_groups = np.asarray([sym_info['space_groups'][sg] for sg in unique_space_group_inds])

        for k in range(n_space_groups):
            all_sample_score = sampling_dict[label].flatten()[space_groups == unique_space_groups[k]]
            if label == 'score':
                all_sample_score = sampling_dict[label].flatten()[space_groups == unique_space_groups[k]] - real_samples_dict[label].mean()

            fig.add_trace(go.Violin(x=all_sample_score, y=['all samples' for _ in range(len(all_sample_score))],
                                    side='positive', orientation='h', width=2, line_color=colors[k],
                                    meanline_visible=True, bandwidth=np.ptp(sampling_dict[label].flatten()) / 100, opacity=opacity,
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
    fig.update_xaxes(row=1, col=scores_labels.index('vdw overlap') + 1, range=[0, 2])

    fig.update_layout(yaxis_showgrid=True)  # legend_traceorder='reversed',

    fig.layout.margin = layout.margin

    if config.wandb.log_figures:
        wandb.log({'Mini-CSP Scores': fig})
    if (config.machine == 'local') and False:
        fig.show()

    return None
