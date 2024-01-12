from copy import copy

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from bulk_molecule_classification.paper1_figs_utils import COLORS, FONTSIZE, LEGEND_FONTSIZE


def embedding_fig(results_dict, ordered_classes, max_samples=1000, perplexity=30):
    num_samples = len(results_dict['Targets'])
    sample_inds = np.random.choice(num_samples, size=min(max_samples, num_samples), replace=False)
    from sklearn.manifold import TSNE
    embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=20000,
                     init='pca', perplexity=perplexity).fit_transform(results_dict['Latents'][sample_inds])

    target_colors = copy(COLORS)
    melt_ind = len(ordered_classes)
    target_colors[melt_ind - 1] = COLORS[-1]

    fig = go.Figure()
    for t_ind in range(len(ordered_classes)):
        inds = np.argwhere((results_dict['Targets'][sample_inds] == t_ind)
                           )[:, 0]

        fig.add_trace(go.Scattergl(x=embedding[inds, 0], y=embedding[inds, 1],
                                   mode='markers',
                                   marker_size=5,
                                   marker_color=target_colors[t_ind],
                                   legendgroup=ordered_classes[t_ind],
                                   name=ordered_classes[t_ind],
                                   showlegend=True,
                                   opacity=.65))

    fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True, xaxis_zeroline=True, yaxis_zeroline=True,
                      xaxis_title='tSNE1', yaxis_title='tSNE2', xaxis_showticklabels=False, yaxis_showticklabels=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(linecolor='black', mirror=True)  # , gridcolor='grey', zerolinecolor='grey')
    fig.update_xaxes(linecolor='black', mirror=True)  # , gridcolor='grey', zerolinecolor='grey')
    fig.update_layout(font=dict(size=FONTSIZE), legend_font_size=LEGEND_FONTSIZE)
    return fig


def dual_embedding_fig(subplot_titles_list, results_dict1, results_dict2, ordered_classes, embed_keys, max_samples=1000, perplexity=30):
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles_list, x_title='tSNE1', y_title='tSNE2')

    for ic, results_dict in enumerate([results_dict1, results_dict2]):
        num_samples = len(results_dict['Targets'])
        sample_inds = np.random.choice(num_samples, size=min(max_samples, num_samples), replace=False)
        from sklearn.manifold import TSNE
        embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=20000,
                         init='pca', perplexity=perplexity).fit_transform(results_dict[embed_keys[ic]][sample_inds])

        target_colors = copy(COLORS)
        melt_ind = len(ordered_classes)
        target_colors[melt_ind - 1] = COLORS[-1]

        for t_ind in range(len(ordered_classes)):
            inds = np.argwhere((results_dict['Targets'][sample_inds] == t_ind)
                               )[:, 0]

            fig.add_trace(go.Scattergl(x=embedding[inds, 0], y=embedding[inds, 1],
                                       mode='markers',
                                       marker_size=5,
                                       marker_color=target_colors[t_ind],
                                       legendgroup=ordered_classes[t_ind],
                                       name=ordered_classes[t_ind],
                                       showlegend=True if ic == 0 else False,
                                       opacity=.65),
                          row=1, col=ic + 1)

    fig.update_layout(xaxis1_showgrid=True, yaxis1_showgrid=True, xaxis1_zeroline=True, yaxis1_zeroline=True,
                      xaxis1_showticklabels=False, yaxis1_showticklabels=False,
                      xaxis2_showgrid=True, yaxis2_showgrid=True, xaxis2_zeroline=True, yaxis2_zeroline=True,
                      xaxis2_showticklabels=False, yaxis2_showticklabels=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(linecolor='black', mirror=True)  # , gridcolor='grey', zerolinecolor='grey')
    fig.update_xaxes(linecolor='black', mirror=True)  # , gridcolor='grey', zerolinecolor='grey')
    fig.update_layout(font=dict(size=FONTSIZE),
                      legend_font_size=LEGEND_FONTSIZE,
                      legend_itemsizing='constant')

    return fig


def paper_defect_accuracy_fig(results_dict, defect_names, temp_series):
    scores = {}
    fig = go.Figure()
    for temp_ind in range(1, 2):
        if temp_ind == 0:
            inds = np.argwhere(results_dict['Temperature'] == temp_series[0])[:, 0]
            temp_type = "Low"
        else:
            inds = np.argwhere(results_dict['Temperature'] > temp_series[0])[:, 0]
            temp_type = "High"

        probs = results_dict['Defect_Prediction'][inds]
        predicted_class = np.argmax(probs, axis=1)
        true_labels = results_dict['Defects'][inds]

        cmat = confusion_matrix(true_labels, predicted_class, normalize='true', labels=np.arange(2))

        try:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo')
        except ValueError:
            auc = 1

        f1 = f1_score(true_labels, predicted_class, average='micro')

        fig.add_trace(go.Heatmap(z=cmat, x=defect_names, y=defect_names, colorscale='blues',
                                 text=np.round(cmat, 2), texttemplate="%{text:.2f}", showscale=False),
                      )

        scores[str(temp_type) + '_F1'] = f1
        scores[str(temp_type) + '_ROC_AUC'] = auc

    fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)
    fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)
    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Class")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(font=dict(size=FONTSIZE))
    return fig, scores


