import numpy as np
from _plotly_utils.colors import sample_colorscale
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from classify_lammps_trajs.NICOAM_constants import class_names
import plotly
from scipy.ndimage import gaussian_filter1d


def embedding_fig(results_dict, num_samples):

    sample_inds = np.random.choice(num_samples, size=min(1000, num_samples), replace=False)
    from sklearn.manifold import TSNE
    embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=5000,
                     init='pca', perplexity=30).fit_transform(results_dict['Latents'][sample_inds])

    # target_colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', 10, colortype='rgb')
    target_colors = sample_colorscale('portland', 10)
    symbols = ['x', 'diamond', 'cross']

    fig = go.Figure()
    for temp_ind, temperature in enumerate([100, 350, 950]):
        for target in range(10):
            inds = np.argwhere((results_dict['Temperature'][sample_inds] == temperature) * (results_dict['Targets'][sample_inds] == target))[:, 0]

            fig.add_trace(go.Scatter(x=embedding[inds, 0], y=embedding[inds, 1],
                                     mode='markers',
                                     marker_color=target_colors[target],
                                     marker_symbol=symbols[temp_ind],
                                     legendgroup=class_names[target],
                                     name=class_names[target] + ', ' + str(temperature) + 'K',
                                     showlegend=True,  # if temperature == 100 else False,
                                     opacity=0.75))
    return fig


def classifier_accuracy_figs(results_dict):
    scores = {}
    fig = make_subplots(cols=3, rows=1, subplot_titles=['100K', '350K', '950K'], y_title="True Forms", x_title="Predicted Forms")
    for temp_ind, temperature in enumerate([100, 350, 950]):
        inds = np.argwhere(results_dict['Temperature'] == temperature)[:, 0]
        probs = results_dict['Prediction'][inds]
        predicted_class = np.argmax(probs, axis=1)
        true_labels = results_dict['Targets'][inds]

        if temperature == 950:
            true_labels = np.ones_like(true_labels)
            predicted_class = np.asarray(predicted_class != 9).astype(int)
            probs_0 = probs[:, -2:]
            probs_0[:, 0] = probs[:, :-1].sum(1)
            probs = probs_0

        cmat = confusion_matrix(true_labels, predicted_class, normalize='true')

        try:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo')
        except ValueError:
            auc = 1

        f1 = f1_score(true_labels, predicted_class, average='micro')

        if temperature == 950:
            fig.add_trace(go.Heatmap(z=cmat, x=['Ordered', 'Disordered'], y=['Ordered', 'Disordered'],
                                     text=np.round(cmat, 2), texttemplate="%{text:.2g}", showscale=False),
                          row=1, col=temp_ind + 1)
        else:
            fig.add_trace(go.Heatmap(z=cmat, x=class_names, y=class_names,
                                     text=np.round(cmat, 2), texttemplate="%{text:.2g}", showscale=False),
                          row=1, col=temp_ind + 1)

        fig.layout.annotations[temp_ind].update(text=f"{temperature}K: ROC AUC={auc:.2f}, F1={f1:.2f}")

        scores[str(temperature) + '_F1'] = f1
        scores[str(temperature) + '_ROC_AUC'] = auc

    return fig, scores


def classifier_trajectory_analysis_fig(sorted_molwise_results_dict, time_steps):

    """trajectory analysis figure"""
    pred_frac_traj = np.zeros((len(time_steps), 10))
    pred_frac_traj_in = np.zeros((len(time_steps), 10))
    pred_frac_traj_out = np.zeros((len(time_steps), 10))

    for ind, pred in enumerate(sorted_molwise_results_dict['Molecule_Prediction']):
        inside_pred = pred[np.argwhere(sorted_molwise_results_dict['Molecule_Coordination_Numbers'][ind] > 20)][:, 0]
        outside_pred = pred[np.argwhere(sorted_molwise_results_dict['Molecule_Coordination_Numbers'][ind] < 20)][:, 0]

        uniques, counts = np.unique(pred, return_counts=True)
        count_sum = sum(counts)
        for thing, count in zip(uniques, counts):
            pred_frac_traj[ind, thing] = count / count_sum

        uniques, counts = np.unique(inside_pred, return_counts=True)
        count_sum = sum(counts)
        for thing, count in zip(uniques, counts):
            pred_frac_traj_in[ind, thing] = count / count_sum

        uniques, counts = np.unique(outside_pred, return_counts=True)
        count_sum = sum(counts)
        for thing, count in zip(uniques, counts):
            pred_frac_traj_out[ind, thing] = count / count_sum


    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    sigma = 5
    fig = make_subplots(cols=3, rows=1, subplot_titles=['All Molecules', 'Core', 'Surface'], x_title="Time (ns)", y_title="Form Fraction")
    for ind in range(10):
        fig.add_trace(go.Scattergl(x=time_steps / 1000000,
                                   y=gaussian_filter1d(pred_frac_traj[:, ind], sigma),
                                   name=class_names[ind],
                                   legendgroup=class_names[ind],
                                   marker_color=colors[ind]),
                      row=1, col=1)
        fig.add_trace(go.Scattergl(x=time_steps / 1000000,
                                   y=gaussian_filter1d(pred_frac_traj_in[:, ind], sigma),
                                   name=class_names[ind],
                                   legendgroup=class_names[ind],
                                   showlegend=False,
                                   marker_color=colors[ind]),
                      row=1, col=2)
        fig.add_trace(go.Scattergl(x=time_steps / 1000000,
                                   y=gaussian_filter1d(pred_frac_traj_out[:, ind], sigma),
                                   name=class_names[ind],
                                   legendgroup=class_names[ind],
                                   showlegend=False,
                                   marker_color=colors[ind]),
                      row=1, col=3)

    return fig
