import numpy as np
from _plotly_utils.colors import sample_colorscale
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from classify_lammps_trajs.NICOAM_constants import class_names, defect_names, form2index
import plotly
from scipy.ndimage import gaussian_filter1d


def embedding_fig(results_dict, num_samples):

    sample_inds = np.random.choice(num_samples, size=min(1000, num_samples), replace=False)
    from sklearn.manifold import TSNE
    embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=5000,
                     init='pca', perplexity=30).fit_transform(results_dict['Latents'][sample_inds])

    # target_colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', 10, colortype='rgb')
    target_colors = (
        'rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)', 'rgb(165, 170, 153)')
    # sample_colorscale('hsv', 10)
    linewidths = [0, 1]
    linecolors = [None, 'DarkSlateGrey']
    symbols = ['circle', 'diamond', 'square']

    fig = go.Figure()
    for temp_ind, temperature in enumerate([100, 350, 950]):
        for t_ind in range(10):
            for d_ind in range(len(defect_names)):
                t_ind = form2index[t_ind]
                inds = np.argwhere((results_dict['Temperature'][sample_inds] == temperature) *
                                   (results_dict['Targets'][sample_inds] == t_ind) *
                                   (results_dict['Defects'][sample_inds] == d_ind)
                                   )[:, 0]

                fig.add_trace(go.Scattergl(x=embedding[inds, 0], y=embedding[inds, 1],
                                           mode='markers',
                                           marker_color=target_colors[t_ind],
                                           marker_symbol=symbols[temp_ind],
                                           marker_line_width=linewidths[d_ind],
                                           marker_line_color=linecolors[d_ind],
                                           legendgroup=class_names[t_ind],
                                           name=class_names[t_ind],  # + ', ' + defect_names[d_ind],# + ', ' + str(temperature) + 'K',
                                           showlegend=True if (temperature == 100 or temperature == 950) and d_ind == 0 else False,
                                           opacity=0.75))

    return fig


def form_accuracy_fig(results_dict):
    scores = {}
    fig = make_subplots(cols=3, rows=1, subplot_titles=['100K', '350K', '950K'], y_title="True Forms", x_title="Predicted Forms")
    for temp_ind, temperature in enumerate([100, 350, 950]):
        inds = np.argwhere(results_dict['Temperature'] == temperature)[:, 0]
        probs = results_dict['Type_Prediction'][inds]
        predicted_class = np.argmax(probs, axis=1)
        true_labels = results_dict['Targets'][inds]

        if temperature == 950:
            true_labels = np.ones_like(true_labels)
            predicted_class = np.asarray(predicted_class == 9).astype(int)
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


def defect_accuracy_fig(results_dict):
    scores = {}
    fig = make_subplots(cols=3, rows=1, subplot_titles=['100K', '350K', '950K'], y_title="True Defects", x_title="Predicted Defects")
    for temp_ind, temperature in enumerate([100, 350, 950]):
        inds = np.argwhere(results_dict['Temperature'] == temperature)[:, 0]
        probs = results_dict['Defect_Prediction'][inds]
        predicted_class = np.argmax(probs, axis=1)
        true_labels = results_dict['Defects'][inds]

        cmat = confusion_matrix(true_labels, predicted_class, normalize='true')

        try:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo')
        except ValueError:
            auc = 1

        f1 = f1_score(true_labels, predicted_class, average='micro')

        fig.add_trace(go.Heatmap(z=cmat, x=defect_names, y=defect_names,
                                 text=np.round(cmat, 2), texttemplate="%{text:.2g}", showscale=False),
                      row=1, col=temp_ind + 1)

        fig.layout.annotations[temp_ind].update(text=f"{temperature}K: ROC AUC={auc:.2f}, F1={f1:.2f}")

        scores[str(temperature) + '_F1'] = f1
        scores[str(temperature) + '_ROC_AUC'] = auc

    return fig, scores


def all_accuracy_fig(results_dict):  # todo fix class ordering
    scores = {}
    fig = make_subplots(cols=2, rows=1, subplot_titles=['100K', '350K'], y_title="True Class", x_title="Predicted Class")
    for temp_ind, temperature in enumerate([100, 350]):
        inds = np.argwhere(results_dict['Temperature'] == temperature)[:, 0]
        defect_probs = results_dict['Defect_Prediction'][inds]
        form_probs = results_dict['Type_Prediction'][inds]
        # form_probs = form_probs[:, list(form2index.values())]

        probs = np.stack([np.outer(defect_probs[ind], form_probs[ind]).T.reshape(len(class_names) * len(defect_names)) for ind in range(len(form_probs))])

        predicted_class = np.argmax(probs, axis=1)
        true_defects = results_dict['Defects'][inds]
        true_forms = results_dict['Targets'][inds]
        # true_forms = [form2index[form] for form in true_forms]

        true_labels = np.asarray([target * 2 + defect for target, defect in zip(true_forms, true_defects)])

        # ordered_class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'Disordered']
        combined_names = [class_name + ' ' + defect_name for class_name in class_names for defect_name in defect_names]

        cmat = confusion_matrix(true_labels, predicted_class, normalize='true')

        try:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo')
        except ValueError:
            auc = 1

        f1 = f1_score(true_labels, predicted_class, average='micro')

        fig.add_trace(go.Heatmap(z=cmat, x=combined_names, y=combined_names,
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
    pred_confidence_traj = np.zeros(len(time_steps))
    pred_confidence_traj_in = np.zeros(len(time_steps))
    pred_confidence_traj_out = np.zeros(len(time_steps))

    def get_prediction_confidence(p1):
        return -np.log10(p1.prod(1)) / len(p1[0]) / np.log10(len(p1[0]))

    for ind, probs in enumerate(sorted_molwise_results_dict['Molecule_Type_Prediction']):
        inside_probs = probs[np.argwhere(sorted_molwise_results_dict['Molecule_Coordination_Numbers'][ind] > 20)][:, 0]
        outside_probs = probs[np.argwhere(sorted_molwise_results_dict['Molecule_Coordination_Numbers'][ind] < 20)][:, 0]

        pred = np.argmax(probs, axis=-1)
        inside_pred = np.argmax(inside_probs, axis=-1)
        outside_pred = np.argmax(outside_probs, axis=-1)

        pred_confidence_traj[ind] = probs.max(1).mean()  # get_prediction_confidence(probs).mean()
        pred_confidence_traj_in[ind] = inside_probs.max(1).mean()  # get_prediction_confidence(inside_probs).mean()
        pred_confidence_traj_out[ind] = outside_probs.max(1).mean()  # get_prediction_confidence((outside_probs)).mean()

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

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS
        sigma = 5
        fig2 = make_subplots(cols=3, rows=1, subplot_titles=['All Molecules', 'Core', 'Surface'], x_title="Time (ns)", y_title="Prediction ")
        fig2.add_trace(go.Scattergl(x=time_steps / 1000000,
                                    y=gaussian_filter1d(pred_confidence_traj[:], sigma),
                                    name=class_names[ind],
                                    legendgroup=class_names[ind],
                                    marker_color=colors[ind]),
                       row=1, col=1)
        fig2.add_trace(go.Scattergl(x=time_steps / 1000000,
                                    y=gaussian_filter1d(pred_confidence_traj_in[:], sigma),
                                    name=class_names[ind],
                                    legendgroup=class_names[ind],
                                    showlegend=False,
                                    marker_color=colors[ind]),
                       row=1, col=2)
        fig2.add_trace(go.Scattergl(x=time_steps / 1000000,
                                    y=gaussian_filter1d(pred_confidence_traj_out[:], sigma),
                                    name=class_names[ind],
                                    legendgroup=class_names[ind],
                                    showlegend=False,
                                    marker_color=colors[ind]),
                       row=1, col=3)

    return fig, fig2
