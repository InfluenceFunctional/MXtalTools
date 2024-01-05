import numpy as np
from _plotly_utils.colors import sample_colorscale
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from bulk_molecule_classification.classifier_constants import defect_names, nic_ordered_class_names, urea_ordered_class_names, form2index, index2form, identifier2form
import plotly
from scipy.ndimage import gaussian_filter1d


def embedding_fig(results_dict, num_samples, classes, ordered_classes, run_temperatures, max_samples=1000, perplexity=30):

    sample_inds = np.random.choice(num_samples, size=min(max_samples, num_samples), replace=False)
    from sklearn.manifold import TSNE
    embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=20000,
                     init='pca', perplexity=perplexity).fit_transform(results_dict['Latents'][sample_inds])

    # target_colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', 10, colortype='rgb')
    target_colors = (
        'rgb(229, 134, 6)', 'rgb(93, 105, 177)', 'rgb(82, 188, 163)', 'rgb(153, 201, 69)', 'rgb(204, 97, 176)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)', 'rgb(165, 170, 153)')
    # sample_colorscale('hsv', 10)
    linewidths = [0, 0.75]
    linecolors = [None, 'DarkSlateGrey']

    fig = go.Figure()
    for t_ind in range(len(classes)):
        for d_ind in range(len(defect_names)):
            inds = np.argwhere((results_dict['Targets'][sample_inds] == t_ind) *
                               (results_dict['Defects'][sample_inds] == d_ind)
                               )[:, 0]

            fig.add_trace(go.Scattergl(x=embedding[inds, 0], y=embedding[inds, 1],
                                       mode='markers',
                                       marker_color=target_colors[t_ind],
                                       # marker_line_width=linewidths[d_ind],
                                       # marker_line_color=linecolors[d_ind],
                                       legendgroup=ordered_classes[t_ind],
                                       name=ordered_classes[t_ind],  # + ', ' + defect_names[d_ind],# + ', ' + str(temperature) + 'K',
                                       showlegend=True if d_ind == 0 else False,
                                       opacity=0.75 if t_ind != len(classes) - 1 else 0.25))
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                      xaxis_title='tSNE1', yaxis_title='tSNE2', xaxis_showticklabels=False, yaxis_showticklabels=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig, embedding


def form_accuracy_fig(results_dict, ordered_classes, temp_series):
    scores = {}
    melt_names = ['Crystal', 'Melt']
    fig = make_subplots(cols=2, rows=1, subplot_titles=["Low Temperature", "High Temperature"], horizontal_spacing=0.1)
    for temp_ind in range(2):
        if temp_ind == 0:
            inds = np.argwhere(results_dict['Temperature'] == temp_series[0])[:, 0]
            temp_type = "Low"
        else:
            inds = np.argwhere(results_dict['Temperature'] > temp_series[0])[:, 0]
            temp_type = "High"
        probs = results_dict['Type_Prediction'][inds]
        predicted_class = np.argmax(probs, axis=1)
        true_labels = results_dict['Targets'][inds]

        cmat = confusion_matrix(true_labels, predicted_class, normalize='true', labels=np.arange(len(ordered_classes)) if temp_ind < 2 else np.arange(len(melt_names)))

        try:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo')
        except ValueError:
            auc = 1

        f1 = f1_score(true_labels, predicted_class, average='micro')

        fig.add_trace(go.Heatmap(z=cmat, x=ordered_classes, y=ordered_classes,
                                 text=np.round(cmat, 2), texttemplate="%{text:.2g}", showscale=False,
                                 colorscale='blues'),
                      row=1, col=temp_ind + 1)

        fig.layout.annotations[temp_ind].update(text=f"{temp_type} T: ROC AUC={auc:.2f}, F1={f1:.2f}")

        scores[str(temp_type) + '_F1'] = f1
        scores[str(temp_type) + '_ROC_AUC'] = auc

    fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)
    fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)

    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Class", row=1, col=1)

    return fig, scores


def defect_accuracy_fig(results_dict, temp_series):
    scores = {}
    fig = make_subplots(cols=2, rows=1, subplot_titles=['a', 'b'], horizontal_spacing=0.1)
    for temp_ind in range(2):
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
                                 text=np.round(cmat, 2), texttemplate="%{text:.2g}", showscale=False),
                      row=1, col=temp_ind + 1)

        fig.layout.annotations[temp_ind].update(text=f"{temp_type} T: ROC AUC={auc:.2f}, F1={f1:.2f}")

        scores[str(temp_type) + '_F1'] = f1
        scores[str(temp_type) + '_ROC_AUC'] = auc

    fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)
    fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)
    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Class", row=1, col=1)

    return fig, scores


def all_accuracy_fig(results_dict, ordered_classes, temp_series):  # todo fix class ordering
    scores = {}
    fig = make_subplots(cols=2, rows=1, subplot_titles=['100K', '350K'], horizontal_spacing=0.2)
    for temp_ind in range(2):
        if temp_ind == 0:
            inds = np.argwhere(results_dict['Temperature'] == temp_series[0])[:, 0]
            temp_type = "Low"
        else:
            inds = np.argwhere(results_dict['Temperature'] > temp_series[0])[:, 0]
            temp_type = "High"

        defect_probs = results_dict['Defect_Prediction'][inds]
        form_probs = results_dict['Type_Prediction'][inds]
        probs = np.stack([np.outer(defect_probs[ind], form_probs[ind]).T.reshape(len(ordered_classes) * len(defect_names)) for ind in range(len(form_probs))])

        predicted_class = np.argmax(probs, axis=1)
        true_defects = results_dict['Defects'][inds]
        true_forms = results_dict['Targets'][inds]

        true_labels = np.asarray([target * 2 + defect for target, defect in zip(true_forms, true_defects)])

        combined_names = [class_name + ' ' + defect_name for class_name in ordered_classes for defect_name in defect_names]

        cmat = confusion_matrix(true_labels, predicted_class, normalize='true', labels=np.arange(len(combined_names)))

        try:
            auc = roc_auc_score(true_labels, probs, multi_class='ovo')
        except ValueError:
            auc = 1

        f1 = f1_score(true_labels, predicted_class, average='micro')

        fig.add_trace(go.Heatmap(z=cmat, x=combined_names, y=combined_names, colorscale='blues',
                                 text=np.round(cmat, 2), texttemplate="%{text:.2g}", showscale=False),
                      row=1, col=temp_ind + 1)

        fig.layout.annotations[temp_ind].update(text=f"{temp_type} T: ROC AUC={auc:.2f}, F1={f1:.2f}")

        scores[str(temp_type) + '_F1'] = f1
        scores[str(temp_type) + '_ROC_AUC'] = auc

    fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True, tickangle=90)
    fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)

    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Class", row=1, col=1)

    return fig, scores


def classifier_trajectory_analysis_fig(sorted_molwise_results_dict, time_steps, molecule_type, inside_radius=None, interface_mode=False):
    if molecule_type == 'urea':
        ordered_class_names = urea_ordered_class_names + ['Uncertain']
        mol_num_atoms = 8
    elif molecule_type == 'nicotinamide':
        ordered_class_names = nic_ordered_class_names + ['Uncertain']
        mol_num_atoms = 15

    stacked_plot = True
    colors = None
    inside_mode = 'radius'

    if interface_mode:
        ordered_class_names = ['I', 'IV', 'Other']
        stacked_plot = False
        colors = ['rgb(50, 50, 150)', 'rgb(50, 150, 50)', 'rgb(150, 50, 50)']
        inside_radius = 20
        inside_mode = 'z'

    num_classes = len(ordered_class_names)

    """trajectory analysis figure"""
    traj_dict = process_trajectory_data(
        inside_radius, mol_num_atoms, num_classes,
        ordered_class_names, sorted_molwise_results_dict, time_steps, inside_mode=inside_mode)

    if colors is None:
        colors = plotly.colors.DEFAULT_PLOTLY_COLORS + ['rgb(50, 50, 50)']
    sigma = min(5, len(traj_dict['overall_number']) / 100)
    fig = make_subplots(cols=3, rows=1,
                        subplot_titles=['All Molecules',
                                        'Core' if not interface_mode else 'Interface',
                                        'Surface' if not interface_mode else 'Bulk'],
                        x_title="Time (ns)", y_title="Form Fraction")
    for i2, key in enumerate(['overall_fraction', 'inside_fraction', 'outside_fraction']):
        traj = traj_dict[key]
        for ind in range(num_classes):
            fig.add_trace(go.Scatter(x=time_steps / 1000000,
                                     y=gaussian_filter1d(traj[:, ind], sigma),
                                     name=ordered_class_names[ind],
                                     legendgroup=ordered_class_names[ind],
                                     line_color=colors[ind],
                                     mode='lines',
                                     showlegend=True if i2 == 0 else False,
                                     stackgroup='one' if stacked_plot else None),
                          row=1, col=i2 + 1)
        if not interface_mode:
            fig.add_trace(go.Scattergl(x=time_steps / 1000000,
                                       y=gaussian_filter1d(traj_dict[key.split('_')[0] + '_confidence'][:], sigma),
                                       name="Confidence",
                                       legendgroup="Confidence",
                                       showlegend=True if i2 == 0 else False,
                                       marker_color='Grey'),
                          row=1, col=i2 + 1)
    if not interface_mode:
        fig.update_yaxes(range=[0, 1])

    fig2 = go.Figure()
    for ind in range(num_classes):
        fig2.add_trace(go.Scatter(x=time_steps / 1000000,
                                  y=gaussian_filter1d(traj_dict['overall_fraction'][:, ind], sigma),
                                  name=ordered_class_names[ind],
                                  legendgroup=ordered_class_names[ind],
                                  line_color=colors[ind],
                                  mode='lines',
                                  stackgroup='one' if stacked_plot else None),
                       )
    fig2.add_trace(go.Scattergl(x=time_steps / 1000000,
                                y=gaussian_filter1d(traj_dict['overall_fraction'][:], sigma),
                                name="Confidence",
                                marker_color='Grey'),
                   )
    fig2.update_yaxes(range=[0, 1])
    fig2.update_layout(xaxis_title="Time (ns)", yaxis_title='Form Prediction')

    return fig, fig2, traj_dict


def process_trajectory_data(inside_radius, mol_num_atoms, num_classes, ordered_class_names, sorted_molwise_results_dict, time_steps, inside_mode='radius'):
    pred_frac_traj = np.zeros((len(time_steps), num_classes))
    pred_frac_traj_in = np.zeros((len(time_steps), num_classes))
    pred_frac_traj_out = np.zeros((len(time_steps), num_classes))
    pred_confidence_traj = np.zeros(len(time_steps))
    pred_confidence_traj_in = np.zeros(len(time_steps))
    pred_confidence_traj_out = np.zeros(len(time_steps))
    pred_num = np.zeros_like(pred_frac_traj)
    pred_num_in = np.zeros_like(pred_frac_traj)
    pred_num_out = np.zeros_like(pred_frac_traj)
    for ind, probs in enumerate(sorted_molwise_results_dict['Molecule_Type_Prediction']):
        if inside_mode == 'radius':
            if inside_radius is not None:
                coords = sorted_molwise_results_dict['Coordinates'][ind]
                centroids = coords.reshape(coords.shape[0] // mol_num_atoms, mol_num_atoms, 3).mean(1)
                centroid_dists = np.linalg.norm(centroids - centroids.mean(0), axis=1)
                inside_inds = np.argwhere(centroid_dists < inside_radius)[:, 0]
                outside_inds = np.argwhere(centroid_dists >= inside_radius)[:, 0]
            else:
                inside_inds = np.argwhere(sorted_molwise_results_dict['Molecule_Coordination_Numbers'][ind] > 20)[:, 0]
                outside_inds = np.argwhere(sorted_molwise_results_dict['Molecule_Coordination_Numbers'][ind] <= 20)[:, 0]
        elif inside_mode == 'z':
            coords = sorted_molwise_results_dict['Coordinates'][ind]
            centroids = coords.reshape(coords.shape[0] // mol_num_atoms, mol_num_atoms, 3).mean(1)
            centroid_dists = np.abs(centroids - centroids.mean(0))[:, 2]  # only track the z axis
            inside_inds = np.argwhere(centroid_dists < inside_radius)[:, 0]
            outside_inds = np.argwhere(centroid_dists >= inside_radius)[:, 0]

        inside_probs = probs[inside_inds]
        outside_probs = probs[outside_inds]

        pred = np.argmax(probs, axis=-1)
        max_confidence = probs.max(1)
        pred[max_confidence < 0.5] = num_classes - 1  # insufficiently high confidence gets assigned 'uncertain'
        inside_pred = pred[inside_inds]
        outside_pred = pred[outside_inds]

        pred_confidence_traj[ind] = probs.max(1).mean()  # get_prediction_confidence(probs).mean()
        pred_confidence_traj_in[ind] = inside_probs.max(1).mean()  # get_prediction_confidence(inside_probs).mean()
        pred_confidence_traj_out[ind] = outside_probs.max(1).mean()  # get_prediction_confidence((outside_probs)).mean()

        uniques, counts = np.unique(pred, return_counts=True)
        count_sum = sum(counts)
        for thing, count in zip(uniques, counts):
            pred_frac_traj[ind, thing] = count / count_sum
            pred_num[ind, thing] = count

        uniques, counts = np.unique(inside_pred, return_counts=True)
        count_sum = sum(counts)
        for thing, count in zip(uniques, counts):
            pred_frac_traj_in[ind, thing] = count / count_sum
            pred_num_in[ind, thing] = count

        uniques, counts = np.unique(outside_pred, return_counts=True)
        count_sum = sum(counts)
        for thing, count in zip(uniques, counts):
            pred_frac_traj_out[ind, thing] = count / count_sum
            pred_num_out[ind, thing] = count

    traj_dict = {'overall_fraction': pred_frac_traj,
                 'inside_fraction': pred_frac_traj_in,
                 'outside_fraction': pred_frac_traj_out,
                 'overall_confidence': pred_confidence_traj,
                 'inside_confidence': pred_confidence_traj_in,
                 'outside_confidence': pred_confidence_traj_out,
                 'inside_number': pred_num_in,
                 'outside_number': pred_num_out,
                 'overall_number': pred_num,
                 'time_steps': time_steps,
                 'ordered_classes': ordered_class_names}
    return traj_dict


def check_for_extra_values(row, extra_axes, extra_values):
    if extra_axes is not None:
        bools = []
        for iv, axis in enumerate(extra_axes):
            bools.append(extra_values[iv] == row[axis])
        return all(bools)
    else:
        return True


def collate_property_over_multiple_runs(target_property, results_df, xaxis, xaxis_title, yaxis, yaxis_title, unique_structures, extra_axes=None, extra_axes_values=None, take_mean=False):
    n_samples = np.zeros((len(unique_structures), len(xaxis), len(yaxis)))

    for iX, xval in enumerate(xaxis):
        for iC, struct in enumerate(unique_structures):
            for iY, yval in enumerate(yaxis):
                for ii, row in results_df.iterrows():
                    if row['structure_identifier'] == struct:
                        if row[xaxis_title] == xval:
                            if row[yaxis_title] == yval:
                                if check_for_extra_values(row, extra_axes, extra_axes_values):
                                    try:
                                        aa = row[target_property]  # see if it's non-empty
                                        n_samples[iC, iX, iY] += 1
                                    except:
                                        pass

    shift_heatmap = np.zeros((len(unique_structures), len(xaxis), len(yaxis)))
    for iX, xval in enumerate(xaxis):
        for iC, struct in enumerate(unique_structures):
            for iY, yval in enumerate(yaxis):
                for ii, row in results_df.iterrows():
                    if row['structure_identifier'] == struct:
                        if row[xaxis_title] == xval:
                            if row[yaxis_title] == yval:
                                if check_for_extra_values(row, extra_axes, extra_axes_values):
                                    try:
                                        if take_mean:
                                            shift_heatmap[iC, iX, iY] += row[target_property].mean() / n_samples[iC, iX, iY]  # take mean over seeds
                                        else:
                                            shift_heatmap[iC, iX, iY] += row[target_property] / n_samples[iC, iX, iY]
                                    except:
                                        shift_heatmap[iC, iX, iY] = 0

    return shift_heatmap, n_samples


def plot_classifier_pies(results_df, xaxis_title, yaxis_title, class_names, extra_axes=None, extra_axes_values=None):
    xaxis = np.unique(results_df[xaxis_title])
    yaxis = np.unique(results_df[yaxis_title])
    unique_structures = np.unique(results_df['structure_identifier'])
    heatmaps, samples = [], []

    for classo in class_names:
        shift_heatmap, n_samples = collate_property_over_multiple_runs(
            classo, results_df, xaxis, xaxis_title, yaxis, yaxis_title, unique_structures,
            extra_axes=extra_axes, extra_axes_values=extra_axes_values, take_mean=False)
        heatmaps.append(shift_heatmap)
        samples.append(n_samples)
    heatmaps = np.stack(heatmaps)
    heatmaps = np.transpose(heatmaps, axes=(0, 1, 3, 2))
    samples = np.stack(samples)

    xlen = len(xaxis)
    ylen = len(yaxis)

    for form_ind, form in enumerate(unique_structures):
        titles = []
        ind = 0
        for i in range(ylen):
            for j in range(xlen):
                row = xlen - ind // ylen - 1
                col = ind % xlen
                titles.append(f"{xaxis_title}={xaxis[j]} <br> {yaxis_title}={yaxis[i]}")
                ind += 1

        fig = make_subplots(rows=ylen, cols=xlen, subplot_titles=titles,
                            specs=[[{"type": "domain"} for _ in range(xlen)] for _ in range(ylen)])

        ind = 0
        for i in range(xlen):
            for j in range(ylen):
                row = j + 1
                col = i + 1
                fig.add_trace(go.Pie(labels=class_names, values=heatmaps[:, form_ind, j, i], sort=False
                                     ),
                              row=row, col=col)
                ind += 1
        fig.update_traces(hoverinfo='label+percent+name', textinfo='none', hole=0.4)
        fig.layout.legend.traceorder = 'normal'
        fig.update_layout(title=form + " Clusters Classifier Outputs")
        fig.update_annotations(font_size=10)

        if extra_axes is not None:
            property_name = form + ' ' + str(extra_axes) + ' ' + str(extra_axes_values)
        else:
            property_name = form
        fig.update_layout(title=property_name)
        fig.show(renderer="browser")
        fig.write_image(form + "_classifier_pies.png")


def cluster_property_heatmap(results_df, property, xaxis_title, yaxis_title, extra_axes=None, extra_axes_values=None, take_mean=False, norm_against_zero_y=False):
    xaxis = np.unique(results_df[xaxis_title])
    yaxis = np.unique(results_df[yaxis_title])
    unique_structures = np.unique(results_df['structure_identifier'])

    shift_heatmap, n_samples = collate_property_over_multiple_runs(property,
                                                                   results_df, xaxis, xaxis_title, yaxis, yaxis_title, unique_structures,
                                                                   extra_axes=extra_axes, extra_axes_values=extra_axes_values, take_mean=take_mean)
    unique_structures = ["Form " + str(identifier2form[struct]) for struct in unique_structures]

    fig = make_subplots(rows=1, cols=len(unique_structures), subplot_titles=unique_structures)

    for i in range(1, len(unique_structures) + 1):
        if norm_against_zero_y:
            heatmap = shift_heatmap[i - 1] / shift_heatmap[i - 1][:, 0, None]
            max_val, min_val = None, None
        else:
            heatmap = shift_heatmap[i - 1]
            max_val = np.amax(shift_heatmap[i - 1])
            min_val = np.amin(shift_heatmap[i - 1])

        fig.add_trace(go.Heatmap(z=heatmap.T,
                                 text=n_samples[i - 1].T,
                                 texttemplate="%{text}",
                                 colorscale='Viridis', zmax=max_val, zmin=min_val,
                                 ), row=1, col=i)

        fig.update_xaxes(title_text=xaxis_title, row=1, col=i)
        fig.update_yaxes(title_text=yaxis_title, row=1, col=i)

    fig.update_layout(xaxis=dict(
        tickmode='array',
        tickvals=np.arange(len(xaxis)),
        ticktext=xaxis
    ))
    fig.update_layout(yaxis=dict(
        tickmode='array',
        tickvals=np.arange(len(yaxis)),
        ticktext=yaxis
    ))
    if len(unique_structures) > 1:
        fig.update_layout(xaxis2=dict(
            tickmode='array',
            tickvals=np.arange(len(xaxis)),
            ticktext=xaxis
        ))
        fig.update_layout(yaxis2=dict(
            tickmode='array',
            tickvals=np.arange(len(yaxis)),
            ticktext=yaxis
        ))
    if len(unique_structures) > 2:
        fig.update_layout(xaxis3=dict(
            tickmode='array',
            tickvals=np.arange(len(xaxis)),
            ticktext=xaxis
        ))
        fig.update_layout(yaxis3=dict(
            tickmode='array',
            tickvals=np.arange(len(yaxis)),
            ticktext=yaxis
        ))
    fig.update_traces(colorbar=dict(tickmode="array", tickvals=[0, 0.25, 0.5, 0.75, 1], ticks="outside"), coloraxis='coloraxis', showscale=True)
    fig.update_layout(coloraxis={'colorscale': 'viridis', 'cmin': 0})
    fig.update_layout()
    if extra_axes is not None:
        property_name = property + ' ' + str(extra_axes) + ' ' + str(extra_axes_values)
    else:
        property_name = property
    fig.update_layout(title=property_name)
    fig.show(renderer="browser")
    fig.write_image(property + "_heatmap.png")

    return fig
