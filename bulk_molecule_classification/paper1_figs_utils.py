import os
from copy import copy

import numpy as np
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

from bulk_molecule_classification.traj_analysis_figs import process_trajectory_data


def combined_embedding_fig(mk_results_dict, d_results_dict1, d_results_dict2,
                           ordered_classes, molecule_name,
                           max_samples=1000, perplexity=30):

    if molecule_name == 'urea':
        n_images = 7
    elif molecule_name == 'nicotinamide':
        n_images = 10

    image_path = r'D:\crystals_extra\classifier_training\polymorph_images/'
    if molecule_name == 'nicotinamide':
        filenames = ['NICOAM13.png',
                     'NICOAM14.png',
                     'NICOAM15.png',
                     'NICOAM16.png',
                     'NICOAM07.png',
                     'NICOAM18.png',
                     'NICOAM08.png',
                     'NICOAM09.png',
                     'NICOAM17.png',
                     'NIC_melt.png']
        stits = ['Form I',
                 'Form II',
                 'Form III',
                 'Form IV',
                 'Form V',
                 'Form VI',
                 'Form VII',
                 'Form VIII',
                 'Form IX',
                 'Melt']
    elif molecule_name == 'urea':
        filenames = ['UREAA.png',
                     'UREAB.png',
                     'UREAC.png',
                     'UREAI.png',
                     'UREAIII.png',
                     'UREAIV.png',
                     'UREA_melt.png']
        stits = ['Form A',
                 'Form B',
                 'Form C',
                 'Form I',
                 'Form III',
                 'Form IV',
                 'Melt']

    from PIL import Image

    images = [Image.open(image_path + pathi) for pathi in filenames]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["(a) Graph Embedding",
                                        "(b) GNN Final Layer",
                                        "(c) SFC Input",
                                        "(d) SFC Final Layer"],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.05,
                        )

    embed_keys = ['Embeddings', 'Latents']

    for ic, results_dict in enumerate([mk_results_dict, mk_results_dict, d_results_dict1, d_results_dict2]):
        if ic == 0:
            row = 1
            col = 1
            cind = 0
        elif ic == 1:
            row = 1
            col = 2
            cind = 1
        elif ic == 2:
            row = 2
            col = 1
            cind = 0
        elif ic == 3:
            row = 2
            col = 2
            cind = 1

        num_samples = len(results_dict['Targets'])
        sample_inds = np.random.choice(num_samples, size=min(max_samples, num_samples), replace=False)
        from sklearn.manifold import TSNE

        embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=20000,
                         init='pca', perplexity=perplexity).fit_transform(results_dict[embed_keys[cind]][sample_inds])

        target_colors = copy(COLORS)
        melt_ind = len(ordered_classes)
        target_colors[melt_ind - 1] = COLORS[-1]

        for t_ind in range(len(ordered_classes)):
            inds = np.argwhere((results_dict['Targets'][sample_inds] == t_ind)
                               )[:, 0]

            fig.add_trace(go.Scattergl(x=embedding[inds, 0] / np.amax(np.abs(embedding[:, 0])), y=embedding[inds, 1] / np.amax(np.abs(embedding[:, 1])),
                                       mode='markers',
                                       marker_size=5,
                                       marker_color=target_colors[t_ind],
                                       # legendgroup=ordered_classes[t_ind],
                                       name=ordered_classes[t_ind],
                                       showlegend=False,  # True if ic == 0 else False,
                                       opacity=.65),
                          row=row, col=col)

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)  # , showticklabels=False)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)  # , showticklabels=False)

    fig.update_layout(#xaxis1_title='tSNE1',
                      #xaxis2_title='tSNE1',
                      xaxis3_title='tSNE1',
                      xaxis4_title='tSNE1',
                      yaxis1_title='tSNE2',
                      #yaxis2_title='tSNE2',
                      yaxis3_title='tSNE2',
                      #yaxis4_title='tSNE2')
    )
    fig.update_layout(font=dict(size=FONTSIZE))
    fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))

    if molecule_name == 'nicotinamide':
        ylevels = [-0.2 - 0.325 * (ind % 2) for ind in range(n_images)]
        xlevels = np.linspace(-0.025, 0.875, int(np.ceil(n_images/2))).repeat(2)
    elif molecule_name == 'urea':
        ylevels = [-0.2 for ind in range(n_images)]
        xlevels = np.linspace(-.1, 0.875, n_images)

    imsize = 0.3 if molecule_name == 'nicotinamide' else 0.25
    for ind in range(n_images):

        fig.add_layout_image(
            dict(source=images[ind],
                 y=ylevels[ind], x=xlevels[ind])
        )
        fig.add_annotation(y=ylevels[ind] + 0.05, x=xlevels[ind] + 0.075,
                           text=stits[ind],
                           showarrow=False,
                           xref='paper',
                           yref='paper',
                           xanchor='left',
                           yanchor='top',
                           font_size=int(FONTSIZE * 0.8))
    fig.update_annotations(font_size=FONTSIZE)
    fig.update_layout_images(dict(
        xref="paper",
        yref="paper",
        sizex=imsize,
        sizey=imsize,
        xanchor="left",
        yanchor="top"
    ))
    if molecule_name == 'nicotinamide':
        fig.layout.margin.b = 375
    elif molecule_name == 'urea':
        fig.layout.margin.b = 275
    # fig.show()
    return fig


def pretty_embedding(mk_results_dict,
                     ordered_classes,
                     ):

    molecule_name = 'nicotinamide'
    n_images = 10

    image_path = r'D:\crystals_extra\classifier_training\polymorph_images/'
    if molecule_name == 'nicotinamide':
        filenames = ['NICOAM13.png',
                     'NICOAM14.png',
                     'NICOAM15.png',
                     'NICOAM16.png',
                     'NICOAM07.png',
                     'NICOAM18.png',
                     'NICOAM08.png',
                     'NICOAM09.png',
                     'NICOAM17.png',
                     'NIC_melt.png']
        stits = ['Form I',
                 'Form II',
                 'Form III',
                 'Form IV',
                 'Form V',
                 'Form VI',
                 'Form VII',
                 'Form VIII',
                 'Form IX',
                 'Melt']

    from PIL import Image

    images = [Image.open(image_path + pathi) for pathi in filenames]
    fig = go.Figure()

    num_samples = len(mk_results_dict['Targets'])
    sample_inds = np.random.choice(num_samples, size=min(1000, num_samples), replace=False)
    from sklearn.manifold import TSNE

    embedding = TSNE(n_components=2, learning_rate='auto', verbose=1, n_iter=20000,
                     init='pca', perplexity=30).fit_transform(mk_results_dict['Latents'][sample_inds])

    target_colors = copy(COLORS)
    melt_ind = len(ordered_classes)
    target_colors[melt_ind - 1] = COLORS[-1]

    for t_ind in range(len(ordered_classes)):
        inds = np.argwhere((mk_results_dict['Targets'][sample_inds] == t_ind)
                           )[:, 0]

        fig.add_trace(go.Scattergl(x=embedding[inds, 0] / np.amax(np.abs(embedding[:, 0])), y=embedding[inds, 1] / np.amax(np.abs(embedding[:, 1])),
                                   mode='markers',
                                   marker_size=5,
                                   marker_color=target_colors[t_ind],
                                   # legendgroup=ordered_classes[t_ind],
                                   name=ordered_classes[t_ind],
                                   showlegend=False,  # True if ic == 0 else False,
                                   opacity=.65))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)  # , showticklabels=False)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)  # , showticklabels=False)

    fig.update_layout(xaxis_title='tSNE1',
                      yaxis_title='tSNE2'
                      )

    fig.update_layout(font=dict(size=FONTSIZE))
    fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))

    ylevels = [-0.2 for _ in range(n_images)]
    xlevels = np.linspace(-0.075, 0.9, n_images)

    for ind in range(n_images):
        fig.add_layout_image(
            dict(source=images[ind],
                 y=ylevels[ind], x=xlevels[ind])
        )
        fig.add_annotation(y=ylevels[ind] + 0.05, x=xlevels[ind] + 0.05,
                           text=stits[ind],
                           showarrow=False,
                           xref='paper',
                           yref='paper',
                           xanchor='left',
                           yanchor='top',
                           font_size=int(FONTSIZE * 0.8))
    fig.update_annotations(font_size=FONTSIZE)
    imsize = 0.28 if molecule_name == 'urea' else 0.18
    fig.update_layout_images(dict(
        xref="paper",
        yref="paper",
        sizex=imsize,
        sizey=imsize,
        xanchor="left",
        yanchor="top"
    ))
    fig.layout.margin.b = 270

    return fig


def process_daisuke_dats():
    """
    daisuke's dat format
    index, p(x) for N classes, one-hot truth(x), T
    """
    dat_path = r'D:\crystals_extra\classifier_training\daisuke_confusion_mats/'
    dats = os.listdir(dat_path)
    data_dict = {}
    for ind in range(len(dats)):
        with open(dat_path + dats[ind], 'r') as f:
            lines = f.readlines()
            data_dict[dats[ind]] = np.zeros((len(lines), len(lines[0].split())))

            for il, line in enumerate(lines):
                data_dict[dats[ind]][il, :] = np.asarray(line.split(), dtype=float)

    d_results_dict = {}
    for key, data in data_dict.items():
        if key[0].lower() == 'u':
            n_types = 7
            #  I, IV, liq, A, B, C, III.
            old2new = {3: 0,
                       4: 1,
                       5: 2,
                       0: 3,
                       6: 4,
                       1: 5,
                       2: 6}
            # p_reindex = [3, 4, 5, 0, 6, 1, 2]

        elif key[0].lower() == 'n':
            n_types = 10
            # # ['V', 'VII', 'VIII', 'I', 'II', 'III', 'IV', 'IX', 'VI', 'Melt']
            # old2new = {3: 0,
            #            4: 1,
            #            5: 2,
            #            6: 3,
            #            0: 4,
            #            8: 5,
            #            1: 6,
            #            2: 7,
            #            7: 8,
            #            9: 9}
            #
            old2new = {0: 0,
                       1: 1,
                       2: 2,
                       3: 3,
                       4: 4,
                       5: 5,
                       6: 6,
                       7: 7,
                       8: 8,
                       9: 9}

            # p_reindex = [4, 6, 7, 0, 1, 2, 3, 8, 5, 9]

        else:
            assert False

        target_reindex = list(old2new.keys())

        d_results_dict[key] = {
            'Type_Prediction': data[:, 1:1 + n_types],
            'Targets': np.argmax(data[:, 1 + n_types:1 + 2 * n_types], axis=1),
            'Temperature': data[:, -1],
        }
        assert round(sum(d_results_dict[key]['Type_Prediction'].sum(1))) == len(d_results_dict[key]['Type_Prediction'])

        # reindex targets
        d_results_dict[key]['Targets'] = np.asarray([old2new[thing] for thing in d_results_dict[key]['Targets']])
        d_results_dict[key]['Type_Prediction'] = d_results_dict[key]['Type_Prediction'][:, target_reindex]

    return d_results_dict['UTmixedonly_w_o_inter_MK_style.dat'], d_results_dict['Nmixed_MK_style.dat']


def combined_accuracy_fig(results_dict, ordered_classes, temp_series):
    scores = {}
    melt_names = ['Crystal', 'Melt']
    defect_names = ['Bulk', 'Surface']

    fig = make_subplots(cols=2, rows=1,
                        subplot_titles=["(a) Polymorph", "(b) Topology"],
                        horizontal_spacing=0.1)
    temp_ind = 1
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

    scores[str(temp_type) + '_F1'] = f1
    scores[str(temp_type) + '_ROC_AUC'] = auc

    fig.add_trace(go.Heatmap(z=cmat, x=ordered_classes, y=ordered_classes,
                             text=np.round(cmat, 2), texttemplate="%{text:.2f}", showscale=False,
                             colorscale='blues'),
                  row=1, col=1)

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
                  row=1, col=2)

    fig.update_xaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)
    fig.update_yaxes(linewidth=1, linecolor='black', mirror=True, ticks='inside',
                     showline=True)

    scores[str(temp_type) + 'top_F1'] = f1
    scores[str(temp_type) + 'top_ROC_AUC'] = auc

    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Class", row=1, col=1)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(font=dict(size=FONTSIZE))

    return fig, scores


def urea_interface_fig(sorted_molwise_results_dict, stacked_plot=False):

    from PIL import Image
    image_paths = [r'C:\Users\mikem\crystals\classifier_runs/Interface0000.png',
                   r'C:\Users\mikem\crystals\classifier_runs/Interface0250.png',
                   r'C:\Users\mikem\crystals\classifier_runs/Interface0500.png',
                   r'C:\Users\mikem\crystals\classifier_runs/Interface0750.png',
                   r'C:\Users\mikem\crystals\classifier_runs/Interface1000.png']

    images = [Image.open(pathi) for pathi in image_paths]

    num_classes = 3
    colors = [COLORS[0], COLORS[3], OTHER_COLOR]
    ordered_class_names = ['I','IV','Other']
    sigma = min(5, len(sorted_molwise_results_dict['inside_number']) / 100)
    fig = go.Figure()
    key = 'inside_number'
    i2 = 0
    traj = sorted_molwise_results_dict[key]
    for ind in range(num_classes):
        fig.add_trace(go.Scatter(x=sorted_molwise_results_dict['time_steps'] / 1000,
                                 y=gaussian_filter1d(traj[:, ind], sigma),
                                 name=ordered_class_names[ind],
                                 legendgroup=ordered_class_names[ind],
                                 line_color=colors[ind],
                                 line_width=4,
                                 mode='lines',
                                 showlegend=True if i2 == 0 else False,
                                 stackgroup='one' if stacked_plot else None),
                      )

    fig.update_xaxes(range=[-0.1, 1.1], zeroline=False, title='Time (ps)')
    fig.update_yaxes(range=[0, 500])
    fig.update_yaxes(title='Number of Molecules')
    fig.update_xaxes(zerolinecolor='black')
    fig.update_yaxes(zerolinecolor='black')  # , gridcolor='grey')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(font=dict(size=FONTSIZE))
    fig.update_annotations(font=dict(size=FONTSIZE))

    timepoints = np.asarray([0, 250, 500, 750, 1000]) / 1e3
    for time in timepoints:
        fig.add_vline(x=time, line_dash='dash', line_color='grey')  #

    fig.add_vline(x=-0.01, line_color='black')
    ylevel = -0.25
    xlevels = np.linspace(-0.1, 0.9, len(images))
    stits = ['0', '250 ns', '500 ns', '750 ns', '1 ps']
    for ind in range(len(images)):
        fig.add_layout_image(
            dict(source=images[ind],
                 y=ylevel, x=xlevels[ind])
        )
        fig.add_annotation(y=ylevel + 0.05, x=xlevels[ind] + 0.015,
                           text=stits[ind],
                           showarrow=False,
                           xref='paper',
                           yref='paper',
                           xanchor='left',
                           yanchor='top')
    fig.update_layout_images(dict(
        xref="paper",
        yref="paper",
        sizex=.8,
        sizey=.8,
        xanchor="left",
        yanchor="top"
    ))
    fig.layout.margin.b = 425
    # fig.show()

    return fig


def nic_clusters_fig(traj_dict1, traj_dict2):
    from PIL import Image
    image_paths = [r'C:\Users\mikem\crystals\classifier_runs/stable_nic_0000.png',
                   r'C:\Users\mikem\crystals\classifier_runs/stable_nic_0100.png',
                   r'C:\Users\mikem\crystals\classifier_runs/stable_nic_0200.png',
                   r'C:\Users\mikem\crystals\classifier_runs/stable_nic_0300.png',
                   r'C:\Users\mikem\crystals\classifier_runs/melt_nic_0000.png',
                   r'C:\Users\mikem\crystals\classifier_runs/melt_nic_0005.png',
                   r'C:\Users\mikem\crystals\classifier_runs/melt_nic_0010.png',
                   r'C:\Users\mikem\crystals\classifier_runs/melt_nic_0050.png']
    images = [Image.open(pathi) for pathi in image_paths]

    def collect_unimportant_fractions(trajectory, keep_dims: list = None):
        num_keep_dims = len(keep_dims)
        toss_dims = [ind for ind in range(trajectory.shape[1]) if ind not in keep_dims]

        collected_trajectory = np.zeros((len(trajectory), num_keep_dims + 1))
        collected_trajectory[:, :num_keep_dims] = trajectory[:, keep_dims]
        collected_trajectory[:, -1] = trajectory[:, toss_dims].sum(1)

        return collected_trajectory

    stable_traj_dict = {'inside_number': collect_unimportant_fractions(traj_dict1['inside_number'], [0, 9]),
                        'outside_number': collect_unimportant_fractions(traj_dict1['outside_number'], [0, 9]), 'classes': ['I', 'Melt', 'Other'],
                        'time_steps': traj_dict1['time_steps']}

    melt_traj_dict = {'inside_number': collect_unimportant_fractions(traj_dict2['inside_number'], [0, 9]),
                      'outside_number': collect_unimportant_fractions(traj_dict2['outside_number'], [0, 9]), 'classes': ['I', 'Melt', 'Other'],
                      'time_steps': traj_dict2['time_steps']}

    colors_list = [[COLORS[0], COLORS[-1],
                    OTHER_COLOR],
                   [COLORS[0], COLORS[-1],
                    OTHER_COLOR
                    ]]

    fig = make_subplots(cols=2, rows=2,
                        subplot_titles=['(a) 100K Core',
                                        '(b) 100K Surface',
                                        '(c) 350K Core',
                                        '(d) 350K Surface',
                                        ],
                        vertical_spacing=0.5,
                        )

    for i3, traj_dict in enumerate([stable_traj_dict, melt_traj_dict]):
        for i2, key in enumerate(['inside_number', 'outside_number']):
            traj = traj_dict[key]
            for ind in range(3):
                fig.add_trace(go.Scatter(x=traj_dict['time_steps'] / 1000000,
                                         y=gaussian_filter1d(traj[:, ind], 2, mode='nearest'),
                                         name=traj_dict['classes'][ind],
                                         line_color=colors_list[i3][ind],
                                         mode='lines',
                                         line_width=2,
                                         showlegend=True if i3 == 0 and i2 == 0 else False,
                                         stackgroup='one' if False else None),
                              row=i3 + 1, col=i2 + 1)

    fig.update_xaxes(range=[-0.1, 5.1], zeroline=False)
    fig.add_vline(x=-0.01, line_color='black')
    fig.update_xaxes(range=[-.0025, .525], row=2, col=1, zeroline=True)
    fig.update_xaxes(range=[-.0025, .525], row=2, col=2, zeroline=True)
    fig.update_yaxes(rangemode='nonnegative')

    # fig.update_yaxes(range=[0, 1])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(zerolinecolor='black', showgrid=False)
    fig.update_yaxes(zerolinecolor='black', showgrid=False)
    fig.update_yaxes(title='Number of Molecules', row=1, col=1)
    fig.update_yaxes(title='Number of Molecules', row=2, col=1)
    fig.update_xaxes(title='Time (ns)', row=2, col=1)
    fig.update_xaxes(title='Time (ns)', row=2, col=2)
    fig.update_layout(font=dict(size=FONTSIZE))
    fig.update_annotations(font=dict(size=FONTSIZE))

    timepoints = np.asarray([10, 1e6, 2e6, 3e6]) / 1e6
    for time in timepoints:
        for row in [1]:
            for col in [1, 2]:
                fig.add_vline(x=time, line_dash='dash', line_color='grey', row=row, col=col)

    timepoints = np.asarray([10, 5e4, 1e5, 5e5]) / 1e6
    for time in timepoints:
        for row in [2]:
            for col in [1, 2]:
                fig.add_vline(x=time, line_dash='dash', line_color='grey', row=row, col=col)

    ylevel = .65
    xlevels = np.linspace(-0.05, 0.86, 4)
    stits = ['0 ns', '1 ns', '2 ns', '3 ns']
    for ind in range(4):
        fig.add_layout_image(
            dict(source=images[ind],
                 y=ylevel, x=xlevels[ind])
        )
        fig.add_annotation(y=ylevel - 0.05, x=xlevels[ind] - 0.05,
                           text=stits[ind],
                           showarrow=False,
                           xref='paper',
                           yref='paper',
                           xanchor='left',
                           yanchor='top')

    stits = ['0 ps', '50 ps', '100 ps', '500 ps']
    ylevel = -0.1
    for ind in range(4):
        fig.add_layout_image(
            dict(source=images[ind + 4],
                 y=ylevel, x=xlevels[ind])
        )
        fig.add_annotation(y=ylevel - 0.05, x=xlevels[ind] - 0.1,
                           text=stits[ind],
                           showarrow=False,
                           xref='paper',
                           yref='paper',
                           xanchor='left',
                           yanchor='top')

    fig.update_layout_images(dict(
        xref="paper",
        yref="paper",
        sizex=0.22,
        sizey=0.22,
        xanchor="left",
        yanchor="top"
    ))

    fig.layout.margin.b = 250
    # fig.show()
    return fig


OTHER_COLOR = 'rgb(50, 50, 50)'
FONTSIZE = 22
LEGEND_FONTSIZE = 14
COLORS = ['rgb(141,211,199)',  # NICOAM13, ureaA
          'rgb(200,200,115)',  # NICOAM14, ureaB
          'rgb(145,90,218)',  # NICOAM15, ureaC
          'rgb(251,128,114)',  # NICOAM16, ureaI / UREAXX12
          'rgb(128,177,211)',  # NICOAM07, ureaIII / UREAXX28
          'rgb(253,180,98)',  # NICOAM18, ureaIV / UREAXX26
          'rgb(179,222,105)',  # NICOAM08
          'rgb(252,205,229)',  # NICOAM09
          'rgb(217,217,217)',  # NICOAM17
          'rgb(188,35,189)']  # MELT

'''
urea I(UREAXX12)
urea III (UREAXX28)
urea IV (UREAXX26)
'''

identifier2form = {'NICOAM07': 5,
                   'NICOAM08': 7,
                   'NICOAM09': 8,
                   'NICOAM13': 1,
                   'NICOAM14': 2,
                   'NICOAM15': 3,
                   'NICOAM16': 4,
                   'NICOAM17': 9,
                   'NICOAM18': 6,
                   'NIC_Melt': 10,
                   'ureaA': 1,
                   'ureaB': 2,
                   'ureaC': 3,
                   'ureaI': 4,
                   'ureaIII': 5,
                   'ureaIV': 6,
                   'UREA_Melt': 7,
                   }


def paper_form_accuracy_fig(results_dict, ordered_classes, temp_series):
    scores = {}
    melt_names = ['Crystal', 'Melt']
    fig = go.Figure()  # make_subplots(cols=1, rows=1, subplot_titles=["Low Temperature", "High Temperature"], horizontal_spacing=0.1)
    # letts = ['a', 'b']
    for temp_ind in range(1, 2):
        # lett = letts[temp_ind]
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
                                 text=np.round(cmat, 2), texttemplate="%{text:.2f}", showscale=False,
                                 colorscale='blues'),
                      )

        # fig.layout.annotations[temp_ind].update(text=f"({lett}) {temp_type} T: ROC AUC={auc:.2f}, F1={f1:.2f}")

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
