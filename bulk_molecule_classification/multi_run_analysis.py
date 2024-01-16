import os
import pandas as pd
import numpy as np
import warnings

# from reporting.cluster_figs import cluster_property_heatmap, collate_property_over_multiple_runs, plot_classifier_pies
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from bulk_molecule_classification.classifier_constants import nic_ordered_class_names, identifier2form
from bulk_molecule_classification.traj_analysis_figs import plot_classifier_pies, cluster_property_heatmap
from common.utils import delete_from_dataframe

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error

os.chdir(r'D:\crystals_extra\classifier_training\results')

files = os.listdir()
results_dicts = [file for file in files if 'analysis' in file and 'crystal_in_melt_test8' in file]

dfs = []
for ind, dict_path in enumerate(results_dicts):
    results_dict = np.load(dict_path, allow_pickle=True).item()
    ldict = {key: [value] for key, value in results_dict.items()}
    for key, value in results_dict['run_config'].items():
        ldict[key] = [value]

    for key, value in results_dict['eval_config'].items():
        ldict[key] = [value]

    dfs.append(pd.DataFrame().from_dict(ldict))

results_df = pd.concat(dfs)
results_df.reset_index(drop=True, inplace=True)
del (dfs)

"""crystal melts analysis"""
# we want to know
# for runs that completed
# initial equil form fraction
# post liquification form fraction
# final form fraction

# filter unfinished runs
run_lengths = np.asarray([ts[-1] for ts in results_df['time_steps']])
bad_inds = np.argwhere(run_lengths < run_lengths.max())[:, 0]
results_df = delete_from_dataframe(results_df, bad_inds)
results_df = results_df.reset_index().drop(columns='index')
assert all(np.array([len(traj) for traj in results_df['time_steps']]) == len(results_df['time_steps'].iloc[0]))

# get the timesteps for the relevant shifts
equil_time = results_df['equilibration_time'].iloc[0]
timesteps = results_df['time_steps'].iloc[0]


# want the closest time indices before the end of these phases
def get_closest_prior_time(time, timesteps):
    diffs = time - timesteps
    prior_times = timesteps[diffs >= 0]
    return np.argmin(time - prior_times)


init_equil_time = equil_time * 1
init_equil_time_ind = get_closest_prior_time(init_equil_time, timesteps)
melt_finished_time = equil_time * 3  # finish melt
melt_finished_time_ind = get_closest_prior_time(melt_finished_time, timesteps)
final_time = timesteps[-1]
final_time_ind = len(timesteps) - 1

init_stability = np.zeros(len(results_df))
melt_stability = np.zeros_like(init_stability)
final_stability = np.zeros_like(melt_stability)
stability_change = np.zeros_like(final_stability)

for ind in range(len(results_df)):
    identifier = results_df['structure_identifier'][ind]
    form = identifier2form[identifier]
    form_ind = form - 1
    init_stability[ind] = results_df['inside_fraction'].iloc[ind][init_equil_time_ind, form_ind]
    melt_stability[ind] = results_df['inside_fraction'].iloc[ind][melt_finished_time_ind, form_ind]
    final_stability[ind] = results_df['inside_fraction'].iloc[ind][final_time_ind, form_ind]
    stability_change[ind] = melt_stability[ind] - final_stability[ind]

results_df['init_stability'] = init_stability
results_df['melt_stability'] = melt_stability
results_df['final_stability'] = final_stability
results_df['stability_change'] = stability_change

figs = []
for property in ['init_stability', 'melt_stability', 'final_stability', 'stability_change']:
    figs.append(cluster_property_heatmap(results_df,
                                         property=property,
                                         xaxis_title='max_sphere_radius',
                                         yaxis_title='temperature',
                                         ))

aa = 0

from scipy.ndimage import gaussian_filter1d
from _plotly_utils.colors import n_colors

colors = n_colors('rgb(5,120,200)', 'rgb(250,50,5)', len(np.unique(results_df['temperature'])), colortype='rgb')
temps = np.sort(np.unique(results_df['temperature']))
temp_dict = {temp: ind for ind, temp in enumerate(temps)}

tstart = get_closest_prior_time(equil_time * 3, timesteps) + 1
sigma=5

fig = make_subplots(cols=2, rows=2, subplot_titles=['Raw Values', 'Normed against t=0'], horizontal_spacing=0.05, vertical_spacing=0.075)
for temp in temps:
    for ind in range(len(results_df)):
        if results_df['temperature'].iloc[ind] == temp:
            x = results_df['time_steps'].iloc[ind][tstart:]
            y = results_df['inside_number'].iloc[ind][tstart:, form_ind]
            fig.add_scatter(x=x, y=gaussian_filter1d(y, sigma=sigma),
                            name=str(results_df['temperature'].iloc[ind]),
                            legendgroup=str(results_df['temperature'].iloc[ind]),
                            line_color=colors[temp_dict[results_df['temperature'].iloc[ind]]],
                            showlegend=ind % 2 == 0, row=1, col=1)
for temp in temps:
    for ind in range(len(results_df)):
        if results_df['temperature'].iloc[ind] == temp:
            x = results_df['time_steps'].iloc[ind][tstart:]
            y = results_df['inside_number'].iloc[ind][tstart:, form_ind] / results_df['inside_number'].iloc[ind][tstart, form_ind]
            fig.add_scatter(x=x, y=gaussian_filter1d(y, sigma=sigma),
                            name=str(results_df['temperature'].iloc[ind]),
                            legendgroup=str(results_df['temperature'].iloc[ind]),
                            line_color=colors[temp_dict[results_df['temperature'].iloc[ind]]],
                            showlegend=False, row=1, col=2)


tstart = 0
sigma=5

for temp in temps:
    for ind in range(len(results_df)):
        if results_df['temperature'].iloc[ind] == temp:
            x = results_df['time_steps'].iloc[ind][tstart:]
            y = results_df['inside_number'].iloc[ind][tstart:, form_ind]
            fig.add_scatter(x=x, y=gaussian_filter1d(y, sigma=sigma),
                            name=str(results_df['temperature'].iloc[ind]),
                            legendgroup=str(results_df['temperature'].iloc[ind]),
                            line_color=colors[temp_dict[results_df['temperature'].iloc[ind]]],
                            showlegend=False, row=2, col=1)
for temp in temps:
    for ind in range(len(results_df)):
        if results_df['temperature'].iloc[ind] == temp:
            x = results_df['time_steps'].iloc[ind][tstart:]
            y = results_df['inside_number'].iloc[ind][tstart:, form_ind] / results_df['inside_number'].iloc[ind][tstart, form_ind]
            fig.add_scatter(x=x, y=gaussian_filter1d(y, sigma=sigma),
                            name=str(results_df['temperature'].iloc[ind]),
                            legendgroup=str(results_df['temperature'].iloc[ind]),
                            line_color=colors[temp_dict[results_df['temperature'].iloc[ind]]],
                            showlegend=False, row=2, col=2)

fig.update_xaxes(title='Time (ns)')
fig.update_yaxes(title='Num Molecules')

fig.show()

#
#
# """NN output analysis"""
# classes = nic_ordered_class_names
# NNout_means = np.zeros((len(results_df), 10))
# for i in range(len(results_df)):
#     for j in range(len(classes)):
#         # trailing frames (pre-smoothed)
#         NNout_means[i, j] = np.mean(results_df['overall_fraction'][i][-10:, j])
#
# for i, label in enumerate(nic_ordered_class_names):
#     results_df[label] = NNout_means[:, i]
#
# """classifier outputs"""
# plot_classifier_pies(results_df, 'max_sphere_radius', 'temperature', nic_ordered_class_names)
#
# """NN output analysis"""
# classes = nic_ordered_class_names
# NNout_means = np.zeros((len(results_df), 10))
# for i in range(len(results_df)):
#     for j in range(len(classes)):
#         # trailing frames (pre-smoothed)
#         NNout_means[i, j] = np.mean(results_df['outside_fraction'][i][-10:, j])
#
# for i, label in enumerate(nic_ordered_class_names):
#     results_df[label] = NNout_means[:, i]
#
# """classifier outputs"""
# plot_classifier_pies(results_df, 'max_sphere_radius', 'temperature', nic_ordered_class_names)
#
# """NN output analysis"""
# classes = nic_ordered_class_names
# NNout_means = np.zeros((len(results_df), 10))
# for i in range(len(results_df)):
#     for j in range(len(classes)):
#         # trailing frames (pre-smoothed)
#         NNout_means[i, j] = np.mean(results_df['inside_fraction'][i][-10:, j])
#
# for i, label in enumerate(nic_ordered_class_names):
#     results_df[label] = NNout_means[:, i]
#
# """classifier outputs"""
# plot_classifier_pies(results_df, 'max_sphere_radius', 'temperature', nic_ordered_class_names)
#
# aa = 1
