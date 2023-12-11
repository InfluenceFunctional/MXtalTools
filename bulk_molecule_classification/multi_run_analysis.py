import os
import pandas as pd
import numpy as np
import warnings

# from reporting.cluster_figs import cluster_property_heatmap, collate_property_over_multiple_runs, plot_classifier_pies
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from bulk_molecule_classification.classifier_constants import nic_ordered_class_names
from bulk_molecule_classification.traj_analysis_figs import plot_classifier_pies

warnings.filterwarnings("ignore", category=FutureWarning)  # ignore numpy error

os.chdir(r'C:\Users\mikem\crystals\classifier_runs')

files = os.listdir()
results_dicts = [file for file in files if 'analysis' in file]

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

"""NN output analysis"""
classes = nic_ordered_class_names
NNout_means = np.zeros((len(results_df), 10))
for i in range(len(results_df)):
    for j in range(len(classes)):
        # trailing frames (pre-smoothed)
        NNout_means[i, j] = np.mean(results_df['overall_fraction'][i][-10:, j])

for i, label in enumerate(nic_ordered_class_names):
    results_df[label] = NNout_means[:, i]

"""classifier outputs"""
plot_classifier_pies(results_df, 'max_sphere_radius', 'temperature', nic_ordered_class_names)

"""NN output analysis"""
classes = nic_ordered_class_names
NNout_means = np.zeros((len(results_df), 10))
for i in range(len(results_df)):
    for j in range(len(classes)):
        # trailing frames (pre-smoothed)
        NNout_means[i, j] = np.mean(results_df['outside_fraction'][i][-10:, j])

for i, label in enumerate(nic_ordered_class_names):
    results_df[label] = NNout_means[:, i]

"""classifier outputs"""
plot_classifier_pies(results_df, 'max_sphere_radius', 'temperature', nic_ordered_class_names)

"""NN output analysis"""
classes = nic_ordered_class_names
NNout_means = np.zeros((len(results_df), 10))
for i in range(len(results_df)):
    for j in range(len(classes)):
        # trailing frames (pre-smoothed)
        NNout_means[i, j] = np.mean(results_df['inside_fraction'][i][-10:, j])

for i, label in enumerate(nic_ordered_class_names):
    results_df[label] = NNout_means[:, i]

"""classifier outputs"""
plot_classifier_pies(results_df, 'max_sphere_radius', 'temperature', nic_ordered_class_names)

aa = 1
