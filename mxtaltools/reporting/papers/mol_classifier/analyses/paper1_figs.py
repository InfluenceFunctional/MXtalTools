import numpy as np
import os

from mxtaltools.constants.classifier_constants import (nic_ordered_class_names, urea_ordered_class_names,
                                                       )
from mxtaltools.reporting.papers.mol_classifier.analyses import (process_daisuke_dats, urea_interface_fig, nic_clusters_fig, combined_accuracy_fig,
                                                                 paper_form_accuracy_fig)

max_tsne_samples = 100

os.chdir(r'D:\crystals_extra\classifier_training\results')
urea_eval_path = 'dev_urea_evaluation_results_feb6_dict.npy'
#nic_eval_path = 'dev_nic_evaluation_results_march13_seed2_dict.npy'
nic_eval_path = 'dev_nic_evaluation_results_march13_seed0_dict.npy'
urea_interface_path = 'crystals_extra_classifier_training_urea_melt_interface_T200_analysis.npy'
# nic_traj_path1 = 'paper_nic_clusters2_1__analysis.npy' # 15 Angstrom radius
# nic_traj_path2 = 'paper_nic_clusters2_7__analysis.npy'
nic_traj_path1 = 'paper_nic_clusters2_2__analysis.npy' # 17 Angstrom radius
nic_traj_path2 = 'paper_nic_clusters2_8__analysis.npy'
d_nic_tsne_path1 = 'daisuke_nic_tsne1.npy'  # raw input
d_nic_tsne_path2 = 'daisuke_nic_tsne2.npy'  # scaled
d_nic_tsne_path3 = 'daisuke_nic_tsne3.npy'  # embedding
d_urea_tsne_path1 = '../daisuke_results/input_no_factor.pkl'  # raw input
d_urea_tsne_path2 = '../daisuke_results/input_w_factor.pkl'  # scaled
d_urea_tsne_path3 = '../daisuke_results/before_out.pkl'  # embedding

fig_dict = {}
scores_dict = {}
'''
urea form confusion matrix
'''
results_dict = np.load(urea_eval_path, allow_pickle=True).item()

fig_dict['urea_cmat'], scores_dict['urea'] = combined_accuracy_fig(
    results_dict, urea_ordered_class_names, [100, 200, 350])
#
# '''
# urea tSNE
# '''
# d_urea_embed_dict2 = np.load(d_urea_tsne_path2, allow_pickle=True)
# d_urea_embed_dict3 = np.load(d_urea_tsne_path3, allow_pickle=True)
# old2new = {3: 0,  # reindex urea targets
#            4: 1,
#            5: 2,
#            0: 3,
#            6: 4,
#            1: 5,
#            2: 6}
# d_urea_embed_dict2['Targets'] = np.asarray([old2new[tgt] for tgt in d_urea_embed_dict2['Targets']])
# d_urea_embed_dict3['Targets'] = np.asarray([old2new[tgt] for tgt in d_urea_embed_dict3['Targets']])
# d_urea_embed_dict2['Embeddings'] = d_urea_embed_dict2['Latents']
#
# fig_dict['urea_tSNE'] = combined_embedding_fig(
#     results_dict, d_urea_embed_dict2, d_urea_embed_dict3, urea_ordered_class_names,
#     max_samples=max_tsne_samples, perplexity=30, molecule_name='urea'
# )
del results_dict

'''
nic form confusion matrix
'''
results_dict = np.load(nic_eval_path, allow_pickle=True).item()

fig_dict['nic_cmat'], scores_dict['nic'] = combined_accuracy_fig(
    results_dict, nic_ordered_class_names, [100, 350])

# '''
# nic tSNE
# '''
# d_nic_embed_dict2 = np.load(d_nic_tsne_path2, allow_pickle=True).item()
# d_nic_embed_dict3 = np.load(d_nic_tsne_path3, allow_pickle=True).item()
# d_nic_embed_dict2['Embeddings'] = d_nic_embed_dict2['Latents']
#
# fig_dict['nicotinamide_tSNE'] = combined_embedding_fig(
#     results_dict, d_nic_embed_dict2, d_nic_embed_dict3, nic_ordered_class_names,
#     max_samples=max_tsne_samples, perplexity=30, molecule_name='nicotinamide'
# )

del results_dict

'''
daisuke's cmats
'''
urea_results, nic_results = process_daisuke_dats()

fig_dict['d_urea_form_cmat'], scores_dict['d_urea_form'] = paper_form_accuracy_fig(
    urea_results, urea_ordered_class_names, [100, 200])  # it only does > 100

fig_dict['d_nic_form_cmat'], scores_dict['d_nic_form'] = paper_form_accuracy_fig(
    nic_results, nic_ordered_class_names, [100, 350])  # it only does > 100

# '''
# urea interface trajectory
# '''
traj_dict = np.load(urea_interface_path, allow_pickle=True).item()
fig_dict['urea_interface_traj'] = urea_interface_fig(traj_dict)
del traj_dict

'''
nic cluster trajectory (stable & melt)
'''
traj_dict1 = np.load(nic_traj_path1, allow_pickle=True).item()
traj_dict2 = np.load(nic_traj_path2, allow_pickle=True).item()

fig_dict['nic_trajectories'] = nic_clusters_fig(traj_dict1, traj_dict2)

for key, fig in fig_dict.items():
    if 'cmat' in key:
        if key[0] != 'd':
            fig.write_image(key + '.png', width=1920 // 2 + 100, height=1080 // 2)
        else:
            fig.write_image(key + '.png', width=1920 // 4, height=1080 // 2)
    elif 'tSNE' in key and 'urea' in key:
        fig.write_image(key + '.png', width=1920, height=720)
    elif 'tSNE' in key and 'nic' in key:
        fig.write_image(key + '.png', width=1920, height=770)
    elif 'traj' in key:
        fig.write_image(key + '.png', width=960, height=900)
    else:
        fig.write_image(key + '.png', scale=4)
# for key, fig in fig_dict.items():
#     fig.show(renderer='browser')

for item in scores_dict.items():
    print(item)

aa = 1
