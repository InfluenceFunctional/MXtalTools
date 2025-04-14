import os
from pathlib import Path

head_path = Path(r'C:\Users\mikem\crystals\CSP_runs\models\ae_draft2_models_and_artifacts')
#--------------------------------------
ae_path = head_path.joinpath('autoencoder')
ae_paths = ['with_protons.pt',
            'without_protons.pt',
            #'inferred_protons.pt'
            ]
ae_results_paths = ['with_protons_results.npy',
                    #'without_protons_results.npy',
                    'with_protonsfull__results.npy'
                    ]
ae_paths = [ae_path.joinpath(path) for path in ae_paths]
ae_results_paths = [ae_path.joinpath(path) for path in ae_results_paths]

#---------------------------------------
er_path = head_path.joinpath('embedding_regressor')
er_ae_path = er_path.joinpath('autoencoder.pt')
er_paths = os.listdir(er_path)
er_paths.remove('autoencoder.pt')
er_paths.remove('results')
import numpy as np

sort_inds = np.argsort([int(path.split('_')[9]) for path in er_paths])
er_paths = [er_paths[ind] for ind in sort_inds]
er_paths = [er_path.joinpath(path) for path in er_paths]

targets = [
    ["rotational_constant_a", 'scalar', 1, 'qm9_dataset.pt'],  # 0
    ["rotational_constant_b", 'scalar', 1, 'qm9_dataset.pt'],  # 1
    ["rotational_constant_c", 'scalar', 1, 'qm9_dataset.pt'],  # 2
    ["dipole_moment", 'scalar', 1, 'qm9_dataset.pt'],  # 3
    ["isotropic_polarizability", 'scalar', 1, 'qm9_dataset.pt'],  # 4
    ["HOMO_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 5
    ["LUMO_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 6
    ["gap_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 7
    ["el_spatial_extent", 'scalar', 1, 'qm9_dataset.pt'],  # 8
    ["zpv_energy", 'scalar', 1, 'qm9_dataset.pt'],  # 9
    ["internal_energy_0", 'scalar', 1, 'qm9_dataset.pt'],  # 10
    ["internal_energy_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 11
    ["enthalpy_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 12
    ["free_energy_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 13
    ["heat_capacity_STP", 'scalar', 1, 'qm9_dataset.pt'],  # 14
    ["dipole", 'vector', 1, 'qm9s_dataset.pt'],  # 15
    ["polar", '2-tensor', 64, 'qm9s_dataset.pt'],  # 16
    ["quadrupole", '2-tensor', 64, 'qm9s_dataset.pt'],  # 17
    ["octapole", '3-tensor', 64, 'qm9s_dataset.pt'],  # 18
    ["hyperpolar", '3-tensor', 64, 'qm9s_dataset.pt'],  # 19
]
er_results_path = er_path.joinpath('results')
er_results_paths = os.listdir(er_results_path)
sort_inds = np.argsort([int(path.split('_')[9]) for path in er_results_paths])
er_results_paths = [er_results_paths[ind] for ind in sort_inds]
er_results_paths = [er_results_path.joinpath(path) for path in er_results_paths]

#------------------------------
proxy_path = head_path.joinpath('proxy_discriminator')
proxy_ae_path = proxy_path.joinpath('autoencoder.pt')
proxy_model_path = os.listdir(proxy_path)
proxy_model_path.remove('autoencoder.pt')
proxy_model_path.remove('results')
proxy_model_path = [proxy_path.joinpath(path) for path in proxy_model_path]
proxy_results_paths = [proxy_path.joinpath('results').joinpath(elem) for elem in os.listdir(proxy_path.joinpath('results')) if '.npy' in elem]
