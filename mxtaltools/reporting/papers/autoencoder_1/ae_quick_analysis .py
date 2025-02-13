"""
refresh model results for Autoencoder paper
"""
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader.dataloader import Collater
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_scatter import scatter

from mxtaltools.common.config_processing import load_yaml, process_main_config
from mxtaltools.common.geometry_utils import scatter_compute_Ip
from mxtaltools.modeller import Modeller
from mxtaltools.reporting.ae_reporting import gaussian_3d_overlap_plots

if __name__ == '__main__':
    max_dataset_length = 10000
    single_run_analysis = True
    multi_run_comparison = False

    model_path = r'C:\Users\mikem\crystals\CSP_runs\models\cluster/best_autoencoder_experiments_autoencoder_tests_otf_zinc_test3_7_05-12-14-03-45'
    results_path = model_path[:-3] + '_results.npy'

    if single_run_analysis:

        if True: #not os.path.exists(results_path):
            mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'

            os.chdir(mxt_path)

            config = load_yaml('configs/analyses/ae_analysis.yaml')
            config['model_paths']['autoencoder'] = model_path
            config['dataset']['filter_protons'] = False
            config['autoencoder']['filter_protons'] = False
            config['autoencoder']['infer_protons'] = False
            config['dataset']['max_dataset_length'] = max_dataset_length
            config['dataset_name'] = 'test_qm9_dataset.pt'
            config['positional_noise']['autoencoder'] = 0.01

            with open('configs/analyses/analysis.yaml', 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

            override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
            config = process_main_config(override_args, append_model_paths=False)

            predictor = Modeller(config)
            predictor.ae_analysis(save_results=True)

        collater = Collater(0, 0)
        stats = np.load(results_path, allow_pickle=True).item()
        stats_dict = stats['train_stats']

        ''' eyeball the best and worst samples '''
        overlaps = np.concatenate(stats_dict['evaluation_overlap'])
        sample_ranking = np.argsort(overlaps)

        mol_batch = collater(stats_dict['sample'])
        decoded_mol_batch = collater(stats_dict['decoded_sample'])
        num_to_show = 5
        for ind in range(num_to_show):
            fig = (
                gaussian_3d_overlap_plots(mol_batch, decoded_mol_batch,
                                          [1, 6, 7, 8, 9],
                                          graph_ind=sample_ranking[ind]
                                          ))
            fig.update_layout(title=mol_batch.smiles[sample_ranking[ind]])
            print(mol_batch.smiles[sample_ranking[ind]])
            fig.show(renderer='browser')
            fig = (
                gaussian_3d_overlap_plots(mol_batch, decoded_mol_batch,
                                          [1, 6, 7, 8, 9],
                                          graph_ind=sample_ranking[-ind - 1]
                                          ))
            fig.update_layout(title=mol_batch.smiles[sample_ranking[-ind-1]])
            print(mol_batch.smiles[sample_ranking[-ind-1]])
            fig.show(renderer='browser')

        fig = go.Figure(go.Histogram(x=overlaps, nbinsx=100)).show()

        ''' high-symmetry samples appear worst, let's confirm this statistically '''
        Ip, Ipm, I = scatter_compute_Ip(mol_batch.pos, mol_batch.batch)
        symmetric_factor = (torch.diff(Ipm, dim=1).sum(1) / Ipm.sum(1)[:]).cpu()

        properties_dict = {
            'asymmetry': symmetric_factor.numpy(),
            'num_atoms': mol_batch.num_atoms.cpu().numpy(),
            'radius': mol_batch.radius.cpu().numpy(),
            'volume': mol_batch.mol_volume.cpu().numpy(),
            'density': scatter(mol_batch.x.float(), mol_batch.batch, dim=0, dim_size=mol_batch.num_graphs,
                               reduce='mean').cpu().numpy(),
            'fluorine_content': scatter((mol_batch.x == 9).float(), mol_batch.batch, dim=0, dim_size=mol_batch.num_graphs,
                                        reduce='mean').cpu().numpy(),
            'oxygen_content': scatter((mol_batch.x == 8).float(), mol_batch.batch, dim=0, dim_size=mol_batch.num_graphs,
                                      reduce='mean').cpu().numpy(),
            'nitrogen_content': scatter((mol_batch.x == 7).float(), mol_batch.batch, dim=0, dim_size=mol_batch.num_graphs,
                                        reduce='mean').cpu().numpy(),
            'carbon_content': scatter((mol_batch.x == 6).float(), mol_batch.batch, dim=0, dim_size=mol_batch.num_graphs,
                                      reduce='mean').cpu().numpy(),
        }

        corr_dict = {key: np.corrcoef(val, overlaps)[0, 1].astype(np.float16) for (key, val) in properties_dict.items()}
        fig = go.Figure(go.Bar(x=list(corr_dict.keys()), y=list(corr_dict.values()))).show()

        aa = 1

    #
    # if multi_run_comparison:
    #     '''compare noise levels'''
    #
    #     noises = np.linspace(0, 0.2, 20)
    #     overlaps = []
    #     for ind, noise in enumerate(noises):
    #         mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'
    #         os.chdir(mxt_path)
    #         config = load_yaml('configs/analyses/ae_analysis.yaml')
    #         config['model_paths']['autoencoder'] = model_path
    #         config['dataset']['filter_protons'] = False
    #         config['autoencoder']['filter_protons'] = False
    #         config['autoencoder']['infer_protons'] = False
    #         config['dataset']['max_dataset_length'] = max_dataset_length
    #         config['dataset_name'] = 'test_qm9_dataset.pt'
    #         config['positional_noise']['autoencoder'] = float(noise)
    #
    #         with open('configs/analyses/analysis.yaml', 'w') as outfile:
    #             yaml.dump(config, outfile, default_flow_style=False)
    #
    #         override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
    #         config = process_main_config(override_args, append_model_paths=False)
    #
    #         predictor = Modeller(config)
    #         predictor.ae_analysis(save_results=True)
    #
    #         collater = Collater(0, 0)
    #         stats = np.load(results_path, allow_pickle=True).item()
    #         stats_dict = stats['train_stats']
    #
    #         ''' eyeball the best and worst samples '''
    #         overlaps.append(np.concatenate(stats_dict['evaluation_overlap']))
    #
    #     fig = go.Figure()
    #     for ind, ov in enumerate(overlaps):
    #         fig.add_trace(
    #             go.Violin(x=ov, name=noises[ind], side='positive', orientation='h', width=4, meanline_visible=True,
    #                       bandwidth=0.005, points=False), )
    #     fig.show()
    aa = 1
