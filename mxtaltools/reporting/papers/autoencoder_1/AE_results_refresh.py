"""
refresh model results for Autoencoder paper
"""
import os
import yaml
from model_paths import ae_paths

from mxtaltools.common.config_processing import load_yaml, process_main_config
from mxtaltools.modeller import Modeller

if __name__ == '__main__':
    filter_protons = [False, True]
    infer_protons = [False, False]
    max_dataset_length = 1000000000
    #
    mxt_path = r'C:\\Users\\mikem\\PycharmProjects\\Python_Codes\\MXtalTools'

    for path, filter, infer in zip(ae_paths, filter_protons, infer_protons):
        os.chdir(mxt_path)

        config = load_yaml('configs/analyses/ae_analysis.yaml')
        config['model_paths']['autoencoder'] = str(path)
        config['dataset']['filter_protons'] = filter
        config['autoencoder']['filter_protons'] = filter
        config['autoencoder']['infer_protons'] = infer
        config['dataset']['max_dataset_length'] = max_dataset_length
        config['dataset']['test_fraction'] = 1

        with open('configs/analyses/analysis.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
        config = process_main_config(override_args, append_model_paths=False)

        predictor = Modeller(config)
        predictor.ae_analysis()
    #
    # '''separately, full QM9 dataset'''
    # path = ae_paths[0]
    # filter = False
    # infer = False
    # os.chdir(mxt_path)
    #
    # config = load_yaml('configs/analyses/ae_analysis.yaml')
    # config['model_paths']['autoencoder'] = str(path)
    # config['dataset']['filter_protons'] = filter
    # config['autoencoder']['filter_protons'] = filter
    # config['autoencoder']['infer_protons'] = infer
    # config['dataset']['max_dataset_length'] = max_dataset_length
    # config['dataset']['test_fraction'] = 1
    # config['dataset_name'] = 'qm9_dataset.pt'
    #
    # with open('configs/analyses/analysis.yaml', 'w') as outfile:
    #     yaml.dump(config, outfile, default_flow_style=False)
    #
    # override_args = ['--user', 'mkilgour', '--yaml_config', mxt_path + r'/configs/analyses/analysis.yaml']
    # config = process_main_config(override_args, append_model_paths=False)
    #
    # predictor = Modeller(config)
    # predictor.ae_analysis(path_prepend='full_')


    # generate figures
    from mxtaltools.reporting.papers.autoencoder_1.AE_figures import RMSD_fig, UMAP_fig

    fig = RMSD_fig()
    #fig2 = UMAP_fig(max_entries=1000000)
    if False:
        fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\RMSD.png', width=1920, height=840)
        #fig2.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\latent_space.png', width=1920, height=840)

    aa = 1


