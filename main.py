"""import statements"""
import argparse, warnings
from mxtaltools.common.config_processing import process_main_config
from mxtaltools.modeller import Modeller

warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=DeprecationWarning)  # ignore numpy error
warnings.filterwarnings("ignore", category=UserWarning)  # ignore w&b error
warnings.filterwarnings("ignore", category=FutureWarning)

# ====================================
if __name__ == '__main__':
    '''
    parse arguments from config and command line and generate config namespace
    '''
    parser = argparse.ArgumentParser()
    _, override_args = parser.parse_known_args()

    config = process_main_config(override_args)

    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))

    '''
    run the code in selected mode
    '''
    if config.sweep_id is None and config.sweep_path is not None:
        import wandb
        from mxtaltools.common.config_processing import load_yaml

        sweep_config = load_yaml(config.sweep_path)
        config.sweep_id = wandb.sweep(sweep=sweep_config,
                                      project="MXtalTools",
                                      entity='mkilgour',
                                      )

    if config.sweep_id is not None:
        import wandb
        from mxtaltools.common.config_processing import load_yaml

        sweep_config = load_yaml(config.sweep_path)
        predictor = Modeller(config, sweep_config=sweep_config)

        wandb.agent(config.sweep_id, project="MXtalTools", function=predictor.train_models, count=1)

    else:
        predictor = Modeller(config)
        if (config.mode == 'discriminator'
                or config.mode == 'gan'
                or config.mode == 'regression'
                or config.mode == 'autoencoder'
                or config.mode == 'embedding_regression'
                or config.mode == 'polymorph_classification'):
            predictor.train_models()

        elif config.mode == 'search':
            _, dataloader, _ = predictor.load_dataset_and_dataloaders(override_test_fraction=1)
            predictor.crystal_search(molecule_data=dataloader.dataset[0], data_contains_ground_truth=True)

        elif config.mode == 'embedding':
            predictor.ae_embedding_analysis()

        elif config.mode == 'mol_generation':
            predictor.autoencoder_molecule_generation()
