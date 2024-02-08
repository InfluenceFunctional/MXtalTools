import wandb
from common.config_processing import load_yaml

sweep_configuration = load_yaml('C:/Users/mikem/OneDrive/NYU/CSD/MCryGAN/configs/autoencoder_tests/sweeps/qm9_sweep1.yaml')

sweep_id = wandb.sweep(sweep=sweep_configuration,
                       project="MXtalTools",
                       entity='mkilgour',
                       )



wandb.agent(sweep_id, function=main, count=1)
