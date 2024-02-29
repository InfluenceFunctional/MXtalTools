import wandb
from mxtaltools.common.config_processing import load_yaml

sweep_configuration = load_yaml('/configs/autoencoder_tests/qm9_sweep1/qm9_sweep1.yaml')

sweep_id = wandb.sweep(sweep=sweep_configuration,
                       project="MXtalTools",
                       entity='mkilgour',
                       )



wandb.agent(sweep_id, function=main, count=1)
