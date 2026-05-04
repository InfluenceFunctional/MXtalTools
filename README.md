![image](https://github.com/InfluenceFunctional/MXtalTools/assets/30198118/ecc49717-b9b4-4901-9b59-8e4c8b919813)
# MXtalTools: Toolbox for machine learning on molecular crystals

A Python library built on PyTorch and PyTorch Geometric for machine learning tasks on molecules and molecular crystals.

**Features:**
- Crystal building -- fast, differentiable molecular crystal construction from asymmetric unit parameters
- Crystal density prediction -- predict packing coefficients from molecular structure
- Molecule autoencoder -- equivariant molecular encodings via pre-trained Mo3ENet
- Crystal scoring -- evaluate crystal structures against CSD statistics
- Crystal structure search -- optimize crystal packing with ML potentials
- Dataset utilities -- build molecular/crystal datasets from CSD, .cif, and .xyz files

## Documentation

See our detailed docs at [readthedocs](https://mxtaltools.readthedocs.io/).

## Quick Start

```python
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.common.training_utils import load_molecule_scalar_regressor

# Create molecule from SMILES
mol = MolData.from_smiles("c1ccccc1", protonate=True, minimize=True, partial_charges=True)
batch = collate_data_list([mol])

# Predict crystal packing coefficient
model = load_molecule_scalar_regressor("checkpoints/cp_regressor.pt")
prediction = model(batch.clone())
```

## Installation for Users

1. Install PyTorch and PyTorch Geometric (including torch-scatter, torch-sparse, torch-cluster) for your CUDA version:
   - [PyTorch installation guide](https://pytorch.org/get-started/locally/)
   - [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

2. Install MXtalTools:

    ```bash
    pip install mxtaltools
    ```

## Installation for Developers

1. Clone the repository:

    ```bash
    git clone git@github.com:InfluenceFunctional/MXtalTools.git
    cd MXtalTools
    ```

2. Create a Python environment (pip+virtualenv recommended).

3. Install PyTorch and PyG as described above.

4. Install remaining dependencies:

    ```bash
    poetry install
    ```

5. For model training, login to Weights & Biases:

    ```bash
    wandb login
    ```

6. Create a user config in `configs/users/YOUR_USERNAME.yaml` with your paths and W&B settings. Pass `--user YOUR_USERNAME` when running.

7. (Optional) For crystal dataset construction from `.cif` files, install the [CSD Python API](https://www.ccdc.cam.ac.uk/) with a valid CCDC license.

## Reference

If you use this code in a publication, please cite:

```bibtex
@article{kilgour2023geometric,
  title={Geometric deep learning for molecular crystal structure prediction},
  author={Kilgour, Michael and Rogal, Jutta and Tuckerman, Mark},
  journal={Journal of Chemical Theory and Computation},
  volume={19},
  number={14},
  pages={4743--4756},
  year={2023},
  publisher={American Chemical Society}
}
```
