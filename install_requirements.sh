# PIP installation method
# optionally install virutualenv
# python -m pip install --user virtualenv

python -m virtualenv mxtal_venv

source mxtal_venv/bin/activate

python -m pip install --upgrade pip
# Install PyTorch family - NOTE update with your pytorch/cuda version, or exclude if running on CPU
# get the correct link here https://pytorch.org/get-started/locally/
# you can get your cuda version via "nvidia-smi | grep CUDA" in the terminal, if you have an NVIDIA GPU
python -m pip install torch --extra-index-url --index-url https://download.pytorch.org/whl/cu126

# likewise make sure your versions are aligned here
# get the correct links here https://pytorch-geometric.readthedocs.io/en/2.5.2/notes/installation.html
python -m pip install torch_geometric
# these in particular are finnicky about torch & cuda version
python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# non-torch packages
python -m pip install ase kaleido matplotlib msgpack numpy pandas pillow plotly pyyaml scikit-learn tqdm umap-learn wandb lmdb


#todo add pynvml
# optionally install rdkit, CSD if you have licence / need to access / process new crystal datasets


'''alternate version on linux'''

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install torch_geometric

# Optional dependencies: SPECIFY TORCH AND CUDA VERSIONS
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html