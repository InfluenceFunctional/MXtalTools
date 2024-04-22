# PIP installation method
python -m virtualenv mxtal_pip_env

source mxtal_pip_env/bin/activate

python -m pip install --upgrade pip
# Install PyTorch family - NOTE update with your cuda version, or exclude if running on CPU
python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install torch_geometric
python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# non-torch packages
python -m pip install ase kaleido matplotlib msgpack numpy pandas pillow plotly pyyaml scikit-learn tqdm umap-learn wandb

# optionally install CSD if you have licence / need to access / process new crystal datasets
