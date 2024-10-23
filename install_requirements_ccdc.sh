# PIP installation method ON WINDOWS
python -m virtualenv --python=python3.9 windows_mxtal_venv

.\windows_mxtal_venv\scripts\activate

python -m pip install --upgrade pip
# Install PyTorch family - NOTE update with your pytorch/cuda version, or exclude if running on CPU
python -m pip install numpy scipy pandas scikit-learn tqdm
python -m pip install torch==2.4.0  --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch_geometric
# these in particular are finnicky about torch & cuda version
python -m pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# non-torch packages
python -m pip install ase kaleido matplotlib msgpack pillow plotly pyyaml umap-learn wandb lmdb

# optionally install rdkit, CSD if you have licence / need to access / process new crystal datasets
python -m pip install rdkit
python -m pip install --extra-index-url https://pip.ccdc.cam.ac.uk/ csd-python-api
