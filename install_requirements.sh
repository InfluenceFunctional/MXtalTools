# alternate possibly faster installation method # TODO test
python -m virtualenv mxtal_pip_env

source mxtal_pip_env/bin/activate

python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==2.1.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
 # Requirements to run
python -m pip install numpy pandas tqdm scipy scikit-learn plotly numba ase #polars
# optionally install CSD #TODO add

#pip install torch torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
