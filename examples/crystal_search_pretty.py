import os
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm

from examples.crystal_search_reporting import batch_compack, density_funnel, compack_fig
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance

# add MXtalTools to path by relative reference
sys.path.insert(0, os.path.abspath("../"))

import torch.nn.functional as F
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import softmax_and_score

torch.set_grad_enabled(False)

device = 'cuda'
dafmuv_path = "D:\crystal_datasets\DAFMUV.pt"
mini_dataset_path = '../mini_datasets/mini_CSD_dataset.pt'
score_checkpoint = r"../checkpoints/crystal_score.pt"
density_checkpoint = r"../checkpoints/cp_regressor.pt"
opt_path = r"D:\crystal_datasets\opt_outputs\DAFMUV.pt"

"""
Load crystal score model
"""
score_model = load_crystal_score_model(score_checkpoint, device).to(device)
score_model.eval()

"""
Load and analyze reference crystal 
"""
dafmuv_data = torch.load(dafmuv_path, weights_only=False)
ref_crystal_batch = collate_data_list(dafmuv_data).to(device)
ref_computes, ref_cluster_batch = ref_crystal_batch.analyze(
    computes=['lj'], return_cluster=True, cutoff=10, supercell_size=10)
model_output_ref = score_model(ref_cluster_batch.to(device), force_edges_rebuild=True).cpu()
ref_score = softmax_and_score(model_output_ref[:, :2]).cpu()
ref_pred_rdfemd = F.softplus(model_output_ref[:, 2]).cpu()
rdf, bin_edges, _ = ref_cluster_batch.compute_rdf()
ref_rdf = rdf.cpu()

ref_lj_energy = ref_computes['lj'].cpu()
ref_cp = ref_cluster_batch.packing_coeff.cpu()

"""
Run optimization via standalone script & config
"""
subprocess.run(["python", "run_search.py", "--input", "dafmuv_example.yaml"], check=True)

"""
Analyze optimized samples
"""
opt_sample_list = torch.load(opt_path, weights_only=False)
batch_size = 25
num_batches = len(opt_sample_list) // batch_size + int((len(opt_sample_list) % batch_size) > 0)
opt_score, opt_pred_rdfemd, opt_rdfs, opt_lj_energy, opt_cp = [], [], [], [], []
for batch_idx in tqdm(range(num_batches)):
    opt_crystal_batch = collate_data_list(opt_sample_list[batch_size * batch_idx:batch_size * (1 + batch_idx)]).to(
        device)
    computes, opt_cluster_batch = opt_crystal_batch.analyze(
        computes=['lj'], return_cluster=True, cutoff=10, supercell_size=10
    )

    model_output = score_model(opt_cluster_batch.to(device), force_edges_rebuild=True).cpu()
    opt_score.append(softmax_and_score(model_output[:, :2]).cpu())
    opt_pred_rdfemd.append(F.softplus(model_output[:, 2]).cpu())
    rdf, bin_edges, _ = opt_cluster_batch.compute_rdf()
    opt_rdfs.append(rdf.cpu())
    opt_lj_energy.append(computes['lj'].cpu())
    opt_cp.append(opt_crystal_batch.packing_coeff.cpu())

opt_score = torch.cat(opt_score)
opt_pred_rdfemd = torch.cat(opt_pred_rdfemd)
opt_rdfs = torch.cat(opt_rdfs)
opt_lj_energy = torch.cat(opt_lj_energy)
opt_cp = torch.cat(opt_cp)

"""
Compute true RDF distances
"""
rdf_dists = torch.zeros(len(opt_rdfs), device=opt_rdfs.device, dtype=torch.float32)
for i in range(len(opt_rdfs)):
    rdf_dists[i] = compute_rdf_distance(ref_rdf[0], opt_rdfs[i], bin_edges.to(opt_rdfs.device)) / \
                   ref_cluster_batch.num_atoms[0]
rdf_dists = rdf_dists.cpu()

"""
COMPACK analysis
"""
best_sample_inds = torch.argwhere((opt_cp > 0.6) * (opt_cp < 0.8)).squeeze()
matches, rmsds = batch_compack(best_sample_inds, opt_sample_list, ref_crystal_batch)

all_matched = np.argwhere(matches == 20).flatten()
matched_rmsds = rmsds[all_matched]

"""
Figures
"""
good_inds = torch.argwhere((opt_pred_rdfemd < 0.015) * (opt_lj_energy < -350)).flatten()
density_funnel(opt_pred_rdfemd[good_inds],
               opt_cp[good_inds],
               rdf_dists[good_inds],
               ref_pred_rdfemd,
               ref_cp,
               yaxis_title='Predicted Distance',
               write_fig=True)
density_funnel(opt_lj_energy[good_inds],
               opt_cp[good_inds],
               rdf_dists[good_inds],
               ref_lj_energy,
               ref_cp,
               yaxis_title='LJ Energy (Arb Units)',
               write_fig=True)

compack_fig(matches, rmsds, write_fig=True)
