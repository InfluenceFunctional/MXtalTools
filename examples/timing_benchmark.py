import plotly.graph_objects as go
from torch.cuda import synchronize
from time import time
from mxtaltools.dataset_utils.utils import collate_data_list
import torch
import numpy as np


def run_task(task_name, device, bsz, samples):
    batch = collate_data_list(samples[:bsz], max_z_prime = 1).to(device)

    if task_name == 'build_ucell':
        t0 = time()

        batch.pose_aunit(std_orientation=True)
        batch.build_unit_cell()

        return time() - t0

    elif task_name == 'build_cluster':
        batch.pose_aunit(std_orientation=True)
        batch.build_unit_cell()

        t0 = time()

        batch.build_cluster(cutoff=10, supercell_size=5)

        return time() - t0

    elif task_name == 'energy':
        cluster = batch.mol2cluster(cutoff=10, supercell_size=5)

        t0 = time()
        cluster.construct_radial_graph()
        cluster.compute(['lj'])

        return time() - t0

    elif task_name == 'opt':
        t0 = time()

        batch.optimize_crystal_parameters(
            mol_orientation='standardized',
            max_num_steps=100,
            optimizer_func=torch.optim.Rprop,
            show_tqdm=True,
            convergence_eps=1e-12,
            optim_target='lj',
            do_box_restriction=False,
            cutoff=10,
            compression_factor=0,
            enforce_reduced=False,
        )

        return time() - t0

samples = torch.load(r"D:\crystal_datasets\test_new_new_csd.pt", weights_only=False)
samples = [elem for elem in samples if elem.z_prime == 1]

# init cuda
batch = collate_data_list(samples[:10], max_z_prime=1).to('cuda')
batch.pose_aunit(std_orientation=True)
batch.build_unit_cell()
del batch

batch_sizes = [10, 100, 1000]
devices = ['cpu','cuda']
n_repeats = 10
tasks = ['build_ucell', 'build_cluster', 'energy', 'opt']
times = {}

for device in devices:
    for rep in range(n_repeats):
        for task in tasks:
            for bsz in batch_sizes:
                synchronize()
                if rep == 0:
                    times[(device, task, bsz)] = [run_task(task, device, bsz, samples)]
                else:
                    if task != 'opt':
                        times[(device, task, bsz)].append(run_task(task, device, bsz, samples))
                synchronize()


task_labels = {
    'build_ucell': 'Unit cell build (ms)',
    'build_cluster': 'Cluster build (ms)',
    'energy': 'Neighbor list + LJ (ms)',
    'opt': '100 Local Opt Steps (s)'
}

def fmt(vals):
    mean = np.mean(vals)
    std = np.std(vals)
    if mean < 2:
        return f"{mean*1000:.0f}"# ± {std:.2f}"
    else:
        return f"{mean:.0f}"

rows = []

for device in devices:
    for bsz in batch_sizes:
        row = {
            'Device': device.upper(),
            'Batch size': bsz,
        }
        for task in tasks:
            key = (device, task, bsz)
            if key in times:
                row[task_labels[task]] = fmt(times[key])
            else:
                row[task_labels[task]] = '—'
        rows.append(row)

headers = ['Device', 'Batch size'] + list(task_labels.values())

columns = {h: [] for h in headers}
for r in rows:
    for h in headers:
        columns[h].append(r[h])

fig = go.Figure(
    data=[go.Table(
        header=dict(
            values=headers,
            fill_color='lightgrey',
            align='center',
            font=dict(size=20),
            height=50
        ),
        cells=dict(
            values=[columns[h] for h in headers],
            align='center',
            font=dict(size=20),
            height=35,
        )
    )]
)

fig.update_layout(
    # width=1100,
    # height=420,
    margin=dict(l=20, r=20, t=20, b=20)
)

fig.show()

aa = 0

