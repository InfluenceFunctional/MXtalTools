import os
from pathlib import Path
import argparse
import numpy as np
import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

"""
Alternate version of this script, for generating just structures of urea
"""

if __name__ == '__main__':
    device = 'cuda'
    #Create the parser
    parser = argparse.ArgumentParser(description='Process an integer.')

    # Add an argument for the integer
    parser.add_argument('chunk_ind', type=int,
                        help='An integer passed from the command line', default=0)

    # Parse the arguments
    args = parser.parse_args()
    chunk_ind = args.chunk_ind
    #chunk_ind = 999
    mode = 'train'
    # initialize
    space_group = 2
    batch_size = 10
    #chunks_path = os.getcwd()
    chunks_path = Path(r'/scratch/mk8347/csd_runs/datasets/qm9_crystals/')

    #qm9_mols = torch.load(r'D:/crystal_datasets/test_csd_free_qm9_dataset.pt', weights_only=False)
    qm9_mols = torch.load(r'/scratch/mk8347/csd_runs/datasets/test_csd_free_qm9_dataset.pt', weights_only=False)

    rng = np.random.RandomState(0)
    rands = rng.choice(len(qm9_mols), len(qm9_mols), replace=False)
    bp = int(len(rands) * 0.8)

    train_mol_list = [qm9_mols[ind] for ind in rands[:bp]]
    test_mol_list = [qm9_mols[ind] for ind in rands[bp:]]
    if mode == 'train':
        mol_list = train_mol_list
        del test_mol_list
    elif mode == 'test':
        mol_list = test_mol_list
        del train_mol_list
    else:
        assert False

    # select some random molecules
    rng = np.random.Generator(np.random.PCG64(int(space_group * chunk_ind * 200)))
    mol_inds = rng.choice(len(mol_list), size=batch_size, replace=False)

    crystal_batch = collate_data_list([MolCrystalData(
        molecule=mol_list[ind].clone(),
        sg_ind=space_group,
        aunit_handedness=torch.ones(1),
        cell_lengths=torch.ones(3, device=device),
        # if we don't put dummies in here, later ops to_data_list fail
        # but if we do put dummies in here, it does box analysis one-by-one which is super slow
        cell_angles=torch.ones(3, device=device),
        aunit_centroid=torch.ones(3, device=device),
        aunit_orientation=torch.ones(3, device=device),
        skip_box_analysis=True,
    ) for ind in mol_inds]).to(device)

    chunks_path = Path(chunks_path)  # where to save outputs
    os.chdir(chunks_path)
    chunk_path = os.path.join(chunks_path, f'qm9_sg_{space_group}_chunk_{chunk_ind}.pkl')

    crystal_batch.sample_reasonable_random_parameters(
        target_packing_coeff=0.5,
        tolerance=5,
        max_attempts=100,
        sample_niggli=True,
    )

    opt1_trajectory = (
        crystal_batch.optimize_crystal_parameters(
            optim_target='silu',
            show_tqdm=True,
            convergence_eps=1e-3,
            compression_factor=0.1,
            do_box_restriction=True,
            enforce_niggli=True,
            cutoff=6,
        ))
    crystal_batch = collate_data_list(opt1_trajectory[-1]).to(device)
    crystal_batch.box_analysis()
    crystal_batch.to('cpu')
    torch.save(crystal_batch.to_data_list(), chunk_path)

    if False:  # visualize
        crystal_batch.plot_batch_cell_params()
        import plotly.graph_objects as go

        lj_pots = torch.stack(
            [torch.tensor([sample.scaled_lj_pot for sample in sample_list]) for sample_list in opt1_trajectory])
        coeffs = torch.stack(
            [torch.tensor([sample.packing_coeff for sample in sample_list]) for sample_list in opt1_trajectory])

        fig = go.Figure()
        fig.add_scatter(x=coeffs[0, :], y=lj_pots[0, :], mode='markers', marker_size=20, marker_color='grey',
                        name='Initial State')
        fig.add_scatter(x=coeffs[-1, :], y=lj_pots[-1, :], mode='markers', marker_size=20, marker_color='black',
                        name='Final State')
        for ind in range(coeffs.shape[1]):
            fig.add_scatter(x=coeffs[:, ind], y=lj_pots[:, ind], name=f"Run {ind}")
        fig.update_layout(xaxis_title='Packing Coeff', yaxis_title='Scaled LJ')
        fig.show()

        fig = go.Figure()
        for ind in range(lj_pots.shape[-1]):
            fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
                            showlegend=True if ind == 0 else False)
        fig.show()

        fig = go.Figure()
        for ind in range(lj_pots.shape[-1]):
            fig.add_scatter(y=coeffs[..., ind], marker_color='blue', name='packing coeff', legendgroup='lg',
                            showlegend=True if ind == 0 else False)
        fig.update_layout(yaxis_range=[0, 1])
        fig.show()

    aa = 0
