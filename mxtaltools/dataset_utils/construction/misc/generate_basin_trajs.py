import argparse
import os
from pathlib import Path
import numpy as np
import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list

"""
Alternate version of this script, for generating just structures of urea
"""

if __name__ == '__main__':
    space_group = 2
    batch_size = 500
    num_of_test_mol = 1
    try:
        device = 'cpu'
        #Create the parser
        parser = argparse.ArgumentParser(description='Process an integer.')

        # Add an argument for the integer
        parser.add_argument('chunk_ind', type=int,
                            help='An integer passed from the command line', default=0)
        # Parse the arguments
        args = parser.parse_args()
        chunk_ind = args.chunk_ind
        mode = 'test'
        chunks_path = Path(r'/scratch/mk8347/csd_runs/datasets/qm9_crystals/')
        qm9_mols = torch.load(r'/scratch/mk8347/csd_runs/datasets/csd_free_qm9_dataset.pt', weights_only=False)

    except:
        chunk_ind = 999
        device = 'cuda'
        mode = 'test'
        chunks_path = os.getcwd()
        qm9_mols = torch.load(r'D:/crystal_datasets/test_csd_free_qm9_dataset.pt', weights_only=False)

    rng = np.random.RandomState(0)
    rands = rng.choice(len(qm9_mols), len(qm9_mols), replace=False)
    bp = int(len(rands) * 0.8)

    if mode == 'train':
        mol_list = [qm9_mols[ind] for ind in rands[:bp]]

    elif mode == 'test':
        mol_list = [qm9_mols[ind] for ind in rands[:bp]][num_of_test_mol]

    else:
        assert False

    rng = np.random.Generator(np.random.PCG64(int(space_group * chunk_ind * 200)))
    mol_inds = rng.choice(len(mol_list), size=batch_size, replace=True if len(mol_list) < batch_size else False)

    chunks_path = Path(chunks_path)  # where to save outputs
    os.chdir(chunks_path)
    chunk_path = os.path.join(chunks_path, f'mol_{int(mol_inds[0])}_qm9_sg_{space_group}_chunk_{chunk_ind}.pkl')

    # select some random molecules
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

    crystal_batch.sample_random_reduced_crystal_parameters(
        target_packing_coeff=0.5,
    )

    opt1_trajectory = (
        crystal_batch.optimize_crystal_parameters(
            optim_target='silu',
            show_tqdm=True,
            convergence_eps=1e-1,
            compression_factor=0.1,
            max_num_steps=300,
            do_box_restriction=True,
            enforce_niggli=True,
            cutoff=6,

        ))

    torch.save(opt1_trajectory, chunk_path)
