import os
from pathlib import Path
import argparse
import torch
from mxtaltools.dataset_utils.construction.parallel_synthesis import generate_smiles_dataset, \
    process_smiles_to_crystal_opt
from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

"""
Alternate version of this script, for generating just structures of urea
"""

if __name__ == '__main__':
    device = 'cpu'
    #Create the parser
    parser = argparse.ArgumentParser(description='Process an integer.')

    # Add an argument for the integer
    parser.add_argument('chunk_ind', type=int,
                        help='An integer passed from the command line', default=0)

    # Parse the arguments
    args = parser.parse_args()
    chunk_ind = args.chunk_ind

    # initialize
    space_group = 2
    batch_size = 200
    chunks_path = Path(r'/scratch/mk8347/csd_runs/datasets')
    # # NICOTINAMIDE
    atom_coords = torch.tensor([
        [-2.3940, 1.1116, -0.0088],
        [1.7614, -1.2284, -0.0034],
        [-2.4052, -1.1814, 0.0027],
        [-0.2969, 0.0397, 0.0024],
        [0.4261, 1.2273, 0.0039],
        [0.4117, -1.1510, -0.0013],
        [1.8161, 1.1886, 0.0018],
        [-1.7494, 0.0472, 0.0045],
        [2.4302, -0.0535, -0.0018]
    ], dtype=torch.float32, device=device)
    atom_coords -= atom_coords.mean(dim=0)
    atom_types = torch.tensor([8, 7, 7, 6, 6, 6, 6, 6, 6], dtype=torch.long, device=device)

    mol = MolData(
        z=atom_types,
        pos=atom_coords,
        x=atom_types,
        skip_mol_analysis=False,
    )

    crystal_batch = collate_data_list([MolCrystalData(
        molecule=mol.clone(),
        sg_ind=space_group,
        aunit_handedness=torch.ones(1),
        cell_lengths=torch.ones(3, device=device),
        # if we don't put dummies in here, later ops to_data_list fail
        # but if we do put dummies in here, it does box analysis one-by-one which is super slow
        cell_angles=torch.ones(3, device=device),
        aunit_centroid=torch.ones(3, device=device),
        aunit_orientation=torch.ones(3, device=device),
        skip_box_analysis=True,
    ) for _ in range(batch_size)]).to(device)

    chunks_path = Path(chunks_path)  # where to save outputs
    os.chdir(chunks_path)
    chunk_path = os.path.join(chunks_path, f'nic_sg_{space_group}_chunk_{chunk_ind}.pkl')

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
