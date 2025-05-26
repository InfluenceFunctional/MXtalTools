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
    # # Create the parser
    # parser = argparse.ArgumentParser(description='Process an integer.')
    #
    # # Add an argument for the integer
    # parser.add_argument('chunk_ind', type=int, help='An integer passed from the command line')
    #
    # # Parse the arguments
    # args = parser.parse_args()
    # chunk_ind = args.chunk_ind
    chunk_ind = 0

    # initialize
    space_group = 2
    num_chunks = 1
    batch_size = 10
    chunks_path = os.getcwd()  #Path(r'/scratch/mk8347/csd_runs/datasets')

    atom_coords = torch.tensor([
        [-1.3042, - 0.0008, 0.0001],
        [0.6903, - 1.1479, 0.0001],
        [0.6888, 1.1489, 0.0001],
        [- 0.0749, - 0.0001, - 0.0003],
    ], dtype=torch.float32, device=device)
    atom_coords -= atom_coords.mean(0)
    atom_types = torch.tensor([8, 7, 7, 6], dtype=torch.long, device=device)

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
    chunk_path = os.path.join(chunks_path, f'urea_sg_{space_group}_chunk_{chunk_ind}.pkl')

    crystal_batch.sample_reasonable_random_parameters(
        target_packing_coeff=0.5,  # diffuse target
        tolerance=5,
        max_attempts=500,
        sample_niggli=True,
    )

    #target_packing_coeff = (torch.randn(crystal_batch.num_graphs, device=device) * 0.0447 + 0.6226).clip(min=0.45, max=0.95)

    opt1_trajectory = (
        crystal_batch.optimize_crystal_parameters(
            optim_target='silu',
            show_tqdm=True,
            convergence_eps=1e-6,
            # score_model=score_model,
            # target_packing_coeff=target_packing_coeff,
            compression_factor=0.1,
            do_box_restriction=True,
            enforce_niggli=True,
            cutoff=6,
        ))
    crystal_batch = collate_data_list(opt1_trajectory[-1]).to(device)
