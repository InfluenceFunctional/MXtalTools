import os
from pathlib import Path
import argparse

from mxtaltools.dataset_utils.construction.parallel_synthesis import generate_smiles_dataset, \
    process_smiles_to_crystal_opt

if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description='Process an integer.')

    # Add an argument for the integer
    parser.add_argument('chunk_ind', type=int, help='An integer passed from the command line')

    # Parse the arguments
    args = parser.parse_args()
    chunk_ind = args.chunk_ind

    # initialize
    space_group = 1
    num_smiles = 100
    num_chunks = 1
    smiles_path = "/scratch/mk8347/zinc22"
    chunks_path = Path(r'/scratch/mk8347/csd_runs/datasets')

    chunks_path = Path(chunks_path)  # where to save outputs
    smiles_path = Path(smiles_path)  # where to get inputs
    os.chdir(smiles_path)
    chunks = generate_smiles_dataset(num_smiles, num_chunks, smiles_path, seed=chunk_ind)  # get batch of smiles and chunkify
    os.chdir(chunks_path)
    chunk_path = os.path.join(chunks_path, f'sg_{space_group}_chunk_{chunk_ind}.pkl')

    conf_kwargs = {
        'max_num_atoms': 30,
        'max_num_heavy_atoms': 9,
        'pare_to_size': 9,
        'max_radius': 15,
        'protonate': True,
        'allow_methyl_rotations': True,
        'compute_partial_charges': True,
    }

    process_smiles_to_crystal_opt(chunks[0],
                                  chunk_path,
                                  [1, 6, 7, 8, 9],
                                  space_group,
                                  10,
                                  False,
                                  'principal_axes',
                                  None,
                                  True,
                                  **conf_kwargs)


