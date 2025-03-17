#!/usr/bin/env python3
"""
Script: single_point.py
Description:
  This script performs a single-point calculation on an input XYZ file.
  It reads configuration options from a YAML file—including cell parameters,
  periodic boundary conditions, and MACE calculator settings.
  The script computes the energy and forces for the structure and saves
  the output structure to an XYZ file.
Usage:
  python single_point.py <input_file.xyz> config_file="config.yaml"
"""

import os

import numpy as np

from mxtaltools.conformer_generation.conformer_generator import embed_mol, extract_mol_info
from mxtaltools.mace_sp.utils import sp_calculation

# Set PyTorch CUDA options if needed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------------------------------------------------------------
# Main Single-Point Calculation Routine
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     raise ValueError("Usage: python single_point.py <input_file.xyz> config_file=\"config.yaml\"")
    #
    # input_file = os.path.abspath(sys.argv[1])
    device = 'cuda'
    smiles = r"C=C=CCOC(=O)C(CC)(CC)CC"
    mol = embed_mol(smiles, True)
    conf = mol.GetConformer()
    coords, atom_types = extract_mol_info(mol, conf,
                                          do_adjacency_analysis=False)

    energies = []
    for scale in np.linspace(0.5, 2, 10):
        energy = sp_calculation(coords,
                                atom_types,
                                np.array([10, 10, 10]) * scale,
                                np.array([90, 90, 90]),
                                device=device
                                )
        energies.append(energy)

    aa = 1
