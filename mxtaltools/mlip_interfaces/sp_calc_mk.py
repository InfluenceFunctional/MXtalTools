#!/usr/bin/env python3
"""
Ad-hoc driver for `SPMaceCalculator` on one embedded molecule.

Run as a script, it embeds a hard-coded SMILES with hydrogens (`embed_mol`),
takes its coordinates and atomic numbers (`extract_mol_info`), builds an
`SPMaceCalculator` on ``'cuda'``, and calls `lattice_energy_calculation` with
that molecule as both the unit-cell contents (one molecule per cell) and the
gas-phase reference, in a cubic cell (all angles 90 degrees) whose edge length
takes 10 evenly spaced values from 5 to 20. The returned
(gas-phase energy, lattice energy) tuples are collected in a local list. It takes
no command-line arguments and writes no files; the only file it reads is the
model file loaded by `SPMaceCalculator`.

Importing the module sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` in
the process environment.
"""

import os

import numpy as np

from mxtaltools.dataset_utils.mol_building import extract_mol_info, embed_mol
from mxtaltools.mlip_interfaces.mace_utils import SPMaceCalculator

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

    calculator = SPMaceCalculator(device)
    energies = []
    for scale in np.linspace(0.5, 2, 10):
        energy = calculator.lattice_energy_calculation(
            coords,
            atom_types,
            coords,
            atom_types,
            np.array([10, 10, 10]) * scale,
            np.array([90, 90, 90]),
            1
            )
        energies.append(energy)

    aa = 1
