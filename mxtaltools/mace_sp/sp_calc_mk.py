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

import os, sys, re, yaml
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.geometry.cell import Cell
import torch
from mace.calculators import MACECalculator

from mxtaltools.conformer_generation.conformer_generator import embed_mol, extract_mol_info

# Set PyTorch CUDA options if needed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def load_config(config_filename):
    with open(config_filename, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Main Single-Point Calculation Routine
# ----------------------------------------------------------------------

def sp_calculation(pos: np.ndarray,
                   z: np.ndarray,
                   cell_lengths: np.ndarray,
                   cell_angles: np.ndarray,
                   device: str):
    cell_params = np.concatenate([cell_lengths, cell_angles])
    atoms = Atoms(
        numbers=z,
        positions=pos,
    )
    atoms.set_cell(Cell.fromcellpar(cell_params))
    pbc_flag = True
    atoms.set_pbc(pbc_flag)

    if device == "cpu":
        device_str = "cpu"
    else:
        gpu_devices = ["cuda:0"]
        device_str = ",".join(gpu_devices)

    model_paths = "mace-mpa-0-medium.model"  #config.get("model_paths", ["/path/to/your/model"])
    mace_calc = MACECalculator(model_paths=model_paths, device=device_str)
    atoms.calc = mace_calc

    # Perform the single-point calculation
    return atoms.get_potential_energy()


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
