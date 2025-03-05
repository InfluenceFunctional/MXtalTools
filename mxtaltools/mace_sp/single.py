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
from ase.geometry.cell import Cell
import torch

# Set PyTorch CUDA options if needed
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def load_config(config_filename):
    with open(config_filename, "r") as f:
        return yaml.safe_load(f)


def parse_xyz_header(filename):
    """
    Parses the header of an XYZ file to extract cell parameters.
    Assumes the second line of the file contains cell information in the format:
      "Cell: a b c alpha beta gamma"
    """
    with open(filename, 'r') as f:
        _ = f.readline()  # number of atoms
        comment_line = f.readline().strip()
    cell_match = re.search(
        r'Cell:\s*([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+'
        r'([\d\.\-eE]+)\s+([\d\.\-eE]+)\s+([\d\.\-eE]+)', comment_line)
    if cell_match:
        a = float(cell_match.group(1))
        b = float(cell_match.group(2))
        c = float(cell_match.group(3))
        alpha = float(cell_match.group(4))
        beta = float(cell_match.group(5))
        gamma = float(cell_match.group(6))
        return [a, b, c, alpha, beta, gamma]
    else:
        raise ValueError("Cell parameters not found in the XYZ header.")


# ----------------------------------------------------------------------
# Main Single-Point Calculation Routine
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python single_point.py <input_file.xyz> config_file=\"config.yaml\"")

    input_file = os.path.abspath(sys.argv[1])
    config_filename = sys.argv[2].split("=")[-1].strip('"')
    config = load_config(config_filename)

    # Determine output directory (default: current working directory)
    output_dir = os.path.abspath(config.get("output_dir", "."))
    os.makedirs(output_dir, exist_ok=True)

    # Determine cell parameters: either read from file or use config value.
    if config.get("readcell_info", True):
        cell_params = parse_xyz_header(input_file)
        print(f"Cell parameters from file: {cell_params}")
    else:
        cell_params = config.get("cell_params")
        if cell_params is None:
            raise ValueError("readcell_info is False, but no cell_params provided in the config file.")
        print(f"Cell parameters from config: {cell_params}")

    # Read input structure and set cell and PBC
    atoms = read(input_file, index=0)
    atoms.set_cell(Cell.fromcellpar(cell_params))
    pbc_flag = config.get("pbc", True)
    atoms.set_pbc(pbc_flag)
    print("Using periodic boundary conditions (PBC):", pbc_flag)
    print("Cell used for simulation:", atoms.get_cell())

    # Initialize the MACE Calculator based on config options
    from mace.calculators import MACECalculator

    device_option = config.get("device", "gpu").lower()
    if device_option == "cpu":
        device_str = "cpu"
    else:
        gpu_devices = config.get("gpus", ["cuda:0"])
        device_str = ",".join(gpu_devices)

    model_paths = config.get("model_paths", ["/path/to/your/model"])
    mace_calc = MACECalculator(model_paths=model_paths, device=device_str)
    atoms.calc = mace_calc

    # Perform the single-point calculation
    print("\nPerforming single-point calculation...")
    sp_energy = atoms.get_potential_energy()
    sp_forces = atoms.get_forces()

    # Save output structure
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    sp_file = os.path.join(output_dir, f"{base_name}_sp.xyz")
    write(sp_file, atoms)

    # Print the results
    print(f"Single-point calculation results:")
    print(f"  Energy: {sp_energy:.3f} eV")
    print(f"  Forces:\n{sp_forces}")
    print(f"Structure saved as {sp_file}")

    # Clear CUDA cache (if using GPU)
    torch.cuda.empty_cache()
