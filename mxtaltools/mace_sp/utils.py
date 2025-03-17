import numpy as np
from ase import Atoms
from ase.cell import Cell
from mace.calculators import MACECalculator
import os
from pathlib import Path


class SPMaceCalculator:
    def __init__(self, device):

        if device == "cpu":
            device_str = "cpu"
        else:
            gpu_devices = ["cuda:0"]
            device_str = ",".join(gpu_devices)

        dir_path = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = dir_path.joinpath("mace-mpa-0-medium.model")  # config.get("model_paths", ["/path/to/your/model"])
        self.mace_calc = MACECalculator(model_paths=str(model_path), device=device_str)

    def sp_calculation(self,
                       pos: np.ndarray,
                       z: np.ndarray,
                       cell_lengths: np.ndarray,
                       cell_angles: np.ndarray,
                       ):
        if np.any(cell_angles < 10):  # automatically detect and convert to degrees
            cell_angles *= 90 / (np.pi / 2)

        cell_params = np.concatenate([cell_lengths, cell_angles])
        atoms = Atoms(
            numbers=z,
            positions=pos,
        )
        atoms.set_cell(Cell.fromcellpar(cell_params))
        pbc_flag = True
        atoms.set_pbc(pbc_flag)
        atoms.calc = self.mace_calc

        # Perform the single-point calculation
        return atoms.get_potential_energy()
