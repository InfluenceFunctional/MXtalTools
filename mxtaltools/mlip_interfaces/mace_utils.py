"""
Single-point MACE lattice energies through the ASE calculator interface.

`SPMaceCalculator` wraps one ``mace.calculators.MACECalculator`` and evaluates one
crystal and one isolated molecule per call from host numpy arrays. It is separate
from the batched on-device MACE route in `AL_mace_utils`.
"""
import numpy as np
from ase import Atoms
from ase.cell import Cell
from mace.calculators import MACECalculator
import os
from pathlib import Path


class SPMaceCalculator:
    """
    ASE-driven single-point MACE calculator for per-structure lattice energies.

    Parameters
    ----------
    device : str
        ``"cpu"`` runs the calculator on CPU; any other value (including other
        CUDA device strings) selects ``"cuda:0"``.

    Attributes
    ----------
    mace_calc : mace.calculators.MACECalculator
        Calculator loaded from the file ``mace-mpa-0-medium.model`` in the
        directory containing this module; that file must be present there.
    """
    def __init__(self, device):

        if device == "cpu":
            device_str = "cpu"
        else:
            gpu_devices = ["cuda:0"]
            device_str = ",".join(gpu_devices)

        dir_path = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = dir_path.joinpath("mace-mpa-0-medium.model")  # config.get("model_paths", ["/path/to/your/model"])
        self.mace_calc = MACECalculator(model_paths=str(model_path), device=device_str)

    def lattice_energy_calculation(self,
                                   cell_pos: np.ndarray,
                                   cell_z: np.ndarray,
                                   mol_pos: np.ndarray,
                                   mol_z: np.ndarray,
                                   cell_lengths: np.ndarray,
                                   cell_angles: np.ndarray,
                                   mols_per_cell: int
                                   ):
        """
        Crystal and gas-phase single-point energies for one structure.

        Builds an ASE ``Atoms`` for the crystal (periodic on all three axes, cell
        from ``Cell.fromcellpar`` on the concatenated lengths and angles) and one
        for the isolated molecule (non-periodic), and evaluates each with
        ``self.mace_calc``. Forces are not requested.

        Parameters
        ----------
        cell_pos : np.ndarray
            [num_cell_atoms, 3] Cartesian positions of every atom in the unit cell.
        cell_z : np.ndarray
            [num_cell_atoms] atomic numbers matching `cell_pos`.
        mol_pos : np.ndarray
            [num_mol_atoms, 3] Cartesian positions of the isolated molecule.
        mol_z : np.ndarray
            [num_mol_atoms] atomic numbers matching `mol_pos`.
        cell_lengths : np.ndarray
            [3] cell lengths a, b, c.
        cell_angles : np.ndarray
            [3] cell angles alpha, beta, gamma in degrees. If any entry is below
            10, the whole array is treated as radians and multiplied by
            180 / pi IN PLACE, so the caller's array is modified.
        mols_per_cell : int
            Number of molecules in the unit cell; the crystal energy is divided
            by it.

        Returns
        -------
        gas_phase_pot : float
            Potential energy of the isolated molecule.
        lattice_pot : float
            ``crystal_energy / mols_per_cell - gas_phase_pot``.

        Notes
        -----
        Inputs are handed to ASE unchanged and both energies come from ASE's
        ``Atoms.get_potential_energy``, so positions and lengths are in ASE's
        length unit (Angstrom) and energies in ASE's energy unit (eV).
        """
        if np.any(cell_angles < 10):  # automatically detect and convert to degrees
            cell_angles *= 90 / (np.pi / 2)

        # crystal
        cell_params = np.concatenate([cell_lengths, cell_angles])
        atoms = Atoms(
            numbers=cell_z,
            positions=cell_pos,
        )
        atoms.set_cell(Cell.fromcellpar(cell_params))
        pbc_flag = True
        atoms.set_pbc(pbc_flag)
        atoms.calc = self.mace_calc

        # Perform the single-point gas phase calculation
        total_pot = atoms.get_potential_energy()

        # gas phase molecule
        atoms = Atoms(
            numbers=mol_z,
            positions=mol_pos,
        )
        pbc_flag = False
        atoms.set_pbc(pbc_flag)
        atoms.calc = self.mace_calc

        # Perform the single-point calculation
        gas_phase_pot = atoms.get_potential_energy()

        return gas_phase_pot, total_pot/mols_per_cell - gas_phase_pot
