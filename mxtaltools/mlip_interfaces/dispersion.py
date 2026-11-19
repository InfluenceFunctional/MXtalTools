"""Grimme D3 dispersion added to an MLIP trained without it.

For models whose reference energies carry no dispersion, e.g. MACE-CSP-25.2 (Hafizi, Gittins, Taylor and Day, 2026:
VASP PBE, dispersion to be added at use as D3 with Becke-Johnson damping). The D3 energy is computed by torch-dftd
(``pip install --no-deps torch-dftd``; an optional dependency, imported on first use) on this package's own batched
periodic neighbour list, so it is differentiable in the Cartesian positions and, through the edge shifts, in the cell.
"""
import torch

from mxtaltools.common.geometry_utils import fractional_transform
from mxtaltools.mlip_interfaces.pbc_neighbours import batched_pbc_neighbour_list


class D3Dispersion:
    """D3 dispersion energy of a batch of periodic cells or isolated molecules.

    Parameters
    ----------
    xc : str, default 'pbe'
        Functional whose D3 parameters are used (``torch_dftd.dftd3_xc_params.get_dftd3_default_params``).
    damping : str, default 'bj'
        'zero', 'bj', 'zerom' or 'bjm'.
    cutoff : float, default 15.0
        Pair cutoff, Angstrom. torch-dftd's default is 95 bohr (50.3 A); at the MACE-CSP-25.2 panel anchors,
        2026-11-19, 15 A put the D3 lattice term 0.2-1.1 kJ/mol above its 50 A value (the shift is nearly the same
        for two polymorphs of one molecule: paracetamol I and II differ by 3.59 at 50 A, 3.56 at 15 A) and 25 A
        within 0.25 kJ/mol.
    cnthr : float, default 15.0
        Coordination-number cutoff, Angstrom; clipped to `cutoff`.
    n_chunks : int or None
        Passed to torch-dftd to split the C6 computation and lower peak memory.
    """

    def __init__(self, xc='pbe', damping='bj', cutoff=15.0, cnthr=15.0, n_chunks=None):
        from torch_dftd.dftd3_xc_params import get_dftd3_default_params
        from torch_dftd.nn.dftd3_module import DFTD3Module
        self.xc, self.damping, self.cutoff = xc, damping, float(cutoff)
        self.module = DFTD3Module(get_dftd3_default_params(damping, xc), cutoff=float(cutoff),
                                  cnthr=min(float(cnthr), float(cutoff)), dtype=torch.float64, bidirectional=True,
                                  n_chunks=n_chunks)
        self._device = None

    def __call__(self, z, pos, batch_ind, T_fc, pbc=True):
        """
        Parameters
        ----------
        z : torch.Tensor
            [N] atomic numbers.
        pos : torch.Tensor
            [N] x 3 Cartesian positions (Angstrom), graphs concatenated, `batch_ind` sorted.
        batch_ind : torch.Tensor
            [N] graph index of each atom.
        T_fc : torch.Tensor
            [G, 3, 3] fractional-to-Cartesian matrices (column convention, as ``MolCrystalData.T_fc``); used for the
            periodic images when `pbc`, and only for the graph count otherwise.
        pbc : bool
            Periodic on all three axes, or isolated (no images).

        Returns
        -------
        torch.Tensor
            [G] D3 energy per graph in eV, in `pos`'s dtype, differentiable in `pos` and `T_fc`.
        """
        if self._device != pos.device:
            self.module = self.module.to(pos.device)
            self._device = pos.device
        cells_t = T_fc.transpose(-2, -1)
        e_i, e_s, e_g = batched_pbc_neighbour_list(pos.detach(), batch_ind, cells_t.detach(), self.cutoff, pbc=pbc)
        shifts = fractional_transform(e_s.to(T_fc.dtype), T_fc[e_g]) if pbc else pos.new_zeros((e_g.numel(), 3))
        # torch-dftd: r_ij = |pos_i - (pos_j + shift)|, the neighbour list's convention; `cell` sizes the per-graph
        # sum (and is read only by the three-body term, which is off)
        e = self.module.calc_energy_batch(z.long(), pos.double(), e_i, cell=cells_t.double(), shift_pos=shifts.double(),
                                          batch=batch_ind, batch_edge=e_g, damping=self.damping)
        return e.to(pos.dtype)
