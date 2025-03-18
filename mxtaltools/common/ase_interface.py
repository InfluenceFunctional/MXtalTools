import numpy as np
import torch
from ase import Atoms
from ase.spacegroup import crystal as ase_crystal
from ase.visualize import view
from torch_scatter import scatter

from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_building.utils import find_coord_in_box_torch
from mxtaltools.crystal_building.utils import fractional_transform

def data_batch_to_ase_mols_list(crystaldata_batch,
                                max_ind: int = np.inf,
                                specific_inds=None,
                                show_mols: bool = False, **kwargs):
    """
    Helper function for converting Crystaldata batches into lists of ase objects.

    Parameters
    ----------
    crystaldata_batch :
        batch of Crystaldata objects
    max_ind : int
        Maximum batch ind to include in the list
    show_mols : bool
        whether to visualize the list of mol objects

    Returns
    -------
    list of ase mol objects
    """
    if hasattr(crystaldata_batch, 'num_graphs'):
        num_graphs = crystaldata_batch.num_graphs
    else:
        num_graphs = 1
    if specific_inds is None:
        mols = [ase_mol_from_crystaldata(crystaldata_batch, ii, **kwargs)
                for ii in range(min(max_ind, num_graphs))]
    else:
        mols = [ase_mol_from_crystaldata(crystaldata_batch, ii, **kwargs)
                for ii in specific_inds]
    if show_mols:
        view(mols)
    return mols


def ase_mol_from_crystaldata(crystal_batch,
                             index: int = None,
                             highlight_canonical_conformer: bool = False,
                             mode=None,
                             cutoff: float = 4,
                             return_crystal: bool = False):
    """
    Extract an atomic structure from a Crystaldata object according to its batch index, and convert it into an ase mol object.
    Several options for visualization of crystals.

    Parameters
    ----------
    crystal_batch : Crystaldata object
    index : int, optional
        Batch index of Crystaldata object to extract. Default is None.
    highlight_canonical_conformer : bool, optional
        Whether to give the canonical conformer a different atom type for color comparison.
    mode : 'conformer', 'unit cell', 'inside cell', 'convolve with', 'distance', or None (default None)
        Assuming the input Crystaldata is a molecule cluster larger than a single unit cell, we have several options to pare down to the desired visualization.
        'conformer' : only the 'canonical conformer'
        'unit cell' : all molecules with centroids inside the unit cell
        'inside cell' : all atoms inside the unit cell
        'convolve with' : all atoms within convolution range of the canonical conformer
        'distance' : all atoms within a certain distance of the canonical conformer
        None : show all atoms in the crystal
    cutoff : float, optional
        Distance fed to the 'distance' exclusion level option
    return_crystal : bool, optional
        Whether to return an ase mol for the crystal. Does not always work properly. Ase does not understand our crystals correctly.


    Returns
    -------
    ase mol object
    """
    crystal_batch = crystal_batch.clone().cpu().detach()
    if crystal_batch.batch is not None:  # more than one crystal in the datafile
        atom_inds = torch.where(crystal_batch.batch == index)[0]
    else:
        atom_inds = torch.arange(len(crystal_batch.z))

    if mode == 'conformer':  # only the canonical conformer itself
        inside_inds = torch.where(crystal_batch.aux_ind == 0)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = crystal_batch.pos[atom_inds].cpu().detach().numpy()

    elif mode == 'unit cell':
        # assume that by construction the first Z molecules are the ones in the unit cell
        mol_size = crystal_batch.num_atoms[index]
        num_molecules = int((crystal_batch.ptr[index + 1] - crystal_batch.ptr[index]) / mol_size)

        molecule_centroids = scatter(crystal_batch.pos[atom_inds],
                                     torch.arange(num_molecules).repeat_interleave(mol_size), dim=0,
                                     dim_size=num_molecules, reduce='mean')

        centroids_fractional = fractional_transform(molecule_centroids, crystal_batch.T_cf[index])
        inside_centroids_inds = find_coord_in_box_torch(centroids_fractional, [1.0, 1.0, 1.0])

        inside_inds = torch.cat(
            [torch.arange(mol_size) + mol_size * inside_centroids_inds[ind]
             for ind in range(len(inside_centroids_inds))]
        ).long()

        inside_inds += crystal_batch.ptr[index]
        atom_inds = inside_inds
        coords = crystal_batch.pos[inside_inds].cpu().detach().numpy()

    elif mode == 'inside cell':
        fractional_coords = torch.inner(torch.linalg.inv(crystal_batch.T_fc[index]), crystal_batch.pos[crystal_batch.batch == index]).T
        inside_coords = torch.prod((fractional_coords < 1) * (fractional_coords > 0), dim=-1)
        inside_inds = torch.where(inside_coords)[0]
        inside_inds += crystal_batch.ptr[index]
        atom_inds = inside_inds
        coords = crystal_batch.pos[inside_inds].cpu().detach().numpy()

    elif mode == 'convolve with':  # atoms potentially in the convolutional field
        inside_inds = torch.where(crystal_batch.aux_ind < 2)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = crystal_batch.pos[atom_inds].cpu().detach().numpy()

    elif mode == 'distance':  # atoms within a certain distance of the conformer radius
        crystal_coords = crystal_batch.pos[atom_inds]
        crystal_inds = crystal_batch.aux_ind[atom_inds]

        canonical_conformer_inds = torch.where(crystal_inds == 0)[0]
        mol_centroid = crystal_coords[canonical_conformer_inds].mean(0)
        mol_radius = torch.max(torch.cdist(mol_centroid[None], crystal_coords[canonical_conformer_inds], p=2))
        in_range_inds = \
            torch.where((torch.cdist(mol_centroid[None], crystal_coords, p=2)
                         < (mol_radius + cutoff))[0])[0]
        atom_inds = atom_inds[in_range_inds]
        coords = crystal_coords[in_range_inds].cpu().detach().numpy()
    else:
        coords = crystal_batch.pos[atom_inds].cpu().detach().numpy()

    if highlight_canonical_conformer:  # highlight the atom aux index
        numbers = crystal_batch.aux_ind[atom_inds].cpu().detach().numpy() + 6
    else:
        numbers = crystal_batch.z[atom_inds].cpu().detach().numpy()

    if hasattr(crystal_batch, "T_fc"):
        if index is not None:
            try:
                cell = crystal_batch.T_fc[index].T.cpu().detach().numpy()
            except IndexError:
                cell = crystal_batch.T_fc[0].T.cpu().detach().numpy()

        else:
            cell = crystal_batch.T_fc[0].T.cpu().detach().numpy()

        mol = Atoms(symbols=numbers, positions=coords, cell=cell)
    else:
        mol = Atoms(symbols=numbers, positions=coords)

    if return_crystal:
        cry = ase_crystal(symbols=mol, cell=cell, setting=1, spacegroup=int(crystal_batch.sg_ind[index]))
        return mol, cry
    else:
        return mol
