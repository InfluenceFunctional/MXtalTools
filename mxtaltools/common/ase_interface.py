import numpy as np
import torch
from ase import Atoms
from ase.spacegroup import crystal as ase_crystal
from ase.visualize import view


def crystaldata_batch_to_ase_mols_list(crystaldata_batch, max_ind: int = np.inf, specific_inds=None,
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
    if specific_inds is None:
        mols = [ase_mol_from_crystaldata(crystaldata_batch, ii, **kwargs)
                for ii in range(min(max_ind, crystaldata_batch.num_graphs))]
    else:
        mols = [ase_mol_from_crystaldata(crystaldata_batch, ii, **kwargs)
                for ii in specific_inds]
    if show_mols:
        view(mols)
    return mols


def ase_mol_from_crystaldata(data,
                             index: int = None,
                             highlight_canonical_conformer: bool = False,
                             exclusion_level=None,
                             inclusion_distance: float = 4,
                             return_crystal: bool = False):
    """
    Extract an atomic structure from a Crystaldata object according to its batch index, and convert it into an ase mol object.
    Several options for visualization of crystals.

    Parameters
    ----------
    data : Crystaldata object
    index : int, optional
        Batch index of Crystaldata object to extract. Default is None.
    highlight_canonical_conformer : bool, optional
        Whether to give the canonical conformer a different atom type for color comparison.
    exclusion_level : 'conformer', 'unit cell', 'inside cell', 'convolve with', 'distance', or None (default None)
        Assuming the input Crystaldata is a molecule cluster larger than a single unit cell, we have several options to pare down to the desired visualization.
        'conformer' : only the 'canonical conformer'
        'unit cell' : all molecules with centroids inside the unit cell
        'inside cell' : all atoms inside the unit cell
        'convolve with' : all atoms within convolution range of the canonical conformer
        'distance' : all atoms within a certain distance of the canonical conformer
        None : show all atoms in the crystal
    inclusion_distance : float, optional
        Distance fed to the 'distance' exclusion level option
    return_crystal : bool, optional
        Whether to return an ase mol for the crystal. Does not always work properly. Ase does not understand our crystals correctly.


    Returns
    -------
    ase mol object
    """
    data = data.clone().cpu().detach()
    if data.batch is not None:  # more than one crystal in the datafile
        atom_inds = torch.where(data.batch == index)[0]
    else:
        atom_inds = torch.arange(len(data.x))

    if exclusion_level == 'conformer':  # only the canonical conformer itself
        inside_inds = torch.where(data.aux_ind == 0)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'unit cell':
        # assume that by construction the first Z molecules are the ones in the unit cell
        mol_size = data.num_atoms[index]
        num_molecules = int((data.ptr[index + 1] - data.ptr[index]) / mol_size)

        molecule_centroids = torch.stack(
            [torch.mean(
                data.pos[data.ptr[index] + int(mol_size * multiplier):data.ptr[index] + int(mol_size * multiplier + 1)],
                dim=0)
                for multiplier in range(num_molecules)])

        fractional_centroids = torch.inner(torch.linalg.inv(data.T_fc[index]), molecule_centroids).T

        inside_centroids = torch.prod((fractional_centroids < 1) * (fractional_centroids > 0), dim=-1)
        # assert inside_centroids.sum() == data.Z[index]  # must be exactly Z molecules in the unit cell
        inside_centroids_inds = torch.where(inside_centroids)[0]

        inside_inds = torch.cat(
            [torch.arange(mol_size) + mol_size * inside_centroids_inds[ind]
             for ind in range(len(inside_centroids_inds))]
        ).long()
        inside_inds += data.ptr[index]
        atom_inds = inside_inds
        coords = data.pos[inside_inds].cpu().detach().numpy()

    elif exclusion_level == 'inside cell':
        fractional_coords = torch.inner(torch.linalg.inv(data.T_fc[index]), data.pos[data.batch == index]).T
        inside_coords = torch.prod((fractional_coords < 1) * (fractional_coords > 0), dim=-1)
        inside_inds = torch.where(inside_coords)[0]
        inside_inds += data.ptr[index]
        atom_inds = inside_inds
        coords = data.pos[inside_inds].cpu().detach().numpy()

    elif exclusion_level == 'convolve with':  # atoms potentially in the convolutional field
        inside_inds = torch.where(data.aux_ind < 2)[0]
        new_atom_inds = torch.stack([ind for ind in atom_inds if ind in inside_inds])
        atom_inds = new_atom_inds
        coords = data.pos[atom_inds].cpu().detach().numpy()

    elif exclusion_level == 'distance':  # atoms within a certain distance of the conformer radius
        crystal_coords = data.pos[atom_inds]
        crystal_inds = data.aux_ind[atom_inds]

        canonical_conformer_inds = torch.where(crystal_inds == 0)[0]
        mol_centroid = crystal_coords[canonical_conformer_inds].mean(0)
        mol_radius = torch.max(torch.cdist(mol_centroid[None], crystal_coords[canonical_conformer_inds], p=2))
        in_range_inds = \
            torch.where((torch.cdist(mol_centroid[None], crystal_coords, p=2) < (mol_radius + inclusion_distance))[0])[
                0]
        atom_inds = atom_inds[in_range_inds]
        coords = crystal_coords[in_range_inds].cpu().detach().numpy()
    else:
        coords = data.pos[atom_inds].cpu().detach().numpy()

    if highlight_canonical_conformer:  # highlight the atom aux index
        numbers = data.aux_ind[atom_inds].cpu().detach().numpy() + 6
    else:
        if data.x.ndim > 1:
            numbers = data.x[atom_inds, 0].cpu().detach().numpy()
        else:
            numbers = data.x[atom_inds].cpu().detach().numpy()

    if index is not None:
        try:
            cell = data.T_fc[index].T.cpu().detach().numpy()
        except IndexError:
            cell = data.T_fc[0].T.cpu().detach().numpy()

    else:
        cell = data.T_fc[0].T.cpu().detach().numpy()

    mol = Atoms(symbols=numbers, positions=coords, cell=cell)
    if return_crystal:
        cry = ase_crystal(symbols=mol, cell=cell, setting=1, spacegroup=int(data.sg_ind[index]))
        return mol, cry
    else:
        return mol
