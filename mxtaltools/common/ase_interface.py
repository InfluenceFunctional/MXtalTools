"""Conversion of crystal data batches to ASE ``Atoms`` objects, CIF writing through ASE, and
ASE Niggli reduction of a structure's cell."""
import numpy as np
import torch
from ase import Atoms
from ase.build import niggli_reduce
from ase.geometry import cell_to_cellpar
from ase.io import read
from ase.spacegroup import crystal as ase_crystal
from ase.visualize import view


def data_batch_to_ase_mols_list(crystaldata_batch,
                                max_ind: int = np.inf,
                                specific_inds=None,
                                show_mols: bool = False, **kwargs):
    """
    Convert several structures of a crystal data batch into ASE ``Atoms`` objects.

    Parameters
    ----------
    crystaldata_batch : MolCrystalData, single or collated
        Structures to convert. ``num_graphs`` is read if present; otherwise 1 is assumed.
    max_ind : int, optional
        When ``specific_inds`` is None, graphs ``0`` to ``min(max_ind, num_graphs) - 1`` are
        converted. Default ``np.inf`` (all graphs).
    specific_inds : iterable of int, optional
        Batch indices to convert, in the given order; takes precedence over ``max_ind``.
    show_mols : bool, optional
        If True, the list is opened with ``ase.visualize.view``. Default False.
    **kwargs
        Forwarded to :func:`ase_mol_from_crystaldata`.

    Returns
    -------
    list
        One :func:`ase_mol_from_crystaldata` result per index: an ``ase.Atoms``, or a tuple if
        ``return_crystal=True`` is forwarded.
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
                             return_crystal: bool = False,
                             highlight_mol_ind: bool=False):
    """
    Extract one structure from a crystal data batch as an ASE ``Atoms`` object.

    The batch is cloned, moved to CPU and detached first; the input is not modified. The atoms
    of graph ``index`` are selected through ``crystal_batch.batch`` (all atoms if ``batch`` is
    None) and then filtered according to ``mode``. The cell is ``T_fc[index].T`` (rows are the
    lattice vectors; ``T_fc[0]`` is used if ``index`` is None or out of range), or no cell if
    ``T_fc`` is None. Periodic boundary flags are not set.

    Parameters
    ----------
    crystal_batch : MolCrystalData, single or collated
        Reads ``batch``, ``z``, ``pos`` and ``T_fc`` and, depending on the options,
        ``aux_ind``, ``mol_ind``, ``ptr``, ``num_atoms``, ``sym_mult``, ``unit_cell_pos``,
        ``unit_cell_batch`` and ``sg_ind``.
    index : int, optional
        Batch index of the structure. Required when ``crystal_batch.batch`` is not None.
    highlight_canonical_conformer : bool, optional
        If True, atomic numbers are replaced by ``aux_ind + 6`` (so ``aux_ind`` 0, 1, 2 appear
        as C, N, O). Default False.
    mode : str or None, optional
        Atom selection within graph ``index``, for an input that is a molecule cluster larger
        than a single unit cell:

        - ``'conformer'``: atoms with ``aux_ind == 0``.
        - ``'unit cell'``: the rows of ``unit_cell_pos`` with ``unit_cell_batch == index``;
          atomic numbers are the asymmetric unit's (the ``num_atoms[index]`` entries of ``z``
          starting at ``ptr[index]`` on a row; the graph's ``aux_ind == 0`` atoms on a cluster),
          each of the crystal's ``z_prime`` units tiled ``sym_mult[index]`` times, unit by unit
          (``crystal_building.utils.unit_cell_z``, the unit cell's layout).
        - ``'inside cell'``: atoms whose fractional coordinates,
          ``inv(T_fc[index]) @ pos``, all lie strictly between 0 and 1.
        - ``'convolve with'``: atoms with ``aux_ind < 2``.
        - ``'distance'``: atoms closer to the centroid of the ``aux_ind == 0`` atoms than the
          largest centroid distance within that group plus ``cutoff``.
        - None or any other value: all atoms of the graph.

        Default None.
    cutoff : float, optional
        Extra distance for ``mode='distance'``, in the units of ``pos``. Default 4.
    return_crystal : bool, optional
        If True, also returns ``ase.spacegroup.crystal(symbols=mol, cell=cell, setting=1,
        spacegroup=int(sg_ind[index]))`` built from the extracted atoms. Does not always work
        properly; ASE does not understand these crystals correctly. Default False.
    highlight_mol_ind : bool, optional
        If True and ``highlight_canonical_conformer`` is False, atomic numbers are replaced by
        ``mol_ind + 6``. Default False.

    Returns
    -------
    ase.Atoms or tuple of ase.Atoms
        The extracted structure, or ``(mol, crystal)`` if ``return_crystal`` is True.
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
        atom_inds = torch.argwhere(crystal_batch.unit_cell_batch == index).flatten()
        coords=crystal_batch.unit_cell_pos[atom_inds]

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
    elif highlight_mol_ind:
        numbers = crystal_batch.mol_ind[atom_inds].cpu().detach().numpy() + 6
    elif mode == 'unit cell':
        from mxtaltools.crystal_building.utils import unit_cell_z   # the unit cell's layout, unit by unit
        aux = getattr(crystal_batch, 'aux_ind', None)
        if aux is not None:   # a cluster: each unit's molecule 0 (aux_ind 0), unit by unit, is the asymmetric unit
            in_graph = (crystal_batch.batch == index) if crystal_batch.batch is not None else torch.ones_like(aux, dtype=torch.bool)
            aunit_z = crystal_batch.z[in_graph & (aux == 0)]
        else:                 # a row: its own atoms
            start = crystal_batch.ptr[index]
            aunit_z = crystal_batch.z[start:start + crystal_batch.num_atoms[index]]
        numbers = unit_cell_z(aunit_z, crystal_batch.z_prime.reshape(-1)[index],
                              crystal_batch.sym_mult[index]).cpu().detach().numpy()
    else:
        numbers = crystal_batch.z[atom_inds].cpu().detach().numpy()

    if crystal_batch.T_fc is not None:
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
        cell = None

    if return_crystal:
        cry = ase_crystal(symbols=mol, cell=cell, setting=1, spacegroup=int(crystal_batch.sg_ind[index]))
        return mol, cry
    else:
        return mol


def get_niggli_cell(crystal_batch, index, radians: bool=False):
    """ASE's Niggli cell parameters: all angles acute or all obtuse. For all-acute cells this is NOT the triclinic
    convention of sym_utils.tri_niggli_reduction_penalty (beta, gamma obtuse), which needs (a, b, c) -> (a, -b, -c):
    beta -> 180 - beta, gamma -> 180 - gamma. Lengths are the same in both.

    The structure from :func:`ase_mol_from_crystaldata` (all atoms of graph ``index``, cell
    ``T_fc[index].T``) is written to ``temp.cif`` in the current working directory, read back
    with ``ase.io.read``, reduced in place with ``ase.build.niggli_reduce`` and converted with
    ``ase.geometry.cell_to_cellpar``. ``temp.cif`` is left on disk.

    Parameters
    ----------
    crystal_batch : MolCrystalData, single or collated
        Source of the structure and its ``T_fc``.
    index : int
        Batch index of the structure.
    radians : bool, optional
        If True, angles are returned in radians; otherwise in degrees. Default False.

    Returns
    -------
    numpy.ndarray
        [6] ``(a, b, c, alpha, beta, gamma)``; lengths in the units of ``T_fc``.
    """

    mol = ase_mol_from_crystaldata(crystal_batch, index=index)
    #mol.info['spacegroup'] = Spacegroup(int(original_cluster_batch.sg_ind[ind]), setting=1)
    mol.write('temp.cif')
    atoms = read("temp.cif")
    niggli_reduce(atoms)
    return cell_to_cellpar(atoms.cell, radians=radians)


def ase_write_cif(batch, inds, path, mode):
    """
    Write selected structures of a batch to CIF files with ASE.

    For each ``ind``, the structure from ``ase_mol_from_crystaldata(batch, ind, mode=mode)`` is
    written to ``f'{path}_{ind}.cif'``, and the first ``data_image0`` in that file is replaced by
    ``data_{path}_{ind}``.

    Parameters
    ----------
    batch : MolCrystalData, single or collated
        Source structures.
    inds : iterable of int
        Batch indices to write.
    path : str
        File-path prefix; also used in the CIF data block name.
    mode : str or None
        Atom selection passed to :func:`ase_mol_from_crystaldata`.
    """
    for ind in inds:
        mol = ase_mol_from_crystaldata(batch, ind, mode=mode)
        cif_path = f"{path}_{ind}.cif"
        mol.write(cif_path)
        with open(cif_path, 'r') as f:
            content = f.read()
        content = content.replace('data_image0', f'data_{path}_{ind}', 1)
        with open(cif_path, 'w') as f:
            f.write(content)