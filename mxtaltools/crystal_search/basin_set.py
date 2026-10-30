"""Basin sets: a search's distinct minima, written as a ``crystal_fd.dataset/1`` folder for prior construction.

Design: GFN ``energy_sampling/docs/design/basin_set.md`` (2026-10-30). One writer (``write_basin_set``) turns the
rows ``minima.minimum_rows`` selects (one row per library minimum, carrying ``id_params``, ``id_handedness``,
``id_hessian`` and ``minimum_f``) into a dataset folder; one reader (``read_basin_set``) returns the dataset and its
header. The streaming search, a partial ``<run>.pt``, or the batch loop's output can all be written through it.

What a row holds
----------------
The dataset row is the basin's canonical Z' = 1 description (``crystal_fd.canon.canonicalize`` on the requested
sheet, Sohncke rows per ``off_sheet``), with its reference molecule, identifier and canonicalisation status. Columns:

``basin_id``        the search's ``minimum_id``.
``p``               [12] float64, the identity-description cell parameters the Hessian was taken at (a, b, c in
                    Angstrom; alpha, beta, gamma in radians; the fractional centre; the rotation vector in radians).
``p_handedness``    the handedness of ``p``. A draw is built in ``p``'s own description and canonicalised afterwards,
                    so ``p`` need not be the canonical row's parameters.
``E``               the physical energy at the hard cutoff, in the header's units, raw (no temperature, no lj scale).
``E_env``           ``minimum_f`` x kT: the enveloped energy the Hessian is of.
``H``               [12, 12] float64, the Hessian of ``E_env`` at ``p`` in header units per coordinate unit squared
                    (``id_hessian`` x kT).
``live_mask``       [12] bool, ``live_mask(sg)``: the coordinates that are degrees of freedom.
``min_eig``         the smallest eigenvalue of ``H`` on the live coordinates.
``hits``, ``seed_hits``, ``polish_status``   as the search wrote them.

The header (the dataset's ``provenance``) carries ``format``, ``energy``, ``search`` (with ``kT``), ``identity``,
``sheets`` and the row counts; the writer's caller supplies the energy, search and identity records.
"""
from typing import Optional, Sequence

import numpy as np
import torch

FORMAT = 'basin_set/1'
D = 12
#: lattice types whose cell angles are fixed by the group: (alpha, beta, gamma) live or not (standard setting, b unique)
_ANGLES_LIVE = {'triclinic': (True, True, True), 'monoclinic': (False, True, False),
                'orthorhombic': (False, False, False)}


def live_mask(sg: int) -> torch.Tensor:
    """[12] bool: the cell parameters that are degrees of freedom in space group ``sg`` at Z' = 1, in the order
    a, b, c, alpha, beta, gamma, centre (3), rotation vector (3). Lengths and rotation are always live; angles fixed
    by the lattice type are not; a free centroid axis (crystal_fd tables ``free``) is not. Lattice types that tie
    lengths (tetragonal and up) raise: their coordinates are not independent in this layout."""
    from mxtaltools.crystal_fd.tables import load_raw
    g = load_raw()[str(int(sg))]
    lt = g['lattice_type']
    if lt not in _ANGLES_LIVE:
        raise NotImplementedError(f"live_mask: sg {sg} is {lt}; tied cell lengths are not handled")
    m = torch.ones(D, dtype=torch.bool)
    m[3:6] = torch.tensor(_ANGLES_LIVE[lt])
    m[6:9] = ~torch.tensor([bool(f) for f in g['free']])
    return m


def _scalar(v) -> float:
    return float(torch.as_tensor(v).reshape(-1)[0])


def write_basin_set(rows: Sequence, path: str, *, kT: float, energy_attr: str, energy: dict, search: dict,
                    identity: dict, sheet: int = 1, off_sheet: str = 'enantiomorph',
                    provenance: Optional[dict] = None) -> dict:
    """Write ``rows`` (MolCrystalData, one per library minimum, as ``minima.minimum_rows`` returns them) as a basin set
    at ``path`` (which must not exist). ``kT`` is the search's, in ``energy['units']``; ``energy_attr`` names the row
    attribute holding the hard-cutoff target energy (e.g. ``'elj'``). Returns the dataset's ``meta.json``.

    Rows the importer refuses are left out and counted in the header (``counts``). Raises if no row is written."""
    from mxtaltools.crystal_fd import canon
    from mxtaltools.crystal_fd.dataset import save_dataset
    from mxtaltools.crystal_fd.io_mxt import from_mol_crystal_data

    rows = list(rows)
    if not rows:
        raise ValueError('write_basin_set: no rows')
    for k in ('target', 'model_id', 'units', 'per'):
        if k not in energy:
            raise ValueError(f"write_basin_set: energy record lacks {k!r}")
    staged = []
    for r in rows:
        if int(_scalar(r.z_prime)) != 1:
            raise NotImplementedError("write_basin_set: Z' = 1 only (the dataset format holds one molecule per row)")
        c = r.clone()
        c.aunit_handedness = torch.as_tensor(r.id_handedness).reshape(c.aunit_handedness.shape).to(
            c.aunit_handedness.dtype)
        c.set_cell_parameters(torch.as_tensor(r.id_params).reshape(1, D).to(c.cell_lengths.dtype))
        # the importer reads pos as the literal posed asymmetric unit; a search row holds the molecule in its own
        # frame, built with std_orientation=True, so pose it the way the search built it
        c.pose_aunit(std_orientation=True)
        staged.append(c)
    desc, info = from_mol_crystal_data(staged, sheet=sheet, off_sheet=off_sheet)
    kept = info.row.long()
    if kept.numel() == 0:
        raise ValueError(f'write_basin_set: the importer refused every row ({set(info.reason)})')
    can = canon.canonicalize(desc, sheet=sheet)

    src = [rows[i] for i in kept.tolist()]
    sg = torch.tensor([int(_scalar(r.sg_ind)) for r in src])
    p = torch.stack([torch.as_tensor(r.id_params, dtype=torch.float64).reshape(D) for r in src])
    ph = torch.tensor([_scalar(r.id_handedness) for r in src], dtype=torch.float64)
    H = torch.stack([torch.as_tensor(r.id_hessian, dtype=torch.float64).reshape(D, D) for r in src]) * float(kT)
    H = 0.5 * (H + H.transpose(1, 2))
    live = torch.stack([live_mask(int(g)) for g in sg.tolist()])
    min_eig = torch.stack([torch.linalg.eigvalsh(H[i][live[i]][:, live[i]]).min() for i in range(len(src))])
    columns = {
        'basin_id': torch.tensor([int(_scalar(r.minimum_id)) for r in src], dtype=torch.long),
        'p': p,
        'p_handedness': ph,
        'E': torch.tensor([_scalar(getattr(r, energy_attr)) for r in src], dtype=torch.float64),
        'E_env': torch.tensor([_scalar(r.minimum_f) for r in src], dtype=torch.float64) * float(kT),
        'H': H,
        'live_mask': live,
        'min_eig': min_eig,
        'hits': torch.tensor([int(_scalar(getattr(r, 'hits', 0))) for r in src], dtype=torch.long),
        'seed_hits': torch.tensor([int(_scalar(getattr(r, 'seed_hits', 0))) for r in src], dtype=torch.long),
        'polish_status': torch.tensor([int(_scalar(getattr(r, 'polish_status', -1))) for r in src],
                                      dtype=torch.long),
    }
    docs = {'basin_id': "the search's minimum_id", 'p': 'identity-description cell parameters H is taken at',
            'p_handedness': 'handedness of p', 'E': f'{energy_attr} at the hard cutoff, raw header units',
            'E_env': 'minimum_f x kT, the enveloped energy H is of', 'H': 'Hessian of E_env at p, header units',
            'live_mask': 'basin_set.live_mask(sg)', 'min_eig': 'smallest eigenvalue of H on live coordinates',
            'hits': 'random starts ending here', 'seed_hits': 'seeds ending here',
            'polish_status': 'minima.POLISH_STATUS code'}
    header = dict(provenance or {})
    header.update({'format': FORMAT, 'energy': dict(energy), 'search': dict(search, kT=float(kT)),
                   'identity': dict(identity), 'sheets': {'sheet': sheet, 'off_sheet': off_sheet},
                   'counts': {'rows_in': len(rows), 'written': int(kept.numel()),
                              'refused': {str(k): int(v) for k, v in zip(*np.unique(
                                  [str(x) for x in info.reason if x is not None], return_counts=True))}}})
    ident = [str(r.identifier if isinstance(r.identifier, str) else r.identifier[0]) for r in src]
    return save_dataset(path, can.desc, info.mols, ident, reference_handedness=info.reference_handedness,
                        status=can.status.long(), status_names=canon.STATUS, columns=columns, column_docs=docs,
                        provenance=header)


def read_basin_set(path: str, status=None):
    """(FDDataset, header) of a basin set. ``status`` is passed to ``crystal_fd.dataset.load_dataset`` (None: every
    row). Raises if the folder is not a basin set of this format."""
    from mxtaltools.crystal_fd.dataset import load_dataset
    ds = load_dataset(path, status=status)
    header = ds.meta.get('provenance', {})
    if header.get('format') != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} basin set (format {header.get('format')!r})")
    return ds, header
