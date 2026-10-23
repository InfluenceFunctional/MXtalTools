"""Crystal descriptions as batch-first tensors, and the molecule each asymmetric unit holds.

A row is one crystal in the standard setting of its space group (``SYM_OPS[sg]``).
Asymmetric unit ``k`` of a row places the reference molecule's body coordinates
``X`` at Cartesian ``T_fc t_k + P_k X`` with pose ``P_k = R_k diag(h_k, 1, 1)``:
``R_k`` a proper rotation and ``h_k = +-1`` the handedness, as the live builder
does (``MolCrystalOps.aunit_orientation`` is the rotation vector of ``R_k``,
``aunit_handedness`` is ``h_k``).
"""
import hashlib
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform


@dataclass(frozen=True, eq=False)
class MoleculeRef:
    """Body coordinates of one molecule, centred as the live builder centres them.

    Attributes
    ----------
    pos : torch.Tensor
        [n, 3] float64 Cartesian body coordinates (Angstrom), centred on the
        heavy-atom (z > 1) centroid (``center_batch(center_on_heavy_atoms=True)``).
    z : torch.Tensor
        [n] long atomic numbers.
    hash : str
        sha256 of ``pos`` rounded to 1e-6 Angstrom and ``z``: an identity of the
        stored coordinates, not a tolerance match (``molecule_hash``).
    """
    pos: torch.Tensor
    z: torch.Tensor
    hash: str

    @classmethod
    def from_coords(cls, pos, z) -> 'MoleculeRef':
        """Centre ``pos`` [n, 3] on the centroid of atoms with ``z > 1`` and hash it.

        For a GFN row, ``pos`` is the stored, admission-oriented ``pos`` (built
        with ``std_orientation=False``); no principal-axis frame is recomputed.
        """
        pos = torch.as_tensor(pos, dtype=torch.float64).reshape(-1, 3)
        z = torch.as_tensor(z, dtype=torch.long).reshape(-1)
        heavy = z > 1
        if not bool(heavy.any()):
            raise ValueError('a molecule needs at least one heavy (z > 1) atom to be centred as the builder centres it')
        pos = pos - pos[heavy].mean(0)
        return cls(pos=pos, z=z, hash=molecule_hash(pos, z))


def molecule_hash(pos, z) -> str:
    """sha256 of coordinates rounded to 1e-6 Angstrom (int64) and atomic numbers.

    An identity of stored coordinates: a coordinate near a rounding boundary flips
    under any noise, so molecules from different pipelines (float32 vs float64,
    recentred) are matched by comparing coordinates within a tolerance, not by
    this hash.
    """
    q = np.round(np.asarray(pos, dtype=np.float64) * 1e6).astype(np.int64)
    h = hashlib.sha256()
    h.update(np.asarray(z, dtype=np.int64).tobytes())
    h.update(q.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class CrystalDescription:
    """A batch of crystal descriptions.

    Attributes
    ----------
    sg : torch.Tensor
        [B] long space-group numbers; the setting is ``SYM_OPS[sg]``.
    cell : torch.Tensor
        [B, 6]: a, b, c (Angstrom) and alpha, beta, gamma (radians).
    z_prime : torch.Tensor
        [B] long number of asymmetric units per row.
    unit_mask : torch.Tensor
        [B, Zmax] bool; True for the first ``z_prime`` slots of a row.
    t : torch.Tensor
        [B, Zmax, 3] fractional centroids.
    R : torch.Tensor
        [B, Zmax, 3, 3] proper rotations.
    h : torch.Tensor
        [B, Zmax] long handedness, +1 or -1.

    Group arithmetic on descriptions is float64; float32 descriptions are
    accepted for building and energies.
    """
    sg: torch.Tensor
    cell: torch.Tensor
    z_prime: torch.Tensor
    unit_mask: torch.Tensor
    t: torch.Tensor
    R: torch.Tensor
    h: torch.Tensor

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_poses(cls, sg, cell, t, P, z_prime=None) -> 'CrystalDescription':
        """From poses ``P`` [B, Z, 3, 3] or [B, 3, 3] (any determinant); ``t`` [B, Z, 3] or [B, 3]."""
        from mxtaltools.crystal_fd.so3 import split_o3
        t = torch.as_tensor(t)
        P = torch.as_tensor(P)
        if t.ndim == 2:
            t, P = t[:, None], P[:, None]
        R, h = split_o3(P.reshape(-1, 3, 3))
        return cls.from_parts(sg, cell, t, R.reshape(P.shape).to(P.dtype), h.reshape(P.shape[:2]), z_prime)

    @classmethod
    def from_parts(cls, sg, cell, t, R, h=None, z_prime=None) -> 'CrystalDescription':
        """From proper rotations ``R`` and handedness ``h`` (default +1); Z' = 1 inputs may omit the unit axis.

        Every field is placed on the device of ``t``.
        """
        t = torch.as_tensor(t)
        dev = t.device
        R = torch.as_tensor(R).to(dev)
        if t.ndim == 2:
            t, R = t[:, None], R[:, None]
            h = None if h is None else torch.as_tensor(h).reshape(-1, 1)
        B, Z = t.shape[:2]
        h = torch.ones(B, Z, dtype=torch.long, device=dev) if h is None \
            else torch.as_tensor(h).to(device=dev, dtype=torch.long).reshape(B, Z)
        z_prime = torch.full((B,), Z, dtype=torch.long, device=dev) if z_prime is None \
            else torch.as_tensor(z_prime).to(device=dev, dtype=torch.long).reshape(B)
        sg = torch.as_tensor(sg).to(device=dev, dtype=torch.long).reshape(-1)
        if sg.numel() == 1 and B > 1:
            sg = sg.expand(B).clone()
        cell = torch.as_tensor(cell).to(dev)
        if cell.ndim == 1:
            cell = cell[None].expand(B, 6).clone()
        mask = torch.arange(Z, device=dev)[None, :] < z_prime[:, None]
        return cls(sg=sg, cell=cell, z_prime=z_prime, unit_mask=mask, t=t, R=R, h=h)

    @classmethod
    def from_crystal_params(cls, sg, cell_lengths, cell_angles, aunit_centroid, aunit_orientation,
                            aunit_handedness=None, z_prime=None) -> 'CrystalDescription':
        """From MolCrystalData-style parameters: centroids and rotation vectors [B, 3 Z'], handedness [B, Z'].

        Rotation vectors are read by ``so3.rotvec_to_matrix`` (any norm); the
        description is float64.
        """
        from mxtaltools.crystal_fd.so3 import rotvec_to_matrix
        c = torch.as_tensor(aunit_centroid, dtype=torch.float64)
        B = c.shape[0]
        Z = c.shape[1] // 3
        rv = torch.as_tensor(aunit_orientation, dtype=torch.float64).reshape(B * Z, 3)
        R = rotvec_to_matrix(rv).reshape(B, Z, 3, 3)
        cell = torch.cat([torch.as_tensor(cell_lengths, dtype=torch.float64),
                          torch.as_tensor(cell_angles, dtype=torch.float64)], dim=1)
        return cls.from_parts(sg, cell, c.reshape(B, Z, 3), R, aunit_handedness, z_prime)

    def crystal_params(self):
        """(cell_lengths [B, 3], cell_angles [B, 3], aunit_centroid [B, 3 Z'], aunit_orientation [B, 3 Z'],
        aunit_handedness [B, Z']) in MolCrystalData's layout; rotation vectors on the trainer's +z hemisphere
        (``so3.matrix_to_rotvec(hemisphere=True)``)."""
        from mxtaltools.crystal_fd.so3 import matrix_to_rotvec
        B, Z = self.t.shape[:2]
        rv = matrix_to_rotvec(self.R.reshape(-1, 3, 3), hemisphere=True).reshape(B, 3 * Z)
        return (self.cell[:, :3], self.cell[:, 3:], self.t.reshape(B, 3 * Z), rv, self.h.clone())

    # ------------------------------------------------------------------ properties

    @property
    def num_rows(self) -> int:
        return int(self.sg.shape[0])

    @property
    def max_z_prime(self) -> int:
        return int(self.t.shape[1])

    @property
    def device(self) -> torch.device:
        return self.t.device

    @property
    def pose(self) -> torch.Tensor:
        """[B, Zmax, 3, 3] poses P = R diag(h, 1, 1)."""
        flip = torch.ones(*self.h.shape, 3, dtype=self.R.dtype, device=self.R.device)
        flip[..., 0] = self.h.to(self.R.dtype)
        return self.R * flip[..., None, :]

    def cell_matrix(self, dtype=torch.float64) -> torch.Tensor:
        """[B, 3, 3] fractional-to-Cartesian matrices T_fc (columns a, b, c; a along x, b in the xy plane)."""
        c = self.cell.to(dtype)
        return batch_compute_fractional_transform(c[:, :3], c[:, 3:])[0]

    # ------------------------------------------------------------------ manipulation

    def replace(self, **changes) -> 'CrystalDescription':
        return replace(self, **changes)

    def clone(self) -> 'CrystalDescription':
        return replace(self, **{f.name: getattr(self, f.name).clone() for f in fields(self)})

    def to(self, device=None, dtype=None) -> 'CrystalDescription':
        """Move to ``device``; ``dtype`` applies to the floating fields (cell, t, R)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to(device=device, dtype=dtype) if (dtype is not None and v.is_floating_point()) \
                else v.to(device=device)
        return CrystalDescription(**out)

    def double(self) -> 'CrystalDescription':
        return self.to(dtype=torch.float64)

    def __getitem__(self, rows) -> 'CrystalDescription':
        if isinstance(rows, int):
            rows = slice(rows, rows + 1)
        return CrystalDescription(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @staticmethod
    def cat(descs) -> 'CrystalDescription':
        """Concatenate along rows; every input must have the same Zmax."""
        descs = list(descs)
        return CrystalDescription(**{f.name: torch.cat([getattr(d, f.name) for d in descs]) for f in fields(descs[0])})

    def validate(self, tol: float = 1e-6) -> None:
        """Raise ValueError unless shapes and devices agree, every used R is a proper rotation within ``tol`` and h
        is +-1."""
        B, Z = self.t.shape[:2]
        for name, shape in (('sg', (B,)), ('cell', (B, 6)), ('z_prime', (B,)), ('unit_mask', (B, Z)),
                            ('t', (B, Z, 3)), ('R', (B, Z, 3, 3)), ('h', (B, Z))):
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f'{name} has shape {tuple(getattr(self, name).shape)}, expected {shape}')
            if getattr(self, name).device != self.device:
                raise ValueError(f'{name} is on {getattr(self, name).device}, t on {self.device}')
        if not torch.equal(self.unit_mask, torch.arange(Z, device=self.device)[None] < self.z_prime[:, None]):
            raise ValueError('unit_mask does not match z_prime')
        m = self.unit_mask
        R = self.R[m].double()
        eye = torch.eye(3, dtype=torch.float64, device=R.device)
        if len(R) and ((R @ R.transpose(1, 2) - eye).abs().amax() > tol or (torch.linalg.det(R) - 1).abs().amax() > tol):
            raise ValueError(f'R is not a proper rotation within {tol}')
        if not bool((self.h[m].abs() == 1).all()):
            raise ValueError('h must be +1 or -1')
        if bool(((self.sg < 1) | (self.sg > 230)).any()):
            raise ValueError('sg must be in 1-230')
