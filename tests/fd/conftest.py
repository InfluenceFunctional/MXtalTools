"""Fixtures and markers for the fundamental-domain gates (fd_tooling.md section 4).

Markers
-------
``cctbx``: re-derives a fact with cctbx (a generator-time dependency); skipped
when cctbx is not importable. ``slow``: not part of the fast tier.

Fixtures
--------
``chiral_molecule``, ``planar_molecule``: small synthetic molecules, float64,
centred on the heavy-atom centroid as the builder centres them.
``cell_sampler``: random cells per lattice family.
``literal_oracle``: a float64 literal-atom builder, independent of MXtalTools'
crystal construction, and a periodic atom-deviation measure.
"""
import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'cctbx: needs cctbx (the table generator); skipped when not importable')
    config.addinivalue_line('markers', 'slow: not part of the fast tier')


def _cctbx_importable() -> bool:
    try:
        import cctbx.sgtbx  # noqa: F401
        return True
    except Exception:
        return False


CCTBX = _cctbx_importable()


@pytest.fixture(autouse=True)
def _skip_without_cctbx(request):
    if request.node.get_closest_marker('cctbx') is not None and not CCTBX:
        pytest.skip('cctbx is not importable')


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------

def _centre_heavy(pos, z):
    heavy = z > 1
    return pos - pos[heavy].mean(0)


@pytest.fixture(scope='session')
def chiral_molecule():
    """CHFClBr-like: four distinct substituents at distinct bond lengths, so no improper symmetry.

    Returns dict ``pos`` float64 [5, 3] (Angstrom) and ``z`` int64 [5].
    """
    d = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)
    z = np.array([6, 1, 9, 17, 35])
    pos = np.vstack([np.zeros(3), 1.09 * d[0], 1.35 * d[1], 1.77 * d[2], 1.94 * d[3]])
    return dict(pos=_centre_heavy(pos, z), z=z)


@pytest.fixture(scope='session')
def planar_molecule():
    """Three heavy atoms in a scalene triangle: planar, so achiral (its plane is a mirror)."""
    z = np.array([8, 6, 7])
    pos = np.array([[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [-0.52, 1.21, 0.0]])
    return dict(pos=_centre_heavy(pos, z), z=z)


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def sample_cells(lattice_type: str, n: int, seed: int = 0) -> np.ndarray:
    """Random cells [n, 6]: a, b, c in Angstrom; alpha, beta, gamma in radians.

    ``lattice_type`` is a LATTICE_TYPE value. Triclinic angles are drawn in
    [65, 115] degrees and redrawn until the cell has positive volume; trigonal
    groups use the hexagonal family.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 6))
    for i in range(n):
        a, b, c = rng.uniform(4.0, 16.0, 3)
        if lattice_type == 'triclinic':
            while True:
                ang = np.radians(rng.uniform(65.0, 115.0, 3))
                ca, cb, cg = np.cos(ang)
                if 1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg > 1e-3:
                    break
        elif lattice_type == 'monoclinic':
            ang = np.radians([90.0, rng.uniform(95.0, 125.0), 90.0])
        elif lattice_type in ('orthorhombic', 'tetragonal', 'cubic'):
            ang = np.radians([90.0, 90.0, 90.0])
        elif lattice_type == 'hexagonal':
            ang = np.radians([90.0, 90.0, 120.0])
        else:
            raise ValueError(lattice_type)
        if lattice_type in ('tetragonal', 'hexagonal'):
            b = a
        if lattice_type == 'cubic':
            b = c = a
        out[i] = [a, b, c, *ang]
    return out


@pytest.fixture(scope='session')
def cell_sampler():
    return sample_cells


# ---------------------------------------------------------------------------
# float64 literal-atom oracle
# ---------------------------------------------------------------------------

def cell_matrix(cell) -> np.ndarray:
    """Fractional-to-Cartesian matrix [3, 3], columns a, b, c; a along x, b in the xy plane."""
    a, b, c, al, be, ga = (float(v) for v in cell)
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    return np.array([[a, b * np.cos(ga), cx],
                     [0.0, b * np.sin(ga), cy],
                     [0.0, 0.0, np.sqrt(c ** 2 - cx ** 2 - cy ** 2)]])


def literal_unit_cell(cell, t, R, X, W, w) -> np.ndarray:
    """Every molecule of a Z' = 1 crystal, literally: [Z, n_atoms, 3] Cartesian, float64.

    Molecule k sits at centroid W_k t + w_k wrapped to [0, 1)^3 with body
    coordinates W_k applied to the asymmetric-unit molecule's fractional offsets,
    the asymmetric-unit molecule being R X (X centred, Cartesian) at centroid t.
    """
    T = cell_matrix(cell)
    off = np.linalg.solve(T, (np.asarray(R, float) @ np.asarray(X, float).T)).T      # fractional offsets [n, 3]
    W = np.asarray(W, float).reshape(-1, 3, 3)
    w = np.asarray(w, float).reshape(-1, 3)
    cen = np.einsum('kij,j->ki', W, np.asarray(t, float)) + w
    cen = cen - np.floor(cen)
    frac = cen[:, None, :] + np.einsum('kij,nj->kni', W, off)
    return frac @ T.T


def max_atom_deviation(A, B, cell) -> float:
    """Largest distance from a molecule of either set to its best match in the other (Angstrom).

    Atoms are compared in order within a molecule, by the minimum lattice image of
    each difference; molecules are matched greedily by that distance.
    """
    T = cell_matrix(cell)
    A, B = np.asarray(A, float), np.asarray(B, float)

    def one_way(P, Q):
        df = np.linalg.solve(T, (P[:, None] - Q[None]).reshape(-1, 3).T).T
        df -= np.round(df)
        d = np.linalg.norm(df @ T.T, axis=1).reshape(len(P), len(Q), -1).max(2)
        return d.min(1).max()

    return float(max(one_way(A, B), one_way(B, A)))


class _Oracle:
    cell_matrix = staticmethod(cell_matrix)
    unit_cell = staticmethod(literal_unit_cell)
    max_atom_deviation = staticmethod(max_atom_deviation)


@pytest.fixture(scope='session')
def literal_oracle():
    return _Oracle
