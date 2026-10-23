"""Shared helpers for the fundamental-domain gates 2-8 (not collected: the name does not start with ``test_``).

- ``physical_cells``: reference triclinic cell samplers, independent of the charts: rejection in coordinates where
  the cell measure (times the V of V d^3 t) is Lebesgue, restricted to the margin-0 zero set of
  ``sym_utils.tri_niggli_reduction_penalty`` and a length box.
- ``scramble`` / ``canonicalise``: a random proper basis change, then ``reduce_cell`` and ``fold``.
- ``chart_draws``: draws from exp(log_reference_density) on a CrystalChart, by rejection from a proposal with a
  known density (the proposal only shapes efficiency; the target is the package's density).
- ``crystal_batch``: a MolCrystalData batch built from descriptions, centroids wrapped to [0, 1) first.
"""
import math

import numpy as np
import torch

from mxtaltools.common.sym_utils import tri_niggli_reduction_penalty
from mxtaltools.crystal_fd import domains, measure, ops
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef

F64 = torch.float64
#: every rejection loop stops after this many chunks and raises (a loop that cannot fill must not spin)
MAX_CHUNKS = 200
ALPHA_RANGE = (math.radians(60.0), math.radians(120.0))   # contains the zero set's alpha
OBTUSE_RANGE = (math.radians(90.0), math.radians(120.0))  # contains its beta and gamma


def mol_ref(mol: dict) -> MoleculeRef:
    return MoleculeRef.from_coords(mol['pos'], mol['z'])


def on_zero_set(cells: torch.Tensor) -> torch.Tensor:
    """[n] True where the live triclinic penalty at margin 0 is exactly zero."""
    c = torch.as_tensor(cells, dtype=F64)
    return tri_niggli_reduction_penalty(c[:, :3], c[:, 3:], 0.0) == 0


def in_box(cells, lo, hi) -> torch.Tensor:
    L = torch.as_tensor(cells, dtype=F64)[:, :3]
    return ((L >= lo) & (L <= hi)).all(1)


def _reduced_ranges(lo, hi):
    """Per-length ranges that contain every a <= b <= c in the box."""
    a = (lo[0], min(hi[0], hi[1], hi[2]))
    b = (max(lo[0], lo[1]), min(hi[1], hi[2]))
    c = (max(lo[0], lo[1], lo[2]), hi[2])
    return a, b, c


def physical_cells(lo, hi, n: int, cell_measure: str, gen: torch.Generator, chunk: int = 400_000) -> torch.Tensor:
    """[n, 6] cells with density mu_cell x V on (tri_niggli zero set) x (length box), independent of the charts.

    ``'latent_v1_lebesgue'``: a, b, c uniform in Angstrom (density abc in log L), angles uniform on a box holding
    the zero set, accepted with probability V / (abc). ``'metric_invariant'``: mu_cell x V = dG, uniform on the
    Gram entries: A, B, C uniform, G_ab, G_ac uniform on [-A/2, 0] and G_bc on [-B/2, B/2] given them, accepted with
    probability A^2 B / (A_max^2 B_max) (the inverse proposal density).
    """
    lo = [float(v) for v in torch.as_tensor(lo, dtype=F64).reshape(3)]
    hi = [float(v) for v in torch.as_tensor(hi, dtype=F64).reshape(3)]
    ra, rb, rc = _reduced_ranges(lo, hi)
    out, have = [], 0
    for _ in range(MAX_CHUNKS):
        if have >= n:
            break
        u = torch.rand(chunk, 7, generator=gen, dtype=F64)
        if cell_measure == 'latent_v1_lebesgue':
            L = torch.stack([r[0] + (r[1] - r[0]) * u[:, k] for k, r in enumerate((ra, rb, rc))], 1)
            ang = torch.stack([ALPHA_RANGE[0] + (ALPHA_RANGE[1] - ALPHA_RANGE[0]) * u[:, 3],
                               OBTUSE_RANGE[0] + (OBTUSE_RANGE[1] - OBTUSE_RANGE[0]) * u[:, 4],
                               OBTUSE_RANGE[0] + (OBTUSE_RANGE[1] - OBTUSE_RANGE[0]) * u[:, 5]], 1)
            cells = torch.cat([L, ang], 1)
            ok = (L[:, 0] <= L[:, 1]) & (L[:, 1] <= L[:, 2])
            vol = measure.cell_volume(torch.where(ok[:, None], cells, torch.tensor([1., 1, 1, 1.5, 1.6, 1.7],
                                                                                   dtype=F64)))
            accept = u[:, 6] < vol / L.prod(1)
        elif cell_measure == 'metric_invariant':
            A = ra[0] ** 2 + (ra[1] ** 2 - ra[0] ** 2) * u[:, 0]
            B = rb[0] ** 2 + (rb[1] ** 2 - rb[0] ** 2) * u[:, 1]
            C = rc[0] ** 2 + (rc[1] ** 2 - rc[0] ** 2) * u[:, 2]
            Gab = -A / 2 * u[:, 3]
            Gac = -A / 2 * u[:, 4]
            Gbc = B * (u[:, 5] - 0.5)
            G = torch.stack([A, Gab, Gac, Gab, B, Gbc, Gac, Gbc, C], 1).reshape(-1, 3, 3)
            ok = (A <= B) & (B <= C) & (torch.linalg.det(G) > 0)
            cells = ops.cell_from_metric_tensor(torch.where(ok[:, None, None], G, torch.eye(3, dtype=F64)))
            accept = u[:, 6] < A ** 2 * B / (ra[1] ** 4 * rb[1] ** 2)
        else:
            raise ValueError(cell_measure)
        keep = ok & accept & in_box(cells, torch.tensor(lo, dtype=F64), torch.tensor(hi, dtype=F64))
        keep &= on_zero_set(torch.where(keep[:, None], cells, cells[:1]))
        out.append(cells[keep])
        have += int(keep.sum())
    if have < n:
        raise RuntimeError(f'physical_cells: {have} of {n} cells after {MAX_CHUNKS} chunks; the box holds too little')
    return torch.cat(out)[:n]


def random_unimodular(n: int, rng: np.random.Generator, span: int = 2) -> torch.Tensor:
    """[n, 3, 3] long integer matrices with entries in [-span, span] and det +1."""
    out = []
    for _ in range(MAX_CHUNKS):            # about 1 in 6 draws has det +1 at span 2
        if sum(len(o) for o in out) >= n:
            return torch.as_tensor(np.concatenate(out)[:n])
        M = rng.integers(-span, span + 1, (4 * n, 3, 3))
        out.append(M[np.round(np.linalg.det(M)) == 1])
    raise RuntimeError('random_unimodular: too few det +1 draws')


def scramble(desc: CrystalDescription, rng: np.random.Generator) -> CrystalDescription:
    """The same triclinic crystals in a random proper basis (no origin shift: every det +1 change keeps P1, P-1)."""
    N = random_unimodular(desc.num_rows, rng)
    return ops.change_basis(desc, N, torch.zeros(desc.num_rows, 3, dtype=F64))


def canonicalise(desc: CrystalDescription):
    """reduce_cell, then fold: (canonical descriptions, CellReduction, FoldResult)."""
    red = ops.reduce_cell(desc)
    fr = domains.fold(red.desc)
    return fr.desc, red, fr


def _proposal(chart, m: int, gen, exps):
    """Draws y on a CrystalChart and their log proposal density.

    Cell rows 0-2 (log c, r2, r1) are exponential / power laws with exponents ``exps`` on their AR bounds, the
    other rows uniform on theirs.
    """
    u = torch.rand(m, chart.ndim, generator=gen, dtype=F64).clamp(1e-15, 1 - 1e-15)
    cols, logq = [], torch.zeros(m, dtype=F64)
    for i in range(chart.ndim):
        prefix = torch.stack(cols, 1) if cols else u.new_zeros(m, 0)
        lo, hi = chart.bounds(prefix, i)
        k = exps[i] if i < 3 else None
        if i == 0:
            span = torch.expm1(k * (hi - lo))
            y = lo + torch.log1p(u[:, i] * span) / k
            logq = logq + math.log(k) + k * (y - lo) - torch.log(span)
        elif i in (1, 2):
            a, b = lo ** (k + 1), hi ** (k + 1)
            y = (a + u[:, i] * (b - a)) ** (1 / (k + 1))
            logq = logq + math.log(k + 1) + k * torch.log(y) - torch.log(b - a)
        else:
            y = lo + u[:, i] * (hi - lo)
            logq = logq - torch.log(hi - lo)
        cols.append(y)
    return torch.stack(cols, 1), logq


#: exponents of the cell proposal per measure: the target's density in (log c, r2, r1) up to bounded factors
PROPOSAL_EXPONENTS = {'latent_v1_lebesgue': (3.0, 3.0, 2.0), 'metric_invariant': (12.0, 9.0, 5.0)}


def chart_draws(chart, n: int, cell_measure: str, gen: torch.Generator, chunk: int = 200_000) -> torch.Tensor:
    """[n, ndim] draws from exp(log_reference_density(chart.forward(y).desc, chart, cell_measure)) on a CrystalChart.

    Rejection from ``_proposal``: the bound is a pilot's largest log weight plus log 2, and every later batch is
    checked against it (an AssertionError means the bound was too small, never a silent bias).
    """
    exps = PROPOSAL_EXPONENTS[cell_measure]

    def logw(m):
        y, logq = _proposal(chart, m, gen, exps)
        lp = measure.log_reference_density(chart.forward(y).desc, chart, cell_measure)
        return y, lp - logq

    _, w0 = logw(chunk)
    bound = float(w0[torch.isfinite(w0)].max()) + math.log(2.0)
    out, have = [], 0
    for _ in range(MAX_CHUNKS):
        if have >= n:
            break
        y, w = logw(chunk)
        assert float(w[torch.isfinite(w)].max()) <= bound, 'rejection bound exceeded; raise the pilot margin'
        keep = torch.rand(chunk, generator=gen, dtype=F64) < torch.exp(w - bound)
        out.append(y[keep])
        have += int(keep.sum())
    if have < n:
        raise RuntimeError(f'chart_draws: {have} of {n} draws after {MAX_CHUNKS} chunks; acceptance too low')
    return torch.cat(out)[:n]


#: the live builder clips fractional centroids to [0, 1 - CLIP] (crystal_ops.assign_aunit_centroid)
CLIP = 1e-4


def crystal_batch(desc: CrystalDescription, mol: MoleculeRef, wrap: bool = True):
    """MolCrystalData batch of Z' = 1 rows, centroids wrapped to [0, 1) by lattice translation before building.

    Raises if a wrapped centroid lies in (1 - CLIP, 1), where the builder's clip would move it. ``wrap=False`` is
    for controls only.
    """
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    d = desc.double()
    if wrap:
        d = d.replace(t=d.t - torch.floor(d.t))
        if bool((d.t > 1 - CLIP).any()):
            raise ValueError('a wrapped centroid lies within the builder clip band (1 - 1e-4, 1)')
    L, A, c, rv, h = d.crystal_params()
    rows = []
    for i in range(d.num_rows):
        m = MolData(z=mol.z.clone(), pos=mol.pos.float().clone(), x=torch.zeros(len(mol.z), 1),
                    do_mol_analysis=True)
        rows.append(MolCrystalData(molecule=m, sg_ind=int(d.sg[i]), cell_lengths=L[i].float(),
                                   cell_angles=A[i].float(), aunit_centroid=c[i].float(),
                                   aunit_orientation=rv[i].float(), aunit_handedness=h[i].float(),
                                   do_box_analysis=True))
    return collate_data_list(rows)


def elj(desc: CrystalDescription, mol: MoleculeRef, chunk: int = 256, wrap: bool = True) -> torch.Tensor:
    """[B] eLJ through MolCrystalData.analyze(['elj']) on CPU (the trainer's kwargs: cutoff 10, supercell 10)."""
    out = []
    for lo in range(0, desc.num_rows, chunk):
        b = crystal_batch(desc[lo:lo + chunk], mol, wrap=wrap)
        with torch.no_grad():
            out.append(b.analyze(['elj'], cutoff=10, supercell_size=10, std_orientation=False)['elj'].double())
    return torch.cat(out)
