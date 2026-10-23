"""Random ellipsoid sampling (axis lengths, rotations) and a Monte Carlo estimate
of the overlap volume of two ellipsoids given by row-wise semi-axis vectors."""
import torch
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F
from tqdm import tqdm

from mxtaltools.common.geometry_utils import compute_ellipsoid_volume, compute_cosine_similarity_matrix


def sample_triangular_right(n_samples, start, stop, device='cpu'):
    """Sample the right-aligned triangular distribution on [start, stop).

    Inverse-CDF sampling: returns ``start + (stop - start) * sqrt(U)`` with
    ``U ~ Uniform[0, 1)``, so the density rises linearly from 0 at ``start`` to
    its maximum at ``stop``.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    start : float
        Lower bound, where the density is 0.
    stop : float
        Upper bound, where the density is largest.
    device : str or torch.device
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        [n_samples] samples in torch's default dtype.
    """
    U = torch.rand(n_samples, device=device)
    return start + (stop - start) * torch.sqrt(U)


def generate_random_eigenvalues(num_samples, device):
    """Sample three non-increasing ellipsoid axis lengths (a, b, c) per sample.

    ``a`` is Uniform[0, 1) clipped below at 0.5 (values under 0.5 become exactly
    0.5).  ``b = a * f_b`` and ``c = b * f_c``, where ``f_b`` and ``f_c`` are
    independent :func:`sample_triangular_right` draws on [0, 1.1) clipped to
    [0.1, 1].  Hence 0 < c <= b <= a and 0.5 <= a < 1.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    device : str or torch.device
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        [num_samples, 3], columns (a, b, c).
    """

    rands = torch.rand(num_samples, device=device).clip(min=0.5)
    b_rands = sample_triangular_right(num_samples, 0, 1.1, device).clip(min=0.1, max=1)
    c_rands = sample_triangular_right(num_samples, 0, 1.1, device).clip(min=0.1, max=1)

    a = rands
    b = a * b_rands
    c = b * c_rands

    return torch.stack([a, b, c]).T


def generate_random_rotations(num_samples, device):
    """Random rotation matrices from ``scipy.spatial.transform.Rotation.random``, as a [num_samples, 3, 3] float32 tensor on ``device``."""
    return torch.tensor(R.random(num_samples).as_matrix(), device=device, dtype=torch.float32)


def compute_cosine_similarity_matrix(e1, e2):
    """Row-to-row dot products for paired batches of matrices (e.g. ellipsoid axes).

    ``out[n, i, k] = e1[n, i] . e2[n, k]``.  The rows are not normalised, so these
    are cosines only for unit rows.  Swapping ``e1`` and ``e2`` transposes the last
    two dimensions of the result.  This definition rebinds, within this module, the
    name imported from ``mxtaltools.common.geometry_utils``.

    Parameters
    ----------
    e1 : torch.Tensor
        [n, i, j].
    e2 : torch.Tensor
        [n, k, j].

    Returns
    -------
    torch.Tensor
        [n, i, k].
    """
    return torch.einsum('nij, nkj -> nik', e1, e2)


def trailing_relative_diffs(record, eps: float = 1e-5):
    """First differences of the running mean of ``record``, scaled by its overall level.

    With ``m[t] = mean(record[:t + 1])``, returns ``diff(m / (mean(m) + eps))``.

    Parameters
    ----------
    record : torch.Tensor
        [T] sequence of estimates.
    eps : float
        Added to the denominator.

    Returns
    -------
    torch.Tensor
        [T - 1].
    """
    cum_vols = torch.cumsum(record, dim=0)
    cum_iters = torch.arange(1, len(record) + 1, device=record.device)
    cum_means = cum_vols / cum_iters
    rel_diffs = torch.diff(cum_means / (cum_means.mean() + eps))
    return rel_diffs


def llipsoid_volume(e):
    """``4/3 * pi * prod_i |e_i|`` over the rows ``e_i`` of ``e`` [..., 3, 3], giving [...].

    This is the ellipsoid volume when the rows are mutually orthogonal semi-axis
    vectors; the computation is the same as
    ``mxtaltools.common.geometry_utils.compute_ellipsoid_volume``.
    """
    return 4 / 3 * torch.pi * e.norm(dim=-1).prod(dim=-1)


def compute_ellipsoid_overlap(e1,
                              e2,
                              v1_true,
                              v2_true,
                              r,
                              num_probes: int = 100000,
                              eps: float = 1e-3,
                              max_iters: int = 1000,
                              min_iters: int = 10,
                              show_tqdm: bool = False,
                              ):
    """Estimate the overlap volume of two ellipsoids by repeated Monte Carlo probing.

    Ellipsoid 1 is centred at the origin and ellipsoid 2 at ``r``; each is the set
    ``{x : x^T (e^T e)^-1 x <= 1}`` about its centre, for its [3, 3] matrix ``e`` of
    row-wise semi-axis vectors.  Each iteration draws ``num_probes`` points
    uniformly in an axis-aligned box containing both ellipsoids (half-widths
    ``abs(e).sum(0)`` about each centre) and records box-volume-scaled estimates
    of the volume of ellipsoid 1, of ellipsoid 2, and of their intersection.

    Once the iteration count exceeds ``min_iters``, iteration stops when all five
    criteria are below ``eps``: for each of the three records, the mean absolute
    value of the last ``min(10, min_iters)`` entries of
    :func:`trailing_relative_diffs`; and, for each ellipsoid, the relative error
    ``abs(mean(v_est) - v_true) / (v_true + 1e-5)`` of its mean volume estimate.
    Otherwise iteration stops after ``max_iters`` iterations.

    Parameters
    ----------
    e1 : torch.Tensor
        [3, 3] row-wise semi-axis vectors of ellipsoid 1.
    e2 : torch.Tensor
        [3, 3] row-wise semi-axis vectors of ellipsoid 2.
    v1_true : torch.Tensor or float
        Reference volume of ellipsoid 1, used only in the convergence test.
    v2_true : torch.Tensor or float
        Reference volume of ellipsoid 2, used only in the convergence test.
    r : torch.Tensor
        [3] centre of ellipsoid 2 relative to the centre of ellipsoid 1.
    num_probes : int
        Points sampled per iteration.
    eps : float
        Threshold applied to every convergence criterion.
    max_iters : int
        Maximum number of iterations.
    min_iters : int
        Convergence is tested only after more than ``min_iters`` iterations; also
        caps the trailing window at ``min(10, min_iters)``.
    show_tqdm : bool
        Show a tqdm progress bar.

    Returns
    -------
    overlap : torch.Tensor or int
        Mean of all per-iteration overlap estimates, a 0-d CPU tensor; the int
        ``0`` on the early exit described in the Notes.
    converged : bool
        Whether the convergence test passed within ``max_iters`` iterations
        (True on the early exit).

    Raises
    ------
    AssertionError
        If ``e1``, ``e2`` and ``r`` differ in length.

    Notes
    -----
    Returns ``(0, True)`` without sampling when the sum of the largest row norms
    of ``e1`` and ``e2`` is at most ``r.norm()``.

    If ``max_iters <= min_iters`` and the early exit does not apply, the
    convergence block never runs, ``ov_rec`` is never assigned, and the final
    ``return`` raises ``UnboundLocalError``.
    """
    if (e1.norm(dim=1).max() + e2.norm(dim=1).max()) <= r.norm():  # impossible for them to intersect
        return 0, True

    assert len(e1) == len(e2) == len(r)

    device = e1.device
    # e1 stays on origin
    # move e2 to point r

    A1 = torch.linalg.inv(e1.T @ e1)  # Metric tensor
    A2 = torch.linalg.inv(e2.T @ e2)  # Metric tensor

    c1 = torch.zeros(3, device=device)[None, ...]
    c2 = r[None, ...]
    centers = torch.cat([c1, c2], dim=0)
    axes = torch.stack([e1, e2])  # shape: (2, 3, 3)

    # Compute corners: c ± each axis vector
    extents = axes.abs().sum(dim=1)  # shape: (2, 3), max extent in each dim
    mins = centers - extents  # shape: (2, 3)
    maxs = centers + extents

    # Bounding box that contains both ellipsoids
    bbox_min = mins.min(dim=0).values  # shape: (3,)
    bbox_max = maxs.max(dim=0).values  # shape: (3,)

    volume_record = []
    converged = False
    iter = 0

    with tqdm(total=max_iters, desc="Optimizing", unit="iter", disable=not show_tqdm) as pbar:
        while not converged and iter < max_iters:
            iter += 1

            points = bbox_min + (bbox_max - bbox_min) * torch.rand((num_probes, 3), device=device)
            bounding_volume = torch.prod(bbox_max - bbox_min)

            # check which points are inside each ellipsoid
            dx1 = points - c1
            in_e1 = (torch.einsum('ni,ij,nj->n', dx1, A1, dx1) <= 1).float()
            dx2 = points - c2
            in_e2 = (torch.einsum('ni,ij,nj->n', dx2, A2, dx2) <= 1).float()

            # Estimate ellipsoid volume
            v1 = bounding_volume * in_e1.mean()
            v2 = bounding_volume * in_e2.mean()
            v_ov = bounding_volume * (in_e1 * in_e2).mean()
            volume_record.append([float(v1), float(v2), float(v_ov)])

            if iter > min_iters:
                recs = torch.tensor(volume_record)
                v1_rec = recs[:, 0]
                v2_rec = recs[:, 1]
                ov_rec = recs[:, 2]

                # stop iterating when the mean estimates are stable
                # AND when the single volume estimates are accurate

                # stable overlap estimate
                v1_relative_diffs = trailing_relative_diffs(v1_rec)
                v2_relative_diffs = trailing_relative_diffs(v2_rec)
                ov_relative_diffs = trailing_relative_diffs(ov_rec)

                criteria1 = v1_relative_diffs[-min(10, min_iters):].abs().mean()
                criteria2 = v2_relative_diffs[-min(10, min_iters):].abs().mean()
                criteria3 = ov_relative_diffs[-min(10, min_iters):].abs().mean()
                criteria4 = ((v1_rec.mean() - v1_true).abs() / (v1_true + 1e-5))
                criteria5 = ((v2_rec.mean() - v2_true).abs() / (v2_true + 1e-5))
                criteria = torch.tensor([criteria1, criteria2, criteria3, criteria4, criteria5])
                if torch.all(criteria < eps):  # relative change in running average less than eps
                    converged = True

                pbar.set_description(f"Iter {iter} | loss={criteria.mean():.4f}")

            pbar.update(1)

    if not converged:
        aa = 1

    return ov_rec.mean(), converged

    """
    # watch mean convergence

    record = ov_rec
    cum_vols = torch.cumsum(record, dim=0)
    cum_iters = torch.arange(1, len(record) + 1, device=record.device)
    cum_means = cum_vols / cum_iters
    rel_diffs = torch.diff(cum_means / (cum_means.mean() + eps))

    import plotly.graph_objects as go
    go.Figure(go.Scatter(y=rel_diffs.abs().cpu().detach())).show()
    """

