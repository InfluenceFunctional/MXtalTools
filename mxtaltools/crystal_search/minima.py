"""Converged minima of a search stage's energy, their Hessians, and capture of descents by known minima.

Opt-in through the search config's ``minima`` block (``run_search.check_minima_config``); nothing here runs otherwise.

A stage returns each row's best state, not a converged minimum, and starts that fall into one basin return several
nearby states. ``finish_minima`` takes the rows of a search's last stage to converged minima (``newton_polish``),
takes the Hessian of E/kT at each, and merges them into a ``MinimaLibrary``. During the last stage of the next batch,
``gradient_descent_optimization(capture_library=...)`` retires a descent once it enters the thermal ellipsoid
d^2 = dx^T H dx < capture_d2 of a known minimum, and ``finish_minima`` reports it as that minimum.

The objective is the stage target's ``analyze`` compute alone, divided by kT: no auxiliary term (compression, box,
packing) and no reduced-cell wall. With ``lj_envelope`` the LJ-family computes are switched off smoothly over the last
``lj_envelope`` A before the cutoff, so the double-backward Hessian exists; the energies a row reports are re-analysed
without it, on the hard cutoff.

Coordinates are a stage's full cell parameters (``full_cell_parameters``: lengths in A, angles in rad, fractional
centres, rotation vectors in rad) in wrap mode (``crystal_opt_utils.wrap_centroid``). One crystal has many such
descriptions (cell settings, symmetry images, handedness under an improper element), so minima are identified, and
descents captured, in one description per crystal (``identity_description``: crystal_fd's canonical form at Z' = 1),
with each minimum's Hessian taken in that description; rows are reported in the asymmetric-unit box
(``canonicalize_aunit``) with the Hessian taken there. d^2 removes the centres' lattice translations (``min_image``)
and the shifts to an equivalent origin (``centre_shifts``: in P-1, half a cell along any axis). Near a face of the
reduced-cell domain two descriptions of one crystal can still differ (two bases nearly equally reduced, the reduction
picking either); ``MinimaLibrary.match`` therefore also merges a new Z' = 1 minimum into a known one of nearly the same
energy within ``FD_MERGE_A`` of it in the glued FD distance (``crystal_fd.metric.GluedFDMetric``, which minimises over
each crystal's near-minimal bases). Capture keeps the ellipsoid test alone.

Termination: ``newton_polish`` evaluates at most ``max_iter + 1`` Hessians and ``max_iter * POLISH_TRIES`` energies
per row and has no other loop; ``finish_minima`` calls it once.
"""
import functools
from typing import Optional

import torch

from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit, fd_fold_aunit, wrap_centroid
from mxtaltools.dataset_utils.utils import collate_data_list

ENERGY_TARGETS = ('lj', 'qlj', 'elj', 'exp6', 'silu', 'uma', 'mace')  # stage targets whose compute is an energy
MLIP_TARGETS = ('uma', 'mace')  # Hessians by central differences of the gradient (fd_step): MACE's TorchScript ops
# refuse the batched double backward, and d sequential ones cost ~2.4x more and twice the memory (MACE-OFF24 on MIPCAS,
# 2026-10-30: 0.77 s per row for 12 passes, 8 rows at most under 6.4 GB; differences agree to 0.2%)
FD_STEP = 1e-3  # central-difference step of the MLIP Hessian, in each coordinate's own unit (A, rad, fraction)
POLISH_CAPS = (0.1, 0.02, 0.01, 0.02)  # largest step per block: lengths (A), angles (rad), centres, rotvecs (rad)
POLISH_TRIES = 8  # Levenberg-Marquardt damping increases tried per iteration before a row counts as stalled
TRUST_RANGE = (1 / 16, 16.0)  # bounds of the per-row factor on POLISH_CAPS (doubled on an accepted step, halved else)
POLISH_STATUS = {0: 'captured (in the stage or the polish)', 1: 'converged', 2: 'stalled', 3: 'iteration cap',
                 4: 'left the cell-parameter box'}
NEG_REL = 1e-6  # a Hessian counts as positive semi-definite when its lowest eigenvalue is above -NEG_REL * its highest
FLAT_REL = 1e-9  # an eigen-direction whose |curvature| is below this times the largest is flat (the energy ignores it)
FLAT_GRAD = 1e-3  # E/kT per unit coordinate: a flat direction with a smaller gradient component is converged along it
FD_MERGE_A = 0.05  # glued FD distance (A) under which a new minimum is a known one (crystal_fd.metric's identity cut)
FD_MERGE_KT = 0.01  # E/kT window around a new minimum inside which known minima are measured by the glued FD distance
# (copies of one minimum differ by ~1e-4: the polish's tolerance; NEHZOR's cell-face pair, 2026-10-30)
F64 = torch.float64


def blocks(d: int):
    """Column slices of a full cell parameter vector of width d = 6 + 6 Z': lengths, angles, centres, rotvecs."""
    nz = (d - 6) // 6
    return slice(0, 3), slice(3, 6), slice(6, 6 + 3 * nz), slice(6 + 3 * nz, 6 + 6 * nz)


def min_image(dx: torch.Tensor) -> torch.Tensor:
    """Parameter differences [..., d] with each fractional centre component moved to [-0.5, 0.5) (a lattice
    translation is the same crystal). Other columns unchanged."""
    dx = dx.clone()
    cen = blocks(dx.shape[-1])[2]
    dx[..., cen] = dx[..., cen] - torch.round(dx[..., cen])
    return dx


@functools.lru_cache(maxsize=None)
def centre_shifts(sg: int) -> torch.Tensor:
    """[k, 3] float64: the translations t in [0, 1)^3, zero first, for which moving every centre by t describes the
    same crystal of space group ``sg``: the conventional cell's centring and the Euclidean normalizer's pure
    translations (an equivalent origin, such as the eight inversion centres of P-1). They are the products with
    identity rotation of a normalizer coset representative (``NORMALIZER_OPS``, or the identity) and a group operator
    (``SYM_OPS``). A translation that normalizes the group is an isometry, so the set is the translation part of the
    affine normalizer as well, and every cell setting of the group (the identity description's reduced cells
    included) has the same set. Continuous origin shifts (``CONTINUOUS_DIMS``, polar axes) are not listed: along
    them a minimum's Hessian is flat."""
    import numpy as np
    from mxtaltools.constants.space_group_info import NORMALIZER_OPS, SYM_OPS
    reps = [(np.eye(3), np.zeros(3))] + [(np.asarray(W, dtype=float), np.asarray(w, dtype=float))
                                         for W, w in NORMALIZER_OPS[str(int(sg))]]
    out = {(0.0, 0.0, 0.0)}
    for W, w in reps:
        for op in SYM_OPS[int(sg)]:
            if np.allclose(W @ op[:3, :3], np.eye(3)):
                t = np.round((W @ op[:3, 3] + w) % 1.0, 6) % 1.0
                out.add(tuple(float(v) for v in t))
    return torch.tensor(sorted(out), dtype=F64)


def centre_images(key: tuple, d: int) -> torch.Tensor:
    """[k, d] float64: ``centre_shifts`` of the key's space group (``row_key``: molecule, space group, Z', ...) as
    full-parameter offsets, the same shift on every unit's centre and zero elsewhere; the zero offset first."""
    sh = centre_shifts(int(key[1]))
    out = torch.zeros(len(sh), d, dtype=F64)
    out[:, blocks(d)[2]] = sh.repeat(1, (d - 6) // 6)
    return out


def ellipsoid_factor(H: torch.Tensor) -> torch.Tensor:
    """W [..., d, d] with d^2 = |W^T dx|^2 = dx^T H_+ dx, H_+ being H with negative eigenvalues set to 0 (a flat
    direction, such as a free centre axis in a polar group, adds nothing)."""
    lam, V = torch.linalg.eigh(H.to(F64))
    return V * lam.clamp_min(0).sqrt().unsqueeze(-2)


def is_psd(H: torch.Tensor) -> torch.Tensor:
    """[...] bool: lowest eigenvalue above -NEG_REL times the highest."""
    lam = torch.linalg.eigvalsh(H.to(F64))
    return lam[..., 0] > -NEG_REL * lam[..., -1].abs()


class StageEnergy:
    """E/kT of a search stage's target at given full cell parameters, row by row.

    Each call builds the crystals as a ``gradient_descent_optimization`` step does in wrap mode -- ``set_cell_parameters``,
    ``clean_cell_parameters(mode='hard', canonicalize_orientations=False)``, centres through ``wrap_centroid`` -- on
    collated copies of ``rows`` (their molecules, space groups and handedness are held fixed), and returns
    ``analyze(computes=[target])[target] / kT``.

    Parameters
    ----------
    rows : list of MolCrystalData
        One crystal per row; ``x`` passed to the methods has one parameter row per crystal (``idx`` selects them).
    optim_target : str
        A stage target in ENERGY_TARGETS.
    kT : float
        In the target's units.
    lj_envelope : float, optional
        Forwarded to ``analyze`` for 'lj', 'qlj' and 'elj' only.
    chunk : int
        Rows per ``analyze`` call.
    hessian : str
        'autograd' (the batched double backward) or 'fd' (central differences of the gradient, ``FD_STEP``); default
        'fd' for ``MLIP_TARGETS``, 'autograd' otherwise.
    """

    def __init__(self, rows, optim_target: str, kT: float, cutoff: float = 10, supercell_size: int = 10,
                 predictor=None, lj_envelope: Optional[float] = None, exp6_kwargs: Optional[dict] = None,
                 device='cpu', chunk: int = 256, hessian: Optional[str] = None):
        key = str(optim_target).lower()
        if key not in ENERGY_TARGETS:
            raise ValueError(f"minima need an energy target ({ENERGY_TARGETS}), got {optim_target!r}")
        if not kT > 0:
            raise ValueError(f"kT must be positive, got {kT!r}")
        self.rows, self.key, self.kT, self.device, self.chunk = list(rows), key, float(kT), device, int(chunk)
        self.hessian = hessian or ('fd' if key in MLIP_TARGETS else 'autograd')
        if self.hessian not in ('autograd', 'fd'):
            raise ValueError(f"hessian must be 'autograd' or 'fd', got {hessian!r}")
        self.kwargs = dict(cutoff=cutoff, rdf_cutoff=cutoff, supercell_size=supercell_size, return_cluster=False,
                           repulsion=1, surface_padding=0, std_orientation=True, margin=0, predictor=predictor,
                           assign_outputs=False, **{f'exp6_{k}': v for k, v in (exp6_kwargs or {}).items()})
        if lj_envelope is not None and key in ('lj', 'qlj', 'elj'):
            self.kwargs['lj_envelope'] = float(lj_envelope)

    def _energy(self, x: torch.Tensor, idx) -> torch.Tensor:
        b = collate_data_list([self.rows[int(i)] for i in idx]).to(self.device)
        b.set_cell_parameters(x, skip_box_analysis=True)
        b.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
        b.aunit_centroid = wrap_centroid(x[:, blocks(x.shape[1])[2]])
        return b.analyze(computes=[self.key], **self.kwargs)[self.key].reshape(-1) / self.kT

    def _chunks(self, n):
        for c in range(0, n, self.chunk):
            yield slice(c, min(c + self.chunk, n))

    @torch.no_grad()
    def f(self, x: torch.Tensor, idx) -> torch.Tensor:
        """[n] float64 E/kT at the rows ``idx`` with parameters ``x`` [n, d]."""
        return torch.cat([self._energy(x[s].to(self.device, torch.float32), idx[s]).double().cpu()
                          for s in self._chunks(len(x))])

    def fg(self, x: torch.Tensor, idx):
        """E/kT [n] and its gradient [n, d], float64 on CPU (one backward pass)."""
        F, G = [], []
        for s in self._chunks(len(x)):
            xi = x[s].to(self.device, torch.float32).clone().requires_grad_(True)
            fi = self._energy(xi, idx[s])
            gi, = torch.autograd.grad(fi.sum(), xi)
            F.append(fi.detach().double().cpu())
            G.append(gi.detach().double().cpu())
        return torch.cat(F), torch.cat(G)

    def fgh(self, x: torch.Tensor, idx):
        """E/kT [n], its gradient [n, d] and its Hessian [n, d, d] (symmetrised), float64 on CPU. The Hessian is the
        double backward of the batch sum: rows share no parameter, so row k of every row's Hessian is the gradient of
        the sum over rows of gradient component k, and the d of them come from one backward batched over the d unit
        vectors (``is_grads_batched``: for 256 MIPCAS rows on a GPU 0.57 s against 1.2 s for d sequential passes, the
        same values to 3e-6, 2026-10-29). With ``hessian='fd'``: (g(x + h e_k) - g(x - h e_k)) / 2h for each
        coordinate k, h = ``FD_STEP``, all 2d displaced copies evaluated together in ``fg``'s chunks."""
        if self.hessian == 'fd':
            n, d = x.shape
            F, G = self.fg(x, idx)
            shift = FD_STEP * torch.eye(d, dtype=x.dtype)
            xs = torch.cat([x[:, None] + shift[None], x[:, None] - shift[None]], 1).reshape(-1, d)  # [n * 2d, d]
            _, Gs = self.fg(xs, torch.as_tensor(idx).repeat_interleave(2 * d))
            Gs = Gs.reshape(n, 2, d, d)
            H = ((Gs[:, 0] - Gs[:, 1]) / (2 * FD_STEP)).to(F64)  # [n, k, :] = d g / d x_k
            return F, G, 0.5 * (H + H.transpose(1, 2))
        F, G, H = [], [], []
        d = x.shape[1]
        for s in self._chunks(len(x)):
            xi = x[s].to(self.device, torch.float32).clone().requires_grad_(True)
            fi = self._energy(xi, idx[s])
            gi, = torch.autograd.grad(fi.sum(), xi, create_graph=True)
            Hi = torch.zeros(len(xi), d, d, dtype=F64)
            if gi.requires_grad:
                eye = torch.eye(d, device=gi.device, dtype=gi.dtype).unsqueeze(1).expand(d, len(xi), d)
                hb, = torch.autograd.grad(gi, xi, grad_outputs=eye, is_grads_batched=True, allow_unused=True)
                if hb is not None:  # [d, n, d] -> [n, d, d]
                    Hi = hb.detach().permute(1, 0, 2).double().cpu()
            F.append(fi.detach().double().cpu())
            G.append(gi.detach().double().cpu())
            H.append(0.5 * (Hi + Hi.transpose(1, 2)))
        return torch.cat(F), torch.cat(G), torch.cat(H)


def newton_polish(energy: StageEnergy, x0: torch.Tensor, max_iter: int = 40, tol: float = 1e-4,
                  hessian_every: int = 1, capture=None, capture_d2: float = 9.0, keys: Optional[list] = None,
                  merge_d2: float = 1.0):
    """Per-row saddle-free Levenberg-Marquardt Newton descent of ``energy`` from ``x0`` [n, d].

    With ``capture``, a callable taking (x [m, d], row indices [m]) and returning (minimum id [m], d^2 [m]) as
    ``MinimaLibrary.capture_query`` does, every active row is tested before each iteration's evaluations (and before
    the first): a row with d^2 < ``capture_d2`` stops there with status 0 and that id in ``capture``.

    With ``keys`` (one hashable per row, e.g. ``identity_key``) and ``capture_d2`` > 0, rows that converged earlier in
    this call capture too: each converged row not within ``merge_d2`` of an earlier one (under its own ellipsoid, at
    its fresh Hessian) becomes a capture target, and before each iteration an active row of the same key within d^2
    < ``capture_d2`` of a target, measured in this call's coordinates (``MinimaLibrary.nearest``), stops with status
    0 and that row's index in ``sibling``. On MIPCAS eLJ (2026-10-29) the first group of a run, polished against an
    empty library, gave 997 converged polishes for 143 new minima, later groups 34 for 27; with this capture the
    same first 1,000 rows gave 525 converged polishes, a fifth fewer Hessians and the same minima.

    Each iteration takes every active row's gradient, and its Hessian H = V diag(lam) V^T when the row's last one is
    ``hessian_every`` iterations old or the row asked for a fresh one; in between, the row steps with its last H.
    The default, 1, takes a Hessian every iteration: on a GPU batch the polish's wall time is its sequential depth,
    and reuse (3 or 5) needed more iterations, about 1.7x the wall time on MIPCAS eLJ for 8% fewer row-evaluations,
    and left some rows stalled or capped (2026-10-28).
    A row stops when its Hessian is fresh (taken at the current x), positive semi-definite (``is_psd``), and the
    Newton decrement 0.5 sum_i (v_i . g)^2 / |lam_i| is below ``tol`` (kT); a decrement below ``tol`` under an older
    Hessian asks for a fresh one at the next iteration instead of a step. Flat directions -- |lam_i| below ``FLAT_REL``
    times the largest, along which the energy does not change: angles a crystal system fixes, and a layer's spacing
    and offsets across a vacuum gap wider than the cutoff -- are left out of the decrement and need only
    |v_i . g| < ``FLAT_GRAD`` (on NEHZOR eLJ, 2026-10-30, 210 of 211 stalled polishes were such layers, b 52-153 A,
    whose decrement was all on directions of curvature ~1e-7 against a largest ~1e5, and whose trial steps changed the
    energy by less than its evaluation noise, ~1e-4 kT). Otherwise it tries
    x - V diag(1 / (|lam| + mu)) V^T g, scaled so no block moves more than its POLISH_CAPS entry times the row's
    trust factor, and keeps it if E/kT falls (mu /= 3, floored at 1e-4; trust doubled up to TRUST_RANGE[1]), else
    mu *= 4 and trust is halved (down to TRUST_RANGE[0]) and it tries again, at most POLISH_TRIES times; a row with
    no accepted step is stalled. Rows still active after ``max_iter`` iterations stop at the cap. Stalled and
    capped rows get a fresh Hessian at their final x.

    Returns
    -------
    dict
        ``x`` [n, d], ``f`` [n] (E/kT), ``g`` [n, d], ``H`` [n, d, d] at ``x`` (float64, CPU; for a row captured
        before its first evaluation, f is NaN and g and H are 0); ``status`` [n] long (0 captured, 1 converged,
        2 stalled, 3 iteration cap: POLISH_STATUS); ``capture`` [n] long (the capturing minimum, else -1);
        ``sibling`` [n] long (the converged row of this call that captured the row, else -1);
        ``n_f``, ``n_g``, ``n_h`` [n] long, the energies, gradients and Hessians evaluated per row (a Hessian
        evaluation includes its energy and gradient).
    """
    x = x0.detach().to(F64).cpu().clone()
    n, d = x.shape
    cap = torch.zeros(d, dtype=F64)
    for sl, c in zip(blocks(d), POLISH_CAPS):
        cap[sl] = c
    status = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    cap_id = torch.full((n,), -1, dtype=torch.long)
    sib_id = torch.full((n,), -1, dtype=torch.long)
    targets = MinimaLibrary() if keys is not None and capture_d2 > 0 else None  # this call's converged rows
    target_row = []  # target id -> row index

    def captured(ia):
        """Rows of ``ia`` inside a known minimum's or a converged row's ellipsoid: marked done (status 0); returns
        the others."""
        if capture is not None and len(ia):
            mid, d2 = capture(x[ia], ia)
            hit = (mid >= 0) & (d2 < capture_d2)
            done[ia[hit]] = True
            cap_id[ia[hit]] = mid[hit]
            ia = ia[~hit]
        if targets is not None and len(targets) and len(ia):
            tid, d2 = targets.nearest(x[ia], [keys[int(i)] for i in ia])
            hit = (tid >= 0) & (d2 < capture_d2)
            done[ia[hit]] = True
            sib_id[ia[hit]] = torch.tensor(target_row, dtype=torch.long)[tid[hit]]
            ia = ia[~hit]
        return ia

    def add_targets(rows_conv):
        """Converged rows (fresh positive semi-definite Hessian at x) as capture targets, one per minimum."""
        if targets is None:
            return
        for i in rows_conv.tolist():
            if targets.match(x[i], ellipsoid_factor(H[i]), keys[i], merge_d2) < 0:
                targets.add(x[i], H[i], float(F[i]), keys[i], {})
                target_row.append(i)

    F = torch.full((n,), float('nan'), dtype=F64)
    G = torch.zeros(n, d, dtype=F64)
    H = torch.zeros(n, d, d, dtype=F64)
    lam_all = torch.ones(n, d, dtype=F64)
    V_all = torch.eye(d, dtype=F64).expand(n, d, d).clone()
    n_h = torch.zeros(n, dtype=torch.long)
    live = captured(torch.arange(n))
    if len(live):
        F[live], G[live], H[live] = energy.fgh(x[live], live)
        lam_all[live], V_all[live] = torch.linalg.eigh(H[live])
        n_h[live] = 1
    age = torch.zeros(n, dtype=torch.long)  # iterations since the row's Hessian was taken at its current x
    want = torch.zeros(n, dtype=torch.bool)
    n_g = torch.zeros(n, dtype=torch.long)
    n_f = torch.zeros(n, dtype=torch.long)
    mu = torch.ones(n, dtype=F64)
    trust = torch.ones(n, dtype=F64)
    for it in range(max_iter):
        ia = torch.nonzero(~done).flatten()
        if it > 0:
            ia = captured(ia)
        if len(ia) == 0:
            break
        if it > 0:
            fresh = ia[(age[ia] >= hessian_every) | want[ia]]
            stale = ia[(age[ia] < hessian_every) & ~want[ia]]
            if len(fresh):
                F[fresh], G[fresh], H[fresh] = energy.fgh(x[fresh], fresh)
                lam_all[fresh], V_all[fresh] = torch.linalg.eigh(H[fresh])
                n_h[fresh] += 1
                age[fresh] = 0
                want[fresh] = False
            if len(stale):
                F[stale], G[stale] = energy.fg(x[stale], stale)
                n_g[stale] += 1
        lam, V = lam_all[ia], V_all[ia]
        gv = torch.einsum('nij,ni->nj', V, G[ia])
        flat = lam.abs() < FLAT_REL * lam.abs().amax(1, keepdim=True)
        dec = 0.5 * (gv ** 2 / lam.abs().clamp_min(1e-12)).masked_fill(flat, 0.0).sum(1)
        dec = torch.where((gv.abs().masked_fill(~flat, 0.0) < FLAT_GRAD).all(1), dec, torch.full_like(dec, float('inf')))
        is_fresh = age[ia] == 0
        conv = is_fresh & is_psd(H[ia]) & (dec < tol)
        done[ia[conv]] = True
        status[ia[conv]] = 1
        add_targets(ia[conv])
        ask = ~is_fresh & (dec < tol)  # small under an old Hessian: confirm with a fresh one, no step
        want[ia[ask]] = True
        age[ia] += 1
        keep = ~conv & ~ask
        ia, lam, V, gv = ia[keep], lam[keep], V[keep], gv[keep]
        pending = torch.ones(len(ia), dtype=torch.bool)
        for _ in range(POLISH_TRIES):
            pj = torch.nonzero(pending).flatten()
            if len(pj) == 0:
                break
            r = ia[pj]
            step = -torch.einsum('nij,nj->ni', V[pj], gv[pj] / (lam[pj].abs() + mu[r, None]))
            scale = (trust[r, None] * cap / step.abs().clamp_min(1e-300)).amin(1).clamp(max=1.0)
            xn = x[r] + scale[:, None] * step
            fn = energy.f(xn, r)
            n_f[r] += 1
            acc = torch.isfinite(fn) & (fn < F[r])
            x[r[acc]], F[r[acc]] = xn[acc], fn[acc]
            mu[r[acc]] = (mu[r[acc]] / 3).clamp_min(1e-4)
            mu[r[~acc]] *= 4
            trust[r[acc]] = (trust[r[acc]] * 2).clamp(max=TRUST_RANGE[1])
            trust[r[~acc]] = (trust[r[~acc]] / 2).clamp(min=TRUST_RANGE[0])
            pending[pj[acc]] = False
        stalled = ia[pending]
        done[stalled] = True
        status[stalled] = 2
    capped = torch.nonzero(~done).flatten()
    status[capped] = 3
    redo = torch.nonzero(status >= 2).flatten()  # stalled or capped: the Hessian in hand may be older than x
    if len(redo):
        F[redo], G[redo], H[redo] = energy.fgh(x[redo], redo)
        n_h[redo] += 1
    return dict(x=x, f=F, g=G, H=H, status=status, capture=cap_id, sibling=sib_id, n_f=n_f, n_g=n_g, n_h=n_h)


def polish_evals(effort: dict) -> int:
    """Row-evaluation equivalents of a finish_minima effort record: energies, gradients x 2 (forward and backward),
    Hessians x (d + 1) (forward and d backward passes)."""
    return int(effort['n_f'] + 2 * effort.get('n_g', 0) + effort['n_h'] * (effort['d'] + 1))


def _ident(value) -> str:
    return str(value[0] if isinstance(value, (list, tuple)) and len(value) else value)


def row_key(row) -> tuple:
    """(identifier, space group, Z') of one crystal: minima of different molecules, groups or Z' never match."""
    return _ident(getattr(row, 'identifier', None)), int(row.sg_ind), int(row.z_prime)


def batch_keys(batch) -> list:
    """``row_key`` of every crystal of a collated batch, without unpacking it."""
    ident = getattr(batch, 'identifier', None)
    n = batch.num_graphs
    idents = [_ident(i) for i in ident] if isinstance(ident, (list, tuple)) and len(ident) == n else [_ident(ident)] * n
    sg = batch.sg_ind.reshape(-1).tolist()
    zp = batch.z_prime.reshape(-1).tolist()
    return [(idents[i], int(sg[i]), int(zp[i])) for i in range(n)]


def identity_key(base: tuple, hand) -> tuple:
    """``row_key`` extended by the signs of the units' handedness in the identity description."""
    zp = int(base[2])
    return tuple(base) + (tuple(int(v) for v in torch.as_tensor(hand).reshape(-1)[:zp].sign().tolist()),)


def identity_description(batch):
    """In place, on a built batch (current box analysis, centres in the cell): re-describe every crystal in the
    description minima are identified in. Z' = 1 batches: crystal_fd's canonical form (``fd_fold_aunit``: reduced
    cell, folded pose, handedness on the +1 sheet where an improper element allows), which gives one description per
    crystal away from the fundamental domain's seams; a row crystal_fd does not take keeps its wrapped centre and
    pose. Batches with a Z' > 1 row: ``canonicalize_aunit`` (an asymmetric-unit image only: the same crystal in two
    cell settings is then two minima)."""
    if int(batch.z_prime.max()) == 1:
        fd_fold_aunit(batch)
    else:
        canonicalize_aunit(batch)


class MinimaLibrary:
    """The distinct minima found so far. A minimum's id is its position in insertion order.

    Each minimum is held twice. In the identity description (``identity_description``): parameters x, the Hessian H of
    E/kT there, E/kT and its key (``identity_key``: molecule, space group, Z', handedness signs); ``nearest`` and
    ``match`` measure against these, d^2 = dx^T H_+ dx (``ellipsoid_factor``) at the nearest image of dx
    (``min_image`` over the offsets of ``centre_images``), minima of the same key only. In the report description (asymmetric-unit box, ``canonicalize_aunit``): parameters, handedness and Hessian,
    which a row reported as the minimum carries."""

    def __init__(self, fd_merge: Optional[float] = FD_MERGE_A):
        self.fd_merge = fd_merge  # A; None: merge by the ellipsoids alone
        self.fd_merged = 0  # matches made by the glued FD distance (the ellipsoids missed)
        self.x, self.H, self.W, self.f, self.key = [], [], [], [], []
        self.report = []  # dicts: x [d], hand [zp], H [d, d] in the asymmetric-unit-box description
        self._ids = {}  # key -> list of minimum ids
        self._stack = {}  # key -> (ids [m], x [m, d], W [m, d, d]); rebuilt after a change

    def __len__(self):
        return len(self.x)

    def add(self, x, H, f, key, report: dict) -> int:
        """Append one minimum (identity-description x, H, E/kT, key; report-description dict); returns its id."""
        i = len(self.x)
        self.x.append(torch.as_tensor(x, dtype=F64).cpu().reshape(-1))
        self.H.append(torch.as_tensor(H, dtype=F64).cpu())
        self.W.append(ellipsoid_factor(self.H[-1]))
        self.f.append(float(f))
        self.key.append(tuple(key))
        self.report.append({k: torch.as_tensor(v).cpu().clone() for k, v in report.items()})
        self._ids.setdefault(tuple(key), []).append(i)
        self._stack.pop(tuple(key), None)
        return i

    def truncate(self, n: int):
        """Keep the first ``n`` minima (undo the adds of a batch that is redone)."""
        for name in ('x', 'H', 'W', 'f', 'key', 'report'):
            del getattr(self, name)[n:]
        self._ids = {}
        for i, k in enumerate(self.key):
            self._ids.setdefault(k, []).append(i)
        self._stack = {}

    def _stacked(self, key):
        if key not in self._stack:
            ids = self._ids.get(key, [])
            if not ids:
                return None
            self._stack[key] = (torch.tensor(ids), torch.stack([self.x[i] for i in ids]),
                                torch.stack([self.W[i] for i in ids]))
        return self._stack[key]

    def nearest(self, x: torch.Tensor, keys: list, chunk: int = 256):
        """For each identity-description row of ``x`` [n, d] (key ``keys[i]``): the id of the minimum with the lowest
        d^2 under that minimum's own ellipsoid, and that d^2. Rows with no minimum of their key: id -1, d^2 inf."""
        x = x.to(F64).cpu()
        ids = torch.full((len(x),), -1, dtype=torch.long)
        best = torch.full((len(x),), float('inf'), dtype=F64)
        for key in set(tuple(k) for k in keys):
            st = self._stacked(key)
            if st is None:
                continue
            mids, X, W = st
            shifts = centre_images(key, X.shape[1])
            rows = torch.tensor([i for i, k in enumerate(keys) if tuple(k) == key])
            for c in range(0, len(rows), chunk):
                r = rows[c:c + chunk]
                d2 = None
                for s in shifts:
                    dx = min_image(x[r, None, :] - X[None] - s)  # [b, m, d]
                    d2s = torch.einsum('mji,bmj->bmi', W, dx).pow(2).sum(-1)
                    d2 = d2s if d2 is None else torch.minimum(d2, d2s)
                v, a = d2.min(1)
                best[r], ids[r] = v, mids[a]
        return ids, best

    @torch.no_grad()
    def capture_query(self, batch, base_keys: list):
        """``nearest`` for the crystals of a built batch (e.g. a descent step's): a detached copy is re-described by
        ``identity_description`` and keyed by ``identity_key``. The batch itself is not changed."""
        b = batch.clone().detach()
        identity_description(b)
        hand = b.aunit_handedness.reshape(b.num_graphs, -1)
        keys = [identity_key(base_keys[i], hand[i]) for i in range(b.num_graphs)]
        return self.nearest(b.full_cell_parameters().detach(), keys)

    def match(self, x, W, key, merge_d2: float, f: Optional[float] = None, rg: Optional[float] = None) -> int:
        """Id of a minimum of ``key`` within ``merge_d2`` of the identity-description point ``x`` [d] under both
        ellipsoids (the minimum's and the point's own factor ``W``), the lowest such d^2 first; -1 if none. Both,
        not either: a Hessian that is nearly flat along some direction puts distant points inside its own ellipsoid
        (with "either", 2.6% of MIPCAS merges joined minima more than 0.05 A apart in the glued FD distance, up to
        6 A; 2026-10-28). Failing that, with ``fd_merge``, a Z' = 1 point of E/kT ``f`` for a molecule of radius of
        gyration ``rg`` (A, ``molecule_rg``) matches the nearest known minimum within ``FD_MERGE_KT`` of ``f`` whose
        glued FD distance (``glued_fd_distance``) is below ``fd_merge``: the same crystal described across a face of
        the reduced-cell domain, which the ellipsoids, measured in the parameters, see as far apart."""
        st = self._stacked(tuple(key))
        if st is None:
            return -1
        mids, X, Wm = st
        d2 = None
        for s in centre_images(key, X.shape[1]):
            dx = min_image(x.to(F64)[None] - X - s)  # [m, d]
            d2_lib = torch.einsum('mji,mj->mi', Wm, dx).pow(2).sum(-1)
            d2_own = (dx @ W.to(F64)).pow(2).sum(-1)
            d2s = torch.maximum(d2_lib, d2_own)
            d2 = d2s if d2 is None else torch.minimum(d2, d2s)
        j = int(torch.argmin(d2))
        if float(d2[j]) < merge_d2:
            return int(mids[j])
        if self.fd_merge is None or f is None or rg is None or int(key[2]) != 1:
            return -1
        fs = torch.tensor([self.f[int(i)] for i in mids], dtype=F64)
        near = torch.nonzero((fs - float(f)).abs() < FD_MERGE_KT).flatten()
        if not len(near):
            return -1
        d = glued_fd_distance(int(key[1]), float(rg), int(key[3][0]), x, X[near])
        k = int(torch.argmin(d))
        if float(d[k]) >= float(self.fd_merge):
            return -1
        self.fd_merged += 1
        return int(mids[near[k]])

    @classmethod
    def from_rows(cls, rows):
        """Rebuild from search outputs: every row with ``minimum_id`` >= 0 that is not a recapture defines its minimum
        (``id_params``, ``id_hessian``, ``id_handedness``, ``minimum_f``; its own parameters, handedness and
        ``hessian`` as the report), added in ``minimum_id`` order (a resumed run continues the ids). Raises
        ValueError when those ids are not 0, 1, ..., m - 1: a recapture's ``minimum_id`` would then name another
        minimum."""
        lib = cls()
        own = sorted([r for r in rows if 'id_hessian' in r.keys() and int(r.minimum_id) >= 0
                      and not bool(r.recapture)], key=lambda r: int(r.minimum_id))
        if [int(r.minimum_id) for r in own] != list(range(len(own))):
            raise ValueError("minima library: the saved rows' own minimum ids are not 0..m-1; the outputs and the "
                             "library they were written with disagree")
        for r in own:
            lib.add(r.id_params[0], r.id_hessian[0], float(r.minimum_f), identity_key(row_key(r), r.id_handedness),
                    dict(x=r.full_cell_parameters().reshape(-1), hand=r.aunit_handedness.reshape(-1),
                         H=r.hessian[0]))
        return lib


def molecule_rg(row) -> float:
    """Radius of gyration (A) of a crystal row's molecule as ``crystal_fd.metric`` takes it: the RMS distance of all
    atoms from the heavy-atom centroid."""
    pos, z = row.pos.double(), row.z
    X = pos - pos[z > 1].mean(0)
    return float(X.pow(2).sum(1).mean().sqrt())


_GLUED = {}  # (space group, rg) -> crystal_fd.metric.GluedFDMetric (its basis-change set is costly to build)


def glued_fd_distance(sg: int, rg: float, hand: int, x: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """[k] glued FD distances (A, ``crystal_fd.metric.GluedFDMetric``, margin 1 A, CPU) from the Z' = 1 crystal of
    full cell parameters ``x`` [12] to each of ``X`` [k, 12], all of handedness ``hand`` in space group ``sg``, for a
    molecule of radius of gyration ``rg``: each put in crystal_fd's canonical form, then the symmetric distance."""
    from mxtaltools.crystal_fd.metric import GluedFDMetric, canonical
    key = (int(sg), round(float(rg), 6))
    if key not in _GLUED:
        _GLUED[key] = GluedFDMetric(int(sg), float(rg))
    metric = _GLUED[key]
    P = torch.cat([x.reshape(1, -1), X.reshape(len(X), -1)]).to(F64).cpu()
    can = canonical(int(sg), P, torch.full((len(P),), int(hand)))
    prep = metric.prepare_glued(can)
    return metric.distance(prep, torch.zeros(len(X), dtype=torch.long), torch.arange(1, len(P)))


def missing_mass(rows, library: MinimaLibrary, band_kT: float, conf: float = 0.9, seeds: bool = False) -> dict:
    """Good-Turing estimate of the chance that the next random start (with ``seeds``: the next seed, the seeds
    taken in a random order) ends at a minimum not yet found within ``band_kT`` of the lowest minimum in ``library``.

    The sample is the rows of random starts (``start_index`` >= 0), or with ``seeds`` the seeds' rows (negative
    indices); rows without a ``minimum_id`` are left out. Each is a draw of the minimum it ended at (``minimum_id``; -1, not converged, counts
    as a draw of no minimum). With n such rows and f1 the band's minima reached by exactly one of them, the estimate
    is p = f1 / n, and p_hi = poisson_upper(f1, conf) / n its one-sided upper bound (coordinator.py's bound on f1). A
    minimum known from a seed but reached by one random start counts in f1, so p over-states the chance of an
    unknown minimum. Band membership is by the library's E/kT."""
    from mxtaltools.crystal_search.coordinator import poisson_upper
    if not len(library):
        return dict(n=0, f1=0, band_minima=0, p=float('nan'), p_hi=float('inf'))
    lo = min(library.f)
    in_band = torch.tensor([f - lo <= band_kT for f in library.f])
    ids = torch.tensor([int(r.minimum_id) for r in rows if 'minimum_id' in r.keys() and 'start_index' in r.keys()
                        and (int(r.start_index) < 0) == bool(seeds)], dtype=torch.long)
    hits = torch.bincount(ids[ids >= 0], minlength=len(library)) if len(ids) else torch.zeros(len(library))
    n = len(ids)
    f1 = int(((hits == 1) & in_band).sum())
    return dict(n=n, f1=f1, band_minima=int(in_band.sum()), p=f1 / n if n else float('nan'),
                p_hi=poisson_upper(f1, conf) / n if n else float('inf'))


def minimum_rows(rows) -> list:
    """One row per minimum, in id order: the row that defined it (``minimum_id`` >= 0, not a recapture), cloned, with
    ``hits`` (rows of random starts that ended at it, captures included) and ``seed_hits`` (rows of seeds)."""
    hits, seed_hits, own = {}, {}, {}
    for r in rows:
        if 'minimum_id' not in r.keys() or int(r.minimum_id) < 0:
            continue
        m = int(r.minimum_id)
        tally = seed_hits if 'start_index' in r.keys() and int(r.start_index) < 0 else hits
        tally[m] = tally.get(m, 0) + 1
        if not bool(r.recapture):
            own[m] = r
    out = []
    for m in sorted(own):
        r = own[m].clone()
        r.hits = torch.tensor([hits.get(m, 0)])
        r.seed_hits = torch.tensor([seed_hits.get(m, 0)])
        out.append(r)
    return out


def _inside_clean_box(rows, x, device, tol: float = 1e-5):
    """[n] bool: ``clean_cell_parameters(mode='hard')`` leaves the cell lengths and angles of ``x`` [n, d] unchanged
    (the crystal StageEnergy evaluates is the one ``x`` describes)."""
    b = collate_data_list(rows).to(device)
    b.set_cell_parameters(x.to(device, torch.float32), skip_box_analysis=True)
    before = torch.cat([b.cell_lengths, b.cell_angles], 1).detach().clone()
    b.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
    after = torch.cat([b.cell_lengths, b.cell_angles], 1).detach()
    return ((after - before).abs().amax(1) <= tol * (1 + before.abs().amax(1))).cpu()


def _described(rows, x, device, how):
    """Rows (list) of the crystals ``rows`` at parameters ``x`` [n, d], built and re-described: ``how`` is
    ``identity_description`` or ``canonicalize_aunit``."""
    b = collate_data_list(rows).to(device)
    b.set_cell_parameters(x.to(device, torch.float32), skip_box_analysis=False)
    b.aunit_centroid = wrap_centroid(b.aunit_centroid)
    how(b)
    return b.cpu().detach().batch_to_list()


def finish_minima(rows: list, library: MinimaLibrary, optim_target: str, kT: float, cutoff: float = 10,
                  supercell_size: int = 10, predictor=None, exp6_kwargs: Optional[dict] = None,
                  repulsion: float = 1.0, lj_envelope: Optional[float] = 1.0, max_iter: int = 40, tol: float = 1e-4,
                  merge_d2: float = 1.0, device='cpu', energy_chunk: int = 256,
                  capture_d2: Optional[float] = None):
    """Turn a last stage's output rows into converged minima, and update ``library``.

    - A row the stage captured (``capture_min`` >= 0) is reported as that library minimum: its parameters, handedness
      and Hessian are the minimum's report description.
    - Every other row is re-described in the identity description and polished there (``newton_polish`` on
      ``StageEnergy``). With ``capture_d2``, the polish captures as a stage does (``MinimaLibrary.capture_query``
      against the library as it was before this call, before each iteration), and a row captured there is reported
      as that minimum too, with status 0; rows of this call that converged earlier capture as well (``newton_polish``
      ``keys``), and a row captured by one is reported as the minimum that row becomes (status 0), or, if it becomes
      none, with that row's description and status. A row whose polished cell ``clean_cell_parameters`` would change gets
      status 4. Each remaining row has its Hessian taken again in two descriptions of the polished crystal: the
      identity description (``identity_description``) and the report description (``canonicalize_aunit``, the
      asymmetric-unit box). In ascending E/kT, a converged row whose
      identity Hessian is positive semi-definite is matched against the library in the identity description
      (``MinimaLibrary.match`` at ``merge_d2``): a match makes it a recapture of that minimum (it keeps its own
      polished state), otherwise it is added as a new minimum. Rows with any other status are kept and not added.

    Every returned row is in the report description, re-analysed without ``lj_envelope`` under ``torch.no_grad`` as
    ``gradient_descent_optimization`` re-analyses its outputs (computes 'lj', 'elj' and the target's, ``repulsion``,
    atomwise), and carries ``minimum_id`` ([1] long, -1 when not converged), ``recapture`` ([1] bool),
    ``polish_status`` ([1] long, POLISH_STATUS), ``minimum_f`` ([1] float64, E/kT under the enveloped energy),
    ``hessian`` ([1, d, d] float32, the Hessian of E/kT at the row's parameters), and the identity description
    ``id_params`` ([1, d]), ``id_handedness`` ([1, max Z']) and ``id_hessian`` ([1, d, d]) that ``from_rows`` reads.

    Returns
    -------
    (list, dict)
        The rows in their input order, and the effort: ``n_f`` energies, ``n_g`` gradients and ``n_h`` Hessians
        evaluated, ``d`` (``polish_evals`` sums them in row-evaluations).
    """
    n = len(rows)
    cap = [int(r.capture_min) if 'capture_min' in r.keys() else -1 for r in rows]
    todo = [i for i in range(n) if cap[i] < 0]
    d = int(rows[0].full_cell_parameters().shape[-1])
    effort = dict(n_f=0, n_g=0, n_h=0, d=d)
    rep = [None] * n  # report description: dict(x, hand, H)
    ident = [None] * n  # identity description: dict(x, hand, H)
    out_f = [float('nan')] * n
    mid = [-1] * n
    recap = [False] * n
    status = [0] * n

    def as_captured(i, m):
        rep[i] = dict(library.report[m])
        ident[i] = dict(x=library.x[m], hand=torch.tensor(library.key[m][3], dtype=torch.float32), H=library.H[m])
        out_f[i], mid[i], recap[i] = library.f[m], m, True

    for i in range(n):
        if cap[i] >= 0:
            as_captured(i, cap[i])

    kw = dict(optim_target=optim_target, kT=kT, cutoff=cutoff, supercell_size=supercell_size, predictor=predictor,
              lj_envelope=lj_envelope, exp6_kwargs=exp6_kwargs, device=device, chunk=energy_chunk)
    sib_of = {}  # row -> the row of this call whose converged polish captured it
    if todo:
        # polish in the identity description: at Z' = 1 a reduced cell, well inside clean_cell_parameters' box. A
        # stage's own description can sit at or past the angle clamp, where StageEnergy (which cleans) is flat in
        # that coordinate, so a polish there could end on a point that is not the crystal it reports.
        sub0 = [rows[i] for i in todo]
        x00 = torch.stack([r.full_cell_parameters().reshape(-1) for r in sub0]).double()
        sub = _described(sub0, x00, device, identity_description)
        x0 = torch.stack([r.full_cell_parameters().reshape(-1) for r in sub]).double()
        capture = None
        if capture_d2 is not None and len(library):
            def capture(xc, ids):  # the polish's rows re-described and measured as a stage's capture measures them
                b = collate_data_list([sub[int(j)] for j in ids]).to(device)
                b.set_cell_parameters(xc.to(device, torch.float32), skip_box_analysis=False)
                b.aunit_centroid = wrap_centroid(b.aunit_centroid)
                return library.capture_query(b, batch_keys(b))
        keys = [identity_key(row_key(r), r.aunit_handedness) for r in sub] if capture_d2 is not None else None
        res = newton_polish(StageEnergy(sub, **kw), x0, max_iter=max_iter, tol=tol, capture=capture,
                            capture_d2=float(capture_d2) if capture_d2 is not None else 0.0, keys=keys,
                            merge_d2=merge_d2)
        effort['n_f'] += int(res['n_f'].sum())
        effort['n_g'] += int(res['n_g'].sum())
        effort['n_h'] += int(res['n_h'].sum())
        for j in torch.nonzero(res['status'] == 0).flatten().tolist():  # captured in the polish
            if int(res['sibling'][j]) >= 0:  # by a row of this call: reported as that row's minimum, below
                sib_of[todo[j]] = todo[int(res['sibling'][j])]
            else:
                as_captured(todo[j], int(res['capture'][j]))
        keep = torch.nonzero(res['status'] != 0).flatten()
        todo, sub = [todo[j] for j in keep.tolist()], [sub[j] for j in keep.tolist()]
        res = {k: v[keep] for k, v in res.items()}
    if todo:
        res['status'][~_inside_clean_box(sub, res['x'], device)] = 4
        both = {}
        for name, how in (('id', identity_description), ('rep', canonicalize_aunit)):
            desc = _described(sub, res['x'], device, how)
            xd = torch.stack([c.full_cell_parameters().reshape(-1) for c in desc]).double()
            Fd, Hd = res['f'].clone(), res['H'].clone()
            hand_d = torch.stack([c.aunit_handedness.reshape(-1) for c in desc])
            hand_s = torch.stack([c.aunit_handedness.reshape(-1) for c in sub])
            # the polish's own Hessian is the identity description's where re-describing did not move the row
            same = (min_image(xd - res['x']).abs().amax(1) < 1e-5) & (hand_d == hand_s).all(1) & (res['status'] == 1)
            if name == 'rep':
                same = torch.zeros_like(same)
            redo = torch.nonzero(~same).flatten()
            if len(redo):
                Fd[redo], _, Hd[redo] = StageEnergy([desc[j] for j in redo.tolist()], **kw).fgh(
                    xd[redo], torch.arange(len(redo)))
                effort['n_h'] += len(redo)
            both[name] = (desc, xd, Fd, Hd)
        id_rows, x_id, F_id, H_id = both['id']
        rep_rows, x_rep, _, H_rep = both['rep']
        psd = is_psd(H_id)
        for j in sorted(range(len(todo)), key=lambda j: float(F_id[j])):
            i = todo[j]
            rep[i] = dict(x=x_rep[j], hand=rep_rows[j].aunit_handedness.reshape(-1), H=H_rep[j])
            ident[i] = dict(x=x_id[j], hand=id_rows[j].aunit_handedness.reshape(-1), H=H_id[j])
            out_f[i] = float(F_id[j])
            status[i] = int(res['status'][j])
            if status[i] != 1 or not bool(psd[j]):
                continue
            key = identity_key(row_key(rows[i]), ident[i]['hand'])
            m = library.match(x_id[j], ellipsoid_factor(H_id[j]), key, merge_d2, f=float(F_id[j]),
                              rg=molecule_rg(rows[i]))
            if m >= 0:
                mid[i], recap[i] = m, True
            else:
                mid[i] = library.add(x_id[j], H_id[j], F_id[j], key, rep[i])
    for i, k in sib_of.items():  # as its capturing row ended: that row's minimum, or no minimum with its status
        rep[i], ident[i], out_f[i], mid[i] = dict(rep[k]), dict(ident[k]), out_f[k], mid[k]
        recap[i] = mid[k] >= 0
        status[i] = 0 if mid[k] >= 0 else status[k]

    plain = [r.clone() for r in rows]
    for r in plain:  # consumed here (and rows of one stage carry it all or none)
        if 'capture_min' in r.keys():
            del r.capture_min
    b = collate_data_list(plain).to(device)
    b.aunit_handedness = torch.stack([rep[i]['hand'].reshape(-1) for i in range(n)]).to(b.aunit_handedness)
    b.set_cell_parameters(torch.stack([rep[i]['x'] for i in range(n)]).to(device, torch.float32),
                          skip_box_analysis=False)
    computes = list({'lj', 'elj', str(optim_target).lower()})
    with torch.no_grad():
        b.analyze(computes=computes, cutoff=cutoff, rdf_cutoff=cutoff, supercell_size=supercell_size,
                  return_cluster=False, repulsion=repulsion, surface_padding=0, predictor=predictor,
                  assign_outputs=True, elementwise=False, atomwise=True, bins=100,
                  **{f'exp6_{k}': v for k, v in (exp6_kwargs or {}).items()})
    out = b.cpu().detach().batch_to_list()
    for i, r in enumerate(out):
        r.minimum_id = torch.tensor([mid[i]])
        r.recapture = torch.tensor([recap[i]])
        r.polish_status = torch.tensor([status[i]])
        r.minimum_f = torch.tensor([out_f[i]], dtype=F64)
        r.hessian = torch.as_tensor(rep[i]['H']).to(torch.float32).reshape(1, d, d)
        r.id_params = torch.as_tensor(ident[i]['x']).to(torch.float32).reshape(1, d)
        r.id_handedness = torch.as_tensor(ident[i]['hand']).to(torch.float32).reshape(1, -1)
        r.id_hessian = torch.as_tensor(ident[i]['H']).to(torch.float32).reshape(1, d, d)
    return out, effort
