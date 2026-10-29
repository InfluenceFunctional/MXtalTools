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
(``canonicalize_aunit``) with the Hessian taken there. d^2 removes the centres' lattice translations (``min_image``);
near a seam of the fundamental domain two descriptions of one crystal can still differ, and such a pair is two minima.

Termination: ``newton_polish`` evaluates at most ``max_iter + 1`` Hessians and ``max_iter * POLISH_TRIES`` energies
per row and has no other loop; ``finish_minima`` calls it once.
"""
from typing import Optional

import torch

from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit, fd_fold_aunit, wrap_centroid
from mxtaltools.dataset_utils.utils import collate_data_list

ENERGY_TARGETS = ('lj', 'qlj', 'elj', 'exp6', 'silu', 'uma', 'mace')  # stage targets whose compute is an energy
POLISH_CAPS = (0.1, 0.02, 0.01, 0.02)  # largest step per block: lengths (A), angles (rad), centres, rotvecs (rad)
POLISH_TRIES = 8  # Levenberg-Marquardt damping increases tried per iteration before a row counts as stalled
TRUST_RANGE = (1 / 16, 16.0)  # bounds of the per-row factor on POLISH_CAPS (doubled on an accepted step, halved else)
POLISH_STATUS = {0: 'not polished (captured)', 1: 'converged', 2: 'stalled', 3: 'iteration cap',
                 4: 'left the cell-parameter box'}
NEG_REL = 1e-6  # a Hessian counts as positive semi-definite when its lowest eigenvalue is above -NEG_REL * its highest
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
    """

    def __init__(self, rows, optim_target: str, kT: float, cutoff: float = 10, supercell_size: int = 10,
                 predictor=None, lj_envelope: Optional[float] = None, exp6_kwargs: Optional[dict] = None,
                 device='cpu', chunk: int = 256):
        key = str(optim_target).lower()
        if key not in ENERGY_TARGETS:
            raise ValueError(f"minima need an energy target ({ENERGY_TARGETS}), got {optim_target!r}")
        if not kT > 0:
            raise ValueError(f"kT must be positive, got {kT!r}")
        self.rows, self.key, self.kT, self.device, self.chunk = list(rows), key, float(kT), device, int(chunk)
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
        double backward of the batch sum: rows share no parameter, so column k of every row's Hessian is the gradient
        of the sum over rows of gradient component k (d further backward passes)."""
        F, G, H = [], [], []
        d = x.shape[1]
        for s in self._chunks(len(x)):
            xi = x[s].to(self.device, torch.float32).clone().requires_grad_(True)
            fi = self._energy(xi, idx[s])
            gi, = torch.autograd.grad(fi.sum(), xi, create_graph=True)
            Hi = torch.zeros(len(xi), d, d, dtype=F64)
            for k in range(d):
                if gi[:, k].requires_grad:
                    hk, = torch.autograd.grad(gi[:, k].sum(), xi, retain_graph=k < d - 1, allow_unused=True)
                    if hk is not None:
                        Hi[:, k] = hk.detach().double().cpu()
            F.append(fi.detach().double().cpu())
            G.append(gi.detach().double().cpu())
            H.append(0.5 * (Hi + Hi.transpose(1, 2)))
        return torch.cat(F), torch.cat(G), torch.cat(H)


def newton_polish(energy: StageEnergy, x0: torch.Tensor, max_iter: int = 40, tol: float = 1e-4,
                  hessian_every: int = 1):
    """Per-row saddle-free Levenberg-Marquardt Newton descent of ``energy`` from ``x0`` [n, d].

    Each iteration takes every active row's gradient, and its Hessian H = V diag(lam) V^T when the row's last one is
    ``hessian_every`` iterations old or the row asked for a fresh one; in between, the row steps with its last H.
    The default, 1, takes a Hessian every iteration: on a GPU batch the polish's wall time is its sequential depth,
    and reuse (3 or 5) needed more iterations, about 1.7x the wall time on MIPCAS eLJ for 8% fewer row-evaluations,
    and left some rows stalled or capped (2026-10-28).
    A row stops when its Hessian is fresh (taken at the current x), positive semi-definite (``is_psd``), and the
    Newton decrement 0.5 sum_i (v_i . g)^2 / |lam_i| is below ``tol`` (kT); a decrement below ``tol`` under an older
    Hessian asks for a fresh one at the next iteration instead of a step. Otherwise it tries
    x - V diag(1 / (|lam| + mu)) V^T g, scaled so no block moves more than its POLISH_CAPS entry times the row's
    trust factor, and keeps it if E/kT falls (mu /= 3, floored at 1e-4; trust doubled up to TRUST_RANGE[1]), else
    mu *= 4 and trust is halved (down to TRUST_RANGE[0]) and it tries again, at most POLISH_TRIES times; a row with
    no accepted step is stalled. Rows still active after ``max_iter`` iterations stop at the cap. Stalled and
    capped rows get a fresh Hessian at their final x.

    Returns
    -------
    dict
        ``x`` [n, d], ``f`` [n] (E/kT), ``g`` [n, d], ``H`` [n, d, d] at ``x`` (float64, CPU); ``status`` [n] long
        (1 converged, 2 stalled, 3 iteration cap: POLISH_STATUS); ``n_f``, ``n_g``, ``n_h`` [n] long, the energies,
        gradients and Hessians evaluated per row (a Hessian evaluation includes its energy and gradient).
    """
    x = x0.detach().to(F64).cpu().clone()
    n, d = x.shape
    idx = torch.arange(n)
    cap = torch.zeros(d, dtype=F64)
    for sl, c in zip(blocks(d), POLISH_CAPS):
        cap[sl] = c
    F, G, H = energy.fgh(x, idx)
    lam_all, V_all = torch.linalg.eigh(H)
    age = torch.zeros(n, dtype=torch.long)  # iterations since the row's Hessian was taken at its current x
    want = torch.zeros(n, dtype=torch.bool)
    n_h = torch.ones(n, dtype=torch.long)
    n_g = torch.zeros(n, dtype=torch.long)
    n_f = torch.zeros(n, dtype=torch.long)
    mu = torch.ones(n, dtype=F64)
    trust = torch.ones(n, dtype=F64)
    status = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    for it in range(max_iter):
        ia = torch.nonzero(~done).flatten()
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
        dec = 0.5 * (gv ** 2 / lam.abs().clamp_min(1e-12)).sum(1)
        is_fresh = age[ia] == 0
        conv = is_fresh & is_psd(H[ia]) & (dec < tol)
        done[ia[conv]] = True
        status[ia[conv]] = 1
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
    return dict(x=x, f=F, g=G, H=H, status=status, n_f=n_f, n_g=n_g, n_h=n_h)


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
    ``match`` measure against these, d^2 = dx^T H_+ dx (``ellipsoid_factor``, ``min_image``), minima of the same key
    only. In the report description (asymmetric-unit box, ``canonicalize_aunit``): parameters, handedness and Hessian,
    which a row reported as the minimum carries."""

    def __init__(self):
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
            rows = torch.tensor([i for i, k in enumerate(keys) if tuple(k) == key])
            for c in range(0, len(rows), chunk):
                r = rows[c:c + chunk]
                dx = min_image(x[r, None, :] - X[None])  # [b, m, d]
                d2 = torch.einsum('mji,bmj->bmi', W, dx).pow(2).sum(-1)
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

    def match(self, x, W, key, merge_d2: float) -> int:
        """Id of a minimum of ``key`` within ``merge_d2`` of the identity-description point ``x`` [d] under both
        ellipsoids (the minimum's and the point's own factor ``W``), the lowest such d^2 first; -1 if none. Both,
        not either: a Hessian that is nearly flat along some direction puts distant points inside its own ellipsoid
        (with "either", 2.6% of MIPCAS merges joined minima more than 0.05 A apart in the glued FD distance, up to
        6 A; 2026-10-28)."""
        st = self._stacked(tuple(key))
        if st is None:
            return -1
        mids, X, Wm = st
        dx = min_image(x.to(F64)[None] - X)  # [m, d]
        d2_lib = torch.einsum('mji,mj->mi', Wm, dx).pow(2).sum(-1)
        d2_own = (dx @ W.to(F64)).pow(2).sum(-1)
        d2 = torch.maximum(d2_lib, d2_own)
        j = int(torch.argmin(d2))
        return int(mids[j]) if float(d2[j]) < merge_d2 else -1

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
                  merge_d2: float = 1.0, device='cpu', energy_chunk: int = 256):
    """Turn a last stage's output rows into converged minima, and update ``library``.

    - A row the stage captured (``capture_min`` >= 0) is reported as that library minimum: its parameters, handedness
      and Hessian are the minimum's report description.
    - Every other row is re-described in the identity description, polished there (``newton_polish`` on
      ``StageEnergy``; a row whose polished cell ``clean_cell_parameters`` would change gets status 4), and its
      Hessian taken again in two descriptions of the polished crystal: the identity description (``identity_description``) and the report
      description (``canonicalize_aunit``, the asymmetric-unit box). In ascending E/kT, a converged row whose
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
    for i in range(n):
        if cap[i] >= 0:
            m = cap[i]
            rep[i] = dict(library.report[m])
            ident[i] = dict(x=library.x[m], hand=torch.tensor(library.key[m][3], dtype=torch.float32), H=library.H[m])
            out_f[i], mid[i], recap[i] = library.f[m], m, True

    if todo:
        kw = dict(optim_target=optim_target, kT=kT, cutoff=cutoff, supercell_size=supercell_size, predictor=predictor,
                  lj_envelope=lj_envelope, exp6_kwargs=exp6_kwargs, device=device, chunk=energy_chunk)
        # polish in the identity description: at Z' = 1 a reduced cell, well inside clean_cell_parameters' box. A
        # stage's own description can sit at or past the angle clamp, where StageEnergy (which cleans) is flat in
        # that coordinate, so a polish there could end on a point that is not the crystal it reports.
        sub0 = [rows[i] for i in todo]
        x00 = torch.stack([r.full_cell_parameters().reshape(-1) for r in sub0]).double()
        sub = _described(sub0, x00, device, identity_description)
        x0 = torch.stack([r.full_cell_parameters().reshape(-1) for r in sub]).double()
        res = newton_polish(StageEnergy(sub, **kw), x0, max_iter=max_iter, tol=tol)
        effort['n_f'] += int(res['n_f'].sum())
        effort['n_g'] += int(res['n_g'].sum())
        effort['n_h'] += int(res['n_h'].sum())
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
            m = library.match(x_id[j], ellipsoid_factor(H_id[j]), key, merge_d2)
            if m >= 0:
                mid[i], recap[i] = m, True
            else:
                mid[i] = library.add(x_id[j], H_id[j], F_id[j], key, rep[i])

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
