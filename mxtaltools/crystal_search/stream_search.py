"""Streaming search: a fixed set of slots, each carrying one start through every stage on its own clock.

Opt-in through the search config's ``stream`` block (``run_search.crystal_search``); ``run_search._search_loop`` is
not used then.

In a batch search every row of a batch steps until the batch stops (its slowest row, the 95% quorum or the step
cap), so rows retired early (cascade, capture, convergence) leave the batch smaller but the batch's sequential depth
unchanged, and a GPU step of a small batch costs much the same at any occupancy. Here each of ``slots`` slots holds
one start and its own state: the stage it is in, its step count in that stage, its Rprop step sizes, its best state
and loss in the stage, and its convergence record. Every step evaluates all occupied slots together; every
``check_every`` steps each slot is tested and, when its row has finished its stage, moved to the next stage in place
(from its best state, Rprop state reset to the new stage's lr) or, after the last stage, emitted; an emptied slot
takes the next start at once.

Per stage, as ``gradient_descent_optimization`` would run it (wrap mode, Rprop):
- the loss is the stage target's compute plus, with ``compression_factor``, its pressure term
  (``compute_auxiliary_loss``), per row;
- a row leaves the stage at ``max_num_steps - 1`` evaluated steps; with ``convergence_eps`` > 0, from 50 steps on,
  when its own EMA-smoothed parameter change (``ema_trajectory``, alpha 0.1; the mean absolute step-to-step change
  over its last 50 smoothed steps) is below ``convergence_eps`` -- ``check_convergence``'s per-row test, without
  the batch's 95% quorum;
- ``early_stop`` retires a row whose loss at its first check past each listed step exceeds the lowest emitted
  final energy + margin x ``early_stop_kT`` (inactive until a row has been emitted); with ``minima`` the final energy
  is the enveloped one at the last stage's cutoff (``minimum_f`` x kT), as the stages evaluate, and a bfgs row's loss
  is that of its accepted point (a stage at another cutoff still reads its own loss against it);
- Rprop is torch's rule (etas 0.5 / 1.2, step sizes in [1e-6, 50], first step lr); the gradient-norm clip and
  anneal_lr, which do not change Rprop's steps, are not applied;
- each stage is evaluated at its own ``cutoff`` and ``supercell_size`` (rows of stages that share both are evaluated
  in one call, so stages with different cutoffs cost one call each per step).
The last stage may instead be ``optimizer_func: bfgs`` (needs ``minima``, no compression): per row, gradient-only
BFGS on the stage's loss, one trial point per step (``_bfgs_step``), each step capped per block by
``minima.POLISH_CAPS`` times a trust factor; a row leaves the stage when, after at least ``BFGS_MIN_UPDATES``
updates, the BFGS decrement 0.5 g^T B g / kT is below ``convergence_eps`` (kT; default 1e-4), at ``max_num_steps - 1``
steps, or after ``minima.POLISH_TRIES`` rejected trials twice in a row (the inverse Hessian reset to the identity in
between). It takes the polish's work into the stream's
batched calls: rows reach ``finish_minima`` close to converged, so its Newton polish needs few iterations.
With the search config's ``minima`` block every stage is evaluated with ``minima.lj_envelope`` (the LJ-family
computes switched off smoothly over the last ``lj_envelope`` A before the stage's cutoff, as the polish evaluates
them), the last stage captures (``MinimaLibrary.capture_query`` every ``capture_every`` steps; on MIPCAS eLJ,
2026-10-30, rows captured in the first stage reached the captured minimum on a full polish 80% of the time, with or
without its compression term, against 89-95% in the last stage, and capturing there saved no time), and emitted rows are finished in groups of
``polish_chunk`` (default 1,000) by ``minima.finish_minima``, whose polish captures too; at the end the run
writes ``<run>_minima.pt``, one row per minimum with its hit counts (``minima.minimum_rows``).

Starts are drawn in blocks of ``slots`` (``get_initial_state`` seeded by the block index, so a stream with ``slots``
equal to a batch search's ``batch_size`` relaxes the same starts) and carry ``start_index``. With ``seeds`` (a saved
list of crystal rows, e.g. another search's output or a set of known structures; needs ``minima``) the seeds come
first, each put on this search's copy of its molecule and polished directly, without a descent stage, in groups of
``polish_chunk``, with start index -(j + 1): the polish relaxes them, takes their Hessians and merges or captures
the repeats, and their minima are in the library before any random start reaches it. (On MIPCAS eLJ, 2026-10-29:
the 13,505 anchors of an earlier eLJ prior polished to 100 minima and the 9,237 of a UMA prior to 80; each set
reached all 15 minima within 2 kT of the lowest that a random run found. Entering at the last stage instead took
about as many evaluations for the eLJ seeds and about 30% fewer for the UMA seeds, and the UMA seeds then reached
14 of the 15.) The seeds are taken in a random order (seeded by ``opt_seed``), so that with ``stop_p`` they stop as
the random starts do: before each seed group the Good-Turing bound over the seeds polished so far
(``minima.missing_mass(seeds=True)``) is compared with ``stop_p``, and once below it the remaining seeds are skipped.
(Simulated on the eLJ-prior seeds above, 50 orders: stopped after 1,000-2,000 of the 13,505, with all 15 minima
within 2 kT found.) In a file of distinct minima, such as another run's ``_minima.pt``, most seeds are singletons, the
bound stays high, and the file is polished whole. Outputs are saved after every finished group; a resumed run skips
the starts already in its output and redraws the rest (rows in flight at a stop are relaxed again).

With ``stop_p`` (needs ``minima``, and one molecule, space group and Z' in the search), after every finished group
the Good-Turing estimate of the chance that the next random start finds a minimum not yet found within
``stop_band_kT`` of the lowest (``minima.missing_mass``, its 90% upper bound) is compared with ``stop_p``; below it the
run ends at once: the rows in flight and the emitted rows not yet polished are dropped, by the same estimate unlikely
to hold a new minimum (finishing them took about 30% of a MIPCAS eLJ run, 2026-10-30), and their count is recorded
(``stopped.dropped``). A resumed run draws them again. The estimates are recorded in the effort file. A stop request
(``stop``) still drains: no new starts, the slots in flight finish.

Termination: a start is in a slot for at most the sum of its stages' max_num_steps - 1 steps (rounded up to
check_every), the starts (seeds and ``num_samples`` random ones) are finite, and ``newton_polish`` is bounded, so the
loop ends; ``stop_p`` can only end it sooner.
"""
import gc
import json
import math
import os
import time

import torch

from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit, wrap_centroid
from mxtaltools.crystal_search.run_state import atomic_json_dump, atomic_torch_save, committed_outputs
from mxtaltools.dataset_utils.utils import collate_data_list

STREAM_DEFAULTS = dict(slots=None, check_every=5, polish_chunk=None, seeds=None, stop_p=None, stop_band_kT=5.0)
STAGE_KEYS = {'optim_target', 'enforce_reduced', 'compression_factor', 'cutoff', 'init_lr', 'convergence_eps',
              'optimizer_func', 'anneal_lr', 'grad_norm_clip', 'show_tqdm', 'max_num_steps', 'target_packing_coeff',
              'centroid_boundary', 'early_stop', 'early_stop_kT', 'supercell_size', 'convergence'}
ETAS, STEP_SIZES = (0.5, 1.2), (1e-6, 50.0)  # torch.optim.Rprop defaults (init_opt uses them)
EMA_ALPHA, CONV_WINDOW = 0.1, 50  # check_convergence's smoothing and window
REGROW_STEPS = 200  # clean steps before a halved evaluation chunk is doubled again
BFGS_MIN_UPDATES = 3  # accepted BFGS updates before a row's decrement can end its stage (B starts as a guess)
BFGS_SHRINK = 0.25  # a rejected trial's step is shortened by this factor for the next trial


def check_stream_config(config):
    """The ``stream`` block with STREAM_DEFAULTS filled in (``slots`` defaults to ``batch_size``, ``polish_chunk`` to
    1,000: the polish's wall time is its sequential depth, so larger groups cost less per row -- 400 MIPCAS rows
    polished in one call took 33 s against 74 s in four calls of 100, 2026-10-28 -- but the library, and so capture
    and the cascade, waits for the first group: at 4 x 500 slots a 2,000-start run captured nothing, 2026-10-30), or
    None without it. Raises
    ValueError on an unknown key, a campaign (``coord_dir``), a start method other than 'random' or 'reasonable',
    ``stop_p`` outside (0, 1) or without the ``minima`` block, a ``seeds`` file that does not exist or ``seeds``
    without the ``minima`` block, or a stage it
    does not reproduce: an optimizer other than rprop (bfgs is allowed on the last stage, with the ``minima`` block, no
    compression and ``convergence_eps`` > 0), an rprop stage without ``init_lr``, a cutoff not above the minima
    block's ``lj_envelope``, clamp mode,
    ``enforce_reduced``, ``fd_refold``, ``target_packing_coeff``, ``keep_lowest_fraction``, ``convergence: grad``,
    any other stage key, or stages with different optimisation targets."""
    block = getattr(config, 'stream', None)
    if block is None:
        return None
    block = dict(block if isinstance(block, dict) else vars(block))
    unknown = sorted(set(block) - set(STREAM_DEFAULTS))
    if unknown:
        raise ValueError(f"cfg:stream: unknown key(s) {unknown}; known {sorted(STREAM_DEFAULTS)}")
    s = {**STREAM_DEFAULTS, **block}
    s['slots'] = int(s['slots'] or config.batch_size)
    s['polish_chunk'] = int(s['polish_chunk'] or 1000)
    s['check_every'] = int(s['check_every'])
    if s['slots'] < 1 or s['check_every'] < 1 or s['polish_chunk'] < 1:
        raise ValueError(f"cfg:stream: slots, check_every and polish_chunk must be positive, got {s}")
    if s['stop_p'] is not None:
        s['stop_p'], s['stop_band_kT'] = float(s['stop_p']), float(s['stop_band_kT'])
        if not 0 < s['stop_p'] < 1 or not s['stop_band_kT'] > 0:
            raise ValueError(f"cfg:stream: stop_p must be in (0, 1) and stop_band_kT positive, got {s}")
        if getattr(config, 'minima', None) is None:
            raise ValueError("cfg:stream.stop_p counts minima: it needs the cfg:minima block")
    if s['seeds'] is not None:
        if not os.path.exists(str(s['seeds'])):
            raise ValueError(f"cfg:stream.seeds: no file {s['seeds']!r}")
        if getattr(config, 'minima', None) is None:
            raise ValueError("cfg:stream.seeds are polished: they need the cfg:minima block")
    if getattr(config, 'coord_dir', None):
        raise ValueError("cfg:stream does not run in a campaign (cfg:coord_dir) yet")
    if config.init_sample_method not in ('random', 'reasonable'):
        raise ValueError(f"cfg:stream draws its own starts: init_sample_method 'random' or 'reasonable', "
                         f"got {config.init_sample_method!r}")
    stages = [st if isinstance(st, dict) else vars(st) for st in config.opt]
    for i, st in enumerate(stages):
        bad = sorted(set(st) - STAGE_KEYS)
        if bad:
            raise ValueError(f"cfg:stream: cfg:opt[{i}] key(s) {bad} are not reproduced by the stream")
        opt = str(st.get('optimizer_func', '')).lower()
        if opt == 'bfgs':
            if i != len(stages) - 1:
                raise ValueError(f"cfg:stream: cfg:opt[{i}]: only the last stage may be bfgs")
            if getattr(config, 'minima', None) is None:
                raise ValueError("cfg:stream: a bfgs stage measures its decrement in kT: it needs the cfg:minima block")
            if st.get('compression_factor'):
                raise ValueError(f"cfg:stream: cfg:opt[{i}]: a bfgs stage takes no compression_factor")
            if not float(st.get('convergence_eps', 1e-4)) > 0:
                raise ValueError(f"cfg:stream: cfg:opt[{i}]: a bfgs stage needs convergence_eps > 0 (kT)")
        elif opt != 'rprop':
            raise ValueError(f"cfg:stream: cfg:opt[{i}].optimizer_func must be rprop (or bfgs on the last stage)")
        elif st.get('init_lr') is None:
            raise ValueError(f"cfg:stream: cfg:opt[{i}]: an rprop stage needs init_lr")
        if st.get('centroid_boundary', 'clamp') != 'wrap':
            raise ValueError(f"cfg:stream: cfg:opt[{i}].centroid_boundary must be 'wrap'")
        if st.get('enforce_reduced') or st.get('target_packing_coeff') is not None:
            raise ValueError(f"cfg:stream: cfg:opt[{i}] enforce_reduced / target_packing_coeff are not supported")
        if st.get('convergence', 'params') != 'params':
            raise ValueError(f"cfg:stream: cfg:opt[{i}].convergence must be 'params' (per row, no quorum)")
    minima = getattr(config, 'minima', None)
    if minima is not None:
        from mxtaltools.crystal_search.run_search import MINIMA_DEFAULTS
        env = {**MINIMA_DEFAULTS, **dict(minima if isinstance(minima, dict) else vars(minima))}['lj_envelope']
        if env is not None and any(float(st.get('cutoff') or 10) <= float(env) for st in stages):
            raise ValueError(f"cfg:stream: every stage's cutoff must exceed cfg:minima.lj_envelope ({env} A), "
                             "which every stage is evaluated with")
    targets = {str(st['optim_target']).lower() for st in stages}
    if len(targets) != 1:
        raise ValueError(f"cfg:stream: every stage must share one optim_target, got {sorted(targets)}")
    return s


class _Slots:
    """Per-slot state, on ``device`` (parameters, Rprop) and the CPU (counters, records)."""

    def __init__(self, n, d, device):
        self.row = [None] * n
        self.start = torch.full((n,), -1, dtype=torch.long)
        self.stage = torch.zeros(n, dtype=torch.long)
        self.nstep = torch.zeros(n, dtype=torch.long)
        self.steps_in = torch.zeros(n, 8, dtype=torch.long)  # steps per stage, for the output record
        self.X = torch.zeros(n, d, device=device)
        self.step_size = torch.zeros(n, d, device=device)
        self.prev = torch.zeros(n, d, device=device)
        self.best_loss = torch.full((n,), float('inf'))
        self.best_x = torch.zeros(n, d)
        self.last_loss = torch.full((n,), float('inf'))
        self.casc_k = torch.zeros(n, dtype=torch.long)
        self.ema_num = torch.zeros(n, d)
        self.ema_den = torch.zeros(n)
        self.smooth = torch.zeros(n, CONV_WINDOW, d)  # ring of the last smoothed parameters
        self.cap_id = torch.full((n,), -1, dtype=torch.long)
        # bfgs stage, per row (CPU, float64): accepted point, its loss and gradient, inverse-Hessian estimate, trust
        self.bx = torch.zeros(n, d, dtype=torch.float64)
        self.bf = torch.zeros(n, dtype=torch.float64)
        self.bg = torch.zeros(n, d, dtype=torch.float64)
        self.B = torch.eye(d, dtype=torch.float64).repeat(n, 1, 1)
        self.btrust = torch.ones(n, dtype=torch.float64)
        self.btries = torch.zeros(n, dtype=torch.long)  # rejected trials since the last accepted one
        self.bhave = torch.zeros(n, dtype=torch.bool)  # an accepted point exists
        self.bupd = torch.zeros(n, dtype=torch.bool)  # B has had an update (the first one is scaled)
        self.breset = torch.zeros(n, dtype=torch.bool)  # B was reset after a run of rejections
        self.bnupd = torch.zeros(n, dtype=torch.long)  # BFGS updates since B was last set
        self.balpha = torch.ones(n, dtype=torch.float64)  # fraction of the capped step the next trial takes
        self.bdone = torch.zeros(n, dtype=torch.bool)  # converged, or given up: leaves at the next check

    def occupied(self):
        return [i for i, r in enumerate(self.row) if r is not None]

    def begin_stage(self, i, x, lr):
        self.X[i] = x.to(self.X)
        self.step_size[i] = lr
        self.prev[i] = 0
        self.nstep[i] = 0
        self.best_loss[i] = float('inf')
        self.last_loss[i] = float('inf')
        self.casc_k[i] = 0
        self.ema_num[i] = 0
        self.ema_den[i] = 0
        self.smooth[i] = 0
        self.B[i] = torch.eye(self.B.shape[-1], dtype=torch.float64)
        self.btrust[i] = 1.0
        self.btries[i] = 0
        self.bhave[i] = self.bupd[i] = self.breset[i] = self.bdone[i] = False
        self.bnupd[i] = 0
        self.balpha[i] = 1.0


def _seed_starts(path, samples_to_optim, device, done, chunk: int = 256):
    """Seed starts: each row of the list saved at ``path`` (crystal rows, e.g. another search's output) whose
    molecule identifier, space group and Z' (``minima.row_key``) this search has, with its cell parameters and
    handedness put on this search's copy of that molecule (the first start with that key, oriented as a drawn start
    is), as start index -(j + 1) for row j. Rows already in the output (``done``) are left out. Raises ValueError
    when no row matches."""
    from mxtaltools.crystal_search.minima import row_key
    seeds = torch.load(path, weights_only=False)
    if not isinstance(seeds, list):
        seeds = seeds.batch_to_list() if hasattr(seeds, 'batch_to_list') else [seeds]
    tmpl = {}
    for r in samples_to_optim:
        tmpl.setdefault(row_key(r), r)
    use = [(j, s) for j, s in enumerate(seeds) if row_key(s) in tmpl]
    if not use:
        raise ValueError(f"cfg:stream.seeds: none of the {len(seeds)} rows of {path} has a molecule, space group and "
                         f"Z' of this search ({sorted(tmpl)})")
    out = []
    for c in range(0, len(use), chunk):
        part = use[c:c + chunk]
        b = collate_data_list([tmpl[row_key(s)].clone() for _, s in part]).to(device)
        b.aunit_handedness = torch.stack([s.aunit_handedness.reshape(-1) for _, s in part]).reshape(
            b.aunit_handedness.shape).to(b.aunit_handedness)
        b.set_cell_parameters(torch.stack([s.full_cell_parameters().reshape(-1) for _, s in part]).to(
            device, torch.float32), skip_box_analysis=False)
        b.orient_molecule(mode='standardized', target_handedness=b.aunit_handedness)
        out.extend((-(j + 1), r) for (j, _), r in zip(part, b.cpu().detach().batch_to_list()))
    print(f"cfg:stream.seeds: {len(use)} of {len(seeds)} rows have a molecule, space group and Z' of this search")
    return [(k, r) for k, r in out if k not in done]


def _converged(slots, i, eps):
    """check_convergence's per-row test on slot i's ring of smoothed parameters."""
    n = int(slots.nstep[i])
    if eps <= 0 or n < CONV_WINDOW:
        return False
    order = (torch.arange(CONV_WINDOW) + n) % CONV_WINDOW  # oldest first
    ring = slots.smooth[i, order]
    return float(ring.diff(dim=0).abs().mean()) < eps


def _bfgs_step(slots, ib, f, g, kT, tol, d):
    """One bfgs-stage step for the slots ``ib`` [m] (long), whose trial points (``slots.X``) evaluated to loss ``f``
    [m] and gradient ``g`` [m, d]. A slot's first evaluation is its accepted point. Later trials are accepted on the
    Armijo test (c1 1e-4, and a strict decrease): the inverse Hessian estimate B gets the BFGS update (the first one
    preceded by the scaling s^T y / y^T y; skipped when s^T y <= 1e-10 |s| |y|), the trust factor doubles (up to
    ``minima.TRUST_RANGE[1]``) and the step fraction returns to 1; a rejected trial halves the trust (down to
    TRUST_RANGE[0]) and multiplies the step fraction by ``BFGS_SHRINK``, so a retry is a shorter step even where no
    cap binds. A slot with at least ``BFGS_MIN_UPDATES`` updates since B was last set and 0.5 g^T B g / kT < ``tol``
    at its accepted point is done (``slots.bdone``); after ``minima.POLISH_TRIES`` rejections in a row, B is reset to
    the identity, and after a second such run with no accepted trial between, the slot is done. Every other slot gets
    its next trial x + alpha * scale * (-B g): scale the largest <= 1 keeping each block within
    ``minima.POLISH_CAPS`` times its trust, alpha its step fraction."""
    from mxtaltools.crystal_search.minima import POLISH_CAPS, POLISH_TRIES, TRUST_RANGE, blocks
    F64 = torch.float64
    x_t = slots.X[ib].detach().to('cpu', F64)
    f_t, g_t = f.detach().to('cpu', F64), g.detach().to('cpu', F64)
    first = ~slots.bhave[ib]
    s_ = x_t - slots.bx[ib]
    slope = (slots.bg[ib] * s_).sum(1)
    acc = first | (torch.isfinite(f_t) & (f_t <= slots.bf[ib] + 1e-4 * slope) & (f_t < slots.bf[ib]))
    upd = acc & ~first
    if bool(upd.any()):
        r = ib[upd]
        sv, yv = s_[upd], g_t[upd] - slots.bg[r]
        sy = (sv * yv).sum(1)
        ok = sy > 1e-10 * sv.norm(dim=1) * yv.norm(dim=1)
        if bool(ok.any()):
            ro, so, yo, syo = r[ok], sv[ok], yv[ok], sy[ok]
            Bo = slots.B[ro]
            fresh = ~slots.bupd[ro]
            if bool(fresh.any()):
                gam = syo[fresh] / (yo[fresh] * yo[fresh]).sum(1)
                Bo[fresh] = gam[:, None, None] * torch.eye(d, dtype=F64)
            rho = 1.0 / syo
            L = torch.eye(d, dtype=F64) - rho[:, None, None] * torch.einsum('ni,nj->nij', so, yo)
            slots.B[ro] = L @ Bo @ L.transpose(1, 2) + rho[:, None, None] * torch.einsum('ni,nj->nij', so, so)
            slots.bupd[ro] = True
            slots.bnupd[ro] += 1
        slots.btrust[r] = (slots.btrust[r] * 2).clamp(max=TRUST_RANGE[1])
    ra = ib[acc]
    slots.bx[ra], slots.bf[ra], slots.bg[ra] = x_t[acc], f_t[acc], g_t[acc]
    slots.bhave[ra] = True
    slots.btries[ra] = 0
    slots.breset[ra] = False
    slots.balpha[ra] = 1.0
    rj = ib[~acc]
    slots.btrust[rj] = (slots.btrust[rj] / 2).clamp(min=TRUST_RANGE[0])
    slots.balpha[rj] *= BFGS_SHRINK
    slots.btries[rj] += 1
    run = rj[slots.btries[rj] >= POLISH_TRIES]
    give_up = run[slots.breset[run]]
    again = run[~slots.breset[run]]
    slots.bdone[give_up] = True
    slots.B[again] = torch.eye(d, dtype=F64)
    slots.bupd[again] = False
    slots.bnupd[again] = 0
    slots.breset[again] = True
    slots.btries[again] = 0
    slots.btrust[again] = 1.0
    slots.balpha[again] = 1.0
    gk = slots.bg[ib]
    dec = 0.5 * torch.einsum('ni,nij,nj->n', gk, slots.B[ib], gk) / kT
    slots.bdone[ib[(dec < tol) & (slots.bnupd[ib] >= BFGS_MIN_UPDATES)]] = True
    go = ib[~slots.bdone[ib]]
    if len(go):
        cap = torch.zeros(d, dtype=F64)
        for sl, c in zip(blocks(d), POLISH_CAPS):
            cap[sl] = c
        p = -torch.einsum('nij,nj->ni', slots.B[go], slots.bg[go])
        scale = (slots.btrust[go, None] * cap / p.abs().clamp_min(1e-300)).amin(1).clamp(max=1.0) * slots.balpha[go]
        slots.X[go] = (slots.bx[go] + scale[:, None] * p).to(slots.X)


def stream_search(config, samples_to_optim, device, out_path, progress, num_samples, lease, stop):
    """Run the streaming search (module docstring); returns the output list, also saved at ``out_path``."""
    from mxtaltools.crystal_search.run_search import check_minima_config
    from mxtaltools.crystal_search.utils import get_initial_state
    from mxtaltools.crystal_search import minima as mn

    scfg = check_stream_config(config)
    mcfg = check_minima_config(config)
    stages = [dict(st if isinstance(st, dict) else vars(st)) for st in config.opt]
    n_st = len(stages)
    if n_st > 8:
        raise ValueError("cfg:stream: at most 8 stages")
    key = str(stages[0]['optim_target']).lower()
    if scfg['stop_p'] is not None and len({mn.row_key(r) for r in samples_to_optim}) > 1:
        raise ValueError("cfg:stream.stop_p compares minima with the lowest one, so it needs one molecule, space group "
                         "and Z' per search")
    cutoff = float(stages[-1].get('cutoff') or 10)  # the polish's, and the emitted rows' re-analysis
    supercell = int(stages[-1].get('supercell_size', 10))
    cuts = [(float(st.get('cutoff') or 10), int(st.get('supercell_size', 10))) for st in stages]
    is_bfgs = [str(st.get('optimizer_func', '')).lower() == 'bfgs' for st in stages]
    envelope = {} if mcfg is None or mcfg['lj_envelope'] is None else {'lj_envelope': float(mcfg['lj_envelope'])}
    stage_lr = lambda k: 0.0 if is_bfgs[k] else float(stages[k]['init_lr'])  # noqa: E731  (bfgs has no lr)
    captures = [k == n_st - 1 for k in range(n_st)]  # the last stage only (module docstring)
    S = scfg['slots']

    # resume: outputs, truncated to the progress file's count
    opt_outs = []
    if os.path.exists(out_path) and not config.force_restart_run:
        opt_outs = torch.load(out_path, weights_only=False)
        if os.path.exists(progress):
            with open(progress) as fh:
                state = json.load(fh)
            if state.get('mode') != 'stream':
                raise ValueError(f"{out_path} was written by a batch search; cfg:stream cannot resume it")
            opt_outs = committed_outputs(opt_outs, state)
    done = {int(r.start_index) for r in opt_outs if 'start_index' in r.keys()}
    library = mn.MinimaLibrary.from_rows(opt_outs) if mcfg is not None else None
    def final_energy(r):
        """A finished row's energy for the cascade's running minimum: with ``minima``, the enveloped E at the last
        stage's cutoff (``minimum_f`` x kT), as the stages evaluate; else the target compute."""
        if mcfg is not None and 'minimum_f' in r.keys() and math.isfinite(float(r.minimum_f)):
            return float(r.minimum_f) * float(mcfg['kT'])
        return float(getattr(r, key))

    running_min = min((final_energy(r) for r in opt_outs if key in r.keys()), default=None)
    effort = dict(stage_evals=[0] * n_st, polish_evals=0, final_evals=0, captured=0, cascaded=0, wall_s=0.0,
                  time_s=dict(starts=0.0, emit=0.0, finish=0.0),  # the rest of wall_s is the descent
                  missing_mass=[], stopped=None, seed_missing_mass=[], seeds_stopped=None)
    saturated = [False]  # cfg:stream.stop_p met: no new starts

    def timed(name):
        def wrap(fn):
            def run(*a, **k):
                t = time.time()
                try:
                    return fn(*a, **k)
                finally:
                    if torch.cuda.is_available() and str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    effort['time_s'][name] += time.time() - t
            return run
        return wrap

    def save():
        lease.check()
        atomic_torch_save(opt_outs, out_path)
        atomic_json_dump({'mode': 'stream', 'cursor': len(opt_outs), 'batch_idx': -1, 'num_samples': num_samples,
                          'n_out': len(opt_outs)}, progress)

    pool, block = [], [0]

    @timed('starts')
    def fill_pool():
        """Draw and orient the next block of starts (those not already in the output)."""
        while block[0] * S < num_samples:
            b0 = block[0]
            block[0] += 1
            lo, hi = b0 * S, min((b0 + 1) * S, num_samples)
            todo = [k for k in range(lo, hi) if k not in done]
            if not todo:
                continue
            batch = collate_data_list(samples_to_optim[lo:hi]).to(device)
            batch = get_initial_state(config, batch, device, b0)  # the seed of a batch search's batch b0
            batch.orient_molecule(mode='standardized', target_handedness=batch.aunit_handedness)
            rows = batch.cpu().detach().batch_to_list()
            pool.extend((k, rows[k - lo]) for k in todo)
            return True
        return False

    d = None
    slots = None
    pending = []  # emitted rows awaiting the polish
    energy_chunk = [256]  # finish_minima's rows per energy evaluation; halved on out-of-memory

    def load(i, k, row):
        slots.row[i] = row
        slots.start[i] = k
        slots.stage[i] = 0
        slots.steps_in[i] = 0
        slots.cap_id[i] = -1
        slots.begin_stage(i, row.full_cell_parameters().reshape(-1), stage_lr(0))

    @timed('emit')
    def emit_seeds(seeds):
        """Seed rows as emitted rows (no stage steps), described and analysed as ``emit`` does."""
        b = collate_data_list([r for _, r in seeds]).to(device)
        b.aunit_centroid = wrap_centroid(b.aunit_centroid)
        canonicalize_aunit(b)
        with torch.no_grad():
            b.analyze(computes=list({'lj', 'elj', key}), cutoff=cutoff, rdf_cutoff=cutoff, supercell_size=supercell,
                      return_cluster=False, repulsion=1.0, surface_padding=0, assign_outputs=True, elementwise=False,
                      atomwise=True, bins=100)
        out = b.cpu().detach().batch_to_list()
        for r, (k, _) in zip(out, seeds):
            r.start_index = torch.tensor([k])
            r.stage_steps = torch.zeros(1, n_st, dtype=torch.long)
            r.capture_min = torch.tensor([-1])
        effort['final_evals'] += len(out)
        return out

    @timed('emit')
    def emit(idx):
        rows = [slots.row[i] for i in idx]
        b = collate_data_list(rows).to(device)
        b.set_cell_parameters(slots.best_x[idx].to(device), skip_box_analysis=False)
        b.aunit_centroid = wrap_centroid(b.aunit_centroid)
        canonicalize_aunit(b)
        with torch.no_grad():
            b.analyze(computes=list({'lj', 'elj', key}), cutoff=cutoff, rdf_cutoff=cutoff, supercell_size=supercell,
                      return_cluster=False, repulsion=1.0, surface_padding=0, assign_outputs=True, elementwise=False,
                      atomwise=True, bins=100)
        out = b.cpu().detach().batch_to_list()
        for r, i in zip(out, idx):
            r.start_index = torch.tensor([int(slots.start[i])])
            r.stage_steps = slots.steps_in[i, :n_st].clone().reshape(1, n_st)
            if mcfg is not None:
                r.capture_min = torch.tensor([int(slots.cap_id[i])])
            slots.row[i] = None
        effort['final_evals'] += len(out)
        pending.extend(out)

    @timed('finish')
    def finish(rows, seeds=False):
        nonlocal running_min
        if mcfg is not None:
            n_lib = len(library)
            while True:  # on a CUDA out-of-memory error: undo this group's adds, halve the energy chunk, redo
                try:
                    done_rows, eff = mn.finish_minima(rows, library, key, float(mcfg['kT']), cutoff=cutoff,
                                                      supercell_size=supercell, lj_envelope=mcfg['lj_envelope'],
                                                      max_iter=int(mcfg['max_iter']), tol=float(mcfg['tol']),
                                                      merge_d2=float(mcfg['merge_d2']), device=device,
                                                      energy_chunk=energy_chunk[0],
                                                      capture_d2=float(mcfg['capture_d2']))
                    break
                except RuntimeError as e:
                    library.truncate(n_lib)
                    if not is_cuda_oom(e) or energy_chunk[0] == 1:
                        raise
                    energy_chunk[0] = max(1, energy_chunk[0] // 2)
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"{config.run_name}: out of memory in the polish; {energy_chunk[0]} rows per evaluation")
            rows = done_rows
            effort['polish_evals'] += mn.polish_evals(eff)
            effort['final_evals'] += len(rows)
        opt_outs.extend(rows)
        lo = min(final_energy(r) for r in rows)
        running_min = lo if running_min is None else min(running_min, lo)
        save()
        if scfg['stop_p'] is not None and not seeds:  # the seeds' own bound is checked by the seed loop
            mm = mn.missing_mass(opt_outs, library, scfg['stop_band_kT'])
            effort['missing_mass'].append(mm)
            if mm['p_hi'] < scfg['stop_p'] and not saturated[0]:
                saturated[0] = True
                effort['stopped'] = dict(reason='stop_p', wall_s=time.time() - t0, **mm)
                print(f"{config.run_name}: a new minimum within {scfg['stop_band_kT']} kT of the lowest is expected "
                      f"in under {scfg['stop_p']} of starts ({mm}); the rows in flight are dropped")

    t0 = time.time()
    if not os.path.exists(progress) or config.force_restart_run:
        save()
    if scfg['seeds'] is not None:  # seeds first, straight to the polish: their minima are in the library before any
        seeds = _seed_starts(scfg['seeds'], samples_to_optim, device, done)  # random start reaches it
        order = torch.randperm(len(seeds), generator=torch.Generator().manual_seed(int(config.opt_seed))).tolist()
        seeds = [seeds[i] for i in order]  # a random order: the seeds' missing-mass sample
        for c in range(0, len(seeds), scfg['polish_chunk']):
            if scfg['stop_p'] is not None:  # checked before the group, so a resumed run skips seeds it had stopped
                mm = mn.missing_mass(opt_outs, library, scfg['stop_band_kT'], seeds=True)
                if mm['n']:
                    effort['seed_missing_mass'].append(mm)
                if mm['p_hi'] < scfg['stop_p']:
                    effort['seeds_stopped'] = dict(skipped=len(seeds) - c, **mm)
                    print(f"{config.run_name}: a new minimum within {scfg['stop_band_kT']} kT of the lowest is "
                          f"expected in under {scfg['stop_p']} of seeds ({mm}); {len(seeds) - c} seeds skipped")
                    break
            finish(emit_seeds(seeds[c:c + scfg['polish_chunk']]), seeds=True)
    while len(pool) < S and fill_pool():
        pass
    if pool:
        d = int(pool[0][1].full_cell_parameters().shape[-1])
        slots = _Slots(S, d, device)
        for i in range(S):
            if pool:
                load(i, *pool.pop(0))
    gstep, templates, tmpl = 0, None, None
    eval_chunk = S  # rows per energy evaluation; halved on a CUDA out-of-memory error (rows are independent)
    clean = 0  # steps since the last out-of-memory error
    stopping = False  # a stop request: no new starts; the loop ends when the slots in flight are done
    nz = (d - 6) // 6 if d is not None else 0
    computes = list({'lj', 'elj', 'reduction_en' if key == 'reduce' else key})

    def step_chunk(template, idx, cut):
        """One evaluation of the rows ``idx`` (built on ``template``) at ``cut`` (cutoff, supercell size): loss [n]
        (CPU), gradient [n, d], recorded parameters [n, d] (cleaned, centre unwrapped as gradient_descent_optimization
        records)."""
        cb = template.clone().detach()
        x = slots.X[idx].clone().requires_grad_(True)
        cb.set_cell_parameters(x, skip_box_analysis=True)
        cb.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
        cb.aunit_centroid = wrap_centroid(x[:, 6:6 + 3 * nz])
        outputs, cluster = cb.analyze(computes=computes, cutoff=cut[0], rdf_cutoff=cut[0], supercell_size=cut[1],
                                      return_cluster=True, repulsion=1, surface_padding=0, std_orientation=True,
                                      margin=0, assign_outputs=True, elementwise=True, atomwise=False, bins=100,
                                      **envelope)
        loss = outputs['reduction_en' if key == 'reduce' else key].reshape(-1)
        cf = torch.tensor([float(stages[int(slots.stage[i])].get('compression_factor') or 0) for i in idx],
                          device=loss.device, dtype=loss.dtype)
        if bool((cf != 0).any()):  # compute_auxiliary_loss's pressure term, per row
            gate = torch.sigmoid((0.65 - cluster.packing_coeff.reshape(-1)) / 0.01)
            loss = loss + cluster.scale_lengths_to_aunit().sum(dim=1) * cf * gate
        loss.sum().backward()
        rec = cb.full_cell_parameters().detach().clone()
        rec[:, 6:6 + 3 * nz] = x.detach()[:, 6:6 + 3 * nz]
        return loss.detach().cpu(), x.grad.detach(), rec

    while slots is not None:
        occ = slots.occupied()
        if not occ:
            break
        if templates is None:  # rows grouped by their stage's (cutoff, supercell size), then chunked
            groups = {}
            for i in occ:
                groups.setdefault(cuts[int(slots.stage[i])], []).append(i)
            tmpl = torch.tensor([i for c in groups for i in groups[c]])
            templates = [(collate_data_list([slots.row[i] for i in g_[c:c + eval_chunk]]).to(device),
                          torch.tensor(g_[c:c + eval_chunk]), cut) for cut, g_ in groups.items()
                         for c in range(0, len(g_), eval_chunk)]
        try:
            parts = [step_chunk(t, i, cut) for t, i, cut in templates]
        except RuntimeError as e:  # nothing has been updated yet: the step is redone in smaller chunks
            if not is_cuda_oom(e) or eval_chunk == 1:
                raise
            eval_chunk = max(1, eval_chunk // 2)
            templates, clean = None, 0
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{config.run_name}: out of memory; evaluating {eval_chunk} rows at a time")
            continue
        lc, g, rec = (torch.cat([p_[k] for p_ in parts]) for k in range(3))
        del parts
        clean += 1
        if eval_chunk < S and clean >= REGROW_STEPS:  # a passing peak (e.g. after a polish) must not halve the rest
            eval_chunk, templates, clean = min(S, 2 * eval_chunk), None, 0
        with torch.no_grad():
            rec_c = rec.cpu()
            better = lc < slots.best_loss[tmpl]
            slots.best_loss[tmpl] = torch.where(better, lc, slots.best_loss[tmpl])
            slots.best_x[tmpl[better]] = rec_c[better]
            slots.last_loss[tmpl] = lc
            slots.ema_num[tmpl] = (1 - EMA_ALPHA) * slots.ema_num[tmpl] + rec_c
            slots.ema_den[tmpl] = (1 - EMA_ALPHA) * slots.ema_den[tmpl] + 1
            ring_pos = slots.nstep[tmpl] % CONV_WINDOW
            slots.smooth[tmpl, ring_pos] = slots.ema_num[tmpl] / slots.ema_den[tmpl, None]
            bf_rows = torch.tensor([is_bfgs[int(st_)] for st_ in slots.stage[tmpl].tolist()], dtype=torch.bool)
            if bool(bf_rows.any()):
                _bfgs_step(slots, tmpl[bf_rows], lc[bf_rows], g[bf_rows.to(g.device)], float(mcfg['kT']),
                           float(stages[-1].get('convergence_eps', 1e-4)), d)
            rp, rpd = tmpl[~bf_rows], (~bf_rows).to(g.device)
            gr = g[rpd]
            sgn = gr * slots.prev[rp]
            ss = slots.step_size[rp]
            ss = torch.where(sgn > 0, ss * ETAS[1], torch.where(sgn < 0, ss * ETAS[0], ss)).clamp(*STEP_SIZES)
            gr = torch.where(sgn < 0, torch.zeros_like(gr), gr)
            slots.X[rp] -= gr.sign() * ss
            slots.step_size[rp], slots.prev[rp] = ss, gr
            slots.nstep[tmpl] += 1
            slots.steps_in[tmpl, slots.stage[tmpl]] += 1
            for s_, c_ in enumerate(torch.bincount(slots.stage[tmpl], minlength=n_st).tolist()):
                effort['stage_evals'][s_] += c_
        gstep += 1
        if gstep % scfg['check_every']:
            continue

        # stage ends, cascade, capture; then advance, emit and refill
        advance, out = [], []
        for i in tmpl.tolist():
            st = stages[int(slots.stage[i])]
            if int(slots.nstep[i]) >= int(st.get('max_num_steps', 500)) - 1 or \
                    (bool(slots.bdone[i]) if is_bfgs[int(slots.stage[i])] else
                     _converged(slots, i, float(st.get('convergence_eps', 1e-3)))):
                (out if int(slots.stage[i]) == n_st - 1 else advance).append(i)
                continue
            casc = [(int(a), float(m)) for a, m in (st.get('early_stop') or [])]
            kT = float(st.get('early_stop_kT', 1.0))
            while running_min is not None and int(slots.casc_k[i]) < len(casc) and \
                    int(slots.nstep[i]) > casc[int(slots.casc_k[i])][0]:
                now = float(slots.bf[i]) if is_bfgs[int(slots.stage[i])] and bool(slots.bhave[i]) else \
                    float(slots.last_loss[i])  # a bfgs row: its accepted point, not a rejected trial
                if now > running_min + casc[int(slots.casc_k[i])][1] * kT:
                    out.append(i)
                    effort['cascaded'] += 1
                    break
                slots.casc_k[i] += 1
        if library is not None and len(library) and gstep % int(mcfg['capture_every']) == 0:
            gone = set(out) | set(advance)
            live = [k for k, i in enumerate(tmpl.tolist()) if i not in gone and captures[int(slots.stage[i])]]
            if live:
                sub = collate_data_list([slots.row[tmpl[k]] for k in live]).to(device)
                sub.set_cell_parameters(rec[live], skip_box_analysis=False)
                sub.aunit_centroid = wrap_centroid(sub.aunit_centroid)
                hid, hd2 = library.capture_query(sub, mn.batch_keys(sub))
                del sub
                for k, m, v in zip(live, hid.tolist(), hd2.tolist()):
                    if v < float(mcfg['capture_d2']):
                        i = int(tmpl[k])
                        slots.cap_id[i] = m
                        out.append(i)
                        effort['captured'] += 1
        for i in advance:  # next stage, in place, from this stage's best state
            slots.stage[i] += 1
            slots.begin_stage(i, slots.best_x[i], stage_lr(int(slots.stage[i])))
        if advance and len(set(cuts)) > 1:
            templates = None  # an advanced row may now belong to another cutoff group
        if out:
            emit(sorted(out))
            if not stopping and stop.reason() is not None:  # drain: no new starts, the slots in flight finish
                stopping = True
                print(f"{config.run_name}: stop requested ({stop.reason()}); finishing the rows in flight")
            if not stopping and not saturated[0]:
                while len(pool) < len(out) and fill_pool():
                    pass
                for i in sorted(out):
                    if pool:
                        load(i, *pool.pop(0))
            templates = None
        while len(pending) >= scfg['polish_chunk'] and not saturated[0]:
            finish(pending[:scfg['polish_chunk']])
            del pending[:scfg['polish_chunk']]
        if saturated[0]:  # stop_p met: end now; rows in flight and emitted rows not yet polished are not finished
            effort['stopped']['dropped'] = len(slots.occupied()) + len(pending)
            pending.clear()
            break
    if pending:
        finish(pending[:])
        pending.clear()
    effort.update(wall_s=time.time() - t0, global_steps=gstep, rows=len(opt_outs))
    if mcfg is not None:
        atomic_torch_save(mn.minimum_rows(opt_outs), str(out_path)[:-3] + '_minima.pt')
    atomic_json_dump(effort, str(out_path)[:-3] + '_stream_effort.json')
    print(f"{config.run_name}: stream effort {json.dumps(effort)}")
    return opt_outs
