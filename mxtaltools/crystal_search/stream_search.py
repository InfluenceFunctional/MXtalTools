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
  final energy + margin x ``early_stop_kT`` (inactive until a row has been emitted);
- Rprop is torch's rule (etas 0.5 / 1.2, step sizes in [1e-6, 50], first step lr); the gradient-norm clip and
  anneal_lr, which do not change Rprop's steps, are not applied.
With the search config's ``minima`` block the last stage also captures (``MinimaLibrary.capture_query`` every
``capture_every`` steps) and emitted rows are finished in groups of ``polish_chunk`` (default 4 x ``slots``) by
``minima.finish_minima``.

Starts are drawn in blocks of ``slots`` (``get_initial_state`` seeded by the block index, so a stream with ``slots``
equal to a batch search's ``batch_size`` relaxes the same starts) and carry ``start_index``. Outputs are saved after
every finished group; a resumed run skips the starts already in its output and redraws the rest (rows in flight at
a stop are relaxed again).

Termination: a start is in a slot for at most the sum of its stages' max_num_steps - 1 steps (rounded up to
check_every), the starts are finite, and ``newton_polish`` is bounded, so the loop ends.
"""
import gc
import json
import os
import time

import torch

from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.crystal_search.crystal_opt_utils import canonicalize_aunit, wrap_centroid
from mxtaltools.crystal_search.run_state import atomic_json_dump, atomic_torch_save, committed_outputs
from mxtaltools.dataset_utils.utils import collate_data_list

STREAM_DEFAULTS = dict(slots=None, check_every=5, polish_chunk=None)
STAGE_KEYS = {'optim_target', 'enforce_reduced', 'compression_factor', 'cutoff', 'init_lr', 'convergence_eps',
              'optimizer_func', 'anneal_lr', 'grad_norm_clip', 'show_tqdm', 'max_num_steps', 'target_packing_coeff',
              'centroid_boundary', 'early_stop', 'early_stop_kT', 'supercell_size', 'convergence'}
ETAS, STEP_SIZES = (0.5, 1.2), (1e-6, 50.0)  # torch.optim.Rprop defaults (init_opt uses them)
EMA_ALPHA, CONV_WINDOW = 0.1, 50  # check_convergence's smoothing and window
REGROW_STEPS = 200  # clean steps before a halved evaluation chunk is doubled again


def check_stream_config(config):
    """The ``stream`` block with STREAM_DEFAULTS filled in (``slots`` defaults to ``batch_size``, ``polish_chunk`` to
    4 x ``slots``: the polish's wall time is its sequential depth, so larger groups cost less per row -- 400 MIPCAS
    rows polished in one call took 33 s against 74 s in four calls of 100, 2026-10-28), or None without it. Raises ValueError on an unknown key, a campaign (``coord_dir``), a start method
    other than 'random' or 'reasonable', or a stage it does not reproduce: an optimizer other than rprop, clamp mode,
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
    s['polish_chunk'] = int(s['polish_chunk'] or 4 * s['slots'])
    s['check_every'] = int(s['check_every'])
    if s['slots'] < 1 or s['check_every'] < 1 or s['polish_chunk'] < 1:
        raise ValueError(f"cfg:stream: slots, check_every and polish_chunk must be positive, got {s}")
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
        if str(st.get('optimizer_func', '')).lower() != 'rprop':
            raise ValueError(f"cfg:stream: cfg:opt[{i}].optimizer_func must be rprop")
        if st.get('centroid_boundary', 'clamp') != 'wrap':
            raise ValueError(f"cfg:stream: cfg:opt[{i}].centroid_boundary must be 'wrap'")
        if st.get('enforce_reduced') or st.get('target_packing_coeff') is not None:
            raise ValueError(f"cfg:stream: cfg:opt[{i}] enforce_reduced / target_packing_coeff are not supported")
        if st.get('convergence', 'params') != 'params':
            raise ValueError(f"cfg:stream: cfg:opt[{i}].convergence must be 'params' (per row, no quorum)")
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


def _converged(slots, i, eps):
    """check_convergence's per-row test on slot i's ring of smoothed parameters."""
    n = int(slots.nstep[i])
    if eps <= 0 or n < CONV_WINDOW:
        return False
    order = (torch.arange(CONV_WINDOW) + n) % CONV_WINDOW  # oldest first
    ring = slots.smooth[i, order]
    return float(ring.diff(dim=0).abs().mean()) < eps


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
    cutoff = float(stages[-1].get('cutoff') or 10)
    supercell = int(stages[-1].get('supercell_size', 10))
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
    running_min = min((float(getattr(r, key)) for r in opt_outs if key in r.keys()), default=None)
    effort = dict(stage_evals=[0] * n_st, polish_evals=0, final_evals=0, captured=0, cascaded=0, wall_s=0.0,
                  time_s=dict(starts=0.0, emit=0.0, finish=0.0))  # the rest of wall_s is the descent

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
        slots.begin_stage(i, row.full_cell_parameters().reshape(-1), float(stages[0]['init_lr']))

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
    def finish(rows):
        nonlocal running_min
        if mcfg is not None:
            n_lib = len(library)
            while True:  # on a CUDA out-of-memory error: undo this group's adds, halve the energy chunk, redo
                try:
                    done_rows, eff = mn.finish_minima(rows, library, key, float(mcfg['kT']), cutoff=cutoff,
                                                      supercell_size=supercell, lj_envelope=mcfg['lj_envelope'],
                                                      max_iter=int(mcfg['max_iter']), tol=float(mcfg['tol']),
                                                      merge_d2=float(mcfg['merge_d2']), device=device,
                                                      energy_chunk=energy_chunk[0])
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
        lo = min(float(getattr(r, key)) for r in rows)
        running_min = lo if running_min is None else min(running_min, lo)
        save()

    t0 = time.time()
    if not os.path.exists(progress) or config.force_restart_run:
        save()
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

    def step_chunk(template, idx):
        """One evaluation of the rows ``idx`` (built on ``template``): loss [n] (CPU), gradient [n, d], recorded
        parameters [n, d] (cleaned, centre unwrapped as gradient_descent_optimization records)."""
        cb = template.clone().detach()
        x = slots.X[idx].clone().requires_grad_(True)
        cb.set_cell_parameters(x, skip_box_analysis=True)
        cb.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
        cb.aunit_centroid = wrap_centroid(x[:, 6:6 + 3 * nz])
        outputs, cluster = cb.analyze(computes=computes, cutoff=cutoff, rdf_cutoff=cutoff, supercell_size=supercell,
                                      return_cluster=True, repulsion=1, surface_padding=0, std_orientation=True,
                                      margin=0, assign_outputs=True, elementwise=True, atomwise=False, bins=100)
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
        if templates is None:
            tmpl = torch.tensor(occ)
            templates = [(collate_data_list([slots.row[i] for i in occ[c:c + eval_chunk]]).to(device),
                          torch.tensor(occ[c:c + eval_chunk])) for c in range(0, len(occ), eval_chunk)]
        try:
            parts = [step_chunk(t, i) for t, i in templates]
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
            sgn = g * slots.prev[tmpl]
            ss = slots.step_size[tmpl]
            ss = torch.where(sgn > 0, ss * ETAS[1], torch.where(sgn < 0, ss * ETAS[0], ss)).clamp(*STEP_SIZES)
            g = torch.where(sgn < 0, torch.zeros_like(g), g)
            slots.X[tmpl] -= g.sign() * ss
            slots.step_size[tmpl], slots.prev[tmpl] = ss, g
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
                    _converged(slots, i, float(st.get('convergence_eps', 1e-3))):
                (out if int(slots.stage[i]) == n_st - 1 else advance).append(i)
                continue
            casc = [(int(a), float(m)) for a, m in (st.get('early_stop') or [])]
            kT = float(st.get('early_stop_kT', 1.0))
            while running_min is not None and int(slots.casc_k[i]) < len(casc) and \
                    int(slots.nstep[i]) > casc[int(slots.casc_k[i])][0]:
                if float(slots.last_loss[i]) > running_min + casc[int(slots.casc_k[i])][1] * kT:
                    out.append(i)
                    effort['cascaded'] += 1
                    break
                slots.casc_k[i] += 1
        if library is not None and len(library) and gstep % int(mcfg['capture_every']) == 0:
            gone = set(out) | set(advance)
            live = [k for k, i in enumerate(tmpl.tolist()) if i not in gone and int(slots.stage[i]) == n_st - 1]
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
            slots.begin_stage(i, slots.best_x[i], float(stages[int(slots.stage[i])]['init_lr']))
        if out:
            emit(sorted(out))
            if not stopping and stop.reason() is not None:  # drain: no new starts, the slots in flight finish
                stopping = True
                print(f"{config.run_name}: stop requested ({stop.reason()}); finishing the rows in flight")
            if not stopping:
                while len(pool) < len(out) and fill_pool():
                    pass
                for i in sorted(out):
                    if pool:
                        load(i, *pool.pop(0))
            templates = None
        while len(pending) >= scfg['polish_chunk']:
            finish(pending[:scfg['polish_chunk']])
            del pending[:scfg['polish_chunk']]
    if pending:
        finish(pending[:])
        pending.clear()
    effort.update(wall_s=time.time() - t0, global_steps=gstep, rows=len(opt_outs))
    atomic_json_dump(effort, str(out_path)[:-3] + '_stream_effort.json')
    print(f"{config.run_name}: stream effort {json.dumps(effort)}")
    return opt_outs
