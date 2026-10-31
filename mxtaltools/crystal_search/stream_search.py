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
- ``early_stop`` retires a row whose loss at its first check past each listed step exceeds the lowest finished
  final energy of its condition + margin x ``early_stop_kT`` (inactive until one has finished); with ``minima`` the
  final energy
  is the enveloped one at the last stage's cutoff (``minimum_f`` x kT), as the stages evaluate, and a bfgs row's loss
  is that of its accepted point (a stage at another cutoff still reads its own loss against it);
- Rprop is torch's rule (etas 0.5 / 1.2, step sizes in [1e-6, 50], first step lr); the gradient-norm clip and
  anneal_lr, which do not change Rprop's steps, are not applied;
- each stage is evaluated at its own ``cutoff``, ``supercell_size`` and ``optim_target`` (rows of stages that share
  all three are evaluated in one call, so stages that differ cost one call each per step): a search may relax under
  eLJ first and finish under an MLIP (``minima.MLIP_TARGETS``, its model loaded once by ``utils.parse_opt_config``
  from ``cfg:mace_predictor_path`` / ``cfg:uma_predictor_path``); the last stage's target is the minima's, and each
  target's rows per call are halved on its own out-of-memory errors (an MLIP call holds far fewer rows than an eLJ
  one: MACE-OFF24 on MIPCAS ~175 MB per row for a gradient, 2026-10-30);
- before each step a row whose cell is flat -- each angle within ``clean_cell_parameters``' range ``ANGLE_RANGE`` but
  together giving volume / (a b c) below ``FLAT_VF`` -- is dropped (not in the output; counted in the effort file's
  ``flat_dropped``; a resumed run draws it again) and its slot refilled: its periodic cluster would take gigabytes
  (on MIPCAS eLJ, 2026-10-30, one such row asked for 6.4 GB, and the out-of-memory fallback then halved every
  call for 200 steps).
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
without its compression term, against 89-95% in the last stage, and capturing there saved no time), and emitted
rows are finished in groups of ``polish_chunk`` (default 1,000) by ``minima.finish_minima``, whose polish captures
too. The first random-start group can be smaller (``first_polish_chunk``, default ``polish_chunk``), so the library
starts filling sooner: on MIPCAS eLJ (2026-10-30) a first group of 250 cut the polish's evaluations by 12% and its
converged polishes from 510 to 272, but the extra group cost wall time, so it suits energies whose calls are costly
rather than eLJ. At the end the run writes ``<run>_minima.pt``, one row per minimum with its hit counts
(``minima.minimum_rows``).

A search's conditions are its (molecule, space group, Z') combinations (``minima.row_key``; the stream gets one
template of each, ``utils.init_samples_to_optim`` with ``num_samples=1``), in identifier order, each with
``cfg:num_samples`` starts at most: start ``k = c N + j`` is start j of condition c. A condition's starts are drawn in
blocks of M = min(``slots``, N) (``get_initial_state`` seeded by c times the blocks per condition plus the block, so a
one-condition stream with ``slots`` equal to a batch search's ``batch_size`` relaxes the same starts) and carry
``start_index``. With a ``quota`` the blocks are of min(``slots``, N, ``quota``). An emptied slot takes the next start of the drawing condition with the fewest rows in slots among those
with fewer rows outstanding (in slots or awaiting the polish) than the minima they still need for their quota (no
bound without one); when no drawing condition may take one, the next condition is admitted, in order, at most
``window`` (default: all) drawing at once, and when none can be, the start goes to the drawing condition least over
its need rather than leave the slot idle. A condition stops drawing when its starts are all drawn or it meets its
quota. The cascade compares a row with its own condition's lowest finished
energy. With ``quota`` (needs ``minima``; not with ``stop_p``) a condition is done once its library minima within
``quota_band_kT`` (default 10) of its lowest number ``quota``: checked after each polish group, it draws no more
starts, and its rows in slots and awaiting the polish are dropped (``quota_dropped`` in the effort file; a resumed
run finds the condition done and draws nothing for it). The run writes ``<run>_conditions.json``: per condition its
finished random starts, minima, minima within the band (``quota_band_kT`` with a quota, else ``stop_band_kT``), lowest
E/kT and whether it met its quota. Admitting conditions only as the slots need them keeps a condition's rows together, so its
library, capture and cascade reference fill early while it is still drawing, and the bound on its outstanding rows
keeps the rows dropped at its quota few. With ``connected_starts``
(default on) a start whose molecules' contact network at the first stage's cutoff does not span three dimensions
(``contact_dimension``: isolated layers, chains or molecules, which no force under the cut-off energy can bring
together) is drawn again, at most ``MAX_REDRAWS`` times; the effort file counts them (``starts_disconnected``,
``starts_redrawn``, ``starts_kept_disconnected``). On NEHZOR eLJ (P2_1/c, 2026-10-30) 28% of the random starts were
such slabs, an axis over 40 A at normal density, and about as many ended as isolated layers 5 kT and more above the
lowest minimum; on MIPCAS (P-1) 1 in 500. With ``seeds`` (a saved
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
check_every), the starts (seeds and ``num_samples`` random ones per condition) are finite, each call for a start draws
a block, retires a condition or returns one (``next_start``), and the polish is bounded, so the loop ends; ``stop_p``
and ``quota`` can only end it sooner.
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

STREAM_DEFAULTS = dict(slots=None, check_every=5, polish_chunk=None, first_polish_chunk=None, seeds=None, stop_p=None,
                       stop_band_kT=5.0, connected_starts=True, quota=None, quota_band_kT=10.0, window=None)
STAGE_KEYS = {'optim_target', 'enforce_reduced', 'compression_factor', 'cutoff', 'init_lr', 'convergence_eps',
              'optimizer_func', 'anneal_lr', 'grad_norm_clip', 'show_tqdm', 'max_num_steps', 'target_packing_coeff',
              'centroid_boundary', 'early_stop', 'early_stop_kT', 'supercell_size', 'convergence'}
ETAS, STEP_SIZES = (0.5, 1.2), (1e-6, 50.0)  # torch.optim.Rprop defaults (init_opt uses them)
EMA_ALPHA, CONV_WINDOW = 0.1, 50  # check_convergence's smoothing and window
REGROW_STEPS = 200  # clean steps before a halved evaluation chunk is doubled again
ANGLE_RANGE = (0.2 * math.pi, 0.8 * math.pi)  # clean_cell_parameters' bound on each cell angle (angle_pad 0.6)
FLAT_VF = 0.01  # volume / (a b c) below which a row's cell counts as flat and the row is dropped before a step
MAX_REDRAWS = 20  # redraws of a start whose contact network is not 3-periodic; after them the last draw is kept
REDRAW_BLOCK = 100000  # a redraw's block index offset per attempt (get_initial_state seeds by block index)
BFGS_MIN_UPDATES = 3  # accepted BFGS updates before a row's decrement can end its stage (B starts as a guess)
BFGS_SHRINK = 0.25  # a rejected trial's step is shortened by this factor for the next trial


def check_stream_config(config):
    """The ``stream`` block with STREAM_DEFAULTS filled in (``slots`` defaults to ``batch_size``, ``polish_chunk`` to
    1,000: the polish's wall time is its sequential depth, so larger groups cost less per row -- 400 MIPCAS rows
    polished in one call took 33 s against 74 s in four calls of 100, 2026-10-28 -- but the library, and so capture
    and the cascade, waits for the first group: at 4 x 500 slots a 2,000-start run captured nothing, 2026-10-30), or
    None without it. Raises
    ValueError on an unknown key, a campaign (``coord_dir``), a start method other than 'random' or 'reasonable',
    ``stop_p`` outside (0, 1) or without the ``minima`` block, a ``quota`` that is not a positive count, without the
    ``minima`` block or with ``stop_p``, a ``window`` that is not a positive count, a ``seeds`` file that does not
    exist or ``seeds`` without the ``minima`` block, or a stage it
    does not reproduce: an optimizer other than rprop (bfgs is allowed on the last stage, with the ``minima`` block, no
    compression and ``convergence_eps`` > 0), an rprop stage without ``init_lr``, a cutoff not above the minima
    block's ``lj_envelope``, ``early_stop`` on a stage whose target is not the last stage's, clamp mode,
    ``enforce_reduced``, ``fd_refold``, ``target_packing_coeff``, ``keep_lowest_fraction``, ``convergence: grad``,
    or any other stage key."""
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
    s['first_polish_chunk'] = int(s['first_polish_chunk'] or s['polish_chunk'])
    s['check_every'] = int(s['check_every'])
    if s['slots'] < 1 or s['check_every'] < 1 or s['polish_chunk'] < 1 or s['first_polish_chunk'] < 1:
        raise ValueError(f"cfg:stream: slots, check_every, polish_chunk and first_polish_chunk must be positive, "
                         f"got {s}")
    if s['stop_p'] is not None:
        s['stop_p'], s['stop_band_kT'] = float(s['stop_p']), float(s['stop_band_kT'])
        if not 0 < s['stop_p'] < 1 or not s['stop_band_kT'] > 0:
            raise ValueError(f"cfg:stream: stop_p must be in (0, 1) and stop_band_kT positive, got {s}")
        if getattr(config, 'minima', None) is None:
            raise ValueError("cfg:stream.stop_p counts minima: it needs the cfg:minima block")
    if s['quota'] is not None:
        s['quota'], s['quota_band_kT'] = int(s['quota']), float(s['quota_band_kT'])
        if s['quota'] < 1 or not s['quota_band_kT'] > 0:
            raise ValueError(f"cfg:stream: quota must be a positive count and quota_band_kT positive, got {s}")
        if getattr(config, 'minima', None) is None:
            raise ValueError("cfg:stream.quota counts minima: it needs the cfg:minima block")
        if s['stop_p'] is not None:
            raise ValueError("cfg:stream: quota and stop_p are two stop rules; set one")
    if s['window'] is not None:
        s['window'] = int(s['window'])
        if s['window'] < 1:
            raise ValueError(f"cfg:stream.window must be a positive count of conditions, got {s['window']}")
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
    last_target = str(stages[-1]['optim_target']).lower()
    for i, st in enumerate(stages):
        if str(st['optim_target']).lower() != last_target and st.get('early_stop'):
            raise ValueError(f"cfg:stream: cfg:opt[{i}].early_stop compares the stage's loss with finished energies "
                             f"of the last stage's target ({last_target}); its target is {st['optim_target']!r}")
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


def _cell_matrix(lengths, angles):
    """[3, 3] float64, columns a, b, c in Cartesian (a along x, b in the xy plane), from lengths (A) and angles
    (rad)."""
    import numpy as np
    a, b, c = lengths
    ca, cb, cg = np.cos(angles)
    sg = np.sin(angles[2])
    cz = np.sqrt(max(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0.0)) / sg
    return np.array([[a, b * cg, c * cb], [0.0, b * sg, c * (ca - cb * cg) / sg], [0.0, 0.0, c * cz]])


def contact_dimension(rows, cutoff: float, max_offsets: int = 200000) -> list:
    """Per crystal row: the dimension (0-3) of the lattice of translations its molecules' contact network reaches,
    or -1 when the network falls into more than one piece.

    The molecules are every unit's copy under the row's symmetry operators (``symmetry_operators``, fractional
    centres ``aunit_centroid``), with their periodic images. Two count as in contact when their centres are closer
    than ``cutoff`` + 2 x ``radius`` (the row's largest atom distance from a molecule's centre): no atom pair can be
    within ``cutoff`` otherwise, so the test over-counts contacts, and a result below 3 means that, under an energy cut
    off at ``cutoff``, no force acts across some gap of the crystal at all -- isolated layers (2), chains (1) or
    molecules (0), or pieces that interpenetrate without touching (-1). Following the contacts out from one molecule
    (breadth first), each contact that closes a loop adds the lattice translation it implies; the dimension is the rank
    of those translations, the dimensionality test used to find layered and chain materials in structure databases.
    A row whose offset grid would exceed ``max_offsets`` (a cell much thinner than the cutoff) is reported as 3."""
    import numpy as np
    out = []
    for r in rows:
        L = r.cell_lengths.reshape(-1).double().numpy()
        A = r.cell_angles.reshape(-1).double().numpy()
        T = _cell_matrix(L, A)
        R = float(cutoff) + 2 * float(r.radius.reshape(-1).max()) + 1e-3
        ops = np.asarray(r.symmetry_operators, dtype=float).reshape(-1, 4, 4)
        cent = r.aunit_centroid.reshape(-1, 3).double().numpy()
        f = np.concatenate([(cent @ op[:3, :3].T) + op[:3, 3] for op in ops]) % 1.0  # [M, 3]
        V = abs(np.linalg.det(T))
        h = np.array([V / np.linalg.norm(np.cross(T[:, (k + 1) % 3], T[:, (k + 2) % 3])) for k in range(3)])
        N = np.ceil(R / h).astype(int) + 1
        if np.prod(2 * N + 1) > max_offsets:
            out.append(3)
            continue
        grid = np.stack(np.meshgrid(*[np.arange(-n, n + 1) for n in N], indexing='ij'), -1).reshape(-1, 3)
        M = len(f)
        diff = f[None, :, None, :] - f[:, None, None, :] + grid[None, None]  # [i, j, n, 3]: f_j + n - f_i
        dist = np.linalg.norm(diff @ T.T, axis=-1)
        i_, j_, k_ = np.nonzero(dist < R)
        self_loop = (i_ == j_) & (np.abs(grid[k_]).sum(1) == 0)
        i_, j_, n_ = i_[~self_loop], j_[~self_loop], grid[k_[~self_loop]]
        pos = {0: np.zeros(3, dtype=np.int64)}
        queue, adj = [0], {}
        for a, b, n in zip(i_.tolist(), j_.tolist(), n_):
            adj.setdefault(a, []).append((b, n))
        while queue:
            a = queue.pop(0)
            for b, n in adj.get(a, []):
                if b not in pos:
                    pos[b] = pos[a] + n
                    queue.append(b)
        if len(pos) < M:
            out.append(-1)
            continue
        loops = np.array([pos[a] + n - pos[b] for a, b, n in zip(i_.tolist(), j_.tolist(), n_)])
        loops = loops[np.abs(loops).sum(1) > 0] if len(loops) else loops
        out.append(int(np.linalg.matrix_rank(loops.astype(float))) if len(loops) else 0)
    return out


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
    key = str(stages[-1]['optim_target']).lower()  # the minima's target: emitted energies, polish, capture, cascade
    if scfg['stop_p'] is not None and len({mn.row_key(r) for r in samples_to_optim}) > 1:
        raise ValueError("cfg:stream.stop_p compares minima with the lowest one, so it needs one molecule, space group "
                         "and Z' per search")
    cutoff = float(stages[-1].get('cutoff') or 10)  # the polish's, and the emitted rows' re-analysis
    supercell = int(stages[-1].get('supercell_size', 10))
    cuts = [(float(st.get('cutoff') or 10), int(st.get('supercell_size', 10)), str(st['optim_target']).lower())
            for st in stages]  # a step's call groups: rows of stages sharing cutoff, supercell size and target
    predictors = {}  # an MLIP target's model, loaded once (utils.parse_opt_config: cfg:mace_predictor_path, ...)
    for st in stages:
        t_ = str(st['optim_target']).lower()
        if t_ in mn.MLIP_TARGETS and t_ not in predictors:
            from mxtaltools.crystal_search.utils import parse_opt_config
            predictors[t_] = parse_opt_config(dict(st), config, device, None)['predictor']
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

    running_min = {}  # per condition (minima.row_key): the lowest finished energy, the cascade's reference
    for r in opt_outs:
        if key in r.keys():
            k_ = mn.row_key(r)
            running_min[k_] = min(running_min.get(k_, math.inf), final_energy(r))
    effort = dict(stage_evals=[0] * n_st, polish_evals=0, final_evals=0, captured=0, cascaded=0, flat_dropped=0,
                  starts_disconnected=0, starts_redrawn=0, starts_kept_disconnected=0, quota_met=0, quota_dropped=0,
                  slot_steps=0, wall_s=0.0,
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

    cutoff0 = cuts[0][0]  # the first stage's cutoff: the one a start is first evaluated at
    # conditions: one template per (molecule, space group, Z') in a fixed order; start k is start j = k - c N of
    # condition c, N = cfg:num_samples at most per condition, drawn in blocks of M seeded by c * n_sub + the block
    tmpl_of = {}
    for r in samples_to_optim:
        tmpl_of.setdefault(mn.row_key(r), r)
    keys = sorted(tmpl_of, key=lambda k_: tuple(str(v) for v in k_))
    cidx = {k_: c for c, k_ in enumerate(keys)}
    C, N = len(keys), int(config.num_samples)
    M = max(1, min(S, N, scfg['quota'] or N))  # one condition without a quota: blocks of S, seeded by the block
    # index, as a batch search's batches; with a quota, blocks of the quota (a condition rarely needs many more)
    n_sub = -(-N // M)
    queue = [[] for _ in range(C)]  # drawn starts not yet in a slot: (k, row)
    next_sub = [0] * C  # a condition's next block to draw
    inflight = [0] * C  # its rows in slots
    waiting = [0] * C  # its rows awaiting the polish
    band_n = [0] * C  # its library minima within quota_band_kT of its lowest (with a quota)
    closed = [False] * C  # its quota met: no new starts, its rows in flight and unpolished rows dropped
    issuing, next_c = [], [0]  # conditions drawing starts now (at most cfg:stream.window), the next to admit
    window = scfg['window'] or C
    slot_c = [None] * S  # the condition of each slot's row
    cond_minima = [[] for _ in range(C)]  # library ids per condition
    for m_, k_ in enumerate(library.key if library is not None else []):
        if tuple(k_[:3]) in cidx:
            cond_minima[cidx[tuple(k_[:3])]].append(m_)

    def drawn(c, m, b0):
        """m starts of condition c, drawn as get_initial_state draws a batch with index b0, oriented."""
        batch = collate_data_list([tmpl_of[keys[c]].clone() for _ in range(m)]).to(device)
        batch = get_initial_state(config, batch, device, b0)
        batch.orient_molecule(mode='standardized', target_handedness=batch.aunit_handedness)
        return batch.cpu().detach().batch_to_list()

    @timed('starts')
    def draw(c, sb):
        """Block sb of condition c: its starts not already in the output, as (k, row). With ``connected_starts``,
        a start whose contact network at the first stage's cutoff is not 3-periodic (``contact_dimension``) is drawn
        again, at most ``MAX_REDRAWS`` times (seeded by the block and the attempt), then kept as last drawn."""
        lo, hi = sb * M, min((sb + 1) * M, N)
        todo = [j for j in range(lo, hi) if c * N + j not in done]
        if not todo:
            return []
        b0 = c * n_sub + sb
        rows = drawn(c, hi - lo, b0)
        if scfg['connected_starts']:
            bad = [j for j, dim in zip(todo, contact_dimension([rows[j - lo] for j in todo], cutoff0)) if dim != 3]
            effort['starts_disconnected'] += len(bad)
            for attempt in range(1, MAX_REDRAWS + 1):
                if not bad:
                    break
                new = drawn(c, len(bad), b0 + REDRAW_BLOCK * attempt)
                effort['starts_redrawn'] += len(bad)
                for j, r in zip(bad, new):
                    rows[j - lo] = r
                bad = [j for j, dim in zip(bad, contact_dimension(new, cutoff0)) if dim != 3]
            effort['starts_kept_disconnected'] += len(bad)
        return [(c * N + j, rows[j - lo]) for j in todo]

    def need(c):
        """Rows condition c may have outstanding (in slots or awaiting the polish): the minima it still needs for
        its quota, unbounded without one."""
        return math.inf if scfg['quota'] is None else scfg['quota'] - band_n[c]

    def next_start():
        """The next start (k, row), from the drawing conditions with fewer rows outstanding than they still need
        (``need``): the one with the fewest rows in slots, the earliest on a tie. When none may take one, the next
        condition is admitted, in order and at most ``window`` drawing at once; when none can be, the start goes to
        the drawing condition least over its need rather than leave a slot idle; None when no condition is drawing.
        Each pass returns, draws a block, retires a condition whose starts are all drawn, or admits one, and each of
        these happens finitely often."""
        while True:
            ok = [t for t, c_ in enumerate(issuing) if inflight[c_] + waiting[c_] < need(c_)]
            if not ok:
                if len(issuing) < window and next_c[0] < C:
                    if not closed[next_c[0]]:
                        issuing.append(next_c[0])
                    next_c[0] += 1
                    continue
                if not issuing:
                    return None
                c = issuing[min(range(len(issuing)), key=lambda t: (
                    inflight[issuing[t]] + waiting[issuing[t]] - need(issuing[t]), t))]
            else:
                c = issuing[min(ok, key=lambda t: (inflight[issuing[t]], t))]
            if queue[c]:
                inflight[c] += 1
                return queue[c].pop(0)
            if next_sub[c] < n_sub:
                queue[c].extend(draw(c, next_sub[c]))
                next_sub[c] += 1
            else:
                issuing.remove(c)  # every start drawn: its rows in flight finish

    def quota_check(cs):
        """Close each condition of ``cs`` holding ``quota`` minima within ``quota_band_kT`` of its lowest."""
        if scfg['quota'] is None:
            return
        for c in sorted(set(cs)):
            if closed[c] or not cond_minima[c]:
                continue
            fs = [library.f[m_] for m_ in cond_minima[c]]
            band_n[c] = sum(f_ - min(fs) <= scfg['quota_band_kT'] for f_ in fs)
            if band_n[c] >= scfg['quota']:
                closed[c] = True
                effort['quota_met'] += 1
                queue[c].clear()
                if c in issuing:
                    issuing.remove(c)

    d = None
    slots = None
    pending = []  # emitted rows awaiting the polish
    groups_done = [0]  # random-start polish groups finished in this run
    energy_chunk = [16 if key in mn.MLIP_TARGETS else 256]  # finish_minima's rows per energy evaluation (and emit's
    # per re-analysis); halved on out-of-memory. An MLIP call holds far fewer rows (MACE-OFF24, MIPCAS: ~175 MB a row)

    def load(i, k, row):
        slot_c[i] = cidx[mn.row_key(row)]
        slots.row[i] = row
        slots.start[i] = k
        slots.stage[i] = 0
        slots.steps_in[i] = 0
        slots.cap_id[i] = -1
        slots.begin_stage(i, row.full_cell_parameters().reshape(-1), stage_lr(0))

    @timed('emit')
    def analysed(b):
        """The rows of a built, described batch re-analysed (the target's energy, lj and elj), in energy_chunk rows."""
        rows_ = b.cpu().detach().batch_to_list()
        out_ = []
        for c in range(0, len(rows_), energy_chunk[0]):
            bc = collate_data_list(rows_[c:c + energy_chunk[0]]).to(device)
            with torch.no_grad():
                bc.analyze(computes=list({'lj', 'elj', key}), cutoff=cutoff, rdf_cutoff=cutoff,
                           supercell_size=supercell, return_cluster=False, repulsion=1.0, surface_padding=0,
                           assign_outputs=True, elementwise=False, atomwise=True, bins=100,
                           predictor=predictors.get(key))
            out_.extend(bc.cpu().detach().batch_to_list())
        return out_

    def emit_seeds(seeds):
        """Seed rows as emitted rows (no stage steps), described and analysed as ``emit`` does."""
        b = collate_data_list([r for _, r in seeds]).to(device)
        b.aunit_centroid = wrap_centroid(b.aunit_centroid)
        canonicalize_aunit(b)
        out = analysed(b)
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
        out = analysed(b)
        for r, i in zip(out, idx):
            r.start_index = torch.tensor([int(slots.start[i])])
            r.stage_steps = slots.steps_in[i, :n_st].clone().reshape(1, n_st)
            if mcfg is not None:
                r.capture_min = torch.tensor([int(slots.cap_id[i])])
            slots.row[i] = None
            inflight[slot_c[i]] -= 1
            waiting[slot_c[i]] += 1
        effort['final_evals'] += len(out)
        pending.extend(out)

    def polish(group):
        """finish() a group of emitted rows (no longer awaiting the polish)."""
        for r in group:
            waiting[cidx[mn.row_key(r)]] -= 1
        finish(group)

    def refill():
        """Load the next starts into the empty slots; the number loaded."""
        n_ = 0
        for i in range(S):
            if slots.row[i] is None:
                s_ = next_start()
                if s_ is None:
                    break
                load(i, *s_)
                n_ += 1
        return n_

    @timed('finish')
    def finish(rows, seeds=False):
        n_lib0 = len(library) if library is not None else 0
        if mcfg is not None:
            n_lib = len(library)
            while True:  # on a CUDA out-of-memory error: undo this group's adds, halve the energy chunk, redo
                try:
                    done_rows, eff = mn.finish_minima(rows, library, key, float(mcfg['kT']), cutoff=cutoff,
                                                      predictor=predictors.get(key),
                                                      supercell_size=supercell, lj_envelope=mcfg['lj_envelope'],
                                                      max_iter=int(mcfg['max_iter']), tol=float(mcfg['tol']),
                                                      polish=mcfg['polish'],
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
        for r in rows:
            k_ = mn.row_key(r)
            running_min[k_] = min(running_min.get(k_, math.inf), final_energy(r))
        if library is not None:
            for m_ in range(n_lib0, len(library)):
                if tuple(library.key[m_][:3]) in cidx:
                    cond_minima[cidx[tuple(library.key[m_][:3])]].append(m_)
        quota_check(cidx[mn.row_key(r)] for r in rows if mn.row_key(r) in cidx)
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
    quota_check(range(C))  # a resumed run: conditions already at their quota draw nothing
    first = next_start()
    if first is not None:
        d = int(first[1].full_cell_parameters().shape[-1])
        slots = _Slots(S, d, device)
        load(0, *first)
        for i in range(1, S):
            s_ = next_start()
            if s_ is None:
                break
            load(i, *s_)
    gstep, templates, tmpl = 0, None, None
    eval_chunk = {c[2]: S for c in cuts}  # rows per energy evaluation, per target; halved on a CUDA out-of-memory
    clean = 0  # steps since the last out-of-memory error                 error in that target's call (rows independent)
    stopping = False  # a stop request: no new starts; the loop ends when the slots in flight are done
    nz = (d - 6) // 6 if d is not None else 0

    def step_chunk(template, idx, cut):
        """One evaluation of the rows ``idx`` (built on ``template``) at ``cut`` (cutoff, supercell size, target):
        loss [n] (CPU), gradient [n, d], recorded parameters [n, d] (cleaned, centre unwrapped as
        gradient_descent_optimization records)."""
        target = cut[2]
        out_key = 'reduction_en' if target == 'reduce' else target
        computes = [target] if target in mn.MLIP_TARGETS else list({'lj', 'elj', out_key})
        cb = template.clone().detach()
        x = slots.X[idx].clone().requires_grad_(True)
        cb.set_cell_parameters(x, skip_box_analysis=True)
        cb.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
        cb.aunit_centroid = wrap_centroid(x[:, 6:6 + 3 * nz])
        outputs, cluster = cb.analyze(computes=computes, cutoff=cut[0], rdf_cutoff=cut[0], supercell_size=cut[1],
                                      return_cluster=True, repulsion=1, surface_padding=0, std_orientation=True,
                                      margin=0, assign_outputs=True, elementwise=True, atomwise=False, bins=100,
                                      predictor=predictors.get(target),
                                      **(envelope if target in ('lj', 'qlj', 'elj') else {}))
        loss = outputs[out_key].reshape(-1)
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
        with torch.no_grad():  # a flat cell (each angle in range, together near zero volume) has a cluster of
            c_ = slots.X[occ, 3:6].clamp(*ANGLE_RANGE).double().cos()  # gigabytes: drop the row before the step
            flat = [i for i, v in zip(occ, (1 - (c_ ** 2).sum(1) + 2 * c_.prod(1)).tolist()) if v < FLAT_VF]
        if flat:
            for i in flat:
                slots.row[i] = None
                inflight[slot_c[i]] -= 1
            effort['flat_dropped'] += len(flat)
            if not stopping and not saturated[0]:
                for i in flat:
                    s_ = next_start()
                    if s_ is not None:
                        load(i, *s_)
            templates = None
            continue
        if templates is None:  # rows grouped by their stage's (cutoff, supercell size), then chunked
            groups = {}
            for i in occ:
                groups.setdefault(cuts[int(slots.stage[i])], []).append(i)
            tmpl = torch.tensor([i for c in groups for i in groups[c]])
            templates = [(collate_data_list([slots.row[i] for i in g_[c:c + eval_chunk[cut[2]]]]).to(device),
                          torch.tensor(g_[c:c + eval_chunk[cut[2]]]), cut) for cut, g_ in groups.items()
                         for c in range(0, len(g_), eval_chunk[cut[2]])]
        effort['slot_steps'] += len(occ)  # occupied slots per step: occupancy = slot_steps / (global_steps x slots)
        parts, failed = [], None
        try:
            for t, i, cut in templates:
                failed, failed_rows = cut[2], (i, cut[0])
                parts.append(step_chunk(t, i, cut))
        except RuntimeError as e:  # nothing has been updated yet: the step is redone in smaller chunks
            if not is_cuda_oom(e) or eval_chunk[failed] == 1:
                raise
            eval_chunk[failed] = max(1, eval_chunk[failed] // 2)
            templates, clean, parts = None, 0, None
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
            alloc, resv = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
            with torch.no_grad():  # the failed chunk's largest periodic cluster, in unit cells: the reach sphere's
                ix, cut0 = failed_rows  # volume over the cell's (as build_cluster's selection, for a cell small
                X_ = slots.X[ix].double().cpu()  # against the reach)
                c_ = X_[:, 3:6].clamp(*ANGLE_RANGE).cos()
                vol = X_[:, :3].prod(1) * (1 - (c_ ** 2).sum(1) + 2 * c_.prod(1)).clamp_min(0).sqrt()
                reach = torch.tensor([float(slots.row[j].radius.reshape(-1).max()) for j in ix.tolist()],
                                     dtype=torch.float64)
                cells = 4 / 3 * math.pi * (float(cut0) + 2 * reach) ** 3 / vol.clamp_min(1e-12)
                k = int(cells.argmax())
            print(f"{config.run_name}: out of memory; evaluating {eval_chunk[failed]} {failed} rows at a time (this "
                  f"process: {alloc:.2f} GiB allocated, {resv:.2f} GiB reserved; device: {free / 2**30:.2f} of "
                  f"{total / 2**30:.2f} GiB free; the chunk's largest cluster about {float(cells[k]):.0f} cells: "
                  f"lengths {[round(float(v), 2) for v in X_[k, :3]]} A, volume {float(vol[k]):.1f} A^3)")
            continue
        lc, g, rec = (torch.cat([p_[k] for p_ in parts]) for k in range(3))
        del parts
        clean += 1
        if clean >= REGROW_STEPS and any(v < S for v in eval_chunk.values()):  # a passing peak (e.g. after a
            eval_chunk = {t_: min(S, 2 * v) for t_, v in eval_chunk.items()}    # polish) must not halve the rest
            templates, clean = None, 0
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
            ref = running_min.get(keys[slot_c[i]])  # this condition's lowest finished energy
            while ref is not None and int(slots.casc_k[i]) < len(casc) and \
                    int(slots.nstep[i]) > casc[int(slots.casc_k[i])][0]:
                now = float(slots.bf[i]) if is_bfgs[int(slots.stage[i])] and bool(slots.bhave[i]) else \
                    float(slots.last_loss[i])  # a bfgs row: its accepted point, not a rejected trial
                if now > ref + casc[int(slots.casc_k[i])][1] * kT:
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
                for i in sorted(out):
                    s_ = next_start()
                    if s_ is not None:
                        load(i, *s_)
            templates = None
        while not saturated[0]:  # the first random-start group may be smaller: the library starts filling sooner
            chunk = scfg['polish_chunk'] if groups_done[0] else scfg['first_polish_chunk']
            if len(pending) < chunk:
                break
            polish(pending[:chunk])
            del pending[:chunk]
            groups_done[0] += 1
        if any(closed):  # a condition at its quota: its rows in slots and awaiting the polish are not finished
            gone = [i for i in slots.occupied() if closed[slot_c[i]]]
            keep = [r for r in pending if not closed[cidx[mn.row_key(r)]]]
            effort['quota_dropped'] += len(gone) + len(pending) - len(keep)
            for r in pending:
                if closed[cidx[mn.row_key(r)]]:
                    waiting[cidx[mn.row_key(r)]] -= 1
            pending[:] = keep
            for i in gone:
                slots.row[i] = None
                inflight[slot_c[i]] -= 1
            if gone:
                templates = None
        if not stopping and not saturated[0] and refill():  # slots a polish group freed: a condition's rows are
            templates = None  # no longer awaiting it, or a condition met its quota
        if saturated[0]:  # stop_p met: end now; rows in flight and emitted rows not yet polished are not finished
            effort['stopped']['dropped'] = len(slots.occupied()) + len(pending)
            pending.clear()
            break
    if pending:
        polish(pending[:])
        pending.clear()
    effort.update(wall_s=time.time() - t0, global_steps=gstep, rows=len(opt_outs),
                  occupancy=effort['slot_steps'] / max(gstep * S, 1),
                  fd_merged=library.fd_merged if library is not None else 0)
    if mcfg is not None:
        atomic_torch_save(mn.minimum_rows(opt_outs), str(out_path)[:-3] + '_minima.pt')
        per = {}
        for r in opt_outs:
            if mn.row_key(r) in cidx and int(r.start_index) >= 0:
                per[cidx[mn.row_key(r)]] = per.get(cidx[mn.row_key(r)], 0) + 1
        cond = []
        for c, k_ in enumerate(keys):
            fs = [library.f[m_] for m_ in cond_minima[c]]
            band = scfg['quota_band_kT'] if scfg['quota'] is not None else scfg['stop_band_kT']
            cond.append(dict(identifier=k_[0], sg=k_[1], z_prime=k_[2], starts_finished=per.get(c, 0),
                             minima=len(fs), band_kT=band,
                             band_minima=sum(f_ - min(fs) <= band for f_ in fs) if fs else 0,
                             lowest_f=min(fs) if fs else None, quota_met=bool(closed[c])))
        atomic_json_dump(cond, str(out_path)[:-3] + '_conditions.json')
    atomic_json_dump(effort, str(out_path)[:-3] + '_stream_effort.json')
    print(f"{config.run_name}: stream effort {json.dumps(effort)}")
    return opt_outs
