"""
A script for loading a batch of molecules and optimizing them against a given property via torch autograd
"""
import gc
import json
import os
import pickle
import time
from pathlib import Path
from time import sleep

import numpy as np
import torch
from tqdm import tqdm

from mxtaltools.common.config_processing import load_yaml, dict2namespace
from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.crystal_search.run_state import RunLease, StopRequest, atomic_json_dump, atomic_torch_save, \
    committed_outputs, read_json
from mxtaltools.crystal_search.utils import get_initial_state, init_samples_to_optim, parse_args, parse_opt_config, \
    recover_opt_state, process_target
from mxtaltools.dataset_utils.utils import collate_data_list


def reject_retired_keys(config):
    """The latent-space umbrella repulsion is removed; a config still carrying its keys
    would otherwise run without it and read as if it had been applied."""
    found = ['umbrella_path'] if hasattr(config, 'umbrella_path') else []
    for i, opt in enumerate(config.opt):
        opt = opt if isinstance(opt, dict) else vars(opt)
        found += [f'opt[{i}].{k}' for k in opt if k.startswith('umbrella')]
    if found:
        raise ValueError(f"retired crystal-search keys (umbrella repulsion was removed): {found}")


def run_files(out_path):
    """Per-run side files next to the output. Named after the run so two searches sharing a working directory can
    never read (or delete) each other's -- the May acridine chunks carry copies across arms, consistent with one
    shared opt_intermediates.pt."""
    stem = str(out_path)[:-3] if str(out_path).endswith('.pt') else str(out_path)
    return stem + '_opt_intermediates.pt', stem + '_progress.json'


def discard_intermediates(path):
    """The intermediates file belongs to ONE batch position (cursor): a retry of that same cursor may resume from it,
    nothing else may. It is removed at the start of a run and after every completed batch. Before, a later batch
    whose retry followed a step-0 OOM (which writes no file) reloaded a stale file and relaxed an EARLIER batch's
    walkers -- 46-55% of the relaxations in the acr_wrap_sep26 random-start jobs repeated earlier starts."""
    if os.path.exists(path):
        os.remove(path)


def crystal_search(config):
    reject_retired_keys(config)
    device = config.device

    if device == 'cuda':
        # prevents from dipping into windows virtual vram which is super slow
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    if config.target_path is not None:
        target, config = process_target(config)
    else:
        target = None

    samples_to_optim = init_samples_to_optim(config, target=target)
    data_mode = config.init_sample_method == 'data'
    if data_mode:
        # outputs carry the index of the seed they came from: the enforce_reduced filter drops rows, so position in
        # the output list is not the seed's position
        for i, sample in enumerate(samples_to_optim):
            sample.dataset_index = torch.tensor([i], dtype=torch.long)

    out_path = Path(config.out_dir + f"/{config.run_name}.pt")  # where to save outputs
    intermediates, progress = run_files(out_path)
    num_samples = len(samples_to_optim)
    print(f"Starting optimization of {num_samples} crystal samples")
    coord = _coordination(config)  # fallible setup first: nothing below may fail while holding a lease it cannot free
    lease = RunLease(str(out_path)[:-3], stale_after_s=getattr(config, 'lease_stale_after_s', 1800),
                     settle_s=getattr(config, 'lease_settle_s', 2.0))
    stop = StopRequest(stop_files=(getattr(config, 'stop_files', None) or []) + (coord['stop_files'] if coord else []))
    try:
        lease.acquire()  # one writer per run name; raises if another live process holds it
        return _search_loop(config, samples_to_optim, target, device, data_mode, out_path, intermediates, progress,
                            num_samples, lease, stop, coord)
    finally:
        stop.close()
        lease.release()  # removes the lease file only if this process owns it


def _coordination(config):
    """Campaign coordination (crystal_search/coordinator.py), opt-in through cfg:coord_dir: this job writes one shard per
    completed batch there and stops between batches when the campaign's STOP or STOP.<coord_stream> file appears."""
    coord_dir = getattr(config, 'coord_dir', None)
    if not coord_dir:
        return None
    from mxtaltools.crystal_search import coordinator
    stream = getattr(config, 'coord_stream', 'default')
    last = config.opt[-1] if isinstance(config.opt[-1], dict) else vars(config.opt[-1])
    return dict(dir=coord_dir, stream=stream, stop_files=coordinator.stop_files(coord_dir, stream),
                model_id=coordinator.energy_model_id(last, config), energy_key=str(last['optim_target']).lower(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                write=coordinator.write_shard, curate_every_s=getattr(config, 'coord_curate_every_s', None),
                maybe_curate=coordinator.maybe_curate)


HOP_WAIT = 'hop parents not available yet'


class HopParentsUnavailable(Exception):
    """hop_parents.pt stayed missing or unreadable for longer than the job's hop wait."""


def _hop_batch(config, samples_to_optim, cursor, num_samples, coord, device, batch_idx, after_pass=0):
    """init_sample_method 'hops': starts are latent log-noise kicks of registry basins (the campaign's hop_parents.pt,
    rewritten by every curate pass), parents drawn with weight 1 / (1 + hop starts so far), seeded by
    opt_seed + batch_idx * 10000. Each start's dataset_index is its parent basin.
    Returns HOP_WAIT while the file is missing or unreadable (no pass has written it yet, or a read failed), and also
    when it lists no parent but was written by a pass older than after_pass (a pass that may not yet have seen this
    job's own last shard, whose children could become parents). None only when a pass numbered >= after_pass lists no
    eligible parent: the stream is exhausted."""
    path = os.path.join(coord['dir'], 'hop_parents.pt')
    try:
        hp = torch.load(path, weights_only=False)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):
        return HOP_WAIT
    if len(hp['basin']) == 0:
        return None if int(hp.get('passes', 0)) >= after_pass else HOP_WAIT
    if hp.get('energy_model_id') != coord['model_id']:
        raise ValueError(f"{path} lists parents scored by {hp.get('energy_model_id')!r}, this job uses "
                         f"{coord['model_id']!r}")
    n = min(config.batch_size, num_samples - cursor)
    seed = int(config.opt_seed) + int(batch_idx) * 10000
    g = torch.Generator().manual_seed(seed)
    pick = torch.multinomial(1.0 / (1.0 + hp['hop_starts'].double()), n, replacement=True, generator=g)
    rows = []
    for k, j in enumerate(pick.tolist()):
        c = samples_to_optim[cursor + k].clone()
        c.dataset_index = torch.tensor([int(hp['basin'][j])], dtype=torch.long)
        rows.append(c)
    b = collate_data_list(rows).to(device)
    b.set_cell_parameters(hp['params'][pick].to(device))
    b.aunit_handedness = hp['hand'][pick].to(device=device, dtype=b.aunit_handedness.dtype).reshape(
        b.aunit_handedness.shape)
    b.box_analysis()
    b.canonicalize_orientation()
    lo, hi = hp['log_noise']
    torch.manual_seed(seed)  # log_noise_latent_parameters draws from the global generator
    # kicked from the parent itself: a parent outside the latent box (a long reduced-cell axis; 1.7% of acridine
    # states within 2 kT, 2026-09-28) was otherwise clipped into a different crystal before the kick
    b.log_noise_latent_parameters(float(lo), float(hi), keep_start_representable=True)
    return b


def _search_loop(config, samples_to_optim, target, device, data_mode, out_path, intermediates, progress, num_samples,
                 lease, stop, coord=None):
    if os.path.exists(out_path) and not config.force_restart_run:
        opt_outs = torch.load(out_path, weights_only=False)
        if os.path.exists(progress):
            # the cursor, not len(opt_outs): enforce_reduced drops rows, and a cursor recomputed from the output
            # length re-relaxed completed samples; batch_idx continues so random starts are never redrawn
            with open(progress) as fh:
                state = json.load(fh)
            cursor, batch_idx = int(state['cursor']), int(state['batch_idx'])
            opt_outs = committed_outputs(opt_outs, state)  # drop a batch saved without its progress update
        else:  # output written before progress files existed
            cursor = len(opt_outs)
            batch_idx = (cursor // config.batch_size) - 1  # so batch_idx+=1 lands on the right value
        if cursor >= num_samples:
            print(f"{config.run_name}: all {num_samples} samples already done")
            return opt_outs
    else:
        opt_outs = []
        cursor = 0
        batch_idx = -1
        # commit the empty state: a kill in the first batch then resumes from here (fresh draws after batch_idx -1)
        # instead of falling back to len(outputs) with no batch index, which redraws the first batch's starts
        atomic_json_dump({'cursor': 0, 'batch_idx': -1, 'num_samples': num_samples, 'n_out': 0}, progress)

    num_opts = len(config.opt)
    finished = False
    pbar = tqdm(total=num_samples, unit="samples")
    prev_best_samples = None
    discard_intermediates(intermediates)  # a file left by an earlier (crashed) attempt of this run is not current
    attempt = dict(cursor=None, n=0, t0=0.0)  # attempts at the current cursor (OOM retries included), for shards
    hops_mode = config.init_sample_method == 'hops'
    last = config.opt[-1] if isinstance(config.opt[-1], dict) else vars(config.opt[-1])
    final_key = str(last['optim_target']).lower()
    # the reference of cfg:opt[i].early_stop: this job's lowest final energy so far (None until a batch completes)
    running_min = min((float(getattr(r, final_key)) for r in opt_outs if final_key in r.keys()), default=None)
    if hops_mode and coord is None:
        raise ValueError("init_sample_method 'hops' draws its starts from a campaign: set cfg:coord_dir")
    # hop mode: how long a job waits for hop_parents.pt (the first curate pass may be running in another job, or a
    # killed curator's lease must go stale first: max(3 x curate interval, 3600 s)), and the pass that must list no
    # parent before the stream counts as exhausted (a pass that started after this job's last shard)
    curate_every = (coord or {}).get('curate_every_s') or 900.0
    hop_wait = dict(since=None, limit=float(getattr(config, 'coord_hop_wait_s', None)
                                            or max(3 * curate_every, 3600.0) + 1800.0),
                    poll=float(getattr(config, 'coord_hop_poll_s', None) or 60.0), after_pass=0, told=False)
    # cfg:oom_ceiling: after an OOM at batch size B, growth stops at 0.95 B (probing 5% higher after 10 clean batches)
    ceiling = dict(value=None, clean=0)

    while not finished:
        if coord is not None and coord['curate_every_s'] is not None:  # this job may run the campaign's curate pass
            coord['maybe_curate'](coord['dir'], coord['curate_every_s'])
        reason = stop.reason()
        if reason is not None:  # between batches: every completed batch is saved, a later resume continues
            print(f"{config.run_name}: stopping at cursor {cursor} ({reason})")
            break
        hop_starts = None
        if hops_mode:  # drawn before the batch bookkeeping, so waiting consumes no batch index or attempt
            hop_starts = _hop_batch(config, samples_to_optim, cursor, num_samples, coord, device, batch_idx + 1,
                                    after_pass=hop_wait['after_pass'])
            if hop_starts is HOP_WAIT:  # bounded: the loop top re-runs maybe_curate and re-checks the stop files
                hop_wait['since'] = hop_wait['since'] or time.time()
                waited = time.time() - hop_wait['since']
                if waited > hop_wait['limit']:
                    raise HopParentsUnavailable(
                        f"{config.run_name}: no usable hop_parents.pt in {coord['dir']} after {waited:.0f} s (limit "
                        f"{hop_wait['limit']:.0f} s): is any job running the curate pass? Resubmit once it exists.")
                if not hop_wait['told']:
                    print(f"{config.run_name}: {HOP_WAIT} (missing, unreadable, or not yet from a pass after this "
                          f"job's last shard); waiting up to {hop_wait['limit']:.0f} s")
                    hop_wait['told'] = True
                sleep(hop_wait['poll'])
                continue
            hop_wait.update(since=None, told=False)
            if hop_starts is None:
                print(f"{config.run_name}: no eligible hop parent in the campaign (the stream is exhausted); "
                      f"stopping at cursor {cursor}")
                break
        crystal_batch, opt_config = None, {}
        if attempt['cursor'] != cursor:
            attempt.update(cursor=cursor, n=0, t0=time.time())
        attempt['n'] += 1
        mlip_evals, other_evals, n_relax = 0, 0, None
        n_starts = min(config.batch_size, num_samples - cursor)
        try:
            batch_idx += 1
            if hops_mode:
                crystal_batch = hop_starts
            else:
                crystal_batch = collate_data_list(samples_to_optim[cursor:cursor + config.batch_size]).to(device)
                if (prev_best_samples is None) or (
                        prev_best_samples is not None and len(prev_best_samples) < crystal_batch.num_graphs):
                    crystal_batch = get_initial_state(config, crystal_batch, device, batch_idx)
                else:
                    crystal_batch = recover_opt_state(crystal_batch, config, device, batch_idx, prev_best_samples)

            for opt_ind, stage in enumerate(config.opt):
                # do optimization in N stages
                opt_config = parse_opt_config(dict(stage), config, device, target)
                # cfg:opt[i].keep_lowest_fraction: after this stage keep only that fraction of rows, lowest energy
                # first (the stage's own target) -- e.g. a cheap eLJ stage choosing which starts the MLIP stages relax
                keep_frac = opt_config.pop('keep_lowest_fraction', None)
                if opt_config.get('early_stop') is not None:
                    opt_config['early_stop_ref'] = running_min
                opt_config['intermediates_path'] = intermediates
                stage_key = str(opt_config['optim_target']).lower()
                is_mlip = coord is None or stage_key == coord['energy_key']
                if is_mlip and n_relax is None:
                    n_relax = crystal_batch.num_graphs  # relaxations by the campaign's energy model in this batch

                'do opt'
                opt_out, opt_record = crystal_batch.optimize_crystal_parameters(return_record=True, **opt_config)
                if isinstance(opt_record, dict) and torch.is_tensor(opt_record.get('loss')):
                    # row-steps evaluated: a row retired by the cascade records +inf for the steps it skipped
                    n_eval = int(opt_record['loss'].numel() - torch.isinf(opt_record['loss']).sum())
                    if is_mlip:
                        mlip_evals += n_eval
                    else:
                        other_evals += n_eval

                if config.save_trajs:
                    opt_record.update({'base_crystal': samples_to_optim[0]})
                    atomic_torch_save(opt_record, Path(str(out_path).replace('.pt', f'_traj{batch_idx}_{opt_ind}.pt')))

                if 'predictor' in opt_config.keys():
                    del opt_config['predictor']
                if 'score_model' in opt_config.keys():
                    del opt_config['score_model']

                if len(opt_out) == 0:  # the enforce_reduced filter dropped every row of this batch
                    crystal_batch = None
                    break
                if keep_frac is not None and len(opt_out) > 1:
                    energies = [float(getattr(r, stage_key)) for r in opt_out]
                    k = max(1, int(np.ceil(float(keep_frac) * len(opt_out))))
                    lowest = sorted(sorted(range(len(opt_out)), key=lambda i: energies[i])[:k])
                    opt_out = [opt_out[i] for i in lowest]
                crystal_batch = collate_data_list(opt_out).to(device)

            new_rows = []
            if crystal_batch is not None:
                crystal_batch.box_analysis()
                new_rows = crystal_batch.cpu().detach().batch_to_list()
                opt_outs.extend(new_rows)
                finals = [float(getattr(r, final_key)) for r in new_rows if final_key in r.keys()]
                if finals:
                    running_min = min(finals + ([running_min] if running_min is not None else []))

            lease.check()  # raises RunLeaseLost, before writing, if another process has taken the run over
            atomic_torch_save(opt_outs, out_path)  # outputs first: a kill before the progress write is undone on resume
            if coord is not None:  # fail-open: returns False rather than raise on any coordination I/O error
                coord['write'](coord['dir'], config.run_name, coord['stream'], cursor, batch_idx, new_rows,
                               coord['energy_key'],
                               dict(n_relaxations=n_relax or 0, n_starts=n_starts, row_evals=mlip_evals,
                                    row_evals_other=other_evals, n_attempts=attempt['n'],
                                    wall_s=time.time() - attempt['t0'], gpu=coord['gpu'],
                                    opt_seed=getattr(config, 'opt_seed', None), energy_model_id=coord['model_id'],
                                    code_version=getattr(config, 'mxt_commit', None)))
                if hops_mode:  # an empty parent list counts as exhaustion only from a pass started after this shard
                    hb = read_json(os.path.join(coord['dir'], 'heartbeat.json')) or {}
                    hop_wait['after_pass'] = int(hb.get('passes', 0)) + 2

            cursor = min(cursor + config.batch_size, num_samples)  # a final partial batch ends at num_samples
            atomic_json_dump({'cursor': cursor, 'batch_idx': batch_idx, 'num_samples': num_samples,
                              'n_out': len(opt_outs)}, progress)
            prev_best_samples = None
            discard_intermediates(intermediates)  # this cursor is done; its saved states must not seed the next one
            pbar.update(min(config.batch_size, num_samples - cursor))  # safe final update
            if cursor >= len(samples_to_optim):
                finished = True
            else:
                if config.grow_batch_size:
                    grown = int(config.batch_size * 1.2)  # keep pushing the batch size between sets
                    if getattr(config, 'oom_ceiling', False) and ceiling['value'] is not None:
                        ceiling['clean'] += 1
                        if ceiling['clean'] >= 10:  # probe: memory needs vary with the crystals drawn
                            ceiling.update(value=int(ceiling['value'] * 1.05) + 1, clean=0)
                        grown = min(grown, ceiling['value'])
                    config.batch_size = max(grown, 1)
                    print(f"Boosting batch size to {config.batch_size}")

            del crystal_batch


        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                if config.batch_size == 1:
                    assert False, "Cascading bsz error"
                if getattr(config, 'oom_ceiling', False):
                    ceiling.update(value=max(1, int(config.batch_size * 0.95)), clean=0)
                config.batch_size = max(int(config.batch_size * 0.9), 1)
                del crystal_batch  # may be None: the OOM can hit while the batch is still being built
                if 'predictor' in opt_config.keys():
                    del opt_config['predictor']
                if 'score_model' in opt_config.keys():
                    del opt_config['score_model']
                print(f"OOM error: dropping batch size to {config.batch_size}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                sleep(0.1)
                # only a file written by an attempt at THIS cursor can exist here (discard_intermediates). Seeded
                # (data-mode) runs never resume from it: after a stage drops rows, its row order no longer matches
                # the cursor's seeds, so a seed slot would resume another seed's state -- redo the seeds instead.
                if os.path.exists(intermediates) and not data_mode and not hops_mode:  # a hop redraw has new parents
                    prev_best_samples = torch.load(intermediates, weights_only=False)
            else:
                raise e


    return opt_outs

    print(f"Sampling complete! Optimized a total of {len(opt_outs)} crystal samples.")

    # batch = collate_data_list(opt_outs)
    # batch.plot_batch_cell_params(space='real', quantiles=[0.1, 0.5], split_by_sg=True)
    #
    # batch.plot_batch_density_funnel(split_by_sg=True)

    aa = 1

if __name__ == '__main__':
    args = parse_args()  # call config with "python run_search.py --config /path/to/config.yaml
    source_dir = Path(__file__).resolve().parent.parent.parent
    if args.config is None:
        config_path = source_dir / 'configs' / 'crystal_searches' / 'base.yaml'
    else:
        config_path = Path(args.config)

    config = dict2namespace(load_yaml(config_path))

    crystal_search(config)

"""


batch = collate_data_list(opt_outs)
batch.plot_batch_cell_params(space='real', quantiles=[0.1, 0.5])

batch.plot_batch_density_funnel()


"""
