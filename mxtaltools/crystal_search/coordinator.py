"""
Coordination of many parallel crystal_search jobs through a shared directory (a "campaign"), with a per-stream stopping
rule: stop a proposal stream once the expected effort to find the next new low-energy basin exceeds Z.

Layout of a campaign directory:
    coord.yaml                 campaign config (see CampaignConfig)
    shards/<run>/<cursor>_<batch>.pt   one per completed search batch, written by the GPU jobs (write_shard)
    registry.pt                the basin registry (single writer: the curate pass)
    stats.json / stats.md      per-stream statistics and verdicts, rewritten each pass
    STOP.<stream> / STOP       stop files the GPU jobs poll between batches
    heartbeat.json             time and pass count of the last curate pass

GPU jobs never wait on anything here: they write shards (fail-open: a coordination I/O error never stops a search) and
check the STOP files between batches. One curate pass (curate) is idempotent; every pass takes the curator lease
(curate_locked: in a job through maybe_curate, by hand through main, in a loop through run_loop), and a pass while
another holds it is skipped, never waited on. If the coordinator stops, the jobs simply run to their own budgets.

Basins. A curate pass takes each physical state within window_kT of the energy reference, re-expresses it in its reduced
cell (standardize.standardize_cells; RDFs do not depend on the cell choice, but the registry keeps reduced
descriptions), computes its RDF, and assigns it to the basin of the nearest leader within identity_cut, else opens a basin
(leader clustering: the first state of a basin is its leader; later states only join). identity_cut must be calibrated
per system against a structure-identity test (COMPACK): the RDF distance at which two structures are the same packing
differs between molecules and between kinds of pair (acridine end states: 0.050 by the rule of the GFN repository's
campaign_compack.py, where 0.085 had held against the experimental forms; MIPCAS and NEHZOR landscapes: 0.12-0.13 at a
fitted P(match) = 0.95). When unsure, a
smaller cut splits one structure into several basins, which only delays the stop; a larger one merges different
structures and stops early. The config records whether the cut was calibrated. Calibrate it on a short pilot campaign
before the real one (the GFN repository's energy_sampling/eval/campaign_compack.py calibrate does this within a fixed
COMPACK budget): changing the cut under a running campaign would reassign the basins it has built.

Stopping rule, per stream s (random starts, eLJ-prescreened starts, hops, ...), for each band b (e.g. 2 kT and 1 kT):
    f1_s  = basins with exactly one state within the band, over all streams and priors, that state from s; a hop state
            that relaxed back into the basin it was kicked from, or one of that basin's ancestors, is left out of every
            count (lineage_returns): it exists because its parent was found, so it is not an independent draw
    W_s   = the stream's effort: row-evaluations of the campaign's energy model (the final stage's target), from the
            shards; cheaper stages (an eLJ pre-screen) are recorded apart (row_evals_other) and not counted; failed
            attempts count through wall time only
    stop s when W_s / f1_hi(f1_s) > Z_b, where f1_hi is the exact one-sided 90% Poisson upper bound on f1_s,
    and N_s >= min_relaxations and the band holds at least min_hits of the stream's states, on passes_to_confirm
    passes in a row, each after new effort from s (two passes over the same data are one look).
The Good-Turing ratio W/f1 estimates the effort per new basin without bias on the sep27 acridine data; the upper bound on
f1 keeps it from stopping early by chance. A stream is exhausted when every band says stop; once STOP.<s> exists the
stream stays stopped. The campaign stops (STOP) when every stream is stopped -- a hop stream with no eligible parent
counts as stopped for this -- or total effort passes hard_cap. Termination: a stream with no new basins stops at
W_s > 2.3 Z (f1_hi(0) = 2.30) once its jobs keep working, or once it has brought no new work for confirm_settle_s or
three times its longest shard (whichever is longer) while the test holds; a stream that never reaches min_hits in a band,
or whose jobs have all ended while its test fails, is bounded only by hard_cap (or a STOP file written by hand); nothing waits on a particular job.

Model-agnostic: energies are whatever the search objective reports (energy_key), kT and the energy reference come from
the config, and every shard carries the id of the energy model that scored it; the curate pass refuses shards (and priors)
whose model id differs from the campaign's.
"""
import argparse
import glob
import hashlib
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from mxtaltools.crystal_search.run_state import RunLeaseLost, atomic_json_dump, atomic_torch_save, read_json

RDF_KW = dict(cutoff=10, rdf_cutoff=10, supercell_size=10, bins=100, rdf_mode='envwise', std_orientation=True)
RDF_DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins', 'uma']
LJ_BLOWN_UP = 1e4  # physicality: no state with lj above this is a crystal (energy_sampling summarize_search.physical)
MIN_ANGULAR_FACTOR = 0.1  # physicality: cell_volume / prod(cell_lengths) (collate_prior.py's criterion)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class StreamRule:
    Z: Dict[str, float]  # band name -> effort threshold per new basin (row-evaluations)
    min_relaxations: int = 2000
    min_hits: int = 20


@dataclass
class CampaignConfig:
    mol_path: str
    sg: int
    z_prime: int
    energy_key: str  # the per-row energy attribute the search stores, e.g. 'mace'
    energy_model_id: str  # must equal every shard's (energy_model_id() of the scoring job)
    kT: float  # in the energy's units
    bands_kT: Dict[str, float]  # band name -> width above the energy reference, in kT
    identity_cut: float  # RDF distance below which two states are one basin
    identity_calibrated: bool = False
    identity_note: str = ''
    energy_ref: Optional[float] = None  # fixed reference; None = lowest energy in the registry
    window_kT: float = 5.0  # states within this of the reference enter the registry
    streams: Dict[str, StreamRule] = field(default_factory=dict)
    hard_cap: float = float('inf')  # total effort (row-evaluations) after which the campaign stops
    priors: List[str] = field(default_factory=list)  # files of known structures, scored by the same model
    passes_to_confirm: int = 2
    # a stream whose stop test holds is confirmed at once after max(confirm_settle_s, 3 x its longest shard's wall time)
    # with no new work from it (verdicts): its statistics are final, so another pass cannot change the verdict
    confirm_settle_s: float = 3600.0
    rdf_batch: int = 32
    # RDF channels of the identity metric: 'envwise' (atoms pooled by environment class, so a molecule's own symmetry
    # relabelling costs nothing) or 'atomwise' (every atom its own channel); recorded in the registry and checked
    rdf_mode: str = 'envwise'
    # hop stream (optional): stream name, basins within window_kT of the reference are parents, at most max_per_basin
    # hop starts each and max_generation generations deep (a basin first found by a hop is one generation below its
    # parent); log_noise is the (low, high) latent log-noise range of the kicks; exhaust_settle_s (default 3600) is how
    # long the hop stream must bring no new work, with no eligible parent, before it counts as stopped for the campaign
    # STOP. Termination: the quota and the depth cap bound the number of hop starts; a hop job with no eligible parent
    # stops.
    hops: Optional[dict] = None

    @staticmethod
    def load(path):
        raw = yaml.safe_load(open(path))
        raw['streams'] = {k: StreamRule(**v) for k, v in (raw.get('streams') or {}).items()}
        cfg = CampaignConfig(**raw)
        if cfg.rdf_mode not in ('envwise', 'atomwise'):
            raise ValueError(f"rdf_mode must be 'envwise' or 'atomwise', not {cfg.rdf_mode!r}")
        return cfg


def energy_model_id(opt_stage, config):
    """A string identifying the energy model that scores a search's final stage: its target and, for an MLIP, the model
    file's size and the SHA1 of its first and last MB (so a replaced checkpoint at the same path gets a new id)."""
    target = opt_stage['optim_target'] if isinstance(opt_stage, dict) else opt_stage.optim_target
    path = {'mace': getattr(config, 'mace_predictor_path', None),
            'uma': getattr(config, 'uma_predictor_path', None)}.get(str(target).lower())
    if path is None:
        return str(target)
    h = hashlib.sha1()
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        h.update(fh.read(1 << 20))
        if size > 2 << 20:
            fh.seek(-(1 << 20), os.SEEK_END)
            h.update(fh.read(1 << 20))
    return f'{target}:{os.path.basename(path)}:{size}:{h.hexdigest()[:16]}'


# ---------------------------------------------------------------------------
# GPU-job side
# ---------------------------------------------------------------------------

def stop_files(coord_dir, stream):
    return [os.path.join(coord_dir, 'STOP'), os.path.join(coord_dir, f'STOP.{stream}')]


def write_shard(coord_dir, run_name, stream, cursor, batch_idx, rows, energy_key, meta):
    """Write one shard for a completed batch. rows: list of MolCrystalData outputs. Fail-open: returns False (and prints)
    instead of raising, so a coordination problem never stops a search job."""
    try:
        if rows:
            from mxtaltools.dataset_utils.utils import collate_data_list
            b = collate_data_list([r.clone().cpu() for r in rows], exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
            params = b.full_cell_parameters().detach().float()
            hand = b.aunit_handedness.detach().float().reshape(len(rows), -1)
            E = torch.tensor([float(getattr(r, energy_key)) for r in rows])
            lj = torch.tensor([float(getattr(r, 'lj', float('nan'))) for r in rows])
            didx = torch.tensor([int(r.dataset_index) if 'dataset_index' in r.keys() else -1 for r in rows])
            kick = torch.tensor([float(r.hop_kick) if 'hop_kick' in r.keys() else float('nan') for r in rows])
        else:
            params, hand = torch.zeros(0, 18), torch.zeros(0, 2)
            E, lj, didx = torch.zeros(0), torch.zeros(0), torch.zeros(0, dtype=torch.long)
            kick = torch.zeros(0)
        blob = dict(run=run_name, stream=stream, cursor=int(cursor), batch_idx=int(batch_idx), params=params,
                    handedness=hand, energy=E, lj=lj, dataset_index=didx, kick=kick, **meta)
        d = os.path.join(coord_dir, 'shards', run_name)
        os.makedirs(d, exist_ok=True)
        atomic_torch_save(blob, os.path.join(d, f'{int(cursor):08d}_{int(batch_idx):06d}.pt'))
        return True
    except Exception as e:  # noqa: BLE001 -- fail-open by design
        print(f'coordinator: shard write failed ({type(e).__name__}: {e}); the search continues without it')
        return False


# ---------------------------------------------------------------------------
# RDFs and basin assignment
# ---------------------------------------------------------------------------

def rebuild_crystals(cfg, params, hand):
    """MolCrystalData crystals from stored parameters and the campaign's conformer (as search seeds are built)."""
    from mxtaltools.dataset_utils.data_classes import MolCrystalData
    mol = torch.load(cfg.mol_path, weights_only=False)
    mol = mol[0] if isinstance(mol, list) else mol
    for k in ('num_atoms', 'mass', 'mol_volume', 'radius'):  # per-molecule scalars: a [1] tensor (as a collated or
        v = getattr(mol, k, None)                            # dataset crystal carries) is concatenated, not summed,
        if torch.is_tensor(v) and v.numel() == 1:            # over the Z' molecules of the new crystal
            setattr(mol, k, v.reshape(()))
    zp = cfg.z_prime
    out = []
    for p, h in zip(params, hand):
        p = p.float()
        out.append(MolCrystalData(
            molecule=[mol.clone() for _ in range(zp)] if zp > 1 else mol.clone(), sg_ind=cfg.sg, z_prime=zp,
            max_z_prime=zp, cell_lengths=p[:3].clone(), cell_angles=p[3:6].clone(),
            aunit_centroid=p[6:6 + 3 * zp].clone(), aunit_orientation=p[6 + 3 * zp:6 + 6 * zp].clone(),
            aunit_handedness=h[:zp].float().clone() if zp > 1 else float(h[0]), do_box_analysis=True))
    return out


def compute_rdfs(crystals, batch_size=32, rdf_mode='envwise'):
    from mxtaltools.dataset_utils.utils import collate_data_list
    out = []
    for lo in range(0, len(crystals), batch_size):
        b = collate_data_list([c.clone() for c in crystals[lo:lo + batch_size]], exclude_keys=RDF_DROP)
        with torch.no_grad():
            o = b.analyze(['rdf'], assign_outputs=False, **dict(RDF_KW, rdf_mode=rdf_mode))
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        out.append(r.detach().cpu().float())
    return torch.cat(out) if out else torch.zeros(0)


def rdf_distance_matrix(ra, rb, bin_width=10 / 99, chunk=64):
    """[na, nb] distances equal to analysis.crystal_rdf.compute_rdf_distance: bin width times the mean, over channels
    active (nonzero) in either RDF, of the L1 distance between the per-channel normalised CDFs."""
    def prep(r):
        s = r.sum(-1, keepdim=True)
        return torch.cumsum(r / (s + 1e-10), -1).permute(1, 0, 2).contiguous(), (s[..., 0] > 1e-12).float()
    ca, aa = prep(ra)
    cb, ab = prep(rb)
    out = torch.empty(len(ra), len(rb))
    na, nb = aa.sum(1), ab.sum(1)
    for i in range(0, len(ra), chunk):
        s = torch.cdist(ca[:, i:i + chunk], cb, p=1).sum(0)
        inter = aa[i:i + chunk] @ ab.T
        union = (na[i:i + chunk, None] + nb[None] - inter).clamp_min(1)
        out[i:i + chunk] = bin_width * s / union
    return out


def physical(params, lj):
    """Rows that are crystals: lj below LJ_BLOWN_UP and angular factor V / (a b c) above MIN_ANGULAR_FACTOR."""
    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    al, be, ga = params[:, 3], params[:, 4], params[:, 5]
    cos_a, cos_b, cos_g = torch.cos(al), torch.cos(be), torch.cos(ga)
    ang = torch.sqrt((1 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2 * cos_a * cos_b * cos_g).clamp(min=0))
    return (torch.nan_to_num(lj, nan=0.0) < LJ_BLOWN_UP) & (ang > MIN_ANGULAR_FACTOR)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def new_registry(cfg):
    return dict(version=1, energy_model_id=cfg.energy_model_id, identity_cut=cfg.identity_cut, rdf_mode=cfg.rdf_mode,
                ingested=[],
                leaders=torch.zeros(0), basin_E=[], basin_params=[], basin_hand=[], basin_first=[], basin_gen=[],
                basin_parent=[], hits=dict(basin=[], stream=[], energy=[], run=[], cursor=[], parent=[]),
                effort={}, refused=[], passes=0, confirm={}, hop_starts={}, hop_blocked=[])


def _stream_effort(reg, stream):
    return reg['effort'].setdefault(stream, dict(relaxations=0, row_evals=0.0, attempts=0, wall_s=0.0, shards=0,
                                                 gpus={}))


def _assign(reg, cfg, rdfs, energies, params, hand, stream, run, cursor, parents=None, kicks=None):
    """Leader clustering of new states against the registry (and each other, in arrival order). parents: for hop
    states, the basin each was kicked from (sets the generation of a basin it opens)."""
    leaders = reg['leaders']
    n = len(rdfs)
    if n == 0:
        return
    if len(leaders):
        d = rdf_distance_matrix(rdfs, leaders)
        best_d, best_j = d.min(1)
    else:
        best_d, best_j = torch.full((n,), float('inf')), torch.zeros(n, dtype=torch.long)
    for i in range(n):
        par = -1 if parents is None else int(parents[i])
        kick = None if kicks is None else float(kicks[i])
        if best_d[i] < cfg.identity_cut:
            j = int(best_j[i])
        else:
            if len(reg['leaders']) > len(leaders):  # compare with basins opened earlier in this pass
                dn = rdf_distance_matrix(rdfs[i:i + 1], reg['leaders'][len(leaders):])[0]
                k = int(dn.argmin())
                if dn[k] < cfg.identity_cut:
                    j = len(leaders) + k
                    _record_hit(reg, j, stream, energies[i], params[i], hand[i], run, cursor, parent=par,
                                kick=kick)
                    continue
            j = len(reg['leaders'])
            reg['leaders'] = torch.cat([reg['leaders'].reshape(-1, *rdfs.shape[1:]), rdfs[i:i + 1]])
            reg['basin_E'].append(float('inf'))
            reg['basin_params'].append(params[i].clone())
            reg['basin_hand'].append(hand[i].clone())
            reg['basin_first'].append(dict(stream=stream, run=run, cursor=int(cursor)))
            reg['basin_gen'].append(reg['basin_gen'][par] + 1 if 0 <= par < len(reg['basin_gen']) else 0)
            reg['basin_parent'].append(par)
        _record_hit(reg, j, stream, energies[i], params[i], hand[i], run, cursor, parent=par, kick=kick)


def _record_hit(reg, j, stream, E, params, hand, run, cursor, parent=-1, kick=None):
    h = reg['hits']
    h['basin'].append(int(j))
    h['stream'].append(stream)
    h['energy'].append(float(E))
    h['run'].append(run)
    h['cursor'].append(int(cursor))
    h['parent'].append(int(parent))
    if kick is not None and kick == kick:  # a hop start's latent kick length (NaN or None: not a hop start)
        reg.setdefault('hit_kick', {})[len(h['basin']) - 1] = float(kick)
    if float(E) < reg['basin_E'][j]:
        reg['basin_E'][j] = float(E)
        reg['basin_params'][j] = params.clone()
        reg['basin_hand'][j] = hand.clone()


def _energy_ref(cfg, reg):
    if cfg.energy_ref is not None:
        return float(cfg.energy_ref)
    return min(reg['basin_E']) if reg['basin_E'] else float('inf')


def _ingest_rows(reg, cfg, params, hand, E, lj, stream, run, cursor, parents=None, kicks=None):
    """Physical states within window of the reference: standardise, RDF, assign. Returns the number admitted."""
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.dataset_utils.utils import collate_data_list
    keep = physical(params, lj)
    ref = _energy_ref(cfg, reg)
    if np.isfinite(ref):
        keep &= E <= ref + cfg.window_kT * cfg.kT
    idx = torch.nonzero(keep).flatten()
    if len(idx) == 0:
        return 0
    crystals = rebuild_crystals(cfg, params[idx], hand[idx])
    std, info = standardize_cells(collate_data_list(crystals), on_failure='flag')
    if not info['ok'].all():
        reg['refused'].append(dict(run=run, cursor=int(cursor), reason=f'{int((~info["ok"]).sum())} rows had no '
                                                                         f'reduced cell; kept as stored'))
    sparams = std.full_cell_parameters().detach().float()
    shand = std.aunit_handedness.detach().float().reshape(len(idx), -1)
    rdfs = compute_rdfs(std.batch_to_list(), cfg.rdf_batch, cfg.rdf_mode)
    _assign(reg, cfg, rdfs, E[idx], sparams, shand, stream, run, cursor,
            parents=None if parents is None else parents[idx], kicks=None if kicks is None else kicks[idx])
    return len(idx)


def _file_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ingest_priors(reg, cfg):
    """Known structures (earlier searches, experimental forms) as stream 'prior'. Each prior file is a list of
    MolCrystalData carrying energy_key and an 'energy_model_id' attribute (or the file sits beside <file>.model_id), or
    a compact dict (params, handedness, energy, energy_model_id). A file is keyed by its path and the SHA1 of its
    content, so a file whose content has changed is ingested again (its states are added; earlier hits stay)."""
    for path in cfg.priors:
        try:
            key = f'prior:{os.path.abspath(path)}@{_file_sha1(path)[:16]}'  # new content at a path is new knowledge
        except OSError as e:  # moved or deleted from the checkout: skip it this pass rather than stop every pass
            print(f'coordinator: prior {path} unreadable ({type(e).__name__}); skipped this pass')
            continue
        if key in reg['ingested']:
            continue
        try:
            _ingest_prior_file(reg, cfg, path, key)
        except Exception as e:  # noqa: BLE001 -- one broken file must not stop every pass; fixing it changes its key
            reg['refused'].append(dict(run=key, cursor=-1, reason=f'prior not ingested ({type(e).__name__}: '
                                                                  f'{str(e)[:120]})'))
            reg['ingested'].append(key)


def _ingest_prior_file(reg, cfg, path, key):
    """One prior file of ingest_priors, under its key: ingested, or refused for another energy model."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    lst = torch.load(path, weights_only=False)
    if isinstance(lst, dict) and 'params' in lst:  # compact prior: params [N, 18], handedness, energy, model id
        mid = lst.get('energy_model_id')
        if mid != cfg.energy_model_id:
            reg['refused'].append(dict(run=key, cursor=-1, reason=f'prior scored by {mid!r}, campaign model '
                                                                  f'is {cfg.energy_model_id!r}'))
        else:
            n = len(lst['params'])
            _ingest_rows(reg, cfg, torch.as_tensor(lst['params']).float(),
                         torch.as_tensor(lst['handedness']).float().reshape(n, -1),
                         torch.as_tensor(lst['energy']).float(),
                         torch.as_tensor(lst.get('lj', torch.zeros(n))).float(), 'prior', key, 0)
        reg['ingested'].append(key)
        return
    side = path + '.model_id'
    mid = open(side).read().strip() if os.path.exists(side) else getattr(lst[0], 'energy_model_id', None)
    if mid != cfg.energy_model_id:
        reg['refused'].append(dict(run=key, cursor=-1, reason=f'prior scored by {mid!r}, campaign model is '
                                                              f'{cfg.energy_model_id!r}'))
        reg['ingested'].append(key)
        return
    b = collate_data_list([c.clone() for c in lst], exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    params = b.full_cell_parameters().detach().float()
    hand = b.aunit_handedness.detach().float().reshape(len(lst), -1)
    E = torch.tensor([float(getattr(c, cfg.energy_key)) for c in lst])
    lj = torch.tensor([float(getattr(c, 'lj', 0.0)) for c in lst])
    _ingest_rows(reg, cfg, params, hand, E, lj, 'prior', key, 0)
    reg['ingested'].append(key)


def ingest_shards(reg, cfg, coord_dir):
    n_new = 0
    for path in sorted(glob.glob(os.path.join(coord_dir, 'shards', '*', '*.pt'))):
        key = os.path.relpath(path, coord_dir).replace('\\', '/')
        if key in reg['ingested']:
            continue
        try:
            s = torch.load(path, weights_only=False)
        except Exception as e:  # noqa: BLE001 -- a file mid-copy; retried next pass
            print(f'coordinator: cannot read {key} yet ({type(e).__name__})')
            continue
        reg['ingested'].append(key)
        if s.get('energy_model_id') != cfg.energy_model_id:
            reg['refused'].append(dict(run=s.get('run'), cursor=s.get('cursor'),
                                       reason=f"shard scored by {s.get('energy_model_id')!r}"))
            continue
        eff = _stream_effort(reg, s['stream'])
        eff['relaxations'] += int(s.get('n_relaxations', len(s['energy'])))
        eff['row_evals'] += float(s.get('row_evals', 0.0))
        eff['attempts'] += int(s.get('n_attempts', 1))
        eff['wall_s'] += float(s.get('wall_s', 0.0))
        eff['shards'] += 1
        eff['max_shard_wall_s'] = max(float(eff.get('max_shard_wall_s', 0.0)), float(s.get('wall_s', 0.0)))
        gpu = str(s.get('gpu', 'unknown'))
        eff['gpus'][gpu] = eff['gpus'].get(gpu, 0) + 1
        parents = None
        if cfg.hops and s['stream'] == cfg.hops.get('stream', 'hops'):
            parents = s['dataset_index']  # a hop job stores each start's parent basin there
            for b in parents.tolist():
                reg['hop_starts'][int(b)] = reg['hop_starts'].get(int(b), 0) + 1
        _ingest_rows(reg, cfg, s['params'], s['handedness'], s['energy'], s['lj'], s['stream'], s['run'], s['cursor'],
                     parents=parents, kicks=s.get('kick'))
        n_new += 1
    return n_new


# ---------------------------------------------------------------------------
# statistics and the stopping rule
# ---------------------------------------------------------------------------

def lineage_returns(reg):
    """[n_hits] bool: hop hits that relaxed back into the basin they were kicked from, or into one of its ancestors
    (the basins its lineage of hops came from). Such a hit exists only because its parent had been found, so it is not
    an independent draw: counted, it would turn the parent's finder's singleton into a double and stop that stream
    early. compute_stats leaves these hits out of every count."""
    h = reg['hits']
    basin = np.asarray(h['basin'], dtype=np.int64)
    parent = np.asarray(h.get('parent', [-1] * len(basin)), dtype=np.int64)
    bp = reg.get('basin_parent', [])
    out = np.zeros(len(basin), dtype=bool)
    for i in np.nonzero(parent >= 0)[0]:
        p, b = int(parent[i]), int(basin[i])
        for _ in range(len(bp) + 1):  # the chain is at most max_generation long; the bound guards a corrupt cycle
            if p == b:
                out[i] = True
                break
            if not 0 <= p < len(bp):
                break
            p = int(bp[p])
    return out


def backfill_lineage(reg, cfg, coord_dir):
    """Rebuild the lineage record (hits['parent'], basin_parent) from the shards, for a registry written before hits
    carried their hop parent or extended since by such code. Hits are appended one per admitted row of each ingested
    shard, in row order, and the admitted rows are the shard's physical rows at or below the admission window's edge
    at that time; so each run of consecutive hits with one (stream, run, cursor) is matched to the lowest-energy
    physical rows of that shard, as many as the hits, which must carry exactly the hits' energies in row order (no
    dependence on the current reference); a hop-stream hit then takes its row's dataset_index (the basin its start was kicked
    from) as its parent. A basin's parent is that of its first hit, the one that opened it. A group that does not match
    as a whole (a missing, rewritten or ambiguous shard) keeps -1 and is counted: MACE energies are quantised, so
    energy ties make a partial match unsafe. Appends a record to
    reg['migrations']; returns (hop hits, unmatched hits)."""
    h = reg['hits']
    n = len(h['basin'])
    parent = [-1] * n
    hop_stream = (cfg.hops or {}).get('stream', 'hops') if cfg.hops else None
    n_hop = n_miss = 0
    i = 0
    while i < n:
        st, run, cur = h['stream'][i], h['run'][i], int(h['cursor'][i])
        j = i
        while j < n and h['stream'][j] == st and h['run'][j] == run and int(h['cursor'][j]) == cur:
            j += 1
        if st == hop_stream:
            n_hop += j - i
            parents = None
            files = glob.glob(os.path.join(coord_dir, 'shards', str(run), f'{cur:08d}_*.pt'))
            if len(files) == 1:
                try:
                    s = torch.load(files[0], weights_only=False)
                    phys = torch.nonzero(physical(s['params'], s['lj'])).flatten()
                    if j - i <= len(phys):  # admitted = physical rows at or below the window's edge at ingestion
                        e = s['energy'][phys]
                        edge = torch.sort(e).values[j - i - 1]
                        idx = phys[e <= edge]
                        if [float(x) for x in s['energy'][idx].tolist()] == [float(x) for x in h['energy'][i:j]]:
                            parents = [int(d) for d in s['dataset_index'][idx].tolist()]
                except Exception:  # noqa: BLE001 -- an unreadable shard leaves its hits unmatched, counted below
                    parents = None
            if parents is None:  # missing, rewritten or ambiguous shard: energy ties make a partial match unsafe
                n_miss += j - i
            else:
                parent[i:j] = parents
        i = j
    h['parent'] = parent
    first = {}
    for t, b in enumerate(h['basin']):
        first.setdefault(int(b), t)
    reg['basin_parent'] = [parent[first[b]] if b in first else -1 for b in range(len(reg['basin_E']))]
    reg.setdefault('migrations', []).append(dict(what='lineage backfilled from the shards', hop_hits=n_hop,
                                                 unmatched=n_miss, at_pass=reg.get('passes', 0)))
    print(f'coordinator: lineage backfilled from the shards for {n_hop} hop hits ({n_miss} unmatched)')
    return n_hop, n_miss


def poisson_upper(k, conf=0.9):
    """Exact one-sided upper confidence bound on a Poisson mean after observing k (Garwood)."""
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (k + 1)) / 2)


def compute_stats(reg, cfg):
    ref = _energy_ref(cfg, reg)
    h = reg['hits']
    E = np.asarray(h['energy'], dtype=float)
    basin = np.asarray(h['basin'], dtype=np.int64)
    stream = np.asarray(h['stream'], dtype=object)
    streams = sorted(set(reg['effort']) | set(cfg.streams))
    lineage = lineage_returns(reg)  # hop returns into their own lineage: not independent draws, never counted
    out = dict(energy_ref=ref, n_basins=len(reg['basin_E']), bands={}, streams={}, lineage_returns=int(lineage.sum()))
    for band, width in cfg.bands_kT.items():
        m = (E <= ref + width * cfg.kT) & ~lineage
        counts = np.bincount(basin[m], minlength=len(reg['basin_E'])) if m.any() else np.zeros(len(reg['basin_E']), int)
        single = np.nonzero(counts == 1)[0]
        single_owner = {}
        for i in np.nonzero(m)[0]:
            if counts[basin[i]] == 1:
                single_owner[int(basin[i])] = stream[i]
        out['bands'][band] = dict(width_kT=width, n_states=int(m.sum()), n_basins=int((counts > 0).sum()),
                                  f1=int(len(single)))
        for s in streams:
            eff = reg['effort'].get(s, {})
            W = float(eff.get('row_evals', 0.0))
            f1 = sum(1 for o in single_owner.values() if o == s)
            n_hits = int((m & (stream == s)).sum())
            first = sum(1 for j, f in enumerate(reg['basin_first']) if f['stream'] == s and counts[j] > 0)
            f1_hi = poisson_upper(f1)
            st = out['streams'].setdefault(s, dict(relaxations=int(eff.get('relaxations', 0)), row_evals=W,
                                                   attempts=int(eff.get('attempts', 0)), bands={}))
            st['bands'][band] = dict(hits=n_hits, basins_first=first, f1=f1,
                                     effort_per_new=(W / f1) if f1 else float('inf'),
                                     effort_per_new_lo=W / f1_hi)
    return out


def verdicts(stats, reg, cfg, now=None):
    """Per-stream stop decisions (confirmed over passes_to_confirm passes after new effort, or at once when the test
    holds and the stream has brought no new work for max(confirm_settle_s, 3 x its longest shard's wall_s)) and the
    campaign STOP."""
    now = time.time() if now is None else float(now)
    decisions = {}
    for s, rule in cfg.streams.items():
        st = stats['streams'].get(s)
        if st is None:
            decisions[s] = dict(stop=False, why='no shards yet')
            continue
        why = []
        ok = st['relaxations'] >= rule.min_relaxations
        if not ok:
            why.append(f"relaxations {st['relaxations']} < {rule.min_relaxations}")
        for band, Z in rule.Z.items():
            b = st['bands'].get(band)
            if b is None:
                ok = False
                why.append(f'no band {band}')
                continue
            if b['hits'] < rule.min_hits:
                ok = False
                why.append(f"{band}: hits {b['hits']} < {rule.min_hits}")
            elif b['effort_per_new_lo'] <= Z:
                ok = False
                why.append(f"{band}: effort per new basin >= {b['effort_per_new_lo']:.3g} (<= Z {Z:.3g})")
        seen = reg.setdefault('confirm_effort', {})
        since = reg.setdefault('confirm_since', {})
        W = float(st.get('row_evals', 0.0))
        new_work = W > seen.get(s, -1.0)
        if new_work or s not in since:  # the settle clock starts on the stream's first verdict, restarts on new work
            since[s] = now
        eff = reg['effort'].get(s, {})
        longest = float(eff.get('max_shard_wall_s', float(eff.get('wall_s', 0.0)) / max(int(eff.get('shards', 0)), 1)))
        settle = max(float(cfg.confirm_settle_s), 3.0 * longest)
        idle = now - since[s]
        if not ok:
            streak = 0
        elif new_work:  # a pass counts toward confirmation only if the stream did new work since
            streak = reg['confirm'].get(s, 0) + 1
        else:
            streak = reg['confirm'].get(s, 0)
        reg['confirm'][s] = streak
        seen[s] = W
        # final: the test holds and no new work has come for settle seconds, longer than a batch takes, so none is in
        # flight and another pass would see the same statistics; bounded, so a stream whose jobs have all ended stops
        final = ok and idle >= settle
        if why:
            reason = '; '.join(why)
        elif streak < cfg.passes_to_confirm and final:
            reason = f'all bands exhausted (final: no new work for {idle:.0f} s >= {settle:.0f} s)'
        else:
            reason = f'all bands exhausted ({streak} pass(es))'
        decisions[s] = dict(stop=streak >= cfg.passes_to_confirm or final, streak=streak, why=reason)
    total = sum(float(e.get('row_evals', 0.0)) for e in reg['effort'].values())
    campaign_stop = (bool(decisions) and all(d['stop'] for d in decisions.values())) or total >= cfg.hard_cap
    return decisions, campaign_stop, total


def write_outputs(coord_dir, cfg, reg, stats, decisions, campaign_stop, total):
    atomic_torch_save(reg, os.path.join(coord_dir, 'registry.pt'))
    blob = dict(stats=stats, decisions=decisions, campaign_stop=campaign_stop, total_row_evals=total,
                identity_cut=cfg.identity_cut, identity_calibrated=cfg.identity_calibrated,
                energy_model_id=cfg.energy_model_id, refused=reg['refused'][-50:], passes=reg['passes'])
    atomic_json_dump(json.loads(json.dumps(blob, default=float)), os.path.join(coord_dir, 'stats.json'))
    lines = [f"# campaign {os.path.basename(os.path.abspath(coord_dir))}: pass {reg['passes']}", '',
             f"energy model `{cfg.energy_model_id}`; reference {stats['energy_ref']:.4f}; kT {cfg.kT}; identity cut "
             f"{cfg.identity_cut} ({cfg.rdf_mode}; {'calibrated' if cfg.identity_calibrated else 'NOT CALIBRATED'}"
             f"{'; ' + cfg.identity_note if cfg.identity_note else ''}); {stats['n_basins']} basins in the registry; "
             f"total effort {total:.4g} row-evaluations" + (f" (hard cap {cfg.hard_cap:.3g})" if np.isfinite(cfg.hard_cap) else ''),
             '']
    for band, bs in stats['bands'].items():
        lines += [f"*Band {band} (within {bs['width_kT']} kT): {bs['n_states']} states in {bs['n_basins']} basins, "
                  f"{bs['f1']} seen once. Effort per new basin = stream effort / basins seen once from that stream "
                  f"(Good-Turing); 'lower' uses the 90% upper bound on the singleton count.*", '',
                  '| stream | relaxations | row-evaluations | states in band | basins first found | seen once | '
                  'effort per new basin | lower bound | Z |', '|---|---|---|---|---|---|---|---|---|']
        for s, st in stats['streams'].items():
            b = st['bands'][band]
            Z = cfg.streams[s].Z.get(band) if s in cfg.streams else None
            lines.append(f"| {s} | {st['relaxations']} | {st['row_evals']:.3g} | {b['hits']} | {b['basins_first']} | "
                         f"{b['f1']} | {b['effort_per_new']:.3g} | {b['effort_per_new_lo']:.3g} | "
                         f"{'' if Z is None else f'{Z:.3g}'} |")
        lines.append('')
    lines += ['| stream | verdict | reason |', '|---|---|---|']
    for s, d in decisions.items():
        lines.append(f"| {s} | {'STOP' if d['stop'] else 'run'} | {d['why']} |")
    lines += ['', f"Campaign: {'STOP' if campaign_stop else 'running'}."]
    if reg['refused']:
        lines += ['', f"Refused inputs ({len(reg['refused'])}; latest shown): " +
                  '; '.join(f"{r['run']}@{r['cursor']}: {r['reason']}" for r in reg['refused'][-5:])]
    tmp = os.path.join(coord_dir, f'stats.md.{os.getpid()}.tmp')
    with open(tmp, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    os.replace(tmp, os.path.join(coord_dir, 'stats.md'))
    for s, d in decisions.items():
        path = os.path.join(coord_dir, f'STOP.{s}')
        if d['stop'] and not os.path.exists(path):
            open(path, 'w').write(d['why'] + '\n')
    if campaign_stop and not os.path.exists(os.path.join(coord_dir, 'STOP')):
        open(os.path.join(coord_dir, 'STOP'), 'w').write(f'campaign stop at pass {reg["passes"]}\n')


def latent_representable(cfg, params, hand, rtol=1e-3):
    """True where the latent round trip returns the same cell: kicking a cell the latent cannot represent (e.g. an axis
    beyond the latent's range, which it clips) would compress it instead of perturbing it."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    if len(params) == 0:
        return torch.zeros(0, dtype=torch.bool)
    b = collate_data_list(rebuild_crystals(cfg, params, hand))
    b.canonicalize_orientation()
    before = b.full_cell_parameters().detach().clone()
    b.latent_to_cell_params(b.latent_params(gauge_fix_free_axes=False))
    after = b.full_cell_parameters().detach()
    return ((after[:, :3] - before[:, :3]).abs() / before[:, :3]).amax(1) < rtol


def write_hop_parents(coord_dir, cfg, reg):
    """hop_parents.pt: the basins a hop job may kick from, least-hopped first. Empty when none is eligible."""
    h = cfg.hops or {}
    ref = _energy_ref(cfg, reg)
    E = np.asarray(reg['basin_E'], dtype=float)
    gen = np.asarray(reg['basin_gen'], dtype=int)
    starts = np.array([reg['hop_starts'].get(j, 0) for j in range(len(E))], dtype=int)
    blocked = np.zeros(len(E), dtype=bool)
    blocked[[j for j in reg['hop_blocked'] if j < len(E)]] = True
    ok = (E <= ref + float(h.get('window_kT', 2.0)) * cfg.kT) & (gen < int(h.get('max_generation', 3))) &          (starts < int(h.get('max_per_basin', 32))) & ~blocked
    idx = np.nonzero(ok)[0]
    if len(idx):
        rep = latent_representable(cfg, torch.stack([reg['basin_params'][j] for j in idx]),
                                   torch.stack([reg['basin_hand'][j] for j in idx]))
        for j in idx[~rep.numpy()]:
            reg['hop_blocked'].append(int(j))
        idx = idx[rep.numpy()]
    order = idx[np.lexsort((E[idx], starts[idx]))] if len(idx) else idx
    blob = dict(basin=torch.as_tensor(order, dtype=torch.long),
                params=torch.stack([reg['basin_params'][j] for j in order]) if len(order) else torch.zeros(0, 18),
                hand=torch.stack([reg['basin_hand'][j] for j in order]) if len(order) else torch.zeros(0, cfg.z_prime),
                E=torch.as_tensor(E[order]), hop_starts=torch.as_tensor(starts[order]),
                generation=torch.as_tensor(gen[order]), log_noise=tuple(h.get('log_noise', (-0.5, -0.5))),
                sg=cfg.sg, z_prime=cfg.z_prime, energy_model_id=cfg.energy_model_id, passes=reg['passes'],
                n_basins=len(E))  # 0: nothing found yet, so an empty list is not exhaustion (run_search._hop_batch)
    atomic_torch_save(blob, os.path.join(coord_dir, 'hop_parents.pt'))
    return len(order)


def curate(coord_dir):
    """One idempotent pass: ingest priors and new shards, update the registry, statistics, verdicts and stop files."""
    cfg = CampaignConfig.load(os.path.join(coord_dir, 'coord.yaml'))
    reg_path = os.path.join(coord_dir, 'registry.pt')
    reg = torch.load(reg_path, weights_only=False) if os.path.exists(reg_path) else new_registry(cfg)
    for k, v in dict(basin_gen=[0] * len(reg['basin_E']), hop_starts={}, hop_blocked=[],
                     basin_parent=[-1] * len(reg['basin_E'])).items():
        reg.setdefault(k, v)  # registries written before the hop stream (or the lineage record) existed
    if 'parent' not in reg['hits'] or len(reg['hits']['parent']) != len(reg['hits']['basin']) or \
            len(reg['basin_parent']) != len(reg['basin_E']):  # no lineage record, or hits added by older code
        backfill_lineage(reg, cfg, coord_dir)
    if 'confirm_effort' not in reg:  # streaks counted under the older rule (a pass needed no new work) do not carry over
        reg['confirm'] = {}
        reg['confirm_effort'] = {}
    if reg['energy_model_id'] != cfg.energy_model_id or reg['identity_cut'] != cfg.identity_cut or \
            reg.get('rdf_mode', 'envwise') != cfg.rdf_mode:  # a registry without the key was built envwise
        raise ValueError('coord.yaml energy_model_id, identity_cut or rdf_mode differs from the existing registry; start a new '
                         'campaign directory rather than mixing definitions')
    ingest_priors(reg, cfg)
    n_new = ingest_shards(reg, cfg, coord_dir)
    reg['passes'] += 1
    stats = compute_stats(reg, cfg)
    decisions, campaign_stop, total = verdicts(stats, reg, cfg)
    for s, d in decisions.items():  # sticky: a stream whose STOP.<s> exists has had its jobs stop, whatever the verdict
        if not d['stop'] and os.path.exists(os.path.join(coord_dir, f'STOP.{s}')):
            d.update(stop=True, why=f"STOP.{s} present (verdict now: {d['why']})")
    done = {s: d['stop'] for s, d in decisions.items()}
    if cfg.hops:
        stats['hop_parents'] = write_hop_parents(coord_dir, cfg, reg)
        hs = cfg.hops.get('stream', 'hops')
        w_hop = float(reg['effort'].get(hs, {}).get('row_evals', 0.0))
        now = time.time()
        # a hop batch finished since the previous pass, or the settle clock starts (a registry written before
        # hop_effort_since existed carries hop_effort_seen alone)
        if w_hop != reg.get('hop_effort_seen', -1.0) or 'hop_effort_since' not in reg:
            reg['hop_effort_seen'], reg['hop_effort_since'] = w_hop, now
        # settled: no new hop work for exhaust_settle_s, longer than a hop batch takes, so none can still be in flight;
        # if every hop job has ended, the effort stays put and this holds after that span (it cannot wait forever)
        idle = now - reg.get('hop_effort_since', now) >= float(cfg.hops.get('exhaust_settle_s', 3600.0))
        if hs in done and stats['hop_parents'] == 0 and idle:  # nothing to kick and nothing in flight: only another
            done[hs] = True                                     # stream's new basin could revive it
    campaign_stop = campaign_stop or (bool(done) and all(done.values()))
    write_outputs(coord_dir, cfg, reg, stats, decisions, campaign_stop, total)
    atomic_json_dump(dict(time=time.time(), host=socket.gethostname(), passes=reg['passes'], new_shards=n_new),
                     os.path.join(coord_dir, 'heartbeat.json'))
    return stats, decisions, campaign_stop


def _heartbeat_age(coord_dir):
    """Seconds since the last curate pass (heartbeat.json), on the filesystem's clock; inf when there is none."""
    hb = os.path.join(coord_dir, 'heartbeat.json')
    if not os.path.exists(hb):
        return float('inf')
    probe = os.path.join(coord_dir, f'.probe.{os.getpid()}')
    with open(probe, 'w'):
        pass
    try:
        return os.path.getmtime(probe) - os.path.getmtime(hb)
    finally:
        os.remove(probe)


def curate_locked(coord_dir):
    """One curate pass under the curator lease, as the jobs take it: for passes run by hand, by run_loop and by export.
    Raises RunLeaseLost while another process is curating (it is never waited on)."""
    from mxtaltools.crystal_search.run_state import RunLease
    lease = RunLease(os.path.join(coord_dir, 'curator'), stale_after_s=3600, settle_s=0.5)
    lease.acquire()
    try:
        return curate(coord_dir)
    finally:
        lease.release()


def pending_shards(coord_dir, reg):
    """Shard files on disk that the registry has not ingested (keys as ingest_shards writes them)."""
    done = set(reg['ingested'])
    keys = (os.path.relpath(p, coord_dir).replace('\\', '/')
            for p in sorted(glob.glob(os.path.join(coord_dir, 'shards', '*', '*.pt'))))
    return [k for k in keys if k not in done]


def maybe_curate(coord_dir, every_s):
    """Run a curate pass from inside a search job, between batches, when the last pass (heartbeat.json, judged on the
    filesystem's clock) is older than every_s and no other job is curating (a curator lease, never waited on).
    Fail-open: any error is printed and the search continues. Returns True if this call ran a pass."""
    from mxtaltools.crystal_search.run_state import RunLease, RunLeaseLost
    try:
        if _heartbeat_age(coord_dir) < every_s:
            return False
        lease = RunLease(os.path.join(coord_dir, 'curator'), stale_after_s=max(3 * every_s, 3600), settle_s=0.5)
        try:
            lease.acquire()
        except RunLeaseLost:
            return False  # another job is curating
        try:
            if _heartbeat_age(coord_dir) < every_s:  # a pass finished while this job was acquiring the lease
                return False
            curate(coord_dir)
            return True
        finally:
            lease.release()
    except Exception as e:  # noqa: BLE001 -- fail-open by design
        print(f'coordinator: curate pass from a search job failed ({type(e).__name__}: {e}); continuing')
        return False


def run_loop(coord_dir, interval_s=900.0, max_hours=None):
    """Curate every interval_s (each pass under the curator lease; a pass while a search job holds it is skipped)
    until two consecutive passes report the campaign stop, or max_hours."""
    t0 = time.time()
    stopped_passes = 0
    while True:
        try:
            _, _, campaign_stop = curate_locked(coord_dir)
            stopped_passes = stopped_passes + 1 if campaign_stop else 0
        except RunLeaseLost as e:  # a search job is curating: this interval's pass is its own
            print(f'coordinator: pass skipped, curator lease held ({e})')
        except Exception as e:  # noqa: BLE001 -- one failed pass must not end the curator (the campaign stop or,
            # when given, max_hours ends the loop; without max_hours, the job's walltime does)
            print(f'coordinator: pass failed ({type(e).__name__}: {e}); retrying next interval')
        if stopped_passes >= 2:
            print('coordinator: campaign stopped; exiting')
            return
        if max_hours is not None and time.time() - t0 > 3600 * max_hours:
            print('coordinator: max_hours reached; exiting (jobs continue to their own budgets)')
            return
        time.sleep(interval_s)


def export(coord_dir, band_kT, out_path, catch_up=True):
    """The campaign's product: one crystal per basin within band_kT of the reference, in its reduced cell (the lowest-energy
    state seen), with energy, basin id, number of states found in it, and the stream that found it first; plus
    <out>.csv. With catch_up (default), a curate pass under the curator lease first ingests the shards the jobs wrote
    after their last pass; raises if any shard on disk is still not in the registry. Returns the number exported."""
    if catch_up:  # the jobs' last shards arrive after the last pass they ran: ingest everything on disk first
        curate_locked(coord_dir)
    cfg = CampaignConfig.load(os.path.join(coord_dir, 'coord.yaml'))
    reg = torch.load(os.path.join(coord_dir, 'registry.pt'), weights_only=False)
    left = pending_shards(coord_dir, reg)
    if left:
        raise RuntimeError(f'{len(left)} shard(s) on disk are not in the registry (unreadable, or written since the '
                           f'pass): {left[:5]}. Rerun the export once no job is writing.')
    ref = _energy_ref(cfg, reg)
    E = np.asarray(reg['basin_E'], dtype=float)
    idx = np.nonzero(E <= ref + band_kT * cfg.kT)[0]
    idx = idx[np.argsort(E[idx])]
    counts = np.bincount(np.asarray(reg['hits']['basin'], dtype=np.int64), minlength=len(E))
    crystals = rebuild_crystals(cfg, torch.stack([reg['basin_params'][j] for j in idx]),
                                torch.stack([reg['basin_hand'][j] for j in idx])) if len(idx) else []
    rows = ['basin,energy,above_ref_kT,states,first_stream,first_run']
    for c, j in zip(crystals, idx):
        setattr(c, cfg.energy_key, torch.tensor([E[j]]))
        c.basin_id = torch.tensor([int(j)])
        c.basin_states = torch.tensor([int(counts[j])])
        c.energy_model_id = cfg.energy_model_id
        f = reg['basin_first'][j]
        rows.append(f"{j},{E[j]:.5f},{(E[j] - ref) / cfg.kT:.4f},{counts[j]},{f['stream']},{f['run']}")
    atomic_torch_save(crystals, out_path)
    with open(os.path.splitext(out_path)[0] + '.csv', 'w') as fh:
        fh.write('\n'.join(rows) + '\n')
    return len(idx)


def main():
    ap = argparse.ArgumentParser(description='crystal_search campaign coordinator')
    ap.add_argument('coord_dir')
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--interval', type=float, default=900.0)
    ap.add_argument('--max_hours', type=float, default=None)
    ap.add_argument('--export', nargs=2, metavar=('BAND_KT', 'OUT'), default=None,
                    help='write the basins within BAND_KT of the reference to OUT (.pt) and OUT.csv')
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))  # unwind: release the curator lease
    if a.export is not None:
        print(f'exported {export(a.coord_dir, float(a.export[0]), a.export[1])} basins to {a.export[1]}')
    elif a.loop:
        run_loop(a.coord_dir, a.interval, a.max_hours)
    else:
        curate_locked(a.coord_dir)
        print(open(os.path.join(a.coord_dir, 'stats.md')).read())


if __name__ == '__main__':
    main()
