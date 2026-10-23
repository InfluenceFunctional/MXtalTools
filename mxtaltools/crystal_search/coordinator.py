"""
Coordination of many parallel crystal_search jobs through a shared directory (a "campaign"), with a per-stream stopping
rule: stop a proposal stream once the expected effort to find the next new low-energy basin exceeds Z.

Layout of a campaign directory::

    coord.yaml                 campaign config (see CampaignConfig)
    shards/<run>/<cursor>_<batch>.pt   one per completed search batch, written by the GPU jobs (write_shard)
    registry.pt                the basin registry (single writer: the curate pass)
    stats.json / stats.md      per-stream statistics and verdicts, rewritten each pass
    STOP.<stream> / STOP       stop files the GPU jobs poll between batches
    heartbeat.json             time, host, pass count and new-shard count of the last curate pass
    hop_parents.pt             basins a hop job may kick from (write_hop_parents), only when hops is configured
    curator_owner.json         the curator lease, held during a pass run through maybe_curate or curate_locked

GPU jobs never wait on the coordinator: they write shards (fail-open: a coordination I/O error never stops a search)
and check the STOP files between batches; the one exception is a hop job, which waits a bounded time (then raises),
still checking the STOP files, while hop_parents.pt is missing or unreadable, is empty but was written on a registry
with no basin or by a pass that may predate the job's last shard, or lists no parent with a kick left for the job; an
empty list from a later pass on a registry with basins ends the job (run_search._hop_batch). A curate pass (curate) is
safe to repeat: each shard and prior is ingested once, and a stream's stop-confirmation streak advances only on a pass
after new effort from that stream; every pass advances the pass count, and a pass without new effort can still stop a
stream whose test holds once its settle time has passed (stopping rule below). curate itself takes no lease; a pass run
inside a search job (maybe_curate), by hand (main), in a loop by a cheap CPU job (run_loop) or before an export (export)
holds the curator lease, and a pass attempted while another process holds it does not run (the lease is never waited
on). If the coordinator stops, the jobs simply run to their own budgets.

Basins. A curate pass takes each physical state within window_kT of the energy reference (every physical state while
there is no reference), re-expresses it in its reduced cell (standardize.standardize_cells; RDFs do not depend on the
cell choice, but the registry keeps reduced descriptions; a state with no reduced cell is kept as stored and noted in
the refused list), computes its RDF (rdf_mode: one channel per pair of bonded-environment labels, 'envwise', the
default, or per pair of atom indices, 'atomwise'), and assigns it to the basin of the nearest leader within
identity_cut -- leaders registered before its shard (or prior file) first, then those opened by earlier states of the
same shard -- or else opens a new basin (leader clustering: the first state of a basin is its leader; later states only
join; the basin's stored energy and parameters are those of its lowest-energy state). A hop state records the basin its
start was kicked from as its parent (hits['parent']) and, where its shard gives one, its latent kick length (hit_kick);
a basin it opens records the same parent (basin_parent) and is one generation below it (basin_gen).
identity_cut must be calibrated per system against a structure-identity test (COMPACK): the RDF distance
at which two structures are the same packing differs between molecules and between kinds of pair (acridine end states:
0.050 by the rule of the GFN repository's campaign_compack.py, where 0.085 had held against the experimental forms;
MIPCAS and NEHZOR landscapes: 0.12-0.13 at a fitted P(match) = 0.95). When unsure, a smaller cut splits one structure
into several basins, which only delays the stop; a larger one merges different structures and stops early. The config
records whether the cut was calibrated. Calibrate it on a short pilot campaign before the real one (the GFN repository's
energy_sampling/eval/campaign_compack.py calibrate does this within a fixed COMPACK budget): changing the cut under a
running campaign would reassign the basins it has built.
A pass raises ValueError when identity_cut, rdf_mode or energy_model_id in coord.yaml differs from the registry's
(curate).

Stopping rule, per stream s (random starts, eLJ-prescreened starts, hops, ...), for each band b (e.g. 2 kT and 1 kT)::

    f1_s  = basins with exactly one state within the band, over all streams and priors, that state from s; a hop state
            that relaxed back into the basin it was kicked from, or into one of that basin's ancestors, is left out of
            every count (lineage_returns)
    W_s   = the stream's effort: row-evaluations of the campaign's energy model (the final stage's target), from the
            shards; cheaper stages (an eLJ pre-screen) are recorded apart (row_evals_other) and not counted; failed
            attempts count through wall time only
    stop s when W_s / f1_hi(f1_s) > Z_b, where f1_hi is the exact one-sided 90% Poisson upper bound on f1_s,
    and N_s >= min_relaxations and the band holds at least min_hits of the stream's states, on passes_to_confirm
    (default 2) passes in a row, each after new effort from s (a pass with no new effort from s leaves the streak
    as it was if the test holds, and resets it if the test fails), or on one pass at least
    max(confirm_settle_s, 3 x the wall time of the stream's longest shard) seconds after the pass that last saw
    new effort from s, or, on a registry with no confirm_since entry for s, after the first pass to judge it
    (confirm_settle_s: default 3600; verdicts).

The Good-Turing ratio W/f1 estimates the effort per new basin without bias on the sep27 acridine data; the upper bound on
f1 keeps it from stopping early by chance. A stream is exhausted when every band says stop; once STOP.<s> exists the
stream stays stopped (a pass never removes a stop file and treats a present STOP.<s> as a stop). The campaign stops
(STOP) when there is at least one stream with a rule and every such stream is stopped, or when total effort reaches
hard_cap; for the campaign stop only, a hop stream with a rule also counts as stopped on a pass that finds no eligible
hop parent once hops exhaust_settle_s (default 3600 s) has passed since a pass last saw its effort (row_evals) change
(curate). Termination: a stream with no new basins that meets min_relaxations and min_hits stops at W_s > 2.3 Z
(f1_hi(0) = 2.30): after passes_to_confirm passes with new effort while its jobs keep delivering shards, or, once they
deliver none, on the first pass after the settle time of the stopping rule. A stream with a rule that never delivers a
shard, never reaches min_relaxations or min_hits in a band, or whose jobs all end while its test fails does not stop by
the rule unless states of other streams later make its test hold (they can lower f1_s); the campaign stop then rests
on hard_cap (default inf, i.e. no cap), a STOP file written by hand, or, for a hop stream only, the no-parent condition
above (with no eligible parent left and the hop jobs ended, it holds on the first pass at least exhaust_settle_s after
the one that saw their last effort, or after the first pass on a registry that had no hop_effort_since); nothing waits
on a particular job.

Model-agnostic: energies are whatever the search objective reports (energy_key), kT and the energy reference come from
the config, and every shard carries the id of the energy model that scored it; the curate pass refuses shards (and priors)
whose model id differs from the campaign's, and refuses a prior file that fails to load or ingest (ingest_priors).
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
    """Stopping rule for one stream: Z maps a band name (a key of CampaignConfig.bands_kT) to the effort threshold per
    new basin, in row-evaluations; the stream can stop only once it has min_relaxations relaxations and at least
    min_hits of its states in each band of Z (lineage returns, see lineage_returns, not counted)."""
    Z: Dict[str, float]  # band name -> effort threshold per new basin (row-evaluations)
    min_relaxations: int = 2000
    min_hits: int = 20


@dataclass
class CampaignConfig:
    """Campaign configuration, read from <coord_dir>/coord.yaml by load. mol_path is a torch file holding the molecule
    (or a list whose first element is used) from which rebuild_crystals builds every crystal, in space group sg with
    Z' = z_prime. identity_calibrated and identity_note are only reported, in the stats outputs. passes_to_confirm is
    the number of passes in a row, each after new effort from the stream, on which a stream's stop test must hold
    (see verdicts); confirm_settle_s (default 3600) is the time in seconds without new effort from a stream after which
    a pass on which its test holds confirms the stop without that streak, the time applied being the longer of it and
    three times the wall time of the stream's longest shard (see verdicts); rdf_batch is compute_rdfs' batch size in a
    curate pass. rdf_mode sets the RDF channels of the identity metric, 'envwise' (default; one channel per pair of
    bonded-environment labels) or 'atomwise' (one channel per pair of atom indices); it is recorded in the registry,
    and curate raises when it differs from an existing registry's.
    hops, when set, enables the hop stream and hop_parents.pt; keys read: stream (default 'hops'), window_kT (default
    2.0), max_generation (default 3), max_per_basin (default 32), log_noise (default (-0.5, -0.5)), exhaust_settle_s
    (default 3600: the seconds that must pass after a pass last saw the hop stream's effort change before, with no
    eligible hop parent, it counts as stopped for the campaign stop; see curate)."""
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
        """Read a CampaignConfig from a YAML file; each entry of streams becomes a StreamRule. Unknown keys raise
        TypeError; an rdf_mode other than 'envwise' or 'atomwise' raises ValueError."""
        raw = yaml.safe_load(open(path))
        raw['streams'] = {k: StreamRule(**v) for k, v in (raw.get('streams') or {}).items()}
        cfg = CampaignConfig(**raw)
        if cfg.rdf_mode not in ('envwise', 'atomwise'):
            raise ValueError(f"rdf_mode must be 'envwise' or 'atomwise', not {cfg.rdf_mode!r}")
        return cfg


def energy_model_id(opt_stage, config):
    """A string identifying the energy model that scores a search's final stage (opt_stage: a dict or object with
    optim_target). For target 'mace' or 'uma' (any case) with config.mace_predictor_path / uma_predictor_path set:
    '<target>:<model file name>:<size in bytes>:<16 hex digits of the SHA1 of the file's first MiB and, when the file
    exceeds 2 MiB, its last MiB>' (so a replaced checkpoint at the same path gets a new id); otherwise str(target)."""
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
    """The stop-file paths for a stream: [<coord_dir>/STOP, <coord_dir>/STOP.<stream>]."""
    return [os.path.join(coord_dir, 'STOP'), os.path.join(coord_dir, f'STOP.{stream}')]


def write_shard(coord_dir, run_name, stream, cursor, batch_idx, rows, energy_key, meta):
    """Write one shard for a completed batch, atomically, to
    <coord_dir>/shards/<run_name>/<cursor:08d>_<batch_idx:06d>.pt.
    rows: list of MolCrystalData outputs, each carrying energy_key. The shard is a dict of run, stream, cursor,
    batch_idx, params [n, ``6 + 6 * max_z_prime``] float32 (full_cell_parameters), handedness [n, max_z_prime],
    energy [n] (energy_key), lj [n] (NaN where a row has none), dataset_index [n] (-1 where absent), kick [n] (the
    row's hop_kick, a hop start's latent kick length; NaN where absent) -- zero-row tensors when rows is empty (params
    [0, 18], handedness [0, 2]) -- and every entry of meta (the curate pass reads energy_model_id, n_relaxations,
    row_evals, n_attempts, wall_s and gpu). Returns True on success. Fail-open: returns False (and prints) instead of
    raising, so a coordination problem never stops a search job."""
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
    """List of MolCrystalData crystals, one per row, from stored parameters and the campaign's conformer (as search
    seeds are built): params [n, ``>= 6 + 6 * z_prime``] in full_cell_parameters order for max_z_prime = z_prime (3
    lengths, 3 angles, ``3 * z_prime`` centroid columns, then ``3 * z_prime`` orientation columns; further columns are
    not read), hand [n, >= z_prime]; space group cfg.sg, z_prime = max_z_prime = cfg.z_prime, the molecule loaded from
    cfg.mol_path on every call (the first element if the file holds a list; its one-element num_atoms, mass,
    mol_volume and radius tensors reshaped to 0-d), box analysis done."""
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
    """RDFs of a list of MolCrystalData, [n, n_channels, bins] float32 on the CPU (a zero-length tensor for an empty
    list): clones of batch_size crystals at a time, collated without the RDF_DROP keys and analysed with
    ``analyze(['rdf'], assign_outputs=False, **RDF_KW)``, RDF_KW's rdf_mode replaced by the rdf_mode argument (a
    campaign passes 'envwise' or 'atomwise'), under no_grad. The input crystals are not modified."""
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
    """[na, nb] float32 distances between RDFs ra [na, n_channels, bins] and rb [nb, n_channels, bins], equal to
    analysis.crystal_rdf.compute_rdf_distance given a grid rr of spacing bin_width: bin width times the mean, over
    channels active (nonzero) in either RDF, of the L1 distance between the per-channel normalised CDFs. The default
    bin_width, ``10 / 99``, is the spacing of torch.linspace(0, 10, 100); the bin positions analyze returns with RDF_KW
    are 0.1 apart. ra is processed chunk rows at a time."""
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
    """[n] bool, rows that are crystals: lj below LJ_BLOWN_UP (NaN counts as 0, so passes) and angular factor
    V / (a b c) above MIN_ANGULAR_FACTOR. params [n, >= 6]: cell lengths, then cell angles in radians; lj [n]."""
    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    al, be, ga = params[:, 3], params[:, 4], params[:, 5]
    cos_a, cos_b, cos_g = torch.cos(al), torch.cos(be), torch.cos(ga)
    ang = torch.sqrt((1 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2 * cos_a * cos_b * cos_g).clamp(min=0))
    return (torch.nan_to_num(lj, nan=0.0) < LJ_BLOWN_UP) & (ang > MIN_ANGULAR_FACTOR)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def new_registry(cfg):
    """An empty basin registry (a dict, version 1) recording cfg's energy_model_id, identity_cut and rdf_mode: per-basin
    records (leaders, a tensor of the leaders' RDFs; the lists basin_E, basin_params, basin_hand, basin_first,
    basin_gen, basin_parent), per-hit lists in hits (basin, stream, energy, run, cursor, parent), per-stream effort,
    the ingested keys, refused inputs, the pass count, confirmation streaks (confirm) and the hop bookkeeping
    (hop_starts, hop_blocked). Keys added later: confirm_effort (curate, verdicts) and confirm_since (verdicts: each
    stream's row_evals at its last verdict, and the time of the verdict that last saw them grow or, failing that, of
    the first verdict that found no time recorded for the stream), hit_kick (_record_hit: hit index -> latent kick
    length, for hop starts whose shard gives one), hop_effort_seen and hop_effort_since (curate, with hops configured:
    the hop stream's row_evals when a pass last saw them change, and that pass's time) and migrations
    (backfill_lineage)."""
    return dict(version=1, energy_model_id=cfg.energy_model_id, identity_cut=cfg.identity_cut, rdf_mode=cfg.rdf_mode,
                ingested=[],
                leaders=torch.zeros(0), basin_E=[], basin_params=[], basin_hand=[], basin_first=[], basin_gen=[],
                basin_parent=[], hits=dict(basin=[], stream=[], energy=[], run=[], cursor=[], parent=[]),
                effort={}, refused=[], passes=0, confirm={}, hop_starts={}, hop_blocked=[])


def _stream_effort(reg, stream):
    """reg['effort'][stream], created on first use with relaxations, row_evals, attempts, wall_s and shards at zero and
    an empty gpus count (ingest_shards adds max_shard_wall_s)."""
    return reg['effort'].setdefault(stream, dict(relaxations=0, row_evals=0.0, attempts=0, wall_s=0.0, shards=0,
                                                 gpus={}))


def _assign(reg, cfg, rdfs, energies, params, hand, stream, run, cursor, parents=None, kicks=None):
    """Leader clustering of new states against the registry (and each other, in arrival order): each state joins the
    basin of the nearest leader registered before this call if within identity_cut, else the nearest leader opened
    earlier in this call if within identity_cut, else opens a basin with itself as leader. One hit is recorded per
    state (_record_hit). parents: for hop states, the basin each was kicked from; recorded as the hit's parent and, for
    a basin the state opens, as its basin_parent, with basin_gen one more than the parent's (0 when the parent is -1 or
    not a registered basin). kicks: per state, a hop start's latent kick length (NaN where the state has none), passed
    to _record_hit; with None no kick is recorded."""
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
    """Append one hit to reg['hits']: basin j, stream, energy E, run, cursor and parent (the basin a hop start was
    kicked from; -1 otherwise). A kick that is neither None nor NaN (a hop start's latent kick length) is stored in
    reg['hit_kick'], a dict created on first use, under the hit's index. When E is below the basin's stored energy,
    the basin's energy, params and hand become this state's."""
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
    """cfg.energy_ref if set, else the lowest basin energy in the registry (inf while it has no basin)."""
    if cfg.energy_ref is not None:
        return float(cfg.energy_ref)
    return min(reg['basin_E']) if reg['basin_E'] else float('inf')


def _ingest_rows(reg, cfg, params, hand, E, lj, stream, run, cursor, parents=None, kicks=None):
    """Physical states within window of the reference: standardise, RDF, assign. Returns the number admitted. Admitted
    rows are the physical ones (physical) at most cfg.window_kT kT above the energy reference as it stands when the
    call starts (every physical row while there is none); they are rebuilt (rebuild_crystals), put in their reduced
    cells (rows without one are kept as stored and counted in one reg['refused'] entry), given RDFs (cfg.rdf_batch,
    cfg.rdf_mode) and assigned by _assign with the standardised params and handedness. parents and kicks, when given,
    are per-row tensors, cut to the admitted rows and passed to _assign."""
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
    """Hex SHA1 of a file's whole content, read in 1 MiB chunks; raises OSError if the file cannot be read."""
    h = hashlib.sha1()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ingest_priors(reg, cfg):
    """Known structures (earlier searches, experimental forms) as stream 'prior': ingests each file of cfg.priors whose
    key is not yet in reg['ingested'] (mutates reg; no effort is counted). A file's key is its absolute path and the
    first 16 hex digits of the SHA1 of its content (``prior:<path>@<sha1>``), so a file whose content has changed is
    ingested again (its states are added; earlier hits stay); a file that cannot be read for hashing (OSError) is
    skipped this pass, with a message, and tried again next pass. A prior file is either a list of MolCrystalData
    carrying energy_key, whose model id is read from <file>.model_id if that exists, else from the first crystal's
    'energy_model_id' attribute (lj 0 where absent); or a dict with params [N, ``6 + 6 * Z'``], handedness, energy,
    energy_model_id and optional lj (default 0). Rows go through the same physical/window filter, standardisation and
    basin assignment as shard rows (_ingest_prior_file). A file whose model id differs from cfg.energy_model_id is
    recorded in reg['refused'], and so is a file that raises any Exception while it is loaded or ingested (reason
    'prior not ingested (<type>: <message>)'); either way its key is marked ingested, so that content is not tried
    again, while changed content (a new key) is ingested on a later pass."""
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
    """One prior file of ingest_priors (formats there), under key (mutates reg): its rows ingested, or the file recorded
    in reg['refused'] for another energy model; either way key is appended to reg['ingested']. Errors propagate to
    ingest_priors."""
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
    """Ingest every ``<coord_dir>/shards/*/*.pt`` not yet in reg['ingested'], in sorted path order (mutates reg). A file
    that cannot be loaded is skipped and retried next pass; a shard whose energy_model_id differs from cfg's is recorded
    in reg['refused']. Otherwise its effort (n_relaxations, default the row count; row_evals; n_attempts, default 1;
    wall_s; gpu) is added to reg['effort'][stream], whose max_shard_wall_s (0 where absent) is raised to the shard's
    wall_s when that is larger; for the hop stream (cfg.hops['stream'], default 'hops') its dataset_index is read as
    each row's parent basin, counted in reg['hop_starts'] and recorded as the parent of the row's hit; and its rows are
    filtered, standardised, RDF'd and assigned to basins (_ingest_rows), the shard's kick, when it has one, going with
    them (a hit whose kick is not NaN gets an entry in reg['hit_kick']). Returns the number of shards accepted (refused
    ones not counted)."""
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
    (the basins its lineage of hops came from, followed through basin_parent); compute_stats leaves these hits out of
    every count. Hits with parent -1 (every non-hop hit), and every hit of a registry without hits['parent'], are
    False; reg is not modified."""
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
    """Rebuild the lineage record (hits['parent'], basin_parent) from the shards, for a registry whose hits carry no
    parent or whose lineage lists differ in length from its hits or basins (curate calls it then). Hits are appended one
    per admitted row of each ingested shard, in row order, and the admitted rows are the shard's physical rows at or
    below the admission window's edge at ingestion; so each run of k consecutive hop-stream hits with one (stream, run,
    cursor) is matched to the rows of the single shard file ``<coord_dir>/shards/<run>/<cursor:08d>_*.pt`` that are
    physical with energy at or below the k-th lowest physical energy, in row order (the energy reference plays no part),
    which must carry exactly the hits' energies in order; each hit then takes its row's dataset_index (the basin its
    start was kicked from) as its parent. A group that does not match as a whole (no such file or more than one, an
    unreadable or rewritten shard, fewer than k physical rows, or a tie at the k-th energy that brings in more than k
    rows) keeps parent -1, so its hits stay in every count, and is reported as unmatched; non-hop hits get -1. A basin's
    parent is that of its first hit, the one that opened it. Mutates reg, appends a record to reg['migrations'] and
    prints a summary; returns (hop hits, unmatched hits)."""
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
    """Exact one-sided upper confidence bound, at level conf, on a Poisson mean after observing k (Garwood):
    chi2.ppf(conf, 2 (k + 1)) / 2."""
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * (k + 1)) / 2)


def compute_stats(reg, cfg):
    """Per-band and per-stream statistics from reg's hits, with lineage returns (lineage_returns) left out of every
    count (reg is not modified). Returns a dict: energy_ref, n_basins, lineage_returns (the number of hits left out);
    bands {band: width_kT, n_states, n_basins, f1}, over the counted hits of all streams and priors within the band;
    streams {stream: relaxations, row_evals, attempts, bands {band: hits, basins_first, f1, effort_per_new = row_evals /
    f1 (inf when f1 = 0), effort_per_new_lo = row_evals / poisson_upper(f1)}}, for every stream with recorded effort or
    a rule in cfg.streams (streams is empty when cfg.bands_kT is). basins_first counts the basins the stream found first
    that have a counted hit in the band."""
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
    """Per-stream stop decisions (confirmed over passes_to_confirm passes, each after new effort from the stream, or at
    once when the test holds and the stream has brought no new effort for max(confirm_settle_s, 3 x its longest shard's
    wall_s)) and the campaign STOP, from compute_stats' stats; now is the time of the verdict in seconds (default
    time.time(), the clock of the host running the pass). Returns (decisions, campaign_stop, total). decisions has an
    entry {stop, streak, why} for every stream in cfg.streams (one without statistics, which happens only when
    cfg.bands_kT is empty, gets stop False, why 'no shards yet' and no streak; a stream with a rule but no shards has
    statistics with zero relaxations and hits). A stream's test holds on a pass when relaxations >= min_relaxations
    and, in every band of its Z, hits >= min_hits and effort_per_new_lo > Z (a band of Z that cfg.bands_kT lacks fails
    it). Its streak, reg['confirm'][stream] (mutated), is reset to 0 when the test fails, advanced by one when the test
    holds and the stream's row_evals exceed those recorded at its previous verdict (reg['confirm_effort'][stream],
    mutated, recorded whenever the stream has statistics; with none recorded, as on its first verdict, the pass counts
    as new effort), and otherwise kept. reg['confirm_since'][stream] (mutated) is set to now on a verdict with new
    effort, and on the first verdict that finds no entry for the stream. stop is streak >= passes_to_confirm, or the
    test holds and now - confirm_since >= max(cfg.confirm_settle_s, 3 x the wall_s of the stream's longest shard), that
    wall_s read from the effort record's max_shard_wall_s or, without the key, taken as its mean wall_s per shard. why
    lists the failed conditions, or reads 'all bands exhausted (<streak> pass(es))', or, when the settle time alone
    gives the stop, 'all bands exhausted (final: no new work for <idle> s >= <settle> s)'. total is the row-evaluations
    summed over every stream's effort; campaign_stop is True when cfg.streams is nonempty and every stream stops, or
    total >= hard_cap. curate may add stops to both (see curate)."""
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
    """Write a pass's outputs to coord_dir: registry.pt (atomic), stats.json (atomic; the latest 50 refused inputs),
    stats.md (a temporary file moved into place), STOP.<stream> for each stream whose decision is stop and STOP when
    campaign_stop, each only if absent. Never removes a stop file."""
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
    """[n] bool, True where the latent round trip (canonicalize_orientation, latent_params(gauge_fix_free_axes=False),
    latent_to_cell_params) returns every cell length within relative tolerance rtol; angles and positions are not
    compared. Kicking a cell the latent cannot represent (e.g. an axis beyond the latent's range, which it clips) would
    compress it instead of perturbing it."""
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
    """Write hop_parents.pt: the basins a hop job may kick from, least-hopped first (ties: lowest energy first); returns
    their number. Eligible: basin energy within hops window_kT (default 2.0) kT of the reference, generation <
    max_generation (default 3), hop starts < max_per_basin (default 32), not in reg['hop_blocked'], and
    latent_representable; a basin failing the last test is appended to reg['hop_blocked'] (mutated). The file holds
    basin, params, hand, E, hop_starts, generation, log_noise (default (-0.5, -0.5)), sg, z_prime, energy_model_id,
    passes, n_basins (the number of basins in the registry, eligible or not) and max_per_basin (the limit above);
    run_search._hop_batch waits rather than stops on an empty list whose n_basins is 0, and draws a parent only while
    the parent's listed hop starts plus the job's own kicks that no listing has counted yet are below max_per_basin.
    The lists are empty when no basin is eligible (params [0, 18], hand [0, z_prime])."""
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
                n_basins=len(E),  # 0: nothing found yet, so an empty list is not exhaustion (run_search._hop_batch)
                max_per_basin=int(h.get('max_per_basin', 32)))  # the limit a hop job holds itself to between passes
    atomic_torch_save(blob, os.path.join(coord_dir, 'hop_parents.pt'))
    return len(order)


def curate(coord_dir):
    """One pass over the campaign in coord_dir: load coord.yaml and registry.pt (a new registry if absent), ingest
    priors and new shards, update the registry, statistics, verdicts, hop_parents.pt (when hops is configured), stop
    files and heartbeat.json (time, host, passes, new_shards). Takes no lease: callers hold the curator lease
    (curate_locked, maybe_curate). A registry lacking the hop or lineage fields gets them (basin_gen 0, basin_parent -1,
    empty hop bookkeeping); one whose hits lack a parent, or whose lineage lists differ in length from its hits or
    basins, is rebuilt by backfill_lineage; one without reg['confirm_effort'] has its confirmation streaks reset. Raises
    ValueError if the registry's energy_model_id, identity_cut or rdf_mode (a registry without rdf_mode counts as
    'envwise') differs from coord.yaml. After verdicts, a stream whose STOP.<s> already exists is marked stop whatever
    its verdict (why names the file and the current verdict). With hops configured, stats['hop_parents'] is the number
    of parents hop_parents.pt lists, and the hop stream, if it has a rule, also counts as stopped for the campaign stop
    (its decision and stop file unchanged) on a pass where that number is 0 and at least hops exhaust_settle_s (default
    3600) seconds, on the clock of the host running the pass, have passed since a pass last saw its row_evals change: a
    pass whose hop row_evals differ from reg['hop_effort_seen'] (-1 when absent, so the first such pass is a change),
    or a pass on a registry without reg['hop_effort_since'], records them there and its time in
    reg['hop_effort_since'] (both mutated), so the settle clock starts on the first pass of any registry. campaign_stop
    is verdicts' campaign_stop, or True when there is at least one stream with a rule and every such stream is stopped
    under these additions. Safe to repeat -- each shard and prior is ingested once and a streak advances only after new
    effort -- but every pass increments reg['passes'], and the settle times are judged at the time of the pass, so a
    pass without new effort can stop a stream (verdicts) or the campaign. Returns (stats, decisions, campaign_stop)."""
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
    """Seconds since the last curate pass (heartbeat.json), on the filesystem's clock: the modification time of a probe
    file written and removed in coord_dir now, minus that of heartbeat.json; inf when there is no heartbeat.json."""
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
    """One curate pass under the curator lease (<coord_dir>/curator_owner.json, the lease maybe_curate takes; here
    taken over once not renewed for 3600 s), released when the pass ends or raises: for passes run by hand (main), by
    run_loop and by export. Raises RunLeaseLost while another process holds the lease (it is never waited on). Returns
    curate's result."""
    from mxtaltools.crystal_search.run_state import RunLease
    lease = RunLease(os.path.join(coord_dir, 'curator'), stale_after_s=3600, settle_s=0.5)
    lease.acquire()
    try:
        return curate(coord_dir)
    finally:
        lease.release()


def pending_shards(coord_dir, reg):
    """Shard files on disk (``<coord_dir>/shards/*/*.pt``) that the registry has not ingested: a sorted list of their
    keys as ingest_shards records them in reg['ingested'] (paths relative to coord_dir, '/'-separated). A refused shard
    counts as ingested; one that could not be loaded does not."""
    done = set(reg['ingested'])
    keys = (os.path.relpath(p, coord_dir).replace('\\', '/')
            for p in sorted(glob.glob(os.path.join(coord_dir, 'shards', '*', '*.pt'))))
    return [k for k in keys if k not in done]


def maybe_curate(coord_dir, every_s):
    """Run a curate pass from inside a search job, between batches, when the last pass (heartbeat.json, judged on the
    filesystem's clock by _heartbeat_age) is older than every_s, or absent, and no other process is curating (the
    curator lease <coord_dir>/curator_owner.json, here stale after ``max(3 * every_s, 3600)`` s, never waited on). The
    heartbeat is read again once the lease is held, and no pass runs if one finished meanwhile. Fail-open: any error is
    printed and the search continues. Returns True if this call ran a pass."""
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
    """Curate every interval_s seconds, each pass through curate_locked, until two completed passes in a row report
    campaign_stop, or until max_hours has elapsed (checked after each attempt; None: no time limit, so it returns only
    through the campaign stop). A pass while another process holds the curator lease is skipped, and a pass that raises
    any other Exception is printed and retried next interval; neither counts toward nor resets the two-pass count.
    SystemExit and KeyboardInterrupt propagate."""
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
    """The campaign's product: one crystal per basin within band_kT of the reference, lowest energy first, rebuilt from
    the basin's lowest-energy state (in its reduced cell unless standardisation failed for that state) and carrying
    energy_key (that energy), basin_id, basin_states (states assigned to the basin) and energy_model_id, saved as a list
    to out_path; plus <out_path without extension>.csv with columns basin, energy, above_ref_kT, states, first_stream
    (the stream that found the basin first) and first_run. With catch_up (default), a pass through curate_locked runs
    first, ingesting the shards written since the last pass (RunLeaseLost if another process holds the curator lease).
    Raises RuntimeError, before writing out_path, if any shard file on disk is still not in the registry
    (pending_shards). Returns the number exported."""
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
    """Command line on coord_dir: with --export BAND_KT OUT, export (with its catch-up pass) and print the count; else
    with --loop, run_loop(--interval, default 900 s, --max_hours); else one pass through curate_locked, then print
    stats.md. SIGTERM is turned into SystemExit(128 + signum), so a pass in progress unwinds and releases the curator
    lease."""
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
