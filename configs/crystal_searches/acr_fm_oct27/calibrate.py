"""acr_fm_oct27 calibration of one model, on a GPU (submit_calibrate.sbatch; `launch.sh calibrate` submits one job per
model): the identity cut of the model's campaign and the throughput its job counts are sized by.

    python calibrate.py MODEL OUT_DIR

1. relax: calib_anchors.pt (the known forms ACRDIN04 and ACRDIN12, and 14 pairwise-distinct low anchors of the old Z'=1
   prior), rebuilt with the model's conformer and relaxed under the model on the campaign schedule, every row polished
   (campaign.stages(kind, cascade=False, polish_cascade=False)); each relaxed state's lattice energy is reported by name;
2. thermal radius, as acr_zp1_sep30's seed_pool_build/thermal_newmodel.py: the N_ANCHOR lowest relaxed states pairwise
   > SEP apart (atomwise RDF); per anchor and kick size, N_DRAW hop-style kicks (canonicalize_orientation,
   log_noise_latent_parameters at a fixed size, keep_start_representable) -> median crystal-energy rise (kT 2.494) and
   median atomwise RDF distance to the zero-kick decode; the 1/2/3 kT crossings log-log interpolated per anchor. The
   identity cut is the median over anchors of the RDF distance at the 1 kT crossing (owner 2026-09-28: the identity cut
   is the thermal radius);
3. throughput: N_THRU random starts, BATCH per batch (fixed), through the random stream's schedule (stage-2 cascade,
   polish stage) -> wall time, relaxations per GPU-hour, peak GPU memory.
Writes OUT_DIR/result.json (and log.txt, starts.pt, relaxed.pt, thermal.pt, thru/). A rerun reuses relaxed.pt.
"""
import json
import os
import platform
import sys
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign as C  # noqa: E402
from mxtaltools.common.config_processing import dict2namespace  # noqa: E402
from mxtaltools.crystal_search import coordinator as co  # noqa: E402
from mxtaltools.crystal_search import run_search as rs  # noqa: E402
from mxtaltools.crystal_search.utils import parse_opt_config  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

N_ANCHOR, SEP, N_DRAW, KBATCH = 6, 0.15, 32, 16
SIZES = [0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.13]
N_THRU, BATCH = 192, 96
DEV = 'cuda'
SMOKE = os.environ.get('CALIB_SMOKE') == '1'  # a local CPU run of every code path at toy sizes; its numbers mean nothing
if SMOKE:
    N_ANCHOR, N_DRAW, KBATCH, SIZES, N_THRU, BATCH, DEV = 2, 3, 3, [0.01, 0.13], 4, 2, 'cpu'
T0 = time.time()
OUT = None


def log(msg):
    line = f'[{time.time() - T0:7.1f}s] {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'log.txt'), 'a') as fh:
        fh.write(line + '\n')


def smoke_stages(opt):
    return [dict(st, max_num_steps=4) for st in opt] if SMOKE else opt


def search_cfg(model, out_dir, run_name, **kw):
    kw['opt'] = smoke_stages(kw['opt'])
    cfg = dict(yaml.safe_load(C.BASE.read_text()), device=DEV, mol_path=C.mol_path(model), out_dir=out_dir,
               run_name=run_name, save_trajs=False, force_restart_run=False, sgs_to_search=[14], zp_to_search=[1],
               coord_dir=None, grow_batch_size=False, oom_ceiling=False, lease_settle_s=0.0,
               min_intermolecular_contact=C.MIN_CONTACT, **C.predictor_paths(model))
    cfg.update(kw)
    return cfg


def relax(model):
    path = os.path.join(OUT, 'relaxed.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    kind = C.MODELS[model]['kind']
    a = torch.load(os.path.join(HERE, 'calib_anchors.pt'), weights_only=False)
    if SMOKE:
        a = dict(params=a['params'][:3], hand=a['hand'][:3], identifier=a['identifier'][:3])
    starts = co.rebuild_crystals(SimpleNamespace(mol_path=C.mol_path(model), sg=14, z_prime=1), a['params'], a['hand'])
    torch.save(starts, os.path.join(OUT, 'starts.pt'))
    cfg = search_cfg(model, OUT, 'relax', init_sample_method='data', dataset_path=os.path.join(OUT, 'starts.pt'),
                     num_samples=len(starts), batch_size=len(starts), opt_seed=0,
                     opt=C.stages(kind, cascade=False, polish_cascade=False))
    outs = rs.crystal_search(dict2namespace(cfg))
    res = dict(rows=outs, names=[a['identifier'][int(o.dataset_index)] for o in outs],
               E=[float(getattr(o, kind)) for o in outs])
    torch.save(res, path)
    return res


def params_of(rows):
    b = collate_data_list(rows, exclude_keys=co.RDF_DROP)
    p = torch.cat([b.cell_lengths, b.cell_angles, b.aunit_centroid[:, :3], b.aunit_orientation[:, :3]], 1).float()
    return p.cpu(), b.aunit_handedness.float().reshape(len(rows), -1)[:, :1].cpu()


def energy(crystals, pred, kind):
    out = []
    for k in range(0, len(crystals), KBATCH):
        b = collate_data_list([c.clone() for c in crystals[k:k + KBATCH]], exclude_keys=co.RDF_DROP).to(DEV)
        with torch.no_grad():
            pot = (b.compute_crystal_mace(pred, std_orientation=True) if kind == 'mace'
                   else b.compute_crystal_uma(pred, std_orientation=True))
        out += (pot.double().flatten() / (b.sym_mult * b.z_prime).double().flatten() * 96.485).cpu().tolist()
    return np.array(out)


def kicked(cfg, p, h, n, size, seed):
    b = collate_data_list(co.rebuild_crystals(cfg, p[None].repeat(n, 1), h[None].repeat(n, 1)))
    b.canonicalize_orientation()
    torch.manual_seed(seed)
    lg = float(np.log10(size)) if size > 0 else -30.0
    b.log_noise_latent_parameters(lg, lg, keep_start_representable=True)
    return b.batch_to_list()


def crossing(x, y, level):
    for k in range(1, len(x)):
        if y[k - 1] < level <= y[k] and y[k - 1] > 0:
            t = (np.log(level) - np.log(y[k - 1])) / (np.log(y[k]) - np.log(y[k - 1]))
            return float(np.exp(np.log(x[k - 1]) + t * (np.log(x[k]) - np.log(x[k - 1]))))
    return float('nan')


def thermal(model, rel):
    kind = C.MODELS[model]['kind']
    cfg = SimpleNamespace(mol_path=C.mol_path(model), sg=14, z_prime=1)
    P, H = params_of(rel['rows'])
    E = np.array(rel['E'])
    R = co.compute_rdfs(co.rebuild_crystals(cfg, P, H), 8, 'atomwise')
    D = co.rdf_distance_matrix(R, R)
    chosen = []
    for k in np.argsort(E):
        if all(float(D[k, j]) > SEP for j in chosen):
            chosen.append(int(k))
        if len(chosen) == N_ANCHOR:
            break
    log(f'anchors {[rel["names"][k] for k in chosen]}: lattice energies {[round(float(E[k]), 2) for k in chosen]} '
        f'kJ/mol')
    pred = parse_opt_config(dict(optim_target=kind), Namespace(**C.predictor_paths(model)), DEV, None)['predictor']
    res = []
    for a_i, k in enumerate(chosen):
        c0 = kicked(cfg, P[k], H[k], 1, 0.0, 0)
        e0, r0 = energy(c0, pred, kind)[0], co.compute_rdfs(c0, 8, 'atomwise')
        dE, dd = [], []
        for s_i, s in enumerate(SIZES):
            t = time.time()
            cs = kicked(cfg, P[k], H[k], N_DRAW, s, 1000 * a_i + s_i)
            dE.append(float(np.median(energy(cs, pred, kind) - e0)) / C.KT)
            dd.append(float(np.median(co.rdf_distance_matrix(co.compute_rdfs(cs, 8, 'atomwise'), r0)[:, 0].numpy())))
            log(f'  anchor {a_i} size {s}: dE median {dE[-1]:+.3f} kT, RDF d median {dd[-1]:.4f} '
                f'({time.time() - t:.0f} s)')
        res.append((np.array(dE), np.array(dd)))
        torch.save(dict(chosen=chosen, res=res, sizes=SIZES), os.path.join(OUT, 'thermal.pt'))
    table = {}
    for n in (1, 2, 3):
        s = [crossing(SIZES, dE, n) for dE, _ in res]
        d = [float(np.exp(np.interp(np.log(x), np.log(SIZES), np.log(np.maximum(dd, 1e-9))))) if np.isfinite(x)
             else float('nan') for x, (_, dd) in zip(s, res)]
        table[f'{n}kT'] = dict(latent_kick=s, rdf_distance=d)
    log(f"Caption: {model}, its gas-phase conformer, sg14 Z'=1; {len(chosen)} relaxed distinct low basins, {N_DRAW} "
        f'hop-style kicks per size; median over anchors [min, max] of the per-anchor crossing.')
    log('| median rise | latent kick size | atomwise RDF distance |')
    f = lambda v: f'{np.nanmedian(v):.4f} [{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]'  # noqa: E731
    for n, row in table.items():
        log(f"| {n} | {f(row['latent_kick'])} | {f(row['rdf_distance'])} |")
    return dict(n_draw=N_DRAW, anchors=[rel['names'][k] for k in chosen], anchor_E=[float(E[k]) for k in chosen],
                table=table, dE_kT=[r[0].tolist() for r in res], rdf_d=[r[1].tolist() for r in res], sizes=SIZES)


def throughput(model):
    kind = C.MODELS[model]['kind']
    cfg = search_cfg(model, os.path.join(OUT, 'thru'), 'thru', init_sample_method='random', dataset_path=None,
                     num_samples=N_THRU, batch_size=BATCH, opt_seed=7, opt=C.stages(kind))
    os.makedirs(cfg['out_dir'], exist_ok=True)
    if DEV == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    outs = rs.crystal_search(dict2namespace(cfg))
    dt = time.time() - t
    E = np.array([float(getattr(o, kind)) for o in outs])
    r = dict(starts=N_THRU, batch=BATCH, returned=len(outs), wall_s=dt, relax_per_gpu_hour=N_THRU * 3600 / dt,
             peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if DEV == 'cuda' else None,
             lowest_E=float(np.nanmin(E)) if len(E) else None,
             within_polish=int(np.sum(E <= np.nanmin(E) + C.POLISH_KT * C.KT)) if len(E) else 0)
    log(f"throughput: {N_THRU} random starts in batches of {BATCH} on {DEV}: {dt:.0f} s, "
        f"{r['relax_per_gpu_hour']:.0f} relaxations per GPU-hour, peak {r['peak_gpu_gb']} GB")
    return r


def main(model, out):
    global OUT
    OUT = out
    os.makedirs(OUT, exist_ok=True)
    m = C.MODELS[model]
    import fairchem.core
    from importlib.metadata import version
    env = dict(python=platform.python_version(), torch=torch.__version__, fairchem=version('fairchem-core'),
               fairchem_file=fairchem.core.__file__, mace=version('mace-torch'),
               gpu=torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu', mxt_commit=os.environ.get('MXT_COMMIT'))
    assert not m['fairchem'] or env['fairchem'] == m['fairchem'], f"fairchem {env['fairchem']}, need {m['fairchem']}"
    mid = co.energy_model_id({'optim_target': m['kind']}, Namespace(**C.predictor_paths(model)))
    log(f'{model}: {mid}; {env}')
    rel = relax(model)
    order = np.argsort(rel['E'])
    log('relaxed anchors (lattice energy, kJ/mol): ' + ', '.join(f"{rel['names'][k]} {rel['E'][k]:.2f}" for k in order))
    th = thermal(model, rel)
    cut = float(np.nanmedian(th['table']['1kT']['rdf_distance']))
    kick = float(np.nanmedian(th['table']['1kT']['latent_kick']))
    thru = throughput(model)
    result = dict(model=model, energy_model_id=mid, env=env, identity_cut=cut, latent_kick_1kT=kick,
                  relaxed={rel['names'][k]: rel['E'][k] for k in order}, thermal=th, throughput=thru,
                  stages=C.stages(m['kind']), wall_s=time.time() - T0)
    json.dump(result, open(os.path.join(OUT, 'result.json'), 'w'), indent=1)
    log(f'identity cut {cut:.4f} (latent kick {kick:.4f}); wrote {os.path.join(OUT, "result.json")}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
