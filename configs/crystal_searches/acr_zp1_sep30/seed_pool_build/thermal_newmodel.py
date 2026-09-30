"""Thermal radius of acridine sg14 Z'=1 basins under acr_newmodel.model with the universal conformer, CPU only, for the
identity cut of the acr_zp1 campaign (atomwise RDF).

1. anchors: the N_POOL lowest (stored MACE energy) rows of the Z'=1 GFN prior's thinned anchors
   (acridine_sg14_zp1_mace_prior_dataset_w3.pt ['prior'], old MACE, old conformer), rebuilt with the new conformer
   (coordinator.rebuild_crystals) and relaxed under acr_newmodel with the campaign's schedule (run_search.crystal_search,
   acr_finish_sep27/0.yaml stages, convergence_eps x10, wrap boundary, no reduction wall; no cascade);
2. the N_ANCHOR lowest relaxed states that are pairwise > SEP apart (atomwise RDF);
3. per anchor and kick size: N_DRAW hop-style kicks (canonicalize_orientation, log_noise_latent_parameters at a fixed
   size, keep_start_representable) -> median crystal-energy rise (kJ/mol per molecule, kT 2.494) and median atomwise RDF
   distance to the zero-kick decode; the 1/2/3 kT crossings are log-log interpolated per anchor.
"""
import os
import sys
import time

DEV = os.environ.get('THERMAL_DEVICE', 'cpu')  # 'cuda' only when the GPU is free
if DEV == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from argparse import Namespace  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(6)
from mxtaltools.common.config_processing import dict2namespace  # noqa: E402
from mxtaltools.crystal_search import coordinator as co  # noqa: E402
from mxtaltools.crystal_search import run_search as rs  # noqa: E402
from mxtaltools.crystal_search.utils import parse_opt_config  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'newmodel')
MXT = 'C:/Users/mikem/Projects/mxt_gfn/mxtaltools'
PRIOR = 'D:/crystal_datasets/conditional/priors/acridine_sg14_zp1_mace_prior_dataset_w3.pt'
MODEL = 'D:/crystal_datasets/acr_newmodel.model'
MOL = 'D:/crystal_datasets/acridine/acr_newmodel_conformer.pt'
CFG = SimpleNamespace(mol_path=MOL, sg=14, z_prime=1)
KT = 2.494
N_POOL, N_ANCHOR, SEP, N_DRAW, BATCH = 16, 6, 0.15, 32, 16
SIZES = [0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.13]
T0 = time.time()


def log(msg):
    line = f'[{time.time() - T0:7.1f}s] {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'log.txt'), 'a') as fh:
        fh.write(line + '\n')


def relax_pool():
    path = os.path.join(OUT, 'relaxed.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = torch.load(PRIOR, weights_only=False)['prior']
    rows = d if isinstance(d, list) else d.batch_to_list()
    E = np.array([float(r.mace) for r in rows])
    order = np.argsort(E)[:N_POOL]
    b = collate_data_list([rows[i] for i in order])
    params = torch.cat([b.cell_lengths, b.cell_angles, b.aunit_centroid[:, :3], b.aunit_orientation[:, :3]], 1).float()
    hand = b.aunit_handedness.float().reshape(len(order), -1)[:, :1]
    starts = co.rebuild_crystals(CFG, params, hand)
    torch.save(starts, os.path.join(OUT, 'starts.pt'))
    log(f'pool: {len(order)} lowest prior anchors, stored old-model energies {E[order].min():.2f}..{E[order].max():.2f}')
    base = yaml.safe_load(open(MXT + '/configs/crystal_searches/acr_finish_sep27/0.yaml'))
    for st in base['opt']:
        st.update(optim_target='mace', show_tqdm=False, convergence_eps=10 * float(st['convergence_eps']))
        st.pop('early_stop', None)
        st.pop('early_stop_kT', None)
    cfg = dict(base, device=DEV, mol_path=MOL, mace_predictor_path=MODEL, uma_predictor_path=None,
               init_sample_method='data', dataset_path=os.path.join(OUT, 'starts.pt'), out_dir=OUT,
               run_name='relax', num_samples=len(order), batch_size=BATCH, grow_batch_size=False, oom_ceiling=False,
               force_restart_run=False, save_trajs=False, sgs_to_search=[14], zp_to_search=[1], coord_dir=None,
               lease_settle_s=0.0, opt_seed=0)
    outs = rs.crystal_search(dict2namespace(cfg))
    torch.save(outs, path)
    log(f'relaxed {len(outs)}: new-model lattice energies {sorted(round(float(o.mace), 2) for o in outs)}')
    return outs


def energy(crystals, pred):
    out = []
    for k in range(0, len(crystals), BATCH):
        b = collate_data_list([c.clone() for c in crystals[k:k + BATCH]], exclude_keys=co.RDF_DROP).to(DEV)
        with torch.no_grad():
            pot = b.compute_crystal_mace(pred, std_orientation=True)
        out += (pot.double() / (b.sym_mult * b.z_prime).double() * 96.485).flatten().cpu().tolist()
    return np.array(out)


def kicked(p, h, n, size, seed):
    b = collate_data_list(co.rebuild_crystals(CFG, p[None].repeat(n, 1), h[None].repeat(n, 1)))
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


def main():
    os.makedirs(OUT, exist_ok=True)
    outs = relax_pool()
    ob = collate_data_list(outs)
    P = torch.cat([ob.cell_lengths, ob.cell_angles, ob.aunit_centroid[:, :3], ob.aunit_orientation[:, :3]], 1).float()
    H = ob.aunit_handedness.float().reshape(len(outs), -1)[:, :1]
    E = np.array([float(o.mace) for o in outs])
    R = co.compute_rdfs(co.rebuild_crystals(CFG, P, H), 8, 'atomwise')
    D = co.rdf_distance_matrix(R, R)
    chosen = []
    for k in np.argsort(E):
        if all(float(D[k, j]) > SEP for j in chosen):
            chosen.append(int(k))
        if len(chosen) == N_ANCHOR:
            break
    log(f'anchors {chosen}: lattice energies {[round(float(E[k]), 2) for k in chosen]} kJ/mol')
    pred = parse_opt_config(dict(optim_target='mace'), Namespace(mace_predictor_path=MODEL), DEV, None)['predictor']
    res = []
    for a_i, k in enumerate(chosen):
        c0 = kicked(P[k], H[k], 1, 0.0, 0)
        e0, r0 = energy(c0, pred)[0], co.compute_rdfs(c0, 8, 'atomwise')
        dE, dd = [], []
        for s_i, s in enumerate(SIZES):
            t = time.time()
            cs = kicked(P[k], H[k], N_DRAW, s, 1000 * a_i + s_i)
            dE.append(float(np.median(energy(cs, pred) - e0)) / KT)
            dd.append(float(np.median(co.rdf_distance_matrix(co.compute_rdfs(cs, 8, 'atomwise'), r0)[:, 0].numpy())))
            log(f'  anchor {a_i} size {s}: dE median {dE[-1]:+.3f} kT, RDF d median {dd[-1]:.4f} ({time.time() - t:.0f} s)')
        res.append((np.array(dE), np.array(dd)))
        torch.save(dict(chosen=chosen, res=res, sizes=SIZES), os.path.join(OUT, 'thermal.pt'))
    log(f'Caption: acr_newmodel, universal conformer, sg14 Z\'=1; {len(chosen)} relaxed distinct low basins, {N_DRAW} '
        f'hop-style kicks per size; median over anchors [min, max] of the per-anchor crossing.')
    log('| median rise | latent kick size | atomwise RDF distance |')
    for n in (1, 2, 3):
        s = [crossing(SIZES, dE, n) for dE, _ in res]
        d = [float(np.exp(np.interp(np.log(x), np.log(SIZES), np.log(np.maximum(dd, 1e-9))))) if np.isfinite(x)
             else float('nan') for x, (_, dd) in zip(s, res)]
        f = lambda v: f'{np.nanmedian(v):.4f} [{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]'
        log(f'| {n} kT | {f(s)} | {f(d)} |')


if __name__ == '__main__':
    main()
