"""The seed list of the acr_fm_oct27 campaigns: the distinct acridine sg14 Z'=1 states of the two earlier landscapes, for
the seeded stream to re-relax under each foundation model.

    python build_seed_pool.py [--cut 0.031] [--rdf_cut 0.134 [0.08 ...]] [--write] [--anchors 14]

Sources (dev-box paths, read only), in priority order; within a source, lowest stored energy first:
  premerge   the basin leaders of pool_acr_oct03's premerge (cluster job 19108908): one coordinator curate pass, atomwise
             RDF at 0.134, over every acr_zp1_sep30 end state within 10 kT under acr_newmodel -- 24,022 basins. That
             campaign re-relaxed the May and December search states under acr_newmodel, so their outcomes are here
             (copied from /scratch/mk8347/data/crystal_datasets/pooled_oct02/acridine_mace/pool/registry.pt)
  old_prior  anchors of acridine_sg14_zp1_mace_prior_dataset_w3.pt, the old fine-tuned MACE's landscape
The known forms ACRDIN04 and ACRDIN12 are added by preflight.py (../acr_zp1_sep30/known_forms.pt). The first version of
this list (MXtalTools 16950107) pooled the premerge's own inputs as well (the acr_zp1_sep30 registry and seed list, the
pooled_oct05 anchors) and thinned in the latent only: 51,690 seeds, mostly copies.
Each state is rebuilt with the reference conformer (acr_newmodel_conformer.pt; the campaigns rebuild seeds with their own
molecule in preflight) and kept if physical, as acr_zp1_sep30's thin_zp1.py: angular factor > 0.5, packing coefficient in
[0.5, 0.85]. Then greedy leader thinning in the latent (periodic coordinates wrapped), at --cut (default 0.031, the 1 kT
latent kick under acr_newmodel, the cut acr_zp1_sep30 thinned its seeds at). Latent distance is not a physical identity:
the same crystal in another setting, a relabelled twin or an unconverged copy of one minimum lies far away in the latent.
--rdf_cut then thins the latent leaders, in the same order, by atomwise RDF distance (coordinator.rdf_distance_matrix,
the campaigns' identity metric) at the first cut given; further cuts are only reported. Writes seed_pool.pt beside this
script: params [n, 12], hand [n, 1], source per row.
The committed seed_pool.pt (2026-10-27): `--cut 0 --rdf_cut 0.134 --write`, no latent thinning. Physical: premerge
22,114 of 24,022, old_prior 14,671 of 14,671. RDF leaders: 21,882 (premerge 19,601, old_prior 2,281). Peak 11.8 GB RAM.
--anchors N writes calib_anchors.pt instead, the start states of calibrate.py: the known forms (ACRDIN04, ACRDIN12) and
the N lowest physical old_prior anchors by stored old-MACE energy that are pairwise distinct (atomwise RDF distance
> 0.15, thermal_newmodel.py's separation of anchors; the lowest anchors are mostly copies of one basin), from the 400
lowest.
"""
import argparse
import os
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import torch
from scipy.spatial import cKDTree

torch.set_num_threads(4)
from mxtaltools.crystal_search import coordinator as co  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A = 'D:/crystal_datasets/acridine/'
CFG = SimpleNamespace(mol_path=A + 'acr_newmodel_conformer.pt', sg=14, z_prime=1)
PERIODIC = [6, 8, 10, 11]  # sg 14 latent rows with period 2: centroid x and z, the two angular orientation rows


def from_registry(path):
    r = torch.load(path, weights_only=False)
    return (torch.as_tensor(np.asarray(r['basin_params']), dtype=torch.float32),
            torch.as_tensor(np.asarray(r['basin_hand']), dtype=torch.float32).reshape(-1, 1),
            torch.as_tensor(np.asarray(r['basin_E']), dtype=torch.float64))


def from_batch(b, key='mace'):
    p = torch.cat([b.cell_lengths, b.cell_angles, b.aunit_centroid[:, :3], b.aunit_orientation[:, :3]], 1).float()
    h = b.aunit_handedness.float().reshape(b.num_graphs, -1)[:, :1]
    e = getattr(b, key).double().flatten() if key in b.keys() else torch.full((b.num_graphs,), float('nan'))
    return p, h, e


def sources():
    return {'premerge': from_registry('D:/crystal_datasets/pooled_oct02/acridine_mace/pool/registry.pt'),
            'old_prior': from_batch(torch.load('D:/crystal_datasets/conditional/priors/'
                                               'acridine_sg14_zp1_mace_prior_dataset_w3.pt', weights_only=False)['prior'])}


def rebuilt(params, hand, chunk=4000):
    lat, cp, ang = [], [], []
    for k in range(0, len(params), chunk):
        b = collate_data_list(co.rebuild_crystals(CFG, params[k:k + chunk], hand[k:k + chunk]))
        lat.append(b.latent_params().detach().double())
        cp.append(b.packing_coeff.double().flatten())
        ang.append((b.cell_volume.flatten() / b.cell_lengths.prod(-1)).double())
    return torch.cat(lat), torch.cat(cp), torch.cat(ang)


def rdf_leaders(P, H, rows, cuts, block=1024, lead_chunk=4096):
    """Greedy leader clustering of P[rows] (visited in that order) by atomwise RDF distance, per cut: a row is kept
    unless a kept row lies within the cut. RDFs held in float16 (276 channels x 100 bins per acridine crystal); a block
    is compared with the kept rows lead_chunk at a time (all at once in float32, with the metric's copies, passed 16 GB
    at ~20,000 kept rows)."""
    R = []
    for k in range(0, len(rows), block):
        idx = torch.as_tensor(rows[k:k + block])
        R.append(co.compute_rdfs(co.rebuild_crystals(CFG, P[idx], H[idx]), 64, 'atomwise').half())
        print(f'  rdfs {k + len(idx)}/{len(rows)}', flush=True)
    R = torch.cat(R)
    out = {}
    for cut in cuts:
        kept = []
        for k in range(0, len(rows), block):
            blk = R[k:k + block].float()
            covered = torch.zeros(len(blk), dtype=torch.bool)
            for c in range(0, len(kept), lead_chunk):
                lead = R[torch.as_tensor(kept[c:c + lead_chunk])].float()
                covered |= co.rdf_distance_matrix(blk, lead).min(1).values < cut
            d_in = co.rdf_distance_matrix(blk, blk)
            new = []
            for i in range(len(blk)):
                if not covered[i] and all(float(d_in[i, j]) >= cut for j in new):
                    new.append(i)
            kept += [k + i for i in new]
        out[cut] = [rows[i] for i in kept]
        print(f'  rdf cut {cut}: {len(rows)} -> {len(kept)}', flush=True)
    return out


def calib_anchors(n):
    p, h, e = from_batch(torch.load('D:/crystal_datasets/conditional/priors/acridine_sg14_zp1_mace_prior_dataset_w3.pt',
                                    weights_only=False)['prior'])
    _, cp, ang = rebuilt(p, h)
    phys = torch.nonzero((ang > 0.5) & (cp >= 0.5) & (cp <= 0.85)).flatten()
    cand = phys[torch.argsort(e[phys])[:400]]
    D = co.rdf_distance_matrix(*[co.compute_rdfs(co.rebuild_crystals(CFG, p[cand], h[cand]), 32, 'atomwise')] * 2)
    picked = []
    for k in range(len(cand)):
        if all(float(D[k, j]) > 0.15 for j in picked):
            picked.append(k)
        if len(picked) == n:
            break
    low = cand[picked]
    known = torch.load(os.path.join(HERE, '..', 'acr_zp1_sep30', 'known_forms.pt'), weights_only=False)
    out = dict(params=torch.cat([known['params'].float(), p[low]]), hand=torch.cat([known['hand'].float(), h[low]]),
               identifier=list(known['identifier']) + [f'old_prior#{int(i)}' for i in low],
               note=f'calibrate.py start states: the known forms and the {n} lowest pairwise-distinct physical old_prior anchors '
                    f'(old-MACE lattice energies {float(e[low].min()):.2f}..{float(e[low].max()):.2f} kJ/mol)')
    torch.save(out, os.path.join(HERE, 'calib_anchors.pt'))
    print(out['note'], '->', os.path.join(HERE, 'calib_anchors.pt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cut', type=float, default=0.031)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--anchors', type=int, default=None)
    ap.add_argument('--rdf_cut', type=float, nargs='*', default=[])
    a = ap.parse_args()
    if a.anchors:
        return calib_anchors(a.anchors)
    P, H, src, order_key = [], [], [], []
    for name, (p, h, e) in sources().items():
        lat, cp, ang = rebuilt(p, h)
        phys = (ang > 0.5) & (cp >= 0.5) & (cp <= 0.85) & torch.isfinite(lat).all(1)
        rank = torch.argsort(torch.argsort(torch.where(torch.isfinite(e), e, torch.arange(len(e), dtype=e.dtype)),
                                           stable=True), stable=True)
        print(f'{name}: {len(p)} states, physical {int(phys.sum())}', flush=True)
        P.append(p[phys]); H.append(h[phys]); src += [name] * int(phys.sum())
        order_key.append(torch.stack([torch.full((int(phys.sum()),), float(len(order_key))), rank[phys].double()], 1))
        if 'L' not in locals():
            L = lat[phys]
        else:
            L = torch.cat([L, lat[phys]])
    P, H, key = torch.cat(P), torch.cat(H), torch.cat(order_key)
    order = np.lexsort((key[:, 1].numpy(), key[:, 0].numpy()))
    x = (L.numpy() + 1.0).copy()
    box = np.full(x.shape[1], 1e3)
    for j in PERIODIC:
        box[j] = 2.0
        x[:, j] = np.mod(x[:, j], 2.0)
    x = np.clip(x, 0.0, None)
    pairs = cKDTree(x, boxsize=box).query_pairs(a.cut, output_type='ndarray')
    nbr = [[] for _ in range(len(x))]
    for i, j in pairs:
        nbr[i].append(j)
        nbr[j].append(i)
    state = np.zeros(len(x), dtype=np.int8)
    for i in order:
        if state[i]:
            continue
        state[i] = 1
        for j in nbr[i]:
            if state[j] == 0:
                state[j] = -1
    keep = np.sort(np.nonzero(state == 1)[0])
    src = np.asarray(src)
    print(f'latent cut {a.cut}: {len(x)} physical states -> {len(keep)} seeds; by source: '
          + ', '.join(f'{s} {int((src[keep] == s).sum())}' for s in dict.fromkeys(src)), flush=True)
    if a.rdf_cut:
        by_cut = rdf_leaders(P, H, [int(i) for i in order if state[i] == 1], a.rdf_cut)
        for cut, rows in by_cut.items():
            print(f'atomwise RDF cut {cut}: {len(keep)} -> {len(rows)} seeds; by source: '
                  + ', '.join(f'{s} {int((src[rows] == s).sum())}' for s in dict.fromkeys(src)), flush=True)
        keep = np.sort(np.asarray(by_cut[a.rdf_cut[0]]))
    if a.write:
        out = os.path.join(HERE, 'seed_pool.pt')
        torch.save(dict(params=P[keep], hand=H[keep], source=list(src[keep]), latent_cut=a.cut,
                        rdf_cut=a.rdf_cut[0] if a.rdf_cut else None,
                        note='distinct acridine sg14 Zp=1 states of the earlier landscapes (build_seed_pool.py); '
                             'rebuilt with the campaign molecule by preflight'), out)
        print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()
