"""The seed list of the acr_fm_oct27 campaigns: every distinct acridine sg14 Z'=1 state from the earlier landscapes, for the
seeded stream to re-relax under each foundation model.

    python build_seed_pool.py [--cut 0.031] [--write] [--anchors 14]

Sources (dev-box paths, read only), in priority order; within a source, lowest stored energy first (file order where none
is stored; acr_campaign_sep28 is a Z'=2 campaign and is not a source):
  sep30_registry  acr_zp1_sep30 campaign basins (acr_newmodel), local registry after the final curate pass
  oct05_anchors   the pooled_oct05 acridine anchors (acr_zp1_sep30 basins within 5 kT, trainer chart)
  old_prior       anchors of acridine_sg14_zp1_mace_prior_dataset_w3.pt (old fine-tuned MACE)
  sep30_seeds     acr_zp1_sep30's seed list (May and December production searches)
Each state is rebuilt with the reference conformer (acr_newmodel_conformer.pt; the campaigns rebuild seeds with their own
molecule in preflight) and kept if physical, as acr_zp1_sep30's thin_zp1.py: angular factor > 0.5, packing coefficient in
[0.5, 0.85]. Then greedy leader thinning in the latent (periodic coordinates wrapped), at --cut (default 0.031, the 1 kT
latent kick under acr_newmodel, the cut acr_zp1_sep30 thinned its seeds at). Writes seed_pool.pt beside this script:
params [n, 12], hand [n, 1], source per row.
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
    out = {}
    out['sep30_registry'] = from_registry(A + 'campaigns/acr_zp1_sep30/registry.pt')
    out['oct05_anchors'] = from_batch(torch.load('D:/crystal_datasets/pooled_oct02/acridine_mace/cluster/anchors.pt',
                                                 weights_only=False)['prior'])
    out['old_prior'] = from_batch(torch.load('D:/crystal_datasets/conditional/priors/'
                                             'acridine_sg14_zp1_mace_prior_dataset_w3.pt', weights_only=False)['prior'])
    s = torch.load('C:/Users/mikem/Projects/mxt_gfn/mxtaltools/configs/crystal_searches/acr_zp1_sep30/seed_pool.pt',
                   weights_only=False)
    out['sep30_seeds'] = (s['params'].float(), s['hand'].float().reshape(-1, 1),
                          torch.full((len(s['params']),), float('nan'), dtype=torch.float64))
    return out


def rebuilt(params, hand, chunk=4000):
    lat, cp, ang = [], [], []
    for k in range(0, len(params), chunk):
        b = collate_data_list(co.rebuild_crystals(CFG, params[k:k + chunk], hand[k:k + chunk]))
        lat.append(b.latent_params().detach().double())
        cp.append(b.packing_coeff.double().flatten())
        ang.append((b.cell_volume.flatten() / b.cell_lengths.prod(-1)).double())
    return torch.cat(lat), torch.cat(cp), torch.cat(ang)


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
    if a.write:
        out = os.path.join(HERE, 'seed_pool.pt')
        torch.save(dict(params=P[keep], hand=H[keep], source=list(src[keep]), latent_cut=a.cut,
                        note='distinct acridine sg14 Zp=1 states of the earlier landscapes (build_seed_pool.py); '
                             'rebuilt with the campaign molecule by preflight'), out)
        print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()
