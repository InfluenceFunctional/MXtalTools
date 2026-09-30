"""Thin the pooled sg14 Z'=1 states (pool.pt) into the seed list of acr_zp1_sep30.

usage: python thin_zp1.py <latent cut> [--write]

Filters: physical (angular factor > 0.5, packing coefficient in [0.5, 0.85], lj < 0); May states within MAY_KT kT
(2.494 kJ/mol) of the May minimum under the old MACE; December states (LJ minima, no MACE energy) the DEC_FRAC lowest
by lj. Each state rebuilt with the universal conformer, its latent vector computed (latent_params), then greedy
lowest-energy-first leader thinning in latent space at the cut: May first (by MACE), then December (by lj) against the
May leaders and each other. Writes <acr_zp1_sep30>/seed_pool.pt: params [n, 12], hand [n, 1], source per row.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
from types import SimpleNamespace  # noqa: E402

from mxtaltools.crystal_search import coordinator as co  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

HERE = os.environ.get('POOL_DIR', os.path.dirname(os.path.abspath(__file__)))
OUT = 'C:/Users/mikem/Projects/mxt_gfn/mxtaltools/configs/crystal_searches/acr_zp1_sep30/seed_pool.pt'
CFG = SimpleNamespace(mol_path='D:/crystal_datasets/acridine/acr_newmodel_conformer.pt', sg=14, z_prime=1)
MAY_KT, DEC_FRAC = 10.0, 0.2


def latents(params, hand):
    out = []
    for k in range(0, len(params), 4000):
        b = collate_data_list(co.rebuild_crystals(CFG, params[k:k + 4000], hand[k:k + 4000]))
        out.append(b.latent_params().detach().float())
        print(f'  latents {k + len(out[-1])}/{len(params)}', flush=True)
    return torch.cat(out)


def leaders(lat, order, cut, existing=None):
    """greedy: visit in order, keep a state unless a kept state (or an existing leader) lies within cut"""
    kept = [] if existing is None else [existing]
    keep_idx = []
    buf = torch.zeros(0, lat.shape[1])
    for i in order:
        x = lat[i:i + 1]
        near = any(bool((torch.cdist(x, L) < cut).any()) for L in kept if len(L)) or \
            (len(buf) and bool((torch.cdist(x, buf) < cut).any()))
        if near:
            continue
        keep_idx.append(int(i))
        buf = torch.cat([buf, x])
        if len(buf) >= 2048:
            kept.append(buf)
            buf = torch.zeros(0, lat.shape[1])
    kept.append(buf)
    return keep_idx, torch.cat([L for L in kept if len(L)])


def main():
    cut = float(sys.argv[1])
    p = torch.load(os.path.join(HERE, 'pool.pt'), weights_only=False)
    src = np.array([s.split(':')[0] for s in p['src']])
    phys = (p['ang'] > 0.5) & (p['cp'] >= 0.5) & (p['cp'] <= 0.85) & (p['lj'] < 0)
    may = torch.as_tensor(src == 'may') & phys
    e = p['mace'].clone()
    may &= e <= e[torch.as_tensor(src == 'may')].min() + MAY_KT * 2.494
    dec = torch.as_tensor(src == 'dec') & phys
    lj = p['lj'][dec]
    dec_idx = torch.nonzero(dec).flatten()[torch.argsort(lj)[:int(DEC_FRAC * len(lj))]]
    may_idx = torch.nonzero(may).flatten()
    print(f'May: {int((torch.as_tensor(src == "may")).sum())} -> physical, within {MAY_KT} kT: {len(may_idx)}; '
          f'December: {int((torch.as_tensor(src == "dec")).sum())} -> physical {int(dec.sum())}, lowest '
          f'{DEC_FRAC:.0%} by lj: {len(dec_idx)}', flush=True)
    sel = torch.cat([may_idx, dec_idx])
    lat = latents(p['params'][sel], p['hand'][sel])
    nm = len(may_idx)
    k_may, L = leaders(lat, np.argsort(p['mace'][may_idx].numpy()), cut)
    k_dec, _ = leaders(lat, nm + np.argsort(p['lj'][dec_idx].numpy()), cut, existing=L)
    keep = sel[torch.as_tensor(k_may + k_dec)]
    print(f'latent cut {cut}: May {nm} -> {len(k_may)}; December {len(dec_idx)} -> {len(k_dec)}; seeds {len(keep)}')
    if '--write' in sys.argv:
        torch.save(dict(params=p['params'][keep], hand=p['hand'][keep], source=[p['src'][int(i)] for i in keep],
                        latent_cut=cut, may_kT=MAY_KT, dec_frac=DEC_FRAC,
                        note='acridine sg14 Zp=1 search end states (May acr_production chunks, old MACE; December '
                             'production, LJ), filtered and latent-thinned by thin_zp1.py; rebuilt with the campaign '
                             'molecule by preflight.py'), OUT)
        print('wrote', OUT, os.path.getsize(OUT) / 1e6, 'MB')


if __name__ == '__main__':
    main()
