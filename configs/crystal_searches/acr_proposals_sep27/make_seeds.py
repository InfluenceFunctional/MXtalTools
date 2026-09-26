"""
Compact seed files for the acr_proposals_sep27 proposal arms (acridine sg14 Z'=2). LOCAL provenance script: it reads
local files (D:/crystal_datasets/acridine/...) and writes seeds/*.pth here (.pth, not .pt: *.pt is Git LFS-tracked in this repo); the cluster only runs build_seeds.py.

hops.pth           one parent per low-energy family of the pooled acr_wrap_sep26 family clustering (sep26 in-band end
                   states, aug21 unseeded in-band set, the relaxed forms; families made only of doubled Z'=1 structures
                   are left to dblkick), lowest-energy non-doubled member; latent log-noise kicks at -1.0 and -0.5.
doubled_kicked.pth the 121 doubled Z'=1 families: one unkicked copy each, plus latent log-noise kicks at -2.0/-1.5/-1.0.
elj_starts.pth     the lowest-eLJ half of the distinct physical end states of the local eLJ pre-search.

Kicks use MolCrystalData.log_noise_latent_parameters (the aug21 seeded-ladder operator) on canonical rotvecs. A parent
whose cell the latent cannot represent (e.g. a or c beyond the latent's range: the latent clips, and a "kick" would
compress that axis by 20-30%) is detected by a latent round trip and gets no kicked copies. Torch seeds are fixed, so
the files are reproducible.

    python make_seeds.py [hops] [dblkick] [elj]      # default: all three
"""
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, 'seeds')
INPUTS = 'D:/crystal_datasets/acridine/sep27_seeds/inputs'
DOUBLED = 'D:/crystal_datasets/acridine/doubled_zp1_sg14_2026-09-26.pt'
ELJ_PRE = 'D:/crystal_datasets/acridine/sep27_seeds/elj_pre_acridine_sg14_zp2_0.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
KT, FLOOR = 2.494, -62.812
BAND = FLOOR + 2 * KT
TORCH_SEED = {'hops': 27001, 'dblkick': 27002}


def strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


def latent_representable(crystals, rtol=1e-3):
    """True where a latent round trip (latent_params -> latent_to_cell_params) returns the same cell parameters."""
    b = collate_data_list([strip(c) for c in crystals])
    b.canonicalize_orientation()
    before = b.full_cell_parameters().detach().clone()
    b.latent_to_cell_params(b.latent_params(gauge_fix_free_axes=False))
    after = b.full_cell_parameters().detach()
    rel = ((after[:, :3] - before[:, :3]).abs() / before[:, :3]).amax(1)
    return (rel < rtol).numpy()


def kicked(parents, meta, levels, tag):
    """levels: list of (log_noise or None, copies). Returns params [N, 18], handedness [N, 2], meta list."""
    params, hand, info = [], [], []
    torch.manual_seed(TORCH_SEED[tag])
    for lvl, copies in levels:
        b = collate_data_list([strip(c) for c in parents for _ in range(copies)])
        b.canonicalize_orientation()
        if lvl is not None:
            b.log_noise_latent_parameters(lvl, lvl)
        params.append(b.full_cell_parameters().detach().float())
        hand.append(b.aunit_handedness.detach().float().reshape(-1, 2))
        info += [dict(meta[i], log_noise=lvl, copy=k) for i in range(len(parents)) for k in range(copies)]
    return torch.cat(params), torch.cat(hand), info


def save(name, params, hand, info):
    assert torch.isfinite(params).all() and params.shape[1] == 18 and len(params) == len(info) == len(hand)
    path = os.path.join(SEEDS, name)
    torch.save({'params': params, 'handedness': hand, 'meta': info}, path)
    print(f'{name}: {len(params)} seeds, {os.path.getsize(path) / 1e6:.2f} MB')


def make_hops():
    F = torch.load(f'{INPUTS}/sep26_families.pt', weights_only=False)
    A = torch.load(f'{INPUTS}/sep26_arms_low.pt', weights_only=False)
    zb = torch.load(f'{INPUTS}/zp2_band.pt', weights_only=False)
    dbl = torch.load(DOUBLED, weights_only=False)
    pool = []
    for name in F['arm_band_rows']:
        pool += [(name, c) for c in A[name]['low'] if float(c.mace) <= BAND]
    pool += [('doubled', c) for c in dbl] + [('aug21', c) for c in zb['band']]
    pool += [('ACRDIN07', zb['forms']['ACRDIN07']), ('ACRDIN06', zb['forms']['ACRDIN06'])]
    owner, lab, E = F['owner'], F['lab'], F['E']
    assert len(pool) == len(owner) and all(p[0] == o for p, o in zip(pool, owner)), 'pool differs from the clustering'
    parents, meta = [], []
    for fam in sorted(set(lab.tolist())):
        rows = np.nonzero((lab == fam) & (owner != 'doubled'))[0]
        if len(rows):
            i = int(rows[np.argmin(E[rows])])
            parents.append(pool[i][1])
            meta.append(dict(family=int(fam), source=str(owner[i]), parent_E=float(E[i])))
    ok = latent_representable(parents)
    print(f'hops: {len(parents)} parent families, {int((~ok).sum())} not latent-representable (no kicks)')
    parents = [p for p, k in zip(parents, ok) if k]
    meta = [m for m, k in zip(meta, ok) if k]
    copies = int(np.ceil(4000 / (2 * len(parents))))
    save('hops.pth', *kicked(parents, meta, [(-1.0, copies), (-0.5, copies)], 'hops'))


def make_dblkick():
    dbl = torch.load(DOUBLED, weights_only=False)
    meta = [dict(family=int(getattr(c, 'zp1_family', -1)), source=str(getattr(c, 'zp1_source', 'doubled')),
                 parent_E=float(c.mace)) for c in dbl]
    ok = latent_representable(dbl)
    print(f'dblkick: {len(dbl)} doubled families, {int((~ok).sum())} not latent-representable (unkicked only)')
    p0, h0, i0 = kicked(dbl, meta, [(None, 1)], 'dblkick')
    kp = [c for c, k in zip(dbl, ok) if k]
    km = [m for m, k in zip(meta, ok) if k]
    p1, h1, i1 = kicked(kp, km, [(-2.0, 11), (-1.5, 11), (-1.0, 11)], 'dblkick')
    save('doubled_kicked.pth', torch.cat([p0, p1]), torch.cat([h0, h1]), i0 + i1)


def make_elj():
    from energy_sampling.eval.nikos_comparison.summarize_search import physical  # gfn_diffusion on PYTHONPATH
    raw = torch.load(ELJ_PRE, weights_only=False)
    phys, _ = physical(raw)
    e = np.array([float(c.elj) for c in phys])
    key = [(round(float(ei), 4), tuple(np.round(c.cell_lengths.reshape(-1).double().numpy(), 3))) for ei, c in zip(e, phys)]
    seen, keep = set(), []
    for i, k in enumerate(key):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    keep = np.array(keep)
    sel = keep[np.argsort(e[keep])][:len(keep) // 2]
    print(f'elj: {len(raw)} eLJ end states, {len(phys)} physical, {len(keep)} distinct; lowest half {len(sel)} '
          f'(eLJ {e[sel].min():.1f} .. {e[sel].max():.1f}; median of all distinct {np.median(e[keep]):.1f})')
    b = collate_data_list([strip(phys[int(i)]) for i in sel])
    b.canonicalize_orientation()
    info = [dict(source=os.path.basename(ELJ_PRE), index=int(i), elj=float(e[i]), rank=r) for r, i in enumerate(sel)]
    save('elj_starts.pth', b.full_cell_parameters().detach().float(), b.aunit_handedness.detach().float().reshape(-1, 2),
         info)


if __name__ == '__main__':
    todo = sys.argv[1:] or ['hops', 'dblkick', 'elj']
    torch.set_num_threads(6)
    os.makedirs(SEEDS, exist_ok=True)
    for what in todo:
        {'hops': make_hops, 'dblkick': make_dblkick, 'elj': make_elj}[what]()
