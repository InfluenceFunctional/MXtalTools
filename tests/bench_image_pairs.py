"""
Named benchmark: `crystal_building.image_pairs` against `mol2cluster` + `construct_radial_graph`.

    python tests/bench_image_pairs.py [dataset.pt] [--key prior] [--batch 1000] [--device cpu]
                                      [--cutoff 10] [--reps 5] [--vram-frac 0.4]

`dataset.pt` is either a list of single crystals (the CSD sets; Z' = 1 rows are used) or a
dict holding a collated batch under `--key` (GFN prior files). It defaults to this suite's
`datasets/mini_new_csd.pt`. Stages are timed separately, each as the median of `--reps`
calls after one warm-up, holding one result at a time: for the existing route the cluster
build, the radius search and the eLJ sum; for the closed form the image-centre selection,
the pair distances and the same eLJ sum. Not collected by pytest (the name does not start
with `test_`). Correctness is `tests/test_image_pairs.py`; this file measures only.

On a shared GPU the numbers are contended: compare the two routes within one run, not
across runs.
"""
import argparse
import os
import time

import torch

from mxtaltools.analysis.vdw_analysis import elj_analysis
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.image_pairs import build_image_tables, select_images, pair_distances
from mxtaltools.dataset_utils.utils import collate_data_list


def load(path, key, n, seed=0):
    d = torch.load(path, weights_only=False, map_location='cpu')
    g = torch.Generator().manual_seed(seed)
    if isinstance(d, list):
        rows = [r for r in d if int(r.z_prime) == 1]
        idx = torch.randperm(len(rows), generator=g)[:n]
        return collate_data_list([rows[i].clone() for i in idx.tolist()])
    batch = d[key] if isinstance(d, dict) else d
    # batch_to_list on a whole stored prior copies shared tensors into every row; subset the batch itself
    return batch.subsample_new_batch(torch.randperm(batch.num_graphs, generator=g)[:n])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('dataset', nargs='?', default=os.path.join(here, 'datasets', 'mini_new_csd.pt'))
    ap.add_argument('--key', default='prior')
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--cutoff', type=float, default=10.0)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--vram-frac', type=float, default=0.4)
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)

    def sync():
        if dev.type == 'cuda':
            torch.cuda.synchronize()

    def med(fn):
        fn()
        out, ts = None, []
        for _ in range(a.reps):
            out = None
            sync()
            t = time.perf_counter()
            out = fn()
            sync()
            ts.append(time.perf_counter() - t)
        ts.sort()
        return out, ts[len(ts) // 2] * 1e3

    batch = load(a.dataset, a.key, a.batch).to(dev)
    n = batch.num_graphs
    vdw = torch.tensor(list(VDW_RADII.values()), device=dev)
    geom = (batch.T_fc, batch.T_cf, batch.aunit_centroid, batch.aunit_orientation)
    with torch.no_grad():
        cluster, t_cluster = med(lambda: batch.mol2cluster(cutoff=a.cutoff, supercell_size=10, std_orientation=False))
        _, t_search = med(lambda: cluster.construct_radial_graph(cutoff=a.cutoff))
        _, t_elj_old = med(lambda: cluster.compute_eLJ_energy())
        pairs_old = cluster.edges_dict['intermolecular_dist'].numel() / n
        at_cap = int((torch.bincount(cluster.edges_dict['edge_index_inter'][1]) >= 1001).sum())
        atoms = cluster.pos.shape[0] / n
        del cluster

        tables = build_image_tables(batch)
        sel, t_select = med(lambda: select_images(tables, *geom, a.cutoff))
        pr, t_pairs = med(lambda: pair_distances(tables, sel, a.cutoff))
        _, t_elj_new = med(lambda: elj_analysis(
            vdw, {'intermolecular_dist': pr['dist'], 'intermolecular_dist_atoms': [pr['z_src'], pr['z_tgt']],
                  'intermolecular_dist_batch': pr['graph']}, n, stiffness=2.5, envelope=None))
        pairs_new = pr['dist'].numel() / n

    t_old, t_new = t_cluster + t_search + t_elj_old, t_select + t_pairs + t_elj_new
    print(f"Table. Wall time per batch of {n} stored crystals ({os.path.basename(a.dataset)}, "
          f"{float(batch.num_atoms.float().mean()):.1f} atoms/molecule), {dev.type}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == 'cuda' else f" ({torch.get_num_threads()} threads)")
          + f", pair cutoff {a.cutoff:g} A, median of {a.reps} calls per stage after a warm-up. The existing route "
          f"wrote out {atoms:.0f} atoms per crystal and kept {pairs_old:.0f} pairs per crystal "
          f"({at_cap} reference atoms at its 1000-neighbour cap); the closed form kept {pairs_new:.0f} pairs per "
          f"crystal from {float(sel['images_kept'].float().mean()):.0f} image molecules "
          f"({int(sel['capped'].sum())} crystals with a clamped translation window).")
    print("stage | existing route (ms/batch) | closed form (ms/batch)")
    print(f"instantiation (cluster build / image-centre selection) | {t_cluster:.1f} | {t_select:.1f}")
    print(f"neighbours and distances | {t_search:.1f} | {t_pairs:.1f}")
    print(f"eLJ sum over the pairs | {t_elj_old:.1f} | {t_elj_new:.1f}")
    print(f"total | {t_old:.1f} | {t_new:.1f}")


if __name__ == '__main__':
    main()
