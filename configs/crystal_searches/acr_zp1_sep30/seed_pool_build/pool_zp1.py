"""Pool every earlier acridine sg14 Z'=1 search end state into compact arrays (read-only on the sources):
  may   D:/crystal_datasets/acridine/prior_chunks/may_acridine_sg14_zp1_*.pt (acr_production, May, old MACE)
  dec   D:/crystal_datasets/acridine/acridine_sg14_zp1.pt (December production, if it is sg14 Z'=1 end states)
Per state: params [12] (cell lengths, angles, aunit centroid, aunit orientation), hand [1], the stored energy under the
source's model (mace if present), lj, packing coefficient, angular factor; source and row index. Writes pool.pt."""
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import psutil  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

HERE = os.environ.get('POOL_DIR', os.path.dirname(os.path.abspath(__file__)))
A = 'D:/crystal_datasets/acridine/'


def compact(rows, src):
    b = collate_data_list(rows, exclude_keys=['fingerprint', 'rdf'])
    n = b.num_graphs
    assert (b.sg_ind == 14).all() and (b.z_prime == 1).all(), src
    get = lambda k: getattr(b, k).double().flatten() if k in b.keys() else torch.full((n,), float('nan'), dtype=torch.float64)
    ang = b.cell_volume.flatten() / b.cell_lengths.prod(-1)
    return dict(params=torch.cat([b.cell_lengths, b.cell_angles, b.aunit_centroid[:, :3], b.aunit_orientation[:, :3]],
                                 1).float(),
                hand=b.aunit_handedness.float().reshape(n, -1)[:, :1], mace=get('mace'), lj=get('lj'), elj=get('elj'),
                cp=get('packing_coeff'), ang=ang.double(), src=[src] * n, n_atoms0=int(b.z[0]))


out = []
for f in sorted(glob.glob(A + 'prior_chunks/may_acridine_sg14_zp1_*.pt')):
    rows = torch.load(f, weights_only=False)
    if not rows:
        continue
    c = compact(rows, 'may:' + os.path.basename(f))
    out.append(c)
    print(f'{os.path.basename(f)}: {len(rows)} rows; RSS {psutil.Process().memory_info().rss / 2 ** 30:.2f} GB', flush=True)
    del rows
print('dec file ...', flush=True)
d = torch.load(A + 'acridine_sg14_zp1.pt', weights_only=False)
print('dec type', type(d).__name__, (list(d.keys()) if isinstance(d, dict) else len(d)),
      f'RSS {psutil.Process().memory_info().rss / 2 ** 30:.2f} GB', flush=True)
rows = d if isinstance(d, list) else (d.batch_to_list() if hasattr(d, 'batch_to_list') else None)
if rows is not None:
    for k in range(0, len(rows), 2000):
        out.append(compact(rows[k:k + 2000], 'dec'))
del d, rows
pool = {k: (torch.cat([o[k] for o in out]) if torch.is_tensor(out[0][k]) else sum((o[k] for o in out), []))
        for k in ('params', 'hand', 'mace', 'lj', 'elj', 'cp', 'ang', 'src')}
torch.save(pool, os.path.join(HERE, 'pool.pt'))
for s in ('may', 'dec'):
    m = torch.tensor([x.startswith(s) for x in pool['src']])
    if m.any():
        e = pool['mace'][m]
        print(f'{s}: {int(m.sum())} states; mace finite {int(torch.isfinite(e).sum())}, min {e.nanmin():.2f}, '
              f'within 6/10/15 kT (2.494) of its min: {[int((e <= e.nanmin() + w * 2.494).sum()) for w in (6, 10, 15)]}')
