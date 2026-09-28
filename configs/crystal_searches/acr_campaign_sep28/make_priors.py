"""
The campaign's prior map, priors/known_map.pth: every state already known for acridine sg14 Z'=2 under the old MACE
checkpoint (acr_112025_mh1_stagetwo.model) within the campaign's window, as a compact file the coordinator ingests as
stream 'prior'. LOCAL provenance script (reads local files); the cluster only reads the .pth (.pth, not .pt: *.pt is Git
LFS-tracked here).

Candidates: the aug21 unseeded band (1781 states within 2 kT of -62.812), the two relaxed forms (ACRDIN07, ACRDIN06), the
121 doubled Z'=1 structures, and every physical sep27 end state within 3 kT (acr_proposals_sep27 and acr_finish_sep27
outputs), energies as the searches stored them (kJ/mol per molecule). Every physical candidate within the window is
shipped, in the order the coordinator ingests them, so the campaign's first curate pass rebuilds exactly the basins of
a temporary campaign run here with the same cut and window (leader clustering depends on which state arrives first: one
representative per basin, re-clustered, merged some basins and left some known states outside every prior basin). The
build's basin counts are stored in the file and printed. The first pass on the cluster computes all their RDFs (about
10 min on CPU, inside one job).

    python make_priors.py [CUT OUT]   # ~20 min on CPU (RDFs of the candidates); default: make_campaign's cut and
                                      # priors/known_map.pth
"""
import os
import shutil
import sys
from argparse import Namespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_campaign import COORD  # noqa: E402  (identity cut, window, reference: the campaign's own)
from mxtaltools.crystal_search import coordinator as co  # noqa: E402
from mxtaltools.dataset_utils.utils import collate_data_list  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = 'D:/crystal_datasets/acr_112025_mh1_stagetwo.model'
MOL = 'D:/crystal_datasets/acridine/opt_acridine_conformer.pt'
Q = 'C:/Users/mikem/AppData/Local/Temp/claude/C--Users-mikem-Projects-mxt-gfn-gfn-diffusion-energy-sampling/688e73d7-8d05-4713-b822-09124ad10a11/scratchpad/acridine/q_double'
ARMS = 'C:/Users/mikem/AppData/Local/Temp/claude/C--Users-mikem-Projects-mxt-gfn-gfn-diffusion-energy-sampling/9d87e030-cb0d-4ca3-b966-a69c6aa4723b/scratchpad/sep27/arms27.pt'
DOUBLED = 'D:/crystal_datasets/acridine/doubled_zp1_sg14_2026-09-26.pt'
WORK = 'D:/crystal_datasets/acridine/campaigns/acr_campaign_sep28_priors_build'
KT, FLOOR = 2.494, -62.812


def compact(crystals, energies, source):
    b = collate_data_list([c.clone() for c in crystals], exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    return (b.full_cell_parameters().detach().float(), b.aunit_handedness.detach().float().reshape(len(crystals), -1),
            torch.as_tensor(energies, dtype=torch.float32), [source] * len(crystals))


def candidates():
    parts = []
    zb = torch.load(os.path.join(Q, 'zp2_band.pt'), weights_only=False)
    parts.append(compact(zb['band'], np.asarray(zb['band_E'], float), 'aug21_band'))
    forms = ['ACRDIN07', 'ACRDIN06']
    parts.append(compact([zb['forms'][f] for f in forms], [zb['forms_E'][f] for f in forms], 'forms'))
    dbl = torch.load(DOUBLED, weights_only=False)
    parts.append(compact(dbl, [float(c.mace) for c in dbl], 'doubled_zp1'))
    A = torch.load(ARMS, weights_only=False)
    for arm, a in A.items():
        keep = np.nonzero(a['phys'] & (a['E'] <= FLOOR + 3 * KT))[0]
        parts.append((torch.as_tensor(a['params'][keep]).float(), torch.as_tensor(a['hand'][keep]).float(),
                      torch.as_tensor(a['E'][keep]).float(), [f'sep27:{arm}'] * len(keep)))
    return (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]),
            sum((p[3] for p in parts), []))


def main():
    cut = float(sys.argv[1]) if len(sys.argv) > 1 else COORD['identity_cut']
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'priors', 'known_map.pth')
    work = f'{WORK}_{cut:g}'
    mid = co.energy_model_id({'optim_target': 'mace'}, Namespace(mace_predictor_path=MODEL))
    params, hand, energy, source = candidates()
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    cand = os.path.join(work, 'candidates.pth')
    keep = co.physical(params, torch.zeros(len(energy))) & (energy <= COORD['energy_ref'] + COORD['window_kT'] * KT)
    idx = torch.nonzero(keep).flatten()  # the rows the coordinator would admit, in their order
    params, hand, energy = params[idx], hand[idx], energy[idx]
    source = [source[i] for i in idx.tolist()]
    torch.save(dict(params=params, handedness=hand, energy=energy, lj=torch.zeros(len(energy)), source=source,
                    energy_model_id=mid), cand)
    cfg = dict(COORD, identity_cut=cut, mol_path=MOL, energy_model_id=mid, priors=[cand], streams={}, hops=None)
    yaml.safe_dump(cfg, open(os.path.join(work, 'coord.yaml'), 'w'))
    co.curate(work)
    reg = torch.load(os.path.join(work, 'registry.pt'), weights_only=False)
    E = np.asarray(reg['basin_E'], dtype=float)
    basins = {'all': len(E), '2kT': int((E <= COORD['energy_ref'] + 2 * KT).sum()),
              '1kT': int((E <= COORD['energy_ref'] + KT).sum())}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(dict(params=params, handedness=hand, energy=energy, lj=torch.zeros(len(energy)), source=source,
                    energy_model_id=mid, identity_cut=cut, build_basins=basins,
                    provenance=f'{len(energy)} states ({ {s: source.count(s) for s in dict.fromkeys(source)} }) within '
                               f'{COORD["window_kT"]} kT; {basins} basins at identity cut {cut}'),
               out)
    print(f'{out}: {len(energy)} states, {basins} basins at cut {cut}; model {mid}; {os.path.getsize(out) / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
