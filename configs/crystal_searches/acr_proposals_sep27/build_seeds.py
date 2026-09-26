"""
Rebuild a compact seed file into the MolCrystalData list that run_search's `init_sample_method: data` reads.

Compact file (torch.save dict): params [N, 18] float (cell lengths, angles in rad, 2 x 3 centroids, 2 x 3 rotvecs),
handedness [N, 2], meta (list of dicts, provenance). Crystals are built the way init_samples_to_optim builds sg14
Z'=2 search crystals -- the conformer at mol_path duplicated -- so a seed file of a few hundred KB can be committed
and expanded on the cluster instead of shipping ~10 KB per MolCrystalData.

    python build_seeds.py <compact.pt> <mol_path> <out.pt>
"""
import sys

import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData


def build(compact_path, mol_path, out_path=None):
    blob = torch.load(compact_path, weights_only=False)
    params = torch.as_tensor(blob['params'], dtype=torch.float32)
    hand = torch.as_tensor(blob['handedness'], dtype=torch.float32)
    if params.ndim != 2 or params.shape[1] != 18 or hand.shape != (len(params), 2):
        raise ValueError(f"expected params [N, 18] and handedness [N, 2], got {tuple(params.shape)} {tuple(hand.shape)}")
    mol = torch.load(mol_path, weights_only=False)
    mol = mol[0] if isinstance(mol, list) else mol
    out = []
    for p, h in zip(params, hand):
        out.append(MolCrystalData(
            molecule=[mol.clone(), mol.clone()], sg_ind=14, z_prime=2, max_z_prime=2,
            cell_lengths=p[:3].clone(), cell_angles=p[3:6].clone(), aunit_centroid=p[6:12].clone(),
            aunit_orientation=p[12:18].clone(), aunit_handedness=h.clone(), do_box_analysis=True))
    if out_path is not None:
        torch.save(out, out_path)
    return out


if __name__ == '__main__':
    crystals = build(*sys.argv[1:4])
    print(f'built {len(crystals)} sg14 Z\'=2 seeds -> {sys.argv[3]}')
