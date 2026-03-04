from mxtaltools.common.geometry_utils import simple_latent_distance
import torch
from tqdm import tqdm


def greedy_bottom_up_anchors(params, cps, ens, d_cut, e_cut):
    valid_density = (cps > 0.55) & (cps < 0.95)
    valid_energy = (ens <= e_cut)
    valid = valid_density & valid_energy

    en_sort_inds = torch.argsort(ens, descending=False).flatten()
    anchor_tensor = params[en_sort_inds[0:1]]
    anchors = [en_sort_inds[0].item()]
    for ind in tqdm(en_sort_inds):
        if not valid[ind]:
            continue
        sample = params[ind:ind + 1]
        diff = simple_latent_distance(anchor_tensor, sample)
        keep = (diff > d_cut).all()

        if keep:
            anchors.append(ind.item())
            anchor_tensor = torch.cat([anchor_tensor, sample], dim=0)

    anchors = torch.tensor(anchors, dtype=torch.long, device=params.device)
    return anchors
