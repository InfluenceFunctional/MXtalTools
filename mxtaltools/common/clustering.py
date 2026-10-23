from mxtaltools.common.geometry_utils import simple_latent_distance
import torch
from tqdm import tqdm


def greedy_bottom_up_anchors(params, cps, ens, d_cut, e_cut = None):
    """
    Greedily select mutually distant samples ("anchors"), visiting samples in ascending energy.

    A sample is eligible if its packing coefficient is strictly between 0.55 and 0.95 and, when ``e_cut`` is given,
    its energy is ``<= e_cut``. An eligible sample becomes an anchor if its distance to every anchor chosen so far is
    greater than ``d_cut``. Distances are ``mxtaltools.common.geometry_utils.simple_latent_distance``: Euclidean, with
    the dimensions that function treats as periodic wrapped into [-1, 1). The lowest-energy sample is always the
    first anchor, whether or not it is eligible. A tqdm progress bar is shown.

    Parameters
    ----------
    params : torch.Tensor
        [num_samples, num_params] latent parameters, in the layout ``simple_latent_distance`` expects.
    cps : torch.Tensor
        [num_samples] packing coefficients.
    ens : torch.Tensor
        [num_samples] energies.
    d_cut : float
        Distance that a new anchor must exceed to every existing anchor.
    e_cut : float or torch.Tensor, optional
        Inclusive energy ceiling for eligibility; None applies no energy filter.

    Returns
    -------
    anchors : torch.Tensor
        [num_anchors] long indices into ``params``, in ascending energy, on ``params.device``.
    """
    valid_density = (cps > 0.55) & (cps < 0.95)
    if e_cut is not None:
        valid_energy = (ens <= e_cut)
    else:
        valid_energy = torch.zeros_like(valid_density).fill_(True)
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



def greedy_bottom_up_anchors2(params, cps, ens, d_cut, e_cut = None):
    """
    Greedily select mutually distant samples ("anchors") as ``greedy_bottom_up_anchors`` does, but with plain
    Euclidean distance (``torch.cdist``, no periodic wrapping).

    Samples are visited in ascending energy. A sample is eligible if its packing coefficient is strictly between 0.55
    and 0.95 and, when ``e_cut`` is given, its energy is ``<= e_cut``; it becomes an anchor if its distance to every
    anchor chosen so far is greater than ``d_cut``. The lowest-energy sample is always the first anchor, whether or not
    it is eligible. A tqdm progress bar is shown.

    Parameters
    ----------
    params : torch.Tensor
        [num_samples, num_params] coordinates the distance is computed on.
    cps : torch.Tensor
        [num_samples] packing coefficients.
    ens : torch.Tensor
        [num_samples] energies.
    d_cut : float
        Distance that a new anchor must exceed to every existing anchor.
    e_cut : float or torch.Tensor, optional
        Inclusive energy ceiling for eligibility; None applies no energy filter.

    Returns
    -------
    anchors : torch.Tensor
        [num_anchors] long indices into ``params``, in ascending energy, on ``params.device``.
    """
    valid_density = (cps > 0.55) & (cps < 0.95)
    if e_cut is not None:
        valid_energy = (ens <= e_cut)
    else:
        valid_energy = torch.zeros_like(valid_density).fill_(True)
    valid = valid_density & valid_energy

    en_sort_inds = torch.argsort(ens, descending=False).flatten()
    anchor_tensor = params[en_sort_inds[0:1]]
    anchors = [en_sort_inds[0].item()]
    for ind in tqdm(en_sort_inds):
        if not valid[ind]:
            continue
        sample = params[ind:ind + 1]
        diff = torch.cdist(anchor_tensor, sample)
        keep = (diff > d_cut).all()

        if keep:
            anchors.append(ind.item())
            anchor_tensor = torch.cat([anchor_tensor, sample], dim=0)

    anchors = torch.tensor(anchors, dtype=torch.long, device=params.device)
    return anchors