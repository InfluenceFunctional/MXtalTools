import torch
from mxtaltools.common.utils import torch_ptp


def parallel_compute_rdf_torch(dists_list: torch.tensor, raw_density=True, rrange=None, bins=None, remove_radial_scaling=False):
    """
    Compute the radial distribution given a list of distances with parallel execution for speed.

    Parameters
    ----------
    dists_list : list of torch tensors
    raw_density : bool
        If true, use uniform density of 1 everywhere, else estimate the density from the dist list.
    rrange : [min_range, max_range], optional
    bins : int
    remove_radial_scaling : removes inverse scaling in RDF, giving something more like the unnormalized radial CDF.

    Returns
    -------
    rdf : torch.tensor(n, n_bins)
    bin_edges : torch.tensor(n_bins + 1)
    """

    hist_range = [0.5, 10] if rrange is None else rrange
    hist_bins = 100 if bins is None else bins

    if not raw_density:  # estimate the density from the distances
        rdf_density = torch.zeros(len(dists_list)).to(dists_list[0].device)
        for i in range(len(dists_list)):
            dists = dists_list[i]
            try:
                sorted_dists = torch.sort(dists)[0][:len(dists) // 2]  # we will use 1/2 the dist radius to avoid edge / cutoff effects
                volume = 4 / 3 * torch.pi * torch_ptp(sorted_dists) ** 3  # volume of a sphere #np.ptp(sorted_dists[:, 0]) * np.ptp(sorted_dists[:, 1]) * np.ptp(sorted_dists[:, 2])
                # number of particles divided by the volume
                rdf_density[i] = len(sorted_dists) / volume
            except:
                rdf_density[i] = 1
    else:
        rdf_density = torch.ones(len(dists_list), device=dists_list[0].device, dtype=torch.float32)

    hh_list = torch.stack([torch.histc(dists, min=hist_range[0], max=hist_range[1], bins=hist_bins) for dists in dists_list])
    rr = torch.linspace(hist_range[0], hist_range[1], hist_bins + 1, device=hh_list.device)
    if remove_radial_scaling:
        rdf = hh_list / rdf_density[:, None]  # un-smoothed radial density
    else:
        shell_volumes = (4 / 3) * torch.pi * ((rr[:-1] + torch.diff(rr)) ** 3 - rr[:-1] ** 3)  # volume of the shell at radius r+dr
        rdf = hh_list / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density

    return rdf, (rr[:-1] + torch.diff(rr)).requires_grad_()  # rdf and x-axis
