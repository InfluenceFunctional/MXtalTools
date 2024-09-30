import torch
from tqdm import tqdm

from mxtaltools.common.utils import torch_ptp, compute_rdf_distance


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


def compute_rdf_distmat(rdf_record, rr, show_tqdm=True, chunk_size=250):
    # todo parallelize this function via async
    rdf_dists = torch.zeros(rdf_record.shape[0], rdf_record.shape[0])

    for i in tqdm(range(1, len(rdf_record)), disable=not show_tqdm):
        num_chunks = i // chunk_size + 1
        for j in range(num_chunks):
            start_ind = j * chunk_size
            stop_ind = min(i, (j + 1) * chunk_size)
            rdf_dists[i, start_ind:stop_ind] = compute_rdf_distance(
                rdf_record[i],
                rdf_record[start_ind:stop_ind],  # save on energy & memory
                rr,
                n_parallel_rdf2=stop_ind - start_ind)
    rdf_dists = rdf_dists + rdf_dists.T  # symmetric distance matrix
    rdf_dists = torch.log10(1 + rdf_dists)
    return rdf_dists


def compute_rdf_distmat_block(rdf_record: torch.Tensor,
                              rr,
                              i_range,
                              j_range) -> torch.Tensor:

    device = rdf_record.device

    num_rdfs = (j_range[1] - j_range[0]) * (i_range[1] - i_range[0])
    i_inds = torch.arange(i_range[0], min(i_range[1], num_rdfs), device=device)
    j_inds = torch.arange(j_range[0], min(j_range[1], num_rdfs), device=device)

    mg = torch.meshgrid(i_inds, j_inds)
    ind1, ind2 = mg[0].flatten(), mg[1].flatten()
    good_bools = ind1 > ind2
    ind1_g = ind1[good_bools]
    ind2_g = ind2[good_bools]
    dists = compute_rdf_distance(
        rdf_record[ind1_g],
        rdf_record[ind2_g],
        rr,
    )
    distmat = torch.zeros((i_range[1]-i_range[0], j_range[1]-j_range[0]), dtype=torch.float32, device=device)
    distmat[ind1_g - i_range[1], ind2_g - j_range[1]] = dists

    return distmat


def compute_rdf_distmat_parallel(rdf_record, rr, num_cpus, chunk_size=250):

    import multiprocessing as mp
    from math import ceil

    pool = mp.Pool(num_cpus)
    num_rdfs = len(rdf_record)
    num_chunks = ceil(len(rdf_record) / chunk_size)

    out = []
    for i in range(num_chunks):
        for j in range(num_chunks):
            out.append(
                pool.apply_async(
                    compute_rdf_distmat_block,
                    (rdf_record,
                     rr,
                     [i * chunk_size, min((i + 1) * chunk_size, num_rdfs)],
                     [j * chunk_size, min((j + 1) * chunk_size, num_rdfs)],
                     )
                )
            )
    pool.close()
    pool.join()
    out = [thing.get() for thing in out]

    rdf_dists = torch.zeros(rdf_record.shape[0], rdf_record.shape[0])
    ind = 0
    for i in range(num_chunks):
        for j in range(num_chunks):
            rdf_dists[i * chunk_size: min((i + 1) * chunk_size, num_rdfs),
                     j * chunk_size: min((j + 1) * chunk_size, num_rdfs)] = out[ind]
            ind += 1

    rdf_dists = rdf_dists + rdf_dists.T  # symmetric distance matrix
    rdf_dists = torch.log10(1 + rdf_dists)
    return rdf_dists
