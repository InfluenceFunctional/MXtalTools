import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from common.utils import torch_ptp


def single_point_compute_rdf(dists, density=None, range=None, bins=None, sigma=None):
    '''
    compute the radial distribution for a single particle
    dists: array of pairwise distances of nearby particles from the reference
    '''
    hist_range = [0.5, 10] if range is None else range
    hist_bins = 100 if bins is None else bins
    gauss_sigma = 1 if sigma is None else sigma

    if density is None:  # estimate the density from the distances
        sorted_dists = np.sort(dists)[:len(dists) // 2]  # we will use 1/2 the dist radius to avoid edge / cutoff effects
        volume = 4 / 3 * np.pi * np.ptp(sorted_dists) ** 3  # volume of a sphere #np.ptp(sorted_dists[:, 0]) * np.ptp(sorted_dists[:, 1]) * np.ptp(sorted_dists[:, 2])
        rdf_density = len(sorted_dists) / volume  # number of particles divided by the volume
    else:
        rdf_density = density

    hh, rr = np.histogram(dists, range=hist_range, bins=hist_bins)
    shell_volumes = (4 / 3) * np.pi * ((rr[:-1] + np.diff(rr)) ** 3 - rr[:-1] ** 3)  # volume of the shell at radius r+dr
    rdf = gaussian_filter1d(hh / shell_volumes / rdf_density, sigma=gauss_sigma)  # radial distribution function

    return rdf, rr[:-1] + np.diff(rr)  # rdf and x-axis


def single_point_compute_rdf_torch(dists, density=None, range=None, bins=None):
    '''
    compute the radial distribution for a single particle
    dists: array of pairwise distances of nearby particles from the reference
    '''
    hist_range = [0.5, 10] if range is None else range
    hist_bins = 100 if bins is None else bins

    if density is None:  # estimate the density from the distances
        sorted_dists = torch.sort(dists)[0][:len(dists) // 2]  # we will use 1/2 the dist radius to avoid edge / cutoff effects
        volume = 4 / 3 * torch.pi * torch_ptp(sorted_dists) ** 3  # volume of a sphere #np.ptp(sorted_dists[:, 0]) * np.ptp(sorted_dists[:, 1]) * np.ptp(sorted_dists[:, 2])
        rdf_density = len(sorted_dists) / volume  # number of particles divided by the volume
    else:
        rdf_density = density

    hh = torch.histc(dists, min=hist_range[0], max=hist_range[1], bins=hist_bins)
    rr = torch.linspace(hist_range[0], hist_range[1], hist_bins + 1).to(hh.device)
    shell_volumes = (4 / 3) * torch.pi * ((rr[:-1] + torch.diff(rr)) ** 3 - rr[:-1] ** 3)  # volume of the shell at radius r+dr
    rdf = hh / shell_volumes / rdf_density  # un-smoothed radial density

    return rdf, rr[:-1] + torch.diff(rr)  # rdf and x-axis


def parallel_compute_rdf_torch(dists_list, density=None, rrange=None, bins=None, remove_radial_scaling=False):
    '''
    compute the radial distribution for a single particle
    dists: array of pairwise distances of nearby particles from the reference
    '''
    hist_range = [0.5, 10] if rrange is None else rrange
    hist_bins = 100 if bins is None else bins

    if density is None:  # estimate the density from the distances
        rdf_density = torch.zeros(len(dists_list)).to(dists_list[0].device)
        for i in range(len(dists_list)):
            dists = dists_list[i]
            sorted_dists = torch.sort(dists)[0][:len(dists) // 2]  # we will use 1/2 the dist radius to avoid edge / cutoff effects
            volume = 4 / 3 * torch.pi * torch_ptp(sorted_dists) ** 3  # volume of a sphere #np.ptp(sorted_dists[:, 0]) * np.ptp(sorted_dists[:, 1]) * np.ptp(sorted_dists[:, 2])
            # number of particles divided by the volume
            rdf_density[i] = len(sorted_dists) / volume
    else:
        rdf_density = density

    hh_list = torch.stack([torch.histc(dists, min=hist_range[0], max=hist_range[1], bins=hist_bins) for dists in dists_list])
    rr = torch.linspace(hist_range[0], hist_range[1], hist_bins + 1).to(hh_list.device)
    if remove_radial_scaling:
        rdf = hh_list / rdf_density[:, None]  # un-smoothed radial density
    else:
        shell_volumes = (4 / 3) * torch.pi * ((rr[:-1] + torch.diff(rr)) ** 3 - rr[:-1] ** 3)  # volume of the shell at radius r+dr
        rdf = hh_list / shell_volumes[None, :] / rdf_density[:, None]  # un-smoothed radial density

    return rdf, (rr[:-1] + torch.diff(rr)).requires_grad_()  # rdf and x-axis


def compute_rdf_distance_old(target_rdf: np.ndarray, sample_rdf: np.ndarray):
    '''
    earth mover's distance
    assuming dimension [sample, element-pair, radius]
    normed against target rdf (sample is not strictly a PDF in this case)
    averaged over nnz elements - only works for single type of molecule per call
    OLD way of doing this
    '''

    nonzero_element_pairs = np.sum(np.sum(target_rdf, axis=1) > 0)
    target_CDF = np.cumsum(target_rdf, axis=-1)
    sample_CDF = np.cumsum(sample_rdf, axis=-1)
    norm = target_CDF[:, -1]
    target_CDF = np.nan_to_num(target_CDF / norm[:, None])
    sample_CDF = np.nan_to_num(sample_CDF / norm[None, :, None])
    emd = np.sum(np.abs(target_CDF - sample_CDF), axis=(1, 2))
    return emd / nonzero_element_pairs  # manual normalization elementwise


def compute_rdf_distance(target_rdf: torch.tensor, sample_rdf: torch.tensor, rr):
    '''
    earth mover's distance
    assuming dimension [sample, radius]
    norm both incoming rdfs to make a symmetric distance metric
    averaged over nnz elements - only works for single type of molecule per call
    '''

    nonzero_element_pairs = torch.sum(torch.sum(target_rdf + sample_rdf, axis=1) > 0)

    target_pdf = torch.nan_to_num(target_rdf / target_rdf.sum(1)[:, None])
    sample_pdf = torch.nan_to_num(sample_rdf / sample_rdf.sum(1)[:, None])

    target_cdf = torch.cumsum(target_pdf, axis=-1)
    sample_cdf = torch.cumsum(sample_pdf, axis=-1)

    # norm distance according to bin width
    emd = torch.sum(torch.abs(target_cdf - sample_cdf), axis=(0, 1)) * (rr[1] - rr[0])

    return emd / nonzero_element_pairs  # manual normalizaion elementwise
