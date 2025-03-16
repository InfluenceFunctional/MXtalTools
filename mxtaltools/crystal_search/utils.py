from math import ceil

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat_parallel, compute_rdf_distmat


def rdf_clustering(packing_coeff, rdf, rdf_cutoff, rr, samples, vdw, num_cpus=None):
    'cluster samples according to rdf distances'
    #rdf_dists = compute_rdf_distmat(rdf, rr)
    if num_cpus is not None:
        rdf_dists = compute_rdf_distmat_parallel(rdf, rr, num_cpus, 50)
    else:
        rdf_dists = compute_rdf_distmat(rdf, rr)

    good_inds, n_clusters, cluster_inds = agglomerative_cluster(vdw,
                                                                threshold=rdf_cutoff,
                                                                dists=rdf_dists)

    vdw, samples, rdf, packing_coeff = (
        vdw[good_inds], samples[good_inds], rdf[good_inds], packing_coeff[good_inds])

    print(f'RDF clustering found {n_clusters} clusters for {len(rdf_dists)} samples')
    updated_distmat = rdf_dists[good_inds]
    updated_distmat = updated_distmat[:, good_inds]
    return packing_coeff, rdf, samples, vdw, updated_distmat


def cell_clustering(cell_params_cutoff, packing_coeff, rdf, samples, vdw, norm_factors, max_iters=10):
    # cluster samples according to cell parameters
    init_num_samples = len(samples)
    # iterate over a random subsample so we don't have issues with OOM
    chunk_size = 10000
    converged = False
    iteration = 0
    while not converged and iteration < max_iters:
        if chunk_size > len(samples):
            converged = True
        num_samples = len(samples)

        inds_to_keep = []
        num_chunks = ceil(num_samples / chunk_size)
        randoms = np.random.choice(num_samples, size=num_samples, replace=False)
        for ind in range(num_chunks):  # subsample and cluster
            chunk_inds = randoms[ind * chunk_size:(ind + 1) * chunk_size]
            dists = torch.cdist(samples[chunk_inds]/ (norm_factors + 1e-4)[None, :], samples[chunk_inds]/(norm_factors + 1e-4)[None, :])
            good_inds, n_clusters, cluster_inds = agglomerative_cluster(
                vdw[chunk_inds],
                threshold=cell_params_cutoff,
                dists=dists)
            inds_to_keep.extend(chunk_inds[good_inds])

        good_inds = np.array(inds_to_keep)
        print(f'Cell clustering eliminated {len(samples) - len(good_inds)} samples')

        vdw, samples, rdf, packing_coeff = (
            vdw[good_inds], samples[good_inds], rdf[good_inds], packing_coeff[good_inds])

        iteration += 1

    print(f'Cell clustering found {len(samples)} clusters for {init_num_samples} samples')
    return packing_coeff, rdf, samples, vdw


def coarse_filter(packing_coeff, packing_coeff_range, rdf, samples, vdw, vdw_threshold):
    # filtering
    init_len = len(vdw)
    bad_inds, good_inds = coarse_crystal_filter(
        vdw, vdw_threshold, packing_coeff, packing_coeff_range)
    vdw, samples, rdf, packing_coeff = (
        vdw[good_inds], samples[good_inds], rdf[good_inds], packing_coeff[good_inds])
    return init_len, packing_coeff, rdf, samples, vdw


def get_topk_samples(packing_coeff, rdf, samples, vdw, k):
    good_inds = torch.argsort(vdw).flatten()[:k]
    vdw, samples, rdf, packing_coeff = (
        vdw[good_inds], samples[good_inds], rdf[good_inds], packing_coeff[good_inds])
    return packing_coeff, rdf, samples, vdw


def agglomerative_cluster(sample_score, threshold, dists=None, samples=None):
    # first, check if any samples are closer than the cutoff

    if dists is not None:
        if torch.sum(dists.flatten() < threshold) == len(dists.flatten()):
            n_clusters = len(dists)
            classes = torch.arange(n_clusters)
        else:
            model = AgglomerativeClustering(distance_threshold=threshold,
                                            linkage="average",
                                            affinity='precomputed',
                                            n_clusters=None)
            model = model.fit(dists.numpy())
            n_clusters = model.n_clusters_
            classes = model.labels_
    elif dists is None and samples is not None:
        model = AgglomerativeClustering(distance_threshold=threshold,
                                        linkage="average",
                                        affinity='euclidean',
                                        n_clusters=None)
        model = model.fit(samples.numpy())
        n_clusters = model.n_clusters_
        classes = model.labels_
    else:
        assert False, "Need either samples or distances"
    # select representative samples from each class
    if n_clusters < len(dists):
        unique_classes, num_uniques = np.unique(classes, return_counts=True)
        good_inds = []
        for group, uniques in zip(unique_classes, num_uniques):
            if uniques == 1:  # only one sample
                good_inds.append(int(np.argwhere(classes == group)[0]))
            else:
                class_inds = np.where(classes == group)[0]
                best_sample = np.argmin(sample_score[class_inds])
                good_inds.append(class_inds[best_sample])
    else:
        good_inds = torch.arange(len(sample_score))

    return torch.LongTensor(good_inds), n_clusters, classes


def coarse_crystal_filter(lj_record, lj_cutoff, packing_coeff_record, packing_cutoff):
    """filtering - samples with exactly 0 LJ energy are too diffuse, and more than CUTOFF are overlapping"""
    bad_inds = []
    bad_bools1 = lj_record == 0
    bad_bools2 = lj_record >= lj_cutoff
    bad_bools3 = packing_coeff_record >= packing_cutoff[-1]
    bad_bools4 = packing_coeff_record <= packing_cutoff[0]

    # if we got any of these, cut the sample
    good_bools = (~bad_bools1) * (~bad_bools2) * (~bad_bools3) * (~bad_bools4)
    good_inds = torch.argwhere(good_bools).flatten()

    print(f"{bad_bools1.sum()} with zero vdW, "
          f"{bad_bools2.sum()} above vdW cutoff, "
          f"{bad_bools3.sum()} outside density cutoff,"
          f"leaving {len(good_inds)} samples")

    return bad_inds, good_inds
