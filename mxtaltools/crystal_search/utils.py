"""
Crystal-search helpers: the command-line parser, loading the target and the crystals to optimize, sampling initial
cell parameters, preparing per-stage optimization settings, and filtering, ranking and clustering of sampled crystals.
"""
import argparse
from math import ceil

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat_parallel, compute_rdf_distmat
from mxtaltools.common.training_utils import load_crystal_score_model
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor


def rdf_clustering(packing_coeff, rdf, rdf_cutoff, rr, samples, vdw, num_cpus=None):
    """
    Cluster samples according to RDF distances and keep the lowest-``vdw`` sample of each cluster.

    The distance matrix is ``compute_rdf_distmat_parallel(rdf, rr, num_cpus, 50)`` when ``num_cpus`` is given and
    ``compute_rdf_distmat(rdf, rr)`` otherwise; clusters come from `agglomerative_cluster` with
    ``threshold=rdf_cutoff``. Prints the number of clusters found.

    Parameters
    ----------
    packing_coeff : torch.Tensor
        Per-sample values indexed along the first dimension; filtered alongside the samples.
    rdf : torch.Tensor
        [num_samples, num_channels, num_bins] RDFs.
    rdf_cutoff : float
        Distance threshold passed to `agglomerative_cluster`.
    rr : torch.Tensor
        Radial grid of the RDFs; only its spacing ``rr[1] - rr[0]`` is used by the distance functions.
    samples : torch.Tensor
        Per-sample values indexed along the first dimension; filtered alongside.
    vdw : torch.Tensor
        [num_samples] scores; the sample with the lowest score in each cluster is kept.
    num_cpus : int, optional
        Worker processes for the parallel distance computation; None computes the matrix serially.

    Returns
    -------
    packing_coeff, rdf, samples, vdw : torch.Tensor
        The inputs restricted to the kept samples, in the order of the indices returned by `agglomerative_cluster`.
    updated_distmat : torch.Tensor
        [num_kept, num_kept] distance matrix among the kept samples.

    Notes
    -----
    ``compute_rdf_distmat_parallel`` returns ``log10(1 + d)`` and ``compute_rdf_distmat`` returns the distance ``d``
    itself, so the same ``rdf_cutoff`` is applied to differently scaled distances on the two paths.
    """
    # rdf_dists = compute_rdf_distmat(rdf, rr)
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
    """
    Cluster samples according to normalized cell parameters and keep the lowest-``vdw`` sample of each cluster.

    Each pass permutes the samples with ``np.random.choice`` (the global NumPy RNG), splits them into chunks of 10000,
    and clusters each chunk separately with `agglomerative_cluster` (``threshold=cell_params_cutoff``) on the
    ``torch.cdist`` distances between ``samples / (norm_factors + 1e-4)``. Passes repeat on the survivors until a
    pass that starts with fewer than 10000 samples (clustered as a single chunk) has run, or ``max_iters`` passes
    have run. Prints the number of samples eliminated per pass and the final count.

    Parameters
    ----------
    cell_params_cutoff : float
        Distance threshold on the normalized cell parameters.
    packing_coeff : torch.Tensor
        Per-sample values indexed along the first dimension; filtered alongside the samples.
    rdf : torch.Tensor
        Per-sample values indexed along the first dimension; filtered alongside the samples.
    samples : torch.Tensor
        [num_samples, num_params] cell parameters; must be on the CPU, since the distances are passed to
        scikit-learn through ``.numpy()``.
    vdw : torch.Tensor
        [num_samples] scores; the sample with the lowest score in each cluster is kept.
    norm_factors : torch.Tensor
        [num_params] per-column scales dividing ``samples`` before the distances are taken.
    max_iters : int
        Maximum number of passes.

    Returns
    -------
    packing_coeff, rdf, samples, vdw : torch.Tensor
        The inputs restricted to the surviving samples.
    """
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
            dists = torch.cdist(samples[chunk_inds] / (norm_factors + 1e-4)[None, :],
                                samples[chunk_inds] / (norm_factors + 1e-4)[None, :])
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
    """
    Keep only the samples accepted by `coarse_crystal_filter`.

    Parameters
    ----------
    packing_coeff : torch.Tensor
        [num_samples] packing coefficients.
    packing_coeff_range : sequence of float
        Exclusive bounds; element 0 is the lower and element -1 the upper bound on ``packing_coeff``.
    rdf : torch.Tensor
        Per-sample values indexed along the first dimension; filtered alongside the samples.
    samples : torch.Tensor
        Per-sample values indexed along the first dimension; filtered alongside.
    vdw : torch.Tensor
        [num_samples] LJ energies checked against ``vdw_threshold``.
    vdw_threshold : float
        Samples with ``vdw`` at or above this value are removed.

    Returns
    -------
    init_len : int
        Number of samples before filtering.
    packing_coeff, rdf, samples, vdw : torch.Tensor
        The inputs restricted to the kept samples.
    """
    # filtering
    init_len = len(vdw)
    bad_inds, good_inds = coarse_crystal_filter(
        vdw, vdw_threshold, packing_coeff, packing_coeff_range)
    vdw, samples, rdf, packing_coeff = (
        vdw[good_inds], samples[good_inds], rdf[good_inds], packing_coeff[good_inds])
    return init_len, packing_coeff, rdf, samples, vdw


def get_topk_samples(packing_coeff, rdf, samples, vdw, k):
    """Keep the ``k`` samples with the lowest ``vdw``, ordered by ascending ``vdw``; returns ``(packing_coeff, rdf,
    samples, vdw)`` restricted to them."""
    good_inds = torch.argsort(vdw).flatten()[:k]
    vdw, samples, rdf, packing_coeff = (
        vdw[good_inds], samples[good_inds], rdf[good_inds], packing_coeff[good_inds])
    return packing_coeff, rdf, samples, vdw


def agglomerative_cluster(sample_score, threshold, dists=None, samples=None):
    """
    Average-linkage agglomerative clustering, keeping the lowest-score sample of each cluster.

    With ``dists`` given: if every entry of ``dists`` is below ``threshold``, no clustering is run and each sample is
    its own cluster (``classes = torch.arange(len(dists))``); otherwise scikit-learn ``AgglomerativeClustering`` with
    ``distance_threshold=threshold``, ``linkage="average"``, ``affinity='precomputed'`` and ``n_clusters=None`` is
    fit on ``dists.numpy()``. With only ``samples`` given, the same estimator with ``affinity='euclidean'`` is fit on
    ``samples.numpy()``.

    Parameters
    ----------
    sample_score : torch.Tensor
        [num_samples] scores; in each cluster with more than one member the sample with the lowest score is kept.
    threshold : float
        Linkage distance threshold, in the units of ``dists`` (or of ``samples``).
    dists : torch.Tensor, optional
        [num_samples, num_samples] precomputed distances, on the CPU.
    samples : torch.Tensor, optional
        [num_samples, num_features] points, on the CPU; used only when ``dists`` is None.

    Returns
    -------
    good_inds : torch.LongTensor
        Indices of the kept samples: one per cluster, ordered by cluster label, or ``arange(len(sample_score))``
        when there are as many clusters as samples.
    n_clusters : int
        Number of clusters.
    classes : torch.Tensor or np.ndarray
        Cluster label of each sample: a torch tensor when every distance is below ``threshold``, the scikit-learn
        ``labels_`` array otherwise.

    Raises
    ------
    AssertionError
        If both ``dists`` and ``samples`` are None.

    Notes
    -----
    The ``samples`` path evaluates ``len(dists)`` with ``dists`` None when selecting representatives, which raises
    ``TypeError``. ``AgglomerativeClustering`` is constructed with the ``affinity`` keyword; scikit-learn versions
    whose constructor takes ``metric`` and no ``affinity`` parameter raise ``TypeError`` there.
    """
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
    """
    Filtering - samples with exactly 0 LJ energy are too diffuse, and those at or above ``lj_cutoff`` are overlapping.

    A sample is kept when its LJ energy is nonzero and below ``lj_cutoff`` and its packing coefficient lies strictly
    between ``packing_cutoff[0]`` and ``packing_cutoff[-1]``. Prints the number of samples with zero energy, at or
    above ``lj_cutoff``, and at or above ``packing_cutoff[-1]`` (printed as "outside density cutoff"), then the
    number kept; the count at or below ``packing_cutoff[0]`` is not printed.

    Parameters
    ----------
    lj_record : torch.Tensor
        [num_samples] LJ energies.
    lj_cutoff : float
        Exclusive upper bound on ``lj_record``.
    packing_coeff_record : torch.Tensor
        [num_samples] packing coefficients.
    packing_cutoff : sequence of float
        Exclusive bounds; element 0 is the lower and element -1 the upper bound.

    Returns
    -------
    bad_inds : list
        Always an empty list.
    good_inds : torch.Tensor
        [num_kept] indices of the kept samples, ascending.
    """
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


def get_initial_state(config, crystal_batch, device, batch_idx):
    """
    Sample initial cell parameters for a crystal batch according to ``config.init_sample_method``.

    ``'data'`` and ``'in_config'`` return ``crystal_batch`` unchanged. Otherwise a target packing coefficient is
    chosen from ``config.init_target_cp``:

    - ``'std'``: per crystal, normal with mean 0.6226 and standard deviation 0.0447, clipped to [0.45, 0.95];
    - ``'wide'``: per crystal, uniform on [0.1, 0.7);
    - any other value that is not None: used as given;
    - None: None.

    ``'reasonable'`` calls ``sample_reasonable_random_parameters`` with that target, ``tolerance=5`` and
    ``max_attempts=50``; ``'random'`` calls ``sample_random_reduced_crystal_parameters`` if ``config.init_reduced``
    and ``sample_random_crystal_parameters`` otherwise. On both paths, if any crystal's ``packing_coeff`` differs
    from the target by more than 0.1, every crystal's ``cell_lengths`` is multiplied by
    ``(packing_coeff / target) ** (1 / 3)``, then ``clean_cell_parameters(mode='hard')`` and ``box_analysis()`` are
    called. ``'target'`` samples as ``'random'`` does and then fails an assertion.

    Parameters
    ----------
    config : object
        Reads ``init_sample_method``, ``init_target_cp``, ``opt_seed`` and, for ``'random'`` and ``'target'``,
        ``init_reduced``.
    crystal_batch : MolCrystalData
        Batch to initialize; modified in place.
    device : torch.device or str
        Device of the sampled target packing coefficients.
    batch_idx : int
        The sampling seed is ``config.opt_seed + int(batch_idx * 10000)``; the sampling methods pass it to
        ``torch.manual_seed``.

    Returns
    -------
    MolCrystalData
        ``crystal_batch``.

    Raises
    ------
    AssertionError
        For ``init_sample_method`` ``'target'`` or any unrecognized method.

    Notes
    -----
    The packing-coefficient comparison subtracts the target, so a None target raises ``TypeError`` on the
    ``'reasonable'`` and ``'random'`` paths.
    """
    # sample initial parameters
    if config.init_sample_method == 'data' or config.init_sample_method == 'in_config':
        return crystal_batch

    if config.init_target_cp == 'std':
        target_cp = (torch.randn(crystal_batch.num_graphs, device=device) * 0.0447 + 0.6226).clip(min=0.45, max=0.95)
    elif config.init_target_cp == 'wide':
        target_cp = torch.rand(crystal_batch.num_graphs, device=device) * 0.6 + 0.1  # range of 0.1 to 0.7
    elif config.init_target_cp is not None:
        target_cp = config.init_target_cp
    else:
        target_cp = None

    if config.init_sample_method == 'reasonable':
        crystal_batch.sample_reasonable_random_parameters(
            target_packing_coeff=target_cp,
            tolerance=5,
            max_attempts=50,
            # sample_niggli=config.init_sample_reduced,
            seed=config.opt_seed + int(batch_idx * 10000),
        )
        if ((crystal_batch.packing_coeff - target_cp).abs() > 0.1).any():  # todo make this a general method
            ratio = crystal_batch.packing_coeff / target_cp
            crystal_batch.cell_lengths *= ratio.pow(1.0 / 3.0)[:, None]
            crystal_batch.clean_cell_parameters(mode='hard')
            crystal_batch.box_analysis()
    elif config.init_sample_method == 'random':
        if config.init_reduced:
            crystal_batch.sample_random_reduced_crystal_parameters(
                target_packing_coeff=target_cp,
                seed=config.opt_seed + int(batch_idx * 10000))
        else:
            crystal_batch.sample_random_crystal_parameters(
                target_packing_coeff=target_cp,
                seed=config.opt_seed + int(batch_idx * 10000))
        if ((crystal_batch.packing_coeff - target_cp).abs() > 0.1).any():
            ratio = crystal_batch.packing_coeff / target_cp
            crystal_batch.cell_lengths *= ratio.pow(1.0 / 3.0)[:, None]
            crystal_batch.clean_cell_parameters(mode='hard')
            crystal_batch.box_analysis()
    elif config.init_sample_method == 'target':
        if config.init_reduced:
            crystal_batch.sample_random_reduced_crystal_parameters(
                target_packing_coeff=target_cp,
                seed=config.opt_seed + int(batch_idx * 10000))
        else:
            crystal_batch.sample_random_crystal_parameters(
                target_packing_coeff=target_cp,
                seed=config.opt_seed + int(batch_idx * 10000))
        assert False, "Below is deprecated and probably unnecessary for now"
        # standard_cell = target.compute_standard_cell()
        # crystal_batch.cell_lengths = torch.tensor(standard_cell[0, :3], dtype=torch.float32,
        #                                           device=crystal_batch.device).repeat(crystal_batch.num_graphs, 1)
        # crystal_batch.cell_angles = torch.tensor(standard_cell[0, 3:], dtype=torch.float32,
        #                                          device=crystal_batch.device).repeat(crystal_batch.num_graphs,
        #                                                                              1) * torch.pi / 2 / 90
        # crystal_batch.box_analysis()

    else:
        assert False
    return crystal_batch


def init_samples_to_optim(config, target=None):
    """
    Load and select molecules to optimize, returning the list of crystals a search starts from.

    - ``config.init_sample_method == 'data'``: ``torch.load(config.dataset_path)``; a loaded object that is not a
      list is converted with ``batch_to_list()`` if it has ``is_batch``, else wrapped in a list.
      ``config.num_samples`` is first set in place to ``n = min(config.num_samples, len(loaded))``, and the loaded
      items ``[lo, min(lo + n, len(loaded)))`` with ``lo = config.mol_seed * n`` are returned: block
      ``config.mol_seed`` (zero-based) of the list cut into consecutive blocks of ``n``, the last block being shorter
      when ``len(loaded)`` is not a multiple of ``n``. ``config.num_samples`` is then set in place to the number of
      items returned; ``lo == len(loaded)`` returns an empty list and sets it to 0.
    - ``'in_config'``: returns ``config.samples_to_optim``.
    - Any other method: the molecules are ``[target]`` if ``target`` is given, else ``torch.load(config.mol_path)``
      (wrapped in a list if it is not one), selected by ``config.sampling_mode``: ``'all'``; ``'random'``,
      ``config.mols_to_sample`` indices drawn with replacement by ``np.random.RandomState(config.mol_seed)``; or
      ``'ordered'``, the first ``config.mols_to_sample``. For every molecule, space group in
      ``config.sgs_to_search``, index in ``range(config.num_samples)`` and Z' in ``config.zp_to_search``, one
      ``MolCrystalData`` is built with ``sg_ind``, ``z_prime`` (``z_prime`` clones of the molecule when it exceeds
      1), ``max_z_prime=max(config.zp_to_search)``, all-ones placeholder cell lengths, angles, centroid,
      orientation and handedness, and ``do_box_analysis=True``; ``skip_box_analysis=True`` is also passed and
      reaches the constructor's ``**kwargs``. The list is shuffled with ``random.shuffle`` (the global Python RNG).

    Parameters
    ----------
    config : object
        Search configuration; the attributes read are named above.
    target : object, optional
        Data object passed (cloned) as the ``molecule`` of every crystal on the construction path, in place of the
        contents of ``config.mol_path``.

    Returns
    -------
    list
        Data objects to optimize.

    Raises
    ------
    AssertionError
        If ``config.sampling_mode`` is not ``'all'``, ``'random'`` or ``'ordered'`` on the construction path; in
        ``'data'`` mode, from ``batch_to_list()`` when the loaded non-list object has an ``is_batch`` attribute
        that is False.
    RuntimeError
        In ``'data'`` mode, from ``torch.arange`` when ``config.mol_seed * n`` exceeds ``len(loaded)``.
    """
    if config.init_sample_method == 'data':
        samples_to_optim = torch.load(config.dataset_path, weights_only=False)
        if not isinstance(samples_to_optim, list):
            if hasattr(samples_to_optim, 'is_batch'):
                samples_to_optim = samples_to_optim.batch_to_list()
            else:
                samples_to_optim = [samples_to_optim]
        config.num_samples = min(config.num_samples, len(samples_to_optim))
        lo = config.mol_seed * config.num_samples
        # the last block of a file whose length is not a multiple of num_samples is partial (it raised IndexError)
        index_block = torch.arange(lo, min(lo + config.num_samples, len(samples_to_optim)))
        samples_to_optim = [samples_to_optim[ind] for ind in index_block]
        config.num_samples = len(samples_to_optim)
        return samples_to_optim
    elif config.init_sample_method == 'in_config':
        samples_to_optim = config.samples_to_optim
        return samples_to_optim
    else:
        if target is None:
            mol_list = torch.load(config.mol_path, weights_only=False)
        else:
            mol_list = [target]  # must use identical conformer with ordering or rdf comparison will fail
        if not isinstance(mol_list, list):
            mol_list = [mol_list]
        if config.sampling_mode == 'all':
            mols_to_optim = mol_list
        elif config.sampling_mode == 'random':
            rng = np.random.RandomState(config.mol_seed)
            inds = rng.randint(0, len(mol_list), config.mols_to_sample)
            mols_to_optim = [mol_list[ind] for ind in inds]
        elif config.sampling_mode == 'ordered':
            mols_to_optim = [elem for elem in mol_list[:config.mols_to_sample]]
        else:
            assert False, "Sampling mode must be 'all' or 'random"
        """
        Initialize full set of crystals to optimize
        """
        max_zp = max(config.zp_to_search)
        samples_to_optim = []
        ones3 = torch.ones(3, device='cpu')
        ones1 = torch.ones(1, device='cpu')
        print("Initializing crystals to optimize")
        for mol in mols_to_optim:
            for sg in config.sgs_to_search:
                for s_ind in range(config.num_samples):
                    for zp in config.zp_to_search:
                        opt_sample = MolCrystalData(
                            molecule=[mol.clone() for _ in range(zp)] if zp > 1 else mol.clone(),
                            # duplicate molecules here
                            sg_ind=sg,
                            aunit_handedness=ones1,
                            cell_lengths=ones3,
                            cell_angles=ones3,
                            aunit_centroid=ones3,
                            aunit_orientation=ones3,
                            skip_box_analysis=True,
                            max_z_prime=max_zp,
                            z_prime=zp,
                            do_box_analysis=True,  # need this just to instantiate the tensors
                        )
                        samples_to_optim.append(opt_sample)
        from random import shuffle
        shuffle(samples_to_optim)
        return samples_to_optim


def parse_args():
    """Parse the command line, which takes one optional ``--config`` path (default None); returns the
    ``argparse.Namespace``."""
    parser = argparse.ArgumentParser(
        description="Optimize a batch of molecules against a given property via torch autograd"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If not provided, defaults to configs/crystal_searches/base.yaml",
    )
    return parser.parse_args()


def parse_opt_config(opt_config, config, device, target):
    """
    Add the models and target quantities that an optimization stage needs to its settings dict, in place.

    By ``opt_config['optim_target']``:

    - ``'rdf_score'`` or ``'classification_score'``: ``'score_model'``, from
      ``load_crystal_score_model(config.score_model_checkpoint, device)``, moved to ``device`` and set to eval mode;
    - ``'uma'``: ``'predictor'``, from ``init_uma_crystal_predictor(config.uma_predictor_path, device=device)``;
    - ``'mace'``: ``'predictor'``, from ``load_mace_model(config.mace_predictor_path, device=device,
      dtype=torch.float32)``;
    - ``'rdf_dist'``: ``'elementwise'`` set to False, ``'atomwise'`` to True, and ``'target_rdf'`` to the RDF
      tensor (element 0 of the ``'rdf'`` entry) returned by ``analyze(['rdf'], cutoff=opt_config['cutoff'],
      rdf_cutoff=opt_config['cutoff'], elementwise=False, atomwise=True, bins=100)`` on ``target`` collated alone;
    - ``'latent_dist'``: ``'target_latent'``, the ``latent_params()`` of ``target`` collated alone.

    Other targets leave the dict unchanged.

    Parameters
    ----------
    opt_config : dict
        Stage settings; must contain ``'optim_target'``, and ``'cutoff'`` for ``'rdf_dist'``.
    config : object
        Search configuration supplying the model paths named above.
    device : torch.device or str
        Device the models are loaded onto.
    target : MolCrystalData
        Target crystal; used only for ``'rdf_dist'`` and ``'latent_dist'``.

    Returns
    -------
    dict
        ``opt_config``.
    """
    if opt_config['optim_target'] in ['rdf_score', 'classification_score']:
        score_model = load_crystal_score_model(config.score_model_checkpoint, device).to(device)
        score_model.eval()
        opt_config['score_model'] = score_model
    if opt_config['optim_target'] in ['uma']:
        pred_path = config.uma_predictor_path  # smaller mol crystal model
        opt_config['predictor'] = init_uma_crystal_predictor(pred_path, device=device)
    if opt_config['optim_target'] in ['mace']:
        pred_path = config.mace_predictor_path  # smaller mol crystal model
        opt_config['predictor'] = load_mace_model(pred_path, device=device, dtype=torch.float32)
    if opt_config['optim_target'] in ['rdf_dist']:
        opt_config['elementwise'] = False
        opt_config['atomwise'] = True
        tbatch = collate_data_list([target])
        out = tbatch.analyze(['rdf'], cutoff=opt_config['cutoff'],
                             rdf_cutoff=opt_config['cutoff'],
                             elementwise=False, atomwise=True, bins=100)
        opt_config['target_rdf'] = out['rdf'][0]
    if opt_config['optim_target'] in ['latent_dist']:
        tbatch = collate_data_list([target])
        opt_config['target_latent'] = tbatch.latent_params()
    # if config.target_path is not None:
    #     opt_config['target_packing_coeff'] = config.init_target_cp
    return opt_config


def recover_opt_state(crystal_batch, config, device, batch_idx, prev_best_samples):
    """
    Restore a batch's cell parameters from ``prev_best_samples``, or reinitialize it if that fails.

    Calls ``crystal_batch.set_cell_parameters`` (which runs ``box_analysis``) with the first ``num_graphs`` rows of
    ``prev_best_samples`` moved to the batch's device; if that raises an ``Exception``, returns
    ``get_initial_state(config, crystal_batch, device, batch_idx)`` instead. ``SystemExit`` and
    ``KeyboardInterrupt``, which are not ``Exception`` subclasses, propagate.

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Batch to restore; modified in place.
    config : object
        Forwarded to `get_initial_state`.
    device : torch.device or str
        Forwarded to `get_initial_state`.
    batch_idx : int
        Forwarded to `get_initial_state`.
    prev_best_samples : torch.Tensor
        ``[num_rows, 6 + 6 * max_z_prime]`` cell parameters in the ``full_cell_parameters`` layout.

    Returns
    -------
    MolCrystalData
        ``crystal_batch``.
    """
    # if we oomed out in the last iter, recover the system state
    try:
        crystal_batch.set_cell_parameters(
            prev_best_samples[:crystal_batch.num_graphs].to(crystal_batch.device)
        )
    except Exception:  # fails often for some reason; a SIGTERM stop (SystemExit) must unwind, not reinitialise
        crystal_batch = get_initial_state(config, crystal_batch, device, batch_idx)
    return crystal_batch


def process_target(config):
    """
    Load the target polymorph named by ``config.target_identifier`` from ``config.target_path``.

    If the file holds a list, its first element whose ``identifier`` matches is taken; otherwise the loaded object's
    ``identifier`` must match. ``len(target) > 0`` is then asserted, and a message saying that the initial and target
    packing coefficients are overwritten from the target is printed; no such assignment is made.

    Parameters
    ----------
    config : object
        Reads ``target_path`` and ``target_identifier``.

    Returns
    -------
    target : object
        The matching loaded data object.
    config : object
        The input ``config``, unmodified.

    Raises
    ------
    IndexError
        If a loaded list has no element with the identifier.
    AssertionError
        If a single loaded object has a different identifier, or if ``len(target) > 0`` fails.
    """
    target = torch.load(config.target_path, weights_only=False)
    if isinstance(target, list):
        target = [elem for elem in target if elem.identifier == config.target_identifier][0]
    else:
        assert target.identifier == config.target_identifier
    assert len(target) > 0, "No target in the file with the given identifier!"
    print("overwriting initial and target packing coeffs from target!")
    # config.init_target_cp = target.packing_coeff.item()
    # config.target_packing_coeff = target.packing_coeff.item()
    return target, config
