"""Local optimization of crystal cell parameters by gradient descent (``gradient_descent_optimization``) and its
helpers: per-step loss terms, records and convergence checks, asymmetric-unit centroid wrapping and
canonicalization, and noised resampling about optimized crystals (``sample_about_crystal``)."""
import copy
import gc
from argparse import Namespace
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import optim
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.geometry_utils import enforce_crystal_system, rotvec2rotmat
from mxtaltools.common.utils import is_cuda_oom
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import enforce_1d_bound, softmax_and_score


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace.

    Nested dictionaries are replaced by namespaces inside ``data_dict`` itself,
    so the input dictionary is modified in place.

    Parameters
    ----------
    data_dict : dict
        The input dictionary.

    Returns
    -------
    data_namespace : argparse.Namespace
        The output namespace.
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


CENTROID_BOUNDARIES = ('clamp', 'wrap')
CONVERGENCE_MODES = ('grad', 'params')  # gradient_descent_optimization(convergence=...)
CELL_EDGE = 1 - 1e-4  # MolCrystalOps.assign_aunit_centroid's upper bound: the builders clip centres there


def wrap_centroid(raw_centroid: torch.Tensor) -> torch.Tensor:
    """Lattice-translate fractional molecule centres into the unit cell -- any position in the cell
    describes a valid crystal -- then clamp to [0, CELL_EDGE] STRAIGHT-THROUGH: the value honours the
    builders' clip while the gradient reaches the raw parameter unchanged. So, unlike the hardtanh
    bound in clean_cell_parameters (zero gradient at and beyond a face, which rprop never leaves),
    no centre can freeze at an asymmetric-unit face or a cell face.
    Elementwise: raw_centroid (fractional, [num_graphs, 3 * max_z_prime] where called here) -> a new tensor of the
    same shape; values that round to exactly 1 after the floor are mapped to 0."""
    wrapped = raw_centroid - torch.floor(raw_centroid).detach()
    wrapped = torch.where(wrapped >= 1, wrapped - 1, wrapped)  # float rounding can land exactly on 1, which is 0
    return wrapped + (wrapped.clamp(0, CELL_EDGE) - wrapped).detach()


def canonicalize_aunit(crystal_batch):
    """In place: re-express every asymmetric-unit molecule by the symmetry image whose centre lies in
    its space group's asymmetric-unit box (ASYM_UNITS). The crystal is unchanged: the image of molecule k under
    (R, t) has centre R f + t (wrapped into the cell) and pose R_cart R(rotvec) diag(h, 1, 1), R_cart = T_fc R T_cf
    (as MolCrystalOps._transform_aunit_params); the pose is split back into a canonical rotvec and handedness by
    zp_doubling.proper_rotvecs, which projects it onto the nearest rotation first (float32 cell angles leave R_cart
    ~1e-7 off orthogonal, which rotmat2rotvec amplifies near a rotation by pi) and puts rotvecs on the +z hemisphere
    the latent transforms expect. Needed after a wrap-mode relaxation, whose centres may sit anywhere in the cell,
    before anything that reads asymmetric-unit coordinates (cleaning, latent transforms, datasets).
    Requires current T_fc / T_cf (box_analysis). Raises ValueError if some molecule has no image in its box
    (best margin below -1e-5), i.e. the tabulated box is not a fundamental domain for that crystal's operator list.
    The image chosen is the one whose centre is deepest inside the box (largest min over axes of 1 - c / box); ties
    keep the lower operator index. Slots k with z_prime <= k are left unchanged. Builds and caches
    crystal_batch.asym_unit_dict if absent; replaces aunit_centroid, aunit_orientation and aunit_handedness with new
    tensors (the cell and T_fc / T_cf are not touched, box_analysis is not rerun) and returns None."""
    from mxtaltools.crystal_building.zp_doubling import proper_rotvecs
    if not hasattr(crystal_batch, 'asym_unit_dict'):
        crystal_batch.asym_unit_dict = crystal_batch.build_asym_unit_dict()
    centroid = crystal_batch.aunit_centroid.clone()
    orientation = crystal_batch.aunit_orientation.clone()
    handedness = crystal_batch.aunit_handedness.clone()
    device = centroid.device
    f64 = torch.float64
    box = torch.stack([crystal_batch.asym_unit_dict[str(int(sg))] for sg in crystal_batch.sg_ind]).to(device, f64)
    t_fc, t_cf = crystal_batch.T_fc.to(f64), crystal_batch.T_cf.to(f64)

    op_lists = [torch.as_tensor(o, dtype=f64, device=device) for o in crystal_batch.symmetry_operators]
    n_ops = max(len(o) for o in op_lists)
    eye = torch.eye(4, dtype=f64, device=device)
    ops = torch.stack([torch.cat([o, eye.expand(n_ops - len(o), 4, 4)]) for o in op_lists])  # [n, n_ops, 4, 4]
    has_op = torch.stack([torch.arange(n_ops, device=device) < len(o) for o in op_lists])  # [n, n_ops]
    rows = torch.arange(len(ops), device=device)

    for k in range(crystal_batch.max_z_prime):
        sl = slice(3 * k, 3 * k + 3)
        present = crystal_batch.z_prime.to(device).reshape(-1) > k  # padding of lower-Z' crystals is left alone
        f = centroid[:, sl].to(f64)
        best_margin, best_c, best_j = None, None, None
        for j in range(n_ops):
            c = torch.einsum('nij,nj->ni', ops[:, j, :3, :3], f) + ops[:, j, :3, 3]
            c = c - torch.floor(c)
            c = torch.where(c >= 1, c - 1, c)  # c = -1e-17 rounds to 1; the builders clip 1 to CELL_EDGE, 1e-4 off
            margin = (1 - c / box).amin(dim=1)  # > 0 strictly inside the box, 0 on a face
            margin = torch.where(has_op[:, j], margin, torch.full_like(margin, -torch.inf))
            if best_margin is None:
                best_margin, best_c, best_j = margin, c, torch.zeros_like(rows)
            else:  # strict: ties (face points) keep the earlier operator, the identity first
                take = margin > best_margin
                best_margin = torch.where(take, margin, best_margin)
                best_c = torch.where(take[:, None], c, best_c)
                best_j = torch.where(take, torch.full_like(best_j, j), best_j)
        bad = present & (best_margin < -1e-5)
        if bad.any():
            raise ValueError(f"canonicalize_aunit: {int(bad.sum())} molecule(s) have no symmetry image inside the "
                             f"asymmetric-unit box (space groups {sorted(set(crystal_batch.sg_ind[bad].tolist()))})")
        if not present.any():
            continue
        r_frac = ops[rows[present], best_j[present], :3, :3]
        r_cart = t_fc[present] @ r_frac @ t_cf[present]
        h_old = handedness[present, k].to(f64)
        pose = rotvec2rotmat(orientation[present, sl].to(f64))  # present rows only: padding may be NaN
        pose = pose * torch.stack([h_old, torch.ones_like(h_old), torch.ones_like(h_old)], 1)[:, None, :]
        rv, h = proper_rotvecs(r_cart @ pose)
        centroid[present, sl] = best_c[present].to(centroid.dtype)
        orientation[present, sl] = rv.to(orientation.dtype)
        handedness[present, k] = h.to(handedness.dtype)

    crystal_batch.aunit_centroid = centroid
    crystal_batch.aunit_orientation = orientation
    crystal_batch.aunit_handedness = handedness


def fd_fold_aunit(crystal_batch):
    """In place, the wrap-mode end of a stage with ``fd_refold``: every Z' = 1 row re-described by its crystal_fd
    canonical form read off the BUILT lattice (``fd_refold.fold_batch``, sheet +1), instead of canonicalize_aunit.
    Never raises for a row (canonicalize_aunit raises in sg 48, 50, 59, 68, 70, 85, 86, 126, 130, 133, 137, 141,
    whose ASYM_UNITS box does not hold an image of every centre); centres wrapped into [0, 1); rotation vectors on the
    +z hemisphere; a row crystal_fd does not take keeps its wrapped centre and rotation. The centres are NOT re-expressed
    inside the ASYM_UNITS box: re-express them before a clamp-mode stage, latent transform or hop kick. Returns [num_graphs] int8
    ``fd_refold.REFOLD_STATUS`` codes (0: folded). Raises ValueError for a batch that is not Z' = 1."""
    from mxtaltools.crystal_search import fd_refold
    return fd_refold.fold_batch(crystal_batch, sheet=1)


def _fd_refold_rows(crystal_batch, param_module, optimizer, active_idx, attempts):
    """One refold pass of ``gradient_descent_optimization(fd_refold=True)`` over the evaluated batch.

    Rows of ``crystal_batch`` (the step's cleaned, centre-wrapped batch: active rows only) that ``refold_triggers``
    flags and that have fewer than ``fd_refold.REFOLD_MAX_ATTEMPTS`` attempts so far are re-described from their built
    lattice (``fd_refold.batch_refold``, sheet None: handedness kept, since it is not an optimised parameter). Each
    applied row's raw parameters (``param_module``, the global row) are overwritten with the canonical values and its
    optimizer state carried over (``_carry_optimizer_state``). Increments ``attempts`` [n] in place for every tried
    row. Returns ([n] bool applied, [n] bool tried), over all n rows of the stage. No loop beyond one pass over the
    flagged rows."""
    from mxtaltools.crystal_search import fd_refold
    n = len(attempts)
    applied = torch.zeros(n, dtype=torch.bool)
    tried = torch.zeros(n, dtype=torch.bool)
    glob = torch.arange(crystal_batch.num_graphs) if active_idx is None else active_idx.cpu()
    flagged = fd_refold.refold_triggers(crystal_batch.cell_angles).cpu() & (attempts[glob] < fd_refold.REFOLD_MAX_ATTEMPTS)
    if not bool(flagged.any()):
        return applied, tried
    if len(param_module.fixed_dims) > 0:
        raise NotImplementedError('fd_refold with fixed parameter columns')
    local = torch.nonzero(flagged).flatten()
    rows = glob[local]
    attempts[rows] += 1
    tried[rows] = True
    res = fd_refold.batch_refold(crystal_batch, local, sheet=None)
    hand = crystal_batch.aunit_handedness.detach().cpu().reshape(crystal_batch.num_graphs, -1)[local, 0].double()
    if bool((res.applied & (res.handedness != hand)).any()):  # cannot happen: det +1 cell steps, Gamma+ fold
        raise RuntimeError('fd_refold changed a handedness with sheet=None; the batch handedness is not optimised')
    for k in torch.nonzero(res.applied).flatten().tolist():
        g = int(rows[k])
        p = param_module.params[g]
        with torch.no_grad():
            p.copy_(res.params[k].to(device=p.device, dtype=p.dtype))
        _carry_optimizer_state(optimizer, p)
        applied[g] = True
    return applied, tried


def _recorded_params(crystal_batch, raw_centroid):
    """[num_graphs, n_params] CPU float32 parameters as ``update_record`` plus the wrap-mode centre overwrite record
    them: cleaned parameters with the raw (unwrapped) centres."""
    p = crystal_batch.full_cell_parameters().detach().cpu().float()
    p[:, 6:6 + raw_centroid.shape[1]] = raw_centroid.detach().cpu().float()
    return p


REFOLD_STATE_BLOCKS = ((0, 3), (3, 6), (6, 9), (9, None))  # lengths, angles, centre, rotation vector (Z' = 1)


def _carry_optimizer_state(optimizer, p):
    """Optimizer state of a refolded row ``p`` (its raw parameters just rewritten as the same crystal).

    Rprop (state with ``'step_size'`` and ``'prev'``): each block of ``REFOLD_STATE_BLOCKS`` keeps the median of its
    learned step sizes (a reduction can permute a, b, c, so per-element sizes need not belong to the same coordinate
    any more) and the previous gradient is zeroed, so the next step moves by those sizes with no sign-based change.
    Dropping the state instead restarts every coordinate at the full lr (about 1 A for a fractional centre in a 20 A
    cell, at stage-1 lr 0.05), which threw rows uphill. Any other optimizer: the state is dropped."""
    st = optimizer.state.get(p)
    if st and 'step_size' in st and 'prev' in st:
        ss = st['step_size'].view(-1)
        with torch.no_grad():
            for lo, hi in REFOLD_STATE_BLOCKS:
                if ss[lo:hi].numel():
                    ss[lo:hi] = ss[lo:hi].median()
            st['prev'].zero_()
    else:
        optimizer.state.pop(p, None)


def get_annealing_factor(start_value, stop_value, total_time, step_iters):
    """Return the multiplicative factor that takes ``start_value`` to ``stop_value`` in ``total_time / step_iters``
    multiplications, ``(stop_value / start_value) ** (step_iters / total_time)``.

    Raises
    ------
    AssertionError
        If ``stop_value`` is not positive.
    """
    assert stop_value > 0, "Setting final value as zero breaks this module"
    return (stop_value / start_value) ** (1 / (total_time / step_iters))


class CrystalParams(nn.Module):
    """Per-row learnable cell parameters with optionally fixed columns.

    Each row of ``init_sample`` becomes its own ``nn.Parameter`` holding that row's free columns
    (``params``, an ``nn.ParameterList``); the fixed columns are kept in the ``fixed_vals`` buffer. Both are
    detached clones of ``init_sample``. With ``fixed_dims`` None the buffer is ``init_sample[:, None]``
    ([n, 1, param_dim]) and is not read. ``stacked()`` reassembles the full [n, param_dim] tensor.

    Parameters
    ----------
    init_sample : torch.Tensor
        [n, param_dim] initial values.
    fixed_dims : list of int, optional
        Column indices held fixed. None (default) leaves every column free.
    """
    def __init__(self, init_sample, fixed_dims=None):
        super().__init__()
        # fixed_dims: list of column indices to freeze
        self.fixed_dims = fixed_dims if fixed_dims is not None else []
        free_dims = [i for i in range(init_sample.shape[1]) if i not in self.fixed_dims]
        self.free_dims = free_dims

        # Only free dims are Parameters
        self.params = nn.ParameterList(
            [nn.Parameter(row[free_dims].clone().detach()) for row in init_sample]
        )
        # Fixed dims stored as buffer (not optimized, but accessible)
        self.register_buffer('fixed_vals', init_sample[:, fixed_dims].clone().detach())

    def stacked(self):
        """Return a new [n, param_dim] tensor: the free columns from the per-row Parameters (differentiable) and the
        fixed columns from ``fixed_vals``."""
        free = torch.stack(list(self.params), dim=0)  # [n, n_free]
        full = torch.zeros(free.shape[0], free.shape[1] + len(self.fixed_dims),
                           device=free.device, dtype=free.dtype)
        full[:, self.free_dims] = free
        if len(self.fixed_dims) > 0:
            full[:, self.fixed_dims] = self.fixed_vals
        return full

    def update_params(self, new_values, mask=None):
        """Copy rows of ``new_values`` into the per-row Parameters in place, without gradient tracking.

        Parameters
        ----------
        new_values : torch.Tensor
            [n, n_free] values; row i is copied into ``params[i]``, so each row must have the shape of the free
            columns (the full param_dim only when no columns are fixed).
        mask : torch.Tensor, optional
            [n] bool; only rows where it is True are copied. None copies every row of ``new_values``.
        """
        with torch.no_grad():
            if mask is None:
                for i, val in enumerate(new_values):
                    self.params[i].copy_(val)
            else:
                # Only update the specific batch indices that were accepted
                indices = torch.where(mask)[0]
                for idx in indices:
                    self.params[idx].copy_(new_values[idx])


def gradient_descent_optimization(  # todo consolidate kwargs somewhere
        init_sample: torch.Tensor,
        init_crystal_batch,
        optimizer_func: Union[torch.optim.Optimizer, str],
        init_lr: Optional[float] = 1e-1,
        max_num_steps: Optional[int] = 500,
        show_tqdm: Optional[bool] = False,
        convergence_eps: float = 1e-3,
        grad_norm_clip: Optional[float] = 0.1,
        optim_target: Optional[str] = 'lj',
        score_model: Optional[torch.nn.Module] = None,
        target_packing_coeff: Optional[torch.Tensor] = None,
        do_box_restriction: Optional[bool] = False,
        cutoff: Optional[float] = None,
        compression_factor: Optional[float] = 0,
        enforce_reduced: Optional[bool] = False,
        supercell_size: Optional[int] = 10,
        anneal_lr: Optional[bool] = False,
        predictor: Optional = None,
        mc_kT: float = 0.5,
        log_noise_range=[-2, -1],
        target_rdf: torch.tensor = None,
        target_latent: torch.tensor = None,
        elementwise: bool = True,
        atomwise: bool = False,
        repulsion: float = 1.0,
        rdf_warmup: Optional[torch.tensor] = 500,
        centroid_boundary: str = 'clamp',
        intermediates_path: str = 'opt_intermediates.pt',
        early_stop: Optional[list] = None,
        early_stop_ref: Optional[float] = None,
        early_stop_kT: float = 1.0,
        exp6_kwargs: Optional[dict] = None,
        fd_refold: bool = False,
        convergence: str = 'params',
        grad_tol: float = 0.1,
        grad_patience: int = 5,
        capture_library=None,
        capture_keys: Optional[list] = None,
        capture_d2: float = 9.0,
        capture_every: int = 10,
):
    """
    Do a local optimization via gradient descent on a per-row loss.

    Each step clones ``init_crystal_batch`` (only its still-active rows), sets the current parameters with
    ``set_cell_parameters``, cleans them with ``clean_cell_parameters(mode='hard', canonicalize_orientations=False)``,
    evaluates ``analyze`` on ``'lj'``, ``'elj'`` and the compute mapped from ``optim_target`` (plus
    ``'reduction_en'`` with ``enforce_reduced``), records parameters and outputs (``update_record``) and takes one
    optimizer step on the mean of the per-row loss (``loss_and_backprop``). The loop runs at most
    ``max_num_steps - 1`` steps; it stops earlier when every row has been retired by ``early_stop`` or, once at
    least ``min(max_num_steps, 50)`` steps have run, when ``check_convergence`` marks every row converged. For each
    row, the parameters recorded at its lowest-loss step are then set on a fresh clone of ``init_crystal_batch``
    (``skip_box_analysis=False``, no cleaning), its rotation vectors moved to the +z half-space (``canonicalize_aunit``
    in wrap mode, ``fd_fold_aunit`` in wrap mode with ``fd_refold``, ``canonicalize_orientation`` otherwise), and
    re-analyzed under ``torch.no_grad`` with ``assign_outputs=True``.

    Parameters
    ----------
    init_sample : torch.Tensor
        [num_graphs, 6 + 6 * max_z_prime] initial cell parameters in the ``full_cell_parameters`` layout, one row
        per graph of ``init_crystal_batch``; optimized as per-row Parameters (``CrystalParams``).
    init_crystal_batch : MolCrystalData
        The crystals to optimize. Only clones are analyzed; ``load_ellipsoid_model`` is called on it when
        ``optim_target`` is ``'ellipsoid'``.
    optimizer_func : torch.optim.Optimizer or str
        Optimizer class or name, resolved by ``init_opt``. ``optimizer_func.lower()`` is also called
        unconditionally, so only a string currently works; ``'mcmc'`` raises AssertionError.
    init_lr : float, optional
        Initial learning rate.
    max_num_steps : int, optional
        Number of rows in the parameter record; at most ``max_num_steps - 1`` steps are taken.
    show_tqdm : bool, optional
        Show a progress bar.
    convergence_eps : float
        At or below 0: no row ever converges (every stage runs to ``max_num_steps``), in either ``convergence`` mode
        (callers pass 0 or -1 for this). Otherwise read only with ``convergence='params'``: the threshold of ``check_convergence``.
    grad_norm_clip : float, optional
        Maximum gradient norm, taken jointly over all rows' parameters (``clip_grad_norm_``).
    optim_target : str, optional
        Loss selector (case-insensitive): ``'lj'``, ``'elj'``, ``'exp6'``, ``'qlj'``, ``'silu'``, ``'ellipsoid'``,
        ``'reduce'`` (compute ``'reduction_en'``), ``'uma'``, ``'mace'`` or ``'rdf_dist'``; the loss is chosen by
        ``compute_loss``. Any other value raises KeyError here, including the further targets ``compute_loss``
        accepts.
    score_model : torch.nn.Module, optional
        Placed in the ``compute_loss`` config; read only by targets this function does not accept.
    target_packing_coeff : torch.Tensor, optional
        If given, adds the squared packing-coefficient deviation (``compute_auxiliary_loss``).
    do_box_restriction : bool, optional
        Adds the asymmetric-unit length penalty of ``compute_auxiliary_loss``.
    cutoff : float, optional
        Passed to ``analyze`` as ``cutoff`` and ``rdf_cutoff`` and used as the RDF range for ``'rdf_dist'``;
        None (default) uses 10.
    compression_factor : float, optional
        Weight of the ``compute_auxiliary_loss`` pressure term; 0 disables it.
    enforce_reduced : bool, optional
        Adds ``'reduction_en'`` to the computes, passes ``margin=0.01`` to the per-step ``analyze`` (0
        otherwise), adds ``10000 * relu(reduction_en)`` to the loss, and drops from the returned list every crystal
        whose final ``compute_cell_reduction_penalty()`` is not below 1e-3.
    supercell_size : int, optional
        Passed to ``analyze``.
    anneal_lr : bool, optional
        Multiply the learning rate after every step by the ``get_annealing_factor`` that reaches 0.01 of
        ``init_lr`` after ``int(0.75 * max_num_steps)`` steps; the decay continues past that point. False keeps
        it constant.
    predictor : optional
        Forwarded to ``analyze`` (the MLIP predictor for ``'uma'`` and ``'mace'``).
    mc_kT : float
        Read only by the Monte Carlo branch, which raises before using it.
    log_noise_range : list
        Read only by the Monte Carlo branch, which raises before using it.
    target_rdf : torch.Tensor, optional
        Target RDF for ``'rdf_dist'``, moved to ``init_sample``'s device; [..., n_channels, n_bins].
    target_latent : torch.Tensor, optional
        Moved to ``init_sample``'s device and placed in the ``compute_loss`` config; read only by a target this
        function does not accept.
    elementwise : bool
        Forwarded to ``analyze`` during the steps; the final analysis passes False.
    atomwise : bool
        Forwarded to ``analyze`` during the steps; the final analysis passes True.
    repulsion : float
        Forwarded to ``analyze`` in the final analysis only; the steps pass ``repulsion=1``.
    rdf_warmup : optional
        Channel warm-up length in steps for ``'rdf_dist'`` (see ``compute_loss``); None weights all channels
        equally.
    centroid_boundary : str
        How molecule centres are bounded while optimising.

        - ``'clamp'`` (default): clean_cell_parameters' hardtanh bound to the asymmetric-unit box -- a centre that
          reaches a face gets zero gradient there and never moves again.
        - ``'wrap'``: centres are wrapped into the unit cell (wrap_centroid) and may cross any asymmetric-unit
          face; the returned samples are re-expressed inside the box by canonicalize_aunit (with ``fd_refold``:
          crystal_fd's canonical form instead, whose centres need not lie in the box; see ``fd_refold``).
          records['params'] then holds the UNWRAPPED centres (a continuous trajectory for the convergence check).
    intermediates_path : str
        Where the best states so far are saved when the batch runs out of memory after step 0
        (run_search passes a per-run file and resumes from it only at the same batch position).
    early_stop : list, optional
        [[step, margin_kT], ...] energy cascade. At each listed step a row whose loss exceeds
        ``early_stop_ref + margin_kT * early_stop_kT`` is retired: it leaves the batch (later steps evaluate only the
        rest, which is what saves compute), keeps its best state so far, and is returned with early_stopped = True.
        Inactive without early_stop_ref (run_search passes its running minimum final energy, so the first batch
        never retires). Retired rows count as converged; the 95% convergence quorum is taken over active rows only.
    early_stop_ref : float, optional
        Reference loss for ``early_stop``; None disables the cascade.
    early_stop_kT : float
        Scale applied to every ``early_stop`` margin.
    exp6_kwargs : dict, optional
        Options of the ``'exp6'`` compute, forwarded to every ``analyze`` call with ``exp6_`` prefixed to each key
        (``{'params': 'w99', 'charges': 'gasteiger'}`` becomes ``exp6_params='w99', exp6_charges='gasteiger'``);
        see ``MolCrystalAnalysis.compute_exp6_energy``, whose pair form is fixed. None uses that method's defaults.
    convergence : str
        How rows stop before ``max_num_steps`` (``CONVERGENCE_MODES``); checked from step ``min_num_steps`` (50).

        - ``'params'`` (default): the earlier rule, ``check_convergence`` on the smoothed parameter record with a 95%
          quorum.
        - ``'grad'`` (opt-in; working assumption, not yet the default): per row, by its gradient -- a generalised force. Each step the gradient of the row's
          own loss with respect to its parameters is read before clipping and reduced to its largest absolute
          component in each block (``row_gradient_blocks``: cell lengths per A, cell angles per rad, centres per A
          (fractional gradient / cell length), rotation vectors per rad). A row whose largest block value stays at or
          below ``grad_tol`` for ``grad_patience`` consecutive steps converges: it leaves the batch, keeps its best
          state so far, and is returned with ``converged_step``. No quorum: the stage ends when every row has
          converged or retired, or at the step cap. A coordinate held by a clean wall (3 A length floor, angle or
          rotation clamp) has zero gradient through that clamp, so a pinned row reads as converged in it.

          Measured 2026-10-27 (one seed each, campaign eLJ stages, ``fd_refold`` on): over 6 space groups x 32 rows
          of a 15-atom molecule it used 17% fewer row-evaluations than ``'params'`` and ended lower; on 32 MIPCAS
          latent starts most stage-1 rows (compression on) never reached the tolerance (gradients staying at 1-8),
          so stage 1 ran to its cap: about 1.9x the row-evaluations at stage-1 ``grad_tol`` 1.0.
    grad_tol : float
        Tolerance of ``convergence='grad'``, in loss units per A or per rad. Default 0.1, set on eLJ campaign stages
        (eLJ units, kT = 6.86); a working assumption for other targets.
    grad_patience : int
        Consecutive steps at or below ``grad_tol`` before a row converges (default 5).
    fd_refold : bool
        Wrap mode, Z' = 1 only (ValueError otherwise). False (default): nothing below happens and the function
        behaves exactly as without the option. True:

        - (a) during the walk, after each step's clean and centre wrap, rows whose built lattice is in trouble -- a
          cleaned cell angle within ``fd_refold.REFOLD_ANGLE_MARGIN`` (0.025 pi) of the [0.2 pi, 0.8 pi] clamp, or
          (V / abc)^2 = F at or below ``fd_refold.REFOLD_MIN_VOLUME_FACTOR`` (0.1; F <= 0 is a left-handed built
          lattice) -- are re-described as the SAME crystal: the lattice is read off the built T_fc (a left-handed
          basis flipped to c -> -c with the centre's z -> -z, Cartesian pose unchanged), canonicalised by crystal_fd
          (reduced cell, folded pose, handedness kept) with centres wrapped into [0, 1), written into the raw
          parameters, and that row's optimizer state is carried over (``_carry_optimizer_state``: rprop keeps its
          learned step sizes, as the median of each parameter block, and forgets the previous gradient; other
          optimizers drop the state) (``_fd_refold_rows``; proofs in the module docstring of
          ``crystal_search.fd_refold``). The step is then rebuilt from the new parameters and evaluated. A result
          that the next clean would change (e.g. a reduced cell with a lattice vector under 3 A) is not applied.
          The thresholds are working assumptions (``fd_refold`` module constants).
        - (b) the returned rows are folded with crystal_fd from the built lattice (``fd_fold_aunit``) instead of
          ``canonicalize_aunit``: no raise in any group, centres in [0, 1), rotation vectors as crystal_fd gives them
          (+z hemisphere), handedness on the +1 sheet where an improper element of G allows. These centres are
          crystal_fd's canonical t wrapped into [0, 1), NOT inside the ASYM_UNITS box: a clamp-mode stage, a latent
          transform or a hop kick (clean_cell_parameters clamps centres into the box) would change the crystal, so
          run_search refuses fd_refold before a clamp-mode stage and in a campaign.

        Convergence: with ``convergence='grad'`` a refold needs nothing (the gradient is read afresh each step). With
        ``'params'`` a refold is a jump in the row's parameters, not in its crystal, so ``check_convergence``
        reads a copy of the record in which the refolded row's earlier steps are shifted by that jump (step-to-step
        changes kept, jump removed); ``records['params']`` and the best-step pick keep the true descriptions.

        Termination: the refold pass runs at most once per step and contains no retry loop, so the stage still ends
        by ``max_num_steps - 1`` steps; per row at most ``fd_refold.REFOLD_MAX_ATTEMPTS`` (8) attempts are made per
        call, after which the row walks as without the option. A canonical cell is never itself a trigger (tri_niggli
        angles in [60, 120] deg with F >= 1/2; monoclinic beta in [90, 135] deg), so an applied refold does not fire
        again until the row has walked back to a trigger.
    capture_library : minima.MinimaLibrary, optional
        Known minima of this stage's energy, each with the Hessian of E/kT there. None (default): nothing below
        happens. Otherwise, every ``capture_every`` steps from step 0, each active row's evaluated crystal is
        re-described as the library identifies crystals (``MinimaLibrary.capture_query``: crystal_fd's canonical form
        at Z' = 1, on a copy) and, if it lies within ``capture_d2`` of a minimum of its key under that minimum's own
        ellipsoid (d^2 = dx^T H dx, centre lattice translations removed), it is captured: it leaves the batch as a
        cascade-retired row does, keeps its best state so far, and is returned with ``capture_min``, the minimum's id
        (-1 for rows never captured). Captured rows count as converged for the ``'params'`` quorum. run_search passes
        it on the last stage under its ``minima`` block (``minima.finish_minima`` then reports a captured row as the
        minimum).
    capture_keys : list of tuple, optional
        One ``minima.row_key`` per row of ``init_crystal_batch``; required with ``capture_library``.
    capture_d2 : float
        Capture threshold on d^2 (default 9: three thermal widths along every direction of the ellipsoid).
    capture_every : int
        Steps between capture tests (default 10; each test re-describes the active rows, about half a step's cost
        for an eLJ batch of 100 measured 2026-10-28).

    Returns
    -------
    samples_list : list
        One data object per crystal (``batch_to_list``) with its lowest-loss parameters and the attributes assigned
        by the final ``analyze``. With an active cascade each also carries ``early_stopped``, a [1] bool tensor
        (retired by the cascade). With ``convergence='grad'`` each carries ``converged_step``, a [1] long tensor (the
        step at which it converged and left the batch; -1 if it did not).
        With ``fd_refold`` each also carries ``fd_refolds`` ([1] long, refolds applied to the row during the walk)
        and ``fd_fold_status`` ([1] long, ``fd_refold.REFOLD_STATUS`` code of the end-of-stage fold, 0 folded).
        With ``capture_library`` each carries ``capture_min`` ([1] long, the capturing minimum's id or -1).
        With ``enforce_reduced`` the list can be shorter than ``num_graphs``.
    records : dict of torch.Tensor
        CPU tensors stacked along a leading step dimension, one per step taken: every tensor output of the first
        step's ``analyze`` other than ``'rdf'``, plus ``'cp'`` (packing coefficient) and ``'loss'``; after a row
        retires its entries are NaN (``'loss'``: inf). ``'params'`` is the float32 [n_steps, num_graphs,
        6 + 6 * max_z_prime] parameter record (cleaned parameters, except the unwrapped centres in wrap mode).
        With ``convergence='grad'`` also ``'grad'``, [n_steps, num_graphs, 4] float: the per-block gradient maxima
        (``row_gradient_blocks``; NaN for rows out of the batch). With ``fd_refold`` also ``'fd_refold'`` and
        ``'fd_refold_tried'``, [n_steps, num_graphs] bool: a refold was
        applied / attempted at that step, before its evaluation (per-row counts: ``.sum(0)``).

    Raises
    ------
    ValueError
        If ``centroid_boundary`` is not in ``CENTROID_BOUNDARIES``; also from ``canonicalize_aunit`` in wrap mode
        (not with ``fd_refold``). With ``fd_refold``, unless ``centroid_boundary`` is ``'wrap'`` and every row is
        Z' = 1.
    KeyError
        If ``optim_target`` is not one of the targets listed above.
    AttributeError
        If ``optimizer_func`` is not a string.
    AssertionError
        If ``optimizer_func`` is ``'mcmc'`` or a name ``init_opt`` does not know, or on an ``'rdf_dist'`` match
        (see Notes).
    RuntimeError
        A RuntimeError or ValueError raised inside the step loop is re-raised unchanged. When ``is_cuda_oom``
        classifies it as out-of-memory and at least one step has run, the [num_graphs, n_params] best
        parameters so far are first saved with ``torch.save`` to ``intermediates_path``.

    Notes
    -----
    With ``optim_target`` exactly ``'rdf_dist'`` (compared case-sensitively here) the function also shows a
    plotly figure of log loss against step (``traj_fig``) and, if the lowest final-step loss has natural log below
    -2.5, prints a message, saves that crystal to ``'<identifier[0]>_standardized_match.pt'`` in the working
    directory and raises AssertionError.
    """  # todo implement wrapping over periodic latent DoF
    if centroid_boundary not in CENTROID_BOUNDARIES:
        raise ValueError(f"centroid_boundary must be one of {CENTROID_BOUNDARIES}, got {centroid_boundary!r}")
    if convergence not in CONVERGENCE_MODES:
        raise ValueError(f"convergence must be one of {CONVERGENCE_MODES}, got {convergence!r}")
    if capture_library is not None and (capture_keys is None or len(capture_keys) != init_crystal_batch.num_graphs):
        raise ValueError("capture_library needs capture_keys, one minima.row_key per row")
    by_grad = convergence == 'grad'
    wrap = centroid_boundary == 'wrap'
    if fd_refold:
        if not wrap:
            raise ValueError("fd_refold needs centroid_boundary='wrap'")
        from mxtaltools.crystal_search.fd_refold import check_zp1
        check_zp1(init_crystal_batch)
        refold_attempts = torch.zeros(init_crystal_batch.num_graphs, dtype=torch.long)
        refold_count = torch.zeros(init_crystal_batch.num_graphs, dtype=torch.long)

    if cutoff is None:
        # lennard jones need 10 angstroms to nicely converge
        cutoff = 10

    energy_computes = ['lj', 'elj']
    min_num_steps = 50
    num_samples = init_crystal_batch.num_graphs

    target_map = {
        'silu': 'silu',
        'qlj': 'qlj',
        'ellipsoid': 'ellipsoid',
        'reduce': 'reduction_en',
        'uma': 'uma',
        'mace': 'mace',
        'rdf_dist': 'rdf',
        'elj': 'elj',
        'exp6': 'exp6',
        'lj':'lj'
    }
    energy_computes.append(target_map[optim_target.lower()])
    energy_computes = list(set(energy_computes))

    if enforce_reduced:  # use an energy based penalty to enforce sampling in the reduced subspace
        energy_computes.append('reduction_en')
        std_margin = 0.01
    else:
        std_margin = 0

    if target_rdf is not None:
        target_rdf = target_rdf.to(init_sample.device)
    if target_latent is not None:
        target_latent = target_latent.to(init_sample.device)

    if False:  # target_rdf is not None:  # assumes we already have the box
        fixed_dims = [0, 1, 2, 3, 4, 5]
    else:
        fixed_dims = None
    param_module = CrystalParams(init_sample, fixed_dims=fixed_dims)

    loss_config = dict2namespace({
        'cutoff': cutoff,
        'target_rdf': target_rdf,
        'score_model': score_model,
        'optim_target': optim_target,
        'target_latent': target_latent,
        'rdf_warmup': rdf_warmup,
    })

    aux_config = dict2namespace({
        'compression_factor': compression_factor,
        'do_box_restriction': do_box_restriction,
        'target_packing_coeff': target_packing_coeff,
        'enforce_reduced': enforce_reduced,
    })

    optimizer = init_opt(init_lr, optimizer_func, param_module)
    monte_carlo = optimizer_func.lower() == 'mcmc'
    if monte_carlo:
        # We need a starting energy for the very first comparison
        assert False, "Rewrite with new args"
        with torch.no_grad():
            T, current_energy, current_state = init_mc_state(compression_factor, cutoff, do_box_restriction,
                                                             energy_computes, enforce_reduced, init_crystal_batch,
                                                             mc_kT, optim_target, param_module, score_model,
                                                             std_margin, supercell_size, target_packing_coeff,
                                                             predictor, target_rdf, target_latent)

    if anneal_lr:
        lr_factor = get_annealing_factor(1, 0.01, int(max_num_steps * 0.75), 1)
    else:
        lr_factor = 1
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                     lr_lambda=lambda epoch: lr_factor)

    params_record = torch.zeros((max_num_steps, init_crystal_batch.num_graphs, init_sample.shape[-1]),
                                dtype=torch.float32)
    conv_record = torch.zeros_like(params_record) if fd_refold and not by_grad else None  # see fd_refold
    grad_streak = torch.zeros(init_crystal_batch.num_graphs, dtype=torch.long)
    converged_step = torch.full((init_crystal_batch.num_graphs,), -1, dtype=torch.long)
    cascade_retired = torch.zeros(init_crystal_batch.num_graphs, dtype=torch.bool)
    capture_min = torch.full((init_crystal_batch.num_graphs,), -1, dtype=torch.long)
    capturing = capture_library is not None and len(capture_library) > 0

    if optim_target.lower() == 'ellipsoid':
        init_crystal_batch.load_ellipsoid_model()
        ellipsoid_model = copy.deepcopy(init_crystal_batch.ellipsoid_model)
        ellipsoid_model = ellipsoid_model.to(init_crystal_batch.device)
        ellipsoid_model.eval()  # unify initialization of this vs other score models, uma models, etc.

    records = None
    converged = torch.zeros(init_crystal_batch.num_graphs, dtype=torch.bool)
    cascade = {int(st): float(m) for st, m in (early_stop or [])} if early_stop_ref is not None else {}
    active = torch.ones(num_samples, dtype=torch.bool)
    active_idx = None  # None: every row active, the batch as given (the default path)
    init_active = init_crystal_batch
    init_list = None
    # torch.autograd.set_detect_anomaly(True)  # for debugging
    try:
        with torch.set_grad_enabled(not monte_carlo):
            with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
                s_ind = 0
                while (s_ind < (max_num_steps - 1)) and not converged.all():
                    optimizer.zero_grad(set_to_none=True)
                    crystal_batch = init_active.clone().detach()
                    stacked = param_module.stacked() if active_idx is None else param_module.stacked()[active_idx]

                    if monte_carlo:
                        crystal_batch.set_cell_parameters(param_module.stacked(), skip_box_analysis=False)
                        noised_latents = get_mc_noised_samples(crystal_batch, log_noise_range)
                        crystal_batch.latent_to_cell_params(noised_latents,
                                                            skip_box_analysis=True,
                                                            skip_enforce_crystal_system=True)
                    else:
                        crystal_batch.set_cell_parameters(stacked,
                                                          skip_box_analysis=True)

                    # no orientation fold here: folded each step, a recorded rotvec would jump where it crosses z = 0,
                    # and check_convergence reads the record; the returned rows are folded once, below
                    crystal_batch.clean_cell_parameters(
                        mode='hard',
                        canonicalize_orientations=False,
                    )  # box analysis included in here
                    if wrap:  # the centres bypass clean's asymmetric-unit clamp: raw parameter, wrapped into the cell
                        raw_centroid = stacked[:, 6:6 + 3 * crystal_batch.max_z_prime]
                        crystal_batch.aunit_centroid = wrap_centroid(raw_centroid)
                    if fd_refold:  # rows at the angle clamp or near / past F = 0: same crystal, canonical description
                        pre = _recorded_params(crystal_batch, raw_centroid)
                        refolded, refold_tried = _fd_refold_rows(crystal_batch, param_module, optimizer, active_idx,
                                                                 refold_attempts)
                        if refolded.any():  # rebuild this step from the new raw parameters (one pass, no re-check)
                            refold_count += refolded.long()
                            crystal_batch = init_active.clone().detach()
                            stacked = param_module.stacked() if active_idx is None else param_module.stacked()[active_idx]
                            crystal_batch.set_cell_parameters(stacked, skip_box_analysis=True)
                            crystal_batch.clean_cell_parameters(mode='hard', canonicalize_orientations=False)
                            raw_centroid = stacked[:, 6:6 + 3 * crystal_batch.max_z_prime]
                            crystal_batch.aunit_centroid = wrap_centroid(raw_centroid)
                            # the convergence record: shift the refolded rows' history by the jump (crystal unchanged)
                            glob = torch.arange(crystal_batch.num_graphs) if active_idx is None else active_idx.cpu()
                            loc = refolded[glob]
                            if conv_record is not None:
                                jump = _recorded_params(crystal_batch, raw_centroid)[loc] - pre[loc]
                                conv_record[:s_ind, glob[loc]] += jump.unsqueeze(0)

                    outputs, cluster_batch = crystal_batch.analyze(
                        computes=energy_computes,
                        cutoff=cutoff,
                        rdf_cutoff=cutoff,
                        supercell_size=supercell_size,
                        return_cluster=True,
                        repulsion=1,
                        surface_padding=0,
                        std_orientation=True,
                        margin=std_margin,
                        predictor=predictor,
                        elementwise=elementwise,
                        atomwise=atomwise,
                        assign_outputs=True,
                        bins=100,
                        **{f'exp6_{k}': v for k, v in (exp6_kwargs or {}).items()},
                    )

                    """ record some stats"""
                    records = update_record(crystal_batch, outputs, params_record, records, s_ind, active_idx)
                    if wrap:  # unwrapped centre: continuous across cell edges, so check_convergence sees no jumps
                        rows = slice(None) if active_idx is None else active_idx.cpu()
                        params_record[s_ind, rows, 6:6 + raw_centroid.shape[1]] = raw_centroid.detach().cpu()
                    if conv_record is not None:
                        conv_record[s_ind] = params_record[s_ind]
                    if fd_refold:
                        records.setdefault('fd_refold', []).append(refolded)
                        records.setdefault('fd_refold_tried', []).append(refold_tried)

                    """loss and backprop"""
                    if monte_carlo:
                        assert False, "Update arguments"
                        parse_mc_step(T, cluster_batch, compression_factor, crystal_batch, current_energy,
                                      current_state, do_box_restriction, enforce_reduced, optim_target, outputs,
                                      param_module, records, score_model, target_packing_coeff, target_rdf,
                                      target_latent, cutoff)

                    else:
                        g_blocks = loss_and_backprop(cluster_batch, crystal_batch, grad_norm_clip,
                                                     optimizer, outputs, param_module, records,
                                                     loss_config, aux_config, s_ind, active_idx, num_samples,
                                                     return_grad_blocks=by_grad)
                    retire = torch.zeros_like(active)
                    if s_ind in cascade:  # retire rows still far above the reference: they leave the batch
                        retire = active & (records['loss'][-1] > early_stop_ref + cascade[s_ind] * early_stop_kT)
                        cascade_retired |= retire
                    if by_grad and not monte_carlo:  # per row: gradient under grad_tol for grad_patience steps
                        records.setdefault('grad', []).append(_full_rows(g_blocks, active_idx, num_samples))
                        rows_g = torch.arange(num_samples) if active_idx is None else active_idx.cpu()
                        below = g_blocks.amax(1) <= grad_tol
                        grad_streak[rows_g] = torch.where(below, grad_streak[rows_g] + 1, 0)
                        if convergence_eps > 0 and s_ind + 1 >= min(max_num_steps, min_num_steps):
                            done = active & ~retire & (grad_streak >= grad_patience)
                            converged_step[done] = s_ind
                            retire = retire | done
                    if capturing and s_ind % capture_every == 0:  # rows inside a known minimum's ellipsoid leave
                        glob = torch.arange(num_samples) if active_idx is None else active_idx.cpu()
                        hit_id, hit_d2 = capture_library.capture_query(crystal_batch,
                                                                       [capture_keys[i] for i in glob.tolist()])
                        hit = (hit_d2 < capture_d2) & ~retire[glob]
                        capture_min[glob[hit]] = hit_id[hit]
                        retire[glob[hit]] = True
                    if retire.any():  # retired rows leave the batch: later steps evaluate only the rest
                        active = active & ~retire
                        if init_list is None:
                            init_list = init_crystal_batch.batch_to_list()
                        active_idx = torch.nonzero(active).flatten()
                        if len(active_idx):
                            init_active = collate_data_list([init_list[i] for i in active_idx.tolist()]).to(
                                init_crystal_batch.device)

                    if s_ind % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                    scheduler1.step()  # shrink
                    s_ind += 1
                    if s_ind % 10 == 0:
                        pbar.update(10)
                    if not active.any():
                        converged = torch.ones_like(converged)
                    elif not by_grad and s_ind >= min(max_num_steps, max(50, min_num_steps)):
                        converged = check_convergence(conv_record if fd_refold else params_record, s_ind,
                                                      convergence_eps, optimizer, init_lr, active=active)


    except (RuntimeError, ValueError) as e:
        if is_cuda_oom(e):
            if s_ind > 0:
                records = {k: torch.stack(v) for k, v in records.items()}
                best_sample_ind = torch.argmin(records['loss'],
                                               dim=0).flatten()  # pick the best sample from each trajectory
                best_samples = torch.stack([params_record[best_sample_ind[ind], ind] for ind in range(num_samples)])
                torch.save(best_samples, intermediates_path)
            raise e  # we want to raise this to fire the oom catcher above
        else:
            raise e

    records = {k: torch.stack(v) for k, v in records.items()}

    """
    Pull out and re-analyze the best samples from each trajectory
    """
    best_sample_ind = torch.argmin(records['loss'], dim=0).flatten()  # pick the best sample from each trajectory
    best_samples = torch.stack([params_record[best_sample_ind[ind], ind] for ind in range(num_samples)])
    crystal_batch = init_crystal_batch.clone().detach()  # this is necessary to not retain lots of intermediate tensors
    crystal_batch.set_cell_parameters(best_samples.to(crystal_batch.device),
                                      skip_box_analysis=False)
    if wrap:  # recorded centres are unwrapped: wrap into the cell, then re-express inside the asymmetric unit
        crystal_batch.aunit_centroid = wrap_centroid(crystal_batch.aunit_centroid)
        if fd_refold:  # crystal_fd's canonical form of the built crystal (never raises for a row)
            fold_status = fd_fold_aunit(crystal_batch)
        else:
            canonicalize_aunit(crystal_batch)  # also puts the rotvecs on the +z half-space
    else:
        crystal_batch.canonicalize_orientation()
    # no_grad: these are reported energies. With grad on, an MLIP energy assigned to each returned sample kept its
    # whole autograd graph alive (~1.5 GB of GPU memory per MACE Z'=2 acridine crystal) for as long as the sample
    # lived -- e.g. through the whole next opt stage in run_search, which collates this stage's outputs.
    with torch.no_grad():
        _ = crystal_batch.analyze(
            computes=energy_computes,
            cutoff=cutoff,
            rdf_cutoff=cutoff,
            supercell_size=supercell_size,
            return_cluster=False,
            repulsion=repulsion,
            surface_padding=0,
            predictor=predictor,
            assign_outputs=True,
            elementwise=False,
            atomwise=True,
            bins=100,
            **{f'exp6_{k}': v for k, v in (exp6_kwargs or {}).items()},
        )
    samples_list = crystal_batch.batch_to_list()
    if cascade:
        for i, elem in enumerate(samples_list):
            elem.early_stopped = torch.tensor([bool(cascade_retired[i])])
    if by_grad:
        for i, elem in enumerate(samples_list):
            elem.converged_step = torch.tensor([int(converged_step[i])])
    if fd_refold:
        for i, elem in enumerate(samples_list):
            elem.fd_refolds = torch.tensor([int(refold_count[i])])
            elem.fd_fold_status = torch.tensor([int(fold_status[i])])
    if capture_library is not None:
        for i, elem in enumerate(samples_list):
            elem.capture_min = torch.tensor([int(capture_min[i])])
    if enforce_reduced:
        penalty = crystal_batch.compute_cell_reduction_penalty()
        samples_list = [elem for i, elem in enumerate(samples_list) if penalty[i] < 1e-3]

    """
    # analyze trajectory information, if we want
    
    from mxtaltools.common.utils import log_rescale_positive
    timesteps = torch.arange(s_ind).repeat(init_crystal_batch.num_graphs, 1).T
    traj_fig(timesteps, log_rescale_positive(records['loss']), names=['time', 'loss'])
    traj_fig(timesteps, (records['cp']), names=['time', 'cp'])
    traj_fig(records['cp'], log_rescale_positive(records['lj']), names=['cp', 'lj'])
    traj_fig(timesteps, torch.log(records['loss']), names=['time', 'loss'])
    
    crystal_batch.plot_batch_cell_params()
    crystal_batch.plot_batch_density_funnel()
    import plotly.graph_objects as go
    penalty = crystal_batch.compute_cell_reduction_penalty()
    go.Figure(go.Histogram(x=(penalty+1e-6).log10().cpu().detach().numpy())).show()
    
    """
    if optim_target == 'rdf_dist':
        timesteps = torch.arange(s_ind).repeat(init_crystal_batch.num_graphs, 1).T
        traj_fig(timesteps, torch.log(records['loss']), names=['time', 'loss'])

        if torch.amin(records['loss'][-1]).log() < -2.5:
            print("Found the crystal!")
            good_ind = torch.argmin(records['loss'][-1]).item()
            sample = crystal_batch.batch_to_list()[good_ind]
            torch.save(sample.cpu().detach(), f'{crystal_batch.identifier[0]}_standardized_match.pt')
            assert False, "Found what we were looking for"

    # timesteps = torch.arange(s_ind).repeat(init_crystal_batch.num_graphs, 1).T
    # traj_fig(timesteps, torch.log(records['loss']), names=['time', 'loss'])

    records['params'] = params_record[:s_ind]
    return samples_list, records


"""
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata

mol = ase_mol_from_crystaldata(cluster_batch, torch.argmin(records['loss'][-1]).item(), mode='unit cell')
from ase.io import write
write('aaa.cif',mol)
"""

"""

# params trajectories
from plotly.subplots import make_subplots
fig = make_subplots(rows=4, cols=3, subplot_titles = ['a','b','c','al','be','ga','x','y','z','u','v','w'])
for ind in range(12):
    row = ind // 3 + 1
    col = ind % 3 + 1
    for ind2 in range(params_record.shape[1]):
        fig.add_scatter(y=params_record[:s_ind, ind2, ind], showlegend=False, row=row, col=col)
fig.show()

# traj vis
t_ind = torch.argmin(records['loss'][-1]).item()
tbatch = collate_data_list([samples_list[0].clone().cpu() for _ in range(s_ind)])
tbatch.set_cell_parameters(params_record[:s_ind, t_ind, :])
tbatch.pose_aunit()
tbatch.build_unit_cell()
tbatch.visualize(mode='unit cell')

"""


def _full_rows(value, active_idx, n, fill=float('nan')):
    """A per-row record of the active rows, padded to all n rows (retired rows get `fill`)."""
    value = value.detach().cpu()
    if active_idx is None:
        return value
    dtype = value.dtype if value.is_floating_point() else torch.float32
    out = torch.full((n,) + tuple(value.shape[1:]), fill, dtype=dtype)
    out[active_idx.cpu()] = value.to(dtype)
    return out


def update_record(crystal_batch, outputs, params_record, records, s_ind, active_idx=None):
    """Record one step's parameters and outputs and return the records dict.

    ``crystal_batch.full_cell_parameters()`` (detached, on CPU) is written into ``params_record[s_ind]`` in place.
    With ``active_idx`` only those rows are written; the others first copy row ``s_ind - 1`` when ``s_ind > 0``.
    When ``records`` is None it is created with an empty list for every key of ``outputs`` except ``'rdf'``, plus
    ``'cp'`` and ``'loss'``. Every tensor output whose key is in ``records``, and ``crystal_batch.packing_coeff``
    under ``'cp'``, is appended as a detached CPU tensor, padded with NaN to all n rows when ``active_idx`` is
    given (``_full_rows``). ``'loss'`` is appended by ``loss_and_backprop``.

    Parameters
    ----------
    crystal_batch : MolCrystalData
        The batch evaluated this step (active rows only).
    outputs : dict
        ``analyze`` results for ``crystal_batch``.
    params_record : torch.Tensor
        [max_num_steps, n, n_params] parameter record, modified in place.
    records : dict or None
        Lists of per-step tensors, or None on the first call.
    s_ind : int
        Step index.
    active_idx : torch.Tensor, optional
        Indices of the evaluated rows among the n; None means every row, in order.

    Returns
    -------
    dict
        ``records`` with this step appended.
    """
    n = params_record.shape[1]
    p = crystal_batch.full_cell_parameters().detach().cpu()  # must put this before the .backward()
    if active_idx is None:
        params_record[s_ind] = p
    else:  # retired rows stay where they were (a flat trajectory: they count as converged)
        if s_ind > 0:
            params_record[s_ind] = params_record[s_ind - 1]
        params_record[s_ind, active_idx.cpu()] = p
    if records is None:
        records = {key: [] for key in outputs if key != 'rdf'}
        records['cp'] = []
        records['loss'] = []
    for key, value in outputs.items():
        if torch.is_tensor(value) and key in records:
            records[key].append(_full_rows(value, active_idx, n))
    records['cp'].append(_full_rows(crystal_batch.packing_coeff, active_idx, n))
    return records


def loss_and_backprop(cluster_batch, crystal_batch, grad_norm_clip, optimizer, outputs, param_module, records,
                      loss_config, aux_config, opt_step, active_idx=None, n_total=None, return_grad_blocks=False):
    """Compute the per-row loss, record it, and take one optimizer step on its mean.

    The loss is ``compute_loss`` followed by ``compute_auxiliary_loss``. It is appended to ``records['loss']``
    detached on CPU and padded to ``n_total`` rows with inf for rows outside ``active_idx`` (``_full_rows``).
    ``loss.mean()`` is backpropagated, the gradients of ``param_module.parameters()`` are clipped to total norm
    ``grad_norm_clip`` and ``optimizer.step()`` is called. Returns None, or with ``return_grad_blocks`` the
    [n_active, 4] ``row_gradient_blocks`` of the rows' own losses, read before the clip.

    Parameters
    ----------
    cluster_batch : MolCrystalData
        Batch returned by ``analyze(return_cluster=True)``.
    crystal_batch : MolCrystalData
        The evaluated crystal batch.
    grad_norm_clip : float
        Maximum total gradient norm.
    optimizer : torch.optim.Optimizer
        Optimizer over ``param_module``.
    outputs : dict
        ``analyze`` results.
    param_module : CrystalParams
        The optimized parameters.
    records : dict
        Records dict with a ``'loss'`` list.
    loss_config : argparse.Namespace
        Passed to ``compute_loss`` as ``config``.
    aux_config : argparse.Namespace
        Passed to ``compute_auxiliary_loss`` as ``config``.
    opt_step : int
        Current step, passed to ``compute_loss``.
    active_idx : torch.Tensor, optional
        Indices of the evaluated rows; None means every row.
    n_total : int, optional
        Total number of rows; needed when ``active_idx`` is given.
    """
    loss = compute_loss(cluster_batch, crystal_batch, outputs, loss_config, opt_step)
    loss = compute_auxiliary_loss(cluster_batch, loss, outputs, aux_config)

    records['loss'].append(_full_rows(loss, active_idx, n_total, fill=float('inf')))  # retired: never the best step
    loss_to_backprop = loss.mean()  # save some effort in backprop
    loss_to_backprop.backward()
    blocks = None
    if return_grad_blocks:  # before the clip
        blocks = row_gradient_blocks(param_module, crystal_batch, active_idx, len(loss))
    torch.nn.utils.clip_grad_norm_(param_module.parameters(), grad_norm_clip)  # gradient clipping
    optimizer.step()  # apply grad
    del loss_to_backprop, cluster_batch, outputs, crystal_batch, loss
    return blocks


def row_gradient_blocks(param_module, crystal_batch, active_idx, n_active):
    """[n_active, 4] CPU float: per evaluated row, the largest absolute component of the gradient of its own loss
    (``n_active`` x the stored gradient of the batch-mean loss; rows do not share parameters) in each block: cell
    lengths (loss / A), cell angles (loss / rad), centres (fractional gradient / that axis's cleaned cell length,
    loss / A) and rotation vectors (loss / rad); Z' > 1 units are pooled per block. Fixed columns count as zero."""
    rows = range(len(param_module.params)) if active_idx is None else active_idx.tolist()
    n_par = len(param_module.free_dims) + len(param_module.fixed_dims)
    full = torch.zeros(n_active, n_par)
    for k, g in enumerate(rows):
        gr = param_module.params[g].grad
        if gr is not None:
            full[k, param_module.free_dims] = gr.detach().cpu().float().view(-1) * n_active
    full = full.abs()
    nz = (n_par - 6) // 6  # Z'
    lengths = crystal_batch.cell_lengths.detach().cpu().float()  # [n_active, 3], cleaned
    cen = full[:, 6:6 + 3 * nz].view(n_active, nz, 3) / lengths[:, None, :]
    return torch.stack([full[:, 0:3].amax(1), full[:, 3:6].amax(1), cen.reshape(n_active, -1).amax(1),
                        full[:, 6 + 3 * nz:].amax(1)], 1)


def get_mc_noised_samples(crystal_batch, log_noise_range):
    """Return ``crystal_batch.latent_params()`` with each row moved along a random unit direction by
    ``10 ** u``, u uniform on [log_noise_range[0], log_noise_range[1]], then clipped to [-1, 1].

    Parameters
    ----------
    crystal_batch : MolCrystalData
        Batch whose latent parameters are perturbed. ``latent_params()`` canonicalizes its asymmetric units (and
        free centroid axes) in place first; the noised latents are not assigned back.
    log_noise_range : sequence of float
        Base-10 exponent range of the step length.

    Returns
    -------
    torch.Tensor
        Noised latents, same shape as ``latent_params()``.
    """
    samples = crystal_batch.latent_params()
    rand_dir = torch.randn_like(samples)
    rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
    # rand_magnitude = torch.randn(len(samples), device=samples.device).abs() * noise_level
    u = torch.rand(len(samples), device=samples.device)
    rand_magnitude = 10 ** (log_noise_range[0] + (log_noise_range[1] - log_noise_range[0]) * u)
    noised_samples = (samples + rand_dir * rand_magnitude[:, None]).clip(min=-1, max=1)
    return noised_samples


def parse_mc_step(T, cluster_batch, compression_factor, crystal_batch, current_energy, current_state,
                  do_box_restriction, enforce_reduced, optim_target, outputs, param_module, records, score_model,
                  target_packing_coeff, target_rdf, target_latent, cutoff):
    """Metropolis acceptance step for the Monte Carlo mode of ``gradient_descent_optimization``, which raises
    before reaching it.

    Written to accept each row with probability ``min(1, exp(-(E_prop - E_curr) / T))``, update ``current_energy``
    and ``current_state`` in place for accepted rows, copy ``current_state`` into ``param_module`` and append
    ``current_energy`` to ``records['loss']``. Its calls to ``compute_loss`` and ``compute_auxiliary_loss`` do not
    match their current signatures, so a call currently raises TypeError at the first of them.
    """
    # 1. Compute energy of the proposal (noised_samples)
    proposal_energy = compute_loss(cluster_batch, crystal_batch, optim_target, outputs, score_model, target_rdf,
                                   target_latent, cutoff)
    proposal_energy = compute_auxiliary_loss(cluster_batch, compression_factor,
                                             do_box_restriction, proposal_energy,
                                             target_packing_coeff, enforce_reduced, outputs)
    # 2. Calculate Metropolis Criterion
    # For minimization: P(accept) = exp(-(E_prop - E_curr) / T)
    delta_e = proposal_energy - current_energy
    acceptance_prob = torch.exp(-delta_e / T).clamp(max=1.0)
    # 3. Batch-wise Acceptance
    accepted = torch.rand_like(acceptance_prob) < acceptance_prob
    # 4. Update the "Current" State
    current_energy[accepted] = proposal_energy[accepted]
    current_state[accepted] = crystal_batch.full_cell_parameters()[accepted]
    # 5. Sync the param_module so the next iteration starts from the accepted state
    param_module.update_params(current_state)  # You'll need a setter in your module
    # Record the energy for your stats
    records['loss'].append(current_energy.detach().cpu())
    # 6. Cooling (Simulated Annealing)
    # T *= cooling_rate
    del proposal_energy, cluster_batch, outputs, crystal_batch


def init_mc_state(compression_factor, cutoff, do_box_restriction, energy_computes, enforce_reduced, init_crystal_batch,
                  mc_kT, optim_target, param_module, score_model, std_margin, supercell_size, target_packing_coeff,
                  uma_predictor, target_rdf, target_latent):
    """Initial state for the Monte Carlo mode of ``gradient_descent_optimization``, which raises before reaching it.

    Sets ``param_module.stacked()`` on a clone of ``init_crystal_batch`` and analyzes it; it would return
    ``(mc_kT, energy, param_module.stacked().clone())``, but its calls to ``compute_loss`` and
    ``compute_auxiliary_loss`` do not match their current signatures, so a call currently raises TypeError.
    """
    # Initialize param_module and get initial energy
    crystal_batch = init_crystal_batch.clone().detach()
    crystal_batch.set_cell_parameters(param_module.stacked())
    outputs, cluster_batch = crystal_batch.analyze(computes=energy_computes,
                                                   cutoff=cutoff,
                                                   rdf_cutoff=cutoff,
                                                   supercell_size=supercell_size,
                                                   return_cluster=True,
                                                   repulsion=1,
                                                   surface_padding=0,
                                                   std_orientation=True,
                                                   margin=std_margin,
                                                   predictor=uma_predictor, )
    current_energy = compute_loss(cluster_batch, crystal_batch, optim_target, outputs, score_model, cutoff, target_rdf,
                                  target_latent)
    current_energy = compute_auxiliary_loss(cluster_batch, compression_factor,
                                            do_box_restriction, current_energy, target_packing_coeff,
                                            enforce_reduced,
                                            outputs)
    current_state = param_module.stacked().clone()
    # Temperature schedule (Initial T should be ~ typical delta energy)
    T = mc_kT
    return T, current_energy, current_state


def init_opt(init_lr, optimizer_func, param_module):
    """Build an optimizer over ``param_module.parameters()`` with learning rate ``init_lr``.

    Parameters
    ----------
    init_lr : float
        Learning rate.
    optimizer_func : str or callable
        An optimizer class (any callable taking ``(params, lr=...)``), or a case-insensitive name: ``'sgd'``,
        ``'adam'``, ``'rprop'``, ``'rmsprop'``, ``'adamw'``, ``'adadelta'``, ``'nadam'``, ``'adagrad'``,
        ``'asgd'``, or ``'mcmc'``, which gives ``optim.SGD``.
    param_module : torch.nn.Module
        Module whose parameters are optimized.

    Returns
    -------
    torch.optim.Optimizer
        The constructed optimizer.

    Raises
    ------
    AssertionError
        If ``optimizer_func`` is a string not listed above.
    """
    if isinstance(optimizer_func, str):
        if optimizer_func.lower() == 'sgd':
            optimizer_func = optim.SGD
        elif optimizer_func.lower() == 'adam':
            optimizer_func = optim.Adam
        elif optimizer_func.lower() == 'rprop':
            optimizer_func = optim.Rprop
        elif optimizer_func.lower() == 'rmsprop':
            optimizer_func = optim.RMSprop
        elif optimizer_func.lower() == 'adamw':
            optimizer_func = optim.AdamW
        elif optimizer_func.lower() == 'adadelta':
            optimizer_func = optim.Adadelta
        elif optimizer_func.lower() == 'nadam':
            optimizer_func = optim.Nadam
        elif optimizer_func.lower() == 'adagrad':
            optimizer_func = optim.Adagrad
        elif optimizer_func.lower() == 'adadelta':
            optimizer_func = optim.Adadelta
        elif optimizer_func.lower() == 'asgd':
            optimizer_func = optim.ASGD
        elif optimizer_func.lower() == 'mcmc':
            optimizer_func = optim.SGD  # this is a dummy
        else:
            assert False, "Must pass an optimizer or an implemented string"
    optimizer = optimizer_func(param_module.parameters(), lr=init_lr)
    return optimizer


def traj_fig(x, y, names=[None, None], yrange=None, xrange=None):
    """Show a plotly figure of ``y`` against ``x``, one line per column, with markers on the first and last rows.

    Parameters
    ----------
    x, y : torch.Tensor
        [T, N]; column i is trajectory "Run i".
    names : list
        x- and y-axis titles.
    yrange, xrange : optional
        Axis ranges passed to plotly.
    """
    import plotly.graph_objects as go
    fig = go.Figure()

    for ind in range(x.shape[1]):
        fig.add_scatter(x=x[:, ind], y=y[:, ind], name=f"Run {ind}", opacity=0.5)
    fig.add_scatter(x=x[-1, :], y=y[-1, :], mode='markers', marker_size=20, marker_color='black',
                    name='Final State')
    fig.add_scatter(x=x[0, :], y=y[0, :], mode='markers', marker_size=20, marker_color='grey',
                    name='Initial State')
    fig.update_layout(xaxis_title=names[0], yaxis_title=names[1], xaxis_range=xrange, yaxis_range=yrange)
    fig.show()


def ema_trajectory(traj: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    Vectorized normalized EMA along time (dim=0).

    Entry t is ``sum_{s<=t} (1 - alpha) ** (t - s) * traj[s] / sum_{s<=t} (1 - alpha) ** (t - s)``, so a constant
    trajectory c gives c. The weights are formed as ``(1 - alpha) ** (T - 1 - s)`` in ``traj``'s dtype, so for long
    trajectories the earliest weights can underflow to zero and the earliest entries become NaN (0 / 0). (Until
    2026-09-28 the result was also multiplied by alpha, so check_convergence's convergence_eps acted as a bound
    1 / alpha = 10x looser than written; configs written before then carry thresholds tuned against that.)

    Parameters
    ----------
    traj : torch.Tensor
        [T, N, D] trajectory.
    alpha : float
        Smoothing factor.

    Returns
    -------
    torch.Tensor
        [T, N, D].
    """
    T = traj.size(0)
    w = (1 - alpha) ** torch.arange(T, device=traj.device, dtype=traj.dtype)  # [T]
    w = w.flip(0).view(T, 1, 1)  # decay weights

    # weighted cumulative sum
    numer = torch.cumsum(traj * w, dim=0)
    denom = torch.cumsum(w, dim=0)

    return numer / denom


def compute_loss(cluster_batch, crystal_batch, outputs, config, opt_step):
    """Return the per-row loss selected by ``config.optim_target`` (case-insensitive).

    - ``'lj'``, ``'silu'``, ``'qlj'``, ``'elj'``, ``'exp6'``, ``'ellipsoid'``, ``'uma'``, ``'mace'``: the ``outputs``
      entry of the same name.
    - ``'reduce'``: ``outputs['reduction_en']``.
    - ``'inter_overlaps'``: ``inter_overlap_loss(cluster_batch, crystal_batch)``; ``inter_overlap_loss`` takes one
      argument, so this currently raises TypeError.
    - ``'classification_score'``: minus ``softmax_and_score`` of the first two output columns of
      ``config.score_model(cluster_batch, force_edges_rebuild=False)``.
    - ``'rdf_score'``: softplus of column 2 of that score-model output.
    - ``'rdf_dist'``: ``compute_rdf_distance(outputs['rdf'][0], config.target_rdf, linspace(0, config.cutoff,
      n_bins), channel_weights)`` plus ``softplus(5 * (outputs['elj'] + 307)) / 5 / 100``, with n_channels and
      n_bins the last two dimensions of ``config.target_rdf``. With ``config.rdf_warmup`` set, channel c has weight
      ``sigmoid((opt_step - onset_c) / (rdf_warmup / (2 * n_channels)))``, the onsets evenly spaced on
      [0, rdf_warmup]; otherwise every weight is 1.
    - ``'latent_dist'``: the per-row Euclidean norm of ``config.target_latent - crystal_batch.latent_params()``.

    Parameters
    ----------
    cluster_batch : MolCrystalData
        Batch returned by ``analyze(return_cluster=True)``.
    crystal_batch : MolCrystalData
        The crystal batch that was analyzed.
    outputs : dict
        ``analyze`` results.
    config : argparse.Namespace
        Needs ``optim_target``, and ``score_model``, ``target_rdf``, ``cutoff``, ``rdf_warmup`` or
        ``target_latent`` as the target requires.
    opt_step : int
        Current step; read by ``'rdf_dist'``.

    Returns
    -------
    torch.Tensor
        Per-row loss.

    Raises
    ------
    UnboundLocalError
        If ``optim_target`` matches none of the targets above.
    """
    if config.optim_target.lower() == 'lj':  # todo obviate this with analysis keys
        loss = outputs['lj']

    elif config.optim_target.lower() == 'silu':
        loss = outputs['silu']

    elif config.optim_target.lower() == 'qlj':
        loss = outputs['qlj']

    elif config.optim_target.lower() == 'elj':
        loss = outputs['elj']

    elif config.optim_target.lower() == 'exp6':
        loss = outputs['exp6']

    elif config.optim_target.lower() == 'inter_overlaps':  # force molecules apart by separating their centroids
        loss = inter_overlap_loss(cluster_batch, crystal_batch)

    elif config.optim_target.lower() == 'classification_score':
        loss = -softmax_and_score(config.score_model(cluster_batch, force_edges_rebuild=False)[:, :2])

    elif config.optim_target.lower() == 'rdf_score':
        loss = F.softplus(config.score_model(cluster_batch, force_edges_rebuild=False)[:, 2])

    elif config.optim_target.lower() == 'ellipsoid':
        loss = outputs['ellipsoid']

    elif config.optim_target.lower() == 'reduce':
        loss = outputs['reduction_en']

    elif config.optim_target.lower() == 'uma':
        loss = outputs['uma']

    elif config.optim_target.lower() == 'mace':
        loss = outputs['mace']

    elif config.optim_target.lower() == 'rdf_dist':
        n_channels = config.target_rdf.shape[-2]  # 120
        if config.rdf_warmup is not None:
            channel_warmup = config.rdf_warmup
            channel_onsets = torch.linspace(0, channel_warmup, n_channels)  # evenly spaced turn-on times
            channel_weights = torch.sigmoid((opt_step - channel_onsets) / (channel_warmup / n_channels * 0.5))

            # n_waves = 3
            # base_periods = torch.tensor([1.0, 1.6, 2.5]) * config.rdf_warmup
            # channel_idx = torch.arange(n_channels, dtype=torch.float32)
            #
            # modulation = torch.zeros(n_channels)
            # for i in range(n_waves):
            #     modulation += torch.sin(
            #         2 * torch.pi * opt_step / base_periods[i] + 2 * torch.pi * channel_idx / n_channels * (i + 1))
            # modulation = modulation / n_waves
            # channel_weights = 0.5 + 0.5 * modulation
        else:
            channel_weights = torch.ones(n_channels)
        rdf_loss = compute_rdf_distance(outputs['rdf'][0], config.target_rdf,
                                        torch.linspace(0, config.cutoff, config.target_rdf.shape[-1]),
                                        channel_weights=channel_weights)

        en_cut = -307  # set equal or higher to the target energy
        beta = 5
        lj_en = outputs['elj']
        lj_loss = F.softplus(beta * (lj_en - en_cut)) / beta

        loss = rdf_loss + lj_loss / 100

    elif config.optim_target.lower() == 'latent_dist':
        loss = (config.target_latent - crystal_batch.latent_params()).norm(dim=-1)

    return loss


def compute_auxiliary_loss(cluster_batch, loss, outputs, config):
    """Return ``loss`` plus the penalties enabled in ``config``, added out of place (``loss`` itself is returned
    when none is enabled).

    With L = ``cluster_batch.scale_lengths_to_aunit()`` ([num_graphs, 3] asymmetric-unit edge lengths) and
    cp = ``cluster_batch.packing_coeff``:

    - ``target_packing_coeff`` not None: ``(cp - target_packing_coeff) ** 2``.
    - ``do_box_restriction``: ``80000 / L ** 12`` summed over the three axes.
    - ``compression_factor`` nonzero: ``compression_factor * L.sum(dim=1) * sigmoid((0.65 - cp) / 0.01)``.
    - ``enforce_reduced``: ``10000 * relu(outputs['reduction_en'])``.

    Parameters
    ----------
    cluster_batch : MolCrystalData
        Batch returned by ``analyze(return_cluster=True)``.
    loss : torch.Tensor
        Per-row base loss; not modified.
    outputs : dict
        ``analyze`` results; ``'reduction_en'`` is read when ``enforce_reduced``.
    config : argparse.Namespace
        Needs ``target_packing_coeff``, ``do_box_restriction``, ``compression_factor`` and ``enforce_reduced``.

    Returns
    -------
    torch.Tensor
        The penalized per-row loss.
    """
    if config.target_packing_coeff is not None:
        cp_loss = (cluster_batch.packing_coeff - config.target_packing_coeff) ** 2
        loss = loss + cp_loss
    if config.do_box_restriction:
        # enforce box shape cannot become too long (3 mol radii) or narrow (3 angstroms) in any direction
        # repulsive from about range 3 #(80000/aunit_lengths**12 + 10*aunit_lengths - 31.25).sum(dim=1)  # forces boxes to be larger than 3 angstroms, but squeezes them otherwise
        aunit_lengths = cluster_batch.scale_lengths_to_aunit()
        box_loss = (80000 / aunit_lengths ** 12).sum(dim=1)
        loss = loss + box_loss
    if config.compression_factor != 0:
        aunit_lengths = cluster_batch.scale_lengths_to_aunit()
        threshold = 0.65
        width = 0.01  # controls sharpness of transition

        gate = torch.sigmoid((threshold - cluster_batch.packing_coeff) / width)
        pressure_loss = aunit_lengths.sum(dim=1) * config.compression_factor * gate
        loss = loss + pressure_loss

    if config.enforce_reduced:
        penalty = F.relu(outputs['reduction_en'])
        loss = loss + 10000 * penalty  # severely punish reduction violations

    return loss


def inter_overlap_loss(cluster_batch):
    """Per-crystal penalty on molecules whose centroids lie within twice the molecular radius.

    Molecule centroids are the mean atom positions per molecule (from ``mol_ind``). Each directed edge of
    ``radius_graph`` over the centroids (radius ``2 * cluster_batch.radius.amax()``, at most 10 neighbours per
    centroid, so a close pair usually appears in both directions) contributes ``relu(2 - d / radius)``, with d the
    centroid distance and radius that crystal's ``radius``; contributions are summed per crystal.

    Parameters
    ----------
    cluster_batch : MolCrystalData
        Cluster batch with ``mol_ind``, ``radius`` and ``num_atoms`` (atoms per molecule).

    Returns
    -------
    torch.Tensor
        [num_graphs] summed penalty.

    Raises
    ------
    AssertionError
        If any crystal has Z' > 1.
    """
    # enforce molecules far enough away that they cannot possibly overlap
    # intermolecular centroid range repulsion
    assert cluster_batch.z_prime.amax() == 1, "molwise business not implemented for Z'>1"
    _, atoms_per_cluster = torch.unique(cluster_batch.batch, return_counts=True)
    mols_per_cluster = (atoms_per_cluster / cluster_batch.num_atoms).long()
    molwise_batch = torch.arange(cluster_batch.num_graphs, device=cluster_batch.device).repeat_interleave(
        mols_per_cluster,
        dim=0)
    flat_mol_inds = cluster_batch.mol_ind + torch.cat(
        [torch.zeros(1, device=cluster_batch.device, dtype=torch.long),
         torch.cumsum(mols_per_cluster, dim=0)]).long()[:-1].repeat_interleave(
        torch.diff(cluster_batch.ptr), dim=0)
    mol_centroids = scatter(cluster_batch.pos, flat_mol_inds, reduce='mean', dim=0, dim_size=flat_mol_inds[-1] + 1)
    edge_i, edge_j = gnn.radius_graph(
        x=mol_centroids,
        batch=molwise_batch,
        max_num_neighbors=10,
        r=float(cluster_batch.radius.amax() * 2))
    inter_dists = torch.linalg.norm(
        mol_centroids[edge_i] - mol_centroids[edge_j], dim=1
    )
    scaled_inter_dists = inter_dists / cluster_batch.radius[molwise_batch[edge_i]]
    edgewise_losses = F.relu(-(scaled_inter_dists - 2))  # push molecules apart if they are within each others' radii
    loss = scatter(edgewise_losses,
                   molwise_batch[edge_i],
                   dim_size=cluster_batch.num_graphs,
                   dim=0,
                   reduce='sum')
    return loss


""" # viz sample distribution

import plotly.graph_objects as go
ljs = torch.tensor([elem.scaled_lj for elem in opt_samples[:len(samples_record[0])]])
ljs2 = torch.tensor([elem.scaled_lj for elem in opt_samples])
fig = go.Figure()
fig.add_histogram(x=ljs, nbinsx=100, histnorm='probability density')
fig.add_histogram(x=ljs2, nbinsx=100, histnorm='probability density')
fig.show()

"""

"""
import plotly.graph_objects as go
fig = go.Figure()
lj_pots = torch.stack(
    [torch.tensor([sample.scaled_lj for sample in sample_list]) for sample_list in samples_record])
es_pots = torch.stack([torch.tensor([sample.es_pot for sample in sample_list]) for sample_list in samples_record])
orig_batch = collate_data_list(opt_samples)
lj_pots = torch.cat([orig_batch.scaled_lj[None, :], lj_pots], dim=0)
es_pots = torch.cat([orig_batch.es_pot[None, :], es_pots], dim=0)
for ind in range(lj_pot.shape[-1]):
    fig.add_scatter(y=lj_pots[..., ind], marker_color='blue', name='lj', legendgroup='lg',
                    showlegend=True if ind == 0 else False)
    fig.add_scatter(y=es_pots[..., ind] * 10, marker_color='red', name='es', legendgroup='es',
                    showlegend=True if ind == 0 else False)
    fig.add_scatter(y=es_pots[..., ind] * 10 + lj_pots[..., ind], marker_color='green', name='combo', legendgroup='combo',
                    showlegend=True if ind == 0 else False)

fig.add_scatter(y=(lj_pots).mean(1), marker_color='black', name='lj', legendgroup='lg',
                showlegend=True if ind == 0 else False)
fig.add_scatter(y=(es_pots * 10).mean(1), marker_color='black', name='es', legendgroup='es',
                showlegend=True if ind == 0 else False)
fig.add_scatter(y=(es_pots * 10 + lj_pots).mean(1), marker_color='black', name='combo', legendgroup='combo',
                showlegend=True if ind == 0 else False)

fig.show()
"""


def _init_for_local_opt(lr, max_num_steps, optimizer_func, sample, num_atoms):
    """Return an optimizer over ``[sample]``, two MultiplicativeLR schedulers (factor 0.975, and the factor that
    grows ``lr`` 100-fold in ``max_num_steps // 10`` steps), ``hit_max_lr`` (False), ``max_lr`` (100 times
    ``lr``) and zero-initialized record tensors, as one tuple in the order of the return statement."""
    optimizer = optimizer_func([sample], lr=lr)
    max_lr_target_time = max_num_steps // 10
    max_lr = lr * 100
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False
    num_samples = len(sample)
    vdw_record = torch.zeros((max_num_steps, num_samples))
    samples_record = torch.zeros((max_num_steps, num_samples, 12))
    raw_samples_record = torch.zeros_like(samples_record)
    handedness_record = torch.zeros((max_num_steps, num_samples))
    loss_record = torch.zeros_like(vdw_record)
    lr_record = torch.zeros(max_num_steps)
    packing_record = torch.zeros_like(vdw_record)
    aunit_poses = torch.zeros((len(vdw_record), num_atoms, 3))
    return (hit_max_lr, loss_record, lr_record, max_lr,
            optimizer, packing_record, samples_record, raw_samples_record, handedness_record,
            scheduler1, scheduler2, vdw_record, aunit_poses)


#
# def cleanup_sample(raw_sample, sg_ind_list, symmetries_dict):
#     # force outputs into physical ranges
#     # cell lengths have to be positive nonzero
#     cell_lengths = raw_sample[:, :3].clip(min=0.01)
#     # range from (0,pi) with 20% padding to prevent too-skinny cells
#     cell_angles = enforce_1d_bound(raw_sample[:, 3:6], x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
#                                    mode='hard')
#     # positions must be on 0-1
#     mol_positions = enforce_1d_bound(raw_sample[:, 6:9], x_span=0.5, x_center=0.5, mode='hard')
#     # for now, just enforce vector norm
#     rotvec = raw_sample[:, 9:12]
#     norm = torch.linalg.norm(rotvec, dim=1)
#     new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode='hard')  # MUST be nonzero
#     new_rotvec = rotvec / norm[:, None] * new_norm[:, None]
#     # invert_inds = torch.argwhere(new_rotvec[:, 2] < 0)
#     # new_rotvec[invert_inds] = -new_rotvec[invert_inds]  # z direction always positive
#     # force cells to conform to crystal system
#     cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list,
#                                                        symmetries_dict)
#     sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)
#     return sample
#

def check_convergence(params_record, s_ind, convergence_eps, optimizer, init_lr, active=None):
    """Per-row convergence flags from the recent parameter trajectory.

    ``params_record[:s_ind]`` is smoothed with ``ema_trajectory`` (alpha 0.1). A row is converged when the absolute
    step-to-step change of its smoothed parameters, averaged over the 49 differences among the last 50 smoothed steps
    and over all parameter columns, is below ``convergence_eps``. With ``active``, inactive rows are marked converged
    and the converged share is taken over active rows only (1 when none are active); without it, over all rows. If
    that share exceeds 0.95, every row is marked converged.

    Parameters
    ----------
    params_record : torch.Tensor
        [max_num_steps, n, n_params] parameter record.
    s_ind : int
        Number of recorded steps; at least 50.
    convergence_eps : float
        Threshold on the averaged absolute change, in the units of ``params_record``.
    optimizer : torch.optim.Optimizer
        Not used.
    init_lr : float
        Not used.
    active : torch.Tensor, optional
        [n] bool mask of active rows.

    Returns
    -------
    torch.Tensor
        [n] bool.
    """
    smoothed = ema_trajectory(params_record[:s_ind])
    diffs = smoothed[s_ind - 50:s_ind, :, :].diff(dim=0).abs().mean((0, 2))
    converged = diffs < convergence_eps
    if active is None:
        share = converged.float().mean()
    else:  # retired rows count as converged but do not vote in the quorum
        converged = converged | ~active
        share = converged[active].float().mean() if active.any() else torch.tensor(1.0)
    if share > 0.95:
        converged.fill_(True)

    return converged


def sample_about_crystal(opt_samples: Union[list],
                         noise_level: float,
                         num_samples: int,
                         cutoff: Optional[float] = 10,
                         do_silu_pot: Optional[bool] = False,
                         ):
    """Draw ``num_samples`` noisy neighbourhoods around each optimized crystal.

    Restored from 8dea6b56^ (deleted 2025-12-10 in the Niggli-sampling removal),
    where its only caller -- parallel_synthesis.otf_synthesize_crystals -- was
    missed.  That dangling import made ``mxtaltools.modeller`` and ``main.py``
    un-importable for ~8.5 months; see docs/design/refactor_plan.md G1.

    Restored WITHOUT the original ``enforce_niggli`` argument: that commit removed
    the parameter from ``clean_cell_parameters``.  The sole caller never passed it
    and its default was False, so dropping it preserves the original behaviour.

    Each draw builds a fresh batch (collated from the list, or cloned),
    applies ``noise_cell_parameters(noise_level)`` then
    ``clean_cell_parameters(mode='hard')``, and evaluates
    ``build_and_analyze(cutoff=cutoff, return_cluster=True)``.

    Parameters
    ----------
    opt_samples : list or MolCrystalData
        The crystals: a list of data objects, or a batch.
    noise_level : float
        Standard deviation passed to ``noise_cell_parameters`` (standardized
        units; its 12-column layout matches only Z' = 1).
    num_samples : int
        Number of noised copies of the whole set.
    cutoff : float, optional
        Passed to ``build_and_analyze``.
    do_silu_pot : bool, optional
        Also attach ``silu`` from ``cluster_batch.compute_silu_energy()``.

    Returns
    -------
    list of list
        ``num_samples`` lists, each holding one noised copy of every
        input sample, with ``lj``, ``scaled_lj`` and ``es_pot`` (and ``silu``
        with ``do_silu_pot``) attached per sample. The samples are detached
        CPU copies; the attached energies are indexed from the computed
        tensors without being moved.
    """
    samples_record = []
    for _ in range(num_samples):
        if isinstance(opt_samples, list):
            crystal_batch = collate_data_list(opt_samples)
        else:
            crystal_batch = opt_samples.clone()

        crystal_batch.noise_cell_parameters(noise_level)
        crystal_batch.clean_cell_parameters(mode='hard')
        lj_pot, es_pot, scaled_lj_pot, cluster_batch = crystal_batch.build_and_analyze(
            cutoff=cutoff, return_cluster=True)
        samples_list = crystal_batch.detach().cpu().batch_to_list()

        if do_silu_pot:
            silu_energy = cluster_batch.compute_silu_energy()

        for si, sample in enumerate(samples_list):
            sample.lj = lj_pot[si]
            sample.scaled_lj = scaled_lj_pot[si]
            sample.es_pot = es_pot[si]
            if do_silu_pot:
                sample.silu = silu_energy[si]

        samples_record.append(samples_list)

    return samples_record
