"""
Local optimization of conformers in internal coordinates.

The intramolecular analogue of ``mxtaltools.crystal_search.crystal_opt_utils.
gradient_descent_optimization``: the same shape of loop, but the optimized latent is the
3N-6 internal DoF instead of the 12 cell parameters, and geometry is rebuilt each step by
``builder.build`` instead of ``set_cell_parameters`` + ``mol2cluster``.

Structure kept from the crystal version
---------------------------------------
* an ``nn.Module`` holding free DoF as Parameters and frozen DoF as buffers (``CrystalParams``)
* rebuild geometry from scratch each step, so no state leaks between iterations
* gradient-norm clipping and multiplicative LR annealing
* per-molecule convergence on a smoothed parameter trajectory, with the batch declared
  converged once >95% of molecules are
* return the best point of each trajectory, not the last

Deliberately changed
--------------------
* **Running-best instead of a full parameter trajectory.** The crystal version keeps
  ``params_record[max_steps, n_samples, param_dim]`` and argmins at the end. Here the DoF
  count is 3N-6 per molecule, so at a few thousand conformers that array runs to
  gigabytes; a running argmin gives the identical answer in O(1) memory. Convergence
  therefore tracks an EMA of per-molecule mean |delta DoF| rather than re-smoothing the
  stored trajectory.
* **Dihedral deltas are wrapped** into (-pi, pi] before entering the convergence
  statistic. Without that, a torsion crossing the branch cut reads as a ~2*pi jump and
  the molecule never converges.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from mxtaltools.conformers.builder import BatchedTree, build, measure
from mxtaltools.conformers.energy import ForceField, intramolecular_energy

_OPTIMIZERS = {"sgd": optim.SGD, "adam": optim.Adam, "adamw": optim.AdamW,
               "rprop": optim.Rprop, "rmsprop": optim.RMSprop,
               "adagrad": optim.Adagrad, "adadelta": optim.Adadelta,
               "lbfgs": optim.LBFGS}


def get_annealing_factor(start_value, stop_value, total_time, step_iters):
    assert stop_value > 0, "Setting final value as zero breaks this module"
    return (stop_value / start_value) ** (1 / (total_time / step_iters))


class InternalParams(nn.Module):
    """Internal DoF as optimizable Parameters, with any subset frozen.

    ``freeze`` names whole DoF classes ('r', 'theta', 'phi') to hold fixed -- the
    semi-rigid variants are exactly this, with no change to the builder. Frozen entries
    live in buffers and are spliced back in by ``dof()``, mirroring ``CrystalParams``'
    ``fixed_dims``/``fixed_vals`` split.
    """

    CLASSES = ("r", "theta", "phi")

    def __init__(self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor,
                 freeze: Sequence[str] = ()):
        super().__init__()
        bad = set(freeze) - set(self.CLASSES)
        if bad:
            raise ValueError(f"unknown DoF class(es) {bad}; choose from {self.CLASSES}")
        self.frozen = tuple(freeze)
        for name, value in zip(self.CLASSES, (r, theta, phi)):
            value = value.detach().clone()
            if name in self.frozen:
                self.register_buffer(name, value)
            else:
                setattr(self, name, nn.Parameter(value))

    def dof(self):
        return self.r, self.theta, self.phi

    @property
    def n_free(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class OptRecords:
    energy: torch.Tensor          # [steps, n_mols]
    best_energy: torch.Tensor     # [n_mols]
    best_dof: tuple               # (r, theta, phi) at each molecule's best step
    n_steps: int
    converged: torch.Tensor       # [n_mols] bool


def _wrapped_delta(new: torch.Tensor, old: torch.Tensor, periodic: bool) -> torch.Tensor:
    d = new - old
    if periodic:
        d = (d + torch.pi) % (2 * torch.pi) - torch.pi
    return d.abs()


def gradient_descent_optimization(
        tree: BatchedTree,
        init_dof: tuple,
        ff: ForceField,
        optimizer_func: Union[torch.optim.Optimizer, str] = "adam",
        init_lr: float = 1e-2,
        max_num_steps: int = 500,
        min_num_steps: int = 50,
        freeze: Sequence[str] = (),
        grad_norm_clip: Optional[float] = 1.0,
        convergence_eps: float = 1e-4,
        anneal_lr: bool = True,
        show_tqdm: bool = False,
):
    """Minimize ``ff`` over the internal DoF of every molecule in ``tree`` at once.

    ``init_dof`` is ``(r, theta, phi)`` as returned by ``measure``. Returns
    ``(best_pos, records)`` where ``best_pos`` is the geometry at each molecule's
    lowest-energy step.

    Every molecule in the batch is optimized independently -- the loss is the *sum* of
    per-molecule energies, whose gradient w.r.t. any one molecule's DoF is exactly that
    molecule's own gradient, so there is no cross-talk *from the energy*.

    ``grad_norm_clip`` is the exception, and it is inherited from the crystal-side
    design: ``clip_grad_norm_`` rescales by a single global norm, so one molecule with an
    exploding gradient throttles the step size of every other molecule in the batch. That
    is tolerable at a few dozen conformers and increasingly wasteful at a few thousand.
    For per-molecule clipping, scale each molecule's DoF gradients by its own norm via
    ``dof_batch`` before ``optimizer.step()`` instead.
    """
    params = InternalParams(*init_dof, freeze=freeze).to(tree.device)
    if params.n_free == 0:
        raise ValueError(f"freeze={freeze} leaves no free DoF to optimize")

    if isinstance(optimizer_func, str):
        optimizer_func = _OPTIMIZERS[optimizer_func.lower()]
    optimizer = optimizer_func(params.parameters(), lr=init_lr)

    lr_factor = get_annealing_factor(1, 0.01, int(max_num_steps * 0.75), 1) if anneal_lr else 1
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda e: lr_factor)

    n_mols = tree.n_mols
    dev = tree.device
    energy_record = torch.zeros(max_num_steps, n_mols)
    best_energy = torch.full((n_mols,), float("inf"), device=dev)
    best_dof = tuple(d.detach().clone() for d in params.dof())
    converged = torch.zeros(n_mols, dtype=torch.bool)
    drift = torch.full((n_mols,), float("inf"), device=dev)
    prev = tuple(d.detach().clone() for d in params.dof())

    # which molecule each DoF entry belongs to, for the per-molecule drift statistic
    dof_batch = (tree.bond_batch, tree.angle_batch, tree.torsion_batch)
    counts = torch.zeros(n_mols, device=dev).index_add(
        0, torch.cat(dof_batch), torch.ones(sum(b.numel() for b in dof_batch), device=dev))

    s_ind = 0
    with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
        while s_ind < max_num_steps and not converged.all():
            optimizer.zero_grad(set_to_none=True)

            pos = build(tree, *params.dof())
            energy = intramolecular_energy(tree, pos, ff)

            with torch.no_grad():
                energy_record[s_ind] = energy.detach().cpu()
                improved = energy.detach() < best_energy
                best_energy = torch.where(improved, energy.detach(), best_energy)
                for store, current, batch_idx in zip(best_dof, params.dof(), dof_batch):
                    store[improved[batch_idx]] = current.detach()[improved[batch_idx]]

            energy.sum().backward()
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(params.parameters(), grad_norm_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                step_drift = torch.zeros(n_mols, device=dev)
                for name, current, old, batch_idx in zip(
                        InternalParams.CLASSES, params.dof(), prev, dof_batch):
                    step_drift = step_drift.index_add(
                        0, batch_idx, _wrapped_delta(current, old, name == "phi"))
                step_drift = step_drift / counts
                # EMA, mirroring the smoothed-trajectory test in the crystal version
                drift = torch.where(torch.isinf(drift), step_drift,
                                    0.9 * drift + 0.1 * step_drift)
                prev = tuple(d.detach().clone() for d in params.dof())

            s_ind += 1
            if s_ind >= min_num_steps:
                converged = (drift < convergence_eps).cpu()
                if converged.float().mean() > 0.95:
                    converged.fill_(True)
            if show_tqdm:
                pbar.update(1)
                pbar.set_postfix(E=f"{energy.mean().item():.1f}",
                                 conv=f"{int(converged.sum())}/{n_mols}")

    best_pos = build(tree, *best_dof).detach()
    records = OptRecords(energy=energy_record[:s_ind], best_energy=best_energy.cpu(),
                         best_dof=best_dof, n_steps=s_ind, converged=converged)
    return best_pos, records


def cartesian_optimization(
        tree: BatchedTree,
        init_pos: torch.Tensor,
        ff: ForceField,
        optimizer_func: Union[torch.optim.Optimizer, str] = "adam",
        init_lr: float = 1e-2,
        max_num_steps: int = 500,
        min_num_steps: int = 50,
        grad_norm_clip: Optional[float] = 1.0,
        convergence_eps: float = 1e-4,
        anneal_lr: bool = True,
        show_tqdm: bool = False,
):
    """All-atom baseline: identical loop, but the latent is the 3N Cartesian coordinates.

    The comparison this exists for: 3N DoF instead of 3N-6, six of which are pure
    translation and rotation that the energy cannot see, and no structural guarantee that
    bonds stay bonded.
    """
    pos_param = nn.Parameter(init_pos.detach().clone())
    if isinstance(optimizer_func, str):
        optimizer_func = _OPTIMIZERS[optimizer_func.lower()]
    optimizer = optimizer_func([pos_param], lr=init_lr)

    lr_factor = get_annealing_factor(1, 0.01, int(max_num_steps * 0.75), 1) if anneal_lr else 1
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda e: lr_factor)

    n_mols, dev = tree.n_mols, tree.device
    energy_record = torch.zeros(max_num_steps, n_mols)
    best_energy = torch.full((n_mols,), float("inf"), device=dev)
    best_pos = pos_param.detach().clone()
    converged = torch.zeros(n_mols, dtype=torch.bool)
    drift = torch.full((n_mols,), float("inf"), device=dev)
    prev = pos_param.detach().clone()
    counts = torch.zeros(n_mols, device=dev).index_add(
        0, tree.batch, torch.ones(tree.n_atoms, device=dev))

    s_ind = 0
    with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
        while s_ind < max_num_steps and not converged.all():
            optimizer.zero_grad(set_to_none=True)
            energy = intramolecular_energy(tree, pos_param, ff)

            with torch.no_grad():
                energy_record[s_ind] = energy.detach().cpu()
                improved = energy.detach() < best_energy
                best_energy = torch.where(improved, energy.detach(), best_energy)
                atom_improved = improved[tree.batch]
                best_pos[atom_improved] = pos_param.detach()[atom_improved]

            energy.sum().backward()
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_([pos_param], grad_norm_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                step_drift = torch.zeros(n_mols, device=dev).index_add(
                    0, tree.batch, (pos_param - prev).abs().mean(-1)) / counts
                drift = torch.where(torch.isinf(drift), step_drift,
                                    0.9 * drift + 0.1 * step_drift)
                prev = pos_param.detach().clone()

            s_ind += 1
            if s_ind >= min_num_steps:
                converged = (drift < convergence_eps).cpu()
                if converged.float().mean() > 0.95:
                    converged.fill_(True)
            if show_tqdm:
                pbar.update(1)
                pbar.set_postfix(E=f"{energy.mean().item():.1f}")

    records = OptRecords(energy=energy_record[:s_ind], best_energy=best_energy.cpu(),
                         best_dof=(), n_steps=s_ind, converged=converged)
    return best_pos.detach(), records
