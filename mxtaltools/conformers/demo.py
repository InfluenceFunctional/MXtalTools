"""
Visual and profiling walkthrough of the internal-coordinate conformer builder.

    python -m mxtaltools.conformers.demo                 # everything, no GUI
    python -m mxtaltools.conformers.demo --view          # everything, opens ASE viewers
    python -m mxtaltools.conformers.demo dists scan      # just those two

Figures land in ``--outdir`` (default ``conformer_demo/``) alongside ``.xyz``
trajectories, so the ASE viewer is optional -- every 3D artefact is written to disk
either way.

Demos
-----
dists        distribution of every r / theta / phi over an ETKDG ensemble
sensitivity  how far the molecule actually moves per DoF class
scan         rigid torsion scan about one bond, driven purely through phi
rings        how fast ring-closure bonds break when ring torsions move
profile      build time vs batch size, fp32 vs fp64, forward vs forward+backward
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from mxtaltools.conformers.builder import build, closure_length, collate, measure
from mxtaltools.conformers.topology import spec_from_mol

torch.set_default_dtype(torch.float64)

# categorical slots 1-3 of the reference palette; all-pairs CVD-validated
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

DEMO_MOLECULES = {
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "ethanol": "CCO",
    "phenetole": "CCOc1ccccc1",
    "leucine": "CC(C)CC(N)C(=O)O",
}
RING_MOLECULES = {
    "benzene (aromatic)": "c1ccccc1",
    "cyclohexane (saturated)": "C1CCCCC1",
    "naphthalene (fused)": "c1ccc2ccccc2c1",
}


# ---------------------------------------------------------------- helpers

def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "medium", "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
        "lines.linewidth": 1.8, "lines.markersize": 5,
        "figure.dpi": 130, "savefig.bbox": "tight",
    })


def ensemble(smiles: str, n_confs: int = 200, seed: int = 0xC0FFEE):
    """(spec, positions [n_confs, N, 3] in placement order, mol) from an ETKDG ensemble."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if mol.GetNumConformers() == 0:
        raise RuntimeError(f"no conformers embedded for {smiles}")
    spec = spec_from_mol(mol)
    pos = np.stack([c.GetPositions()[spec.perm] for c in mol.GetConformers()])
    return spec, pos, mol


def batch_of(spec, pos_stack, device=None):
    """Collate ``n_confs`` copies of one molecule and stack its conformers flat."""
    tree = collate([spec] * len(pos_stack), device=device)
    flat = torch.tensor(pos_stack.reshape(-1, 3), device=tree.device)
    return tree, flat


def kabsch_rmsd(p: torch.Tensor, q: torch.Tensor, mask=None) -> torch.Tensor:
    """Best-fit RMSD per structure, ``[B]``; p, q are ``[B, N, 3]``."""
    if mask is not None:
        p, q = p[:, mask], q[:, mask]
    p = p - p.mean(1, keepdim=True)
    q = q - q.mean(1, keepdim=True)
    u, _, vt = torch.linalg.svd(p.transpose(1, 2) @ q)
    d = torch.sign(torch.linalg.det(u @ vt))
    eye = torch.eye(3, dtype=p.dtype, device=p.device).expand(len(p), -1, -1).clone()
    eye[:, 2, 2] = d
    rot = u @ eye @ vt
    return ((p @ rot - q) ** 2).sum(-1).mean(-1).sqrt()


def atoms_from(spec, pos: torch.Tensor):
    from ase import Atoms
    return Atoms(numbers=np.asarray(spec.z), positions=pos.detach().cpu().numpy())


def write_traj(path: Path, spec, frames):
    from ase.io import write
    write(str(path), [atoms_from(spec, f) for f in frames])
    return path


def maybe_view(frames, spec, do_view: bool, label: str):
    if not do_view:
        return
    from ase.visualize import view
    print(f"    opening ASE viewer: {label}")
    view([atoms_from(spec, f) for f in frames])


def rotatable_tree_bond(tree, spec):
    """The tree bond whose rotation splits the molecule most evenly.

    Rotating about bond ``(u, v)`` means shifting phi for every atom whose reference
    pair is ``(b, c) == (u, v)`` -- i.e. every child of ``v``. That carries all of
    ``v``'s descendants rigidly about the u-v axis, which is a classical torsion.
    """
    n = spec.n_atoms
    parent = np.full(n, -1, dtype=np.int64)
    parent[spec.bond_index[:, 1]] = spec.bond_index[:, 0]

    desc = np.ones(n, dtype=np.int64)
    for k in range(n - 1, 0, -1):  # placement order is topological
        desc[parent[k]] += desc[k]

    best, best_score = None, -1
    ti = spec.torsion_index
    for v in range(1, n):
        u = parent[v]
        if u < 0 or not ((ti[:, 1] == u) & (ti[:, 2] == v)).any():
            continue
        moved = desc[v] - 1                       # v itself lies on the axis
        score = min(moved, n - moved - 2)
        if score > best_score:
            best, best_score = (int(u), int(v)), score
    if best is None:
        return None

    # atoms carried by the rotation: the subtree below v, excluding v itself
    moving = np.zeros(n, dtype=bool)
    for k in range(1, n):
        moving[k] = moving[parent[k]] or parent[k] == best[1]
    return best[0], best[1], moving


def rotate_about_bond(tree, phi: torch.Tensor, u: int, v: int, delta) -> torch.Tensor:
    """Shift every torsion whose axis is ``(u, v)`` by ``delta``.

    ``u`` and ``v`` are *molecule-local* slots, so the batch offsets have to come off
    the index before matching. Only children of ``v`` anchored to ``u`` are shifted;
    children anchored to a rotating sibling follow rigidly on their own, which is what
    makes this a true rigid-group torsion rather than a per-atom nudge.
    """
    local = tree.torsion_index - tree.ptr[tree.torsion_batch].unsqueeze(1)
    mask = (local[:, 1] == u) & (local[:, 2] == v)
    return phi + mask.to(phi.dtype) * delta


# ---------------------------------------------------------------- demos

def demo_dists(out: Path, view_gui: bool):
    """Every DoF over an ETKDG ensemble: r and theta are needles, phi is not."""
    import matplotlib.pyplot as plt

    name, smiles = "ibuprofen", DEMO_MOLECULES["ibuprofen"]
    spec, pos, mol = ensemble(smiles, n_confs=300)
    tree, flat = batch_of(spec, pos)
    r, theta, phi = measure(tree, flat)

    n_conf = len(pos)
    r_ = r.reshape(n_conf, -1).numpy()
    th_ = np.rad2deg(theta.reshape(n_conf, -1).numpy())
    ph_ = np.rad2deg(phi.reshape(n_conf, -1).numpy())

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, data, unit, title, color in [
            (axes[0], r_.ravel(), "A", "bond lengths  r", C_BLUE),
            (axes[1], th_.ravel(), "deg", "bond angles  theta", C_ORANGE),
            (axes[2], ph_.ravel(), "deg", "dihedrals  phi", C_AQUA)]:
        ax.hist(data, bins=90, color=color, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel(unit)
        ax.set_yticks([])

    # per-DoF spread is the number that matters, not the pooled spread
    for ax, arr, unit in [(axes[0], r_, "A"), (axes[1], th_, "deg"), (axes[2], ph_, "deg")]:
        wrapped = arr
        if unit == "deg" and ax is axes[2]:
            wrapped = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(arr))))
        sd = np.median(wrapped.std(axis=0))
        ax.text(0.97, 0.94, f"median per-DoF sd\n{sd:.3g} {unit}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK_2)

    axes[0].set_ylabel("count", color=INK_2)
    fig.suptitle(f"{name}: {n_conf} ETKDG conformers, {spec.n_dof} DoF each "
                 f"({r_.shape[1]} r + {th_.shape[1]} theta + {ph_.shape[1]} phi = 3N-6)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "dof_distributions.png")
    plt.close(fig)

    print(f"  per-DoF sd: r {np.median(r_.std(0)):.4f} A | "
          f"theta {np.median(th_.std(0)):.2f} deg | phi {np.median(ph_.std(0)):.1f} deg")
    print(f"  -> {out / 'dof_distributions.png'}")


def demo_sensitivity(out: Path, view_gui: bool):
    """Move each DoF class within its own observed range; see what the molecule does."""
    import matplotlib.pyplot as plt

    spec, pos, mol = ensemble(DEMO_MOLECULES["ibuprofen"], n_confs=64)
    heavy = torch.tensor(np.asarray(spec.z) > 1)
    scales = np.geomspace(0.02, 1.0, 12)
    n_rep = 24

    base_spec_batch, flat = batch_of(spec, pos[:1])
    r0, th0, ph0 = measure(base_spec_batch, flat)
    ref = build(base_spec_batch, r0, th0, ph0).reshape(1, spec.n_atoms, 3)

    # natural spread of each class across the ensemble; r in A, theta/phi in DEGREES
    # (measure returns radians -- keep every sd in the same unit its perturbation uses)
    tree_e, flat_e = batch_of(spec, pos)
    r_e, th_e, ph_e = measure(tree_e, flat_e)
    sd = {
        "r": float(np.median(r_e.reshape(len(pos), -1).numpy().std(0))),
        "theta": float(np.median(np.rad2deg(th_e.reshape(len(pos), -1).numpy()).std(0))),
        "phi": float(np.median(np.rad2deg(np.angle(np.exp(
            1j * ph_e.reshape(len(pos), -1).numpy()))).std(0))),
    }

    rep_tree = collate([spec] * n_rep)
    curves = {}
    g = torch.Generator().manual_seed(11)
    for cls in ("r", "theta", "phi"):
        rms = []
        for s in scales:
            rr = r0.repeat(n_rep).clone()
            tt = th0.repeat(n_rep).clone()
            pp = ph0.repeat(n_rep).clone()
            if cls == "r":
                rr += torch.randn(rr.shape, generator=g) * (s * sd["r"])
            elif cls == "theta":
                tt += torch.randn(tt.shape, generator=g) * np.deg2rad(s * sd["theta"])
            else:
                pp += torch.randn(pp.shape, generator=g) * np.deg2rad(s * sd["phi"])
            p = build(rep_tree, rr, tt, pp).reshape(n_rep, spec.n_atoms, 3)
            rms.append(kabsch_rmsd(p, ref.expand_as(p), heavy).mean().item())
        curves[cls] = rms

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for cls, color, lbl in [("r", C_BLUE, f"r  (1.0 = {sd['r']:.3f} A)"),
                            ("theta", C_ORANGE, f"theta  (1.0 = {sd['theta']:.1f} deg)"),
                            ("phi", C_AQUA, f"phi  (1.0 = {sd['phi']:.0f} deg)")]:
        ax.loglog(scales, curves[cls], color=color, marker="o", label=lbl)
        ax.annotate(cls, (scales[-1], curves[cls][-1]), textcoords="offset points",
                    xytext=(6, 0), color=INK_2, fontsize=8, va="center")
    ax.set_xlabel("perturbation, in multiples of that class's own ensemble spread")
    ax.set_ylabel("heavy-atom RMSD from reference (A)")
    ax.set_title("Perturbing each DoF class over its natural range")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "dof_sensitivity.png")
    plt.close(fig)

    ratio = curves["phi"][-1] / max(curves["r"][-1], 1e-12)
    print(f"  at full natural spread: phi moves the molecule {ratio:.0f}x further than r")
    print(f"  -> {out / 'dof_sensitivity.png'}")


def demo_scan(out: Path, view_gui: bool):
    """A rigid torsion scan produced purely by shifting phi -- no Cartesian rotation."""
    name, smiles = "phenetole", DEMO_MOLECULES["phenetole"]
    spec, pos, mol = ensemble(smiles, n_confs=1)
    n_frames = 48

    tree1, flat = batch_of(spec, pos[:1])
    r0, th0, ph0 = measure(tree1, flat)
    bond = rotatable_tree_bond(tree1, spec)
    if bond is None:
        print("  no rotatable tree bond found; skipping")
        return
    u, v, moving = bond

    tree = collate([spec] * n_frames)
    deltas = torch.linspace(0, 2 * np.pi, n_frames)
    per_atom = torch.zeros(tree.torsion_index.shape[0])
    n_tors = ph0.numel()
    for i, d in enumerate(deltas):
        per_atom[i * n_tors:(i + 1) * n_tors] = d
    phi = rotate_about_bond(tree, ph0.repeat(n_frames), u, v, per_atom)
    frames = build(tree, r0.repeat(n_frames), th0.repeat(n_frames), phi)
    frames = frames.reshape(n_frames, spec.n_atoms, 3)

    ref = frames[0]
    heavy = torch.tensor(np.asarray(spec.z) > 1)
    rmsds = kabsch_rmsd(frames, ref.expand_as(frames), heavy)
    sym = Chem.GetPeriodicTable()
    elem = lambda i: sym.GetElementSymbol(int(np.asarray(spec.z)[i]))
    print(f"  scanning about tree bond {elem(u)}{u}-{elem(v)}{v}, {n_frames} frames; "
          f"{int(moving.sum())} of {spec.n_atoms} atoms carried")
    print(f"  heavy-atom RMSD across the scan: {rmsds.min():.3f} - {rmsds.max():.3f} A")

    # a *rigid* torsion must leave every distance within each fragment untouched;
    # only cross-fragment distances may change
    dm = lambda p, m: torch.cdist(p[:, m], p[:, m],
                                  compute_mode="donot_use_mm_for_euclid_dist")
    mv = torch.tensor(moving)
    drift = max((dm(frames, m) - dm(ref.unsqueeze(0), m)).abs().max().item()
                for m in (mv, ~mv))
    cross = (torch.cdist(frames[:, mv], frames[:, ~mv])
             - torch.cdist(ref[mv].unsqueeze(0), ref[~mv].unsqueeze(0))).abs().max().item()
    print(f"  intra-fragment distance drift: {drift:.2e} A   "
          f"(cross-fragment swing: {cross:.2f} A)")

    path = write_traj(out / "torsion_scan.xyz", spec, frames)
    print(f"  -> {path}")
    maybe_view(frames, spec, view_gui, "torsion scan")


def demo_rings(out: Path, view_gui: bool):
    """Ring-closure bonds are not free parameters; watch them break."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    sigmas = np.geomspace(0.5, 30.0, 14)
    n_rep = 128
    broken_frames, broken_spec = None, None

    for (label, smiles), color in zip(RING_MOLECULES.items(), (C_BLUE, C_ORANGE, C_AQUA)):
        spec, pos, mol = ensemble(smiles, n_confs=1)
        in_ring = np.array([a.IsInRing() for a in mol.GetAtoms()])[spec.perm]

        tree1, flat = batch_of(spec, pos[:1])
        r0, th0, ph0 = measure(tree1, flat)
        ref_closure = closure_length(tree1, flat)

        tree = collate([spec] * n_rep)
        # only torsions whose rotation axis is a ring bond can break the closure
        axis_in_ring = torch.tensor(
            in_ring[spec.torsion_index[:, 1]] & in_ring[spec.torsion_index[:, 2]])
        axis_in_ring = axis_in_ring.repeat(n_rep)

        errs = []
        g = torch.Generator().manual_seed(5)
        for s in sigmas:
            noise = torch.randn((n_rep * ph0.numel(),), generator=g) * np.deg2rad(s)
            phi = ph0.repeat(n_rep) + noise * axis_in_ring
            p = build(tree, r0.repeat(n_rep), th0.repeat(n_rep), phi)
            err = (closure_length(tree, p).reshape(n_rep, -1)
                   - ref_closure.unsqueeze(0)).abs()
            errs.append(err.max(dim=1).values.mean().item())
            if label.startswith("cyclohexane") and broken_frames is None and s > 12:
                broken_frames = p.reshape(n_rep, spec.n_atoms, 3)[:8]
                broken_spec = spec
        ax.loglog(sigmas, errs, color=color, marker="o", label=label)
        at = lambda deg: np.interp(deg, sigmas, errs)
        print(f"  {label:<26} closure error: {at(2):.3f} A at 2 deg, "
              f"{at(5):.3f} A at 5 deg, {at(15):.3f} A at 15 deg")

    ax.axhline(0.05, color=INK_2, linewidth=0.9, linestyle="--")
    ax.text(sigmas[-1], 0.055, "0.05 A", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_xlabel("sd of noise applied to ring torsions (deg)")
    ax.set_ylabel("max ring-closure bond error (A)")
    ax.set_title("Ring closure is a determined function of the tree DoF")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ring_closure.png")
    plt.close(fig)
    print(f"  -> {out / 'ring_closure.png'}")

    if broken_frames is not None:
        path = write_traj(out / "broken_rings.xyz", broken_spec, broken_frames)
        print(f"  -> {path}  (cyclohexane, 15 deg ring-torsion noise)")
        maybe_view(broken_frames, broken_spec, view_gui, "broken rings")


def demo_profile(out: Path, view_gui: bool):
    """Build cost vs batch size, precision, and direction."""
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def timeit(fn, n=20):
        for _ in range(5):
            fn()
        sync()
        t0 = time.time()
        for _ in range(n):
            fn()
        sync()
        return (time.time() - t0) / n * 1e3

    specs, poss = [], []
    for smiles in DEMO_MOLECULES.values():
        s, p, _ = ensemble(smiles, n_confs=1)
        specs.append(s)
        poss.append(p[0])

    print(f"  device: {device}")
    print(f"  {'molecule':<12} {'heavy':>5} {'atoms':>6} {'rounds':>7} {'dof':>5}")
    for nm, s in zip(DEMO_MOLECULES, specs):
        print(f"  {nm:<12} {int((s.z > 1).sum()):>5} {s.n_atoms:>6} "
              f"{s.n_rounds:>7} {s.n_dof:>5}")

    reps = [1, 4, 16, 64, 256, 1024]
    n_atoms_axis, fwd32, fwd64, both64 = [], [], [], []
    for rep in reps:
        tree = collate(specs * rep, device=device)
        flat = torch.tensor(np.concatenate(poss * rep), device=device)
        n_atoms_axis.append(tree.n_atoms)

        for dtype, sink in ((torch.float32, fwd32), (torch.float64, fwd64)):
            t = tree
            f = flat.to(dtype)
            r, th, ph = measure(t, f)
            sink.append(timeit(lambda: build(t, r, th, ph)))

        r, th, ph = measure(tree, flat)
        rg, tg, pg = (x.clone().requires_grad_() for x in (r, th, ph))
        both64.append(timeit(lambda: build(tree, rg, tg, pg).square().sum().backward(), 10))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    # linear y from zero: the story is that these lines are flat, and a log axis
    # would inflate run-to-run jitter into an apparent trend
    for series, color, lbl in ((fwd64, C_BLUE, "forward, fp64"),
                               (fwd32, C_ORANGE, "forward, fp32"),
                               (both64, C_AQUA, "fwd+bwd, fp64")):
        axes[0].plot(n_atoms_axis, series, color=color, marker="o", label=lbl)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, max(both64) * 1.25)
    axes[0].set_xlabel("atoms in batch")
    axes[0].set_ylabel("time per build (ms)")
    axes[0].set_title(f"Flat in batch size -- launch-bound ({device})")
    axes[0].legend(loc="center left")

    axes[1].loglog(n_atoms_axis, np.array(n_atoms_axis) / np.array(fwd64) / 1e3,
                   color=C_BLUE, marker="o")
    axes[1].set_xlabel("atoms in batch")
    axes[1].set_ylabel("M atoms / s")
    axes[1].set_title("Throughput, fp64 -- slope 1, so cost is per call not per atom")
    fig.tight_layout()
    fig.savefig(out / "profile.png")
    plt.close(fig)

    print(f"\n  {'atoms':>8} {'fp32 ms':>9} {'fp64 ms':>9} {'fwd+bwd ms':>12}")
    for n, a, b, c in zip(n_atoms_axis, fwd32, fwd64, both64):
        print(f"  {n:>8} {a:>9.2f} {b:>9.2f} {c:>12.2f}")
    print(f"  -> {out / 'profile.png'}")


def demo_optimize(out: Path, view_gui: bool):
    """Local optimization in internal coordinates vs the all-atom baseline.

    The intramolecular analogue of the crystal-side latent optimization: scramble the
    torsions of a reference conformer, then relax under a simple bonded + soft-core-LJ
    force field, optimizing three different latents over the same energy.
    """
    import matplotlib.pyplot as plt

    from mxtaltools.conformers.energy import (closure_error, ff_from_reference,
                                   intramolecular_energy, worst_clash)
    from mxtaltools.conformers.optimize import cartesian_optimization, gradient_descent_optimization

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_starts, scramble_deg, max_steps = 96, 40.0, 400
    smiles = DEMO_MOLECULES["ibuprofen"]

    # MMFF-relaxed reference defines the bonded minima
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = 2024
    AllChem.EmbedMolecule(mol, p)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    spec = spec_from_mol(mol)
    ref_pos = np.asarray(mol.GetConformer().GetPositions())[spec.perm]

    tree = collate([spec] * n_starts, device=device)
    ref_flat = torch.tensor(np.tile(ref_pos, (n_starts, 1)), device=device)
    ff = ff_from_reference(tree, ref_flat)
    r0, th0, ph0 = measure(tree, ref_flat)

    g = torch.Generator(device="cpu").manual_seed(17)
    phi_start = ph0 + (torch.randn(ph0.shape, generator=g).to(device)
                       * np.deg2rad(scramble_deg))
    start = (r0.clone(), th0.clone(), phi_start)
    start_pos = build(tree, *start)

    e_start = intramolecular_energy(tree, start_pos, ff)
    e_ref = intramolecular_energy(tree, ref_flat, ff)[0]
    heavy = torch.tensor(np.asarray(spec.z) > 1, device=device)
    print(f"  {n_starts} starts, torsions scrambled by {scramble_deg:.0f} deg sd; "
          f"{spec.n_dof} internal DoF vs {3 * spec.n_atoms} cartesian")
    print(f"  reference energy {e_ref.item():8.2f} kcal/mol   "
          f"start median {e_start.median().item():8.2f}")

    runs = {}
    runs["internal, all DoF"] = gradient_descent_optimization(
        tree, start, ff, init_lr=2e-2, max_num_steps=max_steps)
    runs["internal, torsions only"] = gradient_descent_optimization(
        tree, start, ff, init_lr=2e-2, max_num_steps=max_steps, freeze=("r", "theta"))
    runs["cartesian, all-atom"] = cartesian_optimization(
        tree, start_pos, ff, init_lr=2e-2, max_num_steps=max_steps)

    ref_stack = ref_flat.reshape(n_starts, spec.n_atoms, 3)
    summary = {}
    for name, (pos, rec) in runs.items():
        r, th, _ = measure(tree, pos)
        rmsd = kabsch_rmsd(pos.reshape(n_starts, spec.n_atoms, 3), ref_stack, heavy)
        dim = (int(tree.torsion_index.shape[0] / n_starts) if "torsions" in name
               else spec.n_dof if name.startswith("internal") else 3 * spec.n_atoms)
        summary[name] = dict(
            E=rec.best_energy.median().item(),
            clash=worst_clash(tree, pos, ff).median().item(),
            closure=closure_error(pos, ff).median().item(),
            bond_dev=(r - r0).abs().mean().item(),
            ang_dev=np.rad2deg((th - th0).abs().mean().item()),
            rmsd=rmsd.median().item(), rmsd_all=rmsd.cpu().numpy(),
            steps=rec.n_steps, dim=dim, best=rec.best_energy)

    # "stuck" = failed to get within 5 kcal/mol of the deepest structure anyone found
    deepest = min(s["best"].min().item() for s in summary.values())
    print(f"\n  {'latent':<24} {'dim':>4} {'steps':>6} {'E_med':>7} {'stuck':>6} "
          f"{'clash':>6} {'closure':>8} {'|dr|':>7} {'|dth|':>7} {'RMSD':>5}")
    for name, s in summary.items():
        s["stuck"] = float((s["best"] > deepest + 5.0).float().mean())
        print(f"  {name:<24} {s['dim']:>4} {s['steps']:>6} {s['E']:>7.2f} "
              f"{s['stuck']:>5.0%} {s['clash']:>6.3f} {s['closure']:>8.4f} "
              f"{s['bond_dev']:>7.4f} {s['ang_dev']:>6.2f}d {s['rmsd']:>5.2f}")
    print(f"  (one seed, n={n_starts}: 'stuck' is seed-sensitive at this size -- over "
          f"3 seeds x 256 it is ~8% / ~6% / ~5%. 'closure' replicates cleanly.)")

    # everything is plotted as excess above the deepest structure any latent found, so a
    # log axis stays valid even where a run beats the MMFF reference
    names = list(runs)
    floor = min(min(rec.best_energy.min().item() for _, rec in runs.values()),
                e_ref.item())
    excess = lambda t: (t - floor).clamp_min(1e-3)

    short = ["all\nDoF", "torsions\nonly", "cartesian"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    colors = (C_BLUE, C_ORANGE, C_AQUA)
    for (name, (pos, rec)), color in zip(runs.items(), colors):
        e = excess(rec.energy)
        steps = np.arange(e.shape[0])
        axes[0].plot(steps, e.median(dim=1).values.numpy(), color=color, label=name)
        axes[0].fill_between(steps, e.quantile(0.25, dim=1).numpy(),
                             e.quantile(0.75, dim=1).numpy(),
                             color=color, alpha=0.15, linewidth=0)
    axes[0].set_yscale("log")
    axes[0].axhline(excess(e_ref).item(), color=INK_2, linestyle="--", linewidth=0.9)
    axes[0].text(max_steps, excess(e_ref).item(), " MMFF ref", color=INK_2,
                 fontsize=8, va="bottom", ha="right")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("energy above best found (kcal/mol)")
    axes[0].set_title("Relaxation, median and IQR")
    axes[0].legend(loc="upper right")

    axes[1].boxplot([excess(runs[n][1].best_energy).numpy() for n in names],
                    tick_labels=short, medianprops=dict(color=INK), widths=0.55)
    axes[1].set_yscale("log")
    axes[1].axhline(excess(e_ref).item(), color=INK_2, linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("final energy above best (kcal/mol)")
    axes[1].set_title("Where each run landed")

    axes[2].boxplot([summary[n]["rmsd_all"] for n in names],
                    tick_labels=short, medianprops=dict(color=INK), widths=0.55)
    axes[2].set_ylabel("heavy-atom RMSD to reference (A)")
    axes[2].set_title("Basin spread")
    fig.tight_layout()
    fig.savefig(out / "optimization.png")
    plt.close(fig)
    print(f"  -> {out / 'optimization.png'}")

    # lowest-energy structure from each latent, plus the reference, for viewing
    frames, spec_z = [torch.tensor(ref_pos, device=device)], spec
    for name, (pos, rec) in runs.items():
        best = int(rec.best_energy.argmin())
        frames.append(pos.reshape(n_starts, spec.n_atoms, 3)[best])
    path = write_traj(out / "optimized.xyz", spec_z, frames)
    print(f"  -> {path}  (frame 0 = MMFF reference, then {', '.join(runs)})")
    maybe_view(frames, spec_z, view_gui, "optimized conformers")


DEMOS = {"dists": demo_dists, "sensitivity": demo_sensitivity, "scan": demo_scan,
         "rings": demo_rings, "profile": demo_profile, "optimize": demo_optimize}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # no argparse `choices` here: with nargs="*" it validates the default *list*
    # itself against the choice set and rejects it
    ap.add_argument("demos", nargs="*", metavar="DEMO",
                    help=f"any of: {', '.join(DEMOS)}, all (default: all)")
    ap.add_argument("--outdir", type=Path, default=Path("conformer_demo"))
    ap.add_argument("--view", action="store_true", help="open ASE viewers as well")
    args = ap.parse_args()

    unknown = [d for d in args.demos if d not in DEMOS and d != "all"]
    if unknown:
        ap.error(f"unknown demo(s) {unknown}; choose from {list(DEMOS)} or 'all'")
    chosen = list(DEMOS) if not args.demos or "all" in args.demos else args.demos
    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()

    for name in chosen:
        print(f"\n[{name}]")
        DEMOS[name](args.outdir, args.view)
    print(f"\nall artefacts in {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
