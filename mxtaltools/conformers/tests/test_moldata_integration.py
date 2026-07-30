"""
Internal-coordinate parameterisation as wired into the mxtaltools data classes.

Runs against a real mini dataset rather than synthetic molecules, because the things
most likely to break are integration details -- atom-index conventions, PyG collation,
and the fact that stored MolData carries no bond topology at all.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = Path(__file__).resolve().parents[3] / "mini_datasets" / "mini_qm9_dataset.pt"


def _mols(n=16, min_atoms=8, double=True):
    if not DATASET.exists():
        pytest.skip(f"{DATASET} not present")
    data = torch.load(DATASET, weights_only=False)
    out = [m for m in data if int(m.num_atoms) >= min_atoms][:n]
    for m in out:
        if double:
            m.pos = m.pos.double()
        m.build_conformer_tree()
    return out


def _distmat(p):
    return torch.cdist(p, p, compute_mode="donot_use_mm_for_euclid_dist")


def test_dof_count_on_real_data():
    """3N-6 exactly, per molecule and in aggregate."""
    mols = _mols()
    for m in mols:
        r, theta, phi = m.internal_dof()
        n = int(m.num_atoms)
        assert (r.numel(), theta.numel(), phi.numel()) == (n - 1, n - 2, n - 3)
        assert r.numel() + theta.numel() + phi.numel() == 3 * n - 6


def test_bond_perception_matches_chemistry():
    """Perceived tree bonds are real bonds, not 1-3 contacts.

    This is the check that caught the original index-space bug: tree indices were in
    placement-slot space while `pos` was in atom order, which produced 4 A "bonds".
    """
    mols = _mols()
    for m in mols:
        r, _, _ = m.internal_dof()
        assert r.max() < 2.6, f"implausible tree bond length {r.max():.2f} A"
        assert r.min() > 0.7, f"implausible tree bond length {r.min():.2f} A"


def test_roundtrip_single_and_batched():
    """measure -> set_internal_dof reproduces geometry, alone and collated."""
    mols = _mols()
    for m in mols:
        ref = m.pos.clone()
        m.set_internal_dof(m.internal_dof_vector())
        assert (_distmat(m.pos) - _distmat(ref)).abs().max() < 1e-12

    batch = collate_data_list(_mols())
    ref = batch.pos.clone()
    batch.set_internal_dof(batch.internal_dof_vector())
    for i in range(batch.num_graphs):
        sel = batch.batch == i
        assert (_distmat(batch.pos[sel]) - _distmat(ref[sel])).abs().max() < 1e-12


def test_collation_needs_no_custom_code():
    """PyG batches the tree via the existing __inc__/__cat_dim__ conventions.

    Every index array is named ``*_index`` and stored [k, n], so it inherits node-offset
    incrementing and last-dim concatenation for free. Batched measurement must therefore
    equal per-molecule measurement.
    """
    mols = _mols()
    batch = collate_data_list(mols)
    n_atoms = sum(int(m.num_atoms) for m in mols)
    assert batch.tree_bond_index.shape == (2, n_atoms - len(mols))
    assert batch.tree_angle_index.shape == (3, n_atoms - 2 * len(mols))
    assert batch.tree_torsion_index.shape == (4, n_atoms - 3 * len(mols))

    for got, want in zip(batch.internal_dof(),
                         [torch.cat([m.internal_dof()[i] for m in mols]) for i in range(3)]):
        assert (got - want).abs().max() < 1e-12


def test_frozen_bonds_stay_frozen():
    """Moving only phi leaves every tree bond length untouched -- structurally, not by penalty."""
    batch = collate_data_list(_mols())
    r, theta, phi = batch.internal_dof()
    g = torch.Generator().manual_seed(0)
    batch.set_internal_dof((r, theta, phi + torch.randn(phi.shape, generator=g).double()))
    assert (batch.internal_dof()[0] - r).abs().max() < 1e-12
    assert (batch.internal_dof()[1] - theta).abs().max() < 1e-12


def test_energy_zero_at_reference_and_closure_diagnostic():
    """Bonded terms vanish on the conformer they were fitted to; closure drift is zero."""
    batch = collate_data_list(_mols())
    _, parts = batch.intramolecular_energy(components=True)
    assert parts["bond"].abs().max() < 1e-16
    assert parts["angle"].abs().max() < 1e-16
    assert torch.isfinite(parts["lj"]).all()

    closure = batch.ring_closure_error()
    assert closure.shape == (batch.num_graphs,), "closure error must be per-molecule"
    assert closure.max() < 1e-12

    # scrambling torsions opens rings, and the diagnostic must see it
    r, theta, phi = batch.internal_dof()
    g = torch.Generator().manual_seed(1)
    batch.set_internal_dof((r, theta, phi + torch.randn(phi.shape, generator=g).double()))
    assert batch.ring_closure_error().max() > 0.05


def test_log_jacobian_is_per_molecule():
    batch = collate_data_list(_mols())
    lj = batch.internal_log_jacobian()
    assert lj.shape == (batch.num_graphs,)
    assert torch.isfinite(lj).all()


def test_readout_runs():
    m = _mols(n=1)[0]
    text = m.format_internal_dof(max_rows=5)
    assert "internal DoF for molecule 0" in text
    assert "bond" in text and "angle" in text and "dihedral" in text


def test_lazy_build_and_no_init_cost():
    """Accessors build the tree on demand; construction does not, unless asked."""
    from mxtaltools.dataset_utils.data_classes import MolData

    src = _mols(n=1)[0]
    lazy = MolData(z=src.z.clone(), pos=src.pos.clone())
    assert not lazy.has_conformer_tree
    lazy.internal_dof()
    assert lazy.has_conformer_tree

    eager = MolData(z=src.z.clone(), pos=src.pos.clone(), build_tree=True)
    assert eager.has_conformer_tree


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name:<52} ok")
