"""
Correctness and scale tests for the internal-coordinate conformer builder.

Run directly (``python -m conformers.tests.test_conformers`` from the repo root)
or under pytest.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mxtaltools.conformers import (build, closure_length, collate, log_jacobian, measure,
                        nonbonded_pairs, spec_from_smiles)
from mxtaltools.conformers.topology import reference_dihedral

torch.set_default_dtype(torch.float64)

ACYCLIC = ["CC", "CCO", "CCCC", "CC(C)(C)C", "CC(=O)NC", "CCOC(=O)C",
           "CCCCCCO", "CC(C)CC(N)C(=O)O", "OCC(O)C(O)C(O)C(O)CO"]
CYCLIC = ["c1ccccc1", "C1CCCCC1", "CCOc1ccccc1", "c1ccc2ccccc2c1",
          "C1CC2CCC1CC2", "Cc1ccc(cc1)S(=O)(=O)N"]
LINEAR = ["CC#N", "CC#CC", "N#Cc1ccccc1", "CCC#N"]
ALL = ACYCLIC + CYCLIC + LINEAR


def _reference(smiles, seed=7):
    """(spec, positions-in-placement-order, mol) for one embedded molecule."""
    mol, spec = spec_from_smiles(smiles, seed=seed)
    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    return spec, torch.tensor(pos[spec.perm]), mol


def _distmat(pos):
    # the default cdist compute_mode falls back to the matmul formulation above 25
    # rows, which costs ~float32 precision and would mask real error at this scale
    return torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")


def test_dof_count():
    """The tree parameterisation is exactly minimal: 3N - 6, no redundancy."""
    for smiles in ALL:
        spec, _, _ = _reference(smiles)
        tree = collate([spec])
        total = (tree.bond_index.shape[0] + tree.angle_index.shape[0]
                 + tree.torsion_index.shape[0])
        assert total == 3 * spec.n_atoms - 6, f"{smiles}: {total} != {3 * spec.n_atoms - 6}"


def test_roundtrip_exact():
    """measure -> build reproduces the geometry to machine precision.

    Compared via pairwise distance matrices, which are SE(3)-invariant, so this
    tests the geometry rather than the frame convention.
    """
    worst = 0.0
    for smiles in ALL:
        spec, pos, _ = _reference(smiles)
        tree = collate([spec])
        rebuilt = build(tree, *measure(tree, pos))
        err = (_distmat(rebuilt) - _distmat(pos)).abs().max().item()
        worst = max(worst, err)
        assert err < 1e-9, f"{smiles}: distance-matrix error {err:.3e}"
    return worst


def test_dihedral_convention_matches_rdkit():
    """Our signed dihedral agrees with RDKit for every proper torsion."""
    worst = 0.0
    for smiles in ALL:
        spec, pos, mol = _reference(smiles)
        tree = collate([spec])
        _, _, phi = measure(tree, pos)
        proper = np.flatnonzero(spec.torsion_is_proper)
        for i in proper[:40]:
            ref = reference_dihedral(mol, spec, int(i))
            d = abs(((phi[i].item() - ref + np.pi) % (2 * np.pi)) - np.pi)
            worst = max(worst, d)
            assert d < 1e-8, f"{smiles} torsion {i}: {phi[i].item()} vs rdkit {ref}"
    return worst


def test_batched_matches_individual():
    """Batching is pure bookkeeping: identical geometry to one-at-a-time."""
    specs, poss = zip(*[(s, p) for s, p, _ in map(_reference, ALL)])
    tree = collate(list(specs))
    pos_cat = torch.cat(poss)
    batched = build(tree, *measure(tree, pos_cat))

    worst = 0.0
    for m, (spec, pos) in enumerate(zip(specs, poss)):
        single = collate([spec])
        alone = build(single, *measure(single, pos))
        lo, hi = int(tree.ptr[m]), int(tree.ptr[m + 1])
        worst = max(worst, (batched[lo:hi] - alone).abs().max().item())
    assert worst < 1e-10, f"batched/individual mismatch {worst:.3e}"
    return worst


def test_log_jacobian():
    """log|J| matches the numerical determinant, up to the frame-fixing factor.

    ``log_jacobian`` returns the BAT Jacobian ``prod_k r_k^2 sin(theta_k)``, which is
    the factor relating an internal-coordinate density to the Cartesian Boltzmann
    measure. Differentiating ``build`` instead gives the determinant of the map into
    the *frame-fixed representative*, where the seed convention has already spent 6
    DoF: atom 1 is pinned to +x so it contributes 1 rather than r_1^2, and atom 2 to
    the xy-plane so it contributes r_2 rather than r_2^2 sin(theta_2). Every atom from
    slot 3 on is a genuine spherical placement and contributes r^2 sin(theta), so this
    validates all 3N-9 of the bulk terms exactly.
    """
    worst = 0.0
    for smiles in ["CC", "CCO", "CC#N", "C1CCCCC1", "CC(C)(C)C"]:
        spec, pos, _ = _reference(smiles)
        tree = collate([spec])
        r, theta, phi = measure(tree, pos)
        nb, na = r.numel(), theta.numel()

        def reduced(dof):
            p = build(tree, dof[:nb], dof[nb:nb + na], dof[nb + na:])
            return torch.cat([p[1, :1], p[2, :2], p[3:].reshape(-1)])

        dof = torch.cat([r, theta, phi])
        jac = torch.autograd.functional.jacobian(reduced, dof)
        assert jac.shape[0] == jac.shape[1] == 3 * spec.n_atoms - 6
        numeric = torch.linalg.slogdet(jac)[1].item()

        # r_1 is the first bond, r_2/theta_2 the second bond and first angle
        frame_fix = (2 * torch.log(r[0]) + torch.log(r[1])
                     + torch.log(torch.sin(theta[0]))).item()
        analytic = log_jacobian(tree, r, theta)[0].item() - frame_fix
        worst = max(worst, abs(numeric - analytic))
        assert abs(numeric - analytic) < 1e-8, \
            f"{smiles}: analytic {analytic:.10f} vs numeric {numeric:.10f}"
    return worst


def test_differentiable():
    """gradcheck through the full construction."""
    spec, pos, _ = _reference("CCO")
    tree = collate([spec])
    r, theta, phi = measure(tree, pos)

    def fn(r_, t_, p_):
        return build(tree, r_, t_, p_)

    assert torch.autograd.gradcheck(
        fn, (r.requires_grad_(), theta.requires_grad_(), phi.requires_grad_()),
        eps=1e-6, atol=1e-7)


def test_ring_closure():
    """Non-tree bonds are found, and reconstruction closes them exactly."""
    counts = {}
    for smiles in CYCLIC:
        spec, pos, mol = _reference(smiles)
        tree = collate([spec])
        n_rings = mol.GetRingInfo().NumRings()
        n_broken = tree.broken_bond_index.shape[0]
        assert n_broken == mol.GetNumBonds() - (spec.n_atoms - 1)
        rebuilt = build(tree, *measure(tree, pos))
        err = (closure_length(tree, rebuilt) - closure_length(tree, pos)).abs().max().item()
        assert err < 1e-9, f"{smiles}: closure drift {err:.3e}"
        counts[smiles] = (n_rings, n_broken)
    return counts


def test_acyclic_has_no_broken_bonds():
    for smiles in ACYCLIC:
        spec, _, _ = _reference(smiles)
        assert collate([spec]).broken_bond_index.shape[0] == 0, smiles


def test_nonbonded_exclusions():
    """1-2/1-3/1-4 exclusion counts agree with an independent RDKit path count."""
    from rdkit import Chem
    for smiles in ["CCCC", "c1ccccc1", "CC(C)CC(N)C(=O)O"]:
        spec, _, mol = _reference(smiles)
        tree = collate([spec])
        pair_index, _, _ = nonbonded_pairs(tree, min_separation=4)
        dmat = Chem.GetDistanceMatrix(mol)[np.ix_(spec.perm, spec.perm)]
        expected = int((np.triu(dmat, k=1) >= 4).sum())
        assert pair_index.shape[0] == expected, \
            f"{smiles}: {pair_index.shape[0]} pairs vs expected {expected}"


def _tree_shape(mol):
    """The element-labelled tree: what each slot *means*, independent of which atom fills it."""
    from mxtaltools.conformers.topology import spec_from_mol
    s = spec_from_mol(mol, use_geometry=False)
    z = np.asarray(s.z)
    lab = lambda i: int(z[i]) if i >= 0 else -1
    return tuple((lab(s.ref_a[k]), lab(s.ref_b[k]), lab(s.ref_c[k]), int(z[k]),
                  int(s.round_id[k])) for k in range(s.n_atoms))


def test_tree_shape_is_canonical():
    """Permuting input atom numbering must not change the tree.

    The DoF *values* can still differ, because a graph automorphism (the three
    hydrogens of a methyl) picks out geometrically distinct atoms -- that is not
    resolvable from the graph and is not what this pins down. What must hold is that
    the tree *shape* is a pure function of the molecular graph, so a given slot always
    carries the same chemical meaning and the model's heads always see the same
    elements. Before canonical-rank tie-breaking this failed on most permutations.
    """
    from rdkit import Chem

    rng = np.random.default_rng(0)
    for smiles in ["CCO", "CCOc1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(C)(C)C",
                   "c1ccc2ccccc2c1", "C1CCCCC1", "CC#N", "Cc1ccc(cc1)S(=O)(=O)N"]:
        _, _, mol = _reference(smiles)
        ref = _tree_shape(mol)
        for _ in range(16):
            perm = rng.permutation(mol.GetNumAtoms())
            shuffled = Chem.RenumberAtoms(mol, [int(i) for i in perm])
            assert _tree_shape(shuffled) == ref, f"{smiles}: tree changed under relabelling"


def test_graph_and_mol_paths_agree():
    """spec_from_graph reproduces spec_from_mol, and tolerates edge_index layouts."""
    from mxtaltools.conformers.topology import spec_from_graph, spec_from_mol

    for smiles in ["CCOc1ccccc1", "CC(C)CC(N)C(=O)O"]:
        _, _, mol = _reference(smiles)
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
        ei = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
                       for b in mol.GetBonds()]).T          # [2, E], PyG layout
        ref = spec_from_mol(mol, use_geometry=False)
        for variant in (ei, ei.T, np.concatenate([ei, ei[::-1]], axis=1)):
            got = spec_from_graph(z, variant, use_geometry=False)
            assert np.array_equal(got.ref_a, ref.ref_a), smiles
            assert np.array_equal(got.ref_b, ref.ref_b), smiles
            assert np.array_equal(got.perm, ref.perm), smiles


def test_soft_core_lj():
    """The exponential core matches LJ in value and slope at sigma, and stays finite."""
    from mxtaltools.conformers.energy import soft_core_lj

    sigma = torch.tensor([3.4], dtype=torch.float64)
    eps = torch.tensor([0.1], dtype=torch.float64)
    k = 2.5

    r = sigma.clone().requires_grad_()
    v = soft_core_lj(r, sigma, eps, k)
    (grad,) = torch.autograd.grad(v.sum(), r)
    assert v.abs().item() < 1e-12, f"LJ(sigma) should be 0, got {v.item()}"
    expected = (-24.0 * eps / sigma).item()
    assert abs(grad.item() - expected) < 1e-8, f"slope {grad.item()} vs {expected}"

    # C1 across the switch: one-sided limits agree
    h = 1e-7
    below = soft_core_lj(sigma - h, sigma, eps, k)
    above = soft_core_lj(sigma + h, sigma, eps, k)
    assert (below - above).abs().item() < 1e-6

    at_zero = soft_core_lj(torch.tensor([1e-12], dtype=torch.float64), sigma, eps, k)
    analytic = (24.0 * eps / k * (np.exp(k) - 1)).item()
    assert torch.isfinite(at_zero).all()
    assert abs(at_zero.item() - analytic) < 1e-3, f"{at_zero.item()} vs {analytic}"


def test_energy_minimised_at_reference():
    """Bonded terms vanish at the geometry they were fitted to."""
    from mxtaltools.conformers.energy import ff_from_reference, intramolecular_energy

    spec, pos, _ = _reference("CC(C)CC(N)C(=O)O")
    tree = collate([spec])
    ff = ff_from_reference(tree, pos)
    _, parts = intramolecular_energy(tree, pos, ff, components=True)
    assert parts["bond"].abs().item() < 1e-16, parts["bond"]
    assert parts["angle"].abs().item() < 1e-16, parts["angle"]
    assert torch.isfinite(parts["lj"]).all()


def test_energy_gradient_through_builder():
    """dE/d(internal DoF) exists and is finite -- the whole optimization path."""
    from mxtaltools.conformers.energy import ff_from_reference, intramolecular_energy

    spec, pos, _ = _reference("CCOc1ccccc1")
    tree = collate([spec])
    ff = ff_from_reference(tree, pos)
    r, theta, phi = (x.clone().requires_grad_() for x in measure(tree, pos))
    energy = intramolecular_energy(tree, build(tree, r, theta, phi), ff)
    grads = torch.autograd.grad(energy.sum(), (r, theta, phi))
    for name, g in zip(("r", "theta", "phi"), grads):
        assert torch.isfinite(g).all(), f"non-finite gradient w.r.t. {name}"
    # at the reference the bonded terms are stationary, so only LJ drives phi
    assert grads[2].abs().max() > 0, "phi gradient should be non-zero (LJ acts on it)"


def _bench(n_mols=1024, device="cuda", dtype=torch.float32):
    from mxtaltools.conformers.topology import spec_from_mol
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rng = np.random.default_rng(0)
    pool = ["CC(C)CC(N)C(=O)O", "CCOc1ccccc1", "OCC(O)C(O)C(O)C(O)CO",
            "Cc1ccc(cc1)S(=O)(=O)N", "CCCCCCO", "c1ccc2ccccc2c1"]
    t0 = time.time()
    specs, poss = [], []
    for i in range(n_mols):
        mol = Chem.AddHs(Chem.MolFromSmiles(pool[int(rng.integers(len(pool)))]))
        p = AllChem.ETKDGv3()
        p.randomSeed = int(i)
        AllChem.EmbedMolecule(mol, p)
        spec = spec_from_mol(mol)
        specs.append(spec)
        poss.append(np.asarray(mol.GetConformer().GetPositions())[spec.perm])
    prep = time.time() - t0

    t0 = time.time()
    tree = collate(specs, device=device)
    collate_t = time.time() - t0
    pos = torch.tensor(np.concatenate(poss), dtype=dtype, device=device)
    r, theta, phi = measure(tree, pos)

    for _ in range(3):
        build(tree, r, theta, phi)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        out = build(tree, r, theta, phi)
    if device == "cuda":
        torch.cuda.synchronize()
    per_call = (time.time() - t0) / 20

    err = (_distmat(out[:tree.ptr[1]].double()) - _distmat(pos[:tree.ptr[1]].double())
           ).abs().max().item()
    return dict(n_mols=n_mols, n_atoms=tree.n_atoms, n_rounds=len(tree.rounds),
                n_dof=int(r.numel() + theta.numel() + phi.numel()),
                rdkit_prep_s=prep, collate_s=collate_t, build_ms=per_call * 1e3,
                fp32_err=err)


if __name__ == "__main__":
    print(f"roundtrip max distance-matrix error : {test_roundtrip_exact():.3e}")
    print(f"dihedral vs rdkit, max              : {test_dihedral_convention_matches_rdkit():.3e}")
    print(f"batched vs individual, max          : {test_batched_matches_individual():.3e}")
    print(f"log|J| vs numeric det, max          : {test_log_jacobian():.3e}")
    test_dof_count();                print("dof count == 3N-6                   : ok")
    test_acyclic_has_no_broken_bonds(); print("acyclic: no broken bonds            : ok")
    test_nonbonded_exclusions();     print("nonbonded 1-2/1-3/1-4 exclusions    : ok")
    test_differentiable();           print("gradcheck                           : ok")
    test_tree_shape_is_canonical();  print("tree shape canonical under relabel  : ok")
    test_graph_and_mol_paths_agree(); print("spec_from_graph == spec_from_mol    : ok")
    test_soft_core_lj();             print("soft-core LJ: C1 at sigma, finite   : ok")
    test_energy_minimised_at_reference(); print("bonded energy zero at reference     : ok")
    test_energy_gradient_through_builder(); print("dE/d(internal) through builder      : ok")
    for k, (nr, nb) in test_ring_closure().items():
        print(f"  rings {k:<22} {nr} rings -> {nb} broken bonds")

    if torch.cuda.is_available():
        print("\nscale (fp32, cuda):")
        for k, v in _bench().items():
            print(f"  {k:<14} {v}")
