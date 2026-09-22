"""
LEVEL 1 GROUND TRUTH: our MACE route against a stock MACE workflow.

The MACE twin of test_uma_vs_stock_fairchem.py, and it exists for the same reason.
Most other MACE gates here are INTERNAL parity -- the hoisted/device builder against
the list path -- and a bug that moves both of our routes equally is invisible to
those.

ONE EXISTING GATE IS ALREADY EXTERNAL, and this file does not supersede it:
`test_pbc_neighbours.py` compares the production `batched_pbc_neighbour_list`
against `mace.data.get_neighborhood` (matscipy) on exact edge sets, CPU-only and
with no model. That independently validates the GRAPH. What no gate covered before
this file is the END-TO-END ENERGY: the graph, the batch construction, the shifts and
the model invocation together, against a workflow we did not write.

This file scores the same crystals through `mace.calculators.MACECalculator` --
upstream's own ASE calculator, loading the checkpoint itself from the path, building
its own neighbour list -- and compares against what our production path returns.

WHY THIS IS A CLEANER COMPARISON THAN THE UMA ONE. fairchem's ASE route wraps
positions internally, so that test had to reason about wrapping being a lattice
symmetry. MACE's `get_neighborhood` handles arbitrary (unwrapped) positions directly
-- which is exactly why `test_pbc_neighbours.py` can compare our builder against it
on real unwrapped cells and demand exact edge-set equality. So here both stacks see
the SAME coordinates and the comparison is direct.

TWO HARD CONSTRAINTS, INHERITED AND NOT NEGOTIABLE. Both are documented at length in
test_mace_gpu_real_batches.py and both were discovered expensively:

  BATCH SIZE. The acridine MACE model OOMs above ~5 crystals. MAX_GRAPHS is 4.
  KERNEL DURATION. The forward is ~425 ms at 2 graphs and near-fixed in batch size,
  which puts 16 graphs across the 2 s Windows WDDM watchdog. Two runs of an earlier
  sweep BSOD'd the machine that way with the card verified idle -- a TDR timeout no
  pre-flight can prevent. The only protection is staying small.

The stock calculator scores ONE structure per call, so it is inherently safe; it is
our batched path that has to be held down.

ELEMENT TABLE. The acridine model carries a restricted `atomic_numbers` table, so the
fixture keeps only crystals whose elements it covers. Without that filter the failure
is an index error deep in the model rather than an honest skip.

RUN (needs a free GPU and the checkpoint):
    MACE_CHECKPOINT=D:/crystal_datasets/acr_112025_mh1_stagetwo.model \
        pytest tests/test_mace_vs_stock.py -q -rs
"""
import os

import numpy as np
import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
MODEL_ENV = 'MACE_CHECKPOINT'

#: Never raise this. See KERNEL DURATION above -- the ceiling is a BSOD, not a slow test.
MAX_GRAPHS = 4

EV_TO_KJ = 96.485


@pytest.fixture(scope='module')
def checkpoint_path():
    path = os.environ.get(MODEL_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f'set {MODEL_ENV} to a MACE checkpoint to run this file')
    return path


@pytest.fixture(scope='module')
def our_model(checkpoint_path, gpu):
    """Loaded exactly the way production loads it."""
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    return load_mace_model(checkpoint_path, gpu, torch.float32)


@pytest.fixture(scope='module')
def stock_calculator(checkpoint_path, gpu):
    """
    Upstream's own ASE calculator, pointed at the checkpoint PATH so it performs its
    own load rather than sharing our model object. Sharing the object would still
    exercise the batch construction and neighbour list, but not the load, and the
    point of a ground truth is to hand off as early as possible.
    """
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=checkpoint_path, device=gpu,
                          default_dtype='float32')


@pytest.fixture(scope='module')
def crystals(our_model):
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    allowed = set(int(z) for z in our_model.atomic_numbers)
    good = []
    for c in torch.load(DATASET, weights_only=False, map_location='cpu'):
        try:
            if not set(int(z) for z in c.z).issubset(allowed):
                continue
            p = collate_data_list([c.clone()])
            p.pose_aunit(std_orientation=False)
            p.build_unit_cell()
            good.append(c)
        except Exception:
            pass
    if len(good) < MAX_GRAPHS:
        pytest.skip(f"only {len(good)} crystals in this model's element table")
    return good


@pytest.fixture(scope='module')
def any_crystals():
    """Buildable crystals with NO element filter and NO model, so the convention pin
    below is genuinely CPU-runnable. The `crystals` fixture needs `our_model` for the
    element table, which drags in the GPU and the checkpoint; a cell-convention check
    does not care what the elements are, and gating it behind a GPU meant the one
    thing this file claims to pin on CPU could not run in CI."""
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    good = []
    for c in torch.load(DATASET, weights_only=False, map_location='cpu'):
        try:
            p = collate_data_list([c.clone()])
            p.pose_aunit(std_orientation=False)
            p.build_unit_cell()
            good.append(c)
        except Exception:
            pass
    if len(good) < MAX_GRAPHS:
        pytest.skip(f'only {len(good)} buildable crystals in the fixture')
    return good


def _built(crystals, n, device):
    b = collate_data_list([c.clone() for c in crystals[:n]]).to(device)
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    b.box_analysis()
    return b


def _ase_unit_cells(batch):
    """ase.Atoms from `unit_cell_pos` and `T_fc` directly -- the same tensors the model
    is handed, with no help from our MACE converter. `T_fc` is a column-vector
    operator, so its transpose's rows are the lattice vectors, which is ASE's
    convention. Pinned against ASE's own cellpar in the CPU test below."""
    from ase import Atoms

    z = batch.z.cpu().numpy()
    node_batch = batch.batch.cpu().numpy()
    ucell_batch = batch.unit_cell_batch.cpu().numpy()
    pos = batch.unit_cell_pos.detach().cpu().numpy()
    cells = batch.T_fc.detach().cpu().numpy()
    sym_mult = batch.sym_mult.cpu().numpy()

    out = []
    for i in range(batch.num_graphs):
        numbers = np.tile(z[node_batch == i], int(sym_mult[i]))
        p = pos[ucell_batch == i]
        assert len(numbers) == len(p), (
            f'crystal {i}: {len(numbers)} tiled atomic numbers against {len(p)} '
            f'unit-cell positions -- the batch is inconsistent')
        out.append(Atoms(numbers=numbers, positions=p, cell=cells[i].T, pbc=True))
    return out


def _stock_energies(atoms_list, calculator):
    """One ASE call per crystal, on a fresh copy so the calculator's result cache is
    keyed on a distinct object and cannot serve a stale energy."""
    out = []
    for atoms in atoms_list:
        a = atoms.copy()
        a.calc = calculator
        out.append(float(a.get_potential_energy()))
    return torch.tensor(out, dtype=torch.float64)


def _our_energies(batch, model):
    from mxtaltools.mlip_interfaces.AL_mace_utils import compute_crystal_mace_on_mxt_batch
    return compute_crystal_mace_on_mxt_batch(
        batch.clone(), model, std_orientation=False, pbc=True,
        force_rebuild=False).detach().double().cpu()


def _decompose(delta_ev, batch, label):
    """
    Split the disagreement into BIAS / NOISE / OUTLIER, in reward units.

    WHY SIGN MATTERS AND WHY EVERY OTHER STATISTIC HERE DESTROYS IT. Every assertion
    in this file takes `.abs()` first, so a uniform +d offset and symmetric noise of
    width d are indistinguishable in all of them. Those two are not equally bad:

      * a CONSTANT energy bias cancels exactly in a Boltzmann target -- p(x) is
        proportional to exp(-E/T), so E -> E + c leaves every relative probability
        untouched. It shifts log Z and corrupts absolute energies, nothing else.
      * a STRUCTURE-CORRELATED bias tilts the landscape and is the genuinely
        damaging case: it reweights basins against each other.
      * a LARGE SINGLE-POINT error invents or destroys one minimum, which is how a
        sampler ends up chasing a structure that is not there.
      * symmetric NOISE small against kT (2.5 kJ/mol) is a reward noise floor and
        is tolerable.

    Reported, not asserted. Setting a bias bar before measuring one would be the
    anti-pattern this suite's method doc names -- a bar whose noise floor and defect
    scale are both unknown. Set them from the first runs of this line.
    """
    d = (delta_ev * EV_TO_KJ / (batch.sym_mult * batch.z_prime).double().cpu())
    n = d.numel()
    bias = d.mean().item()
    noise = d.std().item() if n > 1 else 0.0
    outlier = d.abs().max().item()
    # is the error correlated with the crystal's own energy? a tilt, not an offset.
    print(f'[{label}] per molecule kJ/mol -- BIAS(signed mean) {bias:+.4f}  '
          f'NOISE(std) {noise:.4f}  OUTLIER(max abs) {outlier:.4f}  n={n}')
    print(f'[{label}] signed per crystal: {[round(float(x), 4) for x in d]}')
    return {'bias': bias, 'noise': noise, 'outlier': outlier}

def _per_molecule_kj(delta_ev, batch):
    denom = (batch.sym_mult * batch.z_prime).double().cpu()
    return delta_ev.abs() * EV_TO_KJ / denom


# ------------------------------------------------------------------ CPU precondition

def test_ase_cell_convention_round_trips(any_crystals):
    """The one silent way this file could be wrong: if `T_fc.T` is not the
    lattice-vector matrix ASE expects, both stacks score self-consistent but DIFFERENT
    crystals, and a convention error reads as a model disagreement -- or cancels and
    passes."""
    batch = _built(any_crystals, MAX_GRAPHS, 'cpu')
    for i, atoms in enumerate(_ase_unit_cells(batch)):
        cellpar = atoms.cell.cellpar()
        want_len = batch.cell_lengths[i].detach().cpu().numpy()
        want_ang = np.degrees(batch.cell_angles[i].detach().cpu().numpy())
        assert np.abs(cellpar[:3] - want_len).max() < 1e-3, (
            f'crystal {i}: ASE cell lengths {cellpar[:3]} vs stored {want_len}')
        assert np.abs(cellpar[3:] - want_ang).max() < 1e-3, (
            f'crystal {i}: ASE cell angles {cellpar[3:]} vs stored {want_ang}')


# ------------------------------------------------------------------ the gate

def test_crystal_energy_matches_stock_mace_calculator(crystals, our_model,
                                                      stock_calculator, gpu):
    """
    THE LEVEL 1 CLAIM for MACE, stated against a control measured in the same test.

    Our production route (batched device-built neighbour list, hoisted builder,
    `unit_cell_pos` + `T_fc`) against upstream's MACECalculator doing its own load,
    its own `get_neighborhood` and its own collation, on identical coordinates.
    """
    batch = _built(crystals, MAX_GRAPHS, gpu)

    a1 = _our_energies(batch, our_model)
    a2 = _our_energies(batch, our_model)
    control = (a1 - a2).abs().max().item()

    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)
    cross = (a1 - theirs).abs().max().item()
    scale = a1.abs().mean().item()
    per_mol = _per_molecule_kj(a1 - theirs, batch)
    print(f'\n[mace level1] control {control:.4e}  cross-stack {cross:.4e} eV  '
          f'scale {scale:.1f} eV; per molecule max {per_mol.max():.4f} kJ/mol')

    _decompose(a1 - theirs, batch, 'mace level1')
    bar = max(control * 20.0, 1e-5 * max(scale, 1.0))
    assert cross <= bar, (
        f'our MACE energies differ from a stock MACECalculator by {cross:.4e} eV '
        f'({per_mol.max():.4f} kJ/mol per molecule) against a same-stack control of '
        f'{control:.4e}. That is a code difference, not float noise. Per crystal '
        f'(eV): {[round(float(x), 5) for x in (a1 - theirs)]}')


def test_the_residual_is_the_fractional_round_trip_not_our_code(crystals, our_model,
                                                                stock_calculator, gpu):
    """
    ATTRIBUTES the residual rather than tolerating it, which is the difference between
    a gate and a shrug.

    Our path does not hand the model `unit_cell_pos`. The neighbour list IS built from
    the raw positions (`batched_pbc_neighbour_list(pos_all, ...)`), but
    `compute_crystal_mace_on_mxt_batch` then overwrites `input_data['positions']` with
    a fractional round trip (`T_cf` then `T_fc`) that moves atoms by up to ~1.9e-6 A.
    So the model is EVALUATED at positions that differ slightly from the ones its own
    graph was built for.

    WHY THE MECHANISM IS NOT CLAIMED HERE. An earlier version of this docstring said
    the perturbation flips edges at MACE's cutoff. That is false for our side by
    construction -- our edge set is fixed before the round trip is applied -- and the
    magnitude by which a 1.9e-6 A shift produces ~1e-3 eV is NOT established. What is
    established is the attribution, by intervention, below.

    So the honest test is not "is the difference small" but "is it OURS". Scoring the
    STOCK calculator on the same round-tripped positions must move it toward us: if the
    residual came from our construction, matching positions would not help.

    Measured 2026-08-30: ours vs stock(raw) 1.83e-3 eV; ours vs stock(round-tripped)
    4.27e-4 eV; and stock's own sensitivity to the round trip is 2.08e-3 eV -- larger
    than our entire disagreement with it.
    """
    from mxtaltools.common.geometry_utils import fractional_transform

    batch = _built(crystals, MAX_GRAPHS, gpu)
    ours = _our_energies(batch, our_model)

    atoms_raw = _ase_unit_cells(batch)
    stock_raw = _stock_energies(atoms_raw, stock_calculator)

    frac = fractional_transform(batch.unit_cell_pos, batch.T_cf[batch.unit_cell_batch])
    rt = fractional_transform(frac, batch.T_fc[batch.unit_cell_batch])
    shifted = batch.clone()
    shifted.unit_cell_pos = rt
    stock_rt = _stock_energies(_ase_unit_cells(shifted), stock_calculator)

    d_raw = (ours - stock_raw).abs().max().item()
    d_rt = (ours - stock_rt).abs().max().item()
    stock_sensitivity = (stock_raw - stock_rt).abs().max().item()
    print('')
    print(f'[mace attribution] ours-vs-stock(raw) {d_raw:.4e}  '
          f'ours-vs-stock(round-tripped) {d_rt:.4e}  '
          f'stock own sensitivity {stock_sensitivity:.4e} eV')

    assert d_rt < d_raw, (
        f'matching the position round trip did NOT reduce the disagreement '
        f'({d_rt:.4e} vs {d_raw:.4e} eV), so the residual is not attributable to it '
        f'and something in OUR construction is the more likely cause')
    assert stock_sensitivity > d_raw / 2, (
        f'the stock calculator barely moves under the same round trip '
        f'({stock_sensitivity:.4e} eV) yet we differ from it by {d_raw:.4e} -- the '
        f'attribution in this docstring no longer holds and should be re-derived')


def test_energies_are_not_accidentally_identical(crystals, our_model,
                                                 stock_calculator, gpu):
    """Guards the guard: if the stock helper silently returned our own numbers, every
    assertion above would pass vacuously. Two independent stacks on a GPU do not agree
    bitwise."""
    batch = _built(crystals, MAX_GRAPHS, gpu)
    ours = _our_energies(batch, our_model)
    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)
    assert not torch.equal(ours, theirs), (
        'our MACE energies are bitwise identical to the stock calculator -- the '
        'fixture is comparing something against itself')


def test_a_deliberately_broken_cell_is_caught(crystals, our_model,
                                              stock_calculator, gpu):
    """
    NEGATIVE CONTROL. A gate never seen to fail is a gate of unknown power. A 0.3 A
    cell perturbation is far smaller than any real construction bug and must be
    rejected by the same comparison.
    """
    batch = _built(crystals, MAX_GRAPHS, gpu)
    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)

    bent = _built(crystals, MAX_GRAPHS, gpu)
    bent.cell_lengths[0] += 0.3
    bent.box_analysis()
    bent.pose_aunit(std_orientation=False)
    bent.build_unit_cell()
    ours_bent = _our_energies(bent, our_model)

    cross = (ours_bent - theirs).abs().max().item()
    scale = theirs.abs().mean().item()
    bar = max(1e-5 * max(scale, 1.0), 1e-4)
    assert cross > bar, (
        f'a 0.3 A cell perturbation moved the energy by only {cross:.4e} eV, inside '
        f'the {bar:.4e} bar this file asserts. The bar cannot catch a real bug')
