"""
LEVEL 1 GROUND TRUTH: our UMA route against a stock fairchem workflow.

WHAT THIS ADDS THAT THE REST OF THE SUITE CANNOT. Every other UMA gate in this
directory is an INTERNAL parity check -- vectorised construction against the list
path, our neighbour list against fairchem's generator, one predictor against itself
across a flag flip. All of them compare two of OUR routes, judged against a same-run
nondeterminism control. A bug that moves both routes equally is invisible to every
one of them. This file scores the same crystals through a stock fairchem stack --
`load_predict_unit` with fairchem's own 'default' settings, no overrides, no task
surgery, `FAIRChemCalculator`, `ase.Atoms` -- and compares against the energy our
production path actually returns. It is the only test here that can catch a
systematically wrong ENERGY -- with one qualification worth keeping straight:
`test_pbc_neighbours.py` already validates the neighbour list itself against
matscipy (via `mace.data.get_neighborhood`) on exact edge sets, so the graph is not
unvalidated. What was unvalidated before this file is everything downstream of it.

THE WRAPPING QUESTION, AND WHY THE COMPARISON IS FAIR. The two stacks are handed
DIFFERENT COORDINATES on purpose, and that is the point rather than a flaw:

  * ours hands the model UNWRAPPED `unit_cell_pos` together with an externally built
    neighbour list, which is correct because the list is built for those coordinates;
  * the ASE route wraps. `AtomicData.from_ase` calls
    `wrap_positions(pos, cell, pbc=pbc, eps=0)` before it builds anything (checked
    against fairchem-core 2.16.0 in this venv), so the stock stack always sees atoms
    inside the cell -- exactly the case its internal `radius_graph_pbc_v2` documents
    itself correct for.

Wrapping atoms by lattice vectors is a symmetry of the crystal, so the physical
energy is identical and the two stacks must agree. That is what makes this a ground
truth rather than a tautology: if we still had the F-047 edge-dropping bug, or any
convention error in `T_fc` -> cell, this test reads it as a real disagreement.
It is also why the test does NOT wrap by hand -- doing so would hide the very
difference it exists to probe.

PRECISION IS PART OF WHAT IS MEASURED. Our shipping predictor sets `tf32=True`
(`crystal_inference_settings`); fairchem's `InferenceSettings` default is
`tf32=False`. That is a real difference between our stack and a stock one, so
`test_crystal_energy_matches_stock_ase_calculator` leaves it in and sizes its bar
to accommodate it. `test_crystal_energy_matches_stock_at_matched_precision` builds
our predictor with tf32 off so the remaining delta is attributable to code alone;
that one carries the tight bar and is the real correctness gate.

PHYSICAL CELLS ONLY. fairchem's internal graph truncates at max_neighbors=300.
Physical CSD cells reach ~141 and never hit it; degenerate early-training cells
reach ~2710 and would. Comparing on trash cells would therefore measure fairchem's
neighbour cap, not our code, so the fixture here is real crystals only and the
`packing_coeff` guard inside `compute_crystal_uma_on_mxt_batch` is asserted not to
have fired (it mutates `cell_lengths`, which would desynchronise the two stacks).

RUN (needs a free GPU and the checkpoint):
    pytest tests/test_uma_vs_stock_fairchem.py -q -rs \
        --uma-checkpoint D:/crystal_datasets/esen_s.pt

Level 2 (`test_matches_published_reference_energies`) is a mechanism with no data
committed: point $UMA_REFERENCE_JSON at a file of structures fairchem publishes
energies for and it runs. It skips otherwise rather than inventing a reference.
"""
import json
import os

import numpy as np
import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
N_CRYSTALS = 8
EV_TO_KJ = 96.485

#: Separation of concerns for the bars below. The two effects we are willing to
#: tolerate are float noise (tf32 + GPU reduction order, ~4e-3 eV/crystal measured
#: 2026-08-12) and nothing else. The effect we must catch is an F-047-class graph
#: bug, which was 0.243 eV on a 1718 eV batch -- roughly 6 kJ/mol per molecule at
#: sym_mult 4. A bar in reward units at 0.5 kJ/mol/molecule sits an order of
#: magnitude under that and an order of magnitude over the noise.
#:
#: MEASURED 2026-08-30 (RTX 5080, esen_s.pt, 8 mini_new_csd crystals, 948.4 eV scale):
#: our production stack vs a stock fairchem ASE workflow differs by mean 0.126 /
#: max 0.374 kJ/mol per molecule, essentially all of it tf32. Set at 1.0 rather than
#: hugging that 0.374 -- a bar with no headroom fails on the next fixture and teaches
#: everyone to ignore it -- while still sitting ~6x under the F-047 signature.
MAX_REWARD_UNIT_DELTA_KJ = 1.0


@pytest.fixture(scope='module')
def crystals():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    raw = torch.load(DATASET, weights_only=False, map_location='cpu')
    good = []
    for c in raw:
        try:
            probe = collate_data_list([c.clone()])
            probe.pose_aunit(std_orientation=False)
            probe.build_unit_cell()
            good.append(c)
        except Exception:
            pass
    if len(good) < N_CRYSTALS:
        pytest.skip(f'only {len(good)} buildable crystals in the fixture')
    return good


@pytest.fixture(scope='module')
def checkpoint_path(request):
    """Honours BOTH the conftest option and the env var, in that order -- the skip
    message names both, and a message that names a flag the fixture ignores is how a
    run silently tests nothing."""
    path = request.config.getoption('--uma-checkpoint') or os.environ.get('UMA_CHECKPOINT')
    if not path or not os.path.exists(path):
        pytest.skip('pass --uma-checkpoint (or set UMA_CHECKPOINT) to run this file')
    return path


def _built(crystals, n, device):
    b = collate_data_list([c.clone() for c in crystals[:n]]).to(device)
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    b.box_analysis()
    return b


# ------------------------------------------------------------------ the two stacks

@pytest.fixture(scope='module')
def our_predictor(checkpoint_path, gpu):
    """Exactly what production builds: tf32 on, external graph, task heads popped."""
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    return init_uma_crystal_predictor(checkpoint_path, device=gpu)


@pytest.fixture(scope='module')
def our_predictor_fp32(checkpoint_path, gpu):
    """Ours, with the ONE precision knob matched to fairchem's default."""
    from mxtaltools.mlip_interfaces.uma_utils import (
        crystal_inference_settings, _build_uma_crystal_predictor)
    settings = crystal_inference_settings()
    # safe to mutate: crystal_inference_settings constructs a fresh InferenceSettings
    # per call precisely so this cannot reach fairchem's 'default' singleton.
    settings.tf32 = False
    return _build_uma_crystal_predictor(checkpoint_path, gpu, settings)


@pytest.fixture(scope='module')
def stock_calculator(checkpoint_path, gpu):
    """
    A stock fairchem workflow, built the way their docs build one.

    Deliberately NOT `init_uma_crystal_predictor`: no `overrides`, no popping of
    `omc_forces`/`omc_stress`, fairchem's own 'default' InferenceSettings. If any of
    that surgery changes the energy head's output, this is what notices.
    """
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    predictor = pretrained_mlip.load_predict_unit(
        checkpoint_path, inference_settings='default', device=gpu)
    return FAIRChemCalculator(predictor, task_name='omc')


def _ase_unit_cells(batch):
    """
    ase.Atoms built from the SAME tensors the model is handed, with no help from our
    fairchem converter -- `unit_cell_pos` and `T_fc` straight off the crystal.

    `T_fc` maps fractional to cartesian as a column-vector operator, so its ROWS
    transposed are the lattice vectors, which is ASE's convention and matches the
    `cell=batch.T_fc.transpose(1, 2)` that `batch_to_fairchem_batch` passes.
    """
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
            f'unit-cell positions -- the fixture batch is inconsistent, so the '
            f'comparison would be meaningless')
        out.append(Atoms(numbers=numbers, positions=p,
                         cell=cells[i].T, pbc=True))
    return out


def _stock_energies(atoms_list, calculator):
    """One ASE call per crystal. A fresh copy per structure so the calculator's
    result cache is keyed on a distinct object and cannot serve a stale energy --
    silently reusing crystal 0's number for all eight would make every comparison
    below pass for the wrong reason."""
    out = []
    for atoms in atoms_list:
        a = atoms.copy()
        a.calc = calculator
        out.append(float(a.get_potential_energy()))
    return torch.tensor(out, dtype=torch.float64)


def _our_energies(batch, predictor):
    from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
    return compute_crystal_uma_on_mxt_batch(
        batch.clone(), std_orientation=False, predictor=predictor,
        pbc=True, force_rebuild=False).detach().double().cpu()


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
    """eV per unit cell -> kJ/mol per molecule, the units the reward is actually in."""
    denom = (batch.sym_mult * batch.z_prime).double().cpu()
    return delta_ev.abs() * EV_TO_KJ / denom


# ------------------------------------------------------------------ the gates

def test_the_density_guard_did_not_fire(crystals):
    """
    A precondition, asserted rather than assumed, and on CPU so it runs in CI.
    `compute_crystal_uma_on_mxt_batch` grows `cell_lengths` by +2 A in a loop while
    any packing_coeff > 2.0. On real crystals it never fires -- but if it did, our
    stack would be scoring a DIFFERENT structure from the one handed to ASE and
    every comparison below would be quietly invalid.
    """
    batch = _built(crystals, N_CRYSTALS, 'cpu')
    worst = float(batch.packing_coeff.max())
    assert worst <= 2.0, (
        f'packing_coeff reaches {worst:.3f} on the fixture, so the max_cp guard '
        f'would mutate cell_lengths and desynchronise the two stacks')


def test_ase_cell_convention_round_trips(crystals):
    """
    CPU. The one silent way this whole file could be wrong: if `T_fc.T` is not the
    lattice-vector matrix ASE expects, both stacks would score self-consistent but
    DIFFERENT crystals, and the energy comparison would report a convention error as
    a model disagreement -- or worse, pass because the two errors cancelled.

    Asserted by round-tripping through ASE's own cellpar, against the crystal's own
    stored parameters. Measured agreement is ~1e-6 on the fixture.
    """
    batch = _built(crystals, N_CRYSTALS, 'cpu')
    for i, atoms in enumerate(_ase_unit_cells(batch)):
        cellpar = atoms.cell.cellpar()
        want_len = batch.cell_lengths[i].detach().cpu().numpy()
        want_ang = np.degrees(batch.cell_angles[i].detach().cpu().numpy())
        assert np.abs(cellpar[:3] - want_len).max() < 1e-3, (
            f'crystal {i}: ASE cell lengths {cellpar[:3]} against stored {want_len}')
        assert np.abs(cellpar[3:] - want_ang).max() < 1e-3, (
            f'crystal {i}: ASE cell angles {cellpar[3:]} against stored {want_ang}')


def test_the_fixture_is_actually_unwrapped(crystals):
    """
    CPU. The comparison is only meaningful if the two stacks really do see different
    coordinates -- if `unit_cell_pos` happened to be wrapped already, the ASE route's
    internal `wrap_positions` would be a no-op and this file would be comparing
    identical inputs, quietly losing all its power to detect an F-047-class bug.
    """
    batch = _built(crystals, N_CRYSTALS, 'cpu')
    cells = batch.T_fc.detach().cpu().numpy()
    ucell_batch = batch.unit_cell_batch.cpu().numpy()
    pos = batch.unit_cell_pos.detach().cpu().numpy()

    worst = 0.0
    for i in range(batch.num_graphs):
        frac = pos[ucell_batch == i] @ np.linalg.inv(cells[i].T)
        worst = max(worst, float(np.ptp(frac, axis=0).max()))
    assert worst > 1.05, (
        f'the widest fractional spread in the fixture is {worst:.2f} cell widths, so '
        f'the positions are effectively already wrapped and the ASE route sees the '
        f'same coordinates we do -- this file is no longer an independent check')


def test_crystal_energy_matches_stock_ase_calculator(crystals, our_predictor,
                                                     stock_calculator, gpu):
    """
    THE LEVEL 1 CLAIM, with our shipping precision left in.

    Our production route (unwrapped positions + external graph + tf32 + popped task
    heads) against a stock fairchem ASE workflow (wrapped positions + internal graph
    + fp32 + every head). Reported in kJ/mol per molecule because that is the unit
    the number matters in -- a delta is only interesting relative to lattice
    energies of order 100.
    """
    batch = _built(crystals, N_CRYSTALS, gpu)
    ours = _our_energies(batch, our_predictor)
    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)

    delta_ev = ours - theirs
    per_mol = _per_molecule_kj(delta_ev, batch)
    print(f'\n[level1 tf32] max |delta| {delta_ev.abs().max():.4e} eV on a '
          f'{ours.abs().mean():.1f} eV scale; per molecule '
          f'mean {per_mol.mean():.4f} max {per_mol.max():.4f} kJ/mol')

    _decompose(delta_ev, batch, 'level1 tf32')
    assert per_mol.max() < MAX_REWARD_UNIT_DELTA_KJ, (
        f'our energies differ from a stock fairchem workflow by up to '
        f'{per_mol.max():.4f} kJ/mol per molecule, above the {MAX_REWARD_UNIT_DELTA_KJ} '
        f'tolerance. Per crystal (kJ/mol/molecule): '
        f'{[round(float(x), 4) for x in per_mol]}')


def test_crystal_energy_matches_stock_at_matched_precision(crystals, our_predictor_fp32,
                                                           stock_calculator, gpu):
    """
    THE CORRECTNESS GATE, with precision removed as a variable.

    Same comparison with tf32 off on our side, so fairchem's default and ours differ
    only in the code under test: our batch construction, our external neighbour list,
    our predictor overrides and task surgery. The bar is measured against a control
    in the same test rather than pinned, because even at fp32 GPU reduction order is
    not bit-stable across two different graph-construction routes.
    """
    batch = _built(crystals, N_CRYSTALS, gpu)

    a1 = _our_energies(batch, our_predictor_fp32)
    a2 = _our_energies(batch, our_predictor_fp32)
    control = (a1 - a2).abs().max().item()          # same stack, twice

    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)
    cross = (a1 - theirs).abs().max().item()
    per_mol = _per_molecule_kj(a1 - theirs, batch)
    scale = a1.abs().mean().item()
    print(f'\n[level1 fp32] control {control:.4e}  cross-stack {cross:.4e} eV  '
          f'scale {scale:.1f} eV; per molecule max {per_mol.max():.4f} kJ/mol')

    # MEASURED 2026-08-30: control 6.1e-5, cross-stack 7.6e-5 eV -- swapping to a
    # wholly independent fairchem stack adds 1.25x the variation of re-running our
    # own. The reasoned bar (20x control, 1e-5*scale) passed with 125x headroom,
    # which is a gate that cannot fail; tightened to sit ~12x over the measurement
    # and still ~250x under the 0.243 eV F-047 signature.
    _decompose(a1 - theirs, batch, 'level1 fp32')
    bar = max(control * 4.0, 1e-6 * scale)
    assert cross <= bar, (
        f'at matched precision our stack differs from stock fairchem by {cross:.4e} eV '
        f'({per_mol.max():.4f} kJ/mol per molecule), against a same-stack control of '
        f'{control:.4e}. That is a code difference, not float noise')


def test_energies_are_not_accidentally_identical(crystals, our_predictor,
                                                 stock_calculator, gpu):
    """
    Guards the guard. If `_stock_energies` silently returned our own numbers -- a
    fixture wiring mistake -- every assertion above would pass vacuously. Two
    genuinely independent stacks on a GPU do NOT agree bitwise, so requiring a
    nonzero spread proves the comparison is actually crossing a boundary.
    """
    batch = _built(crystals, N_CRYSTALS, gpu)
    ours = _our_energies(batch, our_predictor)
    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)
    assert not torch.equal(ours, theirs), (
        'our energies are bitwise identical to the stock ASE workflow. That is not '
        'possible across two graph-construction routes on a GPU -- the test fixture '
        'is comparing something against itself')


def test_a_deliberately_broken_graph_is_caught(crystals, our_predictor_fp32,
                                               stock_calculator, gpu):
    """
    NEGATIVE CONTROL, and the reason the rest of this file can be believed.

    A gate that has never been seen to fail is a gate of unknown power. Perturbing
    one crystal's cell by 0.3 A is a change far smaller than an F-047-class graph
    bug, and the comparison must reject it. If this test passes while the others do
    too, the bars above are measuring something real; if this one fails, they are
    too loose to catch anything.
    """
    batch = _built(crystals, N_CRYSTALS, gpu)
    theirs = _stock_energies(_ase_unit_cells(batch), stock_calculator)

    bent = batch.clone()
    bent.cell_lengths[0] += 0.3
    bent.box_analysis()
    bent.pose_aunit(std_orientation=False)
    bent.build_unit_cell()
    ours_bent = _our_energies(bent, our_predictor_fp32)

    per_mol = _per_molecule_kj(ours_bent - theirs, batch)
    assert per_mol.max() >= MAX_REWARD_UNIT_DELTA_KJ, (
        f'a 0.3 A cell perturbation moved the energy by only {per_mol.max():.4f} '
        f'kJ/mol per molecule, under the {MAX_REWARD_UNIT_DELTA_KJ} bar this file '
        f'asserts. The bar is too loose to catch a real graph bug')


# ------------------------------------------------------------------ Level 2

#: Checked as a MARKER, not inside the body, so a run without reference data never
#: instantiates the `gpu` fixture and therefore never opens a CUDA context.
_REF_JSON = os.environ.get('UMA_REFERENCE_JSON')


@pytest.mark.skipif(
    not (_REF_JSON and os.path.exists(_REF_JSON)),
    reason='set $UMA_REFERENCE_JSON to a published reference file to run the Level 2 '
           'check (schema is in the test docstring)')
def test_matches_published_reference_energies(gpu):
    """
    LEVEL 2: fairchem's own published outputs, not a second run of their code.

    No reference data is committed here, because inventing one would defeat the
    purpose -- a reference we generated is Level 1 again with extra steps. Point
    $UMA_REFERENCE_JSON at a file of the form

        {"checkpoint": "esen_s.pt",
         "task_name": "omc",
         "structures": [{"numbers": [...], "positions": [[x,y,z], ...],
                         "cell": [[...],[...],[...]], "pbc": true,
                         "energy_ev": -1234.5}, ...]}

    and this runs those structures through the STOCK calculator, confirming our
    fairchem install reproduces the numbers whoever produced them saw. Combined with
    the Level 1 tests above -- which tie our route to that stock stack -- it closes
    the chain from our reward back to a published value.
    """
    path = _REF_JSON
    from ase import Atoms
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    with open(path) as f:
        ref = json.load(f)

    ckpt = os.environ.get('UMA_CHECKPOINT')
    if not ckpt or not os.path.exists(ckpt):
        pytest.skip('UMA_CHECKPOINT must also be set for the Level 2 check')
    if ref.get('checkpoint') and os.path.basename(ckpt) != ref['checkpoint']:
        pytest.skip(f"reference was produced with {ref['checkpoint']}, not "
                    f"{os.path.basename(ckpt)} -- a mismatched checkpoint would "
                    f"fail for the wrong reason")

    predictor = pretrained_mlip.load_predict_unit(
        ckpt, inference_settings='default', device=gpu)
    calc = FAIRChemCalculator(predictor, task_name=ref.get('task_name', 'omc'))

    worst, worst_i = 0.0, -1
    for i, s in enumerate(ref['structures']):
        atoms = Atoms(numbers=s['numbers'], positions=s['positions'],
                      cell=s.get('cell'), pbc=s.get('pbc', True))
        atoms.calc = calc
        delta = abs(float(atoms.get_potential_energy()) - float(s['energy_ev']))
        if delta > worst:
            worst, worst_i = delta, i
    print(f'\n[level2] {len(ref["structures"])} structures, worst |delta| '
          f'{worst:.4e} eV at index {worst_i}')
    assert worst < 1e-2, (
        f'our fairchem install disagrees with the published reference by {worst:.4e} '
        f'eV at structure {worst_i} -- the install or the checkpoint differs from '
        f'whatever produced that file')
