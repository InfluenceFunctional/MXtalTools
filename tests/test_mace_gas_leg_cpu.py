"""
The MACE builders must not write to the batch they read -- above all on the
pbc=False gas leg, on CPU.

THE BUG THIS PINS (found 2026-09-15, fixed 2026-09-28). The builders handed
`mace.data.get_neighborhood` a cell built as
`T_fc.transpose(-2, -1).detach().cpu().numpy()`. On a CPU tensor `.cpu()` does not
copy, so that array was a VIEW of batch.T_fc, and get_neighborhood's non-periodic
branch writes its vacuum box into the cell IN PLACE. The gas leg therefore
overwrote T_fc; compute_crystal_mace_on_mxt_batch then rebuilt the positions
through the overwritten T_fc, and every isolated molecule scored as the same
stretched geometry: -37.017 eV for acridine where MACE's own calculator gives
-159.690 eV, a lattice energy near -11,899 kJ/mol against a stored -62.84. On GPU
`.cpu()` copies, so the GPU route was right throughout. The internal-parity gates
could not see it -- both builders mutated identically -- and the stock-calculator
gate (test_mace_vs_stock.py) scores the crystal leg only, on GPU.

TWO TIERS.
  * Model-free, always on: every builder, at pbc False and True, leaves T_fc,
    T_cf and unit_cell_pos bit-identical; at pbc=False the cell it hands the model
    is the rewritten vacuum box, so the rewrite demonstrably ran -- on a copy.
  * Model-backed, MACE_CHECKPOINT: the gas leg against upstream's MACECalculator on
    the same molecules, on both per-graph builders, and T_fc across the whole
    compute_crystal_mace_on_mxt_batch(pbc=False) call. Skips without a checkpoint.

CPU EVEN WHEN A GPU IS VISIBLE. The model-backed tests set torch.cuda.is_available
to False, so load_mace_model skips its cuequivariance conversion and
safe_predict_mace skips its synchronise: nothing opens a CUDA context on a card the
`gpu` pre-flight in conftest.py never cleared.

RUN:
    MACE_CHECKPOINT=/path/to/acr_112025_mh1_stagetwo.model \
        pytest tests/test_mace_gas_leg_cpu.py -q -rs
"""
import os

import numpy as np
import pytest
import torch

import mxtaltools.mlip_interfaces.AL_mace_utils as M
from mxtaltools.dataset_utils.utils import collate_data_list

CSD = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
ACRIDINE = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_acridine.pt')
MODEL_ENV = 'MACE_CHECKPOINT'

#: Gas leg against MACECalculator, eV per molecule. The bug was 122.7 eV; measured
#: after the fix (2026-09-28, CPU, float32, acr_112025_mh1_stagetwo) at 1.5e-5 eV on
#: this file's four acridine crystals, one float32 ULP at 160 eV, on both builders.
GAS_TOL_EV = 1e-3

BUILDERS = ('batch_to_mace_atomicdata_hoisted', 'batch_to_mace_atomicdata',
            'batch_to_mace_input_dict')


class _StubModel:
    """What the builders read from a MACE model, and nothing else: the cutoff and the
    element table (as in test_mace_atomicdata_vectorisation.py)."""

    def __init__(self, r_max=6.0, atomic_numbers=tuple(range(1, 101))):
        self.r_max = torch.tensor(float(r_max))
        self.atomic_numbers = torch.tensor(list(atomic_numbers))


@pytest.fixture(scope='module')
def csd_crystals():
    """Four buildable crystals; entries that raise inside the unit-cell build are
    dropped (a fixture property, unrelated to this code)."""
    if not os.path.exists(CSD):
        pytest.skip(f'{CSD} not present')
    good = []
    for c in torch.load(CSD, weights_only=False, map_location='cpu'):
        try:
            p = collate_data_list([c.clone()])
            p.pose_aunit(std_orientation=False)
            p.build_unit_cell()
            good.append(c)
        except Exception:
            pass
        if len(good) == 4:
            return good
    pytest.skip(f'only {len(good)} buildable crystals in the fixture')


def _built(crystals):
    b = collate_data_list([c.clone() for c in crystals])
    b.pose_aunit(std_orientation=False)
    b.build_unit_cell()
    return b


@pytest.mark.parametrize('pbc', [False, True], ids=['gas', 'crystal'])
@pytest.mark.parametrize('builder', BUILDERS)
def test_builders_do_not_write_to_the_batch(csd_crystals, builder, pbc):
    if builder == 'batch_to_mace_input_dict' and not pbc:
        pytest.skip('the device-built dict is periodic-only by design')
    model = _StubModel()
    b = _built(csd_crystals)
    before = {k: getattr(b, k).detach().clone() for k in ('T_fc', 'T_cf', 'unit_cell_pos')}

    out = getattr(M, builder)(b, False, model, False, pbc=pbc)

    for k, v in before.items():
        after = getattr(b, k)
        assert torch.equal(after, v), (
            f'{builder}(pbc={pbc}) changed batch.{k} by up to '
            f'{float((after - v).abs().max()):.3e} -- a numpy array handed to a mutating '
            f'callee is a view of a CPU batch tensor')

    if not pbc:
        # THE REWRITE RAN, so an unchanged T_fc means it landed on a copy -- not that
        # nothing was written. The cell the model is handed is get_neighborhood's vacuum
        # box, (max|pos| + 1) * 5 * r_max on the diagonal, not the crystal cell.
        cutoff = float(model.r_max)
        for g, d in enumerate(out):
            pos = b.unit_cell_pos[b.unit_cell_batch == g].detach().numpy()
            box = (np.max(np.abs(pos)) + 1) * 5 * cutoff
            cell = np.asarray(d.cell, dtype=float)
            assert np.allclose(cell, box * np.eye(3), rtol=1e-5), (
                f'graph {g}: the pbc=False cell handed to the model is not the vacuum '
                f'box ({box:.2f} A on the diagonal):\n{cell}')


# ------------------------------------------------------------------ model-backed

@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture(scope='module')
def checkpoint_path():
    path = os.environ.get(MODEL_ENV)
    if not path or not os.path.exists(path):
        pytest.skip(f'set {MODEL_ENV} to a MACE checkpoint to run the model-backed tests')
    return path


@pytest.fixture(scope='module')
def cpu_model(checkpoint_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, 'is_available', lambda: False)
        return M.load_mace_model(checkpoint_path, 'cpu', torch.float32)


@pytest.fixture(scope='module')
def acridine(cpu_model):
    """One Z'=1 crystal from each fixture tier (experimental polymorph, prior sample,
    noised sample -- three different conformers) and the first sg 14 Z'=2 polymorph,
    the shape of the live acridine campaign, whose gas leg goes through
    split_to_zp1_batch and a per-crystal mean."""
    if not os.path.exists(ACRIDINE):
        pytest.skip(f'{ACRIDINE} not present')
    allowed = set(int(z) for z in cpu_model.atomic_numbers)
    raw = [c for c in torch.load(ACRIDINE, weights_only=False, map_location='cpu')
           if set(int(z) for z in c.z).issubset(allowed)]
    picked = []
    for tier in ('polymorph', 'prior', 'noised'):
        zp1 = [c for c in raw if c.fixture_tier == tier and int(c.z_prime.max()) == 1]
        if zp1:
            picked.append(zp1[0])
    zp2 = [c for c in raw if int(c.z_prime.max()) == 2 and int(c.sg_ind) == 14]
    if len(picked) < 3 or not zp2:
        pytest.skip('fixture lacks a Z\'=1 crystal per tier or an sg 14 Z\'=2 polymorph')
    return picked + zp2[:1]


def _collate(crystals):
    """The fixture tiers carry different key sets (polymorphs a fingerprint, samples a
    stored energy); collate on the keys every crystal has."""
    keys = [set(c.keys() if callable(c.keys) else c.keys) for c in crystals]
    return collate_data_list([c.clone() for c in crystals],
                             exclude_keys=sorted(set.union(*keys) - set.intersection(*keys)))


def _stock_gas_energies(crystals, checkpoint_path):
    """Upstream's ASE calculator on each isolated molecule, averaged over a crystal's
    Z' molecules; the atoms of a Z'>1 crystal are Z' equal contiguous blocks."""
    from ase import Atoms
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=checkpoint_path, device='cpu',
                          default_dtype='float32')
    out = []
    for c in crystals:
        zp = int(c.z_prime.max())
        n = c.z.shape[0] // zp
        per_mol = []
        for k in range(zp):
            atoms = Atoms(numbers=c.z[k * n:(k + 1) * n].numpy(),
                          positions=c.pos[k * n:(k + 1) * n].detach().numpy().astype(float),
                          pbc=False)
            atoms.calc = calc
            per_mol.append(float(atoms.get_potential_energy()))
        out.append(np.mean(per_mol))
    return torch.tensor(out, dtype=torch.float64)


@pytest.mark.parametrize('hoisted', [True, False], ids=['hoisted', 'reference'])
def test_gas_leg_matches_stock_mace_calculator(cpu_only, cpu_model, checkpoint_path,
                                               acridine, hoisted, monkeypatch):
    """compute_lattice_gas_phase_mace as production calls it. pbc=False never takes the
    device-built dict or the batched neighbour list, so the hoisted flag is the only
    module switch that changes this leg's builder."""
    monkeypatch.setattr(M, 'USE_HOISTED_MACE_ATOMICDATA', hoisted)
    batch = _collate(acridine)
    with torch.no_grad():
        ours = batch.compute_lattice_gas_phase_mace(cpu_model).double()
    theirs = _stock_gas_energies(acridine, checkpoint_path)

    d = (ours - theirs).abs()
    print(f'\n[mace gas leg, cpu, {"hoisted" if hoisted else "reference"}] max |ours - stock| '
          f'{float(d.max()):.3e} eV over {len(acridine)} crystals')
    assert float(d.max()) <= GAS_TOL_EV, (
        f'gas leg differs from MACECalculator by up to {float(d.max()):.4e} eV; ours '
        f'{[round(float(x), 4) for x in ours]}, stock {[round(float(x), 4) for x in theirs]}')


def test_gas_leg_forward_leaves_T_fc_unchanged(cpu_only, cpu_model, acridine):
    """The whole compute_crystal_mace_on_mxt_batch(pbc=False) call, set up the way
    compute_lattice_gas_phase_mace sets it up: split to one molecule per graph, P1."""
    b = _collate(acridine).split_to_zp1_batch()
    b.reset_sg_info(sg_ind=1)
    b.box_analysis()
    before = b.T_fc.detach().clone()
    with torch.no_grad():
        e = M.compute_crystal_mace_on_mxt_batch(b, cpu_model, std_orientation=False,
                                                pbc=False, force_rebuild=True)
    assert torch.isfinite(e).all(), f'non-finite gas energies {e}'
    assert torch.equal(b.T_fc, before), (
        f'compute_crystal_mace_on_mxt_batch(pbc=False) changed T_fc by up to '
        f'{float((b.T_fc - before).abs().max()):.3e} A')
