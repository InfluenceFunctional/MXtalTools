"""cfg:mlip_gas_reference and cfg:mlip_mirror_average (run_search.attach_mlip_parity; compute_lattice_uma).

UMA/eSEN backbones are rotation- but not reflection-equivariant. Pins, on the GPU with a UMA checkpoint:
- with the mirror-mean gas reference, one crystal scores the same lattice energy in its own description and in the
  identity description (which can swap the enantiomer held by the asymmetric unit); without it the test molecule's
  two descriptions differ, so the check can fail;
- the mirror-averaged periodic energy is the mean of the crystal's and its inverted copy's, and has a finite gradient;
- either key without a uma stage raises.
The test molecule is KEQQON (13 atoms, non-planar) from the mini CSD set, as tests/test_fd_refold.py uses it.
"""
import os
import types

import pytest
import torch

from mxtaltools.crystal_search import minima as mn
from mxtaltools.crystal_search.run_search import attach_mlip_parity
from mxtaltools.dataset_utils.utils import collate_data_list

from tests.test_fd_refold import _case, _rows, mol  # noqa: F401


@pytest.fixture(scope='module')
def predictor(request):
    path = request.config.getoption('--uma-checkpoint') or os.environ.get('UMA_CHECKPOINT')
    if not path or not os.path.exists(path):
        pytest.skip('pass --uma-checkpoint (or set UMA_CHECKPOINT) to run this file')
    if not torch.cuda.is_available():
        pytest.skip('needs a CUDA device')
    from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
    return path, init_uma_crystal_predictor(path, device='cuda', activation_checkpointing=False, tf32=False)


def _row(mol, sg):
    torch.manual_seed(0)
    _, p, h = _case('normal', n=1, seed=31)
    if sg != 2:
        p[:, 3] = p[:, 5] = torch.pi / 2
    rows = _rows(mol, sg, p, h)
    return rows.batch_to_list()[0] if hasattr(rows, 'batch_to_list') else rows[0]


def _lattice(row, pred):
    E = mn.StageEnergy([row], 'uma', 1.0, predictor=pred, device='cuda', chunk=2)
    return float(E.f(row.full_cell_parameters().double(), [0])[0])


def test_gas_reference_makes_descriptions_agree(mol, predictor):
    path, pred = predictor
    a = _row(mol, 14)
    a.identifier = 'KEQQON'
    b = collate_data_list([a]); mn.identity_description(b); r = b.batch_to_list()[0]
    r.identifier = 'KEQQON'
    hand = (float(a.aunit_handedness.reshape(-1)[0]), float(r.aunit_handedness.reshape(-1)[0]))
    cfg = types.SimpleNamespace(mlip_gas_reference='mirror_mean', mlip_mirror_average=False,
                                uma_predictor_path=path, opt=[{'optim_target': 'uma'}])
    a2, r2 = a.clone(), r.clone()
    attach_mlip_parity(cfg, [a2, r2], 'cuda')
    assert abs(_lattice(a2, pred) - _lattice(r2, pred)) < 0.01  # kJ/mol: float32 noise, against ~kJ/mol before
    if hand[0] != hand[1]:  # the descriptions hold opposite enantiomers: without the fix they disagree
        assert abs(_lattice(a, pred) - _lattice(r, pred)) > 0.05


def test_mirror_average_is_the_mean_over_the_inverted_crystal(mol, predictor):
    path, pred = predictor
    a = _row(mol, 4)  # P2_1: chiral, so the crystal and its mirror image are different crystals
    cfg = types.SimpleNamespace(mlip_gas_reference=None, mlip_mirror_average=True, uma_predictor_path=path,
                                opt=[{'optim_target': 'uma'}])
    a2 = a.clone()
    attach_mlip_parity(cfg, [a2], 'cuda')
    from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
    plain = collate_data_list([a]).to('cuda'); plain.mol2ucell(std_orientation=True)
    e = float(plain.compute_crystal_uma(pred, std_orientation=True)[0])
    inv = plain.clone()
    inv.unit_cell_pos = -plain.unit_cell_pos + plain.T_fc.sum(dim=2)[plain.unit_cell_batch.long()]
    e_inv = float(compute_crystal_uma_on_mxt_batch(inv, True, pred, pbc=True, force_rebuild=False)[0])
    sym = collate_data_list([a2]).to('cuda'); sym.mol2ucell(std_orientation=True)
    sym.compute_lattice_uma(pred)
    assert abs(float(sym.uma_pot[0]) - 0.5 * (e + e_inv)) < 1e-3  # eV per cell
    _, g = mn.StageEnergy([a2], 'uma', 2.494, predictor=pred, device='cuda', chunk=2).fg(
        a2.full_cell_parameters().double(), [0])
    assert bool(torch.isfinite(g).all())


def test_parity_keys_without_a_uma_stage_raise():
    cfg = types.SimpleNamespace(mlip_gas_reference='mirror_mean', mlip_mirror_average=False,
                                uma_predictor_path='', opt=[{'optim_target': 'elj'}])
    with pytest.raises(ValueError):
        attach_mlip_parity(cfg, [], 'cpu')
    cfg = types.SimpleNamespace(mlip_gas_reference='average', mlip_mirror_average=False,
                                uma_predictor_path='', opt=[{'optim_target': 'uma'}])
    with pytest.raises(ValueError):
        attach_mlip_parity(cfg, [], 'cpu')
