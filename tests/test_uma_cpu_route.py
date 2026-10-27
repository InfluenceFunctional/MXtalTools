"""
Our UMA crystal route on CPU, against a stock fairchem ASE calculator, under whichever fairchem is installed.

WHAT THIS PINS (2026-10-27). Three defects kept the route from running outside the container's fairchem 2.13 on a GPU:
  * fairchem 2.16 denormalises every task of every dataset; _build_uma_crystal_predictor popped only omc's forces and
    stress, so oc20_forces raised KeyError. The predictor now keeps the energy tasks only;
  * fairchem 2.16's AtomicData asserts integer atomic numbers, and molecule files may store z as float (the acridine
    conformers do). The builders now cast;
  * safe_predict_uma synchronised CUDA unconditionally, so a CPU-only process failed. It now synchronises only with CUDA.
And a search relaxation backpropagates through UMA: with activation checkpointing on, that backward raised torch's
CheckpointError on CPU (UMA-s 1.1 and 1.2, fairchem 2.13 and 2.16); the searches turn it off
(cfg:uma_activation_checkpointing), and the gradient test below runs that way.

CPU EVEN WHEN A GPU IS VISIBLE: torch.cuda.is_available is patched False, so nothing opens a CUDA context.

RUN (any UMA checkpoint; measured 2026-10-27 on 4 crystals: UMA-s 1.1 0.0022 kJ/mol per molecule under fairchem 2.16,
0.0037 under 2.13; UMA-s 1.2 0.0015 under 2.16):
    UMA_CHECKPOINT=D:/crystal_datasets/uma-s-1p1.pt pytest tests/test_uma_cpu_route.py -q -rs
"""
import os

import pytest
import torch

from tests import test_uma_vs_stock_fairchem as T

N = 4
TOL_KJ = 0.05  # per molecule; float32 noise is ~0.004, a graph or element error is kJ/mol


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture(scope='module')
def ckpt():
    path = os.environ.get('UMA_CHECKPOINT')
    if not path or not os.path.exists(path):
        pytest.skip('set UMA_CHECKPOINT to a UMA checkpoint to run this file')
    torch.serialization.add_safe_globals([slice])
    return path


@pytest.fixture(scope='module')
def crystals():
    if not os.path.exists(T.DATASET):
        pytest.skip(f'{T.DATASET} not present')
    good = []
    for c in torch.load(T.DATASET, weights_only=False, map_location='cpu'):
        try:
            p = T.collate_data_list([c.clone()])
            p.pose_aunit(std_orientation=False)
            p.build_unit_cell()
            good.append(c)
        except Exception:
            pass
        if len(good) == N:
            return good
    pytest.skip(f'only {len(good)} buildable crystals in the fixture')


def _predictor(ckpt, activation_checkpointing=True):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, 'is_available', lambda: False)
        from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
        return init_uma_crystal_predictor(ckpt, device='cpu', activation_checkpointing=activation_checkpointing)


def test_predictor_keeps_energy_tasks_only(ckpt):
    p = _predictor(ckpt)
    assert p.tasks and all(t.property == 'energy' for t in p.tasks.values()), sorted(p.tasks)
    assert all(t.property == 'energy' for ts in p.dataset_to_tasks.values() for t in ts)
    assert any(t.name == 'omc_energy' for t in p.dataset_to_tasks['omc'])


def test_cpu_energies_match_stock_calculator_with_float_z(crystals, ckpt):
    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    b = T._built(crystals, N, 'cpu')
    ours = T._our_energies(b, _predictor(ckpt))
    bf = b.clone()
    bf.z = bf.z.float()  # as the acridine conformer files store it
    ours_float_z = T._our_energies(bf, _predictor(ckpt))
    calc = FAIRChemCalculator(pretrained_mlip.load_predict_unit(ckpt, inference_settings='default', device='cpu'),
                              task_name='omc')
    theirs = T._stock_energies(T._ase_unit_cells(b), calc)
    per_mol = T._per_molecule_kj(ours - theirs, b)
    print(f'\n[uma cpu] |ours - stock| per molecule {[round(float(x), 4) for x in per_mol]} kJ/mol')
    assert float(per_mol.max()) < TOL_KJ
    assert torch.equal(ours, ours_float_z), 'a float z changed the energy'


def test_gradient_without_activation_checkpointing(crystals, ckpt):
    """What a search relaxation needs: d(energy)/d(positions), with checkpointing off (the searches' setting)."""
    from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
    b = T._built(crystals, 2, 'cpu')
    b.unit_cell_pos = b.unit_cell_pos.detach().requires_grad_(True)
    e = compute_crystal_uma_on_mxt_batch(b, std_orientation=False, predictor=_predictor(ckpt, False), pbc=True,
                                         force_rebuild=False)
    e.sum().backward()
    g = b.unit_cell_pos.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0
