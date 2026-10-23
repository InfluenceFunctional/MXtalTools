"""Stage 4 of the Z'>1 units plan: one element layout for the unit cell (gfn_diffusion
energy_sampling/docs/design/crystal_units_audit.md, item #15; section 5, stage 4).

The builder writes a Z'>1 unit cell unit by unit, ``[unit][image][atom]`` (split, build, join); every reader of
unit-cell elements gathers them through ``crystal_building.utils.unit_cell_gather_index`` (vectorised paths) or
``unit_cell_z`` (per-crystal paths), the same layout by two derivations.

What is pinned here:

- at Z'=1 the index is the image-major tiling ``z[batch == i].repeat(sym_mult[i])`` element for element, so every
  Z'=1 input is unchanged;
- on batches mixing Z'=1 and Z'>1 rows, the index and the per-crystal helper agree;
- on two-compound crystals the gathered elements are the geometric truth (each unit-cell row sits on an image of an
  asymmetric-unit atom of that element, ``charlib.element_truth``), and the old whole-asymmetric-unit tiling is not
  (negative control: the case is sensitive);
- crystals whose atoms do not divide into their units, or with z_prime < 1, are refused (``ValueError``).

- the other readers, on the two-compound fixtures against the geometric truth, each with the old tiling as a control:
  ``ase_mol_from_crystaldata(mode='unit cell')`` on a CLUSTER batch (its asymmetric unit is the graph's aux_ind == 0
  atoms there), the numbers ``compute_standard_cell`` hands spglib, and ``crystal_search.standardize.
  unit_cell_fractional`` (added after the stage-4 review).

The MLIP input builders (fairchem vectorised and list, MACE dict and list), the ASE 'unit cell' mode on rows and
zp_doubling are checked through the characterization suite (``test_characterization_known.py``: the #15 xfails this
stage flipped; the ``elements`` and ``mlip`` pins).
CPU only, a few seconds.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


def _charlib():
    spec = importlib.util.spec_from_file_location('charlib_for_unit_cell_elements',
                                                  str(REPO / 'tests' / 'characterization' / 'charlib.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CL = _charlib()


@pytest.fixture(autouse=True, scope='module')
def _deterministic_cpu():
    with CL.deterministic():
        yield


def _built(case, std=True):
    b = CL.build_case(case)
    b.mol2ucell(std_orientation=std)
    return b


def _old_tiling(b):
    return torch.cat([b.z[b.batch == i].repeat(int(b.sym_mult[i])) for i in range(b.num_graphs)])


@pytest.mark.parametrize('std', [True, False])
def test_index_is_the_old_tiling_at_zp1(std):
    from mxtaltools.crystal_building.utils import unit_cell_gather_index
    b = _built('F1', std)                           # four real Z'=1 rows, sg 2, 14 and 9
    assert int(b.z_prime.max()) == 1
    idx, out_n = unit_cell_gather_index(b.batch, b.z_prime, b.sym_mult, b.num_graphs)
    assert torch.equal(b.z[idx], _old_tiling(b))
    assert torch.equal(out_n, torch.bincount(b.batch, minlength=b.num_graphs) * b.sym_mult.long())
    assert len(idx) == len(b.unit_cell_pos)


@pytest.mark.parametrize('case', ['F3', 'F3b', 'F4a', 'two_eth_ether', 'two_pyr_flb'])
def test_index_and_per_crystal_helper_agree(case):
    from mxtaltools.crystal_building.utils import unit_cell_gather_index, unit_cell_z
    b = _built(case)
    idx, out_n = unit_cell_gather_index(b.batch, b.z_prime, b.sym_mult, b.num_graphs)
    ref = torch.cat([unit_cell_z(b.z[b.batch == i], b.z_prime.reshape(-1)[i], b.sym_mult[i])
                     for i in range(b.num_graphs)])
    assert torch.equal(b.z[idx], ref)
    assert int(out_n.sum()) == len(b.unit_cell_pos)


@pytest.mark.parametrize('case', ['two_eth_ether', 'two_pyr_flb'])
@pytest.mark.parametrize('std', [True, False])
def test_two_compound_elements_are_the_geometric_truth(case, std):
    from mxtaltools.crystal_building.utils import unit_cell_gather_index
    b = _built(case, std)
    idx, _ = unit_cell_gather_index(b.batch, b.z_prime, b.sym_mult, b.num_graphs)
    truth = CL.element_truth(b.unit_cell_pos, b.pos, b.z, b.T_fc[0], b.symmetry_operators[0])
    assert int((b.z[idx].long() != truth).sum()) == 0
    # negative control: the whole asymmetric unit tiled image by image (the readers before stage 4) is wrong here
    assert int((_old_tiling(b).long() != truth).sum()) > 0


def test_uneven_units_and_empty_rows_are_refused():
    from mxtaltools.crystal_building.utils import unit_cell_gather_index
    batch_ind = torch.tensor([0] * 5 + [1] * 4)
    with pytest.raises(ValueError, match='z_prime units'):
        unit_cell_gather_index(batch_ind, torch.tensor([2, 2]), torch.tensor([2, 4]), 2)
    with pytest.raises(ValueError, match='z_prime < 1'):
        unit_cell_gather_index(batch_ind, torch.tensor([1, 0]), torch.tensor([2, 4]), 2)
    idx, out_n = unit_cell_gather_index(batch_ind, torch.tensor([1, 2]), torch.tensor([2, 3]), 2)
    # crystal 0: 5 atoms x 2 images; crystal 1: units (5, 6) and (7, 8), each x 3 images, unit by unit
    assert idx.tolist() == [0, 1, 2, 3, 4] * 2 + [5, 6] * 3 + [7, 8] * 3
    assert out_n.tolist() == [10, 12]


# ------------------------------------------------------------------------------------------------
# the other readers of unit-cell elements, each on the two-compound fixtures, with the old tiling as the control
# (added after the stage-4 review, 2026-10-23)
# ------------------------------------------------------------------------------------------------

def _truth(ref, pos):
    return CL.element_truth(pos, ref.pos, ref.z, ref.T_fc[0], ref.symmetry_operators[0])


@pytest.mark.parametrize('case', ['two_eth_ether', 'two_pyr_flb'])
def test_ase_unit_cell_mode_on_a_cluster(case):
    """ase_mol_from_crystaldata(mode='unit cell') on a CLUSTER batch reads the asymmetric unit as the graph's
    aux_ind == 0 atoms; the row slice z[ptr : ptr + num_atoms] it used there is unit 0's sub-cluster (unit 0 twice)."""
    import torch as T
    from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
    from mxtaltools.crystal_building.utils import unit_cell_z
    b = CL.build_case(case)
    ref = b.clone()
    ref.mol2ucell(std_orientation=True)
    _, cl = b.clone().analyze(['elj'], cutoff=6.0, return_cluster=True)
    atoms = ase_mol_from_crystaldata(cl, index=0, mode='unit cell')
    pos = T.as_tensor(atoms.get_positions())
    truth = _truth(ref, pos)
    assert int((T.as_tensor(atoms.numbers).long() != truth).sum()) == 0
    row_slice = cl.z[int(cl.ptr[0]):int(cl.ptr[0]) + int(cl.num_atoms[0])]
    old = unit_cell_z(row_slice, cl.z_prime.reshape(-1)[0], cl.sym_mult[0])
    assert int((old.long() != truth).sum()) > 0                   # the cluster's row slice is the wrong asymmetric unit


@pytest.mark.parametrize('case', ['two_eth_ether', 'two_pyr_flb'])
def test_compute_standard_cell_hands_spglib_the_right_elements(case, monkeypatch):
    """The numbers compute_standard_cell gives spglib follow the unit cell's layout (the whole-z tiling it used labels
    rows of the other compound wrongly, and spglib then finds a different group)."""
    import numpy as np
    import spglib
    import torch as T
    calls = []
    real = spglib.get_symmetry_dataset

    def recording(cell, *a, **k):
        calls.append(cell)
        return real(cell, *a, **k)

    monkeypatch.setattr(spglib, 'get_symmetry_dataset', recording)
    b = CL.build_case(case)
    ref = b.clone()
    ref.mol2ucell()
    try:
        b.clone().compute_standard_cell()
    except ValueError:
        pass                                                      # judged by the numbers, below
    assert calls, 'compute_standard_cell never called spglib'
    _, positions, numbers = calls[0]
    pos = T.as_tensor(np.asarray(positions), dtype=T.float64) @ ref.T_fc[0].double().T
    truth = _truth(ref, pos)
    assert int((T.as_tensor(np.asarray(numbers)).long() != truth).sum()) == 0
    assert int((_old_tiling(ref).long() != truth).sum()) > 0


@pytest.mark.parametrize('case', ['two_eth_ether', 'two_pyr_flb'])
def test_search_unit_cell_fractional_labels(case):
    """crystal_search.standardize.unit_cell_fractional labels each unit-cell row with its own element."""
    import numpy as np
    import torch as T
    from mxtaltools.crystal_search.standardize import unit_cell_fractional
    b = CL.build_case(case)
    ref = b.clone()
    ref.mol2ucell()
    (frac, z), = unit_cell_fractional(b)
    truth = _truth(ref, T.as_tensor(frac) @ ref.T_fc[0].double().T)
    assert int((T.as_tensor(np.asarray(z)).long() != truth).sum()) == 0
    assert int((_old_tiling(ref).long() != truth).sum()) > 0
