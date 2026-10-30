"""crystal_search.basin_set: the writer and reader of basin sets (GFN energy_sampling/docs/design/basin_set.md).
What each test would catch:

  * a written row whose p, H, E or E_env do not come back as given (H and E_env are stored x kT, in energy units);
  * the importer refusing rows posed the way the search builds them (pose_aunit, std_orientation=True);
  * a live mask that keeps a group-fixed angle or a free centroid axis;
  * a crystal_fd dataset that is not a basin set read as one.

CPU, synthetic crystals, no energy computed.

    python -m pytest -q tests/test_basin_set.py
"""
import numpy as np
import pytest
import torch

from mxtaltools.crystal_fd import checks, dataset as fdd, domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.io_mxt import _Units
from mxtaltools.crystal_search.basin_set import FORMAT, live_mask, read_basin_set, write_basin_set

F64 = torch.float64
KT = 6.86
ENERGY = {'target': 'elj', 'model_id': 'elj:lj_coeff=1', 'units': 'raw eLJ', 'per': 'asymmetric_unit'}


def _molecule(seed):
    """A chiral 6-atom molecule in the live builder's standard frame, heavy-centred."""
    from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
    g = np.random.default_rng(seed)
    z = torch.tensor([6, 7, 8, 9, 17, 1])
    pos = torch.as_tensor(g.normal(size=(6, 3)) * 1.4, dtype=torch.float32)
    u = _Units(pos.clone(), z, torch.zeros(6, dtype=torch.long), 1)
    align_mol_batch_to_standard_axes(u, handedness=None)
    return MoleculeRef.from_coords(u.pos.double(), z)


def _canonical(sg, n, seed):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    cells = torch.as_tensor(checks.random_cells('triclinic', 4 * n, g), dtype=F64)
    cells[:, :3] = torch.as_tensor(rng.uniform(7.0, 14.0, (4 * n, 3)))
    cells = cells[ops.real_cells(cells)][:n]
    d = CrystalDescription.from_parts(sg, cells, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g),
                                      torch.ones(n, dtype=torch.long))
    d = domains.fold(ops.reduce_cell(d).desc).desc
    return d[~fdd.in_clip_band(d)]


def _search_rows(n=4, seed=3):
    """MolCrystalData rows shaped like minima.minimum_rows output: the identity description in id_params, a Hessian of
    E/kT, minimum_f, the hard-cutoff energy elj, ids and hit counts. pos is the molecule in its own frame."""
    from mxtaltools.crystal_fd.dataset import to_mol_crystal_data
    d = _canonical(2, n, seed)
    mols = [_molecule(seed + i) for i in range(d.num_rows)]
    rows = to_mol_crystal_data(d, mols, torch.ones(d.num_rows, dtype=torch.long), mode='body',
                               identifier=[f'M{i}' for i in range(d.num_rows)])
    L, A, t, rv, h = d.crystal_params()
    g = torch.Generator().manual_seed(seed)
    for i, r in enumerate(rows):
        r.id_params = torch.cat([L[i], A[i], t[i], rv[i]]).reshape(1, 12).float()
        r.id_handedness = torch.ones(1, 1)
        M = torch.randn(12, 12, generator=g)
        r.id_hessian = (M @ M.T + 12 * torch.eye(12)).reshape(1, 12, 12)
        r.minimum_f = torch.tensor([-50.0 + i])
        r.elj = torch.tensor([-350.0 + 7 * i])
        r.minimum_id = torch.tensor([i])
        r.hits, r.seed_hits, r.polish_status = torch.tensor([5 + i]), torch.tensor([0]), torch.tensor([1])
    return rows


def test_written_rows_come_back_with_their_parameters_hessian_and_energies(tmp_path):
    rows = _search_rows()
    path = str(tmp_path / 'basins')
    write_basin_set(rows, path, kT=KT, energy_attr='elj', energy=ENERGY, search={'run': 'test'},
                    identity={'merge_d2': 1.0})
    ds, header = read_basin_set(path)
    assert header['format'] == FORMAT and header['search']['kT'] == KT
    assert header['counts'] == {'rows_in': len(rows), 'written': len(rows), 'refused': {}}
    assert len(ds) == len(rows)
    assert all(ds.status_names[int(s)] == 'canonical' for s in ds.status)
    c = ds.columns
    for i, r in enumerate(rows):
        assert torch.allclose(c['p'][i], r.id_params.double().reshape(12))
        H = r.id_hessian.double().reshape(12, 12) * KT
        assert torch.allclose(c['H'][i], 0.5 * (H + H.T))
        assert float(c['E'][i]) == pytest.approx(float(r.elj))
        assert float(c['E_env'][i]) == pytest.approx(float(r.minimum_f) * KT)
        assert int(c['basin_id'][i]) == i and int(c['hits'][i]) == 5 + i
    assert bool(c['live_mask'].all())        # P-1: every cell parameter is a degree of freedom


def test_the_live_mask_drops_fixed_angles_and_free_axes():
    assert live_mask(2).all()
    m14 = live_mask(14)
    assert m14.tolist() == [True, True, True, False, True, False] + [True] * 6
    m4 = live_mask(4)                          # P2_1: b is a free centroid axis
    assert m4[7].item() is False and m4[6].item() and m4[8].item()
    with pytest.raises(NotImplementedError, match='tetragonal'):
        live_mask(75)


def test_a_plain_crystal_fd_dataset_is_not_read_as_a_basin_set(tmp_path):
    d = _canonical(2, 2, 11)
    mols = [_molecule(11 + i) for i in range(d.num_rows)]
    path = str(tmp_path / 'plain')
    fdd.save_dataset(path, d, mols, [f'P{i}' for i in range(d.num_rows)])
    with pytest.raises(ValueError, match='not a basin_set'):
        read_basin_set(path)


def test_the_writer_needs_an_energy_record_and_rows(tmp_path):
    with pytest.raises(ValueError, match='no rows'):
        write_basin_set([], str(tmp_path / 'a'), kT=KT, energy_attr='elj', energy=ENERGY, search={}, identity={})
    with pytest.raises(ValueError, match="lacks 'units'"):
        write_basin_set(_search_rows(1), str(tmp_path / 'b'), kT=KT, energy_attr='elj',
                        energy={'target': 'elj', 'model_id': 'x', 'per': 'molecule'}, search={}, identity={})
