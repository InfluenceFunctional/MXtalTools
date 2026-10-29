"""MolCrystalData.fd_chart_coordinates and plot_batch_cell_params(space='fd'): batch rows as points s in [-1, 1]^d of
the FD chart (``CrystalChart.for_latent_v1``, hemisphere orientation chart).

What each test would catch:

  * ``charts.to_unit`` not the inverse of ``charts.from_unit`` (a bound evaluated on the wrong prefix, a circle
    coordinate not wrapped);
  * a chart point built into a MolCrystalData batch (float32 storage) not charted back to the same s, in a triclinic,
    monoclinic, sg 6-9 (K cut), tetragonal and trigonal group;
  * a row outside the length box charted instead of reported;
  * the violin figure not laid out one subplot per chart coordinate, titled by the chart's names.
"""
import pytest
import torch

from mxtaltools.crystal_fd import charts

from ._helpers import crystal_batch, mol_ref

F64 = torch.float64
GROUPS = [1, 2, 7, 14, 92, 148]


def _radius(mol):
    from mxtaltools.dataset_utils.data_classes import MolData
    m = MolData(z=torch.as_tensor(mol['z']), pos=torch.as_tensor(mol['pos']).float(),
                x=torch.zeros(len(mol['z']), 1), do_mol_analysis=True)
    return float(m.radius)


def _circle_err(chart, a, b):
    e = (a - b).abs()
    wm = torch.tensor([d.kind == 'circle' for d in chart.dims])
    e[:, wm] = torch.minimum(e[:, wm], 2 - e[:, wm])
    return e


def _chart_batch(sg, mol, n, seed):
    radius = _radius(mol)
    chart = charts.CrystalChart.for_latent_v1(sg, torch.tensor([radius], dtype=F64), 'hemisphere')
    g = torch.Generator().manual_seed(seed)
    u = 0.02 + 0.96 * torch.rand(n, chart.ndim, generator=g, dtype=F64)
    y, _ = charts.from_unit(chart, u)
    desc, _ = chart.forward(y)
    return chart, u, desc


@pytest.mark.parametrize('sg', GROUPS)
def test_to_unit_inverts_from_unit(sg, chiral_molecule):
    chart, u, _ = _chart_batch(sg, chiral_molecule, 64, sg)
    y, _ = charts.from_unit(chart, u)
    assert float((charts.to_unit(chart, y) - u).abs().max()) < 1e-12


@pytest.mark.parametrize('sg', GROUPS)
def test_built_chart_points_chart_back(sg, chiral_molecule):
    chart, u, desc = _chart_batch(sg, chiral_molecule, 24, 100 + sg)
    batch = crystal_batch(desc, mol_ref(chiral_molecule))
    s, ok, status = batch.fd_chart_coordinates()
    assert bool(ok.all()), status
    assert float(_circle_err(chart, s, 2 * u - 1).max()) < 1e-3  # float32 storage


def test_outside_length_box_is_reported(chiral_molecule):
    chart, u, desc = _chart_batch(14, chiral_molecule, 4, 7)
    batch = crystal_batch(desc, mol_ref(chiral_molecule))
    batch.cell_lengths[0] = batch.cell_lengths[0] * 20
    s, ok, status = batch.fd_chart_coordinates()
    assert status[0] == 'outside_length_box' and not bool(ok[0]) and bool(torch.isnan(s[0]).all())
    assert bool(ok[1:].all())


def test_violin_figure_is_one_subplot_per_chart_coordinate(chiral_molecule):
    chart, u, desc = _chart_batch(14, chiral_molecule, 24, 3)
    batch = crystal_batch(desc, mol_ref(chiral_molecule))
    fig = batch.plot_batch_cell_params(space='fd', show=False, return_fig=True)
    assert [a.text for a in fig.layout.annotations] == [d.name for d in chart.dims]
    assert len(fig.data) == chart.ndim
