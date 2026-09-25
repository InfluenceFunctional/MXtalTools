"""The optional C2 cutoff envelope on the LJ family (lj, qlj, elj).

Off by default: analyze()/compute() without `lj_envelope` (or with None) must give energies bit-identical to the plain
sum of the unswitched pair functions. On: every pair energy is multiplied by lj_cutoff_envelope(r, cutoff, width),
which ends at the pair list's own recorded cutoff, so the energy and its forces are continuous as pairs cross it, and
only pairs inside [cutoff - width, cutoff] change.
"""
from pathlib import Path

import pytest
import torch
from torch_scatter import scatter

from mxtaltools.analysis.vdw_analysis import (lj_cutoff_envelope, lj_analysis, qlj_analysis, elj_analysis,
                                              compute_lj_edgewise, quadratic_edgewise_lj_energy,
                                              exponential_edgewise_lj_energy)
from mxtaltools.constants.atom_properties import VDW_RADII

VDW = torch.tensor([VDW_RADII[z] for z in range(100)], dtype=torch.float32)
MINI = Path(__file__).resolve().parents[1] / 'mini_datasets' / 'mini_new_csd.pt'
FAMILY = {'lj': (lj_analysis, lambda dd, v=VDW: compute_lj_edgewise(dd, v)),
          'qlj': (qlj_analysis, lambda dd, v=VDW: quadratic_edgewise_lj_energy(v, dd)),
          'elj': (elj_analysis, lambda dd, v=VDW: exponential_edgewise_lj_energy(v, dd, 2.5))}


def _pairs(r, cutoff=10.0, z=6, dtype=torch.float32):
    r = torch.as_tensor(r, dtype=dtype).flatten()
    n = len(r)
    return {'intermolecular_dist': r, 'intermolecular_dist_atoms': [torch.full((n,), z), torch.full((n,), z)],
            'intermolecular_dist_batch': torch.zeros(n, dtype=torch.long), 'cutoff': cutoff}


# ------------------------------------------------------------------ the switch itself
def _derivs(r, r_off, w):
    x = torch.tensor([r], dtype=torch.float64, requires_grad=True)
    g, = torch.autograd.grad(lj_cutoff_envelope(x, r_off, w).sum(), x, create_graph=True)
    h, = torch.autograd.grad(g.sum(), x)
    return float(g), float(h)


def test_envelope_shape_is_c2():
    r_off, w = 10.0, 1.0
    r = torch.linspace(8.0, 10.5, 2501, dtype=torch.float64)
    s = lj_cutoff_envelope(r, r_off, w)
    assert torch.all(s[r <= r_off - w] == 1.0)
    assert torch.all(s[r >= r_off] == 0.0)
    assert torch.all(s[1:] <= s[:-1] + 1e-15), 'the switch must be monotone non-increasing'
    for r0 in (r_off - w, r_off):                      # at the knots
        g, h = _derivs(r0, r_off, w)
        assert abs(g) < 1e-9 and abs(h) < 1e-9, f'S not C2 at r = {r0}: dS {g}, d2S {h}'
    for r0 in (r_off - w + 1e-4, r_off - 1e-4):        # just inside: a C1-only switch has |d2S| ~ 6 here, C2 ~ 6e-3
        g, h = _derivs(r0, r_off, w)
        assert abs(h) < 0.1, f'second derivative just inside the window is {h} at r = {r0}: not C2'


# ------------------------------------------------------------------ unit level, all three energies
@pytest.mark.parametrize('name', FAMILY)
def test_envelope_off_is_the_plain_sum(name):
    """Independent of the envelope code: the default equals scatter(edgewise pair energy), bit for bit."""
    fn, edgewise = FAMILY[name]
    dd = _pairs(torch.linspace(3.0, 10.0, 400))
    plain = scatter(edgewise(dd), dd['intermolecular_dist_batch'], reduce='sum', dim_size=1)
    assert torch.equal(fn(VDW, dd, 1), plain)
    assert torch.equal(fn(VDW, dd, 1, envelope=None), plain)


@pytest.mark.parametrize('cutoff', [6.0, 10.0])
@pytest.mark.parametrize('name', FAMILY)
def test_envelope_removes_the_cutoff_step(name, cutoff):
    """The switch ends at the dict's recorded cutoff (not a hard-coded radius)."""
    fn, _ = FAMILY[name]
    at_cut = _pairs([cutoff - 1e-4], cutoff=cutoff)
    stock = float(fn(VDW, at_cut, 1))
    switched = float(fn(VDW, at_cut, 1, envelope=1.0))
    assert abs(stock) > 1e-3, f'{name}: the unswitched pair energy just inside the cutoff should be nonzero'
    assert abs(switched) < 1e-9, f'{name}: the enveloped pair energy must vanish at the cutoff, got {switched}'
    inside = _pairs(torch.linspace(3.0, cutoff - 1.01, 200), cutoff=cutoff)
    assert torch.equal(fn(VDW, inside, 1), fn(VDW, inside, 1, envelope=1.0)), \
        f'{name}: pairs below cutoff - width must be untouched'


@pytest.mark.parametrize('name', FAMILY)
def test_envelope_gradient_is_exact(name):
    """Forces include the E * dS/dr term: gradcheck in float64 through the enveloped energy."""
    fn, _ = FAMILY[name]
    vdw64 = VDW.double()
    r = torch.linspace(8.6, 9.95, 12, dtype=torch.float64, requires_grad=True)

    def energy(rr):
        dd = _pairs(rr, dtype=torch.float64); dd['intermolecular_dist'] = rr
        return fn(vdw64, dd, 1, envelope=1.5)
    assert torch.autograd.gradcheck(energy, (r,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize('bad', [0.0, -0.5, 10.5])
def test_envelope_rejects_bad_width(bad):
    with pytest.raises(ValueError, match='width'):
        elj_analysis(VDW, _pairs([5.0]), 1, envelope=bad)


@pytest.mark.parametrize('bad', [True, False, '1.0'])
def test_envelope_rejects_non_numeric_width(bad):
    with pytest.raises(TypeError, match='width'):
        lj_analysis(VDW, _pairs([5.0]), 1, envelope=bad)


def test_envelope_requires_recorded_cutoff():
    dd = _pairs([5.0]); del dd['cutoff']
    with pytest.raises(ValueError, match='cutoff'):
        lj_analysis(VDW, dd, 1, envelope=1.0)


# ------------------------------------------------------------------ through analyze(), real crystals (CPU)
@pytest.fixture(scope='module')
def crystals():
    if not MINI.exists():
        pytest.skip(f'{MINI} not present')
    from mxtaltools.dataset_utils.utils import collate_data_list
    return collate_data_list(list(torch.load(MINI, weights_only=False))[:8])


@pytest.mark.parametrize('name', FAMILY)
def test_analyze_default_is_the_plain_sum(crystals, name):
    """analyze() without the kwarg, and with None, equals scatter of the raw pair energies (bitwise on CPU)."""
    a, cluster = crystals.clone().analyze([name], cutoff=10, supercell_size=5, return_cluster=True)
    b = crystals.clone().analyze([name], cutoff=10, supercell_size=5, lj_envelope=None)
    dd = cluster.edges_dict
    plain = scatter(FAMILY[name][1](dd), dd['intermolecular_dist_batch'], reduce='sum', dim_size=crystals.num_graphs)
    assert torch.equal(a[name], plain), f'{name}: default differs from the plain pair sum'
    assert torch.equal(b[name], plain), f'{name}: lj_envelope=None differs from the plain pair sum'


@pytest.mark.parametrize('name', FAMILY)
def test_analyze_envelope_changes_only_the_window(crystals, name):
    """Through analyze(): enveloped minus stock = sum over window pairs of (S - 1) * E_pair, per crystal; for elj
    times a KNOWN per-crystal lj_coeff."""
    batch = crystals.clone()
    coeff = torch.linspace(0.5, 3.0, batch.num_graphs) if name == 'elj' else torch.ones(batch.num_graphs)
    if name == 'elj':
        batch.lj_coeff = coeff.clone()
    stock, cluster = batch.clone().analyze([name], cutoff=10, supercell_size=5, return_cluster=True)
    switched = batch.clone().analyze([name], cutoff=10, supercell_size=5, lj_envelope=1.5)[name]
    dd = cluster.edges_dict
    assert dd['cutoff'] == 10.0
    d = dd['intermolecular_dist']
    assert d.max() <= dd['cutoff'] and (d > dd['cutoff'] - 0.1).any(), 'pair list must reach the recorded cutoff'
    s = lj_cutoff_envelope(d, 10.0, 1.5)
    expected = scatter(FAMILY[name][1](dd) * (s - 1.0), dd['intermolecular_dist_batch'], reduce='sum',
                       dim_size=batch.num_graphs) * coeff
    # the difference of two float32 sums of magnitude ~500 carries ~1e-3 of rounding; the effect is ~5-40
    assert torch.allclose(switched - stock[name], expected, atol=1e-2, rtol=1e-4)
    assert (switched - stock[name]).abs().max() > 1e-4, 'the envelope should change energies of real crystals'


def test_canonical_route_accepts_envelope(crystals):
    """The GFN reward's call shape, ['reduction_en', 'elj'], with the envelope on."""
    out = crystals.clone().analyze(['reduction_en', 'elj'], cutoff=10, supercell_size=5, lj_envelope=1.0)
    assert torch.isfinite(out['elj']).all() and torch.isfinite(out['reduction_en']).all()
