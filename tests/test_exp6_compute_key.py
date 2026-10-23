"""The 'exp6' compute key and relaxation target: exp6_ff wired through MolCrystalAnalysis.analyze and the optimizer.

1. analyze(['exp6']) equals exp6_ff.crystal_exp6_energy in its 'exp6' pair form on the same crystals (Z' = 1 and
   Z' = 2; FIT and W99; with MMFF charges and without), so the shared cluster analyze builds is the one the module
   builds for itself. The key's pair form is fixed: exp6_form or exp6_k_factor raises.
2. Requesting 'exp6' beside 'elj' leaves the eLJ energy unchanged: the force-field columns written into x for the
   shared cluster are not read by eLJ.
3. compute_exp6_energy refuses a cluster whose x lacks the force-field columns instead of scoring every atom as type 0.
4. The energy is differentiable in the cell parameters through analyze (the typed clone does not detach the path).
5. optimize_crystal_parameters with optim_target 'exp6' descends, and the energy it reports on the returned crystals
   is the one for the parameter set it was given (exp6_kwargs reach the final analyze too).

CPU only.
"""
from pathlib import Path

import pytest
import torch

from mxtaltools.analysis import exp6_ff as ff
from mxtaltools.dataset_utils.utils import collate_data_list

ACRIDINE = Path(__file__).resolve().parent / 'datasets' / 'mini_acridine.pt'
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'exp6', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']


def _strip(c):
    c = c.clone()
    for k in DROP:
        if k in c.keys():
            delattr(c, k)
    return c


@pytest.fixture(scope='module')
def acridine():
    if not ACRIDINE.exists():
        pytest.skip(f'{ACRIDINE} not present')
    data = torch.load(ACRIDINE, weights_only=False)
    zp1 = [_strip(c) for c in data if int(c.sg_ind) == 14 and int(c.z_prime) == 1]
    zp2 = [_strip(c) for c in data if int(c.sg_ind) == 14 and int(c.z_prime) == 2]
    return zp1, zp2


def _batch(acridine, which, n):
    zp1, zp2 = acridine
    rows = zp1 if which == 'zp1' else zp2
    return collate_data_list([c.clone() for c in rows[:n]], max_z_prime=1 if which == 'zp1' else 2)


@pytest.mark.parametrize('which', ['zp1', 'zp2'])
@pytest.mark.parametrize('params', ['fit', 'w99'])
@pytest.mark.parametrize('charges', ['mmff', None])
def test_analyze_equals_the_module(acridine, which, params, charges):
    b = _batch(acridine, which, 2)
    via_analyze = b.analyze(['exp6'], cutoff=10, exp6_params=params, exp6_charges=charges)['exp6']
    direct = ff.crystal_exp6_energy(b, params, charges=charges, form='exp6', cutoff=10)
    assert torch.isfinite(via_analyze).all()
    assert torch.allclose(via_analyze, direct, rtol=1e-5, atol=1e-4), (via_analyze, direct)


def test_defaults_are_fit_with_mmff_in_the_exp6_form(acridine):
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import EXP6_DEFAULT_CHARGES, EXP6_DEFAULT_PARAMS
    assert (EXP6_DEFAULT_PARAMS, EXP6_DEFAULT_CHARGES) == ('fit', 'mmff')
    b = _batch(acridine, 'zp1', 2)
    assert torch.allclose(b.analyze(['exp6'], cutoff=10)['exp6'],
                          ff.crystal_exp6_energy(b, 'fit', charges='mmff', form='exp6', cutoff=10),
                          rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize('option', [{'exp6_form': 'elj'}, {'exp6_form': 'exp6'}, {'exp6_k_factor': 5.0}])
def test_the_pair_form_cannot_be_set(acridine, option):
    b = _batch(acridine, 'zp1', 1)
    with pytest.raises(ValueError, match='pair form'):
        b.analyze(['exp6'], cutoff=10, **option)


def test_exp6_beside_elj_leaves_elj_unchanged(acridine):
    b = _batch(acridine, 'zp2', 2)
    alone = b.analyze(['elj'], cutoff=10)['elj']
    together = b.analyze(['elj', 'exp6'], cutoff=10)
    assert torch.allclose(alone, together['elj'], rtol=1e-6, atol=1e-6)
    assert torch.isfinite(together['exp6']).all()


def test_a_cluster_without_force_field_columns_is_refused(acridine):
    b = _batch(acridine, 'zp1', 2)
    _, cluster = b.analyze(['elj'], cutoff=10, return_cluster=True)
    cluster.x = torch.zeros(len(cluster.z), 1)
    with pytest.raises(RuntimeError, match='force-field columns'):
        cluster.compute_exp6_energy()


def test_gradient_through_analyze(acridine):
    """float32 end to end: a finite gradient that agrees with a central difference to within float32 rounding and
    the curvature of the step; catches a detached path through the typed clone."""
    b = _batch(acridine, 'zp1', 1)
    p0 = b.full_cell_parameters().detach().clone()

    def energy(p):
        c = b.clone()
        c.set_cell_parameters(p, skip_box_analysis=False)
        return c.analyze(['exp6'], cutoff=10, exp6_envelope=1.0)['exp6'].sum()

    p = p0.clone().requires_grad_(True)
    g, = torch.autograd.grad(energy(p), p)
    assert torch.isfinite(g).all() and float(g.abs().sum()) > 0
    for k, h in ((0, 1e-2), (2, 1e-2), (7, 1e-3)):
        dp = torch.zeros_like(p0)
        dp[0, k] = h
        with torch.no_grad():
            fd = (energy(p0 + dp) - energy(p0 - dp)) / (2 * h)
        assert float(g[0, k]) == pytest.approx(float(fd), rel=0.05, abs=1.0), (k, float(g[0, k]), float(fd))


def test_optimizer_descends_and_reports_its_own_parameter_set(acridine, tmp_path):
    b = _batch(acridine, 'zp1', 2)
    kwargs = {'params': 'w99', 'charges': 'mmff'}
    out, rec = b.optimize_crystal_parameters(optim_target='exp6', exp6_kwargs=kwargs, max_num_steps=60,
                                             convergence_eps=0.0, init_lr=0.01, optimizer_func='rprop', cutoff=10,
                                             show_tqdm=False, return_record=True,
                                             intermediates_path=str(tmp_path / 'opt_intermediates.pt'))
    e = rec['exp6'].detach().reshape(rec['exp6'].shape[0], -1)
    assert torch.isfinite(e).all()
    assert bool((e[-1] < e[0]).all()), (e[0], e[-1])
    relaxed = collate_data_list(out, max_z_prime=1)
    with torch.no_grad():
        again = relaxed.analyze(['exp6'], cutoff=10, exp6_params='w99', exp6_charges='mmff')['exp6']
    reported = torch.stack([torch.as_tensor(c.exp6).reshape(()) for c in out])
    assert torch.allclose(reported, again, rtol=1e-4, atol=1e-3), (reported, again)
