"""
Triclinic Niggli walls: mxtaltools/common/sym_utils.py::tri_niggli_reduction_penalty, which cell_reduction_penalty uses for
every sg 1/2 row by default. MXT_NIGGLI_TRICLINIC is retired and fails loudly if set; MXT_LEGACY_TRICLINIC_WALLS=1 is the
rarely-used way back to legacy_tri_reduction_penalty.

Contract. The penalty is 0 on exactly one cell per lattice among all unimodular basis changes (checked for |entries| <= 1). Its sign convention is beta, gamma obtuse with alpha free, not Niggli's all-acute / all-obtuse: spglib's
all-obtuse Niggli cells score 0 as they are, all-acute ones only after (a, b, c) -> (a, -b, -c).

CPU only.
"""
import itertools
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from mxtaltools.common import sym_utils
from mxtaltools.common.sym_utils import cell_reduction_penalty, legacy_tri_reduction_penalty, tri_niggli_reduction_penalty


def random_cells(n, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    L = torch.exp(torch.empty(n, 3, dtype=torch.float64).uniform_(np.log(2), np.log(40), generator=g))
    A = torch.empty(n, 3, dtype=torch.float64).uniform_(0.25 * np.pi, 0.75 * np.pi, generator=g)
    sg = torch.randint(1, 231, (n,), generator=g)
    sg[: n // 4] = 1
    sg[n // 4: n // 2] = 2
    return L.to(dtype), A.to(dtype), sg


def random_lattices(n, seed):
    """Random well-conditioned lattice bases, rows as vectors, float64."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        M = rng.normal(size=(3, 3)) * np.exp(rng.uniform(np.log(3), np.log(30), size=(3, 1)))
        if abs(np.linalg.det(M)) > 0.2 * np.prod(np.linalg.norm(M, axis=1)):
            out.append(M)
    return np.array(out)


def params_from_rows(L):
    G = L @ L.transpose(-1, -2)
    a, b, c = G[..., 0, 0].sqrt(), G[..., 1, 1].sqrt(), G[..., 2, 2].sqrt()
    al = (G[..., 1, 2] / (b * c)).clamp(-1, 1).acos()
    be = (G[..., 0, 2] / (a * c)).clamp(-1, 1).acos()
    ga = (G[..., 0, 1] / (a * b)).clamp(-1, 1).acos()
    return torch.stack([a, b, c], -1), torch.stack([al, be, ga], -1)


def hard_niggli(L, A):
    """Main conditions (beta, gamma obtuse convention) from the metric, and the smallest distance to any wall (relative)."""
    a, b, c = L.unbind(-1)
    Aa, Bb, Cc = a * a, b * b, c * c
    xi, eta, zeta = 2 * b * c * A[:, 0].cos(), 2 * a * c * A[:, 1].cos(), 2 * a * b * A[:, 2].cos()
    s = torch.stack([xi / Bb, eta / Aa, zeta / Aa], -1)
    t = (Aa + Bb + xi + eta + zeta) / (Aa + Bb)
    inside = ((Aa <= Bb) & (Bb <= Cc) & (s.abs() <= 1).all(-1) & (s[:, 1:] <= 0).all(-1) & (t >= 0))
    slack = torch.stack([(Bb - Aa) / Bb, (Cc - Bb) / Cc, (1 - s.abs()).abs().min(-1).values,
                         s[:, 1:].abs().min(-1).values, t.abs()], -1).abs().min(-1).values
    return inside, slack


def unimodular(k):
    rows = np.array(list(itertools.product(range(-k, k + 1), repeat=9)), dtype=np.float64).reshape(-1, 3, 3)
    return rows[np.abs(np.rint(np.linalg.det(rows))) == 1]


def test_legacy_walls_are_off_unless_the_env_asks():
    assert sym_utils.LEGACY_TRICLINIC_WALLS is (os.environ.get('MXT_LEGACY_TRICLINIC_WALLS', '0') == '1')
    code = 'from mxtaltools.common import sym_utils; print(sym_utils.LEGACY_TRICLINIC_WALLS)'
    env = {k: v for k, v in os.environ.items() if k != 'MXT_LEGACY_TRICLINIC_WALLS'}
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == 'False'
    out = subprocess.run([sys.executable, '-c', code], env={**env, 'MXT_LEGACY_TRICLINIC_WALLS': '1'}, capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == 'True' and 'legacy_tri_reduction_penalty' in out.stderr
    for bad in ('true', 'yes', '2'):
        out = subprocess.run([sys.executable, '-c', code], env={**env, 'MXT_LEGACY_TRICLINIC_WALLS': bad},
                             capture_output=True, text=True)
        assert out.returncode != 0 and 'MXT_LEGACY_TRICLINIC_WALLS' in out.stderr


@pytest.mark.parametrize('margin', [0.0, 0.1])
def test_legacy_walls_reproduce_the_old_penalty(monkeypatch, margin):
    """With the escape hatch on, sg 1/2 rows equal the old dispatch bit for bit (walls on the subset, the overlap term on the
    full batch, added after); every other row is untouched."""
    L, A, sg = random_cells(20000, 7)
    E_default = cell_reduction_penalty(A, L, sg, margin)
    monkeypatch.setattr(sym_utils, 'LEGACY_TRICLINIC_WALLS', True)
    E = cell_reduction_penalty(A, L, sg, margin)
    tri = (sg == 1) | (sg == 2)
    l, g = L[tri], A[tri]
    a, b, c = l.unbind(1)
    al, be, ga = g.unbind(1)
    bnd = lambda x: (torch.relu(x - (1 - margin)) ** 2) + (torch.relu((-1 + margin) - x) ** 2)
    old = (F.relu(l[:, 1] / l[:, 2] - (1 - margin)) ** 2 + F.relu(l[:, 0] / l[:, 1] - (1 - margin)) ** 2
           + bnd(al.cos() / (b / 2 / c).clamp(min=1e-6)) + bnd(be.cos() / (a / 2 / c).clamp(min=1e-6))
           + bnd(ga.cos() / (a / 2 / b).clamp(min=1e-6)))
    A_, B_, C_ = L.split(1, dim=1)
    overlap = (A_ * B_ * torch.cos(A[:, 2:3]) + A_ * C_ * torch.cos(A[:, 1:2]) + B_ * C_ * torch.cos(A[:, 0:1])).flatten()
    old = old + F.relu(overlap[tri] - margin) ** 2
    assert torch.equal(E[tri], old)
    assert torch.equal(E[tri], legacy_tri_reduction_penalty(l, g, margin))
    assert torch.equal(E[~tri], E_default[~tri])
    assert (E[tri] != E_default[tri]).float().mean() > 0.5


def test_retired_switch_fails_loudly():
    code = 'from mxtaltools.common import sym_utils'
    for value in ('0', '1'):
        env = {**os.environ, 'MXT_NIGGLI_TRICLINIC': value}
        out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        assert out.returncode != 0 and 'MXT_NIGGLI_TRICLINIC is retired' in out.stderr


@pytest.mark.parametrize('margin', [0.0, 0.1])
def test_triclinic_rows_use_the_niggli_walls(margin):
    L, A, sg = random_cells(20000, 0)
    E = cell_reduction_penalty(A, L, sg, margin)
    tri = (sg == 1) | (sg == 2)
    assert torch.equal(E[tri], tri_niggli_reduction_penalty(L[tri], A[tri], margin))
    # the positive-overlap term of the retired walls is gone: it would fire on these all-acute-ish cells
    overlap = L[:, 0] * L[:, 1] * A[:, 2].cos() + L[:, 0] * L[:, 2] * A[:, 1].cos() + L[:, 1] * L[:, 2] * A[:, 0].cos()
    assert ((overlap[tri] > 0) & (E[tri] == 0)).any()


def test_zero_set_is_the_niggli_main_conditions():
    L, A, _ = random_cells(200000, 2, dtype=torch.float64)
    L[::2] = L[::2].sort(dim=-1).values  # half the rows ordered, angles in [60, 120]: enough of them land inside
    A[::2] = np.pi / 3 + (A[::2] - 0.25 * np.pi) * (2 / 3)
    inside, slack = hard_niggli(L, A)
    far = slack > 1e-9
    E = tri_niggli_reduction_penalty(L, A, 0.0)
    assert inside.sum() > 1000 and (~inside).sum() > 1000
    assert torch.equal((E == 0)[far], inside[far])


def test_margin_zero_set_is_inside_the_margin_free_zero_set():
    L, A, _ = random_cells(200000, 3, dtype=torch.float64)
    Z0 = tri_niggli_reduction_penalty(L, A, 0.0) == 0
    Zm = tri_niggli_reduction_penalty(L, A, 0.05) == 0
    assert Zm.sum() > 100
    assert not (Zm & ~Z0).any()


def test_spglib_niggli_cells_score_zero_after_flipping_all_acute():
    spglib = pytest.importorskip('spglib')
    raw = random_lattices(300, 4)
    nig = torch.tensor(np.array([spglib.niggli_reduce(M, eps=1e-8) for M in raw]))
    L, A = params_from_rows(nig)
    acute = (A < np.pi / 2).all(-1)
    assert 50 < acute.sum() < 250
    E = tri_niggli_reduction_penalty(L, A, 0.0)
    assert (E[~acute] == 0).all() and (E[acute] > 0).all()
    # (a, -b, -c): alpha kept, beta and gamma -> 180 - angle
    flipped = torch.where(acute[:, None, None], nig * torch.tensor([1.0, -1.0, -1.0])[:, None], nig)
    L, A = params_from_rows(flipped)
    assert (tri_niggli_reduction_penalty(L, A, 0.0) == 0).all()
    L_raw, A_raw = params_from_rows(torch.tensor(raw))
    assert (tri_niggli_reduction_penalty(L_raw, A_raw, 0.0) > 0).float().mean() > 0.9
    # production dispatch, float32
    E = cell_reduction_penalty(A.float(), L.float(), torch.full((300,), 2), 0.0)
    assert E.max() < 1e-6


def test_alpha_crosses_90_without_a_jump():
    """The reason for the convention: sweep angle(b, c) through 90 deg; the zero-penalty cell moves continuously
    (Niggli's all-same-sign rule jumps by ~36 deg here)."""
    spglib = pytest.importorskip('spglib')
    mats = torch.tensor(unimodular(1))
    a = np.array([5.0, 0, 0])
    b = 6.3 * np.array([np.cos(np.deg2rad(97)), np.sin(np.deg2rad(97)), 0])
    bh = b / np.linalg.norm(b)
    n = np.cross(a, b) / np.linalg.norm(np.cross(a, b))
    w = 0.93 * n + 0.37 * np.cross(n, bh)
    w /= np.linalg.norm(w)
    reps = []
    for t in np.linspace(-6, 6, 25):
        th = np.deg2rad(90 + t)
        c = 7.4 * (np.cos(th) * bh + np.sin(th) * w)
        L, A = params_from_rows(mats @ torch.tensor(spglib.niggli_reduce(np.stack([a, b, c]), eps=1e-8)))
        zero = tri_niggli_reduction_penalty(L, A, 0.0) == 0
        assert zero.any()
        reps.append(torch.rad2deg(A[zero][0]))
    reps = torch.stack(reps)
    assert reps[0, 0] > 90 > reps[-1, 0]  # alpha did cross 90
    assert (reps[1:] - reps[:-1]).abs().max() < 1.0


def test_exactly_one_zero_penalty_cell_per_lattice():
    spglib = pytest.importorskip('spglib')
    mats = torch.tensor(unimodular(1))
    assert len(mats) == 6960
    raw = random_lattices(200, 5)
    for M in raw:
        base = torch.tensor(spglib.niggli_reduce(M, eps=1e-8))
        L, A = params_from_rows(mats @ base)
        E = tri_niggli_reduction_penalty(L, A, 0.0)
        cells = np.unique(torch.cat([L, A], -1)[E == 0].numpy().round(6), axis=0)
        assert len(cells) == 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('margin', [0.0, 0.1])
def test_values_and_gradients_finite_at_extremes(dtype, margin):
    L = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1e-7, 1e3, 1e3], [1e3, 1e-7, 1e3],
                      [5.0, 5.0, 5.0], [3.0, 7.0, 11.0], [1e3, 1.0, 1e-3]], dtype=dtype, requires_grad=True)
    A = torch.tensor([[0.0, np.pi, np.pi / 2], [np.pi / 2] * 3, [np.pi / 3] * 3, [2.0, 2.0, 2.0],
                      [np.pi / 2, 1.0, np.pi], [np.pi / 2] * 3, [1.2, 1.9, 1.5], [0.1, 3.0, 1.5]], dtype=dtype,
                     requires_grad=True)
    E = tri_niggli_reduction_penalty(L, A, margin)
    assert torch.isfinite(E).all() and (E >= 0).all()
    E.sum().backward()
    assert torch.isfinite(L.grad).all() and torch.isfinite(A.grad).all()
