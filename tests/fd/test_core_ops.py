"""Core operations of crystal_fd: orientation charts, affine actions, basis changes, images and handedness.

Contracts
1. Every orientation chart inverts exactly (matrix -> chart -> matrix within 1e-12), including at r = 0, pi,
   2 pi and the poles; Euler (alpha, cos beta, gamma) is exact for beta >= 1e-4 and limited by the float64
   resolution of cos beta below that.
2. The hemisphere chart is MXtalTools' trainer convention: rotmat2rotvec_stable -> canonicalize_rotvec ->
   cart2sph_rotvec, and back through sph2cart_rotvec -> rotvec2rotmat.
3. Each chart's Haar log-density equals log(sqrt(det g) / 8 pi^2), g the bi-invariant metric pulled back by the
   chart (central differences); the quaternion hemisphere is uniform.
4. ops.apply is MolCrystalOps._transform_aunit_params on matrices; compose / inverse are the affine algebra.
5. change_basis moves no atom (float64 literal oracle) in sg 1, 2, 14 and 15, origin shifts included.
6. images, to_sheet and the free-axis gauge agree with the table and with canonicalize_free_axes.
7. Guards: descriptions keep every field on the device of t; reduce_cell refuses cells with no real metric;
   lattice_automorphisms counts the proper holohedry of a reduced cell from any basis; round_trip compares atoms
   of a float32 path within a given cell tolerance; same_crystal's chunking bounds memory without changing it.
"""
import math
import types

import numpy as np
import pytest
import torch

from mxtaltools.common.geometry_utils import cart2sph_rotvec, rotmat2rotvec_stable, rotvec2rotmat, sph2cart_rotvec
from mxtaltools.constants.space_group_info import LATTICE_TYPE
from mxtaltools.crystal_building.utils import canonicalize_rotvec
from mxtaltools.crystal_fd import domains, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.tables import group_table
from mxtaltools.crystal_search.standardize import setting_shift
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import MolCrystalOps

F64 = torch.float64


def _special_rotations():
    """Rotations at and near the chart singularities: identity, angle pi, near 2 pi (hemisphere), poles, equator."""
    axes = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [1, -2, 0.5], [0.3, 0.4, -1e-9]], dtype=F64)
    axes = axes / axes.norm(dim=1, keepdim=True)
    angles = torch.tensor([0.0, 1e-14, 1e-9, 1e-6, math.pi - 1e-9, math.pi, math.pi - 1e-6, 1.0], dtype=F64)
    v = (axes[:, None] * angles[None, :, None]).reshape(-1, 3)
    return so3.rotvec_to_matrix(v)


@pytest.mark.parametrize('chart', so3.CHARTS)
def test_chart_round_trip(chart):
    g = torch.Generator().manual_seed(1)
    R = torch.cat([so3.random_rotations(20000, g), _special_rotations()])
    back = so3.from_chart(chart, so3.to_chart(chart, R))
    err = (back - R).abs().amax((1, 2))
    if chart == 'euler_zyz':
        beta = torch.arccos(R[:, 2, 2].clamp(-1, 1))
        far = (beta >= 1e-4) & (math.pi - beta >= 1e-4)
        assert err[far].max() <= 1e-12
        assert err.max() <= 3e-8  # u = cos(beta) resolves beta only to ~1.5e-8 in float64
    else:
        assert err.max() <= 1e-12
    eye = torch.eye(3, dtype=F64)
    assert (back @ back.transpose(1, 2) - eye).abs().max() <= 1e-12


@pytest.mark.parametrize('chart', so3.CHARTS)
def test_chart_coordinates_round_trip(chart):
    """chart -> matrix -> chart on the open chart domain, away from its seams."""
    g = torch.Generator().manual_seed(2)
    n = 5000
    u = torch.rand(n, 3, generator=g, dtype=F64) * 0.98 + 0.01
    if chart == 'hemisphere':
        c = torch.stack([u[:, 0] * math.pi / 2, (2 * u[:, 1] - 1) * math.pi, u[:, 2] * 2 * math.pi], 1)
    elif chart == 'euler_zyz':
        c = torch.stack([u[:, 0] * 2 * math.pi, 2 * u[:, 1] - 1, u[:, 2] * 2 * math.pi], 1)
    elif chart == 'axis_angle':
        d = torch.randn(n, 3, generator=g, dtype=F64)
        c = d / d.norm(dim=1, keepdim=True) * (u[:, :1] * math.pi)
    else:
        q = torch.randn(n, 4, generator=g, dtype=F64)
        q = q / q.norm(dim=1, keepdim=True)
        c = torch.where(q[:, :1] < 0, -q, q)
    back = so3.to_chart(chart, so3.from_chart(chart, c))
    assert (back - c).abs().max() <= 1e-10


def test_hemisphere_is_the_trainer_convention():
    g = torch.Generator().manual_seed(3)
    R = torch.cat([so3.random_rotations(20000, g), _special_rotations()])
    sph = so3.matrix_to_hemisphere(R)
    ref = cart2sph_rotvec(canonicalize_rotvec(rotmat2rotvec_stable(R)))
    generic = sph[:, 2] > 1e-4                         # cart2sph_rotvec adds 1e-12 under its sqrt
    assert (sph[generic] - ref[generic]).abs().max() <= 1e-10
    assert bool((sph[:, 0] <= math.pi / 2 + 1e-15).all()) and bool((sph[:, 2] <= 2 * math.pi).all())
    away = (sph[:, 2] > 1e-6) & (sph[:, 2] < 2 * math.pi - 1e-6)
    live = rotvec2rotmat(sph2cart_rotvec(sph[away]))
    assert (live - R[away]).abs().max() <= 1e-10
    assert (so3.hemisphere_to_matrix(sph) - R).abs().max() <= 1e-12


def _pullback_log_density(chart, c, h=1e-6):
    """log(sqrt(det g) / 8 pi^2), g_ij = tr(dR_i^T dR_j) / 2, by central differences in chart coordinates."""
    J = []
    for i in range(3):
        e = torch.zeros(3, dtype=F64)
        e[i] = h
        J.append(((so3.from_chart(chart, c + e) - so3.from_chart(chart, c - e)) / (2 * h)).reshape(-1, 9))
    J = torch.stack(J, 1)                                                      # [n, 3, 9]
    g = J @ J.transpose(1, 2) / 2
    return 0.5 * torch.logdet(g) - math.log(8 * math.pi ** 2)


@pytest.mark.parametrize('chart', ['hemisphere', 'euler_zyz', 'axis_angle'])
def test_haar_density_is_the_pullback(chart):
    g = torch.Generator().manual_seed(4)
    R = so3.random_rotations(500, g)
    c = so3.to_chart(chart, R)
    if chart == 'hemisphere':
        keep = (c[:, 0] > 0.05) & (c[:, 0] < math.pi / 2 - 0.05) & (c[:, 2] > 0.05) & (c[:, 2] < 2 * math.pi - 0.05)
    elif chart == 'euler_zyz':
        keep = c[:, 1].abs() < 0.95
    else:
        keep = (c.norm(dim=1) < math.pi - 0.05)
    c = c[keep]
    assert (so3.haar_log_density(chart, c) - _pullback_log_density(chart, c)).abs().max() <= 1e-6
    if chart == 'axis_angle':
        assert torch.isfinite(so3.haar_log_density(chart, torch.zeros(1, 3, dtype=F64))).all()


def test_quaternion_hemisphere_is_uniform():
    """Haar rotations give w with density (4/pi) sqrt(1 - w^2) on [0, 1]: mean 4 / (3 pi)."""
    g = torch.Generator().manual_seed(5)
    n = 200000
    w = so3.matrix_to_quaternion(so3.random_rotations(n, g))[:, 0]
    assert bool((w >= 0).all())
    sd = math.sqrt(0.25 - (4 / (3 * math.pi)) ** 2)  # E[w^2] = 1/4
    assert abs(w.mean().item() - 4 / (3 * math.pi)) <= 5 * sd / math.sqrt(n)


def test_split_o3_and_pose():
    g = torch.Generator().manual_seed(6)
    R = so3.random_rotations(100, g)
    h = torch.randint(0, 2, (100,), generator=g) * 2 - 1
    P = R * torch.stack([h.to(F64), torch.ones(100, dtype=F64), torch.ones(100, dtype=F64)], 1)[:, None]
    R2, h2 = so3.split_o3(P)
    assert torch.equal(h2, h) and (R2 - R).abs().max() <= 1e-14


# ---------------------------------------------------------------------------
# description
# ---------------------------------------------------------------------------

def test_description_round_trip_and_molecule_ref(chiral_molecule):
    g = torch.Generator().manual_seed(7)
    n = 50
    cell = torch.tensor([[7.0, 8.0, 9.0, 1.4, 1.7, 1.8]], dtype=F64).expand(n, 6)
    d = CrystalDescription.from_parts(2, cell, torch.rand(n, 3, generator=g, dtype=F64), so3.random_rotations(n, g),
                                      torch.randint(0, 2, (n,), generator=g) * 2 - 1)
    d.validate()
    L, A, c, rv, h = d.crystal_params()
    e = CrystalDescription.from_crystal_params(d.sg, L, A, c, rv, h)
    assert (e.R - d.R).abs().max() <= 1e-12 and torch.equal(e.h, d.h) and torch.equal(e.t, d.t)
    assert (rv.reshape(-1, 3)[:, 2] >= 0).all()
    pos = torch.as_tensor(chiral_molecule['pos']) + torch.tensor([1.0, -2.0, 3.0], dtype=F64)
    ref = MoleculeRef.from_coords(pos, chiral_molecule['z'])
    heavy = ref.z > 1
    assert ref.pos[heavy].mean(0).abs().max() <= 1e-14
    assert ref.hash == MoleculeRef.from_coords(ref.pos, ref.z).hash
    with pytest.raises(ValueError):
        d.replace(h=d.h * 2).validate()


# ---------------------------------------------------------------------------
# affine actions and basis changes
# ---------------------------------------------------------------------------

def _rand_desc(sg, n, seed, cell_sampler, h=None):
    g = torch.Generator().manual_seed(seed)
    cell = torch.tensor(cell_sampler(LATTICE_TYPE[sg], n, seed=seed), dtype=F64)
    if h is None:
        h = torch.randint(0, 2, (n,), generator=g) * 2 - 1
    return CrystalDescription.from_parts(sg, cell, torch.rand(n, 3, generator=g, dtype=F64),
                                         so3.random_rotations(n, g), h)


def test_compose_inverse():
    tab = group_table(92)
    W, w = tab.gamma_W, tab.gamma_w
    a, b = (W[3], w[3]), (W[17], w[17])
    Wi, wi = ops.inverse(a)
    Wc, wc = ops.compose(a, (Wi, wi))
    assert (Wc - torch.eye(3, dtype=F64)).abs().max() == 0 and wc.abs().max() <= 1e-15
    Wab, wab = ops.compose(a, b)
    x = torch.rand(10, 3, dtype=F64)
    lhs = (a[0] @ ((b[0] @ x.T).T + b[1]).T).T + a[1]
    assert ((Wab @ x.T).T + wab - lhs).abs().max() <= 1e-14


@pytest.mark.parametrize('sg', [2, 14, 19, 92, 148])
def test_apply_matches_transform_aunit_params(sg, cell_sampler):
    """ops.apply equals MolCrystalOps._transform_aunit_params for every element of the normaliser table."""
    tab = group_table(sg)
    d = _rand_desc(sg, 200, sg, cell_sampler)
    T = d.cell_matrix()
    ns = types.SimpleNamespace(T_fc=T, T_cf=torch.linalg.inv(T))
    rv = so3.matrix_to_rotvec(d.R[:, 0])
    for k in range(len(tab.gamma_W)):
        W, w = tab.gamma_W[k], tab.gamma_w[k]
        c_ref, o_ref, h_ref = MolCrystalOps._transform_aunit_params(ns, d.t[:, 0], rv, d.h[:, 0].to(F64), W, w)
        out = ops.apply(d, W, w, wrap=True)
        R_ref = rotvec2rotmat(o_ref)
        angle = so3.matrix_to_rotvec(out.R[:, 0]).norm(dim=1)
        ok = (angle > 1e-3) & (angle < math.pi - 1e-3)  # rotmat2rotvec is replaced near 0 and pi
        assert (out.t[:, 0] - c_ref).abs().max() <= 1e-12
        assert torch.equal(out.h[:, 0], h_ref.round().long())
        assert (out.R[ok, 0] - R_ref[ok]).abs().max() <= 1e-9


def _atoms(desc, X, oracle):
    tab = group_table(int(desc.sg[0]))
    return [oracle.unit_cell(desc.cell[i].numpy(), desc.t[i, 0].numpy(), desc.pose[i, 0].numpy(), X,
                             tab.g_W.numpy(), tab.g_w.numpy()) for i in range(desc.num_rows)]


def _setting_changes(sg, rng, n):
    out = []
    while len(out) < n:
        if sg <= 2:
            N = rng.integers(-2, 3, (3, 3))
        else:
            p, q, r, s = rng.integers(-2, 3, 4)
            dd = p * s - q * r
            N = np.array([[p, 0, r], [0, dd, 0], [q, 0, s]])
        if abs(round(np.linalg.det(N))) == 1 and round(np.linalg.det(N)) == 1 and setting_shift(N, sg) is not None:
            out.append(N)
    return np.stack(out)


@pytest.mark.parametrize('sg', [1, 2, 14, 15])
def test_change_basis_moves_no_atom(sg, chiral_molecule, cell_sampler, literal_oracle):
    rng = np.random.default_rng(sg)
    d = _rand_desc(sg, 24, 10 + sg, cell_sampler)
    Ns = _setting_changes(sg, rng, d.num_rows)
    shifts = np.stack([setting_shift(N, sg) for N in Ns])
    if sg == 15:
        assert (np.abs(shifts).sum(1) > 0).any()  # some C2/c changes need the b/4 origin shift
    e = ops.change_basis(d, torch.as_tensor(Ns))
    X = chiral_molecule['pos']
    A0, A1 = _atoms(d, X, literal_oracle), _atoms(e, X, literal_oracle)
    T0, T1 = d.cell_matrix().numpy(), e.cell_matrix().numpy()
    for i in range(d.num_rows):
        Q = T1[i] @ np.linalg.inv(Ns[i]) @ np.linalg.inv(T0[i])
        assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-12) and np.linalg.det(Q) > 0
        back = A1[i] @ Q + T0[i] @ shifts[i]                      # new frame -> old frame (Q^T x + T s)
        assert literal_oracle.max_atom_deviation(back, A0[i], d.cell[i].numpy()) <= 1e-9
    with pytest.raises(ValueError):
        ops.change_basis(d[:1], torch.tensor([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_triclinic_reduction_is_on_the_live_zero_set():
    from mxtaltools.common.sym_utils import cell_reduction_penalty
    rng = np.random.default_rng(8)
    n = 400
    L = rng.uniform(3, 25, (n, 3))
    A = np.radians(rng.uniform(40, 140, (n, 3)))
    ca, cb, cg = np.cos(A).T
    ok = 1 - ca ** 2 - cb ** 2 - cg ** 2 + 2 * ca * cb * cg > 1e-2
    cells = torch.as_tensor(np.concatenate([L, A], 1)[ok])
    N, red, tied, viol = ops.triclinic_basis(cells)
    assert viol.max() <= 1e-10
    for sg in (1, 2):
        pen = cell_reduction_penalty(red[:, 3:], red[:, :3], torch.full((len(red),), sg), margin=0.0)
        assert pen.max() <= 1e-18
    G0 = ops.metric_tensor(cells)
    G1 = ops.metric_tensor(red)
    Nf = N.to(F64)
    assert ((Nf.transpose(1, 2) @ G0 @ Nf - G1).abs() / G1.abs().amax((1, 2), keepdim=True)).max() <= 1e-12
    assert bool((torch.round(torch.linalg.det(Nf)) == 1).all())


# ---------------------------------------------------------------------------
# images, sheets, gauge
# ---------------------------------------------------------------------------

def test_gauge_is_canonicalize_free_axes():
    """Section.gauge on free axes equals what MolCrystalOps.canonicalize_free_axes writes, in every free-axis group."""
    sgs = [sg for sg in range(1, 231) if bool(group_table(sg).free.any())]
    assert len(sgs) == 68
    ns = types.SimpleNamespace(device=torch.device('cpu'), max_z_prime=1, z_prime=torch.ones(len(sgs), dtype=torch.long),
                               sg_ind=torch.tensor(sgs), aunit_centroid=torch.rand(len(sgs), 3))
    ns.build_free_axis_tensor = types.MethodType(MolCrystalOps.build_free_axis_tensor, ns)
    ns.build_asym_unit_tensor = types.MethodType(MolCrystalOps.build_asym_unit_tensor, ns)
    assert MolCrystalOps.canonicalize_free_axes(ns)
    for i, sg in enumerate(sgs):
        sec = domains.translation_section(sg)
        assert torch.allclose(ns.aunit_centroid[i][sec.free].double(), sec.gauge[sec.free], atol=1e-7)


@pytest.mark.parametrize('sg', [2, 14, 92])
def test_images(sg, cell_sampler):
    tab = group_table(sg)
    d = _rand_desc(sg, 400, 20 + sg, cell_sampler, h=torch.ones(400, dtype=torch.long))
    canon = domains.fold(d).desc
    imgs, src, el = domains.images(d)
    assert torch.equal(torch.bincount(src, minlength=d.num_rows), torch.full((d.num_rows,), len(tab.gamma_plus)))
    assert (domains.fold(imgs).desc.R - canon.R[src]).abs().max() <= 1e-9
    box, src_b, _ = domains.images(d, within='latent_v1_box')
    counts = torch.bincount(src_b, minlength=d.num_rows)
    hist = {int(k): v for k, v in tab.raw['trainer_box']['images_histogram'].items()}
    assert set(counts.tolist()) <= set(hist)
    assert bool((box.t[:, 0] < tab.trainer_box).all()) and bool((box.h == 1).all())
    assert (domains.fold(box).desc.t - canon.t[src_b]).abs().max() <= 1e-9
    _, src_g, _ = domains.images(d, group='gamma', within='g_asu')
    expect = tab.raw['orders']['N_E'] // tab.raw['orders']['G']
    assert torch.equal(torch.bincount(src_g, minlength=d.num_rows), torch.full((d.num_rows,), expect))


@pytest.mark.parametrize('sg, sohncke', [(14, False), (2, False), (19, True), (92, True)])
def test_to_sheet(sg, sohncke, chiral_molecule, cell_sampler, literal_oracle):
    d = _rand_desc(sg, 30, 30 + sg, cell_sampler, h=-torch.ones(30, dtype=torch.long))
    out, ok = domains.to_sheet(d, +1)
    if sohncke:
        assert not bool(ok.any()) and torch.equal(out.h, d.h)
        return
    assert bool(ok.all()) and bool((out.h == 1).all())
    X = chiral_molecule['pos']
    for a, b, cell in zip(_atoms(d, X, literal_oracle), _atoms(out, X, literal_oracle), d.cell):
        assert literal_oracle.max_atom_deviation(a, b, cell.numpy()) <= 1e-9


def test_fold_refuses_z_prime_above_one():
    d = CrystalDescription.from_parts(14, torch.tensor([7.0, 8, 9, 1.57, 1.9, 1.57], dtype=F64)[None],
                                      torch.rand(1, 2, 3, dtype=F64), so3.random_rotations(2).reshape(1, 2, 3, 3))
    with pytest.raises(NotImplementedError):
        domains.fold(d)


def test_fold_float32_input(cell_sampler):
    d = _rand_desc(92, 100, 40, cell_sampler)
    a = domains.fold(d)
    b = domains.fold(d.to(dtype=torch.float32))
    assert torch.equal(a.element, b.element)
    assert (a.desc.t - b.desc.t).abs().max() <= 1e-6


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_description_fields_share_the_device_of_t():
    """from_parts places sg, cell, z_prime, unit_mask and h on t's device (meta stands in for a GPU); validate checks."""
    t = torch.rand(4, 3, dtype=F64, device='meta')
    R = torch.eye(3, dtype=F64).expand(4, 3, 3).to('meta')
    for z_prime in (None, torch.ones(4, dtype=torch.long, device='meta')):
        d = CrystalDescription.from_parts(2, torch.ones(4, 6, dtype=F64), t, R, z_prime=z_prime)
        assert {str(getattr(d, k).device) for k in ('sg', 'cell', 'z_prime', 'unit_mask', 't', 'R', 'h')} == {'meta'}
    cpu = CrystalDescription.from_parts(2, torch.ones(4, 6, dtype=F64), torch.rand(4, 3, dtype=F64),
                                        so3.random_rotations(4))
    cpu.validate()
    with pytest.raises(ValueError, match='is on'):
        cpu.replace(sg=cpu.sg.to('meta')).validate()


def test_reduce_cell_refuses_unreal_cells():
    """A cell whose angle triple has no real metric (Gram det <= 0) is refused by row, not passed to the SVD."""
    cell = torch.tensor([[6.0, 7.0, 8.0, 1.7, 1.8, 1.9]] * 3, dtype=F64)
    cell[1, 3:] = torch.deg2rad(torch.tensor([60.0, 60.0, 170.0], dtype=F64))
    d = CrystalDescription.from_parts(2, cell, torch.rand(3, 3, dtype=F64), so3.random_rotations(3))
    with pytest.raises(ValueError, match=r'rows \[1\]'):
        ops.reduce_cell(d)
    assert ops.reduce_cell(d[[0, 2]]).violation.max() <= 1e-10


@pytest.mark.parametrize('name, cell, expected', [
    ('triclinic', (6.0, 7.5, 8.0, 100, 95, 97), 1), ('monoclinic', (6.0, 7.5, 8.0, 90, 104, 90), 2),
    ('orthorhombic', (6.0, 7.5, 8.0, 90, 90, 90), 4), ('tetragonal', (7.0, 7.0, 9.0, 90, 90, 90), 8),
    ('rhombohedral', (7.0, 7.0, 7.0, 100, 100, 100), 6), ('hexagonal', (7.0, 7.0, 9.0, 90, 90, 120), 12),
    ('cubic P', (7.0, 7.0, 7.0, 90, 90, 90), 24), ('cubic F (primitive)', (5.0, 5.0, 5.0, 60, 60, 60), 24)])
def test_lattice_automorphisms_count_the_holohedry(name, cell, expected):
    """The proper holohedry of each lattice system, counted on the reduced cell of 20 random proper bases."""
    rng = np.random.default_rng(len(name))
    c = torch.tensor([list(cell[:3]) + [math.radians(v) for v in cell[3:]]] * 20, dtype=F64)
    M = rng.integers(-2, 3, (2000, 3, 3))                         # about 1 in 6 has det +1
    M = M[np.round(np.linalg.det(M)) == 1][:20]
    assert len(M) == 20
    Nf = torch.as_tensor(M, dtype=F64)
    scr = ops.cell_from_metric_tensor(Nf.transpose(1, 2) @ ops.metric_tensor(c) @ Nf)
    _, red, _, _ = ops.triclinic_basis(scr)
    assert ops.lattice_automorphisms(red).tolist() == [expected] * 20


def test_round_trip_through_the_trainer_latent_compares_atoms(chiral_molecule):
    """round_trip(desc, LatentV1Chart, mol): the float32 path's atoms are compared with a cell tolerance it meets."""
    from mxtaltools.crystal_fd import charts, checks, measure
    lv = charts.LatentV1Chart(2, 1.7561)
    y = (-0.9 + 1.8 * torch.rand(400, 12, generator=torch.Generator().manual_seed(1))).float()
    desc, _ = lv.describe(y)
    desc = desc[torch.isfinite(measure.cell_volume(desc.cell))]
    mol = MoleculeRef.from_coords(chiral_molecule['pos'], chiral_molecule['z'])
    with pytest.raises(ValueError, match='cells differ'):
        checks.round_trip(desc, lv, mol)                           # the float64 default refuses a float32 cell
    rt = checks.round_trip(desc, lv, mol, cell_rtol=1e-5)
    assert rt.cell <= 1e-5 and rt.atoms <= 1e-4


def test_same_crystal_chunks_within_a_row(chiral_molecule):
    """A high-symmetry row (sg 227, |G| = 192) compared under a small memory bound gives the unbounded result."""
    from mxtaltools.crystal_fd import checks
    mol = MoleculeRef.from_coords(chiral_molecule['pos'], chiral_molecule['z'])
    g = torch.Generator().manual_seed(3)
    cell = torch.tensor([[9.0, 9.0, 9.0] + [math.pi / 2] * 3] * 2, dtype=F64)
    a = CrystalDescription.from_parts(227, cell, torch.rand(2, 3, generator=g, dtype=F64), so3.random_rotations(2, g))
    b = a.replace(R=so3.random_rotations(2, g)[:, None])
    full = checks.same_crystal(a, b, mol, max_elements=2 ** 40)
    small = checks.same_crystal(a, b, mol, max_elements=192 * 5 * 27 * 3 * 16)    # 16 molecules of a at a time
    assert torch.equal(full, small) and float(full.min()) > 0.1
    assert float(checks.same_crystal(a, a, mol, max_elements=192 * 5 * 27 * 3 * 16).max()) <= 1e-12
