"""
Unit tests for mxtaltools/common/geometry_utils.py

Covers pure-math functions that can be verified analytically or via roundtrips.
Skips functions that require a full molecule graph (batch_compute_molecule_volume,
probe/grid volume estimators) — those belong with the data-class integration tests.
"""
import math

import numpy as np
import pytest
import torch

from mxtaltools.common.geometry_utils import (
    angle2components,
    batch_cell_vol_torch,
    batch_compute_fractional_transform,
    cell_parameters_to_box_vectors,
    cell_vol_angle_factor,
    cell_vol_np,
    cell_vol_torch,
    center_batch,
    components2angle,
    compute_Ip_handedness,
    compute_mol_radius,
    coor_trans_matrix_np,
    enforce_crystal_system2,
    extract_batching_info,
    fractional_transform,
    get_batch_centroids,
    lat2sph_rotvec,
    norm_circular_components,
    rotmat2rotvec,
    rotvec2rotmat,
    sample_random_valid_rotvecs,
    sph2cart_rotvec,
    cart2sph_rotvec,
    sph_rotvec2lat,
)

PI = math.pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cubic(a=1.0):
    """Return lengths and angles for a cubic cell with side length a."""
    lengths = np.array([a, a, a], dtype=np.float64)
    angles = np.array([PI / 2, PI / 2, PI / 2], dtype=np.float64)
    return lengths, angles


def _cubic_torch(a=1.0, n=1):
    lengths = torch.full((n, 3), a, dtype=torch.float32)
    angles = torch.full((n, 3), PI / 2, dtype=torch.float32)
    return lengths, angles


# ===========================================================================
# 1. Cell volume
# ===========================================================================

class TestCellVolume:

    def test_cubic_unit_np(self):
        v, a = _cubic(1.0)
        assert math.isclose(cell_vol_np(v, a), 1.0, rel_tol=1e-6)

    def test_cubic_scaled_np(self):
        v, a = _cubic(3.0)
        assert math.isclose(cell_vol_np(v, a), 27.0, rel_tol=1e-6)

    def test_orthorhombic_np(self):
        v = np.array([2.0, 3.0, 4.0])
        a = np.array([PI / 2, PI / 2, PI / 2])
        assert math.isclose(cell_vol_np(v, a), 24.0, rel_tol=1e-5)

    def test_cubic_torch(self):
        v = torch.tensor([2.0, 2.0, 2.0])
        a = torch.tensor([PI / 2, PI / 2, PI / 2])
        assert torch.isclose(cell_vol_torch(v, a), torch.tensor(8.0), atol=1e-4)

    def test_batch_cell_vol_agrees_with_single(self):
        lengths, angles = _cubic_torch(a=2.0, n=4)
        batch_vols = batch_cell_vol_torch(lengths, angles)
        single_vol = cell_vol_torch(lengths[0], angles[0])
        assert torch.allclose(batch_vols, single_vol.expand(4), atol=1e-4)

    def test_cell_vol_angle_factor_cubic(self):
        angles = torch.tensor([PI / 2, PI / 2, PI / 2])
        factor = cell_vol_angle_factor(angles)
        assert torch.isclose(factor, torch.tensor(1.0), atol=1e-5)

    def test_numpy_torch_consistency(self):
        v_np = np.array([3.0, 4.0, 5.0])
        a_np = np.array([PI / 2, PI * 0.6, PI * 0.55])
        vol_np = cell_vol_np(v_np, a_np)

        v_t = torch.tensor(v_np, dtype=torch.float32)
        a_t = torch.tensor(a_np, dtype=torch.float32)
        vol_t = cell_vol_torch(v_t, a_t).item()

        assert math.isclose(vol_np, vol_t, rel_tol=1e-4)


# ===========================================================================
# 2. Fractional / Cartesian transforms
# ===========================================================================

class TestFractionalTransform:

    def test_cubic_fc_is_diagonal(self):
        v, a = _cubic(2.0)
        m = coor_trans_matrix_np('f_to_c', v, a)
        expected = np.diag([2.0, 2.0, 2.0])
        np.testing.assert_allclose(m, expected, atol=1e-10)

    def test_cubic_cf_is_diagonal(self):
        v, a = _cubic(2.0)
        m = coor_trans_matrix_np('c_to_f', v, a)
        expected = np.diag([0.5, 0.5, 0.5])
        np.testing.assert_allclose(m, expected, atol=1e-10)

    def test_np_roundtrip(self):
        v = np.array([3.0, 4.0, 5.0])
        a = np.array([PI / 2, PI * 0.6, PI * 0.55])
        m_fc = coor_trans_matrix_np('f_to_c', v, a)
        m_cf = coor_trans_matrix_np('c_to_f', v, a)
        product = m_fc @ m_cf
        np.testing.assert_allclose(product, np.eye(3), atol=1e-8)

    def test_batch_fc_cf_inverses(self):
        n = 8
        lengths = torch.rand(n, 3) * 5 + 2
        angles = torch.rand(n, 3) * 0.4 + PI / 2 - 0.2
        T_fc, T_cf, _ = batch_compute_fractional_transform(lengths, angles)
        products = torch.bmm(T_fc, T_cf)
        identity = torch.eye(3).unsqueeze(0).expand(n, -1, -1)
        assert torch.allclose(products, identity, atol=1e-4)

    def test_batch_volume_positive(self):
        lengths, angles = _cubic_torch(a=3.0, n=5)
        _, _, vols = batch_compute_fractional_transform(lengths, angles)
        assert torch.all(vols > 0)

    def test_batch_cubic_volume(self):
        a = 3.0
        lengths, angles = _cubic_torch(a=a, n=3)
        _, _, vols = batch_compute_fractional_transform(lengths, angles)
        assert torch.allclose(vols, torch.full((3,), a ** 3), atol=1e-3)

    def test_single_torch_matches_numpy(self):
        v = np.array([3.0, 4.0, 5.0])
        a_angles = np.array([PI / 2, PI * 0.6, PI * 0.55])
        m_np = coor_trans_matrix_np('f_to_c', v, a_angles)

        vt = torch.tensor(v, dtype=torch.float32)
        at = torch.tensor(a_angles, dtype=torch.float32)
        m_t = cell_parameters_to_box_vectors('f_to_c', vt, at).numpy()
        np.testing.assert_allclose(m_np, m_t, atol=1e-5)

    def test_fractional_transform_roundtrip_torch(self):
        n = 10
        lengths = torch.rand(n, 3) * 4 + 2
        angles = torch.rand(n, 3) * 0.3 + PI / 2 - 0.15
        T_fc, T_cf, _ = batch_compute_fractional_transform(lengths, angles)
        coords = torch.rand(n, 3)
        cart = fractional_transform(coords, T_fc)
        back = fractional_transform(cart, T_cf)
        assert torch.allclose(back, coords, atol=1e-4)

    def test_fractional_transform_roundtrip_numpy(self):
        v = np.array([3.0, 4.0, 5.0])
        a = np.array([PI / 2, PI * 0.6, PI * 0.55])
        m_fc = coor_trans_matrix_np('f_to_c', v, a)
        m_cf = coor_trans_matrix_np('c_to_f', v, a)
        coords = np.random.rand(20, 3)
        cart = fractional_transform(coords, m_fc)
        back = fractional_transform(cart, m_cf)
        np.testing.assert_allclose(back, coords, atol=1e-8)


# ===========================================================================
# 3. Spherical / Cartesian rotation-vector conversions
# ===========================================================================

class TestSphCartRotvec:

    # --- numpy branch ---

    def test_sph2cart_x_axis_np(self):
        angles = np.array([[PI / 2, 0.0, 1.0]])
        rotvec = sph2cart_rotvec(angles)
        np.testing.assert_allclose(rotvec, [[1.0, 0.0, 0.0]], atol=1e-6)

    def test_sph2cart_y_axis_np(self):
        angles = np.array([[PI / 2, PI / 2, 1.0]])
        rotvec = sph2cart_rotvec(angles)
        np.testing.assert_allclose(rotvec, [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_sph2cart_z_axis_np(self):
        angles = np.array([[0.0, 0.0, 1.0]])
        rotvec = sph2cart_rotvec(angles)
        np.testing.assert_allclose(rotvec, [[0.0, 0.0, 1.0]], atol=1e-6)

    def test_sph2cart_scales_with_r_np(self):
        angles2 = np.array([[PI / 2, 0.0, 2.0]])
        angles1 = np.array([[PI / 2, 0.0, 1.0]])
        np.testing.assert_allclose(sph2cart_rotvec(angles2), 2 * sph2cart_rotvec(angles1), atol=1e-6)

    def test_roundtrip_np(self):
        np.random.seed(0)
        rotvec = np.random.randn(20, 3)
        rotvec[:, 2] = np.abs(rotvec[:, 2])  # upper half-sphere
        rotvec /= np.linalg.norm(rotvec, axis=1, keepdims=True)
        rotvec *= np.random.uniform(0.1, 2 * PI, size=(20, 1))
        back = sph2cart_rotvec(cart2sph_rotvec(rotvec))
        np.testing.assert_allclose(back, rotvec, atol=1e-6)

    def test_cart2sph_norm_is_r_np(self):
        rotvec = np.array([[3.0, 4.0, 0.0]])
        sph = cart2sph_rotvec(rotvec)
        assert math.isclose(sph[0, 2], 5.0, rel_tol=1e-6)

    # --- torch branch (batched) ---

    def test_sph2cart_x_axis_torch(self):
        angles = torch.tensor([[PI / 2, 0.0, 1.0]])
        rotvec = sph2cart_rotvec(angles)
        assert torch.allclose(rotvec, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)

    def test_roundtrip_torch(self):
        torch.manual_seed(42)
        rotvec = torch.randn(30, 3)
        rotvec[:, 2] = rotvec[:, 2].abs()
        rotvec = rotvec / rotvec.norm(dim=1, keepdim=True)
        rotvec = rotvec * (torch.rand(30, 1) * 2 * PI + 0.05)
        back = sph2cart_rotvec(cart2sph_rotvec(rotvec))
        assert torch.allclose(back, rotvec, atol=1e-4)


# ===========================================================================
# 4. Rotation matrices
# ===========================================================================

class TestRotationMatrices:

    def test_rotvec2rotmat_is_SO3(self):
        torch.manual_seed(0)
        rotvecs = sample_random_valid_rotvecs(50)
        rmats = rotvec2rotmat(rotvecs)
        # orthogonality: R^T R = I
        eye = torch.eye(3).unsqueeze(0).expand(50, -1, -1)
        assert torch.allclose(torch.bmm(rmats.transpose(1, 2), rmats), eye, atol=1e-4)
        # determinant = +1
        dets = torch.linalg.det(rmats)
        assert torch.allclose(dets, torch.ones(50), atol=1e-4)

    def test_rotvec2rotmat_z_pi(self):
        """Rotation by pi around z → flip x and y."""
        rotvec = torch.tensor([[0.0, 0.0, PI]])
        R = rotvec2rotmat(rotvec)[0]
        expected = torch.tensor([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
        assert torch.allclose(R, expected, atol=1e-5)

    def test_rotvec2rotmat_x_pi(self):
        """Rotation by pi around x → flip y and z."""
        rotvec = torch.tensor([[PI, 0.0, 0.0]])
        R = rotvec2rotmat(rotvec)[0]
        expected = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        assert torch.allclose(R, expected, atol=1e-5)

    def test_roundtrip_rotvec_rotmat(self):
        torch.manual_seed(1)
        rotvecs = sample_random_valid_rotvecs(40)
        rmats = rotvec2rotmat(rotvecs)
        recovered = rotmat2rotvec(rmats, warn_on_bad_determinant=False)
        # rotvecs and recovered should produce the same rotation matrix
        rmats2 = rotvec2rotmat(recovered)
        assert torch.allclose(rmats, rmats2, atol=1e-3)

    def test_sample_random_valid_rotvecs_shape(self):
        vecs = sample_random_valid_rotvecs(100)
        assert vecs.shape == (100, 3)

    def test_sample_random_valid_rotvecs_z_nonneg(self):
        vecs = sample_random_valid_rotvecs(200)
        assert torch.all(vecs[:, 2] >= 0)

    def test_sample_random_valid_rotvecs_norms_in_range(self):
        vecs = sample_random_valid_rotvecs(200)
        norms = vecs.norm(dim=1)
        assert torch.all(norms > 0)
        assert torch.all(norms <= 2 * PI + 1e-5)


# ===========================================================================
# 5. Angle / component encoding
# ===========================================================================

class TestAngleComponents:

    def test_angle2components_shape(self):
        angles = torch.linspace(-PI, PI, 20)
        components = angle2components(angles)
        assert components.shape == (20, 2)

    def test_angle2components_values(self):
        angles = torch.tensor([0.0, PI / 2, PI, -PI / 2])
        c = angle2components(angles)
        # column 0 = sin, column 1 = cos
        assert torch.allclose(c[:, 0], torch.sin(angles), atol=1e-6)
        assert torch.allclose(c[:, 1], torch.cos(angles), atol=1e-6)

    def test_roundtrip(self):
        torch.manual_seed(3)
        angles = (torch.rand(50) * 2 - 1) * PI
        recovered = components2angle(angle2components(angles))
        assert torch.allclose(recovered, angles, atol=1e-5)

    def test_norm_circular_components_unit_norm(self):
        torch.manual_seed(4)
        components = torch.randn(30, 2) * 5
        normed = norm_circular_components(components)
        norms = torch.sqrt((normed ** 2).sum(dim=1))
        assert torch.allclose(norms, torch.ones(30), atol=1e-5)


# ===========================================================================
# 6. Inertial-axes handedness
# ===========================================================================

class TestHandedness:

    def test_right_handed_numpy(self):
        Ip = np.eye(3)
        assert compute_Ip_handedness(Ip) == 1.0

    def test_left_handed_numpy(self):
        # swap two axes to get left-handed
        Ip = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        assert compute_Ip_handedness(Ip) == -1.0

    def test_right_handed_torch_2d(self):
        Ip = torch.eye(3)
        assert compute_Ip_handedness(Ip).item() == 1.0

    def test_left_handed_torch_2d(self):
        Ip = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        assert compute_Ip_handedness(Ip).item() == -1.0

    def test_batched_torch(self):
        right = torch.eye(3).unsqueeze(0)
        left = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]).unsqueeze(0)
        batch = torch.cat([right, left, right], dim=0)
        h = compute_Ip_handedness(batch)
        assert h[0].item() == 1.0
        assert h[1].item() == -1.0
        assert h[2].item() == 1.0


# ===========================================================================
# 7. Crystal system enforcement
# ===========================================================================

class TestEnforceCrystalSystem:

    def _make_inputs(self, n=2):
        lengths = torch.rand(n, 3) * 3 + 2  # 2-5 Å
        angles = torch.rand(n, 3) * 0.5 + PI / 3
        return lengths, angles

    def test_triclinic_passthrough(self):
        lengths, angles = self._make_inputs()
        fl, fa = enforce_crystal_system2(lengths, angles, ['triclinic'] * 2)
        assert torch.allclose(fl, lengths)
        assert torch.allclose(fa, angles)

    def test_monoclinic_fixes_alpha_gamma(self):
        lengths, angles = self._make_inputs()
        fl, fa = enforce_crystal_system2(lengths, angles, ['monoclinic'] * 2)
        assert torch.allclose(fl, lengths)
        assert torch.allclose(fa[:, 0], torch.full((2,), PI / 2), atol=1e-5)
        assert torch.allclose(fa[:, 2], torch.full((2,), PI / 2), atol=1e-5)
        assert torch.allclose(fa[:, 1], angles[:, 1])  # beta free

    def test_orthorhombic_all_angles_90(self):
        lengths, angles = self._make_inputs()
        fl, fa = enforce_crystal_system2(lengths, angles, ['orthorhombic'] * 2)
        assert torch.allclose(fl, lengths)
        assert torch.allclose(fa, torch.full((2, 3), PI / 2), atol=1e-5)

    def test_tetragonal_a_eq_b_all_angles_90(self):
        lengths, angles = self._make_inputs()
        fl, fa = enforce_crystal_system2(lengths, angles, ['tetragonal'] * 2)
        assert torch.allclose(fl[:, 0], fl[:, 1], atol=1e-5)
        assert torch.allclose(fa, torch.full((2, 3), PI / 2), atol=1e-5)

    def test_hexagonal_a_eq_b_gamma_120(self):
        lengths, angles = self._make_inputs()
        fl, fa = enforce_crystal_system2(lengths, angles, ['hexagonal'] * 2)
        assert torch.allclose(fl[:, 0], fl[:, 1], atol=1e-5)
        assert torch.allclose(fa[:, 0], torch.full((2,), PI / 2), atol=1e-5)
        assert torch.allclose(fa[:, 1], torch.full((2,), PI / 2), atol=1e-5)
        assert torch.allclose(fa[:, 2], torch.full((2,), 2 * PI / 3), atol=1e-5)

    def test_rhombohedral_all_equal(self):
        lengths, angles = self._make_inputs()
        fl, fa = enforce_crystal_system2(lengths, angles, ['rhombohedral'] * 2)
        assert torch.allclose(fl[:, 0], fl[:, 1], atol=1e-5)
        assert torch.allclose(fl[:, 0], fl[:, 2], atol=1e-5)
        assert torch.allclose(fa[:, 0], fa[:, 1], atol=1e-5)
        assert torch.allclose(fa[:, 0], fa[:, 2], atol=1e-5)

    def test_cubic_all_equal_90(self):
        lengths = torch.tensor([[3.0, 4.0, 5.0]])
        angles = torch.tensor([[PI / 3, PI / 4, PI / 5]])
        fl, fa = enforce_crystal_system2(lengths, angles, ['cubic'])
        mean_len = (3.0 + 4.0 + 5.0) / 3
        assert torch.allclose(fl[0], torch.full((3,), mean_len), atol=1e-5)
        assert torch.allclose(fa[0], torch.full((3,), PI / 2), atol=1e-5)


# ===========================================================================
# 8. Latent orientation transforms
# ===========================================================================

class TestLatentOrientationTransforms:

    def test_roundtrip_lat_sph_lat(self):
        torch.manual_seed(7)
        z_prime = 2
        lat = torch.rand(16, z_prime * 3) * 2 - 1  # [-1, 1]
        sph = lat2sph_rotvec(lat, z_prime)
        recovered = sph_rotvec2lat(sph, z_prime)
        assert torch.allclose(recovered, lat, atol=1e-5)

    def test_output_shape(self):
        z_prime = 3
        lat = torch.rand(8, z_prime * 3)
        sph = lat2sph_rotvec(lat, z_prime)
        assert sph.shape == lat.shape


# ===========================================================================
# 9. Batch / scatter utilities
# ===========================================================================

class TestBatchUtilities:

    def test_extract_batching_info_ptrs(self):
        nodes_list = [torch.zeros(3, 3), torch.zeros(5, 3), torch.zeros(2, 3)]
        batch, ptrs = extract_batching_info(nodes_list)
        expected_ptrs = torch.tensor([0, 3, 8, 10])
        assert torch.all(ptrs == expected_ptrs)

    def test_extract_batching_info_batch(self):
        nodes_list = [torch.zeros(3, 3), torch.zeros(5, 3), torch.zeros(2, 3)]
        batch, _ = extract_batching_info(nodes_list)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
        assert torch.all(batch == expected)

    def test_compute_mol_radius_unit_sphere(self):
        coords = torch.tensor([
            [1., 0., 0.], [-1., 0., 0.],
            [0., 1., 0.], [0., -1., 0.],
            [0., 0., 1.], [0., 0., -1.],
        ])
        r = compute_mol_radius(coords)
        assert torch.isclose(r, torch.tensor(1.0), atol=1e-6)

    def test_compute_mol_radius_offset(self):
        """Radius should be measured from centroid, not origin."""
        base = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        shifted = base + 10.0  # uniform shift doesn't change radius
        assert torch.isclose(compute_mol_radius(base), compute_mol_radius(shifted), atol=1e-5)

    def test_get_batch_centroids(self):
        coords = torch.tensor([[1., 0., 0.], [3., 0., 0.], [0., 4., 0.], [0., 2., 0.]])
        batch = torch.tensor([0, 0, 1, 1])
        centroids = get_batch_centroids(coords, batch, num_graphs=2)
        expected = torch.tensor([[2., 0., 0.], [0., 3., 0.]])
        assert torch.allclose(centroids, expected, atol=1e-5)

    def test_center_batch_zero_mean(self):
        torch.manual_seed(9)
        nodes_per_graph = torch.tensor([4, 6, 3])
        coords = torch.randn(13, 3)
        batch = torch.repeat_interleave(torch.arange(3), nodes_per_graph)
        centered = center_batch(coords, batch, num_graphs=3, nodes_per_graph=nodes_per_graph)
        means = get_batch_centroids(centered, batch, num_graphs=3)
        assert torch.allclose(means, torch.zeros(3, 3), atol=1e-5)
