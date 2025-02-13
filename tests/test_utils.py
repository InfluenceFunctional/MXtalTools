import numpy as np
import torch
import torch.nn.functional as F

from mxtaltools.common.utils import torch_ptp, softmax_np
from mxtaltools.common.geometry_utils import norm_circular_components, components2angle, angle2components
from mxtaltools.analysis.crystal_rdf import earth_movers_distance_torch, earth_movers_distance_np


def test_torch_ptp():
    numbers = np.random.randn(10)
    ptp = np.ptp(numbers)
    tptp = torch_ptp(torch.Tensor(numbers))
    assert np.abs(ptp - tptp.detach().numpy()) < 1e-5


def test_softmax_np():
    numbers = np.random.randn(100).reshape(10, 10)
    torch_output = F.softmax(torch.Tensor(numbers), dim=-1).detach().numpy()
    np_output = softmax_np(numbers)

    assert np.mean(np.abs(torch_output - np_output)) < 1e-5


def test_earth_movers_distance_torch():
    assert earth_movers_distance_torch(torch.zeros(100), torch.ones(100)) == 5050


def test_earth_movers_distance_np_vs_torch():
    pdf1 = np.random.randn(100)
    pdf2 = np.random.randn(100)
    pdf1 /= np.sum(pdf1)
    pdf2 /= np.sum(pdf2)

    np_emd = earth_movers_distance_np(pdf1, pdf2)
    torch_emd = earth_movers_distance_torch(torch.Tensor(pdf1), torch.Tensor(pdf2)).detach().numpy()

    assert np.abs(np_emd - torch_emd) < 5e-5


def test_components2angle():
    components = torch.randn((100, 2))
    normed_components = norm_circular_components(components)

    angles = components2angle(components)
    angles_from_norm = components2angle(normed_components)
    angles_from_norm2 = components2angle(components, norm_components=True)

    assert torch.mean(torch.abs(angles - angles_from_norm)) < 1e-5
    assert torch.mean(torch.abs(angles - angles_from_norm2)) < 1e-5

    rebuilt_components = angle2components(angles)

    assert torch.mean(torch.abs(rebuilt_components - normed_components)) < 1e-5  # rebuilt components are automatically normed

    rebuilt_angles = components2angle(rebuilt_components)

    assert torch.mean(torch.abs(rebuilt_angles - angles)) < 1e-5


def test_norm_circular_components():
    components = torch.randn((100, 2))
    normed_components = norm_circular_components(components)
    assert torch.mean(torch.abs(torch.sum(normed_components ** 2, dim=1) - torch.ones(100))) < 1e-5
