"""Hop kicks from parents outside the latent box (log_noise_latent_parameters(keep_start_representable=True)).

The acridine campaign's lowest known state has a = 35.1 A, beyond the latent box; the default kick clipped it to a
different crystal (a = 27.4 A, RDF distance 0.42) before kicking. These tests use the campaign's prior map and conformer
exactly as the hop stream does (coordinator.rebuild_crystals, canonicalize_orientation, then the kick). CPU only."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mxtaltools.crystal_search import coordinator as co
from mxtaltools.dataset_utils.utils import collate_data_list

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'configs' / 'crystal_searches' / 'acr_campaign_sep28' / 'priors' / 'known_map.pth'
CONFORMER = Path('D:/crystal_datasets/acridine/opt_acridine_conformer.pt')


@pytest.fixture(scope='module')
def states():
    if not (PRIOR.exists() and CONFORMER.exists()):
        pytest.skip('acridine prior map or conformer not present')
    cfg = SimpleNamespace(mol_path=str(CONFORMER), sg=14, z_prime=2)
    prior = torch.load(PRIOR, weights_only=False)
    order = torch.argsort(prior['energy'])[:400]
    b = collate_data_list(co.rebuild_crystals(cfg, prior['params'][order], prior['handedness'][order]))
    b.canonicalize_orientation()
    b.canonicalize_zp_aunits()
    over = b.latent_transform(b.full_cell_parameters()).abs().max(1).values
    out_rows = order[over > 1.02][:3]
    in_rows = order[over < 0.9][:4]
    assert len(out_rows) >= 1 and len(in_rows) >= 2
    return cfg, prior, out_rows, in_rows


def _kicked(cfg, prior, rows, size, keep, seed=3):
    b = collate_data_list(co.rebuild_crystals(cfg, prior['params'][rows], prior['handedness'][rows]))
    b.canonicalize_orientation()
    torch.manual_seed(seed)
    lg = float(torch.log10(torch.tensor(size)))
    b.log_noise_latent_parameters(lg, lg, keep_start_representable=keep)
    return b


def _rdf_d(cfg, prior, rows, b):
    parents = co.compute_rdfs(co.rebuild_crystals(cfg, prior['params'][rows], prior['handedness'][rows]))
    kicked = co.compute_rdfs(b.batch_to_list())
    return torch.diagonal(co.rdf_distance_matrix(kicked, parents))


def test_a_vanishing_kick_returns_the_out_of_box_parent_only_when_kept_representable(states):
    cfg, prior, out_rows, _ = states
    kept = _rdf_d(cfg, prior, out_rows, _kicked(cfg, prior, out_rows, 1e-9, keep=True))
    clipped = _rdf_d(cfg, prior, out_rows, _kicked(cfg, prior, out_rows, 1e-9, keep=False))
    assert float(kept.max()) < 1e-3, kept
    assert float(clipped.min()) > 0.02, clipped  # the defect this flag exists for


def test_a_small_kick_from_an_out_of_box_parent_stays_local(states):
    cfg, prior, out_rows, _ = states
    d = _rdf_d(cfg, prior, out_rows, _kicked(cfg, prior, out_rows, 1e-2, keep=True))
    assert float(d.max()) < 0.06, d  # a 1e-2 kick moves in-box acridine states ~0.02 (thermal scan, 2026-09-28)


def test_in_box_parents_are_kicked_exactly_as_before(states):
    cfg, prior, _, in_rows = states
    for size in (1e-2, 10 ** -0.5):
        a = _kicked(cfg, prior, in_rows, size, keep=True)
        b = _kicked(cfg, prior, in_rows, size, keep=False)
        assert torch.equal(a.full_cell_parameters(), b.full_cell_parameters())
        assert torch.equal(a.aunit_handedness, b.aunit_handedness)
