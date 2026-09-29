"""standardize.normalizer_images: every description of a crystal under its space group's Euclidean normaliser.

Checked on real states: acridine sg 14 Z'=2 (the acr_campaign_sep28 prior map with its conformer) and, when present,
the MIPCAS sg 2 Z'=1 prior dataset of the GFN's canonical config. Each image must be the same crystal (atomwise RDF
distance ~0 to its source), the images must be distinct descriptions (distinct latents), and the set must be closed:
the images of an image are images of the source. CPU only."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mxtaltools.crystal_search import coordinator as co
from mxtaltools.crystal_search.standardize import normalizer_images
from mxtaltools.dataset_utils.utils import collate_data_list

ROOT = Path(__file__).resolve().parents[1]
ACR_PRIOR = ROOT / 'configs' / 'crystal_searches' / 'acr_campaign_sep28' / 'priors' / 'known_map.pth'
ACR_CONFORMER = Path('D:/crystal_datasets/acridine/opt_acridine_conformer.pt')
MIPCAS_PRIOR = Path('D:/crystal_datasets/conditional/priors/mipcas_sg2_zp1_elj_prior_dataset.pt')
RDF_ATOMWISE = dict(co.RDF_KW, rdf_mode='atomwise')


def _acridine(n=12):
    if not (ACR_PRIOR.exists() and ACR_CONFORMER.exists()):
        pytest.skip('acridine prior map or conformer not present')
    prior = torch.load(ACR_PRIOR, weights_only=False)
    rows = torch.argsort(prior['energy'])[:400:400 // n][:n]
    cfg = SimpleNamespace(mol_path=str(ACR_CONFORMER), sg=14, z_prime=2)
    return collate_data_list(co.rebuild_crystals(cfg, prior['params'][rows], prior['handedness'][rows]))


def _mipcas(n=12):
    if not MIPCAS_PRIOR.exists():
        pytest.skip('MIPCAS prior dataset not present')
    b = torch.load(MIPCAS_PRIOR, weights_only=False)['equalized_prior']
    # rows with a centre ON an asymmetric-unit box face are left out: 28% of this file's rows (clamp-era anchors, not
    # minima along the face normal, 2026-09-28), and on a face the fold into the box has two images, so the latent
    # description is not unique there (closure holds for the crystal, not for its coordinates)
    lat = b.latent_transform(b.full_cell_parameters())
    ok = torch.nonzero((lat[:, 6:].abs() < 1 - 1e-3).all(1)).flatten()
    pick = ok[:: max(1, len(ok) // n)][:n].tolist()
    rows = b.batch_to_list()
    rows = [rows[i] for i in pick]
    return collate_data_list([r.clone() for r in rows], exclude_keys=co.RDF_DROP)


def _rdf(batch):
    out = []
    for lo in range(0, batch.num_graphs, 32):
        b = collate_data_list([c.clone() for c in batch.batch_to_list()[lo:lo + 32]], exclude_keys=co.RDF_DROP)
        with torch.no_grad():
            r = b.analyze(['rdf'], assign_outputs=False, **RDF_ATOMWISE)['rdf']
        out.append((r[0] if isinstance(r, (tuple, list)) else r).float())
    return torch.cat(out)


def _latents(batch):
    b = batch.clone()
    b.canonicalize_zp_aunits()
    return b.latent_transform(b.full_cell_parameters()).double()


@pytest.fixture(scope='module', params=['acridine_sg14_zp2', 'mipcas_sg2_zp1'])
def case(request):
    batch = _acridine() if request.param.startswith('acridine') else _mipcas()
    images, source, coset = normalizer_images(batch)
    return batch, images, source, coset


def test_eight_images_per_crystal_and_the_identity_is_the_input(case):
    batch, images, source, coset = case
    assert images.num_graphs == 8 * batch.num_graphs
    assert torch.equal(torch.bincount(source), torch.full((batch.num_graphs,), 8))
    ident = coset == 0
    assert torch.equal(images.full_cell_parameters()[ident], batch.full_cell_parameters())


def test_every_image_is_the_same_crystal(case):
    batch, images, source, _ = case
    d = torch.diagonal(co.rdf_distance_matrix(_rdf(images), _rdf(batch)[source]))
    assert float(d.max()) < 1e-3, d.max()


def test_images_are_distinct_descriptions(case):
    _, images, source, _ = case
    lat = _latents(images)
    distinct = []
    for s in torch.unique(source):
        L = lat[source == s]
        D = torch.cdist(L, L)
        close = D < 1e-4
        distinct.append(int(len(L) // close.sum(1).max()))  # orbit size: 8 / the stabiliser's order
        assert bool((close.sum(1) == close.sum(1)[0]).all()), 'coincident images must come in equal groups'
    # a crystal a coset maps to itself has fewer distinct descriptions (acridine: exact Z'=1 crystals written as Z'=2,
    # where the a/2 origin shift swaps the two molecules), always a divisor of 8; generic crystals have all 8
    assert all(8 % k == 0 for k in distinct) and max(distinct) == 8, distinct


def test_the_image_set_is_closed(case):
    _, images, source, _ = case
    lat = _latents(images)
    again, src2, _ = normalizer_images(images)
    lat2 = _latents(again)
    for s in torch.unique(source):
        mine = lat[source == s]
        theirs = lat2[torch.isin(src2, torch.nonzero(source == s).flatten())]
        assert float(torch.cdist(theirs, mine).min(1).values.max()) < 1e-4


def test_cosets_with_a_point_part_are_refused():
    batch = _acridine(n=2)
    with pytest.raises(NotImplementedError):
        normalizer_images(batch, table={'14': [(np.diag([-1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])]})
