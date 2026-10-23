"""
MolCrystalData.analyze on a batch holding one crystal with no defined cluster: a non-finite or singular cell, or a
non-finite pose (centroid or orientation), as one diverged sampler draw gives. That crystal's per-graph outputs are
NaN, the batch is not stopped, and every other crystal's outputs are exactly those of the batch without the bad row.
Trainers mask non-finite rewards (fwd/nonfinite_frac); before 2026-10-19 a non-finite cell raised inside
get_cart_translations and a non-finite pose scored a finite eLJ of 0 (NaN distances make no edges).

CPU only, four small crystals from the mini dataset; a few seconds.
"""
import math
import os

import pytest
import torch

from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
DROP = ['mace', 'mace_pot', 'mace_gas_pot', 'elj', 'lj', 'reduction_en', 'rdf', 'fingerprint', 'rdf_bins']
BAD = 1


@pytest.fixture(scope='module')
def clean():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    mols = [c for c in torch.load(DATASET, weights_only=False, map_location='cpu')
            if int(c.z_prime) == 1 and int(c.num_atoms) <= 30][:4]
    if len(mols) < 4:
        pytest.skip('fewer than four small Z\'=1 crystals in the fixture')
    rows = []
    for m in mols:
        m = m.clone()
        for k in DROP:
            if k in m.keys():
                del m[k]
        rows.append(m)
    b = collate_data_list(rows)
    b.box_analysis()
    return b


def _spoil(b, what):
    b = b.clone()
    if what == 'nan length':
        b.cell_lengths[BAD, 0] = float('nan')
    elif what == 'inf length':
        b.cell_lengths[BAD, 1] = float('inf')
    elif what == 'zero length (singular cell)':
        b.cell_lengths[BAD, 0] = 0.0
    elif what == 'nan orientation':
        b.aunit_orientation[BAD, 0] = float('nan')
    elif what == 'nan centroid':
        b.aunit_centroid[BAD, 0] = float('nan')
    b.box_analysis()
    return b


def _analyze(b):
    with torch.no_grad():
        return b.analyze(['elj', 'reduction_en', 'rdf'], cutoff=10, supercell_size=10, std_orientation=False)


@pytest.mark.parametrize('what', ['nan length', 'inf length', 'zero length (singular cell)', 'nan orientation',
                                  'nan centroid'])
def test_undefined_crystal_is_nan_and_the_rest_unchanged(clean, what):
    ref = _analyze(clean)
    with pytest.warns(UserWarning, match='zero translation'):
        out = _analyze(_spoil(clean, what))
    good = torch.arange(clean.num_graphs) != BAD
    for key in ('elj', 'reduction_en'):
        assert bool(torch.isnan(out[key][BAD])), (key, out[key])
        assert torch.equal(out[key][good], ref[key][good]), key
    assert bool(torch.isnan(out['rdf'][0][BAD]).all())
    assert torch.equal(out['rdf'][0][good], ref['rdf'][0][good])


def test_clean_batch_has_no_nan(clean):
    out = _analyze(clean)
    assert bool(torch.isfinite(out['elj']).all()) and bool(torch.isfinite(out['rdf'][0]).all())
