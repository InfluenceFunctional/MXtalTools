"""Contract tests for the package's COMPACK producer (crystal_analysis.compack_compare on the engine compack_engine
builds, and MolCrystalAnalysis.batch_compack on top of it) and its opt-in CIF repair (repair_compack_crystal).

The CSD reader bonds NEHZOR's close Br...Br contacts between molecules (eLJ search minima), fusing a cell of 4
molecules into 2 components, and PackingSimilarity then under-matches or fails on crystals that are the same. With
repair off the producer must be exactly the engine on the crystals as read; with repair on it must recover those
matches, leave a CIF the reader does not fuse (MIPCAS) as read, use a CSD file the reader returns whole as read, and
refuse a CIF of another molecule. Fixtures and their provenance: compack_fixtures/MANIFEST.json.
"""

import contextlib
import io
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / 'compack_fixtures'
EXPERIMENTAL = str(FIXTURES / 'nehzor_experimental_NEHZOR.cif')


def _manifest():
    return json.loads((FIXTURES / 'MANIFEST.json').read_text())


def _template(molecule):
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_bond_template
    m = _manifest()['molecules'][molecule]
    return compack_bond_template(m['atomic_numbers'], m['positions'])


def _pairs(molecule=None):
    return [pytest.param(p, id=f"{p['molecule']}_{p['name']}") for p in _manifest()['pairs']
            if molecule in (None, p['molecule'])]


def _paths(pair):
    return str(FIXTURES / pair['ref']), str(FIXTURES / pair['test'])


def _engine_as_read(ref, test):
    """The independent oracle: PackingSimilarity (shell 20, CCDC defaults) on the two crystals as the reader returns
    them; (status, n_matched, rmsd)."""
    from ccdc.crystal import PackingSimilarity
    from ccdc.io import CrystalReader
    engine = PackingSimilarity()
    engine.settings.packing_shell_size = 20
    try:
        r = engine.compare(CrystalReader(ref)[0], CrystalReader(test)[0])
        return 'ok', int(r.nmatched_molecules), float(r.rmsd)
    except AttributeError:
        return 'failed', 0, 0.0


@pytest.mark.ccdc
@pytest.mark.parametrize('pair', _pairs())
def test_repair_off_is_the_engine_on_the_crystals_as_read(pair):
    """The default route (no repair) and single_compack_run give what the engine gives on the CIFs as read."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_compare, single_compack_run
    ref, test = _paths(pair)
    status, n_matched, rmsd = _engine_as_read(ref, test)

    r = compack_compare(ref, test)
    with contextlib.redirect_stdout(io.StringIO()):
        old = single_compack_run(0, test, ref)

    assert (r['status'], r['n_matched'], r['rmsd']) == (status, n_matched, rmsd)
    assert r['repair_ref'] is None and r['repair_test'] is None
    assert (float(old[0]), int(old[1])) == (rmsd, n_matched)


@pytest.mark.ccdc
@pytest.mark.parametrize('pair', _pairs('nehzor'))
def test_fused_nehzor_pair_matches_after_repair(pair):
    """A NEHZOR pair the reader fuses matches 20/20 once repaired."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_compare
    tmpl = _template('nehzor')
    assert tmpl['n_atoms'] == 24 and len(tmpl['bonds']) == 25 and tmpl['formulas'] == ['C12 H9 Br1 O2']
    ref, test = _paths(pair)

    r = compack_compare(ref, test, tmpl, repair=True)

    # the fixture exercises the fault: the reader returned fewer components than molecules
    assert r['repair_ref']['mode'] == 'repaired' and r['repair_ref']['Z'] == 4
    assert r['repair_ref']['components_as_read'] < 4 and r['repair_ref']['inter_bonds_removed'] > 0
    assert r['status'] == 'ok'
    assert r['n_matched'] == 20
    if pair['name'] == 'canonical_control':  # one crystal written two ways
        assert r['rmsd'] < 1e-4


@pytest.mark.ccdc
@pytest.mark.parametrize('pair', _pairs('mipcas'))
def test_unfused_mipcas_pair_is_unchanged_by_repair(pair):
    """On MIPCAS CIFs the repair changes nothing and COMPACK returns what it returns without it."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_compare, repair_compack_crystal
    tmpl = _template('mipcas')
    ref, test = _paths(pair)
    for path in (ref, test):
        _, info = repair_compack_crystal(path, tmpl)
        assert info['mode'] == 'repaired' and not info['changed']
        assert info['components_as_read'] == info['Z']

    r = compack_compare(ref, test, tmpl, repair=True)
    plain = compack_compare(ref, test)
    assert r['status'] == 'ok'
    assert (r['rmsd'], r['n_matched']) == (plain['rmsd'], plain['n_matched'])


@pytest.mark.ccdc
def test_csd_reference_is_used_as_read():
    """A CSD file (asymmetric unit, its own atom order) that the reader returns whole is compared as read."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_compare
    tmpl = _template('nehzor')
    fused = next(p for p in _manifest()['pairs'] if p['name'] == 'canonical_control')

    self_match = compack_compare(EXPERIMENTAL, EXPERIMENTAL, tmpl, repair=True)
    mixed = compack_compare(EXPERIMENTAL, str(FIXTURES / fused['ref']), tmpl, repair=True)

    assert self_match['status'] == 'ok' and self_match['n_matched'] == 20
    assert self_match['repair_ref']['mode'] == 'as_read' and not self_match['repair_ref']['changed']
    assert mixed['status'] == 'ok'
    assert mixed['repair_ref']['mode'] == 'as_read'
    assert mixed['repair_test']['mode'] == 'repaired' and mixed['repair_test']['changed']


@pytest.mark.ccdc
def test_wrong_molecule_is_a_repair_failure():
    """A CIF that is not whole molecules of the template is refused, not compared, in either layout."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_compare
    mipcas = next(p for p in _manifest()['pairs'] if p['molecule'] == 'mipcas')
    ref, test = _paths(mipcas)
    for r in (compack_compare(ref, test, _template('nehzor'), repair=True),
              compack_compare(EXPERIMENTAL, EXPERIMENTAL, _template('mipcas'), repair=True)):
        assert r['status'].startswith('repair_failed')
        assert r['n_matched'] == -1


@pytest.mark.ccdc
def test_settings_reach_the_engine():
    """Engine settings are applied (a 15-molecule shell matches 15), an unknown one raises, and repair without a
    template raises."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import compack_compare, compack_engine
    pair = next(p for p in _manifest()['pairs'] if p['name'] == 'canonical_control')
    ref, test = _paths(pair)

    r = compack_compare(ref, test, _template('nehzor'), repair=True, settings={'packing_shell_size': 15})

    assert r['status'] == 'ok' and r['n_matched'] == 15
    with pytest.raises(ValueError):
        compack_engine({'packing_shell_sise': 15})
    with pytest.raises(ValueError):
        compack_compare(ref, test, repair=True)


@pytest.mark.ccdc
def test_batch_compack_repair_flag(tmp_path, monkeypatch):
    """batch_compack without repair is the engine on the CIF it writes as read; with repair it matches the fused NEHZOR
    crystal to another description of itself 20/20."""
    pytest.importorskip('ccdc', reason='CCDC licence required')
    import torch
    from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    man = _manifest()
    mo, cr = man['molecules']['nehzor'], man['crystals']['nehzor_canonical_control_ref']
    mol = MolData(z=torch.tensor(mo['atomic_numbers']), pos=torch.tensor(mo['positions'], dtype=torch.float32),
                  do_mol_analysis=True)
    p = torch.tensor(cr['params'])
    crystal = MolCrystalData(molecule=mol, sg_ind=cr['sg'], z_prime=1, max_z_prime=1, cell_lengths=p[:3],
                             cell_angles=p[3:6], aunit_centroid=p[6:9], aunit_orientation=p[9:12],
                             aunit_handedness=cr['handedness'], do_box_analysis=True)
    other_description = str(FIXTURES / 'nehzor_canonical_control_test.cif')
    monkeypatch.chdir(tmp_path)

    m_off, r_off = collate_data_list([crystal.clone()]).batch_compack(other_description, [0], n_cpus=1)
    status, n_matched, rmsd = _engine_as_read(other_description, 'compack_0.cif')
    m_on, _ = collate_data_list([crystal.clone()]).batch_compack(other_description, [0], n_cpus=1, repair=True)

    assert status == 'ok' and (int(m_off[0]), float(r_off[0])) == (n_matched, rmsd)
    assert int(m_on[0]) == 20
