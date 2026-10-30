"""
`crystal_building.image_pairs` against the route it shortcuts, on experimental crystals.

WHAT IS PINNED. For Z' = 1 crystals of many space groups, the closed-form image
instantiation must give the same intermolecular pair list as
`mol2cluster` -> `construct_radial_graph` (the eLJ route `MolCrystalData.analyze` takes):
the same number of pairs per crystal, the same distances, the same eLJ sum, and the same
gradient of that sum with respect to the crystal latent.

WHY PAIRS NEAR THE CUTOFF ARE SET ASIDE. The two routes compute a distance by different
float32 arithmetic, so a pair sitting on the cutoff can land on either side of the
comparison. Both lists are therefore trimmed to `CUTOFF - EDGE` before they are compared;
everything inside that radius has to agree pair for pair.

WHAT IS NOT PINNED. Crystals with a reference atom at the existing route's 1000-neighbour
cap (overlapped, unphysical cells): the existing list is truncated there and the closed
form is not. None occur in this fixture. Z' > 1 is refused by the module and checked only
for that refusal.

CPU-only, no checkpoint. Timing lives in `tests/bench_image_pairs.py`.
"""
import os

import pytest
import torch

from mxtaltools.analysis.vdw_analysis import elj_analysis
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.image_pairs import build_image_tables, image_pairs, pair_distances, select_images
from mxtaltools.dataset_utils.utils import collate_data_list

DATASET = os.path.join(os.path.dirname(__file__), 'datasets', 'mini_new_csd.pt')
CUTOFF = 10.0       # the cutoff GFN's eLJ call uses
EDGE = 1e-3         # Angstrom; pairs this close to the cutoff are not compared
SUPERCELL = 10
DIST_TOL = 2e-4     # Angstrom, on float32 distances of order 1-10 A between atoms up to ~40 A from the origin
MIN_CRYSTALS = 20
MIN_SPACE_GROUPS = 5


@pytest.fixture(scope='module')
def crystals():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    rows = [c for c in torch.load(DATASET, weights_only=False, map_location='cpu') if int(c.z_prime) == 1]
    if len(rows) < MIN_CRYSTALS:
        pytest.skip(f"only {len(rows)} Z'=1 crystals in the fixture")
    return rows


def _existing(batch):
    _, cluster = batch.analyze(['elj'], cutoff=CUTOFF, supercell_size=SUPERCELL,
                               std_orientation=False, return_cluster=True)
    ed = cluster.edges_dict
    per_target = torch.bincount(ed['edge_index_inter'][1])
    return ed, int((per_target >= 1001).sum())


def _trim(dist, graph, z_src, z_tgt):
    keep = dist < CUTOFF - EDGE
    return dist[keep], graph[keep], z_src[keep], z_tgt[keep]


def _elj(dist, graph, z_src, z_tgt, n):
    vdw = torch.tensor(list(VDW_RADII.values()))
    return elj_analysis(vdw, {'intermolecular_dist': dist, 'intermolecular_dist_atoms': [z_src, z_tgt],
                              'intermolecular_dist_batch': graph}, n, stiffness=2.5, envelope=None)


def _sorted(dist, graph):
    order = torch.argsort(graph.double() * 1e3 + dist.double())
    return dist[order], graph[order]


def test_pair_list_matches_the_cluster_route(crystals):
    """Same pairs, same distances and the same eLJ sum, crystal by crystal, across space groups."""
    space_groups = set()
    n_compared = 0
    for start in range(0, len(crystals), 32):
        batch = collate_data_list([c.clone() for c in crystals[start:start + 32]])
        n = batch.num_graphs
        with torch.no_grad():
            ed, at_cap = _existing(batch)
            assert at_cap == 0, ('a fixture crystal hit the existing 1000-neighbour cap; the existing list is '
                                 'truncated there and cannot serve as the reference')
            new = image_pairs(build_image_tables(batch), batch.T_fc, batch.T_cf, batch.aunit_centroid,
                              batch.aunit_orientation, CUTOFF, SUPERCELL)
        assert not bool(new['capped'].any()), 'the lattice-translation window was clamped on a fixture crystal'

        old = _trim(ed['intermolecular_dist'], ed['intermolecular_dist_batch'],
                    ed['intermolecular_dist_atoms'][0], ed['intermolecular_dist_atoms'][1])
        cur = _trim(new['dist'], new['graph'], new['z_src'], new['z_tgt'])
        n_old = torch.bincount(old[1], minlength=n)
        n_new = torch.bincount(cur[1], minlength=n)
        assert torch.equal(n_old, n_new), (
            f'pair counts differ inside {CUTOFF - EDGE} A for crystals '
            f'{(n_old != n_new).nonzero().flatten().tolist()} of the batch starting at {start}: '
            f'existing {n_old[n_old != n_new].tolist()}, closed form {n_new[n_old != n_new].tolist()}')
        assert int(n_old.min()) > 0, 'a crystal with no intermolecular pair inside the cutoff proves nothing'

        d_old, _ = _sorted(old[0], old[1])
        d_new, _ = _sorted(cur[0], cur[1])
        worst = float((d_old - d_new).abs().max())
        assert worst < DIST_TOL, f'sorted pair distances differ by up to {worst:.2e} A'

        e_old, e_new = _elj(*old, n), _elj(*cur, n)
        rel = ((e_old - e_new).abs() / e_old.abs().clamp(min=1.0)).max()
        assert float(rel) < 1e-3, f'eLJ sums differ by up to {float(rel):.2e} (relative)'

        space_groups.update(int(s) for s in batch.sg_ind.flatten().tolist())
        n_compared += n
    assert n_compared >= MIN_CRYSTALS
    assert len(space_groups) >= MIN_SPACE_GROUPS, (
        f'only space groups {sorted(space_groups)} were exercised; the point of this fixture is their spread')


def test_gradient_with_respect_to_the_latent_matches(crystals):
    """d(eLJ)/d(latent) through the closed form equals the existing route's, by autograd."""
    vdw = torch.tensor(list(VDW_RADII.values()))
    base = collate_data_list([c.clone() for c in crystals[:24]])
    n = base.num_graphs
    x0 = base.latent_params(gauge_fix_free_axes=True)

    x = x0.clone().requires_grad_(True)
    cb = base.clone()
    cb.latent_to_cell_params(x)
    out, cluster = cb.analyze(['elj'], cutoff=CUTOFF, supercell_size=SUPERCELL, std_orientation=False,
                              return_cluster=True)
    assert int((torch.bincount(cluster.edges_dict['edge_index_inter'][1]) >= 1001).sum()) == 0
    g_old, = torch.autograd.grad(out['elj'].sum(), x)

    x = x0.clone().requires_grad_(True)
    cb = base.clone()
    cb.latent_to_cell_params(x)
    new = image_pairs(build_image_tables(cb), cb.T_fc, cb.T_cf, cb.aunit_centroid, cb.aunit_orientation,
                      CUTOFF, SUPERCELL)
    e_new = elj_analysis(vdw, {'intermolecular_dist': new['dist'],
                               'intermolecular_dist_atoms': [new['z_src'], new['z_tgt']],
                               'intermolecular_dist_batch': new['graph']}, n, stiffness=2.5, envelope=None)
    g_new, = torch.autograd.grad(e_new.sum(), x)

    assert torch.isfinite(g_old).all() and torch.isfinite(g_new).all()
    live = g_old.norm(dim=1) > 0
    assert int(live.sum()) >= 12, 'too few crystals with a non-zero gradient to compare'
    rel = (g_new[live] - g_old[live]).norm(dim=1) / g_old[live].norm(dim=1)
    assert float(rel.max()) < 1e-3, f'latent gradients differ by up to {float(rel.max()):.2e} (relative)'


def test_distances_carry_second_derivatives(crystals):
    """The returned distances are plain arithmetic of the cell and pose, so a force-matching loss can
    differentiate through a gradient taken with create_graph."""
    base = collate_data_list([c.clone() for c in crystals[:4]])
    x = base.latent_params(gauge_fix_free_axes=True).clone().requires_grad_(True)
    cb = base.clone()
    cb.latent_to_cell_params(x)
    new = image_pairs(build_image_tables(cb), cb.T_fc, cb.T_cf, cb.aunit_centroid, cb.aunit_orientation, 6.0)
    energy = (new['dist'] - 4.0).square().sum()
    g, = torch.autograd.grad(energy, x, create_graph=True)
    gg, = torch.autograd.grad(g.square().sum(), x)
    assert torch.isfinite(gg).all() and float(gg.abs().max()) > 0


def test_node_indices_address_the_batch_atoms(crystals):
    """`node_ref` / `node_img` index the batch's own atoms and carry the pair's atomic numbers."""
    batch = collate_data_list([c.clone() for c in crystals[:8]])
    with torch.no_grad():
        new = image_pairs(build_image_tables(batch), batch.T_fc, batch.T_cf, batch.aunit_centroid,
                          batch.aunit_orientation, 6.0)
    assert torch.equal(batch.z.long()[new['node_ref']], new['z_tgt'])
    assert torch.equal(batch.z.long()[new['node_img']], new['z_src'])
    assert torch.equal(batch.batch[new['node_ref']], new['graph'])
    assert torch.equal(batch.batch[new['node_img']], new['graph'])


def test_caps_leave_ordinary_crystals_alone_and_bound_squeezed_ones(crystals):
    """`max_images` and `max_pairs` are a memory bound for cells far below any physical density:
    with them set above what a real crystal needs the lists are unchanged, entry for entry; on a
    squeezed cell every crystal holds at most the caps, keeps its nearest images and shortest
    pairs, and is flagged."""
    batch = collate_data_list([c.clone() for c in crystals])
    tables = build_image_tables(batch)
    geom = (batch.T_fc, batch.T_cf, batch.aunit_centroid, batch.aunit_orientation)
    free = image_pairs(tables, *geom, 6.0)
    loose = image_pairs(tables, *geom, 6.0, max_images=100_000, max_pairs=10_000_000)
    for key in ('dist', 'graph', 'ia', 'ib', 'image', 'node_ref', 'node_img'):
        assert torch.equal(free[key], loose[key]), key
    assert not bool(loose['image_capped'].any()) and not bool(loose['pair_capped'].any())

    # the same molecules in cells a third the size: far more images and pairs in range
    small = (batch.T_fc / 3, batch.T_cf * 3, batch.aunit_centroid, batch.aunit_orientation)
    wide = select_images(tables, *small, 6.0)
    wide_pairs = pair_distances(tables, wide, 6.0)
    n_img, n_pair = 40, 300
    assert int(wide['images_kept'].max()) > n_img
    assert int(torch.bincount(wide_pairs['graph'], minlength=batch.num_graphs).max()) > n_pair

    sel = select_images(tables, *small, 6.0, max_images=n_img)
    assert int(sel['images_kept'].max()) == n_img
    assert torch.equal(sel['image_capped'], wide['images_kept'] > n_img)
    # the kept images are the nearest ones: none dropped is closer than any kept
    centre2 = lambda s_: s_['rel'].square().sum(1)
    for g in sel['image_capped'].nonzero().flatten().tolist():
        kept = centre2(sel)[sel['graph'] == g].max()
        every = centre2(wide)[wide['graph'] == g].sort().values
        assert torch.isclose(kept, every[n_img - 1], rtol=1e-5)

    capped = pair_distances(tables, sel, 6.0, max_pairs=n_pair)
    per = torch.bincount(capped['graph'], minlength=batch.num_graphs)
    full = pair_distances(tables, sel, 6.0)
    full_per = torch.bincount(full['graph'], minlength=batch.num_graphs)
    assert int(per.max()) == n_pair and torch.equal(capped['pair_capped'], full_per > n_pair)
    assert torch.equal(per, full_per.clamp(max=n_pair))
    for g in capped['pair_capped'].nonzero().flatten().tolist():
        kept = capped['dist'][capped['graph'] == g].max()
        every = full['dist'][full['graph'] == g].sort().values
        assert torch.isclose(kept, every[n_pair - 1], rtol=1e-5)
    # the block-level bound gives the same list as one block would (small blocks force the split)
    split = pair_distances(tables, sel, 6.0, max_pairs=n_pair, max_block=4096)
    assert torch.equal(torch.bincount(split['graph'], minlength=batch.num_graphs), per)
    assert torch.allclose(split['dist'].sort().values, capped['dist'].sort().values, atol=1e-5)
    # a capped list still carries gradients
    x = batch.T_fc.clone().requires_grad_(True)
    sel_g = select_images(tables, x / 3, torch.linalg.inv(x / 3), batch.aunit_centroid, batch.aunit_orientation, 6.0,
                          max_images=n_img)
    d = pair_distances(tables, sel_g, 6.0, max_pairs=n_pair)['dist']
    assert torch.isfinite(torch.autograd.grad(d.sum(), x)[0]).all()


def test_z_prime_above_one_is_refused():
    if not os.path.exists(DATASET):
        pytest.skip(f'{DATASET} not present')
    rows = [c for c in torch.load(DATASET, weights_only=False, map_location='cpu') if int(c.z_prime) > 1]
    if not rows:
        pytest.skip("no Z' > 1 crystal in the fixture")
    with pytest.raises(NotImplementedError):
        build_image_tables(collate_data_list([rows[0].clone()]))
