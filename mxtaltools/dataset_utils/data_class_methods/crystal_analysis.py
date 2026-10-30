"""Analysis and energy methods for molecular crystal data objects.

``MolCrystalAnalysis`` is a mixin of ``MolCrystalData``. Its entry point is
``MolCrystalAnalysis.analyze``, which builds whatever structure the requested
compute keys need (periodic cluster, unit cell, or neither) and dispatches each
key to a method of this class. Every CCDC COMPACK comparison in the package
runs through ``compack_compare`` (one pair of CIFs, on the engine
``compack_engine`` builds), which ``single_compack_run`` and
``MolCrystalAnalysis.batch_compack`` call; with ``repair`` it compares the
crystals after ``repair_compack_crystal`` has undone the CSD reader's bonds
between molecules (``compack_bond_template`` gives the molecule's own bonds).

Module constants:

``COMPUTES_REQUIRE_CLUSTER``
    Every compute key bound by ``MolCrystalAnalysis._init_computes``, mapped to
    whether it needs a periodic cluster with a built radial graph
    (``mol2cluster`` followed by ``construct_radial_graph``). ``analyze`` treats
    a key absent from this dict as False.
``COMPUTES_REQUIRE_UNIT_CELL``
    Keys that need a built unit cell (``unit_cell_pos``, ``sym_mult``,
    ``T_fc``) but no cluster: ``'uma_pot'``, ``'uma'``, ``'mace_pot'`` and
    ``'mace'``. The gas-phase keys are absent because they build their own P1
    cell.
``EXP6_DEFAULT_PARAMS``, ``EXP6_DEFAULT_CHARGES``
    The parameter set (``'fit'``) and charge model (``'mmff'``) of the
    ``'exp6'`` key when its kwargs do not name them.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mxtaltools.analysis.crystal_rdf import crystal_rdf
from mxtaltools.analysis.vdw_analysis import get_intermolecular_dists_dict, lj_analysis, electrostatic_analysis, \
    buckingham_energy, silu_energy, vdW_analysis, qlj_analysis, elj_analysis
from mxtaltools.common.sym_utils import niggli_reduction_penalty, cell_reduction_penalty
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import compute_crystal_mace_on_mxt_batch
from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
from mxtaltools.models.functions.radial_graph import build_radial_graph


def compack_engine(settings=None):
    """The CCDC COMPACK engine every comparison in this package runs on.

    A ``ccdc.crystal.PackingSimilarity`` with ``packing_shell_size = 20`` and
    every other setting at its CCDC default, then each ``name: value`` of
    ``settings`` set on ``engine.settings`` (``packing_shell_size``,
    ``distance_tolerance``, ``angle_tolerance``,
    ``allow_molecular_differences``, ``ignore_bond_types``, ``timeout_ms``,
    ...). Requires the CSD Python API (``ccdc``), imported on call.

    Raises
    ------
    ValueError
        If ``settings`` names a setting the engine does not have (assigning it
        would add an attribute the engine never reads).
    """
    from ccdc.crystal import PackingSimilarity
    engine = PackingSimilarity()
    engine.settings.packing_shell_size = 20
    for name, value in (settings or {}).items():
        if not hasattr(engine.settings, name):
            raise ValueError(f'PackingSimilarity settings have no {name!r}')
        setattr(engine.settings, name, value)
    return engine


def compack_compare(ref_path, test_path, template=None, repair=False, settings=None):
    """One CCDC COMPACK comparison of two CIF files, ``ref_path`` the
    reference: the package's COMPACK producer (``single_compack_run`` and
    ``MolCrystalAnalysis.batch_compack`` run through it).

    Reads the first crystal of each file with ``ccdc.io.CrystalReader`` or,
    with ``repair``, takes it from ``repair_compack_crystal``, and compares the
    two on ``compack_engine(settings)``. Prints nothing. Requires the CSD
    Python API (``ccdc``), imported on call. Errors raised while reading the
    files propagate.

    Parameters
    ----------
    ref_path : str
        CIF of the reference crystal.
    test_path : str
        CIF of the crystal to compare.
    template : dict, optional
        ``compack_bond_template`` of the molecule (or asymmetric unit) both
        CIFs hold; required with ``repair``.
    repair : bool, optional
        Repair both CIFs before the comparison. Default False: the crystals as
        the reader perceives them.
    settings : dict, optional
        Engine settings, as ``compack_engine`` takes them.

    Returns
    -------
    dict
        ``rmsd`` (Angstrom) and ``n_matched`` (molecules matched) as COMPACK
        reports them; ``status``: ``'ok'``; ``'failed'`` when comparing or
        reading the result raises AttributeError (rmsd 0, n_matched 0);
        ``'repair_failed: <reason>'`` when either CIF fails the repair (rmsd
        nan, n_matched -1). ``repair_ref`` and ``repair_test``: each CIF's
        repair info; None without ``repair`` or after a repair failure.
    """
    if repair and template is None:
        raise ValueError('repair needs the molecule template (compack_bond_template)')
    info_ref = info_test = None
    if repair:
        try:
            ref_crystal, info_ref = repair_compack_crystal(ref_path, template)
            test_crystal, info_test = repair_compack_crystal(test_path, template)
        except ValueError as e:
            return dict(rmsd=float('nan'), n_matched=-1, status=f'repair_failed: {str(e)[:200]}', repair_ref=None,
                        repair_test=None)
    else:
        from ccdc.io import CrystalReader
        ref_crystal = CrystalReader(ref_path)[0]
        test_crystal = CrystalReader(test_path)[0]
    engine = compack_engine(settings)
    try:
        result = engine.compare(ref_crystal, test_crystal)
        rmsd, n_matched, status = float(result.rmsd), int(result.nmatched_molecules), 'ok'
    except AttributeError:
        rmsd, n_matched, status = 0.0, 0, 'failed'
    return dict(rmsd=rmsd, n_matched=n_matched, status=status, repair_ref=info_ref, repair_test=info_test)


def single_compack_run(ind, test_path, ref_path):
    """One CCDC COMPACK comparison of two CIF files, printed:
    ``compack_compare(ref_path, test_path)`` (no repair, ``packing_shell_size
    = 20``, CCDC defaults otherwise), ``ref_path`` being the reference.

    Parameters
    ----------
    ind : object
        Label used only in the printed message.
    test_path : str
        CIF of the crystal to compare.
    ref_path : str
        CIF of the reference crystal.

    Returns
    -------
    rmsd : float
        RMSD reported by COMPACK (printed in Angstrom); 0 if the comparison
        failed.
    nmatched_molecules : int
        Number of molecules COMPACK matched; 0 if the comparison failed.

    Notes
    -----
    A comparison counts as failed when ``compack_compare`` reports
    ``'failed'`` (comparing or reading the result raised AttributeError); the
    function then prints "Analysis failed" and returns ``(0, 0)``.
    """
    r = compack_compare(ref_path, test_path)
    if r['status'] != 'ok':
        print("Analysis failed")
        return 0, 0
    print(f"Crystal {ind} RMSD = {r['rmsd']:.3f} Å, {r['n_matched']} mols matched")
    return r['rmsd'], r['n_matched']


def _batch_compack_job(ind, test_path, ref_path, template, repair, settings):
    """``MolCrystalAnalysis.batch_compack``'s worker: ``compack_compare``,
    printed and returned as ``single_compack_run`` does, a repair failure as
    ``(nan, -1)``."""
    r = compack_compare(ref_path, test_path, template=template, repair=repair, settings=settings)
    if r['status'] == 'ok':
        print(f"Crystal {ind} RMSD = {r['rmsd']:.3f} Å, {r['n_matched']} mols matched")
        return r['rmsd'], r['n_matched']
    if r['status'] == 'failed':
        print("Analysis failed")
        return 0, 0
    print(f"Crystal {ind}: {r['status']}")
    return float('nan'), -1


def compack_bond_template(atomic_numbers, positions):
    """The bond graph the CSD reader perceives on one isolated molecule, or on
    one isolated asymmetric unit of several, for ``repair_compack_crystal``.

    Writes the atoms alone, centred, in a 60 A cubic P1 cell (labels
    ``<symbol><index + 1>``), reads them back with
    ``ccdc.crystal.Crystal.from_string`` and records the reader's bonds and
    components by atom index. Requires the CSD Python API (``ccdc``), imported
    on call.

    Parameters
    ----------
    atomic_numbers : array-like of int
        [n_atoms], in the atom order the crystal CIFs write each molecule (or
        asymmetric unit) in.
    positions : array-like of float
        [n_atoms, 3] Cartesian coordinates (Angstrom), same order.

    Returns
    -------
    dict
        ``n_atoms``; ``symbols``, the element symbol of each atom; ``bonds``,
        sorted ``(i, j)`` index pairs with ``i < j``; ``components``, the atom
        indices of each component the reader returns (one per molecule);
        ``formulas``, their formulas as the reader writes them, sorted;
        ``distances``, the [n_atoms, n_atoms] interatomic distances (Angstrom,
        numpy).
    """
    import numpy as np
    from ccdc.crystal import Crystal
    from ccdc.molecule import Atom
    z = np.asarray(atomic_numbers).astype(int).reshape(-1)
    X = np.asarray(positions, dtype=float).reshape(-1, 3)
    X = X - X.mean(0)
    L = 60.0
    sym = [Atom(atomic_number=int(v)).atomic_symbol for v in z]
    rows = [f'{s} {s}{i + 1} {(x[0] + L / 2) / L:.10f} {(x[1] + L / 2) / L:.10f} {(x[2] + L / 2) / L:.10f}'
            for i, (s, x) in enumerate(zip(sym, X))]
    txt = '\n'.join(['data_template'] + [f'_cell_length_{a} {L}' for a in 'abc'] +
                    [f'_cell_angle_{a} 90' for a in ('alpha', 'beta', 'gamma')] +
                    ["_space_group_name_H-M_alt 'P 1'", '_space_group_IT_number 1', 'loop_',
                     '_space_group_symop_operation_xyz', "'x, y, z'", 'loop_', '_atom_site_type_symbol',
                     '_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
                    + rows) + '\n'
    m = Crystal.from_string(txt, 'cif').molecule
    li = {f'{s}{i + 1}': i for i, s in enumerate(sym)}
    bonds = sorted(tuple(sorted((li[b.atoms[0].label], li[b.atoms[1].label]))) for b in m.bonds)
    components = [sorted(li[a.label] for a in c.atoms) for c in m.components]
    return dict(n_atoms=len(z), symbols=sym, bonds=bonds, components=components,
                formulas=sorted(c.formula for c in m.components),
                distances=np.linalg.norm(X[:, None] - X[None], axis=-1))


def compack_cif_sites(path):
    """Labels, element symbols and fractional coordinates of a CIF's
    ``_atom_site`` loop.

    Reads the first data block with the package's own CIF tokenizer
    (``dataset_utils.construction.cif.parse_cif``).

    Returns
    -------
    labels : list of str
        ``_atom_site_label`` values, in file order.
    symbols : list of str or None
        ``_atom_site_type_symbol`` values, same order; None if the loop has no
        such column.
    fractional : numpy.ndarray
        [n_sites, 3] ``_atom_site_fract_x/y/z``, same order.
    """
    import numpy as np
    from mxtaltools.dataset_utils.construction.cif import as_float, parse_cif
    with open(path, encoding='utf-8') as fh:
        block = parse_cif(fh.read())[0]
    loop = block.loop_with('_atom_site_label')
    if loop is None:
        raise ValueError(f'no _atom_site loop in {path}')
    frac = np.array([[as_float(v) for v in loop.column(f'_atom_site_fract_{a}')] for a in 'xyz']).T
    symbols = list(loop.column('_atom_site_type_symbol')) if '_atom_site_type_symbol' in loop else None
    return list(loop.column('_atom_site_label')), symbols, frac


def _template_deviation(B, template):
    """Largest deviation (A) of block B's interatomic distances ([n_atoms, 3] Cartesian) from the template's, within
    each template component (distances between the molecules of one asymmetric unit are not compared)."""
    import numpy as np
    dev = 0.0
    for comp in template['components']:
        D = np.linalg.norm(B[comp][:, None] - B[comp][None], axis=-1)
        dev = max(dev, float(np.abs(D - template['distances'][np.ix_(comp, comp)]).max()))
    return dev


def repair_compack_crystal(path, template):
    """The first crystal of a CIF as ``ccdc.io.CrystalReader`` reads it, its
    molecules restored to the template's bond graph where the reader fused
    them.

    The reader perceives bonds from interatomic distances. Where molecules sit
    close it bonds atoms of different molecules, adds image atoms to close the
    resulting polymer and moves atoms by lattice vectors, so a cell of Z
    molecules reads as fewer components or a polymer and PackingSimilarity
    fails or under-matches. Seen on NEHZOR eLJ search minima, whose Br...Br
    contacts between molecules of 2.0-2.8 A the reader bonds: a crystal against
    itself rebuilt from another description matched 9 of 20 molecules at RMSD
    1.5e-6 A.

    A CIF in the writer's layout -- distinct site labels, the sites Z blocks of
    ``template['n_atoms']``, each block the template's elements in its order
    with its interatomic distances (to 1e-3 A within each component), as
    ``MolCrystalData.write_cif(..., mode='unit cell')`` writes a unit cell --
    is repaired: one atom per site label is kept (atoms the reader added, with
    a repeated label or one not in the file, are removed), every atom is put at
    the file's coordinates through the reader's own fractional-to-Cartesian
    map, and block k (sites ``k * n_atoms`` to ``(k + 1) * n_atoms - 1``) is
    given exactly the template's bonds: every other bond, between molecules or
    within one, is removed, and a missing template bond is added with type
    ``'Unknown'``. A crystal the reader did not fuse comes back as read. Any
    other CIF (a CSD file: an asymmetric unit, its own atom order) is used as
    read when the reader returns a non-polymeric molecule whose every component
    has one of the template's formulas, and refused otherwise. Requires the CSD
    Python API (``ccdc``), imported on call.

    Parameters
    ----------
    path : str
        CIF file.
    template : dict
        ``compack_bond_template`` of the molecule (or asymmetric unit).

    Returns
    -------
    crystal : ccdc.crystal.Crystal
        The repaired crystal, or the crystal as read.
    info : dict
        ``mode``: ``'repaired'`` (writer layout) or ``'as_read'``; ``Z`` (the
        number of blocks; None as read); ``components_as_read``;
        ``image_atoms_removed``; ``atoms_moved`` (moved back to the written
        coordinates by more than 1e-4 A in any Cartesian component);
        ``inter_bonds_removed``, ``intra_bonds_removed``,
        ``intra_bonds_added``; ``changed`` (any of these happened).

    Raises
    ------
    ValueError
        If a writer-layout CIF does not repair to Z blocks of the template's
        components, each with the template's interatomic distances to 1e-3 A,
        or a CIF of any other layout is polymeric or holds a component of
        another formula as read.
    """
    import numpy as np
    from ccdc.io import CrystalReader
    from ccdc.molecule import Bond
    na, bonds = template['n_atoms'], template['bonds']
    c = CrystalReader(path)[0]
    lab, sym, fr = compack_cif_sites(path)
    m = c.molecule
    n_comp_read = len(m.components)
    Fc = np.array([a.fractional_coordinates for a in m.atoms])
    Cc = np.array([a.coordinates for a in m.atoms])
    Mx = np.linalg.lstsq(np.c_[Fc, np.ones(len(Fc))], Cc, rcond=None)[0]  # the reader's fractional -> Cartesian map
    target = np.c_[fr, np.ones(len(fr))] @ Mx
    Z = len(lab) // na
    writer = (len(set(lab)) == len(lab) and Z > 0 and len(lab) == Z * na and sym is not None
              and all(sym[k * na:(k + 1) * na] == template['symbols'] for k in range(Z))
              and all(_template_deviation(target[k * na:(k + 1) * na], template) <= 1e-3 for k in range(Z)))
    if not writer:
        formulas = [x.formula for x in m.components]
        if m.is_polymeric or any(f not in template['formulas'] for f in formulas):
            raise ValueError(f'not in the writer layout, and read as {"a polymer of " if m.is_polymeric else ""}'
                             f'components {sorted(formulas)}, not the template\'s {template["formulas"]}')
        return c, dict(mode='as_read', Z=None, components_as_read=n_comp_read, image_atoms_removed=0, atoms_moved=0,
                       inter_bonds_removed=0, intra_bonds_removed=0, intra_bonds_added=0, changed=False)
    labset, idx = set(lab), {l: i for i, l in enumerate(lab)}
    seen, extra = set(), []
    for a in m.atoms:
        if a.label in seen or a.label not in labset:
            extra.append(a)
        seen.add(a.label)
    if extra:
        m.remove_atoms(extra)
    at = {a.label: a for a in m.atoms}
    if len(at) != len(lab) or len(m.atoms) != len(lab):
        raise ValueError(f'{len(m.atoms)} atoms after removing images, {len(lab)} sites')
    moved = 0
    for l, x in zip(lab, target):
        if np.abs(np.array(at[l].coordinates) - x).max() > 1e-4:
            at[l].coordinates = tuple(float(v) for v in x)
            moved += 1
    want = {(k * na + p, k * na + q) for k in range(Z) for p, q in bonds}
    have, rem = set(), []
    for b in m.bonds:
        e = tuple(sorted((idx[b.atoms[0].label], idx[b.atoms[1].label])))
        have.add(e)
        if e not in want:
            rem.append(b)
    n_inter = sum(1 for b in rem if idx[b.atoms[0].label] // na != idx[b.atoms[1].label] // na)
    add = sorted(want - have)
    if rem:
        m.remove_bonds(rem)
    for p, q in add:
        m.add_bond(Bond.BondType('Unknown'), at[lab[p]], at[lab[q]])
    changed = bool(extra or moved or rem or add)
    if changed:
        c.molecule = m
    comps = c.molecule.components
    info = dict(mode='repaired', Z=Z, components_as_read=n_comp_read, image_atoms_removed=len(extra),
                atoms_moved=moved, inter_bonds_removed=n_inter, intra_bonds_removed=len(rem) - n_inter,
                intra_bonds_added=len(add), changed=changed)
    if sorted(x.formula for x in comps) != sorted(template['formulas'] * Z):
        raise ValueError(f'repaired: components {sorted(x.formula for x in comps)}, not {Z} x {template["formulas"]}')
    P = {a.label: np.array(a.coordinates) for a in c.molecule.atoms}
    for k in range(Z):
        dev = _template_deviation(np.stack([P[l] for l in lab[k * na:(k + 1) * na]]), template)
        if dev > 1e-3:
            raise ValueError(f'repaired block {k}: internal distances off the template by {dev:.3g} A')
    return c, info


# defaults of the 'exp6' compute key (MolCrystalAnalysis.compute_exp6_energy), read by analyze when it builds the
# cluster and by the compute itself: FIT exp-6 (DMACRYS manual Appendix D) in its exp-6 pair form, with RDKit MMFF
# point charges
EXP6_DEFAULT_PARAMS = 'fit'
EXP6_DEFAULT_CHARGES = 'mmff'

# noinspection PyAttributeOutsideInit
COMPUTES_REQUIRE_CLUSTER = {'lj': True,
                            'qlj': True,
                            'elj': True,
                            'exp6': True,
                            'es': True,
                            'bh': True,
                            'silu': True,
                            'vdw': True,
                            'vdw_max': True,
                            'ellipsoid': True,
                            'niggli_overlap': False,
                            'reduction_en': False,
                            'rdf': True,
                            'ellipsoid_emb': True,
                            'uma_pot': False,
                            'uma_gas_pot': False,
                            'uma': False,
                            'mace_pot': False,
                            'mace_gas_pot': False,
                            'mace': False,
                            'latent_harmonic': False,
                            'latent_multiharmonic': False,
                            'latent_gaussian': False,
                            }

# these need a built unit cell (`unit_cell_pos`/`sym_mult`/`T_fc`) but never touch
# the exploded supercell cluster or edges_dict, so they don't belong in
# COMPUTES_REQUIRE_CLUSTER above. The '_gas_phase' variants are excluded here too:
# they always build their own fresh P1 diffuse cell with force_rebuild=True,
# ignoring whatever unit cell (if any) is already on the batch.
COMPUTES_REQUIRE_UNIT_CELL = {'uma_pot': True,
                              'uma': True,
                              'mace_pot': True,
                              'mace': True,
                              }


class MolCrystalAnalysis:
    """Energy, overlap, RDF, cell-reduction and latent-space analyses of crystal batches.

    Mixed into ``MolCrystalData``; relies on its other base classes for
    ``clone``, ``add_graph_attr``, ``mol2cluster``, ``mol2ucell``,
    ``latent_params``, ``split_cell_params`` and ``get_ellipsoid_embedding``. The pairwise energies and overlaps are implemented
    for Batch objects only and read the pair list stored in ``edges_dict`` by
    ``construct_radial_graph``.
    """

    def construct_intra_radial_graph(self, cutoff: float = 6):
        """Build an all-atom radius graph within each graph of a non-cluster object.

        Sets ``self.edges_dict`` to ``build_radial_graph(self.pos, self.batch,
        self.ptr, cutoff, max_num_neighbors=10000)``, which, with no ``aux_ind``,
        holds only ``'edge_index'`` [2, num_edges]: pairs of atoms in the same
        graph closer than ``cutoff``, without self-loops. Also sets
        ``self.edge_index`` to the same tensor.

        Parameters
        ----------
        cutoff : float, optional
            Edge radius, in the units of ``pos`` (Angstrom).

        Raises
        ------
        RuntimeError
            If ``self.aux_ind`` is not None, i.e. the object is already a
            cluster.
        """
        if self.aux_ind is not None:
            raise RuntimeError("Cannot build molecular graph when we have already built the cluster")
        self.edges_dict = build_radial_graph(self.pos,
                                             self.batch,
                                             self.ptr,
                                             cutoff,
                                             max_num_neighbors=10000
                                             )
        self.edge_index = self.edges_dict['edge_index']

    def construct_radial_graph(self,
                               cutoff: float = 6,
                               max_num_neighbors=1000):
        """Build the pair list of a cluster batch and store it as ``self.edges_dict``.

        Calls ``vdw_analysis.get_intermolecular_dists_dict`` on ``self`` (or, for
        a single object, on ``collate_data_list([self])``), which runs
        ``build_radial_graph(pos, batch, ptr, cutoff, max_num_neighbors,
        aux_ind=aux_ind, mol_ind=mol_ind)`` and adds distances. The object must
        be a cluster as produced by ``mol2cluster``: ``aux_ind`` (0 for the
        asymmetric unit, 1 for surrounding image atoms, 2 for image atoms that
        enter no pair) and ``mol_ind`` set.
        Without ``aux_ind`` the lookup of ``'edge_index_inter'`` raises KeyError.

        Keys of ``edges_dict``:

        - ``'edge_index'`` [2, n_intra]: pairs of asymmetric-unit atoms in the
          same molecule within ``cutoff``.
        - ``'edge_index_inter'`` [2, n_inter]: row 1 is an asymmetric-unit atom;
          row 0 is an image atom (``aux_ind == 1``) or, for Z' > 1, an
          asymmetric-unit atom of a different ``mol_ind``, within ``cutoff``.
        - ``'inside_inds'``, ``'outside_inds'``, ``'inside_batch'``,
          ``'n_repeats'``: as returned by ``build_radial_graph``.
        - ``'cutoff'`` (float), ``'num_graphs'``, ``'graph_size'``
          (``num_atoms``), ``'outside_batch'`` (``batch``).
        - ``'intramolecular_dist'`` [n_intra] and ``'intramolecular_dist_atoms'``
          (the two atoms' atomic numbers) for ``'edge_index'``.
        - ``'intermolecular_dist'`` [n_inter], ``'intermolecular_dist_batch'``
          [n_inter] (graph index) and ``'intermolecular_dist_atoms'`` for
          ``'edge_index_inter'``.
        - ``'intermolecular_partial_charges'``: only when ``x`` has one row per
          atom; taken from ``x[:, 0]`` (or ``x`` itself if 1-D).

        Parameters
        ----------
        cutoff : float, optional
            Pair radius, in the units of ``pos`` (Angstrom).
        max_num_neighbors : int, optional
            Cap on neighbours per asymmetric-unit atom, applied separately to
            the within-asymmetric-unit search and the image-atom search.
        """
        if self.is_batch:
            self.edges_dict = get_intermolecular_dists_dict(
                self,
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

        else:
            self.edges_dict = get_intermolecular_dists_dict(
                collate_data_list([self]),
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

    def _pre_compute_checks(self):
        """Create ``vdw_radii_tensor`` (radii indexed by atomic number) if absent; raise RuntimeError if ``edges_dict`` is absent."""
        if not hasattr(self, 'vdw_radii_tensor'):
            self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing energies!")

    def compute(self, requests: dict, **kwargs):
        """Evaluate each requested compute on this object and collect the results.

        Builds ``self.computes`` with ``_init_computes()`` only if it is absent;
        an existing dict is used as is, without rebinding. Performs no
        structure building: cluster computes need ``edges_dict`` already built
        (see ``analyze``).

        Parameters
        ----------
        requests : iterable of str
            Compute keys. Annotated as dict; any iterable of keys works
            (iterating a dict yields its keys). Each key is lower-cased for
            lookup in ``self.computes``.
        **kwargs
            Passed unchanged to every requested compute function.

        Returns
        -------
        dict
            ``{key: output}``, keyed by the request strings as given.

        Raises
        ------
        KeyError
            If a lower-cased key is not in ``self.computes``.
        """
        if not hasattr(self, 'computes'):
            self._init_computes()
        return {key: self.computes[key.lower()](**kwargs) for key in requests}

    def _init_computes(self, override: bool = False):
        """Bind ``self.computes`` (compute key -> callable) and attach the requirement tables.

        Does nothing if ``self.computes`` exists, unless ``override`` is True.
        Otherwise sets ``self.computes``, ``self.computes_requires_cluster``
        (``COMPUTES_REQUIRE_CLUSTER``) and ``self.computes_requires_unit_cell``
        (``COMPUTES_REQUIRE_UNIT_CELL``). Every value in ``self.computes`` is a
        method bound to ``self`` except ``'niggli_overlap'``, which is the
        module-level function ``sym_utils.niggli_reduction_penalty``.
        """
        if not hasattr(self, 'computes') or override:
            self.computes = {'lj': self.compute_LJ_energy,
                             'qlj': self.compute_qLJ_energy,
                             'elj': self.compute_eLJ_energy,
                             'exp6': self.compute_exp6_energy,
                             'es': self.compute_ES_energy,
                             'bh': self.compute_buckingham_energy,
                             'silu': self.compute_silu_energy,
                             'vdw': self.compute_vdW_overlap,
                             'vdw_max': self.compute_max_vdW_overlap,
                             'ellipsoid': self.compute_ellipsoidal_overlap,
                             'niggli_overlap': niggli_reduction_penalty,
                             'reduction_en': self.compute_cell_reduction_penalty,
                             'rdf': self.compute_rdf,
                             'ellipsoid_emb': self.compute_ellipsoid_embedding,
                             'uma_pot': self.compute_crystal_uma,
                             'uma_gas_pot': self.compute_lattice_gas_phase_uma,
                             'uma': self.compute_lattice_uma,
                             'mace_pot': self.compute_crystal_mace,
                             'mace_gas_pot': self.compute_lattice_gas_phase_mace,
                             'mace': self.compute_lattice_mace,
                             'latent_harmonic': self.latent_harmonic_en,
                             'latent_multiharmonic': self.latent_multiharmonic_en,
                             # Same compute as latent_harmonic, deliberately a distinct
                             # NAME so the GFN can keep is_crystal=True: the crystal latent
                             # layout, enforce_crystal_system and the dead-row machinery all
                             # stay live while the energy stays a cheap analytic gaussian.
                             # See gfn docs/decisions.md D33.
                             'latent_gaussian': self.latent_harmonic_en,
                             }
            self.computes_requires_cluster = COMPUTES_REQUIRE_CLUSTER
            self.computes_requires_unit_cell = COMPUTES_REQUIRE_UNIT_CELL

    def compute_ellipsoid_embedding(self):
        """Scatter-sum ellipsoid pair features into one row per graph.

        Calls ``get_ellipsoid_embedding(surface_padding=1)`` on a cluster batch
        whose ``edges_dict`` is built, and sums its feature tensor ``x`` (one
        row per retained ellipsoid pair) into [num_graphs, num_features] using
        ``molwise_batch`` (one entry per molecule in the cluster) as the
        scatter index. The scatter index and ``x`` therefore have matching
        lengths only when the number of retained pairs equals the number of
        cluster molecules; otherwise the scatter fails. Takes no keyword
        arguments, so any kwargs forwarded by ``compute``/``analyze`` raise
        TypeError.

        Returns
        -------
        torch.Tensor
            [num_graphs, num_features], when the scatter succeeds.
        """
        edge_j_good, molwise_batch, norm_factor, normed_v1, normed_v2, v1, v2, x = self.get_ellipsoid_embedding(
            surface_padding=1
        )
        return scatter(x, molwise_batch, dim=0, dim_size=self.num_graphs, reduce='sum')

    def compute_LJ_energy(self, lj_envelope: Optional[float] = None, **kwargs):
        """Per-graph 12-6 Lennard-Jones energy over the intermolecular pair list.

        Pair energy ``4 * ((s / r)**12 - (s / r)**6)`` with ``s`` the sum of the
        two atoms' van der Waals radii (Angstrom) and unit well depth, summed per
        graph over ``edges_dict['intermolecular_dist']``. Not scaled by
        ``lj_coeff``.

        Parameters
        ----------
        lj_envelope : float, optional
            Width (Angstrom) of a smooth C2 switch taking each pair energy to
            zero at the pair-list cutoff (see vdw_analysis.lj_cutoff_envelope);
            must lie in (0, cutoff]. None (default) leaves the energy
            unswitched.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs].

        Raises
        ------
        RuntimeError
            If ``edges_dict`` has not been built.
        NotImplementedError
            If ``self`` is not a Batch.
        """
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = lj_analysis(self.vdw_radii_tensor,
                              self.edges_dict,
                              self.num_graphs,
                              envelope=lj_envelope,
                              )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_qLJ_energy(self, lj_envelope: Optional[float] = None, **kwargs):
        """Per-graph LJ energy with a quadratic wall below the van der Waals radius sum.

        As ``compute_LJ_energy``, except that for ``r < s`` the pair energy is
        ``24 * ((r - s)**2 / s**2 - (r - s) / s)``. Not scaled by ``lj_coeff``.

        Parameters
        ----------
        lj_envelope : float, optional
            As in ``compute_LJ_energy``; None (default) = off.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs].

        Raises
        ------
        RuntimeError
            If ``edges_dict`` has not been built.
        NotImplementedError
            If ``self`` is not a Batch.
        """
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = qlj_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               envelope=lj_envelope,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_eLJ_energy(self, repulsion: Optional[float] = 1.0, lj_envelope: Optional[float] = None, **kwargs):
        """Per-graph eLJ energy, SCALED by the batch's own `lj_coeff`.

        One value per crystal graph: the sum, over the intermolecular pair list
        ``edges_dict['intermolecular_dist']`` of the asymmetric unit (all Z'
        molecules when Z' > 1), of the eLJ pair energy. For ``r >= s``, with
        ``s`` the sum of the two atoms' van der Waals radii (Angstrom), the pair
        energy is the 12-6 LJ ``4 * ((s / r)**12 - (s / r)**6)`` with unit well
        depth; for ``r < s`` it is an exponential wall matching that LJ value
        and slope at ``s``, with stiffness ``2.5 * repulsion``
        (``vdw_analysis.elj_analysis``). The sum is then multiplied by
        ``lj_coeff`` (``_lj_coeff_for``).

        The coefficient rides on the data as a per-graph attribute rather than
        being passed at each call site or applied by each consumer. A bare eLJ
        sum is meaningless in scale -- it has to be multiplied by the calibration
        factor before it means anything -- and the two alternatives both fail in
        practice: passing it per call is forgettable at any one of many sites,
        and applying it per use leaves the same attribute holding different
        quantities depending on who last touched it.

        DEFAULTS TO 1.0 when the attribute is absent, so a batch built by any
        other consumer of this library behaves exactly as before. Callers that
        require the calibrated value are expected to assert the attribute's
        PRESENCE themselves -- permissive library, strict application.

        Parameters
        ----------
        repulsion : float, optional
            Multiplies the baseline wall stiffness 2.5.
        lj_envelope : float, optional
            Width (Angstrom) of a smooth C2 switch taking each pair energy to
            zero at the pair-list cutoff (see vdw_analysis.lj_cutoff_envelope);
            None (default) leaves it unswitched.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs] eLJ energy times ``lj_coeff``.

        Raises
        ------
        RuntimeError
            If ``edges_dict`` has not been built.
        NotImplementedError
            If ``self`` is not a Batch.
        ValueError
            If ``lj_coeff`` is a tensor with neither 1 nor num_graphs entries.
        """
        self._pre_compute_checks()
        stiffness = repulsion * 2.5  # baseline value is 2.5
        if self.is_batch:
            molwise_lj_pot \
                = elj_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               stiffness=stiffness,
                               envelope=lj_envelope,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot * self._lj_coeff_for(molwise_lj_pot)

    def compute_exp6_energy(self,
                            exp6_params: str = EXP6_DEFAULT_PARAMS,
                            exp6_charges=EXP6_DEFAULT_CHARGES,
                            exp6_h_shift: Optional[float] = None,
                            exp6_tail: bool = True,
                            exp6_envelope: Optional[float] = None,
                            **kwargs):
        """Typed exp-6 lattice energy per molecule, kJ/mol, of a cluster batch built by ``analyze`` for ``'exp6'``.

        The key is the exp-6 field and its point charges as one package: every pair is evaluated in the
        ``'exp6'`` form of ``exp6_ff`` (Buckingham, continued linearly below its inner inflection point), never
        in the module's ``'elj'`` form. That form's wall below each pair's sigma is finite at contact, and an
        attractive pair of point charges, clamped only at ``exp6_ff.COULOMB_R_MIN``, can pull through it. The
        eLJ comparison is the ``'elj'`` compute key.

        Calls ``exp6_ff.cluster_exp6_energy`` on ``self`` and its ``edges_dict``. ``analyze`` builds that cluster
        from ``exp6_ff.with_ff_columns(batch, exp6_params, exp6_charges)``, so ``x`` holds three columns
        (partial charge, force-field type, offset from each H to its bonded atom); ``exp6_params`` and
        ``exp6_charges`` must therefore be the values that cluster was built with, which ``analyze`` ensures by
        reading both from the same kwargs. The pair list is ``construct_radial_graph``'s, at ``analyze``'s
        ``cutoff`` and its neighbour cap, as for ``'elj'``. Not scaled by ``lj_coeff``.

        Parameters
        ----------
        exp6_params : str, optional
            ``'fit'`` or ``'w99'`` (``exp6_ff.PARAMETER_SETS``); default ``EXP6_DEFAULT_PARAMS``.
        exp6_charges : str, tensor or None, optional
            As ``exp6_ff.crystal_exp6_energy``'s ``charges``: ``'mmff'``, ``'gasteiger'``, ``'x'``, a tensor, or
            None for no Coulomb term; default ``EXP6_DEFAULT_CHARGES``.
        exp6_h_shift, exp6_tail, exp6_envelope
            ``h_shift``, ``tail`` and ``envelope`` of ``exp6_ff.cluster_exp6_energy``.
        **kwargs
            Ignored, as ``analyze`` forwards every keyword to every compute, except ``exp6_form`` and
            ``exp6_k_factor``, which raise: the pair form is fixed.

        Returns
        -------
        torch.Tensor
            [num_graphs] lattice energy per molecule, kJ/mol.

        Raises
        ------
        RuntimeError
            If ``edges_dict`` has not been built, or ``x`` lacks the three force-field columns (a cluster not built
            through ``analyze``'s ``'exp6'`` path would otherwise be scored with every atom as type 0).
        ValueError
            If ``exp6_form`` or ``exp6_k_factor`` is passed.
        """
        from mxtaltools.analysis.exp6_ff import cluster_exp6_energy
        fixed = [k for k in ('exp6_form', 'exp6_k_factor') if k in kwargs]
        if fixed:
            raise ValueError(f"the 'exp6' key always evaluates the exp-6 pair form with its charges; {fixed} cannot "
                             f"be set (the eLJ comparison is the 'elj' key)")
        self._pre_compute_checks()
        if self.x is None or self.x.ndim != 2 or self.x.shape[1] != 3:
            raise RuntimeError("compute_exp6_energy needs the force-field columns in x; request 'exp6' through "
                               "analyze, which builds the cluster from exp6_ff.with_ff_columns")
        return cluster_exp6_energy(self, self.edges_dict, exp6_params, coulomb=exp6_charges is not None,
                                   form='exp6', h_shift=exp6_h_shift, tail=exp6_tail, envelope=exp6_envelope)

    def _lj_coeff_for(self, like: torch.Tensor):
        """The per-graph `lj_coeff` broadcast against a molwise quantity, or 1.0.

        Kept separate so every energy that should carry the calibration reads it
        one way. Shape is checked rather than assumed: a scalar is fine, and so
        is one value per graph, but anything else is a silently wrong broadcast
        (a per-ATOM tensor would multiply the wrong axis and still produce
        finite, plausible numbers), so it raises instead.
        """
        coeff = getattr(self, 'lj_coeff', None)
        if coeff is None:
            return 1.0
        if not torch.is_tensor(coeff):
            return float(coeff)
        coeff = coeff.to(device=like.device, dtype=like.dtype).flatten()
        if coeff.numel() == 1:
            return coeff.squeeze()
        if coeff.numel() != like.numel():
            raise ValueError(
                f'lj_coeff has {coeff.numel()} entries against {like.numel()} '
                f'graphs -- it must be a scalar or one value per graph, or the '
                f'multiply silently broadcasts along the wrong axis.')
        return coeff

    def compute_vdW_overlap(self, **kwargs):
        """Per-graph summed van der Waals overlap, ``relu(s - r)`` over the intermolecular pair list.

        ``s`` is the sum of the two atoms' van der Waals radii and ``r`` their
        distance (Angstrom). kwargs are accepted and ignored. Returns
        [num_graphs]. Raises RuntimeError if ``edges_dict`` has not been built
        and NotImplementedError if ``self`` is not a Batch.
        """
        self._pre_compute_checks()

        if self.is_batch:
            molwise_vdw_overlap \
                = vdW_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               reduce='sum'
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_vdw_overlap

    def compute_max_vdW_overlap(self, **kwargs):
        """Per-graph maximum of ``relu(s - r)`` over the intermolecular pair list; otherwise as ``compute_vdW_overlap``."""
        self._pre_compute_checks()

        if self.is_batch:
            molwise_vdw_overlap \
                = vdW_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               reduce='max'
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_vdw_overlap

    def compute_ES_energy(self, **kwargs):
        """Per-graph screened (Yukawa) electrostatic energy over the intermolecular pair list.

        Pair energy ``q_i * q_j / r * exp(-0.92 * r)`` with the charges from
        ``edges_dict['intermolecular_partial_charges']`` (``x[:, 0]``, or ``x``
        if 1-D), which exists only if ``x`` had one row per atom when the pair
        list was built; otherwise KeyError. kwargs are accepted and ignored. Returns
        [num_graphs]. Raises RuntimeError if ``edges_dict`` has not been built
        and NotImplementedError if ``self`` is not a Batch.
        """
        self._pre_compute_checks()

        if self.is_batch:
            molwise_estat_energy = electrostatic_analysis(
                self.edges_dict,
                self.num_graphs)

        else:
            raise NotImplementedError("ES energies not implemented for single crystals")

        return molwise_estat_energy

    def compute_buckingham_energy(self, **kwargs):
        """Per-graph Buckingham-form energy in radius-normalised distance over the intermolecular pair list.

        Pair energy ``A * exp(-B * r / s) - C * (s / r)**6`` with
        ``vdw_analysis.buckingham_energy``'s defaults A = -3.2619, B = 1.0,
        C = -0.2 and ``s`` the van der Waals radius sum. kwargs are accepted and
        ignored. Returns [num_graphs]. Raises RuntimeError if ``edges_dict`` has
        not been built and NotImplementedError if ``self`` is not a Batch.
        """
        self._pre_compute_checks()

        if self.is_batch:
            molwise_buckingham_energy = buckingham_energy(
                self.edges_dict,
                self.num_graphs,
                self.vdw_radii_tensor
            )

        else:
            raise NotImplementedError("BH energies not implemented for single crystals")

        return molwise_buckingham_energy

    def compute_silu_energy(self,
                            repulsion: float = 1.0,
                            **kwargs):
        """Per-graph SiLU pair energy over the intermolecular pair list.

        Pair energy ``silu(-4 * repulsion * (r - repulsion * s)) / 0.28`` with
        ``s`` the van der Waals radius sum. kwargs are accepted and ignored.
        Returns [num_graphs]. Raises RuntimeError if ``edges_dict`` has not been
        built and NotImplementedError if ``self`` is not a Batch.
        """
        self._pre_compute_checks()

        if self.is_batch:
            molwise_silu_energy = silu_energy(
                self.edges_dict,
                self.num_graphs,
                self.vdw_radii_tensor,
                repulsion=repulsion,
            )
        else:
            raise NotImplementedError("SiLU energies not implemented for single crystals")

        return molwise_silu_energy

    def build_and_analyze(self,
                          return_cluster: Optional[bool] = False,
                          noise: Optional[float] = None,
                          cutoff: float = 6,
                          supercell_size: int = 10,
                          ):
        """Build a cluster and compute its LJ and electrostatic energies; superseded by ``analyze``.

        Older fixed-request form of ``analyze``, which should be used instead.
        Steps: ``cluster = self.mol2cluster(cutoff, supercell_size)`` (so
        ``std_orientation=True``, the ``mol2cluster`` default); Gaussian noise
        of standard deviation ``noise`` added to the cluster ``pos`` if given;
        ``cluster.construct_radial_graph(cutoff=cutoff)``;
        ``cluster.compute(requests=['lj', 'es'])`` with no extra kwargs. Unlike
        ``analyze``, it does not force-rebind the cluster's ``computes``, so a
        ``computes`` dict the cluster already carries is used as is.

        Assigns ``self.lj``, ``self.es_pot`` and
        ``self.scaled_lj = log_rescale_positive(self.lj)`` (``log1p`` of values
        above 0, others unchanged) as plain attributes.

        Parameters
        ----------
        return_cluster : bool, optional
            If True, also return the cluster batch.
        noise : float, optional
            Standard deviation of Gaussian noise added to every cluster atom
            position (Angstrom); None adds none.
        cutoff : float, optional
            Radius passed to ``mol2cluster`` and ``construct_radial_graph``.
        supercell_size : int, optional
            Passed to ``mol2cluster``.

        Returns
        -------
        tuple
            ``(lj, es_pot, scaled_lj)``, each [num_graphs]; with
            ``return_cluster``, ``(lj, es_pot, scaled_lj, cluster_batch)``.
        """

        cluster_batch = self.mol2cluster(cutoff, supercell_size)

        if noise is not None:
            cluster_batch.pos += torch.randn_like(cluster_batch.pos) * noise

        cluster_batch.construct_radial_graph(cutoff=cutoff)

        outputs = cluster_batch.compute(requests=['lj', 'es'])
        self.lj = outputs['lj']
        self.es_pot = outputs['es']
        self.scaled_lj = log_rescale_positive(self.lj)

        if return_cluster:
            return self.lj, self.es_pot, self.scaled_lj, cluster_batch
        else:
            return self.lj, self.es_pot, self.scaled_lj

    def analyze(self,
                computes: list,
                return_cluster: Optional[bool] = False,
                noise: Optional[float] = None,
                cutoff: float = 10,
                supercell_size: int = 10,
                std_orientation: Optional[bool] = True,
                assign_outputs: Optional[bool] = False,
                **kwargs
                ):
        """Build the structure the requested computes need, then evaluate them.

        This is the shared dispatch surface for every analysis and energy in
        this class. Each key in ``computes`` is looked up (exactly as passed;
        absent keys count as False) in ``COMPUTES_REQUIRE_CLUSTER`` and
        ``COMPUTES_REQUIRE_UNIT_CELL``, which selects one of three branches:

        1. Any key needs a cluster: ``self.mol2cluster(cutoff, supercell_size,
           std_orientation=std_orientation)`` returns a new cluster batch
           (``aux_ind`` 0 marks the asymmetric unit, 1 the surrounding image
           atoms within range, 2 image atoms outside it; it also carries
           ``unit_cell_pos``); ``noise`` is added to its
           ``pos`` (not to ``unit_cell_pos``); then
           ``construct_radial_graph(cutoff=cutoff)`` builds its
           ``edges_dict``. Every requested key, including those that need no
           cluster, is evaluated on this cluster batch. A crystal with no
           defined cluster (a non-finite or singular ``T_fc``, a non-finite
           ``radius``, or a non-finite position in its cluster, as from a
           non-finite pose) gets NaN in every floating-point per-graph output,
           the RDF included, instead of stopping the batch.
        2. Otherwise, any key needs a unit cell: a ``self.clone()`` gets
           ``mol2ucell(std_orientation=std_orientation)``; ``noise`` is then
           added to its ``pos``, after ``unit_cell_pos`` has been built, so
           ``unit_cell_pos`` is not perturbed.
        3. Otherwise ``self`` itself is evaluated and ``noise`` is ignored.

        The evaluated batch's computes are then rebound to it
        (``_init_computes(override=True)``) and
        ``compute(requests=computes, **kwargs)`` runs.

        Parameters
        ----------
        computes : list of str
            Compute keys; see Notes. Dispatch in ``compute`` lower-cases each
            key, but the requirement lookup above does not, so pass lower-case
            keys.
        return_cluster : bool, optional
            If True, also return the batch that was evaluated.
        noise : float, optional
            Standard deviation of Gaussian noise added to the evaluated batch's
            ``pos`` in branches 1 and 2 (Angstrom); None adds none.
        cutoff : float, optional
            Radius (Angstrom) passed to ``mol2cluster`` and
            ``construct_radial_graph`` in branch 1. Not forwarded to the
            compute functions.
        supercell_size : int, optional
            Passed to ``mol2cluster`` in branch 1.
        std_orientation : bool, optional
            Passed to ``mol2cluster`` (branch 1) or ``mol2ucell`` (branch 2).
            Not forwarded to the compute functions, whose own
            ``std_orientation`` arguments keep their defaults.
        assign_outputs : bool, optional
            If True, each result is attached to ``self`` (not to the evaluated
            batch) with ``add_graph_attr`` under its key; for ``'rdf'`` the RDF
            tensor is stored as ``rdf`` and the bin positions, repeated to
            [num_graphs, bins], as ``rdf_bins``. Requires ``self`` to be a
            Batch and each stored value to have leading dimension num_graphs.
        **kwargs
            Forwarded unchanged to every requested compute function via
            ``compute``. ``compute_crystal_uma``, ``compute_crystal_mace`` and
            ``compute_ellipsoid_embedding`` accept no extra kwargs and raise
            TypeError on any they do not name.

        Returns
        -------
        results : dict
            ``{key: output}`` for each key in ``computes``.
        batch_to_analyze : MolCrystalData batch
            Only if ``return_cluster``: the cluster (branch 1), the unit-cell
            clone (branch 2) or ``self`` (branch 3).

        Raises
        ------
        KeyError
            If a key is not a compute key.
        TypeError
            If a compute function receives a keyword it does not accept or
            lacks a required one (``predictor``; ``cell_lengths`` and
            ``cell_angles`` for ``'niggli_overlap'``), or ``assign_outputs`` is
            True and ``self`` is not a Batch.

        Notes
        -----
        Accepted keys, the method each dispatches to, what it requires and what
        it returns (tensors are per graph unless stated):

        - ``'exp6'``: ``compute_exp6_energy``; cluster, built from
          ``exp6_ff.with_ff_columns(self, exp6_params, exp6_charges)`` (kwargs,
          defaults ``EXP6_DEFAULT_PARAMS`` and ``EXP6_DEFAULT_CHARGES``) so that
          ``x`` carries the force-field columns for every compute of the call;
          [num_graphs] lattice energy per molecule in kJ/mol, always in the
          exp-6 pair form with its point charges.
        - ``'elj'``: ``compute_eLJ_energy``; cluster; [num_graphs] eLJ energy
          times ``lj_coeff``.
        - ``'lj'``, ``'qlj'``: ``compute_LJ_energy``, ``compute_qLJ_energy``;
          cluster; [num_graphs].
        - ``'es'``: ``compute_ES_energy``; cluster with per-atom charges in
          ``x``; [num_graphs].
        - ``'bh'``, ``'silu'``: ``compute_buckingham_energy``,
          ``compute_silu_energy``; cluster; [num_graphs].
        - ``'vdw'``, ``'vdw_max'``: ``compute_vdW_overlap``,
          ``compute_max_vdW_overlap``; cluster; [num_graphs].
        - ``'ellipsoid'``: ``compute_ellipsoidal_overlap`` (ellipsoid mixin);
          cluster; [num_graphs] by default.
        - ``'ellipsoid_emb'``: ``compute_ellipsoid_embedding``; cluster;
          [num_graphs, num_features] when its scatter succeeds (see that
          method).
        - ``'rdf'``: ``compute_rdf``; cluster; tuple ``(rdf, bin_positions,
          pair_dict)``.
        - ``'reduction_en'``: ``compute_cell_reduction_penalty``; neither;
          [num_graphs] from the cell parameters and ``sg_ind``.
        - ``'niggli_overlap'``: ``sym_utils.niggli_reduction_penalty``, a
          module function called as ``niggli_reduction_penalty(**kwargs)``, so
          ``cell_lengths`` and ``cell_angles`` must be passed as kwargs; neither;
          [num_graphs].
        - ``'uma_pot'``, ``'mace_pot'``: ``compute_crystal_uma``,
          ``compute_crystal_mace``; unit cell; [num_graphs] predictor energy of
          the periodic unit cell. Need the ``predictor`` kwarg.
        - ``'uma'``, ``'mace'``: ``compute_lattice_uma``,
          ``compute_lattice_mace``; unit cell; [num_graphs] float32 lattice
          energy per molecule in kJ/mol, with the predictor energies taken as
          eV. Need ``predictor``.
        - ``'uma_gas_pot'``, ``'mace_gas_pot'``:
          ``compute_lattice_gas_phase_uma``, ``compute_lattice_gas_phase_mace``;
          neither (they build their own P1 cell); [num_graphs] isolated-molecule
          predictor energy. Need ``predictor``.
        - ``'latent_harmonic'``, ``'latent_gaussian'``: both
          ``latent_harmonic_en``; neither; [num_graphs].
        - ``'latent_multiharmonic'``: ``latent_multiharmonic_en``; neither;
          [num_graphs].

        ELJ route. ``analyze(['reduction_en', 'elj'])`` takes branch 1 because
        ``'elj'`` needs the cluster: ``mol2cluster`` ->
        ``construct_radial_graph`` -> ``get_intermolecular_dists_dict`` ->
        ``build_radial_graph`` (pairs between asymmetric-unit atoms and atoms of
        other molecules within ``cutoff``) -> ``compute_eLJ_energy`` ->
        ``elj_analysis`` summed per graph and multiplied by ``lj_coeff``.
        ``'reduction_en'`` is evaluated on the same cluster batch, which carries
        the cell parameters and ``sg_ind`` of ``self``.

        Side effects on ``self``: ``computes``, ``computes_requires_cluster``
        and ``computes_requires_unit_cell`` are set if ``self.computes`` is
        absent, and rebound in branch 3; ``assign_outputs`` adds graph
        attributes. In branch 3 the compute functions run on ``self`` and may
        write to it (see ``latent_harmonic_en``).
        """
        if not hasattr(self, 'computes'):
            self._init_computes()
        requires_cluster = any(self.computes_requires_cluster.get(k, False) for k in computes)
        requires_unit_cell = any(self.computes_requires_unit_cell.get(k, False) for k in computes)
        if requires_cluster:
            source = self
            if 'exp6' in computes:
                # the typed force field reads its atom types, H partners and charges from x, which mol2cluster copies
                # to every image: build the one shared cluster from the batch with those columns written
                from mxtaltools.analysis.exp6_ff import with_ff_columns
                source = with_ff_columns(self, kwargs.get('exp6_params', EXP6_DEFAULT_PARAMS),
                                         kwargs.get('exp6_charges', EXP6_DEFAULT_CHARGES))
            batch_to_analyze = source.mol2cluster(
                cutoff, supercell_size,
                std_orientation=std_orientation)
            if noise is not None:
                batch_to_analyze.pos += torch.randn_like(batch_to_analyze.pos) * noise

            batch_to_analyze.construct_radial_graph(cutoff=cutoff)
            batch_to_analyze._init_computes(override=True)
        elif requires_unit_cell:
            # e.g. mace/uma alone: these only ever read unit_cell_pos/sym_mult/T_fc
            # (plus z/batch, un-filtered since aux_ind stays None here) - never
            # edges_dict or periodic images - so build just the home unit cell
            # instead of the full periodic supercell mol2cluster produces.
            batch_to_analyze = self.clone()
            batch_to_analyze.mol2ucell(std_orientation=std_orientation)
            if noise is not None:
                batch_to_analyze.pos += torch.randn_like(batch_to_analyze.pos) * noise

            batch_to_analyze._init_computes(override=True)
        else:
            batch_to_analyze = self
            # NOTE: must force-rebind here too, else a `computes` dict of bound
            # methods inherited via .clone() from an earlier-analyzed ancestor
            # (still bound to that ancestor's tensors) gets silently reused,
            # producing results shaped for the wrong batch.
            batch_to_analyze._init_computes(override=True)

        results = batch_to_analyze.compute(requests=computes, **kwargs)
        if requires_cluster:
            results = self._nan_undefined_crystals(results, batch_to_analyze)

        if assign_outputs:
            for key, value in results.items():
                if key != 'rdf':
                    self.add_graph_attr(value, key)
                else:
                    self.add_graph_attr(value[0], key)
                    self.add_graph_attr(value[1].repeat(self.num_graphs, 1), 'rdf_bins')

        if return_cluster:
            return results, batch_to_analyze
        else:
            return results

    def _nan_undefined_crystals(self, results, cluster):
        """NaN, in every floating-point per-graph output (the RDF's first element for 'rdf'), for the crystals of
        ``self`` with no defined cluster: a non-finite or singular ``T_fc``, a non-finite ``radius``, or a non-finite
        position anywhere in the crystal's cluster (as from a non-finite pose). get_cart_translations gives such
        crystals their untranslated cell only, so their outputs would otherwise be finite and wrong (eLJ 0 for a NaN
        pose, since NaN distances make no edges)."""
        n = self.num_graphs
        undefined = ~torch.isfinite(self.T_fc.reshape(n, 9)).all(1) | ~torch.isfinite(self.radius.reshape(n))
        undefined = undefined | (torch.linalg.det(self.T_fc.detach().double()).abs() <= 0)
        bad_pos = (~torch.isfinite(cluster.pos).all(1)).long()
        undefined = undefined | (torch.zeros(n, dtype=torch.long, device=bad_pos.device)
                                 .index_add_(0, cluster.batch, bad_pos) > 0)
        if not bool(undefined.any()):
            return results

        def nan_rows(value):
            if not (torch.is_tensor(value) and value.is_floating_point() and value.dim() >= 1 and value.shape[0] == n):
                return value
            mask = undefined.to(value.device).reshape(-1, *([1] * (value.dim() - 1)))
            return torch.where(mask, torch.full_like(value, float('nan')), value)

        return {key: ((nan_rows(value[0]),) + tuple(value[1:]) if key == 'rdf' else nan_rows(value))
                for key, value in results.items()}

    def compute_crystal_uma(self,
                            predictor,
                            std_orientation: bool = True,
                            ):
        """UMA energy of each crystal's periodic unit cell.

        Passes a clone of the batch (for a single object,
        ``collate_data_list(self)``, which drops ``unit_cell_pos``) to
        ``compute_crystal_uma_on_mxt_batch(copy, std_orientation, predictor)``
        with ``pbc=True``; ``self`` is not modified. That adapter uses the
        copy's ``unit_cell_pos`` when present and otherwise builds the unit cell
        on the copy with ``std_orientation``; before building its input it adds
        2 to the cell lengths of any crystal whose ``packing_coeff`` exceeds
        2.0, repeatedly, on the copy; and it returns NaN for every graph when
        the prediction fails with a non-out-of-memory RuntimeError (an
        out-of-memory error is re-raised).

        Parameters
        ----------
        predictor
            UMA predictor passed to the adapter.
        std_orientation : bool, optional
            Used only if the unit cell has to be built.

        Returns
        -------
        torch.Tensor
            [num_graphs] ``energy`` output of the predictor, one value per unit
            cell; ``compute_lattice_uma`` treats it as eV.
        """
        if self.is_batch:
            return compute_crystal_uma_on_mxt_batch(self.clone(),
                                                    std_orientation,
                                                    predictor)
        else:
            return compute_crystal_uma_on_mxt_batch(collate_data_list(self),
                                                    std_orientation,
                                                    predictor)

    def compute_lattice_uma(self,
                            predictor,
                            # was `std_orienation` (missing t), so a caller passing
                            # std_orientation=... had it swallowed by **kwargs and BOTH legs ran at
                            # the default True. The GFN's analyze_kwargs asks for False, which means
                            # every uma run so far scored std-oriented crystals while the mace/elj
                            # runs did not -- the two were not on the same convention.
                            std_orientation: bool = True,
                            **kwargs):
        """UMA lattice energy per molecule: crystal energy per molecule minus the isolated-molecule energy.

        Uses ``self.uma_gas_pot`` and ``self.uma_pot`` when they already exist,
        without checking them against the current geometry, and otherwise
        computes and stores them on ``self`` with ``add_graph_attr`` (so
        ``self`` must be a Batch): ``uma_gas_pot`` from
        ``compute_lattice_gas_phase_uma(predictor)`` (its default
        ``std_orientation=False``), ``uma_pot`` from
        ``compute_crystal_uma(predictor, std_orientation=std_orientation)``.

        Parameters
        ----------
        predictor
            UMA predictor.
        std_orientation : bool, optional
            Passed to ``compute_crystal_uma``, where it matters only if the unit
            cell has to be built. ``analyze`` does not forward its own
            ``std_orientation`` here.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs] float32,
            ``(uma_pot / (sym_mult * z_prime) - uma_gas_pot) * 96.485``, i.e.
            kJ/mol per molecule with the predictor energies in eV.
        """
        if not hasattr(self, 'uma_gas_pot'):
            self.add_graph_attr(
                self.compute_lattice_gas_phase_uma(predictor),
                'uma_gas_pot')

        if not hasattr(self, 'uma_pot'):
            self.add_graph_attr(
                self.compute_crystal_uma(predictor, std_orientation=std_orientation),
                'uma_pot')

        # lattice energy per molecule, in eV
        # multiply by 96.485 to get kJ/mol
        return ((self.uma_pot / (self.sym_mult * self.z_prime) - self.uma_gas_pot) * 96.485).float()

    def compute_lattice_gas_phase_uma(self,
                                      predictor,
                                      # off by default for the same reason as the mace twin above:
                                      # isolated molecule, rotation-invariant energy, and the only
                                      # thing on this path that can throw in linalg.eigh
                                      std_orientation: bool = False,
                                      **kwargs,
                                      ):
        """UMA energy of each crystal's asymmetric-unit molecule(s) evaluated without periodic boundaries.

        Works on a clone (for a single object, ``collate_data_list(self)``);
        ``self`` is not modified. If ``self`` is a cluster (``aux_ind`` set), the
        copy's ``pos``, ``batch`` and ``z`` are replaced by the ``aux_ind == 0``
        atoms. The copy is reset to space group 1 (``reset_sg_info(sg_ind=1)``;
        cell lengths and angles unchanged), ``box_analysis()`` is rerun, and
        ``compute_crystal_uma_on_mxt_batch`` is called with ``pbc=False`` and
        ``force_rebuild=True``, so a fresh P1 cell holding one molecule is built
        with ``std_orientation`` and any existing unit cell is ignored. When any
        graph has Z' > 1 the copy is first split into one graph per molecule
        (``split_to_zp1_batch``) and the per-molecule energies are averaged back
        to one value per crystal.

        Parameters
        ----------
        predictor
            UMA predictor.
        std_orientation : bool, optional
            Passed to the adapter for posing the molecule.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs] predictor energy per molecule (mean over the Z'
            molecules).
        """

        if self.is_batch:
            diffuse_batch = self.clone()
        else:
            diffuse_batch = collate_data_list(self)

        if hasattr(self, 'aux_ind'):
            if self.aux_ind is not None:  # if this is a cluster, we have to rebuild the original aunit batch
                diffuse_batch.pos = self.pos[self.aux_ind == 0].detach().clone()
                diffuse_batch.batch = self.batch[self.aux_ind == 0].detach().clone()
                diffuse_batch.z = self.z[self.aux_ind == 0].detach().clone()

        if diffuse_batch.z_prime.amax() > 1:  # do it separately per conformer
            zp1_batch = diffuse_batch.split_to_zp1_batch()
            zp1_batch.reset_sg_info(sg_ind=1)  # big P1 cells
            # zp1_batch.cell_lengths *= 100
            zp1_batch.box_analysis()
            split_ens = compute_crystal_uma_on_mxt_batch(zp1_batch,
                                                         std_orientation,
                                                         predictor,
                                                         pbc=False,
                                                         force_rebuild=True)
            zp_inds = torch.arange(self.num_graphs, dtype=torch.long, device=self.device
                                   ).repeat_interleave(self.z_prime).to(split_ens.device)
            return scatter(split_ens, zp_inds, reduce='mean', dim=0, dim_size=self.num_graphs)

        else:
            # evaluate a diffuse P1 5ell
            diffuse_batch.reset_sg_info(sg_ind=1)  # big P1 cells
            # diffuse_batch.cell_lengths *= 100
            diffuse_batch.box_analysis()
            return compute_crystal_uma_on_mxt_batch(diffuse_batch,
                                                    std_orientation,
                                                    predictor,
                                                    pbc=False,
                                                    force_rebuild=True
                                                    # we don't want to inherit real unit cells so we force rebuild them here
                                                    )

    def compute_crystal_mace(self,
                             predictor,
                             std_orientation: bool = True,
                             ):
        """MACE energy of each crystal's periodic unit cell.

        Passes a clone of the batch (for a single object,
        ``collate_data_list(self)``, which drops ``unit_cell_pos``) to
        ``compute_crystal_mace_on_mxt_batch(copy, predictor,
        std_orientation=std_orientation)`` with its default ``pbc=True``;
        ``self`` is not modified. That adapter uses the copy's
        ``unit_cell_pos`` when present and otherwise builds the unit cell on the
        copy with ``std_orientation``, and returns NaN for every graph when the
        forward raises a non-out-of-memory RuntimeError (an out-of-memory error
        is re-raised).

        Parameters
        ----------
        predictor
            MACE model passed to the adapter.
        std_orientation : bool, optional
            Used only if the unit cell has to be built.

        Returns
        -------
        torch.Tensor
            [num_graphs] ``energy`` output of the model, one value per unit
            cell; ``compute_lattice_mace`` treats it as eV.
        """
        if self.is_batch:
            return compute_crystal_mace_on_mxt_batch(self.clone(),
                                                     predictor,
                                                     std_orientation=std_orientation)
        else:
            return compute_crystal_mace_on_mxt_batch(collate_data_list(self),
                                                     predictor,
                                                     std_orientation=std_orientation)

    def compute_lattice_mace(self,
                             predictor,
                             std_orientation: bool = True,
                             **kwargs):
        """MACE lattice energy per molecule: crystal energy per molecule minus the isolated-molecule energy.

        Uses ``self.mace_gas_pot`` and ``self.mace_pot`` when they already
        exist, without checking them against the current geometry, and
        otherwise computes and stores them on ``self`` with ``add_graph_attr``
        (so ``self`` must be a Batch): ``mace_gas_pot`` from
        ``compute_lattice_gas_phase_mace(predictor)`` (its default
        ``std_orientation=False``), ``mace_pot`` from
        ``compute_crystal_mace(predictor, std_orientation=std_orientation)``.

        Parameters
        ----------
        predictor
            MACE model.
        std_orientation : bool, optional
            Passed to ``compute_crystal_mace``, where it matters only if the
            unit cell has to be built. ``analyze`` does not forward its own
            ``std_orientation`` here.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs] float32,
            ``(mace_pot / (sym_mult * z_prime) - mace_gas_pot) * 96.485``, i.e.
            kJ/mol per molecule with the model energies in eV.
        """
        if not hasattr(self, 'mace_gas_pot'):
            self.add_graph_attr(
                self.compute_lattice_gas_phase_mace(predictor),
                'mace_gas_pot')

        if not hasattr(self, 'mace_pot'):
            self.add_graph_attr(
                self.compute_crystal_mace(predictor, std_orientation=std_orientation),
                'mace_pot')

        return ((self.mace_pot / (self.sym_mult * self.z_prime) - self.mace_gas_pot) * 96.485).float()

    def compute_lattice_gas_phase_mace(self,
                                       predictor,
                                       # OFF by default: this leg scores an ISOLATED molecule
                                       # (pbc=False), and MACE is rotation-equivariant, so standard
                                       # orientation cannot change the number -- measured at 3e-5
                                       # kJ/mol against a reference that is constant to 4 decimals.
                                       # What it CAN do is throw: it is the only thing that puts
                                       # align_mol_batch_to_standard_axes -> scatter_compute_Ip ->
                                       # linalg.eigh on this path, and every June 2026 acridine
                                       # Z'=2 run died there at 18-27k steps ("failed to converge
                                       # ... ill-conditioned or too many repeated eigenvalues",
                                       # on the CPU fallback, so the input is genuinely bad).
                                       # scatter_compute_Ip only lifts degeneracies when
                                       # requires_grad or add_noise, and the energy path is
                                       # no-grad. Cost with no benefit; default it off.
                                       std_orientation: bool = False,
                                       **kwargs,
                                       ):
        """MACE energy of each crystal's asymmetric-unit molecule(s) evaluated without periodic boundaries.

        Works on a clone (for a single object, ``collate_data_list(self)``);
        ``self`` is not modified. If ``self`` is a cluster (``aux_ind`` set), the
        copy's ``pos``, ``batch`` and ``z`` are replaced by the ``aux_ind == 0``
        atoms. The copy is reset to space group 1 (``reset_sg_info(sg_ind=1)``;
        cell lengths and angles unchanged), ``box_analysis()`` is rerun, and
        ``compute_crystal_mace_on_mxt_batch`` is called with ``pbc=False`` and
        ``force_rebuild=True``, so a fresh P1 cell holding one molecule is built
        with ``std_orientation`` and any existing unit cell is ignored. When any
        graph has Z' > 1 the copy is first split into one graph per molecule
        (``split_to_zp1_batch``) and the per-molecule energies are averaged back
        to one value per crystal.

        Parameters
        ----------
        predictor
            MACE model.
        std_orientation : bool, optional
            Passed to the adapter for posing the molecule.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs] model energy per molecule (mean over the Z'
            molecules).
        """
        if self.is_batch:
            diffuse_batch = self.clone()
        else:
            diffuse_batch = collate_data_list(self)

        if hasattr(self, 'aux_ind'):
            if self.aux_ind is not None:
                mask = self.aux_ind == 0
                #
                # # --- diagnostic: every graph must retain at least one canonical-aunit atom ---
                # kept_per_graph = torch.bincount(self.batch[mask], minlength=self.num_graphs)
                # if (kept_per_graph == 0).any():
                #     bad_graphs = (kept_per_graph == 0).nonzero(as_tuple=True)[0]
                #     aux_unique_per_bad = [
                #         self.aux_ind[self.batch == g].unique(return_counts=True)
                #         for g in bad_graphs.tolist()
                #     ]
                #     raise AssertionError(
                #         f"aux_ind == 0 filter produced empty graphs: {bad_graphs.tolist()}\n"
                #         f"  kept_per_graph={kept_per_graph.tolist()}\n"
                #         f"  num_atoms={self.num_atoms.tolist()}\n"
                #         f"  aux_ind values per bad graph (val, count): "
                #         f"{[(v.tolist(), c.tolist()) for v, c in aux_unique_per_bad]}\n"
                #         f"  cell_lengths[bad]={self.cell_lengths[bad_graphs].tolist()}\n"
                #         f"  cell_angles[bad]={self.cell_angles[bad_graphs].tolist()}"
                #     )
                # # --- end diagnostic ---

                diffuse_batch.pos = self.pos[self.aux_ind == 0].detach().clone()
                diffuse_batch.batch = self.batch[self.aux_ind == 0].detach().clone()
                diffuse_batch.z = self.z[self.aux_ind == 0].detach().clone()

        if diffuse_batch.z_prime.amax() > 1:
            zp1_batch = diffuse_batch.split_to_zp1_batch()
            zp1_batch.reset_sg_info(sg_ind=1)
            # zp1_batch.cell_lengths *= 100
            zp1_batch.box_analysis()
            split_ens = compute_crystal_mace_on_mxt_batch(zp1_batch,
                                                          predictor,
                                                          std_orientation=std_orientation,
                                                          pbc=False,
                                                          force_rebuild=True)
            zp_inds = torch.arange(self.num_graphs, dtype=torch.long, device=self.device
                                   ).repeat_interleave(self.z_prime).to(split_ens.device)
            return scatter(split_ens, zp_inds, reduce='mean', dim=0, dim_size=self.num_graphs)

        else:
            diffuse_batch.reset_sg_info(sg_ind=1)
            # diffuse_batch.cell_lengths *= 100
            diffuse_batch.box_analysis()
            return compute_crystal_mace_on_mxt_batch(diffuse_batch,
                                                     predictor,
                                                     std_orientation=std_orientation,
                                                     pbc=False,
                                                     force_rebuild=True)

    def latent_harmonic_en(self, c: torch.Tensor = None, width: float = 0.1, x=None, **kwargs):
        """Isotropic harmonic energy of latent points about the mode ``c``.

        E(x) = 0.5 * ((x - mode) / width)^2.sum(-1)

        T-free harmonic energy. At target temperature T, the corresponding
        sampler has std = sqrt(T) * width.

        Parameters
        ----------
        c : torch.Tensor, optional
            The mode. A single condition ([d] or [1, d], broadcast over the
            whole batch) or a per-sample batch ([B, d], one condition per
            row -- each row scored against its own mode). None uses zeros
            [1, d].
        width : float or torch.Tensor, optional
            A scalar or a per-sample [B] tensor (mixed widths in one batch).
        x : torch.Tensor, optional
            [B, d] points to score. None uses
            ``self.latent_params(gauge_fix_free_axes=False)``, the latent cell
            parameters clipped to [-1, 1], [num_graphs, d]. That call does not
            gauge-fix free centroid axes, but it does reassign
            ``aunit_centroid``, ``aunit_orientation`` and ``aunit_handedness``
            on ``self`` in canonical Z' order (``canonicalize_zp_aunits``).
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [B] energy.

        Notes
        -----
        Log-partition function of exp(-E / T) over unbounded R^d::

            import numpy as np
            logZ = lambda T, d, w: (d/2) * np.log(2*np.pi*T) + d*np.log(w)
        """
        # gauge-FREE, and load-bearing: latent_params() default MUTATES the batch
        # (canonicalize_free_axes writes aunit_centroid), and this runs on
        # latent-energy batches whose centroid dims are REAL data (P1 2026-08-24).
        x = self.latent_params(gauge_fix_free_axes=False) if x is None else x
        d = x.shape[-1]

        if c is None:
            c = torch.zeros((1, d), device=x.device, dtype=x.dtype)
        else:
            c = torch.as_tensor(c, dtype=x.dtype, device=x.device)

        w = torch.as_tensor(width, dtype=x.dtype, device=x.device)
        if w.dim() == 1:
            w = w.unsqueeze(-1)  # [B] (per-sample) or [1] (scalar-as-tensor) -> broadcasts against x's last dim

        return 0.5 * (((x - c) / w) ** 2).sum(-1)

    def sample_latent_harmonic(
            self,
            n_samples: int,
            c: torch.Tensor = None,
            width: float = 0.1,
            target_temperature: float = 1.0,
            seed=None,
            **kwargs,
    ):
        """Exact sampler for the target defined by latent_harmonic_en.

        With

            E(x) = 0.5 * ((x - mode) / width)^2.sum(-1)

        at target temperature T,

            pi_T(x) ~ exp(-E(x) / T)
            x ~ N(mode, T * width^2 * I)

        Draws are ``c + sqrt(T) * width * eps`` with standard-normal ``eps``
        drawn on the CPU and moved to ``self.device``; they are not clipped to
        the [-1, 1] range of ``latent_params``. The dimension d is read from
        ``self.latent_params(gauge_fix_free_axes=False)`` (see
        ``latent_harmonic_en`` for what that call reassigns).

        Parameters
        ----------
        n_samples : int
            Number of draws.
        c : torch.Tensor, optional
            A single condition ([d] or [1, d], broadcast to every drawn sample)
            or a per-sample batch ([n_samples, d], one mode per row -- each row
            is centered on its own condition). None uses zeros [1, d].
        width : float or torch.Tensor, optional
            A scalar or a per-sample [n_samples] tensor.
        target_temperature : float, optional
            T.
        seed : int, optional
            Seed for the new CPU ``torch.Generator`` used for ``eps``. With
            None the generator is not seeded explicitly and starts from torch's
            fixed default seed for a new generator, so calls with None repeat
            the same draws.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [n_samples, d].

        Notes
        -----
        Old convention: scale = 1 / width.
        """
        # dimension read only -- must not mutate the batch (P1 2026-08-24)
        d = self.latent_params(gauge_fix_free_axes=False).shape[-1]

        if c is None:
            c = torch.zeros((1, d), device=self.device)
        else:
            c = torch.as_tensor(c, device=self.device)

        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)

        eps = torch.randn(n_samples, d, generator=g, device="cpu").to(self.device)

        w = torch.as_tensor(width, dtype=eps.dtype, device=self.device)
        if w.dim() == 1:
            w = w.unsqueeze(
                -1)  # [n_samples] (per-sample) or [1] (scalar-as-tensor) -> broadcasts against eps's last dim
        std = (float(target_temperature) ** 0.5) * w

        return c + std * eps

        # -----------------------------------------------------------------------
        # Conditional multi-well toy. FIXED landscape E(x; c, width); the sampling
        # temperature T is applied ONLY in the reward / sampler (not in the energy).
        #
        #   g(x; c, w) = sum_k what_k(c) * N(x; mu_k(c), (w * sigma_k(c))^2 I)   (unnorm.)
        #   E(x; c, w) = -log g(x; c, w)                         # T-FREE landscape
        #   pi_T(x)    = exp(-E / T) / Z(T,c,w) = g^{1/T} / Z    # target at temperature T
        #
        #   Z(1, c, w) = sum_k what_k(c)     closed form; == 1 ONLY at reference c (c=0)
        #   Z(T, c, w) = ∫ g^{1/T}           T!=1: low-variance IS, tempered-GMM proposal
        #
        # Knobs (all smooth, exactly-sampleable per (c, w) at T=1):
        #   width  w : scales every basin's std. Preserves mass ratios pi_k AND Z(1).
        #              => the honest "sigma" knob; safe to read calibration mid-anneal.
        #   temp   T : Boltzmann sharpness. Flattens/sharpens mass ratios, moves Z.
        #              => corrupts pi_k mid-anneal; only T=1 mass ratios are trustworthy.
        #   cond   c : arbitrary-dim vector; smoothly moves positions, widths, depths,
        #              and (via sigmoid gates on "ghost" modes) the number of active wells.
        #              Z(c) grows as ghosts switch on (adding a mode adds mass).
        #
        # Usage:
        #   self._build_latent_field(cond_dim=8, seed=0)                 # once
        #   c = torch.zeros(8)                                           # reference state
        #   E   = self.latent_multiharmonic_en(c=c, width=1.0)          # energy on latent_params()
        #   lr  = self.latent_log_reward(c=c, target_temperature=T, width=w)   # log g^{1/T}
        #   x   = self.sample_latent_multiharmonic(4096, c=c, target_temperature=T, width=w)  # SIR -> true target (default)
        #   x   = self.sample_latent_multiharmonic(4096, c=c, ..., exact=False) # fast, approximate proposal q_T
        #   lZ  = self.log_partition_latent(c=c, target_temperature=T, width=w)  # ground-truth logZ
        # -----------------------------------------------------------------------

    def _build_latent_field(
            self,
            cond_dim: int = 2,
            n_core: int = 16,
            n_ghost: int = 16,
            sigma_range: tuple = (0.04, 0.12),
            aniso_scale: float = 0.0,  # NEW: spread of per-axis log-eigstd around base
            depth_range: tuple = (0.0, 4.0),  # (0.0, 4.0),
            mu_scale: float = 0.7,  # NEW: <1 contracts basin centers toward 0 (tighter field)
            disp_max: float = 0.15,
            logsig_scale: float = 0.5,
            gate_steep: float = 4.0,
            edge_sigmas: float = 0.1,
            max_temperature: float = 3.0,
            max_width: float = 2.5,
            seed: int = 0,
    ):
        """Build & cache the conditional GMM field, now with per-mode dense (rotated)
        covariance Sigma_k = R_k diag(eigstd_k)^2 R_k^T. Deterministic in `seed`."""
        import math

        g = torch.Generator(device="cpu").manual_seed(seed)
        K = n_core + n_ghost
        # dimension read only -- must not mutate the batch (P1 2026-08-24)
        d = self.latent_params(gauge_fix_free_axes=False).shape[-1]

        base_log_sigma = torch.empty(K).uniform_(
            math.log(sigma_range[0]), math.log(sigma_range[1]), generator=g)

        # fixed per-mode random orthonormal frame (not conditioned on c)
        A = torch.randn(K, d, d, generator=g)
        base_rot, _ = torch.linalg.qr(A)  # [K,d,d]

        # per-axis anisotropy, zero-mean across d so a mode's overall scale
        # (and therefore depth/mass bookkeeping) is unaffected by its shape
        raw = torch.randn(K, d, generator=g)
        raw = raw - raw.mean(-1, keepdim=True)
        base_log_eigstd = base_log_sigma[:, None] + aniso_scale * raw  # [K,d]

        base_depth = torch.empty(K).uniform_(*depth_range, generator=g)

        gate_bias = torch.empty(K)
        gate_bias[:n_core] = 3.0
        gate_bias[n_core:] = -3.0

        scale = 1.0 / math.sqrt(cond_dim)
        gate_dir = torch.randn(K, cond_dim, generator=g) * scale
        pos_proj = torch.randn(K, d, cond_dim, generator=g) * scale
        logsig_proj = torch.randn(K, d, cond_dim, generator=g) * scale  # NEW: per-axis (was [K,cond_dim])
        depth_proj = torch.randn(K, cond_dim, generator=g) * scale

        sig_eff_max = (max_width
                       * math.exp(math.log(sigma_range[1]) + logsig_scale + aniso_scale)
                       * math.sqrt(max_temperature))
        B = disp_max + edge_sigmas * sig_eff_max
        assert B < 1.0, (
            f"margin {B:.3f} >= 1: shrink sigma_range / edge_sigmas / disp_max / aniso_scale, "
            f"or lower max_temperature / max_width.")
        lo, hi = -1.0 + B, 1.0 - B
        base_mu = mu_scale * (torch.empty(K, d).uniform_(0, 1, generator=g) * (hi - lo) + lo)

        log_gate0 = F.logsigmoid(gate_steep * gate_bias)
        C0 = torch.logsumexp(log_gate0 + base_depth, dim=0)

        dev = self.device
        self._field = dict(
            cond_dim=cond_dim, dim=d, disp_max=disp_max, logsig_scale=logsig_scale,
            gate_steep=gate_steep, n_core=n_core, n_ghost=n_ghost,
            base_mu=base_mu.to(dev), base_log_eigstd=base_log_eigstd.to(dev),
            base_rot=base_rot.to(dev),
            base_depth=base_depth.to(dev), gate_bias=gate_bias.to(dev),
            gate_dir=gate_dir.to(dev), pos_proj=pos_proj.to(dev),
            logsig_proj=logsig_proj.to(dev), depth_proj=depth_proj.to(dev),
            C0=C0.to(dev),
        )

    def _ensure_latent_field(self, **kwargs):
        """Build the latent field lazily if missing, forwarding only the kwargs
        _build_latent_field actually accepts (so callers can pass through extra,
        unrelated kwargs from compute()/analyze() without erroring)."""
        if getattr(self, "_field", None) is None:
            import inspect
            valid_keys = set(inspect.signature(self._build_latent_field).parameters) - {'self'}
            field_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
            self._build_latent_field(**field_kwargs)

    def _latent_field_params(self, c):
        """Return (mu [...,K,d], R [K,d,d] fixed, log_eigstd [...,K,d], log_w [...,K]) at
        condition c. R is not conditioned on c (fixed per-mode frame); everything else is.
        c=None => reference state (zeros). log_w is reference-shifted (unnormalized)."""
        f = self._field
        if f["base_mu"].device != self.device:
            for k, v in f.items():
                if torch.is_tensor(v):
                    f[k] = v.to(self.device)
        if c is None:
            c = torch.zeros(f["cond_dim"], device=self.device)
        if isinstance(c, list):
            c = torch.tensor(c, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(c):
            c = c.to(self.device)

        disp = f["disp_max"] * torch.tanh(
            torch.einsum('kdc,...c->...kd', f["pos_proj"], c))  # [...,K,d]
        mu = f["base_mu"] + disp
        log_eigstd = f["base_log_eigstd"] + f["logsig_scale"] * torch.tanh(
            torch.einsum('kdc,...c->...kd', f["logsig_proj"], c))  # [...,K,d]
        log_gate = F.logsigmoid(
            f["gate_steep"] * (torch.einsum('kc,...c->...k', f["gate_dir"], c)
                               + f["gate_bias"]))
        raw_depth = f["base_depth"] + torch.einsum('kc,...c->...k', f["depth_proj"], c)
        log_w = (log_gate + raw_depth) - f["C0"]
        return mu, f["base_rot"], log_eigstd, log_w

    def latent_multiharmonic_en(self, c=None, width=1.0, x=None, **kwargs):
        """E(x; c, width) = -log g(x; c, width). T-FREE.

        g is the conditional Gaussian mixture cached in ``self._field`` by
        ``_build_latent_field``: g(x) = sum_k w_k(c) N(x; mu_k(c), Sigma_k) with
        Sigma_k = R_k diag(width * eigstd_k(c))^2 R_k^T and unnormalised
        weights w_k (``_latent_field_params``).

        Parameters
        ----------
        c : torch.Tensor or list, optional
            Condition, [cond_dim] (shared) or [B, cond_dim] (one per sample).
            None is the reference state (zeros).
        width : float or torch.Tensor, optional
            Scale on every mode's standard deviations; a scalar or a per-sample
            tensor [B] (mixed widths in one batch).
        x : torch.Tensor, optional
            [B, d] points to score. None uses
            ``self.latent_params(gauge_fix_free_axes=False)`` (see
            ``latent_harmonic_en``).
        **kwargs
            Forwarded to ``_ensure_latent_field``: when ``self._field`` does not
            yet exist, the ones that are ``_build_latent_field`` arguments are
            used to build it; otherwise they are ignored.

        Returns
        -------
        torch.Tensor
            [B] energy.
        """
        import math

        self._ensure_latent_field(**kwargs)
        # gauge-FREE: latent_params() default mutates the batch (P1 2026-08-24)
        x = self.latent_params(gauge_fix_free_axes=False) if x is None else x  # [B,d]
        mu, R, log_eigstd, log_w = self._latent_field_params(c)  # c broadcasts over B
        w = torch.as_tensor(width, dtype=x.dtype, device=x.device)
        if w.ndim == 1:
            w = w[:, None]  # [B] -> [B,1] over K
        eigstd = w[..., None] * log_eigstd.exp()  # [...,K,d]

        diff = x[..., None, :] - mu  # [...,K,d]
        # rotate into each mode's principal frame: y = R_k^T (x - mu_k)
        y = torch.einsum('kij,...ki->...kj', R, diff)  # [...,K,d]
        sq = (y ** 2 / eigstd ** 2).sum(-1)  # [...,K]
        d = mu.shape[-1]
        log_N = (-0.5 * d * math.log(2 * math.pi)
                 - eigstd.log().sum(-1)
                 - 0.5 * sq)
        log_g = torch.logsumexp(log_w + log_N, dim=-1)
        return -log_g

    def latent_log_reward(self, c=None, target_temperature=1.0, width=1.0, x=None):
        """log pi_T(x) up to log Z: -E / T = (1/T) log g.

        E is ``latent_multiharmonic_en(c=c, width=width, x=x)``, which builds the
        field with default arguments if ``self._field`` does not exist. Returns
        a tensor shaped like E ([B]).
        """
        E = self.latent_multiharmonic_en(c=c, width=width, x=x)
        T = torch.as_tensor(target_temperature, dtype=E.dtype, device=E.device)
        return -E / T

    def _proposal_logprob(self, x, mu, R, eigstd, log_alpha_norm, T):
        """Log-density of the per-component tempered-GMM proposal q_T, dense-covariance
        version. mu [K,d], R [K,d,d], eigstd [K,d] (untempered, T applied here)."""
        import math

        d = mu.shape[-1]
        var = T * eigstd ** 2  # [K,d]
        diff = x[:, None, :] - mu[None]  # [m,K,d]
        y = torch.einsum('kij,mki->mkj', R, diff)  # [m,K,d]
        sq = (y ** 2 / var[None]).sum(-1)  # [m,K]
        log_N = (-0.5 * d * math.log(2 * math.pi)
                 - 0.5 * var.log().sum(-1)[None]
                 - 0.5 * sq)
        return torch.logsumexp(log_alpha_norm[None] + log_N, dim=1)  # [m]

    def sample_latent_multiharmonic(self, n_samples, c=None, target_temperature: float = 1.0,
                                    width: float = 1.0, exact: bool = True,
                                    sir_oversample: int = 8, seed=None,
                                    return_diagnostics: bool = False, **kwargs):
        """Sample the tempered target g^{1/T}/Z at a SINGLE shared condition c.

        The proposal q_T picks mode k with probability proportional to
        w_k^{1/T} Z_k(T) and draws from N(mu_k, T * Sigma_k), with
        Sigma_k = R_k diag(width * eigstd_k)^2 R_k^T.
        (For per-sample c, loop over conditions; energy/reward/logZ are batched.)

        Parameters
        ----------
        n_samples : int
            Number of samples returned.
        c : torch.Tensor or list, optional
            Condition of shape [cond_dim]; None is the reference state (zeros).
            A batched c fails the ``mu.dim() == 2`` assertion.
        target_temperature : float, optional
            T.
        width : float, optional
            Scalar scale on every mode's standard deviations (converted with
            ``float``).
        exact : bool, optional
            False: draws from the per-component tempered-GMM proposal q_T.
            EXACT at T=1; near-exact for well-separated modes at T!=1
            (anisotropy/rotation doesn't change this -- only mode separation
            does; watch overlap if you push aniso_scale up).
            True (default): draws ``n_samples * sir_oversample`` from q_T and
            SIR-resamples ``n_samples`` of them, with replacement, in
            proportion to the target/proposal weights.
        sir_oversample : int, optional
            Proposal draws per returned sample when ``exact`` is True.
        seed : int, optional
            Seed for the CPU ``torch.Generator`` used for mode choice, noise and
            resampling. With None it is not seeded explicitly and starts from
            torch's fixed default seed for a new generator.
        return_diagnostics : bool, optional
            If True, also return a dict with ``ess`` and ``ess_frac``
            (effective-sample-size / m, m the number of proposal draws) and
            ``max_weight_frac`` from the SIR weights. Low ess_frac (e.g. <1%)
            means sir_oversample is too small for this config and the resample
            is dominated by a handful of proposal draws -- not trustworthy
            regardless of what the output looks like. See
            check_latent_sampler_accuracy for a fuller diagnostic including
            ground-truth comparison.
        **kwargs
            Forwarded to ``_ensure_latent_field`` (used only if the field does
            not yet exist).

        Returns
        -------
        torch.Tensor or tuple
            [n_samples, d] samples on ``self.device``; with
            ``return_diagnostics``, ``(samples, diagnostics)``, where
            diagnostics is None when ``exact`` is False.
        """
        import math

        self._ensure_latent_field(**kwargs)
        T = float(target_temperature)
        d = self._field["dim"]
        mu, R, log_eigstd, log_w = self._latent_field_params(c)
        assert mu.dim() == 2, "sampler expects a single shared c of shape [cond_dim]"
        eigstd = float(width) * log_eigstd.exp()  # [K,d]

        # per-component tempered proposal: N(mu_k, T Sigma_k), weight ~ w_k^{1/T} Z_k(T)
        logZk = (0.5 * d * (1 - 1 / T) * math.log(2 * math.pi)
                 + (1 - 1 / T) * eigstd.log().sum(-1)
                 + 0.5 * d * math.log(T))
        log_alpha = log_w / T + logZk
        log_alpha = log_alpha - torch.logsumexp(log_alpha, 0)  # normalized

        m = int(n_samples * (sir_oversample if exact else 1))
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        k = torch.multinomial(log_alpha.exp().cpu(), m, replacement=True, generator=g).to(self.device)
        eps = torch.randn(m, d, generator=g, device="cpu").to(self.device)  # [m,d]
        # x = mu_k + sqrt(T) * R_k @ diag(eigstd_k) @ eps
        scaled_eps = (T ** 0.5) * eigstd[k] * eps  # [m,d]
        x = mu[k] + torch.einsum('mij,mj->mi', R[k], scaled_eps)  # [m,d]
        if not exact:
            return (x, None) if return_diagnostics else x

        log_r = self.latent_log_reward(c=c, target_temperature=T, width=width, x=x)
        log_q = self._proposal_logprob(x, mu, R, eigstd, log_alpha, T)
        logw = log_r - log_q
        w = (logw - logw.max()).exp()
        idx = torch.multinomial(w.cpu(), n_samples, replacement=True, generator=g).to(self.device)
        x_out = x[idx]
        if not return_diagnostics:
            return x_out
        ess = (w.sum() ** 2 / (w ** 2).sum()).item()
        diagnostics = dict(ess=ess, ess_frac=ess / w.shape[0],
                           max_weight_frac=(w.max() / w.sum()).item())
        return x_out, diagnostics

    def _mode_log_resp(self, c, width, x):
        """Log responsibility log[w_k N_k(x) / g(x)] of each mode k for samples x. [...,K]"""
        import math

        mu, R, log_eigstd, log_w = self._latent_field_params(c)
        w = torch.as_tensor(width, dtype=x.dtype, device=x.device)
        if w.ndim == 1:
            w = w[:, None]
        eigstd = w[..., None] * log_eigstd.exp()
        diff = x[..., None, :] - mu
        y = torch.einsum('kij,...ki->...kj', R, diff)
        sq = (y ** 2 / eigstd ** 2).sum(-1)
        d = mu.shape[-1]
        log_N = (-0.5 * d * math.log(2 * math.pi)
                 - eigstd.log().sum(-1) - 0.5 * sq)
        log_joint = log_w + log_N
        return log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)

    def check_latent_sampler_accuracy(self, c=None, target_temperature: float = 1.0,
                                      width: float = 1.0, n_samples: int = 2_000,
                                      sir_oversample: int = 8,
                                      n_samples_ref: int = 2_000, reference_oversample: int = 100,
                                      seed: int = 0, **kwargs):
        """Diagnose whether sample_latent_multiharmonic(exact=True) is trustworthy at
        this (c, T, width, aniso) — vs. just showing the genuine rare high-energy tail
        of the true target, which increasing sir_oversample will NOT remove.

        NOTE: at target_temperature=1.0, the proposal is EXACT by construction
        (log_r - log_q is constant in x, so ess_frac=1.0 and weights are uniform
        REGARDLESS of aniso_scale/overlap) — this check is a no-op there. Run it
        at the T you actually sample at to see anything meaningful.

        The reference pass uses its own n_samples_ref/reference_oversample,
        independent of n_samples/sir_oversample — raw proposal draws scale as
        n_samples_ref * reference_oversample, and per-draw energy evaluation is
        O(K*d), so keep that product bounded rather than reusing n_samples.

        The working pass draws ``n_samples`` with ``sir_oversample`` and
        ``seed``; the reference pass draws ``n_samples_ref`` with
        ``reference_oversample`` and ``seed + 1``. Both use
        ``sample_latent_multiharmonic(..., exact=True)`` at the same ``c``,
        ``target_temperature`` and ``width``, and both are scored with
        ``latent_multiharmonic_en``. kwargs are forwarded to
        ``_ensure_latent_field``.

        Returns a dict:
          ess_frac / max_weight_frac : from the working-config SIR pass (see
              sample_latent_multiharmonic). Low ess_frac => resample is degenerate;
              re-run with larger sir_oversample before trusting anything else here.
          energy_q99_work vs energy_q99_ref, energy_max_work vs energy_max_ref :
              working-config resample vs. a much-larger-oversample reference resample.
              Close agreement => the high-energy points you're seeing are a real
              feature of the target at this config, not SIR bias.
          mode_occupancy_l1 : (T==1 only) L1 distance between the mode
              responsibilities of the working resample, averaged over samples
              (``_mode_log_resp``), and the EXACT mode weights softmax(log_w)
              (closed form, no approximation at T=1). Near 0 confirms the whole
              SIR pipeline is faithful to a case with a known-exact answer.

        All values are Python floats.
        """
        self._ensure_latent_field(**kwargs)

        x_work, diag = self.sample_latent_multiharmonic(
            n_samples, c=c, target_temperature=target_temperature, width=width,
            exact=True, sir_oversample=sir_oversample, seed=seed, return_diagnostics=True)
        x_ref, _ = self.sample_latent_multiharmonic(
            n_samples_ref, c=c, target_temperature=target_temperature, width=width,
            exact=True, sir_oversample=reference_oversample, seed=seed + 1, return_diagnostics=True)

        E_work = self.latent_multiharmonic_en(c=c, width=width, x=x_work)
        E_ref = self.latent_multiharmonic_en(c=c, width=width, x=x_ref)

        out = dict(
            ess_frac=diag["ess_frac"],
            max_weight_frac=diag["max_weight_frac"],
            energy_q99_work=E_work.quantile(0.99).item(),
            energy_q99_ref=E_ref.quantile(0.99).item(),
            energy_max_work=E_work.max().item(),
            energy_max_ref=E_ref.max().item(),
        )

        if abs(float(target_temperature) - 1.0) < 1e-12:
            mu, R, log_eigstd, log_w = self._latent_field_params(c)
            true_occ = log_w.softmax(dim=-1)
            resp = self._mode_log_resp(c, width, x_work).exp().mean(0)
            out["mode_occupancy_l1"] = (resp - true_occ).abs().sum().item()

        return out

    def log_partition_latent(self, c=None, target_temperature: float = 1.0,
                             width: float = 1.0, n_is: int = 200_000, seed: int = 0, **kwargs):
        """Ground-truth log Z(T, c, width) of the tempered target g^{1/T}.

        T==1: closed form logsumexp(unnormalized weights) over the modes; exact,
        independent of ``width``, and a batched c [B, cond_dim] gives [B].
        T!=1: logmeanexp IS with the tempered-GMM proposal: ``n_is`` draws from
        ``sample_latent_multiharmonic(..., exact=False, seed=seed)``, weighted
        by ``latent_log_reward - log q_T``; requires a single shared c.

        Parameters
        ----------
        c : torch.Tensor or list, optional
            Condition; None is the reference state (zeros).
        target_temperature : float, optional
            T.
        width : float, optional
            Scalar scale on every mode's standard deviations.
        n_is : int, optional
            Number of importance-sampling draws (T != 1 only).
        seed : int, optional
            Seed for the proposal draws (T != 1 only).
        **kwargs
            Forwarded to ``_ensure_latent_field`` (used only if the field does
            not yet exist).

        Returns
        -------
        torch.Tensor
            0-dim for a single c; [B] for a batched c at T == 1.
        """
        import math

        T = float(target_temperature)
        self._ensure_latent_field(**kwargs)
        mu, R, log_eigstd, log_w = self._latent_field_params(c)
        if abs(T - 1.0) < 1e-12:
            return torch.logsumexp(log_w, dim=-1)  # exact

        x = self.sample_latent_multiharmonic(
            n_is, c=c, target_temperature=T, width=width, exact=False, seed=seed)
        eigstd = float(width) * log_eigstd.exp()
        d = self._field["dim"]
        logZk = (0.5 * d * (1 - 1 / T) * math.log(2 * math.pi)
                 + (1 - 1 / T) * eigstd.log().sum(-1)
                 + 0.5 * d * math.log(T))
        log_alpha = log_w / T + logZk
        log_alpha = log_alpha - torch.logsumexp(log_alpha, 0)
        log_q = self._proposal_logprob(x, mu, R, eigstd, log_alpha, T)
        log_r = self.latent_log_reward(c=c, target_temperature=T, width=width, x=x)
        logw = log_r - log_q
        return torch.logsumexp(logw, dim=0) - math.log(logw.shape[0])  # logmeanexp

    def compute_rdf(self,  # todo rebuild analyses with a template
                    rdf_cutoff: float = 10,
                    bins: int = 100,
                    rdf_mode: Optional[str] = None,
                    **kwargs,
                    ):
        """Intermolecular radial distribution per crystal, from the cluster pair list.

        Calls ``crystal_rdf(self, self.edges_dict, rrange=(0, rdf_cutoff),
        bins=bins, mode=rdf_mode)``, which histograms the distances of the
        ``edges_dict['edge_index_inter']`` pairs only, so pairs beyond the
        radial-graph cutoff are absent. ``crystal_rdf`` normalises the histogram
        by shell volume (a radial density, not the dimensionless g(r)) and
        multiplies it by a smooth cutoff envelope; this method then divides it
        by ``z_prime`` per graph.

        Parameters
        ----------
        rdf_cutoff : float, optional
            Upper end of the histogram range (0, rdf_cutoff), Angstrom.
        bins : int, optional
            Number of histogram bins.
        rdf_mode : str, optional
            None: one channel over all intermolecular pairs. ``'elementwise'``,
            ``'atomwise'`` or ``'envwise'``: one channel per element pair, atom
            pair or environment-label pair, as implemented by ``crystal_rdf``.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        rdf : torch.Tensor
            [num_graphs, num_channels, bins].
        bin_positions : torch.Tensor
            [bins], ``bin_edges[:-1] + bin_width`` (each bin's upper edge), with
            ``requires_grad`` set.
        rdf_pair_dict : dict
            Channel labels from ``crystal_rdf``; empty when ``rdf_mode`` is
            None.

        Raises
        ------
        RuntimeError
            If ``edges_dict`` has not been built.
        AssertionError
            If ``self`` is not a Batch.
        """
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing RDF!"
                               "Do batch.construct_radial_graph before calling.")

        assert self.is_batch, "RDF method not implemented for single crystals"
        rdf, bin_edges, rdf_pair_dict = crystal_rdf(self,
                                                    self.edges_dict,
                                                    rrange=(0, rdf_cutoff),
                                                    bins=bins,
                                                    mode=rdf_mode
                                                    )
        return rdf / self.z_prime[:, None, None], bin_edges, rdf_pair_dict

    def compute_cell_reduction_penalty(self, margin: float = 0.0, **kwargs):
        """Compute a penalty term for the given crystal system that pushes it towards
        our canonical / standardized / reduced cell geometry.

        monoclinic: one zero-penalty cell per lattice within the SYM_OPS setting, the same cell as spglib 2.7.0's
            standardize_cell(to_primitive=False, no_idealize=True) (checked empirically, not proven for spglib);
            see sym_utils.mono_reduction_penalty.
        triclinic: one zero-penalty cell per lattice, in a beta/gamma-obtuse convention that differs from spglib's Niggli
            cell for all-acute lattices; see sym_utils.tri_niggli_reduction_penalty (MXT_LEGACY_TRICLINIC_WALLS=1 restores
            the pre-Niggli walls).

        every other system: enforces only the crystal-system metric; does not select a unique cell.

        Reads ``cell_lengths`` and ``cell_angles`` (radians) through
        ``split_cell_params`` and ``sg_ind``, and dispatches each graph to its
        crystal system's penalty in ``sym_utils.cell_reduction_penalty``. Does
        not modify ``self``.

        Parameters
        ----------
        margin : float, optional
            Passed to the per-system penalty functions; the orthorhombic and
            higher-symmetry penalties do not use it.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        torch.Tensor
            [num_graphs] penalty, zero for a cell that satisfies its walls, in
            the float dtype of ``cell_lengths``; 0 for ``sg_ind`` outside 1-230.
        """
        # todo check behaviors / correctness for higher crystal systems
        # cell_parameters = self.full_cell_parameters()
        cell_lengths, cell_angles, aunit_positions, aunit_orientations = self.split_cell_params()

        'get cs masks'
        # crystal_systems  #
        sg = self.sg_ind
        E = cell_reduction_penalty(cell_angles, cell_lengths, sg, margin)

        return E

    def batch_compack(self, ref_path, inds_to_check, n_cpus: int = 8, repair: bool = False,
                      settings: Optional[dict] = None):  # todo refactor into analysis code
        """Compare selected crystals of the batch against a reference CIF with CCDC COMPACK.

        Calls ``self.mol2ucell()`` (``std_orientation=True``), which rebuilds
        ``pos`` and the unit cell of ``self`` in place, then writes
        ``compack_{ind}.cif`` (a P1 unit cell, ``write_cif(..., 'unit cell')``)
        in the working directory for each ``ind`` in ``inds_to_check``. A pool
        of ``n_cpus`` processes then compares each written file against
        ``ref_path`` with ``compack_compare`` (``ref_path`` the reference),
        printing each result as ``single_compack_run`` does.

        Parameters
        ----------
        ref_path : str
            CIF of the reference crystal.
        inds_to_check : sequence of int
            Batch indices of the crystals to write and compare.
        n_cpus : int, optional
            Number of worker processes.
        repair : bool, optional
            Repair both CIFs of each comparison (``repair_compack_crystal``)
            with the template of the written crystal's own asymmetric unit
            (``compack_bond_template`` of its ``z`` and ``pos``): the written
            CIF is repaired; the reference is repaired too if it is in the
            writer's layout, used as read if the reader returns whole
            molecules of the template, refused otherwise. Default False: both
            as the CSD reader perceives them.
        settings : dict, optional
            Engine settings (``compack_engine``); default
            ``packing_shell_size = 20`` and CCDC defaults.

        Returns
        -------
        matches : numpy.ndarray
            [len(inds_to_check)] number of matched molecules per comparison.
        rmsds : numpy.ndarray
            [len(inds_to_check)] COMPACK RMSD per comparison; a failed
            comparison gives 0 in both arrays, a repair failure -1 matched and
            RMSD nan.
        """
        import numpy as np
        import multiprocessing as mp

        inds = [int(i) for i in inds_to_check]
        self.mol2ucell()
        self.write_cif(inds, f'compack', 'unit cell')
        templates = [compack_bond_template(self.z[self.batch == ind].cpu().numpy(),
                                           self.pos[self.batch == ind].detach().cpu().double().numpy())
                     if repair else None for ind in inds]
        pool = mp.Pool(n_cpus)
        results = []
        for ind, template in zip(inds, templates):
            results.append(
                pool.apply_async(
                    _batch_compack_job,
                    (ind, f'compack_{ind}.cif', ref_path, template, repair, settings)
                )
            )
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        matches = np.array([res[1] for res in results])
        rmsds = np.array([res[0] for res in results])

        return matches, rmsds
