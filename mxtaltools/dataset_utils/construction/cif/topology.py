"""Component decomposition from the CIF's own bond list.

The finding this rests on (design M2): union-find over `_geom_bond_atom_site_label_1/2`
reproduces CCDC's `len(molecule.components)` on **194/200** random CSD entries, with
**zero organic failures** -- all six misses are metal/ionic/polymeric.

So for the organic corpus this is not chemical perception at all.  The bond list
is data in the file, auditable line by line, which is exactly what makes the
result trustworthy in a way `ccdc.molecule.components` is not.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .errors import CifTopologyError, CifUnsupportedError
from .sites import SiteSet
from .tokenizer import CifBlock

__all__ = ['ComponentSet', 'components_from_block', 'union_find']

_BOND_1 = '_geom_bond_atom_site_label_1'
_BOND_2 = '_geom_bond_atom_site_label_2'
_BOND_SYM_1 = '_geom_bond_site_symmetry_1'
_BOND_SYM_2 = '_geom_bond_site_symmetry_2'
_MOIETY = '_chemical_formula_moiety'

# A moiety formula ending in ')n' declares a polymer, e.g. '(C18 H18 N4 O5 Zn1)n'.
# This is the discriminator that WORKS: the bond-symmetry-code route finds 0 of 12
# CCDC-polymeric structures because the exporter writes only intra-repeat-unit bonds.
_POLYMER_RE = re.compile(r'\)\s*n\b')


def unwrap_components(frac: np.ndarray, edges, n_sites: int):
    """Make every bonded pair contiguous, by BFS over the bond graph.

    Some CIFs write each atom wrapped into [0,1) independently, which SPLITS a
    molecule that straddles a cell boundary. The deposited coordinates then look
    fine per-atom while the molecule is in pieces, and a component centroid
    computed from them lands somewhere meaningless.

    CSD exports do NOT do this -- measured 0/2000 split molecules, and e.g.
    KUNTEV is deposited spanning -0.435 to 1.203 fractional, i.e. deliberately
    outside the cell so the molecule stays whole. But that is a property of the
    exporter, not of CIF, so it must be checked rather than assumed.

    Walks each connected component from an arbitrary root, placing every atom at
    the periodic image NEAREST its already-placed neighbour. Returns the repaired
    fractional coordinates and how many atoms moved.
    """
    adj = [[] for _ in range(n_sites)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    out = frac.copy()
    seen = np.zeros(n_sites, dtype=bool)
    moved = 0
    for root in range(n_sites):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if seen[j]:
                    continue
                shift = np.round(out[i] - out[j])          # nearest image of j to i
                if np.any(shift != 0):
                    out[j] = out[j] + shift
                    moved += 1
                seen[j] = True
                stack.append(j)
    return out, moved


class ComponentSet:
    """Which component each site belongs to, and how that was established.

    The edge list itself is deliberately NOT retained: it is consumed by the
    union-find and then has no consumer, and `_geom_bond` is present in 100 % of
    sampled CSD files, so re-deriving it is one pass whenever something actually
    needs it.
    """

    __slots__ = ('component_index', 'n_components', 'source', 'n_unwrapped')

    def __init__(self, component_index, n_components, source, n_unwrapped=0):
        self.component_index = component_index      # (n_sites,) int
        self.n_components = n_components
        self.source = source                        # 'geom_bond' | 'none'
        # how many sites had to be moved to a different periodic image to make
        # their molecule contiguous. 0 for every CSD export measured; non-zero
        # means the file was written wrapped and the geometry was repaired.
        self.n_unwrapped = n_unwrapped

    def sizes(self) -> List[int]:
        return [int((self.component_index == k).sum()) for k in range(self.n_components)]

    def __repr__(self) -> str:
        return f'ComponentSet({self.n_components} components, sizes {self.sizes()})'


def union_find(n: int, edges) -> np.ndarray:
    """Connected-component labels for `n` nodes under `edges`.

    Labels are assigned in order of first appearance, so the result is a
    deterministic function of the input ordering -- which matters, because
    downstream code indexes components positionally.
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    out = np.empty(n, dtype=np.int64)
    seen: Dict[int, int] = {}
    for i in range(n):
        r = find(i)
        if r not in seen:
            seen[r] = len(seen)
        out[i] = seen[r]
    return out


def components_from_block(block: CifBlock,
                          sites: SiteSet,
                          identifier: str = None,
                          refuse_polymers: bool = True,
                          require_bonds: bool = True) -> ComponentSet:
    """Decompose the deposited set into components using `_geom_bond`.

    Refusals:
      - polymer declared by the moiety formula -> `CifUnsupportedError('polymer')`
      - a bond referring to a site label that does not exist -> `CifTopologyError`
      - no bond loop at all -> `CifTopologyError`, because guessing connectivity
        from distances is chemical perception and belongs to a chemistry backend,
        not here

    Bonds carrying a non-`1_555` symmetry code are DROPPED, not followed: they
    connect to a symmetry image rather than to a site in the deposited set, so
    following them would merge components that are not connected within the set.
    Measured 0/3000 CSD files carry such a code, so this is protection for
    non-CSD input rather than a live path.
    """
    moiety = block.get(_MOIETY)
    if refuse_polymers and moiety and _POLYMER_RE.search(moiety):
        raise CifUnsupportedError(
            f'moiety formula {moiety!r} declares a polymer',
            reason='polymer', identifier=identifier)

    loop = block.loop_with(_BOND_1)
    if loop is None:
        if require_bonds:
            raise CifTopologyError(
                f'no bond loop ({_BOND_1} absent); inferring connectivity from '
                'coordinates is chemical perception and is not done here. Pass '
                'require_bonds=False to read the deposited set with its component '
                'decomposition UNKNOWN', identifier)
        # Explicitly asked to proceed without connectivity. The decomposition is
        # NOT computed and must not be claimed: every site goes into one component
        # and `source='none'` records that this is an absence, not a result.
        # This path exists because MolCrystalData carries no bond topology
        # (`data_classes.py:643`: "there is no stored bond topology to build from"),
        # so a CIF this package writes cannot carry `_geom_bond` and could
        # otherwise never be read back.
        return ComponentSet(np.zeros(len(sites), dtype=np.int64),
                            1 if len(sites) else 0, 'none')
    if _BOND_2 not in loop:
        raise CifTopologyError(f'bond loop has {_BOND_1} but no {_BOND_2}', identifier)

    index_of = {lab: i for i, lab in enumerate(sites.labels)}
    l1 = loop.column(_BOND_1)
    l2 = loop.column(_BOND_2)
    s1 = loop.column(_BOND_SYM_1) if _BOND_SYM_1 in loop else ['1_555'] * len(l1)
    s2 = loop.column(_BOND_SYM_2) if _BOND_SYM_2 in loop else ['1_555'] * len(l2)

    edges: List[Tuple[int, int]] = []
    n_dropped = 0
    for a, b, ca, cb in zip(l1, l2, s1, s2):
        # same as sites.py: a trailing apostrophe is part of the label (S1 vs S1'),
        # not quoting, so only whitespace is stripped
        a = a.strip()
        b = b.strip()
        if a not in index_of or b not in index_of:
            missing = a if a not in index_of else b
            raise CifTopologyError(
                f'bond references site {missing!r}, which is not in the atom-site loop',
                identifier)
        if ca.strip() != '1_555' or cb.strip() != '1_555':
            n_dropped += 1                 # connects to a symmetry image, not to us
            continue
        edges.append((index_of[a], index_of[b]))

    comp = union_find(len(sites), edges)
    n_comp = int(comp.max()) + 1 if len(comp) else 0

    # repair a wrapped deposit IN PLACE on the site set, so everything downstream
    # -- centroids, the unit cell build, the molecule geometry -- sees contiguous
    # molecules. Uses the bond graph, which is the only thing that can distinguish
    # "these two atoms are bonded across a cell edge" from "these are far apart".
    repaired, n_moved = unwrap_components(sites.frac, edges, len(sites))
    if n_moved:
        sites.frac = repaired
    return ComponentSet(comp, n_comp, 'geom_bond', n_moved)
