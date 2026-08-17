"""
A deliberately dumb one-body prior over internal coordinates.

Typed marginals for every acyclic degree of freedom, a joint per ring system, fattened
toward uniform so support is complete. Fitted from any collection of ``MolData`` -- the
statistics are read straight off real conformers via ``internal_dof()``, so nothing
external is needed.

Why this shape
--------------
*Types transfer, molecules do not.* A C(sp3)-C(sp3) torsion has the same marginal in
every molecule, so the table is built once and applies to anything. That is the whole
advantage over enumerating rotamer products per molecule.

*Fattening buys completeness.* Mixing each marginal with uniform gives full support, so
the prior covers every mode in the limit -- which is exactly what trajectory balance
needs, since TB is off-policy consistent and only the support matters, not the weights.

*Rings are sampled jointly.* Ring torsions are the maximally-correlated case (closure is
a hard constraint), so a product of marginals is guaranteed to violate it. Each ring
system instead draws a whole observed DoF block, with a jitter bandwidth that makes
sampling and scoring the same distribution.

That last point is the design constraint throughout: ``sample`` and ``log_prob`` must
describe the *same* density, or the prior is useless as an importance-sampling proposal
-- which is most of why it is worth building rather than reusing ETKDG, whose draws come
without a density.

Typing is deliberately coarse: ``(element, degree)`` tuples, no SMARTS, no bond orders,
no ring-aware refinement beyond the ring/acyclic split. Degree stands in for
hybridization, which separates sp3/sp2/sp cheaply. Anything finer is work the 2-/3-/4-body
heads should be doing, and doing better.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# marginal supports. r in A; theta in (0, pi); phi periodic on (-pi, pi]
R_RANGE = (0.6, 2.6)
THETA_RANGE = (0.0, np.pi)
PHI_RANGE = (-np.pi, np.pi)


@dataclass
class Histogram1D:
    """Fattened histogram with matching sampler and density."""

    counts: np.ndarray
    lo: float
    hi: float
    periodic: bool = False

    @property
    def width(self) -> float:
        return (self.hi - self.lo) / len(self.counts)

    def density(self, fatten: float) -> np.ndarray:
        """Normalised per-bin density, mixed toward uniform by ``fatten``."""
        total = self.counts.sum()
        p = (self.counts / total if total > 0
             else np.full(len(self.counts), 1.0 / len(self.counts)))
        p = (1.0 - fatten) * p + fatten / len(self.counts)
        return p / (p.sum() * self.width)          # per unit x

    def sample(self, n: int, fatten: float, rng) -> np.ndarray:
        d = self.density(fatten) * self.width      # back to bin mass
        idx = rng.choice(len(d), size=n, p=d / d.sum())
        return self.lo + (idx + rng.random(n)) * self.width

    def log_prob(self, x: np.ndarray, fatten: float) -> np.ndarray:
        d = self.density(fatten)
        span = self.hi - self.lo
        u = (x - self.lo) % span if self.periodic else np.clip(x - self.lo, 0, span - 1e-12)
        idx = np.clip((u / self.width).astype(int), 0, len(d) - 1)
        return np.log(d[idx])


@dataclass
class RingBank:
    """Observed DoF blocks for one ring-system signature, as a jittered mixture.

    Sampling draws a row and adds Gaussian jitter; the density is the corresponding
    kernel mixture. Keeping those two consistent is what lets a ring-containing molecule
    still be used as an importance proposal.
    """

    rows: np.ndarray                      # [n_obs, d]
    bandwidth: float = 0.05

    def sample(self, n: int, rng) -> np.ndarray:
        pick = rng.integers(0, len(self.rows), size=n)
        return self.rows[pick] + rng.normal(0.0, self.bandwidth, size=(n, self.rows.shape[1]))

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        # log mean_i N(x; row_i, bw^2 I) -- O(n_obs) per query, fine at bank sizes here
        d = self.rows.shape[1]
        diff = x[:, None, :] - self.rows[None, :, :]
        sq = (diff ** 2).sum(-1) / (2 * self.bandwidth ** 2)
        norm = -0.5 * d * np.log(2 * np.pi * self.bandwidth ** 2)
        m = -sq.max(axis=1, keepdims=True)
        return (norm + np.log(np.exp(-sq + m).mean(axis=1)) - m[:, 0])


@dataclass
class InternalPrior:
    n_bins_r: int = 60
    n_bins_theta: int = 60
    n_bins_phi: int = 36
    fatten: float = 0.15
    ring_bandwidth: float = 0.05

    bonds: Dict = field(default_factory=dict)
    angles: Dict = field(default_factory=dict)
    torsions: Dict = field(default_factory=dict)
    rings: Dict = field(default_factory=dict)
    n_fitted: int = 0
    # Bumped whenever ring_systems' signature changes shape. A fit made under an older
    # version has ring keys that simply do not resolve, and the symptom is indistinguishable
    # from "this molecule's ring was never fitted" -- every ring silently falls through to
    # the caller's fallback. Consumers should read it via getattr(prior, 'ring_sig_version',
    # 1), since priors pickled before this field existed will not carry it.
    #   1 = (size, element multiset)                 -- collides benzene/cyclohexane
    #   2 = (size, sorted (element, degree) multiset)
    ring_sig_version: int = 2

    # ------------------------------------------------------------------- typing

    @staticmethod
    def _atom_keys(mol) -> np.ndarray:
        """(element, degree) per atom. Degree is a cheap hybridisation proxy."""
        z = mol.z.detach().cpu().numpy()
        bonds = mol.mol_bond_index.detach().cpu().numpy()
        deg = np.zeros(len(z), dtype=np.int64)
        np.add.at(deg, bonds[0], 1)
        np.add.at(deg, bonds[1], 1)
        return np.stack([z, deg], axis=1)

    @staticmethod
    def bond_key(a, b) -> Tuple:
        return tuple(sorted([tuple(a), tuple(b)]))

    @staticmethod
    def angle_key(a, b, c) -> Tuple:
        """Apex keeps its degree, outer atoms keep only their element.

        The angle at an sp3 carbon is ~109 deg largely regardless of what hangs off the
        outer atoms, so carrying their degree splits one physical distribution across
        many keys and starves each of them.
        """
        return (tuple(b), tuple(sorted([int(a[0]), int(c[0])])))

    @staticmethod
    def torsion_key(b, c) -> Tuple:
        """Central bond only -- the first-order determinant of torsion preference."""
        return tuple(sorted([tuple(b), tuple(c)]))

    # ------------------------------------------------------------- ring systems

    @staticmethod
    def ring_systems(mol) -> Tuple[np.ndarray, Dict]:
        """Per-atom ring-system id (-1 = acyclic), and a signature per system.

        Ring bonds are the non-bridges; a ring *system* is a connected component of
        them, so fused and bridged rings correctly come out as one joint block.
        """
        import networkx as nx

        bonds = mol.mol_bond_index.detach().cpu().numpy()
        keys = InternalPrior._atom_keys(mol)
        n = len(keys)
        g = nx.Graph([(int(i), int(j)) for i, j in zip(*bonds)])
        g.add_nodes_from(range(n))
        bridges = {tuple(sorted(e)) for e in nx.bridges(g)}
        ring_edges = [(int(i), int(j)) for i, j in zip(*bonds)
                      if tuple(sorted((int(i), int(j)))) not in bridges]

        sysid = np.full(n, -1, dtype=np.int64)
        sigs = {}
        if ring_edges:
            rg = nx.Graph(ring_edges)
            for s, comp in enumerate(nx.connected_components(rg)):
                comp = sorted(comp)
                sysid[comp] = s
                # size + the sorted (element, degree) MULTISET.
                #
                # Element alone is too coarse and collides across hybridisation: benzene
                # and cyclohexane shared a bank, as did pyridine and piperidine. Measured
                # consequence -- benzene drew its ring from a bank half full of chairs and
                # came out at a median |ring torsion| of 47 deg, with 75% of draws past
                # 20 deg. An aromatic ring puckered to 47 deg is not an aromatic ring, and
                # nothing in ff_from_reference objects, since it carries no torsion term
                # and bond angles stay near 120 deg through a pucker.
                #
                # An earlier note here justified dropping degree on the grounds that it
                # made the signature "nearly unique per molecule". That is true of an
                # ORDERED per-atom key; it is not true of the sorted multiset used here,
                # which is substitution-invariant by construction: benzene, toluene,
                # phenol, styrene and ibuprofen's ring all give ((6,3),)*6, and
                # cyclohexane, methylcyclohexane and cyclohexanol all give ((6,4),)*6.
                # Twelve test molecules resolve to six buckets, each grouping exactly the
                # rings that should share statistics.
                sigs[s] = (len(comp),
                           tuple(sorted(tuple(int(v) for v in keys[a]) for a in comp)))
        return sysid, sigs

    # ---------------------------------------------------------------------- fit

    def fit(self, mols, verbose: bool = True):
        acc_b, acc_a, acc_t = defaultdict(list), defaultdict(list), defaultdict(list)
        acc_ring = defaultdict(list)

        for mol in mols:
            try:
                mol.build_conformer_tree()
                r, th, ph = (x.detach().cpu().numpy() for x in mol.internal_dof())
            except Exception:
                continue                       # disconnected / linear / too small
            keys = self._atom_keys(mol)
            sysid, sigs = self.ring_systems(mol)
            bi = mol.tree_bond_index.detach().cpu().numpy()
            ai = mol.tree_angle_index.detach().cpu().numpy()
            ti = mol.tree_torsion_index.detach().cpu().numpy()

            # a DoF belongs to a ring system only if ALL its atoms are in that system
            def owner(idx_cols):
                s = sysid[idx_cols]
                return s[0] if (s[0] >= 0 and np.all(s == s[0])) else -1

            ring_blocks = defaultdict(list)
            for j in range(bi.shape[1]):
                o = owner(bi[:, j])
                (ring_blocks[o] if o >= 0 else acc_b[self.bond_key(*keys[bi[:, j]])]).append(r[j])
            for j in range(ai.shape[1]):
                o = owner(ai[:, j])
                (ring_blocks[o] if o >= 0 else acc_a[self.angle_key(*keys[ai[:, j]])]).append(th[j])
            for j in range(ti.shape[1]):
                o = owner(ti[:, j])
                (ring_blocks[o] if o >= 0
                 else acc_t[self.torsion_key(*keys[ti[1:3, j]])]).append(ph[j])
            for s, block in ring_blocks.items():
                # key on (signature, dof count): the number of DoF a ring system owns
                # depends on how the spanning tree threads it, which varies independently
                # of composition. Keying on signature alone lumped incompatible lengths
                # together and fit() then discarded the whole bank.
                acc_ring[(sigs[s], len(block))].append(np.asarray(block))
            self.n_fitted += 1

        def build(acc, nb, rng, periodic=False):
            out = {}
            for k, vals in acc.items():
                c, _ = np.histogram(np.asarray(vals), bins=nb, range=rng)
                out[k] = Histogram1D(c.astype(np.float64), rng[0], rng[1], periodic)
            return out

        self.bonds = build(acc_b, self.n_bins_r, R_RANGE)
        self.angles = build(acc_a, self.n_bins_theta, THETA_RANGE)
        self.torsions = build(acc_t, self.n_bins_phi, PHI_RANGE, periodic=True)
        for key, blocks in acc_ring.items():
            self.rings[key] = RingBank(np.stack(blocks), self.ring_bandwidth)

        if verbose:
            print(f"fitted on {self.n_fitted} molecules: {len(self.bonds)} bond types, "
                  f"{len(self.angles)} angle types, {len(self.torsions)} torsion types, "
                  f"{len(self.rings)} ring signatures")
        return self

    # ------------------------------------------------------------------- layout

    def _layout(self, mol):
        """Per-DoF type keys and ring-block membership for one molecule."""
        mol.build_conformer_tree()
        keys = self._atom_keys(mol)
        sysid, sigs = self.ring_systems(mol)
        bi = mol.tree_bond_index.detach().cpu().numpy()
        ai = mol.tree_angle_index.detach().cpu().numpy()
        ti = mol.tree_torsion_index.detach().cpu().numpy()

        def owner(cols):
            s = sysid[cols]
            return s[0] if (s[0] >= 0 and np.all(s == s[0])) else -1

        free, blocks = {'r': [], 'theta': [], 'phi': []}, defaultdict(lambda: defaultdict(list))
        for j in range(bi.shape[1]):
            o = owner(bi[:, j])
            if o >= 0: blocks[o]['r'].append(j)
            else: free['r'].append((j, self.bond_key(*keys[bi[:, j]])))
        for j in range(ai.shape[1]):
            o = owner(ai[:, j])
            if o >= 0: blocks[o]['theta'].append(j)
            else: free['theta'].append((j, self.angle_key(*keys[ai[:, j]])))
        for j in range(ti.shape[1]):
            o = owner(ti[:, j])
            if o >= 0: blocks[o]['phi'].append(j)
            else: free['phi'].append((j, self.torsion_key(*keys[ti[1:3, j]])))
        return free, blocks, sigs, (bi.shape[1], ai.shape[1], ti.shape[1])

    # ---------------------------------------------------------------- interface

    def sample(self, mol, n: int = 1, seed: Optional[int] = None):
        """Draw ``n`` full internal-coordinate sets for ``mol``. Returns (r, theta, phi)."""
        rng = np.random.default_rng(seed)
        free, blocks, sigs, (nb, na, nt) = self._layout(mol)
        out = {'r': np.zeros((n, nb)), 'theta': np.zeros((n, na)), 'phi': np.zeros((n, nt))}
        tables = {'r': self.bonds, 'theta': self.angles, 'phi': self.torsions}

        for kind, entries in free.items():
            for j, key in entries:
                h = tables[kind].get(key)
                if h is None:                     # unseen type: fall back to uniform
                    rng_ = {'r': R_RANGE, 'theta': THETA_RANGE, 'phi': PHI_RANGE}[kind]
                    out[kind][:, j] = rng.uniform(*rng_, size=n)
                else:
                    out[kind][:, j] = h.sample(n, self.fatten, rng)

        for s, cols in blocks.items():
            bank = self.rings.get(sigs[s])
            order = [('r', j) for j in cols['r']] + [('theta', j) for j in cols['theta']] \
                    + [('phi', j) for j in cols['phi']]
            if bank is None or bank.rows.shape[1] != len(order):
                for kind, j in order:                       # unseen ring: uniform
                    rr = {'r': R_RANGE, 'theta': THETA_RANGE, 'phi': PHI_RANGE}[kind]
                    out[kind][:, j] = rng.uniform(*rr, size=n)
                continue
            drawn = bank.sample(n, rng)
            for col, (kind, j) in enumerate(order):
                out[kind][:, j] = drawn[:, col]

        dt = mol.pos.dtype if mol.pos is not None else torch.float32
        return tuple(torch.as_tensor(out[k], dtype=dt) for k in ('r', 'theta', 'phi'))

    def log_prob(self, mol, r, theta, phi) -> torch.Tensor:
        """log q(x) for the same distribution ``sample`` draws from. ``[n]``."""
        free, blocks, sigs, _ = self._layout(mol)
        arr = {k: np.atleast_2d(np.asarray(v.detach().cpu() if torch.is_tensor(v) else v))
               for k, v in (('r', r), ('theta', theta), ('phi', phi))}
        n = arr['r'].shape[0]
        total = np.zeros(n)
        tables = {'r': self.bonds, 'theta': self.angles, 'phi': self.torsions}
        spans = {'r': R_RANGE, 'theta': THETA_RANGE, 'phi': PHI_RANGE}

        for kind, entries in free.items():
            for j, key in entries:
                h = tables[kind].get(key)
                if h is None:
                    lo, hi = spans[kind]
                    total += -np.log(hi - lo)
                else:
                    total += h.log_prob(arr[kind][:, j], self.fatten)

        for s, cols in blocks.items():
            bank = self.rings.get(sigs[s])
            order = [('r', j) for j in cols['r']] + [('theta', j) for j in cols['theta']] \
                    + [('phi', j) for j in cols['phi']]
            if bank is None or bank.rows.shape[1] != len(order):
                for kind, j in order:
                    lo, hi = spans[kind]
                    total += -np.log(hi - lo)
                continue
            block = np.stack([arr[kind][:, j] for kind, j in order], axis=1)
            total += bank.log_prob(block)
        return torch.as_tensor(total)
