"""The mirror image (enantiomorph) of Z' = 1 crystals of Sohncke groups, written in the standard setting.

Owner decision 2026-10-21: a Z' = 1 row of a Sohncke group that ``domains.to_sheet`` leaves off the requested sheet is
stored as its mirror-image crystal, with the aligned molecule ``X`` itself as the reference (``io_mxt``,
``off_sheet='enantiomorph'``). Energies are required to be mirror-invariant.

The map. Invert the crystal through the origin, r -> -r. The lattice goes onto itself, so the basis and the cell are
unchanged and fractional coordinates go x -> -x. The unit at ``T t + R diag(h, 1, 1) X`` goes to
``T (-t) - R diag(h, 1, 1) X = T (-t) + R diag(1, -1, -1) diag(-h, 1, 1) X``: t -> -t, R -> R diag(1, -1, -1) (a sign
change of two columns: exact, still proper), h -> -h, X unchanged. An operator (W, w) of G becomes (W, -w): the image
is a crystal of {(W, -w)} in the same basis, which a change of setting ``x_inv = P x_out + p``
(``settings.find_setting``) takes to ``SYM_OPS[sg_out]``. sg_out is sg when Gamma (the Euclidean normaliser of G,
``tables.group_table``) has a det -1 element, 43 of the 65 groups, and otherwise the group's ITA enantiomorphic partner
(``PARTNER``: 11 pairs, 22 groups; the pair list is checked against the table). A row becomes

    sg' = sg_out,   cell' = cell,   t' = (-t - p) mod 1 in [0, 1),   R' = R diag(1, -1, -1),   h' = -h,   X' = X

when P = I, and goes through ``settings.re_express`` otherwise. ``enantiomorph_map`` derives (sg_out, P, p) from the
tables at first use: the current tables give P = I in all 65 groups and p = 0 except sg 80 (0, 1/2, 0), 98 (0, 1/2,
1/4) and 210 (1/4, 1/4, 1/4). Where sg_out = sg the map is the element (-I, -p) of Gamma (checked).

Use. Apply the map to rows in the standard setting (``io_mxt`` does, after re-expressing nonstandard rows, D1) and
canonicalise after it: it keeps the standard setting, not ASU(Gamma+). Applied twice it returns sg, cell, R and h
exactly and t to rounding (about 1e-16). The wrap is strict: ``x - floor(x)`` rounds to exactly 1.0 for x = -t - p
a negative number below half an ulp of 1 in magnitude (t = 1e-17 or a denormal where p = 0), which is set to 0.
"""
from functools import lru_cache
from typing import NamedTuple, Optional

import numpy as np
import torch

from mxtaltools.crystal_fd import settings
from mxtaltools.crystal_fd.description import CrystalDescription
from mxtaltools.crystal_fd.tables import group_table

F64 = torch.float64

#: the 11 enantiomorphic pairs of Sohncke space-group types (ITA)
ENANTIOMORPHIC_PAIRS = ((76, 78), (91, 95), (92, 96), (144, 145), (151, 153), (152, 154), (169, 170), (171, 172),
                        (178, 179), (180, 181), (212, 213))
#: sg -> its enantiomorphic partner, for the 22 groups of ``ENANTIOMORPHIC_PAIRS``
PARTNER = {**{a: b for a, b in ENANTIOMORPHIC_PAIRS}, **{b: a for a, b in ENANTIOMORPHIC_PAIRS}}
#: the inversion's factor on R: -R diag(h, 1, 1) = R diag(1, -1, -1) diag(-h, 1, 1)
FLIP = (1.0, -1.0, -1.0)
#: largest deviation (mod 1) accepted when finding (-I, -p) in Gamma
_GAMMA_TOL = 1e-9


class EnantiomorphMap(NamedTuple):
    """How the enantiomorph of a crystal of ``sg`` is written (``enantiomorph_map``).

    ``sg_out``: its space group (sg, or the partner); ``P`` [3, 3], ``p`` [3]: float64 numpy arrays, read-only, the
    change of setting ``x_inv = P x_out + p`` from the inverted group {(W, -w)} to ``SYM_OPS[sg_out]``;
    ``gamma_index``: index into ``group_table(sg).gamma_*`` of the element (-I, -p mod 1) (free axes ignored) where
    sg_out = sg and P = I, else None.
    """
    sg: int
    sg_out: int
    P: np.ndarray
    p: np.ndarray
    gamma_index: Optional[int]

    @property
    def identity_basis(self) -> bool:
        """P = I: the cell is kept and t' = (-t - p) mod 1."""
        return bool(np.array_equal(self.P, np.eye(3)))


def inverted_ops(sg: int) -> np.ndarray:
    """[n, 4, 4] float64: G = ``group_table(sg)``'s exact operators conjugated by the inversion, (W, w) -> (W, -w)."""
    tab = group_table(sg)
    n = tab.g_W.shape[0]
    o = np.zeros((n, 4, 4))
    o[:, :3, :3] = tab.g_W.numpy()
    o[:, :3, 3] = -tab.g_w.numpy()
    o[:, 3, 3] = 1.0
    return o


@lru_cache(maxsize=None)
def enantiomorph_map(sg: int) -> EnantiomorphMap:
    """The ``EnantiomorphMap`` of Sohncke group ``sg``, derived from the tables and ``settings.find_setting``.

    Raises
    ------
    ValueError
        ``sg`` outside 1-230, or not a Sohncke group (G has an improper element: the mirror image is the same crystal,
        which ``domains.to_sheet`` reaches on the other sheet).
    RuntimeError
        The table disagrees with ``PARTNER`` (an improper element of Gamma in a paired group, or none in an unpaired
        one); no change of setting takes the inverted group to ``SYM_OPS[sg_out]``; or (-I, -p) is not in Gamma.
    """
    from mxtaltools.crystal_fd.io_mxt import is_sohncke
    sg = int(sg)
    if not 1 <= sg <= 230:
        raise ValueError(f'space group {sg} is not in 1-230')
    if not is_sohncke(sg):
        raise ValueError(f'sg {sg} is not a Sohncke group: an improper element of G reaches the other sheet '
                         '(domains.to_sheet), and its mirror image is the same crystal')
    tab = group_table(sg)
    improper = bool((tab.gamma_det < 0).any())
    if improper == (sg in PARTNER):
        raise RuntimeError(f'sg {sg}: Gamma has {"an" if improper else "no"} improper element, but the pair list says '
                           f'{"paired" if sg in PARTNER else "unpaired"}')
    sg_out = sg if improper else PARTNER[sg]
    found = settings.find_setting(inverted_ops(sg), sg_out)
    if found is None:
        raise RuntimeError(f'sg {sg}: no change of setting takes the inverted group to SYM_OPS[{sg_out}]')
    P = np.array(found.P, dtype=np.float64)
    p = np.array(found.p, dtype=np.float64)
    gamma_index = None
    if sg_out == sg and np.array_equal(P, np.eye(3)):
        W = tab.gamma_W.numpy()
        d = tab.gamma_w.numpy() + p[None]                       # gamma_w = -p (mod 1, free axes zero)
        d = np.abs(d - np.round(d))
        d[:, tab.free.numpy()] = 0.0
        hit = np.nonzero(np.all(W == -np.eye(3), axis=(1, 2)) & (d.max(1) <= _GAMMA_TOL))[0]
        if not len(hit):
            raise RuntimeError(f'sg {sg}: (-I, {(-p).tolist()}) is not an element of Gamma')
        gamma_index = int(hit[0])
    P.setflags(write=False)
    p.setflags(write=False)
    return EnantiomorphMap(sg, sg_out, P, p, gamma_index)


def _wrap(x: torch.Tensor) -> torch.Tensor:
    """x mod 1 in [0, 1) strictly: ``x - floor(x)`` rounds to 1.0 for x a tiny negative number, which goes to 0."""
    w = x - torch.floor(x)
    return torch.where(w >= 1, w - 1, w)


def _invert(d: CrystalDescription, emap: EnantiomorphMap, via_re_express: bool = False):
    """(cell, t, R, h) of unit 0 of the Z' = 1 rows ``d`` (float64, all of ``emap.sg``) after the map;
    ``via_re_express`` takes the ``settings.re_express`` path even where P = I (it recomputes the cell from the
    metric)."""
    flip = torch.tensor(FLIP, dtype=F64, device=d.device)
    t = -d.t[:, 0]
    R = d.R[:, 0] * flip                                        # R diag(1, -1, -1): columns 1 and 2 negated
    h = -d.h[:, 0]
    if emap.identity_basis and not via_re_express:
        return d.cell, _wrap(t - torch.tensor(emap.p.tolist(), dtype=F64, device=d.device)), R, h
    inv = CrystalDescription.from_parts(d.sg, d.cell, t, R, h)
    out = settings.re_express(inv, np.array(emap.P), np.array(emap.p), wrap=True)
    return out.cell, _wrap(out.t[:, 0]), out.R[:, 0], h


def enantiomorph(desc: CrystalDescription) -> CrystalDescription:
    """The enantiomorph of every row (module docstring): float64, space groups mixed.

    Every row must be Z' = 1 (``NotImplementedError`` otherwise) and of a Sohncke group (``ValueError`` from
    ``enantiomorph_map``); both are checked before any row is mapped. Unit slots past the first are returned as they
    came. Per row: ``sg`` the map's ``sg_out``; ``cell`` unchanged bit for bit (P = I); ``t`` in [0, 1); ``R`` proper;
    ``h`` negated. The reference molecule is unchanged: the caller keeps it.
    """
    d = desc.double()
    if bool((d.z_prime != 1).any()):
        raise NotImplementedError("Z' > 1: the enantiomorph is built for Z' = 1 rows")
    groups = torch.unique(d.sg).tolist()
    maps = {sg: enantiomorph_map(sg) for sg in groups}
    sg_out, cell, t, R, h = d.sg.clone(), d.cell.clone(), d.t.clone(), d.R.clone(), d.h.clone()
    for sg in groups:
        idx = torch.nonzero(d.sg == sg)[:, 0]
        c_i, t_i, R_i, h_i = _invert(d[idx], maps[sg])
        sg_out[idx] = maps[sg].sg_out
        cell[idx] = c_i
        t[idx, 0] = t_i
        R[idx, 0] = R_i
        h[idx, 0] = h_i
    return d.replace(sg=sg_out, cell=cell, t=t, R=R, h=h)
