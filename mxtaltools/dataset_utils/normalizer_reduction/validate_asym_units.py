"""
Fundamental-domain identification, sanity layer: for every space group MXtalTools has
a real (non-[1,1,1]-placeholder) box for in constants/asymmetric_units.py, Monte Carlo
check whether that box is actually a valid fundamental domain of G (SYM_OPS[sg_ind]) --
i.e. for a generic point, exactly one of the |G| symmetry-equivalent images should land
inside the box. Zero images landing = the box has a gap (identify_canonical_asymmetric_unit
would fall back to its "nearest to origin" rescue path and mark well_defined=False).
More than one landing = the box overlaps itself (also well_defined=False, tie-break path).

This doesn't touch the normalizer/reduction question at all -- it's one level down,
just asking "is the existing G-level box even self-consistent" -- but it's a cheap,
fully-automatic check worth running before building anything on top of a given
space group's box, and it's real data (pulled from the actual repo), not guessed.
"""
import numpy as np
import importlib.util
import sys

from mxtaltools.constants.asymmetric_units import RAW_ASYM_UNITS
from mxtaltools.constants.space_group_info import SYM_OPS, SPACE_GROUPS


rng = np.random.default_rng(0)
N_SAMPLES = 20000

results = []
for sg_str, box in sorted(RAW_ASYM_UNITS.items(), key=lambda kv: int(kv[0])):
    sg_ind = int(sg_str)
    if sg_ind not in SYM_OPS:
        continue
    ops = np.array(SYM_OPS[sg_ind], dtype=np.float64)  # (|G|, 4, 4)
    R = ops[:, :3, :3]
    t = ops[:, :3, 3]
    box = np.array(box, dtype=np.float64)

    pts = rng.uniform(0, 1, size=(N_SAMPLES, 3))
    # images: (|G|, N, 3)
    images = np.einsum('gij,nj->gni', R, pts) + t[:, None, :]
    images -= np.floor(images)
    inside = np.all((images >= 0) & (images <= box[None, None, :]), axis=-1)  # (|G|, N)
    n_inside = inside.sum(axis=0)  # (N,)

    frac_exactly_one = np.mean(n_inside == 1)
    frac_zero = np.mean(n_inside == 0)
    frac_multi = np.mean(n_inside > 1)
    vol_check = box.prod() * len(ops)  # should be ~1.0

    results.append((sg_ind, SPACE_GROUPS.get(sg_ind, '?'), len(ops), vol_check,
                    frac_exactly_one, frac_zero, frac_multi))

print(f"{'sg':>4} {'name':<10} {'|G|':>4} {'vol*|G|':>8} {'=1':>7} {'=0':>7} {'>1':>7}")
n_bad = 0
for sg_ind, name, order, vol_check, f1, f0, fm in results:
    flag = "" if (abs(vol_check - 1) < 1e-6 and f1 > 0.999) else "  <-- CHECK"
    if flag:
        n_bad += 1
    print(f"{sg_ind:>4} {name:<10} {order:>4} {vol_check:>8.4f} {f1:>7.3f} {f0:>7.3f} {fm:>7.3f}{flag}")

print(f"\n{len(results)} space groups checked (those with real box data in RAW_ASYM_UNITS).")
print(f"{n_bad} flagged for a volume mismatch or <99.9% single-image coverage.")
