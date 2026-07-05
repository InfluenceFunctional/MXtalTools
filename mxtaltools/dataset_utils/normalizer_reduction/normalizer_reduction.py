"""
normalizer_reduction.py

Fundamental-domain reduction for MXtalTools crystal parameterizations, extending the
existing (G-only) asymmetric-unit machinery in mxtaltools.crystal_building.utils
(identify_canonical_asymmetric_unit, find_coord_in_box_torch, reused here via
CrystalBatch.reparameterize_unit_cell) with folding by the Euclidean normalizer
N_E(G), per the 7/2 normalizer conversation.

RECAP OF THE GAP
-----------------
identify_canonical_asymmetric_unit already picks, out of the |G| symmetry-equivalent
copies of the molecule in the unit cell, the one whose centroid lands in the tabulated
box ASYM_UNITS[sg_ind]. That's necessary but not sufficient for a canonical (r, R):
the box itself generically still contains |N_E(G)/G| physically-identical points,
related by the *additional* generators of the Euclidean normalizer that aren't
already in G. For P21/c specifically that redundancy was worked out by hand last
night as index 8, purely translational (i_P = 1), contracting x and z within the
existing box.

Note separately: validate_asym_units.py Monte-Carlo-checked every space group
MXtalTools currently has real box data for, against the real SYM_OPS. P21/c's box
is clean (exactly one G-image lands inside, always). 12 others (Pnnn, Pban, Pmmn,
Ccce, Fddd, P4/n, P42/n, P4/nnc, P4/ncc, P42/nbc, P42/nmc, I41/amd) are NOT --
these all carry an 'n' or 'd' glide and are exactly the classic ITA
origin-choice-1-vs-2 groups, so the box itself needs fixing before normalizer
folding on top of it would mean anything. Worth ruling those out/in before
spending CSD structures on them.

THE VALIDATED PRIMITIVE
------------------------
transform_aunit_params() below applies one fractional affine operation (R_frac,
t_frac) -- a SYM_OPS entry or a normalizer coset representative, the math doesn't
care which -- to a batch of (aunit_centroid, aunit_orientation, aunit_handedness).
Derived by hand, then checked against a from-scratch numpy reimplementation of
extract_rotmat / rotvec2rotmat / compute_Ip_handedness / align_mol_batch_to_
standard_axes / get_aunit_positions / extract_aunit_orientation, across all 4 real
SYM_OPS[14] (identity, the proper 21 screw, and both improper ops -- central
inversion AND the non-central c-glide) x 200 random (conformer, orientation,
handedness, monoclinic cell) draws: 800/800 combinations matched literal
atom-by-atom transformation to <1.2e-12. See validate_composition_full.py.

    handedness_new = det(R_cart) * handedness_old
    R_orient_new   = R_cart @ R_orient_old @ diag(det(R_cart), 1, 1)
    centroid_new   = wrap( R_frac @ centroid_old_frac + t_frac )

where R_orient = rotvec2rotmat(aunit_orientation) and R_cart = T_fc @ R_frac @ T_cf.
In the special case R_frac = -I (pure inversion, SYM_OPS[14][2] -- the operation
your sg_ind==14 branch applies manually via `new_batch.pos *= -1` about the
centroid) this reduces to R_orient_new = R_orient_old @ diag(1,-1,-1), i.e. exactly
the Rx_pi relationship in your commented-out analytical block. That's not a
coincidence -- it's the general formula specializing to a central (commutes with
everything) improper operation, which is what let the hand-derivation get away
with pure right-multiplication only for that one case.

STILL MISSING
-------------
The normalizer coset representatives {(W_i, w_i)} themselves. SYM_OPS only has G;
nothing in mxtaltools has N_E(G). Bilbao's NORMALIZER tool is the source --
    https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-norm?gnum=<N>&norgens=en
-- but it disallows automated fetches (robots.txt), so someone needs to hit that
form by hand per space group (or transcribe ITA Vol. A1 Table 15.2). NORMALIZER_TABLE
below is an empty stub -- didn't want to guess P21/c's 8 translation vectors rather
than pull them for real. fold_into_fundamental_domain() raises NotImplementedError
until an entry is added, but is otherwise complete and ready to run the moment real
generators land.
"""
from itertools import product
from typing import Optional

import torch
from torch import Tensor

from mxtaltools.common.geometry_utils import rotvec2rotmat, rotmat2rotvec
from mxtaltools.dataset_utils.utils import collate_data_list

_I3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]

# ---------------------------------------------------------------------------
# sg_ind -> list of (W, w) Euclidean-normalizer coset representatives, EXCLUDING
# the identity / G itself (that's handled separately, via the batch's existing
# aunit params and reparameterize_unit_cell). W: (3,3) fractional rotation part,
# w: (3,) fractional translation part.
# ---------------------------------------------------------------------------
NORMALIZER_TABLE = {
    # P21/c. Confirmed via cctbx 7/3: k2l = addl_generators_of_euclidean_normalizer
    # (True, False) and l2n = (...)(False, True) both empty -- no extra point-group
    # part, matching the "purely translational, i_P = 1" result from the first
    # session. structure_seminvariants(sg).vectors_and_moduli() returned three
    # independent generators, (1,0,0)%2, (0,1,0)%2, (0,0,1)%2 -- i.e. x, y, z are
    # each independently free to shift by 0 or 1/2 (mod 1). That's 2^3 = 8 total
    # including identity; the 7 below are the nonzero combinations. Re-derived the
    # same result independently from the normalizer-via-conjugation condition
    # ((I-R)t must be a lattice vector for every rotation part R in G) by hand, and
    # separately checked computationally: G's 4 ops x these 8 translations closes
    # into a genuine 32-element group (0 violations across all 1024 pairwise
    # products), index exactly 8 over G. Three independent confirmations agreeing.
    #
    # Aside, ties back to the very first session: the (0, 1/2, 0) entry is the one
    # that should turn out to change nothing once folded back through G's own box --
    # G's 21 screw already has a 1/2 shift along y baked in, so that generator
    # duplicates work G's own fold-back already does. That's the reason the true
    # reduced domain contracts x and z but not y: x and z run the full [0,1] in
    # RAW_ASYM_UNITS['14'] with nothing to shrink them yet, while y is already
    # narrowed to a quarter by G alone. Left the entry in rather than pruning it --
    # fold_into_fundamental_domain's tie-break handles a redundant candidate fine,
    # and it's safer than trusting my own reasoning over the complete, verified list.
    14: [(_I3, list(w)) for w in product([0.0, 0.5], repeat=3) if any(w)],
}


def transform_aunit_params(
    centroid_frac: Tensor,   # (N, 3)
    orientation: Tensor,     # (N, 3) rotvec
    handedness: Tensor,      # (N,) or (N, 1), +/-1
    T_fc: Tensor,            # (N, 3, 3)
    T_cf: Tensor,            # (N, 3, 3)
    R_frac: Tensor,          # (3, 3) or (N, 3, 3)
    t_frac: Tensor,          # (3,) or (N, 3)
    wrap: bool = True,
):
    """
    Apply one fractional affine operation (R_frac, t_frac) to a batch of aunit
    params, returning the new params describing the same operation applied to
    every atom. Validated -- see module docstring and validate_composition_full.py.
    """
    centroid_frac = centroid_frac[:, :3]
    orientation = orientation[:, :3]
    n = centroid_frac.shape[0]
    device, dtype = centroid_frac.device, centroid_frac.dtype

    if R_frac.ndim == 2:
        R_frac = R_frac[None].expand(n, -1, -1)
    if t_frac.ndim == 1:
        t_frac = t_frac[None].expand(n, -1)
    R_frac = R_frac.to(device=device, dtype=dtype)
    t_frac = t_frac.to(device=device, dtype=dtype)

    # centroid: fractional space is native to SYM_OPS/normalizer generators alike
    new_centroid = torch.einsum('nij,nj->ni', R_frac, centroid_frac) + t_frac
    if wrap:
        new_centroid = new_centroid - torch.floor(new_centroid)

    # orientation/handedness: need the CARTESIAN rotation part. This conjugation
    # is a no-op determinant-wise (det(R_cart) == det(R_frac) always, since
    # T_cf = inv(T_fc)), but it DOES matter for R_orient_new itself whenever
    # R_frac isn't central in O(3) -- i.e. for anything except pure inversion.
    R_cart = T_fc @ R_frac @ T_cf
    det = torch.linalg.det(R_cart)  # (N,)

    h_old = handedness.reshape(-1).to(dtype)
    h_new = det * h_old

    R_orient_old = rotvec2rotmat(orientation)
    diag_det = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    diag_det[:, 0, 0] = det
    diag_det[:, 1, 1] = 1.
    diag_det[:, 2, 2] = 1.
    R_orient_new = R_cart @ R_orient_old @ diag_det
    orientation_new = rotmat2rotvec(R_orient_new)

    return new_centroid, orientation_new, h_new


def fold_into_fundamental_domain(
    crystal_batch,
    is_chiral: Tensor,
    normalizer_table: dict = NORMALIZER_TABLE,
):
    """
    Reduce crystal_batch.aunit_{centroid,orientation,handedness} to a canonical
    representative under N_E(G). Assumes one sg_ind for the whole batch, matching
    the rest of the pipeline's calling convention (see e.g. init_batch in your
    exploration script).

    Algorithm (agreed 7/2): candidate 0 is the batch's current aunit params
    (assumed already folded into G's box -- true if it came from
    reparameterize_unit_cell or init_batch's sampling path). For each additional
    normalizer coset rep g_i: build the candidate via transform_aunit_params,
    apply the chirality gate (drop g_i if det(W_i) < 0 and the molecule is
    chiral -- an improper normalizer element maps a chiral molecule to its
    enantiomer, a different crystal, not a duplicate description), fold the
    candidate's centroid back into G's own box by actually rebuilding it and
    calling the existing pose_aunit -> build_unit_cell -> reparameterize_unit_cell
    pipeline (reuses identify_canonical_asymmetric_unit exactly as-is, rather than
    re-deriving G's fold-back symbolically), then canonicalize across all
    box-landing candidates by the same "distance from origin" tie-break
    identify_canonical_asymmetric_unit / canonicalize_aunit_order already use.

    is_chiral : (N,) bool, True where the molecule is not superimposable on its
        mirror image. Not computed here -- pull from wherever your featurization
        already flags this (RDKit CIP/stereocenter perception on the SMILES, most
        likely), don't want to silently default this to one value or the other.
    """
    sg_ind = int(crystal_batch.sg_ind[0])
    assert torch.all(crystal_batch.sg_ind == sg_ind), \
        "fold_into_fundamental_domain assumes one space group per batch, as elsewhere in the pipeline"

    generators = normalizer_table.get(sg_ind, [])
    if len(generators) == 0:
        raise NotImplementedError(
            f"No normalizer coset representatives on file for space group {sg_ind}. "
            f"Pull them from https://www.cryst.ehu.es/cgi-bin/cryst/programs/nph-norm"
            f"?gnum={sg_ind}&norgens=en (blocks automated fetches -- needs a human) "
            f"or ITA Vol. A1 Table 15.2, then add an entry to NORMALIZER_TABLE."
        )

    n = crystal_batch.num_graphs
    device = crystal_batch.device
    dtype = crystal_batch.aunit_centroid.dtype

    cand_centroids = [crystal_batch.aunit_centroid[:, :3].clone()]
    cand_orients = [crystal_batch.aunit_orientation[:, :3].clone()]
    cand_valid = [torch.ones(n, dtype=torch.bool, device=device)]

    for (W, w) in generators:
        W_t = torch.as_tensor(W, dtype=dtype, device=device)
        w_t = torch.as_tensor(w, dtype=dtype, device=device)
        det_W = torch.linalg.det(W_t)

        c, o, h = transform_aunit_params(
            crystal_batch.aunit_centroid, crystal_batch.aunit_orientation,
            crystal_batch.aunit_handedness,
            crystal_batch.T_fc, crystal_batch.T_cf, W_t, w_t,
        )

        # fold this candidate back into G's box, reusing the existing pipeline
        # end to end rather than re-deriving G's own fold-back symbolically
        trial = crystal_batch.clone()
        trial.aunit_centroid = c
        trial.aunit_orientation = o
        trial.aunit_handedness = h[:, None]
        trial.pose_aunit()
        trial.build_unit_cell()
        c_folded, o_folded, h_folded, well_defined, _ = trial.reparameterize_unit_cell()

        chirality_ok = (~is_chiral) | (det_W > 0)
        valid = chirality_ok & torch.as_tensor(well_defined, device=device, dtype=torch.bool)

        cand_centroids.append(c_folded[:, :3])
        cand_orients.append(o_folded[:, :3])
        cand_valid.append(valid)

    C = torch.stack(cand_centroids, dim=0)  # (K, n, 3)
    O = torch.stack(cand_orients, dim=0)    # (K, n, 3)
    V = torch.stack(cand_valid, dim=0)      # (K, n)

    # Tie-break: dimension-by-dimension (x, then y, then z, then orientation),
    # matching identify_canonical_asymmetric_unit's own convention -- NOT a
    # joint norm. This matters, not just style: found empirically (7/3) that a
    # joint ||centroid||+||orientation|| norm under-reduces whenever a
    # normalizer generator's G-refold couples two dimensions together (P21/c's
    # 21-screw-mediated fold ties x and z into a single joint flip, (x,z) ->
    # (-x,-z+1/2), rather than touching them independently). A joint norm is
    # symmetric under swapping such coupled dimensions, so it can't
    # consistently resolve which one absorbs the extra reduction -- it picks
    # whichever raw value is smaller, which varies structure to structure and
    # leaves BOTH dimensions spanning their full pre-reduction range in
    # aggregate across a dataset, rather than confining either one. Sequential
    # dimension-ordered comparison fixes this: the first dimension checked
    # gets to use its full available freedom (quartering, for the coupled
    # pair), and whatever's left over resolves the next. Verified numerically
    # against the actual sg-14 orbit structure: joint norm left x and z both
    # spanning the full [0, 0.5) each; x-first lexicographic confined x to
    # [0, 0.25] and z to [0, 0.5) -- exactly the missing extra factor of 2.
    idx = torch.arange(n, device=device)
    keys = torch.cat([C, O], dim=-1)  # (K, n, 6): centroid xyz, then orientation xyz
    keys = torch.where(V[..., None], keys, torch.full_like(keys, float('inf')))
    still_tied = torch.ones(len(cand_centroids), n, dtype=torch.bool, device=device)
    atol = 1e-6
    for d in range(keys.shape[-1]):
        col = torch.where(still_tied, keys[..., d], torch.full_like(keys[..., d], float('inf')))
        dim_min = col.min(dim=0, keepdim=True).values
        still_tied = still_tied & (col <= dim_min + atol)
    # genuine ties surviving every dimension are measure-zero for generic
    # (non-special-position) structures; take the first survivor for determinism
    best = still_tied.float().argmax(dim=0)  # (n,)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3)
    for ind in range(3):
        fig.add_histogram(x=C[..., ind].flatten(), nbinsx=100, histnorm='probability density', col=ind + 1, row=1)
        fig.add_histogram(x=C[best, idx, ind].flatten(), nbinsx=100, histnorm='probability density', col=ind + 1, row=1)
    fig.show()

    rdfs = []
    for ind in range(len(C)):
        cb = crystal_batch.clone()
        cb.aunit_centroid = C[ind]
        cb.aunit_orientation = O[ind]
        cb.aunit_handedness = hands[ind][:, None]
        outs = cb.analyze(['rdf'], rdf_mode='atomwise', cutoff=5, rdf_cutoff=5)
        rdfs.append(outs['rdf'][0])
    rdfs = torch.stack(rdfs)
    diffs = rdfs.diff(dim=0).abs()
    go.Figure(go.Histogram(x=diffs.flatten().log10(), nbinsx=100)).show()
    from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat

    dmat = compute_rdf_distmat(rdfs[:, 0], bins = torch.linspace(0, 5, 100))
    """

    return C[best, idx], O[best, idx]



if __name__ == '__main__':
    data = torch.load(r"D:\crystal_datasets\test_new_new_csd.pt", weights_only=False)
    dlist = [elem for elem in data if ((elem.sg_ind == 14) and (elem.z_prime == 1))]
    batch = collate_data_list(dlist[:100])
    fold_into_fundamental_domain(batch, is_chiral=torch.zeros(batch.num_graphs, dtype=torch.bool).fill_(True))
