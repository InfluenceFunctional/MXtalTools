"""Periodic images and intermolecular atom pairs of Z' = 1 crystals, without writing out a cluster.

``MolCrystalData.mol2cluster`` followed by ``construct_radial_graph`` instantiates every
atom of every candidate unit cell and then runs a radius search. The functions here return
the same intermolecular pair list from three small per-crystal quantities: the cell matrix,
the molecule rotation and the fractional centroid. With H = ``T_fc``, R the rotation of
``aunit_orientation``, c' the clipped fractional centroid, p the molecule about its
heavy-atom centroid and (W_k, w_k) symmetry operator k,

    image atom  =  H (g_k + n)  +  Q_k R p          Q_k = H W_k H^-1
    g_k         =  wrap(W_k c' + w_k)

for an integer lattice translation n. The reference molecule is operator 0 at n = 0, as in
``crystal_building.utils.ucell2cluster``.

Four conventions of the existing builder are reproduced, and each changes the pair list if
dropped: the molecule is centred on its heavy-atom centroid (``geometry_utils.center_batch``);
the fractional centroid is clipped to [0, 1 - 1e-4] (``MolCrystalOps.assign_aunit_centroid``);
image centroids are wrapped into the cell except that an exact 1.0 stays (``aunit2ucell``);
and an image molecule is kept when its all-atom centroid lies within
``cutoff + 2 * radius + 0.1`` of the posed asymmetric unit's all-atom centroid
(``_pare_cluster_molwise``).

Two things differ from the existing route. There is no cap on neighbours per atom
(``construct_radial_graph`` keeps at most 1000), so the list is complete wherever the
lattice-translation window is; and pairs at exactly the cutoff may fall on the other side of
the comparison through float32 rounding. Agreement elsewhere is pinned by
``tests/test_image_pairs.py``.

Z' = 1 only.
"""
from dataclasses import dataclass

import numpy as np
import torch

from mxtaltools.common.geometry_utils import rotvec2rotmat

__all__ = ['ImageTables', 'build_image_tables', 'select_images', 'pair_distances', 'image_pairs']


@dataclass
class ImageTables:
    """What depends on the molecule and space group only; build once, reuse at every cell and pose."""
    p: torch.Tensor        # [B, A, 3] molecule coordinates about the heavy-atom centroid, zero padded
    amask: torch.Tensor    # [B, A]    True on real atoms
    z: torch.Tensor        # [B, A]    atomic numbers, zero padded
    nat: torch.Tensor      # [B]       atoms per molecule
    W: torch.Tensor        # [B, Z, 3, 3] rotation part of each symmetry operator (fractional)
    w: torch.Tensor        # [B, Z, 3]    translation part
    kmask: torch.Tensor    # [B, Z]    True on real operators
    radius: torch.Tensor   # [B]       molecule radius as stored on the batch
    ptr: torch.Tensor      # [B]       offset of each molecule's first atom among the batch's real atoms


def build_image_tables(crystal_batch) -> ImageTables:
    """Tables for a collated Z' = 1 ``MolCrystalData`` batch.

    Reads ``pos``, ``z``, ``batch``, ``ptr``, ``num_atoms``, ``radius`` and
    ``symmetry_operators``. ``pos`` may be in any frame and position; only its shape about
    the heavy-atom centroid is kept, which is what ``pose_aunit(std_orientation=False)``
    rotates.
    """
    if int(torch.as_tensor(crystal_batch.z_prime).max()) != 1:
        raise NotImplementedError("image_pairs covers Z' = 1 crystals only")
    dev, dtype = crystal_batch.pos.device, crystal_batch.pos.dtype
    B = crystal_batch.num_graphs
    batch = crystal_batch.batch
    nat = crystal_batch.num_atoms.long().to(dev)
    ptr = crystal_batch.ptr[:-1].to(dev)
    A = int(nat.max())
    slot = torch.arange(len(batch), device=dev) - ptr[batch]

    zz = crystal_batch.z.long()
    z = torch.zeros(B, A, dtype=torch.long, device=dev)
    z[batch, slot] = zz
    amask = torch.zeros(B, A, dtype=torch.bool, device=dev)
    amask[batch, slot] = True

    heavy = zz > 1
    cen = torch.zeros(B, 3, dtype=dtype, device=dev).index_add_(0, batch[heavy], crystal_batch.pos[heavy])
    cnt = torch.zeros(B, dtype=dtype, device=dev).index_add_(
        0, batch[heavy], torch.ones(int(heavy.sum()), dtype=dtype, device=dev))
    p = torch.zeros(B, A, 3, dtype=dtype, device=dev)
    p[batch, slot] = crystal_batch.pos - (cen / cnt[:, None])[batch]

    ops = crystal_batch.symmetry_operators
    Z = max(len(o) for o in ops)
    ops_np = np.zeros((B, Z, 4, 4), dtype=np.float32)
    kmask_np = np.zeros((B, Z), dtype=bool)
    for i, o in enumerate(ops):
        ops_np[i, :len(o)] = o
        kmask_np[i, :len(o)] = True
    ops_t = torch.from_numpy(ops_np).to(dev)
    return ImageTables(p=p, amask=amask, z=z, nat=nat,
                       W=ops_t[..., :3, :3].contiguous(), w=ops_t[..., :3, 3].contiguous(),
                       kmask=torch.from_numpy(kmask_np).to(dev),
                       radius=crystal_batch.radius.to(dev).to(dtype).flatten(), ptr=ptr)


def _nearest_in_group(group, dist, k: int):
    """Mask of the entries that are among the ``k`` smallest ``dist`` of their ``group``.

    Ties keep the earlier entry. The mask is in the caller's order, so filtering with it
    leaves the survivors where they were and a group with at most ``k`` entries untouched.
    """
    by_dist = torch.argsort(dist, stable=True)
    order = by_dist[torch.argsort(group[by_dist], stable=True)]          # by group, then by distance
    g = group[order]
    rank = torch.arange(g.numel(), device=g.device) - torch.searchsorted(g, g)
    keep = torch.zeros(g.numel(), dtype=torch.bool, device=g.device)
    keep[order[rank < k]] = True
    return keep


def select_images(tables: ImageTables, T_fc, T_cf, aunit_centroid, aunit_orientation,
                  cutoff: float, max_translations: int = 10, max_images: int = None) -> dict:
    """Image molecules that can hold an atom within ``cutoff`` of the reference molecule.

    Works on molecule centres only. The lattice-translation window is set per crystal and
    per axis: an image left out of a window of half-width N has a fractional offset of at
    least N + 1/2 on some axis, hence is at least (N + 1/2) lattice-plane spacings away, so
    N = floor(rho / spacing + 1/2) with rho = ``cutoff + 2 * radius + 0.1`` leaves nothing
    within rho outside the window. N is clamped at ``max_translations``; ``capped`` marks
    the crystals where that clamp was hit and the list may be incomplete.

    ``max_images``, if given, bounds the images kept per crystal: a crystal with more keeps
    the ``max_images`` whose centres are nearest the reference molecule's, and is marked in
    ``image_capped``. A cell squeezed far below any physical density has tens of thousands
    of images in range; the bound is what keeps the pair search that follows finite there.
    A crystal under the bound is returned exactly as without one.

    Returns a dict: ``graph`` [P] crystal index and ``op`` [P] operator index of each kept
    image, ``rel`` [P, 3] its centre relative to the reference molecule's centre, ``skel``
    [B, Z, A, 3] the atom offsets of each operator's copy about its own centre, ``capped``
    [B], ``image_capped`` [B], ``images_kept`` [B] and ``centres_examined`` [B]. ``rel`` and
    ``skel`` carry gradients to the cell and pose arguments; the selection itself does not.
    """
    st = tables
    B = T_fc.shape[0]
    dev = T_fc.device
    H, Hi = T_fc, T_cf
    R = rotvec2rotmat(aunit_orientation[:, :3]).to(H.dtype)
    c = aunit_centroid[:, :3]
    cp = c.clip(min=0, max=1 - 1e-4)

    raw = torch.einsum('bkij,bj->bki', st.W, cp) + st.w                  # [B, Z, 3], before the wrap
    g = torch.where(raw != 1.0, raw - torch.floor(raw), raw)

    Rp = torch.einsum('bij,baj->bai', R, st.p)                           # [B, A, 3]
    Q = H[:, None] @ st.W @ Hi[:, None]                                  # [B, Z, 3, 3]
    skel = torch.einsum('bkij,baj->bkai', Q, Rp)                         # [B, Z, A, 3]

    with torch.no_grad():
        am = st.amask[:, None, :, None].to(H.dtype)
        skel_mean = (skel * am).sum(2) / st.nat[:, None, None]           # [B, Z, 3]
        cc = torch.einsum('bij,bj->bi', H, c) + (Rp * st.amask[..., None]).sum(1) / st.nat[:, None]
        rho = cutoff + 2 * st.radius + 0.1                               # [B]

        # (squared sums, not .norm(dim=-1): that reduction over a length-3 axis is slow on CPU)
        spacing = Hi.square().sum(-1).rsqrt()                            # [B, 3] lattice-plane spacings
        N_need = torch.floor(rho[:, None] / spacing + 0.5).long()
        N = N_need.clamp(max=max_translations)
        capped = (N_need > max_translations).any(1)
        D = 2 * N + 1
        M = D.prod(1)
        ptrM = torch.cumsum(M, 0) - M
        bidx = torch.repeat_interleave(torch.arange(B, device=dev), M)
        j = torch.arange(int(M.sum()), device=dev) - ptrM[bidx]
        d2, d1 = D[bidx, 2], D[bidx, 1]
        delta = torch.stack((torch.div(j, d2 * d1, rounding_mode='floor') - N[bidx, 0],
                             torch.div(j, d2, rounding_mode='floor') % d1 - N[bidx, 1],
                             j % d2 - N[bidx, 2]), dim=1)                # [total, 3] integers

        # the point each operator's lattice has to come near, in that operator's fractional frame
        u = torch.einsum('bij,bkj->bki', Hi, cc[:, None, :] - skel_mean) - g
        n0 = torch.round(u)
        rho2 = rho.square()
        keep_b, keep_k, keep_n, keep_d = [], [], [], []
        for k in range(st.W.shape[1]):
            n = n0[bidx, k] + delta
            off = torch.einsum('tij,tj->ti', H[bidx], n - u[bidx, k])    # image centroid - cc
            off2 = off.square().sum(1)
            ok = (off2 <= rho2[bidx]) & st.kmask[bidx, k]
            if k == 0:
                ok &= (n != 0).any(1)                                    # the reference molecule itself
            keep_b.append(bidx[ok])
            keep_k.append(torch.full((int(ok.sum()),), k, device=dev, dtype=torch.long))
            keep_n.append(n[ok])
            if max_images is not None:
                keep_d.append(off2[ok])
        pb, pk, pn = torch.cat(keep_b), torch.cat(keep_k), torch.cat(keep_n)
        order = torch.argsort(pb, stable=True)
        pb, pk, pn = pb[order], pk[order], pn[order]
        image_capped = torch.zeros(B, dtype=torch.bool, device=dev)
        if max_images is not None:
            image_capped = torch.bincount(pb, minlength=B) > max_images
            if bool(image_capped.any()):
                near = _nearest_in_group(pb, torch.cat(keep_d)[order], max_images)
                pb, pk, pn = pb[near], pk[near], pn[near]

    rel = torch.einsum('pij,pj->pi', H[pb], g[pb, pk] + pn - g[pb, 0])   # [P, 3]
    return {'graph': pb, 'op': pk, 'rel': rel, 'skel': skel, 'capped': capped, 'image_capped': image_capped,
            'images_kept': torch.bincount(pb, minlength=B), 'centres_examined': M * st.kmask.sum(1)}


def pair_distances(tables: ImageTables, sel: dict, cutoff: float, max_block: int = 16_000_000,
                   bucket: int = 4, prefilter: bool = True, return_vectors: bool = False,
                   max_pairs: int = None) -> dict:
    """Atom pairs within ``cutoff`` between the reference molecule and the kept images.

    The pairs are found without gradients, on dense blocks of (reference atom, image atom)
    distances. Image molecules are grouped by their molecule's atom count (rounded up to a
    multiple of ``bucket``), so a batch of mixed molecule sizes is not padded to its largest
    member, and processed at most ``max_block`` distances at a time. With ``prefilter``, an
    image with no atom within ``cutoff`` + (extent of the reference molecule) of the
    reference centre is dropped before any pair distance is formed; a pair within ``cutoff``
    forces its image atom inside that sphere, so nothing in range is lost.

    The distances returned are then recomputed for the kept pairs alone from ``sel['rel']``
    and ``sel['skel']``, by plain arithmetic, so they carry first and second derivatives to
    the cell and pose.

    ``max_pairs``, if given, bounds the pairs kept per crystal: a crystal with more keeps
    its ``max_pairs`` shortest and is marked in ``pair_capped`` [B]. The bound is applied
    inside each block of the search as well as at the end, so no crystal ever holds more
    than ``max_pairs`` pairs per block it appears in. A crystal under the bound is returned
    exactly as without one. Together with ``select_images``' ``max_images`` this bounds the
    memory of the search and of whatever consumes the list, whatever the cell.

    Returns a dict of flat per-pair tensors: ``dist``; ``graph`` (crystal index); ``ia`` and
    ``ib``, the reference atom and the image atom as indices within their molecule;
    ``node_ref`` and ``node_img``, the same two atoms as indices among the batch's real
    atoms (the image atom indexed by the atom it is a copy of); ``image``, the row of the
    selection the pair's image molecule is; ``z_src`` and ``z_tgt``, atomic numbers of the
    image and reference atom; and, with ``return_vectors``, ``vec`` [n, 3], image atom minus
    reference atom. ``dense_distances`` is the number of distances formed in the search.
    """
    st = tables
    pb, pk, rel, skel = sel['graph'], sel['op'], sel['rel'], sel['skel']
    with torch.no_grad():
        # torch.cdist without the matrix-product shortcut is the fastest exact form on CPU and
        # roughly ten times slower than explicit differences on CUDA (measured 2026-10-05)
        on_gpu = rel.is_cuda
        width = torch.div(st.nat[pb] + bucket - 1, bucket, rounding_mode='floor') * bucket
        width = width.clamp(max=st.p.shape[1])
        r_ref = (skel[:, 0].square().sum(-1) * st.amask).amax(dim=1).sqrt()      # [B]
        reach2 = (cutoff + r_ref).square()
        out_i, out_a, out_b, out_d = [], [], [], []
        n_dense = 0
        n_graphs = st.p.shape[0]
        over = torch.zeros(n_graphs, dtype=torch.bool, device=rel.device)
        for wdt in torch.unique(width).tolist():
            members = (width == wdt).nonzero().flatten()
            step = max(1, max_block // (wdt * wdt))
            for i in range(0, len(members), step):
                m = members[i:i + step]
                b, k = pb[m], pk[m]
                img = rel[m][:, None, :] + skel[b, k, :wdt]                      # [P, w, 3]
                am = st.amask[b, :wdt]
                if prefilter:
                    near = ((img.square().sum(-1) <= reach2[b][:, None]) & am).any(1)
                    b, img, am, m = b[near], img[near], am[near], m[near]
                    if b.numel() == 0:
                        continue
                ref = skel[b, 0, :wdt]
                n_dense += b.numel() * wdt * wdt
                if on_gpu:
                    d = (ref[:, :, None, :] - img[:, None, :, :]).square().sum(-1).sqrt()
                else:
                    d = torch.cdist(ref, img, compute_mode='donot_use_mm_for_euclid_dist')
                ok = am[:, :, None] & am[:, None, :] & (d <= cutoff)
                pi, ia, ib = ok.nonzero(as_tuple=True)
                if max_pairs is not None:
                    dd, gg = d[pi, ia, ib], b[pi]
                    found = torch.bincount(gg, minlength=n_graphs)
                    over |= found > max_pairs
                    if bool((found > max_pairs).any()):
                        near = _nearest_in_group(gg, dd, max_pairs)
                        pi, ia, ib, dd = pi[near], ia[near], ib[near], dd[near]
                    out_d.append(dd)
                out_i.append(m[pi])
                out_a.append(ia)
                out_b.append(ib)
        if out_i:
            image, ia, ib = torch.cat(out_i), torch.cat(out_a), torch.cat(out_b)
        else:
            image = ia = ib = pb.new_zeros(0)
        if max_pairs is not None and image.numel():
            total = torch.bincount(pb[image], minlength=n_graphs)
            over |= total > max_pairs
            if bool((total > max_pairs).any()):
                near = _nearest_in_group(pb[image], torch.cat(out_d), max_pairs)
                image, ia, ib = image[near], ia[near], ib[near]
    graph = pb[image]
    vec = rel[image] + skel[graph, pk[image], ib] - skel[graph, 0, ia]
    out = {'dist': vec.square().sum(-1).sqrt(), 'graph': graph, 'ia': ia, 'ib': ib,
           'node_ref': st.ptr[graph] + ia, 'node_img': st.ptr[graph] + ib, 'image': image,
           'z_src': st.z[graph, ib], 'z_tgt': st.z[graph, ia], 'dense_distances': n_dense,
           'pair_capped': over}
    if return_vectors:
        out['vec'] = vec
    return out


def image_pairs(tables: ImageTables, T_fc, T_cf, aunit_centroid, aunit_orientation, cutoff: float,
                max_translations: int = 10, max_images: int = None, **kwargs) -> dict:
    """``select_images`` then ``pair_distances``; the pair dict plus ``capped``, ``image_capped``,
    ``images_kept`` and ``centres_examined``."""
    sel = select_images(tables, T_fc, T_cf, aunit_centroid, aunit_orientation, cutoff, max_translations, max_images)
    out = pair_distances(tables, sel, cutoff, **kwargs)
    out.update({k: sel[k] for k in ('capped', 'image_capped', 'images_kept', 'centres_examined')})
    return out
