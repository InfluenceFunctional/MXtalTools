import torch
import torch.nn.functional as F
from models.asymmetric_radius_graph import asymmetric_radius_graph
import sys


def vdw_overlap(vdw_radii, dist_dict=None, dists=None, batch_numbers=None, atomic_numbers=None,
                num_graphs=None, crystaldata=None, graph_sizes=None,
                return_loss_only=False, return_score_only=False,
                loss_func=None):

    assert not (return_loss_only and return_score_only)

    if dist_dict is not None:
        dists = dist_dict['intermolecular_dist']
        atomic_numbers = dist_dict['intermolecular_dist_atoms']
        batch_numbers = dist_dict['intermolecular_dist_batch']

    abs_overlaps, normed_overlaps = raw_vdw_overlap(
        vdw_radii, crystaldata=crystaldata,
        dists=dists, batch_numbers=batch_numbers, atomic_numbers=atomic_numbers,
        num_graphs=num_graphs
    )

    if graph_sizes is not None:
        mol_sizes = graph_sizes
    elif crystaldata is not None:
        mol_sizes = crystaldata.mol_size
    else:
        assert False

    if num_graphs is None:
        num_graphs = crystaldata.num_graphs

    if loss_func is None:  # treat all errors equally
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(normed_overlaps[ii]) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'mse':  # penalize on big errors
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(normed_overlaps[ii] ** 2) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'log':  # go easy on big errors
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(torch.log(1 + normed_overlaps[ii])) for ii in range(num_graphs)]
            )) / mol_sizes
    elif loss_func == 'max':  # cut the tallest dandelion
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.amax(normed_overlaps[ii]) for ii in range(num_graphs)]
            ))
    elif loss_func == 'inv':  # hard asymptote at 1, but we can't touch it
        vdw_loss = torch.nan_to_num(
            torch.stack(
                [torch.sum(
                    1 / (-torch.minimum(0.99 * torch.ones_like(normed_overlaps[ii]), normed_overlaps[ii]) + 1) - 1
                ) for ii in range(num_graphs)]
            )) / mol_sizes
    else:
        print(f'{loss_func} is not a valid loss function for vdw penalty')
        sys.exit()

    vdw_score = -torch.nan_to_num(
        torch.stack(
            [torch.sum(normed_overlaps[ii]) for ii in range(num_graphs)]
        )) / mol_sizes

    max_overlaps = torch.nan_to_num(
        torch.stack(
            [torch.amax(normed_overlaps[ii]) if len(normed_overlaps[ii] > 0) else
             torch.amax(torch.zeros(1, device=normed_overlaps[ii].device)) for ii in range(num_graphs)]
        ))

    if return_loss_only:
        return vdw_loss
    elif return_score_only:
        return vdw_score
    else:  # return everything
        return vdw_loss, vdw_score, max_overlaps, abs_overlaps, normed_overlaps


def raw_vdw_overlap(vdw_radii, dists=None, batch_numbers=None, atomic_numbers=None, num_graphs=None, crystaldata=None):
    if crystaldata is not None:  # extract distances from the crystal
        if crystaldata.aux_ind is not None:
            in_inds = torch.where(crystaldata.aux_ind == 0)[0]
            # default to always intermolecular_distances
            out_inds = torch.where(crystaldata.aux_ind == 1)[0].to(crystaldata.pos.device)

        else:  # if we lack the info, just do it intramolecular
            in_inds = torch.arange(len(crystaldata.pos)).to(crystaldata.pos.device)
            out_inds = in_inds

        '''
        compute all distances
        '''
        edges = asymmetric_radius_graph(crystaldata.pos,
                                        batch=crystaldata.batch,
                                        inside_inds=in_inds,
                                        convolve_inds=out_inds,
                                        r=6, max_num_neighbors=500, flow='source_to_target')

        crystal_number = crystaldata.batch[edges[0]]

        dists = (crystaldata.pos[edges[0]] - crystaldata.pos[edges[1]]).pow(2).sum(dim=-1).sqrt()
        elements = [crystaldata.x[edges[0], 0].long().to(dists.device), crystaldata.x[edges[1], 0].long().to(dists.device)]
        num_graphs = crystaldata.num_graphs
        molecule_sizes = torch.diff(crystaldata.ptr)
    elif dists is not None:  # precomputed intermolecular crystal distances
        crystal_number = batch_numbers
        elements = atomic_numbers
        num_graphs = num_graphs

    else:
        assert False  # must do one or the other

    '''
    compute vdw radii respectfulness
    '''
    vdw_radii_vector = torch.Tensor(list(vdw_radii.values())).to(dists.device)
    atom_radii = [vdw_radii_vector[elements[0]], vdw_radii_vector[elements[1]]]
    radii_sums = atom_radii[0] + atom_radii[1]

    penalties = F.relu(radii_sums - dists)  # only punish positives (meaning overlaps)
    normed_penalties = F.relu((radii_sums - dists) / radii_sums)  # norm overlaps against internuclear distances

    return [penalties[crystal_number == ii] for ii in range(num_graphs)], [normed_penalties[crystal_number == ii] for ii in range(num_graphs)]
