import torch
import torch.nn.functional as F
from models.asymmetric_radius_graph import asymmetric_radius_graph

def vdw_overlap(vdw_radii, dists=None, batch_numbers=None, atomic_numbers=None, num_graphs=None, crystaldata=None, return_atomwise=False):
    if crystaldata is not None:  # extract distances from the crystal
        if crystaldata.aux_ind is not None:
            in_inds = torch.where(crystaldata.aux_ind == 0)[0]
            # default to always intermolecular distances
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

    # penalties = torch.clip(torch.exp(-overlaps / radii_sums) - 1, min=0) / 1.71828  # strictly normed vdw loss
    penalties = F.relu(-(dists - radii_sums))  # only punish negatives (meaning overlaps)
    assert torch.sum(torch.isnan(penalties)) == 0

    scores = torch.nan_to_num(
        torch.stack(
            # [torch.mean(torch.topk(penalties[crystal_number == ii], 5)[0]) for ii in range(num_graphs)]
            [torch.max(penalties[crystal_number == ii]) if (len(penalties[crystal_number == ii]) > 0) else torch.zeros(1)[0].to(penalties.device) for ii in range(num_graphs)]
        )
    )
    # mean_scores = torch.nan_to_num(
    #     torch.stack(
    #         [torch.mean(penalties[crystal_number == ii]) for ii in range(num_graphs)]
    #     )
    # )
    tot_scores = scores #(scores + mean_scores) / 2 # combine mean score with max score
    assert len(tot_scores) == num_graphs
    if return_atomwise:
        return tot_scores, [penalties[crystal_number == ii] for ii in range(num_graphs)]
    else:
        return tot_scores
