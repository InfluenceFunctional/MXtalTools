from typing import Optional

import torch
from torch import optim
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.analysis.crystal_rdf import new_crystal_rdf
from mxtaltools.common.training_utils import init_sym_info
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.builder import CrystalBuilder
from mxtaltools.crystal_building.utils import set_molecule_alignment, overwrite_symmetry_info
from mxtaltools.dataset_utils.CrystalData import CrystalData

from mxtaltools.models.utils import denormalize_generated_cell_params, enforce_1d_bound
from mxtaltools.analysis.crystals_analysis import get_intermolecular_dists_dict
from mxtaltools.common.geometry_utils import enforce_crystal_system
from mxtaltools.analysis.vdw_analysis import electrostatic_analysis, vdw_analysis


def standalone_add_scrambled_molecule_samples(aunit_poses, dist_dict, handedness_record, hit_max_lr, loss, loss_record,
                                              lr_record, max_lr, mol_batch, optimizer, packing_coeff, packing_record,
                                              post_scramble_each, raw_samples_record, s_ind, sample, sample_to_compare,
                                              samples_record, scheduler1, scheduler2, score_func, standardize_pose,
                                              store_aunit, supercell_batch, vdw_record, supercell_size, cutoff,
                                              supercell_builder):
    s_ind -= 1
    inds_to_scramble = torch.arange(0, s_ind, max(1, s_ind // post_scramble_each))
    scrambled_samples_record = torch.zeros_like(samples_record)[:len(inds_to_scramble)]
    scrambled_packing_record = torch.zeros_like(packing_record)[:len(inds_to_scramble)]
    scrambled_loss_record = torch.zeros_like(packing_record)[:len(inds_to_scramble)]
    scrambled_vdw_record = torch.zeros_like(packing_record)[:len(inds_to_scramble)]
    scrambled_aunit_poses = torch.zeros_like(aunit_poses)[:len(inds_to_scramble)]
    scrambled_handedness_record = torch.zeros_like(handedness_record)[
                                  :len(inds_to_scramble)]
    for s_ind2, scramble_ind in enumerate(inds_to_scramble):
        sample = raw_samples_record[scramble_ind] + torch.cat([
            torch.zeros_like(sample[:, :9]), torch.randn_like(sample[:, -3:])
        ], dim=1)  # scramble molecule orientation
        sample.requires_grad_(True)
        descaled_cleaned_sample, dist_dict, loss, packing_coeff, supercell_batch, vdw_potential = standalone_gd_opt_step(
            hit_max_lr, lr_record, max_lr, mol_batch, optimizer, s_ind, sample, scheduler1,
            scheduler2,
            score_func, standardize_pose, supercell_size, cutoff, supercell_builder)

        sample_to_compare = descaled_cleaned_sample.clone()
        sample_to_compare[:, 9:] = supercell_batch.cell_params[:, 9:]

        scrambled_vdw_record[s_ind2] = vdw_potential.detach()
        scrambled_samples_record[s_ind2] = sample_to_compare.detach()
        scrambled_loss_record[s_ind2] = loss.detach()
        scrambled_packing_record[s_ind2] = packing_coeff.detach()
        scrambled_handedness_record[s_ind2] = supercell_batch.aunit_handedness
        if store_aunit:
            scrambled_aunit_poses[s_ind2] = supercell_batch.pos[
                supercell_batch.aux_ind == 0].detach()
    s_ind += len(inds_to_scramble)
    samples_record = torch.cat([samples_record, scrambled_samples_record], dim=0)
    vdw_record = torch.cat([vdw_record, scrambled_vdw_record], dim=0)
    loss_record = torch.cat([loss_record, scrambled_loss_record], dim=0)
    packing_record = torch.cat([packing_record, scrambled_packing_record], dim=0)
    aunit_poses = torch.cat([aunit_poses, scrambled_aunit_poses], dim=0)
    return aunit_poses, dist_dict, loss, loss_record, packing_coeff, packing_record, s_ind, sample, sample_to_compare, samples_record, supercell_batch, vdw_record


def standalone_score_crystal_batch(mol_batch, score_func, supercell_data, vdw_radii_tensor, cutoff: float = 6, discriminator: Optional = None):
    if score_func == 'discriminator':
        output, extra_outputs = discriminator(
            supercell_data.clone(), return_dists=True, return_latent=False)
        dist_dict = extra_outputs['dists_dict']
    elif score_func.lower() == 'vdw':
        dist_dict = get_intermolecular_dists_dict(supercell_data,
                                                  cutoff, 100)
    else:
        assert False, f"{score_func} is not an implemented score function for gradient descent optimization"
    molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, lj_pot \
        = vdw_analysis(vdw_radii_tensor,
                       dist_dict,
                       mol_batch.num_graphs)
    estat_energy = electrostatic_analysis(dist_dict, supercell_data.num_graphs)
    vdw_potential += estat_energy
    vdw_loss += estat_energy
    if score_func == 'discriminator':
        loss = F.softplus(output[:, 2])
    elif score_func.lower() == 'vdw':
        loss = vdw_loss
    return dist_dict, loss, vdw_potential


def standalone_gradient_descent_optimization(
        init_sample: torch.Tensor,
        mol_batch: CrystalData,
        max_num_steps: int,
        convergence_eps: float,
        lr: float,
        optimizer_func,
        score_func: str,
        store_aunit: bool = False,
        standardize_pose: bool = True,
        post_scramble_each: int = None,
        skip_rdf: bool = True,
        show_tqdm: bool = False,
        device: str = 'cpu',
):
    """
    do a local optimization via gradient descent on some score function
    """
    supercell_builder = CrystalBuilder(device=device,
                                       rotation_basis='cartesian')

    sample = init_sample.clone().detach().requires_grad_(True)

    (hit_max_lr, loss_record, lr_record, max_lr,
     optimizer, packing_record, samples_record, raw_samples_record, handedness_record,
     scheduler1, scheduler2, vdw_record, aunit_poses) = _init_for_local_opt(
        lr, max_num_steps, optimizer_func, sample, mol_batch.num_nodes)

    converged = False
    with (torch.enable_grad()):
        with tqdm(total=max_num_steps, disable=not show_tqdm) as pbar:
            s_ind = 0
            while not converged:
                descaled_cleaned_sample, dist_dict, loss, packing_coeff, supercell_batch, vdw_potential = standalone_gd_opt_step(
                    hit_max_lr, lr_record, max_lr, mol_batch, optimizer, s_ind, sample, scheduler1, scheduler2,
                    score_func, standardize_pose, 6, 6, supercell_builder
                )

                sample_to_compare = descaled_cleaned_sample.clone()
                sample_to_compare[:, 9:] = supercell_batch.cell_params[:, 9:]

                vdw_record[s_ind] = vdw_potential.detach()
                samples_record[s_ind] = sample_to_compare.detach()
                raw_samples_record[s_ind] = sample.detach()
                loss_record[s_ind] = loss.detach()
                packing_record[s_ind] = packing_coeff.detach()
                handedness_record[s_ind] = supercell_batch.aunit_handedness

                if store_aunit:
                    aunit_poses[s_ind] = supercell_batch.pos[supercell_batch.aux_ind == 0].detach()

                s_ind += 1
                if s_ind % 100 == 0:
                    pbar.update(100)

                if s_ind > 10:
                    flag1 = all(vdw_record[s_ind - 10:s_ind, :].std(0) < convergence_eps)  # loss is converged
                    flag2 = s_ind > (max_num_steps - 1)  # run out of time
                    if flag1 or flag2:
                        converged = True
                        if post_scramble_each is not None:
                            (aunit_poses, dist_dict, loss, loss_record, packing_coeff, packing_record, s_ind, sample,
                             sample_to_compare, samples_record, supercell_batch, vdw_record
                             ) = standalone_add_scrambled_molecule_samples(
                                aunit_poses, dist_dict, handedness_record, hit_max_lr, loss, loss_record, lr_record,
                                max_lr, mol_batch, optimizer, packing_coeff, packing_record, post_scramble_each,
                                raw_samples_record, s_ind, sample, sample_to_compare, samples_record, scheduler1,
                                scheduler2, score_func, standardize_pose, store_aunit, supercell_batch, vdw_record,
                                5, 6, supercell_builder)

    good_inds = torch.argwhere(samples_record[:, 0, 0] != 0).flatten()
    sampling_dict = {'std_cell_params': samples_record[good_inds].cpu(),
                     'vdw_potential': vdw_record[good_inds].cpu(),
                     'overall_loss': loss_record[good_inds].cpu(),
                     'packing_coeff': packing_record[good_inds].cpu(),
                     }
    if store_aunit:
        sampling_dict['aunit_poses'] = aunit_poses[good_inds].cpu()

    # do RDF on final sample
    if not skip_rdf:
        rdf, rr, _ = new_crystal_rdf(supercell_batch, dist_dict,
                                     rrange=[0, 6], bins=2000,
                                     mode='intermolecular', atomwise=True,
                                     raw_density=True, cpu_detach=False,
                                     atomic_numbers_override=mol_batch.x.unique().long())
    else:
        rdf, rr = torch.empty(mol_batch.num_graphs), torch.empty(mol_batch.num_graphs)

    return (packing_coeff.detach().cpu(), rdf.detach().cpu(),
            sample_to_compare.detach().cpu(), loss.detach().cpu(),
            sampling_dict)


def standalone_gd_opt_step(hit_max_lr, lr_record, max_lr, mol_batch, optimizer, s_ind, sample, scheduler1,
                           scheduler2,
                           score_func, standardize_pose, cutoff, supercell_size, supercell_builder):
    optimizer.zero_grad()
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=sample.device)
    cleaned_sample = cleanup_sample(sample, mol_batch.sg_ind, supercell_builder.symmetries_dict)
    descaled_cleaned_sample = denormalize_generated_cell_params(
        cleaned_sample, mol_batch, supercell_builder.ASYM_UNITS
    )
    # todo functionalize the below - it seems we do it everywhere
    supercell_batch, generated_cell_volumes = (
        supercell_builder.build_zp1_supercells(
            mol_batch=mol_batch,
            cell_parameters=descaled_cleaned_sample,
            supercell_size=supercell_size,
            graph_convolution_cutoff=cutoff,
            align_to_standardized_orientation=standardize_pose,
            skip_refeaturization=False,
            target_handedness=torch.ones(mol_batch.num_graphs, dtype=torch.long,
                                         device=mol_batch.x.device)
        ))
    reduced_volume = generated_cell_volumes / supercell_batch.sym_mult
    packing_coeff = mol_batch.mol_volume / reduced_volume
    dist_dict, loss, vdw_potential = standalone_score_crystal_batch(
        mol_batch, score_func, supercell_batch,
        vdw_radii_tensor=vdw_radii_tensor, cutoff=cutoff
    )
    loss.mean().backward()  # compute gradients
    optimizer.step()  # apply grad
    lr = optimizer.param_groups[0]['lr']
    lr_record[s_ind] = lr
    if lr >= max_lr:
        hit_max_lr = True
    if hit_max_lr:
        if lr > 1e-5:
            scheduler1.step()  # shrink
    else:
        scheduler2.step()  # grow
    return descaled_cleaned_sample, dist_dict, loss, packing_coeff, supercell_batch, vdw_potential


def standalone_opt_random_crystals(
        mol_batch,
        init_state,
        opt_eps,
        post_scramble_each: int = None,
        device: str = 'cpu'):
    print("embed mols for opt")
    sym_info = init_sym_info()

    mol_batch = set_molecule_alignment(mol_batch,
                                       mode='standardized',
                                       right_handed=True,  # right handed
                                       include_inversion=False)  # force all samples to come with same handedness
    centroids = scatter(mol_batch.pos, mol_batch.batch, reduce='mean', dim=0)
    mol_batch.pos -= torch.repeat_interleave(centroids, mol_batch.num_atoms, dim=0, output_size=mol_batch.num_nodes)
    mol_batch.pos -= torch.repeat_interleave(centroids, mol_batch.num_atoms, dim=0,
                                             output_size=mol_batch.num_nodes)

    mol_batch = overwrite_symmetry_info(mol_batch,
                                        mol_batch.sg_ind,
                                        sym_info,
                                        randomize_sgs=False)

    print("doing opt")
    (_, _, _, _,
     optimization_record) = standalone_gradient_descent_optimization(
        init_state,
        mol_batch.clone(),
        max_num_steps=1000,
        convergence_eps=opt_eps,
        lr=1e-6,
        optimizer_func=torch.optim.Rprop,
        score_func='vdW',
        store_aunit=True,
        standardize_pose=True,
        post_scramble_each=post_scramble_each,
    )
    n_samples = len(optimization_record['std_cell_params'])
    inds_to_sample = torch.unique(
        torch.cat(
            [torch.arange(0, n_samples, 10),
             torch.argwhere(
                 torch.diff(optimization_record['vdw_potential'], dim=0).mean(1) > 1
             ).flatten()]
        )
    )

    # rescale after summing and norming
    rescaled_vdw_loss = scale_molwise_vdw_pot(optimization_record['vdw_potential'], mol_batch.num_atoms)

    return (
        optimization_record['vdw_potential'][inds_to_sample],
        rescaled_vdw_loss[inds_to_sample],
        optimization_record['packing_coeff'][inds_to_sample],
        optimization_record['std_cell_params'][inds_to_sample],
        optimization_record['aunit_poses'][inds_to_sample],
    )


def _init_for_local_opt(lr, max_num_steps, optimizer_func, sample, num_atoms):
    optimizer = optimizer_func([sample], lr=lr)
    max_lr_target_time = max_num_steps // 10
    max_lr = lr * 100
    grow_lambda = (max_lr / lr) ** (1 / max_lr_target_time)
    scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.975)
    scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: grow_lambda)
    hit_max_lr = False
    num_samples = len(sample)
    vdw_record = torch.zeros((max_num_steps, num_samples))
    samples_record = torch.zeros((max_num_steps, num_samples, 12))
    raw_samples_record = torch.zeros_like(samples_record)
    handedness_record = torch.zeros((max_num_steps, num_samples))
    loss_record = torch.zeros_like(vdw_record)
    lr_record = torch.zeros(max_num_steps)
    packing_record = torch.zeros_like(vdw_record)
    aunit_poses = torch.zeros((len(vdw_record), num_atoms, 3))
    return (hit_max_lr, loss_record, lr_record, max_lr,
            optimizer, packing_record, samples_record, raw_samples_record, handedness_record,
            scheduler1, scheduler2, vdw_record, aunit_poses)


def cleanup_sample(raw_sample, sg_ind_list, symmetries_dict):
    # force outputs into physical ranges
    # cell lengths have to be positive nonzero
    cell_lengths = raw_sample[:, :3].clip(min=0.01)
    # range from (0,pi) with 20% padding to prevent too-skinny cells
    cell_angles = enforce_1d_bound(raw_sample[:, 3:6], x_span=torch.pi / 2 * 0.8, x_center=torch.pi / 2,
                                   mode='hard')
    # positions must be on 0-1
    mol_positions = enforce_1d_bound(raw_sample[:, 6:9], x_span=0.5, x_center=0.5, mode='hard')
    # for now, just enforce vector norm
    rotvec = raw_sample[:, 9:12]
    norm = torch.linalg.norm(rotvec, dim=1)
    new_norm = enforce_1d_bound(norm, x_span=0.999 * torch.pi, x_center=torch.pi, mode='hard')  # MUST be nonzero
    new_rotvec = rotvec / norm[:, None] * new_norm[:, None]
    # invert_inds = torch.argwhere(new_rotvec[:, 2] < 0)
    # new_rotvec[invert_inds] = -new_rotvec[invert_inds]  # z direction always positive
    # force cells to conform to crystal system
    cell_lengths, cell_angles = enforce_crystal_system(cell_lengths, cell_angles, sg_ind_list,
                                                       symmetries_dict)
    sample = torch.cat((cell_lengths, cell_angles, mol_positions, new_rotvec), dim=-1)
    return sample
