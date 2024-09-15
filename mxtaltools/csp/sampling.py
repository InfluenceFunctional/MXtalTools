from typing import Tuple

from torch_geometric.loader.dataloader import Collater
import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from tqdm import tqdm

from mxtaltools.common.utils import init_sym_info
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.builder import SupercellBuilder
from mxtaltools.crystal_building.utils import set_molecule_alignment, overwrite_symmetry_info
from mxtaltools.models.functions.crystal_rdf import new_crystal_rdf
from mxtaltools.models.functions.vdw_overlap import vdw_analysis
from mxtaltools.models.utils import denormalize_generated_cell_params, compute_prior_loss, \
    get_intermolecular_dists_dict, crystal_filter_cluster


class Sampler:
    def __init__(self,
                 seed,
                 device,
                 prior,
                 generator,
                 autoencoder,
                 ):
        self.device = device
        self.supercell_size = 5
        self.cutoff = 6
        self.prior = prior
        self.generator = generator
        self.sym_info = init_sym_info()
        self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        self.autoencoder = autoencoder
        self.collater = Collater(0, 0)
        self.supercell_builder = SupercellBuilder(device=self.device,
                                                  rotation_basis='cartesian')

    def sample_and_cluster(self, init_sample, tot_samples,
                           batch_size, cc_orientation,
                           sg_to_sample, sample_source, show_progress=True,
                           rauv_cutoff=1.2,
                           vdw_cutoff=1,
                           cell_params_threshold=0.01,
                           rdf_dist_threshold=0.05,
                           variation_factor=None,
                           ):
        with torch.no_grad():
            samples, lj_potential, rauv, rdf, rr = self.do_sampling(
                batch_size, cc_orientation,
                init_sample, sample_source,
                sg_to_sample, tot_samples,
                show_tqdm=show_progress, variation_factor=variation_factor)

            lj_potential, samples, rdf, rauv, rdf_distmat = \
                crystal_filter_cluster(lj_potential, rdf, rr, samples, rauv,
                                       rauv_cutoff, vdw_cutoff,
                                       cell_params_threshold, rdf_dist_threshold,
                                       )

        return {'lj_potential': lj_potential.numpy(), 'rdf': rdf.numpy(), 'samples': samples.numpy(),
                'rauv': rauv.numpy(), 'rdf_distmat': rdf_distmat}

    def do_sampling(self, batch_size, cc_orientation,
                    init_sample, sample_source,
                    sg_to_sample, tot_samples,
                    show_tqdm=True, variation_factor=None):
        num_batches = tot_samples // batch_size + 1
        sample_record, lj_potential_record, rauv_record, rdf_record = [], [], [], []
        for _ in tqdm(range(num_batches), disable=not show_tqdm):
            samples, lj_potential, rauv, rdf, rr = self.sample_iter(
                init_sample, batch_size, cc_orientation,
                sg_to_sample, sample_source, variation_factor)
            sample_record.append(samples)
            lj_potential_record.append(lj_potential)
            rauv_record.append(rauv)
            rdf_record.append(rdf)

        return (torch.cat(sample_record),
                torch.cat(lj_potential_record),
                torch.cat(rauv_record),
                torch.cat(rdf_record),
                rr
                )

    def sample_iter(self, init_sample, batch_size, cc_orientation, sg_to_sample, sample_source, variation_factor=None):
        data = self.collater([init_sample for _ in range(batch_size)]).to(self.device)

        mol_data, scalar_mol_embedding, vector_mol_embedding = self.process_embed_for_generator(
            data,
            cc_orientation,
            sg_to_sample,
            skip_embedding=sample_source == 'csd_prior'
        )

        if sample_source == 'csd_prior':
            generated_sample, descaled_generated_sample = self.sample_from_prior(mol_data)
        elif sample_source == 'generator':
            generated_sample, descaled_generated_sample = self.sample_from_generator(
                mol_data, scalar_mol_embedding, vector_mol_embedding, variation_factor)
        else:
            assert False, f"sample_source {sample_source} not supported"

        supercell_data, generated_cell_volumes = (
            self.supercell_builder.build_zp1_supercells(
                molecule_data=mol_data,
                cell_parameters=descaled_generated_sample,
                supercell_size=self.supercell_size,
                graph_convolution_cutoff=self.cutoff,
                align_to_standardized_orientation=False,
                skip_refeaturization=False,
            ))

        (eval_lj_loss, lj_loss, lj_potential,
         molwise_normed_overlap, prior_loss,
         sample_rauv, scaled_deviation,
         dist_dict) = self.analyze_generated_crystals(
            mol_data,
            generated_cell_volumes,
            generated_sample,
            supercell_data,
            return_dist_dict=True
        )
        sample_to_compare = generated_sample.clone()
        sample_to_compare[:, 9:] = supercell_data.cell_params[:, 9:]

        rdf, rr, _ = new_crystal_rdf(supercell_data, dist_dict,
                                     rrange=[0, 6], bins=2000,
                                     mode='intermolecular', atomwise=True,
                                     raw_density=True, cpu_detach=False,
                                     atomic_numbers_override=mol_data.x.unique().long())

        return sample_to_compare.detach(), (
                    lj_potential / mol_data.num_atoms).detach(), sample_rauv.detach(), rdf.detach(), rr.detach()

    def sample_from_prior(self, mol_data) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_samples = self.prior(mol_data.num_graphs, mol_data.sg_ind).to(self.device)

        samples_to_build = denormalize_generated_cell_params(
            raw_samples,
            mol_data,
            self.supercell_builder.asym_unit_dict
        )

        return raw_samples, samples_to_build

    def sample_from_generator(self, mol_data,
                              scalar_mol_embedding,
                              vector_mol_embedding,
                              variation_factor) -> Tuple[torch.Tensor, torch.Tensor]:
        prior, descaled_prior = self.sample_from_prior(mol_data)

        variation_tensor = variation_factor * torch.ones(scalar_mol_embedding.shape[0], device=self.device)
        scaling_factor = (self.prior.norm_factors[mol_data.sg_ind, :] + 1e-4)

        absolute_reference = torch.eye(3, dtype=torch.float32, device=self.device
                                       ).reshape(1, 9).repeat(mol_data.num_graphs, 1)

        conditioning_vector = torch.cat((
            scalar_mol_embedding,
            prior,
            absolute_reference,
            variation_tensor[:, None],
            scaling_factor),
            dim=1)

        raw_samples = self.generator.forward(
            conditioning_vector,
            vector_mol_embedding,
            mol_data.sg_ind,
        )

        samples_to_build = denormalize_generated_cell_params(
            raw_samples,
            mol_data,
            self.supercell_builder.asym_unit_dict
        )

        return raw_samples, samples_to_build

    def process_embed_for_generator(self, data, orientation, generate_sgs, skip_embedding=False):
        # clone, center, and orient the molecules
        _, mol_data = self.preprocess_ae_inputs(
            data,
            orientation_override=orientation)

        mol_data = overwrite_symmetry_info(mol_data,
                                           generate_sgs,
                                           self.sym_info,
                                           randomize_sgs=True)

        if not skip_embedding:
            # embed molecules
            with torch.no_grad():
                vector_mol_embedding = self.autoencoder.encode(mol_data.clone())
                scalar_mol_embedding = self.autoencoder.scalarizer(vector_mol_embedding)

        else:
            scalar_mol_embedding, vector_mol_embedding = None, None

        return mol_data, scalar_mol_embedding, vector_mol_embedding

    def preprocess_ae_inputs(self, data,
                             orientation_override=None,
                             deprotonate=False):
        # random global roto-inversion
        if orientation_override is not None:
            data = set_molecule_alignment(data,
                                          mode=orientation_override,
                                          right_handed=False,
                                          include_inversion=True)

        # optionally, deprotonate
        input_data = self.fix_autoencoder_protonation(data,
                                                      deprotonate=deprotonate)

        # subtract mean OF THE INPUT from BOTH reference and input
        centroids = scatter(input_data.pos, input_data.batch, reduce='mean', dim=0)
        data.pos -= torch.repeat_interleave(centroids, data.num_atoms, dim=0, output_size=data.num_nodes)
        input_data.pos -= torch.repeat_interleave(centroids, input_data.num_atoms, dim=0,
                                                  output_size=input_data.num_nodes)

        return data, input_data

    def fix_autoencoder_protonation(self, data, deprotonate=False):
        if deprotonate:
            heavy_atom_inds = torch.argwhere(data.x != 1).flatten()  # protons are atom type 1
            input_cloud = data.detach().clone()
            input_cloud.x = input_cloud.x[heavy_atom_inds]
            input_cloud.pos = input_cloud.pos[heavy_atom_inds]
            input_cloud.batch = input_cloud.batch[heavy_atom_inds]
            a, b = torch.unique(input_cloud.batch, return_counts=True)
            input_cloud.ptr = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(b, dim=0)]).long()
            input_cloud.num_atoms = torch.diff(input_cloud.ptr).long()
        else:
            input_cloud = data.detach().clone()

        return input_cloud

    def analyze_generated_crystals(self,
                                   mol_data,
                                   generated_cell_volumes,
                                   generator_raw_samples,
                                   supercell_data,
                                   variation_factor=None,
                                   prior=None,
                                   return_dist_dict=False,
                                   vdw_turnover_potential=10):
        if prior is not None:
            prior_loss, scaled_deviation = compute_prior_loss(
                self.prior.norm_factors,
                mol_data.sg_ind,
                generator_raw_samples,
                prior,
                variation_factor)
        else:
            prior_loss = torch.zeros_like(generated_cell_volumes)
            scaled_deviation = torch.zeros_like(generated_cell_volumes)

        reduced_volume = generated_cell_volumes / supercell_data.sym_mult
        atom_volumes = scatter(4 / 3 * torch.pi * self.vdw_radii_tensor[mol_data.x[:, 0]] ** 3, mol_data.batch,
                               reduce='sum')
        sample_rauv = reduced_volume / atom_volumes

        dist_dict = get_intermolecular_dists_dict(supercell_data, self.cutoff, 100)
        molwise_overlap, molwise_normed_overlap, lj_potential, lj_loss, eval_lj_loss \
            = vdw_analysis(self.vdw_radii_tensor, dist_dict, mol_data.num_graphs, vdw_turnover_potential)

        if return_dist_dict:
            return (eval_lj_loss, lj_loss, lj_potential,
                    molwise_normed_overlap, prior_loss,
                    sample_rauv, scaled_deviation, dist_dict)

        else:
            return (eval_lj_loss, lj_loss, lj_potential,
                    molwise_normed_overlap, prior_loss,
                    sample_rauv, scaled_deviation)
