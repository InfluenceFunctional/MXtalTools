from typing import Optional

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from mxtaltools.analysis.crystal_rdf import crystal_rdf
from mxtaltools.analysis.vdw_analysis import get_intermolecular_dists_dict, lj_analysis, electrostatic_analysis, \
    buckingham_energy, silu_energy, vdW_analysis, qlj_analysis, elj_analysis
from mxtaltools.common.sym_utils import init_sym_info
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
from mxtaltools.models.functions.radial_graph import build_radial_graph


# noinspection PyAttributeOutsideInit
class MolCrystalAnalysis:

    def construct_intra_radial_graph(self, cutoff: float = 6):
        if self.aux_ind is not None:
            raise RuntimeError("Cannot build molecular graph when we have already built the cluster")
        self.edges_dict = build_radial_graph(self.pos,
                                             self.batch,
                                             self.ptr,
                                             cutoff,
                                             max_num_neighbors=10000
                                             )
        self.edge_index = self.edges_dict['edge_index']

    def construct_radial_graph(self,
                               cutoff: float = 6,
                               max_num_neighbors=1000):
        if self.is_batch:
            self.edges_dict = get_intermolecular_dists_dict(
                self,
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

        else:
            self.edges_dict = get_intermolecular_dists_dict(
                collate_data_list([self]),
                cutoff,
                max_num_neighbors=max_num_neighbors
            )

    def _pre_compute_checks(self):
        if not hasattr(self, 'vdw_radii_tensor'):
            self.vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device=self.device)
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing energies!")

    def compute(self, requests: dict, **kwargs):
        """
        Analyze a crystal cluster according to requests in input dict
        :return:
        """
        self._init_computes()
        return {key: self.computes[key.lower()](**kwargs) for key in requests}

    def _init_computes(self):
        if not hasattr(self, 'computes'):
            self.computes = {'lj': self.compute_LJ_energy,
                             'qlj': self.compute_qLJ_energy,
                             'elj': self.compute_eLJ_energy,
                             'es': self.compute_ES_energy,
                             'bh': self.compute_buckingham_energy,
                             'silu': self.compute_silu_energy,
                             'vdw': self.compute_vdW_overlap,
                             'ellipsoid': self.compute_ellipsoidal_overlap,
                             'niggli_overlap': self.compute_niggli_overlap,
                             'reduction_en': self.compute_cell_reduction_penalty,
                             'rdf': self.compute_rdf,
                             'ellipsoid_emb': self.compute_ellipsoid_embedding,
                             'uma_pot': self.compute_crystal_uma,
                             'uma_gas_pot': self.compute_lattice_gas_phase_uma,
                             'uma': self.compute_lattice_uma,
                             }

    def compute_ellipsoid_embedding(self):
        edge_j_good, molwise_batch, norm_factor, normed_v1, normed_v2, v1, v2, x = self.get_ellipsoid_embedding(
            surface_padding=1
        )
        return scatter(x, molwise_batch, dim=0, dim_size=self.num_graphs, reduce='sum')

    def compute_LJ_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = lj_analysis(self.vdw_radii_tensor,
                              self.edges_dict,
                              self.num_graphs,
                              )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_qLJ_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = qlj_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_eLJ_energy(self, repulsion: Optional[float] = 2.5, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_lj_pot \
                = elj_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               stiffness=repulsion,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_lj_pot

    def compute_vdW_overlap(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_vdw_overlap \
                = vdW_analysis(self.vdw_radii_tensor,
                               self.edges_dict,
                               self.num_graphs,
                               )
        else:
            raise NotImplementedError("LJ energies not implemented for single crystals")

        return molwise_vdw_overlap

    def compute_ES_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_estat_energy = electrostatic_analysis(
                self.edges_dict,
                self.num_graphs)

        else:
            raise NotImplementedError("ES energies not implemented for single crystals")

        return molwise_estat_energy

    def compute_buckingham_energy(self, **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_buckingham_energy = buckingham_energy(
                self.edges_dict,
                self.num_graphs,
                self.vdw_radii_tensor
            )

        else:
            raise NotImplementedError("BH energies not implemented for single crystals")

        return molwise_buckingham_energy

    def compute_silu_energy(self,
                            repulsion: float = 1.0,
                            **kwargs):
        self._pre_compute_checks()

        if self.is_batch:
            molwise_silu_energy = silu_energy(
                self.edges_dict,
                self.num_graphs,
                self.vdw_radii_tensor,
                repulsion=repulsion,
            )
        else:
            raise NotImplementedError("SiLU energies not implemented for single crystals")

        return molwise_silu_energy

    def build_and_analyze(self,
                          return_cluster: Optional[bool] = False,
                          noise: Optional[float] = None,
                          cutoff: float = 6,
                          supercell_size: int = 10,
                          ):
        """  # todo this should be deprecated in favour of analyze
        full procedure for building and analyzing a molecular crystal
        this is the OLD version of this function - best practice is to use the 'analyze' function below
        """

        cluster_batch = self.mol2cluster(cutoff, supercell_size)

        if noise is not None:
            cluster_batch.pos += torch.randn_like(cluster_batch.pos) * noise

        cluster_batch.construct_radial_graph(cutoff=cutoff)

        outputs = cluster_batch.compute(requests=['lj', 'es'])
        self.lj = outputs['lj']
        self.es_pot = outputs['es']
        self.scaled_lj = log_rescale_positive(self.lj)

        if return_cluster:
            return self.lj, self.es_pot, self.scaled_lj, cluster_batch
        else:
            return self.lj, self.es_pot, self.scaled_lj

    def analyze(self,
                computes: list,
                return_cluster: Optional[bool] = False,
                noise: Optional[float] = None,
                cutoff: float = 10,  # todo allow custom cutoff for rdf compute
                supercell_size: int = 10,
                std_orientation: Optional[bool] = True,
                assign_outputs: Optional[bool] = False,
                **kwargs
                ):
        """
        full procedure for building and analyzing a molecular crystal
        """
        cluster_batch = self.mol2cluster(
            cutoff, supercell_size,
            std_orientation=std_orientation)

        if noise is not None:
            cluster_batch.pos += torch.randn_like(cluster_batch.pos) * noise

        cluster_batch.construct_radial_graph(cutoff=cutoff)

        results = cluster_batch.compute(requests=computes, **kwargs)

        if assign_outputs:
            for key, value in results.items():
                if key != 'rdf':
                    self.add_graph_attr(value, key)
                else:
                    self.add_graph_attr(value[0], key)

        if return_cluster:
            return results, cluster_batch
        else:
            return results

    def compute_crystal_uma(self,
                            predictor,
                            std_orientation: bool = True,
                            ):
        if self.is_batch:
            return compute_crystal_uma_on_mxt_batch(self.clone(),
                                                    std_orientation,
                                                    predictor)
        else:
            return compute_crystal_uma_on_mxt_batch(collate_data_list(self),
                                                    std_orientation,
                                                    predictor)

    def compute_lattice_uma(self,
                            predictor,
                            std_orienation: bool = True,
                            **kwargs):
        if not hasattr(self, 'uma_gas_pot'):
            self.add_graph_attr(
                self.compute_lattice_gas_phase_uma(predictor),
                'uma_gas_pot')

        if not hasattr(self, 'uma_pot'):
            self.add_graph_attr(
                self.compute_crystal_uma(predictor, std_orientation=std_orienation),
                'uma_pot')

        # lattice energy per molecule, in eV
        # multiply by 96.485 to get kJ/mol
        return (self.uma_pot / (self.sym_mult * self.z_prime) - self.uma_gas_pot) * 96.485

    def compute_lattice_gas_phase_uma(self,
                                      predictor,
                                      std_orientation: bool = True,
                                      **kwargs,
                                      ):


        if self.is_batch:
            diffuse_batch = self.clone()
        else:
            diffuse_batch = collate_data_list(self)

        if hasattr(self, 'aux_ind'):
            if self.aux_ind is not None:  # if this is a cluster, we have to rebuild the original aunit batch
                diffuse_batch.pos = self.pos[self.aux_ind == 0].detach().clone()
                diffuse_batch.batch = self.batch[self.aux_ind == 0].detach().clone()
                diffuse_batch.z = self.z[self.aux_ind == 0].detach().clone()

        if diffuse_batch.z_prime.amax() > 1:  # do it separately per conformer
            zp1_batch = diffuse_batch.split_to_zp1_batch()
            zp1_batch.reset_sg_info(sg_ind=1)  # big P1 cells
            zp1_batch.cell_lengths *= 100
            zp1_batch.box_analysis()
            split_ens = compute_crystal_uma_on_mxt_batch(zp1_batch,
                                                         std_orientation,
                                                         predictor,
                                                         pbc=False)
            zp_inds = torch.arange(self.num_graphs, dtype=torch.long, device=self.device
                                   ).repeat_interleave(self.z_prime).to(split_ens.device)
            return scatter(split_ens, zp_inds, reduce='mean', dim=0, dim_size=self.num_graphs)

        else:
            # evaluate a diffuse P1 5ell
            diffuse_batch.reset_sg_info(sg_ind=1)  # big P1 cells
            diffuse_batch.cell_lengths *= 100
            diffuse_batch.box_analysis()
            return compute_crystal_uma_on_mxt_batch(diffuse_batch,
                                                    std_orientation,
                                                    predictor,
                                                    pbc=False,
                                                    force_rebuild=True)

    def compute_rdf(self,
                    rdf_cutoff: float = 6,
                    elementwise: bool = True,
                    bins: int = 100,
                    **kwargs,
                    ):
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing RDF!"
                               "Do batch.construct_radial_graph before calling.")

        assert self.is_batch, "RDF method not implemented for single crystals"
        rdf, bin_edges, rdf_pair_dict = crystal_rdf(self,
                                                    self.edges_dict,
                                                    rrange=(0, rdf_cutoff),
                                                    bins=bins,
                                                    elementwise=elementwise,
                                                    )
        return rdf / self.z_prime[:, None, None], bin_edges, rdf_pair_dict

    def compute_niggli_overlap(self, **kwargs):
        a, b, c, al, be, ga = self.zp1_cell_parameters()[:, :6].split(1, dim=1)
        ab = a * b
        ac = a * c
        bc = b * c

        al_cos = torch.cos(al)
        be_cos = torch.cos(be)
        ga_cos = torch.cos(ga)

        return (ab * ga_cos + ac * be_cos + bc * al_cos).flatten()

    def bounding_penalty(self, x, lower, upper, margin: float = 0.0):
        return (torch.relu(x - (upper - margin)) ** 2) + (torch.relu((lower + margin) - x) ** 2)

    def compute_cell_reduction_penalty(self, margin: float = 0.0, **kwargs):
        """Compute a penalty term for the given crystal system that pushes it towards
        our canonical / standardized / reduced cell geometry, following spglib"""
        # todo check behaviors / correctness for higher crystal systems
        cell_parameters = self.full_cell_parameters()
        cell_lengths, cell_angles, aunit_positions, aunit_orientations = self.split_cell_params()
        if not hasattr(self, 'symmetries_dict'):
            self.symmetries_dict = init_sym_info()

        'get cs masks'
        # crystal_systems  #
        sg = self.sg_ind
        masks = {'triclinic': (sg == 1) | (sg == 2),
                 'monoclinic': (sg >= 3) & (sg <= 15),
                 'orthorhombic': (sg >= 16) & (sg <= 74),
                 'tetragonal': (sg >= 75) & (sg <= 142),
                 'trigonal': (sg >= 143) & (sg <= 167),
                 'hexagonal': (sg >= 168) & (sg <= 194),
                 'cubic': (sg >= 195) & (sg <= 230),
                 }

        reduction_penalties = {
            'triclinic': self.tri_reduction_penalty,
            'monoclinic': self.mono_reduction_penalty,
            'orthorhombic': self.ortho_reduction_penalty,
            'tetragonal': self.tetra_reduction_penalty,
            'trigonal': self.trig_reduction_penalty,
            'hexagonal': self.hex_reduction_penalty,
            'cubic': self.cube_reduction_penalty,
        }

        E = torch.zeros(self.num_graphs, dtype=torch.float32, device=self.device)
        for cs, mask in masks.items():
            if mask.sum() > 0:
                E[mask] = reduction_penalties[cs](cell_lengths[mask], cell_angles[mask], margin)
                if cs == 'triclinic':
                    E[mask] = E[mask] + F.relu(
                        self.compute_niggli_overlap()[mask] - margin) ** 2  # penalize positive overlaps

        return E

    def tri_reduction_penalty(self, cell_lengths, cell_angles, margin):
        """triclinic cells reduction ruels"""
        eps = 1e-6
        bc_error = F.relu(cell_lengths[:, 1] / cell_lengths[:, 2] - (1 - margin)) ** 2  # c>b
        ab_error = F.relu(cell_lengths[:, 0] / cell_lengths[:, 1] - (1 - margin)) ** 2  # # b>a

        a, b, c = cell_lengths.unbind(dim=1)
        al, be, ga = cell_angles.unbind(dim=1)
        al_max_cos = b / 2 / c
        be_max_cos = a / 2 / c
        ga_max_cos = a / 2 / b

        alpha_error = self.bounding_penalty(al.cos() / al_max_cos.clamp(min=eps), -1, 1, margin=margin)
        beta_error = self.bounding_penalty(be.cos() / be_max_cos.clamp(min=eps), -1, 1, margin=margin)
        gamma_error = self.bounding_penalty(ga.cos() / ga_max_cos.clamp(min=eps), -1, 1, margin=margin)

        return bc_error + ab_error + alpha_error + beta_error + gamma_error

    def mono_reduction_penalty(self, cell_lengths, cell_angles, margin):

        # enforces our reduction scheme
        ac_error = F.relu(cell_lengths[:, 0] / cell_lengths[:, 2] - (1 - margin)) ** 2  # c>a
        beta_error = self.bounding_penalty(cell_angles[:, 1], torch.pi / 2, torch.pi, margin=margin)

        # enforces the crystal system
        alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
        gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

        return ac_error + beta_error + alpha_error + gamma_error

    def ortho_reduction_penalty(self, cell_lengths, cell_angles, margin):
        # crystal system enforcement
        alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
        beta_error = (cell_angles[:, 1] - torch.pi / 2) ** 2
        gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

        # cell reduction enforcement
        bc_error = F.relu(cell_lengths[:, 1] / cell_lengths[:, 2] - (1 - margin)) ** 2  # c>b
        ab_error = F.relu(cell_lengths[:, 0] / cell_lengths[:, 1] - (1 - margin)) ** 2  # # b>a

        return alpha_error + beta_error + gamma_error + ab_error + bc_error

    def tetra_reduction_penalty(self, cell_lengths, cell_angles, margin):
        a, b, c = cell_lengths.unbind(dim=-1)

        # reduction term
        abc_error = F.relu(b / c - (1 - margin)) ** 2 + F.relu(a / c - (1 - margin)) ** 2

        # crystal system terms

        # enforce a=b
        ab_error = (a - b) ** 2
        # enforce right angles
        alpha_error = (cell_angles[:, 0] - torch.pi / 2) ** 2
        beta_error = (cell_angles[:, 1] - torch.pi / 2) ** 2
        gamma_error = (cell_angles[:, 2] - torch.pi / 2) ** 2

        return ab_error + alpha_error + beta_error + gamma_error + abc_error

    def trig_reduction_penalty(self, cell_lengths, cell_angles, margin):
        a, b, c = cell_lengths.unbind(dim=-1)
        al, be, ga = cell_angles.unbind(dim=-1)

        # todo add reduction terms

        # crystal system enforcement
        # a = b
        ab_error = (a - b) ** 2

        # alpha = beta = 90°
        alpha_error = (al - torch.pi / 2) ** 2
        beta_error = (be - torch.pi / 2) ** 2

        # gamma = 120°
        gamma_error = (ga - 2 * torch.pi / 3) ** 2

        return ab_error + alpha_error + beta_error + gamma_error

    def hex_reduction_penalty(self, cell_lengths, cell_angles, margin):
        a, b, c = cell_lengths.unbind(dim=-1)
        al, be, ga = cell_angles.unbind(dim=-1)

        # todo add reduction terms

        # crystal system enforcement
        ab_error = (a - b) ** 2
        alpha_error = (al - torch.pi / 2) ** 2
        beta_error = (be - torch.pi / 2) ** 2
        gamma_error = (ga - 2 * torch.pi / 3) ** 2

        return ab_error + alpha_error + beta_error + gamma_error

    def rhombo_reduction_penalty(self, cell_lengths, cell_angles, margin):
        a, b, c = cell_lengths.unbind(dim=1)
        al, be, ga = cell_angles.unbind(dim=1)

        # todo add reduction terms

        # crystal system enforcement
        ab_error = (a - b) ** 2
        bc_error = (b - c) ** 2
        angle_eq_error = (al - be) ** 2 + (be - ga) ** 2

        return ab_error + bc_error + angle_eq_error

    def cube_reduction_penalty(self, cell_lengths, cell_angles, margin):
        a, b, c = cell_lengths.unbind(dim=-1)
        al, be, ga = cell_angles.unbind(dim=-1)
        # no reduction terms

        # crystal system enforcement
        # lengths equal
        ab_error = (a - b) ** 2
        bc_error = (b - c) ** 2

        # right angles
        alpha_error = (al - torch.pi / 2) ** 2
        beta_error = (be - torch.pi / 2) ** 2
        gamma_error = (ga - torch.pi / 2) ** 2

        return ab_error + bc_error + alpha_error + beta_error + gamma_error
