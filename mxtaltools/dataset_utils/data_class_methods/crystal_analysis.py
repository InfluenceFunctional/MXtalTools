from typing import Optional

import torch

from mxtaltools.analysis.crystal_rdf import crystal_rdf
from mxtaltools.analysis.vdw_analysis import get_intermolecular_dists_dict, lj_analysis, electrostatic_analysis, \
    buckingham_energy, silu_energy, vdW_analysis
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.utils import collate_data_list
#from mxtaltools.mlip_interfaces.uma_utils import compute_crystal_uma_on_mxt_batch
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
                             'es': self.compute_ES_energy,
                             'bh': self.compute_buckingham_energy,
                             'silu': self.compute_silu_energy,
                             'vdw': self.compute_vdW_overlap,
                             'ellipsoid': self.compute_ellipsoidal_overlap,
                             'niggli': self.compute_niggli_overlap
                             }

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
        """
        full procedure for building and analyzing a molecular crystal
        this is the OLD version of this function - best practice is to use the 'analyze' function below
        """

        cluster_batch = self.mol2cluster(cutoff, supercell_size)

        if noise is not None:
            cluster_batch.pos += torch.randn_like(cluster_batch.pos) * noise

        cluster_batch.construct_radial_graph(cutoff=cutoff)

        outputs = cluster_batch.compute(requests=['lj', 'es'])
        self.lj_pot = outputs['lj']
        self.es_pot = outputs['es']
        self.scaled_lj_pot = log_rescale_positive(self.lj_pot)

        if return_cluster:
            return self.lj_pot, self.es_pot, self.scaled_lj_pot, cluster_batch
        else:
            return self.lj_pot, self.es_pot, self.scaled_lj_pot

    def analyze(self,
                computes: list,
                return_cluster: Optional[bool] = False,
                noise: Optional[float] = None,
                cutoff: float = 6,
                supercell_size: int = 10,
                std_orientation: Optional[bool] = True,
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

        if return_cluster:
            return results, cluster_batch
        else:
            return results

    # def compute_crystal_uma(self,
    #                         predictor,
    #                         std_orientation: bool=True,
    #                         ):
    #     if self.is_batch:
    #         return compute_crystal_uma_on_mxt_batch(self,
    #                                                 std_orientation,
    #                                                 predictor)
    #     else:
    #         return compute_crystal_uma_on_mxt_batch(collate_data_list(self),
    #                                                 std_orientation,
    #                                                 predictor)

    def compute_rdf(self,
                    cutoff: float = 6,
                    mode: str = 'intermolecular',
                    elementwise: bool = True,
                    # atomwise: bool = False  # doesn't work properly right now
                    bins: int = 2000,
                    raw_density: bool = True
                    ):
        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Must build radial graph before computing RDF!"
                               "Do batch.construct_radial_graph before calling.")

        assert self.is_batch, "RDF method not implemented for single crystals"
        rdf, bin_edges, rdf_pair_dict = crystal_rdf(self,
                                                    self.edges_dict,
                                                    rrange=(0, cutoff),
                                                    raw_density=raw_density,
                                                    mode=mode,
                                                    bins=bins,
                                                    elementwise=elementwise,
                                                    atomwise=False,
                                                    )
        return rdf, bin_edges, rdf_pair_dict
