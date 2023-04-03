import pickle
from argparse import Namespace
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from sklearn.ensemble import RandomForestRegressor

from crystal_building.builder import SupercellBuilder
from gflownet.proxy.base import Proxy
from gflownet.utils.common import download_file_if_not_exists
from models.discriminator_models import crystal_discriminator
from models.utils import softmax_and_score, compute_h_bond_score, get_vdw_penalty, cell_density_loss, reload_model

SCORE_MODELS = {
    "score_model_1": 'C:/Users\mikem\crystals\CSP_runs\models/cluster/best_discriminator_10088',
}


def reload_model_checkpoints(config):
    pass


class MolecularCrystalScore(Proxy):
    def __init__(self, model: str = "score_model_1", use_aux_scores=False, **kwargs):
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model to be used for prediction.

        use_aux_scores : bool
            Whether to use auxiliary heuristic scores in addition to score from the proxy model
        """
        super().__init__(**kwargs)
        self.use_aux_scores = use_aux_scores

        if SCORE_MODELS.get(model) is None:
            raise ValueError(
                f'Tried to use model "{model}", '
                f"but only {set(SCORE_MODELS.keys())} are available."
            )
        else:
            self.model_path = SCORE_MODELS[model]

        # prep discriminator model
        discriminator_checkpoint = torch.load(self.model_path)
        self.config.discriminator = Namespace(**discriminator_checkpoint['config'])
        self.model = crystal_discriminator(self.config, self.config.dataDims)  # instantiate the model
        self.model, _ = reload_model(self.model, None, self.config.discriminator_path)  # reload weights

        # prep crystal builder
        self.supercell_builder = SupercellBuilder(self.sym_ops, self.sym_info, self.normed_lattice_vectors,
                                                  self.atom_weights, self.config.dataDims)

    @torch.no_grad()
    def __call__(
            self,
            molecule_data,
            crystal_parameters,
    ):
        """
        Args
        ----
        molecule_data : CrystalData object containing atom, and molecule-scale features

        crystal_parameters : proposed cell params
        SG_ind | a,b,c | alpha,beta,gamma | x,y,z | phi,psi,theta

        Returns
        ----
        scores : model score, optionally augmented by heuristic auxiliary functions

        """
        crystal_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
            molecule_data, crystal_parameters, self.config.supercell_size,
            self.config.discriminator.graph_convolution_cutoff,
            align_molecules=False,
        )
        model_output, dist_dict = self.model(crystal_data.clone(), return_dists=True)

        model_score = softmax_and_score(model_output)

        if self.use_aux_scores:
            h_bond_score = compute_h_bond_score(self.config.feature_richness, self.atom_acceptor_ind, self.atom_donor_ind, self.num_acceptors_ind, self.num_donors_ind, supercell_data)
            vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii, dist_dict, crystal_data.num_graphs, crystal_data)
            packing_loss, packing_prediction, packing_target, = \
                cell_density_loss(self.config.packing_loss_rescaling,
                                  self.config.dataDims['tracking features dict'].index('crystal packing coefficient'),
                                  self.mol_volume_ind,
                                  self.config.dataDims['target mean'], self.config.dataDims['target std'],
                                  crystal_data, crystal_parameters, precomputed_volumes=generated_cell_volumes)

        return model_score
