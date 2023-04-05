from argparse import Namespace

import torch

from crystal_building.builder import SupercellBuilder, update_crystal_symmetry_elements
from gflownet.proxy.base import Proxy
from models.discriminator_models import crystal_discriminator
from models.utils import softmax_and_score, compute_h_bond_score, get_vdw_penalty, cell_density_loss, reload_model

SCORE_MODELS = {
    "score_model_1": 'C:/Users\mikem\crystals\CSP_runs\models/cluster/best_discriminator_10088',
}


class MolecularCrystalScore(Proxy):
    def __init__(self, config, sym_info, mol_volume_ind,
                 model: str = "score_model_1", use_aux_scores=False, **kwargs):
        """
        Parameters
        ----------
        config : namespace
            Config object containing information on the model setup and characteristics of the dataset (dataDims)

        sym_info : dict
            Containing information on symmetry operations, point groups, lattice types and space group symbols for the 230 space groups

        mol_volume_ind : int
            index for the molecule volume in crystal_data.tracking features [n_crystals, n_features]

        model : str
            The name of the pretrained model to be used for prediction.

        use_aux_scores : bool
            Whether to use auxiliary heuristic scores in addition to score from the proxy model
        """
        super().__init__(**kwargs)
        self.use_aux_scores = use_aux_scores
        self.config = config
        self.sym_info = sym_info
        self.mol_volume_ind = mol_volume_ind

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
        self.supercell_builder = SupercellBuilder(self.sym_info, self.config.dataDims)

    @torch.no_grad()
    def __call__(
            self,
            molecule_data,
            crystal_parameters,
    ):
        """
        Args
        ----
        molecule_data : CrystalData object containing atom and molecule-scale features
        (same one used to condition crystal parameters' generation)

        crystal_parameters : proposed cell params in format [n_crystals, 13]
        SG_ind | a,b,c | alpha,beta,gamma | x,y,z | phi,psi,theta

        Returns
        ----
        scores : model score, optionally augmented by heuristic auxiliary functions

        """
        cell_params = crystal_parameters[:, 1:]
        space_groups = crystal_parameters[:, 0]

        # overwrite space group and accompanying symmetry info to molecule objects
        molecule_data = update_crystal_symmetry_elements(molecule_data,
                                                         space_groups,
                                                         self.config.dataDims,
                                                         self.sym_info)

        crystal_data, generated_cell_volumes, _ = self.supercell_builder.build_supercells(
            molecule_data,
            cell_params,
            supercell_size=5,
            graph_convolution_cutoff=6,
            align_molecules=False)

        model_output, dist_dict = self.model(crystal_data.clone(), return_dists=True)
        model_score = softmax_and_score(model_output)  # convert output to score

        if self.use_aux_scores:
            # h_bond_score = compute_h_bond_score(self.config.feature_richness, self.atom_acceptor_ind, self.atom_donor_ind, self.num_acceptors_ind, self.num_donors_ind, crystal_data)
            vdw_penalty, normed_vdw_penalty = get_vdw_penalty(self.vdw_radii, dist_dict, crystal_data.num_graphs, crystal_data)
            packing_loss, packing_prediction, packing_target, = cell_density_loss(
                None,
                self.config.dataDims['tracking features dict'].index('crystal packing coefficient'),
                self.mol_volume_ind,
                self.config.dataDims['target mean'],
                self.config.dataDims['target std'],
                crystal_data,
                cell_params,
                precomputed_volumes=generated_cell_volumes)

            model_score -= (packing_loss + vdw_penalty)  # subtract auxiliary losses

        # todo log auxiliary losses and a few crystal samples for analysis

        return model_score
