"""
standalone code for molecular crystal density prediction model
requirements numpy, scipy, torch, torch_geometric, torch_scatter, torch_cluster, pyyaml, tqdm

"""
import pathlib
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader.dataloader import Collater

from CrystalData import CrystalData
from models import Mo3ENet
from utils import (
    dict2namespace, load_yaml, batch_molecule_vdW_volume,
    VDW_RADII, ATOM_WEIGHTS, reload_model, collate_decoded_data, ae_reconstruction_loss, batch_rmsd, swarm_vs_tgt_fig
)

autoencoder_model_path = pathlib.Path(__file__).parent.resolve().joinpath('autoencoder_model.pt')


# noinspection PyAttributeOutsideInit
class MoleculeEncoder(torch.nn.Module):
    def __init__(self,
                 device='cpu',
                 ):
        super(MoleculeEncoder, self).__init__()

        self.device = device
        self.allowed_types = [1, 6, 7, 8, 9]
        self.num_atom_types = len(self.allowed_types)
        self.type_distance_scaling = 2
        self.sigma = 0.015
        self.vdw_radii = torch.tensor(list(VDW_RADII.values())).to(self.device)
        self.atomic_weights = torch.tensor(list(ATOM_WEIGHTS.values())).to(self.device)
        self.collater = Collater(None, None)
        self.renderer = 'browser'
        self.load_models()


    @torch.no_grad()
    def encode(self,
               atomic_numbers: list,
               atom_coordinates: list,
               ):
        """

        """

        mol_batch = self.prep_molecule_batch(atomic_numbers,
                                             atom_coordinates,
                                             )
        return self.model.encode(mol_batch)

    def decode(self,
               encoding: torch.FloatTensor
               ):
        """

        :param encoding:
        :return:
        """

        return self.model.decode(encoding)

    def encode_decode(self,
                      atomic_numbers: list,
                      atom_coordinates: list,
                      evaluate_encoding: bool = False,
                      visualize_decoding: bool = False,
                      ):
        mol_batch = self.prep_molecule_batch(atomic_numbers,
                                             atom_coordinates,
                                             )
        encoding = self.model.encode(mol_batch.clone())
        decoding = self.model.decode(encoding)

        if visualize_decoding:
            self.visualize_decoding(mol_batch.clone(), decoding)

        if not evaluate_encoding:
            return encoding, decoding
        else:
            report = self.evaluate_encoding(mol_batch.clone(), decoding)
            return encoding, decoding, report

    def evaluate_encoding(self, mol_batch, decoding):

        mol_batch.x = self.model.atom_embedding_vector[mol_batch.x].flatten()
        decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            collate_decoded_data(mol_batch,
                                 decoding,
                                 self.model.num_decoder_nodes,
                                 1,
                                 mol_batch.x.device))

        (nodewise_reconstruction_loss, nodewise_type_loss,
         graph_reconstruction_loss, self_likelihoods,
         nearest_node_loss, graph_clumping_loss,
         nearest_component_dist, nearest_component_loss) = ae_reconstruction_loss(mol_batch,
                                                                                  decoded_mol_batch,
                                                                                  nodewise_weights,
                                                                                  nodewise_weights_tensor,
                                                                                  self.num_atom_types,
                                                                                  self.type_distance_scaling,
                                                                                  self.sigma)

        (rmsd, pred_dists, complete_graph_bools,
         matched_particle_bools,
         pred_particle_points, pred_particle_weights) = batch_rmsd(
            mol_batch,
            decoded_mol_batch,
            F.one_hot(mol_batch.x, self.num_atom_types),
            intrapoint_cutoff=0.5,
            probability_threshold=0.25,
            type_distance_scaling=2
        )

        report = {
            'nodewise_reconstruction_loss': nodewise_reconstruction_loss,
            'nodewise_type_loss': nodewise_type_loss,
            'graph_reconstruction_loss': graph_reconstruction_loss,
            'self_likelihoods': self_likelihoods,
            'nearest_node_loss': nearest_node_loss,
            'graph_clumping_loss': graph_clumping_loss,
            'nearest_component_dist': nearest_component_dist,
            'nearest_component_loss': nearest_component_loss,
            'rmsd': rmsd,
            'pred_dists': pred_dists,
            'complete_graph_bools': complete_graph_bools,
            'matched_particle_bools': matched_particle_bools,
            'pred_particle_points': pred_particle_points,
            'pred_particle_weights': pred_particle_weights
        }
        return report

    def visualize_decoding(self, mol_batch, decoding):
        mol_batch.x = self.model.atom_embedding_vector[mol_batch.x].long().flatten()
        decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
            collate_decoded_data(mol_batch,
                                 decoding,
                                 self.model.num_decoder_nodes,
                                 1,
                                 mol_batch.x.device))

        figs = []
        for i in range(mol_batch.num_graphs):
            figs.append(swarm_vs_tgt_fig(
                mol_batch,
                decoded_mol_batch,
                self.num_atom_types,
                graph_ind=i
            ))
        [fig.show(renderer=self.renderer) for fig in figs]

    def prep_molecule_batch(self,
                            atom_types_list,
                            coords_list,
                            ):

        mol_batch = [
            CrystalData(
                x=atom_types_list[ind],
                pos=coords_list[ind],
                mol_size=torch.ones(1) * len(atom_types_list[ind]),
            )
            for ind in range(len(coords_list))
        ]
        mol_batch = self.collater(mol_batch).to(self.device)

        return mol_batch

    #
    def load_models(self):
        self.load_autoencoder_model()

    def load_autoencoder_model(self):
        """autoencoder_model"""
        checkpoint = torch.load(autoencoder_model_path, map_location=self.device)
        model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model
        self.config = Namespace(**{'autoencoder': Namespace(**{'optimizer': {}, 'model': {}})})

        self.config.autoencoder.optimizer = model_config.optimizer
        self.config.autoencoder.model = model_config.model

        type_translation_index = np.zeros(np.array(self.allowed_types).max() + 1) - 1
        for ind, atype in enumerate(self.allowed_types):
            type_translation_index[atype] = ind
        self.autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

        self.model = Mo3ENet(seed=12345,
                             config=self.config.autoencoder.model,
                             num_atom_types=self.num_atom_types,
                             atom_embedding_vector=self.autoencoder_type_index,
                             radial_normalization=1,  # overwritten
                             infer_protons=False,  # overwritten
                             protons_in_input=True,  # overwritten
                             )
        for param in self.model.parameters():  # freeze volume model
            param.requires_grad = False
        self.model, _ = reload_model(self.model,
                                     device=self.device,
                                     optimizer=None,
                                     path=autoencoder_model_path)
        self.model.eval()
        self.model.to(self.device)


def smiles2conformers(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    positions_list = []
    atom_types_list = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        positions_list.append(torch.tensor(pos - pos.mean(0), dtype=torch.float32, device=device))
        atom_types_list.append(torch.tensor(z, dtype=torch.long, device=device))

    return positions_list, atom_types_list


if __name__ == '__main__':
    # test this class
    device = 'cpu'

    # initialize autoencoder
    encoder = MoleculeEncoder(device=device)

    "load up some smiles"
    smiles_list = [
        "COCCN1CC2(C)CCC1C2",
        "NCC(F)C(=O)NC1CCNC1",
        "CN(CCCO)C(=O)CCCN",
        "CC1=CCC(=O)C1(C)O"
    ]

    # generate conformers
    positions_list, atom_types_list = smiles2conformers(smiles_list)

    # embed, visualize, and evaluate reconstruction statistics
    encoding, decoding, report = encoder.encode_decode(atom_types_list,
                                                       positions_list,
                                                       evaluate_encoding=True,
                                                       visualize_decoding=True
                                                       )

    print(report)