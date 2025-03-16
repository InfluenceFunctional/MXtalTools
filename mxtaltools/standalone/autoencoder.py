from argparse import Namespace

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.autoencoder_utils import ae_reconstruction_loss, batch_rmsd, init_decoded_data, get_node_weights
from mxtaltools.models.task_models.autoencoder_models import Mo3ENet
import torch.nn.functional as F



def collate_decoded_data(data, decoding, num_decoder_nodes, node_weight_temperature, device):
    # generate input reconstructed as a data type
    decoded_mol_batch = init_decoded_data(data,
                                          decoding,
                                          device,
                                          num_decoder_nodes
                                          )
    # compute the distributional weight of each node
    nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = \
        get_node_weights(data, decoded_mol_batch, decoding,
                         num_decoder_nodes,
                         node_weight_temperature)
    decoded_mol_batch.aux_ind = nodewise_weights_tensor
    # input node weights are always 1 - corresponding each to an atom
    data.aux_ind = torch.ones(data.num_nodes, dtype=torch.float32, device=device)
    # get probability distribution over type dimensions
    decoded_mol_batch.x = F.softmax(decoding[:, 3:-1], dim=1)
    return decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor




def evaluate_encoding(autoencoder, mol_batch, decoding):
    mol_batch.x = autoencoder.atom_embedding_vector[mol_batch.z].flatten()
    decoded_mol_batch, nodewise_graph_weights, nodewise_weights, nodewise_weights_tensor = (
        collate_decoded_data(mol_batch,
                             decoding,
                             autoencoder.num_decoder_nodes,
                             1,
                             mol_batch.x.device))

    (nodewise_reconstruction_loss, nodewise_type_loss,
     graph_reconstruction_loss, self_likelihoods,
     nearest_node_loss, graph_clumping_loss,
     nearest_component_dist, nearest_component_loss) = ae_reconstruction_loss(mol_batch,
                                                                              decoded_mol_batch,
                                                                              nodewise_weights,
                                                                              nodewise_weights_tensor,
                                                                              5,
                                                                              2,
                                                                              0.015)

    (rmsd, pred_dists, complete_graph_bools,
     matched_particle_bools,
     pred_particle_points, pred_particle_weights) = batch_rmsd(
        mol_batch,
        decoded_mol_batch,
        F.one_hot(mol_batch.x, 5),
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


def load_encoder(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = Namespace(**checkpoint['config'])  # overwrite the settings for the model

    allowed_types = np.array([1, 6, 7, 8, 9])
    type_translation_index = np.zeros(allowed_types.max() + 1) - 1
    for ind, atype in enumerate(allowed_types):
        type_translation_index[atype] = ind
    autoencoder_type_index = torch.tensor(type_translation_index, dtype=torch.long, device='cpu')

    model = Mo3ENet(
        0,
        model_config.model,
        5,
        autoencoder_type_index,
        1,  # will get overwritten
        protons_in_input=True
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if list(checkpoint['model_state_dict'])[0][
       0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
        for i in list(checkpoint['model_state_dict']):
            checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    return pos, z


if __name__ == "__main__":
    device = 'cpu'
    checkpoint_path = 'autoencoder.pt'
    smiles_list = ['CCC#CCC#N', 'CCC(O)(C=O)C#N', 'CC(O)CC=O', 
                   'OCC1=C(O)C=CN1', 'COCC1(O)CC1O', 'CNC(=O)CCCO',
                   'CCC(=O)C(C)C', 'COCC(=O)N(C)C', 'NC1=C(NC=C1)C#C', 
                   'COC1(C)COC1C']
    mol_list = []
    for smiles in smiles_list:
        pos, z = smiles_to_mol(smiles)
        mol = MolData(
            z=torch.LongTensor(z, device=device),
            pos=torch.ttensor(pos, device=device, dtype=torch.float32),
        )
        mol_list.append(mol)

    mol_batch = collate_data_list(mol_list)

    autoencoder = load_encoder(
        checkpoint_path
    ).to(device)
    
    vector_embedding = autoencoder.encode(mol_batch.clone())
    scalar_embedding = autoencoder.scalarizer(vector_embedding)