"""

MolData Initialization
"""

from mxtaltools.dataset_utils.data_classes import MolData

# explicit enumeration
data = MolData(
    z=atomic_numbers_tensor,
    pos=coordinates_tensor,
    smiles=smiles_string)

# from list of smiles
base_molData = MolData()
molecule = base_molData.from_smiles(
    "CCC",
    compute_partial_charges=True,
    minimize=True,
    protonate=True,
    scramble_dihedrals=False)
"""
Batch Initialization
"""
from mxtaltools.dataset_utils.utils import collate_data_list
molecule_batch = collate_data_list(list_of_MolData_objects)


"""
Molecule crystal initialization
"""
from mxtaltools.dataset_utils.data_classes import MolCrystalData
crystal = MolCrystalData(
    molecule=molecule,
    sg_ind=space_group_number,
    cell_lengths=cell_lengths,
    cell_angles=cell_angles,
    aunit_centroid=aunit_centroids,
    aunit_orientation=aunit_orientation,
    aunit_handedness=aunit_handedness)

"""
Example, density
"""

molecule_batch = collate_data_list(list_of_MolData_objects).to(device)

model = load_molecule_scalar_regressor(model_checkpoint_path, device)  # load model

predicted_packing_coefficient = model(molecule_batch).flatten() * model.target_std + model.target_mean
predicted_asymmetric_unit_volume = molecule_batch.mol_volume / packing_coeff_pred  # A^3
predicted_density = molecule_batch.mass / aunit_volume_pred * 1.6654  # g/cm^3

"""
Example, crystal analysis
"""
from mxtaltools.models.utils import softmax_and_score
from mxtaltools.dataset_utils.utils import collate_data_list
from torch.nn.functional import softplus

crystal_batch = collate_data_list(list_of_MolCrystalData_objects)  # batch crystal objects

model = load_crystal_score_model(checkpoint, device)  # load crystal score model

lj_pot, es_pot, scaled_lj_pot, cluster_batch = (
    crystal_batch.build_and_analyze(return_cluster=True))  # analyze crystal

model_output = model(cluster_batch)  # consult crystal score model
classification_score = softmax_and_score(model_output[:, :2])
predicted_RDF_distance = softplus(model_output[:, 2])

cluster_batch.visualize([ind for ind in range(crystal_batch.num_graphs)],
                        mode='convolve with')  # visualize structures

"""
Example, autoencoder
"""
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.common.training_utils import load_molecule_autoencoder

molecule_batch = collate_data_list(list_of_MolData_objects).to(device)
molecule_batch.recenter_molecules()  # batch molecules

model = load_molecule_autoencoder(checkpoint, device)  # load model

vector_encoding = model.encode(molecule_batch.clone())  # generate embeddings
scalar_encoding = model.scalarizer(vector_encoding)

reconstruction_loss, rmsd, matched_molecule = (  # confirm reconstruction quality
    model.check_embedding_quality(molecule_batch, visualize=True))
