import torch
import numpy as np

from mxtaltools.common.geometry_utils import batch_molecule_vdW_volume, mol_batch_vdW_volume, \
    compute_fractional_transform_torch, get_batch_centroids
from mxtaltools.conformer_generation.conformer_generator import embed_mol, extract_mol_info
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_building.utils import fractional_transform, extract_aunit_orientation
from mxtaltools.dataset_utils.construction.featurization_utils import get_partial_charges
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData, MolClusterData
from mxtaltools.dataset_utils.utils import collate_data_list
from tests.utils import test_smiles

"""
molecules
1. Initialize random molecules
2. generate molecule objects
3. collate to batch
4. check correct properties and indexing behavior

crystals
1. initialize random crystals
2. generate crystal objects
    a. test unit cell construction and parameterization
3. collate to batch
4. check correct properties and indexing behavior

supercells
1. generate supercells
2. confirm correct molecule and aux indexing
3. test analysis functions
"""

device = 'cpu'

"""
Molecules
"""
def t_MolData(device):
    coords_list, atom_types_list, atom_features_list, graph_features_list, targets_list = [], [], [], [], []
    for smile in test_smiles:
        mol = embed_mol(smile, protonate=True)
        if mol:
            conf = mol.GetConformer()
            mol, conf, pos, z, edge_index, adjacency_matrix, G = extract_mol_info(mol, conf)
            charges = get_partial_charges(mol)
            coords_list.append(pos)
            atom_types_list.append(z)
            atom_features_list.append(charges)
            graph_features_list.append(np.zeros(1))
            targets_list.append(np.zeros(1))

    num_mols = len(coords_list)
    mols = [
        MolData(
            z=torch.LongTensor(atom_types_list[ind], device=device),
            pos=torch.FloatTensor(coords_list[ind], device=device),
            x=torch.FloatTensor(atom_features_list[ind], device=device),
            graph_x=torch.FloatTensor(graph_features_list[ind], device=device),
            y=torch.FloatTensor(graph_features_list[ind], device=device),
            smiles=str(ind),
            skip_mol_analysis=False,
        )
        for ind in range(num_mols)
    ]

    mol_batch = collate_data_list(mols)

    # check volume calculation
    assert torch.all(
        torch.isclose(mol_batch.volume_calculation(), mol_batch.mol_volume)), "Error in mol volume calculation"
    # check radius calculation
    assert torch.all(torch.isclose(mol_batch.radius_calculation(), mol_batch.radius)), "Error in mol volume calculation"
    # check mass calculation
    assert torch.all(torch.isclose(mol_batch.mass_calculation(), mol_batch.mass)), "Error in mol mass calculation"
    # check recentering
    mol_batch.recenter_molecules()
    assert torch.all(torch.isclose(get_batch_centroids(mol_batch.pos, mol_batch.batch, mol_batch.num_graphs),
                                   torch.zeros((mol_batch.num_graphs, 3), dtype=torch.float32, device=mol_batch.device),
                                   atol=1e-6)), \
        "Error in recentering calculation"
    return mol_batch


mol_batch = t_MolData(device)
"""
Molecular Crystals
"""
def t_MolCrystalData(mol_batch, device):
    num_mols = mol_batch.num_graphs
    cell_lengths = torch.rand((num_mols, 3)) * 4 * mol_batch.radius[:, None]
    cell_angles = torch.randn((num_mols, 3)) * 0.1 + torch.pi / 2
    aunit_centroids = torch.rand((num_mols, 3))
    aunit_orientation = torch.randn((num_mols, 3))  # rotvec representation is easier to initialize
    aunit_orientation[:, 2] = torch.abs(aunit_orientation[:, 2])  # still has to be upper half-plane
    rotvec_lens = torch.rand(num_mols) * 2 * torch.pi
    aunit_orientation = aunit_orientation / aunit_orientation.norm(dim=1)[:, None] * rotvec_lens[:, None]
    aunit_handedness = [1 for _ in range(num_mols)] #[int(np.random.choice([-1, 1])) for _ in range(num_mols)]
    sg_inds = [int(np.random.randint(1, 2, 1)) for _ in range(num_mols)]#[int(np.random.randint(1, 50, 1)) for _ in range(num_mols)]

    # TODO try one-by-one and batchwise
    crystals = [
        MolCrystalData(
            molecule=mol_batch[ind],
            sg_ind=sg_inds[ind],
            cell_lengths=cell_lengths[ind],
            cell_angles=cell_angles[ind],
            aunit_centroid=aunit_centroids[ind],
            aunit_orientation=aunit_orientation[ind],
            aunit_handedness=aunit_handedness[ind],
            identifier=str(ind),
        )
        for ind in range(num_mols)
    ]

    crystal_batch = collate_data_list(crystals)
    # ensure everything inherited properly
    for key in mol_batch.keys():
        if torch.is_tensor(mol_batch[key]):
            assert torch.all(torch.isclose(mol_batch[key], crystal_batch[key])), "mol->crystal inheritance broken"

    # check box parameters
    T_fc_list, T_cf_list, cell_volumes = compute_fractional_transform_torch(cell_lengths, cell_angles)
    assert torch.all(torch.isclose(T_fc_list, crystal_batch.T_fc, atol=1e-6)), "Error in fractional transform calculation"
    assert torch.all(torch.isclose(T_cf_list, crystal_batch.T_cf, atol=1e-6)), "Error in fractional transform calculation"
    assert torch.all(
        torch.isclose(cell_volumes, crystal_batch.cell_volume, atol=1e-6)), "Error in fractional transform calculation"

    # this conversion is easier to do here
    crystal_batch.aunit_centroid = crystal_batch.scale_centroid_to_unit_cell(crystal_batch.aunit_centroid)
    crystal_batch.pose_aunit()
    # confirm aunit centroids are correct
    assert torch.all(torch.isclose(
        fractional_transform(get_batch_centroids(crystal_batch.pos, crystal_batch.batch, crystal_batch.num_graphs),
                             crystal_batch.T_cf),
        crystal_batch.aunit_centroid,
        atol=1e-5)), "Error in aunit placement/analysis"
    # confirm aunit orientations are correct
    orientations, handedness = extract_aunit_orientation(
        [crystal_batch[ind].pos for ind in range(crystal_batch.num_graphs)],
        False,
        crystal_batch.aunit_centroid,
        canonicalize_orientation=True
    )
    assert torch.all(torch.isclose(crystal_batch.aunit_orientation, orientations, atol=1e-4)), "Error in aunit orientation posing/analysis"

    crystal_batch.build_unit_cell()
    crystal_batch.validate_cell_params(check_crystal_system=False)
    # rebuild cell from parameters and confirm the results are identical
    crystal_batch.reparameterize_unit_cell()

    return crystal_batch


crystal_batch = t_MolCrystalData(mol_batch, device)

"""
Molecule Clusters
"""
def t_MolClusterData(crystal_batch):
    num_mols = crystal_batch.num_graphs
    aux_inds_list, mol_inds_list, cluster_z_list, cluster_pos_list, cluster_x_list = [], [], [], [], []

    clusters = [
        MolClusterData(
            crystal=crystal_batch[ind],
            aux_ind=aux_inds_list[ind],
            mol_ind=mol_inds_list[ind],
            cluster_z=cluster_z_list[ind],
            cluster_pos=cluster_pos_list[ind],
            cluster_x=cluster_x_list[ind],
        )
        for ind in range(num_mols)
    ]
    cluster_batch = collate_data_list(clusters)
    return cluster_batch

cluster_batch = t_MolClusterData(crystal_batch)
test_stop = 1
