import numpy as np
import torch

from mxtaltools.common.geometry_utils import batch_compute_fractional_transform, get_batch_centroids
from mxtaltools.crystal_building.utils import fractional_transform, extract_aunit_orientation
from mxtaltools.dataset_utils.data_classes import MolData, MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from tests.utils import test_smiles

device = 'cuda'

"""
Molecules
"""


def test_MolData(device):
    mol_batch = generate_test_mol_batch(device)

    # check volume calculation
    assert torch.all(
        torch.isclose(mol_batch.volume_calculation(), mol_batch.mol_volume, atol=1)), "Error in mol volume calculation"
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

    # test radial calculation
    mol_batch.construct_radial_graph()  # todo add a test/assertion


def generate_test_mol_batch(device):
    base_molData = MolData()
    num_mols = len(test_smiles)
    mols = [base_molData.from_smiles(test_smiles[ind],
                                     pare_to_size=9,
                                     skip_mol_analysis=False,
                                     allow_methyl_rotations=True,
                                     compute_partial_charges=True,
                                     minimize=False,
                                     protonate=True,
                                     ) for ind in range(num_mols)]
    mols = [mol for mol in mols if mol is not None]
    mol_batch = collate_data_list(mols).to(device)
    return mol_batch


"""
Molecular Crystals
"""


def test_MolCrystalData(device):
    mol_batch = generate_test_mol_batch(device)

    device = mol_batch.device
    num_mols = mol_batch.num_graphs
    cell_lengths = torch.rand((num_mols, 3), device=device) * 4 * mol_batch.radius[:, None] + 4
    cell_angles = torch.randn((num_mols, 3), device=device) * 0.1 + torch.pi / 2
    aunit_centroids = torch.rand((num_mols, 3), device=device)
    aunit_orientation = torch.randn((num_mols, 3), device=device)  # rotvec representation is easier to initialize
    aunit_orientation[:, 2] = torch.abs(aunit_orientation[:, 2])  # still has to be upper half-plane
    rotvec_lens = torch.rand(num_mols, device=device) * 2 * torch.pi
    aunit_orientation = aunit_orientation / aunit_orientation.norm(dim=1)[:, None] * rotvec_lens[:, None]
    aunit_handedness = [int(np.random.choice([-1, 1])) for _ in range(num_mols)]  #[1 for _ in range(num_mols)]  #
    sg_inds = [int(np.random.randint(1, 50, 1)) for _ in range(num_mols)]

    mol_batch.noise_positions(1e-2)  # small asymmetry helps crystals behave better down the line

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
    T_fc_list, T_cf_list, cell_volumes = batch_compute_fractional_transform(cell_lengths, cell_angles)
    assert torch.all(
        torch.isclose(T_fc_list, crystal_batch.T_fc, atol=1e-3)), "Error in fractional transform calculation"
    assert torch.all(
        torch.isclose(T_cf_list, crystal_batch.T_cf, atol=1e-3)), "Error in fractional transform calculation"
    assert torch.all(
        torch.isclose(cell_volumes, crystal_batch.cell_volume, atol=1e-3)), "Error in fractional transform calculation"

    # this conversion is easier to do here
    crystal_batch.aunit_centroid = crystal_batch.scale_centroid_to_unit_cell(crystal_batch.aunit_centroid)

    # pose the asymmetric unit
    crystal_batch.pose_aunit()
    # confirm aunit centroids are correct
    assert torch.all(torch.isclose(
        fractional_transform(get_batch_centroids(crystal_batch.pos,
                                                 crystal_batch.batch,
                                                 crystal_batch.num_graphs),
                             crystal_batch.T_cf),
        crystal_batch.aunit_centroid,
        atol=1e-3)), "Error in aunit placement/analysis"
    # confirm aunit orientations are correct
    orientations, handedness = extract_aunit_orientation(
        crystal_batch,
        False,
        canonicalize_orientation=True
    )
    assert torch.all(torch.isclose(crystal_batch.aunit_orientation, orientations,
                                   atol=1e-2)), "Error in aunit orientation posing/analysis"
    assert torch.all(
        torch.isclose(crystal_batch.aunit_handedness.long(), handedness.long().cpu())), "Error in handedness analysis"

    # build then reanalyze the crystal
    crystal_batch.build_unit_cell()
    crystal_batch.validate_cell_params(check_crystal_system=False)
    # rebuild cell from parameters and confirm the results are identical
    aunit_centroid, aunit_orientation, aunit_handedness, is_well_defined, pos = crystal_batch.reparameterize_unit_cell()

    assert torch.all(torch.isclose(aunit_centroid[is_well_defined], crystal_batch.aunit_centroid[is_well_defined],
                                   atol=1e-3)), "Reparameterization of centroids failed"
    assert torch.all(torch.isclose(aunit_orientation[is_well_defined], crystal_batch.aunit_orientation[is_well_defined],
                                   atol=1e-2)), "Reparameterization of rotvecs failed"
    assert torch.all(torch.isclose(aunit_handedness[is_well_defined].cpu(), crystal_batch.aunit_handedness[is_well_defined],
                                   atol=1e-8)), "Reparamterization of handedness failed"

    """
    cluster generation & analysis
    """
    cluster_batch = crystal_batch.build_cluster()
    cluster_batch.construct_radial_graph()
    lj_en, scaled_lj_en = cluster_batch.compute_LJ_energy()
    es_en = cluster_batch.compute_ES_energy()
    assert torch.all(torch.isfinite(lj_en)), "NaN LJ potentials"
    assert torch.all(torch.isfinite(es_en)), "NaN electrostatic potentials"


test_MolData(device)
test_MolCrystalData(device)
