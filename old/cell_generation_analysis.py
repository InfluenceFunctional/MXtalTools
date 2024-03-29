from ccdc.search import EntryReader
from utils import *
from crystal_builder_tools import *
from pymatgen.symmetry import analyzer
from pymatgen.analysis import structure_matcher
from pymatgen.core import (structure, lattice)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # annoying numpy error
from nikos.coordinate_transformations import coor_trans_matrix


def parallel_cell_build_analyze(data, sym_ops, atom_weights):
    '''
    script to study how the generator is busted
    '''
    '''
    direct from the CSD
    '''
    csd_reader = EntryReader('CSD')  # compare to CSD directly
    csd_entry = csd_reader.entry(data.csd_identifier)
    csd_entry.molecule.remove_hydrogens()
    csd_supercell = csd_entry.crystal.packing(box_dimensions=((0, 0, 0), (1, 1, 1)))
    csd_supercell.remove_hydrogens()

    # recalculate cell matrix
    cell_lengths = np.asarray(csd_entry.crystal.cell_lengths)
    cell_angles = np.asarray(csd_entry.crystal.cell_angles)
    redo_T_fc = coor_trans_matrix('f_to_c', cell_lengths, cell_angles / 180 * np.pi)

    supercell_species = np.asarray([csd_supercell.atoms[n].atomic_symbol for n in range(len(csd_supercell.atoms))])
    supercell_coords = np.asarray([csd_supercell.atoms[n].coordinates for n in range(len(csd_supercell.atoms))])
    supercell_coords_f = np.linalg.inv(data.T_fc[0]).dot(supercell_coords.T).T

    molwise_supercell_coords = supercell_coords.reshape(int(data.Z), int(len(data.pos)), 3)
    molwise_supercell_coords_f = supercell_coords_f.reshape(int(data.Z), int(len(data.pos)), 3)

    '''
    extract cell parameters
    '''
    # canonical fractional centroid
    csd_centroids = molwise_supercell_coords_f.mean(1)
    for i in range(data.Z):
        molwise_supercell_coords_f[i] -= np.floor(csd_centroids[i]) # ensure we're in the cell
        molwise_supercell_coords[i] -= data.T_fc[0].numpy().dot(np.floor(csd_centroids[i]))
    csd_centroids = molwise_supercell_coords_f.mean(1)
    csd_canonical_ind = torch.argmin(torch.linalg.norm(torch.Tensor(csd_centroids), dim=1))
    csd_canonical_centroid = csd_centroids[csd_canonical_ind]
    csd_canonical_mol_coords = torch.Tensor(molwise_supercell_coords[csd_canonical_ind])

    '''
    from our stored CSD data
    '''
    struc_lattice = lattice.Lattice(data.T_fc[0].T.type(dtype=torch.float16))
    pymat_struc = structure.IStructure(species=data.x[:, 0].repeat(data.Z), coords=data.ref_cell_pos.reshape(int(data.Z * len(data.pos)), 3), lattice=struc_lattice, coords_are_cartesian=True)
    sg_analyzer = analyzer.SpacegroupAnalyzer(pymat_struc)
    print(sg_analyzer.get_space_group_symbol())

    '''
    ###
    generated by our differentiable tool
    ###
    '''
    atoms = torch.Tensor([data.x[n, 0] for n in range(len(data.x))])
    masses = torch.Tensor([atom_weights[int(data.x[n, 0])] for n in range(len(data.x))])
    coords = data.pos
    T_fc = data.T_fc[0]
    T_cf = torch.linalg.inv(T_fc)
    coords_f = torch.inner(T_cf, coords).T


    '''
    apply translation
    '''
    coords_f_trans = (coords_f - coords_f.mean(0) + csd_canonical_centroid).float()
    '''
    get canonical conformer
    '''
    symmetries = torch.Tensor(sym_ops[int(data.sg_ind)])
    affine_coords_f = torch.cat((coords_f_trans, torch.ones(len(coords)).to(coords.device)[:, None]), dim=1)
    sym_coords = torch.stack([
        torch.inner(symmetries[n], affine_coords_f).T[:, :-1]
        for n in range(int(data.Z))
    ])
    gen_centroids = sym_coords.mean(1)
    gen_centroids -= torch.floor(gen_centroids)
    gen_canonical_centroid_ind = torch.argmin(torch.linalg.norm(gen_centroids, dim=1))

    # gen_canonical_coords_f = coords_f_trans - coords_f_trans.mean(0) + gen_centroids[gen_canonical_centroid_ind]
    gen_canonical_coords = coords - coords.mean(0) + torch.inner(T_fc, gen_centroids[gen_canonical_centroid_ind])

    cent_dists = torch.cdist(gen_centroids, torch.Tensor(csd_centroids - np.floor(csd_centroids)), p=2)
    cent_overlaps = torch.sum(cent_dists < 0.01)
    assert cent_overlaps / data.Z == 1 # centroids agree
    assert torch.sum(torch.abs((torch.inner(T_cf, gen_canonical_coords.mean(0)) - csd_canonical_centroid))) < 0.05

    '''
    retrieve the rotation of the target
    '''
    Ip_axes, _, _ = compute_principal_axes_torch(csd_canonical_mol_coords, masses)
    target_handedness = compute_Ip_handedness(Ip_axes)

    normed_alignment_target = torch.eye(3)
    normed_alignment_target[0,0] = target_handedness # don't invert during rotation

    rot_matrix = torch.matmul(normed_alignment_target.T, torch.linalg.inv(Ip_axes.float()).T)  # rotation matrix between target and current Ip axes
    components = torch.Tensor(Rotation.from_matrix(torch.linalg.inv(rot_matrix)).as_rotvec())  # CRITICAL we want the inverse transform here

    standardized_csd_canonical_coords = torch.inner(rot_matrix, csd_canonical_mol_coords - csd_canonical_mol_coords.mean(0)).T  # standard, non-inverted rotation
    Ip_axes_csd_std, _, _ = compute_principal_axes_torch(standardized_csd_canonical_coords, masses)

    '''
    get std rotation for the generated conformer
    '''
    Ip_axes, _, _ = compute_principal_axes_torch(gen_canonical_coords, masses)
    handedness = compute_Ip_handedness(Ip_axes)
    if handedness != target_handedness:
        gen_canonical_coords = -(gen_canonical_coords - gen_canonical_coords.mean(0)) + gen_canonical_coords.mean(0)
        Ip_axes, _, _ = compute_principal_axes_torch(gen_canonical_coords, masses)

    normed_alignment_target = torch.eye(3)
    normed_alignment_target[0,0] = target_handedness # don't invert during rotation

    std_rotation_matrix = torch.matmul(normed_alignment_target.T, torch.linalg.inv(Ip_axes).T)  # rotation matrix between target and current Ip axes

    standardized_gen_canonical_coords = torch.inner(std_rotation_matrix, gen_canonical_coords - gen_canonical_coords.mean(0)).T
    Ip_axes_gen_std, _, _ = compute_principal_axes_torch(standardized_gen_canonical_coords, masses)

    assert torch.sum(torch.abs(normed_alignment_target - Ip_axes_gen_std)) < 0.1  # confirm the rotation was done correctly
    assert torch.sum(torch.abs(normed_alignment_target - Ip_axes_csd_std)) < 0.1  # confirm the rotation was done correctly
    assert torch.sum(torch.abs(standardized_csd_canonical_coords - standardized_gen_canonical_coords)) < 0.1

    '''
    generate rotation matrix for the target roation
    '''
    theta = torch.linalg.norm(components)
    unit_vector = components / theta
    q = torch.cat([torch.cos(theta / 2)[None], unit_vector * torch.sin(theta / 2)])

    applied_rotation = torch.Tensor(((1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]),
                                     (2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]),
                                     (2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2)))
    '''
    apply rotations
    '''
    gen_final_coords = torch.inner(torch.matmul(applied_rotation, std_rotation_matrix), gen_canonical_coords - gen_canonical_coords.mean(0)).T + gen_canonical_coords.mean(0)
    assert torch.sum(torch.abs(gen_final_coords - csd_canonical_mol_coords)) < 0.1

    '''
    apply point symmetry
    '''
    gen_cell_coords = torch.zeros_like(torch.Tensor(molwise_supercell_coords))
    gen_final_coords_f = torch.inner(T_cf, gen_final_coords).T
    affine_centroid = torch.cat((gen_final_coords_f.mean(0), torch.ones(1)))
    for i in range(data.Z):
        new_centroid = torch.inner(symmetries[i], affine_centroid)[:-1]
        new_centroid -= torch.floor(new_centroid)
        gen_cell_coords[i] = torch.inner(symmetries[i][:3, :3], gen_final_coords - gen_final_coords.mean(0)).T + torch.inner(T_fc, new_centroid)
    gen_cell_coords_flat = gen_cell_coords.reshape(data.Z * len(coords), 3)
    gen_cell_centroids_f = torch.inner(T_cf, gen_cell_coords.mean(1)).T
    '''
    unit cell comparison
    '''
    # check if the positions are identical
    dists = torch.cdist(gen_cell_coords_flat, torch.Tensor(supercell_coords), p=2)
    overlaps = torch.sum(dists < 0.05)
    assert (overlaps / len(gen_cell_coords_flat)) > .99

    cent_dists = torch.cdist(gen_cell_centroids_f, torch.Tensor(csd_centroids), p=2)
    cent_overlaps = torch.sum(cent_dists < 0.005)
    assert cent_overlaps / data.Z == 1

    gen_struc = structure.Structure(species=supercell_species, coords=gen_cell_coords_flat.numpy(), lattice=struc_lattice, coords_are_cartesian=True)
    csd_struc = structure.Structure(species=supercell_species, coords=supercell_coords, lattice=struc_lattice, coords_are_cartesian=True)
    sg_analyzer1 = analyzer.SpacegroupAnalyzer(gen_struc)
    sg_analyzer2 = analyzer.SpacegroupAnalyzer(csd_struc)

    print(sg_analyzer1.get_space_group_symbol())
    print(sg_analyzer2.get_space_group_symbol())
    matcher = structure_matcher.StructureMatcher()
    return matcher.fit(csd_struc, gen_struc), overlaps / len(gen_cell_coords_flat)
    #
    # '''
    # build supercells with our method and pymatgen and check what sticks
    # issue! pymatgen includes all atoms in the supercell, whereas we include all molecules with inside centroids
    # '''
    #
    # supercell_scale = 1
    # n_cells = (2 * supercell_scale + 1) ** 3
    # fractional_translations = torch.zeros((n_cells, 3))  # initialize the translations in fractional coords
    # i = 0
    # for xx in range(2*supercell_scale + 1):#-supercell_scale, supercell_scale + 1):
    #     for yy in range(2*supercell_scale + 1):#-supercell_scale, supercell_scale + 1):
    #         for zz in range(2*supercell_scale + 1):#-supercell_scale, supercell_scale + 1):
    #             fractional_translations[i] = torch.tensor((xx, yy, zz))
    #             i += 1
    # sorted_fractional_translations = fractional_translations[torch.argsort(fractional_translations.abs().sum(1))].to(T_fc.device)
    #
    # gen_supercell = gen_cell_coords_flat.tile(n_cells, 1)  # duplicate over XxXxX supercell
    # cart_translations_i = torch.mul(T_fc.tile(n_cells, 1), sorted_fractional_translations.reshape(n_cells * 3, 1))  # 3 dimensions
    # cart_translations = torch.stack(cart_translations_i.split(3, dim=0), dim=0).sum(1)
    # gen_supercell += torch.repeat_interleave(cart_translations, data.Z * len(coords),dim=0)
    #
    # struc_lattice = lattice.Lattice(data.T_fc[0].type(dtype=torch.float16))
    # pymat_supercell = structure.Structure(species=supercell_species, coords=gen_cell_coords_flat.numpy(), lattice=struc_lattice, coords_are_cartesian=True)
    # pymat_supercell.make_supercell(scaling_matrix=[3,3,3])
    #
    # dists = torch.cdist(gen_supercell, torch.Tensor(pymat_supercell.cart_coords),p=2)
    # overlaps = torch.sum(dists < 0.05)
    # print(f'{overlaps} overlaps out of {len(gen_supercell)} atoms at 0.05')
    #
    # gen_struc = structure.Structure(species=np.tile(supercell_species,n_cells), coords=gen_supercell.numpy(), lattice=struc_lattice, coords_are_cartesian=True)
    # sg_analyzer1 = analyzer.SpacegroupAnalyzer(pymat_struc)
    # sg_analyzer2 = analyzer.SpacegroupAnalyzer(gen_struc)
    #
    # print(sg_analyzer1.get_space_group_symbol())
    # print(sg_analyzer2.get_space_group_symbol())
    #
