from typing import Any, List, Optional

import numpy as np
import plotly.graph_objects as go
import torch
import torch.utils.data
from plotly.subplots import make_subplots
from rdkit import RDLogger
from torch_geometric.data import Batch

from mxtaltools.common.geometry_calculations import batch_molecule_vdW_volume
from mxtaltools.common.utils import init_sym_info
from mxtaltools.conformer_generation.conformer_generator import generate_random_conformers_from_smiles
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.crystal_search.standalone_crystal_opt import standalone_opt_random_crystals
from mxtaltools.dataset_management.CrystalData import CrystalData
from mxtaltools.models.functions.vdw_overlap import vdw_analysis
from mxtaltools.models.task_models.generator_models import CSDPrior
from mxtaltools.models.utils import get_intermolecular_dists_dict

RDLogger.DisableLog('rdApp.*')


class Collater:
    def __init__(
            self,
    ):
        self.abc = 1

    def __call__(self, batch: List[Any]) -> Any:
        return Batch.from_data_list(
            batch,
        )


def process_smiles_list_to_file(lines: list, file_path, allowed_atom_types, **conf_kwargs):
    samples = []
    for line in lines:
        sample, reason = process_smiles(line, allowed_atom_types, to_dict=False, **conf_kwargs)
        if sample is not None:
            samples.append(sample)

    print(f"finished processing smiles list with {len(samples)} samples")
    torch.save(samples, file_path)


def process_smiles_list(lines: list, allowed_atom_types, **conf_kwargs):
    samples = []
    for line in lines:
        #print(f'processing smile {line}')
        sample, reason = process_smiles(line, allowed_atom_types, to_dict=False, **conf_kwargs)
        if sample is not None:
            samples.append(sample)

    print(f"finished processing smiles list with {len(samples)} samples")
    return samples


def process_smiles_to_crystal_opt(lines: list,
                                  file_path,
                                  allowed_atom_types,
                                  space_group,
                                  run_tests=False,
                                  **conf_kwargs):
    """"""
    'starting chunk pool'
    mol_samples = process_smiles_list(lines, allowed_atom_types, **conf_kwargs)
    if len(mol_samples) == 0:
        assert False, "Zero valid molecules in batch, increase crystal generation batch size"

    collater = Collater()
    mol_batch = collater(mol_samples)

    print('''sample random crystals''')
    crystal_generator = CSDPrior(
        sym_info=init_sym_info(),
        device="cpu",
        cell_means=None,
        cell_stds=None,
        lengths_cov_mat=None)
    normed_cell_params = crystal_generator(mol_batch.num_graphs, space_group * torch.ones(mol_batch.num_graphs))
    mol_batch.sg_ind = space_group * torch.ones(mol_batch.num_graphs)

    print('''batch compute vdw volume''')
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device='cpu')
    mol_batch.mol_volume = batch_molecule_vdW_volume(mol_batch.x.flatten(),
                                                     mol_batch.pos,
                                                     mol_batch.batch,
                                                     mol_batch.num_graphs,
                                                     vdw_radii_tensor)

    print('''do local opt''')
    opt_vdw_pot, opt_vdw_loss, opt_packing_coeff, opt_cell_params, opt_aunits = standalone_opt_random_crystals(
        mol_batch.clone().cpu(),
        normed_cell_params.cpu(),
        opt_eps=1e-1,
        post_scramble_each=10,
        device='cpu',
    )

    print('''extract samples''')
    samples = []
    for graph_ind in range(mol_batch.num_graphs):
        graph_inds = mol_batch.batch == graph_ind
        for sample_ind in range(len(opt_vdw_pot)):
            cell_params = opt_cell_params[sample_ind, graph_ind]
            sample = CrystalData(
                x=mol_batch.x[graph_inds],
                pos=opt_aunits[sample_ind, graph_inds],
                smiles=mol_batch.smiles[graph_ind],
                identifier=mol_batch.smiles[graph_ind],
                y=torch.zeros(1, dtype=torch.float32),
                require_crystal_features=True,
                sg_ind=int(mol_batch.sg_ind[graph_ind]),
                z_prime=1,
                cell_lengths=cell_params[:3],
                cell_angles=cell_params[3:6],
                pose_parameters=cell_params[None, 6:],
                vdw_pot=opt_vdw_pot[sample_ind, graph_ind],
                vdw_loss=opt_vdw_loss[sample_ind, graph_ind],
                packing_coeff=opt_packing_coeff[sample_ind, graph_ind]
            )

            samples.append(sample)

    print(f"finished processing smiles list with {mol_batch.num_graphs} "
          f"molecules and optimizing crystals with {len(samples)} samples")
    if run_tests:
        test_crystal_rebuild_from_embedding(
            mol_batch,
            opt_vdw_pot,
            opt_vdw_loss,
            opt_aunits,
            opt_cell_params,
            denorm=False,
            destd=False,
            renorm=False,
            restd=False,
            make_figs=False,
        )
    torch.save(samples, file_path)


def test_crystal_rebuild_from_embedding(mol_batch,
                                        opt_vdw_pot,
                                        opt_vdw_loss,
                                        opt_aunits,
                                        opt_cell_params,
                                        denorm=False,
                                        destd=False,
                                        renorm=False,
                                        restd=False,
                                        make_figs=True,
                                        ):
    def _score_crystal_batch(mol_batch, supercell_data, vdw_radii_tensor):

        dist_dict = get_intermolecular_dists_dict(supercell_data, 6, 100)
        molwise_overlap, molwise_normed_overlap, vdw_potential, vdw_loss, lj_pot \
            = vdw_analysis(vdw_radii_tensor.cpu(),
                           dist_dict,
                           mol_batch.num_graphs, )
        return vdw_potential, vdw_loss

    from mxtaltools.crystal_building.builder import CrystalBuilder
    from mxtaltools.crystal_building.utils import overwrite_symmetry_info
    from mxtaltools.models.utils import denormalize_generated_cell_params, renormalize_generated_cell_params
    from mxtaltools.constants.atom_properties import VDW_RADII
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()), device='cpu')

    supercell_builder = CrystalBuilder(device='cpu',
                                       rotation_basis='cartesian')

    lattice_means = torch.tensor([
        1, 1, 1,
        torch.pi / 2, torch.pi / 2, torch.pi / 2,
        0.5, 0.5, 0.5,
        torch.pi / 4, 0, torch.pi / 2
    ], device='cpu', dtype=torch.float32)
    lattice_stds = torch.tensor([
        .35, .35, .35,
        .45, .45, .45,
        0.25, 0.25, 0.25,
        0.33, torch.pi / 2, torch.pi / 2
    ], device='cpu', dtype=torch.float32)
    rebuild_pot = torch.zeros_like(opt_vdw_pot)
    rebuild_loss = torch.zeros_like(rebuild_pot)
    rebuild_aunit = torch.zeros_like(opt_aunits)
    for sample_ind in range(len(opt_cell_params)):
        sample = opt_cell_params[sample_ind]

        if renorm:
            sample = renormalize_generated_cell_params(
                sample,
                mol_batch,
                supercell_builder.asym_unit_dict
            )
        if restd:
            sample = (sample - lattice_means[None, :]) / lattice_stds[None, :]

        if destd:
            sample = sample * lattice_stds[None, :] + lattice_means[None, :]
        if denorm:
            sample = denormalize_generated_cell_params(sample,
                                                       mol_batch,
                                                       supercell_builder.asym_unit_dict
                                                       )

        mol_batch2 = mol_batch.clone()
        mol_batch2.pos = opt_aunits[sample_ind]
        mol_batch2 = overwrite_symmetry_info(mol_batch2,
                                             mol_batch2.sg_ind,
                                             supercell_builder.symmetries_dict,
                                             randomize_sgs=False)
        supercell_batch, generated_cell_volumes = (
            supercell_builder.build_zp1_supercells(
                mol_batch=mol_batch2,
                cell_parameters=sample,
                supercell_size=5,
                graph_convolution_cutoff=6,
                align_to_standardized_orientation=False,
                skip_refeaturization=True,
                skip_molecule_posing=True,
            ))
        vdw_potential, vdw_loss = _score_crystal_batch(
            mol_batch2, supercell_batch, vdw_radii_tensor
        )
        rebuild_pot[sample_ind] = vdw_potential
        rebuild_aunit[sample_ind] = supercell_batch.pos[supercell_batch.aux_ind == 0].detach()
        # rebuild_loss[sample_ind] = scale_molwise_vdw_pot(vdw_potential, mol_batch.num_atoms)
        # if ((vdw_loss - opt_vdw_loss[sample_ind]).abs().log10() > 0.1).any():
        #     aa = 1
        #     bad_inds = torch.argwhere((vdw_potential - opt_vdw_pot[sample_ind]).abs() > 0.1).flatten()
        #
        # if ((rebuild_aunit[sample_ind] - opt_aunits[sample_ind]).abs() > 0.1).any():
        #     aa = 1

    if make_figs:
        fig = make_subplots(rows=1, cols=min(5, mol_batch.num_graphs))
        for ind in range(min(5, mol_batch.num_graphs)):
            fig.add_scatter(y=(rebuild_pot[:, ind] - opt_vdw_pot[:, ind].amin() + 0.1).flatten().log(), row=1,
                            col=ind + 1)
            fig.add_scatter(y=(opt_vdw_pot[:, ind] - opt_vdw_pot[:, ind].amin() + 0.1).flatten().log(), row=1,
                            col=ind + 1)
        fig.show()
        fig = go.Figure()
        fig.add_scatter(y=(((rebuild_pot.flatten() - opt_vdw_pot.flatten()).flatten()).abs() + 1).log10(), )
        fig.add_scatter(y=(rebuild_pot.flatten() - opt_vdw_pot.amin() + 1).log10(), )
        fig.add_scatter(y=(opt_vdw_pot.flatten() - opt_vdw_pot.amin() + 1).log10(), )
        fig.show()
        fig = go.Figure()
        fig.add_scatter(
            y=(((rebuild_pot.flatten() - opt_vdw_pot.flatten()) / opt_vdw_pot.flatten()).abs() + 1).log10(), )
        fig.show()

        fig = go.Figure()
        fig.add_scatter(y=(((rebuild_loss.flatten() - opt_vdw_loss.flatten())).abs()), )
        fig.add_scatter(y=(((rebuild_loss.flatten()))), )
        fig.add_scatter(y=(((opt_vdw_loss.flatten()))), )
        fig.show()

        deviations = torch.zeros(mol_batch.num_graphs)
        for ind in range(mol_batch.num_graphs):
            deviations[ind] = (
                    rebuild_aunit[0, mol_batch.batch == ind] - opt_aunits[0, mol_batch.batch == ind]).abs().std()
        fig = go.Figure(go.Scatter(y=deviations)).show()

    mae = (rebuild_loss.flatten() - opt_vdw_loss.flatten()).abs()
    print(mae)
    print(mae.sum())

    return rebuild_pot, rebuild_loss, rebuild_aunit


""" script to use the above
from mxtaltools.dataset_management.dataset_generation.generate_dataset_from_smiles import test_crystal_rebuild
from torch_geometric.loader.dataloader import Collater
collater = Collater(0,0)
mol_batch = mols_to_embed.clone().cpu()
opt_vdw_pot = mol_batch.vdw_pot[None, ...]
opt_aunits = mol_batch.pos[None, ...]
cell_params = torch.cat([
    mol_batch.cell_lengths, mol_batch.cell_angles, mol_batch.pose_params0
], dim=1)
opt_cell_params = cell_params[None,...]

r_pot, r_au = test_crystal_rebuild(
    mol_batch,
    opt_vdw_pot,
    opt_aunits,
    opt_cell_params,
    denorm=True
)
print((r_pot-opt_vdw_pot).abs())
"""


def process_smiles(smile: str,
                   allowed_atom_types,
                   max_num_atoms: int = 1000,
                   max_num_heavy_atoms: int = 100,
                   to_dict: bool = True,
                   max_radius: float = 15,
                   protonate: bool = True,
                   rotamers_per_sample: int = 1,
                   allow_simple_hydrogen_rotations: bool = False,
                   pare_to_size: Optional[int] = None):
    if rotamers_per_sample > 1:
        assert False, "Multiple rotamers not implemented"
    coords, atom_types, mask_rotate, mask_edges = generate_random_conformers_from_smiles(
        smile,
        protonate=protonate,
        max_rotamers_per_samples=rotamers_per_sample,
        allow_simple_hydrogen_rotations=allow_simple_hydrogen_rotations)

    if coords is False:
        return None, 'no coordinates'

    coords = coords[0]
    atom_types = atom_types[0]

    # use rotatable bonds as fragmentation sites to pare the molecule down to an acceptable size
    if pare_to_size is not None:
        while np.sum(atom_types > 1) > pare_to_size and len(mask_rotate) > 0:
            #num_heavy_atoms = np.sum(atom_types > 1)
            fragment_size = np.sum(mask_rotate[:, atom_types > 1], axis=1)  # how many heavy atoms in the fragment
            #min_atoms_to_remove = num_heavy_atoms - pare_to_size
            #resulting_num_atoms = num_heavy_atoms - fragment_size
            #excess_atoms = resulting_num_atoms - pare_to_size
            # select a fragment randomly, weighted towards smaller pieces
            fragment_to_pare = \
                np.random.choice(len(fragment_size), 1, p=np.exp(-fragment_size) / np.sum(np.exp(-fragment_size)))[0]
            atoms_to_pare = mask_rotate[fragment_to_pare, :]
            coords, atom_types = coords[~atoms_to_pare], atom_types[~atoms_to_pare]
            mask_rotate = np.delete(mask_rotate, fragment_to_pare, axis=0)
            mask_rotate = mask_rotate[:, ~atoms_to_pare]

    # molecule sizes filter
    if np.sum(atom_types > 1) > max_num_heavy_atoms:
        return None, "too many heavy atoms"
    elif len(atom_types) < 5:
        return None, "too few atoms"
    elif len(atom_types) > max_num_atoms:
        return None, "too many atoms"

    # atom types filter
    if not set(atom_types).issubset(allowed_atom_types):  #[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]):
        return None, "invalid atom types"

    sample = CrystalData(
        x=torch.tensor(atom_types, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        smiles=smile,
        identifier=smile,
        y=torch.zeros(1, dtype=torch.float32),
        require_crystal_features=False,
    )

    # molecule radius filter
    if sample.radius > max_radius:
        return None, "molecule too large"

    if to_dict:
        return sample.to_dict(), None
    else:
        return sample, None

#
# if __name__ == '__main__':
#     parent_directory = r'D:\crystal_datasets\zinc22'
#     chunks_dir = os.path.join(Path(parent_directory), 'chunks')
#
#     os.chdir(parent_directory)
#     dirs = os.listdir()
#
#     chunk_ind = - 1
#     min_chunk = 0
#     max_chunk = min(100000, len(dirs))
#
#     pool = mp.Pool(mp.cpu_count() - 1)
#
#     datapoint_counter = 0
#     dataset_length = 100000
#     t0 = time()
#     with tqdm(total=max_chunk) as pbar:
#         while chunk_ind < max_chunk - 1:
#             pbar.update(1)
#             chunk_ind += 1
#
#             if not (max_chunk > chunk_ind >= min_chunk):
#                 continue
#
#             if dirs[chunk_ind][0] == 'H':
#                 dirpath = Path(dirs[chunk_ind])
#                 for file_ind, file in enumerate(tqdm(os.listdir(dirpath))):
#                     chunkpath = os.path.join(chunks_dir, fr'chunk_{chunk_ind}_{file_ind}.pkl')
#                     if not os.path.exists(chunkpath):
#                         filepath = Path(file)
#                         combo_path = os.path.join(dirpath, filepath)
#
#                         if combo_path[-3:] == '.gz':
#                             with gzip.open(combo_path, 'r') as f:
#                                 lines = f.readlines()
#                         elif combo_path[-4:] == '.smi':
#                             with open(combo_path, 'r') as f:
#                                 lines = f.readlines()
#                         else:
#                             pass
#
#                         chunks = chunkify(lines, int(np.ceil(len(lines) / 1000)))
#                         del lines
#
#                         for chunk_ind2, chunk in enumerate(chunks):
#                             chunk_path = chunks_dir + f'/chunk_{chunk_ind}_{file_ind}_{chunk_ind2}.pkl'
#                             # process_smiles_list(chunk, chunk_path)
#                             if not os.path.exists(chunk_path):
#                                 pool.apply_async(process_smiles_list_to_file, args=(chunk, chunk_path, {}))
#                                 datapoint_counter += len(chunk)
#
#                             if datapoint_counter >= dataset_length:
#                                 print('Hit required number of samples')
#                                 break
#
#     pool.close()
#     pool.join()
#
#     print(time() - t0)
#
#     miner = DataManager(device='cpu',
#                         datasets_path=r"D:\crystal_datasets/",
#                         chunks_path=chunks_dir,
#                         dataset_type='molecule')
#     miner.process_new_dataset(new_dataset_name='temp_zinc_conf_dataset')
