import multiprocessing as mp
import os
from math import ceil
from pathlib import Path
from random import shuffle

import numpy as np
import torch
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from torch_geometric.loader.dataloader import Collater
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import batch_molecule_vdW_volume
from mxtaltools.analysis.crystal_rdf import earth_movers_distance_torch
from mxtaltools.constants.atom_properties import VDW_RADII
from mxtaltools.dataset_utils.synthesis.otf_conf_gen import async_generate_random_conformer_dataset
from mxtaltools.models.functions.asymmetric_radius_graph import radius
from mxtaltools.models.functions.crystal_rdf import get_elementwise_dists, batch_histogram_1d

HDonorSmarts = Chem.MolFromSmarts(
    '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
HAcceptorSmarts = Chem.MolFromSmarts(
    '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' + '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' + '$([nH0,o,s;+0])]')


def get_atom_types_distribution(mol_batch):
    types, counts = torch.unique(mol_batch.x, return_counts=True)
    return types.cpu().detach().numpy(), (counts / counts.sum()).cpu().detach().numpy()


def get_mol_prop_distribution(mol_batch, prop: str, min: float = 0, max: float = 15, nbins: int = 100):
    bins = torch.linspace(min, max, nbins)
    hist, bins = torch.histogram(mol_batch.__dict__['_store'][prop].float(), bins=bins, density=True)
    return hist, bins


def get_heavy_atom_distribution(mol_batch, min: float = 0, max: float = 15, nbins: int = 100):
    bins = torch.linspace(min, max, nbins)
    atom_is_heavy = mol_batch.x.flatten() > 1
    num_heavy_atoms = scatter(atom_is_heavy.float(), mol_batch.batch, reduce='sum', dim_size=mol_batch.num_graphs)
    hist, bins = torch.histogram(num_heavy_atoms.float(), bins=bins, density=True)
    return hist, bins


def get_edge_distribution(dataset,
                          max_edge_length: float,
                          relevant_elements: list = [1, 6, 7, 8, 9],
                          device: str = 'cpu'):
    collater = Collater(0, 0)
    with torch.no_grad():
        batch_size = 100
        num_chunks = dataset.num_graphs // 100
        if dataset.num_graphs % 100 != 0:
            num_chunks += 1

        for ind in range(num_chunks):
            mol_batch = collater(dataset[ind * batch_size: (ind + 1) * batch_size]).to(device)
            edges = radius(
                x=mol_batch.pos,
                y=mol_batch.pos,
                r=max_edge_length,
                max_num_neighbors=100,
                batch_x=mol_batch.batch,
                batch_y=mol_batch.batch,
            )
            edge_batch = mol_batch.batch[edges[0]]
            dists = (mol_batch.pos[edges[0]] - mol_batch.pos[edges[1]]).norm(dim=1)
            dists_per_hist, sorted_dists, rdfs_dict = get_elementwise_dists(
                mol_batch.x.flatten(),
                edges,
                dists,
                device=device,
                num_graphs=mol_batch.num_graphs,
                edge_in_crystal_number=edge_batch,
                atomic_numbers_override=torch.LongTensor(relevant_elements).to(device),
            )
            num_pairs = len(rdfs_dict.keys())
            batch = torch.arange(len(dists_per_hist), device=device).repeat_interleave(dists_per_hist, dim=0)
            hist, bin_edges = batch_histogram_1d(sorted_dists, batch, mol_batch.num_graphs * num_pairs,
                                                 rrange=[0, max_edge_length],
                                                 nbins=100)
            if ind == 0:
                hist_record = torch.zeros((num_pairs, hist.shape[1]), dtype=torch.float32, device='cpu')

            dist_batch = torch.arange(num_pairs, device=device).repeat(mol_batch.num_graphs)
            agg_hist = scatter(hist, dist_batch, dim=0, dim_size=num_pairs)
            hist_record += (agg_hist / mol_batch.num_graphs).cpu()

    return hist_record, bin_edges.cpu(), rdfs_dict


def rdkit_molecule_analysis(smiles):
    molecule_dict = {}
    rd_mol = Chem.MolFromSmiles(smiles)

    h_donors = list(sum(rd_mol.GetSubstructMatches(HDonorSmarts, uniquify=1), ()))  # convert tuple to list
    h_acceptors = list(sum(rd_mol.GetSubstructMatches(HAcceptorSmarts, uniquify=1), ()))

    molecule_dict['mass'] = Descriptors.MolWt(rd_mol)  # includes implicit protons
    molecule_dict['num_rings'] = rd_mol.GetRingInfo().NumRings()
    molecule_dict['num_donors'] = len(h_donors)
    molecule_dict['num_acceptors'] = len(h_acceptors)
    molecule_dict['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(rd_mol)

    for frag in Fragments.__dict__.keys():  # for all the class methods
        if frag[0:3] == 'fr_':  # if it's a functional group analysis methodad
            molecule_dict[f'{frag[3:]}_fragment_count'] = Fragments.__dict__[frag](rd_mol, countUnique=False)

    return molecule_dict


def rdkit_dataset_analysis(dataset, bins_set=None):
    for ind, smiles in enumerate(dataset.smiles):
        mol_dict = rdkit_molecule_analysis(smiles)
        if ind == 0:
            results_dict = {key: [] for key in mol_dict.keys()}

        for key in mol_dict.keys():
            results_dict[key].append(mol_dict[key])

    hist_dict = {}
    for key in results_dict.keys():
        vals = results_dict[key]
        uniques = np.unique(vals)
        if bins_set is not None:
            old_bins = bins_set[key + '_bins']
            bins = max(1, len(old_bins) - 1)
            range = [old_bins[0], old_bins[-1]]
        else:
            if len(uniques) <= 10:
                bins = uniques
            else:
                bins = 100

            data_range = np.ptp(vals)
            range = [np.amin(vals) - 0.2 * data_range, np.amax(vals) + 0.2 * data_range]

        hist, bins = np.histogram(vals, bins=bins, range=range, density=True)
        hist_dict[key] = hist
        hist_dict[key + '_bins'] = bins

    return hist_dict


def property_comparison_fig(dicts_to_plot, keys_to_plot, dataset_names, num_cols, titles, log=False):
    colors = ['rgb(250, 50, 50)', 'rgb(50,50,250)']

    num_feats = len(keys_to_plot)
    num_rows = ceil(num_feats / num_cols)
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles)
    for ind1, feat_dict in enumerate(dicts_to_plot):
        for ind2, key in enumerate(keys_to_plot):
            if len(feat_dict[key + '_bins']) < 20:
                fig.add_bar(x=feat_dict[key + '_bins'],
                            y=feat_dict[key],
                            name=dataset_names[ind1],
                            legendgroup=dataset_names[ind1],
                            showlegend=True if ind2 == 0 else False,
                            marker_color=colors[ind1],
                            row=ind2 // num_cols + 1,
                            col=ind2 % num_cols + 1
                            )
            else:
                fig.add_scatter(x=feat_dict[key + '_bins'],
                                y=feat_dict[key],
                                name=dataset_names[ind1],
                                legendgroup=dataset_names[ind1],
                                showlegend=True if ind2 == 0 else False,
                                marker_color=colors[ind1],
                                row=ind2 // num_cols + 1,
                                col=ind2 % num_cols + 1
                                )
    if log:
        fig.update_yaxes(type='log')
    return fig


def analyze_mol_dataset(dataset, device, bins_set=None):
    mol_feats_dict = rdkit_dataset_analysis(dataset, bins_set)

    mol_feats_dict['atom_types'], mol_feats_dict['atom_types_bins'] = (
        get_atom_types_distribution(dataset))
    mol_feats_dict['radii'], mol_feats_dict['radii_bins'] = (
        get_mol_prop_distribution(dataset,
                                  'radius', 0, 10,
                                  100))
    mol_feats_dict['volume'], mol_feats_dict['volume_bins'] = (
        get_mol_prop_distribution(dataset,
                                  'mol_volume',
                                  0, 300, 100))
    mol_feats_dict['num_atoms'], mol_feats_dict['num_atoms_bins'] = (
        get_mol_prop_distribution(dataset,
                                  'num_atoms',
                                  0, 40,
                                  40))
    mol_feats_dict['num_heavy_atoms'], mol_feats_dict['num_heavy_atoms_bins'] = (
        get_heavy_atom_distribution(dataset,

                                    0, 15,
                                    15))
    rdfs, bin_edges, rdfs_dict = get_edge_distribution(dataset, max_edge_length=6, device=device)
    for ind in range(len(rdfs)):
        rdf_name = rdfs_dict[ind]
        mol_feats_dict[f'{rdf_name}_rdf'] = rdfs[ind][1:] / rdfs[ind][1:].sum()
        mol_feats_dict[f'{rdf_name}_rdf_bins'] = bin_edges[1:]

    return mol_feats_dict


def generate_otf_dataset(generate_samples: int,
                         dataset_length: int,
                         num_processes: int):
    num_processes = num_processes
    if generate_samples > 0:
        pool = mp.Pool(num_processes)
        async_generate_random_conformer_dataset(dataset_length=generate_samples,
                                                smiles_source=r'D:\crystal_datasets\zinc22',
                                                workdir=r'D:\crystal_datasets\otf_chunks',
                                                allowed_atom_types=[1, 6, 7, 8, 9], num_processes=num_processes,
                                                pool=pool, max_num_atoms=32, max_num_heavy_atoms=9, pare_to_size=9,
                                                max_radius=15, synchronize=True)
    chunks = os.listdir(r'D:\crystal_datasets\otf_chunks')
    chunks = [elem for elem in chunks if 'chunk' in elem]
    os.chdir(r'D:\crystal_datasets\otf_chunks')
    otf_dataset = []
    for chunk in chunks:
        otf_dataset.extend(torch.load(chunk))

    shuffle(otf_dataset)
    otf_dataset = otf_dataset[:dataset_length]
    otf_dataset = collater(otf_dataset)
    vdw_radii_tensor = torch.tensor(list(VDW_RADII.values()))
    otf_dataset.mol_volume = batch_molecule_vdW_volume(otf_dataset.x.flatten(),
                                                       otf_dataset.pos,
                                                       otf_dataset.batch,
                                                       otf_dataset.num_graphs,
                                                       vdw_radii_tensor)
    return otf_dataset


def distribution_comparison(qm9_properties_dict, otf_properties_dict, feat_keys):
    dists_dict = {}
    for key in feat_keys:
        try:
            v1 = torch.tensor(qm9_properties_dict[key])
            v2 = torch.tensor(otf_properties_dict[key])
            bins = otf_properties_dict[key + '_bins']
            dists_dict[key] = earth_movers_distance_torch(v1 / v1.sum(), v2 / v2.sum()) / len(bins)
        except:
            pass

    fig = make_subplots(rows=2, cols=1)
    keys = list(dists_dict.keys())
    vals = list(dists_dict.values())
    fig.add_bar(x=keys[:len(keys) // 2],
                y=vals[:len(keys) // 2],
                marker_color='blue',
                showlegend=False,
                row=1, col=1)
    fig.add_bar(x=keys[len(keys) // 2:],
                y=vals[len(keys) // 2:],
                marker_color='blue',
                showlegend=False,
                row=2, col=1)
    fig.update_layout(title="Distribution Distances")

    return dists_dict, fig


"""

M
A
I
N

"""

if __name__ == '__main__':
    """config"""
    collater = Collater(None, None)
    dataset_path = Path(r'D:\crystal_datasets\qm9_dataset.pt')
    qm9_properties_path = Path('D:\crystal_datasets\otf_chunks\qm9_properties_dict.pt')
    dataset_size = 130000
    generate_samples = 10000
    force_qm9_reanalyzis = False

    """get evaluation dataset"""
    if not os.path.exists(qm9_properties_path) or force_qm9_reanalyzis:
        qm9_dataset = collater(torch.load(dataset_path)[:dataset_size])
        qm9_properties_dict = analyze_mol_dataset(qm9_dataset, 'cpu')
        del qm9_dataset
        torch.save(qm9_properties_dict, 'D:\crystal_datasets\otf_chunks\qm9_properties_dict.pt')
    else:
        qm9_properties_dict = torch.load(qm9_properties_path)

    """generate otf dataset"""
    otf_dataset = generate_otf_dataset(generate_samples=generate_samples,
                                       dataset_length=dataset_size,
                                       num_processes=7)
    qm9_bins_dict = {key: bins for (key, bins) in qm9_properties_dict.items() if '_bin' in key}
    otf_properties_dict = analyze_mol_dataset(otf_dataset, 'cpu', bins_set=qm9_bins_dict)
    del otf_dataset

    """visualize property distributions"""
    dataset_names = ['QM9', 'OTF']
    feat_keys = [key for key in qm9_properties_dict.keys() if 'bins' not in key]
    rdf_keys = [key for key in feat_keys if 'rdf' in key]
    frag_keys = [key for key in feat_keys if 'fragment_count' in key]
    remainder_keys = [key for key in feat_keys if key not in rdf_keys + frag_keys]
    clipped_frag_keys = [key.split('_fragment_count')[0] for key in frag_keys]

    property_comparison_fig([qm9_properties_dict, otf_properties_dict], remainder_keys, dataset_names, num_cols=5, titles=remainder_keys,
                            log=False).show()
    property_comparison_fig([qm9_properties_dict, otf_properties_dict], rdf_keys, dataset_names, num_cols=5, titles=rdf_keys, log=False).show()
    property_comparison_fig([qm9_properties_dict, otf_properties_dict], frag_keys, dataset_names, num_cols=12, titles=clipped_frag_keys,
                            log=True).show()

    """check overlaps"""
    dists_dict, dists_fig = distribution_comparison(qm9_properties_dict, otf_properties_dict, feat_keys)
    dists_fig.show()

    aa = 1
