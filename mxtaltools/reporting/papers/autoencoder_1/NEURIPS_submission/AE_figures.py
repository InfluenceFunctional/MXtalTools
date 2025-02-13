import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from PIL import Image
import os
from scipy.interpolate import interpn
from scipy.stats import linregress

stats_dict_paths = [
    r"C:\Users\mikem\crystals\CSP_runs\_experiments_dev_21-10-12-38-37\best_autoencoder_test_stats_dict.npy",
r"C:\Users\mikem\crystals\CSP_runs\_experiments_dev_21-10-12-38-37\best_autoencoder_test_stats_dict.npy",
r"C:\Users\mikem\crystals\CSP_runs\_experiments_dev_21-10-12-38-37\best_autoencoder_test_stats_dict.npy",
    ]
# best results from previous runs
# stats_dict_paths = [r'C:\Users\mikem\crystals\CSP_runs\models/_tests_qm9_test23_4_26-02-22-29-57.npy',
#                     r'C:\Users\mikem\crystals\CSP_runs\models/_tests_qm9_test23_7_27-02-14-34-41.npy',
#                     r'C:\Users\mikem\crystals\CSP_runs\models/_tests_qm9_test23_8_27-02-15-35-51.npy']

stats_dict_names = ["Without Hydrogen",
                    "With Hydrogen",
                    "Inferred Hydrogens"]

colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', len(stats_dict_paths), colortype='rgb')


def get_fraction(atomic_numbers, target: int):
    """get fraction of atomic numbers equal to target"""
    return np.sum(atomic_numbers == target) / len(atomic_numbers)



def compute_principal_axes_np(coords):
    """
    compute the principal axes for a given set of particle coordinates, ignoring particle mass
    use our overlap rules to ensure a fixed direction for all axes under almost all circumstances
    """
    points = coords - coords.mean(0)

    x, y, z = points.T
    Ixx = np.sum((y ** 2 + z ** 2))
    Iyy = np.sum((x ** 2 + z ** 2))
    Izz = np.sum((x ** 2 + y ** 2))
    Ixy = -np.sum(x * y)
    Iyz = -np.sum(y * z)
    Ixz = -np.sum(x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
    Ipm, Ip = np.linalg.eig(I)  # principal inertial tensor
    Ipm, Ip = np.real(Ipm), np.real(Ip)
    sort_inds = np.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds]  # want eigenvectors to be sorted row-wise (rather than column-wise)

    # cardinal direction is vector from CoM to the farthest atom
    dists = np.linalg.norm(points, axis=1)
    max_ind = np.argmax(dists)
    max_equivs = np.argwhere(np.round(dists, 8) == np.round(dists[max_ind], 8))[:,
                 0]  # if there are multiple equidistant atoms - pick the one with the lowest index
    max_ind = int(np.amin(max_equivs))
    direction = points[max_ind]
    direction = np.divide(direction, np.linalg.norm(direction))
    overlaps = Ip.dot(direction)  # check if the principal components point towards or away from the CoG
    signs = np.sign(overlaps)  # returns zero for zero overlap, but we want it to default to +1 in this case
    signs[signs == 0] = 1

    Ip = (Ip.T * signs).T  # if the vectors have negative overlap, flip the direction

    return Ip, Ipm, I


def get_point_density(xy, bins=500):
    """
    Scatter plot colored by 2d histogram
    """
    x, y = xy
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False)

    # To be sure to plot all data
    z = np.nan_to_num(z, posinf=0, neginf=0, nan=0)
    z -= z.min()
    z /= z.max()

    return z


def RMSD_fig():
    """
    Reconstruction Loss Figure

    RMSD violin for each model, with overlaid summary stats
    RMSD correlates bar plot
    """
    # load test stats
    stats_dicts = {}
    for ind in range(len(stats_dict_paths)):
        d = np.load(stats_dict_paths[ind], allow_pickle=True).item()
        stats_dicts[stats_dict_names[ind]] = d

    bandwidth = 0.005
    fig = make_subplots(rows=1, cols=2)
    for ind, run_name in enumerate(stats_dicts.keys()):
        stats_dict = stats_dicts[run_name]

        rmsd = stats_dict['RMSD_dist']
        node_rmsd = stats_dict['nodewise_dists_dist']
        matching_graph_fraction = stats_dict['matching_graph_fraction'].mean()
        matching_node_fraction = stats_dict['matching_node_fraction'].mean()

        'molwise'
        fig.add_annotation(x=0.5, y=ind + 0.3, showarrow=False,
                           text=f'RMSD: {rmsd.mean():.2f} <br> {(1-matching_graph_fraction) * 100:.0f}% Incomplete',
                           row=1, col=1)
        fig.add_trace(go.Violin(  # y=np.zeros_like(x),
            x=rmsd, side='positive', orientation='h',
            bandwidth=bandwidth, width=4, showlegend=True, opacity=1,  # .5,
            name=run_name,
            # scalegroup='',
            meanline_visible=True,
            line_color=colors[ind],
            points=False),
            row=1, col=1
        )

        'nodewise'
        fig.add_annotation(x=0.5, y=ind + 0.3, showarrow=False,
                           text=f'RMSD: {node_rmsd.mean():.2f} <br> {(1-matching_node_fraction) * 100:.0f}% Unmatched',
                           row=1, col=2)
        fig.add_trace(go.Violin(  # y=np.zeros_like(x),
            x=node_rmsd, side='positive', orientation='h',
            bandwidth=bandwidth, width=4, showlegend=False, opacity=1,  # .5,
            name=run_name,
            # scalegroup='',
            meanline_visible=True,
            line_color=colors[ind],
            points=False),
            row=1, col=2
        )

    fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(
        xaxis1={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'},
        xaxis2={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'}
    )  # , 'linecolor': 'white', 'linewidth': 5})
    fig.update_layout(
        yaxis1={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'},
        yaxis2={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'}
    )
    fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=20))
    fig.update_yaxes(title_font=dict(size=24), tickfont=dict(size=20))
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_layout(legend_traceorder='reversed')  # , yaxis_showgrid=True)

    fig.update_layout(xaxis1_range=[0, .75])
    fig.update_layout(xaxis1_title='Molwise RSMD (Angstrom)')
    fig.update_layout(xaxis1_tick0=0,
                      xaxis1_dtick=0.1)

    fig.update_layout(xaxis2_range=[0, .75])
    fig.update_layout(xaxis2_title='Atomwise RSMD (Angstrom)')
    fig.update_layout(xaxis2_tick0=0,
                      xaxis2_dtick=0.1)

    fig.update_layout(yaxis2_showticklabels=False)

    fig.update_layout(font=dict(size=24))
    fig.show(renderer='browser')

    return fig


def compute_rmsd_correlates(mol_keys, stats_dict, x):
    x[np.isinf(x)] = 1
    correlates_dict = {}
    for mol_key in mol_keys:
        try:
            mol_feat = np.concatenate(stats_dict[mol_key])
        except:
            mol_feat = stats_dict[mol_key]
        coeff = np.corrcoef(x, mol_feat, rowvar=False)[0, 1]
        if np.abs(coeff) > 0.1:
            correlates_dict[mol_key] = coeff
    # sort_inds = np.argsort(np.asarray([(correlates_dict[key]) for key in correlates_dict.keys()]))
    keys_list = list(correlates_dict.keys())
    keys_list2 = {}
    for key in keys_list:
        if 'molecule' in key:
            key2 = key.split('_')[1:]
            key2 = ' '.join(key2)
            keys_list2[key] = key2
        else:
            keys_list2[key] = key
    # sorted_correlates_dict = {keys_list[ind]: correlates_dict[keys_list[ind]] for ind in sort_inds}
    return correlates_dict, keys_list2


def add_mol_diagrams_to_rmsd_violin(fig, finite_x, ind, smiles):
    best_smiles = smiles[np.argmin(finite_x)]
    worst_smiles = smiles[np.argmax(finite_x)]
    # best_rmsd = finite_x.min()
    # worst_rmsd = finite_x.max()
    mol = Chem.MolFromSmiles(best_smiles)
    Draw.MolToImageFile(mol, 'conformer.png')
    fig.add_layout_image(dict(
        source=Image.open('conformer.png'),
        x=0, y=[0.2, 0.4, 0.6][ind],
        sizex=0.2, sizey=0.2, xanchor='left', yanchor='middle',
        opacity=.75, yref='y domain', xref='x domain')
    )
    mol = Chem.MolFromSmiles(worst_smiles)
    Draw.MolToImageFile(mol, 'conformer.png')
    fig.add_layout_image(dict(
        source=Image.open('conformer.png'),
        x=1, y=[0.2, 0.4, 0.6][ind],
        sizex=0.2, sizey=0.2, xanchor='right', yanchor='middle',
        opacity=.75, yref='y domain', xref='x domain')
    )


def mol_point_callout(fig2, xref, yref, ex, ey, embedding, smiles, imsize, row, col):
    tr = np.argmin(
        np.linalg.norm(embedding - [xref, yref] + np.random.randn(len(embedding), embedding.shape[1]) * 0.05, axis=1))
    mol = Chem.MolFromSmiles(smiles[tr])
    Draw.MolToImageFile(mol, 'conformer.png')
    fig2.add_layout_image(dict(
        source=Image.open('conformer.png'),
        x=xref + (0.5 - xref) * imsize, y=yref + (0.5 - yref) * imsize,
        sizex=imsize, sizey=imsize, xanchor='center', yanchor='middle',
        opacity=1, yref='y domain', xref='x domain'), row=row, col=col,
    )

    return dict(ax=xref, ay=yref,
                xref="x" + str(col), yref="y" + str(col),
                axref='x' + str(col), ayref='y' + str(col),
                showarrow=True, arrowhead=5,
                opacity=1, width=8,
                x=ex[tr], y=ey[tr],
                )


def UMAP_fig(max_entries=10000000):
    """
    combine train and test stats dicts
    generate and plot embeddings
    correlate against molecule features - maybe a mixed colour scheme?
    """
    # load test stats
    stats_dicts = {}
    for ind in range(len(stats_dict_paths)):
        d = np.load(stats_dict_paths[ind], allow_pickle=True).item()
        stats_dicts[stats_dict_names[ind]] = d
    run_name = "With Hydrogen"

    stats_dict = stats_dicts[run_name]
    del stats_dicts
    import umap

    from rdkit.Chem import rdMolDescriptors
    smiles = []
    for elem in stats_dict['samples']:
        smiles.extend(elem.smiles)

    from mxtaltools.common.geometry_utils import batch_molecule_principal_axes_torch
    Ipm = []
    for elem in stats_dict['samples']:
        _, ipm, _ = batch_molecule_principal_axes_torch([elem.pos[elem.batch == ind] for ind in range(elem.num_graphs)])
        Ipm.extend(ipm)

    Ipm = np.stack(Ipm)
    Ip1 = Ipm[:, 0]
    Ip2 = Ipm[:, 1]
    Ip3 = Ipm[:, 2]

    c_fraction = np.zeros(len(smiles))
    n_fraction = np.zeros_like(c_fraction)
    o_fraction = np.zeros_like(c_fraction)
    num_rings = np.zeros_like(c_fraction)
    num_rotatable = np.zeros_like(c_fraction)
    for ind in range(len(smiles)):
        mol = Chem.MolFromSmiles(smiles[ind])
        atoms = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        c_fraction[ind] = get_fraction(atoms, 6)
        n_fraction[ind] = get_fraction(atoms, 7)
        o_fraction[ind] = get_fraction(atoms, 8)
        num_rings[ind] = mol.GetRingInfo().NumRings()
        num_rotatable[ind] = rdMolDescriptors.CalcNumRotatableBonds(mol)

    composition_coloration = np.stack([c_fraction, n_fraction, o_fraction]).T[:max_entries]
    point_colors1 = normalize_colors(composition_coloration)
    legend_entries1 = ["Carbon Enriched", "Nitrogen Enriched", "Oxygen Enriched"]

    PR_triangle_points = np.asarray([[1, 1], [0, 1], [0.5, 0.5]])  # principal inertial ratio points on the triangle

    # conventional principal ratios - our convention agreeing with QMUGS
    principal_ratio1 = Ip2 / Ip3  # + 1e-3)
    principal_ratio1 = np.nan_to_num(principal_ratio1, posinf=0, neginf=0)[:max_entries]
    principal_ratio1 /= np.quantile(principal_ratio1, 0.9999)
    principal_ratio2 = Ip1/Ip3  # + 1e-3)
    principal_ratio2 = np.nan_to_num(principal_ratio2)[:max_entries]
    principal_ratio2 /= np.quantile(principal_ratio2, 0.9999)

    PR_stack = np.concatenate([principal_ratio2[:, None], principal_ratio1[:, None]], axis=1)
    sphere_like = -np.linalg.norm(PR_triangle_points[0] - PR_stack, axis=1)
    disc_like = -np.linalg.norm(PR_triangle_points[1] - PR_stack, axis=1)
    rod_like = -np.linalg.norm(PR_triangle_points[2] - PR_stack, axis=1)

    composition_coloration = np.stack([sphere_like, disc_like, rod_like]).T[:max_entries]
    point_colors2 = normalize_colors(composition_coloration)
    legend_entries2 = ["Sphere-like", "Rod-Like", "Disc-Like"]

    # composition_coloration = np.stack([num_rings,
    #                                    num_rotatable,
    #                                    np.concatenate(stats_dict['molecule_radius'])
    #                                    ]).T[:max_entries]
    #
    # point_colors3 = normalize_colors(composition_coloration)
    # legend_entries3 = ["# Rings", "# Rotatable Bonds", "Mol. Radius"]

    'triangle PR plot'
    # xy = np.vstack([principal_ratio1, principal_ratio2])
    # try:
    #     z = get_point_density(xy, bins=100)
    #     z[np.isnan(z)] = 0
    # except:
    #     z = np.ones_like(principal_ratio1)
    #
    # x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0.5, 1, 100))
    # good_inds = np.argwhere((y.flatten() > (1 - x.flatten())) * (y.flatten() > x.flatten())).flatten()
    # x=x.flatten()[good_inds]
    # y=y.flatten()[good_inds]
    # PR_stack = np.concatenate([x[:, None], y[:, None]], axis=1)
    # sphere_like = -np.linalg.norm(PR_triangle_points[0] - PR_stack, axis=1)
    # disc_like = -np.linalg.norm(PR_triangle_points[1] - PR_stack, axis=1)
    # rod_like = -np.linalg.norm(PR_triangle_points[2] - PR_stack, axis=1)
    #
    # test_colors = normalize_colors(composition_coloration)

    # fig = go.Figure() #make_subplots(rows=1, cols=2)
    # fig.add_scattergl(x=principal_ratio2, y=principal_ratio1, mode='markers', opacity=0.25, marker_color=point_colors2)
    # #fig.add_scattergl(x=x, y=y, mode='markers', opacity=0.25, marker_color=test_colors)
    # fig.update_layout(xaxis1_title='I1/I3', yaxis1_title='I2/I1', xaxis1_range=[0, 1.1], yaxis1_range=[.4, 1.1])
    # fig.write_image('triangle.png', width=400, height=400)

    reducer = umap.UMAP(n_components=2,
                        metric='cosine',
                        n_neighbors=10,
                        min_dist=0.2)
    scalar_encodings = stats_dict['scalar_embedding'][:max_entries]

    embedding = reducer.fit_transform((scalar_encodings - scalar_encodings.mean()) / scalar_encodings.std())

    'embeddings'
    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=.01, vertical_spacing=0.05,
                         #specs=[[{'rowspan': 3}, {'rowspan': 3}, {'rowspan': 3}, None],
                         #       [None, None, None, {}],
                         #       [None, None, None, None]],
                         subplot_titles=(
                             "(a) Composition", "(b) Geometry"))
    # fig2.add_layout_image(dict(
    #     source=Image.open('triangle.png'),
    #     x=0, y=[0.2, 0.4, 0.6][ind],
    #     sizex=0.2, sizey=0.2, xanchor='left', yanchor='middle',
    #     opacity=.75, yref='y domain', xref='x domain')
    # )
    legend_groups = ["(a)", "(b)", "(c)"]

    annotations_list = []
    for ind2, (point_colors, legend_entries) in enumerate(
            zip([point_colors1, point_colors2], [legend_entries1, legend_entries2])):

        embedding -= embedding.min(0)
        embedding /= embedding.max(0)
        fig2.add_trace(go.Scattergl(x=embedding[:, 0], y=embedding[:, 1],
                                    mode='markers',
                                    opacity=.1,
                                    showlegend=False,
                                    marker_color=point_colors
                                    ), row=1, col=ind2 + 1)
        fig2.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                           xaxis_showticklabels=False, yaxis_showticklabels=False,
                           plot_bgcolor='rgba(0,0,0,0)')
        fig2.update_yaxes(linecolor='black', mirror=True)
        fig2.update_xaxes(linecolor='black', mirror=True)

        # get corner indices
        ex, ey = embedding[:, 0], embedding[:, 1]

        for ix in np.linspace(0, 1, 6):
            for iy in np.linspace(0, 1, 6):
                if ix == 0 or ix == 1 or iy == 0 or iy == 1:
                    annotations_list.append(
                        mol_point_callout(fig2, ix, iy, ex, ey, embedding, smiles, 0.15, row=1, col=ind2 + 1))
        #
        # fig2.add_scattergl(x=np.zeros(1), y=np.zeros(1), marker_size=0.001, mode='markers',
        #                    marker_color=['rgb(255,0,0)'], name=legend_entries[0], showlegend=True,
        #                    legendgroup=legend_groups[ind2], legendgrouptitle_text=legend_groups[ind2],
        #                    row=1, col=ind2 + 1)
        # fig2.add_scattergl(x=np.zeros(1), y=np.zeros(1), marker_size=0.001, mode='markers',
        #                    marker_color=['rgb(0,255,0)'], name=legend_entries[1], showlegend=True,
        #                    legendgroup=legend_groups[ind2], legendgrouptitle_text=legend_groups[ind2],
        #                    row=1, col=ind2 + 1)
        # fig2.add_scattergl(x=np.zeros(1), y=np.zeros(1), marker_size=0.001, mode='markers',
        #                    marker_color=['rgb(0, 0,255)'], name=legend_entries[2], showlegend=True,
        #                    legendgroup=legend_groups[ind2], legendgrouptitle_text=legend_groups[ind2],
        #                    row=1, col=ind2 + 1)

    fig2.update_annotations(font_size=30)
    fig2.update_yaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig2.update_xaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)

    fig2.update_layout(annotations=annotations_list)

    fig2.update_layout(plot_bgcolor='rgb(255,255,255)')

    xlim = -0.2
    ylim = 1.2
    fig2.update_layout(xaxis1_range=[xlim, ylim], yaxis1_range=[xlim, ylim],
                       xaxis2_range=[xlim, ylim], yaxis2_range=[xlim, ylim],
                       #xaxis3_range=[xlim, ylim], yaxis3_range=[xlim, ylim],
                       #xaxis4_range=[-.2, 1.2], yaxis4_range=[0.3, 1.2],
                       #xaxis5_range=[-.2, 1.2], yaxis5_range=[.3, 1.2],
                       #xaxis6_range=[-.2, 1.2], yaxis6_range=[.3, 1.2],
                       )

    fig2.update_layout(
        xaxis_title='Component 1',
        yaxis1_title='Component 2',
        #yaxis4_title='Ip1/Ip3',
        #xaxis4_title='Ip2/Ip3'
    )
    fig2.update_layout(legend={'itemsizing': 'constant'})  #, 'orientation': 'h'})

    fig2.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig2.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig2.update_layout(font=dict(size=30))

    fig2.show(renderer='browser')

    fig = go.Figure()
    fig.add_trace(go.Scatterternary({
        'mode': 'markers',
        'a': c_fraction + np.random.randn(len(c_fraction)) * 0.05,
        'b': o_fraction + np.random.randn(len(c_fraction)) * 0.05,
        'c': n_fraction + np.random.randn(len(c_fraction)) * 0.05,
        'marker': {'color': point_colors1, 'size': 40, 'opacity': 0.1}
    }))
    fig.update_layout(font_size=1)

    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                      xaxis_showticklabels=False, yaxis_showticklabels=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show(renderer='browser')

    fig = go.Figure()
    fig.add_scattergl(x=principal_ratio2, y=-principal_ratio1, mode='markers',
                      opacity=1, marker_color=point_colors2, showlegend=False)

    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False,
                      xaxis_showticklabels=False, yaxis_showticklabels=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.show(renderer='browser')

    return fig2


def normalize_colors(composition_coloration):
    composition_coloration -= composition_coloration.mean(0)[None, :]  # helpful distortion
    composition_coloration /= composition_coloration.std(0)[None, :]
    composition_coloration = np.clip(composition_coloration, a_min=np.quantile(composition_coloration, 0.01),
                                     a_max=np.quantile(composition_coloration, 0.99))
    composition_coloration -= composition_coloration.min(0)
    composition_coloration /= composition_coloration.max(0)
    composition_coloration *= 200
    point_colors = [f'rgb({int(color[0])},{int(color[1])},{int(color[2])})' for color in composition_coloration]
    return point_colors


def embedding_regression_figure():
    os.chdir(r'C:\Users\mikem\crystals\CSP_runs\er7_results')
    elem = os.listdir()
    ers = [thing for thing in elem if '_embedding_regression_' in thing]
    target_names = ["rotational_constant_a",  # 0
                    "rotational_constant_b",  # 1
                    "rotational_constant_c",  # 2
                    "dipole_moment",  # 3
                    "isotropic_polarizability",  # 4
                    "HOMO_energy",  # 5
                    "LUMO_energy",  # 6
                    "gap_energy",  # 7
                    "el_spatial_extent",  # 8
                    "zpv_energy",  # 9
                    "internal_energy_0",  # 10
                    "internal_energy_STP",  # 11
                    "enthalpy_STP",  # 12
                    "free_energy_STP",  # 13
                    "heat_capacity_STP",  # 14
                    ]
    pretty_target_names = [r"$\large{(a)\ Rotational\ Constant\ A\ /GHz}$",
                           r"$\large{(b)\ Rotational\ Constant\ B\ /GHz}$",
                           r"$\large{(c)\ Rotational\ Constant\ C\ /GHz}$",
                           r"$\large{(d)\ Dipole\ Moment\ /Deb}$",
                           r"$\large{(e)\ Iso.\ Polarizability\ /Bohr^3}$",
                           r"$\large{(f)\ HOMO\ Energy\ /Hartree}$",
                           r"$\large{(g)\ LUMO\ Energy\ /Hartree}$",
                           r"$\large{(h)\ Gap\ Energy\ /Hartree}$",
                           r"$\large{(i)\ Electronic\ Spatial\ Extent\ /Bohr^2}$",
                           r"$\large{(j)\ ZPV\ Energy\ /Hartree}$",
                           r"$\large{(k)\ Internal\ Energy\ (T=0)\ /Hartree}$",
                           r"$\large{(l)\ Internal\ Energy\ 298K\ /Hartree}$",
                           r"$\large{(m)\ Enthalpy\ 298K\ /Hartree}$",
                           r"$\large{(n)\ Free\ Energy\ 298K\ /Hartree}$",
                           r"$\large{(o)\ Heat\ Capacity\ 298K\ /cal\ mol^{-1}\ K^{-1}}$"]

    MAE_dict = {}
    NMAE_dict = {}
    R_dict = {}
    for ind, (path, target_name) in enumerate(zip(ers, target_names)):
        dpath = path + '/best_embedding_regressor_test_stats_dict.npy'
        stats_dict = np.load(dpath, allow_pickle=True).item()
        target = stats_dict['regressor_target']
        prediction = stats_dict['regressor_prediction']

        MAE = np.abs(target - prediction).mean()
        NMAE = (np.abs((target - prediction) / np.abs(target))).mean()

        linreg_result = linregress(target, prediction)
        R_value = linreg_result.rvalue
        slope = linreg_result.slope
        MAE_dict[target_name] = MAE
        NMAE_dict[target_name] = NMAE
        R_dict[target_name] = R_value

    fig3 = make_subplots(cols=5, rows=3, subplot_titles=pretty_target_names, horizontal_spacing=0.06,
                         vertical_spacing=0.1)

    annotations = []
    for ind, (path, target_name) in enumerate(zip(ers, target_names)):
        dpath = path + '/best_embedding_regressor_test_stats_dict.npy'
        stats_dict = np.load(dpath, allow_pickle=True).item()
        target = stats_dict['regressor_target']
        prediction = stats_dict['regressor_prediction']

        xline = np.linspace(max(min(target), min(prediction)),
                            min(max(target), max(prediction)), 2)

        xy = np.vstack([target, prediction])
        try:
            z = get_point_density(xy, bins=1000)
        except:
            z = np.ones_like(target)

        row = ind // 5 + 1
        col = ind % 5 + 1
        opacity = 0.05  # np.exp(-num_points / 10000)
        fig3.add_trace(go.Scattergl(x=target, y=prediction, mode='markers', marker=dict(color=z), opacity=opacity,
                                    showlegend=False),
                       row=row, col=col)
        fig3.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                       row=row, col=col)

        minval = max([np.amin(target), np.amin(prediction)])
        maxval = min([np.amax(target), np.amax(prediction)])
        fig3.update_layout({f'xaxis{ind + 1}_range': [minval, maxval]})

        annotations.append(dict(xref="x" + str(ind + 1), yref="y" + str(ind + 1),
                                x=minval + np.ptp(prediction) * 0.25,
                                y=maxval - np.ptp(prediction) * 0.2,
                                showarrow=False,
                                text=f"R={R_dict[target_name]:.2f} <br> MAE={MAE_dict[target_name]:.3g}"
                                ))

    fig3['layout']['annotations'] += tuple(annotations)
    fig3.update_annotations(font=dict(size=18))
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig3.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig3.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig3.update_yaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig3.update_xaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig3.update_layout(yaxis6_title='Predicted Value')
    fig3.update_layout(xaxis13_title='Target Value')
    fig3.update_layout(font=dict(size=20))
    fig3.update_annotations(font_size=20)
    fig3.update_annotations(yshift=10)
    fig3.update_xaxes(tickangle=0)
    fig3.show(renderer='browser')

    return fig3


def regression_training_curve():
    import numpy as np
    dataset_sizes = np.linspace(1000, 130000, num=20)
    dataset_sizes[-1] = 133000

    x = dataset_sizes
    # y = np.asarray([
    #     .2328,  # 26
    #     .1815,  # 27
    #     .1575,  # 28
    #     .1419,  # 29
    #     .1211,  # 30
    #     .1181,  # 31
    #     .09724,  # 32
    #     .09459,  # 33
    #     .09,  # 34
    #     np.nan,  # 35
    #     .07692,  # 36
    #     0.06927,  # 37
    #     0.06702,  # 38
    #     0.06181,  # 39
    #     0.05746,  # 40
    #     0.05795,  # 41
    #     0.05174,  # 42
    #     0.04831,  # 43 \
    #     0.04734,  # 44
    #     0.04505  # 45
    # ])
    y = np.asarray([
        0.3244,  # 26
        0.2218,  # 27
        0.1831,  # 28
        0.1504,  # 29
        0.139,  # 30
        0.1159,  # 31
        0.09535,  # 32
        0.09786,  # 33
        0.08978,  # 34
        0.08531,  # 35
        0.0769,  # 36
        0.07244,  # 37
        0.07253,  # 38
        0.06615,  # 39
        0.06154,  # 40
        0.06448,  # 41
        0.0610,  # 42
        0.05527,  # 43 \
        0.05574,  # 44
        0.05364  # 45
    ])
    fig4 = go.Figure()
    fig4.add_scatter(x=x, y=y, mode='markers', marker_size=10)

    fig4.update_layout(xaxis_title='Training Set Size', yaxis_title='Best Test Loss')
    fig4.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)')
    fig4.update_layout(
        xaxis={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})  # , 'linecolor': 'white', 'linewidth': 5})
    fig4.update_layout(yaxis={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})

    fig4.update_layout(font=dict(size=28))
    fig4.update_layout(yaxis_range=[0, 0.24])
    fig4.show(renderer='browser')

    return fig4


#
fig = RMSD_fig()
fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\RMSD.png', width=1920, height=840)

fig2 = UMAP_fig(max_entries=1000000)
#fig2.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\latent_space.png', width=1920, height=840)

fig3 = embedding_regression_figure()
fig3.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\QM9_properties.png', width=1920, height=840)

#fig4 = regression_training_curve()
#fig4.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\gap_traning_curve.png', width=1200, height=800)
aa = 1
