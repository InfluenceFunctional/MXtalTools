import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from PIL import Image
from scipy.interpolate import interpn
from scipy.stats import linregress
from model_paths import ae_results_paths, er_results_paths, targets, proxy_results_paths
from mxtaltools.common.geometry_utils import scatter_compute_Ip

stats_dict_names = ["With Hydrogen",
                    "Without Hydrogen"]

colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', max(2, len(ae_results_paths)), colortype='rgb')


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

    fontsize = 28
    # load test stats
    stats_dicts = {}
    for ind in range(2):
        d = np.load(ae_results_paths[ind], allow_pickle=True).item()
        stats_dicts[stats_dict_names[ind]] = d['test_stats']

    bandwidth = 0.005
    fig = make_subplots(rows=2, cols=2, subplot_titles=['(a) Whole Molecule', '(b) Atomwise'],
                        horizontal_spacing=0.15, vertical_spacing=0.15)
    for ind, run_name in enumerate([f'Without Hydrogen',
                                    f'With Hydrogen']):
        row = ind + 1
        stats_dict = stats_dicts[run_name]
        run_name = ' <br> '.join(run_name.split()) + '<br>'
        x = np.concatenate(stats_dict['RMSD_dist'])
        matched = np.concatenate(stats_dict['matched_graphs'])
        unmatched = np.mean(np.invert(matched))
        finite_x = x[matched]

        fig.add_annotation(x=0.6, y=0.5, showarrow=False,
                           text=f'Mean Distance: {finite_x.mean():.2f} <br> Unmatched Frac.: {unmatched * 100:.1f}%',
                           font_size=20,
                           row=row, col=1)
        fig.add_trace(go.Violin(  # y=np.zeros_like(x),
            x=finite_x, side='positive', orientation='h',
            bandwidth=bandwidth, width=4, showlegend=False, opacity=1,  # .5,
            name=f'{run_name}',
            # scalegroup='',
            meanline_visible=True,
            line_color=colors[ind],
            points=False),
            row=row, col=1
        )

        x = stats_dict['nodewise_dists_dist']
        matched = stats_dict['matched_nodes']
        unmatched = np.mean(np.invert(matched))
        finite_x = x[matched]

        fig.add_annotation(x=0.6, y=0.5, showarrow=False,
                           text=f'Mean Distance: {finite_x.mean():.2f} <br> Unmatched Frac.: {unmatched * 100:.2f}%',
                           font_size=20,
                           row=row, col=2)
        fig.add_trace(go.Violin(  # y=np.zeros_like(x),
            x=finite_x, side='positive', orientation='h',
            bandwidth=bandwidth, width=4, showlegend=False, opacity=1,  # .5,
            name=run_name,
            # scalegroup='',
            meanline_visible=True,
            line_color=colors[ind],
            points=False),
            row=row, col=2
        )

    fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)')

    fig.update_layout(font=dict(size=fontsize))
    fig.update_xaxes(title_font=dict(size=fontsize), tickfont=dict(size=fontsize))
    fig.update_yaxes(title_font=dict(size=fontsize), tickfont=dict(size=fontsize))
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_layout(legend_traceorder='reversed')  # , yaxis_showgrid=True)
    fig.update_annotations(font_size=int(fontsize * 1.2))

    fig.update_layout(
        xaxis1={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})  # , 'linecolor': 'white', 'linewidth': 5})
    fig.update_layout(xaxis1_range=[0, .75])
    fig.update_layout(xaxis1_title='Mean Distance (Angstrom)')
    fig.update_layout(xaxis1_tick0=0, xaxis1_dtick=0.1)
    fig.update_layout(yaxis1={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})

    fig.update_layout(
        xaxis2={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})  # , 'linecolor': 'white', 'linewidth': 5})
    fig.update_layout(xaxis2_range=[0, .75])
    fig.update_layout(xaxis2_title='Mean Distance (Angstrom)')
    fig.update_layout(xaxis2_tick0=0, xaxis2_dtick=0.1)
    fig.update_layout(yaxis2={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})

    fig.update_layout(
        xaxis3={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})  # , 'linecolor': 'white', 'linewidth': 5})
    fig.update_layout(xaxis3_range=[0, .75])
    fig.update_layout(xaxis3_title='Mean Distance (Angstrom)')
    fig.update_layout(xaxis3_tick0=0, xaxis3_dtick=0.1)
    fig.update_layout(yaxis3={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})

    fig.update_layout(
        xaxis4={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})  # , 'linecolor': 'white', 'linewidth': 5})
    fig.update_layout(xaxis4_range=[0, .75])
    fig.update_layout(xaxis4_title='Mean Distance (Angstrom)')
    fig.update_layout(xaxis4_tick0=0, xaxis4_dtick=0.1)
    fig.update_layout(yaxis4={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})

    fig.show(renderer='browser')
    fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\RMSD.png', width=1920, height=840)

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
    Draw.MolToFile(mol, 'conformer.png')
    fig.add_layout_image(dict(
        source=Image.open('conformer.png'),
        x=0, y=[0.2, 0.4, 0.6][ind],
        sizex=0.2, sizey=0.2, xanchor='left', yanchor='middle',
        opacity=.75, yref='y domain', xref='x domain')
    )
    mol = Chem.MolFromSmiles(worst_smiles)
    Draw.MolToFile(mol, 'conformer.png')
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
    Draw.MolToFile(mol, 'conformer.png')
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
    stats_dict = np.load(ae_results_paths[-1], allow_pickle=True).item()
    import umap
    # combine train and test stats
    if stats_dict['train_stats'] is not None:
        for key in stats_dict['test_stats'].keys():
            try:
                combo = np.concatenate([stats_dict['train_stats'][key], stats_dict['test_stats'][key]])

            except:
                try:
                    combo = np.concatenate([np.concatenate(stats_dict['train_stats'][key]),
                                            np.concatenate(stats_dict['test_stats'][key])])
                except:
                    try:
                        combo = stats_dict['train_stats'][key] + stats_dict['test_stats'][key]
                    except:
                        print(key)
                        print('noo!')
            stats_dict[key] = combo
    else:
        stats_dict = stats_dict['test_stats']

    #del stats_dict['train_stats'], stats_dict['test_stats']

    from torch_geometric.loader.dataloader import Collater
    collater = Collater(0, 0)
    stats_dict['sample'] = collater(stats_dict['sample'])
    _, Ipm, _ = scatter_compute_Ip(stats_dict['sample'].pos, stats_dict['sample'].batch)

    Ip1 = Ipm[:, 0]
    Ip2 = Ipm[:, 1]
    Ip3 = Ipm[:, 2]

    from rdkit.Chem import rdMolDescriptors
    stats_dict['molecule_smiles'] = stats_dict['molecule_smiles'].flatten()
    c_fraction = np.zeros(len(stats_dict['molecule_smiles']))
    n_fraction = np.zeros_like(c_fraction)
    o_fraction = np.zeros_like(c_fraction)
    num_rings = np.zeros_like(c_fraction)
    num_rotatable = np.zeros_like(c_fraction)
    for ind in range(len(stats_dict['molecule_smiles'])):
        mol = Chem.MolFromSmiles(stats_dict['molecule_smiles'][ind])
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

    principal_ratio1 = Ip2 / Ip3  # + 1e-3)
    principal_ratio1 = np.nan_to_num(principal_ratio1, posinf=0, neginf=0)[:max_entries]
    principal_ratio1 /= np.quantile(principal_ratio1, 0.9999)
    principal_ratio2 = Ip1 / Ip3  # + 1e-3)
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
                        n_neighbors=25,
                        min_dist=0.1)
    scalar_encodings = stats_dict['scalar_embedding'][:max_entries]

    embedding = reducer.fit_transform((scalar_encodings - scalar_encodings.mean()) / scalar_encodings.std())

    'embeddings'
    fontsize = 40
    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=.01, vertical_spacing=0.05,
                         # specs=[[{'rowspan': 3}, {'rowspan': 3}, {'rowspan': 3}, None],
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
        fig2.add_trace(go.Scattergl(x=embedding[:, 0],
                                    y=embedding[:, 1],
                                    mode='markers',
                                    opacity=.15,
                                    marker_size=8,
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
        try:
            smiles = np.concatenate(stats_dict['molecule_smiles'])
        except:
            smiles = stats_dict['molecule_smiles']

        for ix in np.linspace(0, 1, 5):
            for iy in np.linspace(0, 1, 5):
                if ix == 0 or ix == 1 or iy == 0 or iy == 1:
                    annotations_list.append(
                        mol_point_callout(fig2, ix, iy, ex, ey, embedding, smiles, 0.2, row=1, col=ind2 + 1))
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

    fig2.update_annotations(font_size=fontsize)
    fig2.update_yaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig2.update_xaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)

    fig2.update_layout(annotations=annotations_list)

    fig2.update_layout(plot_bgcolor='rgb(255,255,255)')

    xlim = -0.4
    ylim = 1.4
    fig2.update_layout(xaxis1_range=[xlim, ylim], yaxis1_range=[xlim, ylim],
                       xaxis2_range=[xlim, ylim], yaxis2_range=[xlim, ylim],
                       # xaxis3_range=[xlim, ylim], yaxis3_range=[xlim, ylim],
                       # xaxis4_range=[-.2, 1.2], yaxis4_range=[0.3, 1.2],
                       # xaxis5_range=[-.2, 1.2], yaxis5_range=[.3, 1.2],
                       # xaxis6_range=[-.2, 1.2], yaxis6_range=[.3, 1.2],
                       )

    # fig2.update_layout(
    # xaxis_title='Component 1',
    # yaxis1_title='Component 2',
    # yaxis4_title='Ip1/Ip3',
    # xaxis4_title='Ip2/Ip3'
    # )
    fig2.update_layout(legend={'itemsizing': 'constant'})  # , 'orientation': 'h'})

    fig2.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig2.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig2.update_layout(font=dict(size=fontsize))

    fig2.show(renderer='browser')

    #----------------------- legend figs ------------------

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
    fig2.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\latent_space.png', width=1920, height=840)

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
    target_names = [t[0] for t in targets]

    pretty_target_names = [r"$\normalsize{(a)\ Rotational\ Constant\ A\ /GHz}$",
                           r"$\normalsize{(b)\ Rotational\ Constant\ B\ /GHz}$",
                           r"$\normalsize{(c)\ Rotational\ Constant\ C\ /GHz}$",
                           r"$\normalsize{(d)\ Dipole\ Moment\ /Deb}$",
                           r"$\normalsize{(e)\ Iso.\ Polarizability\ /Bohr^3}$",
                           r"$\normalsize{(f)\ HOMO\ Energy\ /E_h}$",
                           r"$\normalsize{(g)\ LUMO\ Energy\ /E_h}$",
                           r"$\normalsize{(h)\ Gap\ Energy\ /E_h}$",
                           r"$\normalsize{(i)\ Electronic\ Spatial\ Extent\ /Bohr^2}$",
                           r"$\normalsize{(j)\ ZPV\ Energy\ /E_h}$",
                           r"$\normalsize{(k)\ Internal\ Energy\ (T=0)\ /E_h}$",
                           r"$\normalsize{(l)\ Internal\ Energy\ 298K\ /E_h}$",
                           r"$\normalsize{(m)\ Enthalpy\ 298K\ /E_h}$",
                           r"$\normalsize{(n)\ Free\ Energy\ 298K\ /E_h}$",
                           r"$\normalsize{(o)\ Heat\ Capacity\ 298K\ /cal\ mol^{-1}\ K^{-1}}$",
                           r"$\normalsize{(p)\ Dipole\ Moment\ /D}$",
                           r"$\normalsize{(q)\ Polarizability\ Tensor\ /a_0^3}$",
                           r"$\normalsize{(r)\ Quadrupole\ Moment\ /D*A}$",
                           r"$\normalsize{(s)\ Octapole\ Moment\ /D*A^2}$",
                           r"$\normalsize{(t)\ Hyperpolarizability\ /a.u.}$",
                           ]

    MAE_dict = {}
    NMAE_dict = {}
    R_dict = {}
    for ind, (path, target_name) in enumerate(zip(er_results_paths, target_names)):
        stats_dict = np.load(path, allow_pickle=True).item()
        target = np.concatenate(stats_dict['test_stats']['regressor_target']).flatten()
        prediction = np.concatenate(stats_dict['test_stats']['regressor_prediction']).flatten()

        MAE = np.abs(target - prediction).mean()
        # NMAE_i = (np.abs((target - prediction) / np.abs(target)))
        #         # NMAE_i[target == 0] = 0
        #         # NMAE = NMAE_i.mean()
        #NMAE = np.abs((target - prediction) / np.mean(target)).mean()
        NMAE = MAE / np.abs(np.mean(target))

        linreg_result = linregress(target, prediction)
        R_value = linreg_result.rvalue
        slope = linreg_result.slope
        MAE_dict[target_name] = MAE
        NMAE_dict[target_name] = NMAE
        R_dict[target_name] = R_value

    fontsize = 32
    fig3 = make_subplots(cols=4, rows=5,
                         #subplot_titles=pretty_target_names,
                         horizontal_spacing=0.04,
                         vertical_spacing=0.033)

    annotations = []
    for ind, (path, target_name) in enumerate(zip(er_results_paths, target_names)):
        stats_dict = np.load(path, allow_pickle=True).item()
        target = np.concatenate(stats_dict['test_stats']['regressor_target']).flatten()
        prediction = np.concatenate(stats_dict['test_stats']['regressor_prediction']).flatten()

        xline = np.linspace(max(min(target), min(prediction)),
                            min(max(target), max(prediction)), 2)

        xy = np.vstack([target, prediction])
        try:
            z = get_point_density(xy, bins=50)
        except:
            z = np.ones_like(target)

        row = ind // 4 + 1
        col = ind % 4 + 1
        opacity = 0.1  # np.exp(-num_points / 10000)
        fig3.add_trace(go.Scattergl(x=target,
                                    y=prediction,
                                    mode='markers',
                                    marker=dict(color=z),
                                    opacity=opacity,
                                    marker_size=8,
                                    showlegend=False),
                       row=row, col=col)
        fig3.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                       row=row, col=col)

        minval = max([np.amin(target), np.amin(prediction)])
        maxval = min([np.amax(target), np.amax(prediction)])
        fig3.update_layout({f'xaxis{ind + 1}_range': [minval, maxval]})

        annotations.append(dict(xref="x" + str(ind + 1), yref="y" + str(ind + 1),
                                x=minval + np.ptp(prediction) * 0.3,
                                y=maxval - np.ptp(prediction) * 0.175,
                                showarrow=False,
                                text=f"R={R_dict[target_name]:.4f}<br>MAE={MAE_dict[target_name]:.3g}"
                                ))

    fig3['layout']['annotations'] += tuple(annotations)
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig3.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig3.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig3.update_yaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig3.update_xaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    #fig3.update_layout(yaxis6_title='Predicted Value')
    #fig3.update_layout(xaxis18_title='Target Value')
    fig3.update_layout(font=dict(size=fontsize))
    fig3.update_annotations(font_size=fontsize)
    fig3.update_annotations(yshift=10)
    fig3.update_xaxes(tickangle=0)
    fig3.update_xaxes(showticklabels=False)
    fig3.update_yaxes(showticklabels=False)

    fig3.show(renderer='browser')
    fig3.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\QM9_properties.png', width=1920, height=2200)

    return fig3


def detailed_er_figure():
    target_names = [t[0] for t in targets]

    pretty_target_names = [r"$\large{(a)\ Rotational\ Constant\ A\ /GHz}$",
                           r"$\large{(b)\ Rotational\ Constant\ B\ /GHz}$",
                           r"$\large{(c)\ Rotational\ Constant\ C\ /GHz}$",
                           r"$\large{(a)\ Dipole\ Moment\ /Deb}$",
                           r"$\large{(b)\ Iso.\ Polarizability\ /Bohr^3}$",
                           r"$\large{(a)\ HOMO\ Energy\ /E_h}$",
                           r"$\large{(b)\ LUMO\ Energy\ /E_h}$",
                           r"$\large{(c)\ Gap\ Energy\ /E_h}$",
                           r"$\large{(c)\ Electronic\ Spatial\ Extent\ /Bohr^2}$",
                           r"$\large{(a)\ ZPV\ Energy\ /E_h}$",
                           r"$\large{(b)\ Internal\ Energy\ (T=0)\ /E_h}$",
                           r"$\large{(c)\ Internal\ Energy\ 298K\ /E_h}$",
                           r"$\large{(a)\ Enthalpy\ 298K\ /E_h}$",
                           r"$\large{(b)\ Free\ Energy\ 298K\ /E_h}$",
                           r"$\large{(c)\ Heat\ Capacity\ 298K\ /cal\ mol^{-1}\ K^{-1}}$",
                           r"$\large{(a)\ Dipole\ Moment\ /D}$",
                           r"$\large{(b)\ Polarizability\ Tensor\ /a_0^3}$",
                           r"$\large{(c)\ Quadrupole\ Moment\ /D*A}$",
                           r"$\large{(d)\ Octapole\ Moment\ /D*A^2}$",
                           r"$\large{(e)\ Hyperpolarizability\ /a.u.}$",
                           ]

    MAE_dict = {}
    NMAE_dict = {}
    R_dict = {}
    for ind, (path, target_name) in enumerate(zip(er_results_paths, target_names)):
        stats_dict = np.load(path, allow_pickle=True).item()
        target = np.concatenate(stats_dict['test_stats']['regressor_target']).flatten()
        prediction = np.concatenate(stats_dict['test_stats']['regressor_prediction']).flatten()

        MAE = np.abs(target - prediction).mean()
        # NMAE_i = (np.abs((target - prediction) / np.abs(target)))
        #         # NMAE_i[target == 0] = 0
        #         # NMAE = NMAE_i.mean()
        #NMAE = np.abs((target - prediction) / np.mean(target)).mean()
        NMAE = MAE / np.abs(np.mean(target))

        linreg_result = linregress(target, prediction)
        R_value = linreg_result.rvalue
        slope = linreg_result.slope
        MAE_dict[target_name] = MAE
        NMAE_dict[target_name] = NMAE
        R_dict[target_name] = R_value

    def er_subfigure(inds_to_plot):
        cols = len(inds_to_plot)
        fontsize = 26 if cols == 3 else 34
        fig3 = make_subplots(cols=cols, rows=1,
                             subplot_titles=[pretty_target_names[ind] for ind in inds_to_plot],
                             horizontal_spacing=0.08 if cols == 3 else 0.065,
                             )

        annotations = []
        for g_ind, ind in enumerate(inds_to_plot):
            path = er_results_paths[ind]
            target_name = target_names[ind]
            stats_dict = np.load(path, allow_pickle=True).item()
            target = np.concatenate(stats_dict['test_stats']['regressor_target']).flatten()
            prediction = np.concatenate(stats_dict['test_stats']['regressor_prediction']).flatten()

            xline = np.linspace(max(min(target), min(prediction)),
                                min(max(target), max(prediction)), 2)

            xy = np.vstack([target, prediction])
            try:
                z = get_point_density(xy, bins=50)
            except:
                z = np.ones_like(target)

            row = 1
            col = g_ind + 1
            opacity = 0.1  # np.exp(-num_points / 10000)
            fig3.add_trace(go.Scattergl(x=target,
                                        y=prediction,
                                        mode='markers',
                                        marker=dict(color=z),
                                        opacity=opacity,
                                        marker_size=8,
                                        showlegend=False),
                           row=row, col=col)
            fig3.add_trace(go.Scattergl(x=xline, y=xline, showlegend=False, marker_color='rgba(0,0,0,1)'),
                           row=row, col=col)

            minval = max([np.amin(target), np.amin(prediction)])
            maxval = min([np.amax(target), np.amax(prediction)])
            fig3.update_layout({f'xaxis{g_ind + 1}_range': [minval, maxval]})

            annotations.append(dict(xref="x" + str(g_ind + 1), yref="y" + str(g_ind + 1),
                                    x=minval + np.ptp(prediction) * 0.37,
                                    y=maxval - np.ptp(prediction) * 0.1,
                                    showarrow=False,
                                    text=f"R={R_dict[target_name]:.3f}<br>MAE={MAE_dict[target_name]:.3g}"
                                    ))

        fig3['layout']['annotations'] += tuple(annotations)
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
        fig3.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
        fig3.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
        fig3.update_yaxes(linecolor='black', mirror=True,
                          showgrid=True, zeroline=True, zerolinewidth=1)
        fig3.update_xaxes(linecolor='black', mirror=True,
                          showgrid=True, zeroline=True, zerolinewidth=1)
        fig3.update_layout(yaxis1_title='Predicted Value')
        if cols == 3:
            fig3.update_layout(
                xaxis=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                           zerolinecolor='lightgrey'),
                xaxis2=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                xaxis3=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),

                yaxis=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                           zerolinecolor='lightgrey'),
                yaxis2=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                yaxis3=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
            )
            fig3.update_layout(xaxis2_title='Target Value')
        elif cols == 5:
            fig3.update_layout(xaxis3_title='Target Value')

            fig3.update_layout(

                xaxis=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                           zerolinecolor='lightgrey'),
                xaxis2=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                xaxis3=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                xaxis4=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                xaxis5=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),

                yaxis=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                           zerolinecolor='lightgrey'),
                yaxis2=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                yaxis3=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                yaxis4=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
                yaxis5=dict(linecolor='black', mirror=True, showgrid=True, zeroline=True, zerolinewidth=1,
                            zerolinecolor='lightgrey'),
            )
        fig3.update_layout(font=dict(size=fontsize))
        fig3.update_annotations(font_size=fontsize)
        fig3.update_annotations(yshift=10)
        fig3.update_xaxes(tickangle=-65)

        return fig3

    figs = []
    for inds_list in [
        [0, 1, 2],
        [3, 4, 8],
        [5, 6, 7],
        [9, 10, 11],
        [12, 13, 14],
        [15, 16, 17, 18, 19]
    ]:
        figs.append(er_subfigure(inds_list))

    for ind, fig in enumerate(figs):
        if ind == 5:
            fig.write_image(rf'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\detailed_er_{ind}.png', width=1920,
                            height=600)
        else:
            fig.write_image(rf'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\detailed_er_{ind}.png', width=1200,
                            height=600)

    return None


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


def proxy_discriminator_figure():
    target_names = []
    MAE_dict = {}
    NMAE_dict = {}
    R_dict = {}

    for ind, path in enumerate(proxy_results_paths):
        stats_dict = np.load(path, allow_pickle=True).item()
        is_mace = stats_dict['config']['proxy_discriminator']['train_on_mace']
        is_bh = stats_dict['config']['proxy_discriminator']['train_on_bh']
        es_factor = stats_dict['config']['proxy_discriminator']['electrostatic_scaling_factor']
        embedding = stats_dict['config']['proxy_discriminator']['embedding_type']

        if is_mace:
            energy_func = 'MACE'
        elif is_bh:
            energy_func = "Buckingham"
        elif es_factor > 0:
            energy_func = f"LJ + {int(es_factor / 1000)}k ES"
        else:
            energy_func = "LJ"

        target_name = str(embedding) + ' ' + energy_func
        target_names.append(target_name)
        target = stats_dict['test_stats']['vdw_score']
        prediction = stats_dict['test_stats']['vdw_prediction']
        if energy_func == "Buckingham":
            maxval = target.max() * 1
            target = (target - maxval) / 1500
            prediction = (prediction - maxval) / 1500
        print(target_name)
        MAE = np.abs(target - prediction).mean()
        NMAE = (np.abs((target - prediction) / np.abs(target))).mean()

        linreg_result = linregress(target, prediction)
        R_value = linreg_result.rvalue
        slope = linreg_result.slope
        MAE_dict[target_name] = MAE
        NMAE_dict[target_name] = NMAE
        R_dict[target_name] = R_value

    inds_reorder = [7, 6, 5, 4, 3, 2, 1, 0, ]
    good_target_names = [target_names[ind] for ind in inds_reorder]
    good_proxy_paths = [proxy_results_paths[ind] for ind in inds_reorder]

    col_labels = ["No Embedding", "Molecule Volume", "Principal Vectors", "Autoencoder"]
    row_labels = ["Buckingham Potential (Arb. Units)", 'MACE Potential (kJ/mol)']

    num_rows = len(row_labels)
    num_cols = len(col_labels)
    fontsize = 28
    fig3 = make_subplots(cols=num_cols, rows=num_rows,
                         #subplot_titles=target_names,
                         horizontal_spacing=0.075,
                         vertical_spacing=0.085)

    annotations = []
    for ind, (path, target_name) in enumerate(zip(good_proxy_paths, good_target_names)):
        stats_dict = np.load(path, allow_pickle=True).item()
        target = stats_dict['test_stats']['vdw_score']
        prediction = stats_dict['test_stats']['vdw_prediction']
        if 'Buckingham' in target_name:
            maxval = target.max() * 1
            target = (target - maxval) / 1500
            prediction = (prediction - maxval) / 1500

        xline = np.linspace(min(target), max(target), 2)

        xy = np.vstack([target, prediction])
        try:
            z = get_point_density(xy, bins=50)
        except:
            z = np.ones_like(target)

        row = ind // num_cols + 1
        col = ind % num_cols + 1
        opacity = 0.15
        fig3.add_trace(go.Scattergl(x=target,
                                    y=prediction,
                                    mode='markers',
                                    marker=dict(color=z),
                                    opacity=opacity,
                                    showlegend=False),
                       row=row, col=col)
        fig3.add_trace(go.Scattergl(x=xline, y=xline,
                                    showlegend=False, marker_color='rgba(0,0,0,1)'),
                       row=row, col=col)

        minval = np.quantile(target, 0.01)
        maxval = np.quantile(target, 0.99)
        fig3.update_layout({f'xaxis{ind + 1}_range': [minval, maxval]})
        fig3.update_layout({f'yaxis{ind + 1}_range': [minval, maxval]})

        annotations.append(dict(xref="x" + str(ind + 1), yref="y" + str(ind + 1),
                                x=minval + np.ptp(target) * 0.225,
                                y=maxval - np.ptp(target) * 0.15,
                                showarrow=False,
                                text=f"R={R_dict[target_name]:.2f}<br>MAE={MAE_dict[target_name]:.3f}"
                                ))

    fig3['layout']['annotations'] += tuple(annotations)
    # fig3.update_layout(yaxis1_title=row_labels[0])
    # fig3.update_layout(yaxis5_title=row_labels[1])
    # fig3.update_layout(xaxis5_title=col_labels[0])
    # fig3.update_layout(xaxis6_title=col_labels[1])
    # fig3.update_layout(xaxis7_title=col_labels[2])
    # fig3.update_layout(xaxis8_title=col_labels[3])

    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig3.update_xaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig3.update_yaxes(gridcolor='lightgrey')  # , zerolinecolor='black')
    fig3.update_yaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig3.update_xaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig3.update_layout(font=dict(size=int(fontsize * 0.75)))
    fig3.update_annotations(font_size=fontsize)
    fig3.update_annotations(yshift=10)
    fig3.update_xaxes(tickangle=0)

    fig3.show(renderer='browser')
    fig3.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\proxy_discrim_i.png', width=1920, height=840)

    return fig3


if __name__ == '__main__':
    #fig = RMSD_fig()

    # fig2 = UMAP_fig(max_entries=100000000)

    #fig3 = embedding_regression_figure()

    #detailed_er_figure()

    fig5 = proxy_discriminator_figure()

aa = 1
