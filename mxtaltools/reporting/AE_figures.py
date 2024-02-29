import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from PIL import Image

stats_dict_paths = [r'C:\Users\mikem\crystals\CSP_runs\models/_tests_qm9_test23_4_26-02-22-29-57.npy',
                    r'C:\Users\mikem\crystals\CSP_runs\models/_tests_qm9_test23_7_27-02-14-34-41.npy',
                    r'C:\Users\mikem\crystals\CSP_runs\models/_tests_qm9_test23_8_27-02-15-35-51.npy']

stats_dict_names = ["Without Protons",
                    "With Protons",
                    "Inferred Protons"]

colors = n_colors('rgb(250,50,5)', 'rgb(5,120,200)', len(stats_dict_paths), colortype='rgb')


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
        stats_dicts[stats_dict_names[ind]] = d['test_stats']

    bandwidth = 0.005
    fig = make_subplots(rows=1, cols=2, subplot_titles=['(a) RMSD Distribution', '(b) RMSD Correlates'], column_widths=[0.75, 0.25], horizontal_spacing=0.15)
    for ind, run_name in enumerate(stats_dicts.keys()):

        stats_dict = stats_dicts[run_name]
        mol_keys = [key for key in stats_dict.keys() if 'molecule_' in key]
        mol_keys.remove('molecule_smiles')
        mol_keys.remove('molecule_halogen_count')
        mol_keys.remove('molecule_alkyl_halide_count')

        try:
            smiles = np.concatenate(stats_dict['molecule_smiles'])
        except:
            smiles = stats_dict['molecule_smiles']

        x = np.concatenate(stats_dict['scaffold_rmsds'])
        unmatched = np.mean(np.isinf(x))
        finite_x = x[np.isfinite(x)]
        print(finite_x.mean())

        fig.add_annotation(x=0.4, y=ind + 0.3, showarrow=False, text=f'Matched RMSD: {finite_x.mean():.2f} <br> Unmatched Frac.: {unmatched * 100:.0f}%', row=1, col=1)
        fig.add_trace(go.Violin(  # y=np.zeros_like(x),
            x=x, side='positive', orientation='h',
            bandwidth=bandwidth, width=4, showlegend=True, opacity=1,  # .5,
            name=run_name,
            # scalegroup='',
            meanline_visible=True,
            line_color=colors[ind],
            points=False),
            row=1, col=1
        )
        add_mol_diagrams_to_rmsd_violin(fig, finite_x, ind, smiles)

        correlates_dict, keys_list2 = compute_rmsd_correlates(mol_keys, stats_dict, x)

        fig.add_trace(go.Bar(
            y=[keys_list2[entry] for entry in list(correlates_dict.keys())],
            x=[corr for corr in correlates_dict.values()],
            textposition='auto',
            orientation='h',
            showlegend=False,
            # text=[corr for corr in correlates_dict.values()],
            # texttemplate='%{text:.2f}',
            marker_color=colors[ind],
        ), row=1, col=2)

    fig.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(xaxis1={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})  # , 'linecolor': 'white', 'linewidth': 5})
    fig.update_layout(yaxis1={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})
    fig.update_layout(xaxis2={'gridcolor': 'lightgrey', 'zerolinecolor': 'black'})

    fig.update_layout(xaxis1_range=[-0.15, 0.9])
    fig.update_layout(font=dict(size=24))
    fig.update_xaxes(title_font=dict(size=20), tickfont=dict(size=14))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_layout(legend_traceorder='reversed')  # , yaxis_showgrid=True)
    fig.update_layout(xaxis1_title='RSMD (Angstrom)', xaxis2_title="Correlation Coefficient")
    fig.update_layout(xaxis1_tick0=0, xaxis1_dtick=0.05)
    fig.update_layout(xaxis2_tick0=0, xaxis2_dtick=0.05)

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
    tr = np.argmin(np.linalg.norm(embedding - [xref, yref] + np.random.randn(len(embedding), embedding.shape[1]) * 0.05, axis=1))
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
        stats_dicts[stats_dict_names[ind]] = d['test_stats']

    import umap
    embeddings = []
    for ind, run_name in enumerate(["Without Protons"]):  # stats_dicts.keys()):
        reducer = umap.UMAP(n_components=2,
                            metric='cosine',
                            n_neighbors=10,
                            min_dist=0.2)
        scalar_encodings = stats_dicts[run_name]['scalar_encoding'][:max_entries]

        embeddings.append(reducer.fit_transform((scalar_encodings - scalar_encodings.mean()) / scalar_encodings.std()))

    ind = 0
    run_name = "Without Protons"

    composition_coloration = np.stack([np.concatenate(stats_dicts[run_name]['molecule_C_fraction']),
                                       np.concatenate(stats_dicts[run_name]['molecule_N_fraction']),
                                       np.concatenate(stats_dicts[run_name]['molecule_O_fraction'])]).T[:max_entries]
    point_colors1 = normalize_colors(composition_coloration)
    legend_entries1 = ["Carbon Enriched", "Nitrogen Enriched", "Oxygen Enriched"]

    composition_coloration = np.stack([np.concatenate(stats_dicts[run_name]['molecule_principal_moment_1']),
                                       np.concatenate(stats_dicts[run_name]['molecule_principal_moment_2']),
                                       np.concatenate(stats_dicts[run_name]['molecule_principal_moment_3'])]).T[:max_entries]
    point_colors2 = normalize_colors(composition_coloration)
    legend_entries2 = ["Ip1", "Ip2", "Ip3"]

    composition_coloration = np.stack([np.concatenate(stats_dicts[run_name]['molecule_num_rings']),
                                       np.concatenate(stats_dicts[run_name]['molecule_num_rotatable_bonds']),
                                       np.concatenate(stats_dicts[run_name]['molecule_radius'])]).T[:max_entries]
    point_colors3 = normalize_colors(composition_coloration)
    legend_entries3 = ["# Rings", "# Rotatable Bonds", "Mol. Radius"]

    fig2 = make_subplots(rows=1, cols=3, horizontal_spacing=.01)
    annotations_list = []
    for ind2, (point_colors, legend_entries) in enumerate(zip([point_colors1, point_colors2, point_colors3], [legend_entries1, legend_entries2, legend_entries3])):

        embedding = embeddings[stats_dict_names.index(run_name)]
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
        try:
            smiles = np.concatenate(stats_dicts[run_name]['molecule_smiles'])
        except:
            smiles = stats_dicts[run_name]['molecule_smiles']

        for ix in np.linspace(0, 1, 8):
            for iy in np.linspace(0, 1, 8):
                if ix == 0 or ix == 1 or iy == 0 or iy == 1:
                    annotations_list.append(mol_point_callout(fig2, ix, iy, ex, ey, embedding, smiles, 0.125, row=1, col=ind2 + 1))

        fig2.add_scattergl(x=np.zeros(1), y=np.zeros(1), marker_size=0.001, mode='markers', marker_color=['rgb(255,0,0)'], name=legend_entries[0], showlegend=True, row=1, col=ind2 + 1)
        fig2.add_scattergl(x=np.zeros(1), y=np.zeros(1), marker_size=0.001, mode='markers', marker_color=['rgb(0,255,0)'], name=legend_entries[1], showlegend=True, row=1, col=ind2 + 1)
        fig2.add_scattergl(x=np.zeros(1), y=np.zeros(1), marker_size=0.001, mode='markers', marker_color=['rgb(0, 0,255)'], name=legend_entries[2], showlegend=True, row=1, col=ind2 + 1)

    fig2.update_layout(annotations=annotations_list)

    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig2.update_yaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)
    fig2.update_xaxes(linecolor='black', mirror=True,
                      showgrid=True, zeroline=True)

    xlim = -0.2
    ylim = 1.2
    fig2.update_layout(xaxis1_range=[xlim, ylim], yaxis1_range=[xlim, ylim],
                       xaxis2_range=[xlim, ylim], yaxis2_range=[xlim, ylim],
                       xaxis3_range=[xlim, ylim], yaxis3_range=[xlim, ylim])

    fig2.update_layout(
        xaxis2_title='Component 1',
        yaxis1_title='Component 2',
    )
    fig2.update_layout(legend={'itemsizing': 'constant', 'orientation': 'h'})

    fig2.update_layout(font=dict(size=18))
    fig2.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig2.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))

    fig2.show(renderer='browser')

    return fig2


def normalize_colors(composition_coloration):
    composition_coloration -= composition_coloration.mean(0)[None, :]
    composition_coloration /= composition_coloration.std(0)[None, :]
    composition_coloration = np.clip(composition_coloration, a_min=np.quantile(composition_coloration, 0.05), a_max=np.quantile(composition_coloration, 0.95))
    composition_coloration -= composition_coloration.min(0)
    composition_coloration /= composition_coloration.max(0)
    composition_coloration *= 255
    point_colors = [f'rgb({int(color[0])},{int(color[1])},{int(color[2])})' for color in composition_coloration]
    return point_colors


fig = RMSD_fig()
fig.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\RMSD.png', width=1920, height=840)
fig2 = UMAP_fig(max_entries=1000)
fig2.write_image(r'C:\Users\mikem\OneDrive\NYU\CSD\papers\ae_paper1\latent_space.png', width=1920, height=840)

aa = 1
