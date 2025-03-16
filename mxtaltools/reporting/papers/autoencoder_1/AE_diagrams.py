import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from torch_geometric.loader.dataloader import Collater
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

from mxtaltools.common.geometry_utils import list_molecule_principal_axes_torch
from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
from mxtaltools.reporting.ae_reporting import scaffolded_decoder_clustering, swarm_vs_tgt_fig

from mxtaltools.standalone.qm9_encoder import Qm9Autoencoder
from mxtaltools.standalone.qm9_loader import QM9Loader

def crude_2d_gaussian():
    n_input_particles = 4
    n_output_particles = 12
    np.random.seed(3)
    pos = np.array([0.1, 0.3, 0.7, 0.9])  # np.random.uniform(0, max_xval - min_xval, size=n_input_particles) + min_xval
    types = np.array([1, 0, 0, 1])  # np.random.randint(max_point_types, size=n_input_particles)
    num_gridpoints = 100
    x = np.linspace(-.25, 1.25, num_gridpoints)
    y = np.linspace(-.25, 1.25, num_gridpoints)
    xx, yy = np.meshgrid(x, y)
    grid_array = np.stack((xx.flatten(), yy.flatten())).T
    sigmas = [0.05, 0.025, 0.005]
    fig = make_subplots(rows=1, cols=len(sigmas), horizontal_spacing=0.1, subplot_titles=[f"(a) Width={sigmas[0]:.2f}", f"(b) Width={sigmas[1]:.2f}", f"(c) Width={sigmas[2]:.2f}"])
    for graph_ind, sigma in enumerate(sigmas):
        row = 1
        col = graph_ind + 1

        pl = [pos + np.random.randn(len(pos)) * sigma * 5 for _ in range(3)]
        tl = [types + np.random.randn(len(types)) * sigma * 5 for _ in range(3)]
        de_pos = np.concatenate(pl)
        de_types = np.clip(np.concatenate(tl), a_min=0, a_max=1)

        points_true = np.concatenate([pos[:, None], types[:, None]], axis=1)
        points_pred = np.concatenate([de_pos[:, None], de_types[:, None]], axis=1)

        pred_dist = np.exp(-(cdist(grid_array, points_pred) ** 2 / sigma)).sum(1).reshape(num_gridpoints, num_gridpoints) / n_output_particles * n_input_particles
        true_dist = np.exp(-(cdist(grid_array, points_true) ** 2 / sigma)).sum(1).reshape(num_gridpoints, num_gridpoints)

        overlap = -np.log(pred_dist * true_dist)

        fig.add_trace(go.Contour(x=x, y=y, z=overlap,
                                 showlegend=False,
                                 name=f'Overlap', legendgroup=f'Overlap',
                                 showscale=False,
                                 reversescale=True,
                                 colorscale='bugn',
                                 line_width=0,
                                 contours=dict(start=np.amin(overlap), end=5, size=5 / 50)
                                 ), row=row, col=col)

        fig.add_trace(go.Contour(x=x, y=y, z=pred_dist,
                                 showlegend=False,
                                 name=f'Predicted type', legendgroup=f'Predicted type',
                                 contours_coloring="none",
                                 line_color='red',
                                 line_width=1,
                                 ncontours=15,
                                 ), row=row, col=col)

        fig.add_trace(go.Scattergl(x=points_pred[:, 0], y=points_pred[:, 1],
                                   mode='markers', marker_color='white', marker_size=10, marker_line_width=2, marker_line_color='red',
                                   showlegend=False,
                                   name=f'Predicted type', legendgroup=f'Predicted type'
                                   ), row=row, col=col)

        fig.add_trace(go.Contour(x=x, y=y, z=true_dist,
                                 showlegend=False,
                                 name=f'True type', legendgroup=f'True type',
                                 contours_coloring="none",
                                 line_color='blue',
                                 line_width=1,
                                 ncontours=15,
                                 ), row=row, col=col)

        fig.add_trace(go.Scattergl(x=points_true[:, 0], y=points_true[:, 1],
                                   mode='markers', marker_color='white', marker_size=10, marker_line_width=2, marker_line_color='blue',
                                   showlegend=False,
                                   name=f'True type', legendgroup=f'True type'
                                   ), row=row, col=col)
    fig.update_xaxes(range=[-.25, 1.25], title='Cartesian Dimension')
    fig.update_yaxes(range=[-.25, 1.25], title='Type Dimension')
    fig.update_layout(coloraxis_showscale=False)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(showlegend=False)
    fig.update_layout(font_size=32)
    fig.update_annotations(font_size=32)
    fig.show(renderer='browser')
    aa = 1


# crude_2d_gaussian()

def get_decoding(data, encoder, noise=None):
    fidelity, encoding, decoding = encoder.evaluate_encoding(data, return_encodings=True, noise=noise)
    print(f"Decoding accuracy at {fidelity.mean():.2f}")
    decoded_data = encoder.generate_decoded_data(data, decoding)
    encoding = encoding.cpu().detach().numpy()
    decoded_data = decoded_data.cpu().detach()
    data.pos *= encoder.config.autoencoder.molecule_radius_normalization
    decoded_data.pos *= encoder.config.autoencoder.molecule_radius_normalization
    return data, decoded_data, fidelity.mean()


def process_flat_molecule(loader, flat_ind, encoder):
    # flat_ind = 3 # good index for proton-less
    collater = Collater(0, 0)
    data = collater([loader.data_manager.datapoints[flat_ind]])
    data.pos -= data.pos.mean(0)
    data.pos /= encoder.config.autoencoder.molecule_radius_normalization
    data.x = encoder.autoencoder_type_index[data.x.long() - 1]
    # align it to x-y plane
    data = align_mol_batch_to_standard_axes(data, 1)
    # print(f"Un-flat by {data.pos[:, 2].abs().sum().numpy():.2f}")
    return data


def converging_gaussians_fig():
    loader = QM9Loader(device='cpu')
    loader.dataset_config.filter_protons = False
    loader.load_dataset(max_dataset_length=10000)
    encoder = Qm9Autoencoder(device='cpu',
                             num_atom_types=loader.dataDims['num_atom_types'],
                             min_num_atoms=loader.dataDims['min_molecule_num_atoms'],
                             max_num_atoms=loader.dataDims['max_molecule_num_atoms'],
                             max_molecule_radius=loader.dataDims['max_molecule_radius'],
                             )
    """
    get a flat molecule
    align it in the xy plane
    run it through the encoder
    plot the typewise gaussian mixtures of input and decoding
    """
    num_atom_types = loader.dataDims['num_atom_types']
    # get 1k samples
    collater = Collater(None, None)
    batch = collater(loader.data_manager.datapoints[:1000])
    # pick out flat molecules
    # flat_mols = []
    # for ind in range(batch.num_graphs):
    #     # pick out flat molecules
    #     flat_ind = ind
    #     # flat_ind = 3 # good index for proton-less
    #     data = collater([loader.data_manager.datapoints[flat_ind]])
    #     data.pos -= data.pos.mean(0)
    #     data.pos /= encoder.config.autoencoder.molecule_radius_normalization
    #     data.x = encoder.autoencoder_type_index[data.x.long() - 1]
    #     # align it to x-y plane
    #     data = align_crystaldata_to_principal_axes(data, 1)
    #     unflatness = data.pos[:, 2].abs().sum().numpy()
    #     if unflatness < 1e-1:
    #         flat_mols.append(ind)
    sigmas = [0.5, 0.25, .05]
    noises = [0.1, 0.05, 0]
    for flat_ind in [980]:  # , 20, 28, 53, 95]:
        data = process_flat_molecule(loader, flat_ind, encoder)
        scaled_data, decoded_data, _ = get_decoding(data.clone(), encoder, noise=None)

        ptp_x = np.ptp(scaled_data.pos[:, 0].numpy())
        ptp_y = np.ptp(scaled_data.pos[:, 1].numpy())

        min_xval, max_xval = -ptp_x / 2, ptp_x / 2
        min_yval, max_yval = -ptp_y / 2, ptp_y / 2

        num_gridpoints = 150
        x = np.linspace(min_xval * 1.75, max_xval * 1.75, num_gridpoints)
        y = np.linspace(min_yval * 1.75, max_yval * 1.75, num_gridpoints)
        xx, yy = np.meshgrid(x, y)
        grid_array = np.stack((xx.flatten(), yy.flatten())).T

        colors = ['rgb(255, 255, 255)', 'rgb(144, 144, 144)', 'rgb(48, 80, 248)', 'rgb(255, 13, 13)', 'rgb(144, 224, 80)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)',
                  'rgb(165, 170, 153)'] * 10
        colorscales = [[[0, 'rgba(0, 0, 0, 0)'], [1, color]] for color in colors]

        overlaps = []
        for ind, (sigma, noise) in enumerate(zip(sigmas, noises)):
            scaled_data, decoded_data, overlap = get_decoding(data.clone(), encoder, noise)
            rmsd, max_dist, weight_mean,match_successful = scaffolded_decoder_clustering(0, scaled_data, decoded_data,
                                                                        num_atom_types,
                                                                        return_fig=False)
            overlaps.append(overlap)

        fig = make_subplots(rows=1, cols=4,
                            specs=[[{'type': 'surface'} for _ in range(4)]],
                            subplot_titles=[f"$\sigma={sigma}, Overlap={rmsd:.2f}$" for sigma, rmsd in zip(sigmas, overlaps)] + ['Input'])

        if num_atom_types == 4:
            atom_type_list = ['Carbon', 'Nitrogen', 'Oxygen', 'Fluorine']
            colors = colors[1:]
        elif num_atom_types == 5:
            atom_type_list = ['Hydrogen', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine']
        else:
            assert False

        for ind, (sigma, noise) in enumerate(zip(sigmas, noises)):
            scaled_data, decoded_data, _ = get_decoding(data.clone(), encoder, noise)
            rmsd, max_dist, weight_mean, match_successful = scaffolded_decoder_clustering(0, scaled_data, decoded_data,
                                                                        num_atom_types,
                                                                        return_fig=False)
            for j in range(num_atom_types):
                points_true = scaled_data.pos[data.x[:, 0] == j]

                points_pred = decoded_data.pos.numpy()
                pred_type_weights = (decoded_data.aux_ind * decoded_data.x[:, j]).cpu().detach().numpy()
                decoded_density = np.sum(
                    pred_type_weights * np.exp(-(cdist(grid_array, points_pred[:, :2]) ** 2 / sigma)), axis=-1).reshape(
                    num_gridpoints, num_gridpoints)

                fig.add_trace(go.Surface(x=x, y=y, z=decoded_density,
                                         surfacecolor=decoded_density,
                                         opacity=1, cmin=0, cmax=1,
                                         colorscale=colorscales[j],
                                         showscale=False, ), row=1, col=ind + 1)

                fig.add_trace(go.Scatter3d(x=points_true[:, 0], y=points_true[:, 1], z=points_true[:, 2] + 0.05,
                                           mode='markers', marker_color=colors[j], marker_size=7, marker_line_width=5, marker_line_color='black',
                                           showlegend=False, name=atom_type_list[j]
                                           ), row=1, col=ind + 1)

        for j in range(num_atom_types):
            points_true = scaled_data.pos[scaled_data.x[:, 0] == j]
            fig.add_trace(go.Scatter3d(x=points_true[:, 0], y=points_true[:, 1], z=points_true[:, 2] + 0.05,
                                       mode='markers', marker_color=colors[j], marker_size=7, marker_line_width=5, marker_line_color='black',
                                       showlegend=True, name=atom_type_list[j]
                                       ), row=1, col=ind + 2)
            input_density = np.sum(np.exp(-(cdist(grid_array, points_true[:, :2]) ** 2 / sigma)), axis=-1).reshape(
                num_gridpoints, num_gridpoints)

            fig.add_trace(go.Surface(x=x, y=y, z=input_density,
                                     surfacecolor=input_density,
                                     opacity=1, cmin=0, cmax=1,
                                     colorscale=colorscales[j],
                                     showscale=False, ), row=1, col=ind + 2)

        fig.update_layout(legend={'itemsizing': 'constant'})
        fig.update_scenes(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="grey",
                showbackground=True,
                zerolinecolor="white", ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="grey",
                showbackground=True,
                zerolinecolor="white"),
            zaxis=dict(
                backgroundcolor="rgba(1, 1, 1,0.1)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white", ),
            aspectratio=dict(x=1, y=1, z=0.25),
            camera=dict(eye=dict(x=0, y=1.5, z=1.45)),
        )

        fig.show(renderer='browser')



def converging_caaussians_gif():
    loader = QM9Loader(device='cpu')
    loader.dataset_config.filter_protons = False
    loader.load_dataset(max_dataset_length=10000)
    encoder = Qm9Autoencoder(device='cpu',
                             num_atom_types=loader.dataDims['num_atom_types'],
                             min_num_atoms=loader.dataDims['min_molecule_num_atoms'],
                             max_num_atoms=loader.dataDims['max_molecule_num_atoms'],
                             max_molecule_radius=loader.dataDims['max_molecule_radius'],
                             )

    flat_ind = 980
    data = process_flat_molecule(loader, flat_ind, encoder)
    scaled_data, decoded_data, _ = get_decoding(data.clone(), encoder, noise=None)

    ptp_x = np.ptp(scaled_data.pos[:, 0].numpy())
    ptp_y = np.ptp(scaled_data.pos[:, 1].numpy())

    min_xval, max_xval = -ptp_x / 2, ptp_x / 2
    min_yval, max_yval = -ptp_y / 2, ptp_y / 2

    num_gridpoints = 150
    x = np.linspace(min_xval * 1.75, max_xval * 1.75, num_gridpoints)
    y = np.linspace(min_yval * 1.75, max_yval * 1.75, num_gridpoints)
    xx, yy = np.meshgrid(x, y)
    grid_array = np.stack((xx.flatten(), yy.flatten())).T

    colors = ['rgb(255, 255, 255)', 'rgb(144, 144, 144)', 'rgb(48, 80, 248)', 'rgb(255, 13, 13)', 'rgb(144, 224, 80)', 'rgb(36, 121, 108)', 'rgb(218, 165, 27)', 'rgb(47, 138, 196)', 'rgb(118, 78, 159)', 'rgb(237, 100, 90)',
              'rgb(165, 170, 153)'] * 10
    colorscales = [[[0, 'rgba(0, 0, 0, 0)'], [1, color]] for color in colors]

    sigmas = np.logspace(np.log10(0.5), np.log10(0.05), 100)
    noises = np.logspace(0, -3, 100)

    num_atom_types = loader.dataDims['num_atom_types']

    if num_atom_types == 4:
        atom_type_list = ['Carbon', 'Nitrogen', 'Oxygen', 'Fluorine']
        colors = colors[1:]
    elif num_atom_types == 5:
        atom_type_list = ['Hydrogen', 'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine']
    else:
        assert False

    for ind, (sigma, noise) in enumerate(tqdm(zip(sigmas, noises))):
        scaled_data, decoded_data, overlap = get_decoding(data.clone(), encoder, noise)
        rmsd, max_dist, weight_mean, match_su = scaffolded_decoder_clustering(0, scaled_data, decoded_data,
                                                                    num_atom_types,
                                                                    return_fig=False)
        fig = go.Figure()
        for j in range(num_atom_types):
            points_true = scaled_data.pos[data.x[:, 0] == j]

            points_pred = decoded_data.pos.numpy()
            pred_type_weights = (decoded_data.aux_ind * decoded_data.x[:, j]).cpu().detach().numpy()
            decoded_density = np.sum(
                pred_type_weights * np.exp(-(cdist(grid_array, points_pred[:, :2]) ** 2 / sigma)), axis=-1).reshape(
                num_gridpoints, num_gridpoints)

            fig.add_trace(go.Surface(x=x, y=y, z=decoded_density,
                                     surfacecolor=decoded_density,
                                     opacity=1, cmin=0, cmax=1,
                                     colorscale=colorscales[j],
                                     showscale=False, ))

            fig.add_trace(go.Scatter3d(x=points_true[:, 0], y=points_true[:, 1], z=points_true[:, 2] + 0.05,
                                       mode='markers', marker_color=colors[j], marker_size=7, marker_line_width=5, marker_line_color='black',
                                       showlegend=True, name=atom_type_list[j]
                                       ))

            fig.update_layout(legend={'itemsizing': 'constant'})
            fig.update_scenes(
                xaxis=dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    gridcolor="grey",
                    showbackground=True,
                    zerolinecolor="white", ),
                yaxis=dict(
                    backgroundcolor="rgba(0, 0, 0,0)",
                    gridcolor="grey",
                    showbackground=True,
                    zerolinecolor="white"),
                zaxis=dict(
                    backgroundcolor="rgba(1, 1, 1,0.1)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white", ),
                aspectratio=dict(x=1, y=1, z=0.25),
                camera=dict(eye=dict(x=0, y=1.5, z=1.45)),
            )
            fig.update_layout(title=f"$\sigma={sigma:.2f}, Overlap={overlap:.2f}$")

            fig.write_image(rf'C:\Users\mikem\crystals\CSP_runs\artifacts\frame_{ind}.png', width=480, height=480)

    import glob
    from PIL import Image

    # from https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/
    import os
    globs = sorted(glob.glob(rf"C:\Users\mikem\crystals\CSP_runs\artifacts/*.png"), key=os.path.getmtime)
    frames = [Image.open(image) for image in globs]
    frame_one = frames[0]
    frame_one.save(r"C:\Users\mikem\crystals\CSP_runs\artifacts/gaussian_convergence.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)

converging_gaussians_fig()


# fig2 = swarm_vs_tgt_fig(data, decoded_data, 4)
# fig2.show(renderer='browser')
aa = 1
