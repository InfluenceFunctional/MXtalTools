import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from _plotly_utils.colors import n_colors, sample_colorscale


def log_rmsd_loss(wandb, data, decoded_data):
    rmsds = np.zeros(data.num_graphs)
    general_rmsds = np.zeros_like(rmsds)
    for g_ind in range(data.num_graphs):
        inds = data.batch == g_ind

        ref_coords = data.pos[inds]
        ref_types = data.x[inds, 0]

        pred_coords = decoded_data.pos[inds]
        pred_types = torch.argmax(decoded_data.x[inds], dim=1)

        dists = cdist(ref_coords.cpu().detach().numpy(), pred_coords.cpu().detach().numpy())
        assignment = linear_sum_assignment(dists)

        general_rmsds[g_ind] = dists[assignment].sum() / len(ref_coords)

        a, b = torch.unique(ref_types, return_counts=True)
        c, d = torch.unique(pred_types, return_counts=True)
        if len(a) == len(c):
            if all(a == c) and all(b == d):
                typewise_rmsds = np.zeros(len(a))

                for it, type_ind in enumerate(a):
                    ref_type_pos = ref_coords[ref_types == type_ind]
                    pred_type_pos = pred_coords[ref_types == type_ind]

                    dists = cdist(ref_type_pos.cpu().detach().numpy(), pred_type_pos.cpu().detach().numpy())
                    assignment = linear_sum_assignment(dists)
                    typewise_rmsds[it] = dists[assignment].sum() / len(ref_type_pos)

                rmsds[g_ind] = np.mean(typewise_rmsds)
            else:
                rmsds[g_ind] = np.nan
        else:
            rmsds[g_ind] = np.nan

    wandb.log({'Matching Clouds Fraction': np.mean(np.isfinite(rmsds)),
               'Matching Clouds RMSDs': np.mean(rmsds[np.isfinite(rmsds)]),
               'All Points RMSDs': np.mean(general_rmsds)})


def log_losses(wandb, losses, step, optimizer, data, batch_size, working_sigma,
               decoded_data, mean_sample_likelihood, min_points, max_points):
    losses_dict = {ltype: np.mean(lval[-10:]) for ltype, lval in losses.items()}
    best_losses_dict = {'best_' + ltype: np.amin(lval) for ltype, lval in losses.items()}

    wandb.log(losses_dict)
    wandb.log(best_losses_dict)

    wandb.log({'batch_size': batch_size,
               'encoder_learning_rate': optimizer.param_groups[0]['lr'],
               'decoder_learning_rate': optimizer.param_groups[1]['lr'],
               'step': step,
               'sigma': working_sigma,
               'min_num_points': min_points,
               'max_num_points': max_points,
               'mean_num_points_actual': data.num_nodes / data.num_graphs,
               'mean_sample_likelihood': mean_sample_likelihood,
               })


def save_checkpoint(encoder, decoder, optimizer, config, step, losses):
    if losses['scaled_reconstruction_loss'][-1] == np.amin(losses['scaled_reconstruction_loss']):
        torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   config.save_directory + config.run_name + '_' + 'autoencoder_ckpt' + '_' + str(step))


def update_losses(losses, num_points_loss, reconstruction_loss, encoding_type_loss, working_sigma, loss,
                  nodewise_type_loss, centroid_dist_loss,
                  constraining_loss):
    losses['num_points_loss'].append(num_points_loss.mean().cpu().detach().numpy())
    losses['reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy())
    losses['overall_type_loss'].append(encoding_type_loss.cpu().detach().numpy())
    losses['scaled_reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy() * working_sigma)  # as sigma decreases, credit the reconstruction loss
    losses['combined_loss'].append(loss.cpu().detach().numpy())
    losses['nodewise_type_loss'].append(nodewise_type_loss.cpu().detach().numpy())
    losses['centroid_mean_loss'].append(centroid_dist_loss.cpu().detach().numpy())
    losses['constraining_loss'].append(constraining_loss.cpu().detach().numpy())

    return losses


def overlap_plot(wandb, data, decoded_data, working_sigma, config, nodewise_weights):

    sigma = working_sigma

    max_xval = max(decoded_data.pos.amax(), data.pos.amax()).cpu().detach().numpy()
    min_xval = min(decoded_data.pos.amax(), data.pos.amin()).cpu().detach().numpy()
    ymax = 1.1  # int(torch.diff(data.ptr).amax())  # max graph height

    if config.cart_dimension == 1:
        fig = make_subplots(rows=config.max_point_types, cols=min(4, data.num_graphs))

        x = np.linspace(min(-config.points_spread, min_xval), max(config.points_spread, max_xval), 1001)

        for j in range(config.max_point_types):
            for graph_ind in range(min(4, data.num_graphs)):
                row = j + 1
                col = graph_ind + 1

                points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
                points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()

                ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()
                pred_type_weights = decoded_data.x[decoded_data.batch == graph_ind, j].cpu().detach().numpy()[:, None]

                fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / sigma), axis=0),
                                  line_color='blue', showlegend=True if (j == 0 and graph_ind == 0) else False,
                                  name=f'True type {j}', legendgroup=f'Predicted type {j}', row=row, col=col)

                fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / sigma), axis=0),
                                  line_color='red', showlegend=True if j == 0 and graph_ind == 0 else False,
                                  name=f'Predicted type {j}', legendgroup=f'Predicted type {j}', row=row, col=col)

                # fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / 0.00001), axis=0), line_color='blue', showlegend=False, name='True', row=row, col=col)
                # fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / 0.00001), axis=0), line_color='red', showlegend=False, name='Predicted', row=row, col=col)

        fig.update_yaxes(range=[0, ymax])
        wandb.log({"Sample Distributions": fig})

    elif config.cart_dimension == 2:
        fig = make_subplots(rows=config.max_point_types, cols=min(4, data.num_graphs))

        num_gridpoints = 25
        x = np.linspace(min(-config.points_spread, min_xval), max(config.points_spread, max_xval), num_gridpoints)
        y = np.copy(x)
        xx, yy = np.meshgrid(x, y)

        grid_array = np.stack((xx.flatten(), yy.flatten())).T

        for j in range(config.max_point_types):
            for graph_ind in range(min(4, data.num_graphs)):
                row = j + 1
                col = graph_ind + 1

                points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
                points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()

                ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()
                pred_type_weights = decoded_data.x[decoded_data.batch == graph_ind, j].cpu().detach().numpy()[:, None]

                pred_dist = np.sum(pred_type_weights.mean() * np.exp(-(cdist(grid_array, points_pred) ** 2 / sigma)), axis=-1).reshape(num_gridpoints, num_gridpoints)

                fig.add_trace(go.Contour(x=x, y=y, z=pred_dist,
                                         showlegend=True if (j == 0 and graph_ind == 0) else False,
                                         name=f'Predicted type', legendgroup=f'Predicted type',
                                         coloraxis="coloraxis",
                                         contours=dict(start=0, end=ymax, size=ymax / 50)
                                         ), row=row, col=col)

                fig.add_trace(go.Scattergl(x=points_true[ref_type_inds][:, 0], y=points_true[ref_type_inds][:, 1],
                                           mode='markers', marker_color='white', marker_size=10, marker_line_width=2, marker_line_color='green',
                                           showlegend=True if (j == 0 and graph_ind == 0) else False,
                                           name=f'True type', legendgroup=f'True type'
                                           ), row=row, col=col)

        fig.update_coloraxes(cmin=0, cmax=ymax, autocolorscale=False, colorscale='viridis')
        wandb.log({"Sample Distributions": fig})

    elif config.cart_dimension == 3:

        num_types = config.max_point_types
        cols = 3
        rows = num_types // cols + ((num_types % cols) != 0)
        fig = make_subplots(
            rows=rows, cols=cols,
            specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)])

        num_gridpoints = 15
        x = np.linspace(min(-config.points_spread, min_xval), max(config.points_spread, max_xval), num_gridpoints)
        y = np.copy(x)
        z = np.copy(x)
        xx, yy, zz = np.meshgrid(x, y, z)

        grid_array = np.stack((xx.flatten(), yy.flatten(), zz.flatten())).T

        for j in range(config.max_point_types):
            graph_ind = 0
            row = j // cols + 1
            col = j % cols + 1

            points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
            points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()

            ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()
            pred_type_weights = decoded_data.x[decoded_data.batch == graph_ind, j].cpu().detach().numpy()[:, None]

            pred_dist = np.sum(pred_type_weights.T * np.exp(-(cdist(grid_array, points_pred) ** 2 / sigma)), axis=-1)

            fig.add_trace(go.Volume(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=pred_dist,
                                    showlegend=True if (j == 0 and graph_ind == 0) else False,
                                    name=f'Predicted type', legendgroup=f'Predicted type',
                                    coloraxis="coloraxis",
                                    isomin=0.001, isomax=ymax, opacity=.025,
                                    cmin=0, cmax=ymax,
                                    surface_count=50,
                                    ), row=row, col=col)

            fig.add_trace(go.Scatter3d(x=points_true[ref_type_inds][:, 0], y=points_true[ref_type_inds][:, 1], z=points_true[ref_type_inds][:, 2],
                                       mode='markers', marker_color='white', marker_size=10, marker_line_width=2, marker_line_color='green',
                                       showlegend=True if (j == 0 and graph_ind == 0) else False,
                                       name=f'True type', legendgroup=f'True type'
                                       ), row=row, col=col)

        fig.update_coloraxes(autocolorscale=False, colorscale='Jet')
        wandb.log({"Sample Distributions": fig})

    return None
