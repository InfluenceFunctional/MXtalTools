import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from plotly.subplots import make_subplots


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


def log_losses(wandb, losses, step, optimizer, data, batch_size, working_sigma, decoded_data, mean_num_points):
    losses_dict = {ltype: np.mean(lval[-10:]) for ltype, lval in losses.items()}
    best_losses_dict = {'best_' + ltype: np.amin(lval) for ltype, lval in losses.items()}

    wandb.log(losses_dict)
    wandb.log(best_losses_dict)

    wandb.log({'batch_size': batch_size,
               'encoder_learning_rate': optimizer.param_groups[0]['lr'],
               'decoder_learning_rate': optimizer.param_groups[1]['lr'],
               'step': step,
               'mean_type_confidence': np.mean(np.amax(decoded_data.x.cpu().detach().numpy(), axis=1)),
               'sigma': working_sigma,
               'mean_num_points_setting': mean_num_points,
               'mean_num_points_actual': data.num_nodes / data.num_graphs,
               })


def save_checkpoint(encoder, decoder, optimizer, config, step, losses):
    if losses['scaled_reconstruction_loss'][-1] == np.amin(losses['scaled_reconstruction_loss']):
        torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   config.save_directory + config.run_name + '_' + 'autoencoder_ckpt' + '_' + str(step))


def update_losses(losses, num_points_loss, reconstruction_loss, encoding_type_loss, working_sigma, type_confidence_loss, loss, nodewise_type_loss, centroid_dist_loss, centroid_std_loss):
    losses['num_points_loss'].append(num_points_loss.mean().cpu().detach().numpy())
    losses['reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy())
    losses['overall_type_loss'].append(encoding_type_loss.cpu().detach().numpy())
    losses['scaled_reconstruction_loss'].append(reconstruction_loss.cpu().detach().numpy() * working_sigma)  # as sigma decreases, credit the reconstruction loss
    losses['type_confidence_loss'].append(type_confidence_loss.cpu().detach().numpy())
    losses['combined_loss'].append(loss.cpu().detach().numpy())
    losses['nodewise_type_loss'].append(nodewise_type_loss.cpu().detach().numpy())
    losses['adjusted_nodewise_type_loss'].append((nodewise_type_loss + type_confidence_loss).cpu().detach().numpy())
    losses['centroid_mean_loss'].append(centroid_dist_loss.cpu().detach().numpy())
    losses['centroid_std_loss'].append(centroid_std_loss.cpu().detach().numpy())

    return losses


def overlap_plot(wandb, data, decoded_data, working_sigma, config, nodewise_weights):

    if config.cart_dimension == 1:
        sigma = working_sigma
        fig = make_subplots(rows=config.max_point_types, cols=min(4, data.num_graphs))

        x = np.linspace(-config.points_spread, config.points_spread, 1001)

        for j in range(config.max_point_types):
            for graph_ind in range(min(4, data.num_graphs)):
                row = j + 1
                col = graph_ind + 1

                points_true = data.pos[data.batch == graph_ind].cpu().detach().numpy()
                points_pred = decoded_data.pos[decoded_data.batch == graph_ind].cpu().detach().numpy()

                ref_type_inds = torch.argwhere(data.x[data.batch == graph_ind] == j)[:, 0].cpu().detach().numpy()
                pred_type_weights = decoded_data.x[decoded_data.batch == graph_ind, j].cpu().detach().numpy()[:, None]

                fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / sigma), axis=0), line_color='blue', showlegend=True if graph_ind == 0 else False, name='True', row=row, col=col)
                fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / sigma), axis=0), line_color='red', showlegend=True if graph_ind == 0 else False, name='Predicted', row=row, col=col)

                fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / 0.00001), axis=0), line_color='blue', showlegend=False, name='True', row=row, col=col)
                fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / 0.00001), axis=0), line_color='red', showlegend=False, name='Predicted', row=row, col=col)

        wandb.log({"Sample Distributions": fig})
