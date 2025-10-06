import os
from pathlib import Path
import torch.nn.functional as F
from typing import Optional
import torch
from torch_scatter import scatter

from mxtaltools.common.geometry_utils import safe_batched_eigh, compute_cosine_similarity_matrix, center_mol_batch, \
    compute_ellipsoid_volume
from mxtaltools.models.functions.radial_graph import asymmetric_radius_graph
from mxtaltools.models.modules.components import ResidualMLP


# noinspection PyAttributeOutsideInit
class MolCrystalEllipsoidOps:

    def compute_ellipsoidal_overlap(self,
                                    surface_padding: float = 0,
                                    return_details: Optional[bool] = False,
                                    model: Optional = None,
                                    **kwargs):
        """
        Compute an energy function given the overlaps of molecules in an ellipsoid representation
        Using our pretrained ellipsoid overlap function

        Only works on cluster batches (rather than crystal batches)

        1) embed all molecules as ellipsoids
        2) get inter-ellipsoidal edges
        3) parameterize ellipsoid pairs
        4) format batch to ellipsoid model
        5) evaluate overlaps
        6) convert overlaps to energies
        """
        if model is None:
            if not hasattr(self, "ellipsoid_model"):
                self.load_ellipsoid_model()
        else:
            ellipsoid_model = model

        if not hasattr(self, 'edges_dict'):
            raise RuntimeError("Need to build radial graph before computing overlaps")

        mols_per_cluster = torch.tensor(self.edges_dict['n_repeats'], device=self.device)
        tot_num_mols = torch.sum(mols_per_cluster)
        tot_mol_index = torch.arange(tot_num_mols, device=self.device).repeat_interleave(
            self.num_atoms.repeat_interleave(mols_per_cluster))
        molwise_batch = torch.arange(self.num_graphs, device=self.device).repeat_interleave(mols_per_cluster, dim=0)

        edge_i_good, edge_j_good, mol_centroids = self.get_intermolecular_ellipsoid_edges(molwise_batch,
                                                                                          surface_padding,
                                                                                          tot_mol_index, tot_num_mols)

        (atoms_per_necessary_mol, mol_id_map, molwise_batch_subset,
         num_necessary_mols, subset_pos, tot_mol_index_subset) = self.reindex_ellipsoid_mols(
            edge_i_good, edge_j_good, molwise_batch, tot_mol_index, tot_num_mols)

        """get ellipsoids"""
        add_noise = 0.01
        cov_eps = 0.01

        eigvals_sorted, eigvecs_sorted = self.compute_ellipsoid_eigvecs(add_noise, atoms_per_necessary_mol, cov_eps,
                                                                        molwise_batch_subset, num_necessary_mols,
                                                                        subset_pos,
                                                                        tot_mol_index_subset)

        """
        Set default as the ellipsoid tip being at the surface of the molecule (assume largest radius)
        Then, plus or minus angstroms to expose or cover surface atoms
        """
        eps = 1e-3

        longest_length = self.radius[molwise_batch_subset]
        padding_scaling_factor = (longest_length + surface_padding) / longest_length
        min_axis_length = torch.amax(
            torch.stack([1.5 * padding_scaling_factor, 0.1 * torch.ones_like(padding_scaling_factor)]),
            dim=0)  # need a finite thickness for flat molecules
        sqrt_eigenvalues = torch.sqrt(eigvals_sorted.clamp(min=0) + eps)
        normed_eigs = sqrt_eigenvalues / sqrt_eigenvalues.amax(1, keepdim=True)  # normalize to relative lengths
        # semi axis scale now controls how much of the surface is revealed - for negative values, surface atoms will poke out
        # if the surface padding is set too small, the ellipsoid will just retreat into a tiny sphere
        semi_axis_lengths = (normed_eigs * longest_length[:, None] + surface_padding).clip(min=min_axis_length[:, None])
        Ip = eigvecs_sorted

        """ featurize ellipsoids """
        norm_factor, normed_v1, normed_v2, v1, v2, x = self.featurize_ellipsoids(Ip, edge_i_good, edge_j_good, eps,
                                                                                 mol_centroids, mol_id_map,
                                                                                 semi_axis_lengths)

        # pass to the model
        if hasattr(self, 'ellipsoid_model') and model is None:
            output = self.ellipsoid_model(x)
        else:
            output = ellipsoid_model(x)

        # process results
        v1_pred, v2_pred, normed_overlap_pred = (output[:, 0] * norm_factor ** 3,
                                                 output[:, 1] * norm_factor ** 3,
                                                 output[:, 2].clip(min=0)
                                                 )
        reduced_volume = (normed_v1 * normed_v2) / (normed_v1 + normed_v2)
        denormed_overlap_pred = normed_overlap_pred * reduced_volume  # model works in the reduced basis
        overlap_pred = denormed_overlap_pred * norm_factor ** 3  # inunits of cubic angstroms
        v_pred_error = (v1_pred - v1).abs() / v1 + (v2_pred - v2).abs() / v2

        # sum of overlaps cubic angstroms per molecule
        molwise_ellipsoid_overlap = scatter(overlap_pred, molwise_batch[edge_j_good], dim=0, dim_size=self.num_graphs,
                                            reduce='sum')

        normed_ellipsoid_overlap = scatter(normed_overlap_pred, molwise_batch[edge_j_good], dim=0,
                                           dim_size=self.num_graphs,
                                           reduce='sum')

        if not return_details:
            return normed_ellipsoid_overlap
        else:
            return normed_ellipsoid_overlap, v1_pred, v2_pred, v1, v2, norm_factor, molwise_ellipsoid_overlap

    def featurize_ellipsoids(self, Ip, edge_i_good, edge_j_good, eps, mol_centroids, mol_id_map, semi_axis_lengths):
        # featurize pairs
        r = mol_centroids[edge_j_good] - mol_centroids[edge_i_good]
        edge_i_local = mol_id_map[edge_i_good]
        edge_j_local = mol_id_map[edge_j_good]
        if not torch.isfinite(Ip).all():
            raise ValueError("Non-finite principal axes!")
        e1 = Ip[edge_i_local] * semi_axis_lengths[edge_i_local, :, None]
        e2 = Ip[edge_j_local] * semi_axis_lengths[edge_j_local, :, None]
        v1 = compute_ellipsoid_volume(e1)
        v2 = compute_ellipsoid_volume(e2)
        if not torch.isfinite(v1).all():
            raise ValueError("Non-finite volume!")
        if not (v1 > 0).all():
            raise ValueError("Non-zero volume!")
        if not torch.isfinite(v2).all():
            raise ValueError("Non-zero volume!")
        if not torch.isfinite(v1).all():
            raise ValueError("Non-zero volume!")
        if not torch.isfinite(semi_axis_lengths).all():
            raise ValueError("Non-finite semi-axes!")
        if not (semi_axis_lengths > 0).all():
            raise ValueError("Non-zero semi-axes!")
        # normalize
        max_e1 = e1.norm(dim=-1).amax(1)
        max_e2 = e2.norm(dim=-1).amax(1)
        max_val = torch.stack([max_e1, max_e2]).T.amax(1) + eps
        normed_e1 = e1 / max_val[:, None, None]
        normed_e2 = e2 / max_val[:, None, None]
        normed_r = r / max_val[:, None]
        normed_v1 = v1 / max_val ** 3
        normed_v2 = v2 / max_val ** 3
        # standardize directions
        dot1 = torch.einsum('nij,ni->nj', normed_e1, normed_r)
        sign_flip1 = (dot1 < 0).float() * -2 + 1  # flips points against r
        std_normed_e1 = normed_e1 * sign_flip1.unsqueeze(-1)
        dot2 = torch.einsum('nij,ni->nj', normed_e2, -normed_r)
        sign_flip2 = (dot2 < 0).float() * -2 + 1  # flips points same way as r
        std_normed_e2 = normed_e2 * sign_flip2.unsqueeze(-1)
        # parameterize
        r_hat = F.normalize(normed_r + eps, dim=-1)
        r1_local = torch.einsum('nij,nj->ni', std_normed_e1, r_hat)  # r in frame of ellipsoid 1
        r2_local = torch.einsum('nij,nj->ni', std_normed_e2, -r_hat)  # r in frame of ellipsoid 2
        unit_std_normed_e1 = std_normed_e1 / std_normed_e1.norm(dim=-1, keepdim=True)
        unit_std_normed_e2 = std_normed_e2 / std_normed_e2.norm(dim=-1, keepdim=True)
        # relative rotation matrix
        # R_rel = torch.einsum('nik, njk -> nij', unit_std_normed_e1, unit_std_normed_e2)
        cmat = compute_cosine_similarity_matrix(unit_std_normed_e1, unit_std_normed_e2)
        x = torch.cat([
            normed_r.norm(dim=-1, keepdim=True),
            cmat.reshape(len(e1), 9),
            std_normed_e1.norm(dim=-1),
            std_normed_e2.norm(dim=-1),
            r1_local,
            r2_local,
        ], dim=1)
        return max_val, normed_v1, normed_v2, v1, v2, x

    def compute_ellipsoid_eigvecs(self, add_noise, atoms_per_necessary_mol, cov_eps, molwise_batch_subset,
                                  num_necessary_mols,
                                  subset_pos, tot_mol_index_subset):
        # get principal axes
        centered_mol_pos = center_mol_batch(subset_pos,
                                            tot_mol_index_subset,
                                            num_graphs=len(molwise_batch_subset),
                                            nodes_per_graph=atoms_per_necessary_mol)
        if centered_mol_pos.requires_grad or (add_noise > 0):
            coords_to_compute = centered_mol_pos + torch.randn_like(centered_mol_pos) * cov_eps
        else:
            coords_to_compute = centered_mol_pos
        # we'll get the eigenvalues of the covariance matrix to approximate the molecule spatial extent
        # Compute outer products: [N, 3, 3]
        outer = coords_to_compute[:, :, None] * coords_to_compute[:, None, :]
        # Accumulate covariance sums per molecule
        cov_sums = torch.zeros((num_necessary_mols, 3, 3), device=self.device)
        cov_sums = cov_sums.index_add(0, tot_mol_index_subset, outer)
        # Normalize by number of atoms per molecule
        atoms_per_necessary_mol = atoms_per_necessary_mol.to(dtype=torch.float32)  # in case it's int
        covariances = cov_sums / atoms_per_necessary_mol[:, None, None]
        # explicitly symmetrize
        covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
        covariances = covariances + torch.eye(3, device=covariances.device).expand(len(covariances), -1, -1) * 1e-3
        eigvals, eigvecs_c = safe_batched_eigh(covariances)
        eigvecs = eigvecs_c.permute(0, 2, 1)  # switch to row-wise eigenvectors
        sort_inds = torch.argsort(eigvals, dim=1, descending=True)  # we want eigenvectors sorted a>b>c row-wise
        eigvals_sorted = torch.gather(eigvals, dim=1, index=sort_inds)
        eigvecs_sorted = torch.gather(eigvecs, dim=1, index=sort_inds.unsqueeze(2).expand(-1, -1, 3))
        return eigvals_sorted, eigvecs_sorted

    def reindex_ellipsoid_mols(self, edge_i_good, edge_j_good, molwise_batch, tot_mol_index, tot_num_mols):
        necessary_mol_inds = torch.unique(torch.cat([edge_i_good, edge_j_good]))
        num_necessary_mols = len(necessary_mol_inds)
        mol_id_map = torch.full((tot_num_mols,), -1, device=self.device)
        mol_id_map[necessary_mol_inds] = torch.arange(len(necessary_mol_inds), device=self.device)
        # atom_mask = torch.isin(tot_mol_index, necessary_mol_inds)
        atom_mask = mol_id_map[tot_mol_index] != -1
        tot_mol_index_subset = mol_id_map[tot_mol_index[atom_mask]]
        molwise_batch_subset = molwise_batch[necessary_mol_inds]
        crystal_with_necessary_mol_ind, mols_per_necessary_cluster = torch.unique(molwise_batch_subset,
                                                                                  return_counts=True)
        full_mols_per_necessary_cluster = torch.zeros(self.num_graphs, device=self.device, dtype=torch.long)
        full_mols_per_necessary_cluster[crystal_with_necessary_mol_ind] = mols_per_necessary_cluster
        # mols_per_necessary_cluster = scatter(torch.ones_like(molwise_batch_subset), molwise_batch_subset, reduce='sum', dim=0)
        # mols_per_necessary_cluster = mols_per_necessary_cluster[mols_per_necessary_cluster>0]
        atoms_per_necessary_mol = scatter(torch.ones_like(tot_mol_index_subset), tot_mol_index_subset, dim=0,
                                          reduce='sum')  # self.num_atoms[molwise_batch_subset].repeat_interleave(mols_per_necessary_cluster, dim=0)
        subset_pos = self.pos[atom_mask]
        return atoms_per_necessary_mol, mol_id_map, molwise_batch_subset, num_necessary_mols, subset_pos, tot_mol_index_subset

    def get_intermolecular_ellipsoid_edges(self, molwise_batch, surface_padding, tot_mol_index, tot_num_mols):
        """get edges"""
        max_ellipsoid_radius = self.radius + surface_padding
        mol_centroids = scatter(self.pos, tot_mol_index, dim=0, dim_size=tot_num_mols, reduce='mean')
        mol_aux_inds = scatter(self.aux_ind, tot_mol_index, dim=0, dim_size=tot_num_mols, reduce='max')
        # get edges
        edge_i, edge_j = asymmetric_radius_graph(
            x=mol_centroids,
            batch=molwise_batch,
            inside_inds=torch.argwhere(mol_aux_inds == 0).flatten(),
            convolve_inds=torch.argwhere(mol_aux_inds >= 1).flatten(),
            # take 1 and 2 here, or we might have indexing issues
            max_num_neighbors=50,
            r=max_ellipsoid_radius.amax() * 2)
        # filter edges longer than 2x mol_radius for each sample
        dists = torch.linalg.norm(mol_centroids[edge_i] - mol_centroids[edge_j], dim=1)
        good_inds = dists < (2 * max_ellipsoid_radius[molwise_batch[edge_i]])
        edge_i_good = edge_i[good_inds]
        edge_j_good = edge_j[good_inds]
        return edge_i_good, edge_j_good, mol_centroids

    """
    # Check predictions
    from mxtaltools.common.ellipsoid_ops import compute_ellipsoid_overlap

    n_samples = 10
    overlaps = torch.zeros(n_samples, device=self.device)
    for ind in range(n_samples):
        overlaps[ind], _ = compute_ellipsoid_overlap(e1[ind], e2[ind], v1[ind], v2[ind], r[ind], num_probes=10000,
                                                     show_tqdm=True)

    import plotly.graph_objects as go


    def simple_parity(x, y):
        go.Figure(
            go.Scatter(x=x, y=y.cpu().detach(), mode='markers', showlegend=True,
                       name=f'R={torch.corrcoef(torch.stack([x, y]))[0, 1].cpu().detach().numpy():.4f}')).show(renderer='browser')


    simple_parity(overlaps.cpu().detach(), overlap_pred[:n_samples].cpu().detach())
    simple_parity(torch.cat([v1, v2]).cpu().detach(), torch.cat([v1_pred, v2_pred]).cpu().detach())


    """

    """  # to visualize ellipsoid fit
    import numpy as np
    import plotly.graph_objects as go


    def plot_ellipsoid_and_points(center, eigvals, eigvecs, points, n=50):

        # Create unit sphere
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack((x, y, z), axis=-1)  # shape (n, n, 3)

        # Scale by sqrt of eigenvalues (semi-axes)
        radii = np.sqrt(eigvals)
        ellipsoid = sphere * radii

        # Rotate by eigenvectors
        ellipsoid = ellipsoid @ eigvecs.T  # (n, n, 3)

        # Translate to center
        ellipsoid += center

        # Extract coordinates for surface plot
        x_e = ellipsoid[..., 0]
        y_e = ellipsoid[..., 1]
        z_e = ellipsoid[..., 2]

        # Create ellipsoid surface
        ellipsoid_surface = go.Surface(
            x=x_e, y=y_e, z=z_e,
            opacity=0.5,
            colorscale='Blues',
            showscale=False,
        )

        # Create scatter plot of points
        scatter_points = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Atoms'
        )

        # Center marker (optional)
        center_marker = go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers',
            marker=dict(size=5, color='black'),
            name='Center'
        )

        fig = go.Figure(data=[ellipsoid_surface, scatter_points, center_marker])
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show()

    eigvecs = Ip[0].cpu().detach().numpy()
    eigvals = (semi_axis_lengths[0].cpu().detach().numpy())**2
    points = self.pos[tot_mol_index == 0].cpu().detach().numpy()
    center = np.mean(points, axis=0)

    plot_ellipsoid_and_points(center, eigvals, eigvecs, points)

    """

    def load_ellipsoid_model(self):
        self.ellipsoid_model = ResidualMLP(
            22, 512, 3, 8, None, 0
        )
        # todo update these paths with saved constants
        self_path = Path(os.path.realpath(__file__)).parent.resolve()
        checkpoint = torch.load(self_path.parent.joinpath(Path('ellipsoid_overlap_model.pt')),
                                weights_only=True, map_location=self.device)
        self.ellipsoid_model.load_state_dict(checkpoint)
        self.ellipsoid_model.to(self.device).eval()
