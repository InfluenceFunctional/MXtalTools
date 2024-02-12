import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.spatial.distance import cdist

n_input_particles = 4
n_output_particles = 12
np.random.seed(3)
cmax = 1
max_point_types = 2
num_graphs = 1
min_xval, max_xval = 0, 1
sigma = 0.001

pos = np.array([0.1, 0.3, 0.7, 0.9])  # np.random.uniform(0, max_xval - min_xval, size=n_input_particles) + min_xval
types = np.array([1, 0, 0, 1])  # np.random.randint(max_point_types, size=n_input_particles)

num_gridpoints = 100
x = np.linspace(-.25, 1.25, num_gridpoints)
y = np.linspace(-.25, 1.25, num_gridpoints)
xx, yy = np.meshgrid(x, y)
grid_array = np.stack((xx.flatten(), yy.flatten())).T

sigmas = [0.05, 0.025, 0.005]
fig = make_subplots(rows=1, cols=len(sigmas), horizontal_spacing=0.1,subplot_titles=[f"(a) Width={sigmas[0]:.2f}", f"(b) Width={sigmas[1]:.2f}", f"(c) Width={sigmas[2]:.2f}"])
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
fig.show()

aa = 1

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
#
# n_input_particles = 4
# n_output_particles = 4
# np.random.seed(3)
# cmax = 4
# max_point_types = 3
# num_graphs = 1
# min_xval, max_xval = -1, 1
# sigma = 0.001
#
# pos = np.random.uniform(0, max_xval - min_xval, size=n_input_particles) + min_xval
# types = np.random.randint(max_point_types, size=n_input_particles)
#
# de_pos = np.random.uniform(0, max_xval - min_xval, size=n_output_particles) + min_xval
# de_types = np.abs(np.random.randn(n_output_particles, max_point_types))
# de_types = de_types / np.sum(de_types, axis=1)[:, None]
#
#
# sigmas = [0.1, 0.01, 0.001]
# fig = make_subplots(rows=max_point_types, cols=len(sigmas))
# x = np.linspace(min(-1, min_xval), max(1, max_xval), 1001)
# for j in range(max_point_types):
#     for graph_ind, sigma in enumerate(sigmas):
#         row = j + 1
#         col = graph_ind + 1
#
#         points_true = pos
#         points_pred = de_pos
#
#         ref_type_inds = np.argwhere(types == j)
#         pred_type_weights = de_types[:, j]
#
#         fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / sigma), axis=0),
#                           line_color='blue', showlegend=True,
#                           name=f'True type {j}', legendgroup=f'True type {j}', row=row, col=col)
#
#         fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x[:, None] - points_pred) ** 2 / sigma), axis=1),
#                           line_color='red', showlegend=True,
#                           name=f'Predicted type {j}', legendgroup=f'Predicted type {j}', row=row, col=col)
#
#         # fig.add_scattergl(x=x, y=np.sum(np.exp(-(x - points_true[ref_type_inds]) ** 2 / 0.00001), axis=0), line_color='blue', showlegend=False, name='True', row=row, col=col)
#         # fig.add_scattergl(x=x, y=np.sum(pred_type_weights * np.exp(-(x - points_pred) ** 2 / 0.00001), axis=0), line_color='red', showlegend=False, name='Predicted', row=row, col=col)
# # fig.update_yaxes(range=[0, cmax])
# fig.update_xaxes(range=[min_xval + min_xval * 0.1, max_xval + max_xval * 0.1])
# fig.show()
#
# aa = 1
