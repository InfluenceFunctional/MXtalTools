from torch.cuda import synchronize
from time import time
from mxtaltools.dataset_utils.utils import collate_data_list
import torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go


samples = torch.load(r"D:\crystal_datasets\test_new_new_csd.pt", weights_only=False)
samples = [elem for elem in samples if elem.z_prime == 1]
#bsz = 1
device = 'cpu'
ind = (collate_data_list(samples).num_atoms == 8).argwhere().flatten()[0]
batch = collate_data_list(samples[ind], max_z_prime=1).to(device)

cutoff = 6
clusters = batch.mol2cluster(cutoff=cutoff, supercell_size=5)
#batch.visualize()
#clusters.visualize(mode='convolve with')

pp = clusters.pos.numpy()
ll = clusters.aux_ind.numpy()
pp = pp[ll < 2]
ll = ll[ll < 2]

pos = {ind: row for ind,row in enumerate(pp)}
labels = {ind: row for ind,row in enumerate(ll)}

G = nx.DiGraph()

for i in pos:
    G.add_node(i)

nodes = list(pos.keys())

for i in nodes:
    for j in nodes:
        if i == j:
            continue

        li, lj = labels[i], labels[j]

        # ignore anything involving label 2
        if li == 2 or lj == 2:
            continue

        if li == 1 and lj == 1:
            continue

        dij = np.linalg.norm(pos[i] - pos[j])
        if dij < cutoff:
            if li == lj:
                G.add_edge(i,j, directed=False, length=dij)
            else:
                G.add_edge(i, j, directed=True, length=dij)

G.remove_nodes_from(list(nx.isolates(G)))

# nodes = list(G.nodes)
# node_index = {n: i for i, n in enumerate(nodes)}
#
# D = np.zeros((len(nodes), len(nodes)))
#
# for i, u in enumerate(nodes):
#     for j, v in enumerate(nodes):
#         D[i, j] = np.linalg.norm(pos[u] - pos[v])

#pos2d = nx.kamada_kawai_layout(G)#, pos=pos)

X = np.stack([pos[i] for i in G.nodes()])
X -= X.mean(axis=0)

_, _, Vt = np.linalg.svd(X, full_matrices=False)
X2 = X @ Vt[:2].T

pos2d = {n: X2[k] for k, n in enumerate(G.nodes())}

pos2d = nx.kamada_kawai_layout(
    G.to_undirected(),
    pos=pos2d,
    scale=1.0
)

edge_x_undirected = []
edge_y_undirected = []

edge_x_directed = []
edge_y_directed = []

for u, v, data in G.edges(data=True):
    x0, y0 = pos2d[u]
    x1, y1 = pos2d[v]

    if data["directed"]:
        edge_x_directed += [x0, x1, None]
        edge_y_directed += [y0, y1, None]
    else:
        edge_x_undirected += [x0, x1, None]
        edge_y_undirected += [y0, y1, None]

node_x = []
node_y = []
node_color = []
node_text = []

for i in G.nodes:
    x, y = pos2d[i]
    node_x.append(x)
    node_y.append(y)
    node_color.append(labels[i])
    node_text.append(f"node {i} (label {labels[i]})")


fig = go.Figure()
label_colors = {
    0: "rgb(0, 0, 200)",   # blue (asymmetric unit)
    1: "rgb(200, 200, 200)",   # gray (cluster)
}

node_marker_colors = [label_colors[l] for l in node_color]

# undirected 0--0 edges
fig.add_trace(go.Scatter(
    x=edge_x_undirected,
    y=edge_y_undirected,
    mode="lines",
    line_color="rgb(150, 150, 200)",
    line=dict(width=1.0),
    hoverinfo="none",
    name="0–0"
))

# directed 1->0 edges
fig.add_trace(go.Scatter(
    x=edge_x_directed,
    y=edge_y_directed,
    mode="lines",
    line_color='grey',
    line=dict(width=0.25),#, dash="dot"),
    hoverinfo="none",
    name="1 → 0"
))

fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",#"markers+text",
    #text=node_text,
    #textposition="top center",
    marker=dict(
        size=8,
        color=node_marker_colors,
        colorscale="Viridis",
        line=dict(width=1)
    ),
    name="nodes"
))

fig.update_layout(
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    margin=dict(l=20, r=20, t=20, b=20),
    width=600,
    height=600,
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
)

fig.show()

aa = 0

