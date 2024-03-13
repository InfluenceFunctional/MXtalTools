from plotly import io, graph_objects
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def nice_dataset_analysis(config, dataset):
    '''
    distributions of dataset features
    - molecule num atoms
    - num rings
    - num donors
    - num acceptors
    - atom fractions CNOFCl Metals
    -
    '''
    import plotly.io as pio
    pio.renderers.default = 'browser'

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
        )
    )

    rows = 3
    cols = 4
    mol_feats = ['molecule num atoms', 'molecule num rings', 'molecule num donors', 'molecule num acceptors',
                 'molecule planarity', 'molecule C fraction', 'molecule N fraction', 'molecule O fraction',
                 'crystal packing coefficient', 'crystal lattice centring', 'crystal system', 'crystal z value']

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=mol_feats, horizontal_spacing=0.04,
                        vertical_spacing=0.1)
    for ii, feat in enumerate(mol_feats):
        fig.add_trace(go.Histogram(x=dataset[feat],
                                   histnorm='probability density',
                                   nbinsx=50,
                                   showlegend=False,
                                   marker_color='#0c4dae'),
                      row=(ii) // cols + 1, col=(ii) % cols + 1)
    fig.update_layout(width=900, height=600)
    fig.layout.margin = layout.margin
    fig.write_image('../paper1_figs/dataset_statistics.png', scale=4)
    if config.machine == 'local':
        fig.show(renderer='browser')

    return None
