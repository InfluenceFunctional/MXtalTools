"""Neural-network building blocks: normalization and activation selectors, residual
scalar and scalar-plus-vector MLPs, vector-to-scalar and vector activation layers, and a
node embedding block.

Vector features are laid out as [num_samples, 3, channels], with the Cartesian axis in
dimension 1.
"""

import sys
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch_geometric import nn as gnn

from mxtaltools.models.modules.vector_LayerNorm import VectorLayerNorm


class Scalarizer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 embedding_dim: int,
                 norm_mode: str,
                 act_func: str,
                 dropout: float = 0,
                 output_dim: int = None,
                 ):
        """
        Map vectors [n, 3, hidden_dim] to rotation-invariant scalars [n, output_dim].

        A bias-free linear layer reduces the ``hidden_dim`` input channels to
        ``r = max(hidden_dim // 4, 1)``. The reduced vectors give their norms [n, r] and
        the dot products of the unit reduced vectors with ``embedding_dim`` unit
        directions, each a bias-free linear combination of the reduced vectors,
        [n, embedding_dim * r]. The concatenation passes through ``Linear(4 * r,
        output_dim)``, :class:`Normalization` (called without a batch vector),
        :class:`Activation` and dropout. Norms in the denominators carry an epsilon of
        1e-3.

        Parameters
        ----------
        hidden_dim : int
            Channel count of the input vectors.
        embedding_dim : int
            Number of learned directions. The concatenated width ``(1 + embedding_dim) * r``
            matches the ``4 * r`` inputs of the final linear layer only when
            ``embedding_dim`` is 3.
        norm_mode : str or None
            Normalization name for :class:`Normalization`.
        act_func : str or None
            Activation name for :class:`Activation`.
        dropout : float, optional
            Dropout probability. Default 0.
        output_dim : int, optional
            Output width. Defaults to ``hidden_dim``.
        """
        super(Scalarizer, self).__init__()
        if output_dim is None:
            output_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        reduced_dim = max(hidden_dim // 4, 1)
        #self.linear = nn.Linear(int(hidden_dim * (1 + embedding_dim)), hidden_dim, bias=True)
        self.embedding = nn.Linear(reduced_dim, embedding_dim, bias=False)
        self.dim_red = nn.Linear(hidden_dim, reduced_dim, bias=False)
        self.linear = nn.Linear(reduced_dim * 4, output_dim, bias=True)
        self.norm = Normalization(norm_mode, output_dim)
        self.activation = Activation(act_func, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                v: torch.Tensor
                ) -> torch.Tensor:
        """Return scalars [n, output_dim] from vectors ``v`` [n, 3, hidden_dim]."""
        eps = 1e-3
        v_red = self.dim_red(v)
        norm = torch.linalg.norm(v_red, dim=1)
        normed_v_red = v_red / (norm[:, None, :] + eps)

        directions = self.embedding(v_red)
        normed_directions = directions / (torch.linalg.norm(directions, dim=1)[:, None, :] + eps)

        projections = torch.einsum('nik,nij->njk', normed_v_red, normed_directions)

        v2 = torch.cat([norm, projections.flatten(1)], dim=1)

        return self.dropout(self.activation(self.norm(self.linear(v2))))


class VectorActivation(nn.Module):
    r"""
    Modified implementation of the vector activation function from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_layers.py

    A bias-free linear combination of the input channels gives one axis per sample,
    normalised to unit length (epsilon 1e-5). For each channel, the component ``c`` of the
    vector along that axis (not normalised by the vector's length) is clipped to
    ``min(c, 0)``, passed through :class:`Activation`, and ``-activation(min(c, 0)) * axis``
    is added to the channel.

    Channels with ``c >= 0`` therefore receive ``-activation(0) * axis``, which is zero for
    ``'relu'``, ``'gelu'``, ``'leaky relu'`` and ``'tanh'``. With ``'relu'`` every
    correction is zero and the layer returns its input unchanged; with ``'gelu'`` a
    channel with ``c < 0`` gets ``-gelu(c) * axis`` added, which shrinks the magnitude of
    its negative component.

    Parameters
    ----------
    hidden_dim : int
        Channel count ``k`` of the input and output vectors [n, 3, k].
    act_func : str or None
        Activation name for :class:`Activation`, applied to the clipped components.
    """

    def __init__(self,
                 hidden_dim: int,
                 act_func: str):
        super(VectorActivation, self).__init__()

        self.embedding = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = Activation(act_func, hidden_dim)

    def forward(self,
                v: torch.Tensor
                ) -> torch.Tensor:
        """Return the activated vectors, same shape as ``v`` [n, 3, hidden_dim]."""
        direction = self.embedding(v)[..., -1]
        normed_direction = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-5)

        projection = torch.einsum('nik,ni->nk', v, normed_direction).clip(max=0)
        correction = -self.activation(projection[..., None]) * normed_direction[:, None, :]

        activated_output = v + correction.permute(0, 2, 1)

        return activated_output

    '''
    # tests # todo write equivariance check
    # assert projection.max() <= 1
    # assert projeciton.min() >= -1
    # assert torch.einsum('nik,ni->nk', activated_output, normed_direction).min() >= 1e-3, "Vector Activation Failed"

    import numpy as np
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_histogram(x=projection.flatten(),nbinsx=100)
    fig.add_histogram(x=projection2.flatten(), nbinsx=100)
    fig.show(renderer='browser')
    '''


# noinspection PyAttributeOutsideInit
class Normalization(nn.Module):
    r"""
    Wrapper module for several normalization options, selected by name.

    Parameters
    ----------
    norm : str or None
        ``'batch'`` (:class:`torch.nn.BatchNorm1d`), ``'layer'``
        (:class:`torch.nn.LayerNorm`), ``'graph layer'``
        (:class:`torch_geometric.nn.LayerNorm`), ``'graph'``
        (:class:`torch_geometric.nn.GraphNorm`), ``'instance'``
        (:class:`torch.nn.InstanceNorm1d`), ``'graph vector layer'``
        (:class:`VectorLayerNorm` in ``'graph'`` mode), ``'vector layer'``
        (:class:`VectorLayerNorm` in ``'node'`` mode), or None (identity). Any other
        string prints a message and calls ``sys.exit()``.
    filters : int
        Feature width of the inputs to be normalized.
    """

    def __init__(self,
                 norm: str,
                 filters: int):
        super().__init__()
        self.norm_type = norm
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'graph layer':
            self.norm = gnn.LayerNorm(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(filters)  # not tested
        elif norm == 'graph':
            self.norm = gnn.GraphNorm(filters)
        elif norm == 'graph vector layer':
            self.norm = VectorLayerNorm(filters, mode='graph')
        elif norm == 'vector layer':
            self.norm = VectorLayerNorm(filters, mode='node')
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self,
                x: torch.Tensor,
                batch: Optional[torch.LongTensor] = None):
        """Normalize ``x``.

        The layer is called as ``norm(x, batch)`` when ``batch`` is given and the mode is
        not ``'batch'``, ``'layer'`` or None, and as ``norm(x)`` otherwise.
        :class:`torch.nn.InstanceNorm1d` takes no second argument, so ``'instance'`` with
        a ``batch`` raises TypeError.

        Parameters
        ----------
        x : torch.Tensor
            Features, [num_samples, filters], or vectors [num_samples, 3, filters] for the
            vector modes.
        batch : torch.LongTensor, optional
            Graph index of each row of ``x``, [num_samples].
        """

        if batch is not None and self.norm_type != 'batch' and self.norm_type != 'layer' and self.norm_type is not None:
            return self.norm(x, batch)

        return self.norm(x)


class Activation(nn.Module):
    r"""
    Wrapper module for several activation options, selected by name.

    Parameters
    ----------
    activation_func : str or None
        Case-insensitive name: ``'relu'``, ``'gelu'``, ``'leaky relu'``, ``'tanh'`` or
        ``'sigmoid'`` (the :mod:`torch.nn.functional` functions), or ``'kernel'``
        (:class:`KernelActivation` with 10 basis functions over span 4). None gives the
        identity. Any other string assigns no activation, so :meth:`forward` raises
        AttributeError.
    filters : int
        Feature width of the inputs; used only by ``'kernel'``.
    """

    def __init__(self,
                 activation_func: str,
                 filters: int):
        super().__init__()
        if activation_func is not None:
            if activation_func.lower() == 'relu':
                self.activation = F.relu
            elif activation_func.lower() == 'gelu':
                self.activation = F.gelu
            elif activation_func.lower() == 'kernel':  # rather expensive
                self.activation = KernelActivation(n_basis=10, span=4, filters=filters)
            elif activation_func.lower() == 'leaky relu':
                self.activation = F.leaky_relu
            elif activation_func.lower() == 'tanh':
                self.activation = F.tanh
            elif activation_func.lower() == 'sigmoid':
                self.activation = F.sigmoid
        elif activation_func is None:
            self.activation = nn.Identity()

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """Apply the selected activation to ``x``."""
        return self.activation(x)


class KernelActivation(nn.Module):
    r"""
    Function for learning an activation function for every node in a given layer, as a linear combination of basis functions over a given span.
    Uses nn.Conv1d groups option for efficient evaluation.

    Each input value is expanded over ``n_basis`` Gaussians ``exp(-gamma * (x - c)**2)``
    with centres ``c = torch.linspace(-span, span, n_basis)`` (buffer ``dict``) and
    ``gamma = 1 / (6 * spacing**2)`` for the centre spacing (buffer ``gamma``). A
    bias-free ``nn.Conv1d`` with ``groups=filters`` combines the basis responses of each
    channel into one output per channel.

    Parameters
    ----------
    n_basis : int
        Number of basis functions.
    span : float
        Basis centres are spread evenly over ``[-span, span]``.
    filters : int
        Feature width of the inputs to be activated.
    """

    def __init__(self,
                 n_basis: int,
                 span: float,
                 filters: int):
        super(KernelActivation, self).__init__()

        self.channels, self.n_basis = filters, n_basis
        # define the space of basis functions
        self.register_buffer('dict',
                             torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[
            -2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(filters * n_basis, filters, kernel_size=(1, 1), groups=int(filters), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self,
               x: torch.Tensor
               ) -> torch.Tensor:
        """Return the Gaussian basis responses of ``x`` [n, filters] as [n, filters, n_basis]."""
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        """Apply the learned activation to ``x`` [n, filters]."""
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3],
                      x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x


# noinspection PyAttributeOutsideInit
class scalarMLP(nn.Module):  # todo simplify and smooth out +1's and other custom methods for a general depth controller
    r"""
    Flexible residual multi-layer perceptron module, with several options.

    Each of the ``layers`` blocks computes
    ``x = r(x) + dropout(act(norm(linear(x))))``, or
    ``x = r(x) + dropout(act(linear(norm(x))))`` when ``norm_after_linear`` is False.
    ``r`` is a clone of ``x`` when every hidden width equals ``input_dim`` and a
    bias-free linear layer (``residue_adjust``) otherwise. A bias-free linear output layer
    maps the last hidden width to ``output_dim``; it is the identity when they match.

    Parameters
    ----------
    layers : int
        Number of residual blocks. Must be at least 1; with 0 the constructor raises
        IndexError.
    filters : int
        Hidden width of every block. Not used when ``ramp_depth`` is True.
    input_dim : int
        Width of the input features.
    output_dim : int
        Width of the output.
    activation : str or None, optional
        Activation name for :class:`Activation`. Default ``'gelu'``.
    seed : int, optional
        Passed to ``torch.manual_seed`` at construction. Default 0.
    dropout : float, optional
        Dropout probability after each block's activation. Default 0.
    norm : str or None, optional
        Normalization name for :class:`Normalization`. Default None (identity).
    norm_after_linear : bool, optional
        Normalize the linear layer's output (True) or the block input (False). Default
        True.
    bias : bool, optional
        Whether the block linear layers have a bias. Default True.
    ramp_depth : bool, optional
        If True, block ``i`` (``i = 0 .. layers - 1``) has width
        ``int(input_dim ** (1 + (p - 1) * i / layers))`` with
        ``p = log(output_dim) / log(input_dim)``: the widths change geometrically from
        ``input_dim`` (block 0) toward ``output_dim``, and the output layer makes the last
        step. Default False.

    Notes
    -----
    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator.
    """

    def __init__(self,
                 layers: int,
                 filters: int,
                 input_dim: int,
                 output_dim: int,
                 activation: str = 'gelu',
                 seed: int = 0,
                 dropout: float = 0,
                 norm: 'str' = None,
                 norm_after_linear: bool = True,
                 bias: bool = True,
                 ramp_depth: bool = False,
                 ):
        super(scalarMLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.bias = bias
        self.norm_after_linear = norm_after_linear

        self.ramp_depth = ramp_depth

        torch.manual_seed(seed)
        self.init_scalar_filters(filters)
        self.init_scalar_transforms()

    def init_scalar_filters(self, filters):
        """
        Set the per-block widths and residual projections; returns None.

        Assigns ``s_filters_in`` and ``s_filters_out`` (lists of length ``n_layers``) and
        ``same_depth``; when the widths are not all equal, also builds ``residue_adjust``,
        one bias-free ``Linear(s_filters_in[i], s_filters_out[i])`` per block.
        """
        if self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # log scaling for consistent growth ratio
            p = np.log(self.output_dim) / np.log(self.input_dim)
            n_filters = [self.input_dim] + [int(self.input_dim ** (1 + (p - 1) * (i / self.n_layers))) for i in
                         range(self.n_layers)]
        else:
            n_filters = [self.input_dim] + [filters for _ in range(self.n_layers)]
        self.s_filters_in = n_filters[:-1]
        self.s_filters_out = n_filters[1:]

        if n_filters.count(n_filters[0]) != len(
                n_filters):  # if they are not all the same, we need residue adjustments
            self.same_depth = False
            self.residue_adjust = torch.nn.ModuleList([
                nn.Linear(self.s_filters_in[i], self.s_filters_out[i], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.same_depth = True

    def init_scalar_transforms(self):
        """Build the per-block linear, activation, normalization and dropout modules and the output layer."""

        '''working layers'''
        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(self.s_filters_in[i],
                      self.s_filters_out[i], bias=self.bias)
            for i in range(self.n_layers)
        ])
        self.fc_activations = torch.nn.ModuleList([
            Activation(self.activation, self.s_filters_out[i])
            for i in range(self.n_layers)
        ])
        if self.norm_after_linear:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.s_filters_out[i])
                for i in range(self.n_layers)
            ])
        else:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode,
                              self.s_filters_in[i]
                              )
                for i in range(self.n_layers)
            ])
        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        '''output layer'''
        if self.output_dim != self.s_filters_out[-1]:
            self.output_layer = nn.Linear(self.s_filters_out[-1], self.output_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                return_latent: bool = False,
                batch: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        """Apply the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input features, [num_samples, input_dim].
        return_latent : bool, optional
            Also return the last hidden activation, before the output layer.
        batch : torch.LongTensor, optional
            Graph index of each row, forwarded to every block's :class:`Normalization`.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            The output, [num_samples, output_dim]; with ``return_latent``,
            ``(output, latent)`` where ``latent`` has the last hidden width.
        """

        for i, (norm, linear, activation, dropout) in enumerate(
                zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts)):

            'get residue'
            if self.same_depth:
                res_x = x.clone()
            else:
                res_x = self.residue_adjust[i](x)

            'linear layer'
            if self.norm_after_linear:
                x = res_x + dropout(activation(norm(linear(x), batch=batch)))
            else:
                x = res_x + dropout(activation(linear(norm(x, batch=batch))))

        if return_latent:
            return self.output_layer(x), x
        else:
            return self.output_layer(x)


# noinspection PyAttributeOutsideInit
class vectorMLP(scalarMLP):
    r"""
    scalarMLP model with l=1 vector track added with o3 equivariance.

    Takes scalars ``x`` [n, input_dim] and vectors ``v`` [n, 3, vector_input_dim]. Each
    block ``i``:

    1. takes residuals of ``x`` and ``v`` (clones when the widths do not change, else
       bias-free linear projections);
    2. adds (``v_to_s_combination='sum'``) or concatenates (``'concatenate'``) a
       :class:`Scalarizer` embedding of ``v`` to ``x``;
    3. updates ``x`` as a :class:`scalarMLP` block does;
    4. updates ``v`` to ``(res_v + g * vact(vnorm(W v))) / 1.6``, where ``W`` is a
       bias-free channel-mixing linear layer, ``vnorm`` is the ``vector_norm``
       normalization, ``vact`` is a :class:`VectorActivation`, and the gate ``g``
       [n, 1, width] is ``act(norm(linear(x)))`` of the updated scalars.

    Bias-free output layers map the scalar track to ``output_dim`` and the vector track to
    ``vector_output_dim`` (identities when the widths already match). Every operation on
    ``v`` mixes only the channel dimension or scales by quantities built from norms and
    dot products, so with ``vector_norm`` None, ``'vector layer'`` or
    ``'graph vector layer'`` and invariant ``x``, ``v`` transforms equivariantly under
    orthogonal transformations of its Cartesian axis.

    Parameters
    ----------
    layers : int
        Number of blocks. Must be at least 1.
    filters : int
        Hidden width of both tracks. Not used when ``ramp_depth`` is True.
    input_dim : int
        Width of the input scalars.
    output_dim : int
        Width of the output scalars.
    vector_input_dim : int
        Channel count of the input vectors.
    vector_output_dim : int
        Channel count of the output vectors.
    activation : str or None, optional
        Activation name for the scalar blocks, the gates, the :class:`Scalarizer` layers
        and the :class:`VectorActivation` layers. Default ``'gelu'``.
    seed : int, optional
        Passed to ``torch.manual_seed`` at construction. Default 0.
    dropout : float, optional
        Dropout probability in the scalar blocks and :class:`Scalarizer` layers. Default 0.
    norm : str or None, optional
        Normalization name for the scalar blocks, the gates and the :class:`Scalarizer`
        layers. Default None.
    norm_after_linear : bool, optional
        As in :class:`scalarMLP`, for the scalar track. Default True.
    bias : bool, optional
        Whether the scalar block linear layers have a bias. Default True.
    vector_norm : str or None, optional
        Normalization name for the vector blocks. Default None.
    ramp_depth : bool, optional
        As in :class:`scalarMLP`. The vector widths use base ``vector_input_dim`` with
        ``p = log(vector_output_dim) / log(input_dim)``, i.e. with the scalar input
        width in the denominator. Default False.
    v_to_s_combination : str, optional
        ``'sum'`` or ``'concatenate'``. ``'concatenate'`` doubles the scalar width entering
        each block's linear layer, which is built for the undoubled width, so that setting
        raises a shape error in :meth:`forward`; any other value raises AssertionError
        there. Default ``'sum'``.

    Notes
    -----
    Construction calls ``torch.manual_seed(seed)``, which reseeds torch's global random
    number generator.
    """

    def __init__(self,
                 layers: int,
                 filters: int,
                 input_dim: int,
                 output_dim: int,
                 vector_input_dim: int,
                 vector_output_dim: int,
                 activation: str = 'gelu',
                 seed: int = 0,
                 dropout: float = 0,
                 norm: str = None,
                 norm_after_linear: bool = True,
                 bias: bool = True,
                 vector_norm: str = None,
                 ramp_depth: bool = False,
                 v_to_s_combination: str = 'sum'):
        super(scalarMLP, self).__init__()
        # initialize constants and layers
        self.n_layers = layers
        self.output_dim = output_dim
        self.v_output_dim = vector_output_dim
        self.v_input_dim = vector_input_dim
        self.input_dim = input_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.bias = bias
        self.norm_after_linear = norm_after_linear
        self.v_norm_mode = vector_norm
        self.ramp_depth = ramp_depth
        self.v_to_s_combination = v_to_s_combination

        torch.manual_seed(seed)

        # addition of two normally distributed 3-vectors increases the norm by roughly this factor on average
        # divide this out to combat vector elongation & poor gradient flow
        self.vector_addition_rescaling_factor = 1.6

        self.init_scalar_filters(filters)
        self.init_vector_filters(filters)

        self.init_scalar_transforms()
        self.init_vector_transforms()

    def init_vector_filters(self, filters):
        """Set the per-block vector widths and residual projections; returns None.

        Assigns ``v_filters_in``, ``v_filters_out`` and ``v_same_depth``; when the widths
        are not all equal, also builds ``v_residue_adjust``, one bias-free linear layer per
        block.
        """
        if self.ramp_depth:  # smoothly ramp feature depth across layers
            # linear scaling
            # self.n_filters = torch.linspace(self.input_dim, self.output_dim, self.n_layers).long().tolist()
            # exp scaling for consistent growth ratio
            p = np.log(self.v_output_dim) / np.log(self.input_dim)
            v_n_filters = [self.v_input_dim] + [int(self.v_input_dim ** (1 + (p - 1) * (i / (self.n_layers)))) for i in
                           range(self.n_layers)]
        else:
            v_n_filters = [self.v_input_dim] + [filters for _ in range(self.n_layers)]
        self.v_filters_in = v_n_filters[:-1]
        self.v_filters_out = v_n_filters[1:]

        # if they are not all the same, we need residue adjustments
        if v_n_filters.count(v_n_filters[0]) != len(v_n_filters):
            self.v_same_depth = False
            self.v_residue_adjust = torch.nn.ModuleList([
                nn.Linear(self.v_filters_in[i], self.v_filters_out[i], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.v_same_depth = True

    def init_vector_transforms(self):
        """Build the per-block vector, gating, scalarizer and vector-activation modules and the vector output layer."""
        '''working layers'''
        self.v_fc_layers = torch.nn.ModuleList([
            nn.Linear(self.v_filters_in[i], self.v_filters_out[i], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_gating_layers = torch.nn.ModuleList([
            nn.Linear(self.s_filters_out[i], self.v_filters_out[i], bias=False)
            for i in range(self.n_layers)
        ])
        self.s_to_v_activations = torch.nn.ModuleList(
            [  # use tanh as gating function rather than standard activation which is unbound
                Activation(self.activation, self.v_filters_out[i])
                # positive outputs only to maintain equivariance (no vectors flipped)
                for i in range(self.n_layers)
            ])
        self.v_fc_norms = torch.nn.ModuleList([
            Normalization(self.v_norm_mode, self.v_filters_out[i])
            for i in range(self.n_layers)
        ])
        self.vector_to_scalar = torch.nn.ModuleList([
            Scalarizer(self.v_filters_in[i], 3, self.norm_mode, self.activation, self.dropout_p,
                       output_dim=self.s_filters_in[i])
            for i in range(self.n_layers)
        ])
        self.scalar_to_vector_norm = torch.nn.ModuleList([
            Normalization(self.norm_mode, self.v_filters_out[i])
            for i in range(self.n_layers)
        ])
        self.vector_activation = torch.nn.ModuleList([
            VectorActivation(self.v_filters_out[i], self.activation)
            for i in range(self.n_layers)
        ])

        '''output layer'''
        if self.v_output_dim != self.v_filters_out[-1]:
            self.v_output_layer = nn.Linear(self.v_filters_out[-1], self.v_output_dim, bias=False)
        else:
            self.v_output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                v: torch.Tensor,
                return_latent: bool = False,
                batch: Optional[torch.LongTensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply the MLP to scalars and vectors.

        Prints a message when ``x`` or ``v`` contains NaN, before each block and after the
        last one; the values are not changed.

        Parameters
        ----------
        x : torch.Tensor
            Input scalars, [num_samples, input_dim].
        v : torch.Tensor
            Input vectors, [num_samples, 3, vector_input_dim].
        return_latent : bool, optional
            Also return the last hidden scalar activation, before the output layer.
        batch : torch.LongTensor, optional
            Graph index of each row, forwarded to the scalar and vector block
            normalizations (not to the gate normalizations or the :class:`Scalarizer`
            layers).

        Returns
        -------
        tuple of torch.Tensor
            ``(x_out, v_out)`` with ``x_out`` [num_samples, output_dim] and ``v_out``
            [num_samples, 3, vector_output_dim]; with ``return_latent``,
            ``(x_out, v_out, latent)``.
        """

        for i, (s_norm, s_linear, s_act, s_dropout,
                v_norm, v_linear, v_act,
                s2v_norm, s2v_linear, s2v_act,
                v2s_linear) in enumerate(
            zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts,
                self.v_fc_norms, self.v_fc_layers, self.vector_activation,
                self.scalar_to_vector_norm, self.s_to_v_gating_layers, self.s_to_v_activations,
                self.vector_to_scalar)):
            res_x, res_v = self.get_residues(i, x, v)

            if torch.sum(torch.isnan(x)) != 0:
                print("NaN values in EMLP scalars")

            if torch.sum(torch.isnan(v)) != 0:
                print("NaN values in EMLP vectors")

            'scalar forward'
            if self.v_to_s_combination == 'concatenate':
                # concatenate vector lengths to scalar values
                x = torch.cat([x, v2s_linear(v)], dim=-1)
            elif self.v_to_s_combination == 'sum':
                x = x + v2s_linear(v)
            else:
                assert False, f'{self.v_to_s_combination} not implemented'

            if self.norm_after_linear:
                x = res_x + s_dropout(s_act(s_norm(s_linear(x), batch=batch)))
            else:
                x = res_x + s_dropout(s_act(s_linear(s_norm(x, batch=batch))))

            'vector forward'  # A(FC(x)) * FC(N(v))   # rescaling factor keeps norm from exploding
            s2v_gating = s2v_act(s2v_norm(s2v_linear(x))[:, None, :])
            v = v_act(v_norm(v_linear(v), batch=batch))
            v = (res_v + s2v_gating * v) / self.vector_addition_rescaling_factor

        if torch.sum(torch.isnan(x)) != 0:
            print("NaN values in EMLP scalars")
        if torch.sum(torch.isnan(v)) != 0:
            print("NaN values in EMLP vectors")

        if return_latent:
            return self.output_layer(x), self.v_output_layer(v), x
        else:
            return self.output_layer(x), self.v_output_layer(v)

    def get_residues(self,
                     i: int,
                     x: torch.Tensor,
                     v: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the block-``i`` residuals of ``x`` and ``v``: clones when the widths do not change, else the bias-free residual projections."""
        if self.same_depth:
            x = x.clone()
        else:
            x = self.residue_adjust[i](x)

        if self.v_same_depth:
            v = v.clone()
        else:
            v = self.v_residue_adjust[i](v)

        return x, v


class EmbeddingBlock(torch.nn.Module):
    """Initial node embedding: a learned embedding of the integer type in column 0,
    concatenated with the remaining columns and passed through a linear layer.

    Parameters
    ----------
    init_node_embedding_dim : int
        Output width.
    num_input_classes : int
        The embedding table has ``num_input_classes + 1`` rows, so type indices
        ``0 .. num_input_classes`` are valid.
    num_scalar_input_features : int
        Total input width, including column 0. The linear layer takes
        ``atom_type_embedding_dim + num_scalar_input_features - 1`` inputs.
    atom_type_embedding_dim : int
        Width of the type embedding.
    """

    def __init__(self,
                 init_node_embedding_dim: int,
                 num_input_classes: int,
                 num_scalar_input_features: int,
                 atom_type_embedding_dim: int):
        super(EmbeddingBlock, self).__init__()

        self.embeddings = nn.Embedding(num_input_classes + 1, atom_type_embedding_dim)
        self.linear = nn.Linear(atom_type_embedding_dim + num_scalar_input_features - 1, init_node_embedding_dim)

    def forward(self,
                x: Tensor) -> Tensor:
        """Embed ``x`` [num_nodes, num_scalar_input_features] (or [num_nodes]) to [num_nodes, init_node_embedding_dim]; column 0 is cast to long as the type index."""

        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        # always embed the first dimension only (by convention, atomic number)
        embedding = self.embeddings(x[:, 0].long())

        return self.linear(torch.cat([embedding, x[:, 1:]], dim=-1))


class ResidualMLP(nn.Module):
    """
    a simpler general-purpose MLP

    Computes ``x = relu(input_layer(x))``, then ``num_layers`` residual blocks
    ``x = x + dropout(relu(norm(linear(x))))``, then a linear output layer. All linear
    layers have biases.

    Parameters
    ----------
    input_dim : int
        Width of the input features.
    hidden_dim : int
        Width of every hidden layer.
    output_dim : int
        Width of the output.
    num_layers : int
        Number of residual blocks; 0 is allowed.
    norm : object or None
        If not None, each block uses ``nn.LayerNorm(hidden_dim)`` (the value itself is
        not inspected); if None, the identity.
    dropout_p : float
        Dropout probability in each block.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 norm, dropout_p):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        if norm is not None:
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.norms = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_p) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Map ``x`` [..., input_dim] to [..., output_dim]."""
        x = F.relu(self.input_layer(x))
        for layer, dropout, norm in zip(self.hidden_layers, self.dropouts, self.norms):
            residual = x
            x = dropout(F.relu(norm(layer(x))))
            x = x + residual  # skip connection
        return self.output_layer(x)
