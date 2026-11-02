"""A graph's vectorMLP output does not depend on its batch-mates under a graph-level norm.

Mo3ENet's encoder runs vectorMLP blocks with ``norm='graph layer'`` (torch_geometric ``LayerNorm`` in graph mode),
which normalises per graph only when it is given each row's graph index. The gate norm and the Scalarizer norm were
called without it, so their statistics pooled over every node in the batch: a molecule's embedding changed with
what it was encoded beside (recorded 2026-10-05: median 0.6-0.9% in batches of 200, 6.8% encoded alone). Both now
take ``batch``.

Pinned here on random scalars and vectors for three graphs of different sizes: each graph's outputs are the same
alone and in the batch. The negative control runs the same batch without ``batch`` and asserts the outputs differ,
so the test cannot pass by not exercising a graph-level norm. CPU only, under a second.
"""
import pytest
import torch

from mxtaltools.models.modules.components import vectorMLP

SIZES = (5, 9, 14)


def _model(norm, vector_norm):
    model = vectorMLP(layers=3, filters=32, input_dim=16, output_dim=8,
                      vector_input_dim=16, vector_output_dim=8,
                      norm=norm, vector_norm=vector_norm, seed=0)
    return model.eval()


def _graphs():
    gen = torch.Generator().manual_seed(0)
    xs = [torch.randn(n, 16, generator=gen) for n in SIZES]
    vs = [torch.randn(n, 3, 16, generator=gen) for n in SIZES]
    return xs, vs


@pytest.mark.parametrize('norm, vector_norm', [('graph layer', None), ('graph layer', 'graph vector layer')])
def test_vector_mlp_output_is_batch_independent(norm, vector_norm):
    model = _model(norm, vector_norm)
    xs, vs = _graphs()
    batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(SIZES)])
    with torch.no_grad():
        x_all, v_all = model(torch.cat(xs), torch.cat(vs), batch=batch)
        start = 0
        for x, v, n in zip(xs, vs, SIZES):
            x_one, v_one = model(x, v, batch=torch.zeros(n, dtype=torch.long))
            torch.testing.assert_close(x_all[start:start + n], x_one, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(v_all[start:start + n], v_one, rtol=1e-5, atol=1e-5)
            start += n


def test_negative_control_without_batch_pools_over_the_batch():
    model = _model('graph layer', None)
    xs, vs = _graphs()
    with torch.no_grad():
        x_all, _ = model(torch.cat(xs), torch.cat(vs))
        x_one, _ = model(xs[0], vs[0])
    assert not torch.allclose(x_all[:SIZES[0]], x_one, rtol=1e-3, atol=1e-3)
