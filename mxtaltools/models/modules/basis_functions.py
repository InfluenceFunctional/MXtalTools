"""Radial basis expansions of distances: a sine basis with a polynomial envelope
(:class:`BesselBasisLayer`, :class:`Envelope`) and Gaussian smearing
(:class:`GaussianEmbedding`)."""

from math import pi as PI

import torch


class BesselBasisLayer(torch.nn.Module):  # NOTE borrowed from DimeNet implementation
    """Radial basis ``envelope(x) * sin(freq * x)`` of the scaled distance
    ``x = dist / cutoff``.

    ``freq`` is a learnable parameter of shape [num_radial], initialised to
    ``[1, 2, ..., num_radial] * pi``.

    Parameters
    ----------
    num_radial : int
        Number of basis functions.
    cutoff : float or torch.Tensor, optional
        Distance scale, stored as the buffer ``cutoff`` (a tensor argument is cloned
        and detached). Default 5.0.
    envelope_exponent : int, optional
        Exponent passed to :class:`Envelope`. Default 5.
    """

    def __init__(self,
                 num_radial: int,
                 cutoff: float = 5.0,
                 envelope_exponent: int = 5):
        super(BesselBasisLayer, self).__init__()
        self.register_buffer('cutoff', torch.tensor(cutoff) if not torch.is_tensor(cutoff) else cutoff.clone().detach())
        self.envelope = Envelope(envelope_exponent)
        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        """Set ``freq`` to ``[1, 2, ..., num_radial] * pi`` and mark it as requiring
        gradients."""
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self,
                dist: torch.Tensor
                ) -> torch.Tensor:
        """Expand distances in the basis.

        Parameters
        ----------
        dist : torch.Tensor
            Distances in the units of ``cutoff``, any shape [...].

        Returns
        -------
        torch.Tensor
            [..., num_radial].

        Notes
        -----
        Envelope values are set to zero where the scaled distance ``dist / cutoff``
        exceeds ``cutoff`` (not 1), i.e. where ``dist > cutoff ** 2``. A distance of
        exactly zero gives NaN (the envelope's ``1 / x`` term is inf and the sine is 0).
        """
        dist = dist.unsqueeze(-1) / self.cutoff
        envelope = self.envelope(dist)
        envelope[dist > self.cutoff] = 0 # this function explodes beyond the cutoff
        return envelope * (self.freq * dist).sin()
        # return self.envelope(dist) * (self.freq * dist).sin()


class Envelope(torch.nn.Module):
    """Polynomial envelope ``1/x + a x^(p-1) + b x^p + c x^(p+1)``.

    With ``p = exponent + 1``, the coefficients ``a = -(p + 1)(p + 2)/2``,
    ``b = p(p + 2)`` and ``c = -p(p + 1)/2`` make the envelope zero at ``x = 1``. No
    cutoff mask is applied: values for ``x > 1`` are returned as computed.

    Parameters
    ----------
    exponent : float
        Envelope exponent; ``p = exponent + 1``.
    """

    def __init__(self,
                 exponent: float):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the envelope elementwise at ``x``; returns a tensor of the same
        shape (inf where ``x`` is 0, for a non-negative ``exponent``)."""
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class GaussianEmbedding(torch.nn.Module):
    """Gaussian smearing of distances on evenly spaced centres.

    Centres are ``torch.linspace(start, stop, num_gaussians)`` (buffer ``offset``).
    Each Gaussian is ``exp(coeff * (dist - centre) ** 2)`` with
    ``coeff = -0.5 / spacing ** 2``, where ``spacing`` is the gap between adjacent
    centres; ``coeff`` is stored as a float32 buffer of shape [1].

    Parameters
    ----------
    start : float, optional
        First centre. Default 0.0.
    stop : float, optional
        Last centre. Default 5.0.
    num_gaussians : int, optional
        Number of centres. Must be at least 2, since the spacing is read from the
        first two centres. Default 50.
    """

    def __init__(self,
                 start: float = 0.0,
                 stop: float = 5.0,
                 num_gaussians: int = 50):
        super(GaussianEmbedding, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

        self.register_buffer('offset', offset)
        self.register_buffer('coeff', torch.tensor([coeff], dtype=torch.float32))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances in the Gaussian basis.

        Parameters
        ----------
        dist : torch.Tensor
            Distances in the units of ``start`` and ``stop``, any shape; flattened.

        Returns
        -------
        torch.Tensor
            [dist.numel(), num_gaussians].
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
