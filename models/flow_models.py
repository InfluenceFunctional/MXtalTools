import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class SimpleAffine(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.zeros(self.dim))  # log_scale
        self.b = nn.Parameter(torch.zeros(self.dim))  # shift

    def forward(self, x):
        y = torch.exp(self.a) * x + self.b

        det_jac = torch.exp(self.a.sum())
        log_det_jac = torch.ones(y.shape[0]) * torch.log(det_jac)

        return y, log_det_jac

    def inverse(self, y):
        x = (y - self.b) / torch.exp(self.a)

        det_jac = 1 / torch.exp(self.a.sum())
        inv_log_det_jac = torch.ones(y.shape[0]).to(y.device) * torch.log(det_jac)

        return x, inv_log_det_jac


class StackSimpleAffine(nn.Module):
    def __init__(self, transforms, dim=2):
        super().__init__()
        self.dim = dim
        self.transforms = nn.ModuleList(transforms)
        self.distribution = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0]).to(x.device)
        for transform in reversed(self.transforms):
            x, inv_log_det_jac = transform.inverse(x)
            log_prob += inv_log_det_jac

        log_prob += self.distribution.log_prob(x.float().cpu()).to(x.device)

        return log_prob

    def rsample(self, num_samples):
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)

        for transform in self.transforms:
            x, log_det_jac = transform.forward(x)
            log_prob += log_det_jac

        return x, log_prob


class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim))

        self.scale = nn.Parameter(torch.Tensor(self.dim))

        self.t_func = nn.Sequential(nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
                                    nn.Linear(in_features=hidden_size, out_features=self.dim))

    def forward(self, x):
        x_mask = x*self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x*torch.exp(s) + t)

        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1-self.mask)*(y - t)*torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)

        return x, inv_log_det_jac


class RealNVP(nn.Module):
    def __init__(self, masks, hidden_size):
        super(RealNVP, self).__init__()

        self.dim = len(masks[0])
        self.hidden_size = hidden_size

        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask), requires_grad=False) for mask in masks])
        self.layers = nn.ModuleList([RealNVPNode(mask, self.hidden_size) for mask in self.masks])

        self.distribution = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0])
        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
        log_prob += self.distribution.log_prob(x)

        return log_prob

    def rsample(self, num_samples):
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)

        for layer in self.layers:
            x, log_det_jac = layer.forward(x)
            log_prob += log_det_jac

        return x, log_prob

    def sample_each_step(self, num_samples):
        samples = []

        x = self.distribution.sample((num_samples,))
        samples.append(x.detach().numpy())

        for layer in self.layers:
            x, _ = layer.forward(x)
            samples.append(x.detach().numpy())

        return samples


class RealNVPNodeCNN(nn.Module):
    def __init__(self, mask, in_channels):
        super(RealNVPNodeCNN, self).__init__()

        self.mask = nn.Parameter(mask, requires_grad=False)

        cnn_channels = [32, 64, 32]

        self.s_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[1], out_channels=cnn_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[2], out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

        self.t_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[1], out_channels=cnn_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[2], out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x_mask = x*self.mask
        s = self.s_func(x_mask)
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x*torch.exp(s) + t)

        # Sum for -1, since for every batch, and 1-mask, since the log_det_jac is 1 for y1:d = x1:d.
        log_det_jac = ((1 - self.mask) * s).view(x.shape[0], -1).sum(-1)
        return y, log_det_jac

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask)
        t = self.t_func(y_mask)

        x = y_mask + (1-self.mask)*(y - t)*torch.exp(-s)

        inv_log_det_jac = ((1 - self.mask) * -s).view(y.shape[0], -1).sum(-1)

        return x, inv_log_det_jac


class RealNVPCNN(nn.Module):
    def __init__(self, masks):
        super(RealNVPCNN, self).__init__()

        self.in_channels = masks[0].size(0)
        self.image_width = masks[0].size(1)
        pixels = self.in_channels*self.image_width**2

        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask), requires_grad=False) for mask in masks])
        self.layers = nn.ModuleList([RealNVPNodeCNN(mask, self.in_channels) for mask in self.masks])

        self.distribution = MultivariateNormal(torch.zeros(pixels), torch.eye(pixels))

    def log_probability(self, x):
        log_prob = torch.zeros(x.shape[0])

        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
        log_prob += self.distribution.log_prob(x.view(x.shape[0], -1))

        return log_prob

    def rsample(self, num_samples):
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)
        x = x.view(num_samples, self.in_channels, self.image_width, self.image_width)
        for layer in self.layers:
            x, log_det_jac = layer.forward(x)
            log_prob += log_det_jac

        return x, log_prob

    def sample_each_step(self, num_samples):
        samples = []

        x = self.distribution.sample((num_samples,))
        samples.append(x.detach().numpy())

        for layer in self.layers:
            x, _ = layer.forward(x)
            samples.append(x.detach().numpy())

        return samples



