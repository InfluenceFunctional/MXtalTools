import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
#import inflect


############ Dataset and function to generate artificiall gaussians #################################
def make_art_gaussian(n_gaussians=3, n_samples=1000):
    radius = 2.5
    angles = np.linspace(0, 2 * np.pi, n_gaussians, endpoint=False)

    cov = np.array([[.1, 0], [0, .1]])
    results = []

    for angle in angles:
        results.append(
            np.random.multivariate_normal(radius * np.array([np.cos(angle), np.sin(angle)]), cov,
                                          int(n_samples / 3) + 1))

    return np.random.permutation(np.concatenate(results))


class FlowDataset(Dataset):
    def __init__(self, dataset_type, num_samples=1000, seed=0, **kwargs):
        """
        Dataset used to load different artificial datasets to train normalizing flows on.

        Args:
        dataset_type (str): Choose type from: MultiVariateNormal, Moons, Circles or MultipleGaussians
        num_samples (int): Number of samples to draw.
        seed (int): Random seed.
        kwargs: Specific parameters for the different distributions.
        """
        np.random.seed(seed)
        if dataset_type == 'MultiVariateNormal':
            mean = kwargs.pop('mean', [0, 3])
            cov = kwargs.pop('mean', np.diag([.1, .1]))
            self.data = np.random.multivariate_normal(mean, cov, num_samples)
        elif dataset_type == 'Moons':
            noise = kwargs.pop('noise', .1)
            self.data = make_moons(num_samples, noise=noise, random_state=seed, shuffle=True)[0]
        elif dataset_type == 'Circles':
            factor = kwargs.pop('factor', .5)
            noise = kwargs.pop('noise', .05)
            self.data = make_circles(num_samples, noise=noise, factor=factor, random_state=seed, shuffle=True)[0]
        elif dataset_type == 'MultipleGaussians':
            num_gaussians = kwargs.pop('num_gaussians', 3)
            self.data = make_art_gaussian(num_gaussians, num_samples)
        else:
            raise NotImplementedError

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).type(torch.FloatTensor)


########################## Plotting functions ################################
def plot_density(model, true_dist=None, num_samples=100, mesh_size=4.):
    x_mesh, y_mesh = np.meshgrid(np.linspace(- mesh_size, mesh_size, num=num_samples),
                                 np.linspace(- mesh_size, mesh_size, num=num_samples))

    cords = np.stack((x_mesh, y_mesh), axis=2)
    cords_reshape = cords.reshape([-1, 2])
    log_prob = np.zeros((num_samples ** 2))

    for i in range(0, num_samples ** 2, num_samples):
        data = torch.from_numpy(cords_reshape[i:i + num_samples, :]).float()
        log_prob[i:i + num_samples] = model.log_probability(data).cpu().detach().numpy()

    plt.scatter(cords_reshape[:, 0], cords_reshape[:, 1], c=np.exp(log_prob))
    plt.colorbar()
    if true_dist is not None:
        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', alpha=.05)
    plt.show()
#
#
# def plot_each_step(model, num_samples=200):
#     data = model.sample_each_step(num_samples)
#     len_data = len(data)
#
#     fig, axis = plt.subplots(2, int((len_data+1)/2), figsize=(15, 10),
#                              sharex=True, sharey=True)
#     p = inflect.engine()
#
#     num_plot = 0
#     for i in range(len_data):
#         if i == round((len_data+1)/2):
#             axis.flatten()[num_plot].axis('off')
#             num_plot += 1
#
#         d = data[i]
#         ax = axis.flatten()[num_plot]
#         if i == 0:
#             title = 'Original data'
#         else:
#             title = p.ordinal(i) + ' layer'
#
#         ax.scatter(d[:, 0], d[:, 1], alpha=.2)
#         ax.set_title(title)
#         num_plot += 1


###########################
def generate_image_mask(in_channels, image_width, num_layers):
    count = 0
    vec = []
    for i in range(image_width**2*in_channels):
        count += 1
        if i % image_width == 0 and image_width % 2 == 0:
            count += 1
        vec.append(count % 2.)
    mask = torch.tensor(vec).reshape(in_channels, image_width, image_width)
    masks = []
    for i in range(num_layers):
        if i % 2 == 0:
            masks.append(mask)
        else:
            masks.append(1. - mask)
    return masks