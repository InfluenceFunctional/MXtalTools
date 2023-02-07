import numpy as np
import matplotlib.pyplot as plt
from utils import compute_principal_axes_np
from nikos.rotations import rodrigues_rotation

mol = np.random.uniform(0,1,size=(20,3))
mol[:,0] = 0
magnitude= np.arange(len(mol))
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(projection='3d')
ax.scatter(mol[:,0], mol[:,1], mol[:,2],c=magnitude)
plt.show()

# get underlying axis
cell_vectors = np.random.randn(3,3) # underlying axis system
normed_cell_vectors = np.asarray([cell_vectors[i] / np.linalg.norm(cell_vectors[i]) for i in range(3)])

# get molecule axis
CoM = mol.T @ magnitude / sum(magnitude)
Ip, _, _ = compute_principal_axes_np(magnitude, mol) # arrive pre-normalized
mol -= CoM

# get rotations
rot_angles = np.random.uniform(-180,180,size=3)

rots = []
for i in range(3):
    rots.append(rodrigues_rotation(normed_cell_vectors[i], mol, rot_angles[i]))
    fig = plt.figure(i + 2)
    plt.clf()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rots[i][:, 0], rots[i][:, 1], rots[i][:, 2], c=magnitude)
    plt.show()