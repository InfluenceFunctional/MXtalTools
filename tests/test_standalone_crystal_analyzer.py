import numpy as np
import torch

from mxtaltools.standalone.crystal_analyzer.crystal_analyzer import CrystalAnalyzer  # todo rewrite this test


class test_crystal_analyzer:
    def __init__(self, device):
        self.device = device
        self.analyzer = CrystalAnalyzer(device)

        """specify urea"""
        self.atom_coords = torch.tensor([
            [-1.3042, - 0.0008, 0.0001],
            [0.6903, - 1.1479, 0.0001],
            [0.6888, 1.1489, 0.0001],
            [- 0.0749, - 0.0001, - 0.0003],
        ], dtype=torch.float32, device=self.device)
        self.atom_coords -= self.atom_coords.mean(0)
        self.atom_types = torch.tensor([8, 7, 7, 6], dtype=torch.long, device=self.device)

        ''' # look at molecule
        from ase import Atoms
        from ase.visualize import view
        
        mol = Atoms(positions=self.atom_coords.numpy(), numbers=self.atom_types.numpy())
        view(mol)
        '''

    def score(self):
        states = torch.tensor(np.random.uniform(0, 1, size=(10, 12)), dtype=torch.float32, device=self.device)
        states = torch.cat([
            torch.tensor(np.random.randint(1, 21, size=(10, 1)), dtype=torch.float32, device=self.device),
            states],
            dim=1)

        states[:, 1:4] *= 20
        states[:, 4:7] += torch.pi / 2

        score = self.analyzer([self.atom_coords for _ in range(len(states))],
                              [self.atom_types for _ in range(len(states))],
                              states[:, 1:],
                              states[:, 0],
                              score_type='heuristic')

        aa = 1
