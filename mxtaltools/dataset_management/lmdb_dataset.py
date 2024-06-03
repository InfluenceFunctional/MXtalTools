from torch_geometric.data import Dataset
import lmdb
import pickle
import numpy as np

from mxtaltools.dataset_management.CrystalData import CrystalData


class GeomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.path_to_datafiles = root
        self.txn = None
        # self.data_keys = list(self.txn.cursor().iternext(values=False)) # slow and unused
        self.data_keys = []
        self.dataset_length = int(np.load(root.split('.lmdb')[0] + '_keys.npy', allow_pickle=True).item())
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return ['null.txt']

    def len(self):
        return self.dataset_length

    def get(self, idx):
        """
        build data key, get sample from database, unpickle it, and convert it to crystal data object
        Parameters
        ----------
        idx

        Returns
        -------
        CrystalData object
        """
        if self.txn is None:
            self._init_env()
        return CrystalData.from_dict(pickle.loads(self.txn.get(str(idx).encode('ascii'))))

    def _init_env(self):
        self.env = lmdb.open(
            self.path_to_datafiles,
            readonly=True,
        )
        self.txn = self.env.begin()


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader

    dataset = GeomDataset(root=r'D:\crystal_datasets\drugs_crude.msgpack\train.lmdb')

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    data = next(iter(dataloader))

    aa = 1
