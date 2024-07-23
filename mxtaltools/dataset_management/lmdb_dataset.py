from torch_geometric.data import Dataset
import lmdb
import pickle
import numpy as np

from mxtaltools.dataset_management.CrystalData import CrystalData


class lmdbDataset(Dataset):
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
        if idx == 0:  # always missing
            idx = 1
        elif idx >= self.dataset_length:
            idx = self.dataset_length - 1

        return self.return_sample(idx)

    def return_sample(self, idx):
        """
        required since some keys map to None, which can be difficult to clean up
        """
        try:
            return CrystalData.from_dict(pickle.loads(self.txn.get(str(idx).encode('ascii'))))
        except TypeError:
            idx -= 1
            return self.return_sample(idx)

    def _init_env(self):
        env = lmdb.open(
            self.path_to_datafiles,
            readonly=True,
            max_readers=128,
            readahead=False,
            meminit=False,
        )
        self.txn = env.begin()


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    #dataset = lmdbDataset(root=r'D:\crystal_datasets\drugs_crude.msgpack\train.lmdb')
    dataset = lmdbDataset(root=r'D:\crystal_datasets\zinc22/zinc.lmdb')
    dataset.get(1)

    dataloader = DataLoader(dataset,
                            batch_size=100,
                            shuffle=True,
                            num_workers=0,
                            )

    sample = next(iter(dataloader))

    for ind, data in enumerate(tqdm(dataloader)):
        pass

    aa = 1
