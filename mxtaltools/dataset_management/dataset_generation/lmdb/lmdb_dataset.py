from torch_geometric.data import Dataset
import lmdb
import pickle
import numpy as np
from random import randint

from mxtaltools.dataset_management.CrystalData import CrystalData


class lmdbDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.path_to_datafiles = root
        self.txn = None
        # self.data_keys = list(self.txn.cursor().iternext(values=False)) # slow and unused
        self.keys_path = root.split('.lmdb')[0] + '_keys.npy'
        self.dataset_length = int(np.load(self.keys_path, allow_pickle=True).item())
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
        # if an index returns a none sample, run align below to reindex the dataset
        return CrystalData.from_dict(pickle.loads(self.txn.get(str(idx).encode('ascii'))))

    def _init_env(self):
        env = lmdb.open(
            self.path_to_datafiles,
            readonly=True,
            max_readers=128,
            readahead=False,
            meminit=False,
        )
        self.txn = env.begin()

    def realign(self):
        """
        set keys so that entries are sequentially numerical with no gaps
        may be expensive for large datasets

        CAUTION overwrites entire dataset
        Returns
        -------

        """
        counter = 1

        with lmdb.open(self.path_to_datafiles,
                       map_size=int(10e9)) as db:
            with db.begin(write=True) as txn:
                for idx in tqdm(range(1, self.dataset_length)):
                    item = txn.get(str(idx).encode('ascii'))
                    if item is not None:
                        txn.put(str(counter).encode('ascii'), item)
                        counter += 1
                    else:
                        pass

        # update dataset length via counter
        np.save(self.keys_path, counter - 1)


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    #dataset = lmdbDataset(root=r'D:\crystal_datasets\drugs_crude.msgpack\train.lmdb')
    dataset = lmdbDataset(root=r'D:\crystal_datasets\zinc22/zinc.lmdb')
    dataset.get(1)

    dataloader = DataLoader(dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=0,
                            )

    sample = next(iter(dataloader))

    for ind, data in enumerate(tqdm(dataloader)):
        pass

    aa = 1
