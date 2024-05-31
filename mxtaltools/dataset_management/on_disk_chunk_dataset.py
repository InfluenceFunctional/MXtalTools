from torch_geometric.data import Dataset
import lmdb
import pickle
import numpy as np

from mxtaltools.dataset_management.CrystalData import CrystalData


class GeomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.path_to_datafiles = root
        self.env = lmdb.open(
            root,
            readonly=True,
        )
        self.txn = self.env.begin()
        self.data_keys = list(self.txn.cursor().iternext(values=False))
        self.samplewise_index, self.chunk_tail_index = self.build_key_indices(np.load(root.split('.lmdb')[0] + '_keys.npy', allow_pickle=True).item())
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return ['null.txt']

    def build_key_indices(self, keys_dict):
        """
        Build helpful indices for index->key conversion

        Parameters
        ----------
        keys_dict : dictionary
            Nested dict [chunk_index][smiles_index] = conformers per smiles

        Returns
        -------

        """
        samplewise_index = [np.concatenate([np.zeros(1), np.cumsum([chunk[s_ind] for s_ind in chunk.keys()])]).astype(int)
                            for chunk in keys_dict.values()]
        chunk_tail_index = np.concatenate([np.zeros(1), np.cumsum([s[-1] for s in samplewise_index])]).astype(int)

        return samplewise_index, chunk_tail_index

    def build_data_key(self, idx):
        """
        Use chunkwise data about the dataset to convert from index to usable database keys
        Parameters
        ----------
        idx : int
            desired sample index

        Returns
        -------
        string which is the key to the relevant database sample

        """
        chunk_ind = np.digitize(idx, self.chunk_tail_index) - 1
        index_in_chunk = idx - self.chunk_tail_index[chunk_ind]
        sample_index = np.digitize(index_in_chunk, self.samplewise_index[chunk_ind]) - 1
        index_in_sample = index_in_chunk - self.samplewise_index[chunk_ind][sample_index]

        sample_key = f'{chunk_ind}_{sample_index}_{index_in_sample}'
        return sample_key.encode('ascii')

    # def process(self):
    #     pass

    def len(self):
        return self.chunk_tail_index[-1]

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
        return CrystalData.from_dict(pickle.loads(self.txn.get(self.build_data_key(idx))))


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader

    dataset = GeomDataset(root=r'D:\crystal_datasets\drugs_crude.msgpack\train.lmdb')

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    data = next(iter(dataloader))

    aa = 1
