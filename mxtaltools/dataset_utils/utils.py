from random import shuffle

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Batch
from torch_geometric.loader.dataloader import DataLoader


def collate_data_list(data_list):
    if not isinstance(data_list, list):
        data_list = [data_list]
    return Batch.from_data_list(data_list,
                                exclude_keys=['edges_dict',
                                              'unit_cell_pos'],
                                )


def quick_combine_dataloaders(dataset, data_loader, batch_size, max_size):
    shuffle(data_loader.dataset)  # randomize order of old dataset
    dataset.extend(data_loader.dataset)  # append old dataset to new one
    dataset = dataset[:max_size]  # truncate from the end of the old dataset
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=data_loader.num_workers,
                            pin_memory=data_loader.pin_memory,
                            drop_last=data_loader.drop_last)

    return dataloader


def quick_combine_crystal_embedding_dataloaders(dataset, data_loader, batch_size, max_size):
    x, y = data_loader.dataset.x, data_loader.dataset.y  # randomize order of old dataset
    rands = torch.tensor(np.random.choice(len(x), len(x), replace=False), dtype=torch.long)
    x = x[rands]
    y = y[rands]

    new_x, new_y = dataset.x, dataset.y  # prepend the new dataset
    new_x = torch.cat((new_x, x), dim=0)[:max_size]
    new_y = torch.cat((new_y, y), dim=0)[:max_size]

    new_dataset = SimpleDataset(x=new_x, y=new_y)

    dataloader = DataLoader(new_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=data_loader.num_workers,
                            pin_memory=data_loader.pin_memory,
                            drop_last=data_loader.drop_last)

    return dataloader


def filter_graph_nodewise(data, keep_index=None, delete_index=None):
    """ # NOTE this does not work because of our custom data structure
    Function to get subgraph of data. Effectively filtering by nodes.
    Args:
        data: pyg data batch
        keep_index: boolean or indexes of which nodes should be kept

    Returns:

    Parameters
    ----------
    data
    keep_index
    delete_index

    """
    assert keep_index is not None or delete_index is not None
    if keep_index is None and delete_index is not None:
        keep_index = [ind for ind in range(len(data)) if ind not in delete_index]

    if data.edge_index is None:
        data.edge_index = torch.arange(2)  # necessary dummy
    return data.subgraph(keep_index)


def basic_stats(values: torch.tensor) -> dict[str, Tensor]:
    clipped_values = values.clip(min=torch.quantile(values[:int(16e6)].float(), 0.05),
                                 max=torch.quantile(values[:int(16e6)].float(), 0.95))

    return {'max': torch.amax(values),
            'min': torch.amin(values),
            'mean': torch.mean(values.float()),
            'std': torch.std(values.float()),
            'tight_mean': torch.mean(clipped_values),
            'tight_std': torch.std(clipped_values),
            'histogram': torch.histogram(values.float(), bins=50),
            'uniques': torch.unique(values, return_counts=True) if values.dtype == torch.long else (None, None),
            }


def get_dataloaders(dataset_builder, machine, batch_size, test_fraction=0.2,
                    shuffle=True,
                    num_workers: int = 0):
    batch_size = batch_size
    train_size = int((1 - test_fraction) * len(dataset_builder))  # split data into training and test sets
    test_size = len(dataset_builder) - train_size

    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size):
        train_dataset.append(dataset_builder[i])
    for i in range(test_size):
        test_dataset.append(dataset_builder[i])

    if machine == 'cluster':  # faster dataloading on cluster with more workers
        if len(train_dataset) > 0:
            tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True, drop_last=False)
        else:
            tr = None
        te = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                        drop_last=False)
    else:
        if len(train_dataset) > 0:
            tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True,
                            drop_last=False)
        else:
            tr = None
        te = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True,
                        drop_last=False)

    return tr, te


def update_dataloader_batch_size(loader, new_batch_size):
    return DataLoader(loader.dataset,
                      batch_size=new_batch_size,
                      shuffle=True,
                      num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory,
                      drop_last=loader.drop_last)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
