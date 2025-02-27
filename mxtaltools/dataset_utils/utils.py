import os

import torch
from torch import Tensor
from torch_geometric.data import DataLoader, Batch
from torch_geometric.loader.dataloader import DataLoader


def collate_data_list(data_list):
    return Batch.from_data_list(data_list,
                                exclude_keys=['edges_dict']
                                )


def quick_combine_dataloaders(dataset, data_loader, batch_size, max_size):
    dataset.extend(data_loader.dataset)
    dataset = dataset[:max_size]  # truncate at expense of old data

    if 'batch' in str(type(dataset)):  # todo switch this to an explicit check for a Data Batch
        # if it's batched, revert to data list - this is slow, so if possible don't store datasets as batches but as data lists
        dataset = dataset.to_data_list()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                            drop_last=False)

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


def get_dataloaders(dataset_builder, machine, batch_size, test_fraction=0.2, shuffle=True):
    batch_size = batch_size
    train_size = int((1 - test_fraction) * len(dataset_builder))  # split data into training and test sets
    test_size = len(dataset_builder) - train_size

    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size):
        train_dataset.append(dataset_builder[i])
    for i in range(test_size):
        test_dataset.append(dataset_builder[i])

    if False: #machine == 'cluster':  # faster dataloading on cluster with more workers
        if len(train_dataset) > 0:
            tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(os.cpu_count(), 8), pin_memory=True, drop_last=False)
        else:
            tr = None
        te = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=min(os.cpu_count(), 8), pin_memory=True, drop_last=False)
    else:
        if len(train_dataset) > 0:
            tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False)
        else:
            tr = None
        te = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False)

    return tr, te


def update_dataloader_batch_size(loader, new_batch_size):
    return DataLoader(loader.dataset,
                      batch_size=new_batch_size,
                      shuffle=True,
                      num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory,
                      drop_last=loader.drop_last)
