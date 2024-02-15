import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import os


def get_range_fraction(atomic_numbers, atomic_number_range: [int, int]):
    """get the fraction of atomic nubmers within the given range"""
    assert len(atomic_number_range) == 2, "atomic_number_range must be in format [low, high]"  # low-to-high
    return np.sum((np.asarray(atomic_numbers) > atomic_number_range[0]) * (np.asarray(atomic_numbers) < atomic_number_range[1])) / len(atomic_numbers)


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

    if machine == 'cluster':  # faster dataloading on cluster with more workers
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
        te = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    return tr, te


def update_dataloader_batch_size(loader, new_batch_size):
    return DataLoader(loader.dataset,
                      batch_size=new_batch_size,
                      shuffle=True,
                      num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory,
                      drop_last=loader.drop_last)


def get_fraction(atomic_numbers, target: int):
    """get fraction of atomic numbers equal to target"""
    return np.sum(atomic_numbers == target) / len(atomic_numbers)


def delete_from_dataframe(df: pd.DataFrame, inds):
    """
    delete rows "inds" from dataframe df
    """
    df.drop(index=inds, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
