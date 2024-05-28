import os

from torch_geometric.data import DataLoader


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
        te = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False)

    return tr, te


def update_dataloader_batch_size(loader, new_batch_size):
    return DataLoader(loader.dataset,
                      batch_size=new_batch_size,
                      shuffle=True,
                      num_workers=loader.num_workers,
                      pin_memory=loader.pin_memory,
                      drop_last=loader.drop_last)
