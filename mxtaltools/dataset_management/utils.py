import numpy as np

import torch

from torch import Tensor
from torch_geometric.loader.dataloader import Collater


def filter_graph_nodewise(data, keep_bools):
    """
    Function to get subgraph of data. Effectively filtering by nodes.
    Args:
        data: pyg data batch
        keep_bools: boolean or indexes of which nodes should be kept

    Returns:

    """
    if data.edge_index is None:
        data.edge_index = torch.arange(2)  # necessary dummy
    return data.subgraph(keep_bools)


def filter_batch_graphwise(data, keep_index=None, delete_index=None):
    """
    Function to remove entire graphs from a pyg data batch.
    Args:
        data: pyg data batch
        keep_index: indices of graphs to be kept
        delete_index: indices of graphs to be deleted - exclusive with `keep_index`

    Returns: pyg data batch

    """
    assert keep_index is not None or delete_index is not None
    collater = Collater(None, None)
    if keep_index is not None:
        assert delete_index is None
        return collater(data[keep_index])
    elif delete_index is not None:
        assert keep_index is None
        keep_index = [ind for ind in range(len(data)) if ind not in delete_index]
        return collater(data[keep_index])


def basic_stats(values: torch.tensor) -> dict[str, Tensor]:
    clipped_values = values.clip(min=torch.quantile(values, 0.05), max=torch.quantile(values, 0.95))

    return {'max': torch.amax(values),
            'min': torch.amin(values),
            'mean': torch.mean(values),
            'std': torch.std(values),
            'tight_mean': torch.mean(clipped_values),
            'tight_std': torch.std(clipped_values),
            'histogram': torch.histogram(values, bins=50)
            }


#
# def df2coords(entry):
#     return entry.reshape(len(entry) // 3, 3)
#

def get_range_fraction(atomic_numbers, atomic_number_range: [int, int]):
    """get the fraction of atomic nubmers within the given range"""
    assert len(atomic_number_range) == 2, "atomic_number_range must be in format [low, high]"  # low-to-high
    return np.sum((np.asarray(atomic_numbers) > atomic_number_range[0]) * (
            np.asarray(atomic_numbers) < atomic_number_range[1])) / len(atomic_numbers)


def get_fraction(atomic_numbers, target: int):
    """get fraction of atomic numbers equal to target"""
    return np.sum(atomic_numbers == target) / len(atomic_numbers)


#
# def delete_from_dataframe(df, inds):
#     """
#     delete rows "inds" from dataframe df
#     """
#     if isinstance(df, pd.DataFrame):
#         return delete_pandas_dataframe_rows(df, inds)
#     elif isinstance(df, pl.DataFrame):
#         return delete_polars_dataframe_rows(df, inds)
#     else:
#         assert False, "df must be either a dataframe or a polars dataframe"

#
# def delete_pandas_dataframe_rows(df: pd.DataFrame, inds):
#     df.drop(index=inds, inplace=True)
#     df.reset_index(drop=True, inplace=True)
#
#     return df
# #
#
# def delete_polars_dataframe_rows(df: pl.DataFrame, inds):
#     good_inds = [ind for ind in range(len(df)) if ind not in inds]
#     df = df[good_inds]
#
#     return df
#
#
# def concatenate_dataframe_column(df, column_name):
#     values = df[column_name]
#     series_type = str(values.dtype)
#
#     if (series_type != 'object'
#             or isinstance(values[0], str)
#             or isinstance(values[0], dict)):  # not a list of lists
#         cat_vals = np.asarray(values)
#     elif isinstance(values[0][0], list) or isinstance(values[0][0], np.ndarray):  # list of list of lists
#         cat_vals = np.concatenate([sublist for lists in values for sublist in lists])
#     else:  # list of lists or list of arrays
#         cat_vals = np.concatenate(values)
#     return cat_vals

#
# def flatten_dataframe(dataset1):
#     """
#     Flatten all elements of dataset1
#     Speeds up I/O and makes compatible with new processing functions + polars integration
#     Args:
#         dataset1:
#
#     Returns:
#
#     """
#     dataset2 = dataset1.copy()
#     for column in dataset1.columns:
#         try:
#             vals = [np.concatenate(entry).flatten() for entry in dataset1[column]]
#         except ValueError:
#             vals = [np.array(entry).flatten() for entry in dataset1[column]]
#         except TypeError:
#             vals = [np.array(entry).flatten() for entry in dataset1[column]]
#
#         dataset2[column] = vals
#         del dataset1[column]  # for RAM purposed
#
#     return dataset2


''' test

df = pd.DataFrame(columns=['list', 'list_of_lists', 'list_of_arrays', 'array'])
df['list'] = [_ for _ in range(1, 10)]
df['list_of_lists'] = [[_ for _ in range(thing)] for thing in range(1, 10)]
df['list_of_arrays'] = [np.asarray([_ for _ in range(thing)]) for thing in range(1, 10)]
df['array'] = np.arange(1, 10)
df['list_of_length_one_lists'] = [[thing] for thing in range(1, 10)]
df['list_of_list_of_lists'] = [[[_ for _ in range(subthing)] for subthing in range(thing % 2 + 2)] for thing in range(1, 10)]

for column in df.columns:
    values = df[column]
    series_type = str(values.dtype)

    print(f"{column}:{series_type}")

    if series_type != 'object':  # not a list of lists
        cat_vals = np.asarray(values)
    elif isinstance(values[0][0], list):  # list of list of lists
        cat_vals = np.concatenate([sublist for lists in values for sublist in lists])
    else:  # list of lists or list of arrays
        cat_vals = np.concatenate(values)
    print(cat_vals)
    '''
