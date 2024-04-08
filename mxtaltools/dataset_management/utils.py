import numpy as np
import pandas as pd
import os


def get_range_fraction(atomic_numbers, atomic_number_range: [int, int]):
    """get the fraction of atomic nubmers within the given range"""
    assert len(atomic_number_range) == 2, "atomic_number_range must be in format [low, high]"  # low-to-high
    return np.sum((np.asarray(atomic_numbers) > atomic_number_range[0]) * (
            np.asarray(atomic_numbers) < atomic_number_range[1])) / len(atomic_numbers)


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


def concatenate_dataframe_column(df, column_name):
    values = df[column_name]
    series_type = str(values.dtype)

    if (series_type != 'object'
            or isinstance(values[0], str)
            or isinstance(values[0], dict)):  # not a list of lists
        cat_vals = np.asarray(values)
    elif isinstance(values[0][0], list) or isinstance(values[0][0], np.ndarray):  # list of list of lists
        cat_vals = np.concatenate([sublist for lists in values for sublist in lists])
    else:  # list of lists or list of arrays
        cat_vals = np.concatenate(values)
    return cat_vals


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
