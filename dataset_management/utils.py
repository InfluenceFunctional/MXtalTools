import numpy as np
import pandas as pd
import os


def get_range_fraction(atomic_numbers, atomic_number_range: [int, int]):
    """get the fraction of atomic nubmers within the given range"""
    assert len(atomic_number_range) == 2, "atomic_number_range must be in format [low, high]"  # low-to-high
    return np.sum((np.asarray(atomic_numbers) > atomic_number_range[0]) * (np.asarray(atomic_numbers) < atomic_number_range[1])) / len(atomic_numbers)


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
