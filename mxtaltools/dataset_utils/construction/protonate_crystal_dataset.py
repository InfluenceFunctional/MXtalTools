import torch

from mxtaltools.crystal_building.utils import protonate_mol
from mxtaltools.dataset_utils.utils import collate_data_list
import os

if __name__ == "__main__":
    data_path = r"D:\crystal_datasets\CSD_dump"
    path = [elem for elem in os.listdir(data_path) if "NICOAM" in elem]


    aa = 1
