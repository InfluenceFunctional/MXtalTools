import os

import numpy as np
import torch
from tqdm import tqdm
import ccdc.io as io

'''
test module for crystal builder

(1) read crystal from CIF
(2) parameterize and generate crystal object
(3) explicitly rebuild the crystal
(4) confirm we have the same crystals

(1) build random crystal in every space group
(2) parameterize
(3) rebuild
(4) confirm correct reconstruction
'''

def test_cif_roundtrip():
    cifs_path = r'D\crystal_datasets\CSD_dump'
    cifs = os.listdir(cifs_path)[:100]
    reader = io.CrystalReader(cif_path, format='cif')
