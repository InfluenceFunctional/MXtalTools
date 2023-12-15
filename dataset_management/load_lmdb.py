import lmdb
import numpy as np
import os
import pickle

os.chdir(r'D:\crystal_datasets')
env = lmdb.open(
    'valid.lmdb',
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256)
txn = env.begin()
keys = list(txn.cursor().iternext(values=False))
all_atoms = []
for idx in keys:
    datapoint_pickled = txn.get(idx)
    data = pickle.loads(datapoint_pickled)
    all_atoms.extend(data['atoms'])

a, b =np.unique(all_atoms, return_counts=True)

for ia, ib in zip(a,b):
    print(f'{ia} : {ib}')