Dataset Creation
================

.. warning::
    This module has not been recently updated and may be out-of-date in some ways

This software generates training datasets of molecules/molecular crystal/molecular cluster structures from collections of .xyz or .cif files, respectively.
Structures collated and processed using the CSD Python API (for crystals only), RDKit, and custom functions.
Small datasets are may be saved and loaded into RAM, whereas large ones are loaded in batches from on-disk lmdb databases.
Note dataset mixing and statistical analysis is not yet automated for such on-disk datasets.

Collation includes filtering of structures which are somehow invalid.
Invalid conditions include: symmetry features disagree (i.e., wrong number of molecules in the crystal), no atoms in the structure, RDKit rejects the structure outright, and so on.

The full Cambridge Structural Database (CSD) can be processed by first dumping it to .cif files, which are then processed sequentially, or directly from the database with minor modifications to the existing scripts.

Customized functions are available for processing CSD Blind Test submissions, as well as general collections of .xyz, and .cif files. The current models require only atomic numbers & coordinates to work, so a very simple featurization is sufficient.

Crystals Datasets from the CSD
------------------------------

To generate a dataset from the CSD, run the following scripts,

- `dump_csd.py`

- `process_cifs_to_dataset.py`

- `collate_and_generate_dataset.py`

with the appropriate paths set within each script.

`process_cifs_to_dataset.py` takes on the order of dozens of hours to process the full CSD (>1M crystals). We recommend running several instances in parallel to reduce this time. As they process datasets chunkwise in random order, this parallelism is quite efficient.

We also have functions for identifying duplicate molecules and polymorphs of the same molecule. When filtering these, we identify all the duplicates and pick a single 'representative' sample at random. Options for duplicate and other types of filtering are set in the dataset configs stored in `configs/dataset`


Molecule Datasets from .xyzs
----------------------------

Generating a molecular dataset from sets of .xyz files is very similar to above. 
Simply run the following scripts,

- `process_xyz_to_daataset.py`

- `collate_and_generate_dataset.py`

with the appropriate paths set within each script.


Molecule Datasets from other sources
------------------------------------

Other datasets, such as GEOM Drugs, may be processed analogously to the .xyzs method, see for example:

- `process_GEOM.py`

The key difference in processing such a new dataset is simply in reading the relevant file to be processed.
In the case of GEOM, one must install msgpack to serially walk through the dataset.


