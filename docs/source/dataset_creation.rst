Dataset Creation
================
This software generates training datasets of molecules/molecular crystal structures from collections of .xyz or .cif files, respectively. Structures collated and processed with the CSD Python API (for crystals only) and RDKit.

Collation includes filtering of structures which are somehow invalid. Invalid conditions include: symmetry features disagree, no atoms in the crystal, RDKit rejects the structure outright.

The Cambridge Structural Database (CSD) can be processed by first dumping it to .cif files, which are then processed sequentially, or directly from the database with minor modifications.

Customized functions are available for processing CSD Blind Test submissions, as well as general collections of .xyz, and .cif files. The current models require only atomic numbers & coordinates to work, so a very simple featurization is sufficient.

Crystals Datasets from the CSD
-----------------------------

To generate a dataset from the CSD, run the following scripts,

- `dump_csd.py`

- `process_cifs_to_dataset.py`

- `collate_and_generate_dataset.py`

with the appropriate paths set within each script.

`process_cifs_to_dataset.py` takes on the order of dozens of hours to process the full CSD (>1M crystals). We recommend running several instances in parallel to reduce this time. As they process datasets chunkwise in random order, this parallelism is quite efficient.

We also have functions for identifying duplicate molecules and polymorphs of the same molecule. When filtering these, we identify all the duplicates and pick a single 'representative' sample at random. Options for duplicate and other types of filtering are set in the dataset configs stored in `configs/dataset`
