Dataset Creation
================
1. This software generates training datasets of molecular crystal structures from collections of .cif files.
.cifs are collated and processed primarily with the CSD Python API and RDKit.
Collation includes filtering of structures which are somehow invalid.
Invalid conditions include: symmetry features disagree, no atoms in the crystal, RDKit rejects the structure outright.
The Cambridge Structural Database (CSD) can be processed by first dumping it to .cif files, or directly with minor modifications.
Customized functions are available for processing CSD Blind Test submissions, as well as general collections of .xyz, and .cif files.
In general, the current models require only atomic numbers & coordinates to work, so a very simple featurization is sufficient.

2. In the most common case, processing the CSD, to generate a dataset, run the following scripts,
`dump_csd.py` --> `cif_processor.py` --> `manager.py`, with the appropriate paths set in each script.
`cif_processor.py` takes on the order of dozens of hours to process the full CSD (>1M crystals).
`manager.py` also may take a few minutes to process a large dataset, as this is where we do pose analysis,
duplicates search, and some indexing tasks.
We recommend running several instances in parallel to reduce this time.
As they process datasets chunkwise in random order, this parallelism is fairly efficient.
Note that the speed here depends strongly on disk read-write speed.

3. The list of features computed for each atom/molecule/crystal are listed in TODO generate list.
If you would like to use a different feature set, you may edit `dataset_management/featurization_utils.py`,
or, if your features are fast to compute, add them at runtime in `dataset_management/modelling_utils.py`.

4. Subsets of features, can be selected in the dataset config used for a given run in `configs/dataset`.
Likewise, filters can be applied to the dataset to limit the crystals included.
Such filters may commonly include: molecule size, crystal setting, atomic number.

5. We also have functions for identifying duplicate molecules and polymorphs of the same molecule.
When filtering these, we identify all the duplicates and pick a single 'representative' sample at random.

