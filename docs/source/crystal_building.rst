Crystal Building
================

Molecular crystals are featurized in the following way.

- Conformation (atom types and coordinates) of the 'canonical conformer'(s), that is, the molecule(s) inside the asymmetric unit.
- Unit cell parameters: vector lengths, a, b, c, and internal angles, alpha, beta, gamma.
- Molecule pose parameters: The fractional coordinates of the canonical conformer(s) centroid(s), and their rotational orientation with respect to a standardized orientation.
- Molecule 'handedness': a secondary parameter defining the chirality of the canonical conformer(s) canonical principal axes

This featurization is effective for all Z'>=1 crystal structures, except in some cases of highly symmetric molecules, where the rotational orientation is not uniquely defined, where there may be inconsistencies.


Construction of a molecular crystal is done via a fast and mostly parallel algorithm, so as to be usable on-the-fly.
For Z'>1 structures, this procedure is done Z' times (once for each molecule in the asymmetric unit) in parallel, and the relevant structures are combined after the unit cells are constructed.

Note: periodicity throughout is considered in terms of molecules' centre of geometry.
If a molecule's centre of geometry is inside a given unit cell or asymmetric unit, all of its atoms are placed around it, regardless of whether they fall inside or outside the cell.

1. Pose the canonical conformer in the asymmetric unit.
2. Pattern the asymmetric unit to a single unit cell.
3. Pattern the unit cell to an NxNxN supercell.
4. Pare away molecules which are too far away from the canonical conformer to fall within the range of a single graph convolution.

Indexes aux_index and mol_index are incorporated in the resulting crystal data object to identify whether each atom is inside or outside the canonical conformer, and to which of the Z' molecules it belongs, respectively.