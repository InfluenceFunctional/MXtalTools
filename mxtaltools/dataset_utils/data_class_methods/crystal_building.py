from typing import Optional

import torch

from mxtaltools.common.geometry_utils import fractional_transform
from mxtaltools.crystal_building.utils import get_aunit_positions, aunit2ucell, ucell2cluster
from mxtaltools.dataset_utils.utils import collate_data_list


# noinspection PyAttributeOutsideInit
class MolCrystalBuilding:
    def split_to_zp1_batch(self):
        # if hasattr(self, "aux_ind"):
        #     assert self.aux_ind is None, "Not implemented for cluster objects"
        # NOTE this is only an intermediate for crystal building, and will not generate all the correct attributes for
        # subunit crystal graphs such as molecule and crystal properties
        assert self.is_batch, "Method not implemented for single data object"
        new_num_graphs = int(torch.sum(self.z_prime))
        out_graphs_per_in_graph = self.z_prime
        graph_ids = torch.arange(new_num_graphs, dtype=torch.long, device=self.device)
        # NOTE requires multiples of the identical molecule in each subunit
        rep_index = torch.arange(self.num_graphs, device=self.device).repeat_interleave(self.z_prime)
        atoms_per_subunit = (self.num_atoms // self.z_prime)[rep_index]
        new_batch = graph_ids.repeat_interleave(atoms_per_subunit)

        zp1_batch = self.clone()
        zp1_batch.num_atoms = atoms_per_subunit
        zp1_batch.batch = new_batch
        assert len(zp1_batch.pos) == len(new_batch)
        zp1_batch.ptr = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.device),
             torch.cumsum(atoms_per_subunit, dim=0)])
        zp1_batch._num_graphs = new_num_graphs
        # copy over relevant crystal properties
        zp1_batch.sg_ind = self.sg_ind[rep_index]
        zp1_batch.sym_mult = self.sym_mult[rep_index]
        zp1_batch.nonstandard_symmetry = self.nonstandard_symmetry[rep_index]
        zp1_batch.T_fc = self.T_fc[rep_index]
        zp1_batch.T_cf = self.T_cf[rep_index]
        zp1_batch.symmetry_operators = [self.symmetry_operators[ind] for ind in rep_index]
        zp1_batch.cell_lengths = self.cell_lengths[rep_index]
        zp1_batch.cell_angles = self.cell_angles[rep_index]
        zp1_batch.num_atoms = (self.num_atoms // self.z_prime)[rep_index]
        zp1_batch.radius = self.radius[rep_index]
        zp1_batch.mol_volume = self.mol_volume[rep_index]
        zp1_batch.mass = self.mass[rep_index]
        zp1_batch.z_prime = torch.ones(new_num_graphs, device=self.device, dtype=torch.long)

        if hasattr(zp1_batch,'unit_cell_pos'):
            if zp1_batch.unit_cell_pos is not None:
                atoms_per_sub_ucell = ((self.num_atoms * self.sym_mult) // self.z_prime)[rep_index]
                zp1_batch.unit_cell_batch = graph_ids.repeat_interleave(atoms_per_sub_ucell)

        # extra handling for aunit properties
        subunit_index = torch.arange(new_num_graphs, device=self.device) - torch.repeat_interleave(
            torch.cumsum(self.z_prime, 0) - self.z_prime, self.z_prime)
        # columns to index from [num_graphs, 3*Z'] format
        col_base = 3 * subunit_index.unsqueeze(1) + torch.arange(3, device=self.device)
        col_base2 = 1 * subunit_index.unsqueeze(1) + torch.arange(1, device=self.device)

        zp1_batch.aunit_centroid = self.aunit_centroid[rep_index.unsqueeze(1), col_base]
        zp1_batch.aunit_orientation = self.aunit_orientation[rep_index.unsqueeze(1), col_base]
        zp1_batch.aunit_handedness = self.aunit_handedness[rep_index.unsqueeze(1), col_base2]

        return zp1_batch

    def join_zp1_aunit_batch(self, zp1_batch):
        self.pos = zp1_batch.pos

    def join_zp1_ucell_batch(self, zp1_batch):
        atoms_per_ucell = self.num_atoms * self.sym_mult
        combined_ucell_batch = torch.arange(self.num_graphs, device=self.device).repeat_interleave(
            atoms_per_ucell
        )
        self.pos = zp1_batch.pos
        self.unit_cell_pos = zp1_batch.unit_cell_pos
        self.unit_cell_batch = combined_ucell_batch
        self.unit_cell_mol_ind = zp1_batch.unit_cell_mol_ind

    def join_zp1_cluster_batch(self, zp1_batch):
        if self.z_prime.amax() > 1:
            cluster_batch = self.clone()
            cluster_batch.pos = zp1_batch.pos
            cluster_batch.x = zp1_batch.x
            cluster_batch.z = zp1_batch.z
            cluster_batch.aux_ind = zp1_batch.aux_ind
            # reindexing molecules properly here is important for intermolecular edge construction
            atoms_per_zp1_crystal = torch.bincount(zp1_batch.batch, minlength=len(zp1_batch.ptr) - 1)
            mols_per_zp1_crystal = atoms_per_zp1_crystal // zp1_batch.num_atoms
            molwise_zp1_ptr = torch.cat([
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(mols_per_zp1_crystal[:-1], dim=0)
            ])
            mol_ind_offset = molwise_zp1_ptr.repeat_interleave(atoms_per_zp1_crystal)
            cluster_batch.mol_ind = zp1_batch.mol_ind + mol_ind_offset

            # reindex from Z'=1 to combined structures
            batch_map = torch.arange(self.num_graphs, device=self.device).repeat_interleave(self.z_prime)
            cluster_batch.batch = batch_map[zp1_batch.batch]
            atoms_per_cluster = torch.bincount(cluster_batch.batch)

            cluster_batch.ptr = torch.cat([
                torch.zeros(1, device=self.device, dtype=torch.long),
                torch.cumsum(atoms_per_cluster, dim=0)
            ])

            cluster_batch.unit_cell_pos = zp1_batch.unit_cell_pos.clone()
            cluster_batch.unit_cell_batch = batch_map[zp1_batch.unit_cell_batch].clone()

            return cluster_batch

        else:
            assert False, "No point in joining batches which area already Z'=1"

    # todo add method to just align to the standard orientation
    def pose_aunit(self, std_orientation: Optional[bool] = True,
                   override_handedness=None):
        if override_handedness is not None:
            handedness = override_handedness
        else:
            handedness = self.aunit_handedness
        if self.is_batch:
            self.pos = get_aunit_positions(
                self,
                std_orientation=std_orientation,
                mol_handedness=handedness,
            )
        else:
            self.pos = get_aunit_positions(
                collate_data_list([self]),
                std_orientation=std_orientation,
                mol_handedness=handedness,
            )

    def build_unit_cell(self):
        if self.is_batch:
            self.unit_cell_pos, self.unit_cell_batch, self.unit_cell_mol_ind = aunit2ucell(self)
        else:
            self.unit_cell_pos, self.unit_cell_batch, self.unit_cell_mol_ind = aunit2ucell(collate_data_list([self]))

    def build_cluster(self, cutoff: float = 6, supercell_size: int = 10, zp_buffer=0):
        if self.is_batch:
            return ucell2cluster(self, cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)
        else:
            crystal_batch = collate_data_list([self])
            crystal_batch.build_unit_cell()
            return ucell2cluster(crystal_batch, cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)

    def de_cluster(self):  # todo check and consider rewrite with new methods
        # delete cluster information and reset this object as a molecule
        if self.aux_ind is not None:
            aunit_bools = self.aux_ind == 0
            self.pos = self.pos[aunit_bools]
            if self.x is not None:
                self.x = self.x[aunit_bools]
            self.z = self.z[aunit_bools]
            self.batch = torch.arange(self.num_graphs, device=self.device
                                      ).repeat_interleave(self.num_atoms)
            self.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device),
                                  torch.cumsum(self.num_atoms, dim=0)]).long()

            self.aux_ind = None
            self.mol_ind = None
            self.edges_dict = None
        else:
            raise RuntimeError("can't de-cluster - this is already not a cluster")

    def mol2cluster(self, cutoff: float = 6,
                    supercell_size: int = 10,
                    std_orientation: Optional[bool] = True):
        if self.max_z_prime > 1:
            # if there are any Z'>1 crystals in the batch, we
            # unzip for unit cell generation then re zip

            # add the intra-aunit centroid distance to cutoffs
            frac_centroids = self.aunit_centroid.reshape(self.num_graphs * self.max_z_prime, 3)
            # THE ERROR, fixed 2026-08-13: this call used to end in `[1]`.
            #
            # `repeat_interleave(Zp, 0)` builds a per-centroid stack [g0, g0, g1, g1, ...] so
            # that each flattened centroid is transformed by ITS OWN crystal's cell. `[1]`
            # reduced that stack to a single (3, 3) -- always graph 0's, since element 1 of
            # [g0, g0, ...] is still g0 -- and fractional_transform dispatches (n,3)+(3,3) as
            # one shared transform. So graph 0's cell metric was applied to every crystal in
            # the batch, and the index threw away the precise job the repeat_interleave it was
            # indexing had just done. Silent: the shape stayed legal, only the values moved.
            #
            # What it feeds: `zp_buffer`, the extra supercell padding for Z'>1, added to
            # `cutoff` at TWO sites in crystal_building/utils.py (unit-cell selection, and the
            # final paring of interacting aunits). Too small truncates the neighbour list.
            #
            # COST, measured through the energy on the real sg 9 Z'=2 prior -- NOT inferred
            # from the buffer geometry, which overstates it badly (that route says 5% of
            # crystals lost more than the whole nominal cutoff, worst 20.3 A). Median
            # |d elj| was 1e-4 kJ/mol, numerically nil; 3 of 400 crystals exceeded 1 kJ/mol;
            # worst was 126 against a median |elj| of ~970, i.e. 13% of that structure's
            # lattice energy. One of four random 100-crystal batches contained no affected
            # crystal at all. So: RARE, and severe where it lands -- a generous supercell
            # absorbs the rest -- and 126 kJ/mol on one sample is enough to make a bad
            # structure look good and be preferentially replayed.
            #
            # Because the governing cell came from whichever crystal sat at index 1, a
            # structure's energy depended on ITS BATCH-MATES: the same crystal scored
            # differently across a reshuffled prior. That non-reproducibility, more than the
            # accuracy loss, is why this is pinned by a batch-composition invariance test
            # (energy_sampling/test_batch_invariance.py) rather than a golden value.
            cart_centroids = fractional_transform(
                frac_centroids,
                self.T_fc.repeat_interleave(self.max_z_prime, dim=0)
            ).reshape(self.num_graphs, self.max_z_prime, 3)
            dists = (cart_centroids[:, :, None, :] - cart_centroids[:, None, :, :]).norm(dim=-1)  # [n, Zp, Zp]
            zp_buffer = dists.amax(dim=(1, 2)).repeat_interleave(self.z_prime, dim=0)

            zp1_batch = self.split_to_zp1_batch()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            zp1_cluster = zp1_batch.build_cluster(cutoff=cutoff, supercell_size=supercell_size, zp_buffer=zp_buffer)
            return self.join_zp1_cluster_batch(zp1_cluster)

        else:
            # split batches here to avoid silently mutating the original crystal
            zp1_batch = self.clone()
            zp1_batch.pose_aunit(std_orientation=std_orientation)
            zp1_batch.build_unit_cell()
            return zp1_batch.build_cluster(cutoff, supercell_size)

    def mol2ucell(self,
                  std_orientation: Optional[bool] = True):
        if self.max_z_prime > 1:
            zp1_batch = self.split_to_zp1_batch()
            zp1_batch.pose_aunit()
            zp1_batch.build_unit_cell()

            self.join_zp1_ucell_batch(zp1_batch)

        else:
            # split batches here to avoid silently mutating the original crystal
            self.pose_aunit(std_orientation=std_orientation)
            self.build_unit_cell()

