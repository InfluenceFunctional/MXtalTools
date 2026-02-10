from typing import Optional

import torch

from mxtaltools.common.geometry_utils import fractional_transform
from mxtaltools.crystal_building.utils import get_aunit_positions, aunit2ucell, ucell2cluster
from mxtaltools.dataset_utils.utils import collate_data_list


# noinspection PyAttributeOutsideInit
class MolCrystalBuilding:
    def split_to_zp1_batch(self):
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
        zp1_batch.ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(atoms_per_subunit, dim=0)])
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
        zp1_batch.num_atoms = (self.num_atoms//self.z_prime)[rep_index]
        zp1_batch.radius = self.radius[rep_index]
        zp1_batch.mol_volume = self.mol_volume[rep_index]
        zp1_batch.mass = self.mass[rep_index]
        zp1_batch.z_prime = torch.ones(new_num_graphs, device=self.device, dtype=torch.long)

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


    def pose_aunit(self, std_orientation: Optional[bool] = True):
        if self.is_batch:
            self.pos = get_aunit_positions(
                self,
                std_orientation=std_orientation,
                mol_handedness=self.aunit_handedness,
            )
        else:
            self.pos = get_aunit_positions(
                collate_data_list([self]),
                std_orientation=std_orientation,
                mol_handedness=self.aunit_handedness,
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
            cart_centroids = fractional_transform(frac_centroids,self.T_fc.repeat_interleave(self.max_z_prime,dim=0)[1]).reshape(self.num_graphs, self.max_z_prime, 3)
            dists = (cart_centroids[:, :, None, :] - cart_centroids[:, None, :, :]).norm(dim=-1) # [n, Zp, Zp, 3]
            zp_buffer = dists.amax(dim=(1,2)).repeat_interleave(self.z_prime, dim=0)

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

