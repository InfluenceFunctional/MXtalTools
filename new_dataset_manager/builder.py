from common.utils import chunkify
import numpy as np

try:
    from ccdc.search import TextNumericSearch, EntryReader
    from ccdc.diagram import DiagramGenerator
    from ccdc import io
except:
    print("Can't load CCDC - won't be able to featurize crystals!")

import tqdm
import os
import pandas as pd
import sys
from dataset_management.molecule_featurizer import CustomGraphFeaturizer
from dataset_management.manager import DataManager

#
# class CrystalDatabaseBuilder():
#     def __init__(self, cifs_path, database_path):
#         self.cifs_path = cifs_path
#         self.database_path = database_path
#
#         self.features = [
#             'asym unit coords',
#             'crystal date',
#             'crystal r factor',
#             'crystal polymorph',
#             'crystal symmetry operations',
#             'crystal has disorder',
#             'crystal is organic',
#             'crystal is organometallic',
#             'crystal packing coefficient',
#             'crystal cell angles',
#             'crystal cell lengths',
#             'crystal cell volume',
#             'crystal lattice centring',
#             'crystal system',
#             'crystal z value',
#             'crystal z prime',
#             'crystal spacegroup number',
#             'crystal spacegroup setting',
#             'crystal spacegroup symbol',
#             'crystal unit cell coords',
#         ]
#
#     def get_crystal_features(self):
#
#         if cifs_path == 'csd':
#             crystal_reader = EntryReader('CSD')
#             self.featurize_crystals(crystal_reader)
#         else:
#             for file in os.listdir(self.cifs_path):
#                 crystal_reader = io.CrystalReader(file, format='cif')
#                 self.featurize_crystals(crystal_reader)
#
#                 # ma = crystal.asymmetric_unit_molecule
#                 if (len(molecule.atoms) > 0) and (len(molecule.components) == 1):
#                     if entry is not None:
#                         if (not entry.is_polymeric) and (entry.has_3d_structure):
#                             new_features = self.featurize_crystal(new_features, crystal, molecule, entry)
#                             good_inds.append(i)
#                         else:
#                             bad_inds.append(i)
#                     else:
#                         new_features = self.featurize_crystal(new_features, crystal, molecule, entry)
#                         good_inds.append(i)
#
#                 else:  # several available failure modes
#                     bad_inds.append(i)
#                     if len(molecule.atoms) == 0:
#                         pass  # print('molecule had no atoms')
#                     if crystal.z_prime != 1:
#                         pass  # print('zprime not 1')
#                     elif len(molecule.components) > 1:
#                         pass  # print('more than one molecule')
#                     if entry is not None:  # cifs don't come with entries, we will assume they are 'good' for now
#                         if entry.is_polymeric:
#                             pass  # print('entry was polymeric')
#                         if not entry.has_3d_structure:
#                             pass  # print('molecule had no 3d structure')
#
#                 lens = [len(feat) for feat in new_features]
#                 assert [lens[0]] * len(lens) == lens  # confirm all the crystal features are the same length
#
#     def featurize_crystal(self, features_list, crystal, molecule, entry):
#         for i, feature in enumerate(self.features):
#             if feature == 'xyz':
#                 value = molecule.to_string('mol2')
#             elif feature == 'identifier':
#                 value = crystal.identifier
#             elif feature == 'crystal atoms on special positions':
#                 value = [atom.index for atom in crystal.atoms_on_special_positions()]
#             elif feature == 'crystal date':
#                 if entry is not None:
#                     value = str(entry.deposition_date)
#                 else:
#                     value = 0
#             elif feature == 'crystal temperature':
#                 if entry is not None:
#                     value = str(entry.temperature)
#                 else:
#                     value = 0
#             elif feature == 'crystal symmetries':
#                 value = self.get_crystal_sym_ops(crystal)
#             elif feature == 'crystal r factor':
#                 if entry is not None:
#                     value = str(entry.r_factor)
#                 else:
#                     value = None
#             elif feature == 'crystal pressure':
#                 if entry is not None:
#                     value = str(entry.pressure)
#                 else:
#                     value = 0
#             elif feature == 'crystal polymorph':
#                 if entry is not None:
#                     value = str(entry.polymorph)
#                 else:
#                     value = 0
#             elif feature == 'crystal has disorder':
#                 value = crystal.has_disorder
#             elif feature == 'crystal is organic':
#                 if entry is not None:
#                     value = str(entry.is_organic)
#                 else:
#                     value = 0
#             elif feature == 'crystal is organometallic':
#                 if entry is not None:
#                     value = str(entry.is_organometallic)
#                 else:
#                     value = 0
#             elif feature == 'crystal calculated density':
#                 value = crystal.calculated_density
#             elif feature == 'crystal packing coefficient':
#                 value = crystal.packing_coefficient
#             elif feature == 'crystal void volume':
#                 value = crystal.void_volume()
#             elif feature == 'crystal cell angles':
#                 value = tuple(crystal.cell_angles)
#             elif feature == 'crystal cell lengths':
#                 value = tuple(crystal.cell_lengths)
#             elif feature == 'crystal cell volume':
#                 value = crystal.cell_volume
#             elif feature == 'crystal lattice centring':
#                 value = crystal.lattice_centring
#             elif feature == 'crystal system':
#                 value = crystal.crystal_system
#             elif feature == 'crystal z value':
#                 value = crystal.z_value
#             elif feature == 'crystal z prime':
#                 value = crystal.z_prime
#             elif feature == 'crystal spacegroup number':
#                 try:
#                     value = crystal.spacegroup_number_and_setting[0]
#                 except:
#                     print("can't get spacegroup number, invalid symmetry ops")
#                     value = 0
#             elif feature == 'crystal spacegroup setting':
#                 try:
#                     value = crystal.spacegroup_number_and_setting[1]
#                 except:
#                     value = 0
#             elif feature == 'crystal spacegroup symbol':
#                 value = crystal.spacegroup_symbol
#             elif feature == 'crystal symmetry operators':
#                 value = crystal.symmetry_operators
#             elif feature == 'crystal reference cell coords':
#                 ref_cell = crystal.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='CentroidIncluded')
#                 ref_cell_coords_c = np.zeros((int(crystal.z_value), len(crystal.molecule.heavy_atoms), 3), dtype=np.float_)
#                 # ref_cell_coords_f = np.zeros((int(crystal.z_value), len(crystal.molecule.heavy_atoms), 3), dtype=np.float_) # we can easily compute the fractional coords later
#
#                 lens = [len(component.heavy_atoms) for component in ref_cell.components]
#                 if (len(ref_cell.components) == crystal.z_value) and (lens.count(lens[0]) == len(lens)) and (lens[0] == len(crystal.molecule.heavy_atoms)):  # correct number of components and molecule size
#                     for ind, component in enumerate(ref_cell.components):
#                         if ind < crystal.z_value:  # some cells have spurious little atoms counted as extra components. Just hope the early components are the good ones
#                             ref_cell_coords_c[ind, :] = np.asarray([atom.coordinates for atom in component.heavy_atoms])  # filter hydrogen
#                             # ref_cell_coords_f[ind, :] = np.asarray([atom.fractional_coordinates for atom in component.heavy_atoms])  # filter hydrogen
#
#                     value = ref_cell_coords_c  # np.concatenate((ref_cell_coords_c, ref_cell_coords_f), axis=-1)
#                 else:
#                     # print('Crystal components not equal to Z value or has incorrect number of heavy atoms in each molecule or had atoms added by the packer for no reason')
#                     value = 'error'
#             else:
#                 print(feature + ' is not an implemented crystal feature!!')
#                 sys.exit()
#             features_list[i].append(value)
#         # check for non-equal lengths
#         lens = [len(feat) for feat in features_list]
#         assert [lens[0]] * len(lens) == lens
#
#         return features_list
#
#
#     def add_single_feature_to_dataset(self, dataset_path, feature):
#         df = pd.read_pickle(dataset_path)
#         if feature == 'crystal symmetries':
#             feature = [[] for n in range(len(df))]
#             csd_reader = EntryReader('CSD')
#             for i in tqdm.tqdm(range(len(df))):
#                 crystal = csd_reader.crystal(df['identifier'][i])
#                 feature[i] = self.get_crystal_sym_ops(crystal)
#             df['crystal symmetries'] = feature
#
#         df.to_pickle(dataset_path + '_with_new_feature')
#
#     def get_crystal_sym_ops(self, crystal):
#         sym_ops = crystal.symmetry_operators  # get symmetry operators
#         sym_elements = [np.eye(4) for m in range(len(sym_ops))]
#         for j in range(1, len(sym_ops)):  # convert to affine transform
#             sym_elements[j][:3, :3] = np.asarray(crystal.symmetry_rotation(sym_ops[j])).reshape(3, 3)
#             sym_elements[j][:3, -1] = np.asarray(crystal.symmetry_translation(sym_ops[j]))
#
#         return sym_elements
#
#
# cifs_path = r'C:\Users\mikem\crystals\new_csp_data\cifs'
# database_path = r'C:\Users\mikem\crystals\new_csp_data\test1'
#
# if __name__ == '__main__':
#     if not os.path.exists(database_path):  # need to initialize relevant directories
#         os.mkdir(database_path)
#         os.chdir(database_path)
#
#     builder = CrystalDatabaseBuilder(cifs_path, database_path)
#     builder.build_database()
