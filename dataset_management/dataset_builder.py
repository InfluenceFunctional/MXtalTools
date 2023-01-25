from utils import chunkify
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
from dataset_management.dataset_manager import Miner


class CCDC_helper():
    def __init__(self, chunk_path, database, add_features=None):
        self.chunk_path = chunk_path
        self.database = database  # todo make this do something
        if add_features is not None:  # for adding features after initial pull
            self.features = [add_features]
        else:
            self.features = [
                'xyz',
                'crystal atoms on special positions',
                'crystal date',
                'crystal temperature',
                'crystal pressure',
                'crystal r factor',
                'crystal polymorph',
                'crystal symmetries',
                'crystal has disorder',
                'crystal is organic', 'crystal is organometallic',
                'crystal calculated density', 'crystal packing coefficient', 'crystal void volume',
                'crystal cell angles', 'crystal cell lengths', 'crystal cell volume',
                'crystal lattice centring', 'crystal system', 'crystal z value', 'crystal z prime',
                'crystal spacegroup number', 'crystal spacegroup setting', 'crystal spacegroup symbol',
                'crystal reference cell coords',
            ]

    def grep_crystal_identifiers(self, file_path=None, chunk_inds=[0, 100], identifiers=None):
        print('Getting hits')
        allhits = []
        if identifiers is not None:
            np.save(self.chunk_path + 'identifiers/chunk_{}_identifiers'.format(0), identifiers)
        else:
            if self.database.lower() == 'csd':
                for i in range(1921, 2023):  # pull the entire CSD
                    searcher = TextNumericSearch()
                    searcher.add_citation(year=i)
                    hitlist = searcher.search()
                    allhits.extend(hitlist)
                    del hitlist
            elif self.database.lower() == 'cif':  # drawing from one or more .cif files
                print('No cif database - skipping identifiers grep')
                return None
                # here = os.getcwd()
                # os.chdir(file_path)
                # for ind, file in enumerate(os.listdir(file_path)):
                #     crystal_reader = io.CrystalReader(file, format='cif')
                #     for ind2, crystal in enumerate(crystal_reader):
                #         crystal.identifier = file.split('.cif')[0] + '_' + crystal.identifier + '_' + str(ind2)
                #         allhits.append(crystal)
                # os.chdir(here)

            elif self.database.lower() == 'csp':  # grepping cifs from the COD
                print('No cif database - skipping identifiers grep')
                return None
                # paths_list = []
                # for root, subdirs, files in os.walk(file_path):
                #     paths_list.extend([root + '/' + files[_] for _ in range(len(files))])
                #
                # for ind, file in enumerate(paths_list):
                #     crystal_reader = io.CrystalReader(file, format='cif')
                #     # some cifs are empty or unreadable
                #     try:
                #         for ind2, crystal in enumerate(crystal_reader):
                #             allhits.append(crystal)
                #     except:
                #         pass

            max_chunks = len(allhits) // 1000  # minimum of 1000 entries per chunk
            chunk_inds[1] = max(min(max_chunks, chunk_inds[1]), 1)  # at least one chunk
            chunklist = np.arange(start=chunk_inds[0], stop=chunk_inds[1])
            chunks = chunkify(allhits, max(chunk_inds))[chunk_inds[0]:chunk_inds[1]]
            del allhits

            print('Getting identifiers')
            for i in tqdm.tqdm(range(len(chunks))):
                if not os.path.exists(self.chunk_path + 'identifiers/chunk_{}_identifiers'.format(chunklist[i]) + '.npy'):
                    print('Grepping crystal identifiers for chunk {}'.format(chunklist[i]))
                    identifiers = []
                    for j in range(len(chunks[i])):
                        if self.database.lower() == 'csd':
                            identifiers.append(chunks[i][j].entry.identifier)
                        elif self.database.lower() == 'cif':
                            identifiers.append(chunks[i][j].identifier)
                    np.save(self.chunk_path + 'identifiers/chunk_{}_identifiers'.format(chunklist[i]), identifiers)

    def collect_chunks_and_initialize_df(self):
        if self.database.lower() == 'cif':  # drawing from one or more .cif files
            print('No cif database - skipping identifiers grep')
            return None
        else:
            os.chdir(self.chunk_path + 'identifiers')
            chunks = os.listdir()

            identifiers = []
            for chunk in chunks:  # collect entries from all chunks
                identifiers.extend(np.load(chunk, allow_pickle=True))

            identifiers = list(set(identifiers))  # kill any duplicates
            df = pd.DataFrame(data=identifiers, index=np.arange(len(identifiers)), columns=['identifier'])  # initialize dataframe
            df.to_pickle(self.chunk_path + 'new_dataframe')

    def get_crystal_features(self, n_chunks=100, chunk_inds=[0, 100], file_path=None, get_pot_energy_from_cif=False):
        os.chdir(self.chunk_path)

        allhits = []
        if self.database.lower() == 'csd':

            df = pd.read_pickle('new_dataframe')
            chunks = chunkify(df, n_chunks)[chunk_inds[0]:chunk_inds[1]]  # featurize a subset of chunks on this pass
            csd_reader = EntryReader('CSD')

        elif self.database.lower() == 'cif':  # drawing from one or more .cif files
            allhits = []
            all_identifiers = []
            here = os.getcwd()
            os.chdir(file_path)
            for file in os.listdir(file_path):
                crystal_reader = io.CrystalReader(file, format='cif')
                for ind2, crystal in enumerate(crystal_reader):
                    crystal.identifier = file.split('.cif')[0] + '_' + crystal.identifier + '_' + str(ind2)
                    allhits.append(crystal)
                    all_identifiers.append(crystal.identifier)
            os.chdir(here)
            self.features = ['identifier'] + self.features

            assert len(list(set(all_identifiers))) == len(all_identifiers)  # assert all unique identifiers
            chunks = chunkify(allhits, n_chunks)[chunk_inds[0]:chunk_inds[1]]  # featurize a subset of chunks on this pass


        elif self.database.lower() == 'csp':  # grepping cifs from the COD, or a nested set of directories containing .cifs
            paths_list = []
            all_identifiers = []
            energies = []
            flag = 'UPACK energy:'

            for root, subdirs, files in os.walk(file_path):
                paths_list.extend([root + '/' + files[_] for _ in range(len(files))])

            paths_list_cleaned = []
            for path in paths_list:
                if path[-4:] == '.cif':
                    paths_list_cleaned.append(path)

            path_chunks = chunkify(paths_list_cleaned, n_chunks)[chunk_inds[0]:chunk_inds[1]]
            for path_chunk in path_chunks:
                for ind, file in enumerate(tqdm.tqdm(path_chunk)):
                    try:
                        crystal_reader = io.CrystalReader(file, format='cif')

                        # some cifs are empty or unreadable
                        for ind2, crystal in enumerate(crystal_reader):
                            crystal.identifier = file.split('.cif')[0] + '_' + crystal.identifier + '_' + str(ind2)
                            allhits.append(crystal)
                            all_identifiers.append(crystal.identifier)

                        f = open(file)
                        text = f.read()
                        f.close()
                        found = 0
                        for i, line in enumerate(text.split('\n')):
                            if flag in line:
                                found = 1
                                energy = float(line.split(' ')[-1])
                                energies.append(energy)
                                break
                        if found == 0:
                            energies.append(666666)  # error code


                    except:
                        pass

            self.features = ['identifier'] + self.features
            # if get_pot_energy_from_cif:
            #     np.save('potential_energy_dict', {ident: en for ident, en in zip(all_identifiers, energies)})
            #

            assert len(list(set(all_identifiers))) == len(all_identifiers)  # assert all unique identifiers
            chunks = chunkify(allhits, len(path_chunks))  # featurize a subset of chunks on this pass

        for n, chunk in enumerate(chunks):
            if not os.path.exists(self.chunk_path + 'crystal_features/{}'.format(n + chunk_inds[0])):  # don't repeat

                print('doing crystal chunk {} out of {} with {} entries'.format(n + chunk_inds[0], len(chunks), len(chunk)))

                if self.database.lower() == 'csd':
                    if 'level_0' in chunk.columns:  # delete unwanted samples
                        chunk = chunk.drop(columns='level_0')
                    chunk = chunk.reset_index()
                    new_features = [[] for _ in range(len(self.features))]
                else:
                    new_features = [[] for _ in range(len(self.features))]  # add identifier

                bad_inds = []
                good_inds = []
                for i in tqdm.tqdm(range(len(chunk))):
                    if self.database.lower() == 'csd':
                        entry = csd_reader.entry(chunk['identifier'][i])
                        crystal = entry.crystal
                        molecule = entry.molecule

                    elif (self.database.lower() == 'cif') or (self.database.lower() == 'csp'):
                        crystal = chunk[i]
                        molecule = crystal.molecule
                        entry = None  # cifs don't have 'entries' as such

                    # ma = crystal.asymmetric_unit_molecule
                    if (len(molecule.atoms) > 0) and (len(molecule.components) == 1):
                        if entry is not None:
                            if (not entry.is_polymeric) and (entry.has_3d_structure):
                                new_features = self.featurize_crystal(new_features, crystal, molecule, entry)
                                good_inds.append(i)
                            else:
                                bad_inds.append(i)
                        else:
                            new_features = self.featurize_crystal(new_features, crystal, molecule, entry)
                            good_inds.append(i)

                    else:  # several available failure modes
                        bad_inds.append(i)
                        if len(molecule.atoms) == 0:
                            pass  # print('molecule had no atoms')
                        if crystal.z_prime != 1:
                            pass  # print('zprime not 1')
                        elif len(molecule.components) > 1:
                            pass  # print('more than one molecule')
                        if entry is not None:  # cifs don't come with entries, we will assume they are 'good' for now
                            if entry.is_polymeric:
                                pass  # print('entry was polymeric')
                            if not entry.has_3d_structure:
                                pass  # print('molecule had no 3d structure')

                lens = [len(feat) for feat in new_features]
                assert [lens[0]] * len(lens) == lens  # confirm all the crystal features are the same length

                if self.database.lower() == 'csd':
                    # delete unwanted samples
                    chunk = chunk.drop(chunk.index[bad_inds])
                    if 'level_0' in chunk.columns:  # hygiene
                        chunk = chunk.drop(columns='level_0')
                    chunk = chunk.reset_index()
                    # load new features into the dataframe

                else:
                    chunk = pd.DataFrame()  # fresh empty dataframe

                for i, feature in enumerate(self.features):
                    chunk[feature] = new_features[i]

                # temperature needs to be fixed manually
                if ('crystal temperature' in self.features) and (self.database.lower() == 'csd'):
                    chunk['crystal temperature'] = self.fix_temperature(chunk['crystal temperature'])  # post-fix temperatures

                chunk.to_pickle(self.chunk_path + 'crystal_features/{}'.format(n + chunk_inds[0]))

    def featurize_crystal(self, features_list, crystal, molecule, entry):
        for i, feature in enumerate(self.features):
            if feature == 'xyz':
                value = molecule.to_string('mol2')
            elif feature == 'identifier':
                value = crystal.identifier
            elif feature == 'crystal atoms on special positions':
                value = [atom.index for atom in crystal.atoms_on_special_positions()]
            elif feature == 'crystal date':
                if entry is not None:
                    value = str(entry.deposition_date)
                else:
                    value = 0
            elif feature == 'crystal temperature':
                if entry is not None:
                    value = str(entry.temperature)
                else:
                    value = 0
            elif feature == 'crystal symmetries':
                value = self.get_crystal_sym_ops(crystal)
            elif feature == 'crystal r factor':
                if entry is not None:
                    value = str(entry.r_factor)
                else:
                    value = None
            elif feature == 'crystal pressure':
                if entry is not None:
                    value = str(entry.pressure)
                else:
                    value = 0
            elif feature == 'crystal polymorph':
                if entry is not None:
                    value = str(entry.polymorph)
                else:
                    value = 0
            elif feature == 'crystal has disorder':
                value = crystal.has_disorder
            elif feature == 'crystal is organic':
                if entry is not None:
                    value = str(entry.is_organic)
                else:
                    value = 0
            elif feature == 'crystal is organometallic':
                if entry is not None:
                    value = str(entry.is_organometallic)
                else:
                    value = 0
            elif feature == 'crystal calculated density':
                value = crystal.calculated_density
            elif feature == 'crystal packing coefficient':
                value = crystal.packing_coefficient
            elif feature == 'crystal void volume':
                value = crystal.void_volume()
            elif feature == 'crystal cell angles':
                value = tuple(crystal.cell_angles)
            elif feature == 'crystal cell lengths':
                value = tuple(crystal.cell_lengths)
            elif feature == 'crystal cell volume':
                value = crystal.cell_volume
            elif feature == 'crystal lattice centring':
                value = crystal.lattice_centring
            elif feature == 'crystal system':
                value = crystal.crystal_system
            elif feature == 'crystal z value':
                value = crystal.z_value
            elif feature == 'crystal z prime':
                value = crystal.z_prime
            elif feature == 'crystal spacegroup number':
                try:
                    value = crystal.spacegroup_number_and_setting[0]
                except:
                    print("can't get spacegroup number, invalid symmetry ops")
                    value = 0
            elif feature == 'crystal spacegroup setting':
                try:
                    value = crystal.spacegroup_number_and_setting[1]
                except:
                    value = 0
            elif feature == 'crystal spacegroup symbol':
                value = crystal.spacegroup_symbol
            elif feature == 'crystal symmetry operators':
                value = crystal.symmetry_operators
            elif feature == 'crystal reference cell coords':
                ref_cell = crystal.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='CentroidIncluded')
                ref_cell_coords_c = np.zeros((int(crystal.z_value), len(crystal.molecule.heavy_atoms), 3), dtype=np.float_)
                # ref_cell_coords_f = np.zeros((int(crystal.z_value), len(crystal.molecule.heavy_atoms), 3), dtype=np.float_) # we can easily compute the fractional coords later

                lens = [len(component.heavy_atoms) for component in ref_cell.components]
                if (len(ref_cell.components) == crystal.z_value) and (lens.count(lens[0]) == len(lens)) and (lens[0] == len(crystal.molecule.heavy_atoms)):  # correct number of components and molecule size
                    for ind, component in enumerate(ref_cell.components):
                        if ind < crystal.z_value:  # some cells have spurious little atoms counted as extra components. Just hope the early components are the good ones
                            ref_cell_coords_c[ind, :] = np.asarray([atom.coordinates for atom in component.heavy_atoms])  # filter hydrogen
                            # ref_cell_coords_f[ind, :] = np.asarray([atom.fractional_coordinates for atom in component.heavy_atoms])  # filter hydrogen

                    value = ref_cell_coords_c  # np.concatenate((ref_cell_coords_c, ref_cell_coords_f), axis=-1)
                else:
                    # print('Crystal components not equal to Z value or has incorrect number of heavy atoms in each molecule or had atoms added by the packer for no reason')
                    value = 'error'
            else:
                print(feature + ' is not an implemented crystal feature!!')
                sys.exit()
            features_list[i].append(value)
        # check for non-equal lengths
        lens = [len(feat) for feat in features_list]
        assert [lens[0]] * len(lens) == lens

        return features_list

    def fix_temperature(self, temperatures):
        '''
        parse all the different ways the CSD records temperature
        '''
        t2 = np.zeros(len(temperatures))
        for i, temp in enumerate(temperatures):
            if temp is None:
                t2[i] = -1
            elif not any([a.isdigit() for a in temp]):  # if there are no digits, just ignore it
                t2[i] = -1
            else:
                if temp[-1] == '.':
                    temp = temp[:-1]
                unit = temp[-1]
                if unit.lower() == 'c':
                    if 'deg' in temp:
                        temp = temp[:-5]
                    else:
                        temp = temp[:-1]

                elif unit.lower() == 'k':
                    temp = temp[:-1]

                l = []
                for t in temp.split():
                    try:
                        l.append(float(t))
                    except ValueError:
                        pass

                try:
                    value = l[0]
                except:
                    print('no temperature value found for {}'.format(temperatures[i]))
                    print(l)
                    value = -1

                if unit.lower() == 'c':
                    value += 273
                    t2[i] = value
                elif unit.lower() == 'k':
                    t2[i] = value
                else:
                    t2[i] = -1

        return list(t2)

    def add_single_feature_to_dataset(self, dataset_path, feature):
        df = pd.read_pickle(dataset_path)
        if feature == 'crystal symmetries':
            feature = [[] for n in range(len(df))]
            csd_reader = EntryReader('CSD')
            for i in tqdm.tqdm(range(len(df))):
                crystal = csd_reader.crystal(df['identifier'][i])
                feature[i] = self.get_crystal_sym_ops(crystal)
            df['crystal symmetries'] = feature

        df.to_pickle(dataset_path + '_with_new_feature')

    def get_crystal_sym_ops(self, crystal):
        sym_ops = crystal.symmetry_operators  # get symmetry operators
        sym_elements = [np.eye(4) for m in range(len(sym_ops))]
        for j in range(1, len(sym_ops)):  # convert to affine transform
            sym_elements[j][:3, :3] = np.asarray(crystal.symmetry_rotation(sym_ops[j])).reshape(3, 3)
            sym_elements[j][:3, -1] = np.asarray(crystal.symmetry_translation(sym_ops[j]))

        return sym_elements


def visualizeEntry(identifier):
    csd_reader = EntryReader('CSD')
    mol = csd_reader.molecule(identifier)

    generator = DiagramGenerator()
    generator.settings.font_size = 12
    generator.settings.line_width = 1.6
    generator.settings.image_width = 500
    generator.settings.image_height = 500
    img = generator.image(mol)
    img.show()


mode = 'csd'  #
chunk_path = 'C:/Users\mikem\Desktop\CSP_runs\datasets/nov_functional_add/'  # where the chunks should be saved during featurization
cifs_directory_path = None

mode = 'csd'
chunk_path = 'C:/Users\mikem\Desktop\CSP_runs\datasets/csd_coumarins/'
cifs_directory_path = None
target_identifiers = ["COUMAR01","COUMAR02","COUMAR10","COUMAR11","COUMAR12","COUMAR13",
                      "COUMAR14","COUMAR15","COUMAR16","COUMAR17", # Z'!=1 or some other weird thing
                      "COUMAR18","COUMAR19"]
# target_identifiers = [
#     'OBEQUJ', 'OBEQOD', 'OBEQET', 'OBEQIX',
#     'NACJAF', 'XAFPAY', 'XAFPAY01','XAFPAY02','XAFPAY03','XAFPAY04', 'XAFQIH'
# ]

# mode = 'cif'
# chunk_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/blind_test_6/'  # where the chunks should be saved during featurization
# cifs_directory_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/test_structures/blind_tests/blind_test_6/gp5080sup2'
# target_identifiers = None

# mode = 'cif' # FYI the sheraga extended submissions and Kendrick submissions for BT5 are mislabelled. Have to manually fix them
# chunk_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/blind_test_5/'  # where the chunks should be saved during featurization
# cifs_directory_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/test_structures/blind_tests/blind_test_5/bk5106sup2/file_dump'
# target_identifiers = None

# mode = 'cod'
# chunk_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/COD/'  # where the chunks should be saved during featurization
# cifs_directory_path = 'F:/cod-cifs-mysql'
# target_identifiers = None

# mode = 'csp'
# chunk_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/sapt_full/'  # where the chunks should be saved during featurization
# cifs_directory_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/bt_31_csp'
# target_identifiers = None

# mode = 'cif'
# chunk_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/bt_31_target_data/'  # where the chunks should be saved during featurization
# cifs_directory_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/bt_31_target'
# target_identifiers = None

if __name__ == '__main__':
    if not os.path.exists(chunk_path):  # need to initialize relevant directories
        os.mkdir(chunk_path)
        os.mkdir(chunk_path + '/identifiers')
        os.mkdir(chunk_path + '/crystal_features')
        os.mkdir(chunk_path + '/molecule_features')

    helper = CCDC_helper(chunk_path, mode)
    # helper.add_single_feature_to_dataset(dataset_path='C:/Users/mikem/Desktop/CSP_runs/datasets/new_dataset',
    #                                      feature='crystal symmetries')
    helper.grep_crystal_identifiers(file_path=cifs_directory_path, identifiers = target_identifiers)
    helper.collect_chunks_and_initialize_df()
    helper.get_crystal_features(n_chunks=1, chunk_inds=[0, 1], file_path=cifs_directory_path)

    featurizer = CustomGraphFeaturizer(chunk_path + '/crystal_features')
    #featurizer.add_single_feature(molecule_chunks_path=chunk_path + '/molecule_features', chunk_inds = [0,1000], feature='molecule freeSASA')
    featurizer.featurize(chunk_inds=[0, 1])

    miner = Miner(chunk_path, collect_chunks=True, database=mode)
    miner.process_new_dataset(dataset_name = 'csd_coumarins')
