from utils import chunkify
import numpy as np
from ccdc.search import TextNumericSearch, EntryReader
from ccdc.diagram import DiagramGenerator
import tqdm
import os
import pandas as pd
import sys


class CCDC_helper():
    def __init__(self, chunk_path, database, add_features=None):
        self.chunk_path = chunk_path
        self.database = database
        if add_features is not None:  # for adding features after initial pull
            self.features = [add_features]
        else:
            self.features = [
                'xyz',
                'crystal atoms on special positions',
                'crystal date',
                'crystal temperature',
                'crystal r factor',
                'crystal polymorph',
                'crystal has disorder',
                'crystal is organic', 'crystal is organometallic',
                'crystal calculated density', 'crystal packing coefficient', 'crystal void volume',
                'crystal cell angles', 'crystal cell lengths', 'crystal cell volume',
                'crystal lattice centring', 'crystal system', 'crystal z value', 'crystal z prime',
                'crystal spacegroup number', 'crystal spacegroup setting', 'crystal spacegroup symbol',
                'crystal symmetry operators',
                'crystal reference cell coords',
            ]

    def grep_crystal_identifiers(self, chunk_inds=[0, 100]):
        print('Getting hits')
        allhits = []
        # for group in self.searchingGroups:
        for i in range(1921, 2023):
            searcher = TextNumericSearch()
            # searcher.add_spacegroup_symbol(group, mode='separate')
            # searcher.add_spacegroup_symbol(group, mode='start')
            searcher.add_citation(year=i)
            hitlist = searcher.search()
            allhits.extend(hitlist)
            del hitlist

        chunklist = np.arange(start=chunk_inds[0], stop=chunk_inds[1])

        chunks = chunkify(allhits, 100)[chunk_inds[0]:chunk_inds[1]]
        del allhits

        print('Getting identifiers')
        for i in tqdm.tqdm(range(len(chunks))):
            if not os.path.exists(self.chunk_path + 'identifiers/chunk_{}_identifiers'.format(chunklist[i]) + '.npy'):
                print('doing chunk {}'.format(chunklist[i]))
                identifiers = []
                for j in tqdm.tqdm(range(len(chunks[i]))):
                    identifiers.append(chunks[i][j].entry.identifier)
                np.save(self.chunk_path + 'identifiers/chunk_{}_identifiers'.format(chunklist[i]), identifiers)

    def collect_chunks_and_initialize_df(self):

        os.chdir(self.chunk_path + 'identifiers')
        chunks = os.listdir()

        identifiers = []
        for chunk in chunks:
            identifiers.extend(np.load(chunk, allow_pickle=True))

        identifiers = list(set(identifiers))
        df = pd.DataFrame(data=identifiers, index=np.arange(len(identifiers)), columns=['identifier'])
        df.to_pickle('../../csd_dataframe2')

    def get_crystal_features(self, n_chunks=100, chunk_inds=[0, 100],source_dataset = None):
        if source_dataset is None:
            os.chdir(self.chunk_path)
            os.chdir('../')
            df = pd.read_pickle('csd_dataframe2')
        else:
            df = pd.DataFrame.from_dict(np.load(source_dataset,allow_pickle=True).item())

        chunks = chunkify(df, n_chunks)[chunk_inds[0]:chunk_inds[1]]

        csd_reader = EntryReader('CSD')

        for n, chunk in enumerate(chunks):
            if not os.path.exists(self.chunk_path + 'crystal_features/{}'.format(n + chunk_inds[0])):  # don't repeat
                print('doing chunk {} with {} entries'.format(n+chunk_inds[0],len(chunk)))
                if 'level_0' in chunk.columns:  # delete unwanted samples
                    chunk = chunk.drop(columns='level_0')
                chunk = chunk.reset_index()
                new_features = [[] for _ in range(len(self.features))]
                bad_inds = []
                good_inds = []
                for i in tqdm.tqdm(range(len(chunk))):
                    entry = csd_reader.entry(chunk['identifier'][i])
                    molecule = entry.molecule
                    crystal = entry.crystal
                    ma = crystal.asymmetric_unit_molecule
                    if (len(molecule.atoms) > 0) and (len(molecule.components) == 1) and (not entry.is_polymeric) and (entry.has_3d_structure):
                        #try:
                        new_features = self.featurize_crystal(new_features, crystal, entry)
                        good_inds.append(i)
                        # except:
                        #     print('featurization failed on chunk {} entry {}'.format(n, i))
                        #     if len(crystal.molecule.atoms) == 0:
                        #         print("crystal had no atoms")
                        #     bad_inds.append(i)
                    else:
                        if len(molecule.atoms) == 0:
                            print('molecule had no atoms')
                        if not entry.has_3d_structure:
                            print('molecule had no 3d structure')
                        if len(molecule.components) > 1:
                            print('more than one molecule')
                        if entry.is_polymeric:
                            print('entry was polymeric')
                        bad_inds.append(i)

                lens = [len(feat) for feat in new_features]
                assert [lens[0]] * len(lens) == lens

                chunk = chunk.drop(chunk.index[bad_inds])
                if 'level_0' in chunk.columns:  # delete unwanted samples
                    chunk = chunk.drop(columns='level_0')
                # delete unwanted samples
                chunk = chunk.reset_index()
                for i, feature in enumerate(self.features):
                    chunk[feature] = new_features[i]

                if 'crystal temperature' in self.features:
                    chunk['crystal temperature'] = self.fix_temperature(chunk['crystal temperature'])  # post-fix temperature

                chunk.to_pickle(self.chunk_path + 'crystal_features/{}'.format(n + chunk_inds[0]))

    def featurize_crystal(self, features_list, crystal, entry):
        for i, feature in enumerate(self.features):
            if feature == 'xyz':
                value = entry.molecule.to_string('mol2')
            elif feature == 'crystal atoms on special positions':
                value = [atom.index for atom in crystal.atoms_on_special_positions()]
            elif feature == 'crystal date':
                value = str(entry.deposition_date)
            elif feature == 'crystal temperature':
                value = entry.temperature
            elif feature == 'crystal r factor':
                value = entry.r_factor
            elif feature == 'crystal polymorph':
                value = entry.polymorph
            elif feature == 'crystal has disorder':
                value = entry.has_disorder
            elif feature == 'crystal is organic':
                value = entry.is_organic
            elif feature == 'crystal is organometallic':
                value = entry.is_organometallic
            elif feature == 'crystal calculated density':
                value = entry.calculated_density
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
                ref_cell_coords_f = np.zeros((int(crystal.z_value), len(crystal.molecule.heavy_atoms), 3), dtype=np.float_)

                lens = [len(component.heavy_atoms) for component in ref_cell.components]
                if (len(ref_cell.components) == crystal.z_value) and (lens.count(lens[0]) == len(lens)) and (lens[0] == len(crystal.molecule.heavy_atoms)): # correct number of components and molecule size
                    for ind, component in enumerate(ref_cell.components):
                        if ind < crystal.z_value:  # some cells have spurious little atoms counted as extra components. Just hope the early components are the good ones
                            ref_cell_coords_c[ind, :] = np.asarray([atom.coordinates for atom in component.heavy_atoms])  # filter hydrogen
                            ref_cell_coords_f[ind, :] = np.asarray([atom.fractional_coordinates for atom in component.heavy_atoms])  # filter hydrogen

                    value = np.concatenate((ref_cell_coords_c, ref_cell_coords_f),axis=-1)
                else:
                    print('Crystal components not equal to Z value or has incorrect number of heavy atoms in each molecule or had atoms added by the packer for no reason')
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
        t2 = np.zeros(len(temperatures))  # fix temperature
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


if __name__ == '__main__':
    helper = CCDC_helper('C:/Users\mikem\Desktop\CSP_runs\datasets\may_new_pull2/', 'CSD',add_features='crystal reference cell coords')

    # # #pull identifiers
    # ind = 0
    # run = 100
    # helper.grep_crystal_identifiers(chunk_inds=[ind,ind+run])
    #
    # #after pull, initialize dataset
    # helper.collect_chunks_and_initialize_df()

    # # then, featurize each crystal
    # offset = 30
    # gap = 10
    # helper.get_crystal_features(n_chunks=100, chunk_inds=[offset + 0, offset + gap])

    # optionally, store all the packings
    helper.get_crystal_features(source_dataset = 'C:/Users\mikem\Desktop\CSP_runs\datasets/full_dataset.npy')