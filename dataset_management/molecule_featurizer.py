from utils import *
import numpy.linalg as linalg
import tqdm
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from nikos.coordinate_transformations import coor_trans_matrix
from mendeleev import element as element_table
from crystal_builder_tools import (get_cell_fractional_centroids, c_f_transform)
from pyxtal import symmetry
from scipy.spatial.transform import Rotation


def get_fraction(atomic_numbers, target):
    return atomic_numbers.count(target) / len(atomic_numbers)


def get_dipole(coords, charges):
    center_of_geometry = np.average(np.asarray(coords), axis=0)
    # center_of_charge = np.average(np.multiply(np.asarray(coords),np.asarray(charges)[:,np.newaxis]), axis=0)
    center_of_charge = np.average(np.asarray(coords), weights=charges, axis=0)
    return np.linalg.norm(center_of_charge - center_of_geometry), center_of_geometry


class CustomGraphFeaturizer():
    def __init__(self, crystal_chunks_path=None, full_dataset_path=None):
        '''
        get atom and molecule level features

        Convert dataset to trainable features
        '''
        if crystal_chunks_path is not None:
            os.chdir(crystal_chunks_path)
            self.crystal_chunks_path = crystal_chunks_path

        if full_dataset_path is not None:
            self.full_dataset_path = full_dataset_path

        self.crystalSystems = ['error', 'triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic', 'rhombohedral']
        self.latticeCentering = ['error', 'primitive', 'A-centred', 'B-centred', 'C-centred', 'F-centred', 'I-centred', 'R-centred']
        self.bravaisLattices = ['P', 'I', 'F', 'A', 'B', 'C', 'R']
        # self.unique_elements = ['-x,-z,1/2-y', '-x,1/2+y,1/2+z', '3/4+z,1/4+y,-x', '3/4-z,3/4-x,1/4-y', '-x,1/2+y,-z', 'y,-x+y,5/6+z', '1/4-y,1/2+z,3/4-x', '3/4+x,3/4+y,3/4-z', 'x-y,-y,1/3-z', '1/2-x,3/4+z,3/4+y', '1/2-z,1/2-x,1/2+y',
        #                         'y,x,1/4-z', '1/2+x,-z,y', '1/2+z,1/2+y,1/2-x', '1/4-x,3/4-z,3/4-y', '3/4+y,3/4+x,1/4+z', '1/4+z,1/4+x,3/4-y', '1/2+y,1/4-x,1/4-z', '1/2+x,z,-y', '1/4+y,1/2-x,1/4+z', '1/4-x,3/4-y,1/4-z', '1/4-y,1/4-x,3/4-z',
        #                         'x,x-y,2/3-z', '1/4-y,1/4+x,1/4-z', '1/4-z,3/4+y,3/4+x', '3/4+x,1/2-z,3/4+y', '3/4-z,3/4-y,1/4-x', 'x,-z,y', '-z,x,-y', '3/4+z,1/4+y,3/4-x', 'x-y,x,1/6+z', '1/2-y,3/4+z,1/4+x', '2/3+x-y,1/3-y,5/6-z',
        #                         '1/4+x,1/4-y,3/4+z', 'z,y,-x', '1/4-y,1/4+x,3/4-z', '1/2-y,x,1/2+z', '3/4-y,1/4-x,1/4+z', '3/4+z,1/4-x,1/4+y', '3/4+y,3/4+x,3/4+z', '3/4-y,x,1/4-z', 'x,-z,1/2+y', 'x,3/4-z,3/4-y', '1/4-z,1/4-y,x',
        #                         '-x,1/2+y,z', '3/4-z,1/4-x,1/4-y', '3/4+x,1/2-y,1/4+z', '1/4+x,1/4+y,3/4-z', '1/2-z,-y,1/2+x', '1/2+x,1/2-z,-y', '3/4+x,1/2-z,1/4+y', 'z,x,y', '-y,1/2+x,1/4+z', '2/3+x-y,1/3+x,1/3-z', '1/2-y,-x,1/2+z',
        #                         '3/4+z,3/4-x,1/4+y', 'x,1/4-y,1/2+z', '1/2-y,z,1/2-x', '1/2-x,1/2+z,y', '1/4+z,3/4+y,1/2-x', 'y,1/2+x,3/4-z', '1/4-x,1/4-y,3/4-z', '1/4+x,1/2-z,3/4+y', 'z,1/2+y,1/2+x', '1/4-x,1/4-z,1/4-y', '1/2-z,y,1/2+x',
        #                         '1/2-z,1/2+y,1/2-x', '1/2+x,-y,1/2-z', '1/4+x,3/4+z,-y', '-z,1/2+y,x', '3/4-x,1/2+z,3/4-y', '1/2-y,-z,1/2+x', '1/4-y,3/4-z,1/4-x', '1/2+x,1/2+z,1/2-y', '3/4-y,1/4-z,1/2+x', '1/4+z,3/4+x,1/2-y',
        #                         '-x,1/2-y,1/2-z', '1/3-y,2/3+x-y,2/3+z', 'z,-x,-y', 'x,-z,1/2-y', '-y,1/2-x,1/2+z', 'x,1/2+z,1/2+y', '1/2-x,-z,1/2+y', '1/4+x,1/2-z,1/4+y', '1/2-x,-z,y', 'x,x-y,1/2+z', 'z,-y,1/2+x', '3/4+x,3/4-y,1/4+z',
        #                         '3/4-x,3/4+y,3/4+z', '1/2-z,y,1/2-x', '-y,-x,2/3-z', '1/4-x,1/4+y,1/4+z', '1/2-z,-x,1/2+y', '1/2-z,1/2+x,1/2-y', '1/4+y,-x,3/4+z', '1/4+x,1/2-y,1/4+z', 'y,x,z', '1/2+y,z,1/2-x', '1/2+z,1/2+y,1/2+x',
        #                         '1/2-z,1/2+x,y', '-x,-x+y,1/3-z', 'x-y,x,1/3+z', '-x,1/2+y,3/4-z', '3/4+z,3/4+x,-y', '3/4+x,3/4-z,1/4+y', '1/2+y,1/2+z,1/2+x', '3/4-z,1/4-y,x', '1/4+y,1/4+z,1/4-x', '1/4+z,1/4-y,1/4+x', '1/4-y,1/2+x,3/4-z',
        #                         'x-y,-y,z', '3/4-z,1/4-x,1/2+y', '3/4-y,3/4-x,3/4-z', '1/2+z,y,x', '1/2+x,y,-z', 'x,1/2-y,1/2-z', 'y,1/4-x,1/4-z', '1/4+z,-y,1/4+x', '1/4+y,1/4-z,1/4+x', '1/2+x,1/2+y,1/2+z', '-y,-x,z', '1/2+z,1/2+y,-x',
        #                         '-y,1/2+x,z', 'z,1/2+y,1/2-x', 'x,1/2-z,y', '1/2+y,3/4-x,1/4-z', '2/3-y,1/3-x,5/6+z', '1/3+x,2/3+x-y,2/3+z', '1/2+x,1/2+z,-y', '1/4+x,3/4+y,3/4-z', '1/2-z,3/4+y,3/4+x', '3/4+z,3/4+y,1/2-x', '1/2-z,x,-y',
        #                         '1/2+y,x,1/4+z', '-y,3/4+x,3/4+z', '2/3+x-y,1/3-y,1/3-z', '1/4-x,3/4-y,1/2+z', '3/4-y,3/4-z,x', '1/2+x,-y,-z', '-y,x-y,-z', '1/4-z,x,1/4-y', '-x+y,-x,1/2-z', '1/2-z,x,1/2-y', '3/4-z,1/4+y,3/4+x',
        #                         '1/4+y,1/4+x,3/4+z', '3/4+x,1/4+z,1/2-y', '-z,1/2-y,x', '1/2+x,-y,3/4-z', '3/4-x,z,1/4-y', '1/2-y,1/2-z,x', '1/2+y,1/2+z,-x', '-z,-y,1/2-x', '3/4-x,1/4-y,3/4-z', '1/2+x,-z,-y', '3/4+y,1/4+z,1/2-x',
        #                         'y,x,1/3-z', '-x,1/2-y,z', '1/2+x,1/2-z,1/2-y', '3/4+x,1/2-y,3/4+z', 'y,1/2+x,z', '-y,-x,3/4-z', '3/4+z,1/4-x,3/4+y', '1/2+y,-x,-z', '3/4-y,1/4+x,1/4+z', '1/2+z,3/4-y,1/4-x', '3/4-x,3/4-z,1/2+y',
        #                         '1/2+y,x,3/4+z', '1/2+x,-y,1/4-z', '3/4+z,1/4+y,1/4-x', '1/4-z,1/4+x,1/4+y', '2/3+y,1/3+x,5/6-z', '1/4-x,1/2+y,3/4-z', '-z,1/2+y,1/2+x', '-y,1/2+z,1/2-x', '1/2-y,-x,1/4+z', '1/2-x,1/2+y,1/4-z',
        #                         '1/3-x+y,2/3+y,1/6+z', '1/4+y,1/2-x,3/4+z', '1/2-x,-z,1/2-y', '1/4+y,3/4+x,1/4-z', '1/4-y,3/4-x,1/2+z', 'y,-x,3/4+z', '1/2-y,-x,1/4-z', '-z,y,-x', '1/2-z,1/4+y,3/4+x', '-y,-x,1/4-z', '3/4+x,-z,1/4+y',
        #                         '3/4+x,-y,3/4+z', '1/2+y,1/2-z,1/2+x', '1/4+z,1/4-x,1/4+y', '3/4+y,1/2-z,1/4+x', '3/4-z,3/4-x,3/4-y', '3/4-x,3/4-y,z', '1/4-z,1/4-y,1/4-x', '1/2+x,1/2-y,1/2-z', '1/4-y,3/4-x,3/4-z', 'x,x-y,1/6-z',
        #                         '1/4-x,3/4+z,3/4+y', '1/2+y,1/2+x,z', '2/3+x,1/3+x-y,5/6+z', 'x,1/2+y,-z', '1/2+y,3/4-z,1/4-x', '3/4+x,3/4+z,-y', '1/4+y,3/4+z,3/4-x', 'x-y,-y,1/2-z', '1/2-y,-x,z', '-y,1/2-x,z', '1/4-z,3/4-x,1/2+y',
        #                         'z,-y,x', '3/4-y,3/4+x,1/4+z', 'x,-y,-z', '1/4+x,-y,3/4+z', 'x,y,1/2-z', '1/2+y,1/2+x,1/2-z', '1/4-x,3/4-z,1/2+y', '1/4+y,3/4+z,1/2-x', '1/4-z,3/4+x,3/4+y', '1/2+z,1/2-y,-x', '1/2+z,1/4-x,3/4-y',
        #                         '1/2+x,1/2-z,y', '1/2+x,1/2-y,z', '3/4+x,3/4+y,-z', '1/3+x,2/3+y,2/3+z', '3/4-x,1/4+y,3/4+z', '1/2+z,-y,x', '1/2+y,-x,z', '1/2+z,1/2+x,y', '-x,z,y', '-z,1/2-x,1/2-y', '3/4-y,1/4+x,3/4+z', '3/4-x,1/4-z,y',
        #                         '-z,1/2-y,-x', '1/2-x,-y,z', '3/4+y,-x,3/4+z', '3/4+y,1/4-x,3/4-z', '1/3-x,2/3-y,2/3-z', 'x,z,-y', 'z,3/4-y,1/4-x', '2/3-y,1/3-x,1/3+z', '-x,1/2+z,y', '3/4-x,3/4-y,3/4-z', 'y,-x,1/2-z', '-y,3/4+z,3/4+x',
        #                         '1/2+x,3/4-z,3/4-y', '1/2+x,1/2-y,-z', '1/3+y,2/3-x+y,1/6-z', '1/2-y,x,z', '1/2+z,y,1/2+x', '1/2+z,1/2-y,1/2+x', '3/4-y,3/4+z,1/4+x', '-x,-z,y', '3/4+y,1/4+x,-z', '3/4-y,3/4-x,1/4-z', '1/2+z,y,-x',
        #                         '1/2+y,1/2+x,-z', '-z,1/2-x,1/2+y', '1/2+y,-x,1/4-z', '1/4+z,1/4+y,1/4-x', '-x,1/2+z,1/2+y', '3/4-x,3/4-z,1/4-y', '3/4-x,1/4-z,1/4+y', 'y,1/2+x,3/4+z', '-x,y,z', '2/3-x,1/3-y,1/3-z', '2/3+y,1/3+x,1/3-z',
        #                         '3/4+y,3/4+x,-z', '1/4-y,1/4-x,z', '3/4+z,3/4+y,1/4-x', '3/4-z,3/4+y,1/4+x', '1/4-z,3/4-y,3/4-x', '-x,1/2+z,1/2-y', '3/4-y,1/4-x,1/2+z', '1/2+z,-x,y', '1/2-x,y,1/2-z', '1/2-y,1/4+z,3/4+x',
        #                         '1/4-z,1/4+y,3/4-x', '1/4+z,1/4-x,3/4+y', '1/2-y,1/4+x,3/4+z', '3/4-x,1/2+y,z', '1/4+y,1/4+x,1/4+z', '1/4+x,1/4-z,3/4+y', '1/2-z,y,x', '1/2+y,1/2-x,z', '3/4+z,1/2-y,1/4+x', '1/2+z,-y,1/2-x', 'x,1/2+z,y',
        #                         '1/4-z,1/4-y,1/2+x', '1/2+x,z,1/2-y', '3/4-z,1/4-y,3/4-x', '1/4-y,1/4+z,1/4+x', '3/4+y,-z,3/4+x', '3/4+y,1/2-x,3/4+z', '1/3+x-y,2/3+x,1/6-z', '3/4+z,3/4+x,1/4-y', '3/4+z,1/4+x,3/4-y', '1/2-x,1/2+z,-y',
        #                         '-x,-y,1/2+z', 'x,x-y,1/2-z', '-y,1/2+x,1/2+z', 'y,-x+y,1/2-z', 'y,1/2-x,-z', '3/4-x,1/4+y,1/4+z', '1/2+x,1/2-y,1/2+z', 'z,1/2+x,1/2+y', '1/4+y,1/4-x,3/4+z', 'y,-x+y,1/6+z', '1/4+y,-z,1/4+x',
        #                         '1/2-y,1/2+z,1/2+x', 'y,-x,-z', '1/2+x,-y,1/2+z', '3/4+y,1/4+x,1/4-z', 'x,x-y,-z', '1/4+y,3/4-z,1/4+x', '1/4+x,-z,3/4+y', 'y,3/4-x,3/4-z', '-z,3/4+y,1/4+x', '1/4-y,x,3/4-z', '1/2-y,x,3/4-z', '3/4-y,3/4-x,z',
        #                         'y,z,x', '1/3+y,2/3+x,1/6-z', 'x,1/2+y,1/2+z', '1/2+y,1/4-x,3/4-z', '-z,-y,x', '3/4-y,1/2+x,3/4-z', '1/4-y,1/4-z,x', 'x,1/4-z,1/4-y', '3/4-y,z,3/4-x', 'y,-x+y,1/3+z', '3/4-x,y,1/4-z', '1/2-y,1/2+x,1/2-z',
        #                         '1/2+y,x,1/2+z', '3/4+x,-z,3/4+y', '1/2-y,1/2+z,-x', '1/4+y,1/4+x,1/4-z', '3/4-y,3/4+x,1/4-z', '1/4+x,3/4-y,1/4+z', '1/2-x,1/2+z,1/2-y', 'z,-y,1/2-x', 'z,-x,1/2-y', '-x,-x+y,z', '3/4+x,-y,1/4+z',
        #                         'z,1/2-x,1/2-y', '1/2-x,1/2-y,-z', '-y,x-y,1/2-z', '1/2-z,x,1/2+y', '1/4-x,3/4-z,y', '-z,-y,-x', '-x,1/2+y,1/4-z', '-y,x-y,z', 'x,x-y,1/3-z', '1/4-x,1/4+z,1/4+y', '1/2-x,z,1/2-y', 'z,-x,y', 'x,1/2-z,1/2-y',
        #                         '1/2-x,y,3/4-z', '1/2-y,1/2+z,x', '3/4-x,1/2+y,1/4-z', 'y,3/4-x,1/4-z', 'y,-x,1/2+z', '1/2+x,1/2-y,3/4-z', '-x+y,y,1/2+z', '1/2+y,3/4-x,3/4-z', 'y,-x,1/4+z', '1/2+x,y,1/2+z', '-z,1/2-x,y',
        #                         '3/4+y,3/4+x,1/2-z', '1/2+x,1/2+z,1/2+y', '1/2-x,1/2-z,1/2+y', '1/2-x,1/2-z,1/2-y', '-z,-x,-y', 'z,3/4-x,3/4-y', '1/4+z,1/4-y,3/4+x', '1/4-y,z,1/4-x', '3/4-z,3/4+x,1/4+y', '3/4+y,1/2-x,1/4+z',
        #                         '3/4-x,y,3/4-z', 'y,1/2+z,1/2-x', '-x,1/2-z,-y', '1/4-z,3/4-x,1/4-y', '-y,-x,1/2+z', '1/4-z,y,1/4-x', '3/4-y,1/4+z,1/4+x', '-x,1/2-z,y', '1/4-z,1/4+y,1/4+x', 'x-y,-y,-z', '1/2-y,-x,3/4+z',
        #                         '1/2+z,3/4-y,3/4-x', '3/4+x,1/4+y,1/2-z', '3/4+z,1/4-y,3/4+x', '1/4+y,1/4-z,3/4+x', 'z,y,x', 'z,1/4-y,1/4-x', 'x,x-y,z', '1/4-y,3/4-z,1/2+x', '1/2+z,1/4-y,1/4-x', '1/2-x,z,y', '1/4-x,1/2+y,1/2-z',
        #                         '1/2+z,x,1/2-y', 'z,1/2+y,x', 'y,z,-x', '-z,1/2+x,1/2-y', '-x+y,y,1/3-z', '1/2+x,1/2-y,1/4-z', '1/4-y,1/4+x,1/4+z', '-y,1/2-z,1/2+x', '3/4-y,1/4-x,z', '-x,z,-y', '1/2-y,1/2+x,-z', '1/4-y,1/4-x,1/4-z',
        #                         'y,1/2-x,1/2-z', '1/4+z,3/4+x,1/4-y', '-x,-y,1/2-z', '1/2+y,1/2-z,-x', '1/2+z,x,1/2+y', '1/4+x,1/2-y,3/4+z', '3/4+x,3/4+z,1/4-y', '3/4+y,3/4+z,1/4-x', '1/4-x,1/4+y,3/4+z', '3/4-y,1/4+z,3/4+x',
        #                         '1/3-x+y,2/3+y,2/3+z', '3/4+y,3/4+x,1/4-z', '1/3-y,2/3-x,2/3+z', '1/2-x,y,z', '2/3-x,1/3-x+y,1/3-z', 'y,1/4-z,1/4-x', '3/4+y,3/4-z,3/4+x', '1/2+y,1/2-x,1/2+z', '1/2+x,z,y', '1/2+x,z,1/2+y',
        #                         '1/4-z,1/2+y,3/4-x', 'y,3/4-z,3/4-x', '1/4+z,3/4-y,1/4-x', '3/4+z,1/2-y,3/4+x', '1/3+y,2/3+x,2/3-z', '3/4-x,z,3/4-y', '1/3+x-y,2/3-y,2/3-z', '3/4-z,3/4-y,x', 'y,1/2-x,3/4+z', 'x-y,-y,1/2+z', 'y,1/2-z,1/2+x',
        #                         '1/2-x,1/2-y,z', '3/4-z,1/2+x,1/4-y', '3/4-z,3/4+y,1/4-x', '-z,y,x', '1/4-z,1/4-x,3/4-y', '1/4+x,1/4-y,1/4+z', '2/3-y,1/3+x-y,1/3+z', '1/4+y,3/4-x,3/4-z', '1/2-x,1/2-z,-y', '3/4-z,x,3/4-y', '1/2-x,1/2+y,z',
        #                         '3/4-y,x,3/4-z', '-x,1/4+z,3/4+y', '1/4-y,3/4-x,1/4+z', '1/4+x,3/4+z,1/2-y', '-z,3/4+y,3/4+x', '1/4+z,1/4+x,-y', '1/3+x-y,2/3+x,2/3-z', '1/4+z,1/4+y,1/2-x', '3/4+z,1/4+x,1/4-y', 'x,-y,1/2-z', '1/2-x,y,-z',
        #                         '1/3+x,2/3+x-y,1/6+z', '1/4-x,1/4-z,1/2+y', '1/4-x,y,3/4-z', '-y,1/2+z,1/2+x', '3/4-z,1/4+x,3/4+y', '3/4+x,3/4+z,3/4+y', '1/2-z,1/2+y,x', '-y,x-y,2/3+z', '3/4-z,1/4-x,3/4-y', '-z,1/2-y,1/2+x',
        #                         '1/2+x,3/4-y,1/4-z', '1/2+y,1/2-z,x', '-x,-x+y,-z', '1/4+y,1/2-z,3/4+x', 'x,1/2+z,-y', '-y,x,-z', '3/4-y,1/2+x,1/4-z', '3/4+y,3/4-x,3/4+z', '1/4-z,3/4-y,1/2+x', '3/4+y,1/4+x,3/4-z', '1/4+x,-z,1/4+y',
        #                         'y,1/2+z,-x', '1/2-y,x,3/4+z', '1/2-x,-y,1/2-z', '-y,z,1/2+x', '-x,1/4+z,1/4+y', 'y,1/2+x,1/4-z', 'z,1/2+x,-y', '-y,1/2-x,3/4+z', '1/4+z,-y,3/4+x', '1/3+y,2/3-x+y,2/3-z', '1/2-y,1/2+z,1/2-x',
        #                         '2/3-x,1/3-y,5/6-z', 'x,x-y,5/6-z', '1/4-z,y,3/4-x', '1/2-z,-x,1/2-y', '1/2+x,1/2-z,1/2+y', '1/2-x,1/2-z,y', '1/4-x,3/4+z,1/4+y', '-z,1/4+y,3/4+x', 'x,y,-z', '1/4-y,1/2+x,1/4-z', 'x,1/2-y,1/2+z',
        #                         '-y,1/4+z,1/4+x', '1/2-y,1/2-z,1/2+x', '1/2-x,-y,-z', '3/4+y,1/4-z,3/4+x', '-y,-x,1/3-z', 'x,1/2-z,-y', '1/4+x,1/4+y,-z', '3/4-x,1/4-y,1/2+z', '1/2-x,1/2+y,1/2+z', '-x,-y,z', '1/2-y,z,-x', '1/2-z,1/2-y,-x',
        #                         '1/2-z,1/2-y,1/2+x', '1/2-y,1/2-z,1/2-x', '3/4+x,1/4-z,3/4+y', '1/2-y,-x,-z', '-z,1/2+y,-x', '1/2-y,1/2-x,1/2+z', '1/4+y,-x,1/4+z', '1/4-x,1/4-y,z', '1/4-y,3/4-x,3/4+z', '3/4-y,3/4-x,1/2+z', '-x,1/4+y,1/4+z',
        #                         'x,-y,z', 'y,1/2-x,1/4+z', '1/4-x,1/4+z,3/4-y', '1/4-x,1/4-z,y', '1/2+z,-y,-x', 'x-y,-y,5/6-z', '1/2+y,z,1/2+x', '1/2-y,1/2+x,1/2+z', '-x,-x+y,1/2+z', '1/4-y,1/4-z,1/4-x', 'z,1/2-y,1/2+x',
        #                         '1/4-z,1/4-x,1/4-y', '-x,-y,-z', '1/2-x,1/2-y,1/2+z', '1/4+x,3/4-z,1/4-y', '3/4+z,3/4+y,-x', '-y,-x,5/6-z', '1/2-z,3/4+y,1/4+x', '1/4-x,1/2+z,3/4-y', 'y,1/2-x,3/4-z', '1/2-x,1/2+z,1/2+y', 'z,1/2-x,1/2+y',
        #                         '3/4+y,1/4-x,3/4+z', '1/2-z,1/2-y,x', '3/4-z,1/4-y,1/4+x', 'x,1/2-z,1/2+y', 'x,1/4-y,1/4-z', '-y,1/2+x,-z', '3/4-y,3/4-z,1/4-x', '1/2+x,-z,1/2+y', '-y,x,3/4+z', 'x-y,x,1/2-z', '-x,1/2-z,1/2-y', '-y,-z,-x',
        #                         '3/4-y,1/4-z,1/4-x', 'x-y,x,1/2+z', 'x-y,x,-z', '3/4-y,1/4-z,3/4-x', '1/2-x,z,-y', '1/2+z,-x,1/2-y', '-x,3/4+y,3/4+z', '1/2-x,1/4+y,3/4+z', '1/4+z,3/4+x,3/4-y', '1/4-y,3/4-z,3/4-x', '3/4-x,1/2+y,3/4-z',
        #                         '-x,-x+y,1/6-z', '1/4-x,y,1/4-z', 'y,1/2+x,1/4+z', '-z,y,1/2-x', '-x,z,1/2-y', '1/2-y,1/2-x,1/2-z', '1/4+x,1/4+z,1/4-y', '1/4-z,3/4+x,1/4+y', '2/3+x,1/3+y,1/3+z', '1/4-z,1/4+x,3/4+y', '3/4+x,1/4-z,3/4-y',
        #                         '1/4+y,1/4+x,-z', '1/4+y,1/4+x,1/2-z', '1/4+x,-y,1/4+z', '3/4-x,1/4+z,3/4+y', '-y,-x,1/6-z', '1/2-y,1/2+x,3/4+z', 'x,-z,-y', '-x,-z,1/2+y', '1/3-x,2/3-x+y,2/3-z', 'y,x,2/3-z', '-y,1/2-x,1/4-z',
        #                         '-y,1/2+x,1/2-z', '1/4-y,3/4+x,1/4+z', '-x+y,y,2/3-z', '-z,-y,1/2+x', '3/4-x,3/4+z,1/4-y', '-z,-x,y', '1/4+z,1/4+x,1/4-y', '3/4-x,3/4-z,y', '3/4+x,1/4+y,3/4-z', '1/2-x,z,1/2+y', '1/2-y,3/4+x,1/4+z',
        #                         '3/4+z,1/4+y,1/2-x', '3/4+z,3/4-x,3/4+y', '1/4-z,1/2+x,3/4-y', 'x-y,x,2/3+z', 'x,1/2-y,3/4-z', '1/2-x,1/2+y,-z', '1/2+y,-x,1/2-z', '1/4-z,3/4+y,1/4+x', '1/2+y,1/2+x,1/2+z', '3/4+z,3/4+x,3/4-y',
        #                         '1/2+z,y,1/2-x', '1/2-x,y,1/4-z', '3/4-y,1/4-x,3/4+z', '3/4+y,3/4+z,3/4-x', '1/2-x,-z,-y', '3/4+z,-x,3/4+y', '-y,1/2-x,1/4+z', '1/3-y,2/3-x,1/6+z', '1/4-x,3/4+y,3/4+z', 'z,1/2+x,1/2-y', '1/2-x,-y,3/4-z',
        #                         '2/3+y,1/3-x+y,5/6-z', '3/4-x,1/4-z,1/2+y', '-x,3/4+z,1/4+y', '1/4+x,3/4+y,1/2-z', '1/2-x,3/4-y,1/2+z', '1/4+y,3/4-x,3/4+z', '1/4+x,1/4+z,1/2-y', 'x,1/2+z,1/2-y', '1/2-x,1/2+y,1/2-z', 'x,3/4-y,3/4-z',
        #                         '3/4+x,1/4+z,1/4-y', '1/4+z,1/4+y,-x', '3/4-z,y,3/4-x', '1/2-z,1/2-x,-y', '1/2+y,1/2-x,-z', '1/2+z,1/4-y,3/4-x', '3/4-z,1/4+x,1/4+y', '-z,1/2-y,1/2-x', 'z,1/2+y,-x', '1/2+z,1/2+x,1/2-y', '1/4+x,3/4-z,3/4+y',
        #                         '1/4+y,3/4-x,1/4-z', 'x,3/4-z,1/4-y', '1/2+y,1/2+z,x', '1/2-z,1/2-x,y', '1/4-y,1/4+z,3/4+x', 'x,z,1/2+y', '1/2-z,1/2-x,1/2-y', '-x,y,-z', '3/4+y,3/4-z,1/4+x', '1/4-x,z,3/4-y', '-x,y,1/2-z', '1/2-y,z,1/2+x',
        #                         'x,1/2-y,-z', '1/2-y,x,-z', '1/4+z,1/2-y,1/4+x', '3/4+y,1/4+z,1/4-x', '3/4+z,1/4+x,1/2-y', '1/4-x,3/4+y,1/4+z', '3/4+x,1/4-y,1/4+z', '-y,-z,x', '1/4-z,3/4-x,3/4-y', '1/2+z,-x,1/2+y', '1/2-y,-x,3/4-z',
        #                         'y,1/2-z,1/2-x', '3/4+x,1/4-y,3/4+z', '3/4+y,1/4-x,1/4-z', '1/2-y,x,1/2-z', 'x-y,x,5/6+z', '1/2+z,1/2-y,x', '-y,-x,-z', '1/3-x,2/3-x+y,1/6-z', '-x,3/4+z,3/4+y', '3/4+x,1/4+z,-y', '1/2-y,3/4+x,3/4+z',
        #                         '1/4-y,x,1/4-z', '1/2+x,y,1/2-z', '-x,y,1/2+z', '1/4-y,3/4+z,3/4+x', '3/4-y,1/4-x,3/4-z', 'x-y,-y,2/3-z', '3/4-y,1/2+z,1/4-x', '1/4-y,1/4-x,1/2+z', '-y,z,-x', '1/2-z,1/2+x,-y', '3/4-z,y,1/4-x',
        #                         '3/4+z,3/4+y,3/4+x', '-y,1/2-z,x', '1/2+y,1/2-x,3/4+z', 'y,-x+y,-z', '1/2-x,3/4+y,1/4+z', '2/3+x-y,1/3+x,5/6-z', '1/2+x,-z,1/2-y', '1/2+z,1/2-y,1/2-x', 'x,z,1/2-y', '3/4-x,3/4-y,1/4-z', '1/2-z,1/2+y,1/2+x',
        #                         '3/4-x,1/4-y,1/4-z', '1/2+z,3/4-x,1/4-y', '1/2-x,1/4+z,1/4+y', '1/2+x,1/4-y,3/4-z', 'x,1/2-y,1/4-z', '1/2+x,y,3/4-z', 'z,y,1/2-x', '1/2+y,1/2-x,1/2-z', 'y,x,3/4-z', '-z,1/4+x,1/4+y', '3/4+z,-y,3/4+x',
        #                         '3/4-y,3/4-z,3/4-x', '1/2+x,1/2+y,z', '1/2+z,1/2-x,1/2+y', 'x,y,z', '1/4-x,z,1/4-y', '1/2+y,x,-z', '3/4-z,3/4-y,3/4-x', 'y,1/2-x,z', '1/4+y,3/4+x,-z', '-x+y,-x,-z', '1/2-y,-x,1/2-z', '1/2-y,1/2+x,1/4+z',
        #                         '1/2-x,3/4+z,1/4+y', '3/4+x,3/4+y,1/4-z', 'z,1/2-y,x', '1/2-z,1/2+y,-x', '1/2+y,x,1/2-z', '1/4-z,3/4-y,x', '3/4-x,3/4+z,1/4+y', '-x+y,y,1/2-z', '1/2+y,-z,x', '-x+y,y,-z', '1/2+z,1/2+y,x', '1/2+y,1/2-x,1/4+z',
        #                         '1/2-x,1/2+y,3/4-z', '1/4+z,3/4+y,3/4-x', '-y,1/2-x,-z', '1/2-y,1/4+x,1/4+z', 'x,z,y', '-z,x,y', '1/4+z,1/4+y,1/4+x', '1/2-x,-y,1/2+z', '-x,1/2-y,1/4-z', '-x,-x+y,2/3-z', '1/2-y,1/2-x,-z', '1/2-y,1/2-x,z',
        #                         '1/4+z,1/2-x,3/4+y', '3/4+z,1/4-y,3/4-x', '1/4+y,3/4+x,1/2-z', '1/4+z,-x,1/4+y', '-y,x,1/2+z', '1/2-z,-y,-x', '1/4-z,1/2+y,1/4-x', 'y,1/2+z,1/2+x', '1/4+y,1/4+z,-x', '1/3+x-y,2/3-y,1/6-z',
        #                         '1/2+x,1/2+y,1/2-z', '1/4+y,3/4+z,1/4-x', 'y,1/2-x,1/2+z', '1/2-y,1/2+x,z', '1/2+z,1/2+x,-y', '3/4+x,3/4+z,1/2-y', '1/2+y,-x,3/4+z', '1/4+x,3/4-y,3/4+z', '2/3-x,1/3-x+y,5/6-z', '3/4+z,-y,1/4+x',
        #                         '1/4-y,3/4+z,1/4+x', '-z,y,1/2+x', '3/4+x,1/4+z,3/4-y', '3/4+y,1/4+z,3/4-x', '1/2-y,-z,1/2-x', '1/2-y,1/2-z,-x', '1/2+y,1/2-z,1/2-x', '2/3-x+y,1/3+y,1/3+z', '1/4+y,3/4-z,3/4+x', '1/2-x,1/4+z,3/4+y',
        #                         '-y,1/2+x,1/4-z', '1/2+y,-z,1/2-x', '-x,1/2+z,-y', '3/4+y,1/4-z,1/4+x', 'x,-y,1/2+z', '2/3+y,1/3-x+y,1/3-z', 'z,1/4-x,1/4-y', '1/2-z,y,-x', 'z,1/4-y,3/4-x', '1/4+y,1/4+z,3/4-x', '1/2-x,y,1/2+z',
        #                         '-z,3/4+x,3/4+y', '1/4+z,1/2-y,3/4+x', '1/2+z,1/2-x,-y', '1/2+y,1/4-z,3/4-x', '1/2+y,x,z', 'y,x,1/2-z', '-x,1/2-z,1/2+y', '3/4+y,1/4+x,1/2-z', '-y,1/2-x,3/4-z', '1/2-z,-y,x', 'y,-x+y,z', '-y,z,x',
        #                         '3/4-z,3/4+x,3/4+y', 'z,x,-y', 'y,x,-z', '3/4+z,3/4-y,1/4+x', '-y,1/2-z,1/2-x', '3/4-x,3/4-z,3/4-y', 'y,1/2+x,-z', '1/4+x,3/4+y,1/4-z', '2/3-x+y,1/3-x,1/3+z', '-x+y,y,z', '1/4+x,1/4+y,1/4-z', 'y,1/4-x,3/4-z',
        #                         '1/4-y,1/4-z,3/4-x', '1/4-y,3/4+x,3/4+z', 'x,1/4-z,3/4-y', '3/4-y,3/4+z,3/4+x', 'x,1/2-y,z', '1/4-x,1/4-y,1/4-z', '1/2-z,1/2+x,1/2+y', '1/4+y,3/4+x,3/4-z', '1/3-x,2/3-y,1/6-z', '1/4-z,1/4-x,y',
        #                         '1/4+z,3/4-x,3/4+y', '1/2+y,1/2+x,3/4-z', 'z,1/2-y,-x', '3/4-z,1/2+y,3/4-x', '1/2+y,x,3/4-z', '-y,1/4+x,3/4+z', 'y,-z,-x', '3/4-z,3/4-x,y', '-x+y,-x,z', '-x,-z,-y', '1/2+x,-y,z', '1/4-y,3/4-x,z',
        #                         '1/4-x,3/4-y,3/4-z', '-y,3/4+x,1/4+z', '1/2+y,-z,1/2+x', 'z,y,1/2+x', '2/3-x+y,1/3+y,5/6+z', '3/4-x,1/4-z,3/4-y', '-z,x,1/2+y', '1/2-z,-y,1/2-x', '-z,1/2+y,1/2-x', 'y,-z,x', '3/4+y,-x,1/4+z',
        #                         '3/4-z,1/4-y,1/2+x', 'y,-x+y,2/3+z', '-y,x,1/2-z', '-y,x-y,1/3+z', '3/4+x,1/4+y,1/4-z', 'y,x,1/2+z', '1/2-y,x,1/4+z', 'y,1/2+x,1/2+z', '3/4-y,3/4+x,3/4-z', '1/2+y,-x,1/2+z', '1/2-z,1/2-y,1/2-x',
        #                         '1/2-z,1/4+y,1/4+x', 'x-y,x,z', '-x+y,-x,2/3+z', 'z,1/2-y,1/2-x', 'z,3/4-y,3/4-x', '3/4+z,1/2-x,1/4+y', '-y,1/4+x,1/4+z', '1/2+x,1/4-z,3/4-y', 'y,-x+y,1/2+z', '3/4-z,1/2+y,1/4-x', '1/2-x,1/2-y,1/2-z',
        #                         '1/2+x,1/2+y,-z', '3/4-z,3/4-y,1/2+x', '2/3+x,1/3+x-y,1/3+z', '1/4-x,1/2+z,1/4-y', '-y,1/2-x,1/2-z', '3/4-x,3/4+y,1/4+z', '1/2+x,3/4-z,1/4-y', '1/2+x,1/4-z,1/4-y', '1/2-z,3/4+x,1/4+y', '1/2+z,1/2+x,1/2+y',
        #                         '1/2+z,-y,1/2+x', '1/2-z,1/4+x,3/4+y', '-y,x,z', '1/4+z,3/4+y,-x', '-y,-x,1/2-z', '-x,1/2-y,1/2+z', 'z,-y,-x', '1/4+x,1/4+z,-y', '3/4+y,3/4+z,-x', '-x+y,-x,1/3+z', '3/4+y,3/4-x,1/4+z', '1/3-x+y,2/3-x,2/3+z',
        #                         '3/4-x,1/2+z,1/4-y', '1/4+x,1/4+z,1/4+y', '1/2+z,1/2-x,1/2-y', '-y,x,1/4+z', '-z,1/2+x,1/2+y', '1/4+y,1/4-x,1/4+z', '1/2+x,1/2+z,y', 'y,-z,1/2-x', '1/2+y,1/2+z,1/2-x', 'y,1/2+x,1/2-z', '1/2+z,1/2-x,y',
        #                         '1/4-x,1/2+y,1/4-z', '1/4-z,3/4-y,3/4+x', '-x,z,1/2+y', '1/4+z,3/4-x,1/4+y', '1/4+z,3/4-y,3/4+x', '-z,1/4+y,1/4+x', 'x,1/2+y,1/4-z', 'x,1/2+y,1/2-z', '3/4+x,3/4-y,3/4+z', '1/2+y,x,1/4-z', '-x,1/2+y,1/2-z',
        #                         '1/4+x,3/4+z,3/4-y', '-x,-x+y,1/2-z', '1/4+x,1/4-z,1/4+y', '1/4-x,3/4-z,3/4+y', 'y,-x,z']
        # self.unique_descriptions = ['2-fold screw axis with direction [0, 1, -1] at 0, y+1/4, -y with screw component [0, -1/4, 1/4]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/2, 1/2]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 3/8, y, 5/8 with screw component [0, 1/4, 0]', '3-fold rotoinversion axis with direction [1, 1, 1] at x+5/8, x+1/8, x+1/8 with inversion at [5/8, 1/8, 1/8]',
        #                             '2-fold screw axis with direction [0, 1, 0] at 0, y, 0 with screw component [0, 1/2, 0]', '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 5/6]',
        #                             '3-fold screw axis with direction [1, -1, -1] at x+5/12, -x+1/6, -x with screw component [-1/3, 1/3, 1/3]', 'Glide plane perpendicular to [0, 0, 1] with glide component [3/4, 3/4, 0]',
        #                             '2-fold rotation axis with direction [1, 0, 0] at x, 0, 1/6', '2-fold screw axis with direction [0, 1, 1] at 1/4, y, y with screw component [0, 3/4, 3/4]',
        #                             '3-fold screw axis with direction [1, -1, -1] at x+2/3, -x+2/3, -x with screw component [-1/6, 1/6, 1/6]', '2-fold rotation axis with direction [1, 1, 0] at x, x, 1/8',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 0, 0 with screw component [1/2, 0, 0]', '4-fold screw axis with direction [0, 1, 0] at 1/2, y, 0 with screw component [0, 1/2, 0]',
        #                             '2-fold rotation axis with direction [0, 1, -1] at 1/8, y+3/4, -y', 'Glide plane perpendicular to [1, -1, 0] with glide component [3/4, 3/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/8, -x+5/8, -x+1/8 with inversion at [3/8, 5/8, 1/8]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 3/8, 7/8, z+1/8 with inversion at [3/8, 7/8, 1/8]', '4-fold screw axis with direction [1, 0, 0] at x, 0, 0 with screw component [1/2, 0, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 3/8, 1/8, z with screw component [0, 0, 1/4]', 'Inversion at [1/8, 3/8, 1/8]', '2-fold rotation axis with direction [1, -1, 0] at x+1/4, -x, 3/8',
        #                             '2-fold rotation axis with direction [2, 1, 0] at 2x, x, 1/3', '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 1/4, z+1/8 with inversion at [0, 1/4, 1/8]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 3/4, y, 1/2 with screw component [0, 3/4, 0]', '4-fold screw axis with direction [1, 0, 0] at x, 7/8, 5/8 with screw component [3/4, 0, 0]',
        #                             '2-fold screw axis with direction [1, 0, -1] at x+1/2, 3/8, -x with screw component [1/4, 0, -1/4]', '4-fold rotation axis with direction [1, 0, 0] at x, 0, 0',
        #                             '3-fold rotation axis with direction [1, 1, -1] at x, x, -x', '4-fold screw axis with direction [0, 1, 0] at 3/4, y, 0 with screw component [0, 1/4, 0]',
        #                             '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/6]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+3/4, -x with inversion at [3/4, 3/4, 0]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 1/6, 5/12 with screw component [1/2, 0, 0]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 3/4]',
        #                             '4-fold rotation axis with direction [0, 1, 0] at 0, y, 0', '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 1/4, z+3/8 with inversion at [0, 1/4, 3/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 1/2]', 'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+5/8, x+5/8, -x+7/8 with inversion at [5/8, 5/8, 7/8]', 'Glide plane perpendicular to [1, -1, 0] with glide component [3/4, 3/4, 3/4]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 3/8, 3/8, z+1/8 with inversion at [3/8, 3/8, 1/8]', '4-fold rotation axis with direction [1, 0, 0] at x, 3/4, 1/4',
        #                             'Mirror plane perpendicular to [0, 1, 1]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/8, y+1/8, 1/8 with inversion at [1/8, 1/8, 1/8]',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/2, 0]', '3-fold rotoinversion axis with direction [1, 1, 1] at x+3/8, x+7/8, x+3/8 with inversion at [3/8, 7/8, 3/8]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 1/4]', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/4, 1/4, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 0, y, 1/2 with inversion at [0, 0, 1/2]', 'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, 1/4, -1/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/8, 3/8 with screw component [3/4, 0, 0]', '3-fold rotation axis with direction [1, 1, 1] at x, x, x',
        #                             '4-fold screw axis with direction [0, 0, 1] at 3/4, 1/4, z with screw component [0, 0, 1/4]', '3-fold rotoinversion axis with direction [0, 0, 1] at 1/3, 2/3, z+1/6 with inversion at [1/3, 2/3, 1/6]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 1/2]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+7/8, x+7/8, -x+1/8 with inversion at [7/8, 7/8, 1/8]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [0, 0, 1/2]', '3-fold rotation axis with direction [1, -1, -1] at x+1/2, -x, -x',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/4, y+1/4, y with screw component [0, 1/4, 1/4]', '4-fold screw axis with direction [0, 1, 0] at 3/8, y, 1/8 with screw component [0, 3/4, 0]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 3/8 with screw component [1/4, 1/4, 0]', 'Inversion at [1/8, 1/8, 3/8]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 7/8, 5/8 with screw component [1/4, 0, 0]', 'Glide plane perpendicular to [1, 0, -1] with glide component [1/4, 1/2, 1/4]',
        #                             '2-fold rotation axis with direction [0, 1, -1] at 1/8, y+1/4, -y', '4-fold rotation axis with direction [0, 1, 0] at 0, y, 1/2', 'Glide plane perpendicular to [1, 0, 1] with glide component [0, 1/2, 0]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 0, 1/4 with screw component [1/2, 0, 0]', '4-fold screw axis with direction [1, 0, 0] at x, 3/8, 5/8 with screw component [1/4, 0, 0]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 0, y, 0 with screw component [0, 1/2, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 5/8, 1/8 with inversion at [3/8, 5/8, 1/8]',
        #                             '3-fold screw axis with direction [1, -1, 1] at x+5/6, -x+1/3, x with screw component [1/3, -1/3, 1/3]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+7/8, x+3/8, x+3/8 with inversion at [7/8, 3/8, 3/8]', '4-fold screw axis with direction [1, 0, 0] at x, 1/2, 0 with screw component [1/2, 0, 0]',
        #                             '3-fold screw axis with direction [1, -1, 1] at x+5/6, -x+7/12, x with screw component [1/3, -1/3, 1/3]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+3/4, -x+3/4 with inversion at [0, 3/4, 3/4]', 'Inversion at [0, 1/4, 1/4]',
        #                             '3-fold screw axis with direction [0, 0, 1] at 0, 1/3, z with screw component [0, 0, 2/3]', '3-fold rotation axis with direction [1, -1, 1] at x, -x, x',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [0, -1/4, 1/4]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 1/2]',
        #                             'Glide plane perpendicular to [0, 1, -1] with glide component [0, 1/2, 1/2]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 3/4, 1/4 with inversion at [1/4, 3/4, 1/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/8, 3/8 with screw component [1/4, 0, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 0, 0 with inversion at [1/4, 0, 0]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [0, 0, 1/2]', '2-fold screw axis with direction [1, 0, 1] at x+3/4, 0, x with screw component [1/4, 0, 1/4]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 1/4]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 3/4, 3/4]', 'Mirror plane perpendicular to [1, 0, 1]',
        #                             '2-fold rotation axis with direction [1, -1, 0] at x, -x, 1/3', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/4, 1/4]',
        #                             '3-fold rotation axis with direction [1, -1, -1] at x+1/2, -x+1/2, -x', '3-fold screw axis with direction [1, 1, -1] at x+1/3, x+2/3, -x with screw component [1/6, 1/6, -1/6]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/8, 7/8, z with screw component [0, 0, 3/4]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 1/4]',
        #                             'Mirror plane perpendicular to [1, -1, 0]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/2, -x, x with inversion at [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [1, 0, -1] with glide component [1/2, 1/2, 1/2]', '3-fold rotoinversion axis with direction [1, -1, 1] at x, -x+1/2, x+1/2 with inversion at [0, 1/2, 1/2]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 0, y, 1/6', '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/3]',
        #                             '2-fold screw axis with direction [0, 1, 0] at 0, y, 3/8 with screw component [0, 1/2, 0]', '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+3/4, -x+1/4 with inversion at [0, 3/4, 1/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/4, 1/2 with screw component [3/4, 0, 0]', '3-fold screw axis with direction [1, 1, 1] at x, x, x with screw component [1/2, 1/2, 1/2]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 3/8, y+1/8, 3/8 with inversion at [3/8, 1/8, 3/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/8, -x+1/8, x+7/8 with inversion at [3/8, 1/8, 7/8]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x, 1/8, x with screw component [1/4, 0, 1/4]', '4-fold rotoinversion axis with direction [0, 0, 1] at 7/8, 3/8, z+3/8 with inversion at [7/8, 3/8, 3/8]',
        #                             'Mirror plane perpendicular to [1, 2, 0]', '3-fold rotation axis with direction [1, -1, -1] at x+3/4, -x+1/2, -x', '2-fold rotation axis with direction [1, -1, 0] at x+3/4, -x, 3/8',
        #                             'Glide plane perpendicular to [1, 0, -1] with glide component [1/4, 0, 1/4]', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/2, 0, 0]',
        #                             '2-fold rotation axis with direction [1, 0, 0] at x, 1/4, 1/4', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/8, 1/8, z+1/8 with inversion at [1/8, 1/8, 1/8]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x, 0, x with screw component [1/4, 0, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/8, -x+7/8, -x+3/8 with inversion at [1/8, 7/8, 3/8]', 'Centring vector [1/2, 1/2, 1/2]', 'Mirror plane perpendicular to [1, 1, 0]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/4, y, 3/4 with screw component [0, 1/2, 0]', '4-fold rotation axis with direction [0, 0, 1] at 3/4, 1/4, z',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/4, y, 1/4 with screw component [0, 1/2, 0]', '4-fold rotation axis with direction [1, 0, 0] at x, 1/4, 1/4',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 5/8, 1/8, z+1/8 with inversion at [5/8, 1/8, 1/8]', 'Glide plane perpendicular to [1, 1, 0] with glide component [1/6, -1/6, 5/6]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [1/3, 1/6, 2/3]', '4-fold screw axis with direction [1, 0, 0] at x, 1/4, 3/4 with screw component [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [0, 0, 1] with glide component [1/4, 3/4, 0]', '4-fold screw axis with direction [0, 1, 0] at 7/8, y, 5/8 with screw component [0, 3/4, 0]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 5/8, y, 7/8 with screw component [0, 3/4, 0]', '3-fold screw axis with direction [1, 1, -1] at x+1/3, x+1/6, -x with screw component [1/6, 1/6, -1/6]',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 1/4]', '4-fold screw axis with direction [0, 0, 1] at 5/8, 3/8, z with screw component [0, 0, 3/4]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 1/6, 1/6 with screw component [1/2, 0, 0]', '2-fold screw axis with direction [0, 0, 1] at 1/8, 3/8, z with screw component [0, 0, 1/2]',
        #                             '3-fold rotation axis with direction [1, -1, 1] at x, -x+3/4, x', '2-fold screw axis with direction [1, 0, 0] at x, 0, 0 with screw component [1/2, 0, 0]',
        #                             '6-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z with inversion at [0, 0, 0]', '3-fold rotation axis with direction [1, 1, -1] at x+1/4, x+1/4, -x',
        #                             '6-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z+1/4 with inversion at [0, 0, 1/4]', '3-fold rotation axis with direction [1, 1, -1] at x+1/2, x+1/2, -x',
        #                             '4-fold screw axis with direction [0, 1, 0] at 0, y, 3/4 with screw component [0, 1/4, 0]', 'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 3/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 3/8, 1/8 with screw component [3/4, 0, 0]', '4-fold rotoinversion axis with direction [0, 1, 0] at 0, y+1/4, 0 with inversion at [0, 1/4, 0]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 0, 3/8 with screw component [1/2, 0, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 1/8, 1/8 with inversion at [3/8, 1/8, 1/8]',
        #                             '3-fold rotation axis with direction [1, -1, 1] at x, -x+1/2, x', '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/2, -x, x+1/2 with inversion at [1/2, 0, 1/2]',
        #                             '2-fold screw axis with direction [1, 0, -1] at x+1/4, 0, -x with screw component [-1/4, 0, 1/4]', 'Inversion at [3/8, 1/8, 3/8]',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, 0, 0]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x, x+3/4 with inversion at [3/4, 0, 3/4]',
        #                             '2-fold rotation axis with direction [1, 1, 0] at x, x, 1/6', '2-fold rotation axis with direction [0, 0, 1] at 0, 1/4, z', 'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 3/4]', 'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 0]',
        #                             '2-fold rotation axis with direction [1, -1, 0] at x, -x, 3/8', '3-fold rotoinversion axis with direction [1, 1, -1] at x+7/8, x+3/8, -x+1/8 with inversion at [7/8, 3/8, 1/8]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 3/4, z with inversion at [1/4, 3/4, 0]', '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/2, z with screw component [0, 0, 1/4]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 3/8, y+3/8, 7/8 with inversion at [3/8, 3/8, 7/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 1/8, 5/8 with inversion at [3/8, 1/8, 5/8]', 'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 3/4]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 0, 1/8 with screw component [1/2, 0, 0]', '4-fold screw axis with direction [0, 1, 0] at 1/2, y, 3/4 with screw component [0, 1/4, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+7/8, -x+1/8, x+3/8 with inversion at [7/8, 1/8, 3/8]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+1/6, x, 5/12 with screw component [1/2, 1/2, 0]', '2-fold screw axis with direction [0, 1, 0] at 1/8, y, 3/8 with screw component [0, 1/2, 0]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 3/4, y, 1/4 with screw component [0, 1/2, 0]', '3-fold screw axis with direction [1, -1, -1] at x+1/6, -x+1/6, -x with screw component [-1/3, 1/3, 1/3]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 1/4]', '2-fold screw axis with direction [0, 1, 0] at 1/4, y, 1/8 with screw component [0, 1/2, 0]',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [1/3, 2/3, 1/6]', '4-fold screw axis with direction [0, 0, 1] at 3/8, 1/8, z with screw component [0, 0, 3/4]',
        #                             '2-fold screw axis with direction [0, 1, -1] at 1/4, y+1/4, -y with screw component [0, -1/4, 1/4]', '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 1/8 with screw component [1/2, 1/2, 0]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 1/2]', '4-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 3/4]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 1/8 with screw component [1/4, -1/4, 0]', 'Mirror plane perpendicular to [1, 0, 1]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 7/8, y, 5/8 with screw component [0, 1/4, 0]', '2-fold rotation axis with direction [1, -1, 0] at x, -x, 1/8',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 7/8, 1/8 with screw component [3/4, 0, 0]', 'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/4, -x+3/4, -x+3/4 with inversion at [1/4, 3/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/8, x+7/8, -x+1/8 with inversion at [3/8, 7/8, 1/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/2, -x+3/4, -x+3/4 with inversion at [1/2, 3/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+3/8, x+3/8, x+3/8 with inversion at [3/8, 3/8, 3/8]', '2-fold rotation axis with direction [0, 0, 1] at 3/8, 3/8, z',
        #                             '2-fold rotation axis with direction [1, 0, -1] at x+1/4, 1/8, -x', '2-fold screw axis with direction [1, 0, 0] at x, 1/4, 1/4 with screw component [1/2, 0, 0]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/2, -x, 3/8 with screw component [-1/4, 1/4, 0]', '2-fold rotation axis with direction [2, 1, 0] at 2x, x, 1/12',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/8, y, y with screw component [0, 3/4, 3/4]', 'Glide plane perpendicular to [1, -1, 0] with glide component [1/2, 1/2, 0]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [2/3, 1/3, 5/6]', 'Glide plane perpendicular to [0, 0, 1] with glide component [0, 1/2, 0]',
        #                             '3-fold screw axis with direction [1, 1, -1] at x+7/12, x+5/12, -x with screw component [1/3, 1/3, -1/3]', '4-fold screw axis with direction [1, 0, 0] at x, 3/8, 5/8 with screw component [3/4, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+7/8, -x+5/8, x+7/8 with inversion at [7/8, 5/8, 7/8]', '2-fold rotation axis with direction [1, 0, 0] at x, 0, 1/4',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 0]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 0]',
        #                             '3-fold screw axis with direction [1, -1, -1] at x+7/12, -x+5/6, -x with screw component [-1/3, 1/3, 1/3]', '2-fold rotation axis with direction [1, 0, 1] at x, 0, x',
        #                             '4-fold screw axis with direction [0, 0, 1] at 0, 3/4, z with screw component [0, 0, 1/4]', '2-fold rotation axis with direction [1, 0, 0] at x, 0, 0',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 3/4]', 'Mirror plane perpendicular to [0, 0, 1]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 1/4 with screw component [1/2, 1/2, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 1/8, 5/8 with inversion at [1/8, 1/8, 5/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x+1/2, x+3/4 with inversion at [3/4, 1/2, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/8, -x+1/8, x+7/8 with inversion at [3/8, 1/8, 7/8]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y+1/4, 3/4 with inversion at [1/4, 1/4, 3/4]',
        #                             '3-fold screw axis with direction [1, -1, 1] at x+1/6, -x+5/12, x with screw component [1/3, -1/3, 1/3]', '4-fold screw axis with direction [1, 0, 0] at x, 1/4, 1/4 with screw component [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [1/2, 0, 0]', 'Glide plane perpendicular to [0, 0, 1] with glide component [3/4, 3/4, 0]', 'Centring vector [1/3, 2/3, 2/3]',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/4, 3/4]', '2-fold screw axis with direction [1, 0, 1] at x+1/4, 0, x with screw component [1/4, 0, 1/4]',
        #                             '4-fold rotation axis with direction [0, 0, 1] at 1/4, 3/4, z', '3-fold screw axis with direction [1, 1, 1] at x+1/6, x+1/3, x with screw component [1/3, 1/3, 1/3]',
        #                             '2-fold rotation axis with direction [0, 1, 1] at 0, y, y', '3-fold rotoinversion axis with direction [1, 1, 1] at x, x+1/2, x with inversion at [0, 1/2, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/2, z with screw component [0, 0, 3/4]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 1/8, 1/8 with inversion at [3/8, 1/8, 1/8]',
        #                             '2-fold rotation axis with direction [1, 0, -1] at x, 1/4, -x', '2-fold rotation axis with direction [0, 0, 1] at 1/4, 0, z',
        #                             '4-fold screw axis with direction [0, 0, 1] at 3/8, 5/8, z with screw component [0, 0, 3/4]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/2, 3/4, z+3/8 with inversion at [1/2, 3/4, 3/8]',
        #                             'Inversion at [1/6, 1/3, 1/3]', '4-fold rotation axis with direction [1, 0, 0] at x, 0, 0', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/8, y+3/8, 1/8 with inversion at [1/8, 3/8, 1/8]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [1/6, -1/6, 1/3]', '2-fold screw axis with direction [0, 1, 1] at 0, y+1/4, y with screw component [0, 1/4, 1/4]',
        #                             'Inversion at [3/8, 3/8, 3/8]', '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z+1/4 with inversion at [0, 0, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/4, x+3/4, -x with inversion at [1/4, 3/4, 0]', 'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, 0, 0]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 1/4, 0 with screw component [1/2, 0, 0]', '3-fold rotoinversion axis with direction [0, 0, 1] at 2/3, 1/3, z+1/12 with inversion at [2/3, 1/3, 1/12]',
        #                             '4-fold rotation axis with direction [0, 0, 1] at 1/4, 1/4, z', 'Glide plane perpendicular to [1, 0, -1] with glide component [1/2, 0, 1/2]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x, 1/4, x with screw component [1/2, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+7/8, x+7/8, -x+1/8 with inversion at [7/8, 7/8, 1/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x, 0, 0 with inversion at [0, 0, 0]', '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 0 with screw component [1/2, 1/2, 0]',
        #                             '2-fold rotation axis with direction [1, -1, 0] at x+3/4, -x, 1/8', '4-fold rotation axis with direction [0, 1, 0] at 1/4, y, 3/4',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 0 with screw component [1/2, 1/2, 0]', '3-fold screw axis with direction [1, -1, -1] at x+1/3, -x+5/6, -x with screw component [-1/3, 1/3, 1/3]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 3/4, z+1/8 with inversion at [1/4, 3/4, 1/8]', '4-fold screw axis with direction [0, 1, 0] at 1/4, y, 0 with screw component [0, 1/4, 0]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 0, y, y with screw component [0, 1/2, 1/2]', '2-fold screw axis with direction [0, 1, -1] at 3/8, y+1/2, -y with screw component [0, 1/4, -1/4]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 0, 1/4 with inversion at [3/8, 0, 1/4]', 'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 3/4]',
        #                             'Mirror plane perpendicular to [1, 0, 0]', 'Inversion at [1/3, 1/6, 1/6]', '2-fold screw axis with direction [1, 1, 0] at x+1/6, x, 1/6 with screw component [1/2, 1/2, 0]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 0 with screw component [3/4, 3/4, 0]', 'Mirror plane perpendicular to [1, 1, 0]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/2, y, 3/4 with screw component [0, 3/4, 0]', '4-fold screw axis with direction [0, 1, 0] at 1/4, y, 1/2 with screw component [0, 3/4, 0]',
        #                             '2-fold screw axis with direction [1, 0, -1] at x+1/2, 3/8, -x with screw component [-1/4, 0, 1/4]', '4-fold rotoinversion axis with direction [1, 0, 0] at x, 1/2, 0 with inversion at [0, 1/2, 0]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 1/2]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/4, x+3/4, -x+3/4 with inversion at [1/4, 3/4, 3/4]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 1/4, y, 1/4', '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+3/4, -x+1/2 with inversion at [3/4, 3/4, 1/2]',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [-1/4, 1/4, 1/4]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+5/8, x+5/8, -x+3/8 with inversion at [5/8, 5/8, 3/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/8, 3/8, z with screw component [0, 0, 3/4]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/2, 0]',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 1/4]', '4-fold screw axis with direction [1, 0, 0] at x, 3/4, 1/2 with screw component [1/4, 0, 0]',
        #                             '4-fold rotation axis with direction [0, 1, 0] at 1/4, y, 1/4', '4-fold rotation axis with direction [0, 0, 1] at 1/2, 0, z',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+1/4, 1/4, x with screw component [1/2, 0, 1/2]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/2, y, 0 with inversion at [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [0, 1, -1] with glide component [0, 1/4, 1/4]', '4-fold rotoinversion axis with direction [0, 1, 0] at 7/8, y+1/8, 3/8 with inversion at [7/8, 1/8, 3/8]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/4, 1/4 with screw component [1/2, 0, 0]', '2-fold rotation axis with direction [1, 0, -1] at x+3/4, 1/8, -x',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+7/8, x+3/8, -x+1/8 with inversion at [7/8, 3/8, 1/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+1/4, -x+3/4 with inversion at [0, 1/4, 3/4]', '4-fold screw axis with direction [0, 0, 1] at 5/8, 7/8, z with screw component [0, 0, 3/4]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 2/3, 1/3, z+1/12 with inversion at [2/3, 1/3, 1/12]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/8, -x+7/8, -x+3/8 with inversion at [1/8, 7/8, 3/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+5/8, -x+7/8, -x+7/8 with inversion at [5/8, 7/8, 7/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 1/4, 3/4 with inversion at [1/4, 1/4, 3/4]', '2-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/2]',
        #                             '2-fold rotation axis with direction [2, 1, 0] at 2x, x, 1/4', '4-fold screw axis with direction [0, 0, 1] at 3/4, 1/4, z with screw component [0, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z+1/4 with inversion at [0, 0, 1/4]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 1/4, z with inversion at [1/4, 1/4, 0]',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/4, 1/4]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/2, 0, 1/2]',
        #                             '3-fold screw axis with direction [1, 1, 1] at x+2/3, x+5/6, x with screw component [1/3, 1/3, 1/3]', '4-fold screw axis with direction [0, 0, 1] at 1/4, 0, z with screw component [0, 0, 3/4]',
        #                             '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/6]', '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+3/4, -x+1/4 with inversion at [0, 3/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+3/4, -x+1/4 with inversion at [3/4, 3/4, 1/4]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z with inversion at [0, 0, 0]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/2, 0, 1/2]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 1/8 with screw component [1/2, 1/2, 0]', '2-fold rotation axis with direction [2, 1, 0] at 2x, x, 0',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/8, -x+1/8, -x+5/8 with inversion at [3/8, 1/8, 5/8]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 5/8, 3/8 with screw component [1/4, 0, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 3/8, 3/8, z+3/8 with inversion at [3/8, 3/8, 3/8]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 7/8, y, 1/8 with screw component [0, 3/4, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/8, 1/8, z+3/8 with inversion at [1/8, 1/8, 3/8]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 1/4, z+3/8 with inversion at [1/4, 1/4, 3/8]', 'Mirror plane perpendicular to [1, 1, 0]',
        #                             '3-fold rotation axis with direction [1, 1, 1] at x, x, x', '2-fold screw axis with direction [1, 1, 0] at x+5/6, x, 1/12 with screw component [1/2, 1/2, 0]', 'Centring vector [0, 1/2, 1/2]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 3/8, 7/8, z+3/8 with inversion at [3/8, 7/8, 3/8]', '4-fold rotoinversion axis with direction [0, 1, 0] at 0, y, 0 with inversion at [0, 0, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/8, 5/8, z+3/8 with inversion at [1/8, 5/8, 3/8]', '3-fold rotation axis with direction [1, -1, 1] at x, -x+1/4, x',
        #                             'Mirror plane perpendicular to [0, 1, 1]', '3-fold rotation axis with direction [1, -1, -1] at x+3/4, -x, -x', '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/3]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 3/8, y, 1/8', '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 1/2, z+1/4 with inversion at [0, 1/2, 1/4]',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 1/2]', '4-fold screw axis with direction [1, 0, 0] at x, 5/8, 3/8 with screw component [3/4, 0, 0]',
        #                             '3-fold rotation axis with direction [1, -1, -1] at x, -x+1/2, -x', '2-fold screw axis with direction [1, 1, 0] at x, x, 1/8 with screw component [1/4, 1/4, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 3/4, z+1/8 with inversion at [0, 3/4, 1/8]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 1/4]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 1/2, 0 with inversion at [1/4, 1/2, 0]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y, 1/4 with inversion at [1/4, 0, 1/4]',
        #                             '3-fold screw axis with direction [1, -1, 1] at x+5/6, -x+1/3, x with screw component [1/6, -1/6, 1/6]', 'Mirror plane perpendicular to [2, 1, 0]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 1/4]', '3-fold rotation axis with direction [1, -1, 1] at x, -x+1/2, x', 'Inversion at [1/4, 1/4, 0]',
        #                             '6-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z+1/4 with inversion at [0, 0, 1/4]', '3-fold rotoinversion axis with direction [1, -1, 1] at x, -x, x+1/2 with inversion at [0, 0, 1/2]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 3/8, 3/8 with inversion at [1/8, 3/8, 3/8]', '2-fold rotation axis with direction [1, 0, -1] at x, 0, -x',
        #                             '2-fold screw axis with direction [0, 1, 0] at 0, y, 1/8 with screw component [0, 1/2, 0]', '3-fold rotation axis with direction [0, 0, 1] at 0, 0, z',
        #                             '2-fold rotation axis with direction [2, 1, 0] at 2x, x, 1/6', '2-fold screw axis with direction [0, 1, 1] at 1/8, y, y with screw component [0, 1/4, 1/4]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 1/4, 1/4 with inversion at [1/4, 1/4, 1/4]', '3-fold rotoinversion axis with direction [1, 1, -1] at x, x, -x with inversion at [0, 0, 0]',
        #                             'Mirror plane perpendicular to [0, 1, 1]', '2-fold rotation axis with direction [0, 1, 0] at 1/4, y, 3/8',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x, x+1/2, -x with inversion at [0, 1/2, 0]', '2-fold screw axis with direction [0, 1, 0] at 3/8, y, 1/8 with screw component [0, 1/2, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 3/8, 3/8, z+1/8 with inversion at [3/8, 3/8, 1/8]', '4-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/2]',
        #                             '2-fold screw axis with direction [1, 0, 0] at x, 1/4, 3/8 with screw component [1/2, 0, 0]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 0, 1/2]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 5/8, 1/8, z+3/8 with inversion at [5/8, 1/8, 3/8]', '4-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/4]',
        #                             'Centring vector [1/2, 0, 1/2]', '3-fold screw axis with direction [1, -1, -1] at x+1/6, -x+1/6, -x with screw component [-1/6, 1/6, 1/6]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 1/4 with screw component [3/4, 3/4, 0]', 'Glide plane perpendicular to [0, 1, -1] with glide component [1/2, 1/2, 1/2]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 0, 1/2 with inversion at [1/4, 0, 1/2]', '2-fold rotation axis with direction [0, 1, -1] at 1/4, y+1/2, -y',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x, x, x with inversion at [0, 0, 0]', '3-fold rotation axis with direction [1, -1, 1] at x, -x+3/4, x',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+3/4, 1/8, x with screw component [1/2, 0, 1/2]', '3-fold rotation axis with direction [1, -1, -1] at x+1/4, -x, -x',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+7/8, -x+5/8, x+7/8 with inversion at [7/8, 5/8, 7/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 5/8, 7/8, z with screw component [0, 0, 1/4]', '2-fold rotation axis with direction [0, 1, 0] at 3/8, y, 3/8',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/2, -x+1/2, x with inversion at [1/2, 1/2, 0]',
        #                             '2-fold screw axis with direction [0, 1, -1] at 0, y+1/4, -y with screw component [0, 1/4, -1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+3/8, x+3/8, x+7/8 with inversion at [3/8, 3/8, 7/8]', 'Glide plane perpendicular to [1, 1, 0] with glide component [0, 0, 1/2]',
        #                             'Mirror plane perpendicular to [1, 0, 1]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/8, x+5/8, -x+3/8 with inversion at [1/8, 5/8, 3/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x, 1/4, 1/4 with inversion at [0, 1/4, 1/4]', '4-fold screw axis with direction [0, 1, 0] at 0, y, 1/4 with screw component [0, 1/4, 0]',
        #                             '2-fold rotation axis with direction [1, 0, 0] at x, 0, 0', 'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 3/4]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 5/8, y+3/8, 1/8 with inversion at [5/8, 3/8, 1/8]', 'Glide plane perpendicular to [0, 0, 1] with glide component [3/4, 1/4, 0]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x, 1/8, x with screw component [3/4, 0, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+7/8, -x+5/8, -x+5/8 with inversion at [7/8, 5/8, 5/8]', 'Mirror plane perpendicular to [1, 0, -1]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/8, y+1/8, 1/8 with inversion at [1/8, 1/8, 1/8]', 'Mirror plane perpendicular to [0, 1, 0]',
        #                             '3-fold rotation axis with direction [1, -1, 1] at x+1/2, -x+3/4, x', '4-fold rotoinversion axis with direction [0, 1, 0] at 3/8, y+1/8, 7/8 with inversion at [3/8, 1/8, 7/8]',
        #                             '2-fold rotation axis with direction [0, 1, 1] at 1/4, y, y', '2-fold screw axis with direction [0, 1, 0] at 1/8, y, 1/4 with screw component [0, 1/2, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/2, -x+1/2, -x with inversion at [1/2, 1/2, 0]', 'Glide plane perpendicular to [1, 0, -1] with glide component [0, 1/2, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x, -x, x with inversion at [0, 0, 0]', '3-fold rotation axis with direction [1, 1, -1] at x, x+1/2, -x',
        #                             '2-fold rotation axis with direction [1, 2, 0] at x, 2x, 1/6', '2-fold screw axis with direction [1, 0, 0] at x, 1/4, 1/8 with screw component [1/2, 0, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 0, 1/4, z with screw component [0, 0, 1/4]', '3-fold rotation axis with direction [1, -1, 1] at x+1/2, -x+1/2, x',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x, 0, 0 with inversion at [0, 0, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 1/2, z with inversion at [0, 1/2, 0]', '2-fold rotation axis with direction [1, -1, 0] at x+1/4, -x, 1/8',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 1/4, z+1/4 with inversion at [1/4, 1/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+7/8, -x+5/8, -x+5/8 with inversion at [7/8, 5/8, 5/8]', 'Inversion at [0, 0, 1/4]',
        #                             '3-fold screw axis with direction [1, 1, -1] at x+1/3, x+1/6, -x with screw component [1/3, 1/3, -1/3]',
        #                             '3-fold screw axis with direction [1, 1, 1] at x+1/6, x+5/6, x with screw component [1/3, 1/3, 1/3]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 3/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/2, 3/4 with screw component [3/4, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+7/8, -x+1/8, x+3/8 with inversion at [7/8, 1/8, 3/8]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+7/8, x+7/8, -x+5/8 with inversion at [7/8, 7/8, 5/8]', 'Glide plane perpendicular to [1, 0, 0] with glide component [1/3, 2/3, 2/3]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 1/8 with screw component [3/4, 3/4, 0]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/6, 1/6, 2/3]',
        #                             'Mirror plane perpendicular to [1, 0, 0]', '2-fold rotation axis with direction [0, 1, 0] at 1/3, y, 1/6', '3-fold rotation axis with direction [1, 1, -1] at x+1/4, x+1/4, -x',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/8, -x+5/8, -x+1/8 with inversion at [3/8, 5/8, 1/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/2, 0, z with screw component [0, 0, 1/2]', 'Glide plane perpendicular to [0, 1, -1] with glide component [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [0, 1, -1] with glide component [1/2, 1/4, 1/4]', 'Glide plane perpendicular to [1, 0, 1] with glide component [-1/4, 1/2, 1/4]',
        #                             '3-fold rotation axis with direction [1, 1, -1] at x+3/4, x+3/4, -x', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y+3/8, 0 with inversion at [1/4, 3/8, 0]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x, 1/4, x with screw component [3/4, 0, 3/4]', '2-fold screw axis with direction [1, 1, 0] at x+5/6, x, 1/3 with screw component [1/2, 1/2, 0]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 3/8, 3/8 with inversion at [3/8, 3/8, 3/8]', '2-fold rotation axis with direction [1, 0, 0] at x, 1/3, 1/3',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 3/8, y+3/8, 3/8 with inversion at [3/8, 3/8, 3/8]', '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 3/4]',
        #                             'Glide plane perpendicular to [1, 2, 0] with glide component [0, 0, 1/2]', '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x, -x+1/2 with inversion at [0, 0, 1/2]',
        #                             '2-fold rotation axis with direction [0, 0, 1] at 1/4, 1/4, z', '3-fold screw axis with direction [1, 1, -1] at x+5/12, x+7/12, -x with screw component [1/3, 1/3, -1/3]',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [1/4, 3/4, -1/4]', '4-fold rotation axis with direction [0, 1, 0] at 0, y, 0',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+7/8, x+3/8, x+3/8 with inversion at [7/8, 3/8, 3/8]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 1/4]',
        #                             '3-fold screw axis with direction [0, 0, 1] at 1/3, 1/3, z with screw component [0, 0, 1/3]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/2, 1/4, z+3/8 with inversion at [1/2, 1/4, 3/8]',
        #                             '2-fold screw axis with direction [0, 1, -1] at 1/4, y+1/4, -y with screw component [0, 1/4, -1/4]', '3-fold rotation axis with direction [1, 1, -1] at x+3/4, x+3/4, -x',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/2, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 3/8, 3/8, z+3/8 with inversion at [3/8, 3/8, 3/8]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 0, y+3/4, y with screw component [0, 1/2, 1/2]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 1/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 5/8, 7/8 with screw component [1/4, 0, 0]', '4-fold screw axis with direction [0, 1, 0] at 5/8, y, 3/8 with screw component [0, 3/4, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+1/4, -x+3/4 with inversion at [0, 1/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 2/3, 1/3, z+1/3 with inversion at [2/3, 1/3, 1/3]', '4-fold screw axis with direction [0, 1, 0] at 3/8, y, 1/8 with screw component [0, 1/4, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/8, -x+5/8, -x+5/8 with inversion at [3/8, 5/8, 5/8]', '2-fold rotation axis with direction [1, 0, 0] at x, 0, 1/4',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 1/4, y, 0', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/3, 1/6, 1/6]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 7/8, 3/8 with inversion at [1/8, 7/8, 3/8]', '2-fold rotation axis with direction [0, 1, 0] at 1/8, y, 3/8',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/2, x+1/2, -x with inversion at [1/2, 1/2, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+7/8, -x+1/8, x+7/8 with inversion at [7/8, 1/8, 7/8]', 'Glide plane perpendicular to [0, 1, -1] with glide component [3/4, 3/4, 3/4]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/4, y, 1/4 with screw component [0, 1/2, 0]', '3-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 2/3]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/8, x+1/8, x+5/8 with inversion at [1/8, 1/8, 5/8]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 3/4, y+1/4, 1/4 with inversion at [3/4, 1/4, 1/4]', '2-fold screw axis with direction [1, 0, 0] at x, 3/8, 1/8 with screw component [1/2, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/2, -x, -x+1/2 with inversion at [1/2, 0, 1/2]', '2-fold rotation axis with direction [0, 1, 0] at 0, y, 0',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+3/4, -x+3/4 with inversion at [0, 3/4, 3/4]', '4-fold rotation axis with direction [1, 0, 0] at x, 1/4, 3/4',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z with inversion at [0, 0, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/8, 5/8, z+1/8 with inversion at [1/8, 5/8, 1/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 3/4, 0, z with screw component [0, 0, 3/4]', '4-fold rotoinversion axis with direction [0, 1, 0] at 7/8, y+3/8, 3/8 with inversion at [7/8, 3/8, 3/8]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 3/8 with screw component [1/2, 1/2, 0]', '4-fold screw axis with direction [1, 0, 0] at x, 7/8, 1/8 with screw component [1/4, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/4, -x+1/4, x+3/4 with inversion at [1/4, 1/4, 3/4]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 3/4]', 'Inversion at [1/4, 0, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+1/4, -x+1/4 with inversion at [3/4, 1/4, 1/4]', '2-fold screw axis with direction [0, 1, 1] at 0, y, y with screw component [0, 1/4, 1/4]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 1/8 with screw component [1/4, 1/4, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/4, -x+1/4, -x+3/4 with inversion at [3/4, 1/4, 3/4]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 3/4]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+3/4, 0, x with screw component [1/2, 0, 1/2]', '3-fold rotoinversion axis with direction [0, 0, 1] at 2/3, 1/3, z+1/3 with inversion at [2/3, 1/3, 1/3]',
        #                             '3-fold screw axis with direction [1, -1, -1] at x+1/3, -x+1/3, -x with screw component [-1/6, 1/6, 1/6]', 'Inversion at [1/3, 1/6, 5/12]', '2-fold rotation axis with direction [2, 1, 0] at 2x, x, 5/12',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [-1/4, 0, 1/4]', '3-fold rotoinversion axis with direction [1, 1, 1] at x, x, x+1/2 with inversion at [0, 0, 1/2]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 0, 1/2 with screw component [1/2, 0, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 1/4, 1/4 with inversion at [1/4, 1/4, 1/4]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/8, y+1/4, y with screw component [0, 1/2, 1/2]', '4-fold screw axis with direction [0, 1, 0] at 5/8, y, 3/8 with screw component [0, 1/4, 0]',
        #                             'Mirror plane perpendicular to [0, 0, 1]', '4-fold rotoinversion axis with direction [0, 0, 1] at 7/8, 3/8, z+1/8 with inversion at [7/8, 3/8, 1/8]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [0, 0, 1/2]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+1/4, -x with inversion at [3/4, 1/4, 0]',
        #                             '3-fold screw axis with direction [1, -1, 1] at x+2/3, -x+2/3, x with screw component [1/6, -1/6, 1/6]', 'Inversion at [1/4, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/8, -x+3/8, -x+7/8 with inversion at [1/8, 3/8, 7/8]', '2-fold rotation axis with direction [1, -1, 0] at x, -x, 1/6',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [0, 1/4, -1/4]', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/4, 1/4, 0]',
        #                             '2-fold screw axis with direction [0, 0, 1] at 3/8, 1/8, z with screw component [0, 0, 1/2]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/2, 1/2]',
        #                             '2-fold rotation axis with direction [0, 0, 1] at 0, 0, z', '3-fold screw axis with direction [1, -1, -1] at x+1/6, -x+1/6, -x with screw component [1/6, -1/6, -1/6]',
        #                             '2-fold screw axis with direction [1, 0, -1] at x+1/4, 1/4, -x with screw component [1/4, 0, -1/4]', '4-fold rotoinversion axis with direction [0, 1, 0] at 0, y+1/4, 1/2 with inversion at [0, 1/4, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/4, x+1/4, x+1/4 with inversion at [1/4, 1/4, 1/4]', '4-fold screw axis with direction [1, 0, 0] at x, 3/4, 1/2 with screw component [3/4, 0, 0]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 0 with screw component [1/4, -1/4, 0]', 'Glide plane perpendicular to [1, 0, 1] with glide component [0, 1/2, 0]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [0, 0, 1/2]', '4-fold screw axis with direction [0, 0, 1] at 1/8, 7/8, z with screw component [0, 0, 1/4]',
        #                             '2-fold rotation axis with direction [0, 0, 1] at 1/8, 1/8, z', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 3/4]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [0, 0, 1/2]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/4, 1/4]', 'Mirror plane perpendicular to [0, 1, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 1/4]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 1/2, 1/4 with inversion at [1/8, 1/2, 1/4]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 1/8, 1/8 with inversion at [1/8, 1/8, 1/8]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y, 3/4 with inversion at [1/4, 0, 3/4]', '2-fold rotation axis with direction [1, 0, 0] at x, 0, 5/12',
        #                             '3-fold screw axis with direction [1, 1, 1] at x+5/6, x+2/3, x with screw component [1/3, 1/3, 1/3]', '4-fold screw axis with direction [0, 0, 1] at 0, 1/2, z with screw component [0, 0, 1/2]',
        #                             'Glide plane perpendicular to [2, 1, 0] with glide component [0, 0, 1/2]', '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/8, x+1/8, x+1/8 with inversion at [1/8, 1/8, 1/8]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+3/4, 1/4, x with screw component [1/4, 0, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/8, x+1/8, x+1/8 with inversion at [1/8, 1/8, 1/8]', 'Inversion at [0, 0, 0]',
        #                             '2-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 1/2]', 'Glide plane perpendicular to [0, 1, 1] with glide component [1/4, 1/4, -1/4]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 3/8, y, 5/8 with screw component [0, 3/4, 0]', '2-fold rotation axis with direction [1, -1, 0] at x, -x, 5/12',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/8, y, 3/8 with screw component [0, 3/4, 0]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 5/8, 1/8 with inversion at [1/8, 5/8, 1/8]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 1/4, z+3/8 with inversion at [1/4, 1/4, 3/8]', '2-fold screw axis with direction [0, 1, 1] at 1/4, y, y with screw component [0, 1/2, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/2, x, -x+1/2 with inversion at [1/2, 0, 1/2]', '4-fold screw axis with direction [0, 0, 1] at 1/2, 3/4, z with screw component [0, 0, 3/4]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y+1/4, 1/4 with inversion at [1/4, 1/4, 1/4]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y+1/8, 1/2 with inversion at [1/4, 1/8, 1/2]', '4-fold rotation axis with direction [1, 0, 0] at x, 0, 1/2',
        #                             '2-fold rotation axis with direction [1, 0, 0] at x, 1/8, 1/8', '4-fold rotoinversion axis with direction [0, 0, 1] at 3/4, 1/4, z with inversion at [3/4, 1/4, 0]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/8, x+5/8, x+1/8 with inversion at [1/8, 5/8, 1/8]', '4-fold screw axis with direction [1, 0, 0] at x, 3/4, 1/4 with screw component [1/2, 0, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 3/4]', '3-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z+1/4 with inversion at [0, 0, 1/4]',
        #                             '2-fold rotation axis with direction [0, 1, -1] at 0, y+1/2, -y', '3-fold rotoinversion axis with direction [1, 1, 1] at x, x, x with inversion at [0, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+3/8, x+3/8, x+7/8 with inversion at [3/8, 3/8, 7/8]', '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z with inversion at [0, 0, 0]', '3-fold rotoinversion axis with direction [1, 1, 1] at x+5/8, x+1/8, x+1/8 with inversion at [5/8, 1/8, 1/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/4, 0, 0 with inversion at [1/4, 0, 0]', '3-fold screw axis with direction [1, -1, 1] at x+1/6, -x+1/6, x with screw component [1/3, -1/3, 1/3]',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [0, 3/4, 3/4]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 1/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/8, -x+7/8, -x+7/8 with inversion at [1/8, 7/8, 7/8]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/8, x+1/8, x+5/8 with inversion at [1/8, 1/8, 5/8]', '2-fold screw axis with direction [0, 1, 0] at 3/8, y, 3/8 with screw component [0, 1/2, 0]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 0, y, 1/12', '2-fold rotation axis with direction [0, 1, 0] at 1/8, y, 1/8',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 1/4]', 'Glide plane perpendicular to [1, 0, 1] with glide component [-1/4, 0, 1/4]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x, 1/4, 1/4 with inversion at [0, 1/4, 1/4]', '2-fold rotation axis with direction [1, -1, 0] at x+1/2, -x, 1/4',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/4, 0 with screw component [1/4, 0, 0]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+5/8, -x+3/8, x+5/8 with inversion at [5/8, 3/8, 5/8]',
        #                             'Centring vector [2/3, 1/3, 1/3]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+5/8, -x+7/8, x+5/8 with inversion at [5/8, 7/8, 5/8]',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [3/4, -1/4, 1/4]', '2-fold screw axis with direction [1, 1, 0] at x, x, 0 with screw component [1/4, 1/4, 0]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 1/4 with screw component [1/4, 1/4, 0]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 1/4]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 3/8, y+3/4, y with screw component [0, 1/2, 1/2]', '2-fold rotation axis with direction [1, -1, 0] at x, -x, 1/12',
        #                             '4-fold screw axis with direction [0, 0, 1] at 0, 1/2, z with screw component [0, 0, 3/4]', 'Mirror plane perpendicular to [0, 1, 1]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x, 3/4, 1/4 with inversion at [0, 3/4, 1/4]', '2-fold screw axis with direction [0, 1, 0] at 1/6, y, 1/3 with screw component [0, 1/2, 0]',
        #                             '2-fold rotation axis with direction [1, 1, 0] at x, x, 1/3', '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 1/8 with screw component [-1/4, 1/4, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 3/4, 1/4, z+1/4 with inversion at [3/4, 1/4, 1/4]', '4-fold screw axis with direction [0, 0, 1] at 3/4, 1/2, z with screw component [0, 0, 1/4]',
        #                             '2-fold rotation axis with direction [1, 2, 0] at x, 2x, 1/3', '4-fold rotoinversion axis with direction [0, 1, 0] at 3/4, y, 1/4 with inversion at [3/4, 0, 1/4]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 1/2, 3/4 with inversion at [3/8, 1/2, 3/4]', '3-fold rotation axis with direction [1, -1, -1] at x, -x, -x',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/8, -x+3/8, -x+7/8 with inversion at [1/8, 3/8, 7/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 3/8, 3/8 with inversion at [3/8, 3/8, 3/8]', 'Glide plane perpendicular to [0, 0, 1] with glide component [3/4, 1/4, 0]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/4, y+3/4, y with screw component [0, 1/4, 1/4]', '4-fold screw axis with direction [0, 0, 1] at 7/8, 5/8, z with screw component [0, 0, 1/4]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 5/8, y, 7/8 with screw component [0, 1/4, 0]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/8, x+5/8, -x+3/8 with inversion at [1/8, 5/8, 3/8]', '3-fold rotation axis with direction [1, 1, -1] at x+1/4, x+3/4, -x',
        #                             '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 2/3]', '2-fold rotation axis with direction [1, 0, 0] at x, 1/4, 3/8',
        #                             '2-fold screw axis with direction [0, 1, 0] at 1/4, y, 0 with screw component [0, 1/2, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 3/4, z+1/4 with inversion at [1/4, 3/4, 1/4]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 0, y, 1/4 with screw component [0, 3/4, 0]', 'Glide plane perpendicular to [1, -1, 0] with glide component [1/2, 1/2, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/8, -x+1/8, -x+5/8 with inversion at [3/8, 1/8, 5/8]', '4-fold rotation axis with direction [0, 1, 0] at 1/2, y, 0',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 1/4, y, 1/8', 'Glide plane perpendicular to [1, 1, 0] with glide component [1/4, -1/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/8, -x+3/8, x+5/8 with inversion at [1/8, 3/8, 5/8]', '2-fold rotation axis with direction [0, 1, -1] at 1/4, y, -y',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+1/4, -x with inversion at [3/4, 1/4, 0]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 1/4]',
        #                             'Glide plane perpendicular to [1, 1, 0] with glide component [-1/6, 1/6, 1/6]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 3/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+1/2, -x with inversion at [0, 1/2, 0]', 'Inversion at [1/4, 0, 3/8]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 1/3, 2/3, z+5/12 with inversion at [1/3, 2/3, 5/12]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 7/8, 3/8 with inversion at [3/8, 7/8, 3/8]', '2-fold screw axis with direction [0, 1, 1] at 0, y+1/4, y with screw component [0, 1/2, 1/2]',
        #                             'Glide plane perpendicular to [0, 0, 1] with glide component [1/4, 3/4, 0]', '2-fold screw axis with direction [0, 0, 1] at 1/4, 3/8, z with screw component [0, 0, 1/2]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/2, 1/4, z with screw component [0, 0, 3/4]', '4-fold screw axis with direction [1, 0, 0] at x, 3/8, 1/8 with screw component [1/4, 0, 0]',
        #                             '4-fold rotation axis with direction [1, 0, 0] at x, 1/2, 0', '2-fold screw axis with direction [0, 1, 0] at 1/4, y, 1/4 with screw component [0, 1/2, 0]',
        #                             '2-fold rotation axis with direction [1, 0, 0] at x, 3/8, 3/8', '4-fold screw axis with direction [1, 0, 0] at x, 1/4, 0 with screw component [3/4, 0, 0]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/8, y, 7/8 with screw component [0, 1/4, 0]', 'Mirror plane perpendicular to [1, 0, 1]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/2, x, x with inversion at [1/2, 0, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/2, 0, z with inversion at [1/2, 0, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 5/8, y+1/8, 1/8 with inversion at [5/8, 1/8, 1/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/8, -x+3/8, x+5/8 with inversion at [1/8, 3/8, 5/8]',
        #                             '2-fold screw axis with direction [1, 0, -1] at x+1/4, 1/4, -x with screw component [-1/4, 0, 1/4]', '4-fold screw axis with direction [0, 1, 0] at 0, y, 0 with screw component [0, 1/2, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/4, -x+3/4, -x+3/4 with inversion at [1/4, 3/4, 3/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 0, 3/4 with screw component [1/4, 0, 0]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/2, 1/4, z+1/8 with inversion at [1/2, 1/4, 1/8]',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [0, 1/4, -1/4]', '3-fold screw axis with direction [1, 1, 1] at x+1/3, x+1/6, x with screw component [1/3, 1/3, 1/3]',
        #                             '3-fold rotation axis with direction [1, -1, -1] at x+1/2, -x, -x', '3-fold rotoinversion axis with direction [1, 1, -1] at x+5/8, x+5/8, -x+3/8 with inversion at [5/8, 5/8, 3/8]',
        #                             'Glide plane perpendicular to [0, 1, -1] with glide component [0, 1/4, 1/4]', '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/4, x+1/4, x+1/4 with inversion at [1/4, 1/4, 1/4]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 0, y, 0', '3-fold rotoinversion axis with direction [1, -1, -1] at x+5/8, -x+7/8, -x+7/8 with inversion at [5/8, 7/8, 7/8]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 3/8, 3/8 with inversion at [1/8, 3/8, 3/8]', '2-fold rotation axis with direction [0, 1, 0] at 0, y, 1/4',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x, x+1/2, -x+1/2 with inversion at [0, 1/2, 1/2]', '2-fold rotation axis with direction [1, 0, 0] at x, 1/4, 0',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 1/4, z with inversion at [1/4, 1/4, 0]', '2-fold screw axis with direction [1, 0, 1] at x, 1/4, x with screw component [1/4, 0, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+5/8, -x+7/8, x+5/8 with inversion at [5/8, 7/8, 5/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/2, -x+3/4, -x+3/4 with inversion at [1/2, 3/4, 3/4]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 3/4, 1/4]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 1/4]', '3-fold rotation axis with direction [1, -1, 1] at x, -x, x',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/8, x+5/8, x+1/8 with inversion at [1/8, 5/8, 1/8]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/2, x+1/2, -x with inversion at [1/2, 1/2, 0]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 3/8 with screw component [1/4, -1/4, 0]', '3-fold rotation axis with direction [1, 1, -1] at x+1/2, x+1/2, -x',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 3/4]', '4-fold rotoinversion axis with direction [0, 0, 1] at 1/2, 3/4, z+1/8 with inversion at [1/2, 3/4, 1/8]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/4, 1/4, z+1/4 with inversion at [1/4, 1/4, 1/4]', '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 5/6]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+1/4, 1/4, x with screw component [1/4, 0, 1/4]', '2-fold rotation axis with direction [1, -1, 0] at x, -x, 0',
        #                             '2-fold screw axis with direction [0, 1, 0] at 1/6, y, 1/12 with screw component [0, 1/2, 0]', '2-fold screw axis with direction [0, 1, 1] at 0, y, y with screw component [0, 3/4, 3/4]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/8, 7/8 with screw component [3/4, 0, 0]', '4-fold screw axis with direction [0, 0, 1] at 7/8, 5/8, z with screw component [0, 0, 3/4]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/8, 1/8, z+1/8 with inversion at [1/8, 1/8, 1/8]', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/2, 0, 0]',
        #                             'Glide plane perpendicular to [1, 0, 0] with glide component [0, 0, 1/2]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/8, x+7/8, -x+1/8 with inversion at [3/8, 7/8, 1/8]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/2, -x, 3/8 with screw component [1/4, -1/4, 0]', '2-fold rotation axis with direction [1, 0, 0] at x, 0, 1/3',
        #                             '3-fold rotation axis with direction [1, -1, -1] at x+1/4, -x+1/2, -x', 'Glide plane perpendicular to [1, 1, 0] with glide component [0, 0, 1/2]',
        #                             '3-fold rotation axis with direction [1, -1, -1] at x, -x, -x', '3-fold screw axis with direction [1, 1, -1] at x+1/6, x+1/3, -x with screw component [1/3, 1/3, -1/3]',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [1/4, 0, -1/4]', 'Glide plane perpendicular to [1, 0, -1] with glide component [3/4, 3/4, 3/4]',
        #                             '3-fold screw axis with direction [1, -1, 1] at x+5/6, -x+1/3, x with screw component [-1/6, 1/6, -1/6]', '4-fold screw axis with direction [0, 0, 1] at 1/2, 0, z with screw component [0, 0, 3/4]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z with inversion at [0, 0, 0]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 3/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 1/3, 2/3, z+5/12 with inversion at [1/3, 2/3, 5/12]', 'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, -1/4, 1/4]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/2, y+1/4, 0 with inversion at [1/2, 1/4, 0]', '4-fold rotation axis with direction [1, 0, 0] at x, 1/4, 1/4', 'Inversion at [3/8, 3/8, 1/8]',
        #                             '4-fold screw axis with direction [0, 1, 0] at 0, y, 1/2 with screw component [0, 1/2, 0]', 'Inversion at [3/8, 1/8, 1/8]', '3-fold rotation axis with direction [1, -1, 1] at x+1/2, -x+1/4, x',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/4, y, y with screw component [0, 1/4, 1/4]', '2-fold screw axis with direction [1, 0, 0] at x, 1/8, 3/8 with screw component [1/2, 0, 0]',
        #                             '2-fold rotation axis with direction [1, 0, 0] at x, 1/4, 1/8', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/2, 0, 0]', '4-fold rotation axis with direction [0, 1, 0] at 1/4, y, 1/4',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/2, 0, z+1/4 with inversion at [1/2, 0, 1/4]', '2-fold rotation axis with direction [1, 1, 0] at x, x, 3/8',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x, x+1/4 with inversion at [3/4, 0, 1/4]', '2-fold screw axis with direction [1, 0, 1] at x, 0, x with screw component [3/4, 0, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+3/8, x+3/8, x+3/8 with inversion at [3/8, 3/8, 3/8]', 'Centring vector [1/2, 1/2, 0]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+3/4, -x+1/4 with inversion at [3/4, 3/4, 1/4]', 'Identity',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 1/8, 1/8 with inversion at [1/8, 1/8, 1/8]', '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 0 with screw component [1/4, 1/4, 0]',
        #                             '2-fold rotation axis with direction [1, 0, -1] at x+3/4, 3/8, -x', '4-fold rotation axis with direction [0, 0, 1] at 1/4, 1/4, z',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 0 with screw component [1/2, 1/2, 0]', '6-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z with inversion at [0, 0, 0]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 1/4 with screw component [1/4, -1/4, 0]', '4-fold screw axis with direction [0, 0, 1] at 0, 1/2, z with screw component [0, 0, 1/4]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/4, y+1/4, y with screw component [0, 1/2, 1/2]', 'Glide plane perpendicular to [0, 0, 1] with glide component [3/4, 3/4, 0]',
        #                             '2-fold rotation axis with direction [1, 0, 1] at x, 1/4, x', 'Glide plane perpendicular to [1, 0, 1] with glide component [1/4, 1/2, -1/4]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 1/4 with screw component [1/4, 1/4, 0]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/8, y+3/8, 1/8 with inversion at [1/8, 3/8, 1/8]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 3/8, y+1/4, y with screw component [0, 1/2, 1/2]', '2-fold rotation axis with direction [1, 2, 0] at x, 2x, 1/4',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/4, -x+3/4, -x+1/4 with inversion at [1/4, 3/4, 1/4]', '2-fold rotation axis with direction [1, 2, 0] at x, 2x, 0',
        #                             'Glide plane perpendicular to [1, 0, -1] with glide component [1/4, 1/2, 1/4]', '4-fold screw axis with direction [0, 0, 1] at 1/2, 0, z with screw component [0, 0, 1/4]',
        #                             '2-fold screw axis with direction [0, 1, 0] at 1/4, y, 3/8 with screw component [0, 1/2, 0]', '4-fold screw axis with direction [0, 1, 0] at 1/2, y, 1/4 with screw component [0, 3/4, 0]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 0 with screw component [-1/4, 1/4, 0]', '4-fold screw axis with direction [0, 0, 1] at 1/8, 3/8, z with screw component [0, 0, 1/4]',
        #                             'Mirror plane perpendicular to [0, 1, -1]', '3-fold rotoinversion axis with direction [1, -1, 1] at x, -x, x with inversion at [0, 0, 0]',
        #                             'Glide plane perpendicular to [1, 0, -1] with glide component [1/4, 1/4, 1/4]', '2-fold screw axis with direction [0, 0, 1] at 1/4, 0, z with screw component [0, 0, 1/2]', 'Inversion at [0, 1/4, 1/8]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 0, y, 1/3', '2-fold rotation axis with direction [1, -1, 0] at x+1/2, -x, 0', 'Mirror plane perpendicular to [1, 1, 0]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+3/4, -x+1/2 with inversion at [3/4, 3/4, 1/2]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 3/4, y+1/8, 0 with inversion at [3/4, 1/8, 0]', '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 1/4 with screw component [1/2, 1/2, 0]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/4, x+3/4, -x with inversion at [1/4, 3/4, 0]', '4-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/2]',
        #                             '2-fold screw axis with direction [1, 0, -1] at x+1/4, 0, -x with screw component [1/4, 0, -1/4]', 'Glide plane perpendicular to [1, 0, 1] with glide component [0, 1/2, 0]',
        #                             '3-fold screw axis with direction [1, 1, 1] at x+5/6, x+1/6, x with screw component [1/3, 1/3, 1/3]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/4, -x, x+3/4 with inversion at [1/4, 0, 3/4]', '2-fold rotation axis with direction [1, 0, 0] at x, 1/3, 1/12',
        #                             'Glide plane perpendicular to [0, 0, 1] with glide component [1/2, 1/2, 0]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+5/8, -x+3/8, x+5/8 with inversion at [5/8, 3/8, 5/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 1/2]', '4-fold rotation axis with direction [0, 0, 1] at 0, 1/2, z',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+1/2, -x+1/2 with inversion at [0, 1/2, 1/2]', '4-fold screw axis with direction [1, 0, 0] at x, 5/8, 7/8 with screw component [3/4, 0, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 3/4, z with screw component [0, 0, 3/4]', 'Glide plane perpendicular to [0, 1, 0] with glide component [1/4, 0, 3/4]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 1/3, y, 5/12', '2-fold screw axis with direction [1, 0, 1] at x+1/4, 0, x with screw component [1/2, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+5/8, x+5/8, -x+7/8 with inversion at [5/8, 5/8, 7/8]', '4-fold rotation axis with direction [0, 1, 0] at 3/4, y, 1/4',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/2, 1/4 with screw component [3/4, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+7/8, -x+1/8, x+7/8 with inversion at [7/8, 1/8, 7/8]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+1/2, x, x with inversion at [1/2, 0, 0]', '3-fold rotoinversion axis with direction [1, 1, 1] at x, x+1/2, x with inversion at [0, 1/2, 0]',
        #                             '3-fold screw axis with direction [1, 1, -1] at x+2/3, x+1/3, -x with screw component [1/6, 1/6, -1/6]', 'Glide plane perpendicular to [1, 0, 0] with glide component [1/6, 1/3, 1/3]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+1/8, -x+7/8, -x+7/8 with inversion at [1/8, 7/8, 7/8]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 1/4, y+3/4, y with screw component [0, 1/2, 1/2]', '4-fold rotoinversion axis with direction [0, 0, 1] at 3/4, 1/4, z+1/8 with inversion at [3/4, 1/4, 1/8]',
        #                             '3-fold rotation axis with direction [1, 1, -1] at x+1/2, x, -x', '4-fold rotoinversion axis with direction [1, 0, 0] at x, 1/4, 3/4 with inversion at [0, 1/4, 3/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x+3/8, -x+5/8, -x+5/8 with inversion at [3/8, 5/8, 5/8]', 'Glide plane perpendicular to [0, 1, 0] with glide component [0, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [0, 0, 1] at 1/3, 2/3, z+1/6 with inversion at [1/3, 2/3, 1/6]', '3-fold rotation axis with direction [1, -1, 1] at x, -x+1/4, x',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [1/4, 0, -1/4]', '4-fold rotoinversion axis with direction [0, 1, 0] at 3/8, y+1/8, 3/8 with inversion at [3/8, 1/8, 3/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+5/8, -x+3/8, x+1/8 with inversion at [5/8, 3/8, 1/8]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/4, -x, x+3/4 with inversion at [1/4, 0, 3/4]', '2-fold screw axis with direction [1, 0, 1] at x+3/4, 1/4, x with screw component [1/2, 0, 1/2]',
        #                             '3-fold rotation axis with direction [1, -1, 1] at x+1/2, -x, x', '3-fold rotation axis with direction [1, 1, -1] at x+3/4, x+1/4, -x',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 0]', '2-fold rotation axis with direction [1, 1, 0] at x, x, 1/4',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x, 0, 1/2 with inversion at [0, 0, 1/2]', '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 1/4 with screw component [1/2, 1/2, 0]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 3/8 with screw component [-1/4, 1/4, 0]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y, 1/4 with inversion at [1/4, 0, 1/4]',
        #                             '6-fold rotation axis with direction [0, 0, 1] at 0, 0, z', '3-fold rotoinversion axis with direction [1, 1, -1] at x, x, -x with inversion at [0, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+5/8, -x+3/8, x+1/8 with inversion at [5/8, 3/8, 1/8]',
        #                             '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x, -x with inversion at [0, 0, 0]', '2-fold rotation axis with direction [1, 1, 0] at x, x, 0',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+1/4, 3/8, x with screw component [1/2, 0, 1/2]', '3-fold rotoinversion axis with direction [1, 1, 1] at x, x, x+1/2 with inversion at [0, 0, 1/2]',
        #                             '2-fold rotation axis with direction [0, 1, -1] at 3/8, y+3/4, -y', '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 0 with screw component [1/4, 1/4, 0]',
        #                             'Glide plane perpendicular to [0, 0, 1] with glide component [1/4, 3/4, 0]', '3-fold screw axis with direction [0, 0, 1] at 1/3, 0, z with screw component [0, 0, 1/3]',
        #                             'Mirror plane perpendicular to [1, 0, 0]', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/4, 1/4, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 0, 1] at 1/8, 1/8, z+3/8 with inversion at [1/8, 1/8, 3/8]',
        #                             '3-fold rotoinversion axis with direction [1, 1, 1] at x+3/8, x+7/8, x+3/8 with inversion at [3/8, 7/8, 3/8]', '4-fold screw axis with direction [0, 0, 1] at 3/4, 1/2, z with screw component [0, 0, 3/4]',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [0, -1/4, 1/4]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+5/8, x+1/8, -x+3/8 with inversion at [5/8, 1/8, 3/8]',
        #                             'Mirror plane perpendicular to [0, 1, 0]', 'Inversion at [1/8, 1/8, 1/8]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x+1/4, x+3/4 with inversion at [3/4, 1/4, 3/4]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 3/8 with screw component [1/2, 1/2, 0]', 'Inversion at [1/6, 1/3, 1/12]', '3-fold rotation axis with direction [1, -1, -1] at x+1/4, -x, -x',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+7/8, x+7/8, -x+5/8 with inversion at [7/8, 7/8, 5/8]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x, x, 3/8 with screw component [1/2, 1/2, 0]', '4-fold rotoinversion axis with direction [0, 1, 0] at 0, y+1/4, 0 with inversion at [0, 1/4, 0]',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [0, 1/2, 0]', '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 3/8 with screw component [1/4, 1/4, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 7/8, 1/8, z with screw component [0, 0, 3/4]', '3-fold rotation axis with direction [1, 1, -1] at x, x, -x',
        #                             '3-fold rotation axis with direction [1, -1, -1] at x+3/4, -x, -x', '3-fold rotation axis with direction [0, 0, 1] at 0, 0, z', '2-fold rotation axis with direction [0, 1, -1] at 0, y, -y',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [1/2, 0, 0]', 'Glide plane perpendicular to [1, 1, 0] with glide component [-1/4, 1/4, 0]', 'Inversion at [1/8, 3/8, 3/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 5/8, 3/8, z with screw component [0, 0, 1/4]', '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x+1/2, -x+1/2 with inversion at [0, 1/2, 1/2]',
        #                             'Glide plane perpendicular to [1, 0, -1] with glide component [1/4, 0, 1/4]', 'Glide plane perpendicular to [1, 0, 0] with glide component [1/6, 1/3, 5/6]',
        #                             '2-fold screw axis with direction [0, 1, -1] at 3/8, y+1/2, -y with screw component [0, -1/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x+3/4, x+1/4 with inversion at [3/4, 3/4, 1/4]', '2-fold rotation axis with direction [1, 0, -1] at x+1/2, 0, -x',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [-1/4, 1/2, 1/4]', '3-fold rotoinversion axis with direction [1, -1, -1] at x, -x, -x with inversion at [0, 0, 0]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 3/8, 5/8, z with screw component [0, 0, 1/4]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/8, y+1/8, 5/8 with inversion at [1/8, 1/8, 5/8]',
        #                             '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 2/3]', '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 0, z+1/4 with inversion at [0, 0, 1/4]',
        #                             '3-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/3]', 'Glide plane perpendicular to [0, 0, 1] with glide component [3/4, 1/4, 0]',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [0, 0, 1/2]', '4-fold screw axis with direction [0, 0, 1] at 1/4, 1/4, z with screw component [0, 0, 1/4]',
        #                             'Glide plane perpendicular to [1, -1, 0] with glide component [1/4, 1/4, 1/2]', '4-fold rotoinversion axis with direction [0, 0, 1] at 0, 3/4, z+3/8 with inversion at [0, 3/4, 3/8]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 3/4, z with screw component [0, 0, 1/2]', '2-fold rotation axis with direction [1, 0, -1] at x+1/2, 1/4, -x',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/8, y, 3/8 with screw component [0, 1/4, 0]', '6-fold rotation axis with direction [0, 0, 1] at 0, 0, z',
        #                             '3-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 2/3]', '4-fold rotoinversion axis with direction [0, 1, 0] at 1/4, y+1/4, 1/4 with inversion at [1/4, 1/4, 1/4]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 3/8, y+3/8, 3/8 with inversion at [3/8, 3/8, 3/8]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+3/4, x+3/4, -x with inversion at [3/4, 3/4, 0]', '4-fold screw axis with direction [0, 0, 1] at 7/8, 1/8, z with screw component [0, 0, 1/4]',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, -1/4, 1/4]', '6-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/2]',
        #                             'Glide plane perpendicular to [1, 0, 1] with glide component [1/4, 1/2, -1/4]', 'Inversion at [1/4, 1/4, 1/4]', 'Glide plane perpendicular to [0, 0, 1] with glide component [1/2, 1/2, 0]',
        #                             '4-fold rotoinversion axis with direction [0, 1, 0] at 1/8, y+3/8, 5/8 with inversion at [1/8, 3/8, 5/8]', 'Glide plane perpendicular to [0, 1, 0] with glide component [2/3, 1/3, 1/3]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 3/8, 7/8 with inversion at [1/8, 3/8, 7/8]',
        #                             '2-fold screw axis with direction [1, -1, 0] at x+1/4, -x, 1/4 with screw component [-1/4, 1/4, 0]', 'Glide plane perpendicular to [1, 0, 0] with glide component [0, 3/4, 1/4]',
        #                             'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, 1/4, -1/4]', 'Glide plane perpendicular to [0, 1, 1] with glide component [1/2, 0, 0]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x+1/2, x+3/4 with inversion at [3/4, 1/2, 3/4]',
        #                             '3-fold screw axis with direction [1, 1, 1] at x, x, x with screw component [1/2, 1/2, 1/2]', '2-fold screw axis with direction [1, 0, 1] at x, 0, x with screw component [1/2, 0, 1/2]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x, x+3/4 with inversion at [3/4, 0, 3/4]', '4-fold rotation axis with direction [0, 0, 1] at 0, 0, z',
        #                             '4-fold screw axis with direction [0, 1, 0] at 1/8, y, 7/8 with screw component [0, 3/4, 0]', '2-fold rotation axis with direction [1, -1, 0] at x, -x, 1/4',
        #                             '2-fold screw axis with direction [0, 0, 1] at 0, 1/4, z with screw component [0, 0, 1/2]', '4-fold rotoinversion axis with direction [0, 1, 0] at 0, y, 0 with inversion at [0, 0, 0]',
        #                             '4-fold screw axis with direction [1, 0, 0] at x, 1/8, 7/8 with screw component [1/4, 0, 0]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x, x+1/4 with inversion at [3/4, 0, 1/4]',
        #                             '3-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/3]', '4-fold screw axis with direction [0, 0, 1] at 3/4, 0, z with screw component [0, 0, 1/4]',
        #                             '3-fold screw axis with direction [0, 0, 1] at 1/3, 1/3, z with screw component [0, 0, 2/3]', '4-fold rotoinversion axis with direction [1, 0, 0] at x+3/8, 3/8, 7/8 with inversion at [3/8, 3/8, 7/8]',
        #                             'Glide plane perpendicular to [0, 1, -1] with glide component [1/4, 1/4, 1/4]', '3-fold screw axis with direction [1, -1, 1] at x+1/3, -x+1/3, x with screw component [1/6, -1/6, 1/6]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 0, 0, z with screw component [0, 0, 1/4]', '3-fold rotoinversion axis with direction [1, -1, 1] at x+1/2, -x, x+1/2 with inversion at [1/2, 0, 1/2]',
        #                             '4-fold screw axis with direction [0, 0, 1] at 1/4, 0, z with screw component [0, 0, 1/4]', 'Glide plane perpendicular to [0, 1, -1] with glide component [1/2, 1/4, 1/4]',
        #                             '3-fold screw axis with direction [1, 1, -1] at x+1/3, x+1/6, -x with screw component [-1/6, -1/6, 1/6]',
        #                             '3-fold rotoinversion axis with direction [1, -1, 1] at x+3/4, -x+1/4, x+3/4 with inversion at [3/4, 1/4, 3/4]',
        #                             '2-fold screw axis with direction [1, 1, 0] at x+3/4, x, 1/4 with screw component [1/4, 1/4, 0]', '3-fold rotoinversion axis with direction [1, 1, -1] at x+1/2, x, -x with inversion at [1/2, 0, 0]',
        #                             '2-fold screw axis with direction [0, 1, 0] at 1/8, y, 1/8 with screw component [0, 1/2, 0]', '4-fold rotoinversion axis with direction [0, 1, 0] at 3/4, y+3/8, 1/2 with inversion at [3/4, 3/8, 1/2]',
        #                             '2-fold screw axis with direction [0, 1, 1] at 0, y+3/4, y with screw component [0, 1/4, 1/4]',
        #                             '3-fold rotoinversion axis with direction [1, 1, -1] at x+5/8, x+1/8, -x+3/8 with inversion at [5/8, 1/8, 3/8]',
        #                             '2-fold screw axis with direction [1, 0, 1] at x+3/4, 3/8, x with screw component [1/2, 0, 1/2]', '4-fold screw axis with direction [0, 1, 0] at 7/8, y, 1/8 with screw component [0, 1/4, 0]',
        #                             'Glide plane perpendicular to [0, 0, 1] with glide component [0, 1/2, 0]', 'Glide plane perpendicular to [0, 0, 1] with glide component [0, 1/2, 0]',
        #                             'Glide plane perpendicular to [0, 1, 0] with glide component [3/4, 0, 3/4]', '2-fold screw axis with direction [1, 1, 0] at x+1/4, x, 1/8 with screw component [1/4, 1/4, 0]',
        #                             '2-fold screw axis with direction [0, 1, 0] at 0, y, 1/4 with screw component [0, 1/2, 0]', '4-fold screw axis with direction [1, 0, 0] at x, 3/4, 0 with screw component [1/4, 0, 0]',
        #                             '2-fold rotation axis with direction [0, 1, 0] at 0, y, 1/4', '4-fold screw axis with direction [1, 0, 0] at x, 0, 1/4 with screw component [1/4, 0, 0]',
        #                             '4-fold rotoinversion axis with direction [1, 0, 0] at x+1/8, 0, 3/4 with inversion at [1/8, 0, 3/4]', '4-fold rotation axis with direction [0, 0, 1] at 0, 0, z']

    def init_features(self):
        periodic_table = Chem.GetPeriodicTable()
        self.vdw_radii = {}
        self.element_symbols = {}
        for i in range(1, 119):
            self.vdw_radii[str(i)] = periodic_table.GetRvdw(i)
            self.element_symbols[str(i)] = periodic_table.GetElementSymbol(i)

        self.electronegativity_dict = {}
        for i in range(1, 101):
            self.electronegativity_dict[i] = element_table(i).electronegativity('pauling')

        for key in self.electronegativity_dict.keys():
            if self.electronegativity_dict[key] is None:
                self.electronegativity_dict[key] = 0

        # break down all symmetry elements

        # self.key_symmetry_elements = ['Inversion', 'rotoinversion', 'Mirror', 'Glide', 'screw', 'rotation',
        #                               '2-fold rotation', '3-fold rotation', '4-fold rotation', '6-fold rotation',
        #                               '3-fold rotoinversion', '4-fold rotoinversion', '6-fold rotoinversion',  # there is no 2-fold rotoinversion
        #                               '2-fold screw', '3-fold screw', '4-fold screw', '6-fold screw',
        #                               ]
        # self.crystal_symmetry_elements = {}
        # for element in self.key_symmetry_elements:
        #     self.crystal_symmetry_elements[element] = [self.unique_elements[i] for i in range(len(self.unique_descriptions)) if element in self.unique_descriptions[i]]

        self.point_groups = {}
        self.lattice_type = {}
        self.sym_ops = {}
        for i in tqdm.tqdm(range(1, 231)):
            sym_group = symmetry.Group(i)
            general_position_syms = sym_group.wyckoffs_organized[0][0]
            self.sym_ops[i] = [general_position_syms[i].affine_matrix for i in range(len(general_position_syms))]  # first 0 index is for general position, second index is superfluous, third index is the symmetry operation
            self.point_groups[i] = sym_group.point_group
            self.lattice_type[i] = sym_group.lattice_type

        self.HDonorSmarts = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]')  # from rdkit lipinski https://github.com/rdkit/rdkit/blob/7c6d9cf4e9d95b4daa954f4f094e026093dbc13f/rdkit/Chem/Lipinski.py#L26
        self.HAcceptorSmarts = Chem.MolFromSmarts(
            '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
            '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
            '$([nH0,o,s;+0])]')

        self.atom_keys = ['atom coords', 'atom mass', 'atom Z', 'atom is H bond donor', 'atom is H bond acceptor',
                          'atom valence', 'atom vdW radius', 'atom on a ring', 'atom chirality', 'atom is aromatic',
                          'atom degree', 'atom electronegativity']

        self.sorted_fractional_rotations, self.normed_fractional_translations = initialize_fractional_vectors()

    def featurize(self, chunk_inds=[0, 100]):
        os.chdir(self.crystal_chunks_path)
        chunks = os.listdir()[chunk_inds[0]:chunk_inds[1]]
        self.init_features()

        for i, chunk in enumerate(chunks):
            if not os.path.exists('../molecule_features/{}'.format(chunk_inds[0] + i)):  # don't repeat

                df = pd.read_pickle(chunk)
                bad_inds = []
                new_features = None
                print('Processing chunk {} with {} entries'.format(i, len(df)))
                for j in tqdm.tqdm(range(len(df))):
                    # hydrogens are completely inconsistent, and can't be efficiently added either by CSD or rdkit
                    mol = Chem.MolFromMol2Block(df['xyz'][j], sanitize=True, removeHs=True)
                    try:
                        mol = Chem.RemoveAllHs(mol)  # strictly clean the molecule of all hydrogens
                    except:
                        if mol is not None:
                            bad_inds.append(j)
                            mol = None
                            print("Error kekulizing")

                    if mol is None:
                        bad_inds.append(j)
                    elif df['crystal reference cell coords'][j] == 'error': # if we didn't get reference coordinates, reject the structure
                        bad_inds.append(j)
                    elif mol.GetNumAtoms() < 3:
                        bad_inds.append(j)
                    else:
                        mol_data = self.featurize_molecule(mol)
                        crystal_data = self.featurize_crystal(df.loc[j])
                        mol_data['molecule volume'] = df['crystal packing coefficient'][j] * df['crystal cell volume'][j]/ df['crystal z value'][j] # much faster this way
                        mol_data.update(crystal_data)

                        if new_features is None:
                            new_features = [[] for _ in range(len(mol_data.keys()))]
                            self.new_features_names = list(mol_data.keys())

                        for k, key in enumerate(mol_data.keys()):
                            new_features[k].append(mol_data[key])

                df = df.drop(df.index[bad_inds])
                if 'level_0' in df.columns:  # delete unwanted samples
                    df = df.drop(columns='level_0')
                df = df.reset_index()
                for k, key in enumerate(self.new_features_names):
                    df[key] = new_features[k]

                df.to_pickle('../molecule_features/{}'.format(chunk_inds[0] + i))

    def featurize_molecule(self, mol):
        '''
        input is an rdkit molecule object
        and the dataset which we are appending to
        :param mol:
        :return:
        '''
        atoms = mol.GetAtoms()
        conformer = mol.GetConformer()
        h_donors = list(sum(mol.GetSubstructMatches(self.HDonorSmarts, uniquify=1), ()))  # convert tuple to list
        h_acceptors = list(sum(mol.GetSubstructMatches(self.HAcceptorSmarts, uniquify=1), ()))

        '''
        atom features
        '''

        dataset = {}
        dataset['atom coords'] = conformer.GetPositions()
        dataset['atom mass'] = [atom.GetMass() for atom in atoms]
        dataset['atom Z'] = [atom.GetAtomicNum() for atom in atoms]
        dataset['atom is H bond donor'] = [1 if i in list(h_donors) else 0 for i in range(len(atoms))]
        dataset['atom is H bond acceptor'] = [1 if i in list(h_acceptors) else 0 for i in range(len(atoms))]
        dataset['atom valence'] = [atom.GetTotalValence() for atom in atoms]
        dataset['atom vdW radius'] = [self.vdw_radii[str(number)] for number in dataset['atom Z']]
        dataset['atom on a ring'] = [atom.IsInRing() for atom in atoms]
        dataset['atom chirality'] = [atom.GetChiralTag().real for atom in atoms]
        dataset['atom is aromatic'] = [atom.GetIsAromatic() for atom in atoms]
        dataset['atom degree'] = [atom.GetDegree() for atom in atoms]
        dataset['atom electronegativity'] = [self.electronegativity_dict[atom] for atom in dataset['atom Z']]

        assert sum(np.asarray(dataset['atom Z']) == 1) == 0  # positively assert there are absolutely no protons in the dataset

        '''
        molecule features
        '''

        dataset['molecule mass'] = Descriptors.MolWt(mol)
        dataset['molecule num atoms'] = len(dataset['atom Z'])  # mol.GetNumAtoms()
        dataset['molecule num rings'] = mol.GetRingInfo().NumRings()
        #dataset['molecule point group'] = self.pointGroupAnalysis(dataset['atom Z'], dataset['atom coords'])  # this is also slow, approx 30% of total effort
        #dataset['molecule volume'] = AllChem.ComputeMolVolume(mol)  # this is very slow - approx 50% of total effort - fill this in later from the CSD
        dataset['molecule num donors'] = len(h_donors)
        dataset['molecule num acceptors'] = len(h_acceptors)
        dataset['molecule polarity'], dataset['molecule centroid'] = get_dipole(dataset['atom coords'], dataset['atom electronegativity'])
        dataset['molecule spherical defect'] = rdMolDescriptors.CalcAsphericity(mol)
        dataset['molecule eccentricity'] = rdMolDescriptors.CalcEccentricity(mol)
        dataset['molecule num rotatable bonds'] = rdMolDescriptors.CalcNumRotatableBonds((mol))
        dataset['molecule planarity'] = rdMolDescriptors.CalcPBF(mol)
        dataset['molecule radius of gyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol)
        dataset['molecule H fraction'] = get_fraction(dataset['atom Z'], 1)
        dataset['molecule C fraction'] = get_fraction(dataset['atom Z'], 6)
        dataset['molecule N fraction'] = get_fraction(dataset['atom Z'], 7)
        dataset['molecule O fraction'] = get_fraction(dataset['atom Z'], 8)
        dataset['molecule smiles'] = Chem.MolToSmiles(mol)
        dataset['molecule chemical formula'] = rdMolDescriptors.CalcMolFormula(mol)
        Ip, Ipm, _ = compute_principal_axes_np(np.asarray(dataset['atom coords']), np.asarray(dataset['atom mass']))  # rdMolTransforms.ComputePrincipalAxesAndMoments(mol.GetConformer(), ignoreHs=False) # this does it column-wise
        dataset['molecule principal axes'] = Ip  # row-wise principal axes
        dataset['molecule principal moment 1'] = Ipm[0]
        dataset['molecule principal moment 2'] = Ipm[1]
        dataset['molecule principal moment 3'] = Ipm[2]
        dataset['molecule point group is C1'] = not Ipm[0] == Ipm[1] == Ipm[2]

        # rings = mol.GetRingInfo().AtomRings()
        # if len(rings) > 0:
        #     coords = mol.GetConformer().GetPositions()
        #     centroids = []
        #     planes = []
        #     for j in range(len(rings)):
        #         ring = coords[list(rings[j])]
        #         centroids.append(np.average(ring, axis=0))
        #         planes.append(np.linalg.svd(ring - centroids[-1])[2][-1])
        # else:
        #     centroids = []
        #     planes = []
        #
        # dataset['molecule ring centroids'] = centroids
        # dataset['molecule ring planes'] = planes

        return dataset

    def featurize_crystal(self, df):
        dataset = {}
        cell_lengths, cell_angles = df['crystal cell lengths'], np.asarray(df['crystal cell angles']) / 180 * np.pi
        dataset['crystal cell a'] = cell_lengths[0]
        dataset['crystal cell b'] = cell_lengths[1]
        dataset['crystal cell c'] = cell_lengths[2]
        # set angles in natural units
        dataset['crystal alpha'] = cell_angles[0]
        dataset['crystal beta'] = cell_angles[1]
        dataset['crystal gamma'] = cell_angles[2]

        T_fc = coor_trans_matrix('f_to_c', cell_lengths, cell_angles)
        T_cf = coor_trans_matrix('c_to_f', cell_lengths, cell_angles)
        dataset['crystal fc transform'] = T_fc
        dataset['crystal cf transform'] = T_cf

        # canonical conformer is the image of the molecule closest to the origin
        cell_coords = df['crystal reference cell coords']
        fractional_centroids = get_cell_fractional_centroids(cell_coords, T_cf).astype('float16')
        fractional_centroids -= np.floor(fractional_centroids) # ensure they are inside the unit cell
        canonical_centroid_ind = np.argmin(np.linalg.norm(fractional_centroids,axis=1))
        dataset['crystal asymmetric unit centroid x'] = fractional_centroids[canonical_centroid_ind, 0]
        dataset['crystal asymmetric unit centroid y'] = fractional_centroids[canonical_centroid_ind, 1]
        dataset['crystal asymmetric unit centroid z'] = fractional_centroids[canonical_centroid_ind, 2]

        # get coordinates for the canonical conformer
        canonical_mol_coords = cell_coords[canonical_centroid_ind] - np.floor(fractional_centroids[canonical_centroid_ind])
        Ip, _, _ = compute_principal_axes_np(canonical_mol_coords) # ignore masses, as we are interested in the geometric property rather than the inertial one
        target_handedness = int(compute_Ip_handedness(Ip))
        rotation_target = np.eye(3) # if the molecule is left handed, allow it to do a left handed rotation
        rotation_target[0,0] = target_handedness

        rotation_matrix = rotation_target.T @ np.linalg.inv(Ip).T # need transposes to agree with cell builder
        rotvec = Rotation.from_matrix(rotation_matrix.T).as_rotvec() # transposed because we actually want the inverse transform
        dataset['crystal asymmetric unit rotvec 1'] = rotvec[0]
        dataset['crystal asymmetric unit rotvec 2'] = rotvec[1]
        dataset['crystal asymmetric unit rotvec 3'] = rotvec[2]
        dataset['crystal asymmetric unit handedness'] = target_handedness # handedness of the canonical conformer

        # compute overlaps with cell vectors out to 5x5x5
        # get mol axes in fractional basis
        Ip_f = c_f_transform(Ip, T_cf)
        normed_Ip = Ip_f / np.linalg.norm(Ip_f, axis=1)[:, None]
        overlaps = np.einsum('ij,nj->ni', self.normed_fractional_translations, normed_Ip).astype('float16')

        for i1 in range(3):
            for i2 in range(overlaps.shape[1]):
                dataset[f'crystal inertial overlap {i1} to {i2}'] = overlaps[i1, i2]

        return dataset

    def checkForSymmetry(self, crystal_symmetries, type=None):
        return len(set(crystal_symmetries).intersection(self.crystal_symmetry_elements[type])) > 0

    def pointGroupAnalysis(self, numbers, coords):
        atoms = [self.element_symbols[str(number)] for number in numbers]
        try:
            molecule = Molecule(atoms, coords)
            analyzer = PointGroupAnalyzer(molecule, matrix_tolerance = 0.2)
            return str(analyzer.get_pointgroup())  # , analyzer.get_symmetry_operations()
        except:
            return 'error'  # , 'error'
