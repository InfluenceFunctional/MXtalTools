from ccdc.search import EntryReader, TextNumericSearch
from ccdc import io
import tqdm
from pymatgen.io import cif
import os
import numpy as np
from mendeleev import element as element_table
from scipy.spatial.distance import cdist

symbol_dict = {}
for i in range(1,100):
    symbol_dict[element_table(i).symbol] = i

cifs_path = r'D:/CSD_dump/'
os.chdir(cifs_path)
cifs_list = os.listdir()
for ind, cif_path in enumerate(cifs_list):
    if ind < 100:
        print(ind)
        try:
            ccdc_reader = io.CrystalReader(cif_path, format='cif')
            ccry = ccdc_reader[0]
            pmg_reader = cif.CifParser(cif_path)
            pstruct = pmg_reader.get_structures()[0]
            pcry = pmg_reader.as_dict()[ccry.identifier]
        except:  # come files cannot be read
            continue

        '''compare outputs'''

        # cell parameters
        clengths = np.asarray(ccry.reduced_cell.cell_lengths)
        cangles = np.asarray(ccry.reduced_cell.cell_angles)
        plengths = np.asarray(pstruct.lattice.abc)
        pangles = np.asarray(pstruct.lattice.angles)

        assert np.sum(np.abs(plengths.sum()-clengths.sum())) < 1e-5
        assert np.sum(np.abs(pangles.sum()-cangles.sum())) < 1e-5

        # space group
        psymbol, psgind = pstruct.get_space_group_info()
        csymbol = ccry.spacegroup_symbol
        csgind, csetting = ccry.spacegroup_number_and_setting

        #assert csymbol == psymbol
        assert psgind == csgind

        # atom types in asymmetric unit
        cnums = np.asarray([atom.atomic_number for atom in ccry.molecule.atoms if atom.atomic_number != 1])

        psymbols = pcry['_atom_site_type_symbol']
        pnums = np.asarray([symbol_dict[symbol] for symbol in psymbols if symbol != 'H'])

        cnums.sort()
        pnums.sort()

        assert np.sum(np.abs(cnums - pnums)) < 1e-5

        # asym unit and full cell fractional positions & coordinates
        # cref_cell = ccry.packing(box_dimensions=((0, 0, 0), (1, 1, 1)), inclusion='OnlyAtomsIncluded')
        # cref_coords = np.asarray([atom.coordinates for atom in cref_cell.atoms])
        # cref_numbers = np.asarray([atom.atomic_number for atom in cref_cell.atoms])
        # pref_coords = pstruct.cart_coords
        # pref_numbers = pstruct.atomic_numbers
        #
        # cref_coords -= cref_coords.mean(0)
        # pref_coords -= pref_coords.mean(0)
        # #
        # # mol = Atoms(positions = cref_coords, numbers = cref_numbers)
        # # mol2 = Atoms(positions = pref_coords, numbers = pref_numbers)
        # # view([mol,mol2])
        #
        # cdistmat = cdist(cref_coords, cref_coords)
        # pdistmat = cdist(pref_coords, pref_coords)
        #
        # csort = cdistmat[np.argsort(np.linalg.norm(cdistmat,axis=-1))]
        # psort = pdistmat[np.argsort(np.linalg.norm(pdistmat,axis=-1))]

        # asym unit fractional coordinates
        cfrac = np.asarray([atom.fractional_coordinates for atom in ccry.asymmetric_unit_molecule.atoms])
        pfrac = np.concatenate((np.asarray(pcry['_atom_site_fract_x'])[:,None],
                                np.asarray(pcry['_atom_site_fract_y'])[:,None],
                                np.asarray(pcry['_atom_site_fract_z'])[:,None]), axis=-1)
        for i in range(3):
            for j in range(len(pfrac)):
                pfrac[j, i] = pfrac[j, i].split('(')[0]  # remove uncertainties

        pfrac = pfrac.astype(float)

        assert np.mean(np.abs(pfrac-cfrac)) < 1e-5


        # symmetry operations


aa = 1

"""
manual cleanup checks
- distmat clears minimum cutoffs
- no fucking protons

"""