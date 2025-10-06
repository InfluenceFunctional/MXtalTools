from ccdc.search import EntryReader, TextNumericSearch
from ccdc import io
import tqdm
"""
simple script for dumping the CSD as .cif files for future processing
optionally can directly pull crystal information from an EntryReader, but 
this way aligns with our pipeline's general cif compatibility
"""

if __name__ == '__main__':
    cif_dump_path = r'D:\crystal_datasets\CSD_dump/'

    allhits = []
    crystal_reader = EntryReader('CSD')
    for i in range(1921, 2030):  # pull the entire CSD
        searcher = TextNumericSearch()
        searcher.add_citation(year=i)
        hitlist = searcher.search()
        allhits.extend(hitlist)
        del hitlist

    save_path = cif_dump_path  # path to save all the .cif files
    for hit in tqdm.tqdm(allhits):
        if hit.entry.has_3d_structure:
            if not hit.entry.is_polymeric:
                if not hit.entry.crystal.has_disorder:
                    if len(hit.molecule.atoms) > 0:
                        with io.CrystalWriter(save_path + '/' + hit.entry.identifier +'.cif') as writer:
                            writer.write(hit.crystal)
