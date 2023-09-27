from ccdc.search import EntryReader, TextNumericSearch
from ccdc import io
import tqdm

allhits = []
crystal_reader = EntryReader('CSD')
for i in range(1921, 2023):  # pull the entire CSD
    searcher = TextNumericSearch()
    searcher.add_citation(year=i)
    hitlist = searcher.search()
    allhits.extend(hitlist)
    del hitlist

save_path = r'D:\CSD_dump/'
for hit in tqdm.tqdm(allhits):
    if hit.entry.has_3d_structure:
        if not hit.entry.is_polymeric:
            if not hit.entry.crysta.has_disorder:
                if len(hit.molecule.atoms) > 0:
                    with io.CrystalWriter(save_path + '/' + hit.entry.identifier +'.cif') as writer:
                        writer.write(hit.crystal)
