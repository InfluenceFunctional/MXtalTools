import os
from shutil import copyfile

path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/test_structures/blind_tests/blind_test_5/bk5106sup2/Submissions/'
dump_path = 'C:/Users/mikem/Desktop/CSP_runs/datasets/test_structures/blind_tests/blind_test_5/bk5106sup2/'

blind_test_targets = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                      'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
                      'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', ]


os.chdir(path)
for dir in os.listdir():
    os.chdir(dir)

    files = os.listdir()
    for file in files:
        if file[-4:] == '.cif':
            old_filename = file

            for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
                if blind_test_targets[-1 - j] in old_filename:
                    target = blind_test_targets[ -1 -j]
                    break

            new_filename = target + '_' + dir + old_filename[len(target):]

            copyfile(file, dump_path + new_filename)

    os.chdir('../')