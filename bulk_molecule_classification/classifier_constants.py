nic_class_names = ['V', 'VII', 'VIII', 'I', 'II', 'III', 'IV', 'IX', 'VI', 'Melt']
nic_ordered_class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'Melt']
urea_class_names = ['A', 'B', 'C', 'I', 'III', 'IV', 'Melt']
urea_ordered_class_names = ['A', 'B', 'C', 'I', 'III', 'IV', 'Melt']

defect_names = ['Bulk', 'Surface']

identifier2form = {'NICOAM07': 5,
                   'NICOAM08': 7,
                   'NICOAM09': 8,
                   'NICOAM13': 1,
                   'NICOAM14': 2,
                   'NICOAM15': 3,
                   'NICOAM16': 4,
                   'NICOAM17': 9,
                   'NICOAM18': 6,
                   'NIC_Melt': 10,
                   'ureaA': 1,
                   'ureaB': 2,
                   'ureaC': 3,
                   'ureaI': 4,
                   'ureaIII': 5,
                   'ureaIV': 6,
                   'UREA_Melt': 7,
                   }

form2index = {0: 4,
              1: 6,
              2: 7,
              3: 0,
              4: 1,
              5: 2,
              6: 3,
              7: 8,
              8: 5,
              9: 9}

index2form = {value: key for key, value in form2index.items()}

type2num = {
    'Ca1': 6,
    'Ca2': 6,
    'Ca': 6,
    'C': 6,
    'Nb': 7,
    'N': 7,
    'O': 8,
    'Hn': 1,
    'H4': 1,
    'Ha': 1,
    'H': 1,
    'N1': 7,
    'N2': 7,
}
#
# num2atomicnum = {  # for nicotinamide
#     1: 1,  # Ha
#     2: 6,  # Hn
#     3: 8,  # o
#     4: 6,  # c
#     5: 6,  # ca
#     6: 7,  # N
#     7: 7,  # nb
#     8: 1,  # H4
#     9: 6,  # ca1
#     10: 6,  # ca2
#     # 11: 1,  # H
#     # 12: 7,  # n1
#     # 13: 7  # n2
# }
# "old"
# what we are actually using
num2atomicnum = {  # for nicotinamide
    1: 1,  #
    8: 1,  #
    2: 1,  #
    6: 7,  #
    7: 7,  #
    4: 6,  #
    5: 6,  #
    3: 8,
    9: 6,
    10: 6,
}

# num2atomicnum_new = {  # for nicotinamide
#     3: 1,  #
#     5: 1,  #
#     4: 1,  #
#     6: 7,  #
#     7: 7,  #
#     1: 6,  #
#     2: 6,  #
#     8: 8,  #
#     9: 6,
#     10: 6,
# }
#
