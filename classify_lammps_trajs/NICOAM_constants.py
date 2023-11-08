class_names = ['V', 'VII', 'VIII', 'I', 'II', 'III', 'IV', 'IX', 'VI', 'Disordered']

ordered_class_names = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'Melt']

defect_names = ['Bulk', 'Surface']

identifier2form = {'NICOAM07': 5,
                   'NICOAM08': 7,
                   'NICOAM09': 8,
                   'NICOAM13': 1,
                   'NICOAM14': 2,
                   'NICOAM15': 3,
                   'NICOAM16': 4,
                   'NICOAM17': 9,
                   'NICOAM18': 6}

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
    'Ca1': 1,
    'Ca2': 2,
    'Ca': 3,
    'C': 4,
    'Nb': 5,
    'N': 6,
    'O': 7,
    'Hn': 8,
    'H4': 9,
    'Ha': 10,
}

num2atomicnum = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 7,
    6: 7,
    7: 8,
    8: 1,
    9: 1,
    10: 1,
}
