FF_PATH_DICT = {'nicotinamide': 'gaff2_nicotinamid_long.lt',
                'acridine': 'gaff2_acridine.lt'}

MOLECULE_SYM_INDICES = {'acridine': {ind: ind for ind in range(1, 24)},
                        'anthracene': {ind: ind for ind in range(1, 25)},
                        '2,7-dihydroxynaphthalene': {ind: ind for ind in range(1, 21)},

                        'nicotinamide': {ind: ind for ind in range(1, 16)},
                        # todo get correct ordering of atom indices
                        'benzamide': {ind: ind for ind in range(1, 17)},
                        'isonicotinamide': {ind: ind for ind in range(1, 16)},
                        }

MOLECULE_ATOM_TYPES_MASSES = {
    'nicotinamide': "Masses\n\n1 1.008  # ha\n2 1.008  # h4\n3 1.008  # hn\n4 14.01  # n\n5 14.01  # nb\n6 12.01  # c\n7 12.01  # ca\n8 16.00  # o\n",
    # 'nicotinamide': "Masses\n\n1 1.008  # H\n2 12.01  # C\n3 14.01  # N\n4 16.00  # O\n",  # simplified atom types
    'acridine': "Masses\n\n1 14.01  # nb\n2 12.01  # ca\n3 1.008  # ha\n"}

MOLECULE_NUM_ATOM_TYPES = {'nicotinamide': 8,  # 1-8
                           'acridine': 3}  # CHNO (oxygen from defects)  # todo adjust this when we have carbon-containing additives

MOLECULE_MASSES = {'acridine': 179.22,
                   'anthracene': 178.23,
                   '2,7-dihydroxynaphthalene': 160.17,

                   'nicotinamide': 122.12,
                   'benzamide': 121.14,
                   'isonicotinamide': 122.12,
                   }

MOLECULE_NUM_ATOMS = {'acridine': 23,
                      'anthracene': 24,
                      '2,7-dihydroxynaphthalene': 20,

                      'nicotinamide': 15,
                      'benzamide': 16,
                      'isonicotinamide': 15,
                      }

MOLECULE_SHORTHAND = {'nicotinamide': 'nic1',
                      'acridine': 'AC1'}

ATOM_TYPES = {
    'nicotinamide': {
        1: 7,
        2: 7,
        3: 5,
        4: 7,
        5: 7,
        6: 7,
        7: 6,
        8: 8,
        9: 4,
        10: 3,
        11: 3,
        12: 2,
        13: 2,
        14: 1,
        15: 1
    },
    'benzamide': {
        1: 7,
        2: 7,
        3: 7,
        4: 7,
        5: 7,
        6: 7,
        7: 6,
        8: 8,
        9: 4,
        10: 3,
        11: 3,
        12: 2,
        13: 2,
        14: 1,
        15: 1,
        16: 1
    },
    'isonicotinamide': {  # todo missing N
        1: 7,
        2: 7,
        3: 7,
        4: 7,
        5: 7,
        6: 7,
        7: 6,
        8: 8,
        9: 4,
        10: 3,
        11: 3,
        12: 2,
        13: 2,
        14: 1,
        15: 1
    },
    'acridine': {
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 3,
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 3,
        22: 3,
        23: 3
    }  # todo add anthracene and isonicotinamide
}

ATOM_CHARGES = {
    'nicotinamide': {
        1: -0.326615,
        2: 0.380568,
        3: -0.585364,
        4: 0.384166,
        5: -0.38538,
        6: 0.173788,
        7: 0.6054,
        8: -0.479634,
        9: -0.779885,
        10: 0.357505,
        11: 0.357505,
        12: 0.027993,
        13: 0.034858,
        14: 0.157212,
        15: 0.077882
    },
    'benzamide': {
        1: -0.073516,
        2: -0.081906,
        3: -0.123099,
        4: -0.104813,
        5: -0.123099,
        6: -0.081906,
        7: 0.58899,
        8: -0.47625,
        9: -0.787316,
        10: 0.350274,
        11: 0.350274,
        12: 0.102542,
        13: 0.119669,
        14: 0.117946,
        15: 0.119669,
        16: 0.102542
    },
    'isonicotinamide': {
        1: 0.228047,
        2: -0.429772,
        3: 0.429792,
        4: -0.608913,
        5: 0.429792,
        6: -0.429772,
        7: 0.560881,
        8: -0.462231,
        9: -0.809865,
        10: 0.366217,
        11: 0.366217,
        12: 0.153507,
        13: 0.026297,
        14: 0.026297,
        15: 0.153507
    },
    'acridine': {
        1: -0.693636,
        2: 0.534823,
        3: -0.002672,
        4: -0.20686,
        5: -0.002672,
        6: 0.534823,
        7: -0.274751,
        8: -0.100851,
        9: -0.140261,
        10: -0.187088,
        11: -0.187088,
        12: -0.140261,
        13: -0.100851,
        14: -0.274751,
        15: 0.151995,
        16: 0.152509,
        17: 0.128637,
        18: 0.130168,
        19: 0.133735,
        20: 0.133735,
        21: 0.130168,
        22: 0.128637,
        23: 0.152509
    }  # todo add anthracene and isonicotinamide
}

structure2polymorph = {
    "nicotinamide/NICOAM07": 4,  # form 5
    "nicotinamide/NICOAM08": 6,  # form 7, Tm 381, eta
    "nicotinamide/NICOAM09": 7,  # form 8, Tm 377.5, theta
    "nicotinamide/NICOAM13": 0,  # form 1, Tm 402K, alpha
    "nicotinamide/NICOAM14": 1,  # form 2, Tm 390K, beta
    "nicotinamide/NICOAM15": 2,  # form 3, Tm 388K, gamma
    "nicotinamide/NICOAM16": 3,  # form 4, Tm 387K, delta
    "nicotinamide/NICOAM17": 8,  # form 9, Tm 376K, iota
    "nicotinamide/NICOAM18": 5,  # form 6, Tm 382.5K, zeta
    "nicotinamide/liquid": 9,  # form 6, Tm 382.5K, zeta
    "acridine/Form2": 0,
    "acridine/Form3": 1,
    "acridine/Form4": 2,
    "acridine/Form6": 3,
    "acridine/Form7": 4,
    "acridine/Form8": 5,
    "acridine/Form9": 6,
    "acridine/liquid": 7,
}

polymorph2form = {
    'nicotinamide': {
        4: 'Nic V',  # form 5
        6: 'Nic VII',  # form 7, Tm 381, eta
        7: 'Nic VIII',  # form 8, Tm 377.5, theta
        0: 'Nic I',  # form 1, Tm 402K, alpha
        1: 'Nic II',  # form 2, Tm 390K, beta
        2: 'Nic III',  # form 3, Tm 388K, gamma
        3: 'Nic IV',  # form 4, Tm 387K, delta
        8: 'Nic IX',  # form 9, Tm 376K, iota
        5: 'Nic V',  # form 6, Tm 382.5K, zeta
        9: 'Nic Liq',  # form 6, Tm 382.5K, zeta
    },
    'acridine': {
        0: 'Acr II',
        1: 'Acr III',
        2: 'Acr IV',
        3: 'Acr VI',
        4: 'Acr VII',
        5: 'Acr VIII',
        6: 'Acr IX',
        7: 'Acr Liq',
    }
}
