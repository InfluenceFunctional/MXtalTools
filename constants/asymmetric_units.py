raw_asym_unit_dict = {  # https://www.lpl.arizona.edu/PMRG/sites/lpl.arizona.edu.PMRG/files/ITC-Vol.A%20%282005%29%28ISBN%200792365909%29.pdf
    '1': [1, 1, 1],  # P1
    '2': [0.5, 1, 1],  # P-1
    '3': [1, 1, 0.5],  # P2
    '4': [1, 1, 0.5],  # P21
    '5': [0.5, 0.5, 1],  # C2
    '6': [1, 0.5, 1],  # Pm
    '7': [1, 0.5, 1],  # Pc
    '8': [1, 0.25, 1],  # Cm
    '9': [1, 0.25, 1],  # Cc
    '10': [0.5, 0.5, 1],  # P2/m
    '11': [1, 0.25, 1],  # P21/m
    '12': [0.5, 0.25, 1],  # C2/m
    '13': [0.5, 1, 0.5],  # P2/c
    '14': [1, 0.25, 1],  # P21/c
    '15': [0.5, 0.5, 0.5],  # C2/c
    '16': [0.5, 0.5, 1],  # P222
    '17': [0.5, 0.5, 1],  # P2221
    '18': [0.5, 0.5, 1],  # P21212
    '19': [0.5, 0.5, 1],  # P212121
    '20': [0.5, 0.5, 0.5],  # C2221
    '21': [0.25, 0.5, 1],  # C222
    '22': [0.25, 0.25, 1],  # F222
    '23': [0.5, 0.5, 0.5],  # I222
    '24': [0.5, 0.5, 0.5],  # I212121
    '25': [0.5, 0.5, 1],  # Pmm2
    '26': [0.5, 0.5, 1],  # Pmc21
    '27': [0.5, 0.5, 1],  # Pcc2
    '28': [0.25, 1, 1],  # Pma2
    '29': [0.25, 1, 1],  # Pca21
    '30': [0.5, 1, 0.5],  # Pnc2
    '31': [0.5, 0.5, 1],  # Pmn21
    '32': [0.5, 0.5, 1],  # Pba2
    '33': [0.5, 0.5, 1],  # Pna21
    '34': [0.5, 0.5, 1],  # Pnn2
    '35': [0.25, 0.5, 1],  # Cmm2
    '36': [0.5, 0.5, 0.5],  # Cmc21
    '37': [0.25, 0.5, 1],  # Ccc2
    '38': [0.5, 0.5, 0.5],  # Amm2
    '39': [0.5, 0.25, 1],  # Aem2
    '40': [0.25, 0.5, 1],  # Ama2
    '41': [0.5, 0.5, 0.5],  # Aea2
    '42': [0.25, 0.25, 1],  # Fmm2
    '43': [0.25, 0.25, 1],  # Fdd2
    '44': [0.5, 0.5, 0.5],  # Imm2
    '45': [0.5, 0.5, 0.5],  # Iba2
    '46': [0.25, 1, 0.5],  # Ima2
    '47': [0.5, 0.5, 0.5],  # Pmmm
    '48': [0.25, 0.5, 1],  # Pnnn
    '49': [0.5, 0.5, 0.5],  # Pccm
    '50': [0.5, 0.5, 0.5],  # Pban
    '51': [0.25, 0.5, 1],  # Pmma
    '52': [1, 0.25, 0.5],  # Pnna
    '53': [0.5, 1, 0.25],  # Pmna
    '54': [0.5, 0.5, 0.5],  # Pcca
    '55': [0.5, 0.5, 0.5],  # Pbam
    '56': [0.25, 1, 0.5],  # Pccn
    '57': [0.5, 1, 0.25],  # Pbcm
    '58': [0.5, 0.5, 0.5],  # Pnnm
    '59': [0.5, 0.5, 0.5],  # Pmmn
    '60': [0.5, 0.5, 0.5],  # Pbcn
    '61': [0.5, 0.5, 0.5],  # Pbca
    '62': [0.5, 0.25, 1],  # Pnma
    '63': [0.5, 0.5, 0.25],  # Cmcm
    '64': [0.25, 0.5, 0.5],  # Cmce
    '65': [0.25, 0.5, 0.5],  # Cmmm
    '66': [0.25, 0.5, 0.5],  # Cccm
    '67': [0.5, 0.25, 0.5],  # Cmme
    '68': [0.25, 0.5, 0.5],  # Ccce
    '69': [0.25, 0.25, 0.5],  # Fmmm
    '70': [0.125, 0.25, 1],  # Fddd
    '71': [0.25, 0.5, 0.5],  # Immm
    '72': [0.25, 0.5, 0.5],  # Ibam
    '73': [0.25, 0.5, 0.5],  # Ibca
    '74': [0.25, 0.25, 1],  # Imma
    '75': [0.5, 0.5, 1],  # P4
    '76': [0.5, 0.5, 1],  # P41
    '77': [0.5, 0.5, 1],  # P42
    '78': [0.5, 0.5, 1],  # P43
    '79': [0.5, 0.5, 0.5],  # I4
    '80': [0.5, 1, 0.25],  # I41
    '81': [0.5, 0.5, 1],  # P-4
    '82': [0.5, 0.5, 0.5],  # I-4
    '83': [0.5, 0.5, 0.5],  # P4/m
    '84': [0.5, 0.5, 0.5],  # P42/m
    '85': [0.5, 0.5, 0.5],  # P4/n
    '86': [0.5, 1, 0.25],  # P42/n
    '87': [0.5, 0.5, 0.25],  # I4/m
    '88': [0.25, 0.25, 1],  # I41/a
    '89': [0.5, 0.5, 0.5],  # P422
    '90': [0.5, 0.5, 0.5],  # P4212
    '91': [1, 1, 0.125],  # P4122
    '92': [1, 1, 0.125],  # P41212
    '93': [0.5, 1, 0.25],  # P4222
    '94': [0.5, 0.5, 0.5],  # P4212
    '95': [1, 1, 0.125],  # P4322
    '96': [1, 1, 0.125],  # P43212
    '97': [0.5, 0.5, 0.25],  # I422
    '98': [0.5, 1, 0.125],  # I4122
    # '99': [0.5, 0.5, 1], x<y  # P4mm
    '103': [0.5, 0.5, 0.5],  # P4cc
    '104': [0.5, 0.5, 0.5],  # P4nc
    '105': [0.5, 0.5, 0.5],  # P42mc
    '106': [0.5, 0.5, 0.5],  # P42bc
    # '107'
    '109': [0.5, 0.5, 0.25],  # I41md
    '110': [0.5, 0.5, 0.25],  # I41cd
    # '111':
    '112': [0.5, 0.5, 0.5],  # P-42c
    # '113'
    '114': [0.5, 0.5, 0.5],  # P-421c
    '115': [0.5, 0.5, 0.5],  # P-4m2
    '116': [0.5, 1, 0.25],  # P-4c2
    '117': [0.5, 0.5, 0.5],  # P-4b2
    '118': [0.5, 1, 0.25],  # P-4n2
    '119': [0.5, 0.5, 0.25],  # I-4m2
    '120': [0.5, 0.5, 0.25],  # I-4c2
    # '121'
    '122': [0.5, 1, 0.125],  # I-42d
    # '123'
    '124': [0.5, 0.5, 0.25],  # P4/mcc
    # '125'
    '126': [0.5, 0.5, 0.25],  # P4/nnc
    # '127'
    '128': [0.5, 0.5, 0.25],  # P4/mnc
    # '129'
    '130': [0.5, 0.5, 0.25],  # P4/ncc
    '131': [0.5, 0.5, 0.25],  # P42/mmc
    # '132'
    '133': [0.5, 0.5, 0.25],  # P42/nbc
    # '134'
    '135': [0.5, 0.5, 0.25],  # p42/mbc
}

'''
if we do not have a parameterization for a particular space group's asymmetric unit
set it for now as unrestricted, so that we can at least use it without crashing the code
effectively, all unparameterized SGs are treated as P1
'''
asym_unit_dict = raw_asym_unit_dict.copy()
for i in range(1, 231):
    if str(i) not in raw_asym_unit_dict.keys():
        asym_unit_dict[str(i)] = [1, 1, 1]
