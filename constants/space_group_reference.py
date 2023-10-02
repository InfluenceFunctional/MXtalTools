import numpy as np
import pandas as pd


def loadSpaceGroups():
    """
    generate dataframe containing all space groups info
    not currently used
    """
    sgDict = pd.DataFrame(columns=['system', 'laue', 'class', 'ce', 'SG'], index=np.arange(0, 230 + 1))
    sgDict.loc[0] = pd.Series({'system': 'n/a', 'laue': 'n/a', 'class': 'n/a', 'ce': 'n/a', 'SG': 'n/a'})  # add an entry for misc / throwaway categories

    # Triclinic
    i = 1
    sgDict.loc[i] = pd.Series({'system': 'triclinic',
                               'laue': '-1',
                               'class': '1',
                               'ce': 'p',
                               'SG': 'P1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'triclinic',
                               'laue': '-1',
                               'class': '-1',
                               'ce': 'p',
                               'SG': 'P-1'
                               })
    i += 1

    # Monoclinic
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'p',
                               'SG': 'P2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'p',
                               'SG': 'P21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'c',
                               'SG': 'C2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'p',
                               'SG': 'Pm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'p',
                               'SG': 'Pc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'c',
                               'SG': 'Cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'c',
                               'SG': 'Cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P2/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P21/m'  # alt - P21m
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'c',
                               'SG': 'C2/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P2/c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P21/c'  # alt - P21c
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'c',
                               'SG': 'C2/c'
                               })
    i += 1

    # Orthorhombic
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P2221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P21212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P212121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'c',
                               'SG': 'C2221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'c',
                               'SG': 'C222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'f',
                               'SG': 'F222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'i',
                               'SG': 'I222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'i',
                               'SG': 'I212121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmc21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pcc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pma2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pca21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pnc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmn21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pna21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pnn2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Cmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Cmc21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Ccc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Amm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Abm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Ama2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Aba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'f',
                               'SG': 'Fmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'f',
                               'SG': 'Fdd2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Imm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Iba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Ima2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnnn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pccm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pban'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnna'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmna'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pcca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbam'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pccn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnnm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmmn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbcn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cccm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Ccca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'f',
                               'SG': 'Fmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'f',
                               'SG': 'Fddd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Immm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Ibam'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Ibca'  # aka Ibcm
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Imma'
                               })
    i += 1

    # Tetragonal
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P41'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P42'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P43'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'i',
                               'SG': 'I4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'i',
                               'SG': 'I41'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '-4',
                               'ce': 'p',
                               'SG': 'P-4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '-4',
                               'ce': 'i',
                               'SG': 'I-4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P4/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P42/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P4/n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P42/n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'i',
                               'SG': 'I4/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'i',
                               'SG': 'I41/a'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4122'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P41212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P42212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4322'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P43212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'i',
                               'SG': 'I422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'i',
                               'SG': 'I4122'  # aka I4212
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4bm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42nm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4nc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42mc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42bc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I4mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I4cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I41md'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I41cd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-42m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-42c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-421m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-421c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4b2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4n2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'i',
                               'SG': 'I-4m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'i',
                               'SG': 'I-4c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'i',
                               'SG': 'I-42m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'I-42d'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mcc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nbm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nnc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mbm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mnc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/ncc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mmc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nbc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nnm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mbc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mnm'  # incorrectly /mcm in source
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nmc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/ncm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I4/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I4/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I41/amd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I41/acd'
                               })
    i += 1

    # Trigonal
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P31'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P32'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'r',
                               'SG': 'R3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '-3',
                               'ce': 'p',
                               'SG': 'P-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '-3',
                               'ce': 'r',
                               'SG': 'R-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P312'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P321'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P3112'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P3121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P3212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P3221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'r',
                               'SG': 'R32'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'p',
                               'SG': 'P3m1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '31m',
                               'ce': 'p',
                               'SG': 'P31m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'p',
                               'SG': 'P3c1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '31m',
                               'ce': 'p',
                               'SG': 'P31c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'r',
                               'SG': 'R3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'r',
                               'SG': 'R3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-31m',
                               'ce': 'p',
                               'SG': 'P-31m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-31m',
                               'ce': 'p',
                               'SG': 'P-31c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'p',
                               'SG': 'P-3m1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'p',
                               'SG': 'P-3c1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'r',
                               'SG': 'R-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'r',
                               'SG': 'R-3c'
                               })
    i += 1

    # Hexagonal
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P6'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P61'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P65'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P62'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P64'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P63'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '-6',
                               'ce': 'p',
                               'SG': 'P-6'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6/m',
                               'ce': 'p',
                               'SG': 'P6/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6/m',
                               'ce': 'p',
                               'SG': 'P63/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P622'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6122'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6522'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6322'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P6mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P6cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P63cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P63mc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-6m2',
                               'ce': 'p',
                               'SG': 'P-6m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-6m2',
                               'ce': 'p',
                               'SG': 'P-6c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-62m',
                               'ce': 'p',
                               'SG': 'P-62m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-62m',
                               'ce': 'p',
                               'SG': 'P-62c'  # error in source, missing negative http://pd.chem.ucl.ac.uk/pdnn/symm3/allsgp.htm
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P6/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P6/mcc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P63/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P63/mmc'
                               })
    i += 1

    # Cubic
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'p',
                               'SG': 'P23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'f',
                               'SG': 'F23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'i',
                               'SG': 'I23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'p',
                               'SG': 'P213'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'i',
                               'SG': 'I213'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pm-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pn-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'f',
                               'SG': 'Fm-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'f',
                               'SG': 'Fd-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'i',
                               'SG': 'Im-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pa-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'Ia-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4232'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'f',
                               'SG': 'F432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'f',
                               'SG': 'F4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'I432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4332'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'I4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'p',
                               'SG': 'P-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'f',
                               'SG': 'F-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'i',
                               'SG': 'I-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'p',
                               'SG': 'P-43n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'f',
                               'SG': 'F-43c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'i',
                               'SG': 'I-43d'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pm-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pn-3n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pm-3n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pn-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fm-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fm-3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fd-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fd-3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'i',
                               'SG': 'Im-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'i',
                               'SG': 'Ia-3d'
                               })

    return sgDict
