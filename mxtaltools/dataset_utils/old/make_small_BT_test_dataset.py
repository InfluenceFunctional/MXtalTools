import pandas as pd
import numpy as np

"""
script to generate a smaller dataset from the complete dataset of blind test samples
currently DEPRECATED - we no longer store datasets in this format
"""

if __name__ == '__main__':
    df = pd.read_pickle(r'D:\crystal_datasets\blind_test_dataset.pkl')

    blind_test_targets = [  # 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
        'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
        'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII']

    target_identifiers = {
        'XVI': 'OBEQUJ',
        'XVII': 'OBEQOD',
        'XVIII': 'OBEQET',
        'XIX': 'XATJOT',
        'XX': 'OBEQIX',
        'XXI': 'KONTIQ',
        'XXII': 'NACJAF',
        'XXIII': 'XAFPAY',
        'XXIII_1': 'XAFPAY01',
        'XXIII_2': 'XAFPAY02',
        'XXXIII_3': 'XAFPAY03',
        'XXXIII_4': 'XAFPAY04',
        'XXIV': 'XAFQON',
        'XXVI': 'XAFQIH',
        'XXXI_1': '2199671_p10167_1_0',
        'XXXI_2': '2199673_1_0',
        # 'XXXI_3': '2199672_1_0',
    }

    # determine which samples go with which targets
    crystals_for_targets = {key: [] for key in blind_test_targets}
    for i in range(len(df['crystal_identifier'])):
        item = df['crystal_identifier'][i]
        for j in range(len(blind_test_targets)):  # go in reverse to account for roman numerals system of duplication
            if blind_test_targets[-1 - j] in item:
                crystals_for_targets[blind_test_targets[-1 - j]].append(i)
                break

    # determine which samples ARE the targets (mixed in the dataloader)
    target_identifiers_inds = {key: [] for key in blind_test_targets}
    for i, item in enumerate(df['crystal_identifier']):
        for key in target_identifiers.keys():
            if item == target_identifiers[key]:
                target_identifiers_inds[key] = i

    inds_to_keep = []
    for key in target_identifiers_inds.keys():
        if target_identifiers_inds[key] != []:
            inds_to_keep.append(target_identifiers_inds[key])

    for key in crystals_for_targets.keys():
        if crystals_for_targets[key] != []:  # take 500 samples per crystal
            num_samples = len(crystals_for_targets[key])
            inds = np.random.choice(min(num_samples, 50), size=min(num_samples, 50), replace=False)
            inds_to_keep.extend([crystals_for_targets[key][ind] for ind in inds])

    assert len(inds_to_keep) == len(list(set(inds_to_keep)))

    df2 = df.iloc[inds_to_keep]

    df2.to_pickle(r'D:\crystal_datasets\test_blind_test_dataset.pkl')