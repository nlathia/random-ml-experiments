from csv import reader
from os import path

import pandas as pd

import dataset


def load_feature_types():
    with open(dataset.FEATURE_TYPES, 'r') as lines:
        rows = reader(lines)
        dtypes = {row[0]: row[1] for row in rows}
    return dtypes


def add_correlated_differences(alpha=0.5):
    dtypes = load_feature_types()
    for subset in xrange(0, 10):
        print 'Subset', subset
        directory = path.join(dataset.DATA_PATH, 'subset' + str(subset))

        train = pd.read_csv(path.join(directory, dataset.TRAIN_FILE))
        test = pd.read_csv(path.join(directory, dataset.TEST_FILE))

        new_train = pd.DataFrame()
        new_test = pd.DataFrame()
        corrs = set()

        for a in (f for f in train.columns if dtypes[f] != 'object'):
            sparse_a = train[a][train[a] != -1]
            for b in (f for f in train.columns if dtypes[f] != 'object'):
                if a != b:
                    new_feature = '-'.join(sorted([a, b]))
                    if new_feature not in corrs:
                        corrs.add(new_feature)
                        corr = sparse_a.corr(train[b][train[b] != -1])  # min_periods?
                        if corr >= alpha:
                            print '\t', a, b, corr
                            new_train[new_feature] = abs(train[a] - train[b])
                            new_test[new_feature] = abs(test[a] - test[b])

        print 'New columns', len(new_train.columns)
        new_train.to_csv(path.join(directory, 'corr-' + str(alpha) + dataset.TRAIN_FILE))
        new_test.to_csv(path.join(directory, 'corr-' + str(alpha) + dataset.TEST_FILE))


add_correlated_differences(alpha=0.9)
add_correlated_differences(alpha=0.5)




