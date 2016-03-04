from csv import reader
from os import path

import pandas as pd
import numpy as np

import dataset


def load_feature_types():
    with open(dataset.FEATURE_TYPES, 'r') as lines:
        rows = reader(lines)
        dtypes = {row[0]: row[1] for row in rows}
    return dtypes


def add_correlated_differences(train, test, dtypes, directory, alpha=0.5):
    new_train = pd.DataFrame()
    new_train[dataset.ROW_ID] = train[dataset.ROW_ID]
    new_test = pd.DataFrame()
    new_test[dataset.ROW_ID] = test[dataset.ROW_ID]
    corrs = set()
    for a in (f for f in train.columns if dtypes[f] != 'object' and f not in dataset.IGNORED_COLUMNS):
        sparse_a_train = train[a][train[a] != -1]
        sparse_a_test = test[a][test[a] != -1]
        for b in (f for f in train.columns if dtypes[f] != 'object' and f not in dataset.IGNORED_COLUMNS):
            if a != b:
                new_feature = '-'.join(sorted([a, b]))
                if new_feature not in corrs:
                    corrs.add(new_feature)
                    sparse_b_train = train[b][train[b] != -1]
                    sparse_b_test = test[b][test[b] != -1]
                    corr = sparse_a_train.corr(sparse_b_train)  # min_periods?
                    if corr >= alpha:
                        print '\t', a, b, corr
                        new_train[new_feature] = abs(sparse_a_train - sparse_b_train)
                        new_train[new_feature].fillna(-1, inplace=True)
                        new_test[new_feature] = abs(sparse_a_test - sparse_b_test)
                        new_test[new_feature].fillna(-1, inplace=True)

    print 'New columns', len(new_train.columns)
    new_train.to_csv(path.join(directory, 'corr-' + str(alpha) + dataset.TRAIN_FILE), index=False)
    new_test.to_csv(path.join(directory, 'corr-' + str(alpha) + dataset.TEST_FILE), index=False)


def scale_features(train, test, dtypes, directory):
    new_train = train.replace(-1, np.nan)
    new_test = test.replace(-1, np.nan)
    for f in (f for f in train.columns if f not in dataset.IGNORED_COLUMNS):
        if dtypes[f] != 'object':
            mean = new_train[f].mean()
            std = new_train[f].std()
            print '\tScaling', f, mean, std
            new_train[f] = (new_train[f] - mean) / std
            new_test[f] = (new_test[f] - mean) / std
    new_train.fillna(0)
    new_train.to_csv(path.join(directory, 'scaled-' + dataset.TRAIN_FILE), index=False)
    new_test.fillna(0)
    new_test.to_csv(path.join(directory, 'scaled-' + dataset.TEST_FILE), index=False)


def apply_to_subsets():
    dtypes = load_feature_types()
    for subset in xrange(0, 10):
        print 'Subset', subset
        directory = path.join(dataset.DATA_PATH, 'subset' + str(subset))
        train = pd.read_csv(path.join(directory, dataset.TRAIN_FILE))
        test = pd.read_csv(path.join(directory, dataset.TEST_FILE))

        # add_correlated_differences(train, test, dtypes, directory, alpha=0.9)
        # add_correlated_differences(train, test, dtypes, directory, alpha=0.5)
        scale_features(train, test, dtypes, directory)

if __name__ == '__main__':
    apply_to_subsets()




