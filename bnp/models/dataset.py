from os import path

import pandas as pd

DATA_PATH = '/Users/neal/Desktop/kaggle-experiments/bnp/data'

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

SUBSET = 'subset'
SUBMIT = 'submit'

FEATURE_TYPES = path.join(DATA_PATH, 'feature-types.csv')

FULL_TRAIN_FILE = path.join(DATA_PATH, TRAIN_FILE)
FULL_TEST_FILE = path.join(DATA_PATH, TEST_FILE)

ENCODED_TRAIN_FILE = path.join(DATA_PATH, 'encoded-' + TRAIN_FILE)
ENCODED_TEST_FILE = path.join(DATA_PATH, 'encoded-' + TEST_FILE)

TARGET_CLASS = 'target'
ROW_ID = 'ID'
PREDICTION = 'PredictedProb'

IGNORED_COLUMNS = [TARGET_CLASS, ROW_ID]

# SUBMISSION_FILE = path.join(DATA_PATH, 'sample_submission.csv')


def load_file(data_file, new_feature_files):
    """
    Loads some data and splits into X, y
    :param data_file: Full path to CSV file
    :param new_feature_files: Additional features
    :return: X, y, ids
    """
    print '\tRead', data_file
    X = pd.read_csv(data_file)
    row_ids = X[ROW_ID]
    X.drop(ROW_ID, axis=1, inplace=True)
    y = None
    if TARGET_CLASS in X:
        print '\tSet y', TARGET_CLASS
        y = X[TARGET_CLASS]
        X.drop(TARGET_CLASS, axis=1, inplace=True)
    print '\tFeatures', new_feature_files
    for new_feature_file in new_feature_files:
        print '\tRead', new_feature_file
        new_features = pd.read_csv(new_feature_file)
        for igf in IGNORED_COLUMNS:
            if igf in new_features.columns:
                new_features.drop(igf, axis=1, inplace=True)
        new_columns = set(new_features.columns)
        old_columns = set(X.columns)
        if len(old_columns.intersection(new_columns)) == 0:
            X = pd.concat([X, new_features], axis=1)
        else:
            for feature in new_features.columns:
                print '\tReplace', feature
                X[feature] = new_features[feature]
    return X, y, row_ids

def load_subset(directory, feature_files):
    """
    Loads a subset of the data
    :param directory: The subset's location
    :param feature_files: Any extra feature files
    :return: training Xs and test ys
    """
    train_features = [path.join(directory, f + TRAIN_FILE) for f in feature_files]
    test_features = [path.join(directory, f + TEST_FILE) for f in feature_files]

    X_train, y_train, _ = load_file(path.join(directory, TRAIN_FILE), train_features)
    X_test, y_test, row_ids = load_file(path.join(directory, TEST_FILE), test_features)
    return X_train, X_test, y_train, y_test, row_ids
