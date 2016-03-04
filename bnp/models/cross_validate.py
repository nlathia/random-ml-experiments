from os import path

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np

DATA_PATH = '/Users/neal/Desktop/kaggle-experiments/bnp/data'
SUBSET = 'subset'
TRAIN = 'train.csv'
TEST = 'test.csv'

TARGET_CLASS = 'target'
ROW_ID = 'ID'
PREDICTION = 'PredictedProb'


def load_file(data_file):
    """
    Loads some data and splits into X, y
    :param data_file: Full path to CSV file
    :return: X, y
    """
    print '\tRead', data_file
    data = pd.read_csv(data_file)
    X = data.drop(TARGET_CLASS, axis=1)
    X.drop(ROW_ID, axis=1, inplace=True)
    y = data[TARGET_CLASS]
    row_ids = data[ROW_ID]
    return X, y, row_ids


def load_subset(directory):
    """
    Loads a subset of the data
    :param directory: The subset's location
    :return: training Xs and test ys
    """
    X_train, y_train, _ = load_file(path.join(directory, TRAIN))
    X_test, y_test, row_ids = load_file(path.join(directory, TEST))
    return X_train, X_test, y_train, y_test, row_ids


def cross_validate(name, model):
    losses = []
    for i in xrange(0, 10):
        print 'Subset', i
        directory = path.join(DATA_PATH, SUBSET + str(i))
        X_train, X_test, y_train, y_test, row_ids = load_subset(directory)

        print '\tTrain', name
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

        loss = log_loss(y_test, y_pred)
        losses.append(loss)
        print '\tLoss: ', loss

        result = pd.DataFrame({ROW_ID: row_ids, PREDICTION: y_pred[:, 1]})
        result.to_csv(path.join(directory, name + '.csv'), index=False)
    print 'CV Result', np.mean(losses), np.std(losses)


if __name__ == '__main__':
    # m = LogisticRegression()
    # n = 'LogisticRegression'

    m = RandomForestClassifier(n_estimators=500, criterion='gini', n_jobs=-1)
    n = 'RandomForest-500-gini'
    cross_validate(n, m)


# models = [
#     ExtraTreesClassifier(n_estimators=100, criterion='entropy', n_jobs=-1, random_state=42)
# ]





