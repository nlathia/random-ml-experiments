from os import path
from csv import writer

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np

import dataset


def load_file(data_file):
    """
    Loads some data and splits into X, y
    :param data_file: Full path to CSV file
    :return: X, y
    """
    print '\tRead', data_file
    data = pd.read_csv(data_file)
    X = data.drop(dataset.TARGET_CLASS, axis=1)
    X.drop(dataset.ROW_ID, axis=1, inplace=True)
    y = data[dataset.TARGET_CLASS]
    row_ids = data[dataset.ROW_ID]
    return X, y, row_ids


def load_subset(directory, train_file, test_file):
    """
    Loads a subset of the data
    :param directory: The subset's location
    :return: training Xs and test ys
    """
    X_train, y_train, _ = load_file(path.join(directory, train_file))
    X_test, y_test, row_ids = load_file(path.join(directory, test_file))
    return X_train, X_test, y_train, y_test, row_ids


def cross_validate(name, model, train_file, test_file):
    losses = []
    for i in xrange(0, 10):
        print 'Subset', i
        directory = path.join(dataset.DATA_PATH, dataset.SUBSET + str(i))
        X_train, X_test, y_train, y_test, row_ids = load_subset(train_file, test_file)

        print '\tTrain', name
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

        loss = log_loss(y_test, y_pred)
        losses.append(loss)
        print '\tLoss: ', loss

        result = pd.DataFrame({dataset.ROW_ID: row_ids, dataset.PREDICTION: y_pred[:, 1]})
        result.to_csv(path.join(directory, name + '.csv'), index=False)
    print 'CV Result', np.mean(losses), np.std(losses)
    return np.mean(losses), np.std(losses)


if __name__ == '__main__':
    TRAIN = 'corr-0.9train.csv'
    TEST = 'corr-0.9test.csv'

    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest-500-gini': RandomForestClassifier(n_estimators=500, criterion='gini', n_jobs=-1),
        'RandomForest-500-entropy': RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1),
        'ExtraTrees-500-gini': ExtraTreesClassifier(n_estimators=500, criterion='gini', n_jobs=-1),
        'ExtraTrees-500-entropy': ExtraTreesClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
    }

    with open('corr-0.9-result.csv', 'w') as out:
        rows = writer(out)
        for n, m in models.iteritems():
            avg, std = cross_validate(n, m, TRAIN, TEST)
            rows.writerow([TRAIN, n, avg, std])
