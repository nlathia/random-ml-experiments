from os import path
from csv import writer

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import xgboost as xgb

import pandas as pd
import numpy as np

import dataset


def result_file_name(directory, features, name):
    if len(features) != 0:
        return path.join(directory, '-'.join(features) + '-' + name + '.csv')
    else:
        return path.join(directory, name + '.csv')


def load_file(data_file, new_feature_files):
    """
    Loads some data and splits into X, y
    :param data_file: Full path to CSV file
    :param new_feature_files: Additional features
    :return: X, y, ids
    """
    print '\tRead', data_file
    data = pd.read_csv(data_file)
    X = data.drop(dataset.TARGET_CLASS, axis=1)
    X.drop(dataset.ROW_ID, axis=1, inplace=True)
    y = data[dataset.TARGET_CLASS]
    row_ids = data[dataset.ROW_ID]

    print '\tFeatures', new_feature_files
    for new_feature_file in new_feature_files:
        print '\tRead', new_feature_file
        new_features = pd.read_csv(new_feature_file)
        for igf in dataset.IGNORED_COLUMNS:
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
    train_features = [path.join(directory, f + dataset.TRAIN_FILE) for f in feature_files]
    test_features = [path.join(directory, f + dataset.TEST_FILE) for f in feature_files]

    X_train, y_train, _ = load_file(path.join(directory, dataset.TRAIN_FILE), train_features)
    X_test, y_test, row_ids = load_file(path.join(directory, dataset.TEST_FILE), test_features)
    return X_train, X_test, y_train, y_test, row_ids


def cross_validate(name, model, features=[]):
    losses = []
    for i in xrange(0, 10):
        print 'Subset', i
        directory = path.join(dataset.DATA_PATH, dataset.SUBSET + str(i))
        X_train, X_test, y_train, y_test, row_ids = load_subset(directory, features)

        print '\tTrain', name
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

        loss = log_loss(y_test, y_pred)
        losses.append(loss)
        print '\tLoss: ', loss
        print name, i, loss

        result = pd.DataFrame({dataset.ROW_ID: row_ids, dataset.PREDICTION: y_pred[:, 1]})
        result.to_csv(result_file_name(directory, features, name), index=False)

    print 'CV Result', np.mean(losses), np.std(losses)
    return np.mean(losses), np.std(losses)


def cross_validate_xgb(name='xgb', features=[]):
    # From: https://www.kaggle.com/director/bnp-paribas-cardif-claims-management/simple-xgboost-0-46146/code
    xgboost_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "logloss",
        "eta": 0.01,  # 0.06, #0.01,  # step size shrinkage used in update to prevents overfitting
        # #"min_child_weight": 240,
        "subsample": 0.75,
        "colsample_bytree": 0.68,
        "max_depth": 7
    }
    boost_round = 1800

    losses = []
    for i in xrange(0, 10):
        print 'Subset', i
        directory = path.join(dataset.DATA_PATH, dataset.SUBSET + str(i))
        X_train, X_test, y_train, y_test, row_ids = load_subset(directory, features)

        xgtrain = xgb.DMatrix(X_train, y_train)
        print '\tTrain...'
        clf = xgb.train(xgboost_params, xgtrain, num_boost_round=boost_round, verbose_eval=False, maximize=False)

        xgtest = xgb.DMatrix(X_test)
        y_pred = clf.predict(xgtest, ntree_limit=clf.best_iteration)
        loss = log_loss(y_test, y_pred)
        losses.append(loss)
        print '\tLoss: ', loss, y_pred.shape

        with open(result_file_name(directory, features, name), 'w') as out:
            rows = writer(out)
            rows.writerow([dataset.ROW_ID, dataset.PREDICTION])
            rows.writerows(zip(row_ids, y_pred))

    print 'CV Result', np.mean(losses), np.std(losses)
    return np.mean(losses), np.std(losses)


if __name__ == '__main__':
    fs = ['scaled-']
    models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest-500-gini': RandomForestClassifier(n_estimators=500, criterion='gini', n_jobs=-1),
        'RandomForest-500-entropy': RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1),
        'ExtraTrees-500-gini': ExtraTreesClassifier(n_estimators=500, criterion='gini', n_jobs=-1),
        'ExtraTrees-500-entropy': ExtraTreesClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
    }

    with open('-'.join(fs) + 'result.csv', 'w') as out:
        rows = writer(out)
        for n, m in models.iteritems():
            avg, std = cross_validate(n, m, fs)
            rows.writerow([n, n, avg, std])
            #avg, std = cross_validate_xgb(features=fs)


