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


def cross_validate(name, model, features=[]):
    losses = []
    for i in xrange(0, 10):
        print 'Subset', i
        directory = path.join(dataset.DATA_PATH, dataset.SUBSET + str(i))
        X_train, X_test, y_train, y_test, row_ids = dataset.load_subset(directory, features)

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
        X_train, X_test, y_train, y_test, row_ids = dataset.load_subset(directory, features)

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


