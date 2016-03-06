from os import path
from csv import writer

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import pandas as pd

import dataset


def result_file_name(features, name):
    if len(features) != 0:
        return path.join(dataset.DATA_PATH, dataset.SUBMIT, '-'.join(features) + '-' + name + '.csv')
    else:
        return path.join(dataset.DATA_PATH, dataset.SUBMIT, name + '.csv')


def load_data(features):
    print 'Read', dataset.ENCODED_TRAIN_FILE
    train = pd.read_csv(dataset.ENCODED_TRAIN_FILE)
    train.drop(dataset.ROW_ID, axis=1, inplace=True)

    X_train = train.drop(dataset.TARGET_CLASS, axis=1)
    y_train = train[dataset.TARGET_CLASS]

    print 'Read', dataset.ENCODED_TEST_FILE
    test = pd.read_csv(dataset.ENCODED_TEST_FILE)
    X_test = test.drop(dataset.ROW_ID, axis=1)
    y_id = test[dataset.ROW_ID]

    for feature in features:
        train_features = path.join(dataset.DATA_PATH, feature + dataset.TRAIN_FILE)
        print 'Read', train_features
        fs = pd.read_csv(train_features)
        train = pd.concat([train, fs], axis=1)
        test_features = path.join(dataset.DATA_PATH, feature + dataset.TEST_FILE)
        print 'Read', test_features
        fs = pd.read_csv(test_features)
        test = pd.concat([test, fs], axis=1)

    return X_train, y_train, X_test, y_id


def predict(name, model, features=[]):
    X_train, y_train, X_test, y_id = load_data(features)
    print 'Train', name
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)

    result_file = result_file_name(features, name)
    print 'Write', result_file
    result = pd.DataFrame({dataset.ROW_ID: y_id, dataset.PREDICTION: y_pred[:, 1]})
    result.to_csv(result_file, index=False)


def predict_xgb(name='xgb', features=[]):
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
    X_train, y_train, X_test, y_id = load_data(features)
    xgtrain = xgb.DMatrix(X_train, y_train)
    clf = xgb.train(xgboost_params, xgtrain, num_boost_round=boost_round, verbose_eval=True, maximize=False)

    xgtest = xgb.DMatrix(X_test)
    y_pred = clf.predict(xgtest, ntree_limit=clf.best_iteration)

    result_file = result_file_name(features, name)
    print 'Write', result_file
    with open(result_file, 'w') as out:
        rows = writer(out)
        rows.writerow([dataset.ROW_ID, dataset.PREDICTION])
        rows.writerows(zip(y_id, y_pred))


if __name__ == '__main__':
    # m = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
    # n = 'RandomForest-500-entropy'
    # predict(n, m)
    predict_xgb()

