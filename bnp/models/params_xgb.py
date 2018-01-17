from os import path

import xgboost as xgb

from sklearn.metrics import log_loss

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import dataset


def load_data():
    train_features = [path.join(dataset.DATA_PATH, f + dataset.TRAIN_FILE) for f in features]
    X, y, _ = dataset.load_file(dataset.ENCODED_TRAIN_FILE, train_features)
    return X, y


def score(params):
    print "Training with params: "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    model = xgb.train(params, dtrain, num_round)
    y_pred = model.predict(dtest)
    loss = log_loss(y_test, y_pred)
    print "\tScore {0}\n\n".format(loss)
    return {'loss': loss, 'status': STATUS_OK}


# From: https://www.kaggle.com/director/bnp-paribas-cardif-claims-management/simple-xgboost-0-46146/code
# And https://github.com/bamine/Kaggle-stuff/blob/master/otto/hyperopt_xgboost.py

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
    'eta': hp.quniform('eta', 0.01, 0.5, 0.025),
    'max_depth': hp.quniform('max_depth', 1, 13, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'num_class': 2,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
    'nthread': 4,
    'silent': 1
}

features = []
with open('xgb-params.txt', 'w') as out:
    for i in xrange(0, 10):
        print 'Subset', i
        directory = path.join(dataset.DATA_PATH, dataset.SUBSET + str(i))
        X_train, X_test, y_train, y_test, row_ids = dataset.load_subset(directory, features)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        # score(ps)

        trials = Trials()
        best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)
        print best
        out.write(best + '\n')
        break

