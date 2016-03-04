from os import path

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np

import dataset


def predict(train_file, test_file, name, model):
    train = pd.read_csv(train_file)
    train.drop(dataset.ROW_ID, axis=1, inplace=True)

    X_train = train.drop(dataset.TARGET_CLASS, axis=1)
    y_train = train[dataset.TARGET_CLASS]

    test = pd.read_csv(test_file)
    X_test = test.drop(dataset.ROW_ID, axis=1)
    y_id = test[dataset.ROW_ID]

    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)

    result = pd.DataFrame({dataset.ROW_ID: y_id, dataset.PREDICTION: y_pred[:,1]})
    result.to_csv(path.join(dataset.DATA_PATH, dataset.SUBMIT, name + '.csv'), index=False)


if __name__ == '__main__':
    # m = LogisticRegression()
    # n = 'LogisticRegression'

    # m = RandomForestClassifier(n_estimators=500, criterion='gini', n_jobs=-1)
    # n = 'RandomForest-500-gini'

    # m = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
    # n = 'RandomForest-500-entropy'

    m = ExtraTreesClassifier(n_estimators=500, criterion='gini', n_jobs=-1)
    n = 'ExtraTrees-500-gini'
    predict(n, m)


