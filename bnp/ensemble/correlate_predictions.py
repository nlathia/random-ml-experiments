from os import path

import pandas as pd

import dataset

methods = [
    'corr-0.9-ExtraTrees-500-gini',
    'corr-0.9-LogisticRegression',
    'corr-0.9-RandomForest-500-entropy',
    'corr-0.9-RandomForest-500-gini',
    'xgb'
]

for subset in xrange(0, 10):
    predictions = {}
    directory = path.join(dataset.DATA_PATH, dataset.SUBSET + str(subset))
    for method in methods:
        pred_file = path.join(directory, method + '.csv')
        if path.exists(pred_file):
            predictions[method] = pd.read_csv(pred_file)[dataset.PREDICTION]

    for i in xrange(0, len(methods)):
        if methods[i] in predictions:
            for j in xrange(i + 1, len(methods)):
                if methods[j] in predictions:
                    print i, '\t', j, '\t', predictions[methods[i]].corr(predictions[methods[j]])



