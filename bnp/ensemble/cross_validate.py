from os import path

import pandas as pd

import dataset

methods = [
    'corr-0.9-ExtraTrees-500-gini',
    'corr-0.9-LogisticRegression',
    'corr-0.9-RandomForest-500-entropy',
    'corr-0.9-RandomForest-500-gini'
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




# test = pd.read_csv(dataset.SUBSET_TEST_FILE)
# y_test = test[dataset.TARGET_CLASS]
#
# logistic = pd.read_csv('data/LogisticRe.csv')
# logistic_loss = log_loss(y_test, logistic[dataset.PREDICTION].values)
#
# rf = pd.read_csv('data/RandomFore.csv')
# rf_loss = log_loss(y_test, rf[dataset.PREDICTION].values)
#
# et = pd.read_csv('data/ExtraTrees.csv')
# et_loss = log_loss(y_test, et[dataset.PREDICTION].values)
#
# preds = pd.DataFrame()
# preds[dataset.ROW_ID] = test[dataset.ROW_ID]
# preds['Logistic'] = logistic[dataset.PREDICTION]
# preds['RF'] = rf[dataset.PREDICTION]
# preds['ET'] = et[dataset.PREDICTION]
# preds['Average'] = preds[['RF', 'ET']].mean(1)
#
# print preds.head()
#
# print 'Logistic', '\t', logistic_loss
# print 'Random Forest', '\t', rf_loss
# print 'Extra Trees', '\t', et_loss
# print ''
# print 'Average', '\t', log_loss(y_test, preds['Average'].values)

# ensemble = LinearRegression()
# ensemble.fit(E_train, y_train)
# y_pred = MinMaxScaler().fit_transform(ensemble.predict(E_test).reshape(-1, 1))
# print 'Linear Ensemble', '\t', log_loss(y_test, y_pred)

