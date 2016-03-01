from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from os import path
import pandas as pd
import numpy as np
import dataset

# Load the training set
print 'Read', dataset.SUBSET_TRAIN_FILE
train = pd.read_csv(dataset.SUBSET_TRAIN_FILE)
train.drop(dataset.ROW_ID, axis=1, inplace=True)

X_train = train.drop(dataset.TARGET_CLASS, axis=1)
y_train = train[dataset.TARGET_CLASS]

# Load the test set
print 'Read', dataset.SUBSET_TEST_FILE
test = pd.read_csv(dataset.SUBSET_TEST_FILE)
test_ids = test[dataset.ROW_ID].values

X_test = test.drop([dataset.TARGET_CLASS, dataset.ROW_ID], axis=1)
y_test = test[dataset.TARGET_CLASS]

models = [
    RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=42),
    ExtraTreesClassifier(n_estimators=250, criterion='entropy', n_jobs=-1, random_state=42)
]

E_train = np.copy(X_train)
E_test = np.copy(X_test)

for i in xrange(0, len(models)):
    name = str(models[i])[:10]
    models[i].fit(X_train, y_train)

    y_pred_train = models[i].predict_proba(X_train)[:, 1]
    y_pred_test = models[i].predict_proba(X_test)[:, 1]

    E_train = np.append(E_train, y_pred_train.reshape(-1, 1), axis=1)
    E_test = np.append(E_test, y_pred_test.reshape(-1, 1), axis=1)

    print name, '\t', log_loss(y_test, y_pred_test)
    # result = pd.DataFrame({dataset.ROW_ID: test_ids, dataset.PREDICTION: y_pred_test[:, 1]})
    # result.to_csv(path.join(dataset.DATA_PATH, name + '.csv'), index=False)

ensemble = LinearRegression()
ensemble.fit(E_train, y_train)
y_pred = MinMaxScaler().fit_transform(ensemble.predict(E_test).reshape(-1, 1))
print 'Linear Ensemble', '\t', log_loss(y_test, y_pred)





