from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
from os import path
import dataset

print 'Read', dataset.SUBSET_TRAIN_FILE
train = pd.read_csv(dataset.SUBSET_TRAIN_FILE)
train.drop(dataset.ROW_ID, axis=1, inplace=True)

X_train = train.drop(dataset.TARGET_CLASS, axis=1)
y_train = train[dataset.TARGET_CLASS]

print 'Read', dataset.SUBSET_TEST_FILE
test = pd.read_csv(dataset.SUBSET_TEST_FILE)
test_ids = test[dataset.ROW_ID].values
X_test = test.drop([dataset.TARGET_CLASS, dataset.ROW_ID], axis=1)
y_test = test[dataset.TARGET_CLASS]

models = [
    LogisticRegression(),
    RandomForestClassifier(n_estimators=500, n_jobs=-1),
    ExtraTreesClassifier(n_estimators=700, criterion='entropy', n_jobs=-1)
]

for model in models:
    name = str(model)[:10]
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    print name, '\t', log_loss(y_test, y_pred)

    result = pd.DataFrame({dataset.ROW_ID: test_ids, dataset.PREDICTION: y_pred[:, 1]})
    result.to_csv(path.join(dataset.DATA_PATH, name + '.csv'), index=False)





