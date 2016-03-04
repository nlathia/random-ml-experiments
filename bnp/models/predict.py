from os import path

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import dataset


def predict(name, model, features=[]):
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

    print 'Train', name
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)

    result_file = path.join(dataset.DATA_PATH, dataset.SUBMIT, name + '.csv')
    print 'Write', result_file
    result = pd.DataFrame({dataset.ROW_ID: y_id, dataset.PREDICTION: y_pred[:, 1]})
    result.to_csv(result_file, index=False)


if __name__ == '__main__':
    m = RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)
    n = 'RandomForest-500-entropy'
    predict(n, m)


