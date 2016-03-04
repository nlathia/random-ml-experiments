from csv import writer

import pandas as pd
import dataset

print 'Read', dataset.FULL_TRAIN_FILE
train = pd.read_csv(dataset.FULL_TRAIN_FILE)
train.fillna(-1, inplace=True)
train.drop_duplicates(inplace=True)

print 'Read', dataset.FULL_TEST_FILE
test = pd.read_csv(dataset.FULL_TEST_FILE)
test.fillna(-1, inplace=True)

with open(dataset.FEATURE_TYPES, 'w') as out:
    rows = writer(out)
    for feature in train.columns:
        print feature, train[feature].dtype
        rows.writerow([feature, train[feature].dtype])
        if feature not in [dataset.TARGET_CLASS, dataset.ROW_ID]:
            if train[feature].dtype == 'object':
                train[feature], indexer = pd.factorize(train[feature])
                test[feature] = indexer.get_indexer(test[feature])

print 'Write', dataset.ENCODED_TRAIN_FILE
train.to_csv(dataset.ENCODED_TRAIN_FILE + '2', index=False)

print 'Write', dataset.ENCODED_TEST_FILE
test.to_csv(dataset.ENCODED_TEST_FILE + '2', index=False)

