import pandas as pd
import numpy as np
import dataset

np.seterr(all='raise')

'''
Notes:
* ID needs to be dropped when training
* v22 has 18,211 categorical values
* v56 has 123 categorical values
* v125 has 91 categorical values
* 47 pairs of features have > 0.9 correlation
'''

print 'Read', dataset.FULL_TRAIN_FILE
train = pd.read_csv(dataset.FULL_TRAIN_FILE)
train.fillna(-1, inplace=True)

print 'Read', dataset.FULL_TEST_FILE
test = pd.read_csv(dataset.FULL_TEST_FILE)
test.fillna(-1, inplace=True)

for feature in train.columns:
    if feature not in [dataset.TARGET_CLASS, dataset.ROW_ID]:
        if train[feature].dtype == 'object':
            print feature, 'encoding..'
            train[feature], indexer = pd.factorize(train[feature])
            test[feature] = indexer.get_indexer(test[feature])
        else:
            print feature, train[feature].dtype

print 'Write', dataset.ENCODED_TRAIN_FILE
train.to_csv(dataset.ENCODED_TRAIN_FILE, index=False)

print 'Write', dataset.ENCODED_TEST_FILE
test.to_csv(dataset.ENCODED_TEST_FILE, index=False)

