import os
from sklearn.cross_validation import train_test_split
import pandas as pd
import dataset

pd.options.mode.chained_assignment = None  # Ignore copy warning

print 'Read', dataset.ENCODED_TRAIN_FILE
data_set = pd.read_csv(dataset.ENCODED_TRAIN_FILE)

X = data_set.drop(dataset.TARGET_CLASS, axis=1)
y = data_set[dataset.TARGET_CLASS]

print 'Shape', X.shape, y.shape
print 'Positive Instances', float(len(y[y == 1]))/len(y)
print 'Negative Instances', float(len(y[y == 0]))/len(y)

for i in xrange(0, 10):
    print 'Subset', i
    target = os.path.join(dataset.DATA_PATH, 'subset' + str(i))
    if not os.path.exists(target):
        os.makedirs(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_train.loc[:, dataset.TARGET_CLASS] = y_train
    X_test.loc[:, dataset.TARGET_CLASS] = y_test

    print 'Split Positive', float(len(y_train[y_train == 1]))/len(y_train), float(len(y_test[y_test == 1]))/len(y_test)
    print 'Split Negative', float(len(y_train[y_train == 0]))/len(y_train), float(len(y_test[y_test == 0]))/len(y_test)

    X_train.to_csv(os.path.join(target, 'train.csv'), index=False)
    X_test.to_csv(os.path.join(target, 'test.csv'), index=False)

