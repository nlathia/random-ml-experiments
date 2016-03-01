from sys import argv
from os import path
import pandas as pd
from sklearn.cross_validation import train_test_split

TARGET_CLASS = 'QuoteConversion_Flag'

data_set = pd.read_csv(argv[1])

X = data_set.drop(TARGET_CLASS, axis=1)
y = data_set[TARGET_CLASS]

print 'Shape', X.shape, y.shape

print 'Full Positive', float(len(y[y == 1]))/len(y)
print 'Full Negative', float(len(y[y == 0]))/len(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_train.loc[:, TARGET_CLASS] = y_train
X_test.loc[:, TARGET_CLASS] = y_test

print 'Split Positive', float(len(y_train[y_train == 1]))/len(y_train), float(len(y_test[y_test == 1]))/len(y_test)
print 'Split Negative', float(len(y_train[y_train == 0]))/len(y_train), float(len(y_test[y_test == 0]))/len(y_test)

source = path.split(argv[1])
X_train.to_csv(path.join(source[0], 'split-train-' + source[1]), index=False)
X_test.to_csv(path.join(source[0], 'split-test-' + source[1]), index=False)

