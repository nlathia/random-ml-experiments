import pandas as pd
from os import path
from sys import argv
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

TARGET_CLASS = 'QuoteConversion_Flag'
QUOTE_NUM = 'QuoteNumber'

print 'Load training.', argv[1]
train = pd.read_csv(argv[1])
train.drop(QUOTE_NUM, axis=1, inplace=True)
X_train = train.drop(TARGET_CLASS, axis=1)
y_train = train[TARGET_CLASS]

print 'Training.'
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)
name = str(model)[:str(model).index('(')]

print 'Load testing.', argv[2]
test = pd.read_csv(argv[2])
if TARGET_CLASS in test.columns:
    print 'Evaluating.'
    X_test = test.drop([TARGET_CLASS, QUOTE_NUM], axis=1)
    y_test = test[TARGET_CLASS]
    y_score = model.predict_proba(X_test)
    print name, roc_auc_score(y_test, y_score[:, 1])
else:
    print 'Creating submission.'
    submission = pd.DataFrame()
    X_test = test.drop(QUOTE_NUM, axis=1)
    submission[QUOTE_NUM] = test[QUOTE_NUM]
    submission[TARGET_CLASS] = model.predict_proba(X_test)[:, 1]
    target = path.join('submissions', name + '-' + str(datetime.now()) + '.csv')
    submission.to_csv(target, index=False)

print 'Done.'

# Without nan-binary features   0.949892513349
# Drop 0 std features           0.948790894142
# With nan-binary features      0.949697746955
