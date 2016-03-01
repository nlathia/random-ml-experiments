import pandas as pd
from sklearn.metrics import log_loss
import dataset

test = pd.read_csv(dataset.SUBSET_TEST_FILE)
y_test = test[dataset.TARGET_CLASS]

logistic = pd.read_csv('data/LogisticRe.csv')
logistic_loss = log_loss(y_test, logistic[dataset.PREDICTION].values)

rf = pd.read_csv('data/RandomFore.csv')
rf_loss = log_loss(y_test, rf[dataset.PREDICTION].values)

preds = pd.DataFrame()
preds[dataset.ROW_ID] = test[dataset.ROW_ID]
preds['Logistic'] = logistic[dataset.PREDICTION]
preds['RF'] = rf[dataset.PREDICTION]
preds['Average'] = preds[['Logistic', 'RF']].mean(1)

print preds.head()

print 'Logistic', '\t', logistic_loss
print 'Random Forest', '\t', rf_loss
print 'Average', '\t', log_loss(y_test, preds['Average'].values)

