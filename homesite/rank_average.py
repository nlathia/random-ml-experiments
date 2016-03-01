import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

preds = pd.DataFrame()
subA = pd.read_csv("subA.csv")
subB = pd.read_csv("subB.csv")

preds['Id'] = subA.Id
preds['PredsSubA'] = subA.Prediction
preds['PredsSubB'] = subB.Prediction
preds['RanksSubA'] = rankdata(subA.Prediction)
preds['RanksSubB'] = rankdata(subB.Prediction)
preds['RankAverage'] = preds[['RanksSubA', 'RanksSubB']].mean(1)
preds['FinalBlend'] = MinMaxScaler().fit_transform(preds['RankAverage'].reshape(-1, 1))

print preds.head()

