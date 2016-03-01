from sys import argv
from os import path
import pandas as pd
import numpy as np

np.seterr(all='raise')

TARGET_CLASS = 'QuoteConversion_Flag'
QUOTE_NUM = 'QuoteNumber'
QUOTE_DATE = 'Original_Quote_Date'
YEAR = 'Year'
MONTH = 'Month'
DAY = 'Day'


def load_data(file_name):
    print 'Load', file_name
    d = pd.read_csv(file_name)
    d[YEAR] = d[QUOTE_DATE].apply(lambda x: int(str(x)[:4]))
    d[MONTH] = d[QUOTE_DATE].apply(lambda x: int(str(x)[5:7]))
    d[DAY] = d[QUOTE_DATE].apply(lambda x: int(str(x)[8:10]))
    d.drop(['SalesField8', QUOTE_DATE], axis=1, inplace=True)
    d.fillna(-1, inplace=True)
    return d

datasets = []
for arg in argv[1:]:
    datasets.append(load_data(arg))

i = 0
old_fields = []
for field in datasets[0].columns:
    if field not in [TARGET_CLASS, QUOTE_NUM]:
        values = set()
        for data_set in datasets:
            values |= set(pd.unique(data_set[field].values))
        print i, field, len(values), values
        for value in (x for x in values if x not in [' ', -1]):
            new_field = field + 'v' + str(value).replace(',', '').replace('.', '')
            print '\tAdd', new_field
            for data_set in datasets:
                data_set[new_field] = data_set[field].apply(lambda x: 1 if x == value else 0)
        old_fields.append(field)
    i += 1

for i in xrange(1, len(argv)):
    source = path.split(argv[i])
    print 'Drop', source
    datasets[i-1].drop(old_fields, axis=1, inplace=True)
    print 'Write', source
    datasets[i-1].to_csv(path.join(source[0], 'binary-' + source[1]), index=False)
