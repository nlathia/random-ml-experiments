from sklearn.linear_model import SGDClassifier
import numpy as np
import csv
import os

from util import (
    load,
    test_entries,
    TRAINING_CATEGORIES
)

from preprocess import PreProcessor

ROOT = '/Users/neal/Desktop/Challenge/'
PROCESS = os.path.join(ROOT, 'best-preprocessor-per-category.csv')
SAMPLE = os.path.join(ROOT, 'sampleSubmission.csv')
RESULT = os.path.join(ROOT, 'TopicPreprocessedPredictions.csv')

with open(SAMPLE, 'r') as lines:
    rows = csv.reader(lines)
    categories = rows.next()[1:]

with open(PROCESS, 'r') as lines:
    rows = csv.reader(lines)
    params = {}
    for row in rows:
        category = row[0]
        stop_words = None if row[2] == '' else row[2]
        tf = bool(row[3])
        idf = bool(row[4])
        scale = bool(row[5])
        params[category] = [stop_words, tf, idf, scale]


with open(RESULT, 'w') as out:
    rows = csv.writer(out)
    for category in categories:
        source = os.path.join(TRAINING_CATEGORIES, category)
        if os.path.exists(source):

            X_train, y_train = load(source, category)
            pms = params[category]
            p = PreProcessor(stop_words=pms[0],
                             tf=pms[1],
                             idf=pms[2],
                             scale=pms[3])

            X_train = p.fit_training(X_train)

            print '.fit()'
            m = SGDClassifier()
            m.fit(X_train, y_train)

            print '.predict()'
            for test_id, text in test_entries():
                text = p.fit_test(np.array([text]))
                result = [category, test_id, m.predict(text)[0]]
                rows.writerow(result)
