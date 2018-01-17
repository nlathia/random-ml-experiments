from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
import multiprocessing
import random
import os

from util import (
    get_categories,
    load,
    TRAINING_CATEGORIES
)
from preprocess import PreProcessor

manager = multiprocessing.Manager()
results = manager.dict()

params = []
for sw in [None, 'english']:
    for tfidf in [[False, False], [True, False], [True, True]]:
        params.append([sw, tfidf[0], tfidf[1], False])


def load_train_test(source, category):
    X, y = load(source, category)
    print 'Train/test split'
    return train_test_split(X, y, test_size=0.2, random_state=0)


def worker(X_train, y_train, X_test, y_test, pm, rs):
    p = PreProcessor(stop_words=pm[0], tf=pm[1], idf=pm[2], scale=pm[3])
    print 'Test', p.get_name()

    m = SGDClassifier()
    m.fit(p.fit_training(X_train), y_train)
    y_pred = m.predict(p.fit_test(X_test))

    f1 = metrics.f1_score(y_test, y_pred)
    rs[float(f1)] = p.get_name()


def results_by_category(limit=None):

    for category in get_categories()[:limit]:
        source = os.path.join(TRAINING_CATEGORIES, category)
        if os.path.exists(source):
            results.clear()
            jobs = []
            X_train, X_test, y_train, y_test = load_train_test(source, category)

            for param in params:
                p = multiprocessing.Process(target=worker, args=(X_train, y_train,
                                                                 X_test, y_test,
                                                                 param, results))
                jobs.append(p)
                p.start()
                while len(jobs) >= 4:
                    jobs[0].join()
                    jobs = [j for j in jobs if j.is_alive()]

            while len(jobs) != 0:
                jobs[0].join()
                jobs = [j for j in jobs if j.is_alive()]

            print results
            best_score = max(results.keys())
            yield [category, best_score] + results[best_score]
