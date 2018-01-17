from sklearn.model_selection import train_test_split
from shallowlearn.models import GensimFastText

from sklearn import metrics
import os

from util import (
    get_categories,
    load_words,
    TRAINING_CATEGORIES
)

ms = [
    GensimFastText(size=100, min_count=0, loss='hs', iter=3, seed=66)
]


def load_train_test(source, category):
    X, y = load_words(source, category)
    return train_test_split(X, y, test_size=0.2, random_state=0)


def results_by_category(limit=None):

    for category in get_categories()[:limit]:
        source = os.path.join(TRAINING_CATEGORIES, category)
        if os.path.exists(source):

            X_train, X_test, y_train, y_test = load_train_test(source, category)
            f1s = [category]
            for m in ms:
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                f1s.append(metrics.f1_score(y_test, y_pred))

            yield f1s
