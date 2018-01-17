from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import gensim

from sklearn import metrics
import os

from util import (
    get_categories,
    load_words,
    TRAINING_CATEGORIES
)

ms = [
    SGDClassifier(),
    ExtraTreesClassifier(n_estimators=100)
]

# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/


def transform(word2vec, X):
    dim = len(word2vec.itervalues().next())
    return np.array([
        np.mean([word2vec[w] for w in words if w in word2vec] or [np.zeros(dim)], axis=0)
        for words in X
    ])


def load_train_test(source, category):
    X, y = load_words(source, category)

    print 'Split train/test'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print 'Train word2vec'
    model = gensim.models.Word2Vec(X, size=100)
    w2v = dict(zip(model.index2word, model.syn0))

    print 'Transform'
    X_train = transform(w2v, X_train)
    X_test = transform(w2v, X_test)
    return X_train, X_test, y_train, y_test


def results_by_category(limit=None):

    for category in get_categories()[:limit]:
        source = os.path.join(TRAINING_CATEGORIES, category)
        if os.path.exists(source):

            X_train, X_test, y_train, y_test = load_train_test(source, category)
            f1s = [category]
            for m in ms:
                print 'Fit', str(m)[:20]
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                f1s.append(metrics.f1_score(y_test, y_pred))

            yield f1s
