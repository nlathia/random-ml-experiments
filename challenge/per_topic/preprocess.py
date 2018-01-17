from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler


class PreProcessor(object):

    def __init__(self, stop_words=None, tf=False, idf=False, scale=False):
        self.cvz = CountVectorizer(stop_words=stop_words)
        self.tft = TfidfTransformer(use_idf=idf)
        self.scaler = StandardScaler(with_mean=False, copy=False)
        self.stop_words = stop_words
        self.tf = tf
        self.idf = idf
        self.scale = scale

    def fit_training(self, X_train):
        print 'Preprocess training', self.get_name()
        X_train = self.cvz.fit_transform(X_train)
        if self.tf:
            X_train = self.tft.fit_transform(X_train)
        if self.scale:
            X_train = self.scaler.fit_transform(X_train)
        return X_train

    def fit_test(self, X_test):
        # print 'Preprocess test'
        X_test = self.cvz.transform(X_test)
        if self.tf:
            X_test = self.tft.transform(X_test)
        if self.scale:
            X_test = self.scaler.transform(X_test)
        return X_test

    def get_name(self):
        return [self.stop_words, self.tf, self.idf, self.scale]
