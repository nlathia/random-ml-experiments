from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn import metrics

import xgboost as xgb

xgboost_params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "logloss",
    "eta": 0.01,  # 0.06, #0.01,  # step size shrinkage used in update to prevents overfitting
    # #"min_child_weight": 240,
    "subsample": 0.75,
    "colsample_bytree": 0.68,
    "max_depth": 7
}
boost_round = 500

def predict_xgb(X_train, X_test, y_train, y_test):
    try:
        xgtrain = xgb.DMatrix(X_train, y_train)
        clf = xgb.train(xgboost_params, xgtrain, num_boost_round=boost_round, verbose_eval=True, maximize=False)
        xgtest = xgb.DMatrix(X_test)
        y_proba = clf.predict(xgtest, ntree_limit=clf.best_iteration)
        y_pred = []
        for y in y_proba:
            y_class = 1 if y > 0.5 else 0
            y_pred.append(y_class)
        return metrics.f1_score(y_test, y_pred)
    except:
        return 'Exception'




