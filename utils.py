import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, log_loss, accuracy_score)

RNG = 2016

# Helper function for evaluating classfier in cross-validataion
def evaluate_clf(clf, X, y, splits):
    '''
    X: np.array(shape=(n_docs, n_features))
    y: np.array(shape=(n_docs,))
    splits: should be a pandas.Series
    '''
    scores = pd.DataFrame(columns=['f1','accuracy', 'logloss'])
    for i, split in enumerate(splits.unique()):
        train_idx = np.where(splits != split)[0]
        valid_idx = np.where(splits == split)[0]
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[valid_idx])
        probas = clf.predict_proba(X[valid_idx])

        f1 = f1_score(y[valid_idx], preds, average='weighted')
        acc = accuracy_score(y[valid_idx], preds) 
        ll = log_loss(y[valid_idx], probas)

        scores.loc[i] = [f1, acc, ll]
    return scores
