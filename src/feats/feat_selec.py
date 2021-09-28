#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.feature_selection import SelectFpr, SelectKBest, SelectFromModel, chi2
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import linear_model, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import numpy as np

def _get_feature_importances(estimator, norm_order=1):
    """Retrieve or aggregate feature importances from estimator"""
    importances = getattr(estimator, "feature_importances_", None)
    coef_ = getattr(estimator, "coef_", None)

    if importances is None and coef_ is not None:
        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)
        else:
            importances = np.linalg.norm(coef_, axis=0,
                                         ord=norm_order)
    elif importances is None:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)
    return importances

class FeatureSelection:
    def __init__(self, threshold):
        self.threshold = threshold
        self.th_dico = []
        self.fs = []

    def fit_transform(self, X, X2=None, X3=None, y=None, algo=None, nb_job=1):
        if algo == 'maxent':
            estimator = linear_model.LogisticRegression(
                solver='lbfgs', n_jobs=nb_job)
        elif algo == 'nb':
            estimator = MultinomialNB()
        elif algo == 'svc':
            estimator = LinearSVC()
        elif algo == 'rf':
            estimator = RandomForestClassifier()
        elif algo == 'pct':
            estimator = Perceptron()
        else:
            raise ValueError("Choose an algorithme between maxent and nb")

        X_selec_np_con = None
        for i, _X in enumerate([X, X2, X3]):
            if _X is None:
                break
            fs = SelectFromModel(estimator=estimator, threshold=None)
            fs.fit_transform(_X, y)

            # Define features to select: mean, median, None and 10 averaged step between 1e-5 and the 50th important value
            importances = _get_feature_importances(fs.estimator_, fs.norm_order)
            coef_ = getattr(fs.estimator_, "coef_", None)

            if self.threshold == 1e-5 or self.threshold == None or self.threshold == '1e-5':
                threshold = 1e-5
                self.th_dico.append(1e-5)
            elif self.threshold == "median":
                threshold = np.median(importances)
                self.th_dico.append(threshold)
            elif self.threshold == "mean":
                threshold = np.mean(importances)
                self.th_dico.append(threshold)
            elif self.threshold in ['1','2','3','4','5','6','7','8','9','10']:
                if len(importances) >= 50: #if more than 50 features, keep all top 50, add 10-scale
                    top50 = importances[np.argsort(importances)[-50]]
                    threshold = 1e-5 + (top50 - 1e-5) * int(self.threshold) / 10
                    self.th_dico.append(threshold)
                else:
                    top1 = importances[np.argsort(importances)[-1]] #less than 50 feats
                    threshold = 1e-5 + (top1 - 1e-5) * int(self.threshold) / 10
                    self.th_dico.append(threshold)
            else:
                raise ValueError('Threshold value is not recognized!')

            self.fs.append(SelectFromModel(
                estimator=estimator, threshold=threshold))
            X_selec = self.fs[i].fit_transform(_X, y)
            if not isinstance(X_selec, np.ndarray):
                X_selec_np = X_selec.toarray()
            else:
                X_selec_np = X_selec
            if X_selec_np_con is None:
                X_selec_np_con = X_selec_np
            else:
                X_selec_np_con = np.concatenate(
                    (X_selec_np_con, X_selec_np), axis=1)

        return X_selec_np_con

    def transform(self, X, X2=None, X3=None):
        X_selec_np_con = None
        for i, _X in enumerate([X, X2, X3]):
            if _X is None:
                break
            X_selec = self.fs[i].transform(_X)

            if not isinstance(X_selec, np.ndarray):
                X_selec_np = X_selec.toarray()
            else:
                X_selec_np = X_selec

            if X_selec_np_con is None:
                X_selec_np_con = X_selec_np
            else:
                X_selec_np_con = np.concatenate(
                    (X_selec_np_con, X_selec_np), axis=1)

        return X_selec_np_con

    def get_support(self, indices=True):
        support = []
        for i, fs in enumerate(self.fs):
            support.append(self.fs[i].get_support(indices=indices))
        return support
    
    def get_coef(self):
        coefs = []
        for i, fs in enumerate(self.fs):
            if getattr(fs.estimator_, "coef_", None) is not None:
                coefs.append(getattr(fs.estimator_, "coef_", None)[0])
            else:
                coefs.append(getattr(fs.estimator_, "feature_importances_", None))
        return coefs

    def get_ind_threshold(self):
        return self.th_dico