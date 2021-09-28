#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import joblib
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn import linear_model, svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from utility import write2txt

try:
    code_path = '../../src/'
except:
    raise FileNotFoundError("Please verify folder path!")

sys.path.insert(1, os.path.join(code_path, 'plot/'))
import plot_confMatrix as plt_cm
sys.path.insert(1, os.path.join(code_path + 'feats/'))
import feat_selec as fs


def define_update(update, mode='search', tuned_para=-1):
    # TRY ALL PARAMS AND SEARCH FOR THE BEST ONE
    if mode == 'search':
        clf2params = {
            "nb": [{'alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100]}],
            "maxent": [{'penalty': ['l2'], 'C':[0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100]}],
            "pa": [{'loss': ['hinge'], 'C':[0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100], 'n_iter':[1, 5, 50, 100, 200, 500]},
                {'loss': ['squared_hinge'], 'C':[0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100], 'n_iter':[1, 5, 50, 100, 200, 500]}],
            "svc": [{'penalty': ['l2'], 'C':[0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000]}],
            'rf': [{'max_depth': [2], 'max_features': ['sqrt', 'log2', None]},
                    {'max_depth': [None], 'max_features': ['sqrt', 'log2', None]}
                    ],
            'pct': [{'penalty': ['l2'],
                    'alpha': [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10, 100]}]
        }

        # DEFINE TUNED PARAMETERS AND UPDATE
        if update == 'nb':
            return MultinomialNB(), clf2params["nb"]
        elif update == 'maxent':
            return linear_model.LogisticRegression(solver='lbfgs', n_jobs=8, class_weight="balanced"), clf2params["maxent"] #solver lbfgs by default, only support l2 penalty
        elif update == "svc":
            return LinearSVC(class_weight="balanced"), clf2params["svc"]
        elif update == 'rf':
            return RandomForestClassifier(class_weight="balanced"), clf2params['rf']
        elif update == 'pct':
            return Perceptron(class_weight="balanced"), clf2params['pct']
        else:
            sys.exit("Unknown algo "+update)

    # PARAMETER TUNED, APPLY THE VALUE
    elif mode == 'test' and tuned_para != -1:
        clf2params = {
            'nb': [{'alpha': [tuned_para]}],
            'maxent':[{'penalty': ['l2'], 'C': [tuned_para]}],
            'svc': [{'penalty': ['l2'], 'C': [tuned_para]}]
        }
        if update == 'nb':
            return MultinomialNB(), clf2params["nb"]
        elif update == 'maxent':
            return linear_model.LogisticRegression(solver='lbfgs', n_jobs=8, class_weight='balanced'), clf2params["maxent"] #solver lbfgs by default, only support l2 penalty
        elif update == "svc":
            return LinearSVC(class_weight='balanced'), clf2params["svc"]
        elif update == 'rf':
            return RandomForestClassifier(), clf2params['rf']
        elif update == 'pct':
            return Perceptron(), clf2params['pct']
        else:
            sys.exit("Unknown algo "+update)
    
    elif mode == 'test' and tuned_para == -1:
        raise ValueError('Tuned parameter is negative!')


def main(trial, indir, feat, algo, threshold, nb_job, write_scores, plt_cmatrix, cla, V, train_equiv, record_params):
    inpath = os.path.join(indir, feat + '/')
    outpath = os.path.join(indir, feat + '/')
    if not os.path.isdir(inpath):
        os.mkdir(inpath)        
    cla = cla
    write_scores = write_scores
    n_jobs = nb_job
    train_equiv = train_equiv
    if threshold == 1e-5 or threshold == "1e-5" or threshold == None:
        feature_selec = False
    else:
        feature_selec = True
    record_params = record_params

    scoring = {"acc": make_scorer(balanced_accuracy_score),
               "rec": make_scorer(recall_score, average='weighted'),
               "pre": make_scorer(precision_score, average='weighted'),
               "f1": make_scorer(f1_score, average='weighted')
               }
    # load vocabulary
    vocab = []
    X = []
    for i, _feat in enumerate(feat.split('+')):
        _vocab = os.path.join(inpath, "vocab-" + _feat + "-" + cla)
        _vocab = joblib.load(_vocab)
        vocab.append({v: k for (k, v) in _vocab.items()})

        if train_equiv:
            train = os.path.join(indir + _feat + '/', "train-" + cla + "-equiv.svmlight")
        else:
            train = os.path.join(indir + _feat + '/', "train-" + cla + ".svmlight")
        _X, _ = load_svmlight_file(train)
        X.append(_X)

    _, y = load_svmlight_file(train)
        
    print("FEATURE:", feat)
    # Number of random trials
    print("NUM OF TRAILS:", trial)
    print("==========\n")

    # Initialize classifier and set up possible values of parameters to optimize
    me, tuned_parameters = define_update(algo, mode='search')

    # Array to store scores
    nested_kf_acc_scores = np.zeros(trial)
    nested_kf_pre_scores = np.zeros(trial)
    nested_kf_rec_scores = np.zeros(trial)
    nested_kf_f1_scores = np.zeros(trial)
    kf_acc_scores = np.zeros(trial)
    kf_pre_scores = np.zeros(trial)
    kf_rec_scores = np.zeros(trial)
    kf_f1_scores = np.zeros(trial)    

    kf_thre = []
    nested_kf_thre = []
    count_feats = []

    if plt_cmatrix:
        cm = (trial, 2, 2)
        nested_kf_cm = np.zeros(cm)
        nested_kf_cm_no_normalized = np.zeros(cm)
        fig = plt.figure(figsize=(14,6))
    
    # params best alpha
    param_alpha = np.zeros((10, 5))
    param_best_score = np.zeros((10, 5))
    
    # Enter trial loop
    for i in range(trial):
        print("\n\n[Trial", i, "]")

        # First shuffle the data
        Xys = shuffle(*X, y, random_state=i)
        Xs = Xys[:-1]
        ys = Xys[-1]

        # Then execute feature selection according to the estimator
        if feature_selec:
            selecFM = fs.FeatureSelection(threshold=threshold)
            Xs_selec = selecFM.fit_transform(*Xs, y=ys, algo=algo, nb_job=n_jobs)
            print('Outer loop Xs shape: ', Xs_selec.shape)

            support = selecFM.get_support(indices=True)
            coef = selecFM.get_coef()
            thre = selecFM.get_ind_threshold()
            kf_thre.append(thre)
            if train_equiv:
                featpath = os.path.join(outpath, "feat_selec_selecFromM_v" + str(V) + "_equiv/")
            else:
                featpath = os.path.join(outpath, "feat_selec_selecFromM_v" + str(V) + "/")

            if not os.path.isdir(featpath):
                os.mkdir(featpath)

            # write down ousider kept features
            for ii, _s in enumerate(support):
                file_path = os.path.join(featpath, 'kept_{}_algo_{}_feat_{}_{}_thres_{}_trial_{}_kf.txt'.format(
                    cla, algo, feat, feat.split('+')[ii], threshold, i))
                with open(file_path, 'w') as f:
                    for jj, indx in enumerate(_s):
                        f.write('{}\t{}\t{}\n'.format(indx, vocab[ii][indx], coef[ii][indx]))
                if len(_s) == 0:
                    write_scores = False
                    return
        else:
            Xs_new = []
            for _Xs in Xs:
                if not isinstance(_Xs, np.ndarray):
                    Xs_new.append(_Xs.toarray())
                else:
                    Xs_new.append(_Xs)
            Xs_selec = np.concatenate(Xs_new, axis=1)
            ys = ys
            print(Xs_selec.shape)
            
            nbf = feat.count('+')
            if nbf == 0:
                kf_thre.append([1e-5])
            elif nbf == 1:
                kf_thre.append([1e-5, 1e-5])
            elif nbf == 2:
                kf_thre.append([1e-5, 1e-5, 1e-5])
            else:
                raise ValueError('Maximum 3 combination of features!')

        # 1) Kfold on all data
        kf_cv = KFold(n_splits=5)
        clf_acc = GridSearchCV(estimator=me, param_grid=tuned_parameters,
                               scoring=scoring, refit='acc', cv=kf_cv, n_jobs=n_jobs)
        clf_acc.fit(Xs_selec, ys)
        # mean cross-validated acc. score of the best_estimator
        kf_acc_scores[i] = clf_acc.best_score_
        # best_para_outer = list(clf_acc.best_params_.items())[0][1]

        clf_pre = GridSearchCV(estimator=me, param_grid=tuned_parameters,
                               scoring=scoring, refit='pre', cv=kf_cv, n_jobs=n_jobs) #TODO can't set precision_score to 'weighted'
        clf_pre.fit(Xs_selec, ys)
        # mean cross-validated pre score of the best_estimator
        kf_pre_scores[i] = clf_pre.best_score_

        clf_rec = GridSearchCV(estimator=me, param_grid=tuned_parameters,
                               scoring=scoring, refit='rec', cv=kf_cv, n_jobs=n_jobs)
        clf_rec.fit(Xs_selec, ys)
        # mean cross-validated recall score of the best_estimator
        kf_rec_scores[i] = clf_rec.best_score_

        clf_f1 = GridSearchCV(estimator=me, param_grid=tuned_parameters,
                              scoring=scoring, refit='f1', cv=kf_cv, n_jobs=n_jobs)
        clf_f1.fit(Xs_selec, ys)
        # mean cross-validated f1 score of the best_estimator
        kf_f1_scores[i] = clf_f1.best_score_

        # Nested with KF as inner and outer
        outer_cv = KFold(n_splits=5)
        inner_cv_kf = KFold(n_splits=5)

        # 2) Kfold outer, Kfold inner
        kf_true, kf_pred = [], []
        nested_kf_thre_nes = []
        lst_outer = outer_cv.split(ys)
        nested_tdplen, nested_tdplen_true = [], []

        for j, (train_index, test_index) in enumerate(lst_outer):
            X_train, X_test = [_X[train_index] for _X in Xs], [_X[test_index] for _X in Xs]
            y_train, y_test = ys[train_index], ys[test_index]
            # keep track of test tdp length, list len = 2064
            tdp_len = [len(X_test[0][i].data) for i in range(np.size(X_test[0], 0))]
            nested_tdplen.extend(tdp_len)

            if feature_selec:
                selecFM = fs.FeatureSelection(threshold=threshold)
                Xs_selec = selecFM.fit_transform(*X_train, y=y_train, algo=algo, nb_job=n_jobs)
                support = selecFM.get_support(indices=True)
                coef = selecFM.get_coef()

                thre = selecFM.get_ind_threshold()
                nested_kf_thre_nes.append(thre)
                Xs_selec_test = selecFM.transform(*X_test)

                # write down the selected features
                for ii, _s in enumerate(support):
                    count_feats.append(len(_s))
                    file_path = os.path.join(featpath, 'kept_{}_algo_{}_feat_{}_{}_thres_{}_trial_{}_nestkf_{}.txt'.format(
                        cla, algo, feat, feat.split('+')[ii], threshold, i, j))
                    with open(file_path, 'w') as f:
                        for jj, indx in enumerate(_s):
                            f.write('{}\t{}\t{}\n'.format(indx, vocab[ii][indx], coef[ii][indx]))
                    if len(_s) == 0:
                        write_scores = False
                        return
            else:
                Xtrain_new = []
                for _Xs in X_train:
                    if not isinstance(_Xs, np.ndarray):
                        Xtrain_new.append(_Xs.toarray())
                    else:
                        Xtrain_new.append(_Xs)
                Xs_selec = np.concatenate(Xtrain_new, axis=1)

                Xtest_new = []
                for _Xs in X_test:
                    if not isinstance(_Xs, np.ndarray):
                        Xtest_new.append(_Xs.toarray())
                    else:
                        Xtest_new.append(_Xs)
                Xs_selec_test = np.concatenate(Xtest_new, axis=1)
                # print("Inner Xs shape (no selec)", Xs_selec.shape)

                count_feats.append(len(Xs_selec[1]))
                nbf = feat.count('+')
                if nbf == 0:
                    nested_kf_thre_nes.append([1e-5])
                elif nbf == 1:
                    nested_kf_thre_nes.append([1e-5, 1e-5])
                elif nbf == 2:
                    nested_kf_thre_nes.append([1e-5, 1e-5, 1e-5])
                else:
                    raise ValueError('Maximum 3 combination of features!')
            
            kf_true.extend(y_test)

            # -- KF as inner, optimised f1
            clf = GridSearchCV(estimator=me, param_grid=tuned_parameters, \
                                scoring=scoring, refit='f1', \
                                cv=inner_cv_kf, n_jobs=n_jobs) #inner loop for tuning the hyper-params
            clf.fit(Xs_selec, y_train)
            # get the best param and best model
            param_alpha[i][j] = list(clf.best_params_.items())[0][1]
            param_best_score[i][j] = clf.best_score_
            # eval on the fold left out
            preds = clf.predict(Xs_selec_test)  
            kf_pred.extend(preds)
        nested_kf_thre.append(nested_kf_thre_nes)

        # Calculate accuracy, precision, recall and f1
        nested_kf_acc_scores[i] = balanced_accuracy_score(kf_true, kf_pred)
        nested_kf_pre_scores[i] = precision_score(kf_true, kf_pred, average='weighted') #v31.2 add average weighted
        nested_kf_rec_scores[i] = recall_score(kf_true, kf_pred, average='weighted')
        nested_kf_f1_scores[i] = f1_score(kf_true, kf_pred, average='weighted')

        # confusion matrix and plot
        if plt_cmatrix:
            cmi = confusion_matrix(kf_true, kf_pred, normalize='true', labels=[-1, 1]) #labels=[-2, -1, 1, 2], -1=tem, 1=scz
            cmi_no_normalize = confusion_matrix(kf_true, kf_pred, normalize=None)
            nested_kf_cm[i] = cmi
            nested_kf_cm_no_normalized[i] = cmi_no_normalize
            print(cmi)
            np.set_printoptions(precision=1) #write 2 decimals
            ax = fig.add_subplot(2, 5, i+1)
            disp = plt_cm.ConfusionMatrixDisplay(confusion_matrix=cmi, display_labels=None)
            disp.plot(include_values=True,
                    cmap=plt.cm.Blues, 
                    ax=ax)

    if plt_cmatrix:
        cmpath = os.path.join(outpath, "cmatrix_v" + str(V) + "/")
        if not os.path.isdir(cmpath):
            os.mkdir(cmpath)
        with open(os.path.join(cmpath, f'cmat_{feat}_{algo}_{threshold}.txt'), 'w') as outs:
            outs.write(np.array_str(nested_kf_cm_no_normalized))
    print('Done')
        
    if plt_cmatrix:
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.9, 0.14, 0.04, 0.735]) #(left, bottom, width, height)
        plt.colorbar(mappable=disp.im_, cax=cax, ax=ax)
        # add common xlabel, ylabel and title in the big picture
        fig.text(0.5, 0.02, 'Étiquette prédicte', ha='center', fontsize=14)
        fig.text(0.02, 0.5, 'Étiquette vraie', va='center', rotation='vertical', fontsize=14)
        plt.savefig(os.path.join(cmpath, f'cmat_{feat}_{algo}_{threshold}.png'))

    if record_params:
        parampath = os.path.join(outpath, "param_selecFromM_v" + str(V) + "/")
        if not os.path.isdir(parampath):
            os.mkdir(parampath)
        param_f = os.path.join(parampath, 'log_{}_{}_{}.txt'.format(feat, algo, thres))
        with open(param_f, 'a') as outstream:
            line1 = '\n\n' + '='*22 + '\n{}_{}_th{}\n'.format(feat, algo, threshold) + '='*22 + '\n' + 'Best_parameter:\n'
            outstream.write(line1)
            outstream.write(str(param_alpha))
            line2 = '\n\nBest_train_score:\n'
            outstream.write(line2)
            outstream.write(str(param_best_score))

    if write_scores:
        values, accs, pres, fns, f1s = write2txt(outpath, V, trial, cla, algo, \
                feat, threshold, train_equiv, kf_thre, \
                kf_acc_scores, kf_pre_scores, kf_rec_scores, kf_f1_scores, \
                nested_kf_thre, nested_kf_acc_scores, nested_kf_pre_scores, \
                nested_kf_rec_scores, nested_kf_f1_scores)
        return [np.array(values).mean()], [np.array(count_feats).mean()], accs, pres, fns, f1s        
       

if __name__ == "__main__":
    trial = 10
    indir = '[local_folder]/features/block_64/'
    nb_job = 16
    write_scores = True
    plt_cmatrix = False
    TEST_V = 1
    record_params = True

    parser = argparse.ArgumentParser(description='Use different features to classify schizo-temoin / psy-psy monologues.')
    parser.add_argument('--cla', dest='cla', action='store', default='st', help='choose st classification or pp classification')
    parser.add_argument('--feat', dest='feat', action='store', default="bow", help='type of features (bow, ngram..)')
    parser.add_argument('--algo', dest='algo', action='store', help='choose algorithmes from "svc, maxent, nb, rf, pct"')
    parser.add_argument('--teq', dest='teq', default=False, action='store_true', help='train with equivalent classes')
    parser.add_argument('--thres', dest='thres', action='store', default='mean', help='threshold choose from None, 1 to 10, mean or median')
    
    args = parser.parse_args()
    cla = args.cla
    feat = args.feat
    algo = args.algo
    train_eq = args.teq
    thres = args.thres
    values = []
    nfs = []
    accs = []
    pres = []
    fns = []
    f1s = []

    for threshold in [thres]:
        if write_scores:
            try:
                v, nf, a, p, fn, f1 = main(trial, indir, feat, algo, threshold, nb_job, write_scores, plt_cmatrix, cla=cla, V=TEST_V, train_equiv=train_eq, record_params=record_params)
                values.append(v)
                nfs.append(nf)
                accs.append(a)
                pres.append(p)
                fns.append(fn)
                f1s.append(f1)
            except:
                raise
        else:
            main(trial, indir, feat, algo, threshold, nb_job, write_scores, plt_cmatrix, cla=cla, V=TEST_V, train_equiv=train_eq)
    
