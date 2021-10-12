# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy


https://stats.stackexchange.com/questions/222558/classification-evaluation-metrics-for-highly-imbalanced-data


Besides the AUC and Kohonen's kappa already discussed in the other answers, I'd also like to add a few metrics I've found useful for imbalanced data. They are both related to precision and recall. Because by averaging these you get a metric weighing TPs and both types of errors (FP and FN):

F1 score, which is the harmonic mean of precision and recall.
G-measure, which is the geometric mean of precision and recall. Compared to F1, I've found it a bit better for imbalanced data.
Jaccard index, which you can think of as the TP/(TP+FP+FN). This is actually the metric that has worked for me the best.
Note: For imbalanced datasets, it is best to have your metrics be macro-averaged.

esides the AUC and Kohonen's kappa already discussed in the other answers, I'd also like to add a few metrics I've found useful for imbalanced data. They are both related to precision and recall. Because by averaging these you get a metric weighing TPs and both types of errors (FP and FN):

F1 score, which is the harmonic mean of precision and recall.
G-measure, which is the geometric mean of precision and recall. Compared to F1, I've found it a bit better for imbalanced data.
Jaccard index, which you can think of as the TP/(TP+FP+FN). This is actually the metric that has worked for me the best.
Note: For imbalanced datasets, it is best to have your metrics be macro-averaged.



Final intuition to metric selection
Use precision and recall to focus on small positive class — When the positive class is smaller and the ability to detect correctly positive samples is our run_train focus (correct detection of negatives examples is less important to the problem) we should use precision and recall.
Use ROC when both classes detection is equally important — When we want to give equal weight to both classes prediction ability we should look at the ROC curve.
Use ROC when the positives are the majority or switch the labels and use precision and recall — When the positive class is larger we should probably use the ROC metrics because the precision and recall would reflect mostly the ability of prediction of the positive class and not the negative class which will naturally be harder to detect due to the smaller number of samples. If the negative class (the minority in this case) is more important, we can switch the labels and use precision and recall (As we saw in the examples above — switching the labels can change everything).
Towards Data Science
Sharing concepts, ideas, and codes.
Following
1.2K

Machine Learning
Data Science






"""
import copy
import os
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pandas as pd
import scipy as sci
from dateutil.parser import parse

import sklearn as sk
from matplotlib import pyplot as plt
from sklearn import covariance, linear_model, model_selection, preprocessing
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import (NMF, PCA, LatentDirichletAllocation,
                                   TruncatedSVD, pca)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer,
                             mean_absolute_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def import_(abs_module_path, class_name=None):
    try:
        module_object = import_module(abs_module_path)
        print("imported", module_object)
        if class_name is None:
            return module_object
        target_class = getattr(module_object, class_name)
        return target_class
    except Exception as e:
        print(abs_module_path, class_name, e)


# from attrdict import AttrDict as dict2
# from kmodes.kmodes import KModes
# from tabulate import tabulate


########### Dynamic Import   #######################################################################
# EvolutionaryAlgorithmSearchCV = import_("evolutionary_search", "EvolutionaryAlgorithmSearchCV")
esearch = import_("evolutionary_search")
lgb = import_("lightgbm")
kmodes = import_("kmodes")
catboost = import_("catboost")
tpot = import_("tpot")

####################################################################################################
DIRCWD = os.getcwd()
print("os.getcwd", os.getcwd())


class dict2(object):
    def __init__(self, d):
        self.__dict__ = d




####################################################################################################
def pd_dim_reduction(
    df,
    colname,
    colprefix="colsvd",
    method="svd",
    dimpca=2,
    model_pretrain=None,
    return_val="dataframe,param",
):
    """
       Dimension reduction technics
       dftext_svd, svd = pd_dim_reduction(dfcat_test, None,colprefix="colsvd",
                         method="svd", dimpca=2, return_val="dataframe,param")
    :param df:
    :param colname:
    :param colprefix:
    :param method:
    :param dimpca:
    :param return_val:
    :return:
    """
    colname = colname if colname is not None else list(df.columns)
    if method == "svd":
        if model_pretrain is None:
            svd = TruncatedSVD(n_components=dimpca, algorithm="randomized")
            svd = svd.fit(df[colname].values)
        else:
            svd = copy.deepcopy(model_pretrain)

        X2 = svd.transform(df[colname].values)
        # print(X2)
        dfnew = pd.DataFrame(X2)
        dfnew.columns = [colprefix + "_" + str(i) for i in dfnew.columns]

        if return_val == "dataframe,param":
            return dfnew, svd
        else:
            return dfnew



def model_lightgbm_kfold(
    df, colname=None, num_folds=2, stratified=False, colexclude=None, debug=False
):
    # LightGBM GBDT with KFold or Stratified KFold
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df.shape[0])
    feature_importance_df = pd.DataFrame()
    # colname = [f for f in df.columns if f not in colexclude]

    regs = []
    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[colname], df["is_match"])):
        train_x, train_y = df[colname].iloc[train_idx], df["is_match"].iloc[train_idx]
        valid_x, valid_y = df[colname].iloc[valid_idx], df["is_match"].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x, label=valid_y, free_raw_data=False)

        # params optimized by optuna
        params = {
            "max_depth": -1,
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 2 ** 12 - 1,
            "colsample_bytree": 0.28,
            "objective": "binary",
            "n_jobs": -1,
        }

        reg = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            valid_names=["train", "test"],
            num_boost_round=10000,
            early_stopping_rounds=200,
            verbose_eval=100,
        )
        regs.append(reg)

    return regs


def model_catboost_classifier(
    Xtrain,
    Ytrain,
    Xcolname=None,
    pars={
        "learning_rate": 0.1,
        "iterations": 1000,
        "random_seed": 0,
        "loss_function": "MultiClass",
    },
    isprint=0,
):
    """
  from catboost import Pool, CatBoostClassifier

TRAIN_FILE = '../data/cloudness_small/train_small'
TEST_FILE = '../data/cloudness_small/test_small'
CD_FILE = '../data/cloudness_small/train.cd'
# Load data from files to Pool
train_pool = Pool(TRAIN_FILE, column_description=CD_FILE)
test_pool = Pool(TEST_FILE, column_description=CD_FILE)
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='MultiClass')
# Fit model
model.fit(train_pool)
# Get predicted classes
preds_class = model.predict(test_pool)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(test_pool)
# Get predicted RawFormulaVal
  preds_raw = model.predict(test_pool, prediction_type='RawFormulaVal')


  https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/

  """
    import catboost

    pa = dict2(pars)

    if Xcolname is None:
        Xcolname = [str(i) for i in range(0, Xtrain.shape[1])]
    train_df = pd.DataFrame(Xtrain, Xcolname)
    cat_features_ids = Xcolname

    clf = catboost.CatBoostClassifier(
        learning_rate=pa.learning_rate,
        iterations=pa.iterations,
        random_seed=pa.random_seed,
        loss_function=pa.loss_function,
    )
    clf.fit(Xtrain, Ytrain, cat_features=cat_features_ids)

    Y_pred = clf.predict(Xtrain)

    cm = sk.metrics.confusion_matrix(Ytrain, Y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)
    return clf, cm, cm_norm



def sk_score_get(name="r2"):
    from sklearn.metrics import make_scorer, r2_score, roc_auc_score, mean_squared_error

    if name == "r2":
        return sk.metrics.make_scorer(r2_score, sample_weight=None)

    if name == "auc":
        return sk.metrics.make_scorer(r2_score, sample_weight=None)


def sk_params_search_best(
    clf,
    X,
    y,
    param_grid={"alpha": np.linspace(0, 1, 5)},
    method="gridsearch",
    param_search={"scorename": "r2", "cv": 5, "population_size": 5, "generations_number": 3},
):
    """
   Genetic: population_size=5, ngene_mutation_prob=0.10,,gene_crossover_prob=0.5, tournament_size=3,  generations_number=3

    :param X:
    :param y:
    :param clf:
    :param param_grid:
    :param method:
    :param param_search:
    :return:
  """
    p = param_search
    myscore = sk_score_get(p["scorename"])

    if method == "gridsearch":
        from sklearn.model_selection import GridSearchCV

        grid = GridSearchCV(clf, param_grid, cv=p["cv"], scoring=myscore)
        grid.fit(X, y)
        return grid.best_score_, grid.best_params_

    if method == "genetic":
        from evolutionary_search import EvolutionaryAlgorithmSearchCV
        from sklearn.model_selection import StratifiedKFold

        # paramgrid = {"alpha":  np.linspace(0,1, 20) , "l1_ratio": np.linspace(0,1, 20) }
        cv = EvolutionaryAlgorithmSearchCV(
            estimator=clf,
            params=param_grid,
            scoring=myscore,
            cv=StratifiedKFold(y),
            verbose=True,
            population_size=p["population_size"],
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=p["generations_number"],
        )

        cv.fit(X, y)
        return cv.best_score_, cv.best_params_


def sk_error(ypred, ytrue, method="r2", sample_weight=None, multioutput=None):
    from sklearn.metrics import r2_score

    if method == "rmse":
        aux = np.sqrt(np.sum((ypred - ytrue) ** 2)) / len(ytrue)
        print("Error:", aux, "Error/Stdev:", aux / np.std(ytrue))
        return aux / np.std(ytrue)

    elif method == "r2":
        r2 = r2_score(ytrue, ypred, sample_weight=sample_weight, multioutput=multioutput)
        r = np.sign(r2) * np.sqrt(np.abs(r2))
        return -1 if r <= -1 else r


def sk_cluster(
    Xmat,
    method="kmode",
    args=(),
    kwds={"metric": "euclidean", "min_cluster_size": 150, "min_samples": 3},
    isprint=1,
    preprocess={"norm": False},
):
    """
   'hdbscan',(), kwds={'metric':'euclidean', 'min_cluster_size':150, 'min_samples':3 }
   'kmodes',(), kwds={ n_clusters=2, n_init=5, init='Huang', verbose=1 }
   'kmeans',    kwds={ n_clusters= nbcluster }

   Xmat[ Xcluster== 5 ]
   # HDBSCAN Clustering
   Xcluster_hdbscan= da.sk_cluster_algo_custom(Xtrain_d, hdbscan.HDBSCAN, (),
                  {'metric':'euclidean', 'min_cluster_size':150, 'min_samples':3})

   print len(np.unique(Xcluster_hdbscan))

   Xcluster_use =  Xcluster_hdbscan

# Calculate Distribution for each cluster
kde= da.plot_distribution_density(Y[Xcluster_use== 2], kernel='gaussian', N=200, bandwith=1 / 500.)
kde.sample(5)

   """
    if method == "kmode":
        # Kmode clustering data nbCategory,  NbSample, NbFeatures
        km = kmodes.kmodes.KModes(*args, **kwds)
        Xclus_class = km.fit_predict(Xmat)
        return Xclus_class, km, km.cluster_centroids_  # Class, km, centroid

    if method == "hdbscan":
        import hdbscan

        Xcluster_id = hdbscan.HDBSCAN(*args, **kwds).fit_predict(Xmat)
        print(("Nb Cluster", len(np.unique(Xcluster_id))))
        return Xcluster_id

    if method == "kmeans":
        from sklearn.cluster import KMeans

        if preprocess["norm"]:
            stdev = np.std(Xmat, axis=0)
            Xmat = (Xmat - np.mean(Xmat, axis=0)) / stdev

        sh = Xmat.shape
        Xdim = 1 if len(sh) < 2 else sh[1]  # 1Dim vector or 2dim-3dim vector
        print((len(Xmat.shape), Xdim))
        if Xdim == 1:
            Xmat = Xmat.reshape((sh[0], 1))

        kmeans = KMeans(**kwds)  # KMeans(n_clusters= nbcluster)
        kmeans.fit(Xmat)
        centroids, labels = kmeans.cluster_centers_, kmeans.labels_

        if isprint:
            import matplotlib.pyplot as plt

            colors = ["g.", "r.", "y.", "b.", "k."]
            if Xdim == 1:
                for i in range(0, sh[0], 5):
                    plt.plot(Xmat[i], colors[labels[i]], markersize=5)
                plt.show()
            elif Xdim == 2:
                for i in range(0, sh[0], 5):
                    plt.plot(Xmat[i, 0], Xmat[i, 1], colors[labels[i]], markersize=2)
                plt.show()
            else:
                print("Cannot Show higher than 2dim")

        return labels, centroids


######## Valuation model template  ##########################################################
class model_template1(sk.base.BaseEstimator):
    def __init__(self, alpha=0.5, low_y_cut=-0.09, high_y_cut=0.09, ww0=0.95):
        from sklearn.linear_model import Ridge

        self.alpha = alpha
        self.low_y_cut, self.high_y_cut, self.ww0 = 1000.0 * low_y_cut, 1000.0 * high_y_cut, ww0
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, Y=None):
        X, Y = X * 100.0, Y * 1000.0

        y_is_above_cut = Y > self.high_y_cut
        y_is_below_cut = Y < self.low_y_cut
        y_is_within_cut = ~y_is_above_cut & ~y_is_below_cut
        if len(y_is_within_cut.shape) > 1:
            y_is_within_cut = y_is_within_cut[:, 0]

        self.model.fit(X[y_is_within_cut, :], Y[y_is_within_cut])

        r2 = self.model.score(X[y_is_within_cut, :], Y[y_is_within_cut])
        print(("R2:", r2))
        print(("Inter", self.model.intercept_))
        print(("Coef", self.model.coef_))

        self.ymedian = np.median(Y)
        return self, r2, self.model.coef_

    def predict(self, X, y=None, ymedian=None):
        X = X * 100.0

        if ymedian is None:
            ymedian = self.ymedian
        Y = self.model.predict(X)
        Y = Y.clip(self.low_y_cut, self.high_y_cut)
        Y = self.ww0 * Y + (1 - self.ww0) * ymedian

        Y = Y / 1000.0
        return Y

    def score(self, X, Ytrue=None, ymedian=None):
        from sklearn.metrics import r2_score

        X = X * 100.0

        if ymedian is None:
            ymedian = self.ymedian
        Y = self.model.predict(X)
        Y = Y.clip(self.low_y_cut, self.high_y_cut)
        Y = self.ww0 * Y + (1 - self.ww0) * ymedian
        Y = Y / 1000.0
        return r2_score(Ytrue, Y)


def sk_model_ensemble_weight(model_list, acclevel, maxlevel=0.88):
    imax = min(acclevel, len(model_list))
    estlist = np.empty(imax, dtype=np.object)
    estww = []
    for i in range(0, imax):
        # if model_list[i,3]> acclevel:
        estlist[i] = model_list[i, 1]
        estww.append(model_list[i, 3])
        # print 5

    # Log Proba Weighted + Impact of recent False discovery
    estww = np.log(1 / (maxlevel - np.array(estww) / 2.0))
    # estww= estww/np.sum(estww)
    # return np.array(estlist), np.array(estww)
    return estlist, np.array(estww)


def sk_model_votingpredict(estimators, voting, ww, X_test):
    ww = ww / np.sum(ww)
    Yproba0 = np.zeros((len(X_test), 2))
    Y1 = np.zeros((len(X_test)))

    for k, clf in enumerate(estimators):
        Yproba = clf.predict_proba(X_test)
        Yproba0 = Yproba0 + ww[k] * Yproba

    for k in range(0, len(X_test)):
        if Yproba0[k, 0] > Yproba0[k, 1]:
            Y1[k] = -1
        else:
            Y1[k] = 1
    return Y1, Yproba0




############## ML metrics    ###################################
def sk_showconfusion(Y, Ypred, isprint=True):
    cm = sk.metrics.confusion_matrix(Y, Ypred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    if isprint:
        print((cm_norm[0, 0] + cm_norm[1, 1]))
        print(cm_norm)
        print(cm)
    return cm, cm_norm, cm_norm[0, 0] + cm_norm[1, 1]


def sk_showmetrics(y_test, ytest_pred, ytest_proba, target_names=["0", "1"], return_stat=0):
    #### Confusion matrix
    mtest = sk_showconfusion(y_test, ytest_pred, isprint=False)
    # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
    auc = roc_auc_score(y_test, ytest_proba)  #
    gini = 2 * auc - 1
    acc = accuracy_score(y_test, ytest_pred)
    f1macro = sk.metrics.f1_score(y_test, ytest_pred, average="macro")

    print("Test confusion matrix")
    print(mtest[0])
    print(mtest[1])
    print("auc " + str(auc))
    print("gini " + str(gini))
    print("acc " + str(acc))
    print("f1macro " + str(f1macro))
    print("Nsample " + str(len(y_test)))

    print(classification_report(y_test, ytest_pred, target_names=target_names))

    # Show roc curve
    try:
        fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(fpr, tpr, marker=".")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.show()
    except Exception as e:
        print(e)

    if return_stat:
        return {"auc": auc, "f1macro": f1macro, "acc": acc, "confusion": mtest}




def sk_metric_roc_optimal_cutoff(ytest, ytest_proba):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    ytest : Matrix with dependent or target data, where rows are observations
    ytest_proba : Matrix with predicted data, where rows are observations

    # Find prediction to the dataframe applying threshold
    data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)
    # Print confusion Matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix(data['admit'], data['pred'])
    # array([[175,  98],
    #        [ 46,  81]])
    Returns: with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(ytest, ytest_proba)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i),
                        'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]

    return roc_t['threshold']


def sk_metric_roc_auc(y_test, ytest_pred, ytest_proba):
    ####sk_showmetrics Confusion matrix
    conf_mat = sk.metrics.confusion_matrix(y_test, ytest_pred)
    print(conf_mat)
    if ytest_proba is None:
        return conf_mat
    
    # mtrain = sk_showconfusion( y_train , ytrain_pred, isprint=False)
    roc_auc = roc_auc_score(y_test, ytest_proba)  #
    fpr, tpr, thresholds = roc_curve(y_test, ytest_proba)
    freport = classification_report(y_test, ytest_pred, target_names=[0,1])

    res = { "roc_auc" : roc_auc, "tpr" : tpr, "fpr" : fpr , "confusion" : conf_mat,
            "freport" : freport }
    return res


def sk_metric_roc_auc_multiclass(n_classes=3, y_test=None, y_test_pred=None, y_predict_proba=None):
    # Compute ROC curve and ROC AUC for each class
    # n_classes = 3
    conf_mat = sk.metrics.confusion_matrix(y_test, y_test_pred)
    print(conf_mat)
    if y_predict_proba is None:
        return conf_mat

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = list(map(lambda x: 1 if x == i else 0, y_test))
        # print(y_test_i)
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        roc_auc[i] = sk.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    roc_auc["average"] = sk.metrics.auc(fpr["average"], tpr["average"])

    print("auc average", roc_auc["average"])

    try  :
      # Plot average ROC Curve
      plt.figure()
      plt.plot(fpr["average"], tpr["average"],
             label='Average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["average"]),
             color='deeppink', linestyle=':', linewidth=4)

      # Plot each individual ROC curve
      for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))


      plt.plot([0, 1], [0, 1], 'k--', lw=2)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Some extension of Receiver operating characteristic to multi-class')
      plt.legend(loc="lower right")
      plt.show()
    except BaseException :
      pass

    res = { "roc_auc" : roc_auc, "tpr" : tpr, "fpr" : fpr , "confusion" : conf_mat  }
    return res




def sk_model_eval_regression(clf, istrain=1, Xtrain=None, ytrain=None, Xval=None, yval=None):
    if istrain:
        clf.fit(Xtrain, ytrain)

    CV_score = -cross_val_score(clf, Xtrain, ytrain, scoring="neg_mean_absolute_error", cv=4)

    print("CV score: ", CV_score)
    print("CV mean: ", CV_score.mean())
    print("CV std:", CV_score.std())

    train_y_predicted_logReg = clf.predict(Xtrain)
    val_y_predicted_logReg = clf.predict(Xval)

    print("\n")
    print("Score on logReg training set:", mean_absolute_error(ytrain, train_y_predicted_logReg))
    print("Score on logReg validation set:", mean_absolute_error(yval, val_y_predicted_logReg))

    return clf, train_y_predicted_logReg, val_y_predicted_logReg


def sk_model_eval_classification(clf, istrain=1, Xtrain=None, ytrain=None, Xtest=None, ytest=None):
    if istrain:
        print("############# Train dataset  ####################################")
        clf.fit(Xtrain, ytrain)
        ytrain_proba = clf.predict_proba(Xtrain)[:, 1]
        ytrain_pred = clf.predict(Xtrain)
        sk_showmetrics(ytrain, ytrain_pred, ytrain_proba)

    print("############# Test dataset  #########################################")
    ytest_proba = clf.predict_proba(Xtest)[:, 1]
    ytest_pred = clf.predict(Xtest)
    sk_showmetrics(ytest, ytest_pred, ytest_proba)

    return clf, {"ytest_pred": ytest_pred}


def sk_metrics_eval(clf, Xtest, ytest, cv=1, metrics=["f1_macro", "accuracy", "precision_macro", "recall_macro"] ) :
  #
  entries = []
  model_name = clf.__class__.__name__
  for metric in  metrics :
    metric_val = cross_val_score(clf, Xtest, ytest, scoring= metric, cv=3)
    for i, metric_val_i in enumerate(metric_val):
       entries.append((model_name, i, metric, metric_val_i ))
  cv_df = pd.DataFrame(entries, columns=['model_class', 'fold_idx', "metric", 'metric_val'])
  return cv_df


def sk_model_eval(clf_list, Xtest, ytest, cv=1,
                  metrics=["f1_macro", "accuracy", "precision", "recall"]):
    df_list = []
    for clf in clf_list:
        df_clf_cv = sk_metrics_eval(clf, Xtest, ytest, cv=cv, metrics=metrics)
        df_list.append(df_clf_cv)
    
    return pd.concat(df_list, axis=0)



###################################################################################################
def sk_feature_impt(clf, colname, model_type="logistic"):
    """
       Feature importance with colname
    :param clf:  model or colnum with weights
    :param colname:
    :return:
    """
    if model_type == "logistic":
        dfeatures = pd.DataFrame(
            {"feature": colname, "weight": clf.coef_[0], "weight_abs": np.abs(clf.coef_[0])}
        ).sort_values("weight_abs", ascending=False)
        dfeatures["rank"] = np.arange(0, len(dfeatures))
        return dfeatures

    else:
        # RF, Xgboost, LightGBM
        if isinstance(clf, list) or isinstance(clf, (np.ndarray, np.generic)):
            importances = clf
        else:
            importances = clf.feature_importances_
        rank = np.argsort(importances)[::-1]
        d = {"col": [], "rank": [], "weight": []}
        for i in range(0, len(colname)):
            d["rank"].append(rank[i])
            d["col"].append(colname[rank[i]])
            d["weight"].append(importances[rank[i]])

        return pd.DataFrame(d)


def sk_feature_selection(clf, method="f_classif", colname=None, kbest=50, Xtrain=None, ytrain=None):
    from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression

    if method == "f_classif":
        clf_best = SelectKBest(f_classif, k=kbest).fit(Xtrain, ytrain)

    if method == "f_regression":
        clf_best = SelectKBest(f_regression, k=kbest).fit(Xtrain, ytrain)

    mask = clf_best.get_support()  # list of booleans
    new_features = []  # The list of your K best features
    for bool, feature in zip(mask, colname):
        if bool:
            new_features.append(feature)

    return new_features


def sk_feature_evaluation(clf, df, kbest=30, colname_best=None, dfy=None):
    clf2 = copy.deepcopy(clf)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        df[colname_best].values, dfy.values, random_state=42, test_size=0.5, shuffle=True
    )
    print(Xtrain.shape, ytrain.shape)

    df = {x: [] for x in ["col", "auc", "acc", "f1macro", "confusion"]}
    for i in range(1, len(colname_best)):
        print("########## ", colname_best[:i])
        if i > kbest:
            break
        clf.fit(Xtrain[:, :i], ytrain)
        ytest_proba = clf.predict_proba(Xtest[:, :i])[:, 1]
        ytest_pred = clf.predict(Xtest[:, :i])
        s = sk_showmetrics(ytest, ytest_pred, ytest_proba, return_stat=1)

        # {"auc": auc, "f1macro": f1macro, "acc": acc, "confusion": mtest}

        df["col"].append(str(colname_best[:i]))
        df["auc"].append(s["auc"])
        df["acc"].append(s["acc"])
        df["f1macro"].append(s["f1macro"])
        df["confusion"].append(s["confusion"])

    df = pd.DataFrame(df)
    return df




####################################################################################################
####################################################################################################
def sk_feature_prior_shift(df) :
    """
     Label is drifting
    https://dkopczyk.quantee.co.uk/covariate_shift/

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pass


def sk_feature_concept_shift(df) :
    """

       (X,y) distribution relation is shifting.
    https://dkopczyk.quantee.co.uk/covariate_shift/

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pass


def sk_feature_covariate_shift(dftrain, dftest, colname, nsample=10000):
    """
      X is drifting
    Parameters
    ----------
    dftrain : TYPE
        DESCRIPTION.
    dftest : TYPE
        DESCRIPTION.
    colname : TYPE
        DESCRIPTION.
    nsample : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    drop_list : TYPE
        DESCRIPTION.

    """
    n1 = nsample if len(dftrain) > nsample else len(dftrain)
    n2 = nsample if len(dftest) > nsample else len(dftest)
    train = dftrain[colname].sample(n1, random_state=12)
    test = dftest[colname].sample(n2, random_state=11)

    ## creating a new feature origin
    train["origin"] = 0
    test["origin"] = 1

    ## combining random samples
    combi = train.append(test)
    y = combi["origin"]
    combi.drop("origin", axis=1, inplace=True)

    ## modelling
    model = RandomForestClassifier(n_estimators=50, max_depth=7, min_samples_leaf=5)
    drop_list = []
    for i in combi.columns:
        score = cross_val_score(model, pd.DataFrame(combi[i]), y, cv=2, scoring="roc_auc")


        if np.mean(score) > 0.8:
            drop_list.append(i)
        print(i, np.mean(score))
    return drop_list



def sk_model_eval_classification_cv(clf, X, y, test_size=0.5, ncv=1, method="random"):
    """
    :param clf:
    :param X:
    :param y:
    :param test_size:
    :param ncv:
    :param method:
    :return:
    """
    if method == "kfold":
        kf = StratifiedKFold(n_splits=ncv, shuffle=True)
        clf_list = {}
        for i, itrain, itest in enumerate(kf.split(X, y)):
            print("###")
            Xtrain, Xtest = X[itrain], X[itest]
            ytrain, ytest = y[itrain], y[itest]
            clf_list[i], _ = sk_model_eval_classification(clf, 1, Xtrain, ytrain, Xtest, ytest)

    else:
        clf_list = {}
        for i in range(0, ncv):
            print("############# CV-{i}######################################".format(i=i))
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, shuffle=True)

            clf_list[i], _ = sk_model_eval_classification(clf, 1, Xtrain, ytrain, Xtest, ytest)

    return clf_list



""" 
def sk_cluster_algo_custom(Xmat, algorithm, args, kwds, returnval=1):
    pass
Plot the cLuster using specific Algo
    distance_matrix = pairwise_distances(blobs)
    clusterer = hdbscan.HDBSCAN(metric='precomputed')
    clusterer.fit(distance_matrix)
    clusterer.labels_

    {'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
 'canberra': hdbscan.dist_metrics.CanberraDistance,
 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
 'dice': hdbscan.dist_metrics.DiceDistance,
 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
 'hamming': hdbscan.dist_metrics.HammingDistance,
 'haversine': hdbscan.dist_metrics.HaversineDistance,
 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
 'jaccard': hdbscan.dist_metrics.JaccardDistance,
 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
 'l1': hdbscan.dist_metrics.ManhattanDistance,
 'l2': hdbscan.dist_metrics.EuclideanDistance,
 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
 'matching': hdbscan.dist_metrics.MatchingDistance,
 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
 'p': hdbscan.dist_metrics.MinkowskiDistance,
 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}

    """

"""
def sk_cluster_kmeans(Xmat, nbcluster=5, isprint=False, isnorm=False) :
  from sklearn.cluster import k_means
  stdev=  np.std(Xmat, axis=0)
  if isnorm  : Xmat=   (Xmat - np.mean(Xmat, axis=0)) / stdev

  sh= Xmat.shape
  Xdim= 1 if len(sh) < 2 else sh[1]   #1Dim vector or 2dim-3dim vector
  print(len(Xmat.shape), Xdim)
  if Xdim==1 :  Xmat= Xmat.reshape((sh[0],1))

  kmeans = sk.cluster.KMeans(n_clusters= nbcluster)
  kmeans.fit(Xmat)
  centroids, labels= kmeans.cluster_centers_,  kmeans.labels_

  if isprint :
   import matplotlib.pyplot as plt
   colors = ["g.","r.","y.","b.", "k."]
   if Xdim==1 :
     for i in range(0, sh[0], 5):  plt.plot(Xmat[i], colors[labels[i]], markersize = 5)
     plt.show()
   elif Xdim==2 :
     for i in range(0, sh[0], 5):  plt.plot(Xmat[i,0], Xmat[i,1], colors[labels[i]], markersize = 2)
     plt.show()
   else :
      print('Cannot Show higher than 2dim')

  return labels, centroids, stdev
"""

"""


    clfrf = sk.ensemble.RandomForestClassifier(
        n_estimators=nbtree,
        max_depth=maxdepth,
        max_features="sqrt",
        criterion="entropy",
        n_jobs=njobs,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced",
    )
    
    
    
https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api

X_train, X_test, up_train, up_test, r_train, r_test, u_train, u_test, d_train, d_test = model_selection.train_test_split(
    X, up, r, universe, d, test_size=0.25, random_state=99)


train_cols = X_train.columns.tolist()

train_data = lgb.Dataset(X_train, label=up_train.astype(int), 
                         feature_name=train_cols)
test_data = lgb.Dataset(X_test, label=up_test.astype(int), 
                        feature_name=train_cols, reference=train_data)
                        
                        
# LGB parameters:
params = {'learning_rate': 0.05,
          'boosting': 'gbdt', 
          'objective': 'binary',
          'num_leaves': 2000,
          'min_data_in_leaf': 200,
          'max_bin': 200,
          'max_depth': 16,
          'seed': 2018,
          'nthread': 10,}


# LGB training:
lgb_model = lgb.train(params, train_data, 
                      num_boost_round=1000, 
                      valid_sets=(test_data,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)
                      
                      

# DF, based on which importance is checked
X_importance = X_test

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_importance)


# Plot summary_plot
shap.summary_plot(shap_values, X_importance)
                      

# Plot summary_plot as barplot:
shap.summary_plot(shap_values, X_importance, plot_type='bar')


shap.dependence_plot("returnsClosePrevRaw10_lag_3_mean", shap_values, X_importance)



"""
