# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas



"""
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci


########### LOCAL ##################################################################################
print("os.getcwd", os.getcwd())


def ztest():
    import sklearn as sk
    print(sk)



def pd_num_segment_limit(
    df, col_score="scoress", coldefault="y", ntotal_default=491, def_list=None, nblock=20.0
):
    """
    Calculate Segmentation of colum using rule based.
    :param df:
    :param col_score:
    :param coldefault:
    :param ntotal_default:
    :param def_list:
    :param nblock:
    :return:
    """

    if def_list is None:
        def_list = np.ones(21) * ntotal_default / nblock

    df["scoress_bin"] = df[col_score].apply(lambda x: np.floor(x / 1.0) * 1.0)
    dfs5 = (
        df.groupby("scoress_bin")
        .agg({col_score: "mean", coldefault: {"sum", "count"}})
        .reset_index()
    )
    dfs5.columns = [x[0] if x[0] == x[1] else x[0] + "_" + x[1] for x in dfs5.columns]
    dfs5 = dfs5.sort_values(col_score, ascending=False)
    # return dfs5

    l2 = []
    k = 1
    ndef, nuser = 0, 0
    for i, x in dfs5.iterrows():
        if k > nblock:
            break
        nuser = nuser + x[coldefault + "_count"]
        ndef = ndef + x[coldefault + "_sum"]
        pdi = ndef / nuser

        if ndef > def_list[k - 1]:
            # if  pdi > pdlist[k] :
            l2.append([np.round(x[col_score], 1), k, pdi, ndef, nuser])
            k = k + 1
            ndef, nuser = 0, 0
        l2.append([np.round(x[col_score], 1), k, pdi, ndef, nuser])
    l2 = pd.DataFrame(l2, columns=[col_score, "kaiso3", "pd", "ndef", "nuser"])
    return l2


def fun_get_segmentlimit(x, l1):
    """
    ##### Get Kaiso limit ###############################################################
    :param x:
    :param l1:
    :return :
    """
    for i in range(0, len(l1)):
        if x >= l1[i]:
            return i + 1
    return i + 1






def np_drop_duplicates(l1):
    """
    :param l1:
    :return :
    """
    l0 = np.array( list(OrderedDict((x, True) for x in l1).keys()) )
    return l0


def model_logistic_score(clf, df1, cols, coltarget, outype="score"):
    """

    :param clf:
    :param df1:
    :param cols:
    :param outype:
    :return:
    """

    def score_calc(yproba, pnorm=1000.0):
        yy = np.log(0.00001 + (1 - yproba) / (yproba + 0.001))
        # yy =  (yy  -  np.minimum(yy)   ) / ( np.maximum(yy) - np.minimum(yy)  )
        # return  np.maximum( 0.01 , yy )    ## Error it bias proba
        return yy

    X_all = df1[cols].values

    yall_proba = clf.predict_proba(X_all)[:, 1]
    yall_pred = clf.predict(X_all)
    try:
        y_all = df1[coltarget].values
        sk_showmetrics(y_all, yall_pred, yall_proba)
    except:
        pass

    yall_score = score_calc(yall_proba)
    yall_score = (
        1000 * (yall_score - np.min(yall_score)) / (np.max(yall_score) - np.min(yall_score))
    )

    if outype == "score":
        return yall_score
    if outype == "proba":
        return yall_proba, yall_pred





def split_train_test(X, y, split_ratio=0.8):
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=split_ratio, random_state=42, shuffle=False
    )
    print("train_X shape:", train_X.shape)
    print("val_X shape:", val_X.shape)

    print("train_y shape:", train_y.shape)
    print("val_y shape:", val_y.shape)

    return train_X, val_X, train_y, val_y


def split_train(df1, ntrain=10000, ntest=100000, colused=None, coltarget=None):
    n1 = len(df1[df1[coltarget] == 0])
    dft = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntest, False), :],
            df1[(df1[coltarget] == 1) & (df1["def"] > 201803)].iloc[:, :],
        )
    )

    X_test = dft[colused].values
    y_test = dft[coltarget].values
    print("test", sum(y_test))

    ######## Train data
    n1 = len(df1[df1[coltarget] == 0])
    dft2 = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntrain, False), :],
            df1[(df1[coltarget] == 1) & (df1["def"] > 201703) & (df1["def"] < 201804)].iloc[:, :],
        )
    )
    dft2 = dft2.iloc[np.random.choice(len(dft2), len(dft2), False), :]

    X_train = dft2[colused].values
    y_train = dft2[coltarget].values
    print("train", sum(y_train))
    return X_train, X_test, y_train, y_test


def split_train2(df1, ntrain=10000, ntest=100000, colused=None, coltarget=None, nratio=0.4):
    n1 = len(df1[df1[coltarget] == 0])
    n2 = len(df1[df1[coltarget] == 1])
    n2s = int(n2 * nratio)  # 80% of default

    #### Test data
    dft = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntest, False), :],
            df1[(df1[coltarget] == 1)].iloc[:, :],
        )
    )

    X_test = dft[colused].values
    y_test = dft[coltarget].values
    print("test", sum(y_test))

    ######## Train data
    n1 = len(df1[df1[coltarget] == 0])
    dft2 = pd.concat(
        (
            df1[df1[coltarget] == 0].iloc[np.random.choice(n1, ntrain, False), :],
            df1[(df1[coltarget] == 1)].iloc[np.random.choice(n2, n2s, False), :],
        )
    )
    dft2 = dft2.iloc[np.random.choice(len(dft2), len(dft2), False), :]

    X_train = dft2[colused].values
    y_train = dft2[coltarget].values
    print("train", sum(y_train))
    return X_train, X_test, y_train, y_test




