from __future__ import absolute_import, division
# -*- coding: utf-8 -*-
MNAME = "utilmy.recsys.metric"
HELP = """"
All about metrics


https://github.com/jacopotagliabue/reclist/blob/main/reclist/recommenders/prod2vec.py


https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb


https://github.com/AstraZeneca/rexmex


# Recommender system ranking metrics derived from Spark source for use with
# original Spark Scala source code for recommender metrics.
# https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala



https://rexmex.readthedocs.io/en/latest/modules/root.html#module-rexmex.metrics.ranking



"""
import os, sys, random, numpy as np, pandas as pd, fire, time, itertools, collections, warnings
from typing import Union,TypeVar, List, Tuple
from tqdm import tqdm
from box import Box
import scipy.stats as scs
import matplotlib.pyplot as plt

import random
from itertools import product
from math import sqrt

import scipy.sparse as sp
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve


##################################################################################################
from utilmy import log, log2
def help():
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    print(ss)

#################################################################################################

def test_all():
    test_metrics()

def test_metrics():
    df, popdict, feature_df = test_get_testdata()
    result = metrics_calc(df,
             methods=['personalization','catalog_coverage','intra_list_similarity',
                      'recall_average_at_k_mean','novelty','recommender_precision','recommender_recall'],
             featuredf=feature_df,
             popdict=popdict
             )
    result = result.set_index('metrics').to_dict()['values']
    
    assert np.isclose(result['catalog_coverage'], 100.0, rtol=1e-1)
    assert np.isclose(result['intra_list_similarity'], 0.167, rtol=1e-1)
    assert np.isclose(result['recall_average_at_k_mean'], 0.667, rtol=1e-1)
    assert np.isclose(result['novelty'], -1.905, rtol=1e-1)
    assert np.isclose(result['personalization'], 0.389, rtol=1e-1)
    assert np.isclose(result['catalog_coverage'], 100.0, rtol=1e-1)
    assert np.isclose(result['recommender_precision'], 0.75, rtol=1e-1)
    assert np.isclose(result['recommender_recall'], 0.75, rtol=1e-1)

   
def test_get_testdata():

    df = pd.DataFrame({
        'user_id': [1,2,3,4],
        'reclist': [[1,2,4],[2,4,5],[1,2,5],[2,3,4]],
        'purchaselist': [[1,4,2],[1,3,4],[2,2,5],[2,3,4]],
        })
    popdict = {1:10,2:20,3:20,4:10,5:20}

    lst_features = {1: {'Action': 0, 'Comedy': 0, 'Romance': 0},
                    2: {'Action': 0, 'Comedy': 0, 'Romance': 0},
                    3: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                    4: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                    5: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                    6: {'Action': 1, 'Comedy': 0, 'Romance': 0},
                    7: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                    8: {'Action': 0, 'Comedy': 0, 'Romance': 0},
                    9: {'Action': 1, 'Comedy': 0, 'Romance': 0},
                    10: {'Action': 1, 'Comedy': 0, 'Romance': 0}}

    feature_df = pd.DataFrame.from_dict(lst_features, orient="index").reset_index().rename(
        columns={"index": "movieId"}).set_index("movieId")

    return df, popdict, feature_df


#################################################################################################
def metrics_calc_batch(dirin:Union[str, pd.DataFrame], dirout:str=None, 
                       colid='userid',  colrec='reclist', coltrue='purchaselist',  colinfo='genrelist',  
                       colts='datetime',    method=[''], 
                  nsample=-1,  nfile=1, **kw):
     """  Distributed metric calculation


     """             
     import glob
     flist = glob.glob(dirin)
     for fi in flist :   
         metrics_calc(dirin= fi, dirout=dirout, colid=colid,  colrec= colrec, coltrue=coltrue,  
                      colinfo= colinfo,  colts= colts,    method=method, 
                  nsample= nsample,  nfile= nfile, **kw)



def metrics_calc(dirin:Union[str, pd.DataFrame], 
                 dirout:str=None, 
                 colid='userid',  
                 colrec='reclist', 
                 coltrue='purchaselist',  
                 colinfo='genrelist',  
                 colts='datetime', 
                 methods=[''], 
                 nsample=-1, 
                 nfile=1,
                 featuredf:pd.DataFrame=None,
                 popdict:dict=None,
                 topk=5,
                 **kw):
    
    from utilmy import pd_read_file, pd_to_file

    if isinstance(dirin, pd.DataFrame):
        df = dirin 
    else :
        df = pd_read_file(dirin, nfile= nfile, npool=4)  

    
    if nsample > 0:  df = df.sample(n=nsample)

    if isinstance(df[colrec].values[0], str) :
        df[colrec] = df[colrec].apply(lambda x : x.split(","))  #### list
        ####  userid ---> colrec: [] 23243,2342,324345,4353,453,45345 ]

    if isinstance(df[coltrue].values[0], str) :
        df[coltrue] = df[coltrue].apply(lambda x : x.split(","))  #### list
        ####  userid ---> colrec: [] 23243,2342,324345,4353,453,45345 ]


    res = Box({})
    for mi in methods :
        if mi == 'personalization':              res[mi] = personalization(df[colrec].tolist())
        if mi == 'catalog_coverage':             res[mi] = catalog_coverage(df[colrec].tolist(), 
                                                            catalog=list(set([p for sublist in df.reclist.tolist() for p in sublist] + \
                                                                    [p for sublist in df.purchaselist.tolist() for p in sublist])))



        if mi == 'intra_list_similarity':        res[mi] = intra_list_similarity(df[colrec].tolist(), featuredf)
        if mi == 'recall_average_at_k_mean':                         res[mi] = recall_average_at_k_mean(actual=df[colrec].tolist(), y_preds=df[coltrue].tolist(), k=topk)
        if mi == 'novelty':                      res[mi] = novelty(y_preds=df[colrec].tolist(), pop=popdict, u=df.shape[0],
                                                                   n=max(df[colrec].apply(lambda x: len(x)).values))[0]
        if mi == 'recommender_precision':        res[mi] = recommender_precision(df[colrec].tolist(),  df[coltrue].tolist())
        if mi == 'recommender_recall':           res[mi] = recommender_recall(df[colrec].tolist(),  df[coltrue].tolist())


    dfres = pd.DataFrame(list(res.items()), columns=['metrics', 'values'])

    if isinstance(dirout, str):
        pd_to_file(dfres, dirout , show=1)
    return dfres




####################################################################
def personalization(y_preds: List[list]) -> float:
    """
    Personalization measures recommendation similarity across users.
    A high score indicates good personalization (user's lists of recommendations are different).
    A low score indicates poor personalization (user's lists of recommendations are very similar).
    A model is "personalizing" well if the set of recommendations for each user is different.
    Parameters:
    ----------
    y_preds : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        The personalization score for all recommendations.
    """

    def make_rec_matrix(y_preds: List[list]) -> sp.csr_matrix:
        df = pd.DataFrame(data=y_preds).reset_index().melt(
            id_vars='index', value_name='item',
        )
        df = df[['index', 'item']].pivot(index='index', columns='item', values='item')
        df = pd.notna(df)*1
        rec_matrix = sp.csr_matrix(df.values)
        return rec_matrix

    #create matrix for recommendations
    y_preds = np.array(y_preds)
    rec_matrix_sparse = make_rec_matrix(y_preds)

    #calculate similarity for every user's recommendation list
    similarity = cosine_similarity(X=rec_matrix_sparse, dense_output=False)

    #calculate average similarity
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))
    return 1-personalization

def catalog_coverage(y_preds: List[list], catalog: list) -> float:
    """
    Computes the catalog coverage for k lists of recommendations
    Parameters
    ----------
    y_preds : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    catalog: list
        A list of all unique items in the training data
        example: ['A', 'B', 'C', 'X', 'Y', Z]
    Returns
    ----------
    catalog_coverage:
        The catalog coverage of the recommendations as a percent
        rounded to 2 decimal places
    ----------    
    Metric Defintion:
    Ge, M., Delgado-Battenfeld, C., & Jannach, D. (2010, September).
    Beyond accuracy: evaluating recommender systems by coverage and serendipity.
    In Proceedings of the fourth ACM conference on Recommender systems (pp. 257-260). ACM.
    """
    predicted_flattened = [p for sublist in y_preds for p in sublist]
    L_predictions = len(set(predicted_flattened))
    catalog_coverage = round(L_predictions/(len(catalog)*1.0)*100,2)
    return catalog_coverage

def intra_list_similarity(y_preds: List[list], feature_df: pd.DataFrame) -> float:
    """
    Computes the average intra-list similarity of all recommendations.
    This metric can be used to measure diversity of the list of recommended items.
    Parameters
    ----------
    y_preds : a list of lists
        Ordered predictions
        Example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
        The average intra-list similarity for recommendations.
    """
    feature_df = feature_df.fillna(0)
    Users = range(len(y_preds))
    ils = [_single_list_similarity(y_preds[u], feature_df, u) for u in Users]
    return np.mean(ils)

def _single_list_similarity(y_preds: list, feature_df: pd.DataFrame, u: int) -> float:
    """
    Computes the intra-list similarity for a single list of recommendations.
    Parameters
    ----------
    y_preds : a list
        Ordered predictions
        Example: ['X', 'Y', 'Z']
    feature_df: dataframe
        A dataframe with one hot encoded or latent features.
        The dataframe should be indexed by the id used in the recommendations.
    Returns:
    -------
    ils_single_user: float
        The intra-list similarity for a single list of recommendations.
    """
    # exception y_preds list empty
    if not(y_preds):
        raise Exception('Predicted list is empty, index: {0}'.format(u))

    #get features for all recommended items
    recs_content = feature_df.loc[y_preds]
    recs_content = recs_content.dropna()
    recs_content = sp.csr_matrix(recs_content.values)

    #calculate similarity scores for all items in list
    similarity = cosine_similarity(X=recs_content, dense_output=False)

    #get indicies for upper right triangle w/o diagonal
    upper_right = np.triu_indices(similarity.shape[0], k=1)

    #calculate average similarity score of all recommended items in list
    ils_single_user = np.mean(similarity[upper_right])
    return ils_single_user

def recall_avg_at_k(actual: list, y_preds: list, k=10) -> int:
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be y_preds
    y_preds : list
        An ordered list of y_preds items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : int
        The average recall at k.
    """
    if len(y_preds)>k:
        y_preds = y_preds[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(y_preds):
        if p in actual and p not in y_preds[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)

def recall_average_at_k_mean(actual: List[list], y_preds: List[list], k=10) -> int:
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be y_preds
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    y_preds : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall_average_at_k_mean: int
            The mean average recall at k (mar@k)
    """
    return np.mean([recall_avg_at_k(a,p,k) for a,p in zip(actual, y_preds)])

def novelty(y_preds: List[list], pop: dict, u: int, n: int) -> Tuple[float, list]:
    """
    Computes the novelty for a list of recommendations
    Parameters
    ----------
    y_preds : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    pop: dictionary
        A dictionary of all items alongside of its occurrences counter in the training data
        example: {1198: 893, 1270: 876, 593: 876, 2762: 867}
    u: integer
        The number of users in the training data
    n: integer
        The length of recommended lists per user
    Returns
    ----------
    novelty:
        The novelty of the recommendations in system level
    mean_self_information:
        The novelty of the recommendations in recommended top-N list level
    ----------    
    Metric Defintion:
    Zhou, T., Kuscsik, Z., Liu, J. G., Medo, M., Wakeling, J. R., & Zhang, Y. C. (2010).
    Solving the apparent diversity-accuracy dilemma of recommender systems.
    Proceedings of the National Academy of Sciences, 107(10), 4511-4515.
    """
    mean_self_information = []
    k = 0
    for sublist in y_preds:
        self_information = 0
        k += 1
        for i in sublist:
            self_information += np.sum(-np.log2(pop[i]/u))
        mean_self_information.append(self_information/n)
    novelty = sum(mean_self_information)/k
    return novelty, mean_self_information


def recommender_precision(y_preds: List[list], actual: List[list]) -> int:
    """
    Computes the precision of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be y_preds
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    y_preds : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        precision: int
    """
    def calc_precision(y_preds, actual):
        prec = [value for value in y_preds if value in actual]
        prec = np.round(float(len(prec)) / float(len(y_preds)), 4)
        return prec

    precision = np.mean(list(map(calc_precision, y_preds, actual)))
    return precision


def recommender_recall(y_preds: List[list], actual: List[list]) -> int:
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be y_preds
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    y_preds : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(y_preds, actual):
        reca = [value for value in y_preds if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, y_preds, actual)))
    return recall




#######################################################################
def statistics(x_train, y_train, x_test, y_true, y_pred):
    train_size = len(x_train)
    test_size = len(x_test)
    #num non-zero preds
    num_preds = len([p for p in y_pred if p])
    return {
        'training_set__size': train_size,
        'test_set_size': test_size,
        'num_non_null_predictions': num_preds
    }


def sample_hits_at_k(y_preds, y_true, x_test=None, k=3, size=3):
    hits = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_true)):
        if _y[0] in _p[:k]:
            hit_info = {
                'Y_TEST': [_y[0]],
                'Y_PRED': _p[:k],
            }
            if x_test:
                hit_info['X_TEST'] = [x_test[idx][0]]
            hits.append(hit_info)

    if len(hits) < size or size == -1:
        return hits
    return random.sample(hits, k=size)


def sample_misses_at_k(y_preds, y_true, x_test=None, k=3, size=3):
    misses = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_true)):
        if _y[0] not in _p[:k]:
            miss_info =  {
                'Y_TEST': [_y[0]],
                'Y_PRED': _p[:k],
            }
            if x_test:
                miss_info['X_TEST'] = [x_test[idx][0]]
            misses.append(miss_info)

    if len(misses) < size or size == -1:
        return misses
    return random.sample(misses, k=size)


def hit_rate_at_k_nep(y_preds, y_true, k=3):
    y_true = [[k] for k in y_true]
    return hit_rate_at_k(y_preds, y_true, k=k)


def hit_rate_at_k(y_preds, y_true, k=3):
    hits = 0
    for _p, _y in zip(y_preds, y_true):
        if len(set(_p[:k]).intersection(set(_y))) > 0:
            hits += 1
    return hits / len(y_true)


def mrr_at_k_nep(y_preds, y_true, k=3):
    """
    Computes MRR
    :param y_preds: y, as lists of lists
    :param y_true: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    y_true = [[k] for k in y_true]
    return mrr_at_k(y_preds, y_true, k=k)


def mrr_at_k(y_preds, y_true, k=3):
    """
    Computes MRR
    :param y_preds: y, as lists of lists
    :param y_true: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    rr = []
    for _p, _y in zip(y_preds, y_true):
        for rank, p in enumerate(_p[:k], start=1):
            if p in _y:
                rr.append(1 / rank)
                break
        else:
            rr.append(0)
    assert len(rr) == len(y_preds)
    return np.mean(rr)


def coverage_at_k(y_preds, product_data, k=3):
    pred_skus = set(itertools.chain.from_iterable(y_preds[:k]))
    all_skus = set(product_data.keys())
    nb_overlap_skus = len(pred_skus.intersection(all_skus))

    return nb_overlap_skus / len(all_skus)


def popularity_bias_at_k(y_preds, x_train, k=3):
    #estimate popularity from training data
    pop_map = collections.defaultdict(lambda : 0)
    num_interactions = 0
    for session in x_train:
        for event in session:
            pop_map[event] += 1
            num_interactions += 1
    #normalize popularity
    pop_map = {k:v/num_interactions for k,v in pop_map.items()}
    all_popularity = []
    for p in y_preds:
        average_pop = sum(pop_map.get(_, 0.0) for _ in p[:k]) / len(p) if len(p) > 0 else 0
        all_popularity.append(average_pop)
    return sum(all_popularity) / len(y_preds)


def precision_at_k(y_preds, y_true, k=3):
    precision_ls = [len(set(_y).intersection(set(_p[:k]))) / len(_p) if _p else 1 for _p, _y in zip(y_preds, y_true)]
    return np.average(precision_ls)


def recall_at_k(y_preds, y_true, k=3):
    recall_ls = [len(set(_y).intersection(set(_p[:k]))) / len(_y) if _y else 1 for _p, _y in zip(y_preds, y_true)]
    return np.average(recall_ls)



def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def _mean_ranking_metric(y, labels, metric):
    """Helper function for precision_at_k and mean_average_precision"""
    #do not zip, as this will require an extra pass of O(N). Just assert
    #equal length and index (compute in ONE pass of O(N)).
    if len(y) != len(labels):
        raise ValueError("dim mismatch in y and labels!")
    return np.mean([
        metric(np.asarray(y[i]), np.asarray(labels[i]))
        for i in range(len(y))
    ])
    
    #Actually probably want lazy evaluation in case preds is a 
    #generator, since preds can be very dense and could blow up 
    #memory... but how to assert lengths equal? 


def _warn_for_empty_labels():
    """Helper for missing ground truth sets"""
    warnings.warn("Empty ground truth set! Check input data")
    return 0.


def precision_at(y, labels, k=10, assume_unique=True):
    """Compute the precision at K.
    Compute the average precision of all the queries, truncated at
    ranking position k. If for a query, the ranking algorithm returns
    n (n is less than k) results, the precision value will be computed
    as #(relevant items retrieved) / k. This formula also applies when
    the size of the ground truth set is less than k.
    If a query has an empty ground truth set, zero will be used as
    precision together with a warning.
    Parameters
    ----------
    y : array-like, shape=(n_predictions,)
        The prediction array. The items that were y_preds, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the precision.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and y are each
        unique. That is, the same item is not y_preds multiple times or
        rated multiple times.
    Examples
    --------
    >>> y for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> precision_at(preds, labels, 1)
    0.33333333333333331
    >>> precision_at(preds, labels, 5)
    0.26666666666666666
    >>> precision_at(preds, labels, 15)
    0.17777777777777778
    """
    #validate K
    _require_positive_k(k)

    def _inner_pk(pred, lab):
        #need to compute the count of the number of values in the y
        #that are present in the labels. We'll use numpy in1d for this (set
        #intersection in O(1))
        if lab.shape[0] > 0:
            n = min(pred.shape[0], k)
            cnt = np.in1d(pred[:n], lab, assume_unique=assume_unique).sum()
            return float(cnt) / k
        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(y, labels, _inner_pk)


def mean_average_precision(y, labels, assume_unique=True):
    """Compute the mean average precision on y and labels.
    Returns the mean average precision (MAP) of all the queries. If a query
    has an empty ground truth set, the average precision will be zero and a
    warning is generated.
    Parameters
    ----------
    y : array-like, shape=(n_predictions,)
        The prediction array. The items that were y_preds, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and y are each
        unique. That is, the same item is not y_preds multiple times or
        rated multiple times.
    Examples
    --------
    >>> y for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> mean_average_precision(preds, labels)
    0.35502645502645497
    """
    def _inner_map(pred, lab):
        if lab.shape[0]:
            #compute the number of elements within the y that are
            #present in the actual labels, and get the cumulative sum weighted
            #by the index of the ranking
            n = pred.shape[0]

            """
            Scala code from Spark source:
            var i = 0
            var cnt = 0
            var precSum = 0.0
            val n = pred.length
            while (i < n) {
                if (labSet.contains(pred(i))) {
                    cnt += 1
                    precSum += cnt.toDouble / (i + 1)
                }
                i += 1
            }
            precSum / labSet.size
            """

            arange = np.arange(n, dtype=np.float32) + 1.  # this is the denom
            present = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            prec_sum = np.ones(present.sum()).cumsum()
            denom = arange[present]
            return (prec_sum / denom).sum() / lab.shape[0]

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(y, labels, _inner_map)


def ndcg_at_k(y, labels, k=10, assume_unique=True):
    """Compute the normalized discounted cumulative gain at K.
    Compute the average NDCG value of all the queries, truncated at ranking
    position k. The discounted cumulative gain at position k is computed as:
        sum,,i=1,,^k^ (2^{relevance of ''i''th item}^ - 1) / log(i + 1)
    and the NDCG is obtained by dividing the DCG value on the ground truth set.
    In the current implementation, the relevance value is binary.
    If a query has an empty ground truth set, zero will be used as
    NDCG together with a warning.
    Parameters
    ----------
    y : array-like, shape=(n_predictions,)
        The prediction array. The items that were y_preds, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the NDCG.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and y are each
        unique. That is, the same item is not y_preds multiple times or
        rated multiple times.
    Examples
    --------
    >>> y for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at_k(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at_k(preds, labels, 10)
    0.48791273434956867
    References
    ----------
    .. [1] K. Jarvelin and J. Kekalainen, "IR evaluation methods for
           retrieving highly relevant documents."
    """
    # validate K
    _require_positive_k(k)

    def _inner_ndcg(pred, lab):
        if lab.shape[0]:
            #if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                lab = np.unique(lab)

            n_lab = lab.shape[0]
            n_pred = pred.shape[0]
            #n = min(max(n_pred, n_lab), k)  min(min(p, l), k)?

            #similar to mean_avg_prcsn, we need an arange, but this time +2
            #since python is zero-indexed, and the denom typically needs +1.
            #Also need the log base2...
            arange = np.arange(n, dtype=np.float32)  # length n

            #since we are only interested in the arange up to n_pred, truncate
            #if necessary
            arange = arange[:n_pred]
            denom = np.log2(arange + 2.)  # length n
            gains = 1. / denom  #length n

            #compute the gains where the prediction is present in the labels
            dcg_mask = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            dcg = gains[dcg_mask].sum()

            #the max DCG is sum of gains where the index < the label set size
            max_dcg = gains[arange < n_lab].sum()
            return dcg / max_dcg

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(y, labels, _inner_ndcg)



###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()  
  
