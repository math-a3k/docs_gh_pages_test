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
from typing import Union,TypeVar, List
from tqdm import tqdm
from box import Box
import scipy.stats as scs
import matplotlib.pyplot as plt



##################################################################################################
from utilmy import log, log2
def help():
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    print(ss)

   


    
#################################################################################################
def test_all()
    test1()

def test1():



   metrics_calc(dirin:str, dirout=None, colid='userid',  colrec='reclist',  coltrue='purchaselist',  colinfo='genre_ist',   method=[''], nsample=-1,  nfile=10, )


def test_data_fake():

    df['reclist'] = [  [str(i) for i in np.random()      ]

    retrun df


#################################################################################################
def metrics_calc(dirin:Union[str, pd.DataFrame], dirout:str=None, colid='userid',  colrec='reclist', coltrue='purchaselist',  colinfo='genrelist',  colts='datetime',    method=[''], 
                 nsample=-1,  nfile=1, **kw):
  """  metrics for recommender in batch model
    example_predictions = [
        ['1', '2', 'C', 'D'],
        ['4', '3', 'm', 'X'],
        ['7', 'B', 't', 'X']
    ]
    recmetrics.personalization(predicted=example_predictions)

   https://github.com/statisticianinstilettos/recmetrics/blob/master/example.ipynb


   recmetrics.precision_recall_plot(targs=actual, preds=model_probs)

  """
  from utilmy import pd_read_file, pd_to_file
  import recmetrics

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
    if mi == 'personalization': res[mi] = recmetrics.personalization( df[colrec].values)
    if mi == 'intralist':       res[mi] = recmetrics.intra_list_similarity( df['rec_list'].values, feature_df)
    if mi == 'mrr_at_k':        res[mi] = mrr_at_k( df['rec_list'].values,  df[coltrue].values, k=5 )





  dfres = pd.DataFrame(res)

  if isinstance(dirout, str):
     pd_to_file(dfres, dirout , show=1)
  return dfres




###############################################
def statistics(x_train, y_train, x_test, y_test, y_pred):
    train_size = len(x_train)
    test_size = len(x_test)
    # num non-zero preds
    num_preds = len([p for p in y_pred if p])
    return {
        'training_set__size': train_size,
        'test_set_size': test_size,
        'num_non_null_predictions': num_preds
    }


def sample_hits_at_k(y_preds, y_test, x_test=None, k=3, size=3):
    hits = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
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


def sample_misses_at_k(y_preds, y_test, x_test=None, k=3, size=3):
    misses = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
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


def hit_rate_at_k_nep(y_preds, y_test, k=3):
    y_test = [[k] for k in y_test]
    return hit_rate_at_k(y_preds, y_test, k=k)


def hit_rate_at_k(y_preds, y_test, k=3):
    hits = 0
    for _p, _y in zip(y_preds, y_test):
        if len(set(_p[:k]).intersection(set(_y))) > 0:
            hits += 1
    return hits / len(y_test)


def mrr_at_k_nep(y_preds, y_test, k=3):
    """
    Computes MRR
    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    y_test = [[k] for k in y_test]
    return mrr_at_k(y_preds, y_test, k=k)


def mrr_at_k(y_preds, y_test, k=3):
    """
    Computes MRR
    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    rr = []
    for _p, _y in zip(y_preds, y_test):
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
    # estimate popularity from training data
    pop_map = collections.defaultdict(lambda : 0)
    num_interactions = 0
    for session in x_train:
        for event in session:
            pop_map[event] += 1
            num_interactions += 1
    # normalize popularity
    pop_map = {k:v/num_interactions for k,v in pop_map.items()}
    all_popularity = []
    for p in y_preds:
        average_pop = sum(pop_map.get(_, 0.0) for _ in p[:k]) / len(p) if len(p) > 0 else 0
        all_popularity.append(average_pop)
    return sum(all_popularity) / len(y_preds)


def precision_at_k(y_preds, y_test, k=3):
    precision_ls = [len(set(_y).intersection(set(_p[:k]))) / len(_p) if _p else 1 for _p, _y in zip(y_preds, y_test)]
    return np.average(precision_ls)


def recall_at_k(y_preds, y_test, k=3):
    recall_ls = [len(set(_y).intersection(set(_p[:k]))) / len(_y) if _y else 1 for _p, _y in zip(y_preds, y_test)]
    return np.average(recall_ls)



def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def _mean_ranking_metric(predictions, labels, metric):
    """Helper function for precision_at_k and mean_average_precision"""
    # do not zip, as this will require an extra pass of O(N). Just assert
    # equal length and index (compute in ONE pass of O(N)).
    # if len(predictions) != len(labels):
    #     raise ValueError("dim mismatch in predictions and labels!")
    # return np.mean([
    #     metric(np.asarray(predictions[i]), np.asarray(labels[i]))
    #     for i in range(len(predictions))
    # ])
    
    # Actually probably want lazy evaluation in case preds is a 
    # generator, since preds can be very dense and could blow up 
    # memory... but how to assert lengths equal? FIXME
    return np.mean([
        metric(np.asarray(prd), np.asarray(labels[i]))
        for i, prd in enumerate(predictions)  # lazy eval if generator
    ])


def _warn_for_empty_labels():
    """Helper for missing ground truth sets"""
    warnings.warn("Empty ground truth set! Check input data")
    return 0.


def precision_at(predictions, labels, k=10, assume_unique=True):
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
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the precision.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> precision_at(preds, labels, 1)
    0.33333333333333331
    >>> precision_at(preds, labels, 5)
    0.26666666666666666
    >>> precision_at(preds, labels, 15)
    0.17777777777777778
    """
    # validate K
    _require_positive_k(k)

    def _inner_pk(pred, lab):
        # need to compute the count of the number of values in the predictions
        # that are present in the labels. We'll use numpy in1d for this (set
        # intersection in O(1))
        if lab.shape[0] > 0:
            n = min(pred.shape[0], k)
            cnt = np.in1d(pred[:n], lab, assume_unique=assume_unique).sum()
            return float(cnt) / k
        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_pk)


def mean_average_precision(predictions, labels, assume_unique=True):
    """Compute the mean average precision on predictions and labels.
    Returns the mean average precision (MAP) of all the queries. If a query
    has an empty ground truth set, the average precision will be zero and a
    warning is generated.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> mean_average_precision(preds, labels)
    0.35502645502645497
    """
    def _inner_map(pred, lab):
        if lab.shape[0]:
            # compute the number of elements within the predictions that are
            # present in the actual labels, and get the cumulative sum weighted
            # by the index of the ranking
            n = pred.shape[0]

            # Scala code from Spark source:
            # var i = 0
            # var cnt = 0
            # var precSum = 0.0
            # val n = pred.length
            # while (i < n) {
            #     if (labSet.contains(pred(i))) {
            #         cnt += 1
            #         precSum += cnt.toDouble / (i + 1)
            #     }
            #     i += 1
            # }
            # precSum / labSet.size

            arange = np.arange(n, dtype=np.float32) + 1.  # this is the denom
            present = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            prec_sum = np.ones(present.sum()).cumsum()
            denom = arange[present]
            return (prec_sum / denom).sum() / lab.shape[0]

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_map)


def ndcg_at(predictions, labels, k=10, assume_unique=True):
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
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the NDCG.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at(preds, labels, 10)
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
            # if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                lab = np.unique(lab)

            n_lab = lab.shape[0]
            n_pred = pred.shape[0]
            n = min(max(n_pred, n_lab), k)  # min(min(p, l), k)?

            # similar to mean_avg_prcsn, we need an arange, but this time +2
            # since python is zero-indexed, and the denom typically needs +1.
            # Also need the log base2...
            arange = np.arange(n, dtype=np.float32)  # length n

            # since we are only interested in the arange up to n_pred, truncate
            # if necessary
            arange = arange[:n_pred]
            denom = np.log2(arange + 2.)  # length n
            gains = 1. / denom  # length n

            # compute the gains where the prediction is present in the labels
            dcg_mask = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            dcg = gains[dcg_mask].sum()

            # the max DCG is sum of gains where the index < the label set size
            max_dcg = gains[arange < n_lab].sum()
            return dcg / max_dcg

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_ndcg)



###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()  
  
