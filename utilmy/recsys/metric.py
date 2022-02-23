# -*- coding: utf-8 -*-
MNAME = "utilmy.recsys.ab"
HELP = """"
All about abtest

cd utilmy/recsys/ab
python ab.py ab_getstat --df  /mypath/data.parquet   



_pooled_prob(N_A, N_B, X_A, X_B)
_pooled_SE(N_A, N_B, X_A, X_B)
_p_val(N_A, N_B, p_A, p_B)
np_calculate_z_val(sig_level=0.05, two_tailed=True)
np_calculate_confidence_interval(sample_mean=0, sample_std=1, sample_size=1, sig_level=0.05)
np_calculate_ab_dist(stderr, d_hat=0, group_type='control')
pd_generate_ctr_data(N_A, N_B, p_A, p_B, days=None, control_label='A'
np_calculate_min_sample_size(bcr, mde, power=0.8, sig_level=0.05)
plot_confidence_interval(ax, mu, s, sig_level=0.05, color='grey')
plot_norm_dist(ax, mu, std, with_CI=False, sig_level=0.05, label=None)
plot_binom_dist(ax, A_converted, A_cr, A_total, B_converted, B_cr, B_total)
plot_null_hypothesis_dist(ax, stderr)
plot_alternate_hypothesis_dist(ax, stderr, d_hat)
show_area(ax, d_hat, stderr, sig_level, area_type='power')
plot_ab(ax, N_A, N_B, bcr, d_hat, sig_level=0.05, show_power=False
zplot(ax, area=0.95, two_tailed=True, align_right=False)
abplot_CI_bars(N, X, sig_level=0.05, dmin=None)
funnel_CI_plot(A, B, sig_level=0.05)
ab_getstat(df,treatment_col,measure_col,attribute_cols,control_label,variation_label,inference_method,hypothesis,alpha,experiment_name)


https://pypi.org/project/abracadabra/


"""
import os, sys, random, numpy as np, pandas as pd, fire, time, itertools, collections
from typing import List
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
  pass


   metrics_calc(dirin:str, dirout=None, colid='userid',  colrec='reclist',  coltrue='purchaselist',  colinfo='genre_ist',   method=[''], nsample=-1,  nfile=10, )


def test_data_fake():

    df['reclist'] = [  [str(i) for i in np.random()      ]

    retrun df

def metrics_calc(dirin:str, dirout=None, colid='userid',  colrec='reclist', coltrue='purchaselist',  colinfo='genrelist',     method=[''], 
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

  if isinstance(df[colrec].valus[0], str) :
     df[colrec] = df[colrec].apply(lambda x : x.split(","))  #### list
     ####  userid ---> colrec: [] 23243,2342,324345,4353,453,45345 ]

  if isinstance(df[coltrue].valus[0], str) :
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








###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()  
  
