# -*- coding: utf-8 -*-
HELP = """merge ranking
Original file is located at
    https://colab.research.google.com/drive/1wMTlaj4PPBlpkh4VE0A3X7D0ZULWnok8


"""
import random, pandas as pd, numpy as np,  scipy, math

##################################################################################################
def log(*s): 
  print(*s, flush=True)





#####################################################################################
#####################################################################################
"""
Goal is to find new formulae for
     rank_score0  = FORMULAE( ...)

using cost_fitness (ie minimize)
and Genetic Programming algo.


"""

def cost_fitness(rank_score):
    """
    
    """
    ##### True list
    ltrue = [ str(i)  for i in range(0, 100) ]   

    #### Create noisy list 
    ltrue_rank = {x:i for i,x in enumerate(ltrue)}
    list_overlap =  ltrue
    ll1  = rank_generate_fake(dict_full, list_overlap, nsize=100, ncorrect=40)
    ll2  = rank_generate_fake(dict_full, list_overlap, nsize=100, ncorrect=50)


    #### Merge them using rank_score
    lnew = rank_merge_v5(ll1, ll2, kk= 1, rank_score= rank_score)

    ### Eval with True Rank
    spearman_rho = scipy.stats.spearmanr(ltrue,  lnew)
    return -spearman_rho



#### Example of rank_scores0
def rank_score0(rank1:list, rank2:list, adjust=1.0, kk=1.0)-> list:
    """     ### take 2 np.array and calculate one list of float (ie NEW scores for position)
 
     list of items:  a,b,c,d, ...

     item      a,b,c,d,e

     rank1 :   1,2,3,4 ,,n     (  a: 1,  b:2, ..)
     rank1 :   5,7,2,1 ,,n     (  a: 5,  b:6, ..)
    
     scores_new :   a: -7.999,  b:-2.2323   
     (item has new scores)
    
    """
    scores_new =  -1.0 / (kk + rank1) - 1.0 / (kk + rank2 * adjust)
    return scores_new


def rank_merge_v5(ll1:list, ll2:list, kk= 1, rank_score=None):
    """ Re-rank elements of list1 using ranking of list2
        20k dataframe : 6 sec ,  4sec if dict is pre-build
        Fastest possible in python
    """
    if len(ll2) < 1: return ll1
    n1, n2 = len(ll1), len(ll2)

    if not isinstance(ll2, dict) :
        ll2 = {x:i for i,x in enumerate( ll2 )  }  ### Most costly op, 50% time.

    adjust, mrank = (1.0 * n1) / n2, n2
    rank2 = np.array([ll2.get(sid, mrank) for sid in ll1])
    rank1 = np.arange(n1)
    rank3 = rank_score(rank1, rank2, adjust=1.0, kk=1.0) ### Score

    #### re-rank  based on NEW Scores.
    v = [ll1[i] for i in np.argsort(rank3)]
    return v  #### for later preprocess
    



def rank_generate_fake(dict_full, list_overlap, nsize=100, ncorrect=20):
    """  Returns a list of random rankings of size nsize where ncorrect
         elements have correct ranks

        Keyword arguments:
        dict_full    : a dictionary of 1000 objects and their ranks
        list_overlap : list items common to all lists
        nsize        : the total number of elements to be ranked
        ncorrect     : the number of correctly ranked objects
    """
    # first randomly sample nsize - len(list_overlap) elements from dict_full
    # of those, ncorrect of them must be correctly ranked
    random_vals = []
    while len(random_vals) <= nsize - len(list_overlap):
      rand = random.sample(list(dict_full), 1)
      if (rand not in random_vals and rand not in list_overlap):
        random_vals.append(rand[0])

    # next create list as aggregate of random_vals and list_overlap
    list2 = random_vals + list_overlap
    
    # shuffle nsize - ncorrect elements from list2 
    copy = list2[0:nsize - ncorrect]
    random.shuffle(copy)
    list2[0:nsize - ncorrect] = copy

    # ensure there are ncorrect elements in correct places
    if ncorrect == 0: 
      return list2
    rands = random.sample(list(dict_full)[0:nsize + 1], ncorrect + 1)
    for r in rands:
      list2[r] = list(dict_full)[r]
    return list2


































































def test1():
    """"
        # evaluate each rank aggregation algorithm by using spearman's rho's and kendall-tau's metrics
        # for varying levels of ncorrect in the generated rankings    
    
    """"

    def rank_merge_v4(ll1, ll2):
        """ Re-rank elements of list1 using ranking of list2
        """        
        n1, n2 = len(ll1), len(ll2)
        adjust, mrank = (1.0 * n1) / n2, n2
        rank3 = np.zeros(n1, dtype='float32')
        kk = 2

        for rank1, sid in enumerate(ll1):
            rank2 = np_find(ll2, sid)
            rank2 = mrank if rank2 == -1 else rank2
            rank3[rank1] = -rank_score(rank1, rank2, adjust, kk=kk)

        # Id of ll1 sorted list
        return [ll1[i] for i in np.argsort(rank3)]

    pd.set_option('display.max_columns', 7)
    for ncorrect in [0, 20, 50, 80, 100]:
      print('ncorrect = {}'.format(ncorrect))

      #### Fake with ncorrect 
      df, rank1, rank2, dict_full = rank_generatefake(ncorrect, 100)
      rank_true = list(dict_full)
        
        
      def rank_score(rank1, rank2, adjust=1.0, kk=1.0):
            return 1.0 / (kk + rank1) + 1.0 / (kk + rank2 * adjust)
        
        
      def loss(rank_score_formulae):        
         rankmerge = rank_merge_v3(rank1, rank2, 100)  
         metric    = kendall_tau(rank_true, rankmerge)  
        
         retrun metric*metric

     #### Find rank_score formulae, such as metric is minimize
    
    
###################################################################################################
###################################################################################################    
def test():
    # evaluate each rank aggregation algorithm by using spearman's rho's and kendall-tau's metrics
    # for varying levels of ncorrect in the generated rankings
    pd.set_option('display.max_columns', 7)
    for ncorrect in [0, 20, 50, 80, 100]:
      print('ncorrect = {}'.format(ncorrect))

      #### Fake with ncorrect 
      df, rank1, rank2, dict_full = rank_generatefake(ncorrect, 100)
      rank_true = list(dict_full)

      #### Algo merged
      borda     = rank_merge(df, method='borda')
      dowdall   = rank_merge(df, method='dowdall')
      mc4_rank  = rank_merge(df, method='mc4')
      custom    = rank_merge_v2(rank1, rank2, 100)
      avg       = rank_merge(df, method='average')
      df2 = pd.DataFrame([borda, dowdall, avg, mc4_rank, custom, rank1, rank2],
                         index = ['borda', 'dowdall', 'average', 'MC4', 'algo v2', 'rank1', 'rank2'])
      df2 = df2.transpose()

      #### Fill NA with Max ranking as default 
      for idx, col in enumerate(df2.columns):
        max_value = df2[col].max()
        df2[col]  = df2[col].replace(np.nan, max_value)


      dfres = rank_eval(rank_true, df2, 100)
      log(dfres)




############################################################################################
############################################################################################
def rank_adjust2(ll1, ll2, kk= 1):
    """ Re-rank elements of list1 using ranking of list2"""
    if len(ll2) < 1: return ll1        
    if isinstance(ll1, str): ll1 = ll1.split(",")
    if isinstance(ll2, str): ll2 = ll2.split(",")
    n1, n2 = len(ll1), len(ll2)

    if not isinstance(ll2, dict) :
       ll2 = {x:i for i,x in enumerate( ll2 )  }

    # log(ll1) ; log(ll2)

    adjust, mrank = (1.0 * n1) / n2, n2        
    rank2 = np.array([ll2.get(sid, mrank) for sid in ll1])
    rank1 = np.arange(n1)
    rank3 = -1.0 / (kk + rank1) - 1.0 / (kk + rank2 * adjust)

    # Id of ll1 sorted list
    v = [ll1[i] for i in np.argsort(rank3)]
    return ",".join( v)




def rank_generatefake(ncorrect=30, nsize=100):
    """
      Generate a fake rank list of size nrank that contains ncorrect elements that
      are correctly ranked. Returns a dataframe where the two fake ranks are merged

      # each column of the dataframe should contain the rank each ranker assigns to each candidate
      # because of the fake generate algorithm, some items might not have ranks assigned to them by the ranker

      Keyword arguments:
      ncorrect: The number of correctly ranked elements in list
      nrank: Total number of elements to be ranked
    """
    ### Reference 1000 elements
    dict_full = {}
    for i in range(1000):
      dict_full[i] = i + 1

    list_overlap = [1, 15, 20, 30, 80]
    list1 = rank_generate_fake(dict_full, list_overlap, nsize=nsize,  ncorrect=ncorrect) 
    list2 = rank_generate_fake(dict_full, list_overlap, nsize=nsize,  ncorrect=ncorrect)
    rank1, rank2 = [None] * 1000, [None] * 1000
    for i in range(1000):
      if i in list1:
        rank1[i - 1] = list1.index(i)

      if i in list2:
        rank2[i - 1] = list2.index(i)

    df = pd.DataFrame([rank1, rank2], index=['rank1', 'rank2'])
    df = df.fillna(np.nan)
    df = df.transpose()
    # normalize by replacing missing values
    df = rank_fillna(df)
    return df, rank1, rank2, dict_full


def rank_generate_fake(dict_full, list_overlap, nsize=100, ncorrect=20):
    """  Returns a list of random rankings of size nsize where ncorrect
         elements have correct ranks

        Keyword arguments:
        dict_full    : a dictionary of 1000 objects and their ranks
        list_overlap : list items common to all lists
        nsize        : the total number of elements to be ranked
        ncorrect     : the number of correctly ranked objects
    """
    # first randomly sample nsize - len(list_overlap) elements from dict_full
    # of those, ncorrect of them must be correctly ranked
    random_vals = []
    while len(random_vals) <= nsize - len(list_overlap):
      rand = random.sample(list(dict_full), 1)
      if (rand not in random_vals and rand not in list_overlap):
        random_vals.append(rand[0])

    # next create list as aggregate of random_vals and list_overlap
    list2 = random_vals + list_overlap
    
    # shuffle nsize - ncorrect elements from list2 
    copy = list2[0:nsize - ncorrect]
    random.shuffle(copy)
    list2[0:nsize - ncorrect] = copy

    # ensure there are ncorrect elements in correct places
    if ncorrect == 0: 
      return list2
    rands = random.sample(list(dict_full)[0:nsize + 1], ncorrect + 1)
    for r in rands:
      list2[r] = list(dict_full)[r]
    return list2



def rank_fillna(df):
    """Replace NaN value with maximum value of column
        Keyword arguments:
        df: a dataframe where each row represents an item and each column represents it's
            assigned rank by a ranker
    """
    cols = df.columns
    for col in cols:
      max_val = df[col].max()
      df[col] = df[col].replace(np.nan, max_val)
    return df





############################################################################################
def rank_eval(rank_true, dfmerged, nrank=100):
  """
    Returns a dataframe where the columns are the rank aggregation algorithms
    and the rows are the metrics used to evaluate the algorithms

    Keyword arguments:
    rank_true: a list containing the accurate rank for each item
    dfmerged: a dataframe where each row represents an item and each column represents
              the rank aggregation algorithms
    nrank: the total number of elements to be ranked
  """
  columns = dfmerged.columns
  df = pd.DataFrame([], index=['spearman_rho', 'kendall-tau'])
  for col in columns:
    rank_list    = dfmerged[col].tolist()
    spearman_rho = scipy.stats.spearmanr(rank_true[0:nrank],  rank_list[0:nrank])
    kendall_tau  = scipy.stats.kendalltau(rank_true[0:nrank], rank_list[0:nrank])
    df[col]      = [spearman_rho[0], kendall_tau[0]]
  return df







##########################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()


