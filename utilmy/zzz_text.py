HELP="""
pip install datasketch

https://github.com/topics/hypothesis-testing?l=python&o=desc&s=stars

https://pypi.org/project/pysie/#description



"""
import os,sys,  pandas as pd, numpy as np
from typing import List






########################################################################
def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    from utilmy import text
    from difflib import SequenceMatcher
    from pandas._testing import assert_series_equal

    list1 = ['dog', 'cat']
    list2 = ['doggy', 'cat']

    cols = ['name','pet_name']
    sample_df = pd.DataFrame(zip(list1, list2), columns=cols)
    original_value = text.pd_text_similarity(sample_df, cols)['score']

    check_similarity = lambda *x: SequenceMatcher(None, *x[0]).ratio()
    
    output_value = pd.Series(sample_df.apply(lambda x: check_similarity(x[[*cols]]), axis=1), name="score")

    assert_series_equal(original_value, output_value, check_names=False)


    from utilmy import text as m
    log(m.pd_text_getcluster )
    m.test_lsh()
      

def test():
    """function test
    Args:
    Returns:
        
    """
    """function test
    Args:
    Returns:
        
    """
    from difflib import SequenceMatcher
    from pandas._testing import assert_series_equal

    list1 = ['dog', 'cat']
    list2 = ['doggy', 'cat']

    cols = ['name','pet_name']
    sample_df = pd.DataFrame(zip(list1, list2), columns=cols)
    original_value = pd_text_similarity(sample_df, cols)['score']

    check_similarity = lambda *x: SequenceMatcher(None, *x[0]).ratio()
    
    output_value = pd.Series(sample_df.apply(lambda x: check_similarity(x[[*cols]]), axis=1), name="score")

    assert_series_equal(original_value, output_value, check_names=False)

    log(pd_text_getcluster )
    test_lsh()
      

def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    print(s, flush=True)


def help():
    """function help
    Args:
    Returns:
        
    """
    ss  = ""
    ss += HELP
    print(ss)


def help_get_codesource(func):
    """ Extract code source from func name"""
    import inspect
    try:
        lines_to_skip = len(func.__doc__.split('\n'))
    except AttributeError:
        lines_to_skip = 0
    lines = inspect.getsourcelines(func)[0]
    return ''.join( lines[lines_to_skip+1:] )


#############################################################################
def test():

  column_name = "question1"
  threshold = 0.7
  num_perm = 10
  num_items = 100000

  url = "https://raw.githubusercontent.com/AlexAdvent/utilmy-data/main/text_question.csv"
  df = pd.read_csv(url)
  print(df.head())

  df1 = pd_text_getcluster(
        df.head(num_items), column_name, threshold, num_perm)
  log(df1.head())

  df2 = pd_text_similarity(df, cols=['question1','question2'])
  matched = df.loc[df['score'] >= 0.8]
  print("match using SequenceMatcher is",matched.shape[0])
  print(matched.head())

  df2 = pd_text_similarity(df, cols=['question1','question2'],algo="rapidfuzz")
  matched = df.loc[df['score'] >= 80]
  print("match using rapidfuzz is",matched.shape[0])

  df2 = pd_text_similarity(df, cols=['question1','question2'],algo="editdistance")
  matched = df.loc[df['score'] >= 80]
  print("match using editdistance is",matched.shape[0])


def test_lsh():
    """function test_lsh
    Args:
    Returns:
        
    """

    ll = ['aa bb cc', 'a b c', 'cc bb cc']
    column_name = "sentence"
    threshold = 0.7
    num_perm = 10
    num_items = 100000

    df = pd.DataFrame(ll, columns=[column_name])
    df1 = pd_text_getcluster(
        df.head(num_items), column_name, threshold, num_perm)
    print(df1)


#############################################################################
def pd_text_hash_create_lsh(df, col, sep=" ", threshold=0.7, num_perm=10, npool=1, chunk = 20000):
    '''
    For each of the entry create a hash function
    '''
    from datasketch import MinHash, MinHashLSH
    lsh = None

    if npool > 1 and len(df) > 20000 :
        from utilmy.parallel import multithread_run
        nchunk  = 1 + len(df) // chunk
        df_list = [  df.iloc[i*chunk:(i+1)*chunk, :] for i in range(0, nchunk ) ]

        # Create LSH
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)   #### Global lsh

        res = multithread_run(pd_text_hash_create_lsh, df_list, npool=1)

        hash_lines = []
        for xi in res :
            hash_lines.extend(xi[0])

        return hash_lines, lsh

    if len(df) < 1 :
        return [],[]

    if lsh is None :
       lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Intialize list
    hash_lines = []
    ll         = df[col].values
    for index, sentence in enumerate(ll):

        # Get tokens of individual elements
        tokens = sentence.split(sep)

        # Create local hash funtion
        v = MinHash(num_perm=num_perm)

        for j in set(tokens):
            v.update(j.encode('utf8'))

        hash_lines.append(v)
        lsh.insert(str(index), v)
    return hash_lines, lsh


def pd_text_getcluster(df:pd.DataFrame, col:str='col', threshold=0.5, num_perm:int=5, npool=1, chunk = 100000):
    '''
    For each of the hash function find a cluster and assign unique id to the dataframe cluster_id
    '''
    # MAster cluster ids
    all_cluster_ids = []
    if len(df)< 1 : return df

    if npool > 1 and len(df) > 200000 :
        from utilmy.parallel import multithread_run
        nchunk  = 1 + len(df) // chunk
        df_list = [  df.iloc[i*chunk:(i+1)*chunk, :] for i in range(0, nchunk ) ]
        res     = multithread_run(pd_text_getcluster, df_list, npool=1)
        df      = pd.concat(res)
        return df

    # REturn from hash list
    hash_lines, lsh = pd_text_hash_create_lsh(
        df, col=col, threshold=threshold, num_perm=num_perm)

    # For each local hash find the cluster ids and assign to the dataframe and return dataframe
    cluster_count = 1
    for ind, i in enumerate(hash_lines):
        if ind in all_cluster_ids:
            continue

        x_duplicate     = lsh.query(i)
        x_duplicate_int = list(map(int, x_duplicate))
        # print(x_duplicate_int)
        df.at[x_duplicate_int, 'cluster_id'] = cluster_count
        cluster_count   += 1
        all_cluster_ids += x_duplicate_int

    return df


def pd_text_similarity(df: pd.DataFrame, cols=[], algo='') -> pd.DataFrame:
    '''
        Return similarities between two columns with 
        python's SequenceMatcher algorithm

        Args:
            df (pd.DataFrame): Pandas Dataframe.
            algo (String)    : rapidfuzz | editdistance 
            cols (list[str]) : List of of columns name (2 columns)

        Returns:
            pd.DataFrame

    '''
    if len(cols) != 2:
        raise Exception("Add two columns")
    for col in cols:
        if col not in df:
            raise Exception(f"Columns not found {col}")
            break

    from difflib import SequenceMatcher
    from rapidfuzz import fuzz
    import editdistance

    def find_similarity(col1, col2):
        if algo == "rapidfuzz":
            similarity_score = fuzz.ratio(col1, col2)
        elif algo == "editdistance":
            similarity_score = editdistance.eval(col1, col2)
        else:
            is_junk = None
            similarity_score = SequenceMatcher(is_junk, col1, col2).ratio()
        return similarity_score

    df['score'] = df.apply(lambda x: find_similarity( x[cols[0]], x[cols[1]]), axis=1)
    return df






###################################################################################################
if __name__ == "__main__":
    import fire ;
    fire.Fire()



