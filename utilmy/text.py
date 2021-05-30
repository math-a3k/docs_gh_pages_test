"""

https://github.com/topics/hypothesis-testing?l=python&o=desc&s=stars

https://pypi.org/project/pysie/#description



"""
import os
import sys
import pandas as pd
import numpy as np
from typing import List


def log(*s):
    print(s)


#############################################################################
#############################################################################

def pd_text_hash_create_lsh(df, col, sep=" ", threshold=0.7, num_perm=10):
    '''
    For each of the entry create a hash function
    '''
    from datasketch import MinHash, MinHashLSH
    # Create LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Intialize list
    hash_lines = []

    ll = df[col].values
    for index, sentence in enumerate(ll):

        # Get tokens of individual elements
        tokens = sentence.split(sep)

        # Create local hash funtion
        v = MinHash(num_perm=num_perm)

        for j in set(tokens):
            v.update(j.encode('utf8'))

        # Append
        hash_lines.append(v)
        lsh.insert(str(index), v)
    return hash_lines, lsh


def pd_text_getcluster(df, col, threshold, num_perm):
    '''
    For each of the hash function find a cluster and assign unique id to the dataframe cluster_id
    '''
    # MAster cluster ids
    all_cluster_ids = []

    # REturn from hash list
    hash_lines, lsh = pd_text_hash_create_lsh(
        df, col=col, threshold=threshold, num_perm=num_perm)

    # For each local hash find the cluster ids and assign to the dataframe and return dataframe
    cluster_count = 1
    for ind, i in enumerate(hash_lines):

        if ind in all_cluster_ids:
            continue

        x_duplicate = lsh.query(i)
        x_duplicate_int = list(map(int, x_duplicate))
        # print(x_duplicate_int)
        df.at[x_duplicate_int, 'cluster_id'] = cluster_count
        cluster_count += 1
        all_cluster_ids += x_duplicate_int

    return df


def pd_similarity(df: pd.DataFrame, cols=[], algo='') -> pd.DataFrame:
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
    df['score'] = df.apply(lambda x: find_similarity(
        x[cols[0]], x[cols[1]]), axis=1)
    return df


def test_lsh():

    ll = ['aa bb cc', 'a b c', 'cc bb cc']
    column_name = "sentence"
    threshold = 0.7
    num_perm = 10
    num_items = 100000

    df = pd.DataFrame(ll, columns=[column_name])
    df1 = pd_text_getcluster(
        df.head(num_items), column_name, threshold, num_perm)
    print(df1)