# -*- coding: utf-8 -*-
HELP= """ Utils for sparse matrix creation


"""
import os, sys, time, datetime,inspect, json, yaml, gc, glob, pandas as pd, numpy as np
from utilmy.parallel import pd_read_file, pd_read_file2


###################################################################################
from utilmy.utilmy import log, log2

def help():
    from utilmy import help_create
    ss = help_create("utilmy.ppandas") + HELP
    print(ss)

####################################################################################
verbose = 0

def log(*s, **kw):  print(*s, flush=True, **kw)
def log2(*s, **kw):  
    if verbose >1 : print(*s, flush=True, **kw)

def help():
    from utilmy import help_create
    print(HELP + help_create("utilmy.tabular.util_sparse"))



###################################################################################
def test_all():
    test1()
 

def test1():
    xdf, nExpectedOnes, genreColNames = test_create_fake_df()
    X = pd_historylist_to_csr(xdf, colslist=genreColNames)
    print('Sparse matrix shape:', X.shape)
    print('Expected no. of Ones: ', nExpectedOnes )
    print('No. of Ones in the Matrix: ', np.count_nonzero(X == 1) )
    assert np.count_nonzero(X == 1) == nExpectedOnes, "Invalid CSR matrix"
    print('Sparse matrix verified')    



def test_create_fake_df():
    """ Creates a fake dataframe:
        :return: xdf: pd.DataFrame
            userid: int
            genreCol: string: "4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545"
        :return:nExpectedOnes = numUsers * numSubGenre * nlist * nGenreCols
    """
    numUsers = 5
    numSubGenre = 3
    nlist = 4
    nGenreCols = 2
    nExpectedOnes = numUsers * numSubGenre * nlist * nGenreCols

    # Create fake user ids
    userid = [i for i in range(numUsers)]
    
    # Create a fake list of genres (one person)
    genreList = '/'.join([str(x) for x in np.random.randint(10000,9999999,numSubGenre)])
    for i in range(nlist-1):
        genreList += ', ' + '/'.join([str(x) for x in np.random.randint(10000,9999999,numSubGenre)])
    
    # Populate a dataframe with fake data
    df = pd.DataFrame()
    df['userid']  = userid
    df['genreCol1'] = genreList
    df['genreCol2'] = genreList
    
    return (df, nExpectedOnes, ['genreCol1', 'genreCol2'])


###################################################################################################
######  ###########################################################################################
def pd_historylist_to_csr(df:pd.DataFrame, colslist:list=None, hashSize:int=5000, dtype=np.float32):
    """ Creates Sparse matrix of dimensions:
         ncol: hashsize * (nlist1 + nlist2 + ....)    X    nrows: nUserID

         xdf:  pd.DataFrame
               genreCol: string: "4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545"

         colist:   list of column names containing history list
         hashSize: size of hash space
         return X: scipy.sparse.coo_matrix
    """
    import mmh3
    from scipy.sparse import coo_matrix 

    # Extract genreCols as df
    #df = xdf.loc[:, xdf.columns != 'userid']
    df = df[colslist]
    dfColsIdx = list(range(len(df.columns)))

    # No. cols for sparse matrix X
    nlist = []
    for genreCol in  colslist:
        nlist.append( len(df[genreCol][0].split(',')) )
    ncols = hashSize * ( sum(nlist) )

    # No. rows for sparse matrix X
    nrows = len(df)

    # Create zeros sparse matrix
    X = coo_matrix((nrows, ncols), dtype=dtype).toarray()

    for idx in range(len(df)):
        bucket = 0
        for colIdx in dfColsIdx:
            genreList = [x.strip() for x in df[colslist].to_numpy()[idx, colIdx].split(',')]
            for genre in genreList:
                for subgenre in genre.split('/'):
                    subgenre = subgenre.strip()
                    colid = mmh3.hash(subgenre, 42, signed=False) % hashSize
                    X[idx][colid+bucket] = 1
                bucket += hashSize
            
    return X


#####################################################################################################
def to_float(x):
    try :
        return float(x)
    except :
        return float("NaN")


def to_int(x):
    try :
        return int(x)
    except :
        return float("NaN")


def is_int(x):
    try :
        int(x)
        return True
    except :
        return False    

def is_float(x):
    try :
        float(x)
        return True
    except :
        return False   




###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()





