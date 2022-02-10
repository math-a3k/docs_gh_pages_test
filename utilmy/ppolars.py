# -*- coding: utf-8 -*-
HELP= """


"""
import os, sys, time, datetime,inspect, json, yaml, gc, glob, pandas as pd, numpy as np
import polars as pl


###################################################################################
from utilmy.utilmy import log, log2

def help():
    from utilmy import help_create
    ss = help_create("utilmy.ppolars") + HELP
    print(ss)


###################################################################################
def test_all():
    os.makedirs("testdata/ppolars", exist_ok=True)



def test2():
    nmin = 2
    nmax=5000
    # df = pd_create_random(nmax=5000000)
    df = pd.DataFrame()
    df['key'] = np.arange(0, nmax)
    for i in range(0, nmin):
        df[ f'int{i}'] = np.random.randint(0, 100,size=(nmax, ))
        df[ f'flo{i}'] = np.random.rand(1, nmax)[0] 
        df[ f'str{i}'] =  [ ",".join([ str(t) for t in np.random.randint(10000000,999999999,size=(500, )) ] )  for k in range(0,nmax) ]
        print(df.head)
    df.to_parquet('myfile')


    #### Pandas version
    dfp = pd.read_parquet('myfile')
    i=0
    dfp['new1'] = dfp.apply(lambda x :  min( x['str1'].split(","))   , axis=1)
    dfp.groupby([f'int{i}']).agg({'count'})
    dfp.groupby(['key']).apply(lambda x:";".join(x[f'str{i}'].values))
    dfp.groupby([f'flo{i}']).agg({'sum'})
    dfp.to_parquet('myfile.parquet')


    ### POLARS Version 
    df = pl.read_parquet('myfile')
    i=0
    df['new1'] = df.select(["*",  pl.col("str1").apply(lambda x : min(x.split(",")) ).alias("booknew")])['booknew']
    df.groupby(f'int{i}').agg(pl.all().count())
    df.groupby('key').agg(pl.col(f'str{i}')).select([pl.col('key'), pl.col(f'str{i}').arr.join(",")])
    df.groupby([f'flo{i}']).agg(pl.all().sum())
    df.to_parquet('myfile.parquet.polars')




def test_create_parquet():
    nmin = 2
    nmax=5000
    # df = pd_create_random(nmax=5000000)
    df = pd.DataFrame()
    df['key'] = np.arange(0, nmax)
    for i in range(0, nmin):
        df[ f'int{i}'] = np.random.randint(0, 100,size=(nmax, ))
        df[ f'flo{i}'] = np.random.rand(1, nmax)[0] 
        df[ f'str{i}'] =  [ ",".join([ str(t) for t in np.random.randint(10000000,999999999,size=(500, )) ] )  for k in range(0,nmax) ]
    print(df.head)
    df.to_parquet('ztest/myfile.parquet')
    return 'ztest/myfile.parquet'



###################################################################################################
###### Polars #####################################################################################
def pl_split(df,  col='colstr', sep=",",  colnew="colstr_split", ):
    """
      dfp['new1'] = dfp.apply(lambda x :  min( x['str1'].split(","))   , axis=1)
      df['new1'] = df.select(["*",  pl.col("str1").apply(lambda x : min(x.split(",")) ).alias("booknew")])['booknew']

    """
    df[ colnew ] = df.select(["*",  pl.col(col).apply(lambda x : x.split(",") ).alias(colnew )])[colnew ]
    return df



def pl_groupby_join(df,  colgroup="colgroup", col='colstr', sep=",",   ):
    """
      dfp['new1'] = dfp.apply(lambda x :  min( x['str1'].split(","))   , axis=1)
      df['new1']  = df.select(["*",  pl.col("str1").apply(lambda x : min(x.split(",")) ).alias("booknew")])['booknew']

    """
    df.groupby(colgroup ).agg(pl.col(col)).select([pl.col(colgroup ), pl.col(col).arr.join(sep)])
    return df










def pl_to_file(df, filei,  check=0, verbose=True, show='shape',   **kw):
  import os, gc
  from pathlib import Path
  parent = Path(filei).parent
  os.makedirs(parent, exist_ok=True)
  ext  = os.path.splitext(filei)[1]
  if ext == ".parquet" :   df.to_parquet(filei, **kw)
  elif ext in [".csv" ,".txt"] :  df.to_csv(filei, **kw)        
  else :
      log('No Extension, using parquet')
      df.to_parquet(filei + ".parquet", **kw)

  if verbose in [True, 1] :  log(filei)        
  if show == 'shape':        log(df.shape)
  if show in [1, True] :     log(df)
     
  if check in [1, True, "check"] : log('Exist', os.path.isfile(filei))
  #  os_file_check( filei )
  gc.collect()



def pd_cartesian(df1, df2) :
  ### Cartesian preoduct
  import pandas as pd
  col1 =  list(df1.columns)
  col2 =  list(df2.columns)
  df1['xxx'] = 1
  df2['xxx'] = 1
  df3 = pd.merge(df1, df2,on='xxx')[ col1 + col2 ]
  try:
        del df3['xxx']
  except:pass

  return df3


def pd_col_bins(df, col, nbins=5):
  ### Shortcuts for easy bin of numerical values
  import pandas as pd, numpy as np
  assert nbins < 256, 'nbins< 255'
  return pd.qcut(df[col], q=nbins,labels= np.arange(0, nbins, 1)).astype('int8')





def pd_del(df, cols:list):
    ### Delete columns without errors
    for col in cols :
        try:
            del df[col]
        except : pass
    return df




###########################################################################################################
class dict_to_namespace(object):
    #### Dict to namespace
    def __init__(self, d):
        self.__dict__ = d


def to_dict(**kw):
  ## return dict version of the params
  return kw


def to_timeunix(datex="2018-01-16"):
  if isinstance(datex, str)  :
     return int(time.mktime(datetime.datetime.strptime(datex, "%Y-%m-%d").timetuple()) * 1000)

  if isinstance(datex, datetime.datetime)  :
     return int(time.mktime( datex.timetuple()) * 1000)


def to_datetime(x) :
  import pandas as pd
  return pd.to_datetime( str(x) )


def np_list_intersection(l1, l2) :
  return [x for x in l1 if x in l2]


def np_add_remove(set_, to_remove, to_add):
    # a function that removes list of elements and adds an element from a set
    result_temp = set_.copy()
    for element in to_remove:
        result_temp.remove(element)
    result_temp.add(to_add)
    return result_temp


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




