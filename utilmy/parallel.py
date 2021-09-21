# coding=utf-8
HELP="""
    python parallel.py test0



    #  python parallel.py test2




"""
import itertools, time, multiprocessing, pandas as pd, numpy as np, pickle, gc
from multiprocessing.pool import ThreadPool
from threading import Thread

from typing import Callable, Tuple, Union

#################################################################################################
def log(*s):
    print(*s, flush=True)


def help():
    ss  = ""
    ss += HELP
    print(ss)





#################################################################################################
def pd_random(nrows=1000, ncols= 5):
    return pd.DataFrame( np.random.randint(0, 10, size= (nrows, ncols)),  columns= [ str(i) for i in range(ncols) ]   )



def test_fun_sum_inv(group, name=None):         # Inverse cumulative sum
       group["inv_sum"] = group.iloc[::-1]["1"].cumsum()[::-1].shift(-1).fillna(0)
       return group

def test_fun_sum(df_group, name=None):         # Inverse cumulative sum
       df_group['1sum'] = df_group['1'].sum()
       return df_group


def test0():

    df  = pd_random(1*10**6, ncols=3)

    ###########  pd_groupby_parallel  ######################################
    colsgroup = ['0']
    t0 = time.time()
    df1 = df.groupby(colsgroup).apply(lambda dfi : test_fun_sum_inv(dfi ) )
    df1 = df1.sort_values( list(df1.columns))
    log(df1, time.time() - t0)

    t0 = time.time()
    df2 = pd_groupby_parallel(df, colsgroup, fun_apply= test_fun_sum_inv, npool=4 )
    df2 = df2.sort_values( list(df2.columns))
    log(df2, time.time() - t0)
    log( 'pd_groupby_parallel: ' , df1.equals(df2))


    ###########  pd_groupby_parallel3  ###########################################
    t0 = time.time()
    df2 = pd_groupby_parallel2(df, colsgroup, fun_apply= test_fun_sum_inv, npool=4 )
    df2 = df2.sort_values( list(df2.columns))
    log(df2, time.time() - t0)
    log( 'pd_groupby_parallel3 : ' , df1.equals(df2))


    ###########  pd_groupby_parallel2  : Buggy one ###############################
    t0 = time.time()
    df2 = pd_groupby_parallel3(df, colsgroup, fun_apply= test_fun_sum_inv, npool=4 )
    df2 = df2.sort_values( list(df2.columns))
    log(df2, time.time() - t0)
    log( 'pd_groupby_parallel2 : ' , df1.equals(df2))




def test1():
    def fun_async(xlist):
        list = []
        for x in xlist:
            stdr = ""
            for y in x:
                stdr += y
            list.append(stdr)
        return list

    def apply_func(x):
       return x ** 2

    #### multithread_run
    li_of_tuples = [("x", "y", "z"),("y", "z", "p"),("yw", "zs", "psd"),("yd", "zf", "pf")]
    res =  multithread_run(fun_async, li_of_tuples, npool=2, start_delay=0.1, verbose=True)
    log([["xyz", "ywzspsd"], ["yzp", "ydzfpf"]]== res )


    #### multiproc_run : Pickle error
    #li_of_tuples = [("x", "y", "z"),("y", "z", "p"),("yw", "zs", "psd"),("yd", "zf", "pf"),]
    #res = multiproc_run(fun_async, li_of_tuples, npool=2, start_delay=0.1, verbose=True)
    #log( res == [["xyz"], ["yzp"],
    #["ywzspsd"], ["ydzfpf"], []])


    #### pd_groupby_parallel
    df = pd.DataFrame(data={'result':[5, 8, 1, 7, 0, 3, 2, 9, 4, 6],
                            'user_id':[1, 1, 2, 3, 4, 4, 5, 8, 9, 9],
                            'value'  :[27, 14, 26, 19, 28, 9, 11, 1, 26, 18],'data_chunk':[1, 1, 2, 3, 4, 4, 5, 8, 9, 9]})
    expected_df = df.copy()
    expected_df["inv_sum"] = [14.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 18.0, 0.0]
    result = pd_groupby_parallel(df.groupby("user_id"), fun_apply=test_fun_apply, npool=5)
    log(expected_df.equals(result))


    ### pd_apply_parallel2
    df = pd.DataFrame({"A": [0, 1, 2, 3, 4],   "B": [100, 200, 300, 400, 500],})
    expected_df = pd.DataFrame({"A": [0, 1, 4, 9, 16], "B": [10000, 40000, 90000, 160000, 250000]})
    result = pd_groupby_parallel2(df=df, colsgroup=["A" "B"], fun_apply=apply_func, npool=4)
    log(expected_df.equals(result))



def test2():
    from multiprocessing import freeze_support
    def addition(x,y):
        return x+y

    def addition1(x):
        return x

    def fun_async(xlist):
        s = 0
        for x in xlist:
            s = x[0]+x[1]
        return s

    def test_print(x):
        print(x)

    freeze_support()
    log("pd_groupby_parallel")
    #s   = pickle.dumps(addition)
    #f   = pickle.loads(s)
    df  = pd.DataFrame({'A': [0, 1], 'B': [100, 200]})
    res = pd_groupby_parallel(df.groupby(df.index), addition )
    log(res)

    log("pd_groupby_parallel2")
    #s   = pickle.dumps(addition1)
    # f   = pickle.loads(s)
    df  = pd.DataFrame({'A': [0, 1], 'B': [100, 200]})

    res = pd_groupby_parallel2(df,['A'], addition )
    log(res)


    log("pd_apply_parallel")
    res = pd_apply_parallel(df,['A'], addition)
    log(res)


    list = [(1,2,3), (1,2,3)]
    log("multiproc_run")
    # multiproc_run(fun_async,list)

    log("multithread_run")
    multithread_run(fun_async,list)

    log("multithread_run_list")
    multithread_run_list(function1=(test_print, ("some text",)),
                             function2=(test_print, ("bbbbb",)),
                             function3=(test_print, ("ccccc",)))




########################################################################################################
def pd_read_file2(path_glob="*.pkl", ignore_index=True,  cols=None,  nrows=-1, concat_sort=True, n_pool=1, 
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None, 
                 fun_apply=None,npool=1, max_file=-1, #### apply function for each sub
                 verbose=False,
                 **kw):
    """  Read file in parallel from disk : very Fast
    :param path_glob: list of pattern, or sep by ";"
    :return:
    """
    import glob, gc,  pandas as pd, os, time
    n_pool = npool ## alias
    def log(*s, **kw):  print(*s, flush=True, **kw)
    readers = {
          ".pkl"     : pd.read_pickle,
          ".parquet" : pd.read_parquet,
          ".tsv"     : pd.read_csv, ".csv"     : pd.read_csv, ".txt"     : pd.read_csv, ".zip"     : pd.read_csv, ".gzip"    : pd.read_csv, ".gz"      : pd.read_csv,
          # ".orc"     : pd.read_orc,
    }

    #### File  ###########################################################
    if isinstance(path_glob, list):  path_glob = ";".join(path_glob)
    path_glob  = path_glob.split(";")
    file_list  = []
    for pi in path_glob :
      if "*" in pi :  file_list.extend( sorted( glob.glob(pi) ) )
      else :          file_list.append( pi )

    file_list = sorted(list(set(file_list)))
    file_list = file_list if max_file == -1 else file_list[:max_file]
    n_file    = len(file_list)
    if verbose: log(file_list)

    #######################################################################
    def fun_async(filei):
        dfall = pd.DataFrame()
        for filei in file_list :
            if verbose: log(filei)
            ext  = os.path.splitext(filei)[1]
            if ext == None or ext == '': ext ='.parquet'

            pd_reader_obj = readers.get(ext, None)
            if pd_reader_obj == None: continue

            dfi = pd_reader_obj(filei)

            # if dtype_reduce is not None:      dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
            if col_filter is not None :       dfi = dfi[ dfi[col_filter] == col_filter_val ]
            if cols is not None :             dfi = dfi[cols]
            if nrows > 0        :             dfi = dfi.iloc[:nrows,:]
            if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates)
            if fun_apply is not None  :       dfi = dfi.apply(lambda  x : fun_apply(x), axis=1)

            dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
        return dfall

    #### Input xi #######################################
    xi_list = [ []  for t in range(n_pool) ]
    for i, xi in enumerate(file_list) :
        jj = i % n_pool
        xi_list[jj].append( tuple(xi) )
    
    if verbose :
        for j in range( len(xi_list) ):
            log('thread ', j, len(xi_list[j]))
        time.sleep(6)    
        
    #### Pool execute ###################################
    from multiprocessing.pool import ThreadPool
    # pool     = multiprocessing.Pool(processes=3)  
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         if verbose : log('starts', i)   
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))

    dfall  = pd.DataFrame()
    for i in range(n_pool):
        if i >= len(job_list): break
        dfi = job_list[ i].get() 
        dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()
        log(i, 'thread finished')

    pool.terminate() ; pool.join()  ; pool = None
    log( 'shape', dfall.shape )
    return dfall 


def pd_read_file(path_glob="*.pkl", ignore_index=True,  cols=None, verbose=False, nrows=-1, concat_sort=True, n_pool=1, 
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None,  **kw):
  """  Read file in parallel from disk : very Fast
  :param path_glob: list of pattern, or sep by ";"
  :return:
  """
  import glob, gc,  pandas as pd, os
  def log(*s, **kw):
      print(*s, flush=True, **kw)
  readers = {
          ".pkl"     : pd.read_pickle, ".parquet" : pd.read_parquet,
          ".tsv"     : pd.read_csv, ".csv"     : pd.read_csv, ".txt"     : pd.read_csv, ".zip"     : pd.read_csv,
          ".gzip"    : pd.read_csv, ".gz"      : pd.read_csv,
   }
  from multiprocessing.pool import ThreadPool

  #### File
  if isinstance(path_glob, list):  path_glob = ";".join(path_glob)
  path_glob  = path_glob.split(";")
  file_list = []
  for pi in path_glob :
      if "*" in pi :
        file_list.extend( sorted( glob.glob(pi) ) )
      else :
        file_list.append( pi )


  file_list = sorted(list(set(file_list)))
  n_file    = len(file_list)
  if verbose: log(file_list)

  #### Pool count
  if n_pool < 1 :  n_pool = 1
  if n_file <= 0:  m_job  = 0
  elif n_file <= 2:
    m_job  = n_file
    n_pool = 1
  else  :
    m_job  = 1 + n_file // n_pool  if n_file >= 3 else 1
  if verbose : log(n_file,  n_file // n_pool )

  pool   = ThreadPool(processes=n_pool)
  dfall  = pd.DataFrame()
  for j in range(0, m_job ) :
      if verbose : log("Pool", j, end=",")
      job_list = []
      for i in range(n_pool):
         if n_pool*j + i >= n_file  : break
         filei         = file_list[n_pool*j + i]
         ext           = os.path.splitext(filei)[1]
         if ext == None or ext == '':
           continue

         pd_reader_obj = readers[ext]
         if pd_reader_obj == None:
           continue

         ### TODO : use with kewyword arguments
         job_list.append( pool.apply_async(pd_reader_obj, (filei, )))
         if verbose : log(j, filei)

      for i in range(n_pool):
        if i >= len(job_list): break
        dfi   = job_list[ i].get()

        # if dtype_reduce is not None: dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
        if col_filter is not None :  dfi = dfi[ dfi[col_filter] == col_filter_val ]
        if cols is not None :        dfi = dfi[cols]
        if nrows > 0        :        dfi = dfi.iloc[:nrows,:]
        if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates)
        gc.collect()

        dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()
                
  pool.terminate()
  pool.join()  
  pool = None          
  if m_job>0 and verbose : log(n_file, j * n_file//n_pool )
  return dfall







############################################################################################################
def pd_groupby_parallel2(df, colsgroup=None, fun_apply=None,
                        npool: int = 1, **kw,
                        ):
    """Performs a Pandas groupby operation in parallel.
    pd.core.groupby.DataFrameGroupBy
    Example usage:
        import pandas as pd
        df = pd.DataFrame({'A': [0, 1], 'B': [100, 200]})
        df.groupby(df.groupby('A'), lambda row: row['B'].sum())
    Authors: Tamas Nagy and Douglas Myers-Turnbull
    """
    import pandas as pd
    from functools import partial

    groupby_df = df.groupby(colsgroup)

    start = time.time()
    npool = int(multiprocessing.cpu_count()) - 1
    log("\nUsing {} CPUs in parallel...".format(npool))
    with multiprocessing.Pool(npool) as pool:
        queue = multiprocessing.Manager().Queue()
        result = pool.starmap_async(fun_apply, [(group, name) for name, group in groupby_df])
        cycler = itertools.cycle('\|/―')
        while not result.ready():
            log("Percent complete: {:.0%} {}".format(queue.qsize() / len(groupby_df), next(cycler)))
            time.sleep(0.4)
        got = result.get()
    # log("\nProcessed {} rows in {:.1f}s".format(len(groupby_df), time.time() - start))
    return pd.concat(got)



def pd_groupby_parallel(df, colsgroup=None, fun_apply=None, npool=4):
    """
    Use of multi-thread on group by apply when order is not important

    """
    import pandas as pd
    import concurrent.futures

    dfGrouped = df.groupby(colsgroup)


    with concurrent.futures.ThreadPoolExecutor(max_workers=npool) as executor:
        futures = []
        for name, group in dfGrouped:
            futures.append(executor.submit(fun_apply, group))

        del dfGrouped; gc.collect()

        df_out = pd.DataFrame()
        for future in concurrent.futures.as_completed(futures):
            dfr    = future.result()
            df_out = pd.concat(( df_out, dfr ))
            del dfr; gc.collect()

    return df_out



def pd_groupby_parallel3(df, colsgroup=None, fun_apply=None, npool=5, verbose=False, **kw ):
    """ Pandas parallel groupby apply

    """
    import pandas as pd, numpy as np, time, gc

    def f2(df_list):
        dfiall = None
        for dfi in df_list:
            dfi    = dfi.apply( lambda dfi : fun_apply(dfi) )
            dfiall = pd.concat((dfiall, dfi)) if dfiall is None else dfi
            del dfi;  gc.collect()
        return dfiall

    #### Pool execute #################################################
    import multiprocessing as mp
    # pool     = multiprocessing.Pool(processes=npool)
    pool = mp.pool.ThreadPool(processes=npool)

    
    ### Need to get the groupby splits
    dfg        = df.groupby(colsgroup)  
    input_list = [[]*1]*npool
    for i, dfi in enumerate(dfg):
        input_list[i % npool].append(dfg)


    job_list = []
    for i, inputi_list in enumerate(input_list):
        if verbose: log(i, len(inputi_list))
        job_list.append(pool.apply_async(f2, (inputi_list,)))

    ##### Aggregate results ##########################################
    dfall = None
    for i in range(len(job_list)):
        dfi   = job_list[i].get()
        dfall = pd.concat((dfall, dfi)) if dfall is not None else dfi
        del dfi ; gc.collect()
        if verbose : log(i, 'job finished')

    pool.terminate(); pool.join(); pool = None
    return dfall















def pd_apply_parallel(df, fun_apply=None, npool=5, verbose=True ):
    """ Pandas parallel apply

    """
    import pandas as pd, numpy as np, time, gc

    def f2(df):
        return df.apply(fun_apply, axis=1)

    size = int(len(df) // npool)

    #### Pool execute ###################################
    import multiprocessing as mp
    # pool     = multiprocessing.Pool(processes=npool)
    pool = mp.pool.ThreadPool(processes=npool)
    job_list = []

    for i in range(npool):
        i2  = 3*(i + 2) if i == npool - 1 else i + 1
        dfi = df.iloc[i * size:(i2 * size), :]
        job_list.append( pool.apply_async(f2, (dfi,)) )
        if verbose: log('start', i, dfi.shape)

    dfall = None
    for i in range(npool):
        if i >= len(job_list): break
        dfi = job_list[i].get()
        dfall = pd.concat((dfall, dfi)) if dfall is not None else dfi
        del dfi
        log(i, 'job finished')

    pool.terminate();
    pool.join();
    pool = None
    return dfall


def multiproc_run(fun_async, input_list: list, npool=5, start_delay=0.1, verbose=True, **kw):
    """  Multiprocessing execute
    input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
    def fun_async(xlist):
      for x in xlist :
            download.upload(x[0], x[1])
    """
    import time
    #### Input xi #######################################
    xi_list = [[] for t in range(npool)]
    for i, xi in enumerate(input_list):
        jj = i % npool
        xi_list[jj].append(tuple(xi))

    if verbose:
        for j in range(len(xi_list)):
            log('thread ', j, len(xi_list[j]))
        time.sleep(6)

    #### Pool execute ###################################
    import multiprocessing as mp
    pool = multiprocessing.Pool(processes=npool)
    # pool     = mp.pool.ThreadPool(processes=n_pool)
    job_list = []
    for i in range(npool):
        time.sleep(start_delay)
        log('starts', i)
        job_list.append(pool.apply_async(fun_async, (xi_list[i],)))
        if verbose: log(i, xi_list[i])

    res_list = []
    for i in range(npool):
        if i >= len(job_list): break
        res_list.append(job_list[i].get())
        log(i, 'job finished')

    pool.terminate(); pool.join(); pool = None
    log('n_processed', len(res_list))
    return res_list


def multithread_run(fun_async, input_list: list, n_pool=5, start_delay=0.1, verbose=True, **kw):
    """  input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
    def fun_async(xlist):
      for x in xlist :
            hdfs.upload(x[0], x[1])
    """
    import time
    #### Input xi #######################################
    xi_list = [[] for t in range(n_pool)]
    for i, xi in enumerate(input_list):
        jj = i % n_pool
        xi_list[jj].append(tuple(xi))

    if verbose:
        for j in range(len(xi_list)):
            log('thread ', j, len(xi_list[j]))
        time.sleep(6)

    #### Pool execute ###################################
    import multiprocessing as mp
    # pool     = multiprocessing.Pool(processes=3)
    pool = mp.pool.ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
        time.sleep(start_delay)
        log('starts', i)
        job_list.append(pool.apply_async(fun_async, (xi_list[i],)))
        if verbose: log(i, xi_list[i])

    res_list = []
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append(job_list[i].get())
        log(i, 'job finished')

    pool.terminate(); pool.join(); pool = None
    log('n_processed', len(res_list))
    return res_list


def multithread_run_list(**kwargs):
    """ Creating n number of threads:  1 thread per function,    starting them and waiting for their subsequent completion
    os_multithread(function1=(test_print, ("some text",)),
                          function2=(test_print, ("bbbbb",)),
                          function3=(test_print, ("ccccc",)))
    """

    class ThreadWithResult(Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
            def function():
                self.result = target(*args, **kwargs)

            super().__init__(group=group, target=function, name=name, daemon=daemon)

    list_of_threads = []
    for thread in kwargs.values():
        t = ThreadWithResult(target=thread[0], args=thread[1])
        list_of_threads.append(t)

    for thread in list_of_threads:
        thread.start()

    results = []
    for thread, keys in zip(list_of_threads, kwargs.keys()):
        thread.join()
        results.append((keys, thread.result))

    return results




############################################################################################################
if __name__ == '__main__':
    import fire; fire.Fire()
    ### python parallel.py test1
    


