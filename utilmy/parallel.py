# coding=utf-8
HELP="""
    python parallel.py test0
    #  python parallel.py test2
"""
import itertools, time, multiprocessing, pandas as pd, numpy as np, pickle, gc, os
from multiprocessing.pool import ThreadPool
from threading import Thread

from typing import Callable, Tuple, Union

#################################################################################################
verbose = 0

def log(*s, **kw):  print(*s, flush=True, **kw)
def log2(*s, **kw):  
    if verbose >1 : print(*s, flush=True, **kw)

def help():
    from utilmy import help_create
    ss  = HELP
    ss += help_create("utilmy.parallel")
    print(ss)


#################################################################################################
#################################################################################################
def pd_random(nrows=1000, ncols= 5):
    return pd.DataFrame( np.random.randint(0, 10, size= (nrows, ncols)),  columns= [ str(i) for i in range(ncols) ]   )


def test_fun_sum_inv(group, name=None):         # Inverse cumulative sum
       group["inv_sum"] = group.iloc[::-1]["1"].cumsum()[::-1].shift(-1).fillna(0)
       return group

def test_fun_sum(df_group, name=None):         # Inverse cumulative sum
       df_group['1sum'] = df_group['1'].sum()
       return df_group


# the funtion for test multi process
def test_fun_sum2(list_vars, const=1, const2=1):
    log( list_vars )
    si = 0
    for xi in list_vars :
        log(xi)
        si = si + xi if isinstance(xi, int) else si + sum(xi)
    return si


def test_fun_run(list_vars, const=1, const2=1):
    ### multithread
    log(f'Var: {list_vars[0]}')
    log('Fixed Const: ', const)
    return f"{const*const2} {str(list_vars[0])}"


def test_run_multithread(thread_name, num, string):
    print(f'Var: {thread_name}, {num}, {string}')
    print(f'Start thread: {thread_name}')
    print(f'End thread: {thread_name}')
    return string*2


def test_run_multithread2(thread_name, arg):
    print(f'Var: {thread_name}, {arg}')
    print(f'Start thread: {thread_name}')
    print(f'End thread: {thread_name}')
    return arg

def test_sum(x):
    return  x['0'] + x['1']



def test0():
    df  = pd_random(1*10**5, ncols=3)

    log("\n\n###########  pd_groupby_parallel  #############################################")
    colsgroup = ['0']
    t0 = time.time()
    df1 = df.groupby(colsgroup).apply(lambda dfi : test_fun_sum_inv(dfi ) )
    df1 = df1.sort_values( list(df1.columns))
    log(df1, time.time() - t0)

    t0 = time.time()
    df2 = pd_groupby_parallel(df, colsgroup, fun_apply= test_fun_sum_inv, npool=4 )
    df2 = df2.sort_values( list(df2.columns))
    log( 'pd_groupby_parallel: ' , df1.equals(df2), df2, time.time() - t0)


    log("\n\n###########  pd_groupby_parallel2  ###########################################")
    t0 = time.time()
    df2 = pd_groupby_parallel2(df, colsgroup, fun_apply= test_fun_sum_inv, npool=4 )
    df2 = df2.sort_values( list(df2.columns))
    log( 'pd_groupby_parallel3 : ' , df1.equals(df2), df2, time.time() - t0)


    log("\n\n###########  pd_groupby_parallel3  : Buggy one #################################")
    t0 = time.time()
    # df2 = pd_groupby_parallel3(df, colsgroup, fun_apply= test_fun_sum_inv, npool=4 )   ### Failed due to groupby part
    df2 = df2.sort_values( list(df2.columns))
    log( 'pd_groupby_parallel2 : ' , df1.equals(df2), df2, time.time() - t0)


    log("\n\n###########  pd_apply_parallel  :  ############################################")
    df = df.iloc[:1037,:]
    df1 = df.copy()  ; df2 = df.copy()

    df1['s1'] = df.apply( lambda x : test_sum(x), axis=1)
    df2['s1'] = pd_apply_parallel(df, fun_apply= test_sum, npool=7 )   ### Failed due to groupby part
    df2 = df2.sort_index() ; df1 = df1.sort_index()
    log( 'pd_groupby_parallel2 : ' ,  df1, df2,)
    assert df1.equals(df2), 'unequal pd_apply_parallel'


    log("\n\n########### multiproc_run Sum ############################################")
    npool =3
    input_list = [ [1,1,], [2,2, ], [3,3, ], [4,4,], [5,5, ], [6,6, ], [7,7, ],  ]
    res = multiproc_run(test_fun_sum2, input_list, n_pool= npool )
    log( 'multiproc_run : ' , res,  )
    assert len(res) == npool and  sum(res) == sum([ sum(t) for t in input_list]), 'failed multiproc_run'

    npool= 7
    input_list = [ i for i in range(0, 67) ]
    res = multiproc_run(test_fun_sum2, input_list, n_pool= npool )
    log( 'multiproc_run : ' ,  res)
    assert len(res) == npool and  sum(res) == sum([ t for t in input_list]), 'failed multiproc_run'



    # the list of input will be used for multiproc_run, multithread_run testing
    input_list = [
        [ "path1", "path2", "path2", ],                                                             # String
        (12, 23, 45, 56, ),                                                                         # number
        ([1,2], [3,4], [5,6], ),                                                                    # list number
        ( [1,2, "Hello"], [2,4, "World"], [3,4, "Thread3"], [4,5, "Thread4"], [5,2, "Thread5"], ),  # list number and string
        ("Hello", "World", "test", "path", ),                                                       # string
        ({"test": 1}, ),                                                                            # dict 1 thread
        ({"test": 1}, {"var1": 1, "var2": "string"}, ),                                             # dict multi thread
        [ [ [  "pa_1", "pa_2" ] ], [ [  "pb_1", "pb_2" ] ],  [ [  "pc_1", "pc_2" ] ], ]             #
    ]

    log("\n\n########### multiproc_run ####################################################")
    i = 0
    for input in input_list:
        i += 1
        log(f"\n\n########### multiproc_run with input list: {input}")
        input_fixed={'const': 50, 'const2': i}
        res = multiproc_run(test_fun_run, input, n_pool=len(input), input_fixed=input_fixed)
        log( 'multiproc_run : ' , res)
        log("########### Validation for multiproc_run response")
        for index in range(len(input)):
            # convert to tuple if input type is list
            input_index = input[index]
            # input_index =  (input[index], ) if type(input[index]) is not list else tuple(input[index])
            assert res[index] == test_fun_run([input_index], const=input_fixed['const'], const2=input_fixed['const2']), \
                f"[FAILED], { res[index] }, {input_index}"



    log("\n\n########### multithread_run ####################################################")
    i = 0
    for input in input_list:
        i += 1
        log(f"\n\n########### multithread_run with input list: {input}")
        input_fixed={'const': 50, 'const2': i}
        res = multithread_run(test_fun_run, input, n_pool=len(input), input_fixed=input_fixed)
        log( 'multithread_run : ' , res)
        log("########### Validation for multithread_run response")
        for index in range(len(input)):
            # convert to tuple if input type is list
            input_index = input[index]
            # input_index = (input[index], ) if type(input[index]) is not list else tuple(input[index])
            assert res[index] == test_fun_run([input_index], const=input_fixed['const'], const2=input_fixed['const2']), \
                f"[FAILED], { res[index] }, {input_index}"


    log("\n\n########### multithread_run_list ################################################")
    res = multithread_run_list(
        thread1=(test_run_multithread, ["Thread1", 5, "test"]),
        thread2=(test_run_multithread, ["Thread2", 6, "1234"]),
        thread_another=(test_run_multithread2, ["Thread_diff", "rtyr"]),
        )
    log( 'multithread_run_list : ' , res)



def test_pdreadfile():
   ##### pd_read_file
   # from utilmy import pd_read_file
   import pandas as pd, random
   ncols = 7
   nrows = 100
   ll = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
   # Required for it to be detected in Session's globals()
   global df
   df = pd.DataFrame(ll, columns = [str(i) for i in range(0,ncols)])
   n0 = len(df)
   s0 = df.values.sum()
   os.makedirs("data/parquet/", exist_ok= True)

   ##### m_job , n_pool tests  ##############################
   ncopy = 20
   for i in range(0, ncopy) :
      df.to_csv( f"data/parquet/ppf_{i}.csv.gz",  compression='gzip' , index=False)

   df1 = pd_read_file("data/parquet/ppf*.gz", verbose=True, n_pool= 7 )

   assert len(df1) == ncopy * n0,         f"df1 {len(df1) }, original {n0}"
   assert round(df1.values.sum(), 5) == round(ncopy * s0,5), f"df1 {df1.values.sum()}, original {ncopy*s0}"


   ####################################################
   df.to_csv( "data/parquet/fa0b2.csv.gz",   compression='gzip' , index=False)
   df.to_csv( "data/parquet/fab03.csv.gz",   compression='gzip' , index=False)
   df.to_csv( "data/parquet/fabc04.csv.gz",  compression='gzip' , index=False)
   df.to_csv( "data/parquet/fa0bc05.csv.gz", compression='gzip' , index=False)

   df1 = pd_read_file("data/parquet/fab*.*", verbose=True)
   assert len(df1) == 2 * n0, f"df1 {len(df1) }, original {n0}"


   ##### Stresss n_pool
   df2 = pd_read_file("data/parquet/fab*.*", n_pool=1000 )
   assert len(df2) == 2 * n0, f"df1 {len(df2) }, original {n0}"





#############################################################################################################
def pd_read_file(path_glob="*.pkl", ignore_index=True,  cols=None, verbose=False, nrows=-1, nfile=1000000, concat_sort=True,
                 n_pool=1, npool=None,
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None,
                 fun_apply=None, use_ext=None,   **kw):
    """  Read file in parallel from disk : very Fast
    :param path_glob: list of pattern, or sep by ";"
    :return:
    """
    import glob, gc,  pandas as pd, os
    
    if isinstance(path_glob, pd.DataFrame ) : return path_glob   ### Helpers
    
    n_pool = npool if isinstance(npool, int)  else n_pool ## alias
    def log(*s, **kw):  print(*s, flush=True, **kw)
    readers = {
            ".pkl"     : pd.read_pickle, ".parquet" : pd.read_parquet,
            ".tsv"     : pd.read_csv, ".csv"     : pd.read_csv, ".txt"     : pd.read_csv, ".zip"     : pd.read_csv,
            ".gzip"    : pd.read_csv, ".gz"      : pd.read_csv,
     }

    #### File
    if isinstance(path_glob, list):  path_glob = ";".join(path_glob)
    path_glob = path_glob.split(";")
    file_list = []
    for pi in path_glob :
        if "*" in pi : file_list.extend( sorted( glob.glob(pi) ) )
        else :         file_list.append( pi )

    file_list = sorted(list(set(file_list)))
    file_list = file_list[:nfile]
    if verbose: log(file_list)

    ### TODO : use with kewyword arguments ###############
    def fun_async(filei):
            ext  = os.path.splitext(filei)[1]
            if ext is None or ext == '': ext ='.parquet'

            pd_reader_obj = readers.get(ext, None)
            # dfi = pd_reader_obj(filei)
            try :
               dfi = pd_reader_obj(filei, **kw)
            except Exception as e:
               log('Error', filei, e)
               return pd.DataFrame()

            # if dtype_reduce is not None:    dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
            if col_filter is not None :       dfi = dfi[ dfi[col_filter] == col_filter_val ]
            if cols is not None :             dfi = dfi[cols]
            if nrows > 0        :             dfi = dfi.iloc[:nrows,:]
            if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates)
            if fun_apply is not None  :       dfi = dfi.apply(lambda  x : fun_apply(x), axis=1)
            return dfi

            
            
    ### Parallel run #################################
    import concurrent.futures
    dfall  = pd.DataFrame()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_pool) as executor:
        futures = []
        for i,fi in enumerate(file_list) :
            if verbose : log("file ", i, end=",")
            futures.append( executor.submit(fun_async, fi ))

        for future in concurrent.futures.as_completed(futures):
            try:
                dfi   = future.result()
                dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
                del dfi; gc.collect()
            except Exception as e:
                log('error', e)
    return dfall



def pd_read_file2(path_glob="*.pkl", ignore_index=True,  cols=None, verbose=False, nrows=-1, nfile=1000000, concat_sort=True, n_pool=1, npool=None,
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None, fun_apply=None, use_ext=None,  **kw):
    """  Read file in parallel from disk : very Fast
    :param path_glob: list of pattern, or sep by ";"
    :return:
    """
    import glob, gc,  pandas as pd, os
    if isinstance(path_glob, pd.DataFrame ) : return path_glob   ### Helpers    
    n_pool = npool if isinstance(npool, int)  else n_pool ## alias
    
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
    file_list = file_list[:nfile]
    n_file    = len(file_list)
    if verbose: log(file_list)

    #### Pool count  ###############################################
    if n_pool < 1 :  n_pool = 1
    if n_file <= 0:  m_job  = 0
    elif n_file <= 2:
      m_job  = n_file
      n_pool = 1
    else  :
      m_job  = 1 + n_file // n_pool  if n_file >= 3 else 1
    if verbose : log(n_file,  n_file // n_pool )

    ### TODO : use with kewyword arguments
    pd_reader_obj2 = None

    def fun_async(filei):
        ext  = os.path.splitext(filei)[1]
        if ext is None or ext == '': ext ='.parquet'

        pd_reader_obj = readers.get(ext, None)
        try :
          dfi = pd_reader_obj(filei)
        except Exception as e :
          log(e)

        # if dtype_reduce is not None:    dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
        if col_filter is not None :       dfi = dfi[ dfi[col_filter] == col_filter_val ]
        if cols is not None :             dfi = dfi[cols]
        if nrows > 0        :             dfi = dfi.iloc[:nrows,:]
        if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates)
        if fun_apply is not None  :       dfi = dfi.apply(lambda  x : fun_apply(x), axis=1)
        return dfi

    pool   = ThreadPool(processes=n_pool)
    dfall  = pd.DataFrame()
    for j in range(0, m_job ) :
        if verbose : log("Pool", j, end=",")
        job_list = []
        for i in range(n_pool):
           if n_pool*j + i >= n_file  : break

           filei         = file_list[n_pool*j + i]
           job_list.append( pool.apply_async(fun_async, (filei, )))
           if verbose : log(j, filei)

        for i in range(n_pool):
            try :
                  if i >= len(job_list): break
                  dfi   = job_list[ i].get()
                  dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
                  #log("Len", n_pool*j + i, len(dfall))
                  del dfi; gc.collect()
            except Exception as e:
                log('error', filei, e)
          

    pool.close() ; pool.join() ;  pool = None
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


def pd_groupby_parallel(df, colsgroup=None, fun_apply=None, n_pool=4, npool=None):
    """
    Use of multi-thread on group by apply when order is not important
    """
    n_pool = npool if isinstance(npool, int)  else n_pool ## alias
    import pandas as pd, concurrent.futures

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


def pd_apply_parallel(df, fun_apply=None, npool=5, verbose=True ):
    """ Pandas parallel apply
    """
    import pandas as pd, numpy as np, time, gc

    def f2(df):
        return df.apply(lambda x : fun_apply(x), axis=1)

    if npool == 1 : return f2(df)

    #### Pool execute ###################################
    import concurrent.futures
    size = int(len(df) // npool)

    with concurrent.futures.ThreadPoolExecutor(max_workers=npool) as executor:
        futures = []
        for i in range(npool):
            i2  = 3*(i + 2) if i == npool - 1 else i + 1
            dfi = df.iloc[i*size:(i2*size), :]
            futures.append( executor.submit(f2, dfi,) )
            if verbose: log('start', i, dfi.shape)
            del dfi

        dfall = None
        for future in concurrent.futures.as_completed(futures):
            dfi = future.result()
            dfall = pd.concat((dfall, dfi)) if dfall is not None else dfi
            del dfi
            print(i, 'done' , end="," )

    return dfall


############################################################################################################
def multiproc_run(fun_async, input_list: list, n_pool=5, start_delay=0.1, verbose=True, input_fixed:dict=None, npool=None, **kw):
    """  Multiprocessing execute
    input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
    def fun_async(xlist):
      for x in xlist :
            download.upload(x[0], x[1])
          def f(i, n):
       return i * i + 2*n
    ..
     from itertools import repeat
     N = 10000
     from pathos.pools import ProcessPool as Pool
     pool = Pool()
     ans = pool.map(f, xrange(1000), repeat(20))
     ans[:10]
    [40, 41, 44, 49, 56, 65, 76, 89, 104, 121]
     # this also works
     ans = pool.map(lambda x: f(x, 20), xrange(1000))
     ans[:10]
    [40, 41, 44, 49, 56, 65, 76, 89, 104, 121]
    input_fixed = {'const': 555}
    """
    import time, functools
    n_pool = npool if isinstance(npool, int)  else n_pool ## alias
    #### Input xi #######################################
    #if not isinstance(input_list[0], list ) and not isinstance(input_list[0], tuple ) :
    #     input_list = [  (t,) for t in input_list]  ## Must be a list of list

    if len(input_list) < 1 : return []

    if input_fixed is not None:  #### Fixed keywword variable
        fun_async = functools.partial(fun_async, **input_fixed)

    xi_list = [[] for t in range(n_pool)]
    for i, xi in enumerate(input_list):
        jj = i % n_pool
        xi_list[jj].append( xi )  ### xi is already a tuple

    if verbose:
        for j in range(len(xi_list)):
            log('proc ', j, len(xi_list[j]))
        # time.sleep(6)

    #### Pool execute ###############################################
    import multiprocessing as mp
    pool = mp.Pool(processes=n_pool)
    # pool     = mp.pool.ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
        time.sleep(start_delay)
        log('starts', i)
        job_list.append(pool.apply_async(fun_async, (xi_list[i],) ))
        if verbose: log(i, xi_list[i])

    res_list = []
    for i in range(len(job_list)):
        res_list.append(job_list[i].get())
        log(i, 'job finished')

    pool.close(); pool.join(); pool = None
    log('n_processed', len(res_list))
    return res_list


def multithread_run(fun_async, input_list: list, n_pool=5, start_delay=0.1, verbose=True, input_fixed:dict=None, npool=None, **kw):
    """  input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
    def fun_async(xlist):
      for x in xlist :
            hdfs.upload(x[0], x[1])
    input_fixed = {'const_var' : 1 }
    """
    import time, functools
    n_pool = npool if isinstance(npool, int)  else n_pool ## alias

    #### Input xi #######################################
    #if not isinstance(input_list[0], list ) and not isinstance(input_list[0], tuple ) :
    #     input_list = [  (t,) for t in input_list]  ## Must be a list of lis
    if len(input_list) < 1 : return []

    if input_fixed is not None:
        fun_async = functools.partial(fun_async, **input_fixed)

    #### Input xi #######################################
    xi_list = [[] for t in range(n_pool)]
    for i, xi in enumerate(input_list):
        jj = i % n_pool
        xi_list[jj].append( xi )  ### xi is already a tuple

    if verbose:
        for j in range(len(xi_list)):
            log('thread ', j, len(xi_list[j]))
        # time.sleep(6)

    #### Pool execute ###################################
    import multiprocessing as mp
    # pool     = multiprocessing.Pool(processes=3)
    pool = mp.pool.ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
        time.sleep(start_delay)
        log('starts', i)
        job_list.append(pool.apply_async(fun_async, (xi_list[i],) ))
        if verbose: log(i, xi_list[i])

    res_list = []
    for i in range(len(job_list)):
        res_list.append(job_list[i].get())
        log(i, 'job finished')

    pool.close(); pool.join(); pool = None
    log('n_processed', len(res_list))
    return res_list


def multiproc_tochunk(flist, npool=2 ):
    ll = []
    chunk = len(flist) // npool
    for i in range( npool ) :
         i2 = i+1 if i < npool-1 else 3*(i+1)
         ll.append( flist[i*chunk:i2*chunk] )
    log(len(ll), str(ll)[:100])
    return ll



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
        # print(thread)
        t = ThreadWithResult(target=thread[0], args=thread[1])
        list_of_threads.append(t)

    for thread in list_of_threads:
        thread.start()

    results = []
    for thread, keys in zip(list_of_threads, kwargs.keys()):
        thread.join()
        results.append((keys, thread.result))

    return results


###############################################################################################################




############################################################################################################
if __name__ == '__main__':
    import fire; fire.Fire()
    ### python parallel.py test1














def z_pd_read_file3(path_glob="*.pkl", ignore_index=True,  cols=None, verbose=False, nrows=-1, concat_sort=True, n_pool=1, npool=None,
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None,  **kw):
    """  Read file in parallel from disk : very Fast
    :param path_glob: list of pattern, or sep by ";"
    :return:
    """
    import glob, gc,  pandas as pd, os
    n_pool = npool if isinstance(npool, int)  else n_pool ## alias
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

    #### Pool count  ###############################################
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



def zz_pd_read_file3(path_glob="*.pkl", ignore_index=True,  cols=None,  nrows=-1, concat_sort=True, n_pool=1, npool=None,
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None,
                 fun_apply=None, max_file=-1, #### apply function for each sub
                 verbose=False,
                 **kw):
    """  Read file in parallel from disk : very Fast
    :param path_glob: list of pattern, or sep by ";"
    :return:
    """
    import glob, gc,  pandas as pd, os, time
    n_pool = npool if npool is not None  else n_pool ## alias
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

            # if dtype_reduce is not None:    dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
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




def zz_pd_groupby_parallel5(df, colsgroup=None, fun_apply=None, npool=5, verbose=False, **kw ):
    """ Pandas parallel groupby apply
    """
    import pandas as pd, numpy as np, time, gc

    def f2(df_list):
        dfiall = pd.DataFrame()
        for dfi in df_list:
            dfi    = dfi.apply( lambda dfi : fun_apply(dfi) )
            dfiall = pd.concat((dfiall, dfi))
            del dfi;  gc.collect()
        return dfiall

    #### Pool execute #################################################
    import multiprocessing as mp
    # pool     = multiprocessing.Pool(processes=npool)
    pool = mp.pool.ThreadPool(processes=npool)

    ### Need to get the groupby splits
    dfg        = df.groupby(colsgroup)
    input_list = [[]*1]*npool
    for i, (name, dfi) in enumerate(dfg):
        input_list[i % npool].append(dfi)


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



def ztest1():
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



def ztest2():
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
