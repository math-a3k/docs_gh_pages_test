# coding=utf-8
from multiprocessing.pool import ThreadPool
from threading import Thread
import itertools
import time
import multiprocessing
from typing import Callable, Tuple, Union

#################################################################################################
from utilmy import log

# def log(*s): print(*s, flush=True)

def fun_async(xlist):
        list = []
        for x in xlist:
            stdr = ""
            for y in x:
                stdr += y
            list.append(stdr)
        return list

def group_function(name, group):
    # Inverse cumulative sum
    group["inv_sum"] = group.iloc[::-1]["value"].cumsum()[::-1].shift(-1).fillna(0)
    return group

def apply_func(x):
    return x ** 2
        
def test_parallel():
    import pandas as pd

    "testing script for multithread_run"
    li_of_tuples = [("x", "y", "z"),("y", "z", "p"),("yw", "zs", "psd"),("yd", "zf", "pf")]
    print([["xyz", "ywzspsd"], ["yzp", "ydzfpf"]]== multithread_run(fun_async, li_of_tuples, n_pool=2, start_delay=0.1, verbose=True))

    ##### 2nd test for multithread run list

    """testing the script for checking the list"""
    li_of_tuples = [["x", "y", "z"],["y", "z", "p"]]
    print(multithread_run_list(function1=(fun_async, (li_of_tuples[0],)),function2=(fun_async, (li_of_tuples[1],)))== [("function1", ["x", "y", "z"]), ("function2", ["y", "z", "p"])])

    #### testing script for multiproc_run

    li_of_tuples = [("x", "y", "z"),("y", "z", "p"),("yw", "zs", "psd"),("yd", "zf", "pf"),]
    print(multiproc_run(fun_async, li_of_tuples, n_pool=2, start_delay=0.1, verbose=True)== [["xyz"], ["yzp"], ["ywzspsd"], ["ydzfpf"], []])

    #### TEST applying groupby
    df = pd.DataFrame(data={'result':[5, 8, 1, 7, 0, 3, 2, 9, 4, 6], 'user_id':[1, 1, 2, 3, 4, 4, 5, 8, 9, 9], 'value':[27, 14, 26, 19, 28, 9, 11, 1, 26, 18],'data_chunk':[1, 1, 2, 3, 4, 4, 5, 8, 9, 9]})
    expected_df = df.copy()
    expected_df["inv_sum"] = [14.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 18.0, 0.0]
    result = pd_groupby_parallel(df.groupby("user_id"), func=group_function, int=1)
    print(expected_df.equals(result))

    # groupby parallel2
    result = pd_groupby_parallel(df.groupby("user_id"), func=group_function, int=1)
    print(expected_df.equals(result))

    ###test for applying groupby in pandas
    df = pd.DataFrame({"A": [0, 1, 2, 3, 4],   "B": [100, 200, 300, 400, 500],})
    expected_df = pd.DataFrame({"A": [0, 1, 4, 9, 16], "B": [10000, 40000, 90000, 160000, 250000]})
    result = pd_apply_parallel(df=df, colsgroup=["A" "B"], fun_apply=apply_func, npool=4)
    print(expected_df.equals(result))


def pd_groupby_parallel(groupby_df,func=None,n_cpu: int = 1,**kw,):
    """Performs a Pandas groupby operation in parallel.
    pd.core.groupby.DataFrameGroupBy
    Example usage:
        import pandas as pd
        df = pd.DataFrame({'A': [0, 1], 'B': [100, 200]})
        df.groupby(df.groupby('A'), lambda row: row['B'].sum())
    Authors: Tamas Nagy and Douglas Myers-Turnbull
    """
    import pandas as pd

    num_cpus = multiprocessing.cpu_count() - 1 if n_cpu == -1 else n_cpu
    log("\nUsing {} CPUs in parallel...".format(n_cpu))
    with multiprocessing.Pool(n_cpu) as pool:
        queue = multiprocessing.Manager().Queue()
        result = pool.starmap_async(func, [(name, group) for name, group in groupby_df])
        cycler = itertools.cycle("\|/―")
        while not result.ready():
            log(
                "Percent complete: {:.0%} {}".format(
                    queue.qsize() / len(groupby_df), next(cycler)
                )
            )
            time.sleep(0.4)
        got = result.get(timeout=1)
    # log("\nProcessed {} rows in {:.1f}s".format(len(got), time.time() - start))
    return pd.concat(got)


lam = None


def pd_groupby_parallel2(df,colsgroup=None,fun_apply=None,npool=5,start_delay=0.01,):
    """Pandas parallel apply"""
    import pandas as pd, numpy as np, time, gc

    dfg = df.groupby(colsgroup)  ### Need to get the splits

    def f2(df_list):
        dfiall = None
        for dfi in df_list:
            dfi = dfi.apply(fun_apply)
            dfiall = pd.concat((dfiall, dfi)) if dfiall is None else dfi
            del dfi
            gc.collect()
        return dfiall

    #### Pool execute #################################################
    import multiprocessing as mp

    # pool     = multiprocessing.Pool(processes=npool)
    pool = mp.pool.ThreadPool(processes=n_pool)
    job_list = []

    input_list = [[] * npool]
    for i, name, dfi in enumerate(dfg):
        input_list[i % npool].append(dfg)

    for i, inputi in enumerate(input_list):
        time.sleep(start_delay)
        log("starts", i)
        job_list.append(pool.apply_async(f2, (inputi,)))
        if verbose:
            log(i, dfi.shape)

    ##### Aggregate results ##########################################
    dfall = None
    for i in range(npool):
        if i >= len(job_list):
            break
        dfi = job_list[i].get()
        dfall = pd.concat((dfall, dfi)) if dfall is not None else dfi
        del dfi
        log(i, "job finished")

    pool.terminate()
    pool.join()
    pool = None
    return dfall


def pd_apply_parallel(df, colsgroup=None, fun_apply=None, npool=5, start_delay=0.01, verbose=True):
    """Pandas parallel apply"""
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
        time.sleep(start_delay)
        log("starts", i)
        i2 = i + 2 if i == npool - 1 else i + 1
        dfi = df.iloc[i * size : (i2 * size), :]
        job_list.append(pool.apply_async(f2, (dfi,)))
        if verbose:
            log(i, dfi.shape)

    dfall = None
    for i in range(npool):
        if i >= len(job_list):
            break
        dfi = job_list[i].get()
        dfall = pd.concat((dfall, dfi)) if dfall is not None else dfi
        del dfi
        log(i, "job finished")

    pool.terminate()
    pool.join()
    pool = None
    return dfall


def multiproc_run(fun_async, input_list: list, npool=5, start_delay=0.1, verbose=True, **kw):
    """Multiprocessing execute
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
            print("thread ", j, len(xi_list[j]))
        time.sleep(6)

    #### Pool execute ###################################
    import multiprocessing as mp

    pool = multiprocessing.Pool(processes=npool)
    # pool     = mp.pool.ThreadPool(processes=n_pool)
    job_list = []
    for i in range(npool):
        time.sleep(start_delay)
        log("starts", i)
        job_list.append(pool.apply_async(fun_async, (xi_list[i],)))
        if verbose:
            log(i, xi_list[i])

    res_list = []
    for i in range(npool):
        if i >= len(job_list):
            break
        res_list.append(job_list[i].get())
        log(i, "job finished")

    pool.terminate()
    pool.join()
    pool = None
    log("n_processed", len(res_list))
    return res_list


result_list = []


def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def multithread_run(
    fun_async, input_list: list, n_pool=2, start_delay=0.1, verbose=True, **kw
):
    """input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
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
            print("thread ", j, len(xi_list[j]))
        time.sleep(6)

    #### Pool execute ###################################
    import multiprocessing as mp

    # pool     = multiprocessing.Pool(processes=3)
    pool = mp.pool.ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
        time.sleep(start_delay)
        log("starts", i)
        job_list.append(pool.apply_async(fun_async, (xi_list[i],), callback=log_result))
        if verbose:
            log(i, xi_list[i])

    res_list = []
    for i in range(n_pool):
        if i >= len(job_list):
            break
        res_list.append(job_list[i].get())
        log(i, "job finished")

    pool.terminate()
    pool.join()
    pool = None
    log("n_processed", len(res_list))
    return res_list


def multithread_run_list(**kwargs):
    """Creating n number of threads:  1 thread per function,    starting them and waiting for their subsequent completion
    os_multithread(function1=(test_print, ("some text",)),
                          function2=(test_print, ("bbbbb",)),
                          function3=(test_print, ("ccccc",)))
    """

    class ThreadWithResult(Thread):
        def __init__(
            self,
            group=None,
            target=None,
            name=None,
            args=(),
            kwargs={},
            *,
            daemon=None,
        ):
            self.result = None

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

if __name__ == '__main__':
    test_parallel()