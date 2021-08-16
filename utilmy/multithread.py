
from utilmy import log
from multiprocessing.pool import ThreadPool
from threading import Thread



def multithread_run(fun_async, input_list:list, n_pool=5, verbose=True):
    """  input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
    def fun_async(xlist):
      for x in xlist :
            hdfs.upload(x[0], x[1])
    """
    #### Input xi #######################################
    xi_list = [ []  for t in range(n_pool) ]
    for i, xi in enumerate(input_list) :
        jj = i % n_pool
        xi_list[jj].append( xi )

    #### Pool execute ###################################
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
         if verbose : log(i, xi_list[i] )

    res_list = []
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append( job_list[ i].get() )
        log(i, 'job finished')

    pool.terminate() ; pool.join()  ; pool = None
    log('n_processed', len(res_list) )


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

