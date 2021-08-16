from threading import Thread
import cProfile, pstats, io, os, errno, signal, time
from functools import wraps
from contextlib import contextmanager




########################################################################################################################
########################################################################################################################
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


def thread_decorator(func):
    """ A decorator to run function in background on thread
	Return:
		background_thread: ``Thread``
    """
    @wraps(func)
    def wrapper(*args, **kwags):
        background_thread = Thread(target=func, args=(*args,))
        background_thread.daemon = True
        background_thread.start()
        return background_thread

    return wrapper





########################################################################################################################
class _TimeoutError(Exception):
    """Time out error"""
    pass


########################################################################################################################
def timeout_decorator(seconds=10, error_message=os.strerror(errno.ETIME)):
    """Decorator to throw timeout error, if function doesnt complete in certain time
    Args:
        seconds:``int``
            No of seconds to wait
        error_message:``str``
            Error message
            
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise _TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def timer_decorator(func):
    """
    Decorator to show the execution time of a function or a method in a class.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'function {func.__name__} finished in: {(end - start):.2f} s')
        return result

    return wrapper



########################################################################################################################
@contextmanager
def profiler_context():
    """
    Context Manager the will profile code inside it's bloc.
    And print the result of profiler.
    Example:
        with profiler_context():
            # code to profile here
    """
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    try:
        yield profiler
    except Exception as e:
        raise e
    finally:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))


def profiler_decorator(func):
    """
    A decorator that will profile a function
    And print the result of profiler.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from pyinstrument import Profiler
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        return result
    return wrapper



def profiler_decorator_base(fnc):
    """
    A decorator that uses cProfile to profile a function
    And print the result
    """
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


