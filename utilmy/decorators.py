from threading import Thread
import cProfile, pstats, io
import os
import errno
import signal
from functools import wraps
import time
from contextlib import contextmanager

########################################################################################################################
class _TimeoutError(Exception):
    """Time out error"""
    pass







########################################################################################################################
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
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


def timer(func):
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


def profiler_deco(func):
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



########################################################################################################################
def os_multithread(**kwargs):
    """
    Creating n number of threads:  1 thread per function,
    starting them and waiting for their subsequent completion
   Example:
        os_multithread(function1=(function_name1, (arg1, arg2, ...)),
                       ...)

    def test_print(*args):
        print(*args)

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


def threading_deco(func):
    """ A decorator to run function in background on thread
	Args:
		func:``function``
			Function with args

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



