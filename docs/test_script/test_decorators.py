from threading import Thread
import cProfile, pstats, io, os, errno, signal, time
from functools import wraps
from contextlib import contextmanager
from utilmy.debug import log



def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    test_decorators()	
    test_decorators2()

def test_decorators():
    """
    #### python test.py   test_decorators
    """
    from utilmy.decorators import thread_decorator, timeout_decorator, profiler_context,profiler_decorator, profiler_decorator_base

    @thread_decorator
    def thread_decorator_test():
        log("thread decorator")


    @profiler_decorator_base
    def profiler_decorator_base_test():
        log("profiler decorator")

    @timeout_decorator(10)
    def timeout_decorator_test():
        log("timeout decorator")

    profiler_decorator_base_test()
    timeout_decorator_test()
    thread_decorator_test()



def test_decorators2():
    """function test_decorators2
    Args:
    Returns:
        
    """
    from utilmy.decorators import profiler_decorator, profiler_context

    @profiler_decorator
    def profiled_sum():
       return sum(range(100000))

    profiled_sum()

    with profiler_context():
       x = sum(range(1000000))
       print(x)


    from utilmy import profiler_start, profiler_stop
    profiler_start()
    print(sum(range(1000000)))
    profiler_stop()


    ###################################################################################
    from utilmy.decorators import timer_decorator
    @timer_decorator
    def dummy_func():
       time.sleep(2)

    class DummyClass:
       @timer_decorator
       def method(self):
           time.sleep(3)

    dummy_func()
    a = DummyClass()
    a.method()




########################################################################################################################
########################################################################################################################
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



def test0():
    """function test0
    Args:
    Returns:
        
    """
    with profiler_context():
        x = sum(range(1000000))
        print(x)
    from utilmy import profiler_start, profiler_stop
    profiler_start()
    print(sum(range(1000000)))
    profiler_stop()

@thread_decorator
def thread_decorator_test():
    """function thread_decorator_test
    Args:
    Returns:
        
    """
    log("thread decorator")

@profiler_decorator_base
def profiler_decorator_base_test():
    """function profiler_decorator_base_test
    Args:
    Returns:
        
    """
    log("profiler decorator")

@timeout_decorator(10)
def timeout_decorator_test():
    """function timeout_decorator_test
    Args:
    Returns:
        
    """
    log("timeout decorator")


@profiler_decorator
def profiled_sum():
    """function profiled_sum
    Args:
    Returns:
        
    """
    return sum(range(100000))

@timer_decorator
def dummy_func():
    """function dummy_func
    Args:
    Returns:
        
    """
    time.sleep(2)
