# -*- coding: utf-8 -*-
HELP = """


https://eliot.readthedocs.io/en/stable/



pip install filprofiler




"""
import itertools, time, multiprocessing, pandas as pd, numpy as np, pickle, gc

#################################################################################################
def log(*s):
    """This is the docstring for function log
    Args:
        *s: input variable *s
    Returns:
        None
    """
    print(*s, flush=True)


def help():
    """This is the docstring for function help
    Args:
    Returns:
        None
    """
    ss  = ""
    ss += HELP
    print(ss)



def print_everywhere():
    """
    https://github.com/alexmojaki/snoop
    """
    txt ="""
    import snoop; snoop.install()  ### can be used anywhere
    
    @snoop
    def myfun():
    
    from snoop import pp
    pp(myvariable)
        
    """
    import snoop
    snoop.install()  ### can be used anywhere"
    print("Decaorator @snoop ")


def log10(*s, nmax=60):
    """ Display variable name, type when showing,  pip install varname

    """
    from varname import varname, nameof
    for x in s :
        print(nameof(x, frame=2), ":", type(x), "\n",  str(x)[:nmax], "\n")


def log5(*s):
    """    ### Equivalent of print, but more :  https://github.com/gruns/icecream
    pip install icrecream
    ic()  --->  ic| example.py:4 in foo()
    ic(var)  -->   ic| d['key'][1]: 'one'

    """
    from icecream import ic
    return ic(*s)


def log_trace(msg="", dump_path="", globs=None):
    """This is the docstring for function log_trace
    Args:
        msg: input variable msg
        dump_path: input variable dump_path
        globs: input variable globs
    Returns:
        None
    """
    print(msg)
    import pdb;
    pdb.set_trace()


def profiler_start():
    """This is the docstring for function profiler_start
    Args:
    Returns:
        None
    """
    ### Code profiling
    from pyinstrument import Profiler
    global profiler
    profiler = Profiler()
    profiler.start()


def profiler_stop():
    """This is the docstring for function profiler_stop
    Args:
    Returns:
        None
    """
    global profiler
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

