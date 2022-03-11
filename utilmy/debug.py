# -*- coding: utf-8 -*-
HELP = """


https://eliot.readthedocs.io/en/stable/



pip install filprofiler




"""
import itertools, time, multiprocessing, pandas as pd, numpy as np, pickle, gc

#################################################################################################
def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    print(*s, flush=True)


def help():
    """function help
    Args:
    Returns:
        
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
    """function log_trace
    Args:
        msg:   
        dump_path:   
        globs:   
    Returns:
        
    """
    print(msg)
    import pdb;
    pdb.set_trace()


def profiler_start():
    """function profiler_start
    Args:
    Returns:
        
    """
    ### Code profiling
    from pyinstrument import Profiler
    global profiler
    profiler = Profiler()
    profiler.start()


def profiler_stop():
    """function profiler_stop
    Args:
    Returns:
        
    """
    global profiler
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

