# -*- coding: utf-8 -*-
HELP= """



"""
import os, sys, time, datetime,inspect, json, yaml, gc

def log(*s):
    print(*s, flush=True)

def log2(*s, verbose=1):
    if verbose >0 : print(*s, flush=True)


def help():
    ss     = ""
    suffix = "\n\n\n###############################"

    funlist = [ globals()[t] for t in globals().keys() if 'test_' in t ]
    for f in funlist:
        ss += help_get_codesource(f) + suffix

    ss += HELP
    print(ss)


def help_get_codesource(func):
    """ Extract code source from func name"""
    import inspect
    try:
        lines_to_skip = len(func.__doc__.split('\n'))
    except AttributeError:
        lines_to_skip = 0
    lines = inspect.getsourcelines(func)[0]
    return ''.join( lines[lines_to_skip+1:] )


def import_function(fun_name=None, module_name=None):
    import importlib

    if isinstance(module_name, str):
       module1 = importlib.import_module(module_name)
       func = getattr(module1, fun_name)
    else :
       func = globals()[fun_name]

    return func




###################################################################################################
if __name__ == "__main__":
    import fire ;
    fire.Fire()




