# -*- coding: utf-8 -*-
HELP = """


https://eliot.readthedocs.io/en/stable/



pip install filprofiler




"""
import itertools, time, multiprocessing, pandas as pd, numpy as np, pickle, gc

#################################################################################################
def log(*s):
    print(*s, flush=True)


def help():
    ss  = ""
    ss += HELP
    print(ss)



