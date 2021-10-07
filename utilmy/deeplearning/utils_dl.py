# -*- coding: utf-8 -*-
HELP="""

utils in DL

"""
import os,io, numpy as np, sys, glob, time, copy, json, functools, pandas as pd
from typing import Union
from box import Box
import diskcache as dc



################################################################################################
verbose = 0

def log(*s):
    print(*s, flush=True)


def log2(*s):
    if verbose >1 : print(*s, flush=True)


def help():
    from utilmy import help_create
    ss  = HELP
    ss += help_create("utilmy.deeplearning.util_dl")
    print(ss)



################################################################################################
def test():
    pass




################################################################################################
################################################################################################
def tensorboard_log(pars_dict:dict=None,  writer=None,  verbose=True):
    """
    #### Usage 1
    logdir = 'logs/params'

    cc = {'arbitray dict' : 1 }

    from tensorboardX import SummaryWriter
    # from tensorboard import SummaryWriter
    tb_writer = SummaryWriter(logdir)
    tensorboard_log(cc, writer= tb_writer)

    %reload_ext tensorboard
    %tensorboard --logdir logs/params/
    """
    import collections
    def dict_flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(dict_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


    flatten_box = dict_flatten(pars_dict)
    if verbose:
        print(flatten_box)


    for k, v in flatten_box.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(str(k), v, 0)
        else :
            writer.add_text(str(k), str(v), 0)

    writer.close()
    return writer



    
def gpu_usage(): 
    
   cmd = "nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv"

   from utilmy import os_system    
   res = os_system(cmd)
   print(res)
        
   ## cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
   ## cmd2= " nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv  "
    
    
def gpu_free():  
    cmd = "nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv  "
    from utilmy import os_system    
    ss = os_system(cmd)

    # ss   = ('pci.bus_id, utilization.gpu [%]\n00000000:01:00.0, 37 %\n00000000:02:00.0, 0 %\n00000000:03:00.0, 0 %\n00000000:04:00.0, 89 %\n', '')
    ss   = ss[0]
    ss   = ss.split("\n")
    ss   = [ x.split(",") for x in ss[1:] if len(x) > 4 ]
    print(ss)
    deviceid_free = []
    for ii, t in enumerate(ss) :
        if  " 0 %"  in t[1]:
            deviceid_free.append( ii )
    print( deviceid_free )        
    return deviceid_free
            
            
            



