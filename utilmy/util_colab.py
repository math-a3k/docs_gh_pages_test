# -*- coding: utf-8 -*-
MNAME = "utilmy.util_colab"
HELP = """ Colab utils

## Usage examples
 
    https://drive.google.com/drive/folders/1FawUmqrNOxabWbepiCchDd1b60pseDXm?usp=sharing
 
    https://colab.research.google.com/drive/12rpbgH3WYcQq3jtl9vzEYeVdu9a9GOM_?usp=sharing
 
    https://colab.research.google.com/drive/1NYQZrfAPqbuLCt9yhVROLMRJM-RrFYWr#scrollTo=Rrho08zYe6Gj

    https://colab.research.google.com/drive/1NYQZrfAPqbuLCt9yhVROLMRJM-RrFYWr#scrollTo=2zMKv6MXOJJu





"""
import os,sys,time, datetime, numpy as np, glob, pandas as pd
from box import Box

#### Types
from typing import List, Optional, Tuple, Union


#############################################################################################
from utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    print( HELP + help_create(MNAME) )



#############################################################################################
def test_all() -> None:
    """function test_all
    Args:
    Returns:
        
    """
    log(MNAME)
    test1()
    test2()


def test1() -> None:
   """function test1
   Args:
   Returns:
       
   """
   pass


#############################################################################################





###################################################################################################
if __name__ == "__main__":
    import fire

    fire.Fire()


