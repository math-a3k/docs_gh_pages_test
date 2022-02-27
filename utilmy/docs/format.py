# -*- coding: utf-8 -*-
MNAME = "utilmy.docs.format"
HELP = """ utils for  formatting/parising python files


cd myutil
python utilmy/docs/format.py  test1


"""
import os,sys,time,gc, glob, numpy as np, pandas as pd
from box import Box



#### Types
from typing import List, Optional, Tuple, Union


#############################################################################################
from utilmy import log, log2
def help():
    from utilmy import help_create
    print( HELP + help_create(MNAME) )



#############################################################################################
def test_all() -> None:
    log(MNAME)
    test1()
    test2()


def test1() -> None:
    pass


def test2() -> None:
    pass



#############################################################################################
def format_addheader(dirin:str=None):

    flist = glob.glob( dirin + "**/*.py")

    for fi in flist :
        with open(fi, mode='r') as fp:
            ll = fp.readlines()

        if not 'MNAME' in  ll : 
            ll2 = "NAME" + ""


        if not 'HELP' in  ll : 
             pass




if 'utilties':
    def os_path_norm(diroot:str):
        """os_path_norm 
        Args:
            diroot:
        Returns:
            _description_
        """
        diroot = diroot.replace("\\", "/")
        return diroot + "/" if diroot[-1] != "/" else  diroot



    def glob_glob_python(dirin, suffix ="*.py", nfile=7, exclude=""):
        """glob_glob_python 
        Args:
            dirin: _description_
            suffix: _description_. Defaults to "*.py".
            nfile: _description_. Defaults to 7.
            exclude: _description_. Defaults to "".

        Returns:
            _description_
        """
        flist = glob.glob(dirin + suffix) 
        flist = flist + glob.glob(dirin + "/**/" + suffix ) 
        if exclude != "":     
           flist = [ fi for fi in flist if exclude not in fi ]
        flist = flist[:nfile]
        log(flist)
        return flist




###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()  ### python utilmy/ZZZZZ/util_xxxx.py  test1



