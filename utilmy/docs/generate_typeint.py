# -*- coding: utf-8 -*-
MNAME = "utilmy.tabular.util_sparse"
HELP  = """ Utils for sparse matrix creation
"""
import os, sys, time, datetime,inspect, json, yaml, gc, glob, pandas as pd, numpy as np
from box import Box
from ast import literal_eval
###################################################################################

from utilmy import log, log2

def test_all():
  test()


def test():
  import utilmy

  print(utilmy.__file__)

  dir0 = utilmy.__file__.replace("\\","/")
  dir0 = "/".join( dir0.split("/")[:-2])  +"/"
  log(dir0)

  os.chdir(dir0)
  dirin  = "utilmy/tabular/"
  dirout = "docs/stub/"
  run_monkeytype(dirin, dirout, mode='stub', diroot=None, nfile=1, exclude="sparse" )
  os.system( f"ls {dirout}/")


def run_monkeytype(dirin, dirout, mode, diroot=None, nfile=10, exclude="" ):
    """
    pip install --upgrade utilmy monkeytype

    was running your test files via monkeytype run MY_SCRIPT.py command

    That create SQLite3 DB locally and stores dtypes of variables

    then to apply it for a files I was using: monkeytype apply MY_SCRIPT

    You can see other documentation here: https://pypi.org/project/MonkeyType/

    """
    from utilmy import os_makedirs

    flist = glob.glob(dirin  +"/*.py")
    flist = flist + glob.glob(dirin  +"/**/*.py")
    flist = [ fi for fi in flist if exclude not in fi ]
    flist = flist[:nfile]
    log(flist)


    if diroot is None :
      diroot = os.getcwd()
      diroot = diroot.replace("\\", "/")
    diroot = diroot + "/" if diroot[-1] != "/" else  diroot


    for fi in flist :
      fi = fi.replace("\\", "/")
      fi = fi.replace(diroot, "")

      log(fi)

      cmd = f"monkeytype run {fi}"
      os.system(cmd)

      fi2 = fi.replace( diroot, ""  )

      dirouti = dirout +"/" + fi2
      os_makedirs(dirouti)

      fi3 = fi2.replace(".py", "").replace("/", ".")
      log(fi3, dirouti)
      cmd = f"monkeytype stub {fi3}  > {dirouti} 2>&1 "





################################################################################
################################################################################
if __name__ == '__main__':
    #import fire
    #fire.Fire()
    test()

