# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
MNAME = "utilmy.tabular.util_sparse"
HELP  = """ Utils for sparse matrix creation
"""
from email.policy import default
import os, sys, time, datetime,inspect, json, yaml, gc, glob, pandas as pd, numpy as np
import subprocess, shutil, re
from box import Box
from ast import literal_eval
###################################################################################
import utilmy
from utilmy import os_makedirs
from utilmy import log, log2
import sysconfig

## required if we want to annotate files in site-packages
os.environ["MONKEYTYPE_TRACE_MODULES"] = 'utilmy,site-packages'


def test_all():
  test1()



def test3():
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("-di", "--dirin", type=str, required=True,
    help="path to input files directory")
  ap.add_argument("-do", "--dirout", type=str, required=True,
    help="path to output files directory")
  ap.add_argument("-nf", "--nfile", type=int, default=10 ,
    help="number of files to be annotated")
  ap.add_argument("-ex", "--exclude", type=str, default="",
    help="Files whose name contains this value will be excluded")
  ap.add_argument("-m", "--mode", type=str,default="stub",
    help="mode stab or apply")
  args = vars(ap.parse_args())
  #test()
  d = dict()
  d['dirin'] = args["dirin"]
  d['dirout'] = args["dirout"]
  d['nfile'] = args["nfile"]
  d['exclude'] = args["exclude"]
  d['mode'] = args["mode"]
  run_monkeytype(**d)



def test1():

  log(utilmy.__file__)

  dir0 = utilmy.__file__.replace("\\","/") 
  dir0 = "/".join( dir0.split("/")[:-2])  +"/"
  log(dir0)


  os.chdir(dir0) 

  dirin  = "utilmy/tabular/" 
  dirout = "docs/stub/"

  run_monkeytype(dirin, dirout, mode='stub', diroot=None, nfile=10, exclude="sparse" )
  os.system( f"ls {dirout}/")


def test2():
  log(utilmy.__file__)

  dir0 = utilmy.__file__.replace("\\","/") 
  dir0 = "/".join( dir0.split("/")[:-2])  +"/"
  log(dir0)
  os.chdir(dir0)

  dirin  = "utilmy/tabular/" 
  dirout = "docs/stub/"

  run_monkeytype(dirin, dirout, mode='stub', diroot=None, nfile=10, exclude="sparse" )
  os.system( f"ls {dirout}/")



def run_monkeytype(dirin:str, dirout:str, diroot:str=None, mode="stub", nfile=10, exclude="" ):
    """  Generate type hints for files
         test files via monkeytype run MY_SCRIPT.py command
          That create SQLite3 DB locally and stores dtypes of variables
           then to apply it for a files I was using: monkeytype apply MY_SCRIPT
          documentation here: https://pypi.org/project/MonkeyType/

    """
    dirin = dirin.replace("\\", "/") + '/'


    if "utilmy." in dirin :
        dir0 =  os.path.dirname( utilmy.__file__) + "/"        
        dirin = dir0 +  dirin.replace(".utilmy", "").replace(".", "/")


    os.chdir(dirin)
    diroot = os.getcwd()
    diroot = diroot.replace("\\", "/")
    diroot = diroot + "/" if diroot[-1] != "/" else  diroot


    flist = glob.glob(diroot +"/*.py") 
    flist = flist + glob.glob(diroot + "/**/*.py") 
    if exclude != "":
      flist = [ fi for fi in flist if exclude not in fi ]

    flist = flist[:nfile]


    dir0 = utilmy.__file__.replace("\\","/") 
    dir0 = "/".join( dir0.split("/")[:-2])  +"/"


    for fi in flist :
      fi = fi.replace("\\", "/")
      fi_dir = '/'.join(fi.split("/")[:-1])+ "/"
      log(f'fi_dir : {fi_dir}')

      fi = fi.replace(diroot, "")
      intensive_path = re.compile(re.escape(dir0), re.IGNORECASE)
      pk = intensive_path.sub("",fi_dir)


      log(f"####### Processing file {fi} ###########")
      log(f"Runing Monkeytype to get traces database")

      ### Monkeytype require using temporary runner script to import packages (Not necessary if the file is a pytest) 
      fi_script = fi.replace(".py","").replace("/",".")
      cmd = r'echo import %s ; %s.test_all() > run_script.py' % (fi_script,fi_script)
      subprocess.call(cmd, shell=True)


      # run monkeytype on temporary script
      cmd = f"monkeytype run run_script.py" #{fi}
      os.system(cmd)

      fi2 = fi.replace( diroot, ""  )
      dirouti = dirout +"/"+ pk + fi2 
      os_makedirs(dirouti)
      fi3 = fi2.replace(".py", "").replace("/", ".")


      ### copy sqlite traces database where our file is located
      try:
        shutil.copy("monkeytype.sqlite3", fi_dir) 
      except:
        pass

      
      log(f"Generate output in mode {mode}")
      # Apply or Stub
      if mode == "apply":
        cmd = r'monkeytype apply %s > %s 2>&1' % (fi3 , dirouti)
        subprocess.call(cmd, shell=True)

      if mode == "stub":
        dirouti = dirouti.replace(".py", ".pyi")
        cmd = r'monkeytype stub %s > %s 2>&1' % (fi3 , dirouti)
        subprocess.call(cmd, shell=True)


      log(f"####### clean up")
      sqlite_fi = f'{fi_dir}/monkeytype.sqlite3' 
      cmd = 'rm -f run_script.py monkeytype.sqlite3 %s' % sqlite_fi
      subprocess.call(cmd, shell=True)





################################################################################
################################################################################
if __name__ == '__main__':
  test1()
