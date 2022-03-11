# -*- coding: utf-8 -*-
MNAME = "utilmy.docs.generate_typehint"
HELP  = """ Utils for type generation


  test files via monkeytype run MY_SCRIPT.py command
  That create SQLite3 DB locally and stores dtypes of variables
    then to apply it for a files I was using: monkeytype apply MY_SCRIPT
  documentation here: https://pypi.org/project/MonkeyType/


"""
from email.policy import default
import os, sys, time, datetime,inspect, json, yaml, gc, glob, pandas as pd, numpy as np
import subprocess, shutil, re, sysconfig
from box import Box
from ast import literal_eval

## required if we want to annotate files in site-packages
os.environ["MONKEYTYPE_TRACE_MODULES"] = 'utilmy,site-packages'

###################################################################################
from utilmy import log, log2, os_makedirs
import utilmy
def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    print( HELP + help_create(MNAME) )


####################################################################################
def test_all():
  """function test_all
  Args:
  Returns:
      
  """
  test1()


def test1():
  """function test1
  Args:
  Returns:
      
  """
  log(utilmy.__file__)

  exclude = ""; nfile= 10
  dir0   = os.getcwd()
  dirin  = dir0 + "/utilmy/tabular/" 
  dirout = dir0 + "/docs/types/"
  diroot = dir0        
  dirin = dirin.replace("\\", "/") + '/'

  run_monkeytype(dirin, dirout, mode='full,stub', diroot=diroot, nfile=3, exclude="sparse" )
  os.system( f"ls {dirout}/")


def run_utilmy(nfile=10000):
  """function run_utilmy
  Args:
      nfile:   
  Returns:
      
  """
  log(utilmy.__file__)
  exclude = "";
  dir0   = os.getcwd()
  dirin  = dir0 + "/utilmy/" 
  dirout = dir0 + "/docs/types/"
  diroot = dir0        
  dirin = dirin.replace("\\", "/") + '/'

  run_monkeytype(dirin, dirout, mode='full,stub', diroot=diroot, nfile=nfile, exclude="z" )
  os.system( f"ls {dirout}/")



def run_utilmy2(nfile=100000):
  """function run_utilmy2
  Args:
      nfile:   
  Returns:
      
  """
  log(utilmy.__file__)
  exclude = ""; 
  dir0   = os.getcwd()
  dirin  = dir0 + "/utilmy/" 
  dirout = dir0 + "/utilmy/"
  diroot = dir0        
  dirin = dirin.replace("\\", "/") + '/'

  run_monkeytype(dirin, dirout, mode='full', diroot=diroot, nfile=nfile, exclude="z" )
  os.system( f"ls {dirout}/")



def test2():
  """function test2
  Args:
  Returns:
      
  """
  log(utilmy.__file__)

  dir0 = utilmy.__file__.replace("\\","/") 
  dir0 = "/".join( dir0.split("/")[:-2])  +"/"
  log(dir0)
  os.chdir(dir0)

  dirin  = "utilmy/tabular/" 
  dirout = "docs/stub/"

  run_monkeytype(dirin, dirout, mode='stub', diroot=None, nfile=10, exclude="sparse" )
  os.system( f"ls {dirout}/")



def os_path_norm(diroot):
    """function os_path_norm
    Args:
        diroot:   
    Returns:
        
    """
    diroot = diroot.replace("\\", "/")
    return diroot + "/" if diroot[-1] != "/" else  diroot



def glob_glob_python(dirin, suffix ="*.py", nfile=7, exclude=""):
    """function glob_glob_python
    Args:
        dirin:   
        suffix :   
        nfile:   
        exclude:   
    Returns:
        
    """
    flist = glob.glob(dirin + suffix) 
    flist = flist + glob.glob(dirin + "/**/" + suffix ) 
    if exclude != "":
      flist = [ fi for fi in flist if exclude not in fi ]
    flist = flist[:nfile]
    log(flist)
    return flist


def run_monkeytype(dirin:str, dirout:str, diroot:str=None, mode="stub", nfile=10, exclude="" ):
    """Generate type hints for files
          Args:
              dirin (str): _description_
              dirout (str): _description_
              diroot (str, optional): _description_. Defaults to None.
              mode (str, optional): _description_. Defaults to "stub".
              nfile (int, optional): _description_. Defaults to 10.
              exclude (str, optional): _description_. Defaults to "".
            exclude = ""; nfile= 10
            dir0 = os.getcwd()
            dirin  = dir0 + "/utilmy/tabular/" 
            dirout = dir0 + "/docs/stub/"
            diroot = dir0        
            dirin = dirin.replace("\\", "/") + '/'
    """   

    import os, sys
    os.makedirs(dirout, exist_ok=True)
    if "utilmy." in dirin :
        dir0 =  os.path.dirname( utilmy.__file__) + "/"        
        dirin = dir0 +  dirin.replace("utilmy", "").replace(".", "/").replace("//","/")

    diroot = os.getcwd()  if diroot is None else diroot
    diroot = os_path_norm(diroot)

    
    flist = glob_glob_python(dirin, suffix ="*.py", nfile=nfile, exclude=exclude)
    log(flist)

    for fi0 in flist :
      try :
        log(f"\n\n\n\n\n\n\n####### Processing file {fi0} ###########")
        fi      = fi0.replace("\\", "/")
        fi_dir  = os.path.dirname(fi).replace("\\", "/")  + "/"

        ### Relative to module root path
        fi_pref  = fi.replace(diroot, "")
        mod_name = fi_pref.replace(".py","").replace("/",".")
        mod_name = mod_name[1:] if mod_name[0] == "." else mod_name
        log(f'fi_dir : {fi_dir},  {fi_pref}')


        log(f"#### Runing Monkeytype to get traces database")
        fi_monkey = os.getcwd() + '/ztmp_monkey.py'
        ### Monkeytype require using temporary runner script to import packages (Not necessary if the file is a pytest) 
        with open( fi_monkey, mode='w' ) as fp :
          fp.write( f"import {mod_name}  as mm ; mm.test_all()" )

        # run monkeytype on temporary script
        os.system(f"monkeytype run ztmp_monkey.py"  )


        log(f"###### Generate output in mode {mode}")
        ### copy sqlite traces database where our file is located
        try:  
          shutil.move("monkeytype.sqlite3", fi_dir + "monkeytype.sqlite3" ) 
        except: pass

        dircur = os.getcwd()
        os.chdir(fi_dir)
        if "full" in mode :  #### Overwrite
            dirouti = dirout +"/full/"+ fi_pref
            os_makedirs(dirouti)
            cmd = f'monkeytype apply {mod_name} > {dirouti} 2>&1' 
            subprocess.call(cmd, shell=True)


        if "stub" in mode:
            dirouti = dirout +"/stub/"+ fi_pref.replace(".py", ".pyi")
            os_makedirs(dirouti)       
            cmd = f'monkeytype stub {mod_name} > {dirouti} 2>&1' 
            subprocess.call(cmd, shell=True)

        log(f"####### clean up")
        try :
          os.remove(f'{fi_dir}/monkeytype.sqlite3' )
          os.remove(fi_monkey)
        except : pass  
        os.chdir( dircur )

      except Exception as e :
         log(e)





################################################################################
################################################################################
if __name__ == '__main__':
  test1()
 