# -*- coding: utf-8 -*-
MNAME = "utilmy.docs.format"
HELP = """ utils for  formatting/parising python files

Goal is to normalize all python files with similar structure.




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



def test1():
    import utilmy
    dirin = os.path.dirname(  utilmy.__file__ )  ### Repo Root folder
    os_file_compile_check_batch(dirin)


def test2() -> None:
    import utilmy
    dirin = os.path.dirname(utilmy.__file__)  ### Repo Root folder
    dirout = os_make_dirs('utilmy\\docs\\test\\')[0] ### test dir
    reformatter(dirin,dirout)





#############################################################################################
def os_file_compile_check_batch(dirin:str, nfile=10) -> None:
    """
    ### DO NOT use Command line  !!!!!!



    """
    flist   = glob_glob_python( dirin, "*.py",nfile= nfile)
    results = []
    for fi in flist :
        res = os_file_compile_check(fi)
        results.append(res)

    #results = [os.system(f"python -m py_compile {i}") for i in flist]
    results = ["Working" if x==0 else "Failed" for x in results]
    return results


def os_file_compile_check(filename:str, verbose=1):
    import ast, traceback
    try : 
        with open(filename, mode='r') as f:
            source = f.read()
        ast.parse(source)
        return True
    except Exception as e:
       if verbose >0 : 
           print(e)
           traceback.print_exc() # Remove to silence any errros
       return False


#############################################################################################
def format_add_logger(dirin:str,dirout:str, nfile=1000):
    """  adding log function import and replace all print( with log(
    """
    def reformat_pyfile(file_path:str):
        """
        """
        with open(file_path,'r',encoding='utf-8') as f:
            txt = f.read()
        import_line = "from utilmy import log, log2"
        if txt.find(import_line)==-1:

            #### It's stupid, did you check.....
            txt = import_line+"\n"+txt
        txt = txt.replace("print(",'log(')
        return txt

    flist = glob_glob_python(dirin, suffix ="*.py", nfile=nfile, exclude="*zz*")

    for fi in flist :
        fi       = os_path_norm(fi)
        fname    = fi.split(os.sep)[-1] 
        file_new = reformat_pyfile(fi) 
        with open(dirout+ "/" + fname,'w',encoding='utf-8') as w:
            w.write(file_new)




#############################################################################################
def format_add_header(dirin:str="./"):

    flist = glob_glob_python(dirin, suffix ="*.py", nfile=10, exclude="*zz*")

    ll2 = ""
    for fi in flist :
        with open(fi, mode='r') as fp:
            ll = fp.readlines()

        if find_str(ll,  'MNAME') < 0  : 
            ll2 = "NAME" + ""


        if find_str(ll,  'HELP') < 0 : 
             pass


        
        ### Write down and check
        to_file(ll2, finew)
        isok  = os_file_compile_check(finew)
        if isok :
            os.remove(fi)
            os.rename(finew, fi)
        else :    
            os.remove(finew)
            err_list.append(fi)
        
  
def to_file(lines, fpath):
    pass
    

def find_str(lines, word):
    for ii,li in enumerate(lines) :
        if word in li : return ii
    return -1    
  



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
        elist = []
        
        if exclude != "":    
           for ei in exclude.split(";"):
               elist = glob.glob(ei + "/" + suffix ) 
        flist = [ fi for fi in flist if fi not in elist ]

        #### Unix format 
        flist = [  fi.replace("\\", "/") for fi in flist]

        flist = flist[:nfile]
        log(flist)
        return flist

    def os_make_dirs(filename):
        if isinstance(filename, str):
            filename = [os.path.dirname(filename)]

        if isinstance(filename, list):
            folder_list = filename
            for f in folder_list:
                try:
                    if not os.path.exists(f):
                        os.makedirs(f)
                except Exception as e:
                    print(e)
            return folder_list


###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()  ### python utilmy/ZZZZZ/util_xxxx.py  test1



