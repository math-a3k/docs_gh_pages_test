# -*- coding: utf-8 -*-
HELP= """ IO



"""
import os, glob, sys, math, string, time, json, logging, functools, random, yaml, operator, gc
from pathlib import Path; from collections import defaultdict, OrderedDict
# from utilmy import  to_file, date_now_jp
from box import Box


#####################################################################################
def log(*s):
    print(*s, flush=True)


def log2(*s):
    if random.random() > 0.999 : print(*s, flush=True)


def help():
    ss  = ""


    ss += HELP
    print(ss)



#####################################################################################
#####################################################################################
def test():
    """


    """
    pass





  
  
  
#####################################################################################
def hdfs_put(from_dir="", to_dir="",  verbose=True, n_pool=25, dirlevel=50,  **kw):
    """
     hdfs_put LocalFile into HDFS in multi-thread
    from_dir = "hdfs://nameservice1/user/
    to_dir   = "data/"

    """
    import glob, gc,os, time, pyarrow as pa
    from multiprocessing.pool import ThreadPool

    def log(*s, **kw):
      print(*s, flush=True)

    #### File ############################################
    hdfs      = pa.hdfs.connect()
    hdfs.mkdir(to_dir  )

    from utilmy import os_walk
    dd = os_walk(from_dir, dirlevel= dirlevel, pattern="*")
    fdirs, file_list = dd['dir'], dd['file']
    file_list = sorted(list(set(file_list)))
    n_file    = len(file_list)
    log('Files', n_file)

    file_list2 = []
    for i, filei in enumerate(file_list) :
        file_list2.append( (filei,   to_dir + filei.replace(from_dir,"")   )  )


    ##### Create Target dirs  ###########################
    fdirs = [ t.replace(from_dir,"") for t in fdirs]
    for di in fdirs :
        hdfs.mkdir(to_dir + "/" + di )

    #### Input xi #######################################
    xi_list = [ []  for t in range(n_pool) ]
    for i, xi in enumerate(file_list2) :
        jj = i % n_pool
        xi_list[jj].append( xi )

    #### function #######################################
    def fun_async(xlist):
      for x in xlist :
         try :
           with open(x[0], mode='rb') as f:
                hdfs.upload(x[1], f,)
         except :
            try :
               time.sleep(60)
               with open(x[0], mode='rb') as f:
                  hdfs.upload(x[1], f,)
            except : print('error', x[1])

    #### Pool execute ###################################
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
         if verbose : log(i, xi_list[i] )

    res_list = []
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append( job_list[ i].get() )
        log(i, 'job finished')


    pool.terminate() ; pool.join()  ;  pool = None
    log('n_processed', len(res_list) )




def hdfs_walk(path="hdfs://nameservice1/user/", dirlevel=3, hdfs=None):   ### python  prepro.py hdfs_walk
    import pyarrow as pa
    hdfs = pa.hdfs.connect() if hdfs is None else hdfs
    path = "hdfs://nameservice1/" + path if 'hdfs://' not in path else path

    def os_walk(fdirs):
        flist3 = []
        for diri  in fdirs :
            flist3.extend( [ t for t in hdfs.ls(diri) ]  )
        fdirs3 = [ t   for t in flist3 if hdfs.isdir(t) ]
        return flist3, fdirs3

    flist0, fdirs0   = os_walk([path])
    fdirs = fdirs0
    for i in range(dirlevel):
       flisti, fdiri = os_walk(fdirs)
       flist0 =  list(set(flist0  + flisti ))
       fdirs0 =  list(set(fdirs0  + fdiri ))
       fdirs  = fdiri
    return {'file': flist0, 'dir': fdirs0}



def hdfs_get(from_dir="", to_dir="",  verbose=True, n_pool=20,   **kw):
    """
    import fastcounter
    counter = fastcounter.FastWriteCounter,()
    counter.increment(1)
    cnt.value
    """
    import glob, gc,os, time
    from multiprocessing.pool import ThreadPool

    def log(*s, **kw):
      print(*s, flush=True, **kw)

    #### File ############################################
    os.makedirs(to_dir, exist_ok=True)
    import pyarrow as pa
    hdfs      = pa.hdfs.connect()
    # file_list = [ t for t in hdfs.ls(from_dir) ]
    file_list = hdfs_walk(from_dir, dirlevel=10)['file']


    def fun_async(xlist):
      for x in xlist :
         try :
            hdfs.download(x[0], x[1])
            # ktot = ktot + 1   ### Not thread safe
         except :
            try :
               time.sleep(60)
               hdfs.download(x[0], x[1])
               # ktot = ktot + 1
            except : pass

    ######################################################
    file_list = sorted(list(set(file_list)))
    n_file    = len(file_list)
    log('Files', n_file)
    if verbose: log(file_list)

    xi_list = [ []  for t in range(n_pool) ]
    for i, filei in enumerate(file_list) :
        jj = i % n_pool
        xi_list[jj].append( (filei,   to_dir + "/" + filei.split("/")[-1]   )  )

    #### Pool count
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
         if verbose : log(i, xi_list[i] )

    res_list = []
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append( job_list[ i].get() )
        log(i, 'job finished')

    pool.terminate()  ;  pool.join()  ; pool = None
    log('n_processed', len(res_list) )
    log('n files', len(os.listdir(to_dir)) )





def screenshot( output='fullscreen.png', monitors=-1):
  """
  with mss() as sct:
    for _ in range(100):
        sct.shot()
  # MacOS X
  from mss.darwin import MSS as mss


  """
  try :
    # GNU/Linux
    from mss.linux import MSS as mss
  except :
    # Microsoft Windows
    from mss.windows import MSS as mss

  filename = sct.shot(mon= monitors, output= output)
  print(filename)
