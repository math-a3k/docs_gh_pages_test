# -*- coding: utf-8 -*-
HELP= """ Interface for Key Value store.

pip install diskcache

db_dir : root storage of all db

db_path : Single DB storage



"""
import os, glob, sys, math, string, time, json, logging, functools, random, yaml, operator, gc
from pathlib import Path; from collections import defaultdict, OrderedDict ;  
# from utilmy import  to_file, date_now_jp
from box import Box

import diskcache as dc

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
def pd_random(nrows=1000, ncols= 5):
    return pd.DataFrame( np.random.randint(0, 10, size= (nrows, ncols)),  columns= [ str(i) for i in range(ncols) ]   )

def test():
    """    


    """    
    n = 10**6
    df      = pd_random(n, ncols=2)    
    df['0'] = [  str(x) for x in np.arange(0, len(df)) ]
    df['1'] = [  'xxxxx' + str(x) for x in np.arange(0, len(df)) ]

    
    log('##### multithread insert  commit ##################################')    
    t0 = time.time()
    diskcache_save2(df, colkey='0', colvalue='1', db_path="./dbcache.db", size_limit=100000000000, timeout=10, shards=1, npool=10,
                    sqlmode= 'fast', verbose=True)    
    log(n, time.time()-t0)
    
    
    log('##### multithread insert commit  ##################################')    
    t0 = time.time()
    diskcache_save2(df, colkey='0', colvalue='1', db_path="./dbcache.db", size_limit=100000000000, timeout=10, shards=1, npool=10,
                    sqlmode= 'commit', verbose=True)    
    log(n, time.time()-t0)

    
    
    
    
    
    
    
########################################################################################################    
########################################################################################################
def db_init(db_dir:str="path"):
    """
      db = Box({    
          'db_itemtag_items_path'  :  f"{db_dir}/map_sampling_itemtag_siid.cache",     
          'db_itemid_itemtag_path' :  f"{db_dir}/map_itemid_itemtag.cache",    
      })

    """
    db_list = list( glob.glob(db_dir + "/*") )

    for path in db_list:
        path = path.replace("\\", "/")
        name = path.split("/")[-1].replace(".", "_").replace("-","_")
        log(name)  #, len(globals()[ name ] ))

        ### Global Access
        globals()[ name ] = diskcache_load(path)
     
        

def db_flush(db_dir):  
    """
     Flush wal files on the disk

    """  
    flist = glob.glob(db_dir + "/*")
    for fi in flist:
        log(fi)
        diskcache_config(db_path=fi, task='commit')



def db_size(db_dir= None):        
    ### DB nb of elements    
    flist = glob.glob(db_dir + "/*")        
    for fi in flist:
        print(fi.replace(db_dir,""), len( diskcache_load(fi, verbose=0) ))
            
    



########################################################################################################
########################################################################################################
def diskcache_load( db_path_or_object="", size_limit=100000000000, verbose=True ):    
    """ val = cache[mykey]
    """
    import diskcache as dc

    if not isinstance(db_path_or_object, str ) :
       return db_path_or_object

    cache = dc.Cache(db_path_or_object, size_limit= size_limit )
    if verbose: print('Cache size/limit', len(cache), cache.size_limit ) 
    return cache



def diskcache_save(df, colkey, colvalue, db_path="./dbcache.db", size_limit=100000000000, timeout=10, shards=1, 
                   tbreak=1,  ## Break during insert to prevent big WAL file                   
                   **kw):    
    """ Create dict type on disk, < 100 Gb
       shards>1 : disk spaced is BLOCKED in advance, so high usage
    
    """
    import time
    if shards == 1 :
       import diskcache as dc
       cache = dc.Cache(db_path, size_limit= size_limit, timeout= timeout )        
    else :
       from diskcache import FanoutCache
       cache = FanoutCache( db_path, shards= shards, size_limit= size_limit, timeout= timeout )

    v  = df[[ colkey, colvalue  ]].drop_duplicates(colkey)
    v  = v.values
    print('Starting insert: ', db_path, v.shape )

    
    for i in range(len(v)):
        if i % 500000 == 0 : time.sleep(tbreak)  #### Ensure WAL does not grow too big, by pausing all writers
        
        try :
           cache[ v[i,0] ] = v[i,1]        
        except :
           time.sleep(2.0)
           cache[ v[i,0] ] = v[i,1]        
        
    print('Cache size', len(cache) )    
    return cache


def diskcache_save2(df, colkey, colvalue, db_path="./dbcache.db", size_limit=100000000000, timeout=10, shards=1, npool=10,
                    sqlmode= 'fast', verbose=True):    
    """ Create dict type on disk, < 100 Gb
       shards>1 : disk spaced is BLOCKED in advance, so high usage       
       Issue, uses too much of DISK
    """
    import time, random
    if shards == 1 :
       import diskcache as dc
       cache = dc.Cache(db_path, size_limit= size_limit, timeout= timeout )        
    else :
       from diskcache import FanoutCache
       cache = FanoutCache( db_path, shards= shards, size_limit= size_limit, timeout= timeout )

    v  = df[[ colkey, colvalue  ]].drop_duplicates(colkey)
    v  = v.values  
    print('Starting insert: ', db_path, v.shape )
    
    # if sqlmode == 'commit':
    diskcache_config(db_path, task='commit')
    tbreak = max(5.0, 3.0 * npool)
    
    def insert_key(vlist):            
       for i, vi in enumerate(vlist) :
          #if i % 500000 == 0 :
          #      time.sleep(tbreak)  #### Ensure WAL does not grow too big, by pausing all writers, checkpoint all
          try :
             cache[ vi[0] ] = vi[1] 
          except :
             time.sleep(3.0)                
             cache[ vi[0] ] = vi[1] 
    
    from utilmy.parallel import multithread_run
    n       = len(v) 
    nbefore =  len(cache) 
    log('Cache size:', nbefore )
    mblock = 500000
    for i in range(0, int(n // mblock)+1) :
       i2 = i +1 
       if i == int(n // mblock) : i2 = 3*(i + 2)
       multithread_run(insert_key, input_list= v[i*mblock:(i2+1)*mblock], n_pool=npool, start_delay=0.5, verbose=verbose)
       # len( diskcache_getkeys(cache) )      
       diskcache_config(db_path, task='commit')                                                 
                                                 
    nafter = len(cache)
    log('Cache size:', nafter, 'added:', nafter - nbefore )    
    for k in cache:
       print("Cache check:",k,  str(cache[k])[:100] ); break
    
    #cache.close()
    #diskcache_config(db_path, task='commit')    
    if sqlmode != 'commit':        
       diskcache_config(db_path, task='fast')   

    
def diskcache_getkeys(cache):
    cache = diskcache_load( cache, size_limit=100000000000, verbose=False )
    
    v = cache._sql('SELECT key FROM Cache').fetchall()
    v = [ t[0] for t in v]
    return v


def diskcache_getall(cache, limit=1000000000):
    cache = diskcache_load( cache, size_limit=100000000000, verbose=False )  
    v = cache._sql( f'SELECT key,value FROM Cache LIMIT {limit}').fetchall()
    v = { t[0]: t[1] for t in v }
    return v

def diskcache_get(cache, key, defaultval=None):
    ss = f'SELECT value FROM Cache WHERE key={key} LIMIT 1'
    v  = cache._sql( ss ).fetchall()    
    if len(v) > 0 :
        return v[0][0]
    return defaultval



def diskcache_config(db_path=None, task='commit'):
    """ .open "//e"    python prepro.py diskcache_config 
    https://sqlite.org/wal.html
    
    PRAGMA journal_mode = DELETE;   (You can switch it back afterwards.)
    PRAGMA wal_checkpoint(TRUNCATE);
    PRAGMA journal_mode = WAL;     
    PRAGMA wal_checkpoint(FULL);
    
    """
    cache   = diskcache_load( db_path, size_limit=100000000000, verbose=1 )
    v = diskcache_getkeys(cache) ; log(len(v) )
    cache.close() ; cache = None
    ss      = ""
    if task == 'commit':
       # ss = f""" PRAGMA journal_mode = DELETE;  """
       ss = f""" PRAGMA wal_checkpoint(TRUNCATE); """ 
    
    elif task == 'fast':
       ss = f""" PRAGMA journal_mode = WAL; """

    elif task == 'copy':
       ss = f""" PRAGMA wal_checkpoint(TRUNCATE); """    
    
    with open( 'ztmp_sqlite.sql', mode='w') as fp :
        fp.write(ss)
    log(ss)            
    os.system( f'sqlite3  {db_path}/cache.db  ".read ztmp_sqlite.sql"    ')     
    # time.sleep(20)
    # keys = diskcache_getkeys(cache)
    
    



