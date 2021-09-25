# -*- coding: utf-8 -*-
HELP= """ Interface for Key Value store.

2 type of storage on disk

    pandas parquet file    as  colkey, colvalue,  1) load pandas in RAM   2) Convert the pandas --> Dict  colkey: colval  
        Perf is 100 nano sec read/write  but high RAM Usage
    
    Save into diskcache  directly as colkey: colval
        Perf is 100 micro read/write  and writ is slow,  Low RAM usage




pip install diskcache

db_dir : root storage of all db

db_path : Single DB storage



"""
import os, glob, sys, math, string, time, json, logging, functools, random, yaml, operator, gc
import pandas as pd, numpy as np
from pathlib import Path; from collections import defaultdict, OrderedDict ;  
from utilmy.utilmy import   pd_read_file, pd_to_file
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
    n = 10**4
    df      = pd_random(n, ncols=2)    
    df['0'] = [  str(x) for x in np.arange(0, len(df)) ]
    df['1'] = [  'xxxxx' + str(x) for x in np.arange(0, len(df)) ]

    
    log('##### multithread insert  commit ##################################')    
    t0 = time.time()
    diskcache_save2(df, colkey='0', colvalue='1', db_path="./dbcache.db", size_limit=100000000000, timeout=10, shards=1, npool=10,
                    sqlmode= 'fast', verbose=True)    
    log(n, time.time()-t0)
    diskcache_config(db_path="./dbcache.db", task='fast')
    
    log('##### multithread insert commit  ##################################')    
    t0 = time.time()
    diskcache_save2(df, colkey='0', colvalue='1', db_path="./dbcache.db", size_limit=100000000000, timeout=10, shards=1, npool=10,
                    sqlmode= 'commit', verbose=True)    
    log(n, time.time()-t0)

    diskcache_config(db_path="./dbcache.db", task='commit')

    
    # loading data from disk
    cache = diskcache_load(db_path_or_object="./dbcache.db")

    # reading data
    t0 = time.time()
    data = diskcache_getall(cache)
    log("time taken in reading", time.time()-t0)
    all_keys = diskcache_getkeys(cache)

    log("# checkinh store data")
    for i in range(10):
      print(diskcache_get(cache, (n*i)//10))
      assert(diskcache_get(cache, (n*i)//10) == "xxxxx"+str((n*i)//10))
      



    
########################################################################################################    
##########  Database Class #############################################################################
def db_cli():
    """
      Command line for db access.
      Need a global config file.... set manuaklly
      https://pypi.org/project/qurcol/

    """
    import argparse
    p   = argparse.ArgumentParser()
    add = p.add_argument

    add('task', metavar='task', type=str, nargs=1, help='list/info/check')
    add("val",   metavar='val', type=str, nargs=1, help='list/info/check')

    #### Extra Options



    args = p.parse_args()

    task =  args.task[0]
    val  =  args.val[0]


    #########################################################################
    db = DB(config_path= os.environ['db_diskcacheconfig'])  ##### Need to set

    if task == 'help':  print(HELP)
    if task == 'setconfig':
        os_environ_set('db_diskcache_config', val)


    if task == 'list':   db.list()
    if task == 'info':   db.info()
    if task == 'check':  db.check()
    else :
        log('list,info')




class DB(object):
    """
      DB == collection of diskcache tables on disk.
         A table == a folder on disk

      root_path/mytable1/
      root_path/mytable2/
      root_path/mytable3/
      root_path/mytable4/
      root_path/mytable5/

     {
       'db_paths' : [ 'root1', 'root2'   ]
     }
    
    """
    def __init__(self, config_dict=None, config_path=None):

        if config_dict is not None :
           self.config_dict = config_dict

        elif config_path is not None :
           self.config_dict = json.load(open(config_path, mode='r'))
           self.config_path = config_path

        else :
            self.config_path = os.environ.get('db_diskcache_config', os.getcwd().replace("\\", "/"))
            self.config_dict = json.load(open(self.config_path, mode='r'))

        self.path_list = config_dict('db_paths', [])


    def add(self, db_path):
        self.path_list = list(set(self.path_list + db_path))

    def remove(self, db_path):
        self.path_list = [ t for t in self.path_list  if t != db_path ]

    def list(self, show=True):
        ## get list of db from the folder, size
        flist = []
        for path in self.path_list :
           flist = flist + glob.glob(path +"/*")

        if show :
            for folder in flist :
                size_mb = os_path_size(folder)
                print(folder, size_mb)

        return flist


    def info(self,):
        ## get list of db from the folder, size
        flist = self.list()

        for folder in flist :
            size_mb = os_path_size(folder)

            if os.path.isfile(f"{folder}/cache.db"):
                dbi   = diskcache_load(folder)
                nkeys = diskcache_keycount(dbi)
                print(folder, size_mb, nkeys)
            else :
                print(folder, size_mb)

    def clean(self,):
        ## clean temp files for each diskcache db
        pass


    def check(self, db_path=None):
        """
           Check Sqlite cache.db if file is fine

        """
        pass


    def show(self, db_path=None, n=4):
        """
           show content for each table

        """
        flist = self.list()
        for pathi in flist:
            dbi = diskcache_load(pathi)
            res = diskcache_getall(dbi, limit=n)
            log(pathi, str(res), "\n\n")




def os_environ_set(name, value):
    """
    https://stackoverflow.com/questions/716011/why-cant-environmental-variables-set-in-python-persist

    """
    cmd = """
    
    
    """


def os_path_size(folder=None):
    """
       Get the size of a folder in bytes
    """
    import os
    if folder is None:
        folder = os.getcwd()
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size



        
def db_init(db_dir:str="path", globs=None):
    """ Initialize in the Global Space Name  globs= globals(), DB
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
        globs[ name ] = diskcache_load(path)
     
        

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
            
    

    
def db_merge():   ### python prepro.py  db_merge    2>&1 | tee -a zlog_prepro.py  &
     #### merge 2 dataframe    
     colkey = 'item_tag_vran'
     dir_rec = ""
        
     df1 = pd_read_file( dir_rec+  "/emb//map_idx_13311813.parquet" ) 
     log(df1 )  ### 13 mio
        
     df2 = pd_read_file( dir_rec+  "/map/*" , n_pool=15, verbose=True)
     log(df2)   ### 130 mio

        
     df1 = df1.rename(columns= {'id': colkey})           
     df2 = df2.drop_duplicates(colkey)
     df1 = df1.drop_duplicates(colkey)
     log(df2)
            
     # df1 = df1.iloc[:1000,:]   ; df2 = df2.iloc[:10000, :]
     df1 = df1.merge(df2, on= 'item_tag_vran', how='left' )   
     del df2; gc.collect()
        
     log(df1)   
     pd_to_file(df1, dir_rec + "/db_pandas/map_idx_13311813_siid.parquet" )   
        
     
        
        
def db_create_dict_pandas(df=None, cols=None, colsu=None):   ####  python prepro.py  db_create_dict_pandas &
    ### Load many files and  drop_duplicate on colkey and save as parquet files.
    dirin  =  "/ic/**.parquet"  
    dirout =  "/map/" + dirin.split("/")[-2] +"/"
    
    cols  = [ 'shop_id', 'item_id', 'item_tag_vran'   ]
    colsu = [ 'item_tag_vran'   ]
    
    tag   = "siid_itemtag"
    
    os.makedirs( dirout, exist_ok=True)    
    flist = sorted(glob.glob(dirin))
    fi2   = []; kk =0
    for fi in flist :        
        fi2.append(fi)
        if len(fi2) < 20 and fi != flist[-1] : continue
        log(fi2)    
        df  = pd_read_file( fi2, cols= cols, drop_duplicates=colsu,
                            n_pool=20, verbose=False)
        log(df.shape)
        fi2=[]
                
        df         = df.drop_duplicates(colsu)        
        df['siid'] = df.apply(lambda x : f"{int(x['shop_id'])}_{int(x['item_id'])}"  , axis=1)
        del df['shop_id'] ; del df['item_id']
        
        pd_to_file(df, dirout + f"/df_map_{tag}_{kk}.parquet", show=1 )    
        kk = kk + 1

        
def db_load_dict(df, colkey, colval, verbose=True):
    ### load Pandas dataframe and convert into dict   colkey --> colval
    if isinstance(df, str):
       dirin = df
       log('loading', df)
       df = pd_read_file(dirin, cols= [colkey, colval ], n_pool=3, verbose=True)
    
    df = df.drop_duplicates(colkey)    
    df = df.set_index(colkey)
    df = df[[ colval ]].to_dict()
    df = df[colval] ### dict
    if verbose: log('Dict Loaded', len(df))
    return df
    


########################################################################################################
########DiskCache Utilities ############################################################################
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

def diskcache_keycount(cache):
    cache = diskcache_load( cache, size_limit=100000000000, verbose=False )

    v = cache._sql('SELECT countt(key) FROM Cache').fetchall()
    v = [ t[0] for t in v]
    return v


def diskcache_getall(cache, limit=1000000000):
    cache = diskcache_load( cache, size_limit=100000000000, verbose=False )  
    v = cache._sql( f'SELECT key,value FROM Cache LIMIT {limit}').fetchall()
    v = { t[0]: t[1] for t in v }
    return v

def diskcache_get(cache, key, defaultval=None):
    ss = f'SELECT value FROM Cache WHERE key="{key}" LIMIT 1'
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
    
    



