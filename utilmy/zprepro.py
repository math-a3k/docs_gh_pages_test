# -*- coding: utf-8 -*-
HELP ="""



"""
if 'import':
    import warnings ;warnings.filterwarnings("ignore")
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    with warnings.catch_warnings():
        # filter sklearn\externals\joblib\parallel.py:268:
        # DeprecationWarning: check_pickle is deprecated
        warnings.simplefilter("ignore")
        import os, glob, sys, math, string, time, json, logging, functools, random, numpy as np, pandas as pd,  matplotlib, scipy, h5py, yaml, operator, gc
        import copy, mmh3
        from itertools import chain, zip_longest
        from pathlib import Path; from collections import defaultdict, OrderedDict ;
        #import matplotlib.pyplot as plt
        import scipy.stats ; from scipy.stats import norm

        import zlocal, zlib, pyarrow as pa
        from utilmy import pd_read_file, pd_to_file,  os_makedirs, pd_read_file2

        from util import log, log_pd, to_file
        # def log(*s): print(*s, flush=True)
        from box import Box


    sys.path.append( "/home/noelkevin01/test_code/" )
    
    PYFILE = "python prepro_prod.py"

    def log(*s):
        print(*s, flush=True)

    def log2(*s):
        if random.random() > 0.999 : print(*s, flush=True)

    def diskcache_load( db_path="", size_limit=100000000000, verbose=1 ):
        """ val = cache[mykey]
        """
        import diskcache as dc
        cache = None
        cache = dc.Cache(db_path, size_limit= size_limit )
        if verbose> 0: print('Cache size/limit', len(cache), cache.size_limit )
        return cache


    #### REMOTE Global Params ##########################################################################
    if  'coo' :
        flag  = 'sim'
        newid        =  9
        N_ROWS       =  599959995999 # 9999599944 # 99995999599
        N_BUCKET     =  499 # 499  # 499
        data_type    = "brw"  #  'brw'
        T_MIN, T_MAX = 18000, 18566

        tag_coo      =  f'n{N_BUCKET}'
        path_data    =  f"/a/adigcb301/ipsvols05/pydata/{data_type}_ran_v14/*/*"
        out_path_coo =  f"/a/adigcb301/ipsvols05/pydata/zranking_l3/out/baseline/{data_type}/{tag_coo}/"



####################################################################################################
if 'path':
    dirs = Box({})
    dirs.pur_dir      = "/a/acb401/ipsvols06/pydata/pur_ran_v15/"
    dirs.brw_dir      = "/a/acb401/ipsvols06/pydata/brw_ran_v15/"   ## User History
    # dirs.clk_dir      = "/a/acb401/ipsvols06/pydata/brw_item_v15/"  ### ~last 120 days of click, Only ITEMS

    dirs.sc_clk_dir   = "/a/acb401/ipsvols06/pydata/sc_widget_clk/"  ####  ~last 400 days of click
    dirs.sc_pur_dir   = "/a/acb401/ipsvols06/pydata/sc_widget_pur/"  ###   ~last 400 days of purchase
    dirs.sc_imp_dir   = "/a/acb401/ipsvols06/pydata/sc_widget_imp/"  ###   ~last 120 days of impression

    pur_dir      = "/a/acb401/ipsvols06/pydata/pur_ran_v15/"
    brw_dir      = "/a/acb401/ipsvols06/pydata/brw_ran_v15/"   ## User History
    # dirs.clk_dir      = "/a/acb401/ipsvols06/pydata/brw_item_v15/"  ### ~last 120 days of click, Only ITEMS

    sc_clk_dir   = "/a/acb401/ipsvols06/pydata/sc_widget_clk/"  ####  ~last 400 days of click
    sc_pur_dir   = "/a/acb401/ipsvols06/pydata/sc_widget_pur/"  ###   ~last 400 days of purchase
    sc_imp_dir   = "/a/acb401/ipsvols06/pydata/sc_widget_imp/"  ###   ~last 120 days of impression


    dir_scitems = "/a/adigcb301/ipsvols05/pydata/sc_items/sc_campaign_items.tsv"
    dir_pydata  = "/a/acb401/ipsvols06/pydata/"

    dir_ngsiid  = "//a/acb401/ipsvols06/pydata/sc_block_user_item/"
    dir_ngsiid2 = "/a/gfs101/ipsvols07/ndata/cpa/hdfs/sc_block_user_item/"


    #dir_rec = zlocal.dir_rec
    dir_rec     = '/data/workspaces/takos01/cpa/'
    dir_out0    = '/data/workspaces/noelkevin01/cpa/'


    dir_hive    = "/a/adigcb204/ipsvolh03/ndata/cpa/input"
    dir_res     = "/a/adigcb204/ipsvolh03/ndata/cpa//res/"

    dir_norec   = "/a/adigcb204/ipsvolh03/ndata/cpa/res/norec.sh"
    dir_nohist  = "/a/adigcb204/ipsvolh03/ndata/cpa/res/nohist.sh"


    dir_rec3    = '/a/gfs101/ipsvols07/ndata/cpa/'   ### 66 Tb
    dir_cpaloc  = "/data/workspaces/takos01/cpa/"
    dir_cpa3    = '/a/gfs101/ipsvols07/ndata/cpa/'    ### 3 Tb


if 'path_ca':
    dir_ca       = '/a/gfs101/ipsvols07/ndata/cpa/ca_check/'
    dir_ca_hdfs  = dir_ca + "/hdfs//"
    dir_ca_stats = dir_ca + "/stats/"
    dir_ca_daily = dir_ca + "/daily/"

    ### New items daily
    dir_ca_dailyitem = "/a/adigcb301/ipsvols05/pydata/sc_items/sc_campaign_items.tsv"

    table_ca_all_100d = "nono3.ca_daily_coupons10c"  ### 100 recent days ranid
    table_ca_all_300d = "nono3.ca_daily_coupons11c"  ### all days with ranid

    table_ca_100d = "nono3.ca_daily_coupons10b"  ### 100 recent days NO ranid
    table_ca_300d = "nono3.ca_daily_coupons11"   ### all days with   NO ranid

    dir_export = "/a/gfs101/ipsvols07/ntmp/export/ca/rec/"
    dir_ca_prod = "/a/acb401/ipsvols06/sc-coupon-advance-batches/"

    flog_warning = "/a/gfs101/ipsvols07/ndata/cpa/log/log_gpu/aaa_Warning.py"



if 'cols':
    ### can be used for training
    cols_scstream_item_vec_df = ['genre_name_path', 'item_emb', 'item_id', 'item_name', 'item_text','price', 'review_num', 'shop_id', 'shop_name', 'siid']
    cols_scstream_item = ['genre_id', 'genre_name_path', 'genre_path', 'image_url', 'item_id', 'item_name', 'price', 'ran_cd', 'review_avg', 'review_num',
                         'shop_id', 'shop_name', 'siid']

    cols_pur   = ['basket_id', 'easy_id', 'gbuy_flg', 'genre_id', 'item_id', 'order_number', 'postage', 'price', 'ran_id', 'sender_zip', 'series_id', 'sg_id',
                  'shop_id', 'super_auction_flg', 'tax', 'units', 'unix_time']

    cols_brw      = ['easy_id', 'genre_id', 'item_id', 'ran_id', 'ref', 'ref_type', 'series_id', 'sg_id', 'shop_id', 'time_key'],
    cols_item     = ['basket_id', 'easy_id', 'gbuy_flg', 'genre_id', 'item_id',  'order_number', 'postage', 'price', 'ran_id', 'sender_zip', 'series_id',
                     'sg_id', 'shop_id', 'super_auction_flg', 'tax', 'units', 'unix_time']

    cols_sc_imp   = ['channel', 'easy_id', 'item_id', 'logic_hash', 'query', 'shop_id',    'sid', 'timestamp']
    cols_sc_clk   = ['channel', 'easy_id', 'item_id', 'logic_hash', 'query', 'shop_id',   'sid', 'timestamp']
    cols_sc_pur   = ['channel', 'discount', 'easy_id', 'item_id', 'logic_hash', 'price', 'shop_id', 'timestamp', 'units' ]


    cols_itemv15 = ['genre_id', 'item_id', 'ran_id', 'series_id', 'sg_id', 'shop_id']   ### int64
    cols_itemv16 = ['genre_id', 'item_id', 'ran_id', 'series_id', 'sg_id', 'shop_id']   ### int64




#### DB Path   ############################################################################
if 'db diskcache':
    db_easyid_topgenre_pur = {}; db_itemid_vran= {}; db_ca_genre_siid= {};
    db_easyid_topgenre_brw = {}; db_easyid_topgenre_intra= {} ; db_easyid_topgenre_merge = {};
    db_ca_siid_genre = {}; db_imaster = {}; db_ivector = {} ; db_uvector = {}
    ca_genreid_global = None  ; clientdrant = None ; ok_global = None

    #db_dir = "/data/workspaces/takos01/cpa/db/"
    db_dir  = "/sys/fs/cgroup/cpa/db"

    db = Box({
        #'db_itemtag_items_path'  :  f"{db_dir}/map_sampling_itemtag_siid.cache",
        #'db_itemid_itemtag_path' :  f"{db_dir}/map_itemid_itemtag.cache",
        # 'db_easyid_hist_path'    :  f"{db_dir}/easyid_hist_day_.db",
        #'db_easyid_group_path'   :  f"{db_dir}/easyid_group.db",
        # 'db_easyid_rec_path'     :  f"{db_dir}/easyid_rec.db",
        #'db_timekey_easyidlist_path' : f"{db_dir}/timekey_easyidlist.db",

        'db_itemid_vran_path' : f"{db_dir}/db_itemid_vran.db",


        ### 2mio tag
        #'db_itemid_itemtag_2m_path' :  f"{db_dir}/map_itemid_itemtag_2m.cache",

        #### Local IMaster for check
        'db_imaster' :  f"{db_dir}/imaster.cache",   ####  IMaster cache
        'db_ivector' :  f"{db_dir}/ivector.cache",   #### Item embedding cache

        'db_uvector' :  f"{db_dir}/uvector.cache",   #### User embedding cache


         #### Topk per itemid
        'db_item_toprank' : f"{db_dir}/db_item_toprank.db",

        #### Genre Rec
        'db_ca_genre_siid'       :  f"{db_dir}/db_ca_genre_siid.cache",
        'db_ca_siid_genre'       :  f"{db_dir}/db_ca_siid_genre.cache",

        'db_easyid_topgenre_pur' :    f"{db_dir}/db_easyid_topg_pur.cache",
        'db_easyid_topgenre_brw' :    f"{db_dir}/db_easyid_topg_brw.cache",
        'db_easyid_topgenre_intra' :  f"{db_dir}/db_easyid_topg_intra.cache",    ### Intra updates

        'db_easyid_topgenre_freqpur' :  f"{db_dir}/db_easyid_topg_freqpur.cache",    ### Intra updates

        # 'db_easyid_topgenre_merge' :  f"{db_dir}/db_easyid_topg_merge.cache",   #### Hybrid scores
    })

    # db['db_easyid_hist_path'] = "/data/workspaces/takos01/cpa/db/easyid_hist_day_20210715.db"
    # db['db_easyid_hist_path'] = "/data/workspaces/takos01/cpa/db/easyid_hist_day_20210901.db"
    # db['db_easyid_hist_path'] = f"{db_dir}/easyid_hist_day_latest.db"

    def db_init(verbose=1):
        for key,val in db.items():
            name = key.replace("_path", "")
            log(name)  #, len(globals()[ name ] ))
            if   'imaster' in name :  globals()[ name ] = diskcache_load(val, size_limit=10 *  10**9,)
            elif 'ivector' in name :  globals()[ name ] = diskcache_load(val, size_limit=35 *  10**9,)
            elif 'uvector' in name :  globals()[ name ] = diskcache_load(val, size_limit=10 *  10**9,)

            elif 'itemid_vran' in name :     globals()[ name ] = diskcache_load(val, size_limit=10 *  10**9,)

            elif 'item_toprank' in name :  globals()[ name ] = diskcache_load(val, size_limit=10 *  10**9,)

            else :                    globals()[ name ] = diskcache_load(val, size_limit=100* 10**9)

    db_init()



    ##############################################################################################
    db_items_scfilter_keys  =  None


    def a(): return 0


    def from_timekey(time_key, fmt='%Y%m%d'):
       #   from_timekey(18500)    YYYMDD
       t = time_key * 86400
       return time.strftime(  fmt, time.gmtime(t) )



    def test2():
        """
        pur_18878 12020, pur_18879 13049, brw_18879 19722, brw_18878 285007
            no history for easyid 307711029,   no history for easyid 350183981,no history for easyid 437882277,
            no history for easyid 385680657, no history for easyid 431723304,no history for easyid 346710171,

        """

        val1 = db_easyid_hist.get(366260203, "")


        llist = db_timekey_easyidlist["pur_18879"]

        v = diskcache_getval(db_easyid_hist)
        print(str(v)[:500] )


        # print( db_easyid_hist.check() )
        ss = 'SELECT key,value FROM Cache WHERE key={key} LIMIT 1'

        for ii, k in enumerate(llist) :
           v = db_easyid_hist._sql( ss.format(key= k) ).fetchall()
           print(str(v)[:100])


        return 1
        #ll = diskcache_getkeys(db_easyid_hist)
        ll = {k:True for k in ll }
        print('histo', len(ll))
        # return 1


        llist = ['na'] + llist

        for ii, k in enumerate(llist) :
            print(ii, k , str(ll.get(k, None))[:100] )

            if not k in db_easyid_hist:
                print(k, 'not in')

            print(ii, k,  str(db_easyid_hist.get(k, None))[:100] )
            if ii > 5 : break

        return 1
        print('res', val1)
        for k in  db_easyid_hist :
            print(k,  str(db_easyid_hist[k])[:100] )
            return 1

        for i in range(0, 10) :
          print( i, 'neasy', len(db_easyid_group[i]), str(db_easyid_group[i])[:100] )


if 'diskcache' :
    def diskcache_load(db_path="./dbcache.db", size_limit=100000000000, timeout=20, shards=1,
                        sqlmode= 'fast', ttl=None, verbose=True):
        """ Create dict type on disk, < 100 Gb, shards>1 : disk spaced is BLOCKED in advance, so high usage
           Issue, uses too much of DISK
        """
        import time, random
        if shards == 1 :
           import diskcache as dc
           cache = dc.Cache(db_path, size_limit= size_limit, timeout= timeout )
        else :
           from diskcache import FanoutCache
           cache = FanoutCache( db_path, shards= shards, size_limit= size_limit, timeout= timeout )
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
            # if i % 500000 == 0 : time.sleep(tbreak)  #### Ensure WAL does not grow too big, by pausing all writers
            try :
               cache[ v[i,0] ] = v[i,1]
            except :
               time.sleep(2.0)
               cache[ v[i,0] ] = v[i,1]

        print('Cache size', len(cache) )
        return cache


    def diskcache_save2(df, colkey, colvalue, db_path="./dbcache.db", size_limit=100000000000, timeout=20, shards=1, npool=10,
                        sqlmode= 'fast', ttl=None, verbose=True):
        """ Create dict type on disk, < 100 Gb, shards>1 : disk spaced is BLOCKED in advance, so high usage
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
           if ttl is None :
               for i, vi in enumerate(vlist) :
                  try :
                     cache[ vi[0] ] = vi[1]
                  except :
                     time.sleep(3.0)
                     cache[ vi[0] ] = vi[1]
           else :
               for i, vi in enumerate(vlist) :
                  try :
                     cache.set( vi[0], vi[1], expire= ttl)
                  except :
                     time.sleep(10.0)
                     cache.set( vi[0], vi[1], expire= ttl)


        # from utilmy.parallel import multithread_run
        n = len(v)
        nbefore =  len(cache)
        log('Cache size:', nbefore )
        mblock = 2000000
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

        #diskcache_config(db_path, task='commit')
        if sqlmode != 'commit':
           diskcache_config(db_path, task='fast')


    def diskcache_getkeys(cache):
        if isinstance(cache, str):
            cache = diskcache_load( db_path= cache, size_limit=100000000000, verbose=0 )
        v = cache._sql('SELECT key FROM Cache').fetchall()
        v = [ t[0] for t in v]
        return v


    def diskcache_getitems(cache, n=10):
        if isinstance(cache, str):
            cache = diskcache_load( db_path= cache, size_limit=100000000000, verbose=0 )
        v = cache._sql( f'SELECT key,value FROM Cache LIMIT {n}').fetchall()
        v = { t[0]: t[1] for t in v }
        return v


    def diskcache_dumpdisk(cache=None, dirout="/", cols=None, tag="dcache" , n=1000000000, kbatch=2000000):
        """ python prepro_prod.py  diskcache_dumpdisk --cache db_ca_siid_genre &
           dirout = dir_cpa3 + "map/map_siid_ranid_dump/"
           cols = ['siid', 'ran_id' ]
           kbatch= 1000 ;    n = kbatch *3
           cache = db_itemid_vran

        """
        if  'itemid_vran' in cache :
           dirout = dir_cpa3 + "/map/map_siid_ranid_dump/"
           cols   = ['siid', 'ran_id' ]
           cache  = db_itemid_vran

        if 'db_ca_siid_genre' in cache:
           dirout = dir_cpa3 + "/map/map_siid_genreid_dump/"
           cols   = ['siid', 'genre_id' ]
           cache  = db_ca_siid_genre


        cols   = ['key', 'value'] if cols is None else cols
        os.makedirs(dirout, exist_ok=True)
        if isinstance(cache, str):
            cache = diskcache_load( db_path= cache, size_limit=100000000000, verbose=0 )
        c = cache._sql( f'SELECT key,value FROM Cache LIMIT {n}')

        iimax = int(n//kbatch)+1
        dfall = pd.DataFrame() ; ii = -1
        while ii < iimax:
            v  = c.fetchmany(kbatch)              # each batch contains up to 100 row
            if not v: break
            ii = ii +1
            v  = [ (t[0], to_int(t[1]) ) for t in v ]
            v  = pd.DataFrame(v, columns=cols)
            v.to_parquet(dirout + f"/{tag}_{ii}.parquet")
            log(dirout + f"/{tag}_{ii}.parquet", v.shape)


    def diskcache_expire(cache=None, **kw ):
        """ python prepro_prod2.py  diskcache_expire --cache db_ca_siid_genre &
        cache = "db_ca_siid_genre"
        """
        if 'db_ca_siid_genre' in cache:
           dirin  = dir_cpa3 +"/hdfs/items/*.parquet"
           cols=['shop_id', 'item_id']
           colkey = 'siid'  ; colg ='genre_id'

        tmin  = get_timekey() - 30 # 18960
        tmax  = tmin
        flist = glob_glob(dirin, 10, tmin=tmin, tmax=tmax)
        log(flist)
        if len(flist) < 1: log('no files') ; return 1

        df = pd_read_file2(flist, cols=cols, drop_duplicates=cols)
        df = df.drop_duplicates()
        df = pd_add_siid(df, delete=True)
        df = df[['siid']]
        df['genre_id'] = 0
        diskcache_save2(df,db_path=  db[cache] , colkey= colkey, colvalue= colg,  npool=4, verbose=False , ttl = 1000 )





    def diskcache_topandas(cache=None, cols=None, n=1000000000, kbatch=1000000):
        """ python prepro_prod.py  diskcache_dumpdisk --cache itemid_vran

        """
        if  'itemid_vran' in cache :
           cols   = ['siid', 'ran_id' ]
           cache  = db_itemid_vran

        cols   = ['key', 'value'] if cols is None else cols
        os.makedirs(dirout, exist_ok=True)
        if isinstance(cache, str):
            cache = diskcache_load( db_path= cache, size_limit=100000000000, verbose=0 )
        c = cache._sql( f'SELECT key,value FROM Cache LIMIT {n}')

        iimax = int(n//kbatch)+1
        dfall = pd.DataFrame() ; ii = -1
        while ii < iimax:
            v  = c.fetchmany(kbatch)              # each batch contains up to 100 row
            if not v: break
            ii = ii +1
            v  = [ (t[0], t[1] ) for t in v ]
            v  = pd.DataFrame(v, columns=cols)
            dfall = pd.concat((dfall, v))
            log(dfall.shape)
        return dfall



    def diskcache_del( db_path="./dbcache.db", verbose=True):
        """ Delete DB File
        """
        from utilmy import os_remove
        log('Removing', db_path)
        try :
           # os.system( f'rm -rf "{db_path}" ' )
           os_remove(db_path)
        except Exception as e:
           log(db_path, e)


    def diskcache_get(cache, key, defaultval=None):
        ss = f'SELECT value FROM Cache WHERE key={key} LIMIT 1'
        log(ss)
        v  = cache._sql( ss ).fetchall()
        if len(v) > 0 :
            return v[0][0]
        return defaultval

    def diskcache_check(cache, limit=100):
        v = cache._sql( f'SELECT key,value FROM Cache LIMIT {limit}').fetchall()
        v = { t[0]: t[1] for t in v }
        return v

    def diskcache_flush():
        flist = glob.glob(db_dir + "/*")
        for fi in flist:
            log(fi)
            diskcache_config(db_path=fi, task='commit')

    def diskcache_config(db_path=None, task='commit'):
        """ .open "/data/workspaces/takos01/cpa/db/e"    python prepro.py diskcache_config
        https://sqlite.org/wal.html

        PRAGMA journal_mode = DELETE;   (You can switch it back afterwards.)
        PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode = WAL;
        PRAGMA wal_checkpoint(FULL);

        """
        db_path = "/data/workspaces/takos01/cpa/db/map_itemid_itemtag_2m.cache" if db_path is None else db_path
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

        elif task == 'vacuum':  ### remove empty space
           ss = f""" VACUUM; """

        with open( 'ztmp_sqlite.sql', mode='w') as fp :
            fp.write(ss)
        log(ss)
        os.system( f'sqlite3  {db_path}/cache.db  ".read ztmp_sqlite.sql"    ')
        # time.sleep(20)
        # keys = diskcache_getkeys(cache)


    def size(path= None):
        if path is None : os.system( f" du -Sh  {db_dir}  -d 1 ")

        ### DB nb of elements
        flist = glob.glob(db_dir + "/*")
        for fi in flist:
            print(fi.replace(db_dir,""), len( diskcache_load(fi, verbose=0) ))

        log('\n\n easyid bucket: ', )
        #for key in db_timekey_easyidlist :
        #     log(key, len(db_timekey_easyidlist[key]) )


    from collections import OrderedDict
    class fixedDict(OrderedDict):
        """  fixed size dict
              ddict = fixedDict(limit=10**6)
        """
        def __init__(self, *args, **kwds):
            self.size_limit = kwds.pop("limit", None)
            OrderedDict.__init__(self, *args, **kwds)
            self._check_size_limit()

        def __setitem__(self, key, value):
            OrderedDict.__setitem__(self, key, value)
            self._check_size_limit()

        def _check_size_limit(self):
            while len(self) > self.size_limit:
                self.popitem(last=False)


################ easyid      ############################################################
if 'easyid' :
    def easyid_create_histo(path=None, verbose=1):   ## python prepro.py easyid_create_histo  &
        """
            easyid --> siid list  (after last purchase)
            easyid --> vran de-duplicated.
            6month click --> 287mio (easyid, siid),  28 Gb RAM

        genre_id      int64
        item_id       int64
        ran_id        int64
        ref          object
        ref_type     object
        series_id     int64
        sg_id         int64
        shop_id       int64
        time_key      int64

        """
        nrows   = 1005009000
        topk    = 10
        today   = date_now_jp(add_days=-1)
        past    = int( date_now_jp(add_days=-20) )
        npool   = 5

        path = 'hive'   ###     path = 'clk_dir'

        # rename()
        if path == 'hive' :
           # dir_in =  dir_hive + "/easyid_hist_top50_recent_20210901_6mth/"   ; npool=10
           dir_in = dir_hive + f"/easyid_hist_top50_recent_{today}/" ;
           npool  = 1
           cols   = [ 'easy_id', 'shop_id', 'item_id', 'ts', 'dateint'  ]  ### 'dateint'
           # fpath = dir_in + "/*01*.parquet"
           fpath_list = [dir_in + "/*_*.parquet" ]

        else :
           tk_today   = get_timekey()-1
           # dir_in   = brw_dir + f"/*/*{tk_today}*"
           dir_in     = dirs[ path ]
           npool      = 3
           cols       = [ 'easy_id', 'shop_id', 'item_id'  ]  ### 'dateint'
           fpath_list = [ dir_in + f"/{t}/*{tk_today}*" for t in range(0, 500)  ]


        def get_siid_list(dfi):
            dfi = ",".join( list(OrderedDict.fromkeys(dfi['siid'].values))[:topk] )    ## unique siids
            # dfi = ",".join( dfi['siid'].values[:topk] )     ## unique siids
            return dfi


        for fpath in fpath_list  :
            log(fpath, npool)
            df = pd_read_file(fpath, n_pool=npool, cols=cols, nrows=nrows, verbose=False)         ## 300 mio click
            # log(df, df.dtypes)
            df['easy_id'] = df['easy_id'].astype('int64')


            ### Hive: Only Recently updated  #############################
            if 'dateint' in df.columns :
               leasyid = list(df[ df[ 'dateint' ] >= past ]['easy_id'].unique() )
               df      = df[ df.easy_id.isin( leasyid ) ]
               df      = df.sort_values([ 'ts'], ascending=[0])    ### Decreating in time
               del df['ts'] ; del df['dateint']

            if len(df) < 1 :  log(df) ; return 0


            log('df shape', df.shape)
            df = pd_easyid_flatten_siid(df, get_siid_list)
            log('N easy_id',  len(df) )


            #### Test
            db_path   = db['db_easyid_hist_path']
            # db_path             = db['db_easyid_hist_path']  + 'test'


            #### Filter Empty  siid list  and not Empty
            df       = df[ df['siid_list'].apply(lambda x : len(x)> 0 ) ]

            #### Filter already Inserted
            # db_easyid_hist_keys = diskcache_getkeys( db_easyid_hist )
            # if verbose : log( 'Existing keys', len( db_easyid_hist_keys  ), str(db_easyid_hist_keys)[:50] )
            # df  = df[ df['easy_id'].apply(  lambda x :  True if x not in db_easyid_hist_keys else False  ) ]
            log('N easy_id to Insert',  df.easy_id.nunique() )
            log(df)
            #### Insert missing ones
            if len(df) > 0 :
              diskcache_save(df, colkey='easy_id', colvalue='siid_list', db_path=db_path)


    def easyid_get_histo(easyid, version="") :
        # if db_easyid_hist is None : db_easyid_hist = diskcache_load(db_easyid_hist_path)
        # return db_easyid_hist.get(easyid, [])
        ss = diskcache_get(db_easyid_hist, easyid, "")
        return ss.split(",")




###############  Utilss ###################################################################
if 'utils' :
    def time0():
        return int(time.time())

    def date_now_jp(fmt="%Y%m%d", add_days=0, add_hours=0, timezone='Asia/Tokyo'):
        # "%Y-%m-%d %H:%M:%S %Z%z"
        from pytz import timezone as tzone
        import datetime
        # Current time in UTC
        now_utc = datetime.datetime.now(tzone('UTC'))
        now_new = now_utc+ datetime.timedelta(days=add_days, hours=add_hours)

        if timezone == 'utc':
           return now_new.strftime(fmt)
        else :
           # Convert to US/Pacific time zone
           now_pacific = now_new.astimezone(tzone('Asia/Tokyo'))
           return now_pacific.strftime(fmt)


    def test4():
        """
       channel  discount    easy_id   item_id   price  shop_id   timestamp  units  pred_rank     model

        8  search     190.0  222803971  10000111   4180.0   360250  1626309020      1         -1  pur_kev1
        22  search      65.0  308456081  10000383   1430.0   361393  1626355231      1         -1  pur_kev1
        21  search      95.0  308456081  10000189   2101.0   361393  1626355231      1         -1  pur_kev1
        3   search     199.0  309956982  10000506   3980.0   318558  1626279152      1         -1  pur_kev1

        34  search     458.0  311232515  10001089  11460.0   307448  1626360896      1         -1  pur_kev1
        35  search     203.0  311232515  10000590   4970.0   307448  1626360896      1         -1  pur_kev1
        'db_easyid_hist_path'  : "/data/workspaces/takos01/cpa/db/easyid_hist_day_.db",
        'db_easyid_group_path' : "/data/workspaces/takos01/cpa/db/easyid_group.db",
        'db_easyid_rec_path'   : "/data/workspaces/takos01/cpa/db/rec_easyid_v0.db",

        18823 : 07/15
        """
        log(from_timekey(18823) )
        easyid = 311232515
        # v    = db_easyid_rec[easyid]

        v    = db_easyid_hist[easyid]


        return v


    def pd_easyid_flatten_siid(df, fun_flatten_siid, colgroup = 'easy_id', colist='siid_list'):
        ### easyid --->  siid_list
        import gc
        df = df[ -df[colgroup].isna() ]
        df = df[ (df.item_id > 0) & (df['shop_id'] > 0 ) ]
        for ci in  [ 'shop_id',  'item_id' ] :
           df[ ci ] = df[ci].astype('int64')


        df['siid']    = df.apply(lambda x :   f"{x['shop_id']}_{x['item_id']}", axis=1 )
        del df['shop_id'] ; del  df['item_id'] ; gc.collect()
        df         = df.groupby(colgroup ).apply(lambda dfi : fun_flatten_siid(dfi) ).reset_index()
        df.columns = [colgroup , colist]
        return df


    ###############################################################################
    def rename():   ## python prepro.py rename
         # flist  = sorted( list(set(glob.glob( dir_cpa1 + "/input/*/*" ))) )
         flist = []
         flist += sorted( list(set(glob.glob( dir_cpa2 + "/input/*/*" ))) )
         flist += sorted( list(set(glob.glob( dir_cpa2 + "/input/*/*/*" ))) )


         for fi in flist :
            if ".sh" in fi or ".py"  in fi or "COPY" in fi :
                continue

            if  '.parquet' in fi.split("/")[-1]  or os.path.isfile(fi)    : continue

            if '.' not in fi.split("/")[-1] :
                try :
                  log(fi)
                  os.rename(fi, fi + ".parquet")
                except Exception as e :
                  log(e)


    def rename2():   ## python prepro.py rename
         flist = sorted( list(set(glob.glob( dir_hive + "/ichiba_order_20210901b_itemtagb2/*" ))) )

         for fi in flist :
            if ".sh" in fi or ".py"  in fi or "COPY" in fi :
                continue

            if '.parquet'  in fi   :
                try :
                  log(fi)
                  os.rename(fi, fi.replace(".parquet", "") )
                except Exception as e :
                  log(e)


    def schema():   ###   python prepro.py test   |& tee -a schema0.sh
        dir_hive = "/a/adigcb201/ipsvols03/ndata/cpa/input/"
        dir_in   = dir_hive  + "/ichiba_hist_202101_20210812_agg2b/"

        flist = glob.glob( dir_hive + "/*/000000_0*" )

        log(flist)
        for fi in flist :
           try :
             log(fi)
             df    = pd.read_parquet(fi)
             print(df, df.columns)
           except: pass


    def test():   ## python prepro.py test
        dir_hive = "/a/adigcb201/ipsvols03/ndata/cpa/input/"
        dir_in   = dir_hive  + "/ichiba_hist_202101_20210812_agg2b/"

        flist = glob.glob( dir_in + "/*" )
        log(dir_in) # , flist)
        df    = pd_read_file(dir_in + "/*")
        log(df)


    def daily():   ## python prepro.py daily
        """20210702 (13 027 042, 9)
           20210814 (1092611, 9)
        """
        dir_in =  dir_hive  + "/ichiba_hist_202107_20210812b_vran/"

        os.makedirs( dir_in + "/daily/", exist_ok=True)
        df = pd.read_parquet(dir_in +"/000000_1")
        log(df)

        dlist = list(df['dateint'].unique())

        for ti in dlist :
            log(ti)
            dfi = df[ df.dateint == ti]
            log(dfi.shape)
            dfi.to_parquet(dir_in + f"/daily/pur_clk_{ti}.parquet")

    ##############################################################################
    def test5():
       df = pd.DataFrame(np.random.randint(0, 10, size=(50, 4)),
                         columns=list('abcd'))


       def fun1(dfi):
            return  dfi['c'].sum()  + dfi['d'].sum()

       dfg = pd_apply_parallel(df, colsgroup=None, fun_apply=fun1, npool=5)
       log(df)
       log(dfg)


    def path_getag(x, minsize=6):    ### Return  single tag for folder creation, parse from end of file
        z  = x.split("/")
        ss = ""
        for i in range(len(z)-1, -1, -1) :
            if len(z[i]) <= 0 : continue
            elif "*" in z[i]  : continue
            else :
                ss = z[i] + "_" + ss  if ss != "" else z[i]
                log(ss)
            if len(ss)> minsize : return ss
        return ss


    def os_system(cmd):
        log("####\n", cmd, "\n")
        os.system(cmd)




########  Word2vec  ########################################################################
if 'word2vec' :
    def c():
        dir_in =  "/a/adigcb201/ipsvols03/ndata/cpa/input/ichiba_clk_202102_202108_itemtagb2_202106_08/000000_0.parquet"
        df = pd.read_parquet(dir_in  )
        log(df, df.dtypes)
        log(df.easy_id.nunique() , )


    def check3():
        sid = "202242_10780283"
        log(sid,  db_itemid_vran.get(sid, 'no')   )
        # return 1

        keys     = diskcache_getitems(db_itemid_vran, 10)
        log( keys)
        return 1



    def itemtag_get(shopid, itemid):

        ssid = f"{shopid}_{itemid}"
        tag  = db_itemtag_v2(siid, "")

        if len(tag) > 0 :
           ran_id = db_itemid_vran(siid, "")
           if len(ran_id) :
                return tag
           hran   = hash(ran_id) % 900
           hran   = hran if hran >= 100 else 900 + hran
           tag    = tag + f"_{hran}"
           return tag
        else :
           return ""


    def create_sequence(in_path="", out_path="", t_min=0, t_max=20211201,
                            seq_min= 3, mode='a', encoder='l3_int', n_pool=1, nmax=9995999599, mode_return='save_disk') :
        """
          87 968 984  itemtag_Vran  with 4 digits hash

           python  prepro.py  create_sequence   2>&1 | tee -a zlog_prepro.py

           --in_path ichiba_order_itemtag_20210526b    --out_path  seq_pur  --t_min 0  --t_max 20211201  --seq_min 3

          Minimum 3 sequences length
          encoder: Improve
             ichiba_order_itemtag_20210526b

        easy_id (int)
        ts (bigint)
        dateint (int)
        item_tag (string)
        easyid_group (int)
             data/workspaces/takos01/cpa/ichiba_order_itemtag_20210526b

        shop_id           int32
        item_id           int32
        ts                int64
        item_tag_vran    object



        """
        dovran = False
        nmax     = int( 10**9 )
        # in_tag   = "ichiba_clk_itemtag_202101_202105"
        # in_tag   = "ichiba_clk_202102_202108_itemtagb2_202106_08"
        # in_tag   = "ichiba_clk_202006_202012d_itemtagb_202009_12"
        # in_tag = "ichiba_clk_202102_202108_itemtagb2_202106_08b"
        # in_tag = "ichiba_clk_202006_202012d_itemtagb2_202009_12b"

        #in_tag = "ichiba_order_20210901b_itemtagb2"
        #in_tag = "ichiba_order_20210901b_itemtagb2"
        in_tag = "ichiba_order_20211001_full3_201901_bkt_part2"

        # coltag   = 'item_tag_vran'
        coltag   = 'siid'

        in_path  = dir_hive + "/" + in_tag

        dir_out  = dir_emb
        out_path = dir_out + '/seq/' + in_tag

        tag  = f"{nmax}"

        log(out_path)
        os.makedirs(out_path, exist_ok=True)
        colt = 'dateint'
        cols = [ 'easy_id',   'ts',  coltag   ]
        # cols = [ 'easy_id',   'ts',  coltag , 'shop_id', 'item_id'  ]
        # cols = [ 'easy_id',   'ts', 'dateint', coltag , 'shop_id', 'item_id'  ]
        log("input", in_path)

        flist = sorted(glob.glob( in_path + "/*/*" ))
        flist = flist[:20000]

        def jjoin(v):
            ss, t1 = "", ""
            for t in v:
                if t == t1 :continue   ### siid should be different consecutively
                ss += " " + str(t)
                t1  = t
            return ss

        if dovran :
          log('load mapping keys')
          db_itemid_vran2 =  diskcache_getitems(db_itemid_vran, n=10**9)


        log('N files:', len(flist)  )
        jj = 0; df = None
        for ii, fi in enumerate(flist) :
            coltag = 'siid'
            log(ii, fi)
            dfi = pd.read_parquet(fi, columns= [ 'easy_id', 'ts', coltag  ])
            dfi = dfi.iloc[:nmax, :]
            df  = dfi if df is None else pd.concat((df, dfi))
            jj  = jj + 1

            if len(df) < 2*10**7 and (ii != len(flist)-1) : continue
            jj = 0
            del dfi ; gc.collect()

            log("\n\n########", ii, df.shape )
            if len(df) < 1 : return False
            # df  = df[ (df.easy_id > 0) & (df.item_id > 0 )]
            # df  = df.fillna('')
            df    = df.sort_values(['easy_id', 'ts' ])
            # df  = df[ (df[colt]  < t_max )  & ( df[colt]  >= t_min )  ]
            # del df[colt];  gc.collect()
            # df['item_tag'] = df[[ 'shop_id', 'item_id'  ]].apply(lambda x: itemtag_get(x['shop_id'], x['item_id']), axis=1 )


            log("\n#############  creating sequence per easy_id #########################")
            df  = df[['easy_id', coltag ]]
            df  = df[ -df[coltag].isna() ]


            #### ranid :
            # if dovran :
            log('map')
            df['ranid']= df['siid'].apply(lambda x : db_itemid_vran2.get(x, 0) )
            coltag = 'ranid'

            log('groupby')
            #dfu = df.groupby('easy_id').apply(lambda dfi :   " ".join(dfi[coltag])  )
            dfu = df.groupby('easy_id').apply(lambda dfi :   jjoin(dfi[coltag])  )
            del df; gc.collect()

            dfu         = pd.DataFrame(dfu, columns=['seq'])
            dfu['nlen'] = dfu['seq'].apply(lambda x : len(x.split(" "))  )       ## Size of sequence
            dfu         = dfu[dfu['nlen'] > seq_min ]
            del dfu['nlen']
            log_pd( dfu[[  'seq'  ]] )
            if mode_return == 'save_disk' :
               write_retry(dfu[[  'seq'  ]], out_path + f'/seq_{tag}.txt', mode=mode, retry=3)
            df = None


    def sequence_add_samples():
        """
           itemid --> list of itemids with higher proba
                  - 8 sammples       i0  + 8 sample

                  eaysId + 10**8, block of 50
           df['a', 'b', cocount ]

           sample(list)
        df['a'], df['b']

        """
        def add_sample():
            res  = ",".join( np.random.choice(dfi['b'].values, nmax, p=  dfi['c'].values / dfi['c'].sum() ) )
            res2 = np.random.choice(dfi['b'].values, nmax, p=  dfi['c'].values / dfi['c'].sum() )
            res2 = ",".join( res2[::-1] )
            res  = f"{res},{df['a'].values[0]},{res2}"
            return res

        df1 = df.groupby('a').apply(lambda dfi : add_sample(dfi)).reset_index()
        df1.columns = [ 'a', 'si']



    def create_vector(fname="seq_browsing_18566_full.txt", tag= "", dim_vec=50, autotune=True):
        """#### Fastext running, create Word2Vec Vectors

         python prepro.py   create_vector  &

         ST_CW_ECZ_DH_000_000_C_629

         ST_CW_ECZ_  DH_000_000 _C_629
         0123456789  0123456789 012345     Vector = sum(vi) + word

         16 char
        -19654520278256
        297840888242460002
        145483554626620169
        297840885117710000
        297840888242460002

         sleep  2 &&   python prepro.py   create_vector

          The following arguments for the dictionary are optional:
          -minCount           minimal number of word occurrences [1]
          -minCountLabel      minimal number of label occurrences [0]
          -wordNgrams         max length of word ngram [1]
          -bucket             number of buckets [2000000]
          -minn               min length of char ngram [0]
          -maxn               max length of char ngram [0]
          -t                  sampling threshold [0.0001]
          -label              labels prefix [__label__]

          The following arguments for training are optional:
          -lr                 learning rate [0.1]
          -lrUpdateRate       change the rate of updates for the learning rate [100]
          -dim                size of word vectors [100]

            -ws                 size of the context window [5]  ,  words +5 around, -5 around

          -epoch              number of epochs [5]
          -neg                number of negatives sampled [5]
          -loss               loss function {ns, hs, softmax} [softmax]
          -thread             number of threads [12]
          -pretrainedVectors  pretrained word vectors for supervised learning []
          -saveOutput         whether output params should be saved [0]


          /a/gfs101/ipsvols07/ndata/cpa/emb/seq/ichiba_order_20210901b_itemtagb2/ccountlog_ranid

        """
        dim_vec = 200    ###95000 tokens

        # fname    = "ichiba_clk_itemtag_202101_202105/seq_100000000.txt"
        #fname    = "ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur.txt"
        # fname   = "ichiba_order_20210901b_itemtagb2/aseq_1000000000_hash3.txt"
        # fname = "ichiba_order_20210901b_itemtagb2/aseq_split_hash3_01.txt"   ## 10 mio lines, 12gb

        #### Reference for order training
        ## fname = "ichiba_order_20210901b_itemtagb2/aseq_full_hash3.txt"   ## 10 mio lines, 12gb

        ### mnual
        #fname = "/ichiba_order_20210901b_itemtagb2/ccountlog_ranid/seq_pur_manual_ccount.txt"   ## 10 mio lines, 12gb
        fname = "/ichiba_order_20210901b_itemtagb2/ccount_ranid_merge/seq_pur_manual_ccount.txt"   ## 10 mio lines, 12gb

        #dir_data = "/a/adigcb204/ipsvolh03/ndata/cpa/emb/"
        #dir_out  = "/data/workspaces/takos01/cpa/"

        dir_data = "/a/gfs101/ipsvols07/ndata/cpa/emb/"
        dir_out  = "/data/workspaces/takos01/cpa/"

        tag = "v16merge_50n"

        pinput   = dir_data + f"/seq/{fname}"
        poutput  = dir_out  + f"/emb/{fname.replace('.txt', '')}{tag}/"


        poutput = poutput.replace("\\","/")
        os.makedirs(poutput,exist_ok=True)
        to_file("[config]", poutput +"/config.txt")
        if not os.path.isfile(pinput) :
            raise Exception("input Not a file, " + pinput)

        #pfast  = "/data/workspaces/noelkevin01/cpa/nono_exe"  #zlocal.dir_rec_fastexe #  "C:\D\gitdev\bin\fasttext\fasttext.exe "
        #pfast   = " ../../../../tensorflow-lightgbm-transformers "  #zlocal.dir_rec_fastexe #  "C:\D\gitdev\bin\fasttext\fasttext.exe "
        pfast   = " ../../../../../tensorflow-lightgbm-transformers "  #zlocal.dir_rec_fastexe #  "C:\D\gitdev\bin\fasttext\fasttext.exe "


        if os.name =='nt' :
            cmd  = f"cd {poutput} && {pfast}  skipgram   -dim  {dim_vec} -verbose 10  "
            cmd += f" -input  {pinput}  -output model    "
            # cmd +=  " -autotune-validation  -autotune-duration  10 "
            cmd += f"   >> log.txt 2>&1 "   ### Log

        else :
            cmd  = f" cd {poutput} && {pfast}   skipgram   "

            cmd +=  f" -dim  {dim_vec}   -verbose 10  -output model   -thread  30 "
            # cmd += f"  -neg 12  -minn 0  -maxn 0 "
            cmd +=  f"  -epoch 5  -lr  0.1 "
            # cmd +=  f"  -neg 5    -minn 25   -maxn 25  -ws 5    -bucket 9900000  "  ### small nmin --> very long
            cmd +=  f"  -neg 50    -minn 18   -maxn 18  -ws 15    -bucket 12100100  "  ### small nmin --> very long
            # cmd += f"  -neg 20  -minn    "
            # cmd += f" -autotune-validation  {val_file}  -autotune-duration  100 " if autotune else ""
            cmd +=  f" -input  {pinput}    "
            cmd +=  f"  2>&1 | tee -a log_train_4.txt "   ### Log

        log(cmd)
        to_file(cmd, poutput +"/config.txt")
        os_sleep_cpu(cpu_min=60, sleep=120, msg = 'create_vector' )  ### Wait Before Start
        os.system(cmd)
        log("\n" + poutput)

        #### Move files to
        backup1()


    def create_vector_parquet(dirin=None, dirout=None, skip=0, nmax=10**8):   ##   python prepro.py   create_vector_parquet  &
        #### FastText/ Word2Vec to parquet files    9808032 for purhase
        nmax = 10**8
        #dirin  = dir_cpaloc +     "/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/model.vec"
        if dirin is None :
           dirin  = dir_cpaloc +  "/emb/ichiba_order_20210901b_itemtagb2/seq_1000000000/model.vec"

        #dirout = dir_cpa2   + "/emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/"
        dirout = dir_cpa2   + "/emb/emb/" +  "/".join(dirin.split("/")[-3:-1])  +"/df/"
        log(dirout) ; os.makedirs(dirout, exist_ok=True)  ; time.sleep(4)

        def is_valid(w):
            return len(w)> 5  ### not too small tag

        i = 1; kk=-1; words =[]; embs= []; ntot=0
        with open(dirin, mode='r') as fp:
            while i < nmax+1  :
                i  = i + 1
                ss = fp.readline()
                if not ss  : break
                if i < skip: continue

                ss = ss.strip().split(" ")
                if not is_valid(ss[0]): continue

                words.append(ss[0])
                embs.append( ",".join(ss[1:]) )

                if i % 200000 == 0 :
                  kk = kk + 1
                  df = pd.DataFrame({ 'id' : words, 'emb' : embs }  )
                  log(df.shape)
                  pd_to_file(df, dirout + f"/df_emb_{kk}.parquet", show=0)
                  ntot += len(df)
                  words, embs = [], []

        kk    = kk + 1
        df    = pd.DataFrame({ 'id' : words, 'emb' : embs }  )
        ntot += len(df)
        dirout2 = dirout + f"/df_emb_{kk}.parquet"
        pd_to_file(df, dirout2, show=1 )
        log('ntotal', ntot )
        return os.path.dirname(dirout2)


    def faiss_create_index(df_or_path=None, colemb='emb', colid = 'id', dirout=None,  faiss_type = "IVF4096,Flat", nfile=1000, emb_dim=200, faiss_pars={}):
        """  1 billion size vector creation
             py   faiss_create_index      2>&1 | tee -a log_faiss.txt
                     #dir_cpa2 + "/emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/faiss/"
        """
        import faiss
        cc = Box(faiss_pars)
        # colid = 'id';  colemb='emb'

        if df_or_path is None :  df_or_path = dir_cpa2 + "/emb/emb//ichiba_order_20210901b_itemtagb2/seq_1000000000/df/*.parquet"
        if dirout is None :      dirout     =  "/".join( os.path.dirname(df_or_path).split("/")[:-1]) + "/faiss/"


        os.makedirs(dirout, exist_ok=True) ;
        log('dirout', dirout)
        log('dirin',  df_or_path)  ; time.sleep(5)

        if isinstance(df_or_path, str) :
           log('Loading', df_or_path)
           flist = glob_glob(df_or_path, nfile)
           df    = pd_read_file(flist, n_pool=20, verbose=False)
        else :
           df = df_or_path

        # df  = df.iloc[:9000, :]
        log(df)

        if colid == 'siid' and colid not in df.columns :
            df['siid'] = df.apply( lambda x : siid(x) , axis=1)


        tag       = f"_" + str(len(df))
        df        = df.sort_values(colid)
        df['idx'] = np.arange(0,len(df))
        pd_to_file( df[[ 'idx', colid ]].rename(columns={"id":'item_tag_vran'}),
                    dirout + f"/map_idx{tag}.parquet", show=1)   #### Keeping maping faiss idx, item_tag


        log("### Convert parquet to numpy   ", dirout)
        X  = np.zeros((len(df), emb_dim  ), dtype=np.float32 )
        vv = df[colemb].values
        del df; gc.collect()
        for i, r in enumerate(vv) :
            try :
              vi      = [ float(v) for v in r.split(',')]
              X[i, :] = vi
            except Exception as e:
              log(i, e)

        log("#### Preprocess X")
        faiss.normalize_L2(X)  ### Inplace L2 normalization
        log( X )

        nt = min(len(X), int(max(400000, len(X) *0.075 )) )  ### Training Size
        Xt = X[ np.random.randint(len(X), size=nt),:]
        log('#### Nsample training', nt)

        ####################################################
        D = emb_dim   ### actual  embedding size
        N = len(X)   #1000000

        if len(cc) < 1:
            # Param of PQ for 1 billion
            M      = 40 # 16  ###  200 / 5 = 40  The number of sub-vector. Typically this is 8, 16, 32, etc.
            nbits  = 8        ### bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
            nlist  = 6000     ###  # Param of IVF,  Number of cells (space partition). Typical value is sqrt(N)
            hnsw_m = 32       ###  # Param of HNSW Number of neighbors for HNSW. This is typically 32
        else :
            M = cc.m; nbits= cc.nbits; nlist= cc.nlist; hnsw_m = cc.hnsw_m



        # Setup  distance -> similarity in uncompressed space is  dis = 2 - 2 * sim, https://github.com/facebookresearch/faiss/issues/632
        ### IndexIVFPQ(coarse_quantizer, d, nlist, m, faiss.METRIC_L2) : euclidean distance
        # IndexFlatIP(d) IP stands for "inner product". If you have normalized vectors, the inner product becomes cosine similarity.

        quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
        index     = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)

        log('###### Train indexer')
        index.train(Xt)      # Train

        log('###### Add vectors')
        index.add(X)        # Add

        log('###### Test values ')
        index.nprobe = 8  # Runtime param. The number of cells that are visited for search.
        dists, ids = index.search(x=X[:3], k=4 )  ## top4
        log(dists, ids)

        log("##### Save Index    ")
        dirout2 = dirout + f"/faiss_trained{tag}.index"
        log( dirout2 )
        faiss.write_index(index, dirout2 )
        return dirout2



    def faiss_topk(df=None, root=None, colid='id', colemb='emb', faiss_index=None, topk=200, npool=1, nrows=10**7, nfile=1000) :  ##  python prepro.py  faiss_topk   2>&1 | tee -a zlog_faiss_topk.txt
       """ id, dist_list, id_list
           ## a/adigcb201/ipsvolh03/ndata/cpa//emb/emb//ichiba_order_20210901b_itemtagb2/seq_1000000000/faiss//faiss_trained_9808032.index

           https://github.com/facebookresearch/faiss/issues/632

           This represents the quantization error for vectors inside the dataset.
            For vectors in denser areas of the space, the quantization error is lower because the quantization centroids are bigger and vice versa.
            Therefore, there is no limit to this error that is valid over the whole space. However, it is possible to recompute the exact distances once you have the nearest neighbors, by accessing the uncompressed vectors.

            distance -> similarity in uncompressed space is

            dis = 2 - 2 * sim

       """
       # nfile  = 1000      ; nrows= 10**7
       # topk   = 500

       if faiss_index is None :
          faiss_index = ""
          # index       = root + "/faiss/faiss_trained_1100.index"
          # faiss_index = root + "/faiss/faiss_trained_13311813.index"
          # faiss_index = root + "/faiss/faiss_trained_9808032.index"
       log('Faiss Index: ', faiss_index)
       if isinstance(faiss_index, str) :
            faiss_path  = faiss_index
            faiss_index = faiss_load_index(db_path=faiss_index)
       faiss_index.nprobe = 12  # Runtime param. The number of cells that are visited for search.

       ########################################################################
       if isinstance(df, list):    ### Multi processing part
            if len(df) < 1 : return 1
            flist = df[0]
            root     = os.path.abspath( os.path.dirname( flist[0] + "/../../") )  ### bug in multipro
            dirin    = root + "/df/"
            dir_out  = root + "/topk/"

       elif df is None : ## Default
            # root =  dir_rec + "/emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/"
            root    = dir_cpa2 + "/emb/emb/ichiba_order_20210901b_itemtagb2/seq_1000000000/"
            dirin   = root + "/df/*.parquet"
            dir_out = root + "/topk/"
            flist = sorted(glob.glob(dirin))

       else : ### df == string path
            root    = os.path.abspath( os.path.dirname(df)  + "/../")
            log(root)
            dirin   = root + "/df/*.parquet"
            dir_out = root + "/topk/"
            flist   = sorted(glob.glob(dirin))

       log('dir_in',  dirin) ;
       log('dir_out', dir_out) ; time.sleep(2)
       flist = flist[:nfile]
       if len(flist) < 1: return 1
       log('Nfile', len(flist), flist )
       # return 1

       ####### Parallel Mode ################################################
       if npool > 1 and len(flist) > npool :
            log('Parallel mode')
            from utilmy.parallel  import multiproc_run
            ll_list = multiproc_tochunk(flist, npool = npool)
            multiproc_run(faiss_topk,  ll_list,  npool, verbose=True, start_delay= 5,
                          input_fixed = { 'faiss_index': faiss_path }, )
            return 1

       ####### Single Mode #################################################
       dirmap       = faiss_path.replace("faiss_trained", "map_idx").replace(".index", '.parquet')
       map_idx_dict = db_load_dict(dirmap,  colkey = 'idx', colval = 'item_tag_vran' )

       chunk  = 200000
       kk     = 0
       os.makedirs(dir_out, exist_ok=True)
       dirout2 = dir_out
       flist = [ t for t in flist if len(t)> 8 ]
       log('\n\nN Files', len(flist), str(flist)[-100:]  )
       for fi in flist :
           if os.path.isfile( dir_out + "/" + fi.split("/")[-1] ) : continue
           # nrows= 5000
           df = pd_read_file( fi, n_pool=1  )
           df = df.iloc[:nrows, :]
           log(fi, df.shape)
           df = df.sort_values('id')

           dfall  = pd.DataFrame()   ;    nchunk = int(len(df) // chunk)
           for i in range(0, nchunk+1):
               if i*chunk >= len(df) : break
               i2 = i+1 if i < nchunk else 3*(i+1)

               x0 = np_str_to_array( df[colemb].iloc[ i*chunk:(i2*chunk)].values   , l2_norm=True )
               log('X topk')
               topk_dist, topk_idx = faiss_index.search(x0, topk)
               log('X', topk_idx.shape)

               dfi                   = df.iloc[i*chunk:(i2*chunk), :][[ colid ]]
               dfi[ f'{colid}_list'] = np_matrix_to_str2( topk_idx, map_idx_dict)  ### to item_tag_vran
               # dfi[ f'dist_list']  = np_matrix_to_str( topk_dist )
               dfi[ f'sim_list']     = np_matrix_to_str_sim( topk_dist )

               dfall = pd.concat((dfall, dfi))

           dirout2 = dir_out + "/" + fi.split("/")[-1]
           # log(dfall['id_list'])
           pd_to_file(dfall, dirout2, show=1)
           kk    = kk + 1
           if kk == 1 : dfall.iloc[:100,:].to_csv( dirout2.replace(".parquet", ".csv")  , sep="\t" )

       log('All finished')
       return os.path.dirname( dirout2 )


    def np_matrix_to_str2(m, map_dict):
        res = []
        for v in m:
            ss = ""
            for xi in v:
                ss += str(map_dict.get(xi, "")) + ","
            res.append(ss[:-1])
        return res


    def np_matrix_to_str(m):
        res = []
        for v in m:
            ss = ""
            for xi in v:
                ss += str(xi) + ","
            res.append(ss[:-1])
        return res

    def np_vector_to_str(m, sep=","):
        ss = ""
        for xi in m:
            ss += f"{xi}{sep}"
        return ss[:-1]


    def np_matrix_to_str_sim(m):   ### Simcore = 1 - 0.5 * dist**2
        res = []
        for v in m:
            ss = ""
            for di in v:
                ss += str(1-0.5*di) + ","
            res.append(ss[:-1])
        return res


    def np_str_to_array(vv,  l2_norm=True,     mdim = 200):
        ### Extract list into numpy
        # log(vv)
        #mdim = len(vv[0].split(","))
        # mdim = 200
        from sklearn import preprocessing
        import faiss
        X = np.zeros(( len(vv) , mdim  ), dtype='float32')
        for i, r in enumerate(vv) :
            try :
              vi      = [ float(v) for v in r.split(',')]
              X[i, :] = vi
            except Exception as e:
              log(i, e)

        if l2_norm:
           # preprocessing.normalize(X, norm='l2', copy=False)
           faiss.normalize_L2(X)  ### Inplace L2 normalization
        log("Normalized X")
        return X


    def topk_convert_id(dirin=None, idtype=None, dirmap=None, npool=1, nfile=10000, overwrite=False):   ###  python prepro.py topk_convert_id   --idtype  ranid    2>&1 | tee -a zlog_prepro.py  &
        ### faiss id into
        ### root   = dir_cpa1  + "emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/"
        ### root0  = dir_cpa2  + "/emb/emb/ichiba_order_20210901b_itemtagb2/seq_1000000000/"

        if idtype is None : idtype = 'item_tag_vran'
        df = dirin

        #### vran, siid, item_tag_vran
        if isinstance(df, list):    ### Multi processing part
            flist = df[0]
            root     = os.path.abspath( os.path.dirname( flist[0] + "/../../") )  ### bug in ultipro
            dirin    = root +  "/topk/"
            dirout   = root + f"/topk_{idtype}/"

        elif df is None : ## Default
            # root =  dir_rec + "/emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/"
            root    = dir_cpa2 + "/emb/emb/ichiba_order_20210901b_itemtagb2/seq_1000000000/"
            dirin   = root +  "/topk/*.parquet"
            dirout  = root + f"/topk_{idtype}/"
            flist = sorted(glob.glob(dirin))

        else : ### df == string path
            root    = os.path.abspath( os.path.dirname(df )  + "/../")
            dirin   = root +  "/topk/*.parquet"
            dirout  = root + f"/topk_{idtype}/"
            flist   = sorted(glob.glob(dirin))

        log('dir_in',  dirin) ;
        log('dir_out', dirout) ; time.sleep(2)
        flist = flist[:nfile]

        if len(flist) < 1:
            log('Empty list') ; return 1

        if dirmap is None  :
          dirmap = dir_cpa1 + "/emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur//faiss/map_idx_13311813_siid_ranid.parquet"

        # dirout = dir_cpa0 + f"/cpa/topk/ichiba_clk_202006_202012d_itemtagb_202009_12/topk_{idtype}/"
        #### {0: '203494_10001474', 1: None, 2: None, 3: '231758_10001886', 4: '225009_10000849', 5:
        #idx_toitemtag  = db_load_dict(dir_rec+  "/e/faiss/map_idx_13311813.parquet",     colkey = 'idx', colval = 'id' )
        #### Dict Loaded 20582020
        #itemtag_tosiid = db_load_dict(dir_rec+  "/map/ichiba_clk_202006_202012d_itemtagb_202009_12/*",  colkey = 'item_tag_vran', colval = 'siid' )

        def to_siid(ss):
            ss = ss.split(",")
            ss = ",".join([ str(map_to_siid.get( t, "")) for t in ss ])
            return ss

        #### Parallel Mode  #####################################################
        if npool > 1  and len(flist) >= npool :
            log('Parallel mode')
            from utilmy.parallel  import multiproc_run
            ll_list = multiproc_tochunk(flist, npool = npool)
            multiproc_run(topk_convert_id,  ll_list,  npool, verbose=True,
                          input_fixed = { 'idtype': idtype, 'dirmap': dirmap }, )
            return 1

        #### Single Mode   ######################################################
        log(dirin, len(flist), str(flist)[:200] )
        # map_to_siid  = db_load_dict(dirmap,  colkey = 'idx', colval = idtype )
        map_to_siid  = db_load_dict(dirmap,  colkey = 'item_tag_vran', colval = idtype )
        log(str(map_to_siid)[:200]) ; time.sleep(1)


        for i, fi in enumerate(flist) :
            fouti = dirout + "/" + fi.split("/")[-1]
            # if os.path.isfile(fouti) and not overwrite : continue

            ### dist_list', 'id', 'id_list']
            log('in:', fi)
            df  = pd_read_file( fi,   n_pool=1, verbose=False)
            #  log(df.columns)
            df['id']      = df['id'].apply(lambda x:  str(map_to_siid.get( x, "") )  )  ### As string to prevent issues
            df['id_list'] = df['id_list'].apply(lambda x: to_siid(x))

            log(df[['id', 'id_list'  ]])
            pd_to_file(df, fouti , show=1 )

            if i == 0 :
               pd_to_file(df.iloc[:50, :], fouti.replace(".parquet", ".csv") , show=0 )
        log('Finished', i)




###########################################################################################
if 'backup' :
    cmd_mountgs = "umount /a/adigcb201/ipsvolh03  &&  mount -t glusterfs -o log-level=WARNING,backup-volfile-servers=adigcb202.prod.hnd2.bdd.local:adigcb203.prod.hnd2.bdd.local adigcb201.prod.hnd2.bdd.local:/ipsvolh03 /a/adigcb201/ipsvolh03   "


    def os_copy_safe(dirin=None, dirout=None,  nlevel=5, nfile=5000, logdir="./", pattern="*", cmd_fallback=""):  ### python prepro.py os_copy_safe
        """
            a2/acb401/ipsvols06/ndata/cpa/cpa/topk/ichiba_pur
        """
        import shutil, time, os, glob
        # dirin  = "/data/workspaces/takos01/cpa/emb/"
        # dirout = "/a/adigcb204/ipsvolh03/ndata/cpa/emb/emb/"
        if dirin is None :
           dirin  = "/a/adigcb201/ipsvolh03/ndata/cpa/emb/emb/ichiba_order_20210901b_itemtagb2/aseq_full_hash3v24full/topk_ranid/"
           dirout = "/a/acb401/ipsvols06/ndata/cpa/cpa/topk/ichiba_pur/topk_ranid/"

        flist = [] ; dirinj = dirin
        for j in range(nlevel):
            dirinj = dirinj + "/" + pattern
            ztmp =  sorted( glob.glob(dirinj ) )
            if len(ztmp) < 1 : break
            flist  = flist + ztmp


        log('n files', len(flist), dirinj, dirin )
        kk = 1 ; ntry = 0 ;i =0
        for i in range(0, len(flist)) :
            fi  = flist[i]
            fi2 = fi.replace(dirin, dirout)

            if not fi.isascii(): continue

            if not os.path.isfile(fi2) and os.path.isfile(fi) :
                 kk = kk + 1
                 if kk > nfile   : return 1
                 if kk % 50 == 0 : time.sleep(0.5)
                 if kk % 10 : log(fi2)
                 os.makedirs(os.path.dirname(fi2), exist_ok=True)
                 try :
                    shutil.copy(fi, fi2)
                    ntry = 0
                    # log(fi2)
                 except Exception as e:
                    log(e)
                    time.sleep(10)
                    log(cmd_fallback)
                    os.system(cmd_fallback)
                    time.sleep(10)
                    i = i - 1
                    ntry = ntry + 1
        log('Finished', i)


    def os_merge_safe(dirin_list=None, dirout=None, nlevel=5, nfile=5000, nrows=10**8,  cmd_fallback=""):
        cmd_fallback = "    umount /a/adigcb201/ipsvols03  && mount -t glusterfs -o log-level=WARNING,backup-volfile-servers=adigcb202.prod.hnd2.bdd.local:adigcb203.prod.hnd2.bdd.local adigcb201.prod.hnd2.bdd.local:/ipsvols03 /a/adigcb201/ipsvols03  "

        nrows = 10**8
        flist = []
        for fi in dirin_list :
            flist = flist + glob.glob(fi)
        log('Nfiles', len(flist)  , str(flist)[:100]); time.sleep(7)

        os_makedirs(dirout)
        fout = open(dirout,'a')
        for fi in flist :
            log(fi)
            ii   = 0
            fin  = open(fi,'r')
            while True:
                try :
                  ii = ii + 1
                  if ii % 100000 == 0 : time.sleep(0.3)
                  if ii > nrows : break
                  x = fin.readline()
                  if not x: break
                  fout.write(x.strip()+"\n")
                except Exception as e:
                  log(e)
                  os.system(cmd_fallback)
                  time.sleep(10)
                  fout.write(x.strip()+"\n")
            fin.close()
        log("All finished")


    def merge1():   #### python prepro.py merge1  &

        dirin  = [ "/data/workspaces/takos01/cpa/emb/seq/ichiba_order_20210901b_itemtagb2/ccount_log2/*.txt" ]

        dirout = "/a/gfs101/ipsvols07/ndata/cpa/emb/seq/ichiba_order_20210901b_itemtagb2/ccountlog_ranid/seq_pur_manual_ccount.txt"

        os_merge_safe(dirin_list= dirin, dirout= dirout, nlevel=5, nfile=5000,  cmd_fallback="")



    def backup_emb():   # python prepro.py backup_emb &
        dirin  = "/data/workspaces/takos01/cpa/emb/"
        dirout = "/a/gfs101/ipsvols07/ndata/cpa/emb/"

        os_copy_safe(dirin, dirout, nlevel=15, nfile=500000, logdir="./", cmd_fallback= cmd_mountgs )

    def backup_img():   # python prepro.py backup_img &
        dirin  = "/data/workspaces/noelkevin01/img/models/"
        dirout = "/a/gfs101/ipsvols07/ndata/cpa/img/models/"

        os_copy_safe(dirin, dirout, nlevel=20, nfile=500000, logdir="./", cmd_fallback= cmd_mountgs )


    def backup_db():   # python prepro.py backup_db &
        ### OS rename
        t0      = str(date_now_jp() )
        dirin   = "/sys/fs/cgroup/cpa/db"
        #dirout = "/a/adigcb201/ipsvolh03/ndata/cpa/db/db"
        dirout  = "/a/gfs101/ipsvols07/ndata/cpa/db/db"
        log(t0, dirin, dirout)
        os.system( f"mv   {dirout}  {dirout + t0}   ")     ### Rename

        os_copy_safe(dirin, dirout, nlevel=6, nfile=5000, logdir="./", cmd_fallback= cmd_mountgs )


    def backup_code():   # py backup_code &
        t0      = str(date_now_jp() )
        dirin   = "/home/noelkevin01/test_code"
        dirout1 = "/a/gfs101/ipsvols07/ndata/zbackup/cd/latest/"
        dirout2 = "/a/gfs101/ipsvols07/ndata/zbackup/cd/" + str(t0) + "/"
        log(t0, dirin, dirout2)
        try :
          os.rename( dirout1, dirout2)
        except: pass
        os.system( f" rm -rf {dirout1}")
        os_copy_safe(dirin, dirout1, nlevel=10, nfile=5000, logdir="./", cmd_fallback= cmd_mountgs )



###########################################################################################
if 'utils0':
    def now_hour_between(hour1="12:45", hour2="13:45", timezone="jp"):
        # Daily Batch time is between 2 time.
        from pytz import timezone as tzone ; import datetime
        timezone = {'jp' : 'Asia/Tokyo', 'utc' : 'utc'}.get(timezone, 'utc')
        format_time = "%H:%M"
        hour1 = datetime.datetime.strptime(hour1, format_time).time()
        hour2 = datetime.datetime.strptime(hour2, format_time).time()
        now_weekday = datetime.datetime.now(tz=tzone(timezone)).time()
        if hour1 <= now_weekday <= hour2:
            return True
        return False

    def pd_del(df, cols):
        for ci in cols :
            if ci in df.columns : del df[ci]
        return df


    def test5():
       df = pd.DataFrame(np.random.randint(0, 10, size=(50, 4)),
                         columns=list('abcd'))


       def fun1(dfi):
            return  dfi['c'].sum()  + dfi['d'].sum()

       dfg = pd_apply_parallel(df, colsgroup=None, fun_apply=fun1, npool=5)
       log(df)
       log(dfg)


    def pd_apply_parallel(df, colsgroup=None, fun_apply=None, npool=5):
        """ Pandas parallel apply

        """
        import pandas as pd, numpy as np, time, gc
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ppe = ProcessPoolExecutor(npool)

        if colsgroup is None :
            def f2(df):
                return df.apply(fun_apply, axis=1)

        else :
            df = df.groupby(colsgroup)  ### Need to get the splits
            def f2(df_groupby):
                return df_groupby.apply(fun_apply, axis=1)

        ksize = int( len(df) // npool )
        futures = []
        for  i in range(npool):
            if i == npool-1 :
                i = i + 1   ## Full size
            dfi = df.iloc[ i*ksize : (i+1):ksize, : ]
            p   = ppe.submit(f2, dfi )
            futures.append(p)
            del dfi
        del df ; gc.collect()


        df_out =  None
        for future in as_completed(futures):
            dfr    = future.result()
            df_out = pd.concat((df_out, dfr )) if df_out is not None else dfr
            del dfr

        return df_out


    def multithread_run(fun_async, input_list:list, n_pool=5, start_delay=0.1, verbose=True, **kw):
        """  input is as list of tuples  [(x1,x2,x3), (y1,y2,y3) ]
        def fun_async(xlist):
          for x in xlist :
                hdfs.upload(x[0], x[1])
        """
        # from threading import Thread
        import time
        #### Input xi #######################################
        xi_list = [ []  for t in range(n_pool) ]
        for i, xi in enumerate(input_list) :
            jj = i % n_pool
            xi_list[jj].append( tuple(xi) )

        if verbose :
            for j in range( len(xi_list) ):
                print('thread ', j, len(xi_list[j]))
            time.sleep(6)

        #### Pool execute ###################################
        import multiprocessing as mp
        # pool     = multiprocessing.Pool(processes=3)
        pool     = mp.pool.ThreadPool(processes=n_pool)
        job_list = []
        for i in range(n_pool):
             time.sleep(start_delay)
             print('ok', i, end=",")
             job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
             if verbose : log(i, xi_list[i] )

        res_list = []
        for i in range(n_pool):
            if i >= len(job_list): break
            res_list.append( job_list[ i].get() )
            print(i, 'done', end=",")

        pool.terminate() ; pool.join()  ; pool = None
        log('n_processed', len(res_list) )



if 'utils daily' :
    def ddt(t0=None):
        if t0 is None : return time.time()
        else : return time.time()-t0

    def now(*s):
        log("\n\n###### ", date_now_jp("%Y%m%d-%H:%M:%S"), *s )


    def pd_add_siid(df, delete=False):
       if 'siid' not in df.columns :
         df['siid'] = df.apply(lambda x : siid(x), axis=1)
         if delete :
            del df['shop_id'] ; del df['item_id']
       return df


    def dict_load(dirin=None, colval='itemtag', colkey='siid', nrows = 1005999000):  ###   python  prepro.py  dict_load   > zlog_dictload.txt
        cols = None
        if dirin is None :     dirin = "/a/adigcb201/ipsvols03/ndata/cpa/input/itemaster_cat_tag_02_20210901b_orc/*"
        if dirin == 'itemtag': dirin = "/a/adigcb201/ipsvols03/ndata/cpa/map/siid_itemtag_full.parquet"
        if 'raw3am' in dirin : cols = ['shop_id', 'item_id', 'genre_id_path']

        df = pd_read_file( dirin, cols= cols )
        if len(df) < 1 : return {}
        if colkey == 'siid' :
            df = pd_add_siid(df) ; del df['shop_id'] ; del df['item_id']
        if 'genre_id_path' in df.columns : df['genre_id'] = df['genre_id_path'].apply(lambda x : to_int( x.split("/")[-1]) )
        df = df.set_index(colkey)
        dd = df[[ colval ]].to_dict('dict')
        dd = dd[colval]
        log('dict loaded', colval, len(dd), str(dd)[:100])
        return dd


    def dict_load_genre(today, today1):
        dgenre = dict_load(    dirin= dir_cpa3 + f'/ca_check/daily/item/ca_items2_{today}/raw3am/*.parquet',  colval='genre_id', colkey='siid', )
        if len(dgenre) < 1:
            dgenre = dict_load(dirin= dir_cpa3 + f'/ca_check/daily/item/ca_items2_{today1}/raw3am/*.parquet', colval='genre_id', colkey='siid', )
        log('Ngenres', len(dgenre) )
        return dgenre




    def hash_hex64(input_value):
        return format((hash_int64(input_value) + (1 << 64)) % (1 << 64), 'x')


    def hash_int64(input_value):
        import mmh3
        return mmh3.hash64(input_value)[0]


    def ca_logic_gethash(campaign_id="20211129", logic_id="kvn-20211130") :
        ### ca_logic_gethash("20211129", "kvn-20211130")
        return hash_hex64("%s_%s" % (campaign_id, logic_id))


    def pd_easy_count(dirin="eod", add_days=0):    #####  py2  pd_easy_count
        ###     pd_easy_count(dirin="userhist", add_days=0)
        tk = get_timekey()+add_days
        dirin2 = dirin
        if dirin == 'eod'  :     dirin2  = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod/*.parquet"
        if dirin == 'useremb'  : dirin2  = dir_cpa3 + f"/hdfs/daily_useremb/topk/{tk}/*.parquet"
        if dirin == 'userhist'  : dirin2 = dir_cpa3 + f"/hdfs/daily_user_hist/{tk-2}/brw_ran/*.parquet"   ### 26 mio in history

        log(dirin2)
        df= pd_read_file2(dirin2, cols=['easy_id'], drop_duplicates=['easy_id'], n_pool=20)
        log(df.shape)
        log('N easyid',      df.easy_id.nunique() )
        # log('Min size topk', df.topk.str.len().max() )



    def pd_easy_stats(dirin="eod", add_days=0):    #####  py2  pd_easy_count
        tk = get_timekey()+add_days
        dirin2 = dirin
        if dirin == 'useremb'  :
            dirin2 = dir_cpa3 + f"/hdfs/daily_useremb/{tk}/brw_ran/*.parquet"
            log(dirin2)
            df= pd_read_file2(dirin2, cols=None, n_pool=20)

            df.groupby('easy_id').agg()

        log(df.shape)
        log('N easyid',      df.easy_id.nunique() )
        # log('Min size topk', df.topk.str.len().max() )


    class Index0(object):
        ### to maintain global index, flist = index.read()  index.save(flist)
        def __init__(self, findex):
            self.findex = findex
            os.makedirs(os.path.dirname(self.findex), exist_ok=True)
            if not os.path.isfile(self.findex):
                with open(self.findex, mode='a') as fp:
                    fp.write("")

        def read(self,):
            import time

            with open(self.findex, mode='r') as fp:
               flist = fp.readlines()

            if len(flist) < 1 : return []
            flist2 = []
            for t  in flist :
                if len(t) > 5 and t[0] != "#"  :
                  flist2.append( t.strip() )
            return flist2

        def save(self, flist):
            if len(flist) < 1 : return True
            ss = ""
            for fi in flist :
              ss = ss + fi + "\n"
            try :
               with open(self.findex, mode='a') as fp:
                  fp.write(ss )
            except :
                time.sleep(5)
                with open(self.findex, mode='a') as fp:
                   fp.write(ss )

            return True


    class Index1(object):
        """ ### to maintain global index, flist = index.read()  index.save(flist)
        """
        def __init__(self, findex:str="ztmp_file.txt", ntry=10):
            self.findex = findex
            os.makedirs(os.path.dirname(self.findex), exist_ok=True)
            if not os.path.isfile(self.findex):
                with open(self.findex, mode='a') as fp:
                    fp.write("")

            self.ntry= ntry

        def read(self,):
            import time
            with open(self.findex, mode='r') as fp:
               flist = fp.readlines()

            flist = [ fi.replace("\n", "") for fi in flist]

            if len(flist) < 1 : return []
            flist2 = []
            for t  in flist :
                if len(t) > 5 and t[0] != "#"  :
                  flist2.append( t.strip() )
            return flist2

        def save(self, flist:list):
            if len(flist) < 1 : return True
            ss = ""
            for fi in flist :
              ss = ss + fi + "\n"
            with open(self.findex, mode='a') as fp:
                fp.write(ss )
            return True


        def save_filter(self, val:list=None):
            """  flist = index.save_isok(flist)
              if not isok : continue   ### Dont process flist
              ### Need locking mechanism Common File to check for Check + Write locking.

            """
            import random, time
            if val is None : return True
            if isinstance(val, str):  val = [val]

            #### Check if files exist  #####################
            fall =  set( self.read() )

            val2 = list( set(val).difference(fall) )
            log('New files', len(val2) )

            if len(val2) < 1 : return []

            #### Write the list of files on disk  #########
            ss = ""
            for fi in val2 :
              ss = ss + str(fi) + "\n"

            i = 1 ;  isok= False
            while i < self.ntry :
                try :
                   with open(self.findex, mode='a') as fp:
                      fp.write( ss )
                   return val2
                except Exception as e:
                    print(f"file lock waiting {i}s")
                    time.sleep( i + random.random() * i )
                    i += 1
            return []


    def os_process_find(name=r"*main\.(py|sh))", ishow=1, isregex=0, verbose=True):
        """ Return a list of processes matching 'name'.
            Regex (./tasks./t./main.(py|sh)|tasks./t.*/main.(py|sh))
            Condensed Regex to:
            ((.*/)?tasks.*/t.*/main\.(py|sh)) - make the characters before 'tasks' optional group.

            os_process_find("python prepro_prod.py daily_eod_user --ii 0") >
        """
        import psutil, re
        ls = []
        for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
            cmdline = " ".join(p.info["cmdline"]) if p.info["cmdline"] else ""
            if verbose: log(cmdline)
            if isregex:
                flag = re.match(name, cmdline, re.I)
            else:
                flag = name and name.lower() in cmdline.lower()

            if flag:
                ls.append({"pid": p.info["pid"], "cmdline": cmdline})

                if ishow > 0:
                    log("Monitor", p.pid, cmdline)
        return ls


    def to_int(x,val=-1):
         try :
            return int(x)
         except :
            return val


    def siid(x):
        return f"{int(x['shop_id'])}_{int(x['item_id'])}"


    def file_get_timekey(fi:str):  ### Extract timekey from file
        fi = fi.split("/")[-1]
        fi = fi.replace(".parquet", "").split("_")
        for t in fi :
            t = to_int(t)
            if t> 17000 and t < 19600 : return t
        return -1

    def file_get_bk(fi:str):      ### Extract bucket from file
        fi = fi.split("/")[-1]
        fi = fi.replace(".parquet", "").split("_")
        for t in fi :
            t = to_int(t)
            if t>= 0 and t < 500 : return t
        return -1

    def glob_glob(dirin, nfile=1000, tmin=None, tmax=19800, bmin=None, bmax=500):
        log('glob', dirin)
        flist  = sorted( glob.glob(dirin  ))
        if isinstance(tmin, int)  : flist = [fi for fi in flist if   tmin <= file_get_timekey(fi) < tmax ]
        if isinstance(bmin, int)  : flist = [fi for fi in flist if   bmin <= file_get_bk(fi)      < bmax ]

        flist  = flist[:nfile]
        log('Nfile: ', len(flist), str(flist)[:100])
        return flist


    def get_timekey(timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        return int((int(timestamp) + 9*3600)/86400)


    def get_timekey2():   ### Correct
        """   time.localtime(  time.time() ) --> date in UTC time
        """
        t = time.time()
        # return int((int(t) )/86400) - 1    ### correct
        return int((int(t) )/86400)

    def os_wait_until(dirin, ntry_max=200, sleep_secs=300):
        import glob, time
        log('####### Check if file ready', "\n", dirin, date_now_jp("%Y%m%d-%H:%M"))
        ntry=0
        while ntry < ntry_max :
           fi = glob.glob(dirin )
           if len(fi) >= 1: break
           ntry += 1
           time.sleep(sleep_secs)
           if ntry % 10 == 0 :
              log('####### Check if file ready', "\n", dirin, date_now_jp("%Y%m%d-%H:%M") )

        log('File is ready', dirin)



if 'utils distributed' :
    def batch_split_run(cmd="", split=10,  bmin1=0, bmax1=500, tag='all', rename_path="",sleep=325):
         ### python prepro_prod.py  daily_create_topk_batch    ### Full batch 10 splits
         ### 2mins per bucket,  50 bucket --> 100mins.
         import random
         if split > 20 : return 'error too many split'

         if len( os_process_find(cmd) ) > 0 :
            log('process Already running, terminating', cmd ); return 1

         if 'rename' in tag and len(rename_path)> 10 :  ## Rename target folder
            os_rename(rename_path, tag="_")
         
        
         if isinstance(sleep, tuple):
            if len(sleep) == 2 :
                smin, smax = sleep[0], sleep[1]
         else :
            smin, smax = sleep, sleep

         #### Launch batches   
         kbatch = int(bmax1 / split)
         for jj in range(0, split+1 ):
            if jj*kbatch >= bmax1 : break
            cmd1= f" {cmd} --bmin {jj*kbatch}  --bmax { (jj+1)*kbatch }   --split 0  &  "  ###  2>&1 | tee -a '{logfile}'
            log(jj, cmd1)
            os.system(cmd1)
            time.sleep( random.randint(smin, smax)  )
         log("all Launched", jj)


    def os_rename(f1, tag="_", use_random_tag=True):
        if os.path.isdir(f1):
          f1 = f1[:-1] if f1[-1] == "/" else f1
          f2 = f1 + tag + str(int(time.time()))
          os.rename(f1, f2)



if 'daily_check':
    def daily_check_batches(add_days=0):   ###   py daily_check_batches > "${logfile}_00_daily_batch_check.py"  2>&1  &
      ###    alias pc2='python prepro_prod2.py daily_check_batches 2>&1 | tee  "${logfile}_00_daily_batch_check.py"    '
      today,today1  =  date_now_jp(add_days=0+add_days), date_now_jp(add_days=-1+add_days)
      today2 =   date_now_jp(add_days=-2+add_days)
      tk2,tk1,tk = get_timekey()-2 , get_timekey()-1 ,  get_timekey()

      ll = [ ('', "\nHDFS Real Time"),
        (f"/hdfs/intraday/sc_stream/{today}/",  "Intra"),
        (f"/hdfs/intraday/sc_stream_item/{today}/",      "items embed"),

        ('', "\nEOD"),
        (f"/hdfs/items_pur/*{tk1}*",  "Pur"),
        (f"/hdfs/items_brw/*{tk1}*",  "Brw"),
        (f"/hdfs/daily_user_hist/{tk2}/brw_ran/",  'user hist brw'),
        (f"/hdfs/daily_user_hist/{tk2}/pur_ran/",  'user hist pur top 10'),

        (f"/hdfs/daily_user_eod/{tk1}/brw_ran_v15/",      'T-2 user list'),
        (f"/hdfs/daily_user_eod/{tk1}/sc_stream/",        'T-2, T-1'),
        (f"/hdfs/daily_user_eod/{tk1}/sc_stream_intra/",  'T-2,T-1, realtime'),
        (f"/hdfs/daily_user_eod/{tk1}/sc_stream_item/",   'T-2,T-1, realtime'),

        (f"/hdfs/daily_useremb/emb/{tk1}/", 'user emb'),

        (f'/log/log_gpu/*{today1}*db*topg_merge*.py', 'DB easyid topmerge'),
        (f'/log/log_gpu/*{today}*db*topg_pur*.py', 'DB easyid topmerge'),


        (f'/a/gfs101/ipsvols07/ndata/cpa/input/*genre_brw*{today2}*', 'HIVE genre BRW'),
        (f'/a/gfs101/ipsvols07/ndata/cpa/input/*genre_pur*{today2}*', 'HIVE genre PUR'),


        ('', "\nCA"),
        (f"/ca_check/daily/item/ca_items2_{today}/raw3am/",     "CA item   Data"),
        (f"/ca_check/daily/item/ca_items2_{today}/clean03am/",   "CA item   Data"),
        (f"/ca_check/daily/item/ca_items2_{today}/clean10am/",  "CA item   Data"),

        (f"/ca_check/daily/item/ca_items2_{today}/score/",    "item  scores"),
        (f"/ca_check/daily/item_vec/ca_items_{today}/faiss/", "Faiss Index"),
        (f"/ca_check/daily/item_vec/ca_items_{today}/hnsw/",  "HNSW  Index"),

        ('', "\ntopk"),
        (f'/hdfs/daily_usertopk/m001/{tk}/daily_user_eod/',       'topk all eod'),
        (f'/hdfs/daily_usertopk/m001/{tk}/daily_user_eod_genre/', 'topk all eod genre'),
        (f'/a/gfs101/ipsvols07/ndata/export/ca/rec/{today}/', 'Neil Export'),

        ('', "\ntopk valid"),
        (f'/hdfs/daily_usertopk/m001/{tk}/daily_user_eod_eval/',       'topk eod valid'),
        (f'/hdfs/daily_usertopk/m001/{tk}/abtest/',                    'abtest'),

      ]

      ll = [ (dir_cpa3 + t[0]  if 'ips' not in t[0] else t[0], t[1] ) + t for t in ll ]

      log(today, tk)
      for fi in ll :
        if len(fi[0]) < 32:
           log(fi[1]); continue
        else :
           flist = glob.glob(fi[0] + "*")
           suffix =  "/".join(fi[0].split("/")[-3:])

           if len(flist) < 1 :
              logformat(suffix,  len(flist),   '-- MISSING --- ' , fi[0],  fi[1] )
           else :
              logformat(suffix,  len(flist),   os_file_date_modified( flist[-1] ) , fi[0], )


    def logformat(*s, nsize=[30,4, 11 ]):
      if isinstance(nsize, int):
        nsize = [nsize]

      s2 = []
      for i,x in enumerate(s) :

        nsizei = nsize[i] if i < len(nsize) else nsize[0]

        nspace = max(1, nsizei - len(str(x)))
        s2.extend([x, " "*nspace])
      print(*s2, flush=True)


    def os_file_date_modified(dirin, fmt="%Y%m%d-%H:%M", timezone='Asia/Tokyo'):
        """last modified date
        """
        import datetime
        from pytz import timezone as tzone
        try :
          mtime  = os.path.getmtime(dirin)

          mtime2 = datetime.datetime.utcfromtimestamp(mtime)
          mtime2 = mtime2.astimezone(tzone(timezone))
          return mtime2.strftime(fmt)
        except:
          return ""



if 'imaster':
    def pd_add_itemaster(df, cols_cass=None):
        """  "ran_cd", "price", "tags", "genre_path", "item_type", "item_status", "mobile_flg", # "shop_id", "image_url", "item_name", # "item_id",
            "tax_flg", "postage_flg", "item_url", "shop_name", "shop_url", "genre_name_path", "rate", "detail_sell_type", "catalog_id",
            "item_number", "adult_flg", "yamiichi_flg", "genre_id", "review_num", "review_avg", # "time_stamp",

        """
        log('getting itemaster`')
        from db.cass_queries import CassQueries
        cass_query  = CassQueries( cass_config )

        cols2 =  ['item_name',   'price', 'shop_name',  'ran_cd', "genre_name_path", "review_num", "review_avg",     ] if cols_cass is None else cols_cass
        if 'siid' not in df.columns :  df['siid'] = df.apply( lambda x : siid(x), axis=1)

        ### Check if Imaster Data is needed
        vv = df['siid'].values

        vres = cass_query.get_si_im_data(siids=vv, fields = cols2)
        dfj  = pd.DataFrame.from_dict(vres, orient='index' , columns  = cols2 ).reset_index()
        dfj.columns = ['siid']  + cols2

        df   = df.merge( dfj  , on='siid', how='left' )
        return df


    def pd_add_itemaster_dict(df, cols=None, cass_query=None, update_cache=False):
        """  "ran_cd", "price", "tags", "genre_path", "item_type", "item_status", "mobile_flg", # "shop_id", "image_url", "item_name", # "item_id",
          "tax_flg", "postage_flg", "item_url", "shop_name", "shop_url", "genre_name_path", "rate", "detail_sell_type", "catalog_id",
          "item_number", "adult_flg", "yamiichi_flg", "genre_id", "review_num", "review_avg", # "time_stamp",

          ss=  ['209332_10005968', '205332_10001258',]
          res = pd_add_itemaster_dict(ss)

        """
        # log('getting itemaster`')
        if cols is None:
            cols = ['item_name',   'price', 'shop_name',  "genre_name_path", "review_num", "review_avg",   'image_url', 'genre_path'  ]

        if isinstance(df, pd.DataFrame) :
            if 'siid' not in df.columns :  df['siid'] = df.apply( lambda x : siid(x), axis=1)
            df = df['siid'].values

        vres = {} ; miss =[]
        for sid in df :
          try :
             v =  db_imaster[sid]
             if not isinstance(v, dict) : raise Exception('error not dict')
             vres[sid] = v
          except :
              miss.append(sid)

        if len(miss) > 0 :
            if cass_query is None :
                from db.cass_queries import CassQueries
                cass_query = CassQueries( cass_config )
            vres2 = cass_query.get_si_im_data(siids=miss, fields = cols)

            vres3= {}
            for k,v in vres2.items():
                v = { cols[i] : v[i] for i in range(0, len(v))    }   ##3 filed :vval
                vres3[k] = v
                if update_cache : db_imaster[k] = v
            vres = {**vres, **vres3}
        return vres


    class imaster(object):
        """  im2 = imaster()   'image_url', 'genre_path'
             im2.get(['209332_10005968' ], 'genre_path')

        """
        def __init__(self, max_ram_size=10**6, use_file="today_ca") :
          from db.cass_queries import CassQueries
          self.cass_query = CassQueries( cass_config )
          self.ddict= {}

          if "/" in str(use_file)  :
            self.ddict = db_load_dict2(use_file, colkey='siid', cols=[ 'genre_path', 'image_url'    ])

          elif 'today_ca' in  str(use_file) :
            ### Local parquet info
            today      = date_now_jp("%Y%m%d", timezone='jp', add_days= -1 )
            ## Index(['genre_name_path', 'item_emb', 'item_id', 'item_name', 'item_text', 'price', 'review_num', 'shop_id', 'shop_name', 'siid'],
            ## ['allow_combined', 'asuraku_closing_time', 'asuraku_destination', 'asuraku_flag', 'batch_date', 'campaign_id', 'discount_amount', 'discount_type', 'genre_id', 'genre_path', 'genre_name_path', 'item_id', 'image_url', 'item_max_price', 'item_min_price', 'item_name', 'item_url', 'postage_flag', 'price', 'ran_cd', 'ran_id', 'ref', 'ref_type', 'review_avg', 'review_num', 'series_id', 'sg_id', 'shop_id', 'shop_mng_id', 'shop_name', 'siid', 'user_availability', 'weekly_campaign_id']
            dirindex   = dir_ca + f"/daily/item/ca_items2_{today}/clean/daily_*.parquet"
            self.ddict = db_load_dict2(dirindex, colkey='siid', cols=[ 'genre_path', 'image_url'    ])

          log('Dict loaded', len(self.ddict))

        def get(self, siid, key=None, update_cache=False):
            """ Return a dict  {siid : {val} }
            """
            res =self.get_multi( [siid], key=key, update_cache=update_cache)
            return res.get(key, "" ) if key is not None else res


        def get_multi(self, siids, key=None, update_cache=False):
            """ Return a dict  {siid : {val} }
            """
            res1 = {} ; miss= [] ; res= {}
            if len(self.ddict) > 0:
                for sid in siids :
                    try :
                       res1[sid] = self.ddict[sid]
                    except :
                       miss.append(sid)
            else :
                miss = siids

            #### Use Diskcache + Cassandra   ###########################
            if len(miss) > 0:
                res = pd_add_itemaster_dict(miss, cols=None, cass_query=  self.cass_query, update_cache=update_cache )

            res = {**res, **res1 }
            if key  :
               res2 = {}
               for k,v in res.items():
                  try :
                     res2[k] = v[key]
                  except :
                     db_imaster.set(k, False, expire=1e-9)  ### Delete items
               return res2
            else :
               return res




###########################################################################################
if 'embed':

    def text_model_load(dirin=""):
      """
         pip install tensorflow_text

      """
      import tensorflow.compat.v2 as tf
      import tensorflow_text, tensorflow_hub as hub

      if dirin is None or 'http' in dirin  :
          module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
          # #@param ['https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',

          model = hub.load(dirin)
      else :
          model = tf.keras.models.load_model(dirin)
          model = model.layers[0].layers[0]

      log('model loaded', dirin, model); time.sleep(5)
      return model


    def text_tovec(model, str_list:list):
      return model(str_list).numpy()   ### np vector


    def text_tovec_batch(model, str_list:list):
        ll = []
        kbatch = 64
        n2 = max(1, int(len(str_list) / kbatch ))  ##m min size 1
        for i in range(0, n2 ) :
            # if i % 1000 : log(i)
            i2 = 2*kbatch*(i+1) if i == n2-1  else kbatch*(i+1)
            mat2d = model( str_list[ kbatch*i:i2 ] ).numpy()
            mat2d = np_matrix_to_str(mat2d)
            ll.extend( mat2d )
        return ll


    #### Normalize the input ################################
    def to_int2(x):
        try :
            return int( int(x)/50.0) * 50
        except : return 0

    def to_int3(x):
        try :
            return int( int(x)/500.0) * 500
        except : return 0


    def item_merge_field(x):
        ss = str(x['shop_name']) + " , "
        ss = ss + f"{x['genre_name_path']} , "
        ss = ss + f"ねだん {to_int3( x['price'])} , "
        ss = ss + f"いけん {to_int2( x['review_num'] )} , "

        ss = ss + str(x['item_name'])
        ss = ss.replace("・", " ").replace(">" , " ").replace("/", " ")
        return ss


    def item_add_text(modelin=None, dirin=None, dirout=None):   ### py item_add_text

        if dirin is None :
          dirin   = dir_cpa3 + "/input/ca_daily_coupons10c_ranid_info2"
          dirout  = dir_ca   + "/daily/item_text_all2/"

        t0    = get_timekey()

        cols  = ['shop_id', 'item_id', 'shop_name', 'genre_name_path', 'item_name', 'price', 'ranid', 'review_num', 'image_url',  'genre_path',  ]
        flist = glob_glob(dirin + "/*", 1000)


        df = pd_read_file( flist ,  cols = cols, nrows=5005005001 )
        log(df.shape, df.columns)
        df = df.drop_duplicates([ 'shop_id', 'item_id' ])

        ### Normalize item_text
        df['item_text'] = df.apply(lambda x : item_merge_field(x), axis=1)

        log(  df[[ 'item_text'  ]], df.columns)
        pd_to_file(df, dirout + f"/item_text_emb1_{t0}.parquet" , show=0)


    def item_add_vector(modelin=None, dirin=None, dirout=None):   ### py item_add_vector
        ### Index(['genre_name_path', 'item_id', 'item_name', 'price', 'ranid', 'review_num', 'shop_id', 'shop_name', 'item_text'],
        if dirin is None :
          dirin   = dir_ca   + "/daily/item_text_all2/"
          dirout  = dir_ca   + "/daily/item_text_all2-vgenre3_1m/"
          modelin = dir_ca   + "/models/static/v_genre3_1m/"

        t0    = get_timekey()

        model = text_model_load(modelin)
        #cols  = ['shop_id', 'item_id', 'shop_name', 'genre_name_path', 'item_name']
        cols  = ['shop_id', 'item_id', 'item_text', 'price', 'genre_name_path'  ]
        flist = glob_glob(dirin + "/*", 1)


        df = pd_read_file( flist ,  cols = cols, nrows=500500500 )
        log(df, df.columns)
        df = df.drop_duplicates([ 'shop_id', 'item_id' ])

        ### Normalize item_text
        #df['item_text'] = df.apply(lambda x : item_merge_field(x), axis=1)
        df = df[df['item_text'].str.len() > 50 ]  ### only with item name
        df['item_emb']  = text_tovec_batch(model, df['item_text'].values )

        log(  df[[ 'item_id', 'item_emb'  ]])
        pd_to_file(df, dirout + f"/item_text_emb1_{t0}.parquet" , show=0)


    def item_add_text_vector(modelin=None, dirin=None, dirout=None, nrows=500500500, nfile=1000, today=None,
                             remove_exist=True, overwrite=False ):   ### py item_add_text_vector
        ## Intraday calc on new item clicks
        ### Index(['genre_name_path', 'item_id', 'item_name', 'price', 'ranid', 'review_num', 'shop_id', 'shop_name', 'item_text'],

        today = date_now_jp("%Y%m%d", timezone='utc' ) if today is None else today

        if dirin is None :
           # modelin = dir_ca   +  "/models/static/v_genre3_1m/"
           dirin   = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/"
           dirout  = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/df/"

        t0    = get_timekey()
        global modelin0 ;log(model0)
        model = text_model_load(model0.dir)   if modelin is None else   text_model_load(modelin)
        cols  = ['shop_id', 'item_id', 'shop_name', 'genre_name_path', 'item_name', 'price',  'review_num',  ]
        flist = glob_glob(dirin + "/*.parquet", nfile)

        if len(flist) < 1: return 'empty'

        for ii,fi in enumerate(flist) :
            dirouti = dirout + fi.split("/")[-1].replace(".csv", ".parquet")
            if not overwrite:
               if os.path.isfile(dirouti)  : continue

            log(fi)
            df = pd_read_file( fi ,  cols = cols,  nrows= nrows )   ## sep="\t",
            if len(df) < 1 : continue
            log(df)
            df = df.drop_duplicates([ 'shop_id', 'item_id' ])
            df = df[df['shop_id'] > 0 ]
            df = df[df['genre_name_path'].str.len() > 10 ]  ### only with genre name
            df = df[df['item_name'].str.len() > 10 ]  ### only with item name


            if len(df) < 1 :
                log('Empty df', fi, df ) ; continue

            log(df, df.columns)
            df['siid'] = df.apply(lambda x : siid(x), axis=1)
            if remove_exist :
                df         = pd_cass_remove_exist(df, prefix = model0.pref)  ### Cass remove existing one

            if len(df) < 1 : continue

            ### Text features + emb Calc
            df['item_text'] = df.apply(lambda x : item_merge_field(x), axis=1)
            df['item_emb']  = text_tovec_batch(model, df['item_text'].values )   ###TF_flow model

            log(  df[[ 'item_id', 'item_emb'  ]])
            pd_to_file(df, dirouti , show=0)

            ### Update Cass
            df = df[[ 'siid', 'item_emb' ]]
            cass_update(df, table=model0.table, prefix= model0.pref, colkey="siid", colval="item_emb")


    def item_vector_tocass(dirin=None, table=None):   ### py item_vector_tocass

        today = date_now_jp("%Y%m%d", timezone='utc', add_days=-1)
        dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/df/*"

        cols  = [ 'shop_id', 'item_id', ]
        flist = glob_glob(dirin, 1000)
        df = pd_read_file(flist , cols= cols + [ 'item_emb'], n_pool=5)
        df = df.drop_duplicates(cols)

        if 'item_text' in df.columns :
             df = df[ df['item_text'].str.len() > 50 ]
        df = df[ df['item_emb'].str.len() > 100 ]

        df['siid'] = df.apply(lambda x : siid(x), axis=1)
        del df['shop_id'] ; del df['item_id'] ;

        # df = df.iloc[:10, :]
        log(df)
        cass_update(df, table=model0.table, prefix= model0.pref, colkey="siid", colval="item_emb")


    def text_faiss_create_index_check() :   ### py text_faiss_create_index
        dirin =  dir_cpa3  + f"/ca_check/models/data/item_text_all2-vgenre3_1m//*.parquet"

        cc = Box({})
        cc.m= 32; cc.nbits= 8; cc.nlist= 5000; cc.hnsw_m=32
        faiss_create_index(df_or_path = dirin,
                           colemb  = 'item_emb', colid = 'siid',
                           dirout  = None,  faiss_type = "IVF4096,Flat",
                           nfile=1, emb_dim=512, faiss_pars = cc)


    def faiss_load_index(db_path=None,  db_type = "IVF4096,Flat"):
        import faiss
        index = faiss.read_index( db_path  )
        log(db_path, index)
        return index


    def db_load_dict(df, colkey='ranid', colval='item_tag', naval='0', colkey_type='str', colval_type='str', npool=5, nrows=900900900, verbose=True):
        ### load Pandas into dict
        if isinstance(df, str):
           dirin = df
           log('loading', dirin)
           flist = glob_glob( dirin , 1000)
           log(  colkey, colval )
           df    = pd_read_file(flist, cols=[ colkey, colval  ], nrows=nrows,  n_pool=npool, verbose=True)

        log( df.columns )
        df = df.drop_duplicates(colkey)
        df = df.fillna(naval)
        log(df.shape)

        df[colkey] = df[colkey].astype(colkey_type)
        df[colval] = df[colval].astype(colval_type)


        df = df.set_index(colkey)
        df = df[[ colval ]].to_dict()
        df = df[colval] ### dict
        if verbose: log('Dict Loaded', len(df), str(df)[:100])
        return df



if 'cassandra':
    cass_config = "../config/config_v14.properties"

    def cass_getconn():
        from db.cass_queries import CassQueries
        return CassQueries(cass_config)

    def cass_update(df, table, prefix="m001", colkey="siid", colval="siid_emb", ttl=100*86400):  ## 240days
       ### Very Slow to insert (!)
       log("Start cass insert", table, colkey, colval)
       df = df[[colkey, colval]].drop_duplicates(colkey)
       df = df.values
       log('Insert size', len(df) )

       from db.cass_queries import CassQueries
       cass_query = CassQueries(cass_config)
       cdata = {}
       for i in range(len(df)):
            key           = f"{prefix}_{df[i, 0]}"
            cdata[  key ] = cass_encode( df[i, 1] )
            if len(cdata) < 5 and i < len(cdata)-1 :  continue
            isok = cass_query.ndata_item_upsert( cdata, ttl_sec= ttl, consistency = 'one', istest=False)
            cdata = {}

    def pd_cass_get_vect(df, prefix="m001", tablename="ndata.item_model" )  :
        """    siids = [  'm001_197550_10255674', 'm001_222208_1009782' ]
        """
        log('getting itemaster`')
        from db.cass_queries import CassQueries
        cass_query = CassQueries(cass_config)

        npref = len(prefix)+1
        cols2 =  ['item_emb',  ]
        if not isinstance(df, pd.DataFrame):
            df = pd_read_file(df)

        if 'siid' not in df : df['siid'] = df.apply( lambda x : siid(x), axis=1)
        siids  = set( df['siid'].values[:] )

        log('Query N:', len(siids))
        siids  = [ f"{prefix}_{t}" for  t in siids ]
        siids  = cass_query.ndata_item_read( siids, tablename= tablename )
        siids  = [ [k[npref:], cass_decode(v) ] for k,v in siids.items() ]  ### keep only siid
        siids  = pd.DataFrame(siids,  columns  = ['siid', 'item_emb'] )

        log( 'Query Res:', len(siids), str(siids)[:100] )
        return siids

    def pd_cass_get_vect2(df, prefix="m001", tablename="ndata.item_model", ivect=None, update_cache=False )  :
        """    siids = [  'm001_197550_10255674', 'm001_222208_1009782' ]
        """
        log('getting ', tablename)
        ivect = ivector(use_dict=False) if ivect is None else ivect

        if not isinstance(df, pd.DataFrame): df = pd_read_file(df)

        if 'siid' not in df : df['siid'] = df.apply( lambda x : siid(x), axis=1)
        siids  = set( df['siid'].values[:] )

        log('Query N:', len(siids))
        siids  = ivect.get_multi(siids, use_dict=False, update_cache=update_cache)
        siids  = pd.DataFrame.from_dict(siids, orient='index',  columns  = [ 'item_emb'] ).reset_index()
        siids.columns = ['siid', 'item_emb' ]
        log( 'Query Res:', len(siids), str(siids)[:45] )

        df = df.merge(siids, on='siid', how='left')
        df['item_emb'] = df['item_emb'].fillna("")
        return siids


    def cass_get_itemvect_dict(siids, cass_query=None, prefix="m001", tablename="ndata.item_model" )  :
        """    siids = [  'm001_197550_10255674', 'm001_222208_1009782' ]
        """
        log('getting', tablename)
        if cass_query is None :
           from db.cass_queries import CassQueries
           cass_query = CassQueries(cass_config)

        npref = len(prefix)+1

        log('Query N:', len(siids))
        siids  = [ f"{prefix}_{t}" for  t in siids ]
        siids  = cass_query.ndata_item_read( siids, tablename= tablename )
        siids  = { k[npref:]: cass_decode(v)  for k,v in siids.items() }  ### keep only siid

        log( 'Query Res:', len(siids), str(siids)[:100] )
        return siids


    def cass_check(siid_list, prefix="m001", tablename=""):
       log("Start cass check if exist", tablename)
       from db.cass_queries import CassQueries
       cass_query = CassQueries(cass_config)
       siid_list  = [ f"{prefix}_{t}" for  t in siid_list ]
       vres = cass_query.ndata_item_check( siid_list )
       vres = set( vres.keys())
       return vres


    def pd_cass_remove_exist(df, prefix="m001") :
        ### df = pd.DataFrame( ['197550_10255674', '222208_10097827', '213663_10107856', 'aa'], columns= [ 'siid' ] )
        if 'siid' not in df.columns:
           df['siid'] = df.apply(lambda x : siid(x), axis=1)

        nprefix =len(prefix) +1
        ddict = cass_check(siid_list= set(df['siid'].values), prefix= prefix )
        ddict = { t[nprefix:] for t in ddict  }
        log('Already exist', len(ddict), str(ddict)[:50])
        df= df[ -df.siid.isin(ddict) ]
        return df


    def cass_encode(v):
        v = v.encode('utf-8')
        v = zlib.compress( v, 9)
        return v

    def cass_decode(x):
        try :
            js = zlib.decompress(x).decode('utf8')
            return js
        except Exception as e:
            log( e, x )


    def db_insert(df, colkey, colvalue, db_path) :
        diskcache_save(df, colkey, colvalue, db_path=db_path, )

    def db_get_topk(x, mode):
        return db_siid_topk.get(x, "")



if 'user_hist':
    def pd_userhist_load(fi, only_last_siid=0 ):  ##### easy_id, siid    5 siid per easy_id
        ### Load histo data
        df = pd_read_file(fi, cols=None)  ###
        log('#### Loaded user histo: ', df.shape, )
        df = pd_add_siid(df, delete=True)

        ### can have mutiple   easyid, siid  pairs (historical)
        if only_last_siid > 0 :
           df = df.groupby("easy_id").apply(lambda dfi: dfi['siid'].values[-1]).reset_index()
           df.columns = ['easy_id', 'siid'] ; log('Neasyid', df.shape, "\n\n")
           return df

        df = df.groupby("easy_id").apply(lambda dfi: dfi['siid'].values[::-1]).reset_index()
        df.columns = ['easy_id', 'siids']
        df['siid'] = df['siids'].apply(lambda x: x[0])  #### Most recent is 1st
        log('Neasyid', df.shape, "\n\n")
        return df



if 'user_emb':
    def daily_user_resplit():  ### replit
        for bk in range(0, 500):
            dfp = pd_read_file(  dir_cpa3 + f"/hdfs/daily_useremb/emb/*/*_{bk}.parquet" , npool=15)
            dfp = dfp.drop_duplicates('easy_id')
            dfp = pd_add_easyid_bk(dfp)
            dfp = dfp[dfp.bk == bk ]
            pd_to_file(dfp, dir_cpa3 + f"/hdfs/daily_useremb/emb/a/19011full/user_emb_{bk}_{len(dfp)}.parquet"  )


    def daily_user_emb_update(bmin=0, bmax=500, split=0, mode="pur",add_days=0,  full=0,  nfile=1000, update_cass=0):   ####
        ### Update user vector daily
        """  python prepro_prod.py  daily_user_emb_update  --split 10  --add_days 0  --full 1  >> "${logfile}_user_emb_tocass_all.py"  2>&1   &
            at 20h00am --> 23h00 time --> 3 hours update of user embd
               python prepro_prod.py  daily_user_emb_update  --split 0  --bmin 8 --add_days -1  --full 1  >> "${logfile}_user_emb_tocass8.py"  2>&1   &
               Processed: 27,461,556 in Cassandra

               mode="pur";add_days=0; bmin=0; bmax=500; full=0; split=0; nfile=1000; update_cass=0

        """
        tk1    = get_timekey() - 0 + add_days  ### today
        tk2    = get_timekey() - 1 + add_days  ### Previous day
        datek1 =  date_now_jp('%Y%m%d', add_days=0 +add_days)
        if split > 0 :
            cmd = f"{PYFILE} daily_user_emb_update --split 0  --full {full} --add_days {add_days}   --update_cass {update_cass} "
            batch_split_run(cmd= cmd, split=split, sleep= (150, 300) ) ; return 'done'


        dirout = dir_cpa3 + f"/hdfs/daily_useremb/emb/{tk1}/"
        cass_query = cass_getconn()
        model_pref = model0.pref
        iv  = ivector(model_prefix= model_pref, cass_query= cass_query  )  ### Emb Vectors
        ttl = 90*86400  ## 240 days
        update_cache= False

        #### T-2 data
        dir0  = dir_cpa3         + f"/hdfs/daily_user_hist/"
        index = Index1( dir_cpa3 + f"/hdfs/daily_useremb/topk/done_cass_useremb_{tk1}.txt"  )
        cols  = ['easy_id', 'item_id', 'shop_id', 'genre_id']
        log("Using full:",full, bmin, bmax, datek1,  full,  dir0, log(index.read() ))


        log("#### Today History Load")
        dirintra = sorted(glob_glob( dir_cpa3 + f"/hdfs/intraday/sc_stream/{datek1}/*brow*.parquet", nfile))
        dfb1     = pd_read_file2( dirintra, cols=cols+['ts'], drop_duplicates= ['easy_id', 'genre_id'],   n_pool=20, nfile= nfile)
        dfb1 = dfb1.sort_values(['easy_id', 'ts']).drop_duplicates(['easy_id', 'genre_id'], keep='last' )
                
        dirintra = sorted(glob_glob( dir_cpa3 + f"/hdfs/intraday/sc_stream/{datek1}/*pur*.parquet", nfile))
        dfu1    = pd_read_file2( dirintra, cols=cols+['ts'], drop_duplicates= ['easy_id', 'genre_id'],   n_pool=20, nfile= nfile)
        dfu1    = dfu1.sort_values(['easy_id', 'ts']).drop_duplicates(['easy_id', 'genre_id'], keep='last' )        
        dfb1 = pd.concat((dfb1, dfu1)) ; del dfu1
        
        log(datek1, dfb1.shape, dirintra)
        dfb1 = dfb1.drop_duplicates(['easy_id', 'genre_id'], keep='last' ) ;  del dfb1['ts']
        dfb1 = dfb1.groupby('easy_id' ).tail(5)

        dfb1 = pd_add_easyid_bk(dfb1)
        dfb1 = dfb1[(dfb1.bk >= bmin) & (dfb1.bk < bmax) ]
        log(dfb1, dfb1.shape, )

        if full<1: ## Only new users
           leasyid_new = set( dfb1.easy_id.unique() )
           log('New users', len(leasyid_new) )
           if len(leasyid_new) < 100 : return 'No new users'
           update_cache=True

        def isexist(tk,bk):
            for u in index.read():
                if f",{tk},{bk}," in u : return True
            return False

        for bk in range(bmin, bmax):
            if isexist(tk2, bk) : continue
            log("\n Bucket:", bk, tk2,  dir0 + f"/{tk2}/brw_ran/"  + f"/*_{bk}_*.parquet"  )
            df   = pd_read_file( dir0 + f"/{tk2}/brw_ran/*_{bk}_*.parquet" , cols=cols)  ;
            df1  = pd_read_file( dir0 + f"/{tk2}/pur_ran/*_{bk}_*.parquet" , cols=cols)  ;
            df1  = df1.groupby('easy_id' ).tail(9)   ###3 Only last 5 purchases
            df   = pd.concat((df, df1))  ; del df1
            df   = df.drop_duplicates(['easy_id', 'genre_id'], keep='last' )
            if full<1      : df = df[df.easy_id.isin(leasyid_new)]
            if len(df) < 1 : log('Empty df', bk) ; continue

            df   = pd.concat((df, dfb1[ dfb1.bk == bk ][cols]  ))   #### Merge  with T-1 data
            df   = df.drop_duplicates(['easy_id', 'genre_id'], keep='last' )
            log('Full size', df.shape, df.easy_id.nunique() )
            # df = df.iloc[:500, :]

            df     = pd_add_siid(df, delete=True)
            dd_emb = iv.get_multi(df['siid'].unique() ,  use_dict=False, update_cache= update_cache)
            if len(dd_emb) < 1:  log('\n\n Error no emb retrieved') ; continue

            df['item_emb'] = df[ 'siid'].apply(lambda x : dd_emb.get(x, ''))
            df             = df[ df['item_emb'].str.len() > 10 ]  ### Remove empty emb

            df             = df.groupby('easy_id').apply(lambda dfi : user_emb_merge(dfi['item_emb'].values[-10:]) ).reset_index()
            df.columns     = ['easy_id', 'user_emb']
            df = df[df.user_emb.str.len() > 200 ] ### Filter out empty


            #if update_cass >0 :
            #    cass_update2(df[[ 'easy_id', 'user_emb'  ]], table='ndata.user_model', prefix=model_pref, colkey="easy_id", colval="user_emb",
            #             cass_query=cass_query, ttl=ttl)  ## 240days

            ### Prev day
            if full < 1:
              dfp = pd_read_file(  dir_cpa3 + f"/hdfs/daily_useremb/emb/{tk2}*/*_{bk}_*.parquet" )
              df  = pd.concat((dfp, df)) ; del dfp
            df  = df.drop_duplicates('easy_id', keep='last')
            pd_to_file(df, dirout + f"/user_emb_{bk}_{len(df)}.parquet")


            # log('Nusers',  df.shape)
            index.save_filter([  f"{datek1},{tk2},{bk},{len(df)}" ])
        log('\n\n###### All finished', bk)


    def user_getlist(datek):        #### eaysif for the day
        flist = glob_glon( dir_cpa3 + f"/hdfs/daily_user_list/*_{datek}_*.parquet" )
        if len(flist) < 1 :
            flist = glob_glob( dir_cpa3 + f"/hdfs/intraday/sc_stream/{datek}/*browsing*.parquet" )

        df = pd_read_file( flist[0], verbose=True, drop_duplicates=['easy_id'] )
        df = set(df.easy_id.values)
        log('New users', len(df))
        return df

    
    def user_emb_merge(ll):
      ### Take average,, ll[-1] is the closest in time
      if len(ll) < 1 : return ""
      weights = None

      try :
         vv = np.array( [ x.split(",") for x in ll  if len(x) > 10 ]  , dtype='float32' )  ### 6 digits
         #vv = vv.mean(axis=0)
         weights = 1.0 / ( 2 + np.arange(len(vv), 0, -1 ) )  #### last one has higher weight
         vv = np.average(vv, axis=0, weights=  weights)
         vv = vv / np.sqrt(np.dot(vv,vv))   ### Norm the vector  !!!!!
         ss = ""
         for t in vv: ss+= str(t) +","
         return ss[:-1]
      except :
         log('incorrect', str(ll)[:200] )
         return ""


    class ivector(object):
        """  im2 = imaster()
             im2.get(['209332_10005968' ] )  -->  vector in string format
        """
        def __init__(self, model_prefix="m001", cass_query=None, nmax=10**6, use_dict=False  ) :
          self.ddict      = fixedDict(limit=nmax)

          #### Cass
          self.cass_query = cass_getconn() if cass_query is None else cass_query
          self.cass_table = "ndata.item_model"
          self.prefix     = model_prefix
          self.npref      = len(self.prefix)+1


        def get_multi(self, siids, use_dict=True, update_cache=True):
            """ Return a dict  {siid : {val} }
            """
            global db_ivector
            log('get item vect', len(siids))
            res = {}; miss = [] ;
            if use_dict :
                for sid in siids :
                  try :       res[sid] = self.ddict[sid]
                  except :
                     try :    res[sid] = db_ivector[sid]
                     except : miss.append(sid)
            else :
                for sid in siids :
                     try :     res[sid] = db_ivector[sid]
                     except :  miss.append(sid)

            log('In cache:', len(res) )
            if len(miss) < 1 : return res

            res2 = cass_get_itemvect_dict(miss, cass_query= self.cass_query, prefix=self.prefix, )
            log('Nemb from Cass', len(res2) )
            for key,val in res2.items():
              try :
                 res[key]        = val
                 if update_cache: db_ivector[key] = val
                 if use_dict:     self.ddict[key] = val
              except Exception as e:
                 log('error', key, val, e)
                 db_ivector.set(key, False, expire=1e-9)  ### Delete items

            log('Got N emb:', len(res))
            return res

        
        
    def daily_useremb_update_cass(bmin=0, bmax=500, split=0, add_days=0,  ):        ###  py daily_useremb_update_cass --split 10  >> "${logfile}_useremb_cass.py"  2>&1  & 
        ### 3 hours to update
        tk = get_timekey() -1 + add_days        
        if split > 0 :
            cmd = f"{PYFILE} daily_useremb_update_cass --split 0  --add_days {add_days}    "
            batch_split_run(cmd= cmd, split=split, sleep= (15, 30) ) ; return 'done'
        
        dirin      = dir_cpa3 + "/hdfs/daily_useremb/emb/{tk}/*_{bk}_*.parquet"
        cass_query = cass_getconn()
        model_pref = model0.pref 
        for bk in range(bmin, bmax):
           fi = glob_glob(dirin.format(tk=tk, bk=bk))
           if len(fi) < 1 : continue
           df = pd_read_file(fi) 
           log(df.shape)
           cass_update2(df[[ 'easy_id', 'user_emb'  ]], table='ndata.user_model', prefix=model_pref, colkey="easy_id", colval="user_emb",
                        cass_query=cass_query, ttl=86400*100)  ## 240days
                        

    def cass_update2(df, table, prefix="m001", colkey="siid", colval="siid_emb", cass_query=None, ttl=21086400, kbatch=10):  ## 240days
       ### Very Slow to insert (!)
       log("Start cass insert", table, colkey, colval)
       df = df[[colkey, colval]].drop_duplicates(colkey)
       df = df.values
       log('Insert size', len(df), df[0,0],  str(df[0,1])[:100] )

       if cass_query is None :
          from db.cass_queries import CassQueries
          cass_query = CassQueries(cass_config)

       if   table == 'ndata.item_model' : cass_fun_upsert = cass_query.ndata_item_upsert
       elif table == 'ndata.user_model' : cass_fun_upsert = cass_query.ndata_user_upsert
       else : return "table not known"

       cdata = {}
       for i in range(len(df)):
            key           = f"{prefix}_{df[i, 0]}"
            cdata[  key ] = cass_encode( df[i, 1] )
            if len(cdata) < kbatch and i < len(cdata)-1 :  continue
            isok = cass_fun_upsert( cdata, ttl_sec= ttl, consistency = 'one', istest=False)
            cdata = {}



    #### User topk generation   ################################
    class uvector(object):
        """  im2 = imaster()
             im2.get(['209332_10005968' ] )  -->  vector in string format
        """
        def __init__(self, model_prefix="m001", cass_query=None, nmax=10**5, ttl_cache=None  ) :
          from functools import partial
          self.ddict      = fixedDict(limit=nmax)

          #### Cache
          #vself.db_cache   = db_uvector
          self.ttl_cache = ttl_cache

          #### Cass
          self.cass_query = cass_getconn() if cass_query is None else cass_query
          self.cass_table = "ndata.user_model"
          self.prefix     = model_prefix
          self.npref      = len(self.prefix)+1
          self.db_fun_get = partial(cass_get_useremb_dict, tablename= self.cass_table, prefix=self.prefix, )


        def get_multi(self, uids, use_dict=True, use_cache=True):
            """ Return a dict  {siid : {val} }
            """
            global db_uvector
            ttl = self.ttl_cache if self.ttl_cache is not None else 0
            log('Get key,val', self.cass_table, len(uids))
            res = {}; miss = [] ;
            if use_dict :
                for sid in uids :
                  try :       res[sid] = self.ddict[sid]
                  except :
                     try :    res[sid] = db_uvector[sid]
                     except : miss.append(sid)
            else :
                if use_cache:
                    for sid in uids :
                         try :     res[sid] = db_uvector[sid]
                         except :  miss.append(sid)
                else : miss = uids  #### All from Cass

            log('In cache:', len(res) )
            if len(miss) < 1 : return res

            res2 = self.db_fun_get(miss, )
            log('Nemb from Cass', len(res2) )

            if not use_cache :
                for key,val in res2.items():
                  try :
                     res[key]        = val
                  except Exception as e:
                     log('error', key, val, e)
                     db_uvector.set(key, False, expire=1e-9)  ### Delete items
            else :
                for key,val in res2.items():
                  try :
                     res[key]        = val
                     if ttl >0 : db_uvector.set(key, val, expire= ttl)  ### TTL
                     else :      db_uvector[key] = val

                     if use_dict: self.ddict[key] = val
                  except Exception as e:
                     log('error', key, val, e)
                     db_uvector.set(key, False, expire=1e-9)  ### Delete items
            log('Got N emb:', len(res))
            return res


    def cass_get_useremb_dict(easyids, cass_query=None, prefix="m001", tablename="ndata.item_model")  :
        """   easyid = [  'm001_197550_10255674', 'm001_222208_1009782' ]
        """
        log('getting', tablename)
        cass_query = cass_getconn()  if cass_query is None else cass_query

        log('Query N:', len(easyids))
        npref  = len(prefix)+1
        easyids  = [f"{prefix}_{t}" for t in easyids]
        easyids  = cass_query.ndata_user_read(easyids, tablename= tablename)
        easyids  = {k[npref:]: cass_decode(v) for k, v in easyids.items()}  ### keep only siid

        log( 'Query Res:', len(easyids), str(easyids)[:100])
        return easyids


    def pd_easyid_topk_useremb(df, uservect=None, faiss_index=None, faiss_map_idx=None, use_dict=False, topk=1000) :
        log('###### Embedding Rec ')
        if len(df) < 1: return pd.DataFrame([], columns=['easy_id', 'topk_user'])
        if faiss_index is None :
            dirindex    = glob_glob(dir_ca + f"/daily/item_vec/ca_items_{today1}/faiss/*.index" , 1)[0]
            faiss_index = dirindex  ;  log(dirindex)

        ### Topk from User Vector
        uservect = uvector() if uservect is None else uservect
        df2 = uservect.get_multi( df['easy_id'].values, use_dict=False, use_cache=False )  ### All from Cass
        df2 = pd.DataFrame.from_dict(df2, orient='index', columns= ['user_emb'] ).reset_index()
        df2.columns = ['easy_id', 'user_emb']

        df2 = faiss_topk2(df = df2,   colid='easy_id', colemb='user_emb', topk= topk, npool=1, nrows=10**9, nfile=1000,
                          faiss_index=faiss_index, map_idx_dict= faiss_map_idx   )

        df2         = df2[[ 'easy_id', 'easy_id_list' ]]
        df2.columns = ['easy_id', 'topk_user' ]
        df2['topk_user'] = df2.apply(lambda x : pd_siid_cleangenre2(x, topk=1000) , axis=1)          ####
        df2['easy_id'] = df2['easy_id'].astype('int64')

        df               = df.merge(df2, on='easy_id', how='left')
        df['topk_user']  = df['topk_user'].fillna("")
        df               = df[ df.topk_user.str.len() >50  ] ; log('N_topk', df2.shape)
        return df


    def pd_siid_cleangenre2(x, topk=1000):
        return x['topk_user']


    def daily_user_topkemb(bmin=0, bmax=100, tag='intraday',  add_days= 0, split=0):          ###
        """  python prepro_prod.py  daily_user_topkemb  --bmax 1   >> "${logfile}_topkemb.py"  2>&1   &
             ONLY User of today, add yesterday topk 1000 : from 00h00 -> 4am
             6mins for 1 bucket, in full mode (30mio)  --> 150 mins in total,  00h30 --> 5am30 --> 5 hours (full mode)

             python prepro_prod.py daily_user_topkemb --tag intraday --add_days 0 --bmin 100 --bmax 125  --tag all  >> "${logfile}_topkem_100.py"  2>&1   &

             python prepro_prod.py daily_user_topkemb --tag eod --add_days -1 --bmin 0 --bmax 3  --tag all  >> "${logfile}_topkem_100c.py"  2>&1   &

             29513484 in cass Now

        """
        today  = date_now_jp("%Y%m%d", add_days= -0 + add_days,  timezone= 'jp')
        tk     = get_timekey() -0 + add_days
        if split > 0 :
            batch_split_run(cmd=f"{PYFILE} daily_user_topkemb --split 0 --tag {tag} --add_days {add_days} ", split=split, sleep=100 ) ; return 'done'

        dirout = dir_cpa3 + f"/hdfs/daily_useremb/topk/"         ### Prev day, Current day emb
        log("\n", dirout)

        log("############## Faiss Loading ##########################################")
        faiss_index, map_idx_dict, dirindex = intraday_load_faiss()
        uservect  = uvector(nmax=10**6, ttl_cache= 86000) ### only 1 day for uservect in Cache

        log("############## easyid Loading #########################################")
        log("##### Load eaysid:", tag)
        if 'intraday' in tag :  flist = glob_glob(dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*browsing*.parquet" , 1000 )
        elif 'all' in tag:      flist = glob_glob(dir_cpa3 + f"/hdfs/daily_user_eod/{tk-1}/sc_stream/*.parquet", 1000 )
        elif 'test' in tag:     flist = glob_glob(dir_cpa3 + f"/hdfs/daily_user_hist/{tk-1}/sc_stream/*.parquet", 1000 )


        df2   = pd_read_file2(flist, cols=['easy_id'], drop_duplicates=['easy_id'], n_pool=20)   ;  log(df2.shape)
        df2   = df2.drop_duplicates('easy_id')
        df2['bk'] = df2['easy_id'] % 500
        df2       = df2[(df2.bk >= bmin) & (df2.bk < bmax ) ]
        log(df2.shape, flist[0])


        log("\n############## Top-k from user emb ##################################")
        for bk in range(bmin, bmax):
            dirouti = dirout + f"/{tk+1}/useremb_topk_{bk}.parquet"
            # if os.path.isfile(dirouti)  : continue
            log("\n", bk, dirouti)

            ##### Current bucket easyid
            df2i = pd_easyid_topk_useremb(df2[df2.bk == bk], uservect=uservect, faiss_index=faiss_index,
                                          faiss_map_idx=map_idx_dict)
            log(df2i.shape, )

            ##### Previous day easyid, topk
            fk  = dirout  +  f"/{tk}/useremb_topk_{bk}.parquet"
            df1 = pd_read_file(fk, cols=['easy_id', 'topk_user'])

            ###  Merge and keep last
            df1 = pd.concat(( df1, df2i[[ 'easy_id', 'topk_user'  ]] ))  ; del df2i ; gc.collect
            df1 = df1.drop_duplicates('easy_id', keep='last')
            pd_to_file(df1, dirouti , show=1 )


    def pd_useremb_load(t0, bk=-1, uemb=0 ):   #### Load user:  easyid, user_emb
        dfe = None
        if uemb > 0:
            dirine = dir_cpa3 + f"/hdfs/daily_useremb/emb/{t0}*/*_{bk}_*.parquet"
            dfe    = pd_read_file(dirine, npool=1)
            if len(dfe) < 1: dfe = None
            log("#### Loaded user embed: ", dfe, dirine)
        return dfe

    
    def pd_useremb_add(df, dfe=None):
        if dfe is not None:  #### Add User Embedding  user_emb
            df = df.merge(dfe, on='easy_id', how='left')
            log(df, type(df['user_emb'].values[0]), 'no-NA user emb: ', len(df[-df.user_emb.isna()]))
            df['user_emb'] = df['user_emb'].fillna('0')  ### user with NO embeddings
            df['user_emb'] = df['user_emb'].apply( lambda vi: np.array([float(x) for x in vi.split(",")], dtype='float32'))
        return df




if 'qdrant':
    def drant_insert_daily_emb2(create=0, add_days=0):   ## python prepro_prod.py  drant_insert_daily_emb  >> "${logfile}_qdrant_update.py"  2>&1   &
        ## python prepro_prod.py  drant_insert_daily_emb  --add_days -1 --create 1  >> "${logfile}_qdrant_update.py"  2>&1   &
        from dbvector import Qdrant
        t0 = date_now_jp(add_days = add_days)
        dirin   = dir_cpa3 + f"/ca_check/daily/item_vec/ca_items_{t0}/*.parquet"
        dirin2  = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{t0}/raw3am/*.parquet"
        log(dirin, dirin2)


        #### Get genre info
        df2 = pd_read_file(dirin2, cols = [ 'item_id', 'shop_id', 'genre_id_path' ])
        df2['genre_id'] = df2['genre_id_path'].apply(lambda  x : to_int(x.split("/")[-1]) )
        df2 = pd_add_siid(df2)
        df2 = df2.drop_duplicates('siid')

        colsfeat = ['genre_id', 'siid']  #  'shop_id', 'item_id']
        cols     = ['siid', 'shop_id', 'item_id', 'item_emb',  'price']
        df = pd_read_file(dirin, cols = cols)
        df = df.merge(df2[[  'siid', 'genre_id' ]], on ='siid', how='left')

        df             = df[df.genre_id > 0 ]
        df['genre_id'] = df['genre_id'].astype('int32')
        df['id']       = df['siid'].apply(lambda x : int(x.replace("_", "")) )
        log(df)

        ######### Log qdrant  ######################################################################################
        client = Qdrant(table='ca_daily')
        if create > 0 :
            'Re-creating+Flushing table, 30 sec sleep'; time.sleep(30)
            client.table_create('ca_daily', 512, 'Euclid') ; time.sleep(10)
            res = client.client.create_payload_index('ca_daily', "genre_id")
            log( res.dict() )

        #### client.table_shape()
        log( client.table_info()  )

        log("insert")  ## 1k /sec 4 threads
        client.put_multi(df.iloc[:,:], colemb ='item_emb', colid= 'id', colsfeat= colsfeat, npool=6 , kbatch= 60000)
        if create >0 :
            log( client.client.create_payload_index('ca_daily', "genre_id") )

        log("check`")
        v0          = np.random.rand(512)
        filter_dict = {'genre_id': 564203 }
        dfr         = client.get_multi(v0, filter_dict= filter_dict, topk=100,  mode='pandas')
        log(dfr)

    def drant_init2(table='ca_daily'):
        try:
            from dbvector import Qdrant;
            global clientdrant ; clientdrant = Qdrant(table= table)
            log('clientdrant ok:', clientdrant)
        except Exception as e:
            log(e)

    def drant_useremb_get_topk_siid2(vecti, genreid=-1, topk=100, dimvect=512, filter_cond='must'):
        ####  genrei: list_siid
        global clientdrant
        if isinstance(vecti, str) :
           vecti = np.array([ float(x) for x in  vecti.split(",")] ,  dtype='float32')

        if len(vecti) < dimvect :
            # log('error drant input vect size', )
            return []

        if isinstance(genreid, list) :
           filt  = [ ('genre_id', int(gi) ) for gi in genreid ]
           topki = clientdrant.get_multi(vecti, filt, topk=topk, mode='records', filter_cond='should')

        elif genreid > 0 :
           topki  = clientdrant.get_multi(vecti, {'genre_id': genreid}, topk=topk, mode='records')

        else :
           topki  = clientdrant.get_multi(vecti, None, topk=topk, mode='records')

        # log(topki)
        sids = [ t.get('siid',[''])[0] for t in  topki if t is not None  ]   #### output is :   'genreid' : [324242]
        sids = [ t for t in sids if len(t)> 5 ]  ### remove empty ones
        # log('uservect', len(sid),  sid[:5] )
        return sids

    def rank_adjust(ll1, ll2, kk= 1):
        """ Re-rank elements of list1 using ranking of list2
            20k dataframe : 6 sec ,  4sec if dict is pre-build
        """
        if len(ll2) < 1: return ll1
        if isinstance(ll1, str): ll1 = ll1.split(",")
        if isinstance(ll2, str): ll2 = ll2.split(",")
        n1, n2 = len(ll1), len(ll2)

        if not isinstance(ll2, dict) :
           ll2 = {x:i for i,x in enumerate( ll2 )  }  ### Most costly op, 50% time.

        adjust, mrank = (1.0 * n1) / n2, n2
        rank2 = np.array([ll2.get(sid, mrank) for sid in ll1])
        rank1 = np.arange(n1)
        rank3 = -1.0 / (kk + rank1) - 1.0 / (kk + rank2 * adjust)  ### Score

        # Id of ll1 sorted list
        v = [ll1[i] for i in np.argsort(rank3)]
        return v  #### for later preprocess
        #return ",".join( v)
    
    
    def rank_adjust2(ll1, ll2, kk= 1):
        """ Re-rank elements of list1 using ranking of list2
            20k dataframe : 6 sec ,  4sec if dict is pre-build
        """
        if len(ll2) < 1: return ll1
        if isinstance(ll1, str): ll1 = ll1.split(",")
        if isinstance(ll2, str): ll2 = ll2.split(",")
        n1, n2 = len(ll1), len(ll2)

        if not isinstance(ll2, dict) :
           ll2 = {x:i for i,x in enumerate( ll2 )  }  ### Most costly op, 50% time.

        adjust, mrank = (1.0 * n1) / n2, n2
        rank2 = np.array([ll2.get(sid, mrank) for sid in ll1])
        rank1 = np.arange(n1)
        rank3 = -1.0 / (kk + rank1) - 1.0 / (kk + rank2 * adjust)  ### Score

        # Id of ll1 sorted list
        v = [ll1[i] for i in np.argsort(rank3)]
        return ",".join( v)


    def rank_adjust3(ll1, ll2,):
        """ Re-rank elements of list1 using ranking of list2"""
        if isinstance(ll1, str): ll1 = ll1.split(",")
        if isinstance(ll2, str): ll2 = ll2.split(",")
        n1, n2 = len(ll1), len(ll2)
        if n2 < 1: return ll1

        ll2 = {x:i for i,x in enumerate( ll2 )  }

        # log(ll1) ; log(ll2)

        adjust, mrank = (1.0 * n1) / n2, n2
        rank3 = np.zeros(n1, dtype='float32')
        kk    = 2
        for rank1, sid in enumerate(ll1):
            rank2        = ll2.get(sid, mrank)
            rank3[rank1] = - rank_score2(rank1, rank2, adjust=adjust, kk= kk)

        # Id of ll1 sorted list
        v = [ll1[i] for i in np.argsort(rank3)]
        return ",".join( v)


    def rank_score2(rank1, rank2, adjust=1.0, kk=1.0):
        return 1.0 / (kk + rank1) + 1.0 / (kk + rank2 * adjust)


    def drant_insert_daily_emb(create=0, add_days=0):   ## py2  hnsw_insert_daily_emb  >> "${logfile}_qdrant_update.py"  2>&1   &
        ## python prepro_prod.py  drant_insert_daily_emb  --add_days -1 --create 1  >> "${logfile}_qdrant_update.py"  2>&1   &
        t0 = date_now_jp(add_days = add_days)
        dirin   = dir_cpa3 + f"/ca_check/daily/item_vec/ca_items_{t0}/*.parquet"
        dirin2  = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{t0}/raw3am/*.parquet"
        log(dirin, dirin2)

        #### Get genre info
        df2 = pd_read_file(dirin2, cols = [ 'item_id', 'shop_id', 'genre_id_path' ])
        df2['genre_id'] = df2['genre_id_path'].apply(lambda  x : to_int(x.split("/")[-1]) )
        df2 = pd_add_siid(df2)
        df2 = df2.drop_duplicates('siid')

        colsfeat = ['genre_id', 'siid']  #  'shop_id', 'item_id']
        cols     = ['siid', 'shop_id', 'item_id', 'item_emb',  'price']
        df = pd_read_file(dirin, cols = cols)
        df = df.merge(df2[[  'siid', 'genre_id' ]], on ='siid', how='left')

        df             = df[df.genre_id > 0 ]
        df['genre_id'] = df['genre_id'].astype('int32')
        # df['id']       = df['siid'].apply(lambda x : int(x.replace("_", "")) )
        log(df)

        log("######### Insert   HNSW")
        idx = np.arange(0, len(df))
        df['idx'] = idx

        #### M=32, ef_cons = 64 , ef_search 32
        import hnswlib
        p = hnswlib.Index(space   = 'l2', dim = 512)
        p.init_index(max_elements = len(df), ef_construction = 64, M = 32 )
        p.set_num_threads(4)
        vlist = [  np.array(vi.split(","), dtype='float32') for vi in df['item_emb'].values ]
        p.add_items( vlist )
        for ii, gi in zip(idx, df['genre_id'].values ) :
            p.add_tags( [ii], gi )

        #### Index tag for better search
        for gi in  df['genre_id'].unique():
          p.index_tagged( gi )

        #### Cross tag neighbor
        # p.index_cross_tagged(neighbors)

        dirout = dirin.split("*")[0] + "/hnsw/"
        log("###### Saving index to", dirout )
        os.makedirs(dirout, exist_ok=True)
        p.save_index(dirout + "/index.bin")
        df['id']  = df['siid']
        pd_to_file(df[[ 'idx', 'id', 'genre_id' ]], dirout +"/index.parquet")


        log("###### Query check ")
        v0        = vlist[10]
        filters   = [[(False,  df['genre_id'].values[10]  )]]
        p.set_ef(32)  # higher ef leads to better accuracy, but slower search, recall
        idx, dist = p.knn_query([v0 ], k=5, conditions=filters )
        log(df[['siid', 'genre_id']].iloc[idx[0], :] , filters )


    def drant_init(dirin="", ndim=512 , efsearch= 32, add_days=0):
        global clientdrant
        if clientdrant is not None :
            log("Not empty clientdrant"); return None

        t0    = date_now_jp(add_days=add_days)
        dirin = dir_cpa3 + f"/ca_check/daily/item_vec/ca_items_{t0}/hnsw/"

        clientdrant = Box({})
        import hnswlib
        p = hnswlib.Index(space= 'l2', dim= ndim)
        try :
           p.load_index(dirin + "/index.bin")
        except Exception as e :
           log(e)
           t1    = date_now_jp(add_days=add_days-1)
           dirin = dir_cpa3 + f"/ca_check/daily/item_vec/ca_items_{t1}/hnsw/"
           log("Load previous day", dirin)
           p.load_index(dirin + "/index.bin")

        p.set_ef( efsearch)   #### Speed/Precision

        dd = pd_read_file(dirin + "/index.parquet")
        dd = dd.set_index('idx').to_dict()
        index  = dd['id']   ## 0 --> siid
        indexg = dd.get('genre_id', {})
        clientdrant.p = p ; clientdrant.index = index ; clientdrant.indexg = indexg
        log('clientdrant loaded',  len(clientdrant.index))


    def drant_useremb_get_topk_siid(vecti, genreid=-1, topk=100, dimvect=512, filter_cond='must'):
        ####  genrei: list_siid
        global clientdrant

        # hnsw_init(dirin=dirout, ndim=512 )
        if isinstance(vecti, str) :
           vecti = np.array([ float(x) for x in  vecti.split(",")] ,  dtype='float32')

        if len(vecti) < dimvect :
            # log('error drant input vect size', )
            return []

        if isinstance(genreid, list) :
           idxall = []
           #vects  = [ vecti ]  #  [ vecti for i in range(0, len(genreid)) ]
           for gi in genreid :
              filters    = [[ (False, int(gi)) ]]    #            ## (isexcluded:false, catid)
              idxallj,_  = clientdrant.p.knn_query( vecti, k = topk, conditions = filters   )
              idxall.append( idxallj[0] )

        else :
           idxall,_   = clientdrant.p.knn_query( vecti, k =topk,  )

        sids_group = [ [clientdrant.index[ix] for ix in ixlist] for ixlist in idxall  ]

        ### Genre mapping
        # gids_group = [ [clientdrant.indexg[ix] for ix in ixlist] for ixlist in idxall  ]

        #df[['siid', 'genre_id']].iloc[idxall, :]
        #df.iloc[idxall, :].groupby('genre_id').agg({'id': 'count'})
        # sids_group = [ t for t in topki if len(t)> 5 ]  ### remove empty ones
        # log('uservect', len(sid),  sid[:5] )
        return sids_group




if 'rec_genre':
    def map_siid_imaster():  ##  python prepro_prod.py  map_siid_imaster &
        log("map_daily_ca_genre_siid  from Index")  ####  CA siid list
        today = date_now_jp("%Y%m%d", add_days=-1)
        # today = "*"

        cols = ['siid', 'genre_id_path', 'item_image_url']

        dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean/*.parquet"
        df = pd_read_file(dirin, cols=cols, nfile=1000, n_pool=20, nrows=5000500050)
        log(dirin, df)
        ##### siid --> Imaster Infos
        df['imaster'] = df.apply(lambda x: {'genre_path': x['genre_id_path'], 'image_url': x['item_image_url']}, axis=1)
        diskcache_save2(df, db_path=db['db_imaster'], colkey='siid', colvalue='imaster', npool=5, verbose=False,
                        ttl=86400 * 10)

    def recgenre_init_proba():
        global lproba_dict
        lproba_dict = {}
        for n in range(1, 5000 + 1):
            # lproba = [ 1.0 / np.log(1+t) for t in range(0, n) ]
            lproba = [1.0 / (2 + t) for t in range(0, n)]  ### Less fait tails
            # lproba = [ 1.0 / (1+t**1.4) for t in range(0, n) ]   ### Less fait tails

            lproba = lproba / np.sum(lproba)
            lproba_dict[n] = lproba

    def np_sample(lname, lproba=None, mode='norm', k=5, replace=False):
        ### Pre calculated proba
        global lproba_dict
        lname = lname[:5000]  ## top 5000
        n = len(lname)
        lproba = lproba_dict[n]

        #if mode == 'inverse'  :
        #  #lproba = [ 1.0 / np.log(2+t) for t in range(0, n) ]
        #  #lproba = lproba / np.sum(lproba)
        #  lproba = lproba_dict[n]
        #
        #elif mode == 'norm'  :
        #  lproba = np.array(lproba)
        #  lproba = np.log(1+lproba)
        #  lproba = lproba / np.sum(lproba)
        # log( np.sum(lproba) )

        ll = np.random.choice(lname, size=min(k, n), p=lproba, replace=replace)
        return list(ll)

    def dict_sorted_keys(dd):
       return [k for k, _ in sorted(dd.items(), key=lambda item: item[1], reverse=True  ) ]

    def create_rec_topgenre_old(df1, topgenre='pur', ngenre=5, topk=6):
        # log('######### Getting top genre', topgenre )
        if topgenre == 'pur':  df1['top_genre'] = df1['easy_id'].apply(lambda x: db_easyid_topgenre_pur.get(x, ""))
        if topgenre == 'brw':  df1['top_genre'] = df1['easy_id'].apply(lambda x: db_easyid_topgenre_brw.get(x, ""))
        if topgenre == 'intra':  df1['top_genre'] = df1['easy_id'].apply(lambda x: db_easyid_topgenre_intra.get(x, ""))
        # log(df1['top_genre'])
        coltop = f"topk_{topgenre}"

        log('######### Adding topk_genre siid', topgenre)
        # log(db_ca_genre_siid, len( db_ca_genre_siid) )
        # df1[coltop] = df1[ 'top_genre'].apply(lambda x : easyid_create_daily_recgenre(x, ngenre=ngenre, topk=topk)  )
        # if 'user_emb' in df1.columns : log( df1['user_emb'].shape )
        df1[coltop] = df1.apply(lambda x: easyid_create_daily_recgenre2(x, ngenre=ngenre, topk=topk), axis=1)

        # log(df1, df1.head(1).T)
        log('Missing', coltop, len(df1[df1[coltop].str.len() < 100]))
        df1[coltop] = df1[coltop].fillna("")
        return df1

    def easyid_create_daily_recgenre2_old(x, ngenre=6, topk=6, nmax=31):
        ####    'g1,g2,g3 ;   3445,4566,566'
        ####    db_ca_genre_siid is updated daily
        global db_ca_genre_siid
        glist = x['top_genre']
        glist = glist.split(";")[0].split(",")
        glist = glist[:ngenre]
        ng = len(glist)

        topk = {1: 30, 2: 15, 3: 10, 4: 8, 5: 6, 6: 5}[ng]

        # log('ng', ng,  len(db_ca_genre_siid) )
        if 'user_emb' in x:  uvecti = x['user_emb']
        ll = []
        # t0 = time.time()
        for gpath in glist:
            #x = db_ca_genre_siid.get( to_int(gpath), None)  ### list of CA siid per genre_id  A lot of Miss
            u = db_ca_genre_siid.get(gpath, None)  ### gpath as STRING
            # log(gpath, x)
            if u is None: continue
            u = u.split(";")
            #  siids, proba = x[0].split(","),  [float(t) for t in x[1].split(",")]
            siids, proba = u[0].split(","), None
            # log('siids', siids)
            rec = np_sample(siids, proba, k=topk, mode='inverse', replace=False)

            #### Adjustment with User x Top-Popular
            #if 'user_emb' in x:
            #    # log('get rec2 useremb', gpath )
            #    rec2 = drant_useremb_get_topk_siid(uvecti, genreid= int(gpath), topk=20)
            #    if len(rec2) > 0:    rec = rank_adjust2(rec, rec2)
            #    # log('uservect', gpath, len(rec2))  #, rec2[:2])

            ll.append(rec)
        #log('dt:',  time.time()-t0)

        ##### Merge together  #############################################################
        kk = 0
        ss = ""
        for xi in chain.from_iterable(zip_longest(*ll)):
            if kk > nmax: break
            if xi is not None:
                ss = ss + xi + ","
                kk = kk + 1

        #### Adjustment with User x Top-Popular, only one time
        if 'user_emb' in x:
            # log('get rec2 useremb', gpath )
            rec2 = drant_useremb_get_topk_siid(uvecti, genreid=-1, topk=200)
            # log(str(rec2)[:80])
            if len(rec2) > 0:   ss = rank_adjust2(ss[:-1].split(","), rec2)
            # log('uservect', gpath, len(rec2))  #, rec2[:2])
            return ",".join(ss)

        return ss[:-1]


    def create_rec_topgenre(df1, ngenre=5, topk=6):
        global ca_genreid_global
        log('######### Adding topk_genre siid',)
        coltop = 'topk_genre'
        ca_genreid_global = diskcache_getkeys(db['db_ca_genre_siid']) if ca_genreid_global is None else ca_genreid_global


        ff = { i : np.exp(-0.1*i) for i in range(0,20) }  ### Adjustement factor for top-genre weighting
        df1['top_genre'] = df1.apply(lambda x: easyid_create_daily_genre(x, ca_genreid_global, nmax=topk, ff=ff), axis=1)
        log( df1.top_genre )
        df1[coltop]      = df1.apply(lambda x: easyid_create_daily_recgenre3b(x, nmax=topk), axis=1)

        # log(df1, df1.head(1).T)
        log('Missing', coltop, len(df1[df1[coltop].str.len() < 10]))
        df1[coltop] = df1[coltop].fillna("")
        return df1


    def easyid_create_daily_genre(x, ca_genreid=None, nmax=40, ff=None, **kw):
        easyid  = x['easy_id']
        ##### Genre extraction/Scoring
        gall = {}
        for i, gi in enumerate( db_easyid_topgenre_pur.get(easyid, "").split(";")[0].split(",") ) :
          if gi in  ca_genreid and i < 20:
             gall[gi] = 2*ff[i]   if gi not in gall else gall[gi] + 2*ff[i]

        for i,gi in enumerate(db_easyid_topgenre_brw.get(easyid, "").split(";")[0].split(",") ):
          if gi in  ca_genreid and i < 20:
             gall[gi] = 1*ff[i]  if gi not in gall else gall[gi] + 1*ff[i]

        for i,gi in enumerate(db_easyid_topgenre_intra.get(easyid, "").split(";")[0].split(",")) :
          if gi in  ca_genreid and i < 20:
             gall[gi] = 3*ff[i] if gi not in gall else gall[gi] + 3*ff[i]

        for i,gi in enumerate(db_easyid_topgenre_freqpur.get(easyid, "").split(";")[0].split(",")) :
          if gi in  ca_genreid and i < 20:
             gall[gi] = 2*ff[i] if gi not in gall else gall[gi] + 2*ff[i]

        ###### List of genre sorted by score
        gall = dict_sorted_keys( gall )
        return ",".join(gall)


    def easyid_create_daily_recgenre3b(x, ca_genreid=None, nmax=40, ff=None, **kw):
        easyid  = x['easy_id']
        ng_list = x['ng_list'] if 'ng_list' in x  else set()
        gall    = [ to_int(t,0) for t in  x['top_genre'].split(",") ]
        gall    = [ t for t in gall if t>0 ]

        ###### Useremb, topk per useremb
        useremb = x['user_emb'] if 'user_emb' in x else []
        if isinstance( useremb, str):
           useremb = [float(t) for t in  useremb.split(",") ]

        def top_popular(gall):        
            topk_groups = []
            for gi in gall :
                u = db_ca_genre_siid.get(str(gi), None)  ### gpath as STRING
                if u is None: continue
                u = u.split(";")
                siids, proba = u[0].split(","), None
                rec = np_sample(siids, proba, k=30, mode='inverse', replace=False)  ### from top-sales
                topk_groups.append(rec)
            return topk_groups    
                
        if len(useremb) < 2:  ### [0] :No embedding
            topk_groups = top_popular(gall)
            
        else :
            topk_groups = drant_useremb_get_topk_siid(useremb, genreid= gall, topk=30, dimvect=512)  ### return list of list
            
            """
            topk_groups1 = drant_useremb_get_topk_siid(useremb, genreid= gall, topk=30, dimvect=512)  ### return list of list            
            topk_groups2 = top_popular(gall)   ### list of list
                
            topk_groups = []    
            for l1, l2 in zip(topk_groups1, topk_groups2):
                if   l1 is None : topk_groups.append(l2)
                elif l2 is None : topk_groups.append(l1)
                else :                   
                  l1 = rank_adjust(l1, l2)  ### Merge Popular with Relevance
                  topk_groups.append(l1)
            """  
        # list of list
        ss          = pd_siid_combine5b(topk_groups, ngs=ng_list, n1=1, n2=0, n3=0, n4=0, nmax=nmax)
        return ss


    ####### Standalone generator
    def daily_eod_recgenre(ii=0, ii2=100, tag='new_rename', add_days=0, uemb=0, split=0):  ### py  daily_eod_topk2  --ii 0  --ii2 1 --tag new_rename  >> "${logfile}_00_eod_topk.py"  2>&1   &
        """  py2  daily_eod_recgenre  --ii 0  --ii2 5  --tag new_rename  --add_days 0   --uemb 1 >> "${logfile}_00_eod_topkgenre.py"  2>&1   &
             daily_rec_genre(ii=112, ii2=113, tag='new_rename', add_days=0, uemb=1)
               195.7356719970703 for 1000 easy with useremb
        """
        t00    = ddt()
        today0 = date_now_jp("%Y%m%d", add_days=add_days, add_hours=0, timezone='jp')
        if split > 0 :
            batch_split_run(f"{PYFILE}  daily_eod_recgenre --split 0 --tag {tag} --add_days {add_days} ", split=split, sleep=130 ) ; return 'done'


        ##### Input Data in T-1
        tk = get_timekey() - 1 + add_days
        if 'new' in tag :    dirin_list = [f"/a/gfs101/ipsvols07/ndata/cpa/hdfs/daily_user_eod/", ]
        elif tag == 'sc':    dirin_list = [f"/a/acb401/ipsvols06/pydata/sc_widget_pur/"]
        else:  return 'no tag early finish'
        log('users', dirin_list);  # time.sleep(1)

        #### Output in T
        tout   = get_timekey()
        dirout = dir_cpa3 + f"/hdfs/daily_usertopk/{model0['pref']}/{tout}/"
        log("\n", dirout)

        log("############## Faiss Loading T ########################################")
        # faiss_index, map_idx_dict, dirindex = intraday_load_faiss()
        ivect = ivector(use_dict=False); log(ivect)
        try:
            global clientdrant ; drant_init(dirin="", ndim=512, add_days= add_days )
        except Exception as e:
            log(e)

        log("############## Top-k ##################################################")
        #### Default topk popular
        recgenre_init_proba()  #### Fixed proba for sampling  topk: 1-K
        bkmin = ii; bkmax = ii2
        for bk in range(bkmin, bkmax):
            flist = [];
            for dirin in dirin_list:
                flist += glob_glob(dirin + f"/{tk}/sc_stream/*_{bk}_*.parquet", 5000)

            df = pd.DataFrame()
            for jj, fi in enumerate(flist):
                log("\n\n", fi)
                group   = dirin.split("/")[-2] + "_genre"
                dirouti = dirout + f"/{group}/topk_{bk}_{tout}"
                diroutj = dirout + f"/{group}/topk_{bk}_{tout}_*.parquet"
                if len(glob.glob(diroutj)) > 0: continue

                df = pd_userhist_load(fi, only_last_siid=1 )     ### eays_id, siid, siids, can have mutiple   easyid, siid  pairs (historical)
                df = pd_add_easyid_bk(df) ; log(df.bk.mean() ) ; del df['bk']

                ################################################################################
                dfe = pd_useremb_load(t0=tk, bk=bk, uemb= uemb )  ### usre emb dfe= None or dataframe
                dfe = pd_add_easyid_bk(dfe) ; log(dfe.bk.mean() )  ; del dfe['bk']
                if uemb > 0 and dfe is not None:
                    # df = df.iloc[:1000, :]
                    df = pd_useremb_add(df, dfe)

                #### 'ng_siid' column, need the bucket bk to match !!!!!!!!
                df = pd_add_ng_easysiid(df, bk=bk)  ####

                log('\n###### Genre pur Rec: eaysid --> genre --> list of siid, intra, brw, pur ')
                df = create_rec_topgenre(df,  ngenre=6, topk=30)

                if 'ng_siid'  in df.columns:  del df['ng_siid']
                if 'user_emb' in df.columns:  del df['user_emb']

                df = pd_topk_count(df, [ 'topk_genre',   ])
                ################################################################################

                df = df[df['topk_genre'].str.len() > 8 ]
                log('  Cleaned topk', df.shape)
                log(df[list(df.columns)[-2:]], df.columns)
                ## ['easy_id', 'siid',  'top_genre',  'topk_genre','topk_genre_n''],
                pd_to_file(df, dirouti  + f"_{len(df)}.parquet", show=0)
        log("############## Finished ", ii2,  ddt(t00) )




if 'rec_item_emb':
    #############################################################################################################
    def  daily_eod_user_hist_item(add_days=-1, tag=""):  ###  python prepro_prod.py  daily_eod_user_hist_item    >> "${logfile}_histitem.py"  2>&1   &
        tk     = get_timekey()  + add_days
        dirin  = dir_cpa3 + f"/hdfs/daily_user_eod/{tk}/sc_stream/*.parquet"
        dirout = dir_cpa3 + f"/hdfs/daily_user_eod/{tk}/sc_stream_item/"

        cols  = [ 'shop_id', 'item_id', 'genre_id'  ]
        log(dirin, dirout)
        flist = glob_glob(dirin, 1000)

        df = pd.DataFrame() ; fj =[]
        for ii, fi in enumerate(flist) :
           fj.append(fi)
           if len(fj)< 100  and ii < len(flist)-1 : continue
           dfi    = pd_read_file2( fj, n_pool=25 , cols=cols,  drop_duplicates = cols )
           dfi    = dfi.drop_duplicates(cols)
           df = pd.concat((df, dfi))
           df = df.drop_duplicates(cols)
           log(ii, df.shape)
           fj = []

        log(df.shape, df.columns)
        df['bk'] = df['item_id'] % 500
        df       = pd_add_siid(df, delete=True)

        for bi in df['bk'].unique():
            dfi = df[df.bk == bi ]
            pd_to_file(dfi, dirout + f"/siid_list_{bi}.parquet", show=0)


    def update_item_topk(test=0, bmin=0, bmax=0, tag="intraday", split=0, add_days=0,
                         use_intra=False) :  ### python prepro_prod.py update_item_topk  --test 1  >>  "${logfile}_item_topk2.py" &
        """   Depend on Faiss Calc
             #dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/df/"

             python prepro_prod.py update_item_topk  --test 1  --tag eod --bmin 13  --bmax 14  --split 3 >>  "${logfile}_item_topk_eod3.py" &

             #### 2h30  to update all db_cache, 7mio items...
             python prepro_prod.py update_item_topk  --test 1  --tag eod --bmin 13  --bmax 14  --split 10 >>  "${logfile}_item_topk_eod5.py" &


        """
        tk = get_timekey()
        today    = date_now_jp("%Y%m%d", timezone='jp', add_days= 0  + add_days )
        today1   = date_now_jp("%Y%m%d", timezone='jp', add_days= -1 + add_days)
        overwrite = True
        log(today, bmin, bmax)

        if split > 0 :
            batch_split_run(cmd=f"{PYFILE}  update_item_topk --split 0 --tag {tag} --add_days {add_days} ", split=split, sleep=130 ) ; return 'done'

        if 'intraday' in tag :   dirin   = dir_cpa3 + f"/hdfs/intraday/sc_stream_item/{today}/" + "/*0.parquet"
        if 'eod' in tag :        dirin   = dir_cpa3 + f"/hdfs/daily_user_eod/{tk-1}/sc_stream_item/*.parquet"


        log("\n\n############## Check new files  ##############################################")
        flist  = glob_glob(dirin , 1000)
        dirout = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/"
        index  = Index1( dirout  + f"/done_item_topk2_{today}.txt"  )
        flist  = [ t for t in flist if t not in index.read()     ]

        def file_isvalid(fi,tag):
            if 'intraday' in tag:
                tr = date_now_jp("%Y-%m-%d", timezone='jp', add_days= -1)   ### --20: 6am,  --21: 7am
                rref = [ tr + "--14",   tr + "--15", tr + "--16", tr + "--17", tr + "--18", tr + "--19", tr + "--20", 'in-progress'  ]  ### tr + "--20",
                for r in rref:
                    if r in t : return False
                return True

            if 'eod' in tag:    ### bucket bmin, bmax
                try :
                   bk= to_int( fi.split("/")[-1].split("_")[2].replace(".parquet", "") )   ###  siid_list_0.parquet
                   # bk= to_int( fi.split("/")[-1].split("_")[1] )   ###  user_eod_intra_0_18978_20211217_214221.parquet
                   return bk >= bmin and bk < bmax
                except : return False

        flist  = [t for t in flist if file_isvalid(t, tag)]
        log(flist)
        if len(flist) < 1 : log('No new files', len(flist)) ; return 1

        log("############## Faiss Loading T   #################################################")
        faiss_index, faiss_map_idx, dirindex = intraday_load_faiss()
        ivect  = ivector(use_dict=False) ; log(ivect)
        dgenre = dict_load_genre(today, today1)   ### siid --> genreid  for today product

        #dfk = pd_load_hist_intra(df0=None, add_days=add_days, tag='intra,clk', bmin=bmin, bmax=bmax)

        log("\n############## Loop over files: topk emb  ######################################")
        cols0 = ['shop_id', 'item_id']
        ii = 0; fj = []
        for ii, fi in enumerate(flist) :
            fj.append(fi)
            #if fi in index.read()      : continue
            index.save([fi])
            if len(fj) < 20 and ii < len(flist)-1  : continue
            log("\n", fj)

            if 'eod' in tag :  df = pd_read_file( fj, n_pool=10  )
            else :             df = pd_read_file( fj, n_pool=5 , drop_duplicates = cols0 )
            # df = pd.concat((df, dfk[  list(df.columns)]))

            log(df.shape)
            if test>0 :       log('using test'); df = df.iloc[:2000, :]
            if len(df) < 1 :  log('Empty df:', fi) ; continue

            if 'shop_id' in df.columns :
                df = df[[ 'shop_id', 'item_id', 'genre_id'  ]].drop_duplicates(cols0, keep='last')
            # if 'ts' in df.columns : df = df.sort_values('ts', ascending=0)
            # df = df.groupby('easy_id').tail(3)
            df = pd_add_siid(df, delete=True)
            log('unique siids', df.shape)

            pd_item_topk_update(df, faiss_index=faiss_index, faiss_map_idx=faiss_map_idx, ivect=ivect, dgenre=dgenre,
                                overwrite= overwrite, npool=1 )
            fj =[]
            now(ii)
        log('end', bmin, bmax)

        
    def pd_siid_cleangenre3(x, topk=8, dgenre=None):
        #### Filter into same genre
        siids = x['topk'].split(",")

        try :
            g0 = str(x['genre_id'])
        except :
            g0 = str(db_ca_siid_genre.get(x['siid'], ""))  ### Unknown one

        ss = "" ; ii = 0 ; lexist = set()
        for sid in siids :
            if sid in lexist : continue
            g1 = dgenre.get(sid, None)
            if g1 is None : g1 = str(db_ca_siid_genre.get(sid,""))
            if str(g1) == g0  :
                 ii + ii + 1
                 ss = ss  + sid + ","
                 lexist.add(sid)
            if ii > topk: break
        return ss[:-1]


    def pd_item_topk_update(df, faiss_index=None, faiss_map_idx=None, ivect=None, dgenre=None, overwrite=True, npool=1, append=False ):
        ####  siid, genreid
        # if 'siid' not in df.columns : df['siid'] = df.apply(lambda x : siid(x) , axis=1)

        def fun_append(x1, x2):
            if len(x2)< 1 : return x1
            else :  return ",".join(x1.split(",")[:4]) + "," + x2

        if not overwrite :
            df = df[ -df['siid'].apply(lambda x : x in db_item_toprank)] ; log('Unique 2', df.shape)

        df2 = pd_cass_get_vect2(df, prefix= model0.pref, tablename="ndata.item_model", ivect=ivect )
        df2 = faiss_topk3(df = df2,   colid='siid', colemb='item_emb', topk= 80, npool=1, nrows=10**9, nfile=1000,
                          faiss_index=faiss_index, map_idx_dict= faiss_map_idx   )
        df2 = df2[[ 'siid', 'siid_list' ]]
        df2.columns = ['siid', 'topk' ]
        df          = df.merge(df2, on='siid', how='left')  ### genre_id    siid       topk
        df = df[ df.topk.str.len() > 10 ]  ; log(df.shape)

        log('Normalizing Genre')  ##### too slow, need in RAM
        df['topk'] = df.apply(lambda x : pd_siid_cleangenre3(x, topk=20, dgenre=dgenre) , axis=1)   #### need db_ca_siid_genre
        log(df[[ 'siid', 'topk' ]])

        ##### Re-Rank using top-sales
        df['topk'] = df.apply(lambda x:  rank_adjust2(x['topk'], db_ca_genre_siid.get(x['genre_id'], "" ) ) , axis=1)
        # df['topk2'] = df.apply(lambda x:  rank_adjust2(x['topk'], x['topk'] ) , axis=1)
        # df['topk'] = df.apply(lambda x : ",".join( x['topk']), axis=1)
        df = df[ df.topk.str.len() > 10 ] ;


        ##### update db_item_topk
        if append : #### Read and append to existing
            siids = df['siid'].values
            df['topk'] = df.apply(lambda x:  fun_append(x['topk'] , db_item_toprank.get(x['siid'], "" ) )  , axis=1)

        npool = 1
        diskcache_save2(df, colkey='siid', colvalue='topk', db_path= db.db_item_toprank, npool= npool, ttl= 86400*3, verbose=False )


    def faiss_topk3(df=None,  colid='id', colemb='emb', faiss_index=None, topk=200, npool=1, nrows=10**7, nfile=1000, faiss_pars={},
                    map_idx_dict=None,     ) :
       """ id, dist_list, id_list
       """
       mdim = 512
       cc   = Box(faiss_pars)
       log('Faiss Index: ', faiss_index)
       faiss_index.nprobe = 12  # Runtime param. The number of cells that are visited for search.

       ####### Single Mode #################################################
       chunk  = 200000
       kk     = 0
       log(df.columns, df.shape)
       df = df.iloc[:nrows, :]

       dfall  = pd.DataFrame()   ;    nchunk = int(len(df) // chunk)
       for i in range(0, nchunk+1):
           if i*chunk >= len(df) : break
           i2 = i+1 if i < nchunk else 3*(i+1)

           x0 = np_str_to_array( df[colemb].iloc[ i*chunk:(i2*chunk)].values   , l2_norm=True, mdim = mdim )
           # log('X topk', x0.shape )
           _, topk_idx = faiss_index.search(x0, topk)
           log('X', topk_idx.shape)

           dfi                   = df.iloc[i*chunk:(i2*chunk), :] # [[ colid ]]
           dfi[ f'{colid}_list'] = np_matrix_to_str2( topk_idx, map_idx_dict)  ### to item_tag_vran
           # dfi[ f'dist_list']  = np_matrix_to_str( topk_dist )
           # dfi[ f'sim_list']     = np_matrix_to_str_sim( topk_dist )
           dfall = pd.concat((dfall, dfi))
           # log(i, dfi[[ f'{colid}_list', 'sim_list'  ]])
       return dfall


    #### During Prediction  ###############################################
    def create_rec_topemb3(df, faiss_index=None, faiss_map_idx=None, ivect=None, maxrec=30, dfe=None):
        """  siids --> Topk for each (siid, genreid) --> topk, merge.

        """
        def get_topk(siids):
            topks = []
            for sid in siids :
                topks.append( db_item_toprank.get(sid, "").split(",") )

            ii = 0 ; ss=""
            for x in chain.from_iterable(zip_longest( *topks )):
               if ii >= maxrec : break
               x  = str(x)
               if len(x)> 5 :
                   ii = ii + 1
                   ss = ss + x +","
            if len(ss) < 8 : return ""
            return ss[:-1]

        ### df = df.drop_duplicates(['easy_id', 'genre_id'], keep='last')
        df = pd_add_siid(df, delete=True)
        ### Most revent at Bottom, end of list --->  on Top of list by reverse
        df         = df.groupby('easy_id').apply(lambda dfi : dfi['siid'].values[::-1] ).reset_index()
        df.columns = ['easy_id', 'siids']
        log('Neasy id', df )

        if dfe is not None:  #### Add User Embedding  user_emb
           df             = df.merge(dfe, on='easy_id', how='left')
           log(df,  type(df['user_emb'].values[0]) , 'no-NA user emb: ', len(df[-df.user_emb.isna() ]) )
           df['user_emb'] = df['user_emb'].fillna('0')   ### user with NO embeddings
           df['user_emb'] = df['user_emb'].apply(lambda vi:  np.array([ float(x) for x in vi.split(",")] ,  dtype='float32') )

        # df = df.iloc[:100000, :]
        df['siid'] = df['siids'].apply(lambda x: x[0]  )  #### Most recent is 1st
        df['topk'] = df['siids'].apply(lambda x: get_topk(x) )
        log(df)
        return df



if 'ng_list':
    def pd_add_easyid_bk(df):   ### Same than Neil  df = pd_easyid_bucket(df
        df['bk']  = df['easy_id'].apply(lambda x : mmh3.hash(str(x),signed=False)%500 )
        return df

    def easyid_bk(easy_id):   ### Same than Neil  df = pd_easyid_bucket(df
        return  mmh3.hash(str(easy_id),signed=False) % 500

    def pd_add_ng_easysiid(df,bk):  ###merge with ng easy_id siid list
        ### in combine  ngs =  set(x['ng_siid'].split(","))
        # tk  = get_timekey()
        dfn           = pd_read_file( dir_ngsiid2 + f"/*_{bk}.parquet" )
        # log('NG easyid', dfn)
        if len(dfn)< 1:
            df['ng_siid'] = '' ;  return df
        df            = df.merge(dfn, on='easy_id', how='left')

        log( 'NG ok: ', len(df[ -df['ng_siid'].isna() ] ) )
        df['ng_siid'] = df['ng_siid'].fillna('')
        df['ng_siid'] = df['ng_siid'].apply(lambda x : set(x.split(',')) )

        return df

    def pd_add_ng_easysiid_intra(df,bk):  ###merge with ng easy_id siid list
        ### in combine  ngs =  set(x['ng_siid'].split(","))
        tk = get_timekey()
        t0 = date_now_jp()

        dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream_userng/{t0}/*.parquet"
        flist = sorted( glob_glob(dirin, 1000) )
        flist = flist[-4:]
        log(flist)
        dfn           = pd_read_file( flist, n_pool=2 )
        dfn = dfn.drop_duplicates('easy_id')
        # log(dirin,   dfn.shape, dfn )

        if len(dfn)< 1:
            df['ng_siid'] = '' ;  return df
        df            = df.merge(dfn, on='easy_id', how='left')

        log( 'No NA easyid', len(df[ -df['ng_siid'].isna() ] ) )
        df['ng_siid'] = df['ng_siid'].fillna('')
        df['ng_siid'] = df['ng_siid'].apply(lambda x : set(x.split(',')) )
        # log('NG easyid',t0)
        return df


    def ng_get_global_siid(add_days=0):
        today = date_now_jp(add_days= add_days)
        dirin = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/siid_ng/*.parquet"
        flist = sorted( glob_glob(dirin) )
        if len(flist) < 1 : return set()
        else :
            log('Global ng file', flist[-1])
            ngs = pd_read_file(flist[-1])
            if ngs is None : return set()
            ngs = set( ngs.siid.values)
            log("### Nb NG siid loaded::", len(ngs))
            return ngs

    def ok_get_global_siid(add_days=0):
        today = date_now_jp(add_days= add_days)
        flist = sorted( glob_glob( dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/siid_ok/*nglist*.parquet" ) )
        flist+= sorted( glob_glob( dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/siid_ok/*realtime*.parquet" ) )
        if len(flist) < 1 :
            today = date_now_jp(add_days= add_days-1)
            flist = sorted( glob_glob( dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/siid_ok/*nglist*.parquet" ) )
            flist+= sorted( glob_glob( dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/siid_ok/*realtime*.parquet" ) )
            if len(flist) < 1 :
                log('No item available in T, T-1') ; 1/0

        log('Global OK file', flist[-1])
        ngs = pd_read_file(flist[-1])
        if ngs is None : return set()
        ngs = set( ngs.siid.values)
        log("### Nb OK siid loaded::", len(ngs))
        return ngs



##########################################################################################
if 'daily_batch':
    def create_siid_ranid( mode='brw', tag="", nfiles=10000):   ####  python prepro.py v_create_map --tag daily &
        """  python prepro_prod.py v_create_map &
             https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html?highlight=read_table#pyarrow.parquet.read_table
        """
        nmax      = 5005000990
        bkt       = '*' #  '0'    ## 0
        tmin,tmax = get_timekey() - 2, get_timekey()

        tag    = "a"
        dirout = '/a/gfs101/ipsvols07/ndata/cpa/map/'    ;  os.makedirs(dirout, exist_ok=True)
        cols   = ['shop_id', 'item_id', 'ran_id']

        if mode == 'pur':  dirin = dir_pydata  + "/pur_ran_v15/"
        else :             dirin = dir_pydata  + "/brw_ran_v15/"

        #keys     = diskcache_getkeys(db_itemid_vran )
        db_path = db.db_itemid_vran_path
        keys2   = set()

        for tk in range(tmax, tmin, -1 ):
            log('Existing keys', tk-tmin, tk, len(keys2))
            #tk  = tk0
            bkt = "*"

            dir_in = dirin + f"/{bkt}/*{tk}*"
            flist  = sorted(glob.glob(dir_in ))
            flist  = flist[: nfiles]
            filei  = []
            n      = len(flist)
            log(str(flist)[:100])

            dfall = pd.DataFrame()
            for ii in range(n) :
                filei.append(flist[ii])
                if len(filei) < 200  and ii < n-1 : continue
                log('nfiles', len(filei) )
                df    = pd_read_file2(filei, cols = cols,  drop_duplicates= ['shop_id', 'item_id' ],
                                      n_pool= 25, verbose= False )
                filei = []
                df    = df.drop_duplicates([ 'shop_id', 'item_id'])
                log(df)
                df         = pd_add_siid(df, delete=True)
                pd_to_file(df, dirout + f"/map_siid_ranid/ranid_{tk}.parquet")

                keysi      = set(df['siid'].values)
                df         = df[ -df['siid'].isin(keys2) ]
                df         = df[ -df['siid'].isin(db_itemid_vran) ]
                if len(df) < 1 : continue
                # df    = df[[ 'siid', 'ran_id' ]]
                log(df.shape)

                # diskcache_save2(df,  'siid', 'ran_id',  db_path= db_path, npool=1, sqlmode='fast', ttl= 86400 * 30, verbose=False )
                keys2 = keys2.union(keysi)  ### Prevent Duplicates keys
                log(dir_in, len(keys2))


    def ca_today_items(path='latest', colkey='siid', colval='dum', add_days=0 ):  ###   python  prepro.py  ca_today_items --path check >> "${logfile}_todayitems.py"  2>&1

        today   = date_now_jp("%Y%m%d", add_days=add_days)
        tk      = get_timekey() + add_days

        if path == 'intra':
            nowhour = int( date_now_jp("%H", add_days=add_days) )
            log('intra hour', nowhour)

            nhour = nowhour
            dd    = set() ; ii =0
            while len(dd) < 50000 and ii < 10 :
                if nhour >= 12 and nhour < 14:   dd  = ca_today_items('12am',  add_days= 0 )
                elif nhour >= 14 and nhour < 17: dd  = ca_today_items('14am',  add_days= 0 )
                elif nhour >= 17 and nhour < 19: dd  = ca_today_items('16am',  add_days= 0 )
                elif nhour >= 19 and nhour < 21: dd  = ca_today_items('18am',  add_days= 0 )
                elif nhour >= 21 and nhour < 24: dd  = ca_today_items('20am',  add_days= 0 )

                ### T+1
                elif nhour >= 0 and nhour < 15:  dd  = ca_today_items('23am',  add_days= -1)

                if len(dd) > 50000 : break  ### Ok good

                nhour = nhour-1
                if nhour < 0 : nhour = 24
                ii = ii +1

            if len(dd) < 50000 :
                 log('using 12am list', nowhour)
                 if nowhour>= 14 and nowhour < 24 :
                    dd = ca_today_items('12am',  add_days= 0 )
                 else :
                    dd = ca_today_items('12am',  add_days= -1 )

            return dd



        if path == 'check':
            log('\nToday item list', today, tk)
            d3am  = ca_today_items('3am',   add_days= add_days)
            d10am = ca_today_items('10am', add_days= add_days)  ### 12am version
            # d12am = ca_today_items( '12am', add_days= add_days)  ### 12am version n
            d15am = ca_today_items( '15am', add_days= add_days)  ### 12am version n

            flist = glob_glob(dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/siid_ok/*.parquet", 1000)

            for fi in flist :
               siidok = pd_read_file(fi)
               siidok = set(siidok.siid.values)
               diff   = d15am.difference(siidok)
               log(fi.split("/")[-1], ", diff 15am vs siiok:",  len(diff), )


            log("\n15am: ",   len(d15am),  )
            ddiff  = d15am.difference(d3am)
            log("03am: ", len(d3am), ',15am Not in 3am:', len(ddiff)  )

            ddiff  = d15am.difference(d10am)
            log("10am: ", len(d10am), ',15am Not in 10am:', len(ddiff)  )
            return ''


        if path == 'latest':
            log('\nToday item list', today, tk)
            d3am  = ca_today_items('3am',  add_days= add_days)
            d10am = ca_today_items('10am', add_days= add_days)  ### 12am version
            d12am = ca_today_items('12am', add_days= add_days)  ### 12am version n

            log("\n12am",   len(d12am),  )
            ddiff  = d3am.difference(d12am)
            log("3am: ", len(d3am), 'Diff with 12am', len(ddiff)  )

            ddiff  = d10am.difference(d12am)
            log("10am: ", len(d10am), 'Diff with 12am', len(ddiff)  )

            if len(d12am) > 50000 :
                log('using 12am item list') ;   return d12am   ### 12am versio

            elif len(d10am)> 50000:  #### Using couch as Reference
                log('using 10am item list') ;   return d10am   ### 10am versio

            log('using 3am list')  ;   return d3am


        ##### Normal mode
        if path == 'eod' or path =='3am' :  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean03am/*.parquet"
        if path == '10am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean10am/*.parquet"
        if path == '12am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean12am/*.parquet"
        if path == '14am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean14am/*.parquet"
        if path == '16am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean16am/*.parquet"
        if path == '18am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean18am/*.parquet"
        if path == '20am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean20am/*.parquet"
        if path == '23am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean23am/*.parquet"

        ### couchbase dump
        if path == '15am':  dirin = dir_ca + f"/daily/item/ca_items2_{today}/clean15am/*.parquet"

        try :
            flist = sorted( glob_glob(dirin, 1000) )
            log(flist[-1])
            df = pd_read_file( flist[-1] )
            log(df.shape)
            df['dum'] = True
            df = df[[ colkey, colval ]]
            df = df.drop_duplicates(colkey)
            dd = df.set_index(colkey).to_dict('dict')
            dd = dd[colval]
            log('N elements',  len(dd))
            dd = set(dd.keys() )
            return dd
        except Exception as e:
            log(e)
            return set()


    def pd_check_topk(dirin="eod", add_days=0):    #####  python prepro_prod.py  pd_check_topk  --dirin eod  --add_days 0   >> "${logfile}_check_topk3.py"  2>&1   &
        tk     = date_now_jp("%Y%m%d", add_days= add_days)
        ti     = get_timekey() + add_days
        dirin2 = dirin
        if dirin == 'eod'    :        dirin2 = dir_cpa3   + f"/hdfs/daily_usertopk/m001/{ti}/daily_user_eod/*.parquet"
        if dirin == 'intra'  :        dirin2 = dir_cpa3   + f"/hdfs/intraday/sc_stream_usertopk/{tk}/*.parquet"
        if dirin == 'export_eod' :    dirin2 = dir_export + f"/{tk}/eod/*.parquet"
        if dirin == 'export_intra' :  dirin2 = dir_export + f"/{tk}/intraday/*.parquet"

        # isok_dict = dict_load()
        dcheck = ca_today_items('latest', add_days= add_days)

        def siid_ok(ll):
            if len(ll)> 0 :
               ll = [  len(t.split("_"))  for t in ll ]
               return max(ll)
            return 0

        def getbad(x):
            return list( set(x).difference(dcheck))

        log(dirin2)
        flist = glob_glob(dirin2, nfile=1000)
        flist = sorted(flist)
        # flist = flist[-1:]

        nfile2 = len(flist)
        log("\n\n Nfiles", nfile2  , "\n" )
        if nfile2 < 500:
            log("\n\n\n\n Warning, N Files<500",  nfile2, "\n\n\n\n\n\n"  )
            log(flist)

        nmax  = 0
        for ii, fi in enumerate(flist):
           if ii > 1000 : break
           log("\n", fi)
           df          = pd_read_file2(fi, cols=['easy_id', 'topk'], n_pool=1)
           if 'export' in dirin :
              df['topk'] = df['topk'].apply(lambda x : json.loads(x)  )
           else :
              df['topk'] = df['topk'].apply(lambda x : x.split(",")  )


           df['nk']   = df['topk'].apply(lambda x :  len(x))
           df['bad']  = df['topk'].apply(lambda x :  getbad(x))
           df['nbad'] = df['bad'].apply(lambda x :   len(x))

           nbad     =  df['nbad'].mean()
           siidsize =  df['topk'].apply(lambda x : siid_ok(x)  ).max()
           log( "nbad mean",        nbad ,
                'nrec min, mean' ,  df['nk'].min() , df['nk'].mean() ,
                'siid_ok',          siidsize
              )

           if siidsize > 2:
              log(df)

           if nbad > 0.0 :
              log( df[['bad', 'nbad' ]] )


    def ca_shop_blocked_file(add_days=0):
        today  = date_now_jp("%Y%m%d", add_days= add_days )
        flist  = sorted( glob_glob( dir_ca + f"/daily/shop_blocked/{today}/*.parquet"  ))
        if len(flist) < 1 : return set()
        df     = pd_read_file(flist[0])
        bshops = set( df['shop_id'].values)

        bcustom = ['203677','261122','193677','212232','306273','204094','217176','193838','193620','211717']
        bcustom = [ int(t) for t in bcustom ]
        bshops =  set( bcustom + list(bshops) )

        log("blocked shops", len(bshops), str(bshops)[:60])
        return bshops


    def pd_topk_check(df2):
        ss = ""
        # ss += str(  len( df2[ df2['topk'].str.len() < 100 ] ) ) + ","
        ss += str(df2.topk.str.len().mean() )  + ","
        # ss += str(  df2['easy_id'].nunique() ) + ","
        return ss



if 'daily_afternoon':
    ###### Afternoon Batch  #####################################################
    def daily_eod_user_hist(ii=0, ii2=500, tag='all', add_days=0):   ### python prepro_prod.py  daily_eod_user_hist --add_days -1   >> "${logfile}_eod_user.py"  2>&1
        """    python prepro_prod.py  daily_eod_user_hist    --ii 0    &
               ### Other 15 mio users
        """
        # today0  = date_now_jp("%Y%m%d", add_days=-1,  timezone= 'jp')
        t0      = get_timekey() - 1 + add_days
        tk_list = [ t for t in range(t0-2  , t0+1  , 1)  ]   ### Past 5 day ONLY
        tmax    = get_timekey() + add_days
        log(tk_list); time.sleep(5)

        if  'all' in tag : dirin_list = [ f"/a/acb401/ipsvols06/pydata/brw_ran_v15/",   ]
        elif 'sc' in tag : dirin_list = [ f"/a/acb401/ipsvols06/pydata/sc_widget_pur/"  ]
        else :             return 'no tag early finish'
        log('users', dirin_list) ; time.sleep(5)

        dirout = dir_cpa3 + f"/hdfs/daily_user_eod/{tmax}/"
        log("\n", dirout); time.sleep(3)

        #### Last purchase users 6 months
        tk_list2 = [ t for t in range(t0-5  , t0+1  , 1)  ]   ### Past 5 day ONLY


        cols = [  'easy_id', 'shop_id',  'item_id', 'genre_id'  ]
        log("############## Merge ######################################################")
        bkmin = ii ; bkmax = ii2 ; dfr =pd.DataFrame()
        for bk in range(bkmin, bkmax):
            flist  = []

            #### Pur only
            #dirin2 = f"/a/acb401/ipsvols06/pydata/pur_ran_v15/"
            #for tk in tk_list2:
            #    flist = flist +  glob.glob(dirin2 + f"/{bk}/*{tk}*"  )

            #### Brw only
            for dirin in dirin_list :
                for tk in tk_list:
                    flist = flist +  glob.glob(dirin + f"/{bk}/*{tk}*"  )

            if len(flist) < 1 : continue
            group   = "brw_ran_v15"   ### flist[-1].split("/")[-3]
            dirouti = dirout + f"/{group}/user_eod_{bk}_{tk}.parquet"
            # if os.path.isfile(dirouti) : continue

            log("\n\n", tk )
            #### prev daay
            df1 = pd.DataFrame() ; tj= tmax-1
            while len(df1) < 1000:
               dirin2 = dir_cpa3 + f"/hdfs/daily_user_eod/{tj}/{group}/*_{bk}_*.parquet"
               df1    = pd_read_file2( dirin2 , nfile=1)
               log('Prev day, T-2 ', df1.shape,  dirin2 )
               tj = tj-1

            log("T-1 day", len(flist) )
            df2 = pd.DataFrame()
            for fi in flist:
               dfi = pd_read_file2(fi, cols=cols, n_pool=1,  )      ### Need to preserve order
               df2 = pd.concat(( df2, dfi ))
            log(df2.shape)
            #df2  = df2.drop_duplicates([ 'easy_id', 'shop_id', 'item_id' ])


            df = pd_add_easyid_bk(df2)
            dfr       = pd.concat(( dfr,  df2[df2.bk != bk ] ))
            df2       = df2[df2.bk == bk ]

            if 'genre_id' not in df1.columns :  df1['genre_id'] = 0
            df2  = pd.concat((df1, dfr[dfr.bk == bk ][cols], df2[cols],  ))
            df2  = df2.drop_duplicates( ['easy_id', 'genre_id'] , keep='last')
            df2  = df2.groupby('easy_id' ).tail(5)  #### Last 5 events

            pd_to_file(df2, dirouti , show=1 )


    def daily_eod_item(mode='pur', tmin=-10, tmax=-1):   ### sleep 600 && python prepro_prod.py  daily_eod_item  &   2>&1 | tee -a zlog_daily5b.py
        """   sleep 600 &&  python prepro_prod.py daily_eod_item --mode pur > zlog_daily6a.py  2>&1   &
                    python prepro_prod.py daily_eod_item --mode pur >  "${logfile}_eod_item_pur.py"  2>&1  &

                    python prepro_prod.py daily_eod_item --mode "pur,custom"  --tmin -175  --tmax -30   >  "${logfile}_eod_item_pur_custom.py"  2>&1  &
        """
        # today2  = date_now_jp("%Y%m%d", add_days=0, add_hours=0, timezone= 'jp')
        # now     = date_now_jp("%Y-%m-%d--%H%M", add_days=0, add_hours=0,  timezone= 'jp')
        tk      = get_timekey() - 7

        if 'pur' in mode :    tk_list = [ t for t in range(get_timekey()-1, get_timekey() - 10 , -1)  ]
        if 'brw' in mode :    tk_list = [ t for t in range(get_timekey()-1, get_timekey() - 5  , -1)  ]
        if 'custom' in mode : tk_list = [ t for t in range(get_timekey()+tmin, get_timekey() + tmax,  1)  ]


        cols   = [  'shop_id',  'item_id',  ]

        model  = text_model_load(model0['dir'] )
        dirout = dir_cpa3 + f"/hdfs/daily_brw_pur/{model0['pref']}/"    ## Related to model
        index = Index1( dirout + "/done.txt"  )


        for tk in tk_list :
            dirin_list = []
            if 'pur' in mode : dirin_list.append( f"/a/acb401/ipsvols06/pydata/pur_ran_v15/*/*{tk}*.parquet"  )
            if 'brw' in mode : dirin_list.append( f"/a/acb401/ipsvols06/pydata/brw_ran_v15/*/*{tk}*.parquet"  )
            # f"/a/acb401/ipsvols06/pydata/sc_widget_clk/*/*{tk}*.parquet"

            for dirin in dirin_list :
                flist = glob_glob(dirin , 5000 )
                flist = [  t for t in flist if t not in set(index.read()) ]
                log('files', len(flist), str(flist)[:10] )
                if len(flist) < 1: continue

                jj = 0; df2 = pd.DataFrame()
                for fi in flist :
                    fil = index.save_filter( [ fi ]  )  ### remove already procssed
                    if len(fil) < 1 : continue
                    fi = fil[0]
                    jj    = jj + 1
                    df2i  = pd_read_file(fi, cols=cols)
                    df2i  = df2i.drop_duplicates( cols )
                    df2   = pd.concat((df2, df2i))
                    # log(df2.shape)
                    df2  = df2.drop_duplicates( cols )
                    # df2 = df2.iloc[:100, :]
                    if len(df2) < 2000000 and jj < len(flist) : continue
                    log('N siids', fi, df2.shape)
                    # df2 = df2[df2.groupby(['shop_id','item_id'])['dum'].transform('count') > 5 ]  ## > 5 clicks

                    df2  = pd_cass_remove_exist(df2, prefix= model0['pref'])
                    df2  = pd_add_itemaster(df2)          ### Too much ressources on Cass

                    df2['item_text'] = df2.apply(lambda x : item_merge_field(x), axis=1)
                    df2['item_emb']  = text_tovec_batch(model, df2['item_text'].values )
                    df2 = df2[[ 'siid', 'item_emb' ]]
                    log(df2)
                    cass_update(df2, table=model0['table'], prefix= model0['pref'], colkey="siid", colval="item_emb")
                    df2 = pd.DataFrame() ; jj =0


    def daily_create_easyid_topg2(ii=0, ii2=500, tag='all'):     ## python prepro_prod.py  daily_create_easyid_topg2  --tag brw   2>&1 | tee -a zlog_genre.py
        """
        """
        t0   = get_timekey() - 1
        tmax = get_timekey()

        if  'brw' in tag :
            dirin   = f"/a/acb401/ipsvols06/pydata/brw_ran_v15/"
            dbname  = 'db_easyid_topgenre_brw'
            tk_list = [ t for t in range(t0-60  , t0+1  , 1)  ]   ### Past 5 day ONLY

        elif 'pur' in tag :
            dirin   = f"/a/acb401/ipsvols06/pydata/pur_ran_v15/"
            dbname  = 'db_easyid_topgenre_pur'
            tk_list = [ t for t in range(t0-120  , t0+1  , 1)  ]   ### Past 5 day ONLY

        else :   return 'no tag early finish'
        log('users', dirin)
        log(tk_list)
        dirout = dir_cpa3 + f"/hdfs/daily_user_eod/{tmax}/"
        log("\n", dirout); time.sleep(3)


        cols  = [  'easy_id',  'genre_id'   ]
        log("############## Merge ##################################################")
        nn    = 0
        bkmin = ii ; bkmax = ii2 ; flist = [] ; jj =0
        for bk in range(bkmin, bkmax):
            jj = jj + 1
            for tk in tk_list:
                flist = flist + glob.glob(dirin  + f"/{bk}/*{tk}*" )

            if jj < 10 and bk < bkmax -1: continue

            log(bk,  len(flist))
            if len(flist) < 1 : continue

            df = pd_read_file2(flist, cols=cols, n_pool=20)
            log(df.shape)
            df['dum']  = 1
            df         = df.groupby(['easy_id', 'genre_id'] ).agg({'dum' : 'count'}).reset_index()
            df.columns = ['easy_id', 'genre_id', 'cnt' ]
            df         = df.sort_values(['easy_id', 'cnt'], ascending=[1,0])

            df['genre_id'] = df['genre_id'].astype('str')
            # df1          = df.groupby('easy_id').apply( lambda x : ",".join( x['genre'] )  + ";" + ",".join( [  str(t) for t in  x['cnt']])   ).reset_index()
            df         = df.groupby('easy_id').apply( lambda x : ",".join(  x['genre_id'][:6] )  + ";" ).reset_index()
            df.columns = [ 'easy_id', 'top_genre']
            log(df.shape, df)
            diskcache_save2(df, db_path  = db[dbname] , colkey='easy_id',  colvalue = 'top_genre',  npool=2, verbose=False )
            flist = [] ; jj = 0


    def intra_user_hist():  ### python prepro_prod.py  intra_user_hist
        ### T-2 End of day +  T-1  Intraday  --->  T-1 Full easyid click  for Rec Compute
        today1 = date_now_jp("%Y%m%d", add_days=0,  timezone= 'jp')
        tk     = get_timekey()
        dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*"
        dirout = dir_cpa3 + f"/hdfs/intraday/user_histo/{today1}"
        log(dirin,  dirout)

        log('####### Load Intraday', today1)
        cols   = ['easy_id',  'item_id',  'shop_id', 'genre_id', 'ts', ]
        df     = pd_read_file( dirin , cols=cols, nfile=200, n_pool=20, nrows=9900010000)
        log(df)
        df['siid'] = df.apply(lambda x : siid(x), axis=1)
        df = df.sort_values(['easy_id', 'ts' ])

        df1               = df.groupby('easy_id').apply(lambda x :  ",".join( x['siid']) )
        #df1['hist_genre'] = df.groupby('easy_id').apply(lambda x :  ",".join([ str(t) for t in x['genre_id'] ]) )
        df1 = df1.reset_index()
        # log(df1)
        df1.columns       = [ 'easy_id', 'hist' ]
        #log(df1)

        pd_to_file(df1, dirout + f"/userhist_intra_{today1}_{tk}_{int(time.time())}.parquet", show=1)


    def pd_groupby_pur(dirin, tk=None, use_cache=True):
        ###  Intraday easy_id  genre_id   item_id  price  shop_id          ts unit
        ###  df = pd_groupby_pur(dirin)
        dir1 = dir_cpa3   + f"/hdfs/items_pur/itempur_{tk}.parquet"
        if use_cache :
           df   = pd_read_file(  dir1 )
           if len(df) > 1 : return df

        if isinstance(dirin, str):
           flist = glob_glob(dirin , 1000)
           df    = pd_read_file2(flist,  n_pool=25)

        if isinstance(dirin, pd.DataFrame): df  = dirin

        df = df.rename(columns={'unit':'units'})
        for ci in ['price', 'units',] :
          df[ci] = df[ci].astype('float32')
        df = df[ df['units'] > 0.0 ]

        for ci in ['genre_id', 'price', 'units', 'shop_id', 'item_id'] :
          df[ci] = df[ci].astype('int64')

        df['siid'] = df.apply(lambda x : siid(x) , axis=1 )
        df['gms']  = df['units'] * df['price']
        df         = df.groupby(['siid', 'genre_id']).agg({ 'item_id' : 'count', 'gms': 'sum', 'price': 'mean',  }).reset_index()
        df.columns = [ 'siid', 'genre_id', 'n_pur', 'gms', 'price'  ]
        for ci in ['genre_id', 'n_pur', 'gms', 'price'] :
          df[ci] = df[ci].astype('int64')

        df = df.sort_values('n_pur', ascending=0)

        if use_cache: pd_to_file(df, dir1, show=0 )
        return df


    def pd_groupby_clk(dirin, tk=None, use_cache=True):
        ###  Intraday easy_id  genre_id   item_id  price  shop_id          ts unit
        ###  df = pd_groupby_pur(dirin)
        if use_cache :
           dir1 = dir_cpa3   + f"/hdfs/items_brw/itembrw_{tk}.parquet"
           df   = pd_read_file(  dir1 )
           if len(df) > 1 : return df

        if isinstance(dirin, str):
           flist = glob_glob(dirin , 1000)
           df    = pd_read_file2(flist, cols=['shop_id', 'item_id', 'genre_id'],  n_pool=20)

        if isinstance(dirin, pd.DataFrame): df  = dirin

        for ci in ['genre_id', 'shop_id', 'item_id'] :
           df[ci] = df[ci].astype('int64')

        df['siid'] = df.apply(lambda x : siid(x) , axis=1 )
        df         = df.groupby(['siid', 'genre_id']).agg({ 'item_id' : 'count',   }).reset_index()
        df.columns = [ 'siid', 'genre_id', 'n_clk',  ]
        for ci in ['genre_id', 'n_clk', ] :
          df[ci] = df[ci].astype('int64')
        df = df.sort_values('n_clk', ascending=0)

        if use_cache: pd_to_file(df, dir1, show=0 )
        return df


    def daily_item_info(mode='create'):  ####   python prepro_prod.py  daily_item_info --mode brw  &
        #### To match series with CA item list
        ##  python prepro_prod.py  daily_item_info --mode merge  &
        t0     =  get_timekey()-1
        tk     =  get_timekey()-1
        tlist  =  [ t0-i for i in range(0, 7) ]
        cols   =  ['shop_id', 'item_id',  'genre_id',  'ran_id', 'ref', 'ref_type', 'series_id', 'sg_id', ]

        if mode == 'merge':
            dirout =  dir_cpa3   + "/hdfs/items/"
            dirin  =  dir_pydata + "/brw_ran_v15/"
            df    = pd.DataFrame()
            flist = sorted( glob_glob(dirout + f"/*18*.parquet") )
            for fi in flist[-1:]:
               dfi = pd_read_file2(fi, cols=cols,  n_pool=1)
               df  = pd.concat((df, dfi))
               df  = df.drop_duplicates(cols)
            # n = len(df)
            pd_to_file(df, dirout + f"/item_merge_latest.parquet", show=1)
            return 1

        if mode == 'daily':
            dirout =  dir_cpa3    + "/hdfs/items/"
            dirin  =  dir_pydata  + f"/brw_ran_v15/"
            for tk in tlist :
                dirouti = dirout + f'/item_{tk}.parquet'
                if os.path.isfile(dirouti) : continue
                flist   = glob_glob(dirin + f"/*/*{tk}*.parquet")
                df = pd_read_file2(flist, cols=cols, drop_duplicates=cols, n_pool=20)
                df = df.drop_duplicates(cols)
                pd_to_file(df, dirouti, show=1)


        if mode == 'pur':
            dirout =  dir_cpa3   + "/hdfs/items_pur/"
            dirin  =  dir_pydata + f"/pur_ran_v15/*/*{tk}*.parquet"
            df  = pd_groupby_pur(dirin, use_cache=False)
            pd_to_file(df, dirout + f'/itempur_{tk}.parquet', show=1)


        if mode == 'brw':
            dirout =  dir_cpa3   + "/hdfs/items_brw/"
            dirin  =  dir_pydata + f"/brw_ran_v15/*/*{tk}*.parquet"
            df  = pd_groupby_clk(dirin, use_cache=False)
            pd_to_file(df, dirout + f'/itembrw_{tk}.parquet', show=1)


    def daily_eod_user_histfull(bmin=0, bmax=500, tag='brw', add_days=0, past= -1, mode='', split=0):   ###
        """ Past cathcup, 2hours to finish
           py  daily_eod_user_histfull   --split 15 --past -200  --tag pur  --add_days 0    --mode full    >> "${logfile}_user_histofull_pur2.py"  2>&1   &

           py  daily_eod_user_histfull   --split 10 --past -60  --tag brw  --add_days 0    --mode full    >> "${logfile}_user_histofull_brw2.py"  2>&1   & 

        """
        # today0  = date_now_jp("%Y%m%d", add_days=-1,  timezone= 'jp')
        t0      = get_timekey() - 1 + add_days  ### at T-1 EOD
        tk_list = [ t for t in range(t0+past  , t0+1  , 1)  ]   ### Past 5 day ONLY
        tmax    = max(tk_list)   ### Last day,
        log(tk_list)

        if split > 0 :
            cmd = f"{PYFILE}  daily_eod_user_histfull --split 0  --tag {tag}   --add_days {add_days} --past {past}  --mode {mode} "
            batch_split_run(cmd, split=split, sleep= random.randint(2, 10) ) ; return 'done'
        
        if  'brw' in tag :
            dirin_list = [ f"/a/acb401/ipsvols06/pydata/brw_ran_v15/",   ]
            group      = "brw_ran" + mode

        elif 'pur' in tag:
            dirin_list = [ f"/a/acb401/ipsvols06/pydata/pur_ran_v15/",   ]
            group      = "pur_ran" + mode

        else :             return 'no tag early finish'

        log('users', dirin_list)
        dirout = dir_cpa3 + f"/hdfs/daily_user_hist/{tmax}/"
        log("\n", dirout)

        cols = [  'easy_id', 'shop_id',  'item_id', 'genre_id'   ]
        log("############## Merge ######################################################")
        for bk in range(bmin, bmax):
            log("\n\n", bk ) 
            dirouti = dirout + f"/{group}/user_{tag}_{bk}_{tmax}"
            # if os.path.isfile(dirouti) : continue

            log("########  T-2 day" )
            df1 = pd.DataFrame() ; tj= tmax-1 ; jj = 0
            if mode == '' :
                while len(df1) < 1000:
                   if jj > 5 : break
                   dirin2 = dir_cpa3 + f"/hdfs/daily_user_hist/{tj}/{group}/*_{bk}_*.parquet"
                   df1    = pd_read_file( dirin2 , nfile=1)
                   log(df1.shape,  dirin2 )
                   tj = tj-1 ; jj = jj + 1


            log("########  T-1 day" )
            flist  = []
            for dirin in dirin_list :
                for tk in tk_list:
                    flist = flist +  glob.glob(dirin + f"/{bk}/*{tk}*"  )
            log('Nfiles', len(flist))

            for fi in flist :
                df2  = pd_read_file(fi, cols=cols, n_pool=1, )      ###
                df2  = df2.drop_duplicates([ 'easy_id', 'genre_id',  ], keep='last')            
                df1  = pd.concat((df1, df2))  ; del df2
            df1  = df1.drop_duplicates([ 'easy_id', 'genre_id',  ], keep='last')

            df1  = df1.groupby('easy_id' ).tail(10)  #### Last 10 <> clicks or purchases
            df1['dum'] = np.arange(0, len(df1))
            df1        = df1.sort_values(['easy_id', 'dum'])
            del df1['dum']
            
            neasy = len(df1.easy_id.unique() )
            log('Neasyid',  neasy )            
            df1 = pd_add_easyid_bk(df1) ;  nbk = df1.bk.unique() ;  log('BK', nbk ) ; del df1['bk']

            pd_to_file(df1, dirouti  + f"-{len(df1)}-{neasy}.parquet", show=1 )

            

    def daily_eod_reorder(ii=0, ii2=500, tag='intra', add_days=0):   ###  python prepro_prod.py  daily_eod_reorder  >> "${logfile}_clean_pur.py"  2>&1   &
        """  Re-ordered into correct easyid Bucket 0, 1,2,3
           python prepro_prod.py  daily_eod_clean  --tag brw  >> "${logfile}_userhist_clean_brw2.py"  2>&1   &

        """
        today1 = date_now_jp("%Y%m%d", add_days=-1,  timezone= 'jp')
        t0     = get_timekey() - 1 + add_days
        log(today1, t0)

        #tag     = 'pur'
        #dirin   = dir_cpa3 + f"/hdfs/daily_user_hist/18971/{tag}_ran/*.parquet"
        #dirout0 = dir_cpa3 + f"/hdfs/daily_user_hist/ok2_{tag}/"

        t0 = "18977"
        dirin   = dir_cpa3 + "/hdfs/daily_user_eod/18978/brw_ran_v15/*.parquet"
        dirout0 = dir_cpa3 + "/hdfs/daily_user_eod/18978/brw_ran_v15_v2/"


        #### Load ALL and re-distribute
        # cols  = ['easy_id', 'shop_id', 'item_id', 'genre_id']
        flist = glob_glob(dirin, 3000 )
        log(len(flist))
        df   = pd_read_file2( flist, n_pool=25, cols=None,   drop_duplicates=None)
        cols = df.columns
        log(df.shape)
        #df['bk']   = df['easy_id'] % 500
        df = pd_add_easyid_bk(df)  ## bk
        df['rank'] = np.arange(0, len(df))

        for bi in df['bk'].unique() :
            #dirout = dirout0 + f"/user_{tag}_{bi}_{t0}.parquet"
            dirout = dirout0 + f"/user_eod_{bi}_{t0}.parquet"
            dfi = df[ df.bk == bi]
            dfi = dfi.sort_values(['easy_id', 'rank'] )
            dfi = dfi[cols]
            pd_to_file(dfi, dirout , show=1 )


    def daily_eod_user_list(ii=0, ii2=500, tag='intra', add_days=0):   ###
        """ 5 days Past Catchup
           python prepro_prod.py  daily_eod_user_list    >> "${logfile}_user_eod_list.py"  2>&1   &

        """
        today1 = date_now_jp("%Y%m%d", add_days=-1,  timezone= 'jp')
        t0     = get_timekey() - 1 + add_days
        log(today1, t0)

        if tag == 'intra':
           dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*.parquet"
           df    = pd_read_file( dirin , n_pool=20, cols=['easy_id']  , drop_duplicates=['easy_id'] )
        else :
            df  = pd_read_file2(  brw_dir + f"/*/*{t0}*", n_pool=20, cols=['easy_id']  , drop_duplicates=['easy_id'] )
            df2 = pd_read_file2(  pur_dir + f"/*/*{t0}*", n_pool=20, cols=['easy_id']  , drop_duplicates=['easy_id'] )
            df  = pd.concat((df, df2)) ; del df2

        log(df.shape)
        df  = df.drop_duplicates('easy_id')
        #df['bk'] = df['easy_id'] % 500
        df = pd_add_easyid_bk(df)
        df = df.sort_values('bk')

        dirout = dir_cpa3 + f"/hdfs/daily_user_list/user_list_{today1}_{t0}_{len(df)}.parquet"
        pd_to_file(df, dirout , show=1 )


    def pd_load_hist_intra(df0=None, add_days=0, tag='intra,clk', bmin=0, bmax=500):
        """
        """
        today1 = date_now_jp("%Y%m%d", add_days=add_days,  timezone= 'jp')
        t0     = get_timekey() + add_days
        log(today1, t0)

        if 'clk' in tag and 'intra' in tag:    dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
        elif 'pur' in tag and 'intra' in tag:  dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*purchase*.parquet"
        elif 'clk' in tag :  dirin = brw_dir + f"/*/*{t0}*.parquet"
        elif 'pur' in tag :  dirin = pur_dir + f"/*/*{t0}*.parquet"

        df = pd_read_file( dirin , n_pool=20, cols=None , ) ;log(df.shape)

        # df['bk'] = df['easy_id'] % 500
        df = pd_add_easyid_bk(df)
        df       = df[ (df.bk >= bmin) & (df.bk < bmax) ]
        if 'ts' in df.columns : df = df.sort_values('ts')

        if isinstance(df0, pd.DataFrame)  :
            cols = [ c for c in df.columns if c in df0.columns ]
            df   = pd.concat((df0, df[cols]))
        return df



if 'daily_rec_topk':
    if 'model' :
        model0 = Box({})
        # model0.dir   =  dir_ca + "/models/static/v_genre3_1m/"
        model0.dir   =  dir_ca + "/models/static/v_genre3_price_4m/"
        model0.pref  =  "m001"
        model0.table =  'item_model'

    def daily_eod_user_hist_intra(add_days=-1):  ### python prepro_prod.py  daily_eod_user_hist_intra
        ### T-2 End of day +  T-1  Intraday  --->  T-1 Full easyid click  for Rec Compute
        #### Most recent at bottom
        today1 = date_now_jp("%Y%m%d", add_days=add_days,  timezone= 'jp')
        tk     = get_timekey()+add_days
        dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*.parquet"
        dirin2 = dir_cpa3 + f"/hdfs/daily_user_eod/{tk}/brw_ran_v15/*.parquet"
        dirout = dir_cpa3 + f"/hdfs/daily_user_eod/{tk}/sc_stream/"
        log(dirin, dirin2, dirout)

        log('\n####### Load past data T-2')
        log(dirin2)
        df2 = pd.DataFrame();  ti = tk
        while len(df2) < 1 :
           dirin2 = dir_cpa3 + f"/hdfs/daily_user_eod/{ti}/brw_ran_v15/*.parquet"
           # cols2  = [ 'easy_id', 'shop_id', 'item_id', 'genre_id'   ]
           df2    = pd_read_file2(dirin2, n_pool=25)  ; log(df2)
           if len(df2) > 10000 : break
           to_file('T-2 is empty:' + dirin2 ,   flog_warning)
           ti     = ti-1

        log('\n ####### Load Intraday T-1')
        log(dirin)
        cols   = ['easy_id',  'item_id',  'shop_id', 'genre_id', 'ts', ]
        df     = pd_read_file2( dirin , cols=cols, nfile=1000, n_pool=25, drop_duplicates=['easy_id', 'genre_id'])
        df = df.sort_values(['easy_id', 'ts' ])
        # df2  = df2.drop_duplicates([ 'easy_id', 'shop_id', 'item_id' ])
        # df2  = df2.groupby('easy_id' ).tail(3)  #### Last 3 clicks
        df = df.drop_duplicates( ['easy_id', 'genre_id'] , keep='last')
        log(df)


        log('\n ####### Merge Full size into  T')
        log(dirout)
        df   = pd.concat((df2, df )) ; del df2
        df   = df.drop_duplicates( ['easy_id', 'genre_id'] , keep='last')
        log(df)
        # df  = df.drop_duplicates([ 'easy_id', 'shop_id', 'item_id' ])
        # df  = df.groupby('easy_id' ).tail(3)  #### Last 3 clicks

        log('######## Save on disk')
        #df['siid']     = df.apply(lambda x : siid(x), axis=1)
        #df['siid_emb'] = ""    ### Fetch embed : Save time during compute

        #df['bk'] = df['easy_id'] % 500
        df = pd_add_easyid_bk(df)  #### bk % 500

        for bk in range(0, 500):
           dfk = df[df.bk== bk ]
           dfk = dfk.groupby('easy_id' ).tail(5)
           nk  = len(dfk)
           dirouti =  dirout + f"/user_eod_intra_{bk}_{tk}_{today1}_{nk}.parquet"
           # if len(glob.glob(dirouti[:-25] + "*" )) > 0 : continue
           pd_to_file( dfk,  dirouti, show=0)
           # log( nk )
        log(df)


    def daily_eod_user_sameday(add_days=0):     ### python prepro_prod.py  daily_eod_user_sameday   >> "${logfile}_eod_sameday.py"  2>&1  &
        ### T-1 End of day +  T  Intraday  --->  T Full easyid click  for Rec Compute
        ### Most revent
        today = date_now_jp("%Y%m%d", add_days=0 + add_days,  timezone= 'jp')
        tk     = get_timekey()-1 + add_days

        dirint  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*.parquet"
        dirin   = dir_cpa3 + f"/hdfs/daily_user_eod/{tk}/sc_stream/"

        dirout  = dir_cpa3 + f"/hdfs/daily_user_eod/{tk}/sc_stream_intra/"
        log(dirin, dirint, dirout)


        log('\n ####### Load Intraday T')
        cols   = ['easy_id',  'item_id',  'shop_id', 'genre_id', 'ts', ]
        df     = pd_read_file2( dirint , cols=cols, nfile=1000, n_pool=20, drop_duplicates=['easy_id', 'genre_id'])
        df     = df.sort_values(['easy_id', 'ts' ])
        # df['bk'] = df['easy_id'] % 500
        df     = pd_add_easyid_bk(df)
        df     = df.drop_duplicates( ['easy_id', 'genre_id'] , keep='last')
        log(df)

        cols1 = ['easy_id',  'item_id',  'shop_id', 'genre_id',  ]
        log('\n ####### Merge Full size into  T')
        for bk in range(0, 500):
           df2 = pd_read_file(dirin + f"*_{bk}_*.parquet", n_pool=1)

           dfk = pd.concat((df2, df[df.bk== bk ][cols1] ))   ### Most Recent at bottom
           dfk = dfk.groupby('easy_id' ).tail(5)
           nk  = len(dfk)
           dirouti =  dirout + f"/user_eod_intra_{bk}_{tk}_{today}_{nk}.parquet"
           pd_to_file( dfk,  dirouti, show=0)


    def daily_create_easyid_topg(mode='pur', add_days=0):   ###   python prepro_prod.py  daily_create_easyid_topg  --mode intra   2>&1 | tee -a zlog_brw.py   &
        """    easyid --> list of top genr Once a week is ok -->  distribution is stable
               86 mio  rows,  15mio easyid for 1 month brwsing,  132 mio rows, 22 mio easyid for 6month purchase

               T-2 Hive + T-1 Rate --->
        """
        # off = 0
        log(" daily_create_easyid_topg  ")
        today  = date_now_jp("%Y%m%d", add_days= 0 + add_days)
        today1 = date_now_jp("%Y%m%d", add_days=-1 + add_days)
        today2 = date_now_jp("%Y%m%d", add_days=-2 + add_days)
        tk     = get_timekey()+add_days

        dirout = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/siid_genre/"
        ttl    = None; cols = None ; ww = {}

        if   'pur'   in mode  :
            dbname = 'db_easyid_topgenre_pur'
            dirin  = [dir_cpa3 + f"/input/ca_user_genre_pur_{today2}/*",
                      dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*pur*.parquet"  ]

        elif 'brw'   in mode :
            dbname = 'db_easyid_topgenre_brw'
            dirin  = [dir_cpa3 + f"/input/ca_user_genre_brw_{today2}/*",
                      dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet", ]

        elif 'intra' in mode :
            dbname = 'db_easyid_topgenre_intra'
            dirin  = [dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*"  ]
            ttl    = 86400 * 4

        elif 'merge' in mode :
            # dbname = 'db_easyid_topgenre_merge'
            dbname = 'db_easyid_topgenre_brw'
            dirin  = [ dir_cpa3 + f"/input/ca_user_genre_brw_{today2}/*" ,     ### easy_id', 'genre_id', 'genre_path', 'n_clk'
                       dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet",   ### browsing in T-1

                       dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*pur*.parquet",        ### Pur  in T-1
                       dir_cpa3 + f"/input/ca_user_genre_pur_{today2}/*",   ### in T-2
                       ### dir_cpa3 + f"/input/ca_user_genre_pur_{20201228}/*",   ### last year, Too big with Super Sales

                     ] ### browsing in T
            ww     = { 0:1, 1:1, 2: 0.5, 3: 0.5, 4: 0.5 }
            # ttl   = 86400 * 10

        df0 = pd.DataFrame()
        for j, fj in enumerate(dirin) :
            log("\n",j, fj)
            df = pd_read_file2(fj, cols = None, n_pool=20, nfile=50000  )
            # df = df.iloc[:1000, :]
            log(df)
            df             = df[-df['genre_id'].isna()]
            df['genre_id'] = df['genre_id'].astype('int32')


            if 'n_pur'  in df.columns or 'n_clk' in df.columns :   ### Hive pre compute
                df = df.rename(columns= {'n_pur': 'cnt',  'n_clk': 'cnt'})  ### genre_id   -->   genre_path xxx
            else :
                log('groupby count dummy')
                df['dum']  = 1
                df         = df.groupby(['easy_id', 'genre_id']).agg({'dum' : 'count'}).reset_index()
                df.columns = ['easy_id', 'genre_id', 'cnt' ]

            df        = df[[ 'easy_id', 'genre_id', 'cnt'  ]]
            df['cnt'] = df['cnt'].astype('int32')
            df['cnt'] = ww.get(j, 1) * df['cnt']
            log(df)

            log("#### concat with prev   ")
            df0 = pd.concat((df0, df))  ; del df; gc.collect()
            df0 = df0.groupby(['easy_id', 'genre_id']).agg({'cnt' : 'sum'}).reset_index()
            df0.columns = ['easy_id', 'genre_id', 'cnt' ]
            log('dfall', df0,)

        log("#### Export #################################################################")
        df0 = df0.sort_values(['easy_id', 'cnt'], ascending=[1,0])
        pd_to_file(df0[ df0.easy_id > 1 ].iloc[:5000,:],   dirout + f"/check_siid_genre_{tk}_{mode}_{int(time.time())}.csv", index=False )

        df0['genre_id'] = df0['genre_id'].astype('str')
        # df1          = df.groupby('easy_id').apply( lambda x : ",".join( x['genre'] )  + ";" + ",".join( [  str(t) for t in  x['cnt']])   ).reset_index()
        df0         = df0.groupby('easy_id').apply( lambda x : ",".join(  x['genre_id'][:8] )  + ";" ).reset_index()
        df0.columns = [ 'easy_id', 'top_genre']
        log(df0 )
        diskcache_save2(df0, db_path  = db[dbname] , colkey='easy_id',  colvalue = 'top_genre',   ttl=None,  npool=8, verbose=False )
        log('finished')


    def daily_eod_user_pur_intra(add_days=0) :  ### python prepro_prod.py    daily_eod_user_pur_intra
        ### T-2 End of day +  T-1  Intraday  --->  T-1 Full easyid click  for Rec Compute
        ### siid  genre_id  n_pur       gms  price
        ###  siid  genre_id   n_pur         gms    price      gms_t1  n_pur_t1      gms_t2  n_pur_t2
        today0 = date_now_jp("%Y%m%d", add_days=  0 + add_days,  timezone= 'jp')
        today1 = date_now_jp("%Y%m%d", add_days= -1 + add_days,  timezone= 'jp')
        today2 = date_now_jp("%Y%m%d", add_days= -2 + add_days,  timezone= 'jp')
        tk     = get_timekey() + add_days
        tk1    = get_timekey() -1 + add_days
        tk2    = get_timekey() -2 + add_days

        path = "pdef"
        hour = int( date_now_jp("%H", add_days= add_days,  timezone= 'jp'))
        if hour <  7    : path = "p3am"
        elif hour <  12 : path = "p10am"
        elif hour <  15 : path = "p12am"
        elif hour <  24 : path = "p15am"

        dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today0}/*purchase*0.parquet"
        dirout = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today0}/score/"
        log(dirin,  dirout)

        log('\n####### Load Intraday T')
        df2   = pd_groupby_pur(dirin, tk, use_cache=False) ; log(df2.shape)


        log('\n####### Load Intraday T-1')
        dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*purchase*0.parquet"
        df    = pd_groupby_pur(dirin, tk1)
        df2   = df2.merge(df[[ 'siid', 'gms', 'n_pur'  ]], on='siid', how='outer', suffixes=(None, "_t1")) ; log(df2.shape)


        log('\n####### Load Intraday T-2')
        dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today2}/*purchase*0.parquet"
        df    = pd_groupby_pur( dirin, tk2 )
        df2   = df2.merge(df[[ 'siid', 'gms', 'n_pur'  ]], on='siid', how='outer', suffixes=(None, "_t2")) ; log(df2.shape)


        df2   = df2.fillna(0.0)
        for ci in ['genre_id']:
            df2[ci] = df2[ci].astype('int64')

        df2['score'] = np.log( 1 + 3*df2['n_pur']) +  0.7*np.log( 1+df2['n_pur_t1']) + 0.4*np.log( 1+df2['n_pur_t2'])

        pd_to_file( df2,  dirout + f"/{path}/pur_intra_{tk}.parquet", show=1)
        pd_to_file( df2,  dirout + f"/pur_intra_{tk}.parquet", show=0)


    def daily_eod_user_brw_intra(add_days=0) :  ### python prepro_prod.py  daily_eod_user_brw_intra    >> "${logfile}_faiss3.py"  2>&1   &&
        ### T-2 End of day +  T-1  Intraday  --->  T-1 Full easyid click  for Rec Compute
        ### siid  genre_id  n_pur       gms  price
        ###  siid  genre_id   n_pur         gms    price      gms_t1  n_pur_t1      gms_t2  n_pur_t2
        today0 = date_now_jp("%Y%m%d", add_days=  0 + add_days,  timezone= 'jp')
        today1 = date_now_jp("%Y%m%d", add_days= -1 + add_days,  timezone= 'jp')
        today2 = date_now_jp("%Y%m%d", add_days= -2 + add_days,  timezone= 'jp')
        tk     = get_timekey() + add_days
        tk2    = get_timekey() -2 + add_days
        tk1    = get_timekey() -1 + add_days

        path = "pdef"
        hour = int( date_now_jp("%H", add_days= add_days,  timezone= 'jp'))
        if hour <  7    : path = "p3am"
        elif hour <  12 : path = "p10am"
        elif hour <  15 : path = "p12am"


        dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today0}/*browsing*0.parquet"
        dirout = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today0}/score/"
        log(dirin,  dirout)

        log('\n####### Load Intraday T')
        df2 = pd_groupby_clk(dirin, use_cache=False) ; log(df2.shape)


        log('\n####### Load Intraday T-2')
        df  = pd_groupby_clk( dir_cpa3 + f"/hdfs/intraday/sc_stream/{today2}/*browsing*0.parquet" , tk2 )
        df2 = df2.merge(df[[ 'siid', 'n_clk'  ]], on='siid', how='outer', suffixes=(None, "_t2")) ; log(df2.shape)


        log('\n####### Load Intraday T-1')
        df  = pd_groupby_clk( dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*0.parquet", tk1 )
        df2 = df2.merge(df[[ 'siid',  'n_clk'  ]], on='siid', how='outer', suffixes=(None, "_t1")) ; log(df2.shape)


        df2 = df2.fillna(0.0)
        for ci in ['genre_id']:
            df2[ci] = df2[ci].astype('int64')

        ### Scores
        df2['score_clk'] = np.log( 1 + 3*df2['n_clk']) +  0.7*np.log( 1+df2['n_clk_t1']) + 0.4*np.log( 1+df2['n_clk_t2'])

        pd_to_file( df2,  dirout + f"/{path}/clk_intra_{tk}.parquet", show=1)
        pd_to_file( df2,  dirout + f"/clk_intra_{tk}.parquet", show=1)
        ### T-2 End of day +  T-1  Intraday  --->  T-1 Full easyid click  for Rec Compute
        ### siid  genre_id  n_pur       gms  price
        ###  siid  genre_id   n_pur         gms    price      gms_t1  n_pur_t1      gms_t2  n_pur_t2
        today0 = date_now_jp("%Y%m%d", add_days=  0 + add_days,  timezone= 'jp')
        today1 = date_now_jp("%Y%m%d", add_days= -1 + add_days,  timezone= 'jp')
        today2 = date_now_jp("%Y%m%d", add_days= -2 + add_days,  timezone= 'jp')
        tk     = get_timekey() + add_days
        tk2    = get_timekey() -2 + add_days
        tk1    = get_timekey() -1 + add_days

        path = "pdef"
        hour = int( date_now_jp("%H", add_days= add_days,  timezone= 'jp'))
        if hour <  7    : path = "p3am"
        elif hour <  12 : path = "p10am"
        elif hour <  15 : path = "p12am"


        dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today0}/*browsing*0.parquet"
        dirout = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today0}/score/"
        log(dirin,  dirout)

        log('\n####### Load Intraday T')
        df2 = pd_groupby_clk(dirin, use_cache=False) ; log(df2.shape)


        log('\n####### Load Intraday T-2')
        df  = pd_groupby_clk( dir_cpa3 + f"/hdfs/intraday/sc_stream/{today2}/*browsing*0.parquet" , tk2 )
        df2 = df2.merge(df[[ 'siid', 'n_clk'  ]], on='siid', how='outer', suffixes=(None, "_t2")) ; log(df2.shape)


        log('\n####### Load Intraday T-1')
        df  = pd_groupby_clk( dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*0.parquet", tk1 )
        df2 = df2.merge(df[[ 'siid',  'n_clk'  ]], on='siid', how='outer', suffixes=(None, "_t1")) ; log(df2.shape)


        df2 = df2.fillna(0.0)
        for ci in ['genre_id']:
            df2[ci] = df2[ci].astype('int64')

        ### Scores
        df2['score_clk'] = np.log( 1 + 3*df2['n_clk']) +  0.7*np.log( 1+df2['n_clk_t1']) + 0.4*np.log( 1+df2['n_clk_t2'])

        pd_to_file( df2,  dirout + f"/{path}/clk_intra_{tk}.parquet", show=1)
        pd_to_file( df2,  dirout + f"/clk_intra_{tk}.parquet", show=1)



    ######
    def daily_create_index_faiss(add_days=0) :   ### python prepro_prod.py daily_create_index_faiss     >> "${logfile}_faiss4.py"  2>&1   &
        """### Only daily items on T, at 4am ( After donwload on new items )
            finih in1 hour,
        """
        today   = date_now_jp("%Y%m%d", timezone='jp', add_days= add_days)

        dirin   = dir_ca + f"/daily/item/ca_items2_{today}/"
        dirout  = dir_ca + f"/daily/item_vec/ca_items_{today}"

        log('####### Wait if nglist finished ', "\n", dirin,)
        os_wait_until(dirin + "/*", ntry_max=9000)

        dirbkp = dirout + f"_{int(time.time())}/"
        os.system( f" cp -r  {dirout}/   {dirbkp}  " )
        dirout = dirout +"/"


        log("\n####### Create data + vector ")
        try :
           item_add_text_vector(modelin=model0.dir, dirin= dirin + "/clean/", dirout= dirout, remove_exist=False, overwrite=True)  ## Keep all
        except Exception as e :
           log(e)


        log("\n####### Clean by score, score_clk > 0 ")
        fi = glob_glob(dirout + "/*.parquet",)
        df = pd_read_file(fi[0]) ; log(df.columns)

        dfs = pd_read_file(dirin +'/score/*clk*.parquet')
        df  = df.merge(dfs, on='siid', how='left', suffixes=(None, "2")) ; del dfs

        dfs = pd_read_file(dirin +'/score/*pur*.parquet')
        df  = df.merge(dfs, on='siid', how='left', suffixes=(None, "3")) ; del dfs

        dfn = df[ -((df['score_clk']> 0.0) | (df['score']> 0.0)) ]        ### Zero Score siid
        df  = df[ ((df['score_clk']> 0.0) | (df['score']> 0.0)) ]

        log(df[[ 'siid', 'score_clk'  ]], df.columns)
        genres = set(dfn['genre_name_path'].values).difference(  set(df['genre_name_path'].values )  )
        log('New genres with NA scores:', len(genres))
        df    = pd.concat((df, dfn[dfn['genre_name_path'].isin(genres)] ))   ### Only add new genres
        pd_to_file(df, fi[0], show=1)



        log("\n####### Create Faiss Index from today's topk ")
        dirfaiss  = dirout + f"/faiss/"
        log('Faiss Index Create:', "\n", dirout, "\n", dirfaiss)

        log("### Delete previous Faiss ")
        os.system( f" rm -rf {dirout}/faiss/ ")

        cc = Box({})
        cc.m= 32; cc.nbits= 8; cc.nlist= 5000; cc.hnsw_m=32
        faiss_create_index(df_or_path = dirout + "/*.parquet",
                           colemb  = 'item_emb', colid = 'siid',
                           dirout  = dirfaiss,    faiss_type = "IVF4096,Flat",
                           nfile=100, emb_dim=512, faiss_pars = cc)

        map_daily_ca_genre_siid()


    def daily_create_topk_batch(ii=0, ii2=500, tag='all', kbatch=20):
         ### python prepro_prod.py  daily_create_topk_batch   2>&1 | tee -a zlog_daily_02_user.py   ### Full batch 10 splits
         ### python prepro_prod.py  daily_create_topk_batch  --tag new  2>&1 | tee -a zlog_daily_02_user2.py   ### Full batch 10 splits
         ### 2mins per bucket,  50 bucket --> 100mins.
         today  = date_now_jp("%Y%m%d", timezone='jp', add_days=0)
         dirin  = dir_ca + f"/daily/item_vec/ca_items_{today}/faiss/"
         os_wait_until(dirin + "/*index", ntry_max=9000)


         if tag in ['all', 'new'] and len( os_process_find("python prepro_prod.py daily_eod_user --ii 0 ") ) > 0 :
            log('process Already running, terminating', ); return 1
         # logfile = f"/a/gfs101/ipsvols07/ndata/cpa/log/log_gpu/log_{today}_daily_ca_02_user_topk_0.py"

         if 'rename' in tag:  ## Rename previous folder
            tk = get_timekey()
            os_rename( dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod/", tag="_")

         # kbatch = 20
         jmax = min(25, int((ii2/kbatch)) ) ; log('max batch', jmax)
         for jj in range(0, jmax ):
            cmd= f" python prepro_prod.py  daily_eod_topk2 --ii {jj*kbatch}  --ii2 { (jj+1)*kbatch } --tag {tag}  &  "  ###  2>&1 | tee -a '{logfile}'
            os.system(cmd)
            time.sleep(20)


    def daily_eod_topk2(ii=0, ii2=100, tag='all', add_days= 0, uemb=0):   ### py daily_eod_topk2  --ii 0  --ii2 1 --tag new_rename  >> "${logfile}_00_eod_topk.py"  2>&1   &
        """  py daily_eod_topk2  --ii 111  --ii2 112 --tag new_rename  --add_days -1   --uemb 1 >> "${logfile}_00_eod_topk1.py"  2>&1   &

             py daily_eod_user --ii 100  &   python prepro_prod.py  daily_eod_user --ii 200 &
        """
        today0  = date_now_jp("%Y%m%d", add_days= add_days, add_hours=0, timezone= 'jp')
        #now     = date_now_jp("%Y-%m-%d--%H%M", add_days=0, add_hours=0,  timezone= 'jp')

        #### Input Data in T-1
        t0      = get_timekey() - 1 + add_days
        tk_list = [ t for t in range(t0-0  , t0 + 1 , 1)  ]   ### Past 5 day ONLY
        tmax = t0

        if    'new' in tag :   dirin_list = [ f"/a/gfs101/ipsvols07/ndata/cpa/hdfs/daily_user_eod/",   ]
        elif  'all' in tag :   dirin_list = [ f"/a/acb401/ipsvols06/pydata/brw_ran_v15/",   ]
        elif tag == 'sc'  :    dirin_list = [ f"/a/acb401/ipsvols06/pydata/sc_widget_pur/"  ]
        elif tag == 'sclk'  :  dirin_list = [ f"/a/acb401/ipsvols06/pydata/sc_widget_clk/"  ]
        else :              return 'no tag early finish'
        log('users', dirin_list) ; # time.sleep(1)

        ### Output in T
        tout   = get_timekey()
        dirout = dir_cpa3 + f"/hdfs/daily_usertopk/{model0['pref']}/{tout}/"
        index  = Index1( dirout + "/done_daily_user_eod.txt"  )
        log("\n", dirout);
        cols  = [  'easy_id', 'shop_id',  'item_id',   ]

        log("############## Faiss Loading T ########################################")
        faiss_index, map_idx_dict, dirindex = intraday_load_faiss()
        ivect = ivector(use_dict=False) ; log(ivect)
        try :
           from dbvector import Qdrant ; global clientdrant
           clientdrant  = Qdrant(table='ca_daily')
        except Exception as e: log(e)


        log("############## Top-k ##################################################")
        bkmin = ii ; bkmax = ii2
        for bk in range(bkmin, bkmax):
            flist = [] ; tk = tmax
            for dirin in dirin_list :
                if 'new' in tag: flist +=  glob_glob(dirin +  f"/{tk}/sc_stream_intra/*intra_{bk}_*.parquet" , 5000 )
                else:            flist +=  glob_glob(dirin +  f"{bk}/*{tk}*.parquet" , 5000 )

            df2 = pd.DataFrame()
            for jj,fi in enumerate( flist) :
                group   = dirin.split("/")[-2]
                log(fi)
                dirouti = dirout + f"/{group}/topk_{bk}_{tout}.parquet"
                diroutj = dirout + f"/{group}/topk_{bk}_{tout}_*.parquet"
                if len(glob.glob(diroutj )) > 0: continue

                dfe = None
                if uemb > 0 :
                    dirine = dir_cpa3 + f"/hdfs/daily_useremb/emb/{t0+1}/*_{bk}*.parquet"
                    dfe    = pd_read_file(dirine)
                    if len(dfe) < 1 : dfe = None
                    log("#### Loaded user embed: ", dfe, dirine)


                df2 = pd_read_file(fi, cols=None)      ###
                if uemb > 0 : df2 = df2.iloc[:1000, :]
                log('#### Loaded user histo: ', df2.shape, fi, "\n\n",)

                df2 = pd_easyid_get_topk(df2, faiss_index=faiss_index, faiss_map_idx=map_idx_dict, ivect=ivect, bk=bk, dfe=dfe )
                # log(df2, df2.head(1).T, df2.columns) ; time.sleep(8)
                # log('  Loaded topk', df2.shape)

                df2 = df2[ df2.topk.str.len() >50  ] ; log('  Cleaned topk', df2.shape)
                # to_file( f"{dirouti}, NA topk, " + pd_topk_check(df2) ,  dirout +"/stats.py", mode='a'  )

                log(df2[ list(df2.columns)[-5:]  ])
                pd_to_file(df2, dirouti.replace(".parquet", f"_{len(df2)}.parquet") , show=0 )
                df2 = pd.DataFrame()
                log("\n\n")
        log("############## Finished ", ii2)


    def daily_export(tag='eod', dirtarget='', add_days=0, bmin=0, bmax=500, split=0):  ### python prepro_prod.py  daily_export  count  >> "${logfile}_00_export_count.py"  2>&1   &
        ### Create user list for Rec
        today = date_now_jp("%Y%m%d", add_days=0,  timezone= 'jp')
        tk    = get_timekey() + add_days

        if split > 0 :
            batch_split_run(cmd=f"{PYFILE}  daily_export --split 0 --tag {tag} --add_days {add_days} ", split=split, sleep=10 ) ; return 'done'


        # bshops  = ca_shop_blocked_file()
        caitems = ca_today_items('latest', add_days= add_days)
        log('Ref items', len(caitems) )
        if len(caitems) < 50000 :
            log('Too small caitems ref', len(caitems)); 1/0

        def isvalid(fi):
            bi = to_int( fi.split("/")[-1].split("_")[1] )
            return  bi >= bmin and bi < bmax

        if tag == 'remove':
             dirout = dir_export + f"/{today}/eod/"
             os.system( f"rm -rf {dirout}" )
             os.system( f"ls {dirout}/../" )
            
            
        if tag == 'count':
             dirout = dir_export + f"/{today}/eod/"
             n = pd_easy_count(dirin= dirout +"/*", add_days=0)
             log(n) ; return 1

        if tag == "eod":
            dirin  = dir_cpa3   + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod/"
            dirout = dir_export + f"/{today}/eod/"
            os_wait_until(dirin + "/*", ntry_max=4000)

            log('####### Export T for Couch')
            cols  = ['easy_id',  'topk',  ]  ; nall =0
            # index = Index1(dirin + "/done_export_eod.txt")
            flist = glob_glob(dirin +"/*.parquet", 1000)
            flist = [fi for fi in flist if  isvalid(fi) ]
            log("Use files", flist )

            for fi in flist :
                diroutj =  dirout + "/" + fi.split("/")[-1]
                if os.path.isfile(diroutj) : continue

                df           = pd_read_file( fi , cols=cols, nfile=1000, n_pool=1)
                df['topk']   = df['topk'].apply(lambda x :  json.dumps([t for t in  x.split(",")  if len(t) >5 and t  in caitems  ][:40] )  )
                df['topk_n'] = df['topk'].apply(lambda x :  len(x) )
                log(df[[ 'topk', 'topk_n'   ]])

                df = df[ df['topk_n'] > 60 ]   ### Filter out users with less < 4 items
                log(df.shape)
                nall += len(df)

                pd_to_file(df[[ 'easy_id', 'topk'  ]], diroutj.replace(".parquet", f"_{len(df)}.parquet") , show=0 )
                # log(diroutj, df['easy_id'].nunique()  )

            #
            return 'Neasyid' + str(nall)


        if tag == "intra":
            dirin  = dir_cpa3   + f"/hdfs/intraday/sc_stream_usertopk/{today}/"
            if dirtarget == 'eod':
                dirout = dir_export + f"/{today}/eod/"
            else :
                dirout = dir_export + f"/{today}/intraday/"


            os_wait_until(dirin + "/*", ntry_max=4000)
            log('####### Export for intraday')
            cols  = ['easy_id',  'topk', 'ts', ]
            flist = sorted( glob_glob(dirin + "/*browsing*.parquet", 1000 ) )[-12:]       ### top most recent files  4 file per hour, 14h00 -5 hours : from 9h30 file  in Rolling
            flist = flist + sorted( glob_glob(dirin + "/*purch*.parquet", 1000 ) )[-12:]  ### Pucchase part

            log("dirin files", len(flist), str(flist))
            df = pd_read_file2(flist , cols=cols, nfile=1000, n_pool=20 )
            log(df)
            if len(df)< 1:
                log('Empty', flist); return "empty"

            df = df.sort_values(['ts' ]).drop_duplicates( 'easy_id' , keep='last')
            log('before filtering', df.shape)
            df = df[ df.topk.str.len() > 200 ]
            df = df[ df['topk'].str[0] != "," ] ### remove wrong updates

            df['topk'] = df['topk'].apply(lambda x :  json.dumps([t for t in  x.split(",")  if len(t) >5 and t  in caitems  ][:40] )  )
            df = df[ df['topk'].str.len() > 100  ]

            diroutj = dirout + f"/zzz_brw_{today}_{int(time.time())}.parquet"
            log( 'Neasy with rec < 90', len( df[ df['topk'].str.len() < 100  ] )   )

            if '/intraday/' in diroutj:
                os.system(f" rm -rf {dirout}")
            pd_to_file( df[[ 'easy_id', 'topk' ]],  diroutj, show= 1)
            pd_easy_count(dirin= dirout +"/*", add_days=0)



    def daily_export_intra(tag='intra', dirtarget='', add_days=0,):  ### py daily_export intra  >> "${logfile}_exporintra.py"  2>&1   &
        ### Create user list for Rec
        datein = date_now_jp("%Y%m%d", add_days=0,  timezone= 'jp')
        tk    = get_timekey() + add_days

        nowhour = int(date_now_jp("%H", add_days=0,  timezone= 'jp') )
        if nowhour > 0  and nowhour < 15 :
            dateout = date_now_jp("%Y%m%d", add_days=-1,  timezone= 'jp')
        else :
            dateout = date_now_jp("%Y%m%d", add_days=0,  timezone= 'jp')

        log('datein', datein, 'dateout', dateout)
        caitems = ca_today_items('intra', add_days= add_days, )
        log('Ref items', len(caitems) )
        if len(caitems) < 50000 :  log('Too small caitems ref', len(caitems)); return None


        dirin  = dir_cpa3   + f"/hdfs/intraday/sc_stream_usertopk/{datein}/"
        dirout = dir_export + f"/{dateout}/intraday_test/"

        os_wait_until(dirin + "/*", ntry_max=4000)
        log('####### Export for intraday')
        cols  = ['easy_id',  'topk', 'ts', ]
        flist =  sorted( glob_glob(dirin + "/*browsing*.parquet", 1000 ) )[-3:]  ### top most recent files  4 file per hour,
        flist += sorted( glob_glob(dirin + "/*purch*.parquet", 1000 ) )[-3:]     ### Pucchase part

        log("dirin files", len(flist), str(flist))
        df = pd_read_file2(flist , cols=cols, nfile=1000, n_pool=20 )
        log(df)
        if len(df)< 1:
            log('Empty', flist); return "empty input, No Export"

        df = df.sort_values(['ts' ]).drop_duplicates( 'easy_id' , keep='last')
        log('before filtering', df.shape)
        df = df[ df.topk.str.len() > 200 ]
        df = df[ df['topk'].str[0] != "," ] ### remove wrong updates

        df['topk'] = df['topk'].apply(lambda x :  json.dumps([t for t in  x.split(",")  if len(t) >5 and t  in caitems  ][:40] )  )
        df = df[ df['topk'].str.len() > 200  ]

        diroutj = dirout + f"/zzz_brw_{datein}_{int(time.time())}.parquet"
        log( 'Neasy with rec < 90', len( df[ df['topk'].str.len() < 200  ] )   )
        if len(df) < 100 : log(df); return 'Emppty df, no Export'


        #if '/intraday' in diroutj:
        #    os.system(f" rm -rf {dirout}")
        pd_to_file( df[[ 'easy_id', 'topk' ]],  diroutj, show= 1)
        pd_easy_count(dirin= dirout +"/*", add_days=0)




if 'daily_genre_score':
    def pd_read_file_waitready(dir0, cols=None, n_pool=1, drop_duplicates=None, date_shiftdown= True, minsize=100,  **kw) :
         #####   {datek} shift down
         df2 = pd.DataFrame() ; ii = 1; dir1 = dir0
         while len(df2) < minsize :
             if date_shiftdown :
                 ii = ii -1
                 datek = date_now_jp("%Y%m%d", add_days=ii)
                 dir1  = dir0.format(datek= datek)

             log('loading', dir1)
             df2 = pd_read_file(dir1, cols=cols, n_pool=n_pool, drop_duplicates=drop_duplicates)
         log(df2.shape, df2.columns)
         return df2

    def ca_campaign_bids(add_days=-1):
        #### Bid price for custom  # df = ca_campaign_bids()
        try :
            today0 = date_now_jp("%Y%m%d", timezone='jp',  add_days= add_days)
            dirin  = dir_ca_prod + f"/hdfs_files/prod/item_bids/item_bid*.csv"
            flist = sorted( glob_glob(dirin) )
            cols   = ['shop_id', 'item_id', 'bid', 'status' ]
            log(flist[-1])
            df     = pd.read_csv(flist[-1], names=cols )
            df['siid'] = df.apply(lambda x : f"{x['shop_id']}_{x['item_id']}" , axis=1)
            log('dfbid', df.shape)
            return df
        except : return pd.DataFrame()


    def pd_siid_score_cnt(df2):
        """  cnt :   2.648320190367988, score avg =1.0, max: 32
        t1.shop_id	t1.item_id	t1.campaign_id
        t2.cnt_pur_1d	t2.cnt_pur_2d	t2.cnt_pur_3d	t2.cnt_pur_1w	t2.cnt_pur_2w	t2.cnt_pur_1m	t2.cnt_pur_3m
        t2.gms_pur_3d	t2.gms_pur_1w	t2.gms_pur_1m	t2.gms_pur_3m	t2.row_number	t3.genre_id	t3.genre_path	t3.price
        t3.genre_name_path		t4.cnt_clk_1d	t4.cnt_clk_2d	t4.cnt_clk_3d	t4.cnt_clk_1w	t4.cnt_clk_2w	t4.cnt_clk_1m
            mean scores cnt, gms 0.7022515882904249 3.614420164306004
                    cnt_pur_1w  gms_pur_1w  cnt_pur_1m  gms_pur_1m  cnt_pur_3m   gms_pur_3m  score_cnt  score_gms      score
            0          30158.0  64466115.0     34111.0  72603175.0     70982.0  140225295.0  19.913661  22.323172  42.236832
            2           7176.0  27898728.0     25321.0  98656272.0     63541.0  245998896.0  17.711106  21.707314  39.418420
            3           9077.0  23887120.0     17015.0  47097450.0     31024.0   75570489.0  17.751672  21.236596  38.988268
            5           7210.0  50955800.0      7938.0  54878040.0     12800.0   84404320.0  16.953216  21.976098  38.929314
            11          5964.0  42079360.0      6209.0  42687640.0     17355.0  115159112.0  16.656778  21.787134  38.443912
        """
        acnt = 1/np.log(2.0 )
        df2_score = (acnt *(  0.5*np.log(1+ 7*df2['cnt_pur_1d']) + np.log(1+df2['cnt_pur_1w']) + 0.25* np.log(1+ 0.25*df2['cnt_pur_1m']) +  0.15*np.log(1 + 0.25*0.333*df2['cnt_pur_3m']) )
                    ) #### cnt_pur_1w
        return df2_score

    def pd_siid_score_gms(df2):
        """  GMS 1week avg :  14271.395711352749, score avg =1.0
        """
        agms = 1/np.log(3.0 )
        df2_score = (agms *(np.log(1+df2['gms_pur_1w']) + 0.25*np.log(1.0+ 0.25* df2['gms_pur_1m']) + 0.15*np.log(1 + 0.15*0.333 * df2['gms_pur_3m']))
                    )
        return df2_score

    def pd_siid_score_cnt_clk(df2):
        """  cnt :   2.648320190367988, score avg =1.0, max: 32
        t3.genre_name_path		t4.cnt_clk_1d	t4.cnt_clk_2d	t4.cnt_clk_3d	t4.cnt_clk_1w	t4.cnt_clk_2w	t4.cnt_clk_1m
        """
        acnt = 1/np.log(10.0 )
        df2_score = (acnt *( np.log(1+df2['cnt_clk_1w']) + 1.0* np.log(1+ 3*df2['cnt_clk_2d']) +  0.3*np.log(1 + 0.5*df2['cnt_clk_2w']) )
                    ) #### cnt_pur_1w
        return df2_score

    def pd_siid_score_cvr(df2):
        """  cnt :   2.648320190367988, score avg =1.0, max: 32
        """
        acnt = 2.3
        df2_score = np.minimum(3.0, acnt *(  (1+df2['cnt_pur_1w']) / (1+df2['cnt_clk_1w']) +  0.2 * (1+df2['cnt_pur_1m']) / (1+df2['cnt_clk_1m']) )
                    ) #### cnt_pur_1w
        return df2_score


    def map_daily_ca_genre_siid(add_days=0):  ###  python prepro_prod.py map_daily_ca_genre_siid  --add_days  0 >>  "${logfile}_ca_genre_siid4.py" 2>&1  &
         """  db_ca_genre_siid:  genre_path --> siid list.  5732 genre, 600k --> genre
         """
         log("map_daily_ca_genre_siid  from Index") ####  CA siid list
         today  = date_now_jp("%Y%m%d", add_days= add_days )
         dirin  = dir_ca + f"/daily/item/ca_items2_{today}/clean/*.parquet"
         dirout = dir_ca + f"/daily/item/ca_items2_{today}/map_ca_genre_siid/map_ca_genre_siid.parquet"

         # dirin  = dir_ca + f"/daily/item_vec/ca_items_{today}/*clean*.parquet"
         df     = pd_read_file(dirin, nfile=1, nrows=5000500050)
         log(df.columns)
         df = df.drop_duplicates([ 'shop_id', 'item_id' ])
         df = df[df['shop_id'] > 0 ]
         df = df[df['genre_name_path'].str.len() > 10 ]  ### only with genre name
         df = df[df['item_name'].str.len() > 10 ]        ### only with item name

         df['genre_id'] = df['genre_id_path'].apply(lambda x: to_int( str(x).split("/")[-1] ) )
         colg = 'genre_id'   ### string
         log(df, dirin)

         log("#### Scores per siid  ################################################")
         dir2 = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/score/clk*.parquet"  #### Use Real time
         df2  = pd_read_file(dir2, cols=None, n_pool=1,)

         dir2 = dir_cpa3 + f"/ca_check/daily/item/ca_items2_{today}/score/pur*.parquet"  #### Use Real time
         dfi  = pd_read_file(dir2, cols=None, n_pool=1,)
         log(dfi.shape)

         df2 = df2.merge(dfi, on='siid', how='left')
         df2['score'] = df2['score'].fillna(0.0)
         df2['score'] = df2['score'] +  0.3* df2['score_clk']
         log(df2)


         log("######### Bidding data   #################################################")
         dfbid = ca_campaign_bids(add_days=  add_days   )  ### Today's bid after 3am
         dfbid = dfbid[[ 'siid', 'bid' ]] ; log(dfbid)
         dfbid['bid'] = dfbid['bid'].astype('float32')
         dfbid['bid'] = np.minimum( 1.3 , 0.4 * np.log( dfbid['bid'] / 40.0 ) )   ###  50/40 * 1.5 --> 1.8

         df2           = df2.merge(dfbid, on='siid', how='left' )
         df2['bid']    = df2['bid'].fillna(1.0)
         df2['score2'] = df2['score'] * df2['bid']  ### Bidding Shift
         log(df2[[ 'siid', 'bid', 'score2'  ]])


         log("######### Merge score, bid  : for popularity per genre   #################")
         df           = df.merge(df2[ ['siid'] + [t for t in df2.columns if t not in df.columns]   ], on=['siid' ], how='left', suffixes=(None,"_2") )
         df['score2'] = df['score2'].fillna(0.0)
         df           = df.sort_values( ['score2'], ascending=[0]).drop_duplicates('siid', keep='first')
         pd_to_file(df, dirout, show=1)

         ### Remove No GMS siid
         df = df[df['score2'] > 0.05 ]  ;  log(df, df.columns)


         log("######### genre ---> list of top siid  (daily), USING score2(bid) #######################################")
         #df2  = df.groupby(colg).apply( lambda x : ",".join(x['siid'])  + ";" +  ",".join([  str(t) for t in  x['score']]) ).reset_index()
         df2  = df.groupby(colg).apply( lambda x : ",".join(x['siid'])  + ";" +  "").reset_index()
         df2.columns = [ colg, 'siids']
         df2[colg]   = df2[colg].astype('str')    #### Need to convert to STRING

         ###  Need to delete the table before
         db_ca_genre_siid.clear() ; log('Size db_ca_genre_siid: ',  len( db_ca_genre_siid ) )
         diskcache_save(df2, db_path=  db['db_ca_genre_siid'] , colkey=colg, colvalue='siids',  npool=1, verbose=False )


         log("#########  Most popular: Sampling, USING score(no bid)   ##############")
         daily_top_popular_update(df=df, colscore_sort = 'score')


         log("#########  Update mapping siid --> genreid   ##########################")        
         diskcache_save2(df,db_path=  db['db_ca_siid_genre'] , colkey= 'siid', colvalue= colg,  npool=5, verbose=False , ttl = 186400 )


         ##### siid --> Imaster Infos
         df['imaster'] = df.apply(lambda x : {'genre_path': x['genre_id_path'],   'image_url': x['item_image_url'] }, axis=1)
         diskcache_save2(df, db_path=  db['db_imaster'] , colkey= 'siid', colvalue= 'imaster',  npool=5, verbose=False )

            
    def daily_top_popular_update(df=None, colscore_sort='score'):
         if df is not None :
             # df1 = df.drop_duplicates('genre_id', keep='first').iloc[:200, :]
             df1 = df.sort_values( [colscore_sort], ascending=[0]).iloc[:100, :]
             log('Top popular',   df1[[ 'siid', 'genre_id', colscore_sort  ]] )
             topk_popu = list( df1['siid'].values)

         else :
             topk_popu = db_ca_genre_siid['topk_popular']
             caitems   = ca_today_items(path='latest', colkey='siid', colval='dum', add_days=0 )
             topk_popu = [ t for t in topk_popu if t in  caitems ]

         db_ca_genre_siid['topk_popular'] =   topk_popu
         log( 'Top popular',   len(db_ca_genre_siid['topk_popular']), str(db_ca_genre_siid['topk_popular'])[:100], )
         log( 'ca genre_siid', len(db_ca_genre_siid ) )

    def daily_map_siid_genre(add_days=0):     ### python prepro_prod.py daily_map_siid_genre  add_days=-1    >>  "${logfile}_siid_genreid.py" 2>&1  &
         """   ####  siid ---> genre  (daily)
               [507510043 rows x 3 columns]
               Starting insert:  /sys/fs/cgroup/cpa/db/db_ca_siid_genre.cache (37881359, 2)
         """
         tk = date_now_jp("%Y%m%d", add_days=add_days)
         dirin = dir_cpa3 + "/hdfs/intraday/sc_stream_item/"
         flist = glob_glob(dirin + f"/{tk}/*0.parquet", 30000)
         log(len(flist))
         index = Index1( dirin +"/done_genreid.txt" )
         # flist = index.save_filter(flist)
         if len(flist) < 1 : return 'No files'
         cols = ['item_id', 'shop_id']
         colg = 'genre_id'
         df = pd_read_file2( flist, n_pool=20,  cols=cols+ [ 'genre_id'],  drop_duplicates=cols, verbose=False)
         log(df)
         df = df.drop_duplicates(cols)
         df['siid'] = df.apply(lambda x: siid(x), axis=1)
         del df['item_id'] ; del df['shop_id']
         diskcache_save2(df,db_path=  db['db_ca_siid_genre'] , colkey= 'siid', colvalue= colg,  npool=5, ttl= 86400 * 20 , verbose=False )

        
    #################    
    def daily_siid_score5():         
         """
         dir2 = dir_cpa3 + "/input/ca_siid_feat_all_{datek}/*"
         df2  = pd_read_file_waitready(dir2, cols=None, n_pool=20, drop_duplicates=None, date_shiftdown= True, )
         log(df2.columns)
         df2['siid'] = df2.apply(lambda x : siid(x), axis=1)
         df2         = df2.drop_duplicates('siid', keep='first')
         cols1 = ['cnt_pur_1w',  'gms_pur_1w', 'cnt_clk_1w',  'cnt_pur_1m',    'gms_pur_1m', 'cnt_clk_2w',   'cnt_pur_3m',  'gms_pur_3m',   'cnt_clk_2d',     ]
         #for x in cols1:  df2[x] = df2[x].fillna(0.0)
         df2 = df2.fillna(0.0)
                
         #### GMS Based scores  #################################
         df2['score_cnt'] = pd_siid_score_cnt(df2)
         df2['score_gms'] = pd_siid_score_gms(df2)
         df2['score_clk'] = pd_siid_score_cnt_clk(df2)
         df2['score_cvr'] = pd_siid_score_cvr(df2)

         #df2['score']   = df2.score_cnt + df2.score_gms + df2.score_clk + df2.score_cvr
         df2['score']    = df2.score_cnt         
         df2['score']    = df2['score'].apply(lambda x : 0.01 if x < 1 else x )  ### to have correct bid price impact
         df2             = df2.sort_values( ['score'], ascending=[0])
            
         log('mean scores cnt, gms', df2.score_cnt.mean(), df2.score_gms.mean(), df2.score_clk.mean(), df2.score_cvr.mean() )
         log(df2[ cols1 + [ 'score_cnt',  'score_gms', 'score_clk',  'score_cvr', 'score'  ]])
         """
         pass   



if 'intraday, easyid topk':
    def intraday_load_faiss():
        log("############## Faiss Loading T-1 ##################################################")
        isok= False
        add_days = 1

        ###3pm japan adjustment
        add_hours = -15

        while not isok :
            add_days = add_days - 1
            today0      = date_now_jp("%Y%m%d", add_days= add_days, add_hours=0, timezone= 'jp')
            dirpath     = dir_ca + f"/daily/item_vec/ca_items_{today0}/faiss/*.index"
            dirindex    = glob_glob( dirpath , 1)
            if len(dirindex) > 0 :
               dirindex     = dirindex[0]
               faiss_index  = faiss_load_index(db_path=dirindex)
               dirmap       = dirindex.replace("faiss_trained", "map_idx").replace(".index", '.parquet')
               map_idx_dict = db_load_dict(dirmap,  colkey = 'idx', colval = "siid", colkey_type='int' )  ### idx --> siid
               log("index loaded,", dirindex)
               return faiss_index, map_idx_dict, dirindex

            if add_days < -2 : break
            log('Index not available: ', dirpath)


    def intra_i_jax(today=None):   intraday_item_vec_calc(today)
    def intraday_item_vec_calc(today= None):    ### py intraday_item_vec_calc     ### every 10mins
        ### */5 * * * * /python prepro_prod.py intraday_item_vec_calc  > "/a/gfs101/ipsvols07/ndata/cpa/log/log_gpu/1hour.py"  2>&1   &
        ###   in  daily_1hour.sh
        today = date_now_jp("%Y%m%d", timezone='jp' ) if today is None else today
        log(' Intraday  Item updater', today)
        dirin   = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/"
        dirout  = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/df/"

        status = item_add_text_vector(modelin=None, dirin=dirin, dirout=dirout, nrows=500500500, nfile=1000,  remove_exist= True
                             #today='20211103',
                            )
        #### Previous day updates
        if status in  ['empty']:
            today = date_now_jp("%Y%m%d", timezone='jp', add_days=-1 )
            log(' Intraday  Item updater', today)
            dirin   = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/"
            dirout  = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/df/"

            status = item_add_text_vector(modelin=None, dirin=dirin, dirout=dirout, nrows=500500500, nfile=1000,  remove_exist= True
                                 #today='20211103',
                                )


    def intra_u_jax(test=0): 1/0; intraday_eaysid_topk(test=test)    ### nice -n 10  python prepro_prod.py  intra_u_jax
    def intraday_eaysid_topk(test=0,) :     ### python prepro_prod.py intraday_eaysid_topk  --test 1   2>&1 | tee -a  "${logfile}_topk_intra_test.py"
        """   Depend on Faiss Calc
             #dirin  = dir_cpa3 + f"/hdfs/intraday/sc_stream_item_vec/{today}/df/"
        """
        today    = date_now_jp("%Y%m%d", timezone='jp', add_days= 0)
        today1   = date_now_jp("%Y%m%d", timezone='jp', add_days= -1)
        # today1   = "20211031"
        tk = get_timekey()

        if now_hour_between(hour1="00:01", hour2="06:00", timezone="jp"):
            log('No compute'); return None

        log(' Intraday  User updater', today)
        dirin   = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/"
        #dirout  = dir_cpa3 + f"/hdfs/intraday/sc_stream_usertopk/{today}/"
        dirout  = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/intraday/out/"


        log("\n\n############## Check new files  ##########################################")
        os_makedirs(dirout)
        flist = glob_glob(dirin + "/*.parquet", 1000)
        index = Index1( dirout  + "/done.txt"  )
        flist = [ t for t in flist if t not in index.read()     ]

        def isvalid(t,):
            tr = date_now_jp("%Y-%m-%d", timezone='jp', add_days= -1)   ### --20: 6am,  --21: 7am
            rref = [ tr + "--14",   tr + "--15", tr + "--16", tr + "--17", tr + "--18", tr + "--19", tr + "--20", ]  ### tr + "--20",  ##  'in-progress'
            for r in rref:
                if r in t : return False
            return True
        flist  = [t for t in flist if isvalid(t)]

        #log(flist, index.read()  )
        if len(flist) < 1 : log('No new files', len(flist)) ; return 1

        log("############## Faiss Loading T   ##################################################")
        faiss_index, map_idx_dict, dirindex = intraday_load_faiss()
        ivect = ivector(use_dict=False) ; log(ivect)
        global ok_global ;  ok_global = ok_get_global_siid(add_days=0)
        global clientdrant ; drant_init( ndim=512 , efsearch= 32, add_days=0)


        log("\n############## Loop over files: topk emb  ######################################")
        log('db ca genre', db_ca_genre_siid,  )
        log('Nfiles', len(flist))
        ii = 0
        for fi in flist :
            if fi in index.read()      : continue
            index.save([fi])
            dirouti = dirout + fi.split("/")[-1]
            if os.path.isfile(dirouti) : continue

            log("\n", fi)
            df = pd_read_file( fi )  ; log(df.shape)
            # df = df.iloc[:5000, :]

            if test>0 :      log('using test'); df = df.iloc[:100, :]
            if len(df) < 1 : log('Empty df:', fi) ; continue
            log('unique', df.shape)

            df = df.sort_values('ts', ascending=1)
            df = pd_easyid_get_topk_intra(df, faiss_index=faiss_index, faiss_map_idx=map_idx_dict, ivect=ivect )

            to_file( f"{dirouti}, easyid NA emb, " + str(  len( df[ df['topk'].str.len() < 200 ] ) ),  dirout +"/stats.py", mode='a'  )

            colsc = [ ci for ci in df.columns if "_n" in ci]
            log(df[ colsc ])
            log(df[ colsc ].mean() )
            pd_to_file(df, dirouti , show=0 )
            ii = ii + 1
        log('finished, nfiles:', ii)



    def pd_easyid_get_topk2(df, topk=50, cass_query=None, today1=None, faiss_index=None, faiss_map_idx=None, ivect=None, bk=0,dfe=None,  **kw):
        today1   = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)  if today1 is None else today1
        today    = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)
        # today1   = "20211031"
        ### can have mutiple   easyid, siid  pairs (historical)
        log(df.shape)

        #### Default topk popular
        recgenre_init_proba()    #### Fixed proba for sampling  topk: 1-K


        log('####### Embedding Rec ')
        df = create_rec_topemb3(df, faiss_index= faiss_index, faiss_map_idx= faiss_map_idx, ivect=ivect, maxrec=60, dfe=dfe)
        df = df.rename(columns={'topk': 'topk_emb'})


        log('\n###### Genre pur Rec: eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, topgenre='pur',   ngenre=6, topk=5 )    ### Diversity

        log('\n###### Genre brw Rec: eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, topgenre='brw',   ngenre=6, topk=5 )    ### Diversity

        log('\n###### Genre Intra Rec : eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, topgenre='intra', ngenre=6, topk=5 )    ### Diversity


        log("###### Merge, add on topk list   ")
        #### 'ng_siid' column, need the bucket bk to match !!!!!!!!
        df = pd_add_ng_easysiid(df, bk=bk)  ####

        #df['topk'] = df.apply(lambda x : pd_siid_combine3( x['topk_emb'], x['topk_brw'], x['topk_pur'], x['topk_intra'],   n1=4,   )   , axis=1  )
        df['topk'] = df.apply(lambda x : pd_siid_combine3( x['topk_emb'], x['topk_brw'], x['topk_pur'], x['topk_intra'],
                                                           ngs= x['ng_siid']  if 'ng_siid' in x else set(),
                                                           n1=5,  nmax=50 )   , axis=1  )

        df = create_rec_addon(df, topop= None, maxrec=40 )
        if 'ng_siid' in df.columns :  del df['ng_siid']

        df = pd_topk_count( df, [ 'topk',  'topk_emb', 'topk_intra',  'topk_pur', 'topk_brw'  ] )
        log(df.shape)
        # log( 'Nitems min', df.topk_n.min() )
        return df


    def pd_easyid_get_topk(df, topk=50, cass_query=None, today1=None, faiss_index=None, faiss_map_idx=None, ivect=None, bk=0,dfe=None,  **kw):
        today1   = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)  if today1 is None else today1
        today    = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)
        # today1   = "20211031"
        ### can have mutiple   easyid, siid  pairs (historical)
        log(df.shape)

        #### Default topk popular
        recgenre_init_proba()    #### Fixed proba for sampling  topk: 1-K


        log('######## Embedding Rec ')
        df = create_rec_topemb3(df, faiss_index= faiss_index, faiss_map_idx= faiss_map_idx, ivect=ivect, maxrec=45, dfe=dfe)
        df = df.rename(columns={'topk': 'topk_emb'})


        log('\n###### Genre pur Rec: eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, ngenre=5, topk=40)  ### topk_genre, topk_genre_n


        log("###### Merge, add on topk list   ")
        #### 'ng_siid' column, need the bucket bk to match !!!!!!!!
        df = pd_add_ng_easysiid(df, bk=bk)  ####
        global ok_global ;  ok_global = ok_get_global_siid(add_days=0)  if ok_global is None else ok_global

        df['topk'] = df.apply(lambda x : pd_siid_combine5( x['topk_emb'], x['topk_genre'],
                                                           ngs= x['ng_siid']  if 'ng_siid' in x else set(),
                                                           ok_global = ok_global,
                                                           n1=8,  nmax=45 )   , axis=1  )

        df = create_rec_addon(df, topop= None, maxrec=40 )

        df = pd_del(df, cols=['ng_siid', 'user_emb'])

        df = pd_topk_count( df, [ 'topk',  'topk_emb', 'topk_genre',    ] )
        log(df.shape)
        # log( 'Nitems min', df.topk_n.min() )
        return df


    def pd_easyid_get_topk_intra2(df, topk=50, cass_query=None, today1=None, faiss_index=None, faiss_map_idx=None, ivect=None, dfe=None, **kw):
        today1   = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)  if today1 is None else today1
        today    = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)
        # today1   = "20211031"

        # df = df.sort_values('ts', ascending=0)  #### bottom is most recent siid
        log(df.shape)

        #### Default topk popular
        ## topop = rec_topopular()
        recgenre_init_proba()    ### Fixed proba for sampling  topk: 1-K


        log('\n###### Embed Rec:  siid --> list of topk ')
        #df = create_rec_topemb2(df, faiss_index= faiss_index, faiss_map_idx= faiss_map_idx, ivect=ivect)
        df = create_rec_topemb3(df, faiss_index= faiss_index, faiss_map_idx= faiss_map_idx, ivect=ivect, dfe=dfe)
        df = df.rename(columns={'topk': 'topk_emb'})

        #### 'ng_siid' column, need the bucket bk to match : cannot in real time
        # df = pd_add_ng_easysiid(df, bk=bk)  ####

        log('\n###### Genre Intra rec : eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, topgenre='intra', ngenre=6, topk=4 )   ### Diversity

        log('\n###### Genre pur Rec:    eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, topgenre='pur',   ngenre=6, topk=4 )     ### Diversity

        log('\n###### Genre brw Rec:    eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, topgenre='brw',   ngenre=6, topk=4 )     ### Diversity

        log("###### Merge, add on topk list   ")
        #df['topk'] = df.apply(lambda x : pd_siid_combine3( x['topk_emb'], x['topk_brw'], x['topk_pur'], x['topk_intra']   )  , axis=1  )
        df = pd_add_ng_easysiid_intra(df, bk=None)  ####  NG siid through real time parsing

        df['topk'] = df.apply(lambda x : pd_siid_combine3( x['topk_emb'], "" , "",  "",
                                                           ngs= x['ng_siid']  if 'ng_siid' in x else "",
                                                           )  , axis=1  )
        # df = create_rec_addon(df, topop= None)
        if 'ng_siid' in df.columns :  del df['ng_siid']


        # df['topk'] = df['topk'].apply(lambda x :  genre_add_on(x  ) )  #### Add Top popular for unknown users
        df = pd_topk_count( df, [ 'topk',  'topk_emb', 'topk_intra',  'topk_pur', 'topk_brw'  ] )
        log( df.shape )
        return df


    def pd_easyid_get_topk_intra(df, topk=50, cass_query=None, today1=None, faiss_index=None, faiss_map_idx=None, ivect=None, dfe=None, **kw):
        today1   = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)  if today1 is None else today1
        today    = date_now_jp("%Y%m%d", timezone='utc', add_days= 0)
        # today1   = "20211031"

        # df = df.sort_values('ts', ascending=0)  #### bottom is most recent siid
        log(df.shape)

        #### Default topk popular
        recgenre_init_proba()    ### Fixed proba for sampling  topk: 1-K


        log('\n###### Embed Rec:  siid --> list of topk ')
        df = create_rec_topemb3(df, faiss_index= faiss_index, faiss_map_idx= faiss_map_idx, ivect=ivect, dfe=dfe, maxrec=35,)
        df = df.rename(columns={'topk': 'topk_emb'})

        #### Addd 'ng_siid', 'useremb' column, need the bucket bk to match : cannot in real time
        df   = pd_add_easy_allinfo_intra(df, bk=None)  ####  NG siid through real time parsing
        df = df[df['user_emb'].str.len() > 7 ]  ; log('Correct with user_emb', df.shape)

        log('\n###### Genre pur Rec: eaysid --> genre --> list of siid ')
        df = create_rec_topgenre(df, ngenre=5, topk=35)  ### topk_genre, topk_genre_n


        log("###### Merge, add on topk list   ")
        global ok_global ;  ok_global = ok_get_global_siid(add_days=0)  if ok_global is None else ok_global
        log('Global ngs', len(ok_global))

        df['topk'] = df.apply(lambda x : pd_siid_combine5intra( x['topk_emb'], x['topk_genre'],
                                                           ngs       = x['ng_siid']  if 'ng_siid' in x else set(),
                                                           ok_global = ok_global,
                                                           n1=9,  nmax=35 )   , axis=1  )

        df = create_rec_addon(df, topop= None, maxrec=35,)

        df = pd_del(df, ['ng_siid', 'user_emb'])
        df = pd_topk_count( df, [ 'topk',  'topk_emb', 'topk_genre',    ] )
        log( df.shape )
        return df


    def pd_add_easy_allinfo_intra(df,bk):  ###merge with ng easy_id siid list , useremb
        ### in combine  ngs =  set(x['ng_siid'].split(","))
        tk = get_timekey()
        t0 = date_now_jp()

        dirin = dir_cpa3 + f"/hdfs/intraday/sc_stream_userng/{t0}/*.parquet"
        flist = sorted( glob_glob(dirin, 1000) )
        flist = flist[-20:]
        log(flist)

        dfn = pd_read_file( flist, n_pool=15, cols= ['easy_id', 'ng_siid'] )
        dfn = dfn.drop_duplicates('easy_id')
        log(dirin,   dfn.shape, dfn )

        if len(dfn)< 1:
            df['ng_siid'] = '' ;  df['useremb'] = '' ;  return df

        df            = df.merge(dfn, on='easy_id', how='left')

        df['ng_siid'] = df['ng_siid'].fillna('')
        log('Correct easyid ngsiid', len(df[ df['ng_siid'].str.len() > 7  ] ) )
        df['ng_siid'] = df['ng_siid'].apply(lambda x : set(x.split(',')) )


        log('### Get useremb ')
        uservect = uvector() # if uservect is None else uservect
        ddemb    = uservect.get_multi( df['easy_id'].unique(), use_dict=False, use_cache=True )  ### All from Cass
        ddemb    = { int(key):val for key,val in ddemb.items() }
        log('useremb_dict key: ', next(iter(ddemb)) )
        #df2 = pd.DataFrame.from_dict(df2, orient='index', columns= ['user_emb'] ).reset_index()
        #df2.columns = ['easy_id', 'user_emb']
        df['user_emb'] = df.apply(lambda x : ddemb.get(x['easy_id'], "0") , axis=1)
        # df['user_emb'] = df.apply(lambda x : ddemb.get(x['easy_id'], x['user_emb']) , axis=1)
        # df['user_emb'] = df['user_emb'].replace('', '0')  #### For vector emb calc
        log( 'Correct easyid useremb', len(df[ df['user_emb'].str.len() >8 ] ) )


        #if 'user_emb' in df.columns :
        #    df['user_emb'] = df['user_emb'].fillna('')
        #    df['user_emb'] = df['user_emb'].replace('', '0')  #### For vector emb calc
        #    log( 'Correct easyid useremb', len(df[ df['user_emb'].str.len() >8 ] ) )

        # log('NG easyid',t0)
        return df


    ########################################################
    def create_rec_addon(df, topop=None, maxrec=40, ngs=None):
        """ Add Missing topk
        """
        topop = rec_topopular()

        def genre_add_on(s, ngs=None): ###Missing genre
            s = [ t for t in s.split(",") if len(t)> 8 ]
            n = len(s)
            if n < maxrec :
                topopi =  np_sample(topop, lproba=None, mode='inverse', k=maxrec-n,  replace=False)  ### Sample from Top popular
                topopi =  [ t for t in topopi if t not in ngs ]
                return  ",".join( s + topopi)   ###3 Need to add ","
            else :
                return  ",".join(s)

        df['topk'] = df.apply(lambda x :  genre_add_on(x['topk'], x['ng_siid']  if 'ng_siid' in x else set() ), axis=1 )  #### Add Top popular for unknown users
        return df


    def rec_topopular():
        topop = db_ca_genre_siid['topk_popular']
        topop = topop[:70]
        log('Load Top Popu', len(topop), str(topop)[:50])
        if len(topop) < 40:  log("Error: db_ca_genre_siid.get(topk_popular   lenghth < 40 ",)  ; 1/0
        return topop



    ########################################################
    def create_rec_topemb2(df, faiss_index=None, faiss_map_idx=None, ivect=None):
        """ siids --> genre for siids --> Topk for each (siid, genreid) --> topk, merge.

        """
        if faiss_index is None :
            dirindex    = glob_glob(dir_ca + f"/daily/item_vec/ca_items_{today1}/faiss/*.index" , 1)[0]
            faiss_index = dirindex  ;  log(dirindex)

        df = df.drop_duplicates(['easy_id', 'genre_id'], keep='last')
        log('easyid, genreid unique', df.shape)

        df1 = df.sort_values(['easy_id', 'ts'], ascending=[1,0] )  ###last is on top
        log(df1)
        df2 = pd_cass_get_vect2(df1, prefix= model0.pref, tablename="ndata.item_model", ivect=ivect )
        df2 = faiss_topk2(df = df2,   colid='siid', colemb='item_emb', topk= 40, npool=1, nrows=10**9, nfile=1000,
                          faiss_index=faiss_index, map_idx_dict= faiss_map_idx   )
        df2 = df2[['siid', 'siid_list' ]]
        df2.columns = [  'siid', 'topk' ]

        ### Aggrefate Mutiple top-k into one.
        df2['topk'] = df2.apply(lambda x : pd_siid_cleangenre(x, topk=12) , axis=1)          ####

        ### Merge back per siid
        df1 = df1.merge(df2, on='siid', how='left')
        df1 = df1[ df1.topk.str.len() > 10 ] ## remove NA
        df1 = df1.groupby('easy_id').apply(lambda dfi:  ",".join( dfi['topk'].values[:3] ) ).reset_index()  ## Merge 5 <> genreid
        df1.columns = ['easy_id', 'topk']
        log(df1.columns)

        ### Merge back per easy_id
        df = df.sort_values('ts').drop_duplicates("easy_id", keep='last')
        df = df.merge(df1, on='easy_id'  , how='left')
        df['topk']  = df['topk'].fillna("")
        log(df['topk'])
        return df


    def create_rec_topemb(df, faiss_index=None, faiss_map_idx=None, ivect=None):
        """
          siids --> genre for siids --> Topk for each (siid, genreid) --> topk, merge.
          siid --> topk from embedding,
        """
        df = df.drop_duplicates(['easy_id'], keep='last')

        if faiss_index is None :
            dirindex    = glob_glob(dir_ca + f"/daily/item_vec/ca_items_{today1}/faiss/*.index" , 1)[0]
            faiss_index = dirindex  ;  log(dirindex)

        df2 = pd_cass_get_vect2(df, prefix= model0.pref, tablename="ndata.item_model", ivect=ivect )
        df2 = faiss_topk2(df = df2,   colid='siid', colemb='item_emb', topk= 40, npool=1, nrows=10**9, nfile=1000,
                          faiss_index=faiss_index, map_idx_dict= faiss_map_idx   )
        df2 = df2[[ 'siid', 'siid_list' ]]
        df2.columns = ['siid', 'topk' ]
        df2['topk'] = df2.apply(lambda x : pd_siid_cleangenre(x, topk=10) , axis=1)          ####
        df          = df.merge(df2, on='siid', how='left')
        df['topk']  = df['topk'].fillna("")
        return df


    def faiss_topk2(df=None,  colid='id', colemb='emb', faiss_index=None, topk=200, npool=1, nrows=10**7, nfile=1000, faiss_pars={},
                    map_idx_dict=None,
                   ) :
       ##  py  faiss_topk   2>&1 | tee -a zlog_faiss_topk.txt
       """ id, dist_list, id_list
       """
       mdim = 512
       cc = Box(faiss_pars)
       log('Faiss Index: ', faiss_index)
       if isinstance(faiss_index, str) :
            faiss_path  = faiss_index
            faiss_index = faiss_load_index(db_path=faiss_index)
       faiss_index.nprobe = 12  # Runtime param. The number of cells that are visited for search.

       ####### Single Mode #################################################
       if map_idx_dict is None :
          dirmap       = faiss_path.replace("faiss_trained", "map_idx").replace(".index", '.parquet')
          map_idx_dict = db_load_dict(dirmap,  colkey = 'idx', colval = colid, colkey_type='int' )  ### idx --> siid

       chunk  = 200000
       kk     = 0

       log(df.columns, df.shape)
       df = df.iloc[:nrows, :]

       dfall  = pd.DataFrame()   ;    nchunk = int(len(df) // chunk)
       for i in range(0, nchunk+1):
           if i*chunk >= len(df) : break
           i2 = i+1 if i < nchunk else 3*(i+1)

           x0 = np_str_to_array( df[colemb].iloc[ i*chunk:(i2*chunk)].values   , l2_norm=True, mdim = mdim )
           log('X topk', x0.shape )
           # topk_dist, topk_idx = faiss_index.search(x0, topk)
           _, topk_idx = faiss_index.search(x0, topk)
           log('X', topk_idx.shape)

           dfi                   = df.iloc[i*chunk:(i2*chunk), :] # [[ colid ]]
           dfi[ f'{colid}_list'] = np_matrix_to_str2( topk_idx, map_idx_dict)  ### to item_tag_vran
           # dfi[ f'dist_list']  = np_matrix_to_str( topk_dist )
           # dfi[ f'sim_list']     = np_matrix_to_str_sim( topk_dist )
           dfall = pd.concat((dfall, dfi))
           # log(i, dfi[[ f'{colid}_list', 'sim_list'  ]])
       return dfall



if 'combine, clean topk list':
    #######################################################
    def t1():
        siids= [ '1223_123', '287670_10001822', '287670_10001822', '247351_10000326', '247351_10000328', '243166_10000340',
             '203146_10054840', '227132_10000357', '275136_10004870',  ]
        log( pd_siid_getranid(siids)  )
        siids= ",".join(siids)

        ss2 = pd_siid_clean(siids)
        log(ss2)

        a = pd_add_itemaster(pd.DataFrame(siids, columns=['siid']) , cols_cass=['genre_path'])

    from itertools import chain, zip_longest
    def pd_siid_combine(t1, t2, t3):   #### interleaving technics
        t1 = t1.split(",")
        t2 = t2.split(",")
        t3 = t3.split(",")

        rdict  = pd_siid_getranid( t1 + t2 + t3 )
        ranid0 = set()
        ss = ""
        for x in chain.from_iterable(zip_longest(t1, t2, t3)):
           if x and len(x) > 4 :
              ranid= rdict.get(x, x)
              # print(ranid)
              if ranid not in ranid0 :
                ranid0.add(ranid)
                ss = ss + x +","
        return ss[:-1]


    def pd_siid_combine2(t1, t2, t3, nrecent=3):
        """
          remove similar genres

        """
        t1 = t1.split(",")
        t2 = t2.split(",")
        t3 = t3.split(",")

        rdict  = pd_siid_getranid( t1 + t2 + t3 )
        ranid0 = set()
        ss = ",".join(t1[:nrecent]) +  "," + ",".join(t2[:2]) + "," + ",".join(t3[:2]) + ","

        try :
            while ss[0] == "," :
                ss = ss[1:]  ### remove the ","
        except : pass

        ii = 7
        for x in chain.from_iterable(zip_longest(t1[nrecent:], t2[2:], t3[2:])):
           # print("")
           if ii >= 31 : break
           if x and len(x) > 4 :
              ranid= rdict.get(x, x)
              # print(ranid)
              if ranid not in ranid0 :
                ii = ii + 1
                ranid0.add(ranid)
                ss = ss + x +","
        return ss[:-1]


    def pd_siid_combine3(t1, t2, t3, t4, ngs=None, n1=2, n2=2, n3=2, nmax=51):
        """ Remove similar genres
        """
        t1 = t1.split(",")   ###  past pur
        t2 = t2.split(",")   ###  past brw
        t3 = t3.split(",")   ###  past pur
        t4 = t4.split(",")   ###  intra brw

        # rdict  = pd_siid_getranid( t1 + t2 + t3 + t4 )   ## ranid is already separated a Index.
        # ranid0 = set()

        t1 = [ t for t in t1 if len(t) >5 ]
        t4 = [ t for t in t4 if t not in t1    and len(t) >5  ]
        t2 = [ t for t in t2 if t not in t4+t1 and len(t) >5  ]
        t3 = [ t for t in t3 if t not in t2+t4 and len(t) >5  ]


        #### NG easyid_siid
        if ngs is None :    ngs = set()
        elif len(ngs) < 1 : ngs = set()
        # else :              ngs = set(ngs.split(","))


        #### Item intra, item genre intra,  brw past, pur past
        ss = t1[:n1] + t4[:n2] + t2[:n3] + t3[:2]
        ii = len(ss)
        ss = ",".join(ss) +  ","

        # ii = 0
        for x in chain.from_iterable(zip_longest(t1[n1:],  t4[n2:], t2[n3:], t3[2:], )):
           # print("")
           if ii >= nmax : break
           if x is not None and x not in ngs :
               # ranid= rdict.get(x, x)
               # print(ranid)
               #if ranid not in ranid0 :
               ii = ii + 1
               #ranid0.add(ranid)
               ss = ss + x +","

        if len(ss) < 8 : return ""
        return ss[:-1]


    def pd_siid_combine0(t1, t2, t3):
        t1 = t1 + "," + t2 +"," +t3
        t1 = t1.split(",")

        rdict  = pd_siid_getranid( set(t1) )
        ranid0 = set()
        ss = ""
        for x in t1 :
           if len(x) > 4 :
              ranid= rdict.get(x, x)
              if not ranid in ranid0 :
                ranid0.add(ranid)
                ss = ss + x +","
        return ss[:-1]



    def pd_siid_combine5intra(t1, t2="", t3="", t4="", ngs=None, n1=0, n2=0, n3=0, n4=0, ok_global=None, nmax=51):
        """ Remove similar genres, remove NG
        """
        t1 = t1.split(",")  ###  past pur
        t2 = t2.split(",")  ###  past brw
        t3 = t3.split(",")  ###  past pur
        t4 = t4.split(",")  ###  intra brw

        t1 = [t for t in t1 if len(t) > 5]
        t2 = [t for t in t2 if t not in t1 and len(t) > 5]
        t3 = [t for t in t3 if t not in t2 and t not in  t1 and len(t) > 5]
        t4 = [t for t in t4 if t not in t3 and t not in  t2 and len(t) > 5]

        #### NG easyid_siid
        if ok_global is None:    ok_global = set()
        elif len(ok_global) < 1: ok_global = set()

        if ngs is None:    ngs = set()
        elif len(ngs) < 1: ngs = set()
        # else :              ngs = set(ngs.split(","))

        #### Item intra, item genre intra,  brw past, pur past
        ss = t1[:n1] + t2[:n2] + t3[:n3] + t4[:n4]
        ii = len(ss)
        ss = ",".join(ss) + ","

        # ii = 0
        for x in chain.from_iterable(zip_longest(t1[n1:], t2[n2:], t3[n3:], t4[n4:], )):
            # print("")
            if ii >= nmax: break
            if x is not None and x not in ngs and x in ok_global and len(x)>8:
                ii = ii + 1
                ss = ss + x + ","

        if len(ss) < 8: return ""
        if ss[0] == "," : return ss[1:-1]
        return ss[:-1]


    def pd_siid_combine5(t1, t2="", t3="", t4="", ngs=None, n1=0, n2=0, n3=0, n4=0, ok_global=None, nmax=51):
        """ Remove similar genres, remove NG
        """
        t1 = t1.split(",")  ###  past pur
        t2 = t2.split(",")  ###  past brw
        t3 = t3.split(",")  ###  past pur
        t4 = t4.split(",")  ###  intra brw

        t1 = [t for t in t1 if len(t) > 5]
        t2 = [t for t in t2 if t not in t1 and len(t) > 5]
        t3 = [t for t in t3 if t not in t2 and t not in  t1 and len(t) > 5]
        t4 = [t for t in t4 if t not in t3 and t not in  t2 and len(t) > 5]

        #### NG easyid_siid
        if ok_global is None:    ok_global = set()
        elif len(ok_global) < 1: ok_global = set()

        if ngs is None:    ngs = set()
        elif len(ngs) < 1: ngs = set()
        # else :              ngs = set(ngs.split(","))

        #### Item intra, item genre intra,  brw past, pur past
        ss = t1[:n1] + t2[:n2] + t3[:n3] + t4[:n4]
        ii = len(ss)
        ss = ",".join(ss) + ","

        # ii = 0
        for x in chain.from_iterable(zip_longest(t1[n1:], t2[n2:], t3[n3:], t4[n4:], )):
            # print("")
            if ii >= nmax: break
            if x is not None and x not in ngs and x not in ok_global and len(x)>8:
                ii = ii + 1
                ss = ss + x + ","

        if len(ss) < 8: return ""
        if ss[0] == "," : return ss[1:-1]
        return ss[:-1]



    def pd_siid_combine5b(tg, ngs=None, n1=1, n2=0, n3=0, n4=0, ok_global=None, nmax=51):
        """ Remove similar genres, remove NG
        """
        ng = len(tg)
        if ng < 1: return ""
        t1 = tg[0] if ng >=1 else []
        t2 = tg[1] if ng >=2 else []
        t3 = tg[2] if ng >=3 else []
        t4 = tg[3] if ng >=4 else []

        #### NG easyid_siid
        if ok_global is None:    ok_global = set()
        elif len(ok_global) < 1: ok_global = set()

        if ngs is None:    ngs = set()
        elif len(ngs) < 1: ngs = set()
        # else :              ngs = set(ngs.split(","))

        #### Item intra, item genre intra,  brw past, pur past
        n1 = 1 if ng == 0  else n1  ### prevent ',2342342' at start
        ss = t1[:n1] + t2[:n2] + t3[:n3] + t4[:n4]
        ii = len(ss)
        ss = ",".join(ss) + ","
        if len(ss) < 5 :  ss = ""

        # ii = 0
        for x in chain.from_iterable(zip_longest(t1[n1:], t2[n2:], t3[n3:], t4[n4:], )):
            # print("")
            if ii >= nmax: break
            # x = str(x)
            if x in ngs : continue
            if x in ok_global : continue
            if x is not None and len(x)>8 :  ### not none, ..
                ii = ii + 1
                ss = ss + x + ","

        if len(ss) < 8: return ""
        if ss[0] == "," : return ss[1:-1]
        return ss[:-1]


    def pd_siid_combine5a(tg, ngs=None, n1=0, n2=0, n3=0, n4=0, ok_global=None, nmax=51):
        """ Remove similar genres, remove NG
        """
        ng = len(tg)
        if ng < 1: return ""
        t1 = [t for t in tg[0] if len(t) > 5] if ng >=1 else []
        t2 = [t for t in tg[1] if t not in t1 and len(t) > 5]  if ng >=2 else []
        t3 = [t for t in tg[2] if t not in t2 and t not in  t1 and len(t) > 5]  if ng >=3 else []
        t4 = [t for t in tg[3] if t not in t3 and t not in  t2 and len(t) > 5]  if ng >=4 else []

        #### NG easyid_siid
        if ok_global is None:    ok_global = set()
        elif len(ok_global) < 1: ok_global = set()

        if ngs is None:    ngs = set()
        elif len(ngs) < 1: ngs = set()
        # else :              ngs = set(ngs.split(","))

        #### Item intra, item genre intra,  brw past, pur past
        ss = t1[:n1] + t2[:n2] + t3[:n3] + t4[:n4]
        ii = len(ss)
        ss = ",".join(ss) + ","

        # ii = 0
        for x in chain.from_iterable(zip_longest(t1[n1:], t2[n2:], t3[n3:], t4[n4:], )):
            # print("")
            if ii >= nmax: break
            if x is not None and x not in ngs and x not in ok_global and len(x)>8:
                ii = ii + 1
                ss = ss + x + ","

        if len(ss) < 8: return ""
        return ss[:-1]

    def pd_topk_count(df, cols):
        for ci in cols :
           df[ ci + '_n'] = df[ci].apply(lambda x : len(x.split(",") ) )
        return df


    def pd_siid_cleangenre(x, topk=8):
        #### Filter into same genre
        siids = x['topk'].split(",")

        try :
            g0 = str(x['genre_id'])
        except :
            g0 = str(db_ca_siid_genre.get(x['siid'], ""))

        if len(g0) < 6 :
            ss= "" ; ii = 0
            for sid in siids :
                g1 = str(db_ca_siid_genre.get(sid, ""))
                if len(g1)> 6 :
                     ii + ii + 1
                     ss = ss  + sid + ","
                if ii > topk: break
            return ss[:-1]

        ss= "" ; ii = 0
        for sid in siids :
            g1 = str(db_ca_siid_genre.get(sid,""))
            if g1 == g0 :
                 ii + ii + 1
                 ss = ss  + sid + ","
            if ii > topk: break

        return ss[:-1]


    def pd_siid_clean(siids):
        siids = [    t for t in      siids.split(",") if len(t)> 8 ]
        siids = siids[1:]   ### Remove the 1st self elements.
        siids = pd_siid_remove_ranid(siids,)
        # siids = pd_siid_remove_mona(siids,)   ### remove mona at Index creation
        return ",".join(siids)


    def pd_siid_getranid(siids):
        ll = { s: db_itemid_vran.get(s, -int(s.replace("_", ""))) for s in siids if len(s) > 0 }
        return ll


    def pd_siid_remove_ranid(siids,):
        """ remove ranid take a list.          """
        rdict  = pd_siid_getranid(siids )
        ranid0 = set()
        ss= []
        for siid in siids :
          ranid= rdict.get(siid, None)
          if ranid and not ranid in ranid0 :
               ranid0.add(ranid)
               ss.append(siid)
        return ss




##########################################################################################
##########################################################################################
if 'daily_eval':
    def db_load_dict2(dirin, colkey='siid',  cols=None):
      ####  dd[siid]['image_url']
      try :
          df = pd_read_file(dirin, cols=None)
          df = df.rename(columns= {'genre_id_path': 'genre_path',  'item_image_url':  'image_url'  })
          log(df.columns)

          if len(df) < 1:
               log('empty file') ; return {}

          if 'siid' not in df.columns :
              df['siid'] = df.apply(lambda x : f"{x['shop_id']}_{x['item_id']}", axis=1)

          if cols is None :
              cols = [ 'genre_path', 'image_url'    ]

          df = df.drop_duplicates('siid')
          df = df[['siid'] + cols]
          df = df.set_index(colkey).to_dict('index')
          return df
      except Exception as e :
         log(e) ; return {}

    def pd_stats_rank(dfa):
        """Compare Rec from couch, kev and user histo browsing Couch rec
            Avg Rank 1.2213831053131945  on Click Avg Rank 0.2642857142857143 on Pur

        # log( "352500_10012744,243930_10002234,352500_1001274".split(",").index(  "352500_10012744" ) )
        #log(  np_find( "352500_10012744"  ,  "352500_10012744,243930_10002234,352500_1001274".split(",")  ) )
        # return 1

        df['rank'] = df.apply(lambda x :  np_find(  f"{x['shop_id']}_{x['item_id']}"  , str(x['rec']).split(",")  ), axis=1)
        df['nrec'] = df.apply(lambda x :  len( str(x['rec']).split(",") ), axis=1)
        log(df[['easy_id', 'channel', 'rank',  'nrec' ]])

        log( 'Avg Rank', df[ df['rank'] > -1   ]['rank'].mean() )
        df.to_csv( dir_out + "/check_couch_clk.csv", index=False )
        """
        dfa['i_topk_hist'] = dfa.apply(lambda x :  len( set(x['hist']).intersection(set(x['topk'])) ) / len(x['hist']), axis=1  )
        dfa['i_rec_hist']  = dfa.apply(lambda x :  len( set(x['hist']).intersection(set(x['rec'])) )  / len(x['hist']), axis=1  )
        dfa['i_rec_topk']  = dfa.apply(lambda x :  len( set(x['rec']).intersection(set(x['topk'])) )  / len(x['rec']),  axis=1  )
        dfa = dfa.reset_index()

        return dfa

    def daily_check_genre_dist(tag='eod', nfile=1000):  ####  python prepro_prod.py daily_check_genre_dist  --mode all 2>&1 | tee -a zlog_check_genre2.py  &
        """  Check if the distance of genre is too high.
             editdistance.eval('banana', 'bahama')
        """
        tk =get_timekey()
        #tk = 1

        dirkev = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod"

        if 'sc'  in tag :   dirkev = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/sc_widget_pur"
        # if 'eod' in tag :  dirkev = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod"

        log('####### Waiting', "\n", dirkev,)
        os_wait_until(dirkev + "/*", ntry_max=400)


        flist = glob_glob(dirkev + "/*", nfile)
        im = imaster()
        key  = "genre_path"
        cols = ['siid', 'topk']

        topk1 = 2

        im2 = {}
        siidcur = set()
        jj = 0; fj = []
        for ii, fi in enumerate(flist) :
            fj.append(fi)
            if len(fj) < 10  and ii < len(flist) -1 : continue
            dirout = dirkev + f"_outlier/rec_siid_outlier_{ii+1}.parquet"
            # if os.path.isfile(dirout) : continue

            df = pd_read_file( fj, cols=cols, n_pool=20, nfile=1000 , verbose=0)
            df = df.drop_duplicates('siid')
            df = df[ -df['siid'].isin(siidcur)]
            fj = []; jj = 0
            # df = df0.iloc[:100, :]
            log(df)
            if len(df) < 1: continue


            #### Pre-fetched ids
            siids      = list(df.siid.values)
            df['topk'] = df['topk'].apply(lambda x: x.split(",")[:topk1] )
            siids.extend( df.explode('topk')['topk'].values )
            siids = set(siids).difference( siidcur )
            log('N siids to fetch: ', len(siids) )
            im1     = im.get_multi( siids, key, update_cache=False)
            log('Retrieved', len(im1), str(im1)[:100] )
            #im2 = {**im2, **im1 }
            siidcur.update( siids )

            #############################################################
            import editdistance as edit

            def genre_dist(x):
                g0 =  im.get( x['siid'], key)
                d  = 0
                for sid in x['topk'][:topk1]:
                    #gi = im1.get(sid, key)
                    gi = im1.get(sid, 10.0)  ### Retrieved fron Dict
                    d  = d + edit.eval( str(gi), str(g0) )
                return d / topk1

            df['genre_dist'] = df.apply(lambda x : genre_dist(x), axis=1)
            df = df.sort_values('genre_dist', ascending=0)
            df = df[ df.genre_dist > 20.0 ]
            del df['topk']

            pd_to_file(df, dirout , show=1)
        log("all finished")


    def daily_update_wrong_vect():                      ####  python prepro_prod.py daily_update_wrong_vect      2>&1 | tee -a zlog_wrong3.py &
        tk =get_timekey()-1
        #tk = 18940
        tk= "*"
        dirin = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/*_outlier/"

        # log('####### Waiting', "\n", dirkev,)
        # os_wait_until(dirin + "/*", ntry_max=400)
        model  = text_model_load(model0['dir'] )

        index = Index1(  dir_cpa3 + f"/hdfs/daily_usertopk/m001/done_outlier.txt"  )
        cols  = ['siid', 'genre_dist']
        flist = glob_glob(dirin + "/*.parquet", 70000)
        flist = [ fi for fi in flist if fi not in set(index.read() )]
        log(flist, )  ; time.sleep(10)
        flist = index.save_filter(flist)
        if len(flist) < 1: return 'no files'
        log('new', flist)

        df = pd_read_file(flist, cols=cols, n_pool= 20, nfile=10000, nrows=500500500)
        df = df.drop_duplicates('siid')
        df = df[df.genre_dist > 8.0 ]
        log(df.shape)
        df = pd_add_itemaster(df)    ### Too much ressources on Cass

        df['item_text'] = df.apply(lambda x : item_merge_field(x), axis=1)
        df['item_emb']  = text_tovec_batch(model, df['item_text'].values )
        df = df[[ 'siid', 'item_emb' ]]
        log(df)
        cass_update(df, table=model0['table'], prefix= model0['pref'], colkey="siid", colval="item_emb")


    ##########################################################################
    def pd_couch_load(df, tk, iscouch=True ):   ### tk-1
      if iscouch :
         dircouch = dir_cpa3 + f"/res/couch/rec/{tk}/*.parquet"
         log( dircouch )
         df2 = pd_read_file( dircouch , n_pool=20 )      ;   log('couch', df2, df2.columns)
         df  = df.merge(df2, on='easy_id', how='left')   ##### Already into List Format
         df['rec'] = df['rec'].fillna('')
         df = df[- (df['rec'].str.len() < 10) ]   ### only with Couch ones
         log(df.shape, df )
      else: df['rec'] = ""
      return df

    def pd_hist_load(df, dirhist, dirhist2="", col=None, remove_missing=False):
      col= 'hist' if col is None else col
      log(dirhist)
      leasy = list(df.easy_id.unique() )

      dfh = pd_read_file2( dirhist ,  n_pool=20, nfile= 10000 )       ;   log(dfh)
      dfh = dfh[dfh.easy_id.isin(leasy)]

      if dirhist2 != "":
          dfh2 = pd_read_file2( dirhist2 ,  n_pool=20, nfile= 10000 )       ;   log(dfh2)
          dfh2 = dfh2[dfh2.easy_id.isin(leasy)]
          dfh   = pd.concat((  dfh2, dfh))

      if len(dfh) < 1:
          df[col] = '' ; return df

      if 'ts' in dfh.columns :
          dfh = dfh.sort_values('ts')

      dfh['siid'] = dfh.apply(lambda x : siid(x), axis=1)
      dfh         = dfh[[ 'easy_id', 'siid' ]]
      dfh         = dfh.groupby("easy_id").apply( lambda x : ",".join(x['siid'] ) ).reset_index()
      dfh.columns = ['easy_id', col ]
      df      = df.merge(dfh, on='easy_id', how='left')
      df[col] = df[col].fillna('')

      if remove_missing:
         df = df[ -(df['hist'].str.len() < 10*2) ]  ### Remove missing history
      log(df, df.columns, df.shape )
      return df

    def imaster_get_data(dfa1, today):
      log("#### Get itemMaster Infos    ")
      dfb= list(  dfa1.explode('hist')['hist'].values )
      dfb.extend( dfa1.explode('hist_pur')['hist_pur'].values[:] )
      dfb.extend( dfa1.explode('topk')['topk'].values[:] )
      dfb.extend( dfa1.explode('rec')['rec'].values[:]   )
      dfb = list( set(dfb) )

      # today    = date_now_jp("%Y%m%d", timezone='jp', add_days= 0 )
      #dirindex = dir_ca + f"/daily/item_vec/ca_items_{today}/daily_item_clean.parquet"
      dirindex = dir_ca + f"/daily/item/ca_items2_{today}/clean/daily_*.parquet"
      im0      = db_load_dict2(dirindex, colkey='siid', cols=[ 'shop_id', 'item_id', 'genre_path', 'image_url'    ])
      dfb = [ t for t in dfb if t not in im0 ]
      dfb = [ t for t in dfb if len(t) > 8 ]

      #im1 = { t: db_imaster.get(t) for t in db if t in db_imaster }
      #dfb = [ t for t in dfb if t not in im1 ]

      dfb = pd.DataFrame( dfb, columns=['siid'] )  ###  'item_name',  'shop_name',  "genre_name_path",
      dfb = pd_add_itemaster(dfb, cols_cass= [  'price', 'genre_path', 'image_url'   ]  )  #### Extra infos.
      im  = dfb.set_index('siid').to_dict('index')  ### Item Master data
      ### Merge with existing
      im = {**im, **im0}
      log( 'imaster', len(im), str(im)[:100] )
      log('N ids', len(dfb))
      return im

    def daily_get_allhisto(add_days=0, tag="scpur", nfile=20,  mode='verbose', iscouch=False):   ####
        """  py  daily_check_html  --mode intra  --tag intra  --nfile 20   --add_days -1   >> "${logfile}_intra_html6.py"  2>&1   &
        """
        tk     = get_timekey() + add_days
        today  = date_now_jp("%Y%m%d", add_days= add_days)
        today1 = date_now_jp("%Y%m%d", add_days= add_days-1)
        today2 = date_now_jp("%Y%m%d", add_days= add_days-2)
        # tk = 18940
        iscouch=False ; dirout = None ; nhist_max = 70 ;  logic11 = ""; tag1= ""
        dirpur = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*purch*.parquet"
        dirhist2=""

        if  mode == 'export':
            iscouch = False
            # tag     = 'eod'
            nhist_max = 30
            dirkev   = dir_export + f"/{today}/eod"
            if tag == 'eod'     :   dirkev   = dir_export + f"/{today}/eod"
            if tag == 'intra'   :   dirkev   = dir_export + f"/{today}/intraday"
            dircouch = dir_cpa3   + f"/res/couch/rec/{tk-1}/*.parquet*"
            dirhist  = dir_cpa3   + f"/ca_check/stats/user_capur_brw/*{tk-1}*"
            dirout   = dir_export + f"/{today}/check/"
            nmax = 2*10**6 ; nmax2= 50000

        elif mode == 'intra':
            iscouch = False
            # tag     = 'eod'
            nhist_max = 30
            # dirkev   = dir_export + f"/{today}/intraday"
            dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/intraday/out/*pur*.parquet"

            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*browsing*.parquet"
            dirhist2 = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
            dirout   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/intra_eval"
            nmax = 2*10**4  ; nmax2= 500  ; nfile = 1000

        elif mode == 'abtest' :  ##ABtest:   18961,  20211130
            logic11  = '76ace6915d79b785'     ####  ca_logic_gethash(campaign_id="20211129", logic_id="kvn-20211130")
            tag1     = 'purkvn2'

            dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/ca_log/*zcheck*pur*"
            # dirkev   = dir_cpa3 + f"/res/couch/rec_capur/{tk-1}/*pur*kvn*.parquet"
            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            # dirhist  = brw_dir  + f"/*/*{tk-1}*.parquet"  ### too slow

            dirpur   = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*purch*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
            dirout   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/abtest_eval"
            nmax = 13*10**6  ; nmax2= 200000  ; nfile= 100

        elif mode == 'useremb' :  ##ABtest:   18961,  20211130
            # dirkev   = dir_cpa3 + f"/hdfs/daily_useremb/topk/{tk}/*.parquet"
            dirkev   = dir_cpa3 + f"/hdfs/daily_useremb/topk/{tk}/*.parquet"

            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            dirpur   = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*purch*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
            dirhist2 = dir_cpa3 + f"/hdfs/daily_user_hist/{today2}/*/*.parquet"

            dirout   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/useremb_eval"
            nmax = 13*10**6  ; nmax2= 200000  ; nfile= 20

        else :  ###3 EOD
            dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod"
            if tag == 'eod'   :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod"
            if tag == 'scpur' :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/sc_widget_pur"
            if tag == 'sclk'  :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/sc_widget_clk"
            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            dirhist  = dir_cpa3 + f"/ca_check/stats/user_capur_brw/*{tk-1}*"
            dirout   = dirkev + f"_eval/"

            # dirhist2 = dir_cpa3 + f"/hdfs/daily_user_hist/{today2}/*/*.parquet"
            # dirhist  = dir_cpa3 + f"/ca_check/stats/user_capur_brw/*{tk-1}*"
            dirout   = dirkev + f"_eval/"
            nmax = 13*10**6  ; nmax2= 500000  ; nfile= 100


        dirkev2 = dirkev + "/*.parquet" if "*" not in dirkev.split("/")[-1] else dirkev
        dirkev  = "/".join( dirkev2.split("/")[:-1] )
        group   = dirkev.split("/")[-1]
        log(dirkev, dirout, dircouch, dirhist) ; time.sleep(6)
        os_makedirs(dirout)

        if len(glob.glob( dirout + f"/ca_{mode}_{tag1}_*.parquet")) < 1  :
            ###### Topk Load   ######################################################
            # os_wait_until(dirkev2 , ntry_max=4000)
            flist = sorted( glob_glob( dirkev2 , nfile ) )
            flist = flist[-2:] if mode =='intra' else flist  ### Take most recent
            flist = flist[:nfile]
            log('File used', flist)

            df = pd_read_file( flist, n_pool=20, nfile= nfile )  ;  log(df)
            df = df.drop_duplicates('easy_id', keep='last')
            df = df.iloc[:nmax,:]
            df = pd_del(df, ['user_emb'])
            df = df.rename(columns= {'rec' : 'topk', 'topk_user': 'topk'})  ### Ab test results --topk
            if  'genre' in tag : df = df.rename(columns={'topk_genre': 'topk' })
            log(df, df.columns )

            log("#### Filter only logic", logic11 )
            if  logic11 != "" and 'logic_hash' in df.columns :  df= df[df.logic_hash == logic11 ]
            log(df.shape)

            log("#### Couch load  #################################################")
            df = pd_couch_load(df, tk, iscouch=iscouch )   ### tk-1

            log("#### Histo load  ################################################")
            df = pd_hist_load(df, dirhist, dirhist2, col='hist', remove_missing=True)

            log("#### Purchase load  #############################################")
            df = pd_hist_load(df, dirpur, col='hist_pur')

            ##### Clean  ###############################################################
            dfa = df.iloc[:nmax2, :]    ; del df
            dfa = dfa.drop_duplicates('easy_id', keep='last')
            # if mode != 'export': pd_to_file(dfa, dirout + f"/ca_{mode}_{tag1}_{tk}_merge.parquet")

        else :
            dfa = pd_read_file( dirout + f"/ca_*.parquet"  )
            if 'index' in dfa.columns:  del dfa['index']  ### create issue later


        log(dfa, dfa.columns, dfa.dtypes)
        if 'clean':
            for x in [ 'rec', 'topk', 'hist', 'hist_pur' ] :
               try :
                  dfa[ x ] = dfa[ x ].apply(lambda x:  "" if len(str(x)) < 10 else x )
                  dfa[ x ] = dfa[ x ].apply(lambda x: x.split(","))    ### x can already be  a numpy array
               except : pass

            dfa['n_hist'] = dfa['hist'].apply( lambda x : len(x)   )
            dfa           = dfa[ (dfa.n_hist < nhist_max) & ( dfa.n_hist >= 2 ) ]
            dfa           = dfa.sort_values('n_hist', ascending=0)
            # dfa = pd_stats_rank(dfa)

        log("#### Clean df   ###############################")
        dfa1 = dfa.iloc[:2000, :]
        dfa1 = dfa.reset_index()
        # dfa1 = dfa1.drop_duplicates('easy_id')
        if 'get_imaster':
            log("#### Get itemMaster Infos    ")
            im = imaster_get_data(dfa1, today)

        cc = Box({}) ; cc.dirout = dirout; cc.tag1= tag1    ; cc.group = group; cc.tk =tk; cc.time0 = time0()
        return dfa1, im, cc



    def daily_check_html(add_days=0, tag="scpur", nfile=20,  mode='eod', iscouch=True):   ####
        """  py  daily_check_html  --mode intra  --tag intra  --nfile 20   --add_days -1   >> "${logfile}_intra_html6.py"  2>&1   &

             py2  daily_check_html  --mode eod  --tag genre  --nfile 20   --add_days 0     >> "${logfile}_abtest_html5.py"  2>&1   &

             #### AB test export
             py2  daily_check_html  --mode abtest  --tag eod  --nfile 20   --add_days 0     >> "${logfile}_abtest_html3.py"  2>&1   &

             daily_check_html(add_days=0, tag="genre", nfile=20,  mode='eod', iscouch=False)

        """
        ### 'rec', 'topk', 'hist', ''hist_pur'
        dfa1, im, cc =daily_get_allhisto(add_days=add_days, tag=tag, nfile=nfile, mode=mode, iscouch=iscouch)
        ############################################################################
        def conv(ll, key):
            ll2 = [ im.get(siid, {}).get(key, "") for siid in ll  ]
            return ll2

        def conv2(ll, key):
            ll2 = [ f"<img width='60'  height='60' src='" + str(im.get(siid, {}).get(key, "")).split(" ")[0] + "' >"  for siid in ll  ]
            ll2 =  "<p>" + ",".join( ll2 ) + "</p>"
            return ll2

        log("##### Export HTML  #################################################")
        dfa1 = dfa1.rename(columns= {'shop_id_x': 'shop_id', 'item_id_x' : 'item_id'  })
        if 'shop_id' not in dfa1.columns:
            dfa1['shop_id'] = dfa1['hist'].apply(lambda x: to_int( x[0].split("_")[0])  )
            dfa1['item_id'] = dfa1['hist'].apply(lambda x: to_int( x[0].split("_")[1])  )

        log('using', dfa1)
        ######### image Export #####################################################
        doc = Doc()  ; i =0
        key = 'image_url'
        for j,x in dfa1.iterrows():
            topk = str(conv2( x['topk'][:50], key))
            if len(topk) < 100 : continue
            i  += 1
            sid = f"{x['shop_id']}_{x['item_id']}"
            # doc.h += "<p style='font-size:16px;'>"
            doc.add(  x['easy_id'])
            doc.add(  sid,  conv( [ sid ], 'genre_path' )  )
            doc.add(  conv2( [ sid ], key ))

            doc.add('hist',  conv2( x['hist'][:50], key) )
            if mode != 'export' : doc.add( 'Pur',  conv2( x['hist_pur'][:50], key) )
            doc.add('topk', topk )
            if iscouch : doc.add('couch',  conv2( x['rec'][:50],  key) )

            # doc.h += "<p style='font-size:16px;'>"
            doc.add( 'hist=',     x['hist'][:50] )
            doc.add( 'hist_pur=', x['hist_pur'][:50] )
            doc.add( 'topk=',     x['topk'][:50] )
            doc.add('')

            doc.add( 'easy_topgenre_intra= ', db_easyid_topgenre_intra.get( x['easy_id'], "") )
            doc.add( 'easy_topgenre_pur= ',   db_easyid_topgenre_pur.get( x['easy_id'], "") )
            doc.add( 'easy_topgenre_brw= ',   db_easyid_topgenre_brw.get( x['easy_id'], "") )
            doc.add('')

            doc.add( 'hist_genre', conv( x['hist'][:50],      'genre_path') )
            doc.add( 'pur_genre',  conv( x['hist_pur'][:50] , 'genre_path') )
            doc.add( 'topk_genre', conv( x['topk'][:50] ,     'genre_path') )

            doc.add("<hr>\n\n")
            if i > 1000 : break
        doc.save( cc.dirout + f'/zcheck_{mode}_{cc.tag1}_{cc.group}_rec_img_{cc.tk}_{cc.time0}.html')


    def daily_check_dist(add_days=0, tag="scpur", nfile=20,  mode='verbose', iscouch=False):   ####
        dfa1, im, cc= daily_get_allhisto(add_days=add_days, tag=tag, nfile=nfile,  mode=mode, iscouch=iscouch)   ####

        def conv(ll, key):
            ll2 = [ im.get(siid, {}).get(key, "") for siid in ll  ]
            return ll2

        def valid_dist_genre(x ):
          #g1 = set([ im.get(si)['genre_path']  for si in l1 ] )
          #g2 = set([ im.get(si)['genre_path']  for si in l2 ] )
          #l1 = set(x['topk'])
          #l2 = set(x['hist'].split(",")

          g1 = set(x['topk_genre'])
          g2 = set(x['hist_genre'])

          nsame     = len( g1.intersection(g2 ) )
          nsame_pct = nsame / len(g1)
          g3 = g2.difference(g1)
          return f"{nsame},{nsame_pct}"    #### nb of same, pct same


        def valid_dist_emb(x):
            #l1 = set(x['topk'])
            #l2 = set(x['hist'].split(",")
            g1 = set(x['topk_genre'])
            g2 = set(x['hist_genre'])

            slist1 = {}
            for gi,si in zip(g1,l1) :
              if gi in slist1: slist1[ gi ].append( si )
              else :  slist1[ gi ] = [ si ]

            slist2 = {}
            for gi,si in zip(g2,l2) :
              if gi in slist2: slist2[ gi ].append( si )
              else :  slist2[ gi ] = [ si ]


            l2l = []
            for gi  in glist :
              v1 = [ ivect.get(sid) for sid in slist1[gi]   ]
              v2 = [ ivect.get(sid) for sid in slist2[gi]   ]

            v1 = np.mean(v1, axis=0)
            v2 = np.mean(v2, axis=0)
            l2l.append( np.sum(v1-v2)**2 )
            return np.mean(l2l)


        dfa1['topk_genre'] = dfa1['topk'].apply(lambda x :  conv(x.split(","), key='genre_path') )
        dfa1['hist_genre'] = dfa1['hist'].apply(lambda x :  conv(x.split(","), key='genre_path') )


        ###  [ 'rec', 'topk', 'hist', 'hist_pur' ]
        dfa1['dist_genre'] = dfa1.apply(lambda x : valid_dist_genre(x ), axis=1  )
        dfa1['dist_emb']   = dfa1.apply(lambda x : valid_dist_emb(x), axis=1  )
        pd_to_file( cc.dirout +"/dist_genre_embd.parquet")




    ##########################################################################
    def pd_remove(df, cols):
        cols2 = [ ci for ci in df.columns if ci not in cols ]
        return df[cols2]

    def np_find(x, v):
        try :    return v.index(x)
        except : return -1


    def itemrec_getrec_k8s(path):
        dir_in = "/a/adigcb201/ipsvols03/ndata/cpa/res/couch/*full*82*simple*"
        df     = pd_read_file(dir_in, n_pool=10, verbose=True)
        df     = df.drop_duplicates('easy_id')
        # log('k8s', df) ; time.sleep(3)
        keys = df['easy_id'].values
        vals = df['rec'].values
        del df
        dd = {}
        for i in range(len(keys)):
            dd[ int(keys[i]) ] = vals[i]
        return dd


    def easyid_create_histo2(verbose=1):   ## py easyid_create_histo
        """

              Setup with recent Data for all data, 3months

              Daily Hadoop batch  + extraction on disk

               easyid, shopid, itemid, vran, event_type

            easyid --> siid list  (after last purchase)
            easyid --> vran de-duplicated.

            nono3.ichiba_hist_202101_20210812b
            JOIN with VRAN and distinct

            20210702 (13 027 042, 9)
            20210814 (1092611, 9)

            DO it in SQL last 10 clicks per easyid

        """
        nmax = 10000000000
        from collections import OrderedDict
        dir_in =  dir_hive  + "/ichiba_hist_202107_20210812b_vran/daily/"

        def get_siid_list(dfi):
            pur_ts = dfi[dfi.event_type ==2 ]['ts'].values
            ts1    = 0.0
            if len(pur_ts) > 0 :
               ts1 = pur_ts[-1]
            dfi = dfi[dfi.ts >= ts1] ### After last purchase
            dfi = list(OrderedDict.fromkeys(dfi['siid'].values))    ## unique siid
            return dfi

        flist = sorted(glob.glob(dir_in + "/*"))
        # fi  = flist[-10]   ### be careful of the date , last one is only purchases
        cols  = [ 'easy_id', 'shop_id', 'item_id', 'ts', 'event_type' ]
        #### Most Recent up to 15 days
        for i in range(14, 0, -1) :
            fi = flist[i]
            log(fi)
            df = pd_read_file(fi, npool=1, cols=cols)
            df = df.iloc[:500, :]
            df['easy_id'] = df['easy_id'].astype('int')
            df = df.sort_values(['easy_id', 'ts'])

            df = pd_easyid_flatten_siid(df, get_siid_list)
            log('N easy_id',  len(df) )
            # log(df)

            #### Filter no Empty
            db_path = db['db_easyid_hist_path']  + 'test'
            db_easyid_hist2 = diskcache_load( db_path )
            db_easyid_hist_keys = diskcache_getkeys( db_easyid_hist2 )
            if verbose : log( 'keys', len( db_easyid_hist_keys  ), str(db_easyid_hist_keys)[:50] )
            df         = df[ df['siid_list'].apply(lambda x : len(x)> 0 ) ]
            df         = df[ df['easy_id'].apply(  lambda x :  True if x not in db_easyid_hist_keys else False  ) ]
            log('N easy_id New',  df.easy_id.nunique() )

            if verbose > 0 : log(df)

            #### Insert missing ones
            if len(df) > 0 :
              diskcache_save(df, colkey='easy_id', colvalue='siid_list', db_path=db_path)


    def save(x, name):
       from utilmy import save as save2
       dirout= "/data/workspaces/takos01/cache/"
       save2(x, dirout + "/" + name + ".pkl" )

    def load(x, name):
       from utilmy import load as load2
       dirout= "/data/workspaces/takos01/cache/"
       return load2(dirout + "/" + name + ".pkl" )


    class Doc(object):
        def  __init__(self, ):
            self.h= "<html><body>\n"
            self.h = """
            <!DOCTYPE html>
            <html><head><style>
            .hide {  display: none;}
            .show1:hover + .hide {
              display: block;
              color: red;
            }
            </style></head><body>
            """

        def  add(self, *s):
            self.h += " ".join( [str(x) for  x in s] ) + "<br>\n"

        def  add2(self, *s):
            s = " ".join( [str(x) for  x in s] ) + "<br>\n"
            s = '<div class="show1">Show</div><div class="hide">' + s +'</div><br>'
            self.h += s

        def save(self, file):
            os_makedirs(file)
            log(file)
            with open(file, mode='w') as fp:
                fp.write(self.h + "\n\n</body></html>\n\n")




if 'old':
    def zz_pd_hist_load2(df, dirhist, dirhist2="", col=None):
        col= 'hist' if col is None else col
        log(dirhist)
        leasy = list(df.easy_id.unique() )

        dfh = pd_read_file2( dirhist ,  n_pool=20, nfile= 10000 )       ;   log(dfh)
        dfh   = dfh[dfh.easy_id.isin(leasy)]

        if dirhist2 != "":
            dfh2 = pd_read_file2( dirhist2 ,  n_pool=20, nfile= 10000 )       ;   log(dfh2)
            dfh2 = dfh2[dfh2.easy_id.isin(leasy)]
            dfh   = pd.concat((  dfh2, dfh))

        if len(dfh) < 1:
            df[col] = '' ; return df

        if 'ts' in dfh.columns :
            dfh = dfh.sort_values('ts')

        dfh['siid'] = dfh.apply(lambda x : siid(x), axis=1)
        dfh         = dfh[[ 'easy_id', 'siid' ]]
        dfh         = dfh.groupby("easy_id").apply( lambda x : ",".join(x['siid'] ) ).reset_index()
        dfh.columns = ['easy_id', col ]
        df      = df.merge(dfh, on='easy_id', how='left')
        df[col] = df[col].fillna('')
        log(df.columns, df.shape )
        return df



    def daily_check_html2(add_days=0, tag="scpur", nfile=20,  mode='verbose'):   ####
        """  py  daily_check_html  --mode intra  --tag intra  --nfile 20   --add_days -1   >> "${logfile}_intra_html6.py"  2>&1   &

             py2  daily_check_html  --mode eod  --tag genre  --nfile 20   --add_days 0     >> "${logfile}_abtest_html4.py"  2>&1   &

             #### AB test export
             py2  daily_check_html  --mode abtest  --tag eod  --nfile 20   --add_days 0     >> "${logfile}_abtest_html3.py"  2>&1   &

        """
        tk     = get_timekey() + add_days
        today  = date_now_jp("%Y%m%d", add_days= add_days)
        today1 = date_now_jp("%Y%m%d", add_days= add_days-1)
        today2 = date_now_jp("%Y%m%d", add_days= add_days-2)
        # tk = 18940
        iscouch=True ; dirout = None ; nhist_max = 70 ;  logic11 = ""; tag1= ""
        dirpur = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*purch*.parquet"
        dirhist2=""

        if  mode == 'export':
            iscouch = False
            # tag     = 'eod'
            nhist_max = 30
            dirkev   = dir_export + f"/{today}/eod"
            if tag == 'eod'     :   dirkev   = dir_export + f"/{today}/eod"
            if tag == 'intra'   :   dirkev   = dir_export + f"/{today}/intraday"
            dircouch = dir_cpa3   + f"/res/couch/rec/{tk-1}/*.parquet*"
            dirhist  = dir_cpa3   + f"/ca_check/stats/user_capur_brw/*{tk-1}*"
            dirout   = dir_export + f"/{today}/check/"
            nmax = 2*10**6 ; nmax2= 50000

        elif mode == 'intra':
            iscouch = False
            # tag     = 'eod'
            nhist_max = 30
            # dirkev   = dir_export + f"/{today}/intraday"
            dirkev   = dir_cpa3 + f"/hdfs/intraday/sc_stream_usertopk/{today}/*purchase*.parquet"

            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*browsing*.parquet"
            dirhist2 = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
            dirout   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/intra_eval"
            nmax = 2*10**4  ; nmax2= 500  ; nfile = 1000

        elif mode == 'abtest' :  ##ABtest:   18961,  20211130
            logic11  = '76ace6915d79b785'     ####  ca_logic_gethash(campaign_id="20211129", logic_id="kvn-20211130")
            tag1     = 'purkvn2'

            dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/ca_log/*zcheck*pur*"
            # dirkev   = dir_cpa3 + f"/res/couch/rec_capur/{tk-1}/*pur*kvn*.parquet"
            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            # dirhist  = brw_dir  + f"/*/*{tk-1}*.parquet"  ### too slow

            dirpur   = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*purch*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
            dirout   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/abtest_eval"
            nmax = 13*10**6  ; nmax2= 200000  ; nfile= 100

        elif mode == 'useremb' :  ##ABtest:   18961,  20211130
            # dirkev   = dir_cpa3 + f"/hdfs/daily_useremb/topk/{tk}/*.parquet"
            dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod_genre/*.parquet"

            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            dirpur   = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*purch*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"
            dirhist2 = dir_cpa3 + f"/hdfs/daily_user_hist/{today2}/*/*.parquet"

            dirout   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/eodgenre_eval"
            nmax = 13*10**6  ; nmax2= 200000  ; nfile= 20

        else :  ###3 EOD
            dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod"
            if tag == 'eod'   :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod"
            if tag == 'scpur' :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/sc_widget_pur"
            if tag == 'sclk'  :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/sc_widget_clk"
            if tag == 'genre' :  dirkev   = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod_genre"

            dircouch = dir_cpa3 + f"/res/couch/rec/{tk-1}/*.parquet"
            dirhist  = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today}/*browsing*.parquet"
            dirhist2 = dir_cpa3 + f"/hdfs/intraday/sc_stream/{today1}/*browsing*.parquet"

            # dirhist2 = dir_cpa3 + f"/hdfs/daily_user_hist/{today2}/*/*.parquet"
            # dirhist  = dir_cpa3 + f"/ca_check/stats/user_capur_brw/*{tk-1}*"
            dirout   = dirkev + f"_eval/"
            nmax = 13*10**6  ; nmax2= 500000  ; nfile= 100


        dirkev2 = dirkev + "/*.parquet" if "*" not in dirkev.split("/")[-1] else dirkev
        dirkev  = "/".join( dirkev2.split("/")[:-1] )
        group   = dirkev.split("/")[-1]
        log(dirkev, dirout, dircouch, dirhist) ; time.sleep(6)
        os_makedirs(dirout)


        if len(glob.glob( dirout + f"/ca_{mode}_{tag1}_*.parquet")) < 1  :
            ###### Topk Load   ######################################################
            # os_wait_until(dirkev2 , ntry_max=4000)
            flist = sorted( glob_glob( dirkev2 , nfile ) )
            flist = flist[-2:] if mode =='intra' else flist  ### Take most recent
            flist = flist[:nfile]
            log('File used', flist)

            df = pd_read_file( flist, n_pool=20, nfile= nfile )  ;  log(df)
            df = df.drop_duplicates('easy_id', keep='last')
            df = df.iloc[:nmax,:]

            if  'genre' in tag : df = df.rename(columns={'topk_genre': 'topk' })
            df = df.rename(columns= {'rec' : 'topk', 'topk_user': 'topk',})  ### Ab test results --topk
            log(df, df.columns )

            log("#### Filter only logic", logic11 )
            if  logic11 != "" and 'logic_hash' in df.columns :  df= df[df.logic_hash == logic11 ]
            log(df.shape)

            log("#### Couch load  #################################################")
            if iscouch :  ####  channel    easy_id   item_id  ... shop_id            sid   timestamp
               log( dircouch )
               df2 = pd_read_file( dircouch , n_pool=20 )      ;   log('couch', df2, df2.columns)
               df  = df.merge(df2, on='easy_id', how='left')   ##### Already into List Format
               df['rec'] = df['rec'].fillna('')
               df = df[- (df['rec'].str.len() < 10) ]   ### only with Couch ones
               del df2
            else: df['rec'] = ""
            log(df.shape, df )


            log("#### Histo load  ################################################")
            # os_wait_until(dirhist + "", ntry_max=4000)
            df = pd_hist_load(df, dirhist, dirhist2, col='hist')
            df = df[ -(df['hist'].str.len() < 10*2) ]  ### Remove missing history
            log(df)


            log("#### Purchase load  #############################################")
            df     = pd_hist_load(df, dirpur, col='hist_pur')
            log(df.shape)


            ##### Clean  ###############################################################
            dfa = df.iloc[:nmax2, :]    ; del df
            dfa = dfa.drop_duplicates('easy_id', keep='last')
            # if mode != 'export': pd_to_file(dfa, dirout + f"/ca_{mode}_{tag1}_{tk}_merge.parquet")

        else :
            dfa = pd_read_file( dirout + f"/ca_*.parquet"  )
            if 'index' in dfa.columns:  del dfa['index']  ### create issue later


        log(dfa, dfa.columns, dfa.dtypes)
        if 'stats':
            for x in [ 'rec', 'topk', 'hist', 'hist_pur' ] :
               try :
                  dfa[ x ] = dfa[ x ].apply(lambda x:  "" if len(str(x)) < 10 else x )
                  dfa[ x ] = dfa[ x ].apply(lambda x: x.split(","))    ### x can already be  a numpy array
               except : pass

            dfa['n_hist'] = dfa['hist'].apply( lambda x : len(x)   )
            dfa           = dfa[ (dfa.n_hist < nhist_max) & ( dfa.n_hist >= 2 ) ]
            dfa           = dfa.sort_values('n_hist', ascending=0)
            dfa = pd_stats_rank(dfa)


        if 'get_imaster':
            log("#### Clean df   ###############################")
            dfa1 = dfa.reset_index()
            # dfa1 = dfa1.drop_duplicates('easy_id')
            dfa1 = dfa1.iloc[:2000, :]

            log("#### Get itemMaster Infos    ")
            dfb= list(  dfa1.explode('hist')['hist'].values )
            dfb.extend( dfa1.explode('hist_pur')['hist_pur'].values[:] )
            dfb.extend( dfa1.explode('topk')['topk'].values[:] )
            dfb.extend( dfa1.explode('rec')['rec'].values[:]   )
            dfb = list( set(dfb) )

            # today    = date_now_jp("%Y%m%d", timezone='jp', add_days= 0 )
            #dirindex = dir_ca + f"/daily/item_vec/ca_items_{today}/daily_item_clean.parquet"
            dirindex = dir_ca + f"/daily/item/ca_items2_{today}/clean/daily_*.parquet"
            im0      = db_load_dict2(dirindex, colkey='siid', cols=[ 'shop_id', 'item_id', 'genre_path', 'image_url'    ])
            dfb = [ t for t in dfb if t not in im0 ]
            dfb = [ t for t in dfb if len(t) > 8 ]

            #im1 = { t: db_imaster.get(t) for t in db if t in db_imaster }
            #dfb = [ t for t in dfb if t not in im1 ]

            dfb = pd.DataFrame( dfb, columns=['siid'] )  ###  'item_name',  'shop_name',  "genre_name_path",
            dfb = pd_add_itemaster(dfb, cols_cass= [  'price', 'genre_path', 'image_url'   ]  )  #### Extra infos.
            im  = dfb.set_index('siid').to_dict('index')  ### Item Master data
            ### Merge with existing
            im = {**im, **im0}
            log( 'imaster', len(im), str(im)[:100] )
            log('N ids', len(dfb))


        ############################################################################
        def conv(ll, key):
            ll2 = [ im.get(siid, {}).get(key, "") for siid in ll  ]
            return ll2

        def conv2(ll, key):
            ll2 = [ f"<img width='60'  height='60' src='" + str(im.get(siid, {}).get(key, "")).split(" ")[0] + "' >"  for siid in ll  ]
            ll2 =  "<p>" + ",".join( ll2 ) + "</p>"
            return ll2

        log("##### Export HTML  #################################################")
        dfa1 = dfa1.rename(columns= {'shop_id_x': 'shop_id', 'item_id_x' : 'item_id'  })
        if 'shop_id' not in dfa1.columns:
            dfa1['shop_id'] = dfa1['hist'].apply(lambda x: to_int( x[0].split("_")[0])  )
            dfa1['item_id'] = dfa1['hist'].apply(lambda x: to_int( x[0].split("_")[1])  )

        log('using', dfa1)
        ######### image Export #####################################################
        doc = Doc()  ; i =0
        key = 'image_url'
        for j,x in dfa1.iterrows():
            topk = str(conv2( x['topk'][:50], key))
            if len(topk) < 100 : continue
            i  += 1
            sid = f"{x['shop_id']}_{x['item_id']}"
            # doc.h += "<p style='font-size:16px;'>"
            doc.add(  x['easy_id'])
            doc.add(  sid,  conv( [ sid ], 'genre_path' )  )
            doc.add(  conv2( [ sid ], key ))

            doc.add('hist',  conv2( x['hist'][:50], key) )
            if mode != 'export' : doc.add( 'Pur',  conv2( x['hist_pur'][:50], key) )
            doc.add('topk', topk )
            if iscouch : doc.add('couch',  conv2( x['rec'][:50],  key) )

            # doc.h += "<p style='font-size:16px;'>"
            doc.add( 'hist=',     x['hist'][:50] )
            doc.add( 'hist_pur=', x['hist_pur'][:50] )
            doc.add( 'topk=',     x['topk'][:50] )
            doc.add('')

            doc.add( 'easy_topgenre_intra= ', db_easyid_topgenre_intra.get( x['easy_id'], "") )
            doc.add( 'easy_topgenre_pur= ',   db_easyid_topgenre_pur.get( x['easy_id'], "") )
            doc.add( 'easy_topgenre_brw= ',   db_easyid_topgenre_brw.get( x['easy_id'], "") )
            doc.add('')

            doc.add( 'hist_genre', conv( x['hist'][:50],      'genre_path') )
            doc.add( 'pur_genre',  conv( x['hist_pur'][:50] , 'genre_path') )
            doc.add( 'topk_genre', conv( x['topk'][:50] ,     'genre_path') )

            doc.add("<hr>\n\n")
            if i > 1000 : break
        doc.save( dirout + f'/zcheck_{mode}_{tag1}_{group}_rec_img_{tk}_{time0()}.html')




if 'ab_test':
    def ab_extract_siid(tag='couch', add_days=-1): #### python prepro_prod.py   ab_extract_siid --tag couch    >> "${logfile}_abtest_siid_couch1.py"  2>&1  &
        tk    = get_timekey() + add_days
        today = date_now_jp(add_days= add_days)

        if 'couch' in tag :  dirin = dir_cpa3 + f"/res/couch/rec/{tk}/*.parquet"
        if 'eod' in tag :    dirin = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod/*.parquet"

        dirout = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/abtest/"
        log(dirin, dirout)

        flist = glob_glob(dirin, 300)
        ddict = {}
        def ddict_update(v):
          for rank, sid in enumerate( v.split(",")[:20] ) :
             if sid in ddict :
               ddict[sid][0] += 1      ### Nuser
               ddict[sid][1] += rank   ### Rank
             else :
               ddict[sid] = [1, rank]

        for fi in flist :
           df = pd_read_file(fi)
           if len(df) < 1: continue
           # df = df.iloc[:10, :]
           df = df.rename(columns={'rec': 'topk'}  )
           # log(df['topk'])
           df = df[df.topk.str.len() > 10 ]
           log(df.shape, fi )
           df['topk'].apply(lambda v: ddict_update(v))

        df1 = pd.DataFrame().from_dict(ddict, orient='index').reset_index()
        df1.columns     = ['siid','nuser', 'rank']
        df1['rank_avg'] =  df1['rank'] / df1['nuser']
        df1 = df1.sort_values('nuser', ascending=0)
        del df1['rank']

        log("Nusers", df1['nuser'].sum(),    df1['rank_avg'].mean(),    )
        pd_to_file(df1, dirout + f"/siid_stats_{tk}_{today}_{tag}_top20.parquet", show=1)


    def ab_extract_stats(tag='couch', add_days=-1): #### python prepro_prod.py   ab_extract_stats  add_days -1  >> "${logfile}_abtest_stats5.py"  2>&1  &
        """  python prepro_prod.py   ab_extract_siid --tag couch  --add_days 0    >> "${logfile}_abtest_siid_couch1.py"  2>&1  &
             python prepro_prod.py   ab_extract_siid --tag eod    --add_days 0    >> "${logfile}_abtest_siid_couch1.py"  2>&1  &

            sleep 2000 && python prepro_prod.py   ab_extract_stats  add_days -1  >> "${logfile}_abtest_stats5.py"  2>&1  &


        """
        tk     = get_timekey() + add_days
        today  = date_now_jp(add_days= add_days)
        dirout = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/abtest/"
        log( dirout)  ;#  time.sleep(4)

        log("\n########  CA Item list")
        df    = pd_read_file(dir_cpa3 + f"/ca_check//daily/item/ca_items2_{today}/raw3am/*.parquet")
        df    = df[df.weekly_campaign_id == df.weekly_campaign_id.values[0] ]
        df['siid'] = df.apply(lambda x : siid(x) , axis=1 )
        df = df.drop_duplicates('siid')
        log( df.columns,"\n", df )


        log("\n#######  topk eod")
        tag = 'eod'
        dfrec = pd_read_file( dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}*/abtest/*{tag}*top20*.parquet" )
        log( dfrec.columns,"\n", dfrec )
        df    = df.merge(dfrec, on ='siid', how='left', suffixes=(None, "_"+tag) ) ; log('\ndfrec', dfrec)


        log("\n#######  topk couch")
        tag = 'couch'
        dfrec = pd_read_file( dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/abtest/*{tag}*top20*.parquet" )
        log( dfrec.columns,"\n", dfrec )
        df    = df.merge(dfrec, on ='siid', how='left', suffixes=(None, "_"+tag) ) ; log('\ndfrec', dfrec)


        log("#########  Item list")
        dfkev = pd_read_file(dir_cpa3 + f"/ca_check//daily/item/ca_items2_{today}/map_ca_genre_siid/map_ca_genre_siid.parquet")
        log(dfkev.columns, dfkev)
        cols  = [ t for t in dfkev.columns if t not in df.columns ]
        df    = df.merge(dfkev[['siid'] + cols  ], on='siid', how='left', suffixes=(None, "_ok" )  ) ; del dfkev

        try :
            log("###### Pur all ")
            tag = 'puran'
            dfpur = pd_read_file(dir_cpa3 + f"/hdfs/items_pur/*{tk}*.parquet" )
            log( dfpur.columns,"\n", dfpur )
            df    = df.merge(dfpur, on='siid', how='left',  suffixes=(None, "_"+ "all" )  ) ; log(tag,'\n', dfpur) ; del dfpur

            log("###### Pur CA ")
            tag   = "ca"
            dfpur = pd_purca_stats(tk)
            df    = df.merge(dfpur, on ='siid', how='left', suffixes=(None, "_"+tag) ) ; log("\n\n", tag, '\n', dfpur)

            df    = df.sort_values(['gms'], ascending=[0])
            cols  = ['siid', 'nuser', 'rank_avg', 'nuser_couch', 'rank_avg_couch', 'score', 'score_clk', 'n_pur_all',   'n_pur_ca', 'n_pur_t1',  'n_clk_t1' ]
            log(df[ cols ], df.columns)
            cols1 = list(set(df.columns))
            df    = df[ cols + [ t for t in cols1 if t not in cols]  ]
            df2   = df[cols]

        except :
            cols1 = list(set(df.columns))
            df    = df[cols1]
            cols2 = [ 'siid', 'nuser', 'rank_avg', 'nuser_couch', 'rank_avg_couch', 'score', 'score_clk',  'n_pur_t1', 'n_clk_t1'  ]
            df2   = df[cols2]

        df2 = df2[ df2.n_clk_t1 > 0.0 ]
        df2 = df2.sort_values('score', ascending=0)
        #### Export
        pd_to_file(df,  dirout + f"/merge_siid_stats_{tk}_{today}_all_{int(time.time())}.parquet", show=0)
        pd_to_file(df2, dirout + f"/merge_siid_stats_{tk}_{today}_all_{int(time.time())}.csv",     show=0, index=False)


    def pd_purca_stats(tk):
        ## ['channel', 'discount', 'easy_id', 'item_id', 'logic_hash', 'price', 'shop_id', 'timestamp', 'units' ]
        log('loading sc_pur_dir')
        flist = glob_glob(dirs.sc_pur_dir  + f"/*/*{tk}*.parquet" , 1000 )
        df = pd_read_file( flist , n_pool=25, nfile=1000)
        # log(df)
        df['siid'] = df.apply(lambda x : siid(x), axis=1)
        df['gms']  = df['price'] * df['units']
        df = df.groupby('siid').agg({ 'item_id': 'count',   'gms':'sum',  'discount':'mean', 'price':'mean' }).reset_index()
        df.columns = [ 'siid', 'n_pur', 'gms',  'discount_pur', 'price_pur' ]
        log(df)
        return df


    def ab_extract_log(mode='imp,clk,pur', tk = 18962):   #### python prepro_prod.py   ab_extract_log --mode imp,pur,clk     >> "${logfile}_abtest_export.py"  2>&1   &
        ### Export Daily CA Into specific folder,, avg rank of 6
        ### tk     = 18962
        ab_list =[ # [  18983 ,'20211220',  'kvn-20211222' ],   ### CTR 1.83, 0.5 CVR
                   [ 18983, 18984 ,'20211220',  'kvn-20211222' ],   ### CTR 1.83, 0.5 CVR
                 ]

        for aa in ab_list :
            tk0 = aa[0]
            tk  = aa[1]
            logic1 = ca_logic_gethash(campaign_id= aa[2],  logic_id= aa[3])
            dirab  = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk0}_abtest/ab_log/"
            log(logic1)

            dfa1 = None
            if 'imp' in mode :
                flist = glob_glob( sc_imp_dir + f"/*/*{tk}*.parquet" )
                dfa = pd.DataFrame()
                for fi in flist:
                    log(fi)
                    dfi = pd_read_file(fi)
                    dfi = dfi[dfi.logic_hash == logic1 ]
                    dfa = pd.concat((dfa, dfi))
                    log(dfa.shape)

                pd_to_file(dfa,  dirab + f"imp_topk_{tk}.parquet")
                pd_to_file( dfa.drop_duplicates('easy_id')    , dirab + f"imp_easyid_{tk}.parquet" )

                ### De-duplicates easyid, topk
                dfa['siid'] = dfa.apply(lambda x : siid(x), axis=1 )
                cols = [ 'easy_id',   'sid', 'timestamp']
                dfa1 = dfa.groupby(cols).apply(lambda dfi : ",".join(dfi['siid'].values)).reset_index()
                dfa1.columns = cols + ['topk']
                dfa1 = dfa1.groupby(['easy_id', 'topk']).agg({'sid':'count'}).reset_index()
                dfa1 = dfa1.sort_values([ 'sid'], ascending=0 )
                dfa1 = dfa1.drop_duplicates('easy_id', keep='first')
                pd_to_file(dfa1,  dirab + f"imp_topk_{tk}_v2.parquet")


            if dfa1 is None :
                dfa1 = pd_read_file( dirab + f"imp_topk_{tk}_v2.parquet" )

            if 'clk' in mode : ### click + topk
                dfc = pd_read_file( sc_clk_dir + f"/*/*{tk}*.parquet",  n_pool=20  )
                dfc = dfc[dfc.logic_hash == logic1 ]

                dfc = dfc.merge(dfa1, on='easy_id', how='left')
                dfc['siid'] = dfc.apply(lambda x : siid(x), axis=1 )
                dfc['rank'] = dfc.apply(lambda x :  np_find(  x['siid']  , str(x['topk']).split(",")  ), axis=1)
                # dfc = dfc.sort_values([ 'sid_y'], ascending=0 )

                pd_to_file(dfc,  dirab + f"zcheck_clk_topk_{tk}.parquet", show=1)
                log('avg rank', dfc['rank'].mean()  )


            if 'pur' in mode :  # #### Average Positin 6 for click
                ### click + topk
                dfc = pd_read_file(sc_pur_dir + f"/*/*{tk}*.parquet", n_pool=20 )
                dfc = dfc[dfc.logic_hash == logic1 ]
                dfc = dfc.merge(dfa1, on='easy_id', how='left')
                dfc['siid'] = dfc.apply(lambda x : siid(x), axis=1 )
                dfc['rank'] = dfc.apply(lambda x :  np_find(  x['siid']  , str(x['topk']).split(",")  ), axis=1)

                # dfc = dfc.sort_values([ 'sid_y'], ascending=0 )

                pd_to_file(dfc,  dirab + f"zcheck_pur_topk_{tk}.parquet", show=1)
                log('avg rank', dfc['rank'].mean()  )


    def pd_hist_groupby(df, colid='easy_id', tag = "pur", tostr=True):

       df = pd_add_siid(df)

       if tostr:
          df1         = df.groupby(colid).apply(lambda dfi : ",".join(dfi['siid'])  ).reset_index()
       else:
          df1         = df.groupby(colid).apply(lambda dfi : dfi['siid'].values  ).reset_index()

       df1.columns = [colid, 'hist' + tag ]
       return df1



    def user_get_histo(easy, add_days=0  ) :
        tk = get_timekey() + add_days
        # bk  = easy % 500
        bk = easyid_bk(easy_id)

        log('brw')
        df1 = pd_read_file( dir_cpa3 + f"/hdfs/daily_user_eod/{tk-1}/sc_stream_intra/*_{bk}_*.parquet" )
        df1 = df1[df1.easy_id == easy] ; log(df1)
        df1 = pd_hist_groupby(df1, colid='easy_id', tag = "", tostr=False)

        log('pur')
        df2 = pd_read_file( dir_cpa3 + f"/hdfs/daily_user_hist/{tk-2}/pur_ran/*_{bk}_*.parquet" )
        df2 = df2[df2.easy_id == easy] ; log(df2)
        df2 = pd_hist_groupby(df2, colid='easy_id', tag = "_pur", tostr=False)
        # log(df1, df2)
        df1 = df1.merge(df2, on='easy_id', how='left')

        log('\neod rec')
        df2 = pd_read_file( dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/daily_user_eod/*_{bk}_*.parquet" )
        df2 = df2[df2.easy_id == easy] ; log(df2)

        for ci in [ 'topk', 'topk_emb', 'topk_intra','topk_brw', 'topk_pur',   ]:
           df2[ci] = df2[ci].apply(lambda x: x.split(","))  ### Lready a list
        df1 = df1.merge(df2, on='easy_id', how='left')


        log('\neod rec Couch', bk)
        df2 = pd_read_file( dir_cpa3 + f"/res/couch/rec/{tk-1}/*_{bk}_*.parquet"  )
        if len(df2) > 1:
            df2 = df2[df2.easy_id == easy].drop_duplicates('easy_id') ; log(df2)
            df2 = df2.rename(columns={'rec': 'topk_couch'})
            for ci in [ 'topk_couch',    ]:
               df2[ci] = df2[ci].apply(lambda x: x.split(","))  ### Lready a list
            df1 = df1.merge(df2, on='easy_id', how='left')
        else :
            df1['topk_couch'] = df1.apply(lambda x: [], axis=1)

        siids = []
        for ci in [ 'hist', 'hist_pur', 'topk', 'topk_emb', 'topk_intra', 'topk_brw',  'topk_couch', ]:
           siids.extend( df1[ci].explode().values )
        siids = list(set(siids))
        log('N siid', len(siids) )
        # siids = siids[:2]

        im1    = imaster(use_file="today_ca")
        imdict = im1.get_multi(siids, update_cache=True)  ### Fetch the dict
        log('N siid retrieved', len(imdict) )

        for ci in [ 'hist', 'hist_pur', 'topk',  'topk_emb', 'topk_intra', 'topk_brw', 'topk_couch'   ]:
           df1[ci +"_genre"] = df1[ci].apply(lambda x : [ imdict.get(t,{}).get('genre_path')    for t in  x ])
           df1[ci +"_image"] = df1[ci].apply(lambda x : [ imdict.get(t,{}).get('image_url', "").split(",")[0] for t in  x ])
           # df1[ci] = df1[ci].apply(lambda x : ",".join(x))

        user_tohtml(df1, add_days= add_days)

        log(df1.head(1).T, df1.columns)
        return df1


    def user_tohtml(dfa1, add_days=0):
        tk     = get_timekey() + add_days
        dirout  = dir_cpa3 + f"/hdfs/daily_usertopk/m001/{tk}/easyid_eval/"
        dirout2 = f"/a/gfs101/ipsvols07/offline/backend_data/log/ztmp/check/{tk}/"


        def conv2(ll):
            ll2 = [ f"<img width='60'  height='60' src='" + str(x.split(" ")[0]) + "' >"  for x in ll  ]
            ll2 =  "<p>" + ",".join( ll2 ) + "</p>"
            return ll2

        ######### image Export #####################################################
        doc = Doc()  ; i =0
        key = 'image_url'
        for j,x in dfa1.iterrows():
            i  += 1
            doc.add(  x['easy_id'])

            doc.add('hist',  conv2( x['hist_image'][:50],) )
            doc.add('Pur',   conv2( x['hist_pur_image'][:50],) )
            doc.add('EOD topk',  conv2( x['topk_image'][:50],) )
            doc.add('EOD topk_emb',  conv2( x['topk_emb_image'][:50],) )
            doc.add('EOD topk_intra',  conv2( x['topk_intra_image'][:50],) )
            doc.add('EOD topk_brw',  conv2( x['topk_brw_image'][:50],) )
            doc.add('EOD Couch',  conv2( x['topk_couch_image'][:50],) )


            ##### doc.h += "<p style='font-size:16px;'>"
            doc.add( 'hist=',       x['hist'][:50] )
            doc.add( 'hist_pur=',   x['hist_pur'][:50] )
            doc.add( 'topk=',       x['topk'][:50] )
            doc.add( 'topk_emb=',   x['topk_emb'][:50] )
            doc.add( 'topk_intra=', x['topk_intra'][:50]  )
            doc.add( 'topk_brw=',   x['topk_brw'][:50]  )
            doc.add( 'topk_couch=', x['topk_couch'][:50]  )
            doc.add('')

            doc.add( 'siid_toprank= ',  db_item_toprank.get( x['hist'][-1], "") )
            doc.add( 'easy_topgenre_intra= ', db_easyid_topgenre_intra.get( x['easy_id'], "") )
            doc.add( 'easy_topgenre_pur= ',   db_easyid_topgenre_pur.get( x['easy_id'], "")   )
            doc.add( 'easy_topgenre_brw= ',   db_easyid_topgenre_brw.get( x['easy_id'], "")   )
            doc.add('')

            doc.add( 'hist_genre', x['hist_genre'][:50],   )
            doc.add( 'pur_genre',  x['hist_pur_genre'][:50] , )
            doc.add( 'EOD topk_genre',        x['topk_genre'][:50] , )
            doc.add( 'EOD topk_emb_genre',    x['topk_emb_genre'][:50] , )
            doc.add( 'EOD topk_intra_genre',  x['topk_intra_genre'][:50] , )
            doc.add( 'EOD topk_brw_genre',    x['topk_brw_genre'][:50] , )
            doc.add( 'EOD topk_couch_genre',  x['topk_couch_genre'][:50] , )

            #doc.add( 'topk_genre', conv( x['topk'][:50] ,     'genre_path') )

            doc.add("<hr>\n\n")
            if i > 1000 : break

        fi = f"/zcheck_{x['easy_id']}_rec_img_{tk}_{time0()}.html"
        doc.save( dirout  + fi )
        #doc.save( dirout2 + fi )
        #log( f"http://ins-adig101.prod.hnd2.bdd.local:17999/log/ztmp/check/{tk}/{fi}" )


if 'daily_stats':
    def daily_all_user_pur(mode='pur'):      ##   py daily_all_user_pur   &  --mode  pur
        #  List of easyid purchase
        #tmin, tmax = 18542, get_timekey()-1
        tmin, tmax = get_timekey()-20, get_timekey()-1

        bk_list  = [ "*" ]
        #tk_list  = [ str(tmax) ]
        tk_list  = [ str(t) for t in range(tmax, tmin, -1) ]

        cols = ['shop_id', 'item_id', 'easy_id', 'units', 'price' ]

        for bkt in bk_list :
            log('bucket', bkt)
            dirin = ""
            for kk, tk in enumerate(tk_list) :
                tk1     = tk.replace("*","")  ### witohou *
                dirouti  = dir_cpa3 + f'/hdfs/user_all_pur/user_all_{tk}.parquet'
                dirouti2 = dir_cpa3 + f'/hdfs/item_all_pur/item_all_{tk}.parquet'
                if os.path.isfile(dirouti): continue
                log( dirin, tk1, dirouti )

                flist = glob_glob( dirs.pur_dir  + f"/{bkt}/*{tk}.parquet"  )
                log(str(flist)[:200])
                df    = pd_read_file(flist , cols= cols, n_pool=4, verbose=False)
                if len(df) < 1 : continue

                df1 = df.groupby('easy_id').agg({  'item_id' :'count', 'price' : 'mean', 'units': 'sum' }).reset_index()
                pd_to_file(df1, dirouti, show=1)
                del df1

                df1 = df.groupby([ 'shop_id', 'item_id']).agg({  'easy_id' :'count', 'price' : 'mean', 'units': 'sum' }).reset_index()
                del df
                pd_to_file(df1, dirouti2, show=1)


    def ca_daily_user_pur(mode='pur'):       ##   python prepro_prod.py  ca_user_daily_pur  --mode  pur
        """  List of easyid purchase     python prepro_prod.py  ca_daily_user_pur  --mode  pur    &&   python prepro_prod.py  ca_daily_user_hist_brw  --mode  pur

        """
        # tmin, tmax = get_timekey()-100, get_timekey()
        #tmin, tmax = 18542, get_timekey()-1
        tmin, tmax = get_timekey()-10, get_timekey()-1

        # + str(i) + "*" for i in range( 18926 , 18545 , -1)
        bk_list  = [ "*" ]
        #tk_list  = [ str(tmax) ]
        tk_list  = [ str(t) for t in range(tmax, tmin, -1) ]

        cols = ['shop_id', 'item_id', 'easy_id', 'timestamp']

        for bkt in bk_list :
            log('bucket', bkt)
            dirin = ""
            for kk, tk in enumerate(tk_list) :
                tk1     = tk.replace("*","")  ### witohou *
                dirouti = dir_cpa3 + f'/ca_check/stats/user_ca_pur/user_all_{tk}.parquet'
                if os.path.isfile(dirouti): continue
                log( dirin, tk1, dirouti )

                flist = glob_glob( sc_pur_dir  + f"/{bkt}/sc_widget_pch_{tk}.parquet"  )
                log(str(flist)[:200])
                df    = pd_read_file(flist , cols=None, n_pool=20, verbose=False)
                if len(df) < 100 : continue

                df = df.groupby('easy_id').agg({  'item_id' :'count', 'price' : 'mean', 'units': 'sum' }).reset_index()

                pd_to_file(df, dirouti, show=1)


    def ca_daily_user_hist_brw(mode='pur'):  ##   py ca_daily_user_hist_brw    --mode  pur
        """  all siid for one easyid
        """
        # tmin, tmax = 18542, get_timekey()-1
        tmin, tmax = get_timekey()-10, get_timekey()-1

        bk_list  = [ "*" ]
        #tk_list  = [ str(tmax) ]
        tk_list  = [ str(t) for t in range(tmax, tmin, -1) ]
        cols = ['easy_id', 'shop_id', 'item_id']

        for bkt in bk_list :
            log('bucket', bkt)
            dirin = ""
            for kk, tk in enumerate(tk_list) :

                tk1     = tk.replace("*","")  ### witohou *
                dirouti = dir_cpa3 + f'/ca_check/stats/user_brw/purca_user_brw_{tk}.parquet'
                if os.path.isfile(dirouti): continue
                log( dirin, tk1, dirouti )

                leasyid = pd_read_file( dir_ca + f"/stats/user_ca_pur/*{tk}*")
                if len(leasyid) < 1 : continue
                leasyid = set( leasyid['easy_id'].values )
                log( 'Neasyid in ca_pur',  len(leasyid))

                flist = glob_glob( brw_dir + f"/{bkt}/brw_ran_{tk}.parquet" , 10000 )

                fj = []; jj=0;  df = pd.DataFrame()
                for fi in flist :
                    fj.append(fi) ; jj = jj +1
                    if len(fj) < 20 and jj < len(flist)-1 : continue
                    dfj = pd_read_file(fj , cols=cols, n_pool=20, verbose=False)
                    dfj = dfj[dfj.easy_id.isin(leasyid)]
                    fj  = [] ; jj = 0
                    df  = pd.concat((df, dfj))

                log(df.shape)

                df['timekey'] = tk

                #df['siid'] = df.apply(lambda x:  siid(x), axis=1)
                #df         = df.groupby('easy_id').apply( lambda dfi:  ",".join(dfi['siid']) ).reset_index()
                #df.columns = ['easy_id', 'siid_brw']
                if len(df) < 10 : continue
                pd_to_file(df, dirouti, show=1)



##############################################################################################
if 'hadoop':

    def hdfs_down():
        """   python prepro.py  hdfs_down

        kdestroy && kinit -kt  /usr/local/hdp26/keytabs/scoupon.prod.keytab scoupon  && klist

        hdfs dfs  -copyToLocal  hdfs://nameservice1/user/hive/warehouse/nono3.db/shopid_itemlist_img_full/   /data/workspaces/takos01/a

        hdfs dfs  -ls   hdfs://nameservice1/user/hive/warehouse/nono3.db/shopid_itemlist_img_full/

        hdfs dfs  -ls   hdfs://nameservice1/user/hive/warehouse/nono3.db/siid_feat_20210526_v2b/

        hdfs dfs  -copyFromLocal   /data/workspaces/noelkevin01/img/models/fashion/dcf_vae/   hdfs://nameservice1/scoupon/nono/zimg/

        https://hadoop.apache.org/docs/current/hadoop-distcp/DistCp.html

        SOURCEDIR="$1"
            TARGETDIR="$2"
            MAX_PARALLEL=4
            nroffiles=$(ls $SOURCEDIR|wc -w)
            setsize=$(( nroffiles/MAX_PARALLEL + 1 ))
            hadoop fs -mkdir -p $TARGETDIR

            ls -1 $SOURCEDIR/* | xargs -n $setsize | while read workset; do
              hadoop fs -put $workset $TARGETDIR &
            done
            wait

          http://hadooptutorial.info/hdfs-distributed-file-copy-tool-distcp/

        """
        ll = [
            #"itemaster_cat_tag_03"
            # 'siid_feat_20210526'
            # 'ichiba_clk_202010_202105',
            # 'siid_feat_20210526_v3b'
            'siid_feat_20210526_v4'
        ]

        for tag in ll :
            pref1    = "hdfs://nameservice1/user/hive/warehouse/nono3.db/"
            from_dir = pref1 + tag
            #to_dir   = "/data/workspaces/takos01/cpa/input/"
            to_dir   = "/data/workspaces/takos01/cpa/input/" + tag + "/"

            cmd      = f"hdfs dfs  -copyToLocal  '{from_dir}'   '{to_dir}'   |& tee -a zlog_hive.py &  "
            log( cmd )

            hdfs_get(from_dir, to_dir, n_pool=20)
            # os.system( cmd )



    def hdfs_rename(dirin=None, ):
        """  python prepro.py  hdfs_rename
        """
        in_dir  = "/data/workspaces/takos01/cpa/input/"
        flist = glob.glob(in_dir + "/*/*" )
        log(str(flist)[:100])
        for fi in flist :
            if ".parquet" not in fi :
               os.rename(fi, fi + ".parquet" );  log(fi)




    def hdfs_put(from_dir="", to_dir="",  verbose=True, n_pool=15, dirlevel=50,  **kw):
        """  python prepro.py  hdfs_put
        /user/scoupon/zexport/z/fashion_emb_100k

        from_dir = "hdfs://nameservice1/user/hive/warehouse/nono3.db/shop_v1_20210526"
        to_dir   = "data/workspaces/noelkevin01/img/models/fashion/dcf_vae/auto_album/"
        verbose= True

        """
        from_dir   = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_best_good_epoch_313/fashion_emb_100k/"
        to_dir     = "hdfs://nameservice1/user/scoupon/zexport/z/fashion_emb_100k/"
        # verbose= True

        import glob, gc,os, time, pyarrow as pa
        from multiprocessing.pool import ThreadPool

        def log(*s, **kw):
          print(*s, flush=True, **kw)

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
        # log('n files',) )


    def hdfs_walk(path="", dirlevel=3, hdfs=None):   ### python  prepro.py hdfs_walk
        # path =  "hdfs://nameservice1/user/scoupon/nono/z/dcf_vae/m_train9pred/"
        import pyarrow as pa
        # path = "/user/scoupon/zexport/"
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
        """  python prepro.py  hdfs_get
        from_dir = "hdfs://nameservice1/user/hive/warehouse/nono3.db/shop_v1_20210526"
        to_dir   = "/data/workspaces/takos01/cpa/ztest/"

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


        ktot = 0
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



##########################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()



