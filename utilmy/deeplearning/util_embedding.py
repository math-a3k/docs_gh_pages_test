# -*- coding: utf-8 -*-
HELP=""" Embedding

https://try2explore.com/questions/10109123

https://mpld3.github.io/examples/index.html


"""
import warnings ;warnings.filterwarnings("ignore")
from warnings import simplefilter  ; simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings():
    import random, os, sys, numpy as np, pandas as pd, time, gc, copy, glob
    from datetime import datetime ; from typing import List

    import matplotlib.pyplot as plt

    from scipy.cluster.hierarchy import ward, dendrogram
    from sklearn.cluster import KMeans
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    from utilmy import pd_read_file, os_makedirs, pd_to_file

    from box import Box 


import os, glob, sys, math, string, time, json, logging, functools, random, yaml, operator, gc
from pathlib import Path; from collections import defaultdict, OrderedDict ;


try :
   import diskcache as dc
   import faiss
except: pass

from utilmy.utilmy import   pd_read_file, pd_to_file

#####################################################################################
from utilmy import log, log2



           
#########################################################################################################
def embedding_table_comparison(embeddings_1:list, embeddings_2:list, labels_1:list, labels_2:list,
                         plot_title,
                         plot_width=1200, plot_height=600,
                         xaxis_font_size='12pt', yaxis_font_size='12pt'):
        """
           list of vectors    vs list of labels
           list of vectors    vs list tof labels


        """
        import bokeh
        import bokeh.models
        import bokeh.plotting

        assert len(embeddings_1) == len(labels_1)
        assert len(embeddings_2) == len(labels_2)

        # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
        sim = 1 - np.arccos(
            sklearn.metrics.pairwise.cosine_similarity(embeddings_1,
                                                       embeddings_2))/np.pi

        embeddings_1_col, embeddings_2_col, sim_col = [], [], []
        for i in range(len(embeddings_1)):
          for j in range(len(embeddings_2)):
            embeddings_1_col.append(labels_1[i])
            embeddings_2_col.append(labels_2[j])
            sim_col.append(sim[i][j])
        df = pd.DataFrame(zip(embeddings_1_col, embeddings_2_col, sim_col),
                          columns=['embeddings_1', 'embeddings_2', 'sim'])

        mapper = bokeh.models.LinearColorMapper(
            palette=[*reversed(bokeh.palettes.YlOrRd[9])], low=df.sim.min(),
            high=df.sim.max())

        p = bokeh.plotting.figure(title=plot_title, x_range=labels_1,
                                  x_axis_location="above",
                                  y_range=[*reversed(labels_2)],
                                  plot_width=plot_width, plot_height=plot_height,
                                  tools="save",toolbar_location='below', tooltips=[
                                      ('pair', '@embeddings_1 ||| @embeddings_2'),
                                      ('sim', '@sim')])
        p.rect(x="embeddings_1", y="embeddings_2", width=1, height=1, source=df,
               fill_color={'field': 'sim', 'transform': mapper}, line_color=None)

        p.title.text_font_size = '12pt'
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_standoff = 16
        p.xaxis.major_label_text_font_size = xaxis_font_size
        p.xaxis.major_label_orientation = 0.25 * np.pi
        p.yaxis.major_label_text_font_size = yaxis_font_size
        p.min_border_right = 300

        bokeh.io.output_notebook()
        bokeh.io.show(p)
        
        
def embedding_to_parquet(dirin=None, dirout=None, skip=0, nmax=10**8, 
                         is_linevalid_fun=None):   ##   python emb.py   embedding_to_parquet  &
    #### FastText/ Word2Vec to parquet files    9808032 for purhase
    log(dirout) ; os_makedirs(dirout)  ; time.sleep(4)

    if is_linevalid_fun is None : #### Validate line
        def is_linevalid_fun(w):
            return len(w)> 5  ### not too small tag

    i = 0; kk=-1; words =[]; embs= []; ntot=0
    with open(dirin, mode='r') as fp:
        while i < nmax+1  :
            i  = i + 1
            ss = fp.readline()
            if not ss  : break
            if i < skip: continue

            ss = ss.strip().split(" ")            
            if not is_linevalid_fun(ss[0]): continue

            words.append(ss[0])
            embs.append( ",".join(ss[1:]) )

            if i % 200000 == 0 :
              kk = kk + 1                
              df = pd.DataFrame({ 'id' : words, 'emb' : embs }  )  
              log(df.shape, ntot)  
              if i < 2: log(df)  
              pd_to_file(df, dirout + f"/df_emb_{kk}.parquet", show=0)
              ntot += len(df)
              words, embs = [], []  

    kk      = kk + 1                
    df      = pd.DataFrame({ 'id' : words, 'emb' : embs }  )  
    ntot   += len(df)
    dirout2 = dirout + f"/df_emb_{kk}.parquet"
    pd_to_file(df, dirout2, show=1 )
    log('ntotal', ntot, dirout2 )
    return os.path.dirname(dirout2)


def embedding_load_parquet(dirin="df.parquet", nmax = 500):
    """  id, emb (string , separated)
    
    """
    log('loading', dirin)     
    col_embed = 'pred_emb'
    colid     = 'id'
    nmax    = nmax if nmax > 0 else  len(df)   ### 5000
    
    flist = list( glob.glob(dirin) )
    
    df  = pd_read_file( flist, npool= max(1, int( len(flist) / 4) ) )
    df  = df.iloc[:nmax, :]
    df  = df.rename(columns={ col_embed: 'emb'})
    
    df  = df[ df['emb'].apply( lambda x: len(x)> 10  ) ]  ### Filter small vector
    log(df.head(5).T, df.columns, df.shape)
    log(df, df.dtypes)    


    ###########################################################################
    ###### Split embed numpy array, id_map list,  #############################
    embs    = np_str_to_array(df['emb'].values,  l2_norm=True,     mdim = 200)
    id_map  = { name: i for i,name in enumerate(df[colid].values) }     
    log(",", str(embs)[:50], ",", str(id_map)[:50] )
    
    #####  Keep only label infos  ####
    del df['emb']                  
    return embs, id_map, df 


def np_str_to_array(vv,  l2_norm=True,     mdim = 200):
    ### Extract list of string into numpy
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
    
    
def viz_run(dirin="in/model.vec", dirout="ztmp/", nmax=100):
   ###   python emb.py run    &  
   nmax    =  500000
   mode    =  'umap'
   tag     = f"{nmax}_{mode}"

   #### Generate HTML  ############################################ 
   log(dirin)
   myviz = vizEmbedding(path = dirin )
   myviz.run_all(nmax=nmax, dir_out= dirout, mode=mode, ntest=50000)

         
        

    
def topk(topk=100, dname=None, pattern="df_*1000*.parquet", filter1=None):
    """  python emb.py  topk    |& tee -a  /zzlog.py
    
    """
    from utilmy import pd_read_file
    if dname is None :
       dname = "seq_100000000"
    
    ksample = 500
         
    ###################################################################
    dname    = dname.replace("/", "_").replace(".", "-")    
    in_dir   = r0 + dname
    out_dir  = in_dir + "/topk/"
    os.makedirs(out_dir, exist_ok=True)
    log(in_dir)
    
    #### Load emb data  ###############################################
    df        = pd_read_file(  in_dir + f"/{pattern}", n_pool=10 )
    df.index = np.arange(0, len(df))
    log(df)
    # df['emb'] = df['emb'].apply(lambda x :  list( np.array(x) /np.sqrt(np.dot(x,x)) ) )   ###Norm Vector

        
    #### Element X0 ####################################################
    llids   = list(df.sample(frac=1.0)['id'].values)
    vectors =  np_str_to_array(df['emb'].values,  mdim=200)   
    
    # faiss_create_index(df_or_path=None, col='emb', dir_out="",  db_type = "IVF4096,Flat", nfile=1000, emb_dim=200)
    
    for ii,idr in enumerate(llids) :        
        if ii >= ksample : break
        dfi     = df[ df['id'] == idr ] 
        if len(dfi) < 1: continue
        x0      = np.array(dfi['emb'].values[0]).astype(np.float32)
        xname   = dfi['id'].values[0]
        log(xname)

        ##### Setup Faiss queey ########################################
        x0      = x0.reshape(1, -1).astype('float32')  
        # log(x0.shape, vectors.shape)
        dist, rank = topk_nearest_vector(x0, vectors, topk= topk) 
        df1              = df.iloc[rank[0], :]
        df1['topk_dist'] = dist[0]
        df1['topk_rank'] = np.arange(0, len(df1))
        log( df1 )
        del df1['emb']
        df1.to_csv( out_dir + f"/topk_{xname}_{filter1}.csv"  , sep=",")


        
        
def topk_nearest_vector(x0, vector_list, topk=3) :
   """ Retrieve top k nearest vectors using FAISS
   """
   import faiss  
   index = faiss.index_factory(x0.shape[1], 'Flat')
   index.add(vector_list)
   dist, indice = index.search(x0, topk)
   return dist, indice
   
        

 
def sim_score2(path=""):
    """
       Sim Score using FAISS
    
    """
    import faiss
    x0 = [ 0.1, .2, 0.3]
    x  = np.array([x0]).astype(np.float32)

    index = faiss.index_factory(3, "Flat", faiss.METRIC_INNER_PRODUCT)
    log(index.ntotal)
    faiss.normalize_L2(x)
    index.add(x)
    distance, index = index.search(x, 5)
    log(f'Distance by FAISS:{distance}')
    return distance, index
    
    #To Tally the results check the cosine similarity of the following example
    #from scipy import spatial
    #result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    #print('Distance by FAISS:{}'.format(result))

    


#####################################################################################
def simscore_cosinus_calc(embs, words):
    """
    
      Calculation
    
    """
    from sklearn.metrics.pairwise import cosine_similarity    
    dfsim = []
    for i in  range(0, len(words) -1) :
        vi = embs[i,:]
        normi = np.sqrt(np.dot(vi,vi))
        for j in range(i+1, len(words) ) :
            # simij = cosine_similarity( embs[i,:].reshape(1, -1) , embs[j,:].reshape(1, -1)     )
            vj = embs[j,:]
            normj = np.sqrt(np.dot(vj, vj))
            simij = np.dot( vi ,  vj  ) / (normi * normj)
            dfsim.append([ words[i], words[j],  simij   ])
            # dfsim2.append([ nwords[i], nwords[j],  simij[0][0]  ])
    
    dfsim  = pd.DataFrame(dfsim, columns= ['l3_genre_a', 'l3_genre_b', 'sim_score' ] )   

    ### Add symmetric part      
    dfsim3 = copy.deepcopy(dfsim)
    dfsim3.columns = ['l3_genre_b', 'l3_genre_a', 'sim_score' ]
    dfsim          = pd.concat(( dfsim, dfsim3 ))
    return dfsim



#####################################################################################
def faiss_create_index(df_or_path=None, col='emb', dir_out="",  db_type = "IVF4096,Flat", nfile=1000, emb_dim=200):
    """
      1 billion size vector creation
      ####  python prepro.py   faiss_create_index      2>&1 | tee -a log_faiss.txt    
    """
    import faiss
    # nfile      = 1000
    emb_dim    = 200   
    
    if df_or_path is None :  df_or_path = "/emb/emb//ichib000000000/df/*.parquet"
    dirout    =  "/".join( os.path.dirname(df_or_path).split("/")[:-1]) + "/faiss/"
    os.makedirs(dirout, exist_ok=True) ; 
    log( 'dirout', dirout)    
    log('dirin',   df_or_path)  ; time.sleep(10)
    
    if isinstance(df_or_path, str) :      
       flist = sorted(glob.glob(df_or_path  ))[:nfile] 
       log('Loading', df_or_path) 
       df = pd_read_file(flist, n_pool=20, verbose=False)
    else :
       df = df_or_path
    # df  = df.iloc[:9000, :]        
    log(df)
        
    tag = f"_" + str(len(df))    
    df  = df.sort_values('id')    
    df[ 'idx' ] = np.arange(0,len(df))
    pd_to_file( df[[ 'idx', 'id' ]].rename(columns={"id":'item_tag_vran'}), 
                dirout + f"/map_idx{tag}.parquet", show=1)   #### Keeping maping faiss idx, item_tag
    

    log("### Convert parquet to numpy   ", dirout)
    X  = np.zeros((len(df), emb_dim  ), dtype=np.float32 )    
    vv = df[col].values
    del df; gc.collect()
    for i, r in enumerate(vv) :
        try :
          vi      = [ float(v) for v in r.split(',')]        
          X[i, :] = vi
        except Exception as e:
          log(i, e)
            
    log("Preprocess X")
    faiss.normalize_L2(X)  ### Inplace L2 normalization
    log( X ) 
    
    nt = min(len(X), int(max(400000, len(X) *0.075 )) )
    Xt = X[ np.random.randint(len(X), size=nt),:]
    log('Nsample training', nt)

    ####################################################    
    D = emb_dim   ### actual  embedding size
    N = len(X)   #1000000

    # Param of PQ for 1 billion
    M      = 40 # 16  ###  200 / 5 = 40  The number of sub-vector. Typically this is 8, 16, 32, etc.
    nbits  = 8        ### bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte    
    nlist  = 6000     ###  # Param of IVF,  Number of cells (space partition). Typical value is sqrt(N)    
    hnsw_m = 32       ###  # Param of HNSW Number of neighbors for HNSW. This is typically 32

    # Setup  distance -> similarity in uncompressed space is  dis = 2 - 2 * sim, https://github.com/facebookresearch/faiss/issues/632
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
        root    = dir_cpa2 + "/emb/emb/i_1000000000/"
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



def topk_predict():
    #### wrapper :      python prepro.py topk_predict 
    
    dname = "m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000.cache/best/best_good_epoch_313"
        
    cmd = f" python train9pred.py '{dname}'  "
    os.system(cmd)

    
    
def topk(topk=100, dname=None, pattern="df_*", filter1=None):
    """  python prepro.py  topk    |& tee -a  /data/worpoch_261/topk/zzlog.py
    

    """
    from utilmy import pd_read_file
    
    filter1 = "all"    #### "article"
    
    if dname is None : 
       dname = "m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_epoch_261"
     
    
    dname    = dname.replace("/", "_").replace(".", "-")    
    r0       = "/data/workspin9pred/res/"
    in_dir   = r0 + dname
    out_dir  = in_dir + "/topk/"
    os.makedirs(out_dir, exist_ok=True)
    log(in_dir)
    
    #### Load emb data  ###############################################
    df        = pd_read_file(  in_dir + f"/{pattern}.parquet", n_pool=10 )
    log(df)
    df['id1'] = df['id'].apply(lambda x : x.split(".")[0])
    
    
    #### Element X0 ######################################################
    colsx = [  'masterCategory', 'subCategory', 'articleType' ]  # 'gender', , 'baseColour' ] 
    df0   = df.drop_duplicates( colsx )    
    log('Reference images', df0)
    llids = list(df0.sample(frac=1.0)['id'].values)
    

    for idr1 in llids :        
        log(idr1)
        #### Elements  ####################################################
        ll = [  (  idr1,  'all'     ),
                # (  idr1,  'article' ),
                (  idr1,  'color'   )
        ]       


        for (idr, filter1) in ll :                
            dfi     = df[ df['id'] == idr ] 
            log(dfi)
            if len(dfi) < 1: continue
            x0      = np.array(dfi['pred_emb'].values[0])
            xname   = dfi['id'].values[0]
            log(xname)

            #### 'gender',  'masterCategory', 'subCategory',  'articleType',  'baseColour',
            g1 = dfi['gender'].values[0]
            g2 = dfi['masterCategory'].values[0]
            g3 = dfi['subCategory'].values[0]
            g4 = dfi['articleType'].values[0]
            g5 = dfi['baseColour'].values[0]
            log(g1, g2, g3, g4, g5)

            xname = f"{g1}_{g4}_{g5}_{xname}".replace("/", "-") 

            if filter1 == 'article' :
                df1 = df[ (df.articleType == g4) ]       

            if filter1 == 'color' :
                df1 = df[ (df.gender == g1) & (df.subCategory == g3) & (df.articleType == g4) & (df.baseColour == g5)  ]    
            else :
                df1 = copy.deepcopy(df)
                #log(df)

            ##### Setup Faiss queey ########################################
            x0      = x0.reshape(1, -1).astype('float32')
            vectors = np.array( list(df1['pred_emb'].values) )    
            log(x0.shape, vectors.shape)

            dist, rank = topk_nearest_vector(x0, vectors, topk= topk) 
            # print(dist)
            df1              = df1.iloc[rank[0], :]
            df1['topk_dist'] = dist[0]
            df1['topk_rank'] = np.arange(0, len(df1))
            log( df1 )
            df1.to_csv( out_dir + f"/topk_{xname}_{filter1}.csv"  )

            img_list = df1['id'].values
            log(str(img_list)[:30])

            log('### Writing images on disk  ###########################################')
            import diskcache as dc
            # db_path = "/data/workspaces/noelkevin01/img/data/fashion/train_npz/small/img_train_r2p2_70k_clean_nobg_256_256-100000.cache"
            db_path = "/dev/shm/train_npz/small//img_train_r2p2_1000k_clean_nobg_256_256-1000000.cache"            
            cache   = dc.Cache(db_path)
            print('Nimages', len(cache) )

            dir_check = out_dir + f"/{xname}_{filter1}/"
            os.makedirs(dir_check, exist_ok=True)
            for i, key in enumerate(img_list) :
                if i > 15: break       
                img  = cache[key]
                img  = img[:, :, ::-1]
                key2 = key.split("/")[-1]
                cv2.imwrite( dir_check + f"/{i}_{key2}"  , img)            
            log( dir_check )    


        
def topk_nearest_vector(x0, vector_list, topk=3) :
   """
      Retrieve top k nearest vectors using FAISS
  
   """
   import faiss  
   index = faiss.index_factory(x0.shape[1], 'Flat')
   index.add(vector_list)
   dist, indice = index.search(x0, topk)
   return dist, indice



def topk_export():     #### python prepro.py  topk_export  
    """   /user/scoon_emb_100k  
                     id gender masterCategory subCategory  ... masterCategory_pred subCategory_pred articleType_pred  baseColour_pred
0     cn3357-01_1-11.png  women        apparel     topwear  ...                   1                1               32                4
1      cs6481-01_1-4.png   kids          shoes       shoes  ...                   5               20              151                7


    """
    dir_in  = "/daain9b_g3_-imh_313/*.parquet" 
    dir_out = "/data/wai13/fashion_emb_500k/"
    
    os.makedirs(dir_out, exist_ok=True)
    flist = glob.glob(dir_in)
    for ii, fi in enumerate(flist):
        if ii > 200 : break
        log(ii, fi)    
        dfi  = pd.read_parquet(fi)
        # log(dfi)
        cols = ['id', 'gender',  'masterCategory', 'baseColour',  'pred_emb' ]
        dfi  = dfi[cols]
        #log(dfi)
        dfi.to_parquet( dir_out + "/" + fi.split("/")[-1]  )
    log(dfi)        
 
    

def convert_txt_to_vector_parquet(dirin=None, dirout=None, skip=0, nmax=10**8):   ##   python prepro.py   create_vector_parquet  &
    #### FastText/ Word2Vec to parquet files    9808032 for purhase
    nmax = 10**8
    if dirin is None :
       dirin  = dir_cpaloc +  "/emb/ichiba_order_model.vec"
    
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


def data_add_onehot(dfref, img_dir, labels_col):
    """
       id, uri, cat1, cat2, .... , cat1_onehot

    """
    import glob
    fpaths = glob.glob(img_dir)
    fpaths = [fi for fi in fpaths if "." in fi.split("/")[-1]]
    log(str(fpaths)[:100])

    df = pd.DataFrame(fpaths, columns=['uri'])
    log(df.head(1).T)
    df['id'] = df['uri'].apply(lambda x: x.split("/")[-1].split(".")[0])
    df['id'] = df['id'].apply(lambda x: int(x))
    df = df.merge(dfref, on='id', how='left')

    # labels_col = [  'gender', 'masterCategory', 'subCategory', 'articleType' ]

    for ci in labels_col:
        dfi_1hot = pd.get_dummies(df, columns=[ci])  ### OneHot
        dfi_1hot = dfi_1hot[[t for t in dfi_1hot.columns if ci in t]]  ## keep only OneHot
        df[ci + "_onehot"] = dfi_1hot.apply(lambda x: ','.join([str(t) for t in x]), axis=1)
        #####  0,0,1,0 format   log(dfi_1hot)

    return df


def test():
    """
       python prepro.py test

    """
    dfref = pd.read_csv(data_label + "/prepro_df.csv")
    img_dir = data_dir + '/train/*'
    labels_col = ['gender', 'masterCategory', 'subCategory', 'articleType']

    df = data_add_onehot(dfref, img_dir, labels_col)
    log(df.head(2).T)


def unzip(in_dir, out_dir):
    # !/usr/bin/env python3
    import sys
    import zipfile
    with zipfile.ZipFile(in_dir, 'r') as zip_ref:
        zip_ref.extractall(out_dir)




    
    
    

################################################################################################################
import mpld3

CSS = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
    """

class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();
      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);
      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """

    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


        
################################################################################################################
class vizEmbedding:
    def __init__(self, path="myembed.parquet", num_clusters=5, sep=";", config:dict=None):
        """ 
           Many issues with numba, numpy, pyarrow !!!!
           pip install  pynndescent==0.5.4  numba==0.53.1  umap-learn==0.5.1  llvmlite==0.36.0   numpy==1.19.1   --no-deps
        
           self = Box({})
           self.path = "C:/D/gitdev/cpa/data/model.vec"

           from utilmy.viz.embedding import vizEmbedding
           myviz = vizEmbedding(path = "C:/D/gitdev/cpa/data/model.vec")
           myviz.run_all(nmax=5000)

           myviz.dim_reduction(mode='mds')
           myviz.create_visualization(dir_out="ztmp/vis/")        
        
        """
        self.path         = path
        self.sep          = sep
        self.num_clusters = num_clusters
        self.dist         = None

    def run_all(self, mode="mds", col_embed='embed', ndim=2, nmax= 5000, dir_out="ztmp/", ntest=10000):
       self.dim_reduction( mode, col_embed, ndim=ndim, nmax= nmax, dir_out=dir_out, ntest=ntest)
       self.create_clusters(after_dim_reduction=True)
       self.create_visualization(dir_out, mode='d3', cols_label=None, show_server=False)


    def dim_reduction(self, mode="mds", col_embed='embed', ndim=2, nmax= 5000, dir_out=None, ntest=10000, npool=2 ): 
        
        if ".vec"     in self.path :        
          embs, id_map, df_labels  = embedding_load_word2vec(self.path, nmax= nmax)
        
        if ".parquet" in self.path :        
          embs, id_map, df_labels  = embedding_load_parquet(self.path, nmax= nmax)

            
        if mode == 'mds' :
            ### Co-variance matrix
            dist = 1 - cosine_similarity(embs)
            mds = MDS(n_components=ndim, dissimilarity="precomputed", random_state=1)
            mds.fit(dist)  # shape (n_components, n_samples)
            pos = mds.transform(dist)  # shape (n_components, n_samples)
            
            
        if mode == 'umap' :
            y_label = None
            from umap import UMAP, AlignedUMAP, ParametricUMAP
            clf = UMAP( set_op_mix_ratio=0.25, ## Preserve outlier
                        densmap=False, dens_lambda=5.0,          ## Preserve density
                        n_components= ndim,
                        n_neighbors=7,  metric='euclidean',
                        metric_kwds=None, output_metric='euclidean',
                        output_metric_kwds=None, n_epochs=None,
                        learning_rate=1.0, init='spectral',
                        min_dist=0.0, spread=1.0, low_memory=True, n_jobs= npool,
                        local_connectivity=1.0,
                        repulsion_strength=1.0, negative_sample_rate=5,
                        transform_queue_size=4.0, a=None, b=None, random_state=None,
                        angular_rp_forest=False, target_n_neighbors=-1,
                        target_metric='categorical', target_metric_kwds=None,
                        target_weight=0.5, transform_seed=42, transform_mode='embedding',
                        force_approximation_algorithm= True, verbose=False,
                        unique=False,  dens_frac=0.3,
                        dens_var_shift=0.1, output_dens=False, disconnection_distance=None)

            clf.fit( embs[ np.random.choice( len(embs), size= ntest )  , : ] , y=y_label )                      
            pos  = clf.transform( embs )          

        self.embs      = embs
        self.id_map    = id_map
        self.df_labels = df_labels        
        self.pos       = pos

        if dir_out is not None :
            os.makedirs(dir_out, exist_ok=True)
            df = pd.DataFrame(pos, columns=['x', 'y'] )
            for ci in [ 'x', 'y' ] :
               df[ ci ] = df[ ci ].astype('float32')
   
            # log(df, df.dtypes)
            pd_to_file(df.iloc[:100, :],  f"{dir_out}/embs_xy_{mode}.csv" )
            pd_to_file(df,                f"{dir_out}/embs_xy_{mode}.parquet" , show=1)
                

    def create_clusters(self, after_dim_reduction=True):
        
        import hdbscan
        #km = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=10)  #.fit_predict(self.pos)
        km = KMeans(n_clusters=self.num_clusters)

        if after_dim_reduction :
           km.fit( self.pos)
        else :
           km.fit( self.embs)


        self.clusters      = km.labels_.tolist()        
        self.cluster_color = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(self.num_clusters)]        
        self.cluster_names = {i: f'Cluster {i}' for i in range(self.num_clusters)}
        
        
    def create_visualization(self, dir_out="ztmp/", mode='d3', cols_label=None, show_server=False,  **kw ):
        """

        """
        os.makedirs(dir_out, exist_ok=True)
        cols_label          = [] if cols_label is None else cols_label 
        text_label_and_text = []
        for i,x in self.df_labels.iterrows():
          ss = x["id"]  
          for ci in cols_label:  
             ss = ss + ":" + x[ci]
          text_label_and_text.append(ss) 

        #######################################################################################
        # create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x= self.pos[:, 0], 
                               y= self.pos[:, 1], 
                               clusters= self.clusters, title=text_label_and_text))
        df.to_parquet(f"{dir_out}/embs_xy_cluster.parquet")


        # group by cluster
        groups_clusters = df.groupby('clusters')

        # set up plot
        fig, ax = plt.subplots(figsize=(25, 15))  # set size
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return
        # the appropriate color/label
        for name, group in groups_clusters:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label= self.cluster_names[name],
                    color=self.cluster_color[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(axis='x',  # changes apply to the x-axis
                           which='both',  # both major and minor ticks are affected
                           bottom='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelbottom='off')
            ax.tick_params(axis='y',  # changes apply to the y-axis
                           which='both',  # both major and minor ticks are affected
                           left='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelleft='off')

        ax.legend(numpoints=1)  # show legend with only 1 point

        # add label in x,y position with the label as the
        for i in range(len(df)):
            ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

        # uncomment the below to save the plot if need be
        plt.savefig(f'{dir_out}/clusters_static-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.png', dpi=200)

        # Plot
        fig, ax = plt.subplots(figsize=(20, 15))  # set plot size
        ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        for name, group in groups_clusters:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=7, label= self.cluster_names[name], mec='none',
                             color=self.cluster_color[name])
            ax.set_aspect('auto')
            labels = [i for i in group.title]

            # set tooltip using points, labels and the already defined 'css'
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=CSS)
            # connect tooltip to fig
            mpld3.plugins.connect(fig, tooltip, TopToolbar())

            # set tick marks as blank
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            # set axis as blank
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        ax.legend(numpoints=1)  # show legend with only one dot


        ##### Export ############################################################
        mpld3.save_html(fig,  f"{dir_out}/embeds.html")
        log('Visualization',    f"{dir_out}/embeds.html" )

        ### Windows specifc
        if os.name == 'nt': os.system(f'start chrome "{dir_out}/embeds.html" ')


        if show_server :
           # mpld3.show(fig=None, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None, **kwargs)[source] 
           mpld3.show()  # show the plot



    def draw_hiearchy(self):
        linkage_matrix = ward(self.dist)  # define the linkage_matrix using ward clustering pre-computed distances
        fig, ax = plt.subplots(figsize=(15, 20))  # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=self.text_labels)
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tight_layout()
        plt.savefig('dendogram_clusters.png', dpi=200)

    
    
 
    
###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()



    
