









    
def topk_predict():
    #### wrapper :      python prepro.py topk_predict 
    
    dname = "m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000.cache/best/best_good_epoch_313"
        
    cmd = f" python train9pred.py '{dname}'  "
    os.system(cmd)

    
    
def topk(topk=100, dname=None, pattern="df_*", filter1=None):
    """  python prepro.py  topk    |& tee -a  /data/worksg3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_epoch_261/topk/zzlog.py
    

         
    """
    from utilmy import pd_read_file
    
    filter1 = "all"    #### "article"
    
    if dname is None :
       # dname = "m_train8bb_g2_-img_train_r2p2_70k_clean_nobg_256_256-100000-cache_best_epoch_64"
       # dname = "m_train9a_g6_-img_train_nobg_256_256-100000-cache_best_epoch_54"
       # dname = "m_train9b_g3_-img_train_r2p2_70k_clean_nobg_256_256-100000-cache_best_epoch_69"
       # dname = "m_train9a_g6_-img_train_nobg_256_256-100000-cache_best_epoch_98"
       # dname = "m_train9b_g3_-img_train_r2p2_70k_clean_nobg_256_256-100000-cache_best_epoch_120" 

       dname = "m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_epoch_261"
     
    
    dname    = dname.replace("/", "_").replace(".", "-")    
    r0       = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/"
    in_dir   = r0 + dname
    out_dir  = in_dir + "/topk/"
    os.makedirs(out_dir, exist_ok=True)
    log(in_dir)
    
    #### Load emb data  ###############################################
    df        = pd_read_file(  in_dir + f"/{pattern}.parquet", n_pool=10 )
    log(df)
    df['id1'] = df['id'].apply(lambda x : x.split(".")[0])
    
    
    """
    name = 'tshirts'
    dfy  = df[df.articleType == name ]
    # dfy[[ 'id1', 'gender', 'pred_emb'  ]].to_parquet(  out_dir + f'/export_{name}.parquet'  )
    del dfy[ 'pred_emb' ] ; dfy.to_csv( out_dir + f'/{name}.csv')
    sys.exit(0)
    """
    
    #### Element X0 ######################################################
    colsx = [  'masterCategory', 'subCategory', 'articleType' ]  # 'gender', , 'baseColour' ] 
    df0   = df.drop_duplicates( colsx )    
    log('Reference images', df0)
    llids = list(df0.sample(frac=1.0)['id'].values)
    
    """
    llids = [ # "ab5517-05_1-10.png",  ###  blouson purple
              # 'as5238-06_1-2.png',   ###  tee shirt kids yellow
              # 'ab5503-06_1-1.png',   ###  men topwear
              "ac4552-01_1-1.png",     ###  White blouson women    
              'ab5509-01_1-1.png',     ###  Men white pants
              'ad4408-07_1-17.png'      ## thsirt men blue
            ]
    """
    
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


     hdfs dfs -put  /data/womg_trche_best_best_good_epoch_313/fashion_emb_500k/                /user/scoupon/zexport/z/

    """
    dir_in  = "/daain9b_g3_-img_trai000-cache_best_best_good_epoch_313/*.parquet" 
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
 
    
