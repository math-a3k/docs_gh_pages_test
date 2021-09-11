# -*- coding: utf-8 -*-
"""


"""
import os,sys,glob,time,gc,copy
os.environ['MPLCONFIGDIR'] = "/tmp/"
try :
  from importall import *
  from utils import *
  from utils import xdim, ydim, cdim, log5
  import util_image
except : pass



##########################################################################################
xdim= 64
ydim= 64

def log(*s):
    print(*s, flush=True)


##########################################################################################
def prepro_images(image_paths, nmax=10000000):
  images = [] 
  for i in range(len(image_paths)):
    if i > nmax : break                
    image =  prepro_image(image_paths[i] )
    images.append(image)        
  return images


def prepro_image0(image_path):
    mean   = [0.5]
    std    = [0.5]  
    try :
        fname      = str(image_path).split("/")[-1]    
        id1        = fname.split(".")[0]
        # print(image_path)

        image = util_image.image_read(image_path)
        image = util_image.image_resize_pad(image, (xdim,ydim), padColor=0)
        image = util_image.image_center_crop(image, (xdim,ydim))
        image = (image / 255)
        image = (image-mean) /std  # Normalize the image to mean and std
        image = image.astype('float32')
        return image, image_path 
    except :
        return [], ""


    
def prepro_images_multi(image_paths, npool=30, prepro_image=None):
    """ Parallel processing
    
    """
    from multiprocessing.dummy import Pool    #### use threads for I/O bound tasks

    pool = Pool(npool) 
    res  = pool.map(prepro_image, image_paths)      
    pool.close()
    pool.join()     
    
    print('len res', len(res))
    images, labels = [], []
    for (x,y) in res :
        if len(y)> 0 and len(x)> 0 :
            images.append(x)            
            labels.append(y)        
            
    print('len images', len(images))
    print(str(labels)[:60])        
    return images, labels
    


def run_multiprocess(myfun, list_args, npool=10, **kwargs):
    """
       res = run_multiprocess(prepro, image_paths, npool=10, )
    """
    from functools import partial
    from multiprocessing.dummy import Pool    #### use threads for I/O bound tasks
    pool = Pool(npool) 
    res  = pool.map( partial(myfun, **kwargs), list_args)      
    pool.close()
    pool.join()   
    return res
    
    

    
def create_train_npz():
    
    import cv2, gc
    #### List of images (each in the form of a 28x28x3 numpy array of rgb pixels)  ############
    #### data_dir = pathlib.Path(data_img)
    nmax =  100000

    log("### Sub-Category  #################################################################")
    # tag   = "-women_topwear"
    tag = "-alllabel_nobg"
    tag = tag + f"-{xdim}_{ydim}-{nmax}"
    log(tag)


    df    = pd.read_csv( data_label  +"/preproed_df.csv")
    # flist = set(list(df[ (df['subCategory'] == 'Watches') & (df.gender == 'Women')   ]['id'].values))
    # flist = set(list(df[ (df['subCategory'] == 'Topwear') & (df.gender == 'Women')   ]['id'].values))
    flist = set(list(df['id'].values))
    log('Label size', len(flist))
    log(tag)


    log("#### Train  ######################################################################")
    image_list = sorted(list(glob.glob( data_dir + '/train_nobg/*.*')))
    image_list = image_list[:nmax]
    log('Size Before', len(image_list))

    ### Filter out
    image_list = [ t for  t in image_list if  int(t.split("/")[-1].split(".")[0]) in flist  ]
    log('Size After', len(image_list))

    images, labels = prepro_images_multi(image_list, prepro_image= prepro_image)
    log5( images )
    train_images = np.array(images)
    train_label  = np.array(labels)
    del images, labels ; gc.collect


    log("#### Test  #######################################################################")
    image_list = sorted(list(glob.glob( data_dir + '/test_nobg/*.*')))
    image_list = image_list[:nmax]
    # log( image_list )

    image_list = [ t for  t in image_list if  int(t.split("/")[-1].split(".")[0]) in flist ]
    log('Size After', len(image_list))
    images, labels = prepro_images_multi(image_list, prepro_image= prepro_image)
    log(str(images)[:100]  )
    test_images = np.array(images)
    test_label  = np.array(labels)
    del images, labels ; gc.collect


    log("#### Save train, test ###########################################################")
    np.savez_compressed( data_train + f"/train_test{tag}.npz", 
                         train = train_images, 
                         test  = test_images,

                         train_label = train_label,
                         test_label  = test_label,

                         df_master   =  df  ###Labels
                       )

    log('size', len(test_images))
    log(data_train + f"/train_test{tag}.npz" )

    util_image.image_check_npz(data_train + f"/train_test{tag}.npz" ,  
                                 keys = None, 
                                 path = data_train + "/zcheck/", 
                                 tag  = tag , n_sample=3
                              )


    
    
 
    
def image_resize(out_dir=""):
    """     python prepro.py  image_resize

          image white color padded
    
    """
    import cv2, gc, diskcache
    
    in_dir   = data_dir + "/train_nobg"
    out_dir  = data_dir + "/train_nobg_256/"
    
    nmax     =  500000000
    global xdim, ydim
    xdim= 256
    ydim= 256  
    padcolor = 0   ## 0 : black
    
    os.makedirs(out_dir, exist_ok= True)
    log('target folder', out_dir); time.sleep(5)

    def prepro_image3b(img_path):
        try :
            fname        = str(img_path).split("/")[-1]    
            id1          = fname.split(".")[0]
            img_path_new = out_dir + "/" + fname

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  
            img = util_image.image_resize_pad(img, (xdim,ydim), padColor=  padcolor)   ### 255 white, 0 for black
            img = img[:, :, ::-1]
            cv2.imwrite(img_path_new, img)        
            # print(img_path_new)            
            return [1], "1"
        except Exception as e:
            # print(image_path, e)
            return [],""
    
    log("#### Process  ######################################################################")
    image_list = sorted(list(glob.glob(  f'/{in_dir}/*.*')))
    image_list = image_list[:nmax]
    log('Size Before', len(image_list))
    

    log("#### Saving disk  #################################################################")        
    images, labels = prepro_images_multi(image_list, prepro_image= prepro_image3b )
    os_path_check(out_dir, n=5)

    
            

def image_check():
    """     python prepro.py  image_check 

          image white color padded
    
    """    
    #print( 'nf files', len(glob.glob("/data/workspaces/noelkevin01/img/data/fashion/train_nobg_256/*")) )
    nmax =  100000
    global xdim, ydim
    xdim= 64
    ydim= 64

    log("### Load  ##################################################")
    # fname    = f"/img_all{tag}.cache"
    # fname    = f"/img_fashiondata_64_64-100000.cache"    
    # fname = "img_train_nobg_256_256-100000.cache"
    fname = "img_train_r2p2_40k_nobg_256_256-100000.cache"
    fname = "img_train_r2p2_40k_nobg_256_256-100000.cache"    
    
    log('loading', fname)
    
    import diskcache as dc
    db_path = data_train + fname
    cache   = dc.Cache(db_path)
    
    lkey = list(cache)
    print('Nimages', len(lkey) )

    ### key check:
    #df = pd_read_file("/data/workspaces/noelkevin01/img/data/fashion/csv/styles_df.csv" )
    #idlist = df['id']
        
    log('### writing on disk  ######################################')
    dir_check = data_train + "/zcheck/"
    os.makedirs(dir_check, exist_ok=True)
    for i, key in enumerate(cache) :
        if i > 10: break
        img = cache[key]
        img = img[:, :, ::-1]
        print(key)
        key2 = key.split("/")[-1]
        cv2.imwrite( dir_check + f"/{key2}"  , img)            
    
   
    
def create_train_parquet():
    
    import cv2, gc
    #### List of images (each in the form of a 28x28x3 numpy array of rgb pixels)  ############
    #### data_dir = pathlib.Path(data_img)
    nmax =  100000

    log("### Sub-Category  #################################################################")
    # tag   = "-women_topwear"
    tag = "-alllabel3_nobg"
    tag = tag + f"-{xdim}_{ydim}-{nmax}"
    log(tag)


    df    = pd.read_csv( data_label  +"/preproed_df.csv")
    # flist = set(list(df[ (df['subCategory'] == 'Watches') & (df.gender == 'Women')   ]['id'].values))
    # flist = set(list(df[ (df['subCategory'] == 'Topwear') & (df.gender == 'Women')   ]['id'].values))
    flist = set(list(df['id'].values))
    log('Label size', len(flist))
    log(tag)


    log("#### Train  ######################################################################")
    image_list = sorted(list(glob.glob( data_dir + '/train_nobg/*.*')))
    image_list = image_list[:nmax]
    log('Size Before', len(image_list))

    ### Filter out
    image_list = [ t for  t in image_list if  int(t.split("/")[-1].split(".")[0]) in flist  ]
    log('Size After', len(image_list))
    
    images, labels = prepro_images_multi(image_list)

    df2        = pd.DataFrame(labels, columns=['uri'])
    df2['id']  = df2['uri'].apply(lambda x : int(x.split("/")[-1].split(".")[0])  ) 
    df2        = df2.merge(df, on='id', how='left')    
    df2['img'] = images
    df2.to_parquet( data_train + f"/train_{tag}.parquet" )    
    log(df2)    
        

    log("#### Test  #######################################################################")
    image_list = sorted(list(glob.glob( data_dir + '/test_nobg/*.*')))
    image_list = image_list[:nmax]
    # log( image_list )

    image_list = [ t for  t in image_list if  int(t.split("/")[-1].split(".")[0]) in flist ]
    
    log('Size After', len(image_list))
    images, labels = prepro_images_multi(image_list)
    log(str(images)[:100]  )
    
     
    df2        = pd.DataFrame(labels, columns=['uri'])
    df2['id']  = df2['uri'].apply(lambda x : int(x.split("/")[-1].split(".")[0])  ) 
    df2['img'] = images
    df2        = df2.merge(df, on='id', how='left')    
    df2.to_parquet( data_train + f"/test_{tag}.parquet" )  


    log("#### Save train, test ###########################################################")
    # img = df2['img'].values 

    
    

def image_remove_bg(in_dir="", out_dir="", level=1):
    """ #### remove background
    
         source activate py38 &&  sleep 5 && python prepro.py   image_remove_bg  
    
    
        python prepro.py rembg  --in_dir  /data/workspaces/noelkevin01/img/data/bing/v4     --out_dir  /data/workspaces/noelkevin01/img/data/bing/v4_nobg &>> /data/workspaces/noelkevin01/img/data/zlog_rembg.py  &

        rembg  -ae 15 -p  /data/workspaces/noelkevin01/img/data/fashion/test2/  /data/workspaces/noelkevin01/img/data/fashion/test_nobg/  
        
        mkdir /data/workspaces/noelkevin01/img/data/fashion/train_nobg/  
        
    """    
    in_dir  = "/data/workspaces/noelkevin01/img/data/gsp/v1000k_clean/"
    out_dir = "/data/workspaces/noelkevin01/img/data/gsp/v1000k_clean_nobg/"

    
    fpaths = glob.glob(in_dir + "/*")
    log( str(fpaths)[:10] )
    for fp in fpaths : 
        if "." not in fp.split("/")[-1] :             
            fp_out = fp.replace(in_dir, out_dir)
            os.makedirs(fp_out, exist_ok=True)
            cmd = f"rembg   -p {fp}  {fp_out} "    #### no adjustment -ae 15
            log(cmd)
            try :
               os.system( cmd )
            except : pass         


def image_create_cache():
    #### source activate py38 &&  sleep 13600  && python prepro.py   image_remove_bg     && python prepro.py  image_create_cache  
    #### List of images (each in the form of a 28x28x3 numpy array of rgb pixels)  ############
    ####   sleep 56000  && python prepro.py  image_create_cache       
    import cv2, gc, diskcache
    nmax =  1000000 #  0000
    global xdim, ydim
    xdim= 256
    ydim= 256

    log("### Sub-Category  ################################################################")
    # in_dir   = data_dir + '/fashion_data/images/'
    # in_dir   = data_dir + "/train_nobg_256/"
    in_dir   = data_dir + "/../gsp/v1000k_clean_nobg/"
    
    image_list = sorted(list(glob.glob(  f'/{in_dir}/*/*.*')))
    image_list = [  t  for t in image_list if "/-1/" not in t  and "/60/" not in t   ]
    log('N images', len(image_list))
    # tag   = "-women_topwear"
    tag      = "train_r2p2_1000k_clean_nobg"
    tag      = f"{tag}_{xdim}_{ydim}-{nmax}"
    # db_path  = data_train + f"/img_{tag}.cache"
    db_path = "/dev/shm/train_npz/small/" + f"/img_{tag}.cache"
    
    log(in_dir)
    log(db_path)
                
    def prepro_image2b(image_path):
        try :
            fname      = str(image_path).split("/")[-1]    
            id1        = fname.split(".")[0]
            # print(image_path)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            # image = util_image.image_resize_pad(image, (xdim,ydim), padColor=255)
            image = util_image.image_center_crop(image, (245, 245))

            # image = image.astype('float32')
            return image, image_path 
            #return [1], "1"
        
        except :
            try :
               # image = image.astype('float32')
               # cache[ fname ] =  image        ### not uulti thread write
               return image, image_path
               # return [1], "1"
            except :
               return [],""
    
    log("#### Converting  ############################################################")
    image_list = image_list[:nmax]
    log('Size Before', len(image_list))

    import diskcache as dc
    #  from diskcache import FanoutCache  ### too much space
    # che = FanoutCache( db_path, shards=4, size_limit=int(60e9), timeout=9999999 )
    cache = dc.Cache(db_path, size_limit=int(100e9), timeout=9999999 )

    log("#### Load  #################################################################")       
    images, labels = prepro_images_multi(image_list, prepro_image= prepro_image2b, npool=32 )
    
    
    import asyncio
    async def set_async(key, val):
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, cache.set, key, val)
        result = await future
        return result

    # asyncio.run(set_async('test-key', 'test-value'))

    
    log(str(images)[:500],  str(labels)[:500],  )
    log("#### Saving disk  #################################################################")           
    for path, img in zip(labels, images) : 
       key = os.path.abspath(path)
       key = key.split("/")[-1] 
       cache[ key ] =  img
       # asyncio.run(set_async( key , img ))   ##only python 3.7
    
    
    print('size cache', len(cache),)
    print( db_path )
    
    for i,key in enumerate(cache):
       if i > 3 : break 
       x0 = cache[key] 
       cv2.imwrite( data_train + f"/check_{i}.png", x0 )
       print(key, x0.shape, str(x0)[:50]  )
             
        
def os_path_check(path, n=5):
    from utilmy import os_system
    print('top files', os_system( f"ls -U   '{path}' | head -{n}") )
    print('nfiles', os_system( f"ls -1q  '{path}' | wc -l") )
               


def image_face_blank(in_dir="", level = "/*", 
                     out_dir=f"", npool=30):
    """  Remove face

     python prepro.py  image_face_blank
     
     python prepro.py  image_face_blank  --in_dir img/data/fashion/test_nobg   --out_dir img/data/fashion/test_nobg_noface

     python prepro.py  image_face_blank  --in_dir img/data/fashion/train_nobg   --out_dir img/data/fashion/train_nobg_noface


      five elements are [xmin, ymin, xmax, ymax, detection_confidence]

    """
    import cv2, glob
    import face_detection

    #in_dir  = "/data/workspaces/noelkevin01/" + in_dir
    #out_dir = "/data/workspaces/noelkevin01/" + out_dir
    npool    = 30
    in_dir   = "/data/workspaces/noelkevin01/img/data/gsp/v70k_clean_nobg/"
    out_dir  = "/data/workspaces/noelkevin01/img/data/gsp/v70k_clean_nobg_noface/"
    fpaths   = glob.glob(in_dir + "/*/*" )
    
    # fpaths   = [  t for t in fpath if "/-1" not in fpaths ]
    # fpaths   = fpaths[:60]
    
    detector = face_detection.build_detector( "RetinaNetMobileNetV1", 
                            confidence_threshold=.5, nms_iou_threshold=.3)

    log(str(fpaths)[:60])

    def myfun(fp):
      try :
          log(fp)  
          img   = cv2.imread(fp)
          im    = img[:, :, ::-1]
          areas = detector.detect(im)

          ### list of areas where face is detected.
          for (x0, y0, x1, y1, proba) in areas:  
             x0,y0, x1, y1     = int(x0), int(y0), int(x1), int(y1)
             img[y0:y1, x0:x1] = 0

          fout = fp.replace(in_dir, out_dir)    
          os.makedirs( os.path.dirname(fout), exist_ok=True)
          cv2.imwrite( fout, img )
      except : pass        


    #for fp in fpaths :
    #  myfun(fp)

    from multiprocessing.dummy import Pool    #### use threads for I/O bound tasks
    pool = Pool(npool) 
    res  = pool.map(myfun, fpaths)      
    pool.close()
    pool.join()     
        
    
    
def image_text_blank(in_dir, out_dir, level="/*"):
    """
        Not working well
        python prepro.py  image_text_blank  --in_dir img/data/fashion/ztest   --out_dir img/data/fashion/ztest_noface
        
    
    """
    import cv2, glob
    from ztext_detector import detect_text_regions
    
    in_dir  = "/data/workspaces/noelkevin01/" + in_dir
    out_dir = "/data/workspaces/noelkevin01/" + out_dir

    fpaths  = glob.glob(in_dir + level )
    log(str(fpaths)[:60])
    for fp in fpaths :
      try :
          log(fp)  
          img   = cv2.imread(fp)
          im    = img[:, :, ::-1]
                        
          areas = detect_text_regions(img)
                                       
          ### list of areas where is detected.
          for (x0, y0, x1, y1) in areas:  
             x0,y0, x1, y1     = int(x0), int(y0), int(x1), int(y1)
             img[y0:y1, x0:x1] = 0

          fout = fp.replace(in_dir, out_dir)    
          os.makedirs( os.path.dirname(fout), exist_ok=True)
          cv2.imwrite( fout, img )
      except : pass
    

    
def model_deletes(dry=0):
    """  ## Delete files on disk
        python prepro.py model_deletes  --dry 0
        
    """
    
    path0  = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/*"
    fpath0 = glob.glob(path0)
    # print(fpath0)
    
    for path in fpath0 :
        print("\n", path)
        try :
            fpaths = glob.glob(path + "/best/*" )
            fpaths = [ t for t in fpaths if  'epoch_' in t ]    
            fpaths = sorted(fpaths, key=lambda x: int(x.split("/")[-1].split('_')[-1].split('.')[0]), reverse=True)
            # print(fpaths)
            fpaths = fpaths[4:]  ### Remove most recents
            fpaths = [ t for t in fpaths if  int(t.split("/")[-1].split('_')[-1]) % 10 != 0  ]  ### _10, _20, _30

            for fp in fpaths :
                cmd = f"rm -rf '{fp}' "
                print(cmd)
                if dry > 0 :
                  os.system( cmd)
            # sys.exit(0)
        except Exception as e:
            print(e)
    

def image_save():
    ##### Write some sample images  ########################
    import diskcache as dc
    db_path = "/data/workspaces/noelkevin01/img/data/fashion/train_npz/small/img_train_r2p2_70k_clean_nobg_256_256-100000.cache"
    cache   = dc.Cache(db_path)
    print('Nimages', len(cache) )

    log('### writing on disk  ######################################')
    dir_check = out_dir + f"/{xname}/"
    os.makedirs(dir_check, exist_ok=True)
    for i, key in enumerate(img_list) :
        if i > 10: break       
        img = cache[key]
        img = img[:, :, ::-1]
        key2 = key.split("/")[-1]
        cv2.imwrite( dir_check + f"/{i}_{key2}"  , img)            
    log( dir_check ) 

    
def topk_predict():
    #### wrapper :      python prepro.py topk_predict 
    
    dname = "m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000.cache/best/best_good_epoch_313"
        
    cmd = f" python train9pred.py '{dname}'  "
    os.system(cmd)

    
    
def topk(topk=100, dname=None, pattern="df_*", filter1=None):
    """  python prepro.py  topk    |& tee -a  /data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_epoch_261/topk/zzlog.py
    
         https://brandavenue.rakuten.co.jp/item/AS5238/?s-id=brn_top_history
         1  as5238-06_1-2.png
         
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
    """   /user/scoupon/zexport/z/fashion_emb_100k  
                     id gender masterCategory subCategory  ... masterCategory_pred subCategory_pred articleType_pred  baseColour_pred
0     cn3357-01_1-11.png  women        apparel     topwear  ...                   1                1               32                4
1      cs6481-01_1-4.png   kids          shoes       shoes  ...                   5               20              151                7


     hdfs dfs -put  /data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_best_good_epoch_313/fashion_emb_500k/                /user/scoupon/zexport/z/

    """
    dir_in  = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_best_good_epoch_313/*.parquet" 
    dir_out = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_best_good_epoch_313/fashion_emb_500k/"
    
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
 
    
    
    
def data_add_onehot(dfref, img_dir, labels_col) :      
    """
       id, uri, cat1, cat2, .... , cat1_onehot

    """
    import glob
    fpaths   = glob.glob(img_dir )
    fpaths   = [ fi for fi in fpaths if "." in fi.split("/")[-1] ]
    log(str(fpaths)[:100])

    df         = pd.DataFrame(fpaths, columns=['uri'])
    log(df.head(1).T)
    df['id']   = df['uri'].apply(lambda x : x.split("/")[-1].split(".")[0]    ) 
    df['id']   = df['id'].apply( lambda x: int(x) )
    df         = df.merge(dfref, on='id', how='left') 

    # labels_col = [  'gender', 'masterCategory', 'subCategory', 'articleType' ]
 
    for ci in labels_col :  
      dfi_1hot           = pd.get_dummies(df, columns=[ci])  ### OneHot
      dfi_1hot           = dfi_1hot[[ t for t in dfi_1hot.columns if ci in t   ]]  ## keep only OneHot      
      df[ci + "_onehot"] = dfi_1hot.apply( lambda x : ','.join([   str(t) for t in x  ]), axis=1)
      #####  0,0,1,0 format   log(dfi_1hot)
        
    return df



def test():
    """
       python prepro.py test
       
    """
    dfref      = pd.read_csv( data_label  +"/prepro_df.csv")
    img_dir    = data_dir + '/train/*'
    labels_col = [ 'gender', 'masterCategory', 'subCategory', 'articleType' ]
    
    df = data_add_onehot(dfref, img_dir, labels_col)     
    log(df.head(2).T)
    
def unzip(in_dir, out_dir):
    #!/usr/bin/env python3
    import sys
    import zipfile
    with zipfile.ZipFile(in_dir, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

        
def gzip():
    #  python prepro.py gzip 
    import sys
    
    # in_dir  = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9a_g6_-img_train_nobg_256_256-100000.cache/check"    
    
    in_dir = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_epoch_261/topk"
    
    name    = "_".join(in_dir.split("/")[-2:])
    
    cmd = f"tar -czf /data/workspaces/noelkevin01/{name}.tar.gz   '{in_dir}/'   "
    print(cmd)
    os.system(cmd)            

        
def predict(name=None):
    ###   python prepro.py  predict 
    if name is None :
       name = "m_train9b_g3_-img_train_r2p2_70k_clean_nobg_256_256-100000.cache/best/epoch_90"
    os.system( f" python train9pred.py  '{name}'   ")
    
                
def folder_size():
    os.system(" du -h --max-depth  13   /data/  | sort -hr  > /home/noelkevin01/folder_size.txt  ")
        

def gpu_usage(): 
    
   cmd = "nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv"

   from utilmy import os_system    
   res = os_system(cmd)
   print(res)
        
   ## cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
   ## cmd2= " nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv  "
    
    
def gpu_free():  
    cmd = "nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv  "
    from utilmy import os_system    
    ss = os_system(cmd)

    # ss   = ('pci.bus_id, utilization.gpu [%]\n00000000:01:00.0, 37 %\n00000000:02:00.0, 0 %\n00000000:03:00.0, 0 %\n00000000:04:00.0, 89 %\n', '')
    ss   = ss[0]
    ss   = ss.split("\n")
    ss   = [ x.split(",") for x in ss[1:] if len(x) > 4 ]
    print(ss)
    deviceid_free = []
    for ii, t in enumerate(ss) :
        if  " 0 %"  in t[1]:
            deviceid_free.append( ii )
    print( deviceid_free )        
    return deviceid_free
            
            
        
def down_ichiba()  :
    """  python prepro.py down_ichiba
    
     TODO :
        Map queries ---> Categories        
        Cateogries  ---> Generate Ichiba Queries
        
    
      https://search.rakuten.co.jp/search/mall/-/566028/tg1003435/?max=4000&min=3000
      
    ### reverse Images  
    https://search.rakuten.co.jp/search/mall/blue/566028/tg1004015/?max=4000&min=3000
    
    
    get random images
       https://search.rakuten.co.jp/search/mall/blue/566028/tg1004015/?max=9000&p=4
       
       from generating URL and queries
       
       
    
    
    """
    dir_data = "/data/workspaces/noelkevin01/img/data/"    
    
    def down2(args):        
        down_page(query= args[0],  out_dir= args[1], genre_en= args[2],
                  id0=   args[3],  cat=args[4],      npage= args[5] )
        
    df   = pd.read_csv( dir_data  +"/df_ichi_genres2.csv" )
    log(df)
    tag0 = "women"
                            
    colors = [ 'blue','red','green', 'white','beige','grey','black','pink','orange','yellow','brown','purple', ]      #   white ### red, blue, green
    for color in colors :    
        ll    = []
        for ii,x in df.iterrows():    
            if ii > 100 : break                    
            query    = x['genre_ja'].replace(">", " ")   + " " + color
            genre_en = x['genre_en'].replace(">", "_").replace("'","").replace("/","")   + "_" + color
            id0      = x['genre_path'] + color        
            cat      = x['category']   + "-" + color        
            out_dir  =  f"/{tag0}/{color}/{genre_en}"        
            ll.append((query, out_dir, genre_en, id0, cat, 3  )    )

        run_multiprocess(down2, ll, npool= 30 )     

 

def down_page(query, out_dir="query1", genre_en='', id0="", cat="", npage=1) :
    """
        python prepro.py down_page  'メンズファッション+トップス+ポロシャツ'    --out_dir men_fashion_topshirts_blue  


    """
    import time, os, json, csv, requests, sys, urllib
    from bs4 import BeautifulSoup as bs
    from urllib.request import Request, urlopen
    import urllib.parse


    path = "/data/workspaces/noelkevin01/img/data/rakuten/" + out_dir + "/"
    os.makedirs(path, exist_ok=True)
    # os.chdir(path)

    query2     = urllib.parse.quote(query, encoding='utf-8')
    url_prefix = 'https://search.rakuten.co.jp/search/mall/' + query2
    ### https://search.rakuten.co.jp/search/mall/%E3%83%A1%E3%8384+blue+/?p=2
    print(url_prefix)
    print(path)

    csv_file   = open( path + 'ameta.csv','w',encoding="utf-8")
    csv_writer = csv.writer(csv_file, delimiter='\t')
    csv_writer.writerow(['path', 'id0', 'cat', 'genre_en', 'image_name', 'price','shop','item_url','page_url',  ])

    page  = 1
    count = 0
    while page < npage+1 :
        try:
            rakuten_url = url_prefix  + f"/?p=+{page}"
            req    = Request(url=rakuten_url)
            source = urlopen(req).read()
            soup   = bs(source,'lxml')

            print('page', page, str(soup)[:5], str(rakuten_url)[-20:],  )

            for individual_item in soup.find_all('div',class_='searchresultitem'):
                count += 1
                save = 0
                shopname     = 'nan'
                count_review = 'nan'

                for names in individual_item.find_all('div',class_='title'):
                    product_name = names.h2.a.text
                    break

                for price in individual_item.find_all('div',class_='price'):
                    product_price = price.span.text
                    product_price = product_price .replace("円", "").replace(",", "") 
                    break
                
                for url in individual_item.find_all('div',class_='image'):
                    product_url = url.a.get('href')
                    break

                for images in individual_item.find_all('div',class_='image'):
                    try:
                        product_image = images.a.img.get('src')
                        urllib.request.urlretrieve(product_image, path + str(count)+".jpg")
                        # upload_to_drive(str(count)+'.jpg')
                        count += 1
                        break
                    except:
                        save = 1
                        print(product_url + " Error Detected")
                    
                for simpleshop in individual_item.find_all('div',class_='merchant'):
                    shopname = simpleshop.a.text
                    break

                for review in individual_item.find_all('a',class_='dui-rating-filter'):
                    count_review = review.text

                if save == 0:
                    csv_writer.writerow([str(count)+'.jpg', id0, cat, genre_en,  product_name, product_price, shopname, product_url, rakuten_url, ])

        except Exception as e :
            print(e)
            time.sleep(2)
            continue

        page += 1

    print("Success", page-1, count)


"""
tar -zcf /data/workspaces/noelkevin01/img/data/rakuten/women.tar.gz  /data/workspaces/noelkevin01/img/data/rakuten/women/


cp -R /data/workspaces/noelkevin01/img/data/fashion/train_npz/small/img_train_nobg_256_256-100000.cache/    /dev/shm/




"""        

def check_tf():
    #### python prepro.py check_tf 
    import tensorflow as tf
    print( tf.config.list_physical_devices())


if __name__ == "__main__":
    import fire
    fire.Fire()

    

    
    
    

    
"""

1. You are using nvidia-gpu
2. You are using conda environment (Anaconda)
Step I: Find out if the tensorflow is able to see the GPU
Command:

import tensorflow as tf
print( tf.config.list_physical_devices())



$ nvcc -V


conda install cudnn=8.2.1=cuda11.3_0
 
 
$ conda install tensorflow-gpu


 linux-64/cudnn-8.2.1-cuda11.3_0.tar.bz2
 
 
"""
    






"""

https://drive.google.com/file/d/1Jf2XOJb078Mu75oUCJjBfxM36TGZ8SFv/view?usp=sharing

gdown 



gdown --id  1Jf2XOJb078Mu75oUCJjBfxM36TGZ8SFv   -O /data/workspaces/noelkevin01/img/data/fashion/fashion_data.zip


unzip -o -qq  /data/workspaces/noelkevin01/img/data/fashion/fashion_data.zip  -d  /data/workspaces/noelkevin01/img/data/fashion/



    subCategory
1   Topwear
2   Bottomwear
3   Watches
4   Bottomwear
5   Topwear
6   Topwear
7   Topwear
8   Topwear
9   Socks
10  Watches
11  Shoes
12  Belts
13  Flip Flops
14  Bags
15  Flip Flops
16  Topwear


"""


"""
Resize((384, 384), interpolation=Image.BICUBIC),
CenterCrop((224, 224)),
ToTensor(),
Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),


pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).resize((self.opt.load_size, self.opt.load_size), resample=Image.NEAREST)
params = get_params(self.opt, pose.size)
transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
tensors_dist = 0
e = 1
for i in range(len(joints)):
im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
im_dist = np.clip((im_dist / 3), 0, 255).astype(np.uint8)
tensor_dist = transform_img(Image.fromarray(im_dist))
tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
e += 1

"""


def prepro_images2(image_paths):
  images = []
  original_first_image = None
  for i in range(len(image_paths)):
    if i > nmax : break
                
    image_path = image_paths[i]
    fname      = str(image_path).split("/")[-1]    
    id1        = fname.split(".")[0]
    
    if (i+100) % 100 == 0: print(fname, id1)
    
    image = matplotlib.image.imread(image_path)

    if images == []:
        temp = (image / 255)
        original_first_image = temp.astype('float32')
    resized_image = cv2.resize(image, dsize=(xdim, ydim), interpolation=cv2.INTER_CUBIC)

    
    if resized_image.shape == (xdim, ydim, cdim):
        resized_image = resized_image / 255
        images.append(resized_image.astype('float32'))
  return images, original_first_image















#labels_dir = pathlib.Path(data_label +"/styles.csv")
#names      = ["id", "gender", "masterCategory", 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName']
#labels     = pd.read_csv(labels_dir, names=names, skiprows=1)
#labels.head(10)


### '''prepro Data'''
#List of paths to each image file











