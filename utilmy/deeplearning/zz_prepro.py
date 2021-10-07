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













