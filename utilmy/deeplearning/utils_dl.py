# -*- coding: utf-8 -*-
import os
os.environ['MPLCONFIGDIR'] = "/tmp/"

import io
import cv2
import numpy as np
import tifffile.tifffile
from typing import Union
from pathlib import Path
import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers
from tensorflow.python.keras.utils.data_utils import Sequence    
import matplotlib.pyplot as plt
import scipy.stats ; from scipy.stats import norm
from PIL import Image ; from tqdm import tqdm
from sklearn import manifold
from utilmy import pd_read_file
# from tf_sprinkles import Sprinkles
from madgrad import MadGrad
from sklearn.preprocessing import OneHotEncoder
from albumentations.core.transforms_interface import ImageOnlyTransform
from skimage import morphology
from sklearn.metrics import accuracy_score
import pandas as pd, numpy as np
from pprint import pprint as print2
from box import Box
import sys, glob, time, gc, copy, math, string, json, logging, functools, random, matplotlib, scipy, h5py, yaml
from ztext_detector import detect_text_regions
import diskcache as dc
from importall import *
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, Resize,
    GridDistortion, ElasticTransform,OpticalDistortion,
    ToGray,
    Equalize, # histogram (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5)
    Cutout, # (num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5
)

  

#####################################################################################
#################################SETTINGS############################################
#####################################################################################

cc = Box({})

### Output naming  ##################################################
#cc.dname       = 'fashion_64_44k'
#cc.dname = "img_fashiondata_64_64-100000.cache"
cc.dname  = "img_train_nobg_256_256-100000.cache"
#cc.dname  = "img_train_r2p2_40k_nobg_256_256-100000.cache"
cc.tag   = "train10a_g2_"
#cc.tag   = "ztest"
tag = cc.tag


# cc.model_reload_name =  "m_train8a_gpu3_-img_train_nobg_256_256-100000.cache/best/epoch_20/"
# cc.model_reload_name = "m_tt_64nobg_5class_300emb_2-img_train_nobg64_64_64-100000.cache"
# "/m_tt_64nobg_5class-img_train_nobg64_64_64-100000.cache/best"
cc.model_reload_name = ""
# cc.model_reload_name = "m_train9a_g6_-img_train_nobg_256_256-100000.cache/best/good_epoch_50/"
epoch0       = 1

cc.gpu_id  = 1    ###3  1,2,3

### Epoch

num_epochs        = 200  # 80 overfits
cc.compute_mode   = "custom" #  "gpu"  # "tpu"  "cpu_only"
cc.n_thread     = 0   #### 0 neans TF optimal

### Input file
image_size     = 256   ### 64
xdim           = 256
ydim           = 256
cdim           = 3

cc.img_suffix  = '.png'
### cc.img_suffix  = '.jpg'  ### fashion dataset

cc.verbosity = 1

#### Data Size
nmax           = 100000

#### Model Parameters
n_filters      = 12  #  12  # Base number of convolutional filters
latent_dim     = 512  # Number of latent dimensions


#### Training parmeter
batch_size     = 64  #8 * cc.n_thread 
learning_rate  = 5e-4

cc.lrate_mode    = 'random'   ##   'step' random
cc.learning_rate = learning_rate
cc.lr_actual     = cc.learning_rate 

cc.latent_dim = latent_dim

cc.patience  = 600
cc.kloop     = int(round( int( 44000.0 / batch_size  /10.0 ) /100, 0) * 100)  # 10 loop per epoch
# cc.kloop     = 2
cc.kloop_img = 2* cc.kloop


#### Paths data  #####################
cc.root = '/img/data/'
cc.root3   = ""
cc.data_train = "/img/data/fashion/train_npz/small/"

cc.path_img_all   = cc.root + "fashion/train_nobg_256/"
cc.path_img_train = cc.root + "fashion/train_nobg_256/"
cc.path_img_test  = cc.root + "fashion/train_nobg_test/"


cc.path_label_raw   = cc.root  + 'fashion/csv/styles_df_normalize.csv'    ### styles_raw.csv
#cc.path_label_raw   = cc.root  + 'fashion/csv/fashion_raw_clean.parquet'    ### styles_raw.csv

cc.path_label_train = cc.root  + 'fashion/csv/styles_train.csv'
cc.path_label_test  = cc.root  + 'fashion/csv/styles_test.csv'
cc.path_label_val   = cc.root  + 'fashion/csv/styles_test.csv'

cc.model_dir     = "/data/workspaces"
cc.code_source   = "/img/dcf_vae/"
# cc.model_reload_name = cc.model_dir +  cc.model_reload_name 


def np_remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def clean1(ll):
    v = [ t for t in ll if len(t) > 1 ]
    return  np_remove_duplicates(v)


cc.labels_map = {
 'gender'         : clean1( [ 'aaother', 'men', 'women',  'kids', 'unisex'   ] ),

 'masterCategory' : [ 
         'aaother','apparel','festival','uniform','accessories','shoes','watch','bags'
                    ],


 'subCategory'    : clean1([ 
 
       'aaother','topwear','bottomwear','dress','innerwear','loungewear and nightwear','office','onepiece','rainwear','set','suits','uniform','unisex','coat','sports','skirt','maternity','swimwear','festival','scarves','shoes','socks','eyewear','bags','belts','wallets','watch','gloves','headwear','jewellery','ties'

                ]) ,

 'articleType'    :  clean1([   
        'aaother', 'bare tops', 'bath robe', 'blouses', 'boxers', 'bra', 'camisoles', 'cardigan', 'culottes', 'dresses', 'hoodies', 'innerware', 'innerwear vests', 'jackets', 'knits', 'knits camisole', 'knits hoodies', 'lounge pants', 'lounge shorts', 'night suits', 'nightdress', 'shirts', 'shirts_polo', 'shorts', 'skirts', 'stockings', 'suspenders', 'sweaters', 'sweatshirts', 'tanktops', 'tops', 'trunk', 'tshirts', 'tunics', 'vest', 'three-quarter sleeve shirt', 'short sleeve shirt', 'long sleeve shirt', '5-9 sleeve knit', 'sleeveless knit / vest', 'short sleeve knit', 'long sleeve knit', 'u neck cut and sew', 'v-neck cut and sew', 'cut and sew other', 'kids cut and sew', 'turtleneck cut and sew', 'design cut and sew', 'sleeveless cut and sew', 'parker', 'sleeveless / camisole shirt', 'coat', 'rain jacket', 'short coat', 'sukajan', 'bal collar coat', 'down jacket', 'duffle coat', 'chester coat', 'tailored jacket', 'denim jacket', 'trench coat', 'nylon jacket', 'collarless jacket', 'half coat', 'pea coat', 'fur coat', 'blouson', 'best', 'poncho', 'mountain parka', 'military jacket', 'mouton coat', 'mod coat', 'riders jacket / leather jacket', 'long coat', 'coveralls', 'track pants', 'trousers', 'leggings', 'pants', 'jeans', 'jeggings', 'short pants', 'cargo pants', 'kids pants', 'cropped / odd length pants', 'salopette / all-in-one', 'short jeans', 'sweat pants', 'skinny jeans', 'straight jeans', 'slacks / dress pants', 'chinos', 'full length', 'wide / buggy pants', '7-9 minutes length jeans', '2way skirt', 'jumper skirt', 'tight skirt', 'denim skirt', 'flare skirt', 'pleated skirt / gathered skirt', 'mini skirt', 'long skirt', 'trapezoidal skirt / cocoon skirt', 'tracksuits', 'suits', 'ensemble', 'formal', 'overall', 'set', 'kimono', 'yutaka', 
        'swimwear', 
        '5-9 sleeve dress', 'kids dress', 'camisole dress', 'shirt dress', 'knit dress', 'sleeveless dress', 'bare dress', 'long dress / maxi dress', 'one piece other', 'short sleeve dress', 'long sleeve dress', 'handbags', 'business bag', 'backpacks', 'trolley bag', 'bags', 'basket bag', 'waist pouch', 'eco bag / sub bag', 'kids bag', 'carry bag', 'clutch bag', 'shoulder bag', 'tote bag', 'party bag', 'boston bag', 'sports', 'flats', 'flip flops', 'casual shoes', 'formal shoes', 'heels', 'sandals', 'sports sandals', 'sports shoes', 'short boots / booties', 'sneakers / slip-ons', 'ballet shoes', 'pumps', 'middle boots', 'mouton boots', 'moccasins', 'rain boots', 'knee-high boots', 'running shoes', 'kids shoes', 'ties', 'belts', 'scarves', 'socks', 'caps', 'gloves', 'jewellery', 'earrings', 'sunglasses', 'cufflinks', 'bracelet', 'watch', 'wallets'

   ]),


 'baseColour' : [   'aaother','white','beige','grey','black','blue','red','pink','orange','yellow','brown','purple','green',
                ]
}


#cc.labels_cols  = [ 'gender',  'masterCategory', 'subCategory',  'articleType',  'baseColour', 'tags', 'price' ]
cc.labels_cols  = [ 'gender',  'masterCategory', 'subCategory',  'articleType',  'baseColour',  ]

cc.labels_count = { ci : len(cc.labels_map[ci]) for ci in cc.labels_map }

cc.labels_onehotdim = sum([  cc.labels_count[ci] for ci  in  cc.labels_cols ])



#### transformation
cc.cutout_color = 0
####

###############################################
##########disk caching#########################
db_path    = cc.data_train + f"/{cc.dname}"
db_path    = "/dev/shm/img_train_nobg_256_256-100000.cache"
# db_path    = "/dev/shm/img_train_nobg_256_256-100000.cache"


img_cache  = dc.Cache(db_path )
print2(cc )
print(db_path , len(img_cache))
if len(list(img_cache)) < 20000: 
   sys.exit(0)


##### Output
model_dir     = cc.model_dir
model_dir2    = model_dir + f"/m_{tag}-{cc.dname}/"
cc.model_dir2 = model_dir2
os.makedirs(model_dir2, exist_ok=True)


time.sleep(3) 


################################################################################
##############################LOG###############################################
################################################################################

def log(*s):
    print(*s, flush=True)


def log3(*s):
    if cc.verbosity >= 3:   
        print(*s, flush=True)
        with open(cc.model_dir2 + "/debug.py", mode='a') as fp:
            fp.write(str(s) + "\n")

def log2(*s): 
    print(*s, flush=True)
    with open(cc.model_dir2 + "/log.py", mode='a') as fp:
        fp.write(str(s) + "\n")




#######################################################################################
##################################PREPROCESSING########################################
#######################################################################################


def prepro_images(image_paths, nmax=10000000):
  images = [] 
  for i in range(len(image_paths)):
    if i > nmax : break                
    image =  prepro_image(image_paths[i])
    images.append(image)        
  return images


def image_center_crop(img, dim):
 	"""Returns center cropped image
 	Args:
 	img: image to be center cropped
 	dim: dimensions (width, height) to be cropped
 	"""
 	width, height = img.shape[1], img.shape[0]

 	# process crop width and height for max available dimension
 	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
 	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
 	mid_x, mid_y = int(width/2), int(height/2)
 	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
 	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
 	return crop_img


def image_resize_pad(img,size=(256,256), padColor=0 ):
     """
       resize and keep into the target Box
     
     """

     h, w = img.shape[:2]
     sh, sw = size

     # interpolation method
     if h > sh or w > sw: # shrinking image
         interp = cv2.INTER_AREA

     else: # stretching image
         interp = cv2.INTER_CUBIC

     # aspect ratio of image
     aspect  = float(w)/h 
     saspect = float(sw)/sh

     if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
         new_h = sh
         new_w = np.round(new_h * aspect).astype(int)
         pad_horz = float(sw - new_w) / 2
         pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
         pad_top, pad_bot = 0, 0

     elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
         new_w = sw
         new_h = np.round(float(new_w) / aspect).astype(int)
         pad_vert = float(sh - new_h) / 2
         pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
         pad_left, pad_right = 0, 0

     # set pad color
     if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
         padColor = [padColor]*3

     # scale and pad
     scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
     scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

     return scaled_img
   


def prepro_image0(image_path):
    mean   = [0.5]
    std    = [0.5]  
    try :
        fname      = str(image_path).split("/")[-1]    
        id1        = fname.split(".")[0]
        # print(image_path)

        image = image_read(image_path)
        image = image_resize_pad(image, (xdim,ydim), padColor=0)
        image = image_center_crop(image, (xdim,ydim))
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



def pd_get_dummies(df, cols_cat, cat_dict:dict, only_onehot=True):
   """ dfi_onehot = pd_get_dummies( df, cols_cat = ['articleType'  ], cat_dict= cc.labels_map, only_onehot= False)
      dfi_onehot.sum()
      dfi_onehot.dtypes
   """ 
   dfall      =  None
   #cols_cat   = list(cat_dict.keys() )
   cols_nocat = [ t for t in df.columns if t not in cols_cat   ]

   for  coli in cols_cat :  # cat_dict.keys() :
     if coli not in cat_dict : continue
     # cat_list = [ coli + "_" + ci for ci in cat_dict[coli] ]
     cat_list = [ ci for ci in cat_dict[coli] ]
     df1      = df[coli].fillna('aaother')
     dfi      = pd.get_dummies( df1.astype(pd.CategoricalDtype(categories= cat_list )))

     dfi.columns = [ coli + "_" + ci for ci in dfi.columns ]
     dfall    = dfi if dfall is None else pd.concat((dfall, dfi))
 
   print(df[ cols_nocat])
   print(dfall)

   if not only_onehot:
      dfall = pd.concat((df[cols_nocat], dfall), axis=1)
      for ci in dfall.columns :
        if ci not in cols_nocat :
            dfall[ci] = dfall[ci].astype('int8')

   return dfall 



###############################################################################
#########Code for generating custom data from the Kaggle dataset###############

def label_get_data():
    
    #### Labels  ##############################
    #df          = pd.read_csv(cc.path_label_raw)  #, error_bad_lines=False, warn_bad_lines=False)
    df          = pd_read_file(cc.path_label_raw)  #, error_bad_lines=False, warn_bad_lines=False)
    
    #df          = df.dropna()
    df = df.fillna('')
    #for ci in cc.labels_cols :
    #        df[ci] = df[ci].str.lower()
    log(df)
    df['id']    = df['id'].astype('int')    
    # df['id']  = df['id'].apply(lambda x :  x.replace(".jpg", ".png").lower() )
    df        = df.drop_duplicates("id")
    
    ### Filter out dirty    
    df    = df[ -df['masterCategory'].isin([ 'aaother', 'accessories', 'watch', 'festival'  ]) ]
    df    = df[ -df['articleType'].isin([ 'aaother'  ]) ]
    df    = df[ -df['baseColour'].isin([  'aaother'  ]) ]
    
    df_img_ids  = set(df['id'].tolist())
    for img1 in df_img_ids : break    
    log('N ids', len(df_img_ids),  img1 )

    #### On Disk Available ########################################################################
    #flist    = glob.glob(cc.path_img_all + "/*.png")
    #flist    = [  t.split("/")[-1] for t in flist ]
    img_ids  = set([int(filename.split(".")[0]) for filename in list(img_cache) ])
    # img_ids  = set([filename for filename in list(img_cache) ])
        
    for img0 in img_ids : break
    log('N images', len(img_ids) , img0 );
    
    
    ##### Intersection  ###########################################################################
    available_img_ids = df_img_ids & img_ids
    print('Total valid images:', len(available_img_ids))
    df = df[df['id'].isin(available_img_ids)]
    print(df.nunique() )
    print(df.shape)
    if len(df) < 1000 : sys.exit(0)    
    time.sleep(10)    
    return df

# df = label_get_data()
# ### Fitlered   ##################################################
# df = df[  ['id'] + cc.labels_cols ]

#### Filter label values  #######################################
def pd_category_filter(df, category_map):
    def cat_filter(s, values):
        if s.lower() in values:
             return s.lower()
        return 'aaother'

    colcat = list(category_map.keys())
    cols   = ['id'] + list(category_map.keys())
    df     = df[cols]
    for col, values in category_map.items():
       df[col] = df[col].apply(cat_filter, args=(values,))
       # print(df[col].unique())
        
    # class_dict = {ci : df[ci].nunique() for ci in category_map }
    # colcat     = 
    return df

# df = pd_category_filter(df, cc.labels_map)
# log(df)
#### Split into  #################################################
# shuffled  = df.sample(frac=1)
# n         = df.shape[0]
# num_train = int(0.97 * n)
# num_val   = n - num_train

# df_train  = df.iloc[:num_train, :]
# df_val    = df.iloc[num_train:, :]

# df_train.to_csv(cc.path_label_train,  index=False)
# df_val.to_csv(  cc.path_label_val,    index=False)

# #### test
# # pathi       = cc.path_img_all  + "/10004.png"
# # image_dataset = [ np.array(  cv2.cvtColor( cv2.imread(pathi), cv2.COLOR_BGR2RGB)  ) ]

# for x in [ 'articleType', 'baseColour' ]:
#     print(df_train.groupby(x).agg({'id': 'count'})) ; time.sleep(4)


def image_load(pathi, mode='cache'):
    keyi =  pathi.split("/")[-1] #  f'{image_id}{self.img_suffix}'
    # image = np.array(Image.open(pathi).convert('RGB') )            
    # image = np.array(  cv2.cvtColor( cv2.imread(pathi), cv2.COLOR_BGR2RGB)  )  ### to RGB format, openCV iR BGR
    # image = tf.keras.preprocessing.image.img_to_array( tf.keras.preprocessing.image.load_img(pathi, color_mode='rgb') )
    # image = ThreadedReader(pathi)
    #### Not Thread sae
    # image = image_dataset[0]
    image = img_cache[keyi]
    return image
    


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



"""## Data loader"""
class RealCustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, label_path, class_dict,
                 split='train', batch_size=8, transforms=None, shuffle=False, img_suffix=".png"):
        self.image_dir              = image_dir
        # self.labels               = np.loadtxt(label_path, delimiter=' ', dtype=np.object)
        self.labels_map             = class_dict
        self.image_ids, self.labels = self._load_data(label_path)
        self.num_classes            = len(class_dict)
        self.batch_size             = batch_size
        self.transforms             = transforms
        self.seed                   = 12

        self.shuffle    = shuffle
        self.img_suffix = img_suffix
        
        #self.nmax            = int( len(self.image_ids) // 2 )
        #self.image_ids_small = self.image_ids[:self.nmax]
        #self.labels_small    = [  dfi.loc[:self.nmax, :] for dfi in self.labels ] 
        

    def _load_data(self, label_path):
        df   = pd.read_csv(label_path, error_bad_lines=False, warn_bad_lines=False)
        keys = ['id'] + list(self.labels_map.keys())
        df   = df[keys]

        # Get image ids
        df        = df.dropna()
        image_ids = df['id'].tolist()
        df        = df.drop('id', axis=1)
        labels    = []
        for col in self.labels_map :
            dfi_onehot = pd_get_dummies(df, cols_cat=[col], cat_dict= self.labels_map, only_onehot=True)
            log2(col, dfi_onehot.sum())
            labels.append(dfi_onehot.values )
            # log2(col, len(self.labels_map[col]) )
        return image_ids, labels

    def on_epoch_end(self):
      if self.shuffle :
        np.random.seed(12)
        indices        = np.arange(len(self.image_ids))
        np.random.shuffle(indices)
        self.image_ids = self.image_ids[indices]
        self.labels    = [label[indices] for label in self.labels]

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x       = []
        for image_id in batch_img_ids:
            pathi = os.path.abspath( self.image_dir + f'/{image_id}{self.img_suffix}' )
            # pathi = os.path.abspath( self.image_dir + f'/{image_id}' )            
            image = image_load(pathi, mode='cache')
            batch_x.append(image)
            # batch_x[-1].start()
            
        ###### X - Category :n (64, 231))  ###################################################
        xcat = None
        for y_head in self.labels:
            yi   = y_head[idx * self.batch_size:(idx + 1) * self.batch_size, :]
            # print(yi.shape)
            # xcat = yi
            xcat = np.concatenate( (xcat, yi), axis=1 ) if xcat is not None else  yi
        xcat =  np.stack( [  xcat[i,:] for i in range(len(xcat)) ] , axis=0 )
                
            
        batch_y = []
        for y_head in self.labels:
            batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size, :])

        if self.transforms is not None:
            batch_x = np.stack([self.transforms(image=x)['image'] for x in batch_x], axis=0)
            
        batch_x = (batch_x, xcat)    
        # image_check(name=f"{idx}.png", img=batch_x[1])            
        return (batch_x, *batch_y)



class CustomDataGenerator_img(Sequence):
    def __init__(self, img_dir, label_path, class_list,
                 split='train', batch_size=8, transforms=None):
        """    
           df_label format :
               id, uri, cat1, cat2, cat3, cat1_onehot, cat1_onehot, ....
        """
        self.image_dir   = img_dir
        self.class_list  = class_list
        self.batch_size  = batch_size
        self.transforms  = transforms

        dfref       = pd.read_csv(label_path)
        self.labels = data_add_onehot(dfref, img_dir, class_list)

    def on_epoch_end(self):
        np.random.seed(12)
        np.random.shuffle(self.labels)

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Create batch targets
        df_batch    = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []  #  list of heads

        for ii, x in df_batch.iterrows():  
            img =  np.array(Image.open(x['uri']).convert('RGB') )  
            batch_x.append(img)

            
        for ci in self.class_list :
               v = [ x.split(",") for x in df_batch[ci + "_onehot" ] ] 
               v = np.array( [ [int(t) for t in vlist ]   for vlist in v    ])
               batch_y.append( v )

                
        if self.transforms is not None:
            batch_x = np.stack([self.transforms(image=x)['image'] for x in batch_x], axis=0)

        return (batch_x, *batch_y)
 


######################################################################################
###############################IMAGE FUNCTIONS########################################
######################################################################################

os.makedirs(model_dir2 + f"/debug/", exist_ok=True)
# def image_check(name, img, renorm=False):
#     img  = img[:, :, ::-1]    
#     if renorm :
#         img = (img *0.5 +0.5) * 255.0        
#     cv2.imwrite( model_dir2 + f"/debug/{name}"  , img) 



def image_check_npz(path_npz,  keys=['train'], path="", tag="", n_sample=3,
                    renorm=True):    
    import cv2
    
    os.makedirs(path, exist_ok=True)
    data_npz = np.load( path_npz  )
    keys     =  data_npz.keys() if keys is None else keys 
    print('check', keys, "\n" )
    for key in keys :        
      print(key, str(data_npz[key])[:100], "\n")  
      for i in range(0, n_sample) :        
         try :
           img = data_npz[key][i]
           if renorm :        
              img = ( img * 0.5 + 0.5)  * 255
           cv2.imwrite( f"{path}/img_{tag}_{key}_{i}.png", img )
         except :
           pass     


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
            img = image_resize_pad(img, (xdim,ydim), padColor=  padcolor)   ### 255 white, 0 for black
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



def image_resize2(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resizes a image and maintains aspect ratio.
    Args:
        image:
        width:
        height:
        inter:
    Returns:
    """
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)



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
 
    

    
def padding_generate(
     paddings_number: int = 1, min_padding: int = 1, max_padding: int = 1
 ) -> np.array:
     """
     Args:
         paddings_number:  4
         min_padding:      1
         max_padding:    100
     Returns: padding list
     """
     return np.random.randint(low=min_padding, high=max_padding + 1, size=paddings_number)





def image_merge(image_list, n_dim, padding_size, max_height, total_width):
    """
    Args:
        image_list:  list of image
        n_dim:
        padding_size: padding size max
        max_height:   max height
        total_width:  total width
    Returns:
    """
    # create an empty array with a size large enough to contain all the images + padding between images
    if n_dim == 2:
        final_image = np.zeros((max_height, total_width), dtype=np.uint8)
    else:
        final_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    current_x = 0  # keep track of where your current image was last placed in the x coordinate
    idx_len = len(image_list) - 1
    for idx, image in enumerate(image_list):
        # add an image to the final array and increment the x coordinate
        height = image.shape[0]
        width = image.shape[1]
        if n_dim == 2:
            final_image[:height, current_x : width + current_x] = image
        else:
            final_image[:height, current_x : width + current_x, :] = image
        # add the padding between the images
        if idx == idx_len:
            current_x += width
        else:
            current_x += width + padding_size[idx]
    return final_image, padding_size




def image_remove_extra_padding(img, inverse=False, removedot=True):
    """TODO: Issue with small dot noise points : noise or not ?
              Padding calc has also issues with small blobs.
    Args:
        img: image
    Returns: image cropped of extra padding
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if removedot:
        ## TODO : renove hard-coding of filtering params
        min_size = max(1, int(img.shape[1] * 1.1))
        graybin = np.where(gray > 0.1, 1, 0)
        processed = morphology.remove_small_objects(
            graybin.astype(bool), min_size=min_size, connectivity=1
        ).astype(int)
        mask_x, mask_y = np.where(processed == 0)
        gray[mask_x, mask_y] = 0

    if inverse:
        gray = 255 * (gray < 128).astype(np.uint8)  # To invert to white

    coords = cv2.findNonZero(gray)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    crop = img[y : y + h, x : x + w]  # Crop the image
    return crop


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
            image = image_center_crop(image, (245, 245))

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


def image_read(filepath_or_buffer: Union[str, io.BytesIO]):
    """
    Read a file into an image object
    Args:
        filepath_or_buffer: The path to the file, a URL, or any object
            with a `read` method (such as `io.BytesIO`)
    """
    image = None
    if isinstance(filepath_or_buffer, np.ndarray):
        return filepath_or_buffer

    if hasattr(filepath_or_buffer, "read"):
        image = np.asarray(bytearray(filepath_or_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

    elif isinstance(filepath_or_buffer, str):
        assert os.path.isfile(filepath_or_buffer), (
            "Could not find image at path: " + filepath_or_buffer
        )
        if filepath_or_buffer.endswith(".tif") or filepath_or_buffer.endswith(".tiff"):
            image = tifffile.imread(filepath_or_buffer)
        else:
            image = cv2.imread(filepath_or_buffer)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    return image  



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


#############################################################################################
########################################PREPARATION##########################################
#############################################################################################


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

    image_check_npz(data_train + f"/train_test{tag}.npz" ,  
                                 keys = None, 
                                 path = data_train + "/zcheck/", 
                                 tag  = tag , n_sample=3
                              )


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
 
    

def data_get_sample(batch_size, x_train, labels_val):
        #### 
        # i_select = 10
        # i_select = np.random.choice(np.arange(train_size), size=batch_size, replace=False)
        i_select = np.random.choice(np.arange(len(labels_val['gender'])), size=batch_size, replace=False)


        #### Images
        x        = np.array([ x_train[i]  for i in i_select ] )

        #### y_onehot Labels  [y1, y2, y3, y4]
        labels_col   = [  'gender', 'masterCategory', 'subCategory', 'articleType' ]
        y_label_list = []
        for ci in labels_col :
           v =  labels_val[ci][i_select]
           y_label_list.append(v)

        return x, y_label_list 


def data_to_y_onehot_list(df, dfref, labels_col) :      
    df       = df.merge(dfref, on = 'id', how='left')

    
    labels_val = {}
    labels_cnt = {}
    for ci in labels_col:
      dfi_1hot  = pd.get_dummies(df, columns=[ci])  ### OneHot
      dfi_1hot  = dfi_1hot[[ t for t in dfi_1hot.columns if ci in t   ]].values  ## remove no OneHot
      labels_val[ci] = dfi_1hot 
      labels_cnt[ci] = df[ci].nunique()
      assert dfi_1hot.shape[1] == labels_cnt[ci],   labels_cnt     
    
    print(labels_cnt)
    return labels_val, labels_cnt


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
    import zipfile
    with zipfile.ZipFile(in_dir, 'r') as zip_ref:
        zip_ref.extractall(out_dir)


def gzip():
    #  python prepro.py gzip 
    
    # in_dir  = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9a_g6_-img_train_nobg_256_256-100000.cache/check"    
    
    in_dir = "/data/workspaces/noelkevin01/img/models/fashion/dcf_vae/m_train9pred/res/m_train9b_g3_-img_train_r2p2_200k_clean_nobg_256_256-500000-cache_best_epoch_261/topk"
    
    name    = "_".join(in_dir.split("/")[-2:])
    
    cmd = f"tar -czf /data/workspaces/noelkevin01/{name}.tar.gz   '{in_dir}/'   "
    print(cmd)
    os.system(cmd)  


def folder_size():
    os.system(" du -h --max-depth  13   /data/  | sort -hr  > /home/noelkevin01/folder_size.txt  ")
        


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
    import time, os, csv, urllib
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



def config_save(cc,path):    
   #path1 = path +"/info.json"  #### model_dir2 +"/info.json" 
   json.dump(  str(cc), open( path , mode='a'))
   print(path)


    
def os_path_copy(in_dir, path, ext="*.py"):
  """ Copy folder recursively 
  """
  import os, shutil, glob
  file_list = glob.glob(in_dir + '/' + ext)
  print(file_list)
  os.makedirs(path, exist_ok=True)
  for f in file_list:
    if os.path.isdir(f):
      shutil.copytree(f, os.path.join(path, os.path.basename(f)))
    else:
      shutil.copy2(f, os.path.join(path, os.path.basename(f)))  

os_path_copy(in_dir= cc.code_source  , path= cc.model_dir2 + "/code/")


######## 1-5) Transform, Dataset
class SprinklesTransform(ImageOnlyTransform):
    def __init__(self, num_holes=100, side_length=10, always_apply=False, p=1.0):
        from tf_sprinkles import Sprinkles
        super(SprinklesTransform, self).__init__(always_apply, p)
        self.sprinkles = Sprinkles(num_holes=num_holes, side_length=side_length)

    def apply(self, image, **params):
        if isinstance(image, Image.Image):
            image = tf.constant(np.array(image), dtype=tf.float32)
        elif isinstance(image, np.ndarray):
            image = tf.constant(image, dtype=tf.float32)

        return self.sprinkles(image).numpy()

  

image_size =  64
train_transforms = Compose([
    Resize(image_size, image_size, p=1),
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                       val_shift_limit=10, p=.9),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    ToFloat(max_value=255),
    SprinklesTransform(num_holes=10, side_length=10, p=0.5),
])

test_transforms = Compose([
    Resize(image_size, image_size, p=1),
    ToFloat(max_value=255)
])



 

##############################################################################################
#########################################MODEL TRAINING#######################################
##############################################################################################

def save_best(model, model_dir2, curr_loss, best_loss, counter, epoch, dd):
    """Save the best model"""
    # curr_loss = valid_loss
    if curr_loss < best_loss or (epoch % 5 == 0 and epoch > 0) :
        save_model_state(model, model_dir2 + f'/best/epoch_{epoch}')
        config_save(cc,model_dir2 + f"/best/epoch_{epoch}/info.json"  )
        config_save(dd,model_dir2 + f"/best/epoch_{epoch}/metrics.json"  )
        #json.dump(dd, open(, mode='w'))
        print(f"Model Saved | Loss improved {best_loss} --> {curr_loss}")
        best_loss = curr_loss
        counter   = 0
    else:
        counter += 1
              
    # odel_delete(path= model_dir2 + "/best/" )          
              
    return best_loss, counter


def save_model_state(model, model_dir2):
    """Save the model"""
    os.makedirs(model_dir2 , exist_ok=True)
    model.save_weights( model_dir2 + f'/model_keras_weights.h5')


def train_stop(counter, patience):
    """Stop the training if meet the condition"""
    if counter == patience :
        log(f"Model not improved from {patience} epochs...............")
        log("Training Finished..................")
        return True
    return False


def model_reload(model_reload_name, cc,):    
    model2      = DFC_VAE(latent_dim= cc.latent_dim, class_dict= cc.labels_count )
    input_shape = (batch_size, xdim, ydim, cdim)   ### x_train = x_train.reshape(-1, 28, 28, 1)
    model2.build(input_shape)
    model2.load_weights( model_reload_name + f'/model_keras_weights.h5')
    return model2


class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule", path=None):
        # compute the set of learning rates for each corresponding
        # epoch
        pass


#####  Learning Rate Schedule         
def learning_rate_schedule(mode="step", epoch=1, cc=None):
    if mode == "step" :
        # compute the learning rate for the current epoch
        if   epoch % 30  < 7 :  return 1e-3 * np.exp(-epoch * 0.005)  ### 0.74 every 10 epoch
        elif epoch % 30  < 17:  return 7e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 25:  return 3e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 30:  return 5e-5 * np.exp(-epoch * 0.005)
        else :  return 1e-4
    
    if mode == "random" :
        # randomize to prevent overfit... and reset learning
        if   epoch < 10 :  return 1e-3 * np.exp(-epoch * 0.004)  ### 0.74 every 10 epoch
        else :
            if epoch % 10 == 0 :
               ll = np.array([ 1e-3, 7e-4, 6e-4, 3e-4, 2e-4, 1e-4, 5e-5 ]) * np.exp(-epoch * 0.004)            
               ix = np.random.randint(len(ll))        
               cc.lr_actual =  ll[ix]            
            return cc.lr_actual
        

        
def loss_schedule(mode="step", epoch=1):
    if mode == "step" :
        ####  Classifier Loss :   2.8920667 2.6134858 
        ####  {'gender': 0.8566666, 'masterCategory': 0.99, 'subCategory': 0.9166, 'articleType': 0.7, 'baseColour': 0.5633333 }
        cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 90.0  ]
        cc.loss.ww_triplet = 1.0

        if epoch % 10 == 0 : dd.best_loss = 10000.0

        if epoch % 30 < 10 :
            cc.loss.ww_triplet  = 0.5   * 1.0
            cc.loss.ww_clf      = 2.0   * 5.0      ### 2.67
            cc.loss.ww_vae      = 100.0 * 10.0      ### 5.4
            cc.loss.ww_percep   = 10.0  * 1.0      ### 5.8   0.015    #### original: 0.015
            cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 100.0  ]

        elif epoch % 30 < 20 :
            cc.loss.ww_triplet  = 0.5   * 5.0
            cc.loss.ww_clf      = 2.0   * 5.0      ### 2.67
            cc.loss.ww_vae      = 100.0 * 5.0      ### 5.4
            cc.loss.ww_percep   = 10.0  * 1.0      ### 5.8   0.015    #### original: 0.015
            cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 100.0  ]

        elif epoch % 30 < 30 :
            cc.loss.ww_triplet  = 0.5   * 20.0
            cc.loss.ww_clf      = 2.0   * 5.0      ### 2.67
            cc.loss.ww_vae      = 100.0 * 1.0      ### 5.4
            cc.loss.ww_percep   = 10.0  * 5.0      ### 5.8   0.015    #### original: 0.015
            cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 100.0  ]



cc.loss= {}
def perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight=0.00005,
                             y_label_heads=None, y_pred_heads=None, clf_loss_fn=None):
    ### log( 'x_recon.shae',  x_recon.shape )
    ### VAE Loss  :  Mean Square : 0.054996297 0.046276666   , Huber: 0.0566 
    ### m = 0.00392156862  # 1/255
    ###   recons_loss = tf.reduce_mean( tf.reduce_mean(tf.abs(x-x_recon), axis=(1,2,3)) )
    ###   recons_loss = tf.reduce_mean( tf.reduce_mean(tf.square(x-x_recon), axis=(1,2,3) ) )    ## MSE error
    #     recons_loss = tf.reduce_mean( tf.square(x-x_recon), axis=(1,2,3) )     ## MSE error    
    recons_loss = recons_loss_global(x, x_recon)
    latent_loss = tf.reduce_mean( 0.5 * tf.reduce_sum(tf.exp(z_logsigma) + tf.square(z_mean) - 1.0 - z_logsigma, axis=1) )
    loss_vae    = kl_weight*latent_loss + recons_loss

    
    ### Efficient Head Loss : Input Need to Scale into 0-255, output is [0,1], 1280 vector :  0.5819792 0.5353247 
    ### https://stackoverflow.com/questions/65452099/keras-efficientnet-b0-use-input-values-between-0-and-255
    ### loss_percep = tf.reduce_mean(tf.square(  tf.subtract(tf.stop_gradient(percep_model(x * 255.0 )), percep_model(x_recon * 255.0  )  )))
    loss_percep = percep_loss_global( tf.stop_gradient(percep_model(x * 255.0 )),   percep_model(x_recon * 255.0  )  )
    
    
    #### Update  cc.loss  weights
    loss_schedule(mode="step", epoch=epoch)

        
    ### Triplet Loss: ####################################################################################################
    loss_triplet = 0.0 
    if cc.loss.ww_triplet > 0.0 :   ### 1.9  (*4)     5.6 (*1)
        ### `y_true` to be provided as 1-D integer `Tensor` with shape `[batch_size]  
        ### `y_pred` must be 2-D float `Tensor` of l2 normalized embedding vectors.
        z1     = tf.math.l2_normalize(z_mean, axis=1)  # L2 normalize embeddings
        loss_triplet = 6*triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[0], axis = -1) ,   y_pred=z1)  + 4*triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[2], axis = -1) ,   y_pred=z1)  +  2 * triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[3], axis = -1) ,   y_pred=z1)   +  triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[4], axis = -1) ,   y_pred=z1)  

    
    ####  Classifier Loss :   2.8920667 2.6134858 
    ####  {'gender': 0.8566666, 'masterCategory': 0.99, 'subCategory': 0.9166, 'articleType': 0.7, 'baseColour': 0.5633333 }
    if y_label_heads is not None:
        loss_clf = []
        for i in range(len(y_pred_heads)):
            head_loss = clf_loss_fn(y_label_heads[i], y_pred_heads[i])
            loss_clf.append(head_loss * cc.loss.ww_clf_head[i] )
        loss_clf = tf.reduce_mean(loss_clf)
        
        ####     0.05    ,  0.5                              2.67
        loss_all = loss_vae * cc.loss.ww_vae  +  loss_percep * cc.loss.ww_percep +  loss_clf * cc.loss.ww_clf  + loss_triplet * cc.loss.ww_triplet
        #  loss_all = loss_triplet 
        #### 0.20  
    else :
        loss_all = loss_vae * cc.loss.ww_vae  +  loss_percep * cc.loss.ww_percep
    return loss_all


class StepDecay(LearningRateDecay):
    def __init__(self, init_lr=0.01, factor=0.25, drop_every=5):
        # store the base initial learning rate, drop factor, and epochs to drop every
        self.init_lr    = init_lr
        self.factor     = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        if   epoch % 30  < 7 :  return 1e-3 * np.exp(-epoch * 0.005)  ### 0.74 every 10 epoch
        elif epoch % 30  < 17:  return 7e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 25:  return 3e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 30:  return 5e-5 * np.exp(-epoch * 0.005)
        else :  return 1e-5



################TEST#####################
if cc.schedule_type == 'step':
    print("Using 'step-based' learning rate decay")
    schedule = StepDecay(init_lr=cc.learning_rate, factor=0.70, drop_every= 3 )
    
    
    
    

#############################################################################################        
################################# 1-3) Define DFC-VAE model #################################
############################################################################################# 

class DFC_VAE(tf.keras.Model):
    """Deep Feature Consistent Variational Autoencoder Class"""
    def __init__(self, latent_dim, class_dict):
        super(DFC_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = make_encoder()
        self.decoder = make_decoder()

        self.classifier = make_classifier(class_dict)

    def encode(self, x):
        z_mean, z_logsigma = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logsigma

    def reparameterize(self, z_mean, z_logsigma):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return eps * tf.exp(z_logsigma * 0.5) + z_mean

    def decode(self, z, apply_sigmoid=False):
        x_recon = self.decoder(z)
        if apply_sigmoid:
            new_x_recon = tf.sigmoid(x_recon)
            return new_x_recon
        return x_recon

    def call(self, x,training=True, mask=None, y_label_list= None):
        # out_classes = None        
        xcat_all = x[1]  ### Category
        x        = x[0]  ### Image
                
        z_mean, z_logsigma = self.encode( [x, xcat_all] )
        z = self.reparameterize(z_mean, z_logsigma)
        x_recon = self.decode(z)

        #### Classifier
        out_classes = self.classifier(z)

        return z_mean, z_logsigma, x_recon, out_classes


def make_encoder(n_outputs=1):
    #Functionally define the different layer types
    #Input = tf.keras.layers.InputLayer
    Input = tf.keras.Input
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                                activity_regularizer=regularizers.l2(1e-5))
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten
    Dense = functools.partial(  tf.keras.layers.Dense, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                                bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))

    input0 = [ Input( shape=(xdim, ydim, 3)),   
               Input( shape=( cc.labels_onehotdim  , ))     ]
    
    
    ##### Build the encoder network using the Sequential API
    encoder1 = tf.keras.Sequential([
        input0[0],

        Conv2D(filters=2*n_filters, kernel_size=5,  strides=2),
        BatchNormalization(),
        layers.Dropout(0.25),

        Conv2D(filters=4*n_filters, kernel_size=3,  strides=2),
        BatchNormalization(),
        layers.Dropout(0.25),
        
        Conv2D(filters=6*n_filters, kernel_size=3,  strides=2),
        BatchNormalization(),

        Flatten(),
        # Dense(512*2, activation='relu'),
    ])
    
    
    ##### Category Inpput
    encoder2 = tf.keras.Sequential([
        input0[1],
        Dense(64, activation='relu'),
    ])
        
        
    x = tf.keras.layers.concatenate([encoder1.output, encoder2.output])    

    x = Dense(512*2, activation='relu')(x)
    x = layers.Dropout(0.1)(x)    
    x = Dense(512*2, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    output0 = Dense(2*latent_dim, activation="sigmoid")(x)
        
    
    encoder = tf.keras.Model( inputs= input0, outputs= output0)
    
    return encoder


def make_decoder():
    """
    ValueError: Dimensions must be equal, but are 3 and 4
    for '{{node sub}} = Sub[T=DT_FLOAT](x, sequential_1/conv2d_transpose_3/Relu)' with input shapes: [8,256,256,3], [8,256,256,4].
    """
    #Functionally define the different layer types
    Input = tf.keras.layers.InputLayer

    # bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                                bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))
    Reshape = tf.keras.layers.Reshape
    Conv2DTranspose = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                                        activity_regularizer=regularizers.l2(1e-5))
    # BatchNormalization = tf.keras.layers.BatchNormalization
    # Flatten = tf.keras.layers.Flatten

    #Build the decoder network using the Sequential API
    if xdim == 64 :   #### 64 x 64 img
        decoder = tf.keras.Sequential([
            Input(input_shape=(latent_dim,)),

            Dense(units= 4*4*6*n_filters),
            Dense(units= 4*4*6*n_filters),
            layers.Dropout(0.2),
            Dense(units= 4*4*6*n_filters),
            Reshape(target_shape=(4, 4, 6*n_filters)),
            #### ValueError: total size of new array must be unchanged, input_shape = [2304], output_shape = [7, 4, 144]

            #### Conv. layer
            Conv2DTranspose(filters=4*n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=2*n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=1*n_filters, kernel_size=5,  strides=2),

            Conv2DTranspose(filters=3, kernel_size=5,  strides=2),
            # Conv2DTranspose(filters=4, kernel_size=5,  strides=2),

        ])

    if ydim == 256 :  ### 256 8 256 img
        decoder = tf.keras.Sequential([
            Input(input_shape=(latent_dim,)),

            Dense(units=16*16*6*n_filters),
            Dense(units=16*16*6*n_filters),
            layers.Dropout(0.2),
            Dense(units=16*16*6*n_filters),
            Reshape(target_shape=(16, 16, 6*n_filters)),

            #### Conv. layer
            Conv2DTranspose(filters=4*n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=2*n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=1*n_filters, kernel_size=5,  strides=2),
            Conv2DTranspose(filters=3, kernel_size=5,  strides=2),

        ])
    return decoder


def make_classifier(class_dict):
    """ Supervised multi class
            self.gender         = nn.Linear(self.inter_features, self.num_classes['gender'])
            self.masterCategory = nn.Linear(self.inter_features, self.num_classes['masterCategory'])
    """
    Input = tf.keras.layers.InputLayer
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                                bias_regularizer=regularizers.l2(1e-4),
                                activity_regularizer=regularizers.l2(1e-5))
    # Reshape = tf.keras.layers.Reshape
    # BatchNormalization = tf.keras.layers.BatchNormalization

    # if xdim == 64 :   #### 64 x 64 img
    base_model = tf.keras.Sequential([
        Input(input_shape=(latent_dim,)),
        Dense(units=1024),
        # layers.Dropout(0.10),
        # Dense(units=512),
        # layers.Dropout(0.10),
        # Dense(units=512),
    ])

    x = base_model.output
    ## x = layers.Flatten()(x) already flatten

    #### Multi-heads
    outputs = [Dense(num_classes, activation='softmax', name= f'{class_name}_out')(x) for class_name, num_classes in class_dict.items()]
    clf = tf.keras.Model(name='clf', inputs=base_model.input , outputs=outputs)

    return clf


"""## 1-4) Build loss function"""
#### input is 0-255, do not normalize input
percep_model = tf.keras.applications.EfficientNetB2(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=(xdim, ydim, cdim), pooling=None, classes=1000,
    classifier_activation='softmax'
)


###### Loss definition ##########################################################
####  reduction=tf.keras.losses.Reduction.NONE  for distributed GPU
clf_loss_global    =  tf.keras.losses.BinaryCrossentropy()

### Classification distance
triplet_loss_global = tfa.losses.TripletSemiHardLoss( margin=  1.0,    distance_metric='L2',    name= 'triplet',)


###  tf.keras.losses.Huber( delta= 0.6) , 
# rec_loss_global = tf.keras.losses.MeanSquaredError()
# recons_loss_global = tf.keras.losses.Huber( delta= 0.6) 
# percep_loss_global = tf.keras.losses.Huber( delta= 0.6) 
# recons_loss_global = tf.keras.losses.MeanSquaredError()
# percep_loss_global = tf.keras.losses.MeanSquaredError()

recons_loss_global = tf.keras.losses.MeanAbsoluteError()  # reduction="sum"
percep_loss_global = tf.keras.losses.MeanSquaredError()

"""
https://www.tensorflow.org/tutorials/distribute/custom_training
# Set reduction to `none` so we can do the reduction afterwards and divide by
# global batch size.
#  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#      from_logits=True,
@tf.function
tfa.losses.triplet_semihard_loss
for speed up
"""




##################################################################################
###################################MODEL PREDICTION###############################
##################################################################################

def predict(name=None):
    ###   python prepro.py  predict 
    if name is None :
       name = "m_train9b_g3_-img_train_r2p2_70k_clean_nobg_256_256-100000.cache/best/epoch_90"
    os.system( f" python train9pred.py  '{name}'   ")
    

##################################################################################
##################################MODEL VALIDATION################################
##################################################################################

def metric_accuracy_test(y_test, y_pred, dd):
   test_accuracy = {} 
   for k,(ytruei, ypredi) in enumerate(zip(y_test, y_pred)) : 
       ytruei = np.argmax(ytruei,         axis=-1)
       ypredi = np.argmax(ypredi.numpy(), axis=-1)
       # log(ytruei, ypredi ) 
       test_accuracy[ dd.labels_col[k] ] = accuracy_score(ytruei, ypredi )
        
   log('accuracy', test_accuracy)     
   return test_accuracy 
    


def metric_accuracy_val(y_val, y_pred_head, class_dict):
    # Val accuracy
    val_accuracies = {class_name: 0. for class_name in class_dict}
    for i, class_name in enumerate(class_dict):
        y_pred = np.argmax(y_pred_head[i], 1)
        y_true = np.argmax(y_val[i], 1)
        val_accuracies[class_name] = (y_pred == y_true).mean()
    print( f'\n {val_accuracies}')
    return val_accuracies


def valid_image_original(img_list, path, tag, y_labels, n_sample=None):
    """Assess image validity"""
    os.makedirs(path, exist_ok=True)
    if n_sample is not None and isinstance(n_sample, int):
        img_list = img_list[:n_sample]
        y_labels = [y[:n_sample].tolist() for y in y_labels]

    for i in range(len(img_list)) :
        img = img_list[i]
        if not isinstance(img, np.ndarray) :
            img = img.numpy()

        img       = img[:, :, ::-1]
        img       = np.clip(img * 255, 0, 255).astype('uint8')
        label_tag = 'label_{' + '-'.join([str(y[i]) for y in y_labels]) + '}'
        save_path = f"{path}/img_{cc.tag}_nimg_{i}_{tag}_{label_tag}.png"
        cv2.imwrite(save_path, img)
        img = None


def valid_image_check(img_list, path="", tag="", y_labels="", n_sample=3, renorm=True):
    """Assess image validity"""
    os.makedirs(path, exist_ok=True)
    if n_sample is not None and isinstance(n_sample, int):
        img_list = img_list[:n_sample]
        y_labels = [y[:n_sample].tolist() for y in y_labels]

    for i in range(len(img_list)) :
        img = img_list[i]
        if not isinstance(img, np.ndarray) :
            img = img.numpy()

        if renorm:
            img = (img * 0.5 + 0.5) * 255

        label_tag = 'label_{' + '-'.join([str(y[i]) for y in y_labels]) + '}'
        save_path = f"{path}/img_{cc.tag}_nimg_{i}_{tag}_r_{label_tag}.png"
        img       = img[:, :, ::-1]
        cv2.imwrite(save_path, img)
        img = None

                    

def metric_accuracy2(y_test, y_pred, dd):
   from sklearn.metrics import accuracy_score
   test_accuracy = {}
   for k,(ytruei, ypredi) in enumerate(zip(y_test, y_pred)) :
       ytruei = np.argmax(ytruei,         axis=-1)
       ypredi = np.argmax(ypredi.numpy(), axis=-1)
       # log(ytruei, ypredi )
       test_accuracy[ dd.labels_col[k] ] = accuracy_score(ytruei, ypredi )

   log('accuracy', test_accuracy)
   return test_accuracy



def clf_loss_macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y     = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp    = tf.reduce_sum(y_hat * y, axis=0)
    fp    = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn    = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost    = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost


##################################################################################
##################################SYSTEM CONFIG###################################
##################################################################################

### CPU Multi-thread 
tf.config.threading.set_intra_op_parallelism_threads(cc.n_thread)
tf.config.threading.set_inter_op_parallelism_threads(cc.n_thread)
strategy = None

if cc.compute_mode == "custom":
    log2("# Disable all GPUS")
    ll = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices( [ ll[ cc.gpu_id ] ], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    print(visible_devices)
       
if cc.compute_mode == "cpu_only":
    try:
        log2("# Disable all GPUS")
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except Exception as e :
        log2(e)
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    
if cc.compute_mode == "gpu" :
    log2("# Create a MirroredStrategy.")
    strategy = tf.distribute.MirroredStrategy()
    log2('Number of devices: {}'.format(strategy.num_replicas_in_sync))



def check_tf():
    #### python prepro.py check_tf 
    import tensorflow as tf
    print( tf.config.list_physical_devices())
    
    
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
            
            
            
#####################################################################################
################################TEST CODE############################################
#####################################################################################

"""
Auto Augmentation :
   data/img/models/fashion/dcf_vae/auto_config
"""      
    
train_transforms = Compose([
    Resize(image_size, image_size, p=1),
    # Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.7),
    ToGray(p=0.3),
    HorizontalFlip(p=0.7),
    #### isseuw with black
    # RandomContrast(limit=0.2, p=0.5),
    # RandomGamma(gamma_limit=(96, 103), p=0.3),
    # RandomBrightness(limit=0.1, p=0.5),
    #HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10,
    #                   val_shift_limit=10, p=.9),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1,
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
    GridDistortion(num_steps=5, p=0.7),
    ElasticTransform(sigma=10, alpha_affine=10, p=0.8),
    OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.8),
    Cutout(num_holes=13, max_h_size=14, max_w_size=14, 
           fill_value= cc.cutout_color , always_apply=False, p=0.8 ) ,   
    
    ToFloat(max_value=255),
    # SprinklesTransform(num_holes=7, side_length=3, p=0.5),
])


test_transforms = Compose([
    Resize(image_size, image_size, p=1),
    ToFloat(max_value=255)
])



"""#### Dataset Train, test"""
#df_train = pd.read_csv( cc.path_label_train,  error_bad_lines=False, warn_bad_lines=False)
#df_val   = pd.read_csv( cc.path_label_val,    error_bad_lines=False, warn_bad_lines=False)
#log(df_train) ; time.sleep(3)
#log(df_val) ; time.sleep(3)

#data_dir        = Path('./fashion_data')
image_dir        = cc.path_img_all
train_label_path = cc.path_label_train
val_label_path   = cc.path_label_val



train_data = RealCustomDataGenerator(image_dir, train_label_path, class_dict= cc.labels_map,
                                     split='train', batch_size=batch_size, transforms=train_transforms, shuffle=True,
                                     img_suffix= cc.img_suffix)

val_data   = RealCustomDataGenerator(image_dir, val_label_path, class_dict= cc.labels_map,
                                     split='val', batch_size= 500, transforms=test_transforms,
                                     img_suffix= cc.img_suffix)    

    
log('N training batches:', len(train_data))
log('N test batches:',     len(val_data))
#############################################################################
#############################################################################
"""## Train"""
@tf.function
def train_step(x, model, y_label_list=None):
    with tf.GradientTape() as tape:
        z_mean, z_logsigma, x_recon, out_classes = model(x, training=True, y_label_list= y_label_list)      #Forward pass through the VAE
        
        loss = perceptual_loss_function(x[0], x_recon, z_mean, z_logsigma,
            y_label_heads = y_label_list,
            y_pred_heads  = out_classes,
            clf_loss_fn   = clf_loss_global
        )

    grads = tape.gradient(loss, model.trainable_variables)   #Calculate gradients
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def validation_step(x, model, y_label_list=None):
    z_mean, z_logsigma, x_recon, out_classes = model(x, training=False)  #Forward pass through the VAE
    loss = perceptual_loss_function(x[0], x_recon, z_mean, z_logsigma,
            y_label_heads = y_label_list, 
            y_pred_heads  = out_classes, 
            clf_loss_fn   = clf_loss_global
        )
    return loss, x_recon, out_classes




#### Instantiate a new DFC_CVAE model and optimizer  ###############################
log("\nBuild the DFC-VAE Model")

if len(cc.model_reload_name) > 0  :
   model = model_reload( cc.model_dir + cc.model_reload_name, cc,)
   log('Model reloaded', model, "\n\n") ; time.sleep(5) 
else : 
   model     = DFC_VAE(latent_dim, class_dict= cc.labels_count )
optimizer = tf.keras.optimizers.Adam(learning_rate)
log(model, optimizer)



#### Setup learning rate scheduler


### Class
for i,c in enumerate(cc.labels_map):
    print(i,c)

    
# time.sleep(20)
### Training loop  ################################################################
dd = Box({})

kbatch          = len(train_data)
dd.train_loss_hist = []
dd.valid_loss_hist = []
counter         = 0
dostop          = False
dd.best_loss    = 100000000.0


config_save(cc,path= cc.model_dir2 +"info.json")
log2("\n\n#########################################################")
log2(cc)
for epoch in range(epoch0, num_epochs):
    log2(f"Epoch {epoch+1}/{num_epochs}, in {kbatch} kbatches ")
    if dostop: break

    ###### Set learning rate
    cc.lr_actual            = learning_rate_schedule(mode= cc.lrate_mode, epoch=epoch, cc= cc)
    optimizer.learning_rate = cc.lr_actual
    
    for batch_idx, (x,  *y_label_list) in enumerate(train_data):
        if dostop: break
        # log("x", x)
        # log("y_label_list", y_label_list)
        # log('[Epoch {:03d} batch {:04d}/{:04d}]'.format(epoch + 1, batch_idx+1, kbatch))
        # print( str(y_label_list)[:100] )
        train_loss = train_step(x, model, y_label_list=y_label_list)
        dd.train_loss_hist.append( np.mean(train_loss.numpy()) )        
        # log( dd.train_loss_hist[-1] )
        # image_check(name= f"{batch_idx}.png", img=x[0], renorm=False)
        
        
        if (batch_idx +1) % cc.kloop  == 0 or batch_idx < 2 :
            log( f'[Epoch {epoch} batch {batch_idx}/{kbatch}],  ')
            for kk, (x_val, *y_val) in enumerate(val_data):
                valid_loss, x_recon, y_pred_head = validation_step(x_val, model, y_val)
                break

            dd.valid_loss_hist.append( np.mean(valid_loss.numpy())  )

            # log3("x",       str(x_recon)[1000:2000])
            # log3("y_label", str(y_pred_head[0])[:300])
            log2(epoch, batch_idx, 'train,valid', dd.train_loss_hist[-1], dd.valid_loss_hist[-1], cc.lr_actual)
            dd.best_loss, counter = save_best(model, model_dir2, dd.train_loss_hist[-1] + dd.valid_loss_hist[-1], dd.best_loss, counter, epoch, dd)
            dostop                = train_stop(counter, cc['patience'])


        if (batch_idx + 1) % cc.kloop_img == 0  or batch_idx < 2 :
            #for (x_val, *y_val) in val_data:
            #    _, x_recon, y_pred_head = validation_step(x_val, model)
            #    break
            y_pred     = [np.argmax(y, 1) for y in y_pred_head]
            valid_image_check(x_recon, path=model_dir2 + "/check/",
                              tag=f"e{epoch+1}_b{batch_idx+1}", y_labels=y_pred, n_sample=15, renorm=True)

            if epoch == 0 and batch_idx < cc.kloop_img+1 :
               y_val_true = [np.argmax(y, 1) for y in y_val]
               valid_image_original(x_val, path=model_dir2 + "/check/",
                                    tag=f'e{epoch+1}_b{batch_idx+1}', y_labels=y_val_true, n_sample=15)

            dd.val_accuracy = metric_accuracy(y_val, y_pred_head, class_dict= cc.labels_map)
            log2(dd.val_accuracy, cc.loss )

              
log('Final valid_loss', str(valid_loss_hist)[:200])


"""## Save the model"""
log("\nSaving Model")
os.makedirs(model_dir, exist_ok=True)
tf.saved_model.save(model, model_dir2)
model.save_weights( model_dir2 + f'/model_keras_weights.h5')
log(model_dir2)


"""## Reload the model"""
log('\nReload Model')
model2 = DFC_VAE(latent_dim, class_dict= cc.labels_count)
input_shape = (batch_size, xdim, ydim, cdim)   ### x_train = x_train.reshape(-1, 28, 28, 1)
model2.build(input_shape)
model2.load_weights( model_dir2 + f'/model_keras_weights.h5')
log('# Reloaded', model2)
# log('rerun eval', validation_step(x_val, model2))



"""
gender ['men', 'women', 'boys', 'girls', 'unisex', 'other']
masterCategory ['apparel', 'accessories', 'footwear', 'personal care', 'free items'
 'sporting goods', 'home', 'other']
 
subCategory ['topwear', 'bottomwear', 'watches', 'socks', 'shoes', 'belts', 'flip flops'
 'bags', 'innerwear', 'sandal', 'shoe accessories', 'fragrance', 'jewellery'
 'lips', 'saree', 'eyewear', 'scarves', 'dress', 'loungewear and nightwear'
 'wallets', 'apparel set', 'headwear', 'mufflers', 'skin care', 'makeup'
 'free gifts', 'ties', 'accessories', 'nails', 'beauty accessories'
 'water bottle', 'skin', 'eyes', 'bath and body', 'gloves'
 'sports accessories', 'cufflinks', 'sports equipment', 'stoles', 'hair'
 'perfumes', 'home furnishing', 'umbrellas', 'wristbands', 'other', 'vouchers']
 
 
articleType ['shirts', 'jeans', 'watches', 'track pants', 'tshirts', 'socks', 'casual shoes'
 'belts', 'flip flops', 'handbags', 'tops', 'bra', 'sandals', 'shoe accessories'
 'sweatshirts', 'deodorant', 'formal shoes', 'bracelet', 'lipstick', 'flats'
 'kurtas', 'waistcoat', 'sports shoes', 'shorts', 'briefs', 'sarees'
 'perfume and body mist', 'heels', 'sunglasses', 'innerwear vests', 'pendant'
 'laptop bag', 'scarves', 'dresses', 'night suits', 'skirts', 'wallets'
 'blazers', 'ring', 'kurta sets', 'clutches', 'shrug', 'backpacks', 'caps'
 'trousers', 'earrings', 'camisoles', 'boxers', 'jewellery set', 'dupatta'
 'capris', 'lip gloss', 'bath robe', 'mufflers', 'tunics', 'jackets', 'trunk'
 'lounge pants', 'face wash and cleanser', 'necklace and chains'
 'duffel bag', 'sports sandals', 'foundation and primer', 'sweaters'
 'free gifts', 'trolley bag', 'tracksuits', 'swimwear', 'shoe laces'
 'fragrance gift set', 'bangle', 'nightdress', 'ties', 'baby dolls', 'leggings'
 'highlighter and blush', 'travel accessory', 'kurtis', 'mobile pouch'
 'messenger bag', 'lip care', 'nail polish', 'eye cream', 'accessory gift set'
 'beauty accessory', 'jumpsuit', 'kajal and eyeliner', 'water bottle'
 'suspenders', 'face moisturisers', 'lip liner', 'robe', 'salwar and dupatta'
 'patiala', 'stockings', 'eyeshadow', 'headband', 'tights', 'nail essentials'
 'churidar', 'lounge tshirts', 'face scrub and exfoliator', 'lounge shorts'
 'gloves', 'wristbands', 'tablet sleeve', 'ties and cufflinks', 'footballs'
 'compact', 'stoles', 'shapewear', 'nehru jackets', 'salwar', 'cufflinks'
 'jeggings', 'hair colour', 'concealer', 'rompers', 'sunscreen', 'booties'
 'mask and peel', 'waist pouch', 'hair accessory', 'body lotion', 'rucksacks'
 'basketballs', 'lehenga choli', 'clothing set', 'mascara', 'cushion covers'
 'key chain', 'rain jacket', 'toner', 'lip plumper', 'umbrellas'
 'face serum and gel', 'other', 'hat', 'mens grooming kit', 'makeup remover'
 'body wash and scrub', 'ipad']
 
baseColour ['navy blue', 'blue', 'silver', 'black', 'grey', 'green', 'purple', 'white'
 'beige', 'brown', 'bronze', 'teal', 'copper', 'pink', 'off white', 'maroon'
 'red', 'khaki', 'orange', 'yellow', 'charcoal', 'gold', 'steel', 'tan', 'multi'
 'magenta', 'lavender', 'sea green', 'cream', 'peach', 'olive', 'skin'
 'burgundy', 'coffee brown', 'grey melange', 'rust', 'rose', 'lime green'
 'mauve', 'turquoise blue', 'metallic', 'mustard', 'taupe', 'nude'
 'mushroom brown', 'fluorescent green', 'other']
 
N ids 15841
N images 44441
Total valid images: 15841
(17719, 8)
id                15841
gender                5
masterCategory        7
subCategory          45
articleType         141
baseColour           46
season                4
usage                 8
"""