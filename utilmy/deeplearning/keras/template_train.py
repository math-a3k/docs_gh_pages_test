# -*- coding: utf-8 -*-
HELP = """ watch nvidia-smi
Focus on reconstruction
"""
import os, pandas as pd, time, numpy as np,sys
from pprint import pprint as print2
from box import Box

import os, glob, sys, math, string, time, json, logging, functools, random, numpy as np, pandas as pd, cv2, matplotlib, scipy, h5py, yaml
from pathlib import Path

import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers

import scipy.stats ; from scipy.stats import norm

from tqdm import tqdm
from utilmy import pd_read_file
from utilmy.deeplearning.keras.util_train import clean_duplicates as clean1


cc = Box({})


def param_set():
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

    #############################
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

    cc.cutout_color = 0





def params_set2():
    import diskcache as dc
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





##### Labels


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




######################################################################################
##### Cache for image loaading
import diskcache as dc
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
### log  #######################################################

from utilmy.deeplearning.keras.util_train import print_log_info, log2
from utilmy.deeplearning.keras.util_train import *

########################################################################################################


########### TPU ########################################################################################
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




########################################################################################################
## 1) Define some helper functions ###############################################


os_path_copy(in_dir= cc.code_source  , path= cc.model_dir2 + "/code/")


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



    

    
    

#################################################################################################
#### Code for generating custom data from the Kaggle dataset   ##################################
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

df = label_get_data()



### Fitlered   ##################################################
df = df[  ['id'] + cc.labels_cols ]

#duplicate ftn removed

df = pd_category_filter(df, cc.labels_map)
log(df)


    
log('N training batches:', len(train_data))
log('N test batches:',     len(val_data))
###############################################################################################################
###############################################################################################################
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

