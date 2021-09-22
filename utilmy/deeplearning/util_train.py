# -*- coding: utf-8 -*-
""" watch nvidia-smi
python train8aa.py  


Focus on reconstruction

"""
import os, pandas as pd, time, numpy as np,sys
from pprint import pprint as print2
from box import Box
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


##### Labels
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
def log3(*s):
    if cc.verbosity >= 3:   
        print(*s, flush=True)
        with open(cc.model_dir2 + "/debug.py", mode='a') as fp:
            fp.write(str(s) + "\n")

def log2(*s): 
    print(*s, flush=True)
    with open(cc.model_dir2 + "/log.py", mode='a') as fp:
        fp.write(str(s) + "\n")

        
        
#####################################################################################
import os, glob, sys, math, string, time, json, logging, functools, random, numpy as np, pandas as pd, cv2, matplotlib, scipy, h5py, yaml
from pathlib import Path

import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers

import matplotlib.pyplot as plt
import scipy.stats ; from scipy.stats import norm

from PIL import Image ; from tqdm import tqdm
from sklearn import manifold
from utilmy import pd_read_file

# from tf_sprinkles import Sprinkles
from madgrad import MadGrad
from sklearn.preprocessing import OneHotEncoder
from albumentations.core.transforms_interface import ImageOnlyTransform
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
def log(*s):
    """Log decorator"""
    print(*s, flush=True)

    
def config_save(cc,path):   
   import json 
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


def metric_accuracy(y_val, y_pred_head, class_dict):
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


os.makedirs(model_dir2 + f"/debug/", exist_ok=True)
def image_check(name, img, renorm=False):
    img  = img[:, :, ::-1]    
    if renorm :
        img = (img *0.5 +0.5) * 255.0        
    cv2.imwrite( model_dir2 + f"/debug/{name}"  , img) 
        

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

class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule", path=None):
        # compute the set of learning rates for each corresponding
        # epoch
        pass


    

#############################################################################################        
####### 1-3) Define DFC-VAE model"""  #######################################################
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
    BatchNormalization = tf.keras.layers.BatchNormalization
    Flatten = tf.keras.layers.Flatten

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
    Reshape = tf.keras.layers.Reshape
    BatchNormalization = tf.keras.layers.BatchNormalization

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

            
#from import argmax as kargmax


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

df = pd_category_filter(df, cc.labels_map)
log(df)



#### Split into  #################################################
shuffled  = df.sample(frac=1)
n         = df.shape[0]
num_train = int(0.97 * n)
num_val   = n - num_train

df_train  = df.iloc[:num_train, :]
df_val    = df.iloc[num_train:, :]

df_train.to_csv(cc.path_label_train,  index=False)
df_val.to_csv(  cc.path_label_val,    index=False)

#### test
# pathi       = cc.path_img_all  + "/10004.png"
# image_dataset = [ np.array(  cv2.cvtColor( cv2.imread(pathi), cv2.COLOR_BGR2RGB)  ) ]

for x in [ 'articleType', 'baseColour' ]:
    print(df_train.groupby(x).agg({'id': 'count'})) ; time.sleep(4)


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



########################################################################
"""
Auto Augmentation :
   data/img/models/fashion/dcf_vae/auto_config

"""    
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, Resize,
    GridDistortion, ElasticTransform,OpticalDistortion,
    ToGray,
    Equalize, # histogram (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5)
    Cutout, # (num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5
)
    
    
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

        
        
if cc.schedule_type == 'step':
    print("Using 'step-based' learning rate decay")
    schedule = StepDecay(init_lr=cc.learning_rate, factor=0.70, drop_every= 3 )
    


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



######## 1-5) Transform, Dataset
class SprinklesTransform(ImageOnlyTransform):
    def __init__(self, num_holes=100, side_length=10, always_apply=False, p=1.0):
        super(SprinklesTransform, self).__init__(always_apply, p)
        self.sprinkles = Sprinkles(num_holes=num_holes, side_length=side_length)

    def apply(self, image, **params):
        if isinstance(image, Image.Image):
            image = tf.constant(np.array(image), dtype=tf.float32)
        elif isinstance(image, np.ndarray):
            image = tf.constant(image, dtype=tf.float32)

        return self.sprinkles(image).numpy()

  
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


















