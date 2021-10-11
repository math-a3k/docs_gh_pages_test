# -*- coding: utf-8 -*-
HELP = """ watch nvidia-smi
Focus on reconstruction
"""
import os, glob, sys, math, string, time, json, logging, functools, random, numpy as np, pandas as pd, cv2, matplotlib, scipy, h5py, yaml
from pprint import pprint as print2
from box import Box

from pathlib import Path

import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers
import scipy.stats ; from scipy.stats import norm


from utilmy import pd_read_file

###########################################################################################




##### Labels
def np_remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def clean_duplicates(ll):   #name changed
    v = [ t for t in ll if len(t) > 1 ]
    return  np_remove_duplicates(v)



### log  #############################################################################
def print_debug_info(*s):   #name changed
    if cc.verbosity >= 3:   
        print(*s, flush=True)
        with open(cc.model_dir2 + "/debug.py", mode='a') as fp:
            fp.write(str(s) + "\n")

def print_log_info(*s):   #name changed
    print(*s, flush=True)
    with open(cc.model_dir2 + "/log.py", mode='a') as fp:
        fp.write(str(s) + "\n")

        
        

########################################################################################################
########### TPU ########################################################################################
### CPU Multi-thread


def tf_compute_set(cc:dict):
    cc = Box(cc)  ### dot notaation
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


def check_valid_image(img_list, path="", tag="", y_labels="", n_sample=3, renorm=True):
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

                    
def save_best_model(model, model_dir2, curr_loss, best_loss, counter, epoch, dd):   #name changed
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




#################################################################################################
#### Code for generating custom data from the Kaggle dataset   ##################################
def get_custom_label_data():
    
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




###############################################################################################################
###############################################################################################################

@tf.function
def train_step_opt(x, y, model, loss_fn, optimizer):   #changed file
    with tf.GradientTape() as tape_w:

        # A separate GradientTape is needed for watching the input.
        with tf.GradientTape() as tape_x:
            tape_x.watch(x)
            # Regular forward pass.
            sample_features, nbr_features, nbr_weights = nbr_features_layer.call(x)
            base_output  = model(sample_features, training=True)
            labeled_loss = loss_fn(y, base_output)

        has_nbr_inputs = nbr_weights is not None and nbr_features
        if (has_nbr_inputs and graph_reg_config.multiplier > 0):
            # Use logits for regularization.
            sample_logits = base_output
            nbr_logits    = model(nbr_features, training=True)
            graph_loss    = regularizer(sources=sample_logits, targets=nbr_logits, weights=nbr_weights)
        else:
            graph_loss = tf.constant(0, dtype=tf.float32)

        scaled_graph_loss = graph_reg_config.multiplier * graph_loss

        # Combines both losses. This could also be a weighted combination.
        total_loss = labeled_loss + scaled_graph_loss

    # Regular backward pass.
    gradients = tape_w.gradient(total_loss,  model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return base_output, total_loss, labeled_loss, scaled_graph_loss


@tf.function
def test_step(x, y, model, loss_fn):   #changed file
    # Regular forward pass.
    sample_features, nbr_features, nbr_weights = nbr_features_layer.call(x)
    base_output  = model(sample_features, training=False)
    labeled_loss = loss_fn(y, base_output)

    has_nbr_inputs = nbr_weights is not None and nbr_features
    if (has_nbr_inputs and graph_reg_config.multiplier > 0):
        # Use logits for regularization.
        sample_logits = base_output
        nbr_logits    = model(nbr_features, training=False)
        graph_loss    = regularizer(sources=sample_logits, targets=nbr_logits, weights=nbr_weights)
    else:
        graph_loss = tf.constant(0, dtype=tf.float32)

    scaled_graph_loss = graph_reg_config.multiplier * graph_loss

    # Combines both losses. This could also be a weighted combination.
    total_loss = labeled_loss + scaled_graph_loss
    return base_output, total_loss, labeled_loss, scaled_graph_loss




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



