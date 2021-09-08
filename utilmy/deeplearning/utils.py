# -*- coding: utf-8 -*-
import io
import os
from typing import Union

import cv2
import numpy as np
#import tifffile.tifffile
from skimage import morphology



###################################################################################################
###################################################################################################
from importall import *
from sklearn.metrics import accuracy_score


def log(*s):
    print(*s, flush=True)

    
    
def metric_accuracy(y_test, y_pred, dd):
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
    

#################################################################################    
def save_best(model, model_dir2, valid_loss, best_loss, counter):
    curr_loss = valid_loss
    if curr_loss < best_loss:
        save_model_state(model, model_dir2 + f'/best/')
        # dd = {"pars" : [ learning_rate, latent_dim, num_epochs ]}
        # json.dump(dd, open(model_dir2 +"/best/info.json" , mode='w'))
        print(f"Model Saved | Loss impoved from {best_loss} -----> {curr_loss}")
        best_loss = curr_loss
        counter   = 0
    else:
        counter += 1    
    return best_loss, counter    


def save_model_state(model, model_dir2):
    os.makedirs(model_dir2 , exist_ok=True)
    model.save_weights( model_dir2 + f'/model_keras_weights.h5')

    
    
def train_stop(counter, patience) :
    if counter == patience :
        log(f"Model not improved from {patience} epochs...............")
        log("Training Finished..................")
        return True
    return False


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
    
    
    

#################################################################################    
from tensorflow.python.keras.utils.data_utils import Sequence    
class CustomDataGenerator(Sequence):
    def __init__(self, x, y, batch_size=32, augmentations=None):
        self.x          = x
        self.y          = y
        self.batch_size = batch_size
        self.augment    = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = []
        for y_head in self.y:
            batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size])
        
        if self.augment is not None:
            batch_x = np.stack([self.augment(image=x)['image'] for x in batch_x], axis=0)
        return (batch_x, *batch_y)

    


#################################################################################   
from PIL import Image
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
                            
                            
                            
                            
###############################################################################
###############################################################################
from albumentations.core.transforms_interface import ImageOnlyTransform
import tensorflow as tf
class SprinklesTransform(ImageOnlyTransform):
    def __init__(self, num_holes=30, side_length=5, always_apply=False, p=1.0):
        from tf_sprinkles import Sprinkles
        super(SprinklesTransform, self).__init__(always_apply, p)
        self.sprinkles = Sprinkles(num_holes=num_holes, side_length=side_length)
    
    def apply(self, image, **params):
        if isinstance(image, PIL.Image.Image):   image = tf.constant(np.array(image), dtype=tf.float32)            
        elif isinstance(image, np.ndarray):      image = tf.constant(image, dtype=tf.float32)
        return self.sprinkles(image).numpy()

    
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, Resize
)

#image_size = 64
train_augments = Compose([
    #Resize(image_size, image_size, p=1),
    HorizontalFlip(p=0.5),
    #RandomContrast(limit=0.2, p=0.5),
    #RandomGamma(gamma_limit=(80, 120), p=0.5),
    #RandomBrightness(limit=0.2, p=0.5),
    #HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
    #                   val_shift_limit=10, p=.9),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, 
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
    # ToFloat(max_value=255),
    SprinklesTransform(p=0.5),
])

test_augments = Compose([
    # Resize(image_size, image_size, p=1),
    # ToFloat(max_value=255)
])



###############################################################################
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








# train_data = CustomDataGenerator(x_train, y_train, augmentations=train_augments)
# val_data   = CustomDataGenerator(x_train, y_train, augmentations=test_augments)


# # Data Augmentation with built-in Keras functions
# train_gen = keras.preprocessing.image.ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=20,
#     height_shift_range=20,
#     brightness_range=[0.2, 1.0],
#     shear_range=20,
#     horizontal_flip=True,
#     rescale=1./255
# )

# test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# train_data = train_gen.flow(x_train, y_train)
# val_data   = test_gen.flow(x_val, y_val)












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


#########################################################################################
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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

