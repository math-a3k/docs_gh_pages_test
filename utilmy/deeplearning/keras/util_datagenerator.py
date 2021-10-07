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
def log(*s):
    print(*s, flush=True)

    



#################################################################################    
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















