# -*- coding: utf-8 -*-
HELP = """
 utils keras for dataloading
"""
import os, numpy as np, glob, pandas as pd
from pathlib import Path
from box import Box

import tensorflow as tf
from tensorflow import keras

# import tifffile.tifffile
# from skimage import morphology
import PIL, cv2
from PIL import Image

from albumentations import (
    Compose, HorizontalFlip, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, Resize,
)
from albumentations.core.transforms_interface import ImageOnlyTransform

#### Types
from albumentations.core.composition import Compose
from numpy import ndarray
from pandas.core.frame import DataFrame
from typing import List, Optional, Tuple




#############################################################################################
from utilmy import log, log2


def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    ss = HELP + help_create("utilmy.deeplearning.keras.util_layers")
    print(ss)




############################################################################################
def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    test()
    test1()
    test2()


def test():
    """Tests train and test images transformation functions"""
    image_size = 64
    train_transforms = Compose([
        Resize(image_size, image_size, p=1),
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        Transform_sprinkle(num_holes=10, side_length=10, p=0.5),
    ])

    test_transforms = Compose([
        Resize(image_size, image_size, p=1),
        ToFloat(max_value=255)
    ])


def test1():
    """Tests image dataloader"""
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()

    train_loader = DataLoader_img(X_train, y_train)
    for i, (image, label) in enumerate(train_loader):
        print('Training : ')
        print(f'image shape : {image.shape}')
        print(f'label shape : {label.shape}')
        break


def test2():  # using predefined df and model training using model.fit()
    """Tests model training process"""
    from PIL import Image
    from pathlib import Path
    from tensorflow import keras
    from tensorflow.keras import layers

    def get_model():
        model = keras.Sequential([
            keras.Input(shape=(28, 28, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(num_labels, activation="softmax"),
        ])
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7), metrics=["accuracy"])
        return model

    #####################################################
    label_file    = 'df.csv'

    dir_img_path  = 'random_images/'
    dir_img       = Path(dir_img_path).as_posix()
    num_images    = 256
    num_labels    = 2

    df = test_create_random_images_ds((28, 28, 3), num_images=num_images, num_labels=num_labels, dirout=dir_img_path)
    # df = create_random_images_ds2(img_shape=(10,10,2), num_images = 10,
    #                              dirout =folder_name,  n_class_perlabel=2,  cols_labels = [ 'label', ] )
    df.to_csv(label_file, index=False)

    label_cols = ['label']
    label_dict = {ci: df[ci].unique() for ci in label_cols}

    log('############   without Transform')
    dt_loader = DataLoader_imgdisk(dir_img, label_dir=df, label_dict=label_dict,
                                   col_img='uri', batch_size=32, transforms=None)

    for i, (image, label) in enumerate(dt_loader):
        log(f'image shape : {(image).shape}')
        log(f'label shape : {(label).shape}')
        break

    model = get_model()
    model.fit(dt_loader, epochs=1, )


    log('############   with Transform + model fit ')
    trans_train = Compose([
        # Resize(image_size, image_size, p=1),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1,
            rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        # ToFloat(max_value=255),
        Transform_sprinkle(p=0.5),
    ])
    dt_loader = DataLoader_imgdisk(dir_img, label_dir=df, label_dict=label_dict,
                                   batch_size=32, col_img='uri', transforms=trans_train)

    for i, (image, label) in enumerate(dt_loader):
        log(f'image shape : {(image).shape}')
        log(f'label shape : {(label).shape}')
        break

    model = get_model()
    model.fit(dt_loader, epochs=1, )




############################################################################################
def test_create_random_images_ds(img_shape: Tuple[int, int, int], num_images: int=10, dirout: str='random_images/',
                                 return_df: bool=True, num_labels: int=2,
                                 label_cols: List[str]=['label']) -> DataFrame:
    """Creates datasets with random images for testing"""
    if not os.path.exists(dirout):
        os.mkdir(dirout)
    for n in range(num_images):
        filename = f'{dirout}/{n}.jpg'
        rgb_img = np.random.rand(img_shape[0], img_shape[1], img_shape[2]) * 255
        image = Image.fromarray(rgb_img.astype('uint8')).convert('RGB')
        image.save(filename)

    label_dict = []

    files = [Path(i) for i in glob.glob(dirout + '/*.jpg')]
    for i in enumerate(label_cols):
        label_dict.append(np.random.randint(num_labels, size=(num_images)))

        df = pd.DataFrame(list(zip(files, *label_dict)), columns=['uri'] + label_cols)
        if return_df:
            return df


def test_create_random_images_ds2(img_shape: Tuple[int, int, int]=(10, 10, 2), num_images: int=10,
                                  dirout:str='random_images/', n_class_perlabel: int=7,
                                  cols_labels: List[str]=['gender', 'color', 'size'],
                                  col_img: str='uri') -> DataFrame:
    """ Image + labels into Folder + csv files.
        Multiple label:
    """
    os.makedirs(dirout, exist_ok=True)
    for n in range(num_images):
        filename = f'{dirout}/{n}.jpg'
        rgb_img  = np.random.rand(img_shape[0],img_shape[1],img_shape[2]) * 255
        image    = Image.fromarray(rgb_img.astype('uint8')).convert('RGB')
        image.save(filename)

    files = [fi.replace("\\", "/") for fi in glob.glob(dirout + '/*.jpg')]
    df = pd.DataFrame(files, columns=[col_img])

    ##### Labels
    for ci in cols_labels:
        df[ci] = np.random.choice(np.arange(0, n_class_perlabel), len(df), replace=True)
    return df


##########################################################################################
class Transform_sprinkle(ImageOnlyTransform):
    def __init__(self, num_holes: int=30, side_length: int=5, always_apply: bool=False, p: float=1.0):
        """ Remove Patches of the image, very efficient for training.
        """
        from tf_sprinkles import Sprinkles
        super(Transform_sprinkle, self).__init__(always_apply, p)
        self.sprinkles = Sprinkles(num_holes=num_holes, side_length=side_length)

    def apply(self, image: ndarray, **params) -> ndarray:
        """ Transform_sprinkle:apply
        Args:
            image (function["arg_type"][i]) :     
            **params:     
        Returns:
           
        """
        if isinstance(image, PIL.Image.Image):
            image = tf.constant(np.array(image), dtype=tf.float32)
        elif isinstance(image, np.ndarray):
            image = tf.constant(image, dtype=tf.float32)
        return self.sprinkles(image).numpy()


def transform_get_basic(pars: dict = None):
    """"  Get Transformation Class for Datalaoder
          Basic version
    cc.resize.p = 1

    """
    if pars is None:
        cc = Box({})
        cc.height = 64
        cc.width = 64
    else:
        Box(pars)

    train_transforms = Compose([
        Resize(cc.height, cc.width, p=1),
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
        Transform_sprinkle(num_holes=10, side_length=10, p=0.5),
    ])

    test_transforms = Compose([
        Resize(cc.height, cc.width, p=1),
        ToFloat(max_value=255)
    ])
    return train_transforms, test_transforms


##########################################################################################
class DataLoader_imgdisk(tf.keras.utils.Sequence):
    """Custom DataGenerator using Keras Sequence for images on disk
        df_label format :
        id, uri, cat1, cat2, cat3, cat1_onehot, cat1_onehot, ....
    """
    def __init__(self, img_dir:str="images/", label_dir:str=None, label_dict:dict=None,
                 col_img: str='uri', 
                 batch_size:int=8, transforms: Optional[Compose]=None, shuffle: bool=True, 
                 label_imbalance_col: str=None, label_imbalance_min_sample:int=100):
        """
        Args:
            img_dir (Path(str)): String path to images directory
            label_dir (DataFrame): Dataset for Generator
            label_cols (list): list of cols for the label (multi label)
            label_dict (dict):    {label_name : list of values }
            split (str, optional): split for train or test. Defaults to 'train'.
            batch_size (int, optional): batch_size for each batch. Defaults to 8.
            transforms (str, optional): type of transformations to perform on images. Defaults to None.
        """
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle    = shuffle

        self.image_dir  = img_dir
        self.col_img    = col_img


        from utilmy import pd_read_file
        dflabel     = pd_read_file(label_dir)
        dflabel     = dflabel.dropna()

        ### Imablance label
        if label_imbalance_col is not None:
            dflabel = pd_sample_equal(dflabel, col= label_imbalance_col, nsample = label_imbalance_min_sample)

        self.dflabel    = dflabel
        self.label_cols = list(label_dict.keys())
        self.label_df   = pd_to_onehot(dflabel, labels_dict=label_dict)  ### One Hot encoding
        self.label_dict = label_dict

        assert col_img in self.label_df.columns


    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(12)
            self.label_df = self.label_df.sample(frac=1).reset_index(drop=True)
            # np.random.shuffle(self.labels.reset_index(drop=True))

    def __len__(self) -> int:
        """ DataLoader_imgdisk:__len__
        Args:
        Returns:
           
        """
        return int(np.ceil(len(self.label_df) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        """ DataLoader_imgdisk:__getitem__
        Args:
            idx (function["arg_type"][i]) :     
        Returns:
           
        """
        # for X,y in mydatagen :
        batch_x, batch_y = self.__get_data(idx, self.batch_size)
        return np.array(batch_x), np.array(batch_y)


    def __get_data(self, idx, batch=8):
        """ DataLoader_imgdisk:__get_data
        Args:
            idx:     
            batch:     
        Returns:
           
        """
        # Create batch targets
        df_batch    = self.label_df[idx * batch:(idx + 1) * self.batch_size]
        batch_x, batch_y = [], []   #  list of output heads

        ##### Xinput
        for ii, x in df_batch.iterrows():
            img =  np.array(Image.open(x['uri']).convert('RGB') )
            batch_x.append(img)

        if self.transforms is not None:
            batch_x = np.stack([self.transforms(image=x)['image'] for x in batch_x], axis=0)

        #### ylabel
        for ci in self.label_cols:
            v = [x.split(",") for x in df_batch[ci + "_onehot"]]
            v = np.array([[int(t) for t in vlist] for vlist in v])
            batch_y.append(v)

        return (batch_x, *batch_y)


##########################################################################################
class DataLoader_img(tf.keras.utils.Sequence):
    """Custom DataGenerator using keras Sequence
    Args: x (np array): The input samples from the dataset
          y (np array): The labels from the dataset
          batch_size (int, optional): batch size for the samples. Defaults to 32.
          augmentations (str, optional): perform augmentations to the input samples. Defaults to None.
    """

    def __init__(self, x: ndarray, y: ndarray, batch_size: int=32, transform: None=None):
        """ DataLoader_img:__init__
        Args:
            x (function["arg_type"][i]) :     
            y (function["arg_type"][i]) :     
            batch_size (function["arg_type"][i]) :     
            transform (function["arg_type"][i]) :     
        Returns:
           
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augment = transform

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[ndarray, ndarray]:
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # for y_head in self.y:                                                         ----
        #     batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size]) ----
        if self.augment is not None:
            batch_x = np.stack([self.augment(image=x)['image'] for x in batch_x], axis=0)
        # return (batch_x, *batch_y)                                                    ----
        return (batch_x, batch_y)


##########################################################################################
if 'utils':
    def pd_sample_equal(df, col, nsampl=100):
        ### Stratified under-sampling, using the least frequent from category col
        n2  = min(nsample, df[col].value_counts().min())
        df_ = df.groupby(col).apply(lambda x: x.sample(n = n2, replace=True))
        df_.index = df_.index.droplevel(0)
        return df_

    def pd_cols_unique_count(df, cols_exclude:list=[], nsample=-1) :
        ### Return cadinat=lity
        clist = {}
        for ci in df.columns :
            ctype   = df[ci].dtype
            if nsample == -1 :
                nunique = len(df[ci].unique())
            else :
                nunique = len(df.sample(n= nsample, replace=True)[ci].unique())

            if 'float' in  str(ctype) and ci not in cols_exclude and nunique > 5 :
               clist[ci] = 0
            else :
               clist[ci] = nunique

        return clist


    def pd_label_normalize_freq(df, cols:list)  -> pd.DataFrame :
        """
            Resample each class


            for ci in cols:
                nuniques[ci] =  df[ci].nunique()



        """
        pass


    def get_data_sample(batch_size, x_train, ylabel_dict, ylabel_name):  # name changed
        """ Get a data sample with batch size from dataset
        Args:
            batch_size (int): Provide a batch size for sampling
            x_train (list): Inputs from the dataset
            ylabel_dict (list): True labels for the dataset
            ylabel_name(list): Samples to select from these columns
        Returns:
            x (numpy array): Selected samples of size batch_size
            y_label_list (list): List of labels from selected samples
        """
        for k, y0_list in ylabel_dict:
            break
        i_select = np.random.choice(np.arange(len(y0_list)), size=batch_size, replace=False)

        #### Images
        x = np.array([x_train[i] for i in i_select])

        #### list of y_onehot Labels  [y1, y2, y3, y4]
        # labels_col   = [  'gender', 'masterCategory', 'subCategory', 'articleType' ] #*To make user-defined
        y_label_list = []
        for ci in ylabel_name:
            v = ylabel_dict[ci][i_select]
            y_label_list.append(v)

        return x, y_label_list


    def pd_get_onehot_dict(df, labels_col: list, dfref=None, category_dict=None):  # name changed
        """
        Args:
            df (DataFrame): Actual DataFrame
            dfref (DataFrame): Reference DataFrame
            labels_col (list): List of label columns
            category_dict:  Provide the category list
        Returns:
            dictionary: label_columns, count
        """
        if dfref is not None:
            for ci in labels_col:
                df[ci] = pd.Categorical(df[ci], categories=dfref[ci].unique())

        if category_dict is not None:
            for ci, catval in category_dict.items():
                df[ci] = pd.Categorical(df[ci], categories=catval)

        labels_val = {}
        labels_cnt = {}
        for ci in labels_col:
            dfi_1hot = pd.get_dummies(df, columns=[ci])  ### OneHot
            dfi_1hot = dfi_1hot[[t for t in dfi_1hot.columns if ci in t]].values  ## remove no OneHot
            labels_val[ci] = dfi_1hot
            labels_cnt[ci] = df[ci].nunique()
            assert dfi_1hot.shape[1] == labels_cnt[ci], labels_cnt

        print(labels_cnt)
        return labels_val, labels_cnt


    def pd_merge_labels_imgdir(dflabels, img_dir="*.jpg", labels_col=[]):  # name changed
        """One Hot encode label_cols
        #    id, uri, cat1, cat2, .... , cat1_onehot
        Args:
            dflabels (DataFrame): DataFrame to perform one hot encoding on
            img_dir (Path(str)): String Path /*.png to image directory
            labels_col (list): Columns to perform One Hot encoding on. Defaults to []
        Returns:
            DataFrame: One Hot encoded DataFrame
        """

        import glob
        fpaths = glob.glob(img_dir)
        fpaths = [fi for fi in fpaths if "." in fi.split("/")[-1]]
        log(str(fpaths)[:100])

        df = pd.DataFrame(fpaths, columns=['uri'])
        log(df.head(1).T)
        df['id'] = df['uri'].apply(lambda x: x.split("/")[-1].split(".")[0])
        # df['id']   = df['id'].apply( lambda x: int(x) )
        df = df.merge(dflabels, on='id', how='left')

        # labels_col = [  'gender', 'masterCategory', 'subCategory', 'articleType' ]
        for ci in labels_col:
            dfi_1hot = pd.get_dummies(df, columns=[ci])  ### OneHot
            dfi_1hot = dfi_1hot[[t for t in dfi_1hot.columns if ci in t]]  ## keep only OneHot
            df[ci + "_onehot"] = dfi_1hot.apply(lambda x: ','.join([str(t) for t in x]), axis=1)
            #####  0,0,1,0 format   log(dfi_1hot)
        return df


    def pd_to_onehot(dflabels: DataFrame, labels_dict: dict = None) -> DataFrame:  # name changed
        """One Hot encode label_cols for predefined df
        #    id, uri, cat1, cat2, .... , cat1_onehot
        Args:
            dflabels (DataFrame): DataFrame to perform one hot encoding on
            labels_col (list): Columns to perform One Hot encoding on. Defaults to []
        Returns: DataFrame: One Hot encoded DataFrame
        """
        if labels_dict is not None:
            for ci, catval in labels_dict.items():
                dflabels[ci] = pd.Categorical(dflabels[ci], categories=catval)
        labels_col = labels_dict.keys()

        for ci in labels_col:
            dfi_1hot = pd.get_dummies(dflabels, columns=[ci])  ### OneHot
            dfi_1hot = dfi_1hot[[t for t in dfi_1hot.columns if ci in t]]  ## keep only OneHot
            dflabels[ci + "_onehot"] = dfi_1hot.apply(lambda x: ','.join([str(t) for t in x]), axis=1)
            #####  0,0,1,0 format   log(dfi_1hot)

        return dflabels


    ####################################################################################
    def tf_byte_feature(value):
        if not isinstance(value, (tuple, list)):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def tf_int64_feature(value):
        if not isinstance(value, (tuple, list)):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    def tf_float_feature(value):
        if not isinstance(value, (tuple, list)):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def build_tfrecord(x, tfrecord_out_path, max_records, dims=(23, 23, 3)):

        extractor = tf.keras.applications.ResNet50V2(
            include_top=False, weights='imagenet',
            input_shape=(dims[0], dims[1], dims[2]),
            pooling='avg'
        )
        with tf.io.TFRecordWriter(tfrecord_out_path) as writer:
            id_cnt = 0
            for i, (_, images, *_) in enumerate(x):
                if i > max_records:
                    break
                batch_embedding = extractor(images, training=False).numpy().tolist()
                for embedding in batch_embedding:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'id': tf_byte_feature(str(id_cnt).encode('utf-8')),
                        'embedding': tf_float_feature(embedding),
                    }))
                    writer.write(example.SerializeToString())
                    id_cnt += 1
        return tfrecord_out_path


###################################################################################################
if __name__ == "__main__":
    import fire

    fire.Fire()


"""
class Dataloader_img_disk_custom(tf.keras.utils.Sequence):        
    def __init__(self, image_dir, label_path, class_dict,
                 split='train', batch_size=8, transforms=None, shuffle=True):
        self.image_dir = image_dir
        # self.labels = np.loadtxt(label_dir, delimiter=' ', dtype=np.object)
        self.class_dict = class_dict
        self.image_ids, self.labels = self._load_data(label_path)
        self.num_classes = len(class_dict)
        self.batch_size = batch_size
        self.transforms = transforms
        self.shuffle = shuffle
    def _load_data(self, label_path):
        df = pd.read_csv(label_path, error_bad_lines=False, warn_bad_lines=False)
        keys = ['id'] + list(self.class_dict.keys())
        df = df[keys]
        # Get image ids
        df = df.dropna()
        image_ids = df['id'].tolist()
        df = df.drop('id', axis=1)
        labels = []
        for col in self.class_dict:
            categories = pd.get_dummies(df[col]).values
            labels.append(categories)
        return image_ids, labels
    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(12)
            indices = np.arange(len(self.image_ids))
            np.random.shuffle(indices)
            self.image_ids = self.image_ids[indices]
            self.labels = [label[indices] for label in self.labels]
    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))
    def __getitem__(self, idx):
        batch_img_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        for image_id in batch_img_ids:
            # Load image
            image = np.array(Image.open(os.path.join(self.image_dir, '%d.jpg' % image_id)).convert('RGB'))
            batch_x.append(image)
        batch_y = []
        for y_head in self.labels:
            batch_y.append(y_head[idx * self.batch_size:(idx + 1) * self.batch_size, :])
        if self.transforms is not None:
            batch_x = np.stack([self.transforms(image=x)['image'] for x in batch_x], axis=0)
        return (idx, batch_x, *batch_y)
"""