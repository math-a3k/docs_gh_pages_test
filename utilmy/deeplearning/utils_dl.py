# -*- coding: utf-8 -*-
HELP="""

 utils in Keras

"""
import os,io, numpy as np, sys, glob, time, copy, json, functools, pandas as pd
from typing import Union


os.environ['MPLCONFIGDIR'] = "/tmp/"

import io, cv2,  tifffile.tifffile
import tensorflow as tf, tensorflow_addons as tfa
from tensorflow.keras import layers, regularizers
from tensorflow.python.keras.utils.data_utils import Sequence    
from PIL import Image 
from albumentations.core.transforms_interface import ImageOnlyTransform
from skimage import morphology
from sklearn.metrics import accuracy_score
from box import Box
import diskcache as dc



from utilmy import pd_read_file


################################################################################################
verbose = 0

def log(*s):
    print(*s, flush=True)


def log2(*s):
    if verbose >1 : print(*s, flush=True)


def help():
    from utilmy import help_create
    ss  = ""
    ss += HELP
    ss += help_create("utilmy.deeplearning.util_dl")
    print(ss)



################################################################################################
def test():
    pass






################################################################################################
def prepro_image(image_path):
    mean   = [0.5]
    std    = [0.5]
    try :
        # fname      = str(image_path).split("/")[-1]
        # id1        = fname.split(".")[0]
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
     if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
         padColor = [padColor]*3

     # scale and pad
     scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
     scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=padColor)

     return scaled_img






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
    in_dir   = "/data//"
    out_dir  = "/data/wonoface/"
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



def image_save_tocache(out_dir, name="cache1"):
    ##### Write some sample images  ########################
    import diskcache as dc
    db_path = "/data/woshion/train_npz/small/img_train_r2p2_70k_clean_nobg_256_256-100000.cache"
    cache   = dc.Cache(db_path)
    print('Nimages', len(cache) )

    log('### writing on disk  ######################################')
    dir_check = out_dir + f"/{name}/"
    os.makedirs(dir_check, exist_ok=True)
    for i, key in enumerate(cache) :
        if i > 10: break
        img = cache[key]
        img = img[:, :, ::-1]
        key2 = key.split("/")[-1]
        cv2.imwrite( dir_check + f"/{i}_{key2}"  , img)
    log( dir_check )












































