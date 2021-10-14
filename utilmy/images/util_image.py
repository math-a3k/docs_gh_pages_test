# -*- coding: utf-8 -*-
HELP="""

 utils images

"""
import os,io, numpy as np, sys, glob, time, copy, json, functools, pandas as pd
from typing import Union
from box import Box

os.environ['MPLCONFIGDIR'] = "/tmp/"

import io, cv2,  tifffile.tifffile
from PIL import Image
from skimage import morphology

import matplotlib

try :
   from albumentations.core.transforms_interface import ImageOnlyTransform
   import diskcache as dc 

except : pass

from utilmy import pd_read_file

################################################################################################
verbose = 0

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbose >1 : print(*s, flush=True)


def help():
    from utilmy import help_create
    ss  = HELP
    ss += help_create("utilmy.deeplearning.util_image")
    print(ss)



################################################################################################
def test():
    pass




################################################################################################
def prep_images(image_paths, nmax=10000000):
  images = []
  for i in range(len(image_paths)):
    if i > nmax : break
    image =  prep_image(image_paths[i] )
    images.append(image)
  return images


def prep_images2(image_paths, nmax=10000000):
    images = []
    original_first_image = None
    for i in range(len(image_paths)):
        if i > nmax: break

        image_path = image_paths[i]
        fname = str(image_path).split("/")[-1]
        id1 = fname.split(".")[0]

        if (i + 100) % 100 == 0: print(fname, id1)

        image = matplotlib.image.imread(image_path)

        if images == []:
            temp = (image / 255)
            original_first_image = temp.astype('float32')
        resized_image = cv2.resize(image, dsize=(xdim, ydim), interpolation=cv2.INTER_CUBIC)

        if resized_image.shape == (xdim, ydim, cdim):
            resized_image = resized_image / 255
            images.append(resized_image.astype('float32'))
    return images, original_first_image


def prep_image(image_path:str, xdim=1, ydim=1):
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


def prep_images_multi(image_path_list:list, prepro_image_fun=None, npool=1):
    """ Parallel processing

    """
    from multiprocessing.dummy import Pool    #### use threads for I/O bound tasks

    pool = Pool(npool)
    res  = pool.map(prepro_image_fun, image_path_list)
    pool.close() ;     pool.join()  ; pool = None

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



################################################################################################
def image_cache_create():
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


def image_cache_check(db_path:str="db_images.cache", dirout:str="tmp/", tag="cache1"):
    ##### Write some sample images  from cache #############################
    import diskcache as dc
    cache   = dc.Cache(db_path, size_limit= 100 * 10**9, timeout= 5 )
    log('Nimages', len(cache) )

    log('### Check writing on disk  ###########################')
    dir_check = dirout + f"/{tag}/"
    os.makedirs(dir_check, exist_ok=True)
    for i, key in enumerate(cache) :
        if i > 10: break
        img = cache[key]
        img = img[:, :, ::-1]
        key2 = key.split("/")[-1]
        cv2.imwrite( dir_check + f"/{i}_{key2}"  , img)
    log( dir_check )


def image_cache_save(image_path_list:str="db_images.cache", db_dir:str="tmp/", tag="cache1"):
    ##### Write some sample images  from cache #############################
    import diskcache as dc
    cache   = dc.Cache(db_dir, size_limit= 100 * 10**9, timeout= 5 )
    log('Nimages', len(cache) )


    log('### Check writing on disk  ###########################')
    for img_path in image_path_list:
        img = image_read(img_path)
        cache[img_path] = img


def image_check_npz(path_npz,  keys=['train'], path="", tag="", n_sample=3, renorm=True):
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



##############################################################################
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




################################################################
def image_show_in_row(image_list:dict=None):
    """ # helper function for data visualization
    Plot images in one row.
    
    """
    import matplotlib.pyplot as plt
   
    if isinstance(image_list, list): 
         image_dict = {i:x for (i,x) in enumerate(image_list) }
    else :
         image_dict = image_list
      
    n = len(image_dict)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(image_dict.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()



def image_resize_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Resizes a image and maintains aspect ratio
    # Grab the image size and initialize dimensions
    import cv2
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
    
    
    
    
    
    
    


############################################################################
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
     if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
         padColor = [padColor]*3

     # scale and pad
     scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
     scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=padColor)

     return scaled_img


def image_resize(out_dir=""):
    """     python prepro.py  image_resize

          image white color padded

    """
    import cv2, gc, diskcache

    in_dir = data_dir + "/train_nobg"
    out_dir = data_dir + "/train_nobg_256/"

    nmax = 500000000
    global xdim, ydim
    xdim = 256
    ydim = 256
    padcolor = 0  ## 0 : black

    os.makedirs(out_dir, exist_ok=True)
    log('target folder', out_dir);
    time.sleep(5)

    def prepro_image3b(img_path):
        try:
            fname = str(img_path).split("/")[-1]
            id1 = fname.split(".")[0]
            img_path_new = out_dir + "/" + fname

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = util_image.image_resize_pad(img, (xdim, ydim), padColor=padcolor)  ### 255 white, 0 for black
            img = img[:, :, ::-1]
            cv2.imwrite(img_path_new, img)
            # print(img_path_new)
            return [1], "1"
        except Exception as e:
            # print(image_path, e)
            return [], ""

    log("#### Process  ######################################################################")
    image_list = sorted(list(glob.glob(f'/{in_dir}/*.*')))
    image_list = image_list[:nmax]
    log('Size Before', len(image_list))

    log("#### Saving disk  #################################################################")
    images, labels = prepro_images_multi(image_list, prepro_image=prepro_image3b)
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

       

def image_padding_generate( paddings_number: int = 1, min_padding: int = 1, max_padding: int = 1
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


def image_check():
    """     python prepro.py  image_check

          image white color padded

    """
    # print( 'nf files', len(glob.glob("/data/workspaces/noelkevin01/img/data/fashion/train_nobg_256/*")) )
    nmax = 100000
    global xdim, ydim
    xdim = 64
    ydim = 64

    log("### Load  ##################################################")
    # fname    = f"/img_all{tag}.cache"
    # fname    = f"/img_fashiondata_64_64-100000.cache"
    # fname = "img_train_nobg_256_256-100000.cache"
    fname = "img_train_r2p2_40k_nobg_256_256-100000.cache"
    fname = "img_train_r2p2_40k_nobg_256_256-100000.cache"

    log('loading', fname)

    import diskcache as dc
    db_path = data_train + fname
    cache = dc.Cache(db_path)

    lkey = list(cache)
    print('Nimages', len(lkey))

    ### key check:
    # df = pd_read_file("/data/workspaces/noelkevin01/img/data/fashion/csv/styles_df.csv" )
    # idlist = df['id']

    log('### writing on disk  ######################################')
    dir_check = data_train + "/zcheck/"
    os.makedirs(dir_check, exist_ok=True)
    for i, key in enumerate(cache):
        if i > 10: break
        img = cache[key]
        img = img[:, :, ::-1]
        print(key)
        key2 = key.split("/")[-1]
        cv2.imwrite(dir_check + f"/{key2}", img)



    



    
    



