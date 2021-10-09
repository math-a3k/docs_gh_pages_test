# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

util_feature: input/output is pandas
"""
import copy
import os
from collections import OrderedDict
import numpy as np, pandas as pd
from tempfile import gettempdir
import requests
import cgi
import os
import re
import uuid


#############################################################################################
# print("os.getcwd", os.getcwd())
def log(*s, n=0, m=1, **kw):
    sspace = "#" * n
    sjump = "\n" * m

    ### Implement Logging
    print(sjump, sspace, s, sspace, flush=True, **kw)

class dict2(object):
    def __init__(self, d):
        self.__dict__ = d



from collections.abc import Mapping
class dictLazy(Mapping):
    """       .parquet --->  disk retrieval in pandas
         hdfs     --->  disk retrieval in pandas
        # Lazy dict allows storing function and argument pairs when initializing the dictionary,
        # it calculates the value only when fetching it.
        # In this examole, if the key starts with '#', it would accept a (function, args) tuple as value and
        # returns the calculated result when fetching the values.

        # Initialize a lazy dict
        d = ictLasy(
            {
                '#1': (lambda x: x + 1, 0),
                '#2': (lambda x: x + 2, 0),
                '#3': (lambda x: x + 3, 0)
            }
        )
        A collection of files, nrows

# Example of iterator=True. Note iterator=False by default.
reader = pd.read_csv('some_data.csv', iterator=True)
reader.get_chunk(100)
This gets the first 100 rows, running through a loop gets the next 100 rows and so on.

import pandas as pd
from glob import glob
files = sorted(glob('dat.parquet/part*'))

data = pd.read_parquet(files[0],engine='fastparquet')
for f in files[1:]:
    data = pd.concat([data,pd.read_parquet(f,engine='fastparquet')])

    # Save into 50,000 row chunks,
# so we should get file saved into two chunks.

df.to_parquet('/users/nick/desktop/test.parquet',
              engine='fastparquet',
              row_group_offsets=50000)

    # Then we have to read it in using the `fastparquet`
    # library itself (there's no way to do this directly from
    # pandas I'm afraid):

    from fastparquet import ParquetFile
    pf = ParquetFile('/users/nick/desktop/test.parquet')

    # Iterates over row groups
    for rg in pf.iter_row_groups():
        print(rg)



    """
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        if key.startswith('#'):
            path = self._raw_dict.__getitem__(key)

            if 'hdfs:' in path :
                valx = load_hdfs(path)

            elif '.parquet' in path :
                pass

            elif 'spark:'  in path  :
                pass
            return valx
        else :
            return self._raw_dict.__getitem__(key)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)




def pd_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    from scipy.sparse import lil_matrix
    arr = lil_matrix(df.shape, dtype=np.float32)
    for i, col in enumerate(df.columns):
        ix = df[col] != 0
        arr[np.where(ix), i] = 1

    return arr.tocsr()



def pd_to_keyvalue_dict(dfa, colkey= [ "shop_id", "l2_genre_id" ]   , col_list='item_id',  to_file=""):
    import copy, pickle
    dfa = copy.deepcopy(dfa)
    def to_key(x):
        return "_".join([ str(x[t]) for t in colkey  ])

    dfa["__key"] = dfa.apply( lambda x :  to_key(x) , axis=1  )
    # dd = pd.DataFrame( dfa.groupby([ "__key"  ]).apply(lambda dfi :  [  int(t) for t in  dfi['item_id'].values] ) )
    dd = pd.DataFrame( dfa.groupby([ "__key"  ]).apply(lambda dfi :    dfi[col_list].values ) )
    dd.columns = ['__val']
    dd = dd.to_dict("dict")['__val']
    save(dd, to_file)








        
#############################################################################################        
# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
https://docs.python-guide.org/writing/logging/
https://docs.python.org/3/howto/logging-cookbook.html

python util_log.py test_log


"""
import logging
import os
import random
import socket
import sys
from logging.handlers import TimedRotatingFileHandler
import datetime
import yaml

################### Logs #################################################################
APP_ID  = __file__ + "_" + str(os.getpid()) + "_" + str(socket.gethostname())
APP_ID2 = str(os.getpid()) + "_" + str(socket.gethostname())

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logfile.log")

FORMATTER_0 = logging.Formatter("%(message)s")
FORMATTER_1 = logging.Formatter("%(asctime)s,  %(name)s, %(levelname)s, %(message)s")
FORMATTER_2 = logging.Formatter("%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s")
FORMATTER_3 = logging.Formatter("%(asctime)s  %(levelname)s %(message)s")
FORMATTER_4 = logging.Formatter("%(asctime)s, %(process)d, %(filename)s,    %(message)s")

FORMATTER_5 = logging.Formatter(
    "%(asctime)s, %(process)d, %(pathname)s%(filename)s, %(funcName)s, %(lineno)s,  %(message)s"
)


#########################################################################################
def create_appid(filename):
    # appid  = filename + ',' + str(os.getpid()) + ',' + str( socket.gethostname() )
    appid = filename + "," + str(os.getpid())
    return appid


def create_logfilename(filename):
    return filename.split("/")[-1].split(".")[0] + ".log"


def create_uniqueid():
    return datetime.datetime.now().strftime(  "_%Y%m%d%H%M%S_"  )   + str(random.randint(1000, 9999))
    # return arrow.utcnow().to("Japan").format("_YYYYMMDDHHmmss_") + str(random.randint(1000, 9999))


########################################################################################
################### Logger #############################################################
class logger_class(object):
    """
    Higher level of verbosity =1, 2, 3
    logger = logger_class()

    def log(*s):
        logger.log(*s, level=1)

    def log1(*s):
        logger.log(*s, level=1)

    """
    def __init__(self, config_file=None, verbose=True) :
        self.config     = self.load_config(config_file)
        if verbose: print(self.config)
        d = self.config['logger_config']
        self.logger     = logger_setup( **d )
        self.level_max  = self.config.get('verbosity', 1)


    def load_config(self, config_file_path=None) :
        try :
            if config_file_path is None :
                config_file_path = 'config.yaml'

            with open(config_file_path, 'r') as f:
                return yaml.load(f)

        except Exception as e :
            return {'logger_config': {}, 'verbosity': 1}  ## Default parameters


    def log(self,*s, level=1) :
        if level <= self.level_max : 
            self.logger.info(*s)


    def debug(self,*s, level=1) :
        if level <= self.level_max : 
            self.logger.debug(*s)



##########################################################################################
##########################################################################################
def logger_setup(logger_name=None, log_file=None, formatter='FORMATTER_0', isrotate=False, 
    isconsole_output=True, logging_level='info',):
    """
    my_logger = util_log.logger_setup("my module name", log_file="")
    APP_ID    = util_log.create_appid(__file__ )
    def log(*argv):
      my_logger.info(",".join([str(x) for x in argv]))
  
   """
    logging_level = {  'info':logging.INFO, 'debug' : logging.DEBUG }[logging_level]
    formatter     = {'FORMATTER_0': FORMATTER_0, 'FORMATTER_1': FORMATTER_1}.get(formatter, formatter)

    if logger_name is None:
        logger = logging.getLogger()  # Gets the root logger
    else:
        logger = logging.getLogger(logger_name)

    logger.setLevel(logging_level)  # better to have too much log than not enough

    if isconsole_output:
        logger.addHandler(logger_handler_console(formatter))

    if log_file is not None:
        logger.addHandler(
            logger_handler_file(formatter=formatter, log_file_used=log_file, isrotate=isrotate)
        )

    # with this pattern, rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def logger_handler_console(formatter=None):
    formatter = FORMATTER_1 if formatter is None else formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler


def logger_handler_file(isrotate=False, rotate_time="midnight", formatter=None, log_file_used=None):
    formatter = FORMATTER_1 if formatter is None else formatter
    log_file_used = LOG_FILE if log_file_used is None else log_file_used
    if isrotate:
        print("Rotate log", rotate_time)
        fh = TimedRotatingFileHandler(log_file_used, when=rotate_time)
        fh.setFormatter(formatter)
        return fh
    else:
        fh = logging.FileHandler(log_file_used)
        fh.setFormatter(formatter)
        return fh


def logger_setup2(name=__name__, level=None):
    _ = level

    # logger defines
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger




###########################################################################################################
def test_log():
    logger =logger_class(verbose=True)

    def log(*s):
       logger.log(*s, level=1)

    def log2(*s):
       logger.log(*s, level=2)

    def log3(*s):
       logger.log(*s, level=3)

    log( "level 1"  )
    log2( "level 2"  )
    log3( "level 3"  )




###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
    


        
        
        
        

############################################################################################################
def download_googledrive(file_list=[ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4",  "path_target":  "data/input/download/test.json"}], **kw):
    """
      Use in dataloader with
         "uri": mlmodels.data:donwload_googledrive
         file_list = [ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4",  "path_target":  "ztest/covid19/test.json"},
                        {  "fileid" :  "GOOGLE URL ID"   , "path_target":  "dataset/test.json"},
                 ]
    """
    try :
      import gdown
    except:
        os.system('pip install gdown')
        import gdown
    import random
    target_list = []

    for d in file_list :
      fileid = d["fileid"]
      target = d.get("path_target", "data/input/adonwload/googlefile_" + str(random.randrange(1000) )  )

      os.makedirs(os.path.dirname(target), exist_ok=True)

      url = f'https://drive.google.com/uc?id={fileid}'
      gdown.download(url, target, quiet=False)
      target_list.append( target  )

    return target_list


def download_dtopbox(data_pars):
  """
  download_data({"from_path" :  "tabular",
                        "out_path" :  path_norm("ztest/dataset/text/") } )
  Open URL
     https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAoFh0aO9RqwwROksGgasIha?dl=0


  """
  # from cli_code.cli_download import Downloader

  folder = data_pars.get('from_path', None)  # dataset/text/

  urlmap = {
     "text" :    "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AADHrhC7rLkd42_CEqK6A9oYa/dataset/text?dl=1&subfolder_nav_tracking=1"
     ,"tabular" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAxZkJTGSumLADzj3B5wbA0a/dataset/tabular?dl=1&subfolder_nav_tracking=1"
     ,"pretrained" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AACL3LHW1USWrvsV5hipw27ia/model_pretrained?dl=1&subfolder_nav_tracking=1"

     ,"vision" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAM4k7rQrkjBo09YudYV-6Ca/dataset/vision?dl=1&subfolder_nav_tracking=1"
     ,"recommender": "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AABIb2JjQ6aQHwfq5CU0ypHOa/dataset/recommender?dl=1&subfolder_nav_tracking=1"

  }

  if data_pars.get('url', None):
      url = data_pars['url']
  elif folder:
      url = urlmap[folder]

  #prefix = "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/"
  #url= f"{prefix}/AADHrhC7rLkd42_CEqK6A9oYa/{folder}?dl=1&subfolder_nav_tracking=1"

  out_path = data_pars['out_path']

  if folder:
      zipname = folder.split("/")[0]


  os.makedirs(out_path, exist_ok=True)
  downloader = Downloader(url)
  downloader.download(out_path)

  if folder:
      import zipfile
      with zipfile.ZipFile( out_path + "/" + zipname + ".zip" ,"r") as zip_ref:
          zip_ref.extractall(out_path)



class Downloader:

    GITHUB_NETLOC = 'github.com'
    GITHUB_RAW_NETLOC = 'raw.githubusercontent.com'

    GDRIVE_NETLOC = 'drive.google.com'
    GDRIVE_LINK_TEMPLATE = 'https://drive.google.com/u/0/uc?id={fileid}&export=download'

    DROPBOX_NETLOC = 'dropbox.com'

    DEFAULT_FILENAME = uuid.uuid4().hex  # To provide unique filename in batch jobs

    def __init__(self, url):
        """Make path adjustments and parse url"""
        self.url = url
        self.parsed = requests.utils.urlparse(url)

        self.clean_netloc()

        if not self.parsed.netloc:
            raise ValueError('Wrong URL (Make sure "http(s)://" included)')

        self.adjust_url()

    def clean_netloc(self):
        clean_netloc = re.sub(r'^www\.', '', self.parsed.netloc)
        self.parsed = self.parsed._replace(netloc=clean_netloc)

    def adjust_url(self):
        if self.parsed.netloc == self.GITHUB_NETLOC:
            self._transform_github_url()
        elif self.parsed.netloc == self.GDRIVE_NETLOC:
            self._transform_gdrive_url()
        elif self.parsed.netloc == self.DROPBOX_NETLOC:
            self._transform_dropbox_url()

    def _transform_github_url(self):
        """Github specific changes to get link to raw file"""
        self.url = (
            self.url
            .replace('/blob/', '/')
            .replace(self.GITHUB_NETLOC, self.GITHUB_RAW_NETLOC)
        )

    def _transform_gdrive_url(self):
        """GDrive specific changes to get link to raw file"""
        fileid = self.parsed.path.replace('/file/d/', '').split('/')[0]
        self.url = self.GDRIVE_LINK_TEMPLATE.format(fileid=fileid)

    def _transform_dropbox_url(self):
        """DropBox specific changes to get link to raw file"""
        self.url = requests.utils.urlunparse(
            self.parsed._replace(query='dl=1'))

    def get_filename(self, headers):
        """Attempt to get filename from content-dispositions header.

        If not found: get filename from parsed path
        If both fail: use DEFAULT_FILENAME to save file
        """
        header = headers.get('content-disposition')

        if header is not None:
            _, params = cgi.parse_header(header)
            filename = params.get('filename')
        else:
            try:
                filename = self.parsed.path.split('/')[-1]
            except IndexError:
                filename = None

        return filename if filename is not None else self.DEFAULT_FILENAME

    def download(self, filepath=''):
        '''Downloading and saving file'''

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        response = requests.get(self.url)
        filename = self.get_filename(response.headers)

        full_filename = os.path.join(filepath, filename)

        if response.status_code == 200:
            with open(full_filename, "wb") as f:
                f.write(response.content)

            print(f'File saved as {full_filename}')
        else:
            print('Bad request')






####################################################################################
def load_dataset_generator(data_pars):
  def sent_generator(TRAIN_DATA_FILE, chunksize):
      import pandas as pd
      reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
      for df in reader:
          val3  = df.iloc[:, 3:4].values.tolist()
          val4  = df.iloc[:, 4:5].values.tolist()
          flat3 = [item for sublist in val3 for item in sublist]
          flat4 = [str(item) for sublist in val4 for item in sublist]
          texts = []
          texts.extend(flat3[:])
          texts.extend(flat4[:])

          sequences  = model.tokenizer.texts_to_sequences(texts)
          data_train = pad_sequences(sequences, maxlen=data_pars["MAX_SEQUENCE_LENGTH"])
          yield [data_train, data_train]

  return sent_generator(data_pars["train_data_path"])

  # model.model.fit(sent_generator(data_pars["train_data_path"], batch_size / 2),
  #                epochs          = epochs,
  #                steps_per_epoch = n_steps,
  #                validation_data = (data_pars["data_1_val"], data_pars["data_1_val"]))




def tf_dataset(dataset_pars):
    """
        dataset_pars ={ "dataset_id" : "mnist", "batch_size" : 5000, "n_train": 500, "n_test": 500,
                            "out_path" : "dataset/vision/mnist2/" }
        tf_dataset(dataset_pars)


        https://www.tensorflow.org/datasets/api_docs/python/tfds
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Here we assume Eager mode is enabled (TF2), but tfds also works in Graph mode.
        print(tfds.list_builders())

        # Construct a tf.data.Dataset
        ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)

        # Build your input pipeline
        ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
        for features in ds_train.take(1):
          image, label = features["image"], features["label"]


        NumPy Usage with tfds.as_numpy
        train_ds = tfds.load("mnist", split="train")
        train_ds = train_ds.shuffle(1024).batch(128).repeat(5).prefetch(10)

        for example in tfds.as_numpy(train_ds):
          numpy_images, numpy_labels = example["image"], example["label"]
        You can also use tfds.as_numpy in conjunction with batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object:

        train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
        numpy_ds = tfds.as_numpy(train_ds)
        numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]


        FeaturesDict({
    'identity_attack': tf.float32,
    'insult': tf.float32,
    'obscene': tf.float32,
    'severe_toxicity': tf.float32,
    'sexual_explicit': tf.float32,
    'text': Text(shape=(), dtype=tf.string),
    'threat': tf.float32,
    'toxicity': tf.float32,
})

    """
    try :
        import tensorflow_datasets as tfds
    except :
        os.system(('pip install tensorflow_datasets'))
        import tensorflow_datasets as tfds

    d          = dataset_pars
    dataset_id = d['dataset_id']
    batch_size = d.get('batch_size', -1)  # -1 neans all the dataset
    n_train    = d.get("n_train", 500)
    n_test     = d.get("n_test", 500)
    out_path   = d['out_path']
    name       = dataset_id.replace(".","-")
    os.makedirs(out_path, exist_ok=True)

    train_ds =  tfds.as_numpy( tfds.load(dataset_id, split= f"train[0:{n_train}]", batch_size=batch_size) )
    test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    # test_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )

    print("train", train_ds.shape )
    print("test",  test_ds.shape )


    def get_keys(x):
       if "image" in x.keys() : xkey = "image"
       if "text" in x.keys() : xkey = "text"
       return xkey


    for x in train_ds:
       #print(x)
       xkey =  get_keys(x)
       np.savez_compressed(out_path + f"{name}_train" , X = x[xkey] , y = x.get('label') )


    for x in test_ds:
       #print(x)
       np.savez_compressed(out_path + f"{name}_test", X = x[xkey] , y = x.get('label') )

    print(out_path, os.listdir( out_path ))


