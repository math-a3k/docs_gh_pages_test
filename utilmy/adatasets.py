# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, json, pandas as pd, numpy as np
from pathlib import Path

from utilmy import (os_makedirs,  )

####################################################################################################
from utilmy import log, log2




##############################################################################################
def test_all():
    """
    #### python test.py   test_adatasets
    """
    def test():
        log("Testing  ...")
        from utilmy.adatasets import test_data_classifier_fake, test_data_classifier_petfinder, test_data_classifier_covtype,\
            test_data_regression_fake,test_data_classifier_pmlb
        test_data_regression_fake(nrows=500, n_features=17)
        test_data_classifier_fake(nrows=10)
        test_data_classifier_petfinder(nrows=10)
        test_data_classifier_covtype(nrows=10)
        test_data_classifier_pmlb(name=2)
    
    def test_pd_utils():
        import pandas as pd
        from utilmy.adatasets import pd_train_test_split,pd_train_test_split2, fetch_dataset
        fetch_dataset("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz",path_target="./testdata/tmp/test")
        df = pd.read_csv("./testdata/tmp/test/crop.data.csv")
        pd_train_test_split(df,"block")
        pd_train_test_split2(df, "block")

    test()
    test_pd_utils()


def test0():
    log("Testing  ...")
    test_data_regression_fake(nrows=500, n_features=17)
    test_data_classifier_fake(nrows=10)
    test_data_classifier_petfinder(nrows=10)
    test_data_classifier_covtype(nrows=10)
    '''TODO:
    dataset_classifier_pmlb(name=2)
    '''


def test1():
    fetch_dataset("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz",path_target="./testdata/tmp/test")
    df = pd.read_csv("./testdata/tmp/test/crop.data.csv")
    ''' TODO : need to add axis on df.drop()
    KeyError: "['block'] not found in axis"
    pd_train_test_split(df,"block")
    '''
    pd_train_test_split2(df, "block")

    

####################################################################################################
def test_data_XXXXXfier_YYYY(nrows=500, nmode='split/pandas', **kw):
    """

    """
    if 'split' in mode or 'numpy' in mode :



        return  X_train, X_test, y_train, y_test, colX, coly



    colnum = []
    colcat = []
    coly = []
    df = pd.DataFrame()
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly, 'info': '' }
    return df, pars  





####################################################################################################
######## Lisr of datasets ##########################################################################
def test_data_classifier_pmlb(name='', return_X_y=False, train_split=True):
    """  pip install pmlb


    """
    from pmlb import fetch_data, classification_dataset_names
    from sklearn.model_selection import train_test_split
    from utilmy import find_fuzzy
    ds = find_fuzzy(name, classification_dataset_names)
    pars = {}

    X,y = fetch_data(ds, return_X_y=  True)

    if train_split:
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y) # split
       return X_train, X_test, y_train, y_test

    colnum = [ str(i) for i in list(range(X.shape[1])) ]
    df = pd.DataFrame(X,columns=colnum)
    df['coly'] = y
    pars = {"colnum":colnum,"coly":y}
    return df, pars


def test_data_classifier_covtype(nrows=500):
    log("start")

    import wget
    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
              "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
              "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  ]

    # Target column
    coly   = ["Covertype"]

    datafile = os.getcwd() + "/ztmp/covtype/covtype.data.gz"
    os_makedirs(os.path.dirname(datafile))
    url      = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    if not Path(datafile).exists():
        wget.download(url, datafile)

    # Read nrows of only the given columns
    feature_columns = colnum + colcat + coly
    df   = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }

    return df, pars


def test_data_regression_fake(nrows=500, n_features=17):
    from sklearn import datasets as sklearn_datasets
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, 17) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_regression( n_samples=nrows, n_features=n_features, n_targets=1,
                                                n_informative=n_features-1)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))

    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }
    return df, pars


def test_data_classifier_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim    =11
    coly    = 'y'
    colnum  = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat  = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(n_samples=nrows, n_features=ndim, n_classes=1,
                                                   n_informative=ndim-2)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))

    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }
    return df, pars


def test_data_classifier_petfinder(nrows=1000):
    # Dense features
    import wget
    colnum = ['PhotoAmt', 'Fee','Age' ]

    # Sparse features
    colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized',
              'Health', 'Breed1' ]

    colembed = ['Breed1']
    coly        = "y"

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    localfile   = os.path.abspath('ztmp/petfinder-mini/')
    filepath    = localfile + "/petfinder-mini/petfinder-mini.csv"

    if not os.path.exists(filepath):
        os.makedirs(localfile, exist_ok=True)
        wget.download(dataset_url, localfile + "/petfinder-mini.zip")
        import zipfile
        with zipfile.ZipFile(localfile + "/petfinder-mini.zip", 'r') as zip_ref:
            zip_ref.extractall(localfile + "/")

    log('Data Frame Loaded')
    df       = pd.read_csv(filepath)
    df       = df.iloc[:nrows, :]
    df[coly] = np.where(df['AdoptionSpeed']==4, 0, 1)
    df       = df.drop(columns=['AdoptionSpeed', 'Description'])

    import shutil
    shutil.rmtree(localfile, ignore_errors=True)


    log2(df.dtypes)
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly, 'colembed' : colembed }
    return df, pars


def test_data_regression_boston(nrows=1000):
    '''load (regression) data on boston housing prices
    '''
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    X, y = load_boston(return_X_y=True)
    X,y = X[:nrows,:], y[:nrows]
    feature_names = load_boston()['feature_names']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25) # split
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names


def test_data_classifier_digits(nrows=1000):
    '''load (classification) data
    '''
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X, y     = digits.data, digits.target
    X,y = X[:nrows,:], y[:nrows]
    feature_names = load_digits()['feature_names']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y) # split
    return X_train, X_test, y_train, y_test, feature_names





####################################################################################################
####  Utilss #######################################################################################
def pd_train_test_split(df, coly=None):
    from sklearn.model_selection import train_test_split
    X,y = df.drop(coly,axis=1), df[[coly]]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)
    return X_train, X_valid, y_train, y_valid, X_test, y_test


def pd_train_test_split2(df, coly):
    from sklearn.model_selection import train_test_split
    log2(df.dtypes)
    X,y = df.drop(coly,  axis=1), df[coly]
    log2('y', np.sum(y[y==1]) , X.head(3))
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)
    num_classes                                = len(set(y_train_full.values.ravel()))
    return X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes



def fetch_dataset(url_dataset, path_target=None, file_target=None):
    """Fetch dataset from a given URL and save it.

    Currently `github`, `gdrive` and `dropbox` are the only supported sources of
    data. Also only zip files are supported.

    :param url_dataset:   URL to send
    :param path_target:   Path to save dataset
    :param file_target:   File to save dataset

    """
    log("###### Download ##################################################")
    from tempfile import mktemp, mkdtemp
    from urllib.parse import urlparse, parse_qs
    import pathlib
    fallback_name        = "features"
    download_path        = path_target
    supported_extensions = [ ".zip" ]

    if path_target is None:
        path_target   = mkdtemp(dir=os.path.curdir)
        download_path = path_target
    else:
        pathlib.Path(path_target).mkdir(parents=True, exist_ok=True)

    if file_target is None:
        file_target = fallback_name # mktemp(dir="")


    if "github.com" in url_dataset:
        """
                # https://github.com/arita37/dsa2_data/raw/main/input/titanic/train/features.zip
 
              https://github.com/arita37/dsa2_data/raw/main/input/titanic/train/features.zip            
              https://raw.githubusercontent.com/arita37/dsa2_data/main/input/titanic/train/features.csv            
              https://raw.githubusercontent.com/arita37/dsa2_data/tree/main/input/titanic/train/features.zip             
              https://github.com/arita37/dsa2_data/blob/main/input/titanic/train/features.zip
                 
        """
        # urlx = url_dataset.replace(  "github.com", "raw.githubusercontent.com" )
        urlx = url_dataset.replace("/blob/", "/raw/")
        urlx = urlx.replace("/tree/", "/raw/")
        log(urlx)

        urlpath = urlx.replace("https://github.com/", "github_")
        urlpath = urlpath.split("/")
        fname = urlpath[-1]  ## filaneme
        fpath = "-".join(urlpath[:-1])[:-1]   ### prefix path normalized
        assert "." in fname, f"No filename in the url {urlx}"

        os.makedirs(download_path + "/" + fpath, exist_ok= True)
        full_filename = os.path.abspath( download_path + "/" + fpath + "/" + fname )
        log('#### Download saving in ', full_filename)

        import requests
        with requests.Session() as s:
            res = s.get(urlx)
            if res.ok:
                print(res.ok)
                with open(full_filename, "wb") as f:
                    f.write(res.content)
            else:
                raise res.raise_for_status()
        return full_filename



    elif "drive.google.com" in url_dataset:
        full_filename = os.path.join(path_target, file_target)
        from util import download_googledrive
        urlx    = urlparse(url_dataset)
        file_id = parse_qs(urlx.query)['id'][0]
        download_googledrive([{'fileid': file_id, "path_target":
                               full_filename}])




    elif "dropbox.com" in url_dataset:
        full_filename = os.path.join(path_target, file_target)
        from util import download_dtopbox
        dbox_path_target = mkdtemp(dir=path_target)
        download_dtopbox({'url':      url_dataset,
                          'out_path': os.path.join(dbox_path_target)})
        dbox_file_target = os.listdir(dbox_path_target)[0]
        full_filename = os.path.join(dbox_path_target, dbox_file_target)


    else :
        if not os.path.exists(path_target):
            os.makedirs(localfile, exist_ok=True)
            wget.download(dataset_url, path_target )
            import zipfile
            with zipfile.ZipFile(localfile, 'r') as zip_ref:
                zip_ref.extractall(localfile + "/")      
            return localfile     

        


    path_data_x = full_filename

    #### Very Hacky : need to be removed.  ######################################
    for file_extension in supported_extensions:
        path_link_x = os.path.join(download_path, fallback_name + file_extension)
        if os.path.exists(path_link_x):
            os.unlink(path_link_x)
        os.link(path_data_x, path_link_x)

    #path_data_x = download_path + "/*"

    return path_data_x
    #return full_filename


def download_googledrive(file_list, **kw):
    """
      Use in dataloader with
         "uri": mlmodels.data:donwload_googledrive
         file_list = [ {  "fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4",  "path_target":  "ztest/covid19/test.json"},
                        {  "fileid" :  "GOOGLE URL ID"   , "path_target":  "dataset/test.json"},
                 ]

    """
    import random
    try :     import gdown
    except :  os.system('pip install gdown')
    finally : import gdown

    # file_list   = kw.get("file_list")
    target_list = []
    
    for d in file_list :
      fileid = d["fileid"]
      target = path_norm( d.get("path_target", "ztest/googlefile_" + str(random.randrange(1000) )  ) )
           
      if not os.path.exists(os.path.dirname(target)):
         os.makedirs(os.path.dirname(target), exist_ok=True)

      url = f'https://drive.google.com/uc?id={fileid}'
      gdown.download(url, target, quiet=False)
      target_list.append( target  )
                         
    return target_list

                         
def download_dtopbox(data_pars):
  """

   dataset/

   Prefix based :
      repo::
      dropbox::

   import_data 

   preprocess_data

  download_data({"from_path" :  "tabular",  
                        "out_path" :  path_norm("ztest/dataset/text/") } )

  Open URL
     https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAoFh0aO9RqwwROksGgasIha?dl=0


  """
  from cli_code.cli_download import Downloader

  folder = data_pars['from_path']  # dataset/text/

  urlmap = {
     "text" :    "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AADHrhC7rLkd42_CEqK6A9oYa/dataset/text?dl=1&subfolder_nav_tracking=1"
     ,"tabular" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAxZkJTGSumLADzj3B5wbA0a/dataset/tabular?dl=1&subfolder_nav_tracking=1"
     ,"pretrained" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AACL3LHW1USWrvsV5hipw27ia/model_pretrained?dl=1&subfolder_nav_tracking=1"

     ,"vision" : "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AAAM4k7rQrkjBo09YudYV-6Ca/dataset/vision?dl=1&subfolder_nav_tracking=1"
     ,"recommender": "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/AABIb2JjQ6aQHwfq5CU0ypHOa/dataset/recommender?dl=1&subfolder_nav_tracking=1"

  }

  url = urlmap[folder]

  #prefix = "https://www.dropbox.com/sh/d2n3hgsq2ycpmjf/"
  #url= f"{prefix}/AADHrhC7rLkd42_CEqK6A9oYa/{folder}?dl=1&subfolder_nav_tracking=1"

  out_path = data_pars['out_path']

  zipname = folder.split("/")[0]


  os.makedirs(out_path, exist_ok=True)
  downloader = Downloader(url)
  downloader.download(out_path)

  import zipfile
  with zipfile.ZipFile( out_path + "/" + zipname + ".zip" ,"r") as zip_ref:
      zip_ref.extractall(out_path)

class Downloader:
    import uuid 
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





################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




