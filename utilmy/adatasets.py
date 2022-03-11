# -*- coding: utf-8 -*-
MNAME = "utilmy.adatasets"
HELP = """ utils for dataset donwloading


"""
import os, sys, time, datetime,inspect, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff



###############################################################################################
from utilmy import (os_makedirs, os_system, global_verbosity,  git_repo_root )
from utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    print( HELP + help_create(MNAME) )


##############################################################################################
def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    log(MNAME)
    test()
    test1()


def test():
    """function test
    Args:
    Returns:
        
    """
    test_dataset_regression_fake(nrows=500, n_features=17)
    test_dataset_classifier_fake(nrows=10)
    test_dataset_classifier_petfinder(nrows=10)
    test_dataset_classifier_covtype(nrows=10)
    test_dataset_classifier_pmlb(name=2)


def test1():
    """function test1
    Args:
    Returns:
        
    """
    fetch_dataset("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz",path_target="./testdata/tmp/test")
    df = pd.read_csv("./testdata/tmp/test/crop.data.csv")
    pd_train_test_split(df,  coly="block")
    pd_train_test_split2(df, coly="block")



####################################################################################################
def template_dataset_classifier_XXXXX(nrows=500, **kw):
    """

    """
    colnum = []
    colcat = []
    coly = []
    df = pd.DataFrame
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly, 'info': '' }
    return df, pars  


####################################################################################################
########## Classification ##########################################################################
def test_dataset_classifier_fake(nrows=500):
    """function test_dataset_classifier_fake
    Args:
        nrows:   
    Returns:
        
    """
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


def test_dataset_classifier_pmlb(name='', return_X_y=False):
    """function test_dataset_classifier_pmlb
    Args:
        name:   
        return_X_y:   
    Returns:
        
    """
    from pmlb import fetch_data, classification_dataset_names
    ds = classification_dataset_names[name]
    pars = {}

    X,y = fetch_data(ds, return_X_y=  True)
    colnum = list(range(X.shape[1]))
    df = pd.DataFrame(X,columns=colnum)
    df['coly'] = y
    pars = {"colnum":colnum,"coly":y}
    return df, pars


def test_dataset_classifier_covtype(nrows=500):
    """function test_dataset_classifier_covtype
    Args:
        nrows:   
    Returns:
        
    """
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


def test_dataset_classifier_petfinder(nrows=1000):
    """function test_dataset_classifier_petfinder
    Args:
        nrows:   
    Returns:
        
    """
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


def test_dataset_classifier_diabetes_traintest():
    '''load (classification) data on diabetes
    '''
    data = loadarff("content/imodels/imodels/tests/test_data/diabetes.arff")
    data_np = np.array(list(map(lambda x: np.array(list(x)), data[0])))
    X = data_np[:, :-1].astype('float32')
    y_text = data_np[:, -1].astype('str')
    y = (y_text == 'tested_positive').astype(int)  # labels 0-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75) # split
    feature_names = ["#Pregnant","Glucose concentration test","Blood pressure(mmHg)","Triceps skin fold thickness(mm)",
                "2-Hour serum insulin (mu U/ml)","Body mass index","Diabetes pedigree function","Age (years)"]
    return X_train, X_test, y_train, y_test, feature_names



#####################################################################################################
######  Regression ##################################################################################
def test_dataset_regression_fake(nrows=500, n_features=17):
    """function test_dataset_regression_fake
    Args:
        nrows:   
        n_features:   
    Returns:
        
    """
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


def test_dataset_regression_boston_traintest():
    '''load (regression) data on boston housing prices
    '''
    X_reg, y_reg = load_boston(return_X_y=True)
    feature_names = load_boston()['feature_names']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.75) # split
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names




###################################################################################################
if 'utils':
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



################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




