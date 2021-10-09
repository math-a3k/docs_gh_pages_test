# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

"""
import logging, os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

####################################################################################################
verbosity = 3

def log(*s):
    print(*s, flush=True)



####################################################################################################
def test_dataset_classifier_covtype(nrows=500):
    import wget
    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
        "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  ]

    # Target column
    coly        = ["Covertype"]

    log("start")
    global model, session

    root     = os.path.join(os.getcwd() ,"ztmp")
    BASE_DIR = Path.home().joinpath( root, 'data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

    # Download the dataset in case it's missing
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    # Read nrows of only the given columns
    feature_columns = colnum + colcat + coly
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    #### Matching Big dict  ##################################################
    # X = df
    # y = df[coly].astype('uint8')
    return df, colnum, colcat, coly



def test_dataset_regress_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, 17) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_regression( n_samples=1000, n_features=17, n_targets=1, n_informative=17)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly




def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim    =11
    coly    = 'y'
    colnum  = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat  = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(n_samples=1000, n_features=ndim, n_targets=1, n_informative=ndim
    )
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly


def test_dataset_petfinder(nrows=1000):
    import tensorflow as tf
    # Dense features
    colnum = ['PhotoAmt', 'Fee','Age' ]

    # Sparse features
    colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized',
              'Health', 'Breed1' ]

    colembed = ['Breed1']
    # Target column
    coly        = "y"

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')

    print('Data Frame Loaded')
    df      = pd.read_csv(csv_file)
    df      = df.iloc[:nrows, :]
    df['y'] = np.where(df['AdoptionSpeed']==4, 0, 1)
    df      = df.drop(columns=['AdoptionSpeed', 'Description'])

    print(df.dtypes)
    return df, colnum, colcat, coly, colembed






###################################################################################################################

def tf_data_create_sparse(cols_type_received:dict= {'cols_sparse' : ['col1', 'col2'],
                                                     'cols_num'    : ['cola', 'colb']

                                                     },
                           cols_ref:list=  [ 'col_sparse', 'col_num'  ], Xtrain:pd.DataFrame=None,
                           **kw):
    """

       Create sparse data struccture in KERAS  To plug with MODEL:
       No data, just virtual data
    https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/master/09_cloudml/flights_model_tf2.ipynb

    :return:
    """
    import tensorflow
    from tensorflow.feature_column import (categorical_column_with_hash_bucket,
        numeric_column, embedding_column, bucketized_column, crossed_column, indicator_column)

    ### Unique values :
    col_unique = {}

    if Xtrain is not None :
        for coli in cols_type_received['col_sparse'] :
                col_unique[coli] = int( Xtrain[coli].nunique())

    dict_cat_sparse, dict_dense = {}, {}
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "

        if cols_groupname == "cols_sparse" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               m_bucket = min(500, col_unique.get(coli, 500) )
               dict_cat_sparse[coli] = categorical_column_with_hash_bucket(coli, hash_bucket_size= m_bucket)

        if cols_groupname == "cols_dense" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               dict_dense[coli] = numeric_column(coli)

        if cols_groupname == "cols_cross" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               m_bucketi = min(500, col_unique.get(coli, 500) )
               m_bucketj = min(500, col_unique.get(coli, 500) )
               dict_cat_sparse[coli[0]+"-"+coli[1]] = crossed_column(coli[0], coli[1], m_bucketi * m_bucketj)

        if cols_groupname == "cols_discretize" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               bucket_list = np.linspace(min, max, 100).tolist()
               dict_cat_sparse[coli +"_bin"] = bucketized_column(numeric_column(coli), bucket_list)


    #### one-hot encode the sparse columns
    dict_cat_sparse = { colname : indicator_column(col)  for colname, col in dict_cat_sparse.items()}

    ### Embed
    dict_cat_embed  = { 'em_{}'.format(colname) : embedding_column(col, 10) for colname, col in dict_cat_sparse.items()}


    #### TO Customisze
    #dict_dnn    = {**dict_cat_embed,  **dict_dense}
    # dict_linear = {**dict_cat_sparse, **dict_dense}

    return  dict_cat_sparse, dict_cat_embed, dict_dense,




def tf_data_pandas_to_dataset(training_df, colsX, coly):
    # tf.enable_eager_execution()
    # features = ['feature1', 'feature2', 'feature3']
    import tensorflow as tf
    print(training_df)
    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(training_df[colsX].values, tf.float32),
                tf.cast(training_df[coly].values, tf.int32)
            )
        )
    )

    for features_tensor, target_tensor in training_dataset:
        print(f'features:{features_tensor} target:{target_tensor}')
    return training_dataset



def tf_data_file_to_dataset(pattern, batch_size, mode=tf.estimator.ModeKeys.TRAIN, truncate=None):
    """  ACTUAL Data reading :
           Dataframe ---> TF Dataset  --> feed Keras model

    """
    import os, json, math, shutil
    import tensorflow as tf

    DATA_BUCKET = "gs://{}/flights/chapter8/output/".format(BUCKET)
    TRAIN_DATA_PATTERN = DATA_BUCKET + "train*"
    EVAL_DATA_PATTERN = DATA_BUCKET + "test*"

    CSV_COLUMNS  = ('ontime,dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' + \
                    ',carrier,dep_lat,dep_lon,arr_lat,arr_lon,origin,dest').split(',')
    LABEL_COLUMN = 'ontime'
    DEFAULTS     = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],\
                    ['na'],[0.0],[0.0],[0.0],[0.0],['na'],['na']]

    def load_dataset(pattern, batch_size=1):
      return tf.data.experimental.make_csv_dataset(pattern, batch_size, CSV_COLUMNS, DEFAULTS)

    def features_and_labels(features):
      label = features.pop('ontime') # this is what we will train for
      return features, label

    dataset = load_dataset(pattern, batch_size)
    dataset = dataset.map(features_and_labels)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(batch_size*10)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(1)
    if truncate is not None:
        dataset = dataset.take(truncate)
    return dataset






if __name__ == "__main__":
    import fire
    fire.Fire()



