# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
ipython source/models/keras_widedeep.py  test  --pdb
python keras_widedeep.py  test
pip install Keras==2.4.3
"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
from utilmy import global_verbosity, os_makedirs, pd_read_file
verbosity = global_verbosity(__file__,"/../../config.json", 3 )

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)


####################################################################################################
global model, session
def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None

def reset():
    global model, session
    model, session = None, None


########Custom Model ################################################################################
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow import feature_column
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers


####################################################################################################
THISMODEL_COLGROUPS = ['cols_cross_input', 'cols_deep_input']  # ['colcontinuous', 'colsparse']

def WideDeep_sparse(model_pars2):
    """ Using TF feature column sparse tensor
    """
    loss             = model_pars2.get('loss', 'binary_crossentropy')
    optimizer        = model_pars2.get('optimizer', 'adam')
    metrics          = model_pars2.get('metrics', ['accuracy'])
    dnn_hidden_units = model_pars2.get('hidden_units', '64,32,16')

    if model_pars2.get('create_tensor', True) :
        #### To plug into Model   #####################################################################
        prepare = tf_FeatureColumns()

        # Numeric Columns creation
        # colnum = ['PhotoAmt', 'Fee', 'Age']

        colnum  = model_pars2['data_pars']['colnum']
        prepare.numeric_columns(colnum)

        #### Categorical Columns
        colcat = model_pars2['data_pars']['colcat']
        # colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized', 'Health','Breed1']
        colcat_unique = model_pars2['data_pars']['colcat_unique']
        prepare.categorical_columns(colcat, colcat_unique)

        ##### Bucketized Columns
        #bucket_cols = {'Age': [1,2,3,4,5]}
        #prepare.bucketized_columns(bucket_cols)

        ##### Embedding Columns
        colembed_dict = model_pars2['data_pars']['colembed_dict']
        prepare.embeddings_columns(colembed_dict)
        #embeddingCol = {'Breed1':8}

        ##### Export
        linear_feature_columns, dnn_feature_columns, inputs = prepare.get_features()

    #else :
    #    inputs                  = model_pars2['inputs']
    #    linear_feature_columns  = model_pars2['linear_cols']
    #    dnn_feature_columns     = model_pars2['dnn_cols']

    deep   = tf.keras.layers.DenseFeatures(dnn_feature_columns.values(), name='deep_inputs')(inputs)
    layers = [int(x) for x in dnn_hidden_units.split(',')]

    for layerno, numnodes in enumerate(layers):
        deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno+1))(deep)

    wide   = tf.keras.layers.DenseFeatures(linear_feature_columns.values(), name='wide_inputs')(inputs)
    both   = tf.keras.layers.concatenate([deep, wide], name='both')
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
    model  = tf.keras.Model(inputs, output)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    log2(model.summary())
    return model


class Model(object):
    global model,session
    def __init__(self, model_pars=None,  data_pars=None, compute_pars=None,):
        self.model_pars, self.data_pars, self.compute_pars= model_pars, data_pars, compute_pars
        self.history = None
        if model_pars is None:
            self.model = None

        else:
            model_class = model_pars.get('model_class', 'WideDeep_sparse')
            if 'sparse' in model_class  :
                cpars = model_pars['model_pars']
                cpars.update(data_pars)
                #cpars = { **cpars, **data_pars['data_pars'] }
                #pprint.pprint(cpars)
                self.model = WideDeep_sparse(cpars)
            #else :
            #    cpars = model_pars['model_pars']
            #    self.model = zz_WideDeep_dense(cpars)


#####################################################################################################
def fit(data_pars=None, compute_pars=None, out_pars=None):
    """  Fit model
    """
    global model,session
    cpars            = copy.deepcopy( compute_pars.get("compute_pars", {}) )
    cpars['epochs']  = cpars.get('epochs', 1)
    cpars['verbose'] = cpars.get('verbose',1)

    batch_size      = compute_pars.get('batch_size',2)
    path_checkpoint = compute_pars.get('path_checkpoint','ztmp_checkpoint/model_.pth')
    early_stopping  = EarlyStopping(monitor='loss', patience=3)
    model_ckpt      = ModelCheckpoint(filepath = path_checkpoint,save_best_only=True, monitor='loss')
    cpars_ex = copy.deepcopy( compute_pars.get("compute_extra", {}) )


    ###  Xtrain can be a path string (not DF)   ###########################################
    Xtrain,Ytrain, Xtest,Ytest= get_dataset_split(data_pars, task_type="train")

    # Rename columns to initial ones
    rename_col_dict = {}

    for col in Xtrain.columns:
        if col.endswith('_bin') or col.endswith('_int'):
            new_col = ''.join(x for x in col.split('_')[:-1])
            rename_col_dict[col] = new_col

    Xtrain = Xtrain.rename(columns=rename_col_dict)
    Xtest = Xtest.rename(columns=rename_col_dict)

    ### Coupling Data X Model Input
    if 'sparse' in  model.model_pars['model_class'] :
        log2('Fitting Sparse input...')
        Xy_train = get_dataset_split_for_model_tfsparse(Xtrain, Ytrain, data_pars)
        Xy_val   = get_dataset_split_for_model_tfsparse(Xtest,  Ytest,  data_pars)

        hist   = model.model.fit(Xy_train, validation_data=Xy_val, **cpars)
        model.history = hist

    else :
        log2('Fitting Dense input...')

        Xtrain,Ytrain = get_dataset_split_for_model_pandastuple(Xtrain, Ytrain, data_pars)
        Xtrain = pd.concat(Xtrain, axis=1)
        Xtrain = tf.data.Dataset.from_tensor_slices(Xtrain)
        Ytrain = tf.data.Dataset.from_tensor_slices(Ytrain)
        Xy_train      = tf.data.Dataset.zip(((Xtrain, Xtrain, Xtrain), Ytrain)).batch(batch_size)

        Xtest,Ytest   = get_dataset_split_for_model_pandastuple(Xtest,  Ytest,  data_pars)
        Xtest = pd.concat(Xtest, axis=1)
        Xtest = tf.data.Dataset.from_tensor_slices(Xtest)
        Ytest = tf.data.Dataset.from_tensor_slices(Ytest)
        Xy_val        = tf.data.Dataset.zip(((Xtest, Xtest, Xtest), Ytest)).batch(batch_size)
        #log(next(Xtrain_tuple.make_initializable_iterator())[0].numpy)

        hist     = model.model.fit(Xy_train, validation_data=Xy_val, **cpars)
        model.history = hist



def predict(Xpred=None,data_pars=None, compute_pars=None, out_pars=None):
    global model, session
    if Xpred is None :
        Xpred = get_dataset_split(data_pars, task_type="predict")

    # Rename columns to initial ones
    rename_col_dict = {}

    for col in Xpred.columns:
        if col.endswith('_bin') or col.endswith('_int'):
            new_col = ''.join(x for x in col.split('_')[:-1])
            rename_col_dict[col] = new_col

    Xpred = Xpred.rename(columns=rename_col_dict)


    if 'sparse' in  model.model_pars['model_class'] :
        Xpred       = get_dataset_split_for_model_tfsparse(Xpred, None, data_pars)
        ypred_proba = model.model.predict(Xpred)
        ypred       = [  1 if t > 0.5 else 0 for t in ypred_proba ]

    else :
        Xpred,_  = get_dataset_split_for_model_pandastuple(Xpred, None, data_pars)
        Xpred = pd.concat(Xpred, axis=1)
        Xpred = tf.data.Dataset.from_tensor_slices(Xpred)
        testdata = tf.data.Dataset.zip(((Xpred, Xpred, Xpred),)).batch(32)
        ypred_proba = model.model.predict(testdata)
        ypred       = [  1 if t > 0.5 else 0 for t in ypred_proba ]

    ################################################################
    if compute_pars.get("probability", False):
        return ypred, ypred_proba
    else :
        return ypred, None


def save(path=None, info=None):
    global model, session
    import dill as pickle, copy
    os.makedirs(path, exist_ok=True)

    ### Keras save
    model.model.save(f"{path}/model_keras.h5")
    model.model.save_weights(f"{path}/model_keras_weights.h5")

    ### Wrapper saving
    modelx = Model()  # Empty model  Issue with pickle
    modelx.model_pars   = model.model_pars
    modelx.data_pars    = model.data_pars
    modelx.compute_pars = model.compute_pars
    # log('model', modelx.model)
    pickle.dump(modelx, open(f"{path}/model.pkl", mode='wb'))  #
    pickle.dump(info, open(f"{path}/info.pkl", mode='wb'))  #
    log('Model Saved', path)


def load_model(path=""):
    global model, session
    import dill as pickle
    session = None

    model0             = pickle.load(open(f"{path}/model.pkl", mode='rb'))
    model              = Model()  # Empty model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars

    #### Kerars Load
    model.model        = keras.models.load_model( f"{path}/model_keras.h5"  )
    #### Issue when loading model due to custom weights, losses, Keras erro
    #model_keras = get_model()
    #model.model.load_weights( f'{path}/model_keras_weights.h5')
    log(model.model.summary())
    return model, session


def model_summary(path="ztmp/"):
    global model
    # os.makedirs(path, exist_ok=True)
    # try :
    #  tf.keras.utils.plot_model(model.model, f'{path}/model.png', show_shapes=False, rankdir='LR')
    # tf.keras.utils.plot_model(model.model, f'{path}/model_shapes.png', show_shapes=True, rankdir='LR')
    #except Exception as e :
    #  log("error", e)


########################################################################################################################
def get_dataset_split(data_pars=None, task_type="train", **kw):
    """
       Grab the splitting of data_pars DICT
    """
    if task_type == "predict":
        d            = data_pars[task_type]
        Xtrain       = d["X"]
        return Xtrain

    if task_type == "eval":
        d               = data_pars[task_type]
        Xtrain, ytrain  = d["X"], d["y"]
        return Xtrain, ytrain

    if task_type == "train":
        d                             = data_pars[task_type]
        Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d.get("ytrain", None), d["Xtest"], d.get("ytest", None)
        return Xtrain, ytrain, Xtest,ytest


def get_dataset_split_for_model_pandastuple(Xtrain, ytrain=None, data_pars=None, ):
    """  Split data for moel input/
    Xtrain  ---> Split INTO  tuple of data  Xtuple= (df1, df2, df3) to fit model input.
    :param Xtrain:
    :param coldataloader_received:
    :param colmodel_ref:
    :return:
    """
    coldataloader_received  = data_pars.get('cols_model_type2', {})
    colmodel_ref             = THISMODEL_COLGROUPS

    ### Into RAM
    if isinstance(Xtrain, str) : Xtrain = pd_read_file(Xtrain, verbose=False)
    if isinstance(ytrain, str) : ytrain = pd_read_file(ytrain, verbose=False)


    if len(colmodel_ref) <= 1 :   ## No split
        return Xtrain

    ### Split the pandas columns into different pieces  ######################
    Xtuple_train = []
    for cols_groupname in colmodel_ref :
        assert cols_groupname in coldataloader_received, "Error missing colgroup in  data_pars[cols_model_type] "
        cols_name_list = coldataloader_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_name_list] )

    return Xtuple_train, ytrain


def get_dataset_split_for_model_petastorm(Xtrain, ytrain=None, pars:dict=None):
    """  Split data for moel input/
    Xtrain  ---> Split INTO  tuple PetaStorm Reader
    https://github.com/uber/petastorm/blob/master/petastorm/reader.py#L61-L134
    :param Xtrain:  path
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    from petastorm.reader import Reader, make_batch_reader
    from petastorm.tf_utils import make_petastorm_dataset

    dataset_url_train = Xtrain
    batch_size = 128
    num_classes = 10
    epochs = 12

    train_reader  = make_batch_reader( dataset_url_train, num_epochs=epochs)
    train_dataset = make_petastorm_dataset(train_reader)

    ### Re-shape  #############################################
    train_dataset = train_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))



    ###########################################################
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    return train_dataset




class tf_FeatureColumns:
    """
       Coupling between Abstract definition of data vs Actual Data values
    """
    def __init__(self,dataframe=None):
        # self.df = dataframe
        self.real_columns = {}
        self.sparse_columns = {}
        self.feature_layer_inputs = {}

    def df_to_dataset(self,dataframe,target,shuffle=True, batch_size=32):
        dataframe = dataframe.copy()

        if target in dataframe.columns:
            labels    = dataframe.pop(target)
            ds        = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        else:
            ds        = tf.data.Dataset.from_tensor_slices((dict(dataframe),))

        if shuffle: ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    def df_to_dataset_dense(self,dataframe,target,shuffle=True,batch_size=32):
        from sklearn.preprocessing import LabelEncoder
        dataframe = dataframe.copy()
        labels = None
        try:
            labels = dataframe.pop(target)
            labels = labels.values.reshape(-1,1)
            labels = tf.data.Dataset.from_tensor_slices(labels)
        except:
            pass
        dataframe = dataframe.apply(LabelEncoder().fit_transform)
        dataframe = dataframe.values
        shape = dataframe.shape[1]
        data = tf.data.Dataset.from_tensor_slices(dataframe)
        if labels:
            return data,labels,shape
        return data,None,shape


    def split_sparse_data(self,df,shuffle_train=False,shuffle_test=False,shuffle_val=False,batch_size=32,test_split=0.2, colnum=[], colcat=[]):
        for c in colcat:
            df[c] =  df[c].astype(str)

        if test_split is not None :
            train_df,test_df = train_test_split(df,test_size=test_split)
            train_df,val_df  = train_test_split(train_df,test_size=0.1)
            log('Files Splitted')
            self.train,self.val,self.test = train_df,val_df,test_df
            return self.train, self.val, self.test

    def data_to_tensorflow_split(self,df, target,model='sparse',shuffle_train=False,shuffle_test=False,shuffle_val=False,batch_size=32,
                           test_split=0.2, colnum=[], colcat=[]):

        for c in colcat:
            df[c] =  df[c].astype(str)

        if test_split is not None :
            train_df,test_df = train_test_split(df,test_size=test_split)
            train_df,val_df  = train_test_split(train_df,test_size=0.1)
            log('Files Splitted')
            self.train,self.val,self.test = train_df,val_df,test_df

        if model=='sparse':
            self.train_df_tf = self.df_to_dataset(self.train,target=target,shuffle=shuffle_train, batch_size=batch_size)
            self.test_df_tf  = self.df_to_dataset(self.test,target=target, shuffle=shuffle_test,  batch_size=batch_size)
            self.val_df_tf   = self.df_to_dataset(self.val,target=target,  shuffle=shuffle_val,   batch_size=batch_size)

            self.exampleData = self.test_df_tf #for visualization purposes
            log('Datasets Converted to Tensorflow')
            return self.train_df_tf,self.test_df_tf,self.val_df_tf
        else:
            self.train_data,self.train_label,shape = self.df_to_dataset_dense(self.train,target=target,shuffle=shuffle_train,batch_size=batch_size)
            self.test_data,self.test_label,_= self.df_to_dataset_dense(self.test,target=target,shuffle=shuffle_train,batch_size=batch_size)
            self.val_data,self.val_label,_ = self.df_to_dataset_dense(self.val,target=target,shuffle=shuffle_train,batch_size=batch_size)

            return self.train_data,self.train_label,self.test_data,self.test_label,self.val_data,self.val_label,shape


    def data_to_tensorflow(self,df, target,model='sparse',shuffle_train=False,shuffle_test=False,shuffle_val=False,batch_size=32,
                           colnum=[], colcat:list=[]):

        for c in colcat:
            df[c] =  df[c].astype(str)  ### TF wants String for category

        self.train = df

        if model=='sparse':
            self.train_df_tf = self.df_to_dataset(self.train,target=target,shuffle=shuffle_train, batch_size=batch_size)
            log('Datasets Converted to Tensorflow')
            return self.train_df_tf, None, None

        else:
            self.train_data,self.train_label,shape = self.df_to_dataset_dense(self.train,target=target,
                                                                              shuffle=shuffle_train,batch_size=batch_size)
            return self.train_data, self.train_label, shape


    ################## To PLUG INTO  model
    def numeric_columns(self,columnsName):
        for header in columnsName:
            numeric = feature_column.numeric_column(header)
            self.real_columns[header] = numeric
            self.feature_layer_inputs[header] = tf.keras.Input(shape=(1,), name=header)
        return numeric


    def bucketized_columns(self,columnsBoundaries):
        for key,value in columnsBoundaries.items():
            col = feature_column.numeric_column(key)
            col_buckets = feature_column.bucketized_column(col,boundaries=value)
            self.sparse_columns[key] = col_buckets
        return col_buckets


    def categorical_columns(self,indicator_column_names, colcat_nunique=None, output=False):
        for col_name in indicator_column_names:

            ###Dependance on actual Data
            #nuniques =  list(self.df[col_name].unique())

            # Must be string to be categorical
            nuniques = [str(x) for x in colcat_nunique[col_name]]

            categorical_column = feature_column.categorical_column_with_vocabulary_list(col_name, nuniques )
            indicator_column   = feature_column.indicator_column(categorical_column)
            self.sparse_columns[col_name] = indicator_column

            #### Specific model keras input
            self.feature_layer_inputs[col_name] = tf.keras.Input(shape=(1,), name=col_name,dtype=tf.string)
        return indicator_column


    def hashed_columns(self,hashed_columns_dict):
        ### Independance
        for col_name,bucket_size in hashed_columns_dict.items():
            hashedCol     = feature_column.categorical_column_with_hash_bucket(col_name, hash_bucket_size=bucket_size)
            hashedFeature = feature_column.indicator_column(hashedCol)
            self.sparse_columns[col_name] = hashedFeature
        return hashedFeature


    def crossed_feature_columns(self,columns_crossed,nameOfLayer,bucket_size=10):
        crossed_feature = feature_column.crossed_column(columns_crossed, hash_bucket_size=bucket_size)
        crossed_feature = feature_column.indicator_column(crossed_feature)
        self.sparse_columns[nameOfLayer] = crossed_feature
        return crossed_feature


    def embeddings_columns(self,coldim_dict):
        for col_name,dimension in coldim_dict.items():
            #embCol    = feature_column.categorical_column_with_vocabulary_list(col_name, colunique )
            bucket_size = dimension*dimension
            embCol      = feature_column.categorical_column_with_hash_bucket(col_name, hash_bucket_size=bucket_size)
            embedding   = feature_column.embedding_column(embCol, dimension=dimension)
            self.real_columns[col_name] = embedding
        return embedding


    def transform_output(self,featureColumn):
        feature_layer = layers.DenseFeatures(featureColumn)
        example = next(iter(self.exampleData))[0]
        log(feature_layer(example).numpy())


    def get_features(self):
        return self.real_columns,self.sparse_columns,self.feature_layer_inputs



def get_dataset_split_for_model_tfsparse(Xtrain, ytrain=None, pars:dict=None):
    """  Split data for moel input/
    Xtrain  ---> Split INTO  tuple of data  Xtuple= (df1, df2, df3) to fit model input.
    :param Xtrain:
    :param coldataloader_received:
    :param colmodel_ref:
    :return:
    """
    log3("Xtrain type", Xtrain)

    if isinstance(Xtrain, str) :
        from utilmy import pd_read_file
        Xtrain  = pd_read_file(Xtrain, verbose=False)  ## in Memory
        log3('Xtrain laoded', Xtrain)

    if isinstance(Xtrain, tf.data.Dataset) :
       train_tfdata = Xtrain

    else :
        colnum   = pars['data_pars']['colnum']
        colcat   = pars['data_pars']['colcat']
        coly     = pars['data_pars']['coly']
        # colembed = pars['data_pars']['colembed_dict']
        assert isinstance(Xtrain, pd.DataFrame), 'Xtrain not dataframe'

        # Concatenate features and targets
        if ytrain is not None:
            Xtrain = pd.concat([Xtrain, ytrain], axis=1)

        prepare          = tf_FeatureColumns()
        train_tfdata,_,_ = prepare.data_to_tensorflow(Xtrain, model='sparse', target= coly,
                                                  colcat=colcat, colnum=colnum)
        log3(train_tfdata)

    return train_tfdata




########################################################################################################################
########################################################################################################################
def test(config='',     n_sample = 100):
    """ ACTUAL DICT
    """
    global model, session
    from adatasets import test_data_classifier_petfinder
    df, d = test_data_classifier_petfinder(nrows=100) ;     print(d)
    colnum, colcat, coly, colembed = d['colnum'], d['colcat'], d['coly'], d['colembed']


    #### Big dict  ##########################################################
    cols_input_type_1 = []
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)

    m = {
    'model_pars': {
         'model_class' :  "keras_widedeep.py::WideDeep_sparse"
         ,'model_pars' : {  }
        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################
            ### Pipeline for data processing ##############################
            'pipe_list': [  #### coly target prorcessing
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },

            ],
            }
    },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score'],
                      'compute_pars': { 'epochs' : 1}
    },

    'data_pars': { 'n_sample' : n_sample,
        'data_pars' :{
        },

        'download_pars'   : None,
        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  ##################
         'cols_model_group': [ 'colnum_bin',   'colcat_bin', ]

        ### Filter data rows   ###########################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' :     colcat,
        }


        ####### ACTUAL data Values #############################################################
        ,'train':   {}  #{'X_train': train_df,'Y_train':train_label, 'X_test':  val_df,'Y_test':val_label }
        ,'val':     {}  #{  'X':  val_df ,'Y':val_label }
        ,'predict': {}

        }
    }


    log("##### Sparse Tests  ############################################### ")
    prepare = tf_FeatureColumns()
    model_type = 'sparse'
    # train_df, test_df, val_df = prepare.data_to_tensorflow_split(df, model=model_type, target='y',
    #                             colcat=colcat, colnum=colnum)
    train_df, test_df, val_df = prepare.split_sparse_data(df, colcat=colcat, colnum=colnum)

    ### Unique values
    colcat_unique = {  col: list(df[col].unique())  for col in colcat }
    ### Embedding size Breed=8
    colembed_dict = {col: 2 + int(np.log(df[col].nunique()))  for col in colembed }

    ##### Dict update
    m['model_pars']['model_pars'] = { 'loss' : 'binary_crossentropy','optimizer':'adam',
                                      'metric': ['accuracy'],'hidden_units': '64,32,16'}

    m['data_pars']['train']     = {'Xtrain': train_df, 'Xtest':  val_df}
    m['data_pars']['predict']   = {'X':       test_df }
    m['data_pars']['data_pars'] = {
        'colcat_unique' : colcat_unique,
        'colcat'        : colcat,
        'colnum'        : colnum,
        'colembed_dict' : colembed_dict,
        'coly' : coly
    }
    test_helper( m['model_pars'], m['data_pars'], m['compute_pars'])


    log("##### Sparse Tests 2 ############################################### ")
    path  = "ztmp/parquets/"
    path2 = "ztmp/parquets/f01.parquet"
    os.makedirs(path, exist_ok=True)
    df.to_parquet( path + "/f01.parquet")
    colcat_unique = {  col: list(df[col].unique())  for col in colcat }
    colembed_dict = {col: 2 + int(np.log(df[col].nunique()))  for col in colembed }

    ##### Dict update
    m['model_pars']['model_pars'] = { 'loss' : 'binary_crossentropy','optimizer':'adam',
                                      'metric': ['accuracy'],'hidden_units': '64,32,16'}

    m['data_pars']['train']     = {'Xtrain':  train_df,
                                   'Xtest':   val_df}
    m['data_pars']['predict']   = {'X':        test_df }

    m['data_pars']['data_pars'] = {
        'colcat'        : colcat,
        'colnum'        : colnum,
        'coly'          : coly,
        'colcat_unique' : colcat_unique,
        'colembed_dict' : colembed_dict,
    }
    test_helper( m['model_pars'], m['data_pars'], m['compute_pars'])





    """
    log("##### Dense Tests  ##################################################")
    model_type = 'dense'
    prepare = tf_FeatureColumns()
    train_df, train_label, test_df, test_label, val_df, val_label, df_shape = prepare.data_to_tensorflow(df, model=model_type,
                                                                                                         target='y',
                                                                                                         colcat=colcat,
                                                                                                         colnum=colnum)
    print(train_df)
    m['model_pars']['model_pars'] = {  'loss' : 'binary_crossentropy','optimizer':'adam',
                           'metric': ['accuracy'],'hidden_units': '64,32,16',
                            'n_wide_cross': train_df.output_shapes[0],
                            'n_wide':       train_df.output_shapes[0],
                            'n_deep':       train_df.output_shapes[0]}
    m['data_pars']['train'] = {'X_train': train_df,'Y_train':train_label, 'X_test':  val_df,'Y_test':val_label }
    m['data_pars']['val']   = {  'X':  val_df ,'Y':val_label }
    test_helper( m['model_pars'], m['data_pars'], m['compute_pars'])
    """



def test2(config=''):
    """
    """
    from adatasets import test_data_classifier_petfinder
    df, d = test_data_classifier_petfinder(nrows=100) ;     print(d)
    colnum, colcat, coly, colembed = d['colnum'], d['colcat'], d['coly'], d['colembed']


    #### For pipeline data feed ###############################################################################
    prepare = tf_FeatureColumns()
    train_df,test_df,val_df = prepare.data_to_tensorflow(df, target='y')

    #### To plug into Model   ##################################################################################
    prepare = tf_FeatureColumns()
    #Numeric Columns creation
    colnum = ['PhotoAmt', 'Fee', 'Age']
    prepare.numeric_columns(colnum)

    #Categorical Columns
    colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized', 'Health','Breed1']
    colcat_unique = {colname:list(df[colname].unique()) for colname in colcat}
    prepare.categorical_columns(colcat,colcat_nunique=colcat_unique)

    #Bucketized Columns
    bucket_cols = {'Age': [1,2,3,4,5]}
    prepare.bucketized_columns(bucket_cols)

    #Embedding Columns
    embeddingCol = {'Breed1':8}
    tf_linear,tf_dnn, tf_inputs = prepare.get_features()


    ########## Dict ######################################################
    model_pars = {
        'model_class' : 'keras_widedeep.py::DeepWide_sparse',
        'model_pars'  : {  'loss' : 'binary_crossentropy','optimizer':'adam','metric': ['accuracy'],'hidden_units': '64,32,16'}
    }

    data_pars = { 'data_pars' : { 'inputs': tf_inputs,
                                  'linear_cols': tf_linear.values(),
                                  'dnn_cols': tf_dnn.values(),
                   },

                ##### Data Feed  #################################
                'train': train_df,
                'test':  test_df,
                'val':   val_df }

    compute_pars = {'epochs':2, 'verbose': 1,'path_checkpoint': 'checkpoint/model.pth','probability':True}

    ######## Run ##########################################################
    test_helper(model_pars, data_pars, compute_pars)



def test_helper(model_pars, data_pars, compute_pars):
    global model,session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars)

    log('Predict data..')
    ypred, ypred_proba = predict(data_pars=data_pars,compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')


    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Model architecture:')
    log(model.model.summary())

    log('Model Snapshot')
    # model_summary()


####################################################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()



































"""
def test_dataset_petfinder(nrows=1000):
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
"""






def zz_WideDeep_dense(model_pars2):
        #n_wide_cross, n_wide, n_deep, n_feat=8, m_EMBEDDING=10, loss='mse', metric ='mean_squared_error'):
        """
           Dense Model of DeepWide
        :param n_wide_cross:
        :param n_wide:
        :param n_deep:
        :param n_feat:
        :param m_EMBEDDING:
        :param loss:
        :param metric:
        :return:
        """

        m = model_pars2
        n_wide_cross = m.get('n_wide_cross', 37)
        n_wide = m.get('n_wide', 37)
        n_deep = m.get('n_deep', 37)
        n_feat = m.get('n_feat', 2)
        m_EMBEDDING = m.get('m_embedding', 2)
        loss      = m.get('loss', 'binary_crossentropy')
        optimizer = m.get('optimizer', 'adam')
        metrics   = m.get('metrics', ['accuracy'])
        dnn_hidden_units = m.get('hidden_units', '64,32,16')


        #### Wide model with the functional API
        col_wide_cross          = tf.keras.layers.Input(shape=(n_wide_cross,))
        col_wide                = tf.keras.layers.Input(shape=(n_wide,))
        merged_layer            = tf.keras.layers.concatenate([col_wide_cross, col_wide])
        merged_layer            = tf.keras.layers.Dense(15, activation='relu')(merged_layer)
        predictions             = tf.keras.layers.Dense(1)(merged_layer)
        wide_model              = tf.keras.Model(inputs=[col_wide_cross, col_wide], outputs=predictions)

        wide_model.compile(loss = loss, optimizer='adam', metrics= metrics)
        #log2(wide_model.summary())

        #### Deep model with the Functional API
        deep_inputs             = tf.keras.layers.Input(shape=(n_deep,))
        #embedding               = tf.keras.layers.Embedding(n_feat, m_EMBEDDING, input_length= n_deep)(deep_inputs)
        embedding               = tf.keras.layers.Flatten()(deep_inputs)

        merged_layer            = tf.keras.layers.Dense(15, activation='relu')(embedding)

        embed_out               = tf.keras.layers.Dense(1)(merged_layer)
        deep_model              = tf.keras.Model(inputs=deep_inputs, outputs=embed_out)
        deep_model.compile(loss=loss,   optimizer='adam',  metrics= metrics)
        # log2(deep_model.summary())


        #### Combine wide and deep into one model
        merged_out = tf.keras.layers.concatenate([wide_model.output, deep_model.output])
        merged_out = tf.keras.layers.Dense(1)(merged_out)
        model      = tf.keras.Model(wide_model.input+[deep_model.input], merged_out)
        model.compile(loss=loss,   optimizer='adam',  metrics= metrics)
        log2(model.summary())
        log("Deep Model")
        return model



def zz_get_dataset(data_pars=None, task_type="train", **kw):
    """
       Coupling  Columns from DataLoading  --->  Columns Input for Model
    """
    # log(data_pars)
    data_type           = data_pars.get('type', 'ram')
    colmodel_reference  = THISMODEL_COLGROUPS
    pars = copy.deepcopy(data_pars)

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname,             ### dict  colgroup ---> list of df
        ### Split into Sparse, Continuous
        coldataloader_received     = data_pars.get('cols_model_type2', {} )

        if task_type == "predict":
            d            = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_split_for_model_pandastuple(Xtrain, coldataloader_received, colmodel_reference)
            return Xtuple_train

        if task_type == "eval":
            d               = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_split_for_model_pandastuple(Xtrain, coldataloader_received, colmodel_reference)
            return Xtuple_train, ytrain

        if task_type == "train":
            d                           = data_pars[task_type]
            Xtrain,Ytrain, Xtest,Ytest  = d["X_train"],d["Y_train"],d["X_test"],d["Y_test"]
            Xtuple_train = get_dataset_split_for_model_pandastuple(Xtrain, coldataloader_received, colmodel_reference)
            Xtuple_test  = get_dataset_split_for_model_pandastuple(Xtest,  coldataloader_received, colmodel_reference)
            log3(type(Xtrain))

            return Xtuple_train,Ytrain, Xtuple_test,Ytest
            #return Xtrain,Ytrain, Xtest,Ytest


    if data_type == "tfsparse":
        ### Split into Sparse, Continuous TF Dataset
        coldataloader_received     = data_pars.get('cols_model_type2', {} )

        if task_type == "predict":
            d            = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_split_for_model_tfsparse(Xtrain, pars)
            return Xtuple_train

        if task_type == "eval":
            d               = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_split_for_model_tfsparse(Xtrain, pars)
            return Xtuple_train, ytrain

        if task_type == "train":
            d                           = data_pars[task_type]
            Xtrain,Ytrain, Xtest,Ytest  = d["X_train"],d["Y_train"],d["X_test"],d["Y_test"]
            Xtuple_train = get_dataset_split_for_model_tfsparse(Xtrain, pars)
            Xtuple_test  = get_dataset_split_for_model_tfsparse(Xtest, pars)
            return Xtuple_train,Ytrain, Xtuple_test,Ytest



    if data_type == "petastorm":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        ### Split into Sparse, Continuous
        coldataloader_received     = data_pars.get('cols_model_type2', {} )

        if task_type == "predict":
            d            = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_split_for_model_petastorm(Xtrain, pars)
            return Xtuple_train

        if task_type == "eval":
            d               = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_split_for_model_petastorm(Xtrain, pars)
            return Xtuple_train, ytrain

        if task_type == "train":
            d                           = data_pars[task_type]
            Xtrain,Ytrain, Xtest,Ytest  = d["X_train"],d["Y_train"],d["X_test"],d["Y_test"]
            Xtuple_train = get_dataset_split_for_model_petastorm(Xtrain, pars)
            Xtuple_test  = get_dataset_split_for_model_petastorm(Xtest,  pars)
            #log(type(Xtrain))
            ### dict  colgroup ---> list of df
            return Xtuple_train,Ytrain, Xtuple_test,Ytest
            #return Xtrain,Ytrain, Xtest,Ytest

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')





####################################################################################################################
####################################################################################################################




def zz_input_template_feed_keras_model(Xtrain, cols_type_received, cols_ref, **kw):
    """
       Create sparse data struccture in KERAS  To plug with MODEL:
       No data, just virtual data
    https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/master/09_cloudml/flights_model_tf2.ipynb
    :return:
    """
    from tensorflow.feature_column import (categorical_column_with_hash_bucket,
        numeric_column, embedding_column, bucketized_column, crossed_column, indicator_column)

    if len(cols_ref) <= 1 :
        return Xtrain

    dict_sparse, dict_dense = {}, {}
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "

        if cols_groupname == "cols_sparse" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               ### dependance on Actual Data
               m_bucket = min(500, int( Xtrain[coli].nunique()) )
               dict_sparse[coli] = categorical_column_with_hash_bucket(coli, hash_bucket_size= m_bucket)

        if cols_groupname == "cols_dense" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               dict_dense[coli] = numeric_column(coli)

        if cols_groupname == "cols_cross" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               m_bucketi = min(500, int( Xtrain[coli[0]].nunique()) )
               m_bucketj = min(500, int( Xtrain[coli[1]].nunique()) )
               dict_sparse[coli[0]+"-"+coli[1]] = crossed_column(coli[0], coli[1], m_bucketi * m_bucketj)

        if cols_groupname == "cols_discretize" :
           col_list = cols_type_received[cols_groupname]
           for coli in col_list :
               bucket_list = np.linspace(min, max, 100).tolist()
               dict_sparse[coli +"_bin"] = bucketized_column(numeric_column(coli), bucket_list)


    #### one-hot encode the sparse columns
    dict_sparse = { colname : indicator_column(col)  for colname, col in dict_sparse.items()}

    ### Embed
    dict_embed  = { 'em_{}'.format(colname) : embedding_column(col, 10) for colname, col in dict_sparse.items()}

    dict_dnn    = {**dict_embed,  **dict_dense}
    dict_linear = {**dict_sparse, **dict_dense}

    return (dict_linear, dict_dnn )










####################################################################################################################
####################################################################################################################

def zz_get_dataset2(data_pars=None, task_type="train", **kw):
    """
      return tuple of Tensoflow
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    cols_ref  = THISMODEL_COLGROUPS

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple_keras(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple_keras(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xytuple_train = get_dataset_tuple_keras(Xtrain, ytrain, cols_type_received, cols_ref)
            Xytuple_test  = get_dataset_tuple_keras(Xtest, ytest, cols_type_received, cols_ref)

            log2("Xtuple_train", Xytuple_train)

            return Xytuple_train, Xytuple_test


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')






########################### Using Sparse Tensor  #######################################################
def ModelCustom2():
    # Build a wide-and-deep model.
    def wide_and_deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
        deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
        layers = [int(x) for x in dnn_hidden_units.split(',')]
        for layerno, numnodes in enumerate(layers):
            deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno+1))(deep)
        wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)
        both = tf.keras.layers.concatenate([deep, wide], name='both')
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    sparse, real =  input_template_feed_keras(cols_type_received, cols_ref)

    DNN_HIDDEN_UNITS = 10
    model = wide_and_deep_classifier(
        inputs,
        linear_feature_columns = sparse.values(),
        dnn_feature_columns = real.values(),
        dnn_hidden_units = DNN_HIDDEN_UNITS)
    #tf.keras.utils.plot_model(model, 'flights_model.png', show_shapes=False, rankdir='LR')
    return model





def zz_get_dataset_tuple_keras(pattern, batch_size, mode=tf.estimator.ModeKeys.TRAIN, truncate=None):
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

    def pandas_to_dataset(training_df, coly):
        # tf.enable_eager_execution()
        # features = ['feature1', 'feature2', 'feature3']
        print(training_df)
        training_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (
                    tf.cast(training_df[features].values, tf.float32),
                    tf.cast(training_df[coly].values, tf.int32)
                )
            )
        )

        for features_tensor, target_tensor in training_dataset:
            print(f'features:{features_tensor} target:{target_tensor}')
        return training_dataset

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

def zz_Modelsparse2():
    """
    https://github.com/GoogleCloudPlatform/data-science-on-gcp/blob/master/09_cloudml/flights_model_tf2.ipynb
    :return:
    """
    import tensorflow as tf
    NBUCKETS = 10

    real = {
        colname : tf.feature_column.numeric_column(colname)
              for colname in
                ('dep_delay,taxiout,distance,avg_dep_delay,avg_arr_delay' +
                 ',dep_lat,dep_lon,arr_lat,arr_lon').split(',')
    }
    inputs = {
        colname : tf.keras.layers.Input(name=colname, shape=(), dtype='float32')
              for colname in real.keys()
    }

    sparse = {
          'carrier': tf.feature_column.categorical_column_with_vocabulary_list('carrier',
                      vocabulary_list='AS,VX,F9,UA,US,WN,HA,EV,MQ,DL,OO,B6,NK,AA'.split(',')),
          'origin' : tf.feature_column.categorical_column_with_hash_bucket('origin', hash_bucket_size=1000),
          'dest'   : tf.feature_column.categorical_column_with_hash_bucket('dest', hash_bucket_size=1000)
    }
    inputs.update({
        colname : tf.keras.layers.Input(name=colname, shape=(), dtype='string')
              for colname in sparse.keys()
    })



    latbuckets = np.linspace(20.0, 50.0, NBUCKETS).tolist()  # USA
    lonbuckets = np.linspace(-120.0, -70.0, NBUCKETS).tolist() # USA
    disc = {}
    disc.update({
           'd_{}'.format(key) : tf.feature_column.bucketized_column(real[key], latbuckets)
              for key in ['dep_lat', 'arr_lat']
    })
    disc.update({
           'd_{}'.format(key) : tf.feature_column.bucketized_column(real[key], lonbuckets)
              for key in ['dep_lon', 'arr_lon']
    })

    # cross columns that make sense in combination
    sparse['dep_loc'] = tf.feature_column.crossed_column([disc['d_dep_lat'], disc['d_dep_lon']], NBUCKETS*NBUCKETS)
    sparse['arr_loc'] = tf.feature_column.crossed_column([disc['d_arr_lat'], disc['d_arr_lon']], NBUCKETS*NBUCKETS)
    sparse['dep_arr'] = tf.feature_column.crossed_column([sparse['dep_loc'], sparse['arr_loc']], NBUCKETS ** 4)
    #sparse['ori_dest'] = tf.feature_column.crossed_column(['origin', 'dest'], hash_bucket_size=1000)

    # embed all the sparse columns
    embed = {
           'embed_{}'.format(colname) : tf.feature_column.embedding_column(col, 10)
              for colname, col in sparse.items()
    }
    real.update(embed)

    # one-hot encode the sparse columns
    sparse = {
        colname : tf.feature_column.indicator_column(col)
              for colname, col in sparse.items()
    }


    print(sparse.keys())
    print(real.keys())


    # Build a wide-and-deep model.
    def wide_and_deep_classifier(inputs, linear_feature_columns, dnn_feature_columns, dnn_hidden_units):
        deep = tf.keras.layers.DenseFeatures(dnn_feature_columns, name='deep_inputs')(inputs)
        layers = dnn_hidden_units
        for layerno, numnodes in enumerate(layers):
            deep = tf.keras.layers.Dense(numnodes, activation='relu', name='dnn_{}'.format(layerno+1))(deep)
        wide = tf.keras.layers.DenseFeatures(linear_feature_columns, name='wide_inputs')(inputs)
        both = tf.keras.layers.concatenate([deep, wide], name='both')
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='pred')(both)
        model = tf.keras.Model(inputs, output)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model


    DNN_HIDDEN_UNITS = 10
    model = wide_and_deep_classifier(
        inputs,
        linear_feature_columns = sparse.values(),
        dnn_feature_columns = real.values(),
        dnn_hidden_units = DNN_HIDDEN_UNITS)
    tf.keras.utils.plot_model(model, 'flights_model.png', show_shapes=False, rankdir='LR')