# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python source/models/keras_widedeep.py  test


pip install Keras==2.4.3


"""
import os
import pandas as pd, numpy as np, scipy as sci
import sklearn
from sklearn.model_selection import train_test_split

import keras
layers = keras.layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin,  RegressorMixin, TransformerMixin


####################################################################################################
VERBOSE = True

# MODEL_URI = get_model_uri(__file__)

def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session


def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


def Modelcustom(n_wide_cross, n_wide, n_feat=8, m_EMBEDDING=10, loss='mse', metric = 'mean_squared_error'):

        #### Wide model with the functional API
        col_wide_cross          = layers.Input(shape=(n_wide_cross,))
        col_wide                = layers.Input(shape=(n_wide,))
        merged_layer            = layers.concatenate([col_wide_cross, col_wide])
        merged_layer            = layers.Dense(15, activation='relu')(merged_layer)
        predictions             = layers.Dense(1)(merged_layer)
        wide_model              = keras.Model(inputs=[col_wide_cross, col_wide], outputs=predictions)

        wide_model.compile(loss = 'mse', optimizer='adam', metrics=[ metric ])
        print(wide_model.summary())


        #### Deep model with the Functional API
        deep_inputs             = layers.Input(shape=(n_wide,))
        embedding               = layers.Embedding(n_feat, m_EMBEDDING, input_length= n_wide)(deep_inputs)
        embedding               = layers.Flatten()(embedding)

        merged_layer            = layers.Dense(15, activation='relu')(embedding)

        embed_out               = layers.Dense(1)(merged_layer)
        deep_model              = keras.Model(inputs=deep_inputs, outputs=embed_out)
        deep_model.compile(loss='mse',   optimizer='adam',  metrics=[ metric ])
        print(deep_model.summary())


        #### Combine wide and deep into one model
        merged_out = layers.concatenate([wide_model.output, deep_model.output])
        merged_out = layers.Dense(1)(merged_out)
        model      = keras.Model( wide_model.input + [deep_model.input], merged_out)
        model.compile(loss=loss,   optimizer='adam',  metrics=[ metric ])
        print(model.summary())

        return model


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
        else:
            model_class = model_pars['model_class']  # globals() removed
            self.model  = Modelcustom(**model_pars['model_pars'])
            if VERBOSE: log(model_class, self.model)
            self.model.summary()


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")

    cpars = compute_pars.get("compute_pars", {})
    assert 'epochs' in cpars, 'epoch'

    hist = model.model.fit( Xtrain_tuple, ytrain,  **cpars)
    model.history = hist


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    # data_pars['train'] = True
    Xval, yval = get_dataset(data_pars, task_type="eval")

    # n_wide_features = data_pars.get('n_wide_features', None)
    # n_deep_features = data_pars.get('n_deep_features', None)

    ypred = predict(Xval, data_pars, compute_pars, out_pars)

    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {'metric_name': 'mae'})

    scorer = {
        "rmse": sklearn.metrics.mean_squared_error,
        "mae": sklearn.metrics.mean_absolute_error
    }[mpars['metric_name']]

    mpars2    = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred[0], **mpars2)

    ddict = [{"metric_val": score_val, 'metric_name': mpars['metric_name']}]
    return ddict


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session
    """

    """
    if Xpred is None:
        # data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    ypred = model.model.predict(Xpred )

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba


def reset():
    global model, session
    model, session = None, None


def save(path=None):
    global model, session
    os.makedirs(path, exist_ok=True)

    filename = "model.h5"
    filepath = path + filename
    model.model.save(filepath)


def load_model(path=""):
    global model, session

    filepath = path + 'model.h5'
    model = keras.models.load_model(filepath)
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df  = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values,
                                                        stratify=dfy.values,test_size=0.1)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')

    if data_type == "ram":
        cols_input_groupname = ['cols_wide_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_input_group     = data_pars['cols_model_type2']

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]

            Xtuple_train = []
            for cols_groupname in cols_input_groupname :
                cols_i = cols_input_group[cols_groupname]
                Xtuple_train.append( Xtrain[cols_i] )

            return Xtuple_train


        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["Xtrain"], d["ytrain"]

            Xtuple_train = []
            for cols_groupname in cols_input_groupname :
                cols_i = cols_input_group[cols_groupname]
                Xtuple_train.append( Xtrain[cols_i] )

            return Xtuple_train, ytrain


        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = []
            for cols_groupname in cols_input_groupname :
                cols_i = cols_input_group[cols_groupname]
                Xtuple_train.append( Xtrain[cols_i] )

            Xtuple_test = []
            for cols_groupname in cols_input_groupname :
                cols_i = cols_input_group[cols_groupname]
                Xtuple_test.append( Xtest[cols_i] )

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



def test(config=''):
    """
        Group of columns for the input model
           cols_input_group = [

          ]
          for cols in cols_input_group,


    :param config:
    :return:
    """
    global model, session

    X = pd.DataFrame( np.random.rand(100,30), columns= [ 'col_' +str(i) for i in range(20)] )
    y = pd.DataFrame( np.random.binomial(n=1, p=0.5, size=[100]), columns = ['coly'] )


    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model_ckpt     = ModelCheckpoint(filepath='', save_best_only=True, monitor='loss')
    callbacks      = [early_stopping, model_ckpt]

    n_features      = X_train.shape[1]  # number of features
    n_wide_features = 20
    n_deep_features = n_features - n_wide_features

    ##### Generate column actual names from
    colg_input = {
      'cols_wide_input':   ['colnum', 'colcat_onehot' ],
      'cols_deep_input':   ['colnum', 'colcat_onehot' ],
    }


    cols_family = {
        'colnum' : [],
        'colcat' : []
    }

    cols_model_input_group = {}
    for colg, colist in colg_input.items() :
        cols_model_input_group[colg] = []
        for colg_i in colist :
          cols_model_input_group[colg].extend( cols_family[colg_i] )


    model_pars = {'model_class': 'WideAndDeep',
                  'model_pars': {'n_wide_cross': n_wide_features,
                                 'n_wide':       n_deep_features},
                 }

    data_pars = {'train': {'Xtrain': X_train,
                           'ytrain': y_train,
                           'Xtest': X_test,
                           'ytest': y_test},
                 'eval': {'X': X_valid,
                          'y': y_valid},
                 'predict': {'X': X_valid},


                 'cols_model_input_group' :cols_model_input_group,

                 'n_features': n_features,
                 'n_wide_features': n_wide_features,
                 'n_deep_features': n_deep_features,
                }

    compute_pars = { 'compute_pars' : { 'epochs': 2,
                    'callbacks': callbacks} }

    test_helper(model_pars, data_pars, compute_pars)





def get_dataset2(data_pars=None, task_type="train", **kw):
    """
      "ram"  :
      "file" :

    n_wide_features = data_pars.get('n_wide_features', None)
    n_deep_features = data_pars.get('n_deep_features', None)

    Xtrain_A, Xtrain_B, Xtrain_C = Xtrain[:, :n_wide_features], Xtrain[:, -n_deep_features:], Xtrain[:, -n_deep_features:]
    Xtest_A, Xtest_B, Xtest_C    = Xtest[:, :n_wide_features], Xtest[:, -n_deep_features:], Xtest[:, -n_deep_features:]


    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        n_wide_features = data_pars.get('n_wide_features', None)
        n_deep_features = data_pars.get('n_deep_features', None)

        if task_type == "predict":
            d = data_pars[task_type]

            Xtrain = d["X"]
            Xtrain_A, Xtrain_B, Xtrain_C = Xtrain[:, :n_wide_features], Xtrain[:, -n_deep_features:], Xtrain[:, -n_deep_features:]
            Xtuple_train = (Xtrain_A, Xtrain_B, Xtrain_C)
            return Xtuple_train
            # return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]

            Xtrain, ytrain  = d["X"], d["y"]
            Xtrain_A, Xtrain_B, Xtrain_C = Xtrain[:, :n_wide_features], Xtrain[:, -n_deep_features:], Xtrain[:, -n_deep_features:]
            Xtuple_train = (Xtrain_A, Xtrain_B, Xtrain_C)

            return Xtuple_train, ytrain
            #return d["X"], d["y"]


        if task_type == "train":
            d = data_pars[task_type]

            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            Xtrain_A, Xtrain_B, Xtrain_C = Xtrain[:, :n_wide_features], Xtrain[:, -n_deep_features:], Xtrain[:, -n_deep_features:]
            Xtest_A, Xtest_B, Xtest_C    = Xtest[:, :n_wide_features], Xtest[:, -n_deep_features:], Xtest[:, -n_deep_features:]

            Xtuple_train = (Xtrain_A, Xtrain_B, Xtrain_C)
            Xtuple_test  = (Xtest_A, Xtest_B, Xtrain_C)

            return Xtuple_train, ytrain, Xtuple_test, ytest
            # return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')

def get_params_sklearn(deep=False):
    return model.model.get_params(deep=deep)


def get_params(param_pars={}, **kw):
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")


def test_helper(model_pars, data_pars, compute_pars):
    global model, session

    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)
    log('Training completed!\n\n')

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')
    log('Data successfully predicted!\n\n')

    log('Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))
    log('Evaluating completed!\n\n')
    #
    log('Saving model..')
    save(path= root + '/model_dir/')
    log('Model successfully saved!\n\n')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.summary())


#######################################################################################

def test2(config=''):

    global model, session

    X = np.random.rand(100,30)
    y = np.random.binomial(n=1, p=0.5, size=[100])

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model_ckpt     = ModelCheckpoint(filepath='', save_best_only=True, monitor='loss')
    callbacks      = [early_stopping, model_ckpt]

    n_features      = X_train.shape[1]  # number of features
    n_wide_features = 20
    n_deep_features = n_features - n_wide_features

    model_pars = {'model_class': 'WideAndDeep',
                  'model_pars': {'n_wide_cross': n_wide_features,
                                 'n_wide':       n_deep_features},
                 }
    data_pars = {'train': {'Xtrain': X_train,
                           'ytrain': y_train,
                           'Xtest': X_test,
                           'ytest': y_test},
                 'eval': {'X': X_valid,
                          'y': y_valid},
                 'predict': {'X': X_valid},
                 'n_features': n_features,
                 'n_wide_features': n_wide_features,
                 'n_deep_features': n_deep_features,
                }
    compute_pars = { 'compute_pars' : { 'epochs': 2,
                    'callbacks': callbacks} }

    test_helper(model_pars, data_pars, compute_pars)





if __name__ == "__main__":
    import fire
    fire.Fire(test)
