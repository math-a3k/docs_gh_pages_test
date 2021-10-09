# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Template explanation

"""

####################################################################################################
#### Generic Wrappers   ############################################################################
"""
source/models/mymodel.py
  Meta Class Model wraps sub-models
      Global variable : model, session  stores Model
    #### 1. Global variables    : global variables **model** and **session**.
    #### 2.  init method        : init method to initialize global variables `model and session`
    #### 3. class Model         :   storing model details and parameters.
    #### 4. preprocess method   :   preprocessing dataset.
    #### 5. fit method          :   fitting the defined model and inputted data.
    #### 6. predict method      :   predicting using the fitted model.
    #### 7. save method         :   saving the model in the pickle file.
    #### 8. load_model method   :   loading, the model saved in a pickle file.
    #### 9. load_info method    :   loading the in mation stored in the pickle file.
    #### 10. get_dataset method :   retrieving the dataset.
    #### 11. get_params method  :   retrieving parameters.


Big dictionnary :    
  model_pars :   Aribitrary dict for model params
  compute_pars:  Arbitrary dict for  compute (ie epochs=1)
  data_pars:     Arbitrary dict for data definition
  out_pars    :  Arbitrary dict for ouput path

"""
global model, session
# model = Model()
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        model.model =  MY_MODEL_CLASS(model_pars['model_pars'])


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    global model, session
    ...
    return None


def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    ...

    return ypred, ypred_proba


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    ...


def load_model(path=""):
    global model, session
    import dill as pickle
    ...
    return model


#### Your custom model ##################################################################
class MY_MODEL_CLASS(object):
    def __init__(cpars):
        """
        YOUR   custom MODEL definition

        """
        n_wide = cpars['n_wide']
        ###### Your model Definition







####################################################################################################
####################################################################################################
#### Global Dictionary definition
Dict can NOT contain Object ( Dict ==  JSON file)
Dict can contain onyl string, float, .. list of string, dict of float...

model_pars : {
    'model_class'      : Name of your Class

    'model_pars'       : Dict to pass DIRECTLY to YOUR MODEL   MY_MODEL_CLASS(**model_pars['model_pars'])
    'model_extra'      : Dict of extra params for YOUR MODEL   , but we use if .. then .. code to map
  
    'post_process_fun' : post_process_function to run  After prediction
    'pre_process_pars' :
         'y_norm_fun' :  pre_process_fun ### Before training 
         'pipe_list'   : [  List of preprocessors{'uri': 'source/prepro.py::pd_coly',      },  ]
}        
    



data_pars : {
    'n_sample' :  nb of sub-samples
    'download_pars' : None,
    'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },
 
    'cols_input_type' :  dict of raw data columns by data type : colcategory, colnumerics
    
    ### Model feed 
    'cols_model_group':  List of column groups to feed the model.
    'cols_model_type2':  dict of colunm groups to feed the model  by TYPE : colmodel_sparse, colmodel_dense
  
    "data_pars":         dict of specific data pars for the model. : tf_feature_column

    #### This part is GENERATED Dynamically from other data_pars   ###############################
    'data_flow' :
        "train" : dict of dataframe or path names for training.
        "val"   : Optional dict of dataframe or path names for validation
        "test"  : Optional dict of dataframe or path names for test.
    ###############################################################    
}



compute_pars = {
    'compute_pars'  :  Dict to pass DIRECTLY to YOUR MODEL .fit( , ** compute_pars['compute_pars'])
    'compute_extra' :  Dict used for training, But the params are NOT used directly (ie IF THEN ...mapping code) 
    "metrics_list"  :  list of sklearn metrics in string
}


global_pars = {
    "global_pars":    Dict of specific global pars for model

    "config_path"       = config_path  
    "config_name"       = config_name
    #### peoprocess input path
    "path_data_preprocess" = dir_data + f"/input/{data_name}/train/"

    #### train input path
    "path_data_train"      = dir_data + f"/input/{data_name}/train/"
    "path_data_test"       = dir_data + f"/input/{data_name}/test/"


    #### train output path
    "path_train_output"    = dir_data + f"/output/{data_name}/{config_name}/"
    "path_train_model"     = dir_data + f"/output/{data_name}/{config_name}/model/"
    "path_features_store"  = dir_data + f"/output/{data_name}/{config_name}/features_store/"
    "path_pipeline"        = dir_data + f"/output/{data_name}/{config_name}/pipeline/"

    #### predict  input path
    "path_pred_data"       = dir_data + f"/input/{data_name}/test/"
    "path_pred_pipeline"   = dir_data + f"/output/{data_name}/{config_name}/pipeline/"
    "path_pred_model"      = dir_data + f"/output/{data_name}/{config_name}/model/"

    #### predict  output path
    "path_pred_output"     = dir_data + f"/output/{data_name}/pred_{config_name}/"

    #####  Generic
    "n_sample"             = Nb samples

}








###########################################################################
#### Example  #############################################################
    m = {'model_pars': {
            # Specify the model
            'model_class':  "torch_tabular.py::RVAE",

            'model_pars' : {
                "activation":'relu',
                "outlier_model":'RVAE',
                "AVI":False,
                "alpha_prior":0.95,
                "embedding_size":50,
                "is_one_hot":False,
                "latent_dim":20,
                "layer_size":400,
            }


        },

        'compute_pars': {
            'compute_extra' :{ 
                "log_interval":50,
                "save_on":True,
                "verbose_metrics_epoch":True,
                "verbose_metrics_feature_epoch":False
            },

            'compute_pars' :{
                "cuda_on":False,
                "number_epochs":1,
                "l2_reg":0.0,
                "lr":0.001,
                "seqvae_bprop":False,
                "seqvae_steps":4,
                "seqvae_two_stage":False,
                "std_gauss_nll":2.0,
                "steps_2stage":4,
                "inference_type":'vae',
                "batch_size":150,
            },

            'metric_list': [
                'accuracy_score',
                'average_precision_score'
            ],


        },

        'data_pars': {
            'data_pars' :
               {"batch_size":150,   ### Mini Batch from data
                # Needed by getdataset
                "clean" : False,
                # "data_path":   path_pkg + '/data_simple/Wine/gaussian_m0s5_categorical_alpha0.0/5pc_rows_20pc_cols_run_1/',
               }
        },

        'global_pars' :{
            "data_path":   path_pkg + '/data_simple/Wine/gaussian_m0s5_categorical_alpha0.0/5pc_rows_20pc_cols_run_1/',
            "output_path": path_pkg + '/outputs_experiments_i/Wine/gaussian_m0s5_categorical_alpha0.0/5pc_rows_20pc_cols_run_1/RVAE_CVI',

        }

    }






















####################################################################################################
####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


class Model(object):
    """
           Generic Wrapper Class for WideDeep_dense
           Actual model is in Model.model

    """
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
        else:
            log2("data_pars", data_pars)
            model_class = model_pars['model_class']  #
            mdict       = model_pars['model_pars']


            ######### Size the model based on data size  ##############
            mdict['model_pars']['n_columns']  = data_pars['n_columns']





            ######### Create Model Instance  #########################
            self.model  = Modelcustom(**mdict)
            log2(self.model)



def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """ Train the model.model
    """
    global model, session

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle
    
    model.model.fit(Xtrain, ytrain, **cpars)





def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")
    else :
        cols_type   = data_pars['cols_model_type2'] 
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    ypred = model.model.predict(Xpred_tuple )

    ###### Add Probability 
    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)

    #####  Return prediction
    return ypred, ypred_proba






def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

#
def load_model(path=""):
    global model, session
    import dill as pickle





####################################################################################################
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :  ## No Tuple
        return Xtrain

    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )


    return Xtuple_train


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes to feed Model
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel   ### Column GROUP defined for the model

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### Defined source/run_train.py ---> data_pars ---> get_dataset
        ### ##3 Sparse, Continuous
        cols_type_received     = data_pars.get('cols_model_type2', {} )  

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)


            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



####################################################################################################
############ Do not change #########################################################################
def test(config=''):
    """
        Group of columns for the input model
           cols_input_group = [ ]
          for cols in cols_input_group,

    :param config:
    :return:
    """

    X = pd.DataFrame( np.random.rand(100,30), columns= [ 'col_' +str(i) for i in range(30)] )
    y = pd.DataFrame( np.random.binomial(n=1, p=0.5, size=[100]), columns = ['coly'] )
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    log(X_train.shape, )
    ##############################################################
    ##### Generate column actual names from
    colnum = [ 'col_0', 'col_11', 'col_8']
    colcat = [ 'col_13', 'col_17', 'col_13', 'col_9']

    cols_input_type_1 = {
        'colnum' : colnum,
        'colcat' : colcat
    }

    ###### Keras has 1 tuple input    ###########################
    colg_input = {
      'cols_cross_input':  ['colnum', 'colcat' ],
      'cols_deep_input':   ['colnum', 'colcat' ],
    }

    cols_model_type2= {}
    for colg, colist in colg_input.items() :
        cols_model_type2[colg] = []
        for colg_i in colist :
          cols_model_type2[colg].extend( cols_input_type_1[colg_i] )


    ##################################################################################
    model_pars = {'model_class': 'WideAndDeep',
                  'model_pars': {},
                }
    
    n_sample = 100
    data_pars = {'n_sample': n_sample,
                  'cols_input_type': cols_input_type_1,

                  'cols_model_group': ['colnum',
                                       'colcat',
                                       # 'colcross_pair'
                                       ],

                  'cols_model_type2' : cols_model_type2


        ### Filter data rows   #######################3############################
        , 'filter_pars': {'ymax': 2, 'ymin': -1}
                  }

    data_pars['train'] ={'Xtrain': X_train,  'ytrain': y_train,
                         'Xtest': X_test,  'ytest': y_test}
    data_pars['eval'] =  {'X': X_test,
                          'y': y_test}
    data_pars['predict'] = {'X': X_test}

    compute_pars = { 'compute_pars' : { 'epochs': 2,
                   } }

    ######## Run ###########################################
    test_helper(model_pars, data_pars, compute_pars)


def test_helper(model_pars, data_pars, compute_pars):
    global model, session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

    log('Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))
    #
    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.summary())


#######################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire(test)






