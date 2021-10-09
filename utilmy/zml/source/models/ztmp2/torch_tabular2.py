# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python torch_tabular.py test --nrows 1000

https://github.com/arita37/pytorch_tabular
https://github.com/manujosephv/pytorch_tabular/tree/main/pytorch_tabular/models

Bug  
    cmd ="python -m pip install git+https://github.com/manujosephv/pytorch_tabular.git@82a30fe2ad1cc8c4f883d86d5f63925e67a0a015 --no-deps"

The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.
Args:
    config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
        the path to the yaml file holding all the config parameters. Defaults to None.
    data_config (Optional[Union[DataConfig, str]], optional): DataConfig object or path to the yaml file. Defaults to None.
    model_config (Optional[Union[ModelConfig, str]], optional): A subclass of ModelConfig or path to the yaml file.
        Determines which model to run from the type of config. Defaults to None.
    optimizer_config (Optional[Union[OptimizerConfig, str]], optional): OptimizerConfig object or path to the yaml file.
        Defaults to None.
    trainer_config (Optional[Union[TrainerConfig, str]], optional): TrainerConfig object or path to the yaml file.
        Defaults to None.
    experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or path to the yaml file.
        If Provided configures the experiment tracking. Defaults to None.
    model_callable (Optional[Callable], optional): If provided, will override the model callable that will be loaded from the config.
        Typically used when providing Custom Models

"""
import os, sys,  numpy as np,  pandas as pd, wget, copy
from pathlib import Path
try :
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import (CategoryEmbeddingModelConfig, TabNetModelConfig, NodeConfig,
                                        CategoryEmbeddingMDNConfig,  AutoIntConfig )

    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

except :
    print(" !! Couldn't import pytorch_tabular, pip install ****************************************")
    cmd ="python -m pip install git+https://github.com/manujosephv/pytorch_tabular.git@82a30fe2ad1cc8c4f883d86d5f63925e67a0a015 "
    # cmd = "pip install pytorch_tabular[all]"
    os.system(cmd)
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import (CategoryEmbeddingModelConfig, TabNetModelConfig, NodeConfig,
                                        CategoryEmbeddingMDNConfig,  AutoIntConfig )
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


print(TabularModel)
MODEL_DICT = { "CategoryEmbeddingModelConfig":  CategoryEmbeddingModelConfig ,
    "TabNetModelConfig" : TabNetModelConfig ,
    "NodeConfig" : NodeConfig,
    "CategoryEmbeddingMDNConfig": CategoryEmbeddingMDNConfig,
    "AutoIntConfig" :  AutoIntConfig
}

# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)
# from torch.utils import data
from sklearn.model_selection import train_test_split
import torch
####################################################################################################
VERBOSE = True

def log(*s):
    print(*s, flush=True)

def log2(*s):
    print(*s, flush=True)

####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None

        else:
            ###############################################################
            dm          = data_pars['cols_model_group_custom']
            data_config = DataConfig(
              target           = dm['coly'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
              continuous_cols  = dm['colnum'],
              categorical_cols = dm['colcat'],
            )

            class_name   = model_pars.get('model_class',  "CategoryEmbeddingModelConfig" ).split("::")[-1]
            assert class_name in MODEL_DICT, "ModelConfig not available"
            log2(class_name)

            # Pick the needed ModelConfig
            #model_class = globals()[ class_name ]

            model_class = None
            if class_name == "CategoryEmbeddingModelConfig":
                model_class = CategoryEmbeddingModelConfig
            elif class_name == "TabNetModelConfig":
                model_class = TabNetModelConfig
            else:
                model_class = NodeConfig
            #model_class  = MODEL_DICT[class_name]

            model_config     = model_class( **self.model_pars.get('model_pars', {})   )
            trainer_config   = TrainerConfig( **compute_pars.get('compute_pars', {} )) # For testing quickly, max_epochs=1 )
            optimizer_config = OptimizerConfig(**compute_pars.get('optimizer_pars', {} ))

            self.config_pars = { 'data_config' : data_config,
                        'model_config'         : model_config,
                        'optimizer_config'     : optimizer_config,
                        'trainer_config'       : trainer_config,
            }

            self.model = TabularModel(**self.config_pars)
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log( self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    # global model, session
    session = None  # Session type for compute
    log2(model)
    log2(model.model)

    # if data_pars is not None :
    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle

    train = pd.concat((Xtrain_tuple[0], Xtrain_tuple[1]), axis=1)
    train = pd.concat((train, ytrain), axis=1)

    val   = pd.concat((Xtest_tuple[0], Xtest_tuple[1]), axis=1)
    val   = pd.concat((val, ytest), axis=1)

    ###############################################################
    model.model.fit(train=train, validation=val) # , **cpars)



def predict(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
    global model, session

    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")
    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)   

    Xpred_tuple_concat = pd.concat((Xpred_tuple[0], Xpred_tuple[1]), axis=1)
    ypred = model.model.predict(Xpred_tuple_concat)
    
    #####################################################################
    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba



def reset():
    global model, session
    model, session = None, None

def save(path=None, info=None):
    """ Custom saving
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(path + "/model/", exist_ok=True)

    #### Torch part
    model.model.save_model(path + "/model/torch_checkpoint")

    #### Wrapper
    model.model = None   ## prevent issues
    pickle.dump(model,  open(path + "/model/model.pkl", mode='wb')) # , protocol=pickle.HIGHEST_PROTOCOL )
    pickle.dump(info, open(path   + "/model/info.pkl", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(path + '/model/model.pkl', mode='rb'))
         
    model = Model()  # Empty model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars
    model.data_pars    = model0.data_pars

    ### Custom part
    # model.model        = TabularModel.load_from_checkpoint( "ztmp/data/output/torch_tabular/torch_checkpoint")
    model.model        = TabularModel.load_from_checkpoint(  path +"/model/torch_checkpoint")
 
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


def get_dataset2(data_pars=None, task_type="train", **kw):
    """
      "ram"  :
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



# cols_ref_formodel = ['cols_single_group']
cols_ref_formodel = ['colcontinuous', 'colsparse']
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3) OR single dataframe
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :
        return Xtrain

    Xtuple_train = []
    # cols_ref is the reference for types of cols groups (sparse/continuous)
    # This will result in deviding the dataset into many groups of features
    for cols_groupname in cols_ref :
        # Assert the group name is in the cols reference
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        # Add the columns of this group to the list
        Xtuple_train.append( Xtrain[cols_i] )

    return Xtuple_train


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname

        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

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
############ Test  #################################################################################
def test_dataset_covtype(nrows=1000):

    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am" , "Hillshade_Noon",  "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3", "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9", "Soil_Type10",  "Soil_Type11",  "Soil_Type12",  "Soil_Type13",  "Soil_Type14", "Soil_Type15",  "Soil_Type16",  "Soil_Type17",  "Soil_Type18",  "Soil_Type19", "Soil_Type40",  ]

    # Target column
    coly        = ["Covertype"]

    log("start")
    global model, session

    root = os.path.join(os.getcwd() ,"ztmp")


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
    return df, colnum, colcat, coly


def test(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    df, colnum, colcat, coly = test_dataset_covtype()

    #### Matching Big dict  ##################################################
    X = df[ colnum + colcat ]
    y = df[coly].astype('uint8')
    log('y', np.sum(y[y==1]) )

    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)
    num_classes = len(set(y_train_full[coly].values.ravel()))
    log(X_train)

    cols_input_type_1 = []
    n_sample          = 100
    def post_process_fun(y):
        return int(y)

    def pre_process_fun(y):
        return int(y)

    m = {'model_pars': {
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  "torch_tabular.py::CategoryEmbeddingModelConfig"
        
        # Type of target prediction, evaluation metrics
        ,'model_pars' : { 'task': "classification",
                        'metrics' : ["f1","accuracy"],
                        'metrics_params' : [{"num_classes":num_classes},{}],

                        # 'layers' : "1024-512-512",  # Number of nodes in each layer
                        # 'activation' : "LeakyReLU", # Activation between each layers
                        'learning_rate' :  1e-3
                        }  

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },

        #### catcol INTO integer,   colcat into OneHot
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },

        ],  }},

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score'],

      'compute_pars' :{  ### Specif
        'batch_size' :1024,'max_epochs' :1, 'gpus': 0

      }
                    },

    'data_pars': { 'n_sample' : n_sample,
        'download_pars' : None,

        'cols_input_type' : { 'colid'  : 'id0',
                              'coly'   : coly,
                              'colnum' : colnum,
                              'colcat' : colcat            },

        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum_bin',
                              'colcat_bin',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                            }
        ###################################################
        ,'train': {'Xtrain': X_train,   'ytrain': y_train,
                   'Xtest': X_valid,    'ytest': y_valid},
                'eval': {'X': X_valid,  'y': y_valid},
                'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' :      colcat,
        },
        }
    }

    ##### Running loop
    """
    Neural Oblivious Decision Ensembles for Deep Learning on 
    Tabular Data is a model presented in ICLR 2020 and 
    according to the authors have beaten well-tuned Gradient Boosting models on many datasets.

    TabNet: Attentive Interpretable Tabular Learning is another model coming out 
    of Google Research which uses Sparse Attention in multiple steps 
    of decision making to model the output.


    """
    ll = [
        ('torch_tabular.py::CategoryEmbeddingModelConfig', {}),
        ('torch_tabular.py::TabNetModelConfig', {} ),
        ('torch_tabular.py::NodeConfig', {}),
        ('torch_tabular.py::CategoryEmbeddingMDNConfig', {}),
        ('torch_tabular.py::AutoIntConfig', {}) ,

    ]
    for cfg in ll:

        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg[0]
        log('\n\n### Config ...', m )

        log('\n\n### Setup model..', cfg[0])
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )



        log('\n\n### Training odel..')
        log2(model.model)
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')


        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

        if cfg != "torch_tabular.py::NodeConfig":
            log('Saving model..')
            save(path= "ztmp/data/output/torch_tabular")
            #  os.path.join(root, 'data\\output\\torch_tabular\\model'))

            log('Load model..')
            model, session = load_model(path="ztmp/data/output/torch_tabular")
            #os.path.join(root, 'data\\output\\torch_tabular\\model'))
        else:
            log('\n*** !!! Saving Bug in pytorch_tabular for NodeConfig !!! ***\n')
            
        log('Model architecture:', model.model )
        log('Model config:', model.model.config._config_name)
        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])

        reset()


def test3():
    pass


def test2(nrows=10000):
    """
       python source/models/torch_tabular.py test

    """
    global model, session


    df,colcat, colnum, coly = test_dataset_covtype(1000)
    target_name =  coly

    df.head()
    train, test = train_test_split(df, random_state=42)
    train, val  = train_test_split(train, random_state=42)
    num_classes = len(set(train[target_name].values.ravel()))


    data_config = DataConfig(
        target=target_name,
        continuous_cols=colnum,
        categorical_cols=colcat,
        continuous_feature_transform=None,#"quantile_normal",
        normalize_continuous_features=False
    )
    model_config = CategoryEmbeddingModelConfig(task="classification",
                                                metrics=["f1","accuracy"],
                                                metrics_params=[{"num_classes":num_classes},{}])

    trainer_config = TrainerConfig(gpus=None, fast_dev_run=True)
    experiment_config = ExperimentConfig(project_name="PyTorch Tabular Example",
                                         run_name="node_forest_cov",
                                         exp_watch="gradients",
                                         log_target="wandb",
                                         log_logits=True)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        # experiment_config=experiment_config,
    )

    tabular_model.fit(  train=train, validation=val)
    result = tabular_model.evaluate(val)
    log(result)
    
    
    test.drop(columns=target_name, inplace=True)
    pred_df = tabular_model.predict(val.iloc[:100,:])

    log(pred_df)
    # pred_df.to_csv("output/temp2.csv")
    # tabular_model.save_model("test_save")
    # new_model = TabularModel.load_from_checkpoint("test_save")
    # result = new_model.evaluate(test)


if __name__ == "__main__":
    import fire
    fire.Fire()
    # test()


