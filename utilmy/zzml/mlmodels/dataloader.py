# -*- coding: utf-8 -*-
"""
Typical user workflow

def get_dataset(data_pars):
    loader = DataLoader(data_pars)
    loader.compute()
    data = loader.get_data()
    [log(x.shape) for x in data]
    return data


data_pars --> Dataloader.py  :
  sequence of pre-processors item
       uri, args
       return some objects in a sequence way.



"data_pars": {  
 "data_info": { 
                  "name" : "text_dataset",   "data_path": "dataset/nlp/WIKI_QA/" , 
                  "train": true
                  } 
         },
 

"preprocessors": [ 
                {  "uri" : "mlmodels.preprocess.generic.:train_test_val_split",
                    "arg" : {   "split_if_exists": true, "frac": 0.99, "batch_size": 64,  "val_batch_size": 64,
                                    "input_path" :    "dataset/nlp/WIKIQA_singleFile/" ,
                                    "output_path" :  "dataset/nlp/WIKIQA" ,   
                                    "format" : "csv"
                               },
                    "output_type" :  "file_csv"
                } ,  


             {  "name" : "loader"  ,
                "uri" :  "mlmodels.model_tch.matchzoo:WIKI_QA_loader",
                "arg" :  {  "name" : "text_dataset",   
                                        "data_path": "dataset/nlp/WIKI_QA/"   ,
                                         "data_pack"  : "",   "mode":"pair",  "num_dup":2,   "num_neg":1,
                                        "batch_size":20,     "resample":true,  
                                        "sort":false,   "callbacks":"PADDING"
                                      },
                 "output_type" :  "pytorch_dataset"
             } ]
}


"""
#### System utilities
import os
import sys
import inspect
from urllib.parse import urlparse
from jsoncomment import JsonComment ; json = JsonComment()
from importlib import import_module
import pandas as pd
import numpy as np
from collections.abc import MutableMapping
from functools import partial

from pprint import pprint as print2



from sklearn.model_selection import train_test_split
import cloudpickle as pickle
import fire

#########################################################################
#### mlmodels-internal imports
from mlmodels.util import load_callable_from_dict, load_callable_from_uri, path_norm, path_norm_dict, log

from mlmodels.preprocess.generic import load_function

from mlmodels.preprocess.generic import pandasDataset, NumpyDataset

#########################################################################
VERBOSE = 0 
DATASET_TYPES = ["csv_dataset", "text_dataset", "NumpyDataset", "pandasDataset"]
TRANSFORMER_LOADERS = ["NLIDataReader", "STSBenchmarkDataReader"]


#########################################################################
def pickle_load(file):
    return pickle.load(open(file, " r"))


def pickle_dump(t, **kwargs):
    with open(kwargs["path"], "wb") as fi:
        pickle.dump(t, fi)
    return t


def image_dir_load(path):
    ## TODO : implement it
    return ImageDataGenerator().flow_from_directory(path)


def batch_generator(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def _validate_data_info(self, data_info):
    dataset = data_info.get("dataset", None)
    if not dataset:
        raise KeyError("Missing dataset key in the dataloader.")

    dataset_type = data_info.get("dataset_type", None)

    if dataset_type and dataset_type not in DATASET_TYPES:
        raise Exception(f"Unknown dataset type {dataset_type}")
    return True

    self.path          = path
    self.dataset_type  = dataset_type
    self.test_size     = data_info.get("test_size", None)
    self.generator     = data_info.get("generator", False)
    self.batch_size    = int(data_info.get("batch_size", 1))

    self.col_Xinput    = data_info.get("col_Xinput", None)
    self.col_Yinput    = data_info.get("col_Yinput", None)
    self.col_miscinput = data_info.get("col_miscinput", None)


def _check_output_shape(self, inter_output, shape, max_len):
    case = 0
    if isinstance(inter_output, tuple):
        if not isinstance(inter_output[0], dict):
            case = 1
        else:
            case = 2
    if isinstance(inter_output, dict):
        if not isinstance(tuple(inter_output.values())[0], dict):
            case = 3
        else:
            case = 4
    # max_len enforcement
    if max_len is not None:
        try:
            if case == 0:
                inter_output = inter_output[0:max_len]
            if case == 1:
                inter_output = [o[0:max_len] for o in inter_output]
            if case == 3:
                inter_output = {
                    k: v[0:max_len] for k, v in inter_output.items()
                }
        except:
            pass
    # shape check
    if shape is not None:
        if (
            case == 0
            and hasattr(inter_output, "shape")
            and tuple(shape) != inter_output.shape
        ):
            raise Exception(f"Expected shape {tuple(shape)} does not match  {inter_output.shape[1:]}")

        if case == 1:
            for s, o in zip(shape, inter_output):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(f"Expected shape {tuple(shape)} does not match  {inter_output.shape[1:]}")

        if case == 3:
            for s, o in zip(shape, tuple(inter_output.values())):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(f"Expected shape {tuple(shape)} does not match  {inter_output.shape[1:]}")

    self.output_shape = shape
    return inter_output



def get_dataset_type(x) :
    from mlmodels.preprocess.generic import PandasDataset, NumpyDataset, Dataset, kerasDataset  #Pytorch
    from mlmodels.preprocess.generic import DataLoader  ## Pytorch


    if isinstance(x, PandasDataset  ) : return "PandasDataset"
    if isinstance(x, NumpyDataset  ) : return "NumpyDataset"
    if isinstance(x, Dataset  ) : return "pytorchDataset"




#################################################################################################
class DataLoader:
    """
      Class which read the json and execute to load the data and send to the model


    """
    default_loaders = {
        ".csv": {"uri": "pandas::read_csv", "pass_data_pars":False},
        ".parquet": {"uri": "pandas::read_parquet", "pass_data_pars":False},
        ".npy": {"uri": "numpy::load", "pass_data_pars":False},
        ".npz": {"uri": "np:load", "arg": {"allow_pickle": True}, "pass_data_pars":False},
        ".pkl": {"uri": "dataloader::pickle_load", "pass_data_pars":False},

        "image_dir": {"uri": "dataloader::image_dir_load", "pass_data_pars":False},
    }
    _validate_data_info = _validate_data_info
    _check_output_shape = _check_output_shape
    
    def __init__(self, data_pars):
        self.final_output             = {}
        self.internal_states          = {}
        self.data_info                = data_pars['data_info']
        self.preprocessors            = data_pars.get('preprocessors', [])
        # self.final_output_type        = data_pars['output_type']



    def check(self):
        # Validate data_info without execution
        self._validate_data_info(self.data_info)

        input_type_prev = "file"   ## HARD CODE , Bad

        for preprocessor in self.preprocessors:
            uri = preprocessor.get("uri", None)
            if not uri:
                log(f"Preprocessor {preprocessor} missing uri")


            ### Compare date type for COMPATIBILITY
            input_type = preprocessor.get("input_type", "")   ### Automatic ???
            if input_type != input_type_prev :
                log(f"Mismatch input / output data type {preprocessor} ")                  

            input_type_prev = preprocessor.get('output_type', "")
       

    def compute(self, docheck=True, verbose=True):
        if docheck :
            self.check()

        input_tmp = None
        for preprocessor in self.preprocessors:
            uri  = preprocessor["uri"]
            args = preprocessor.get("args", {})
            log("URL: ",uri, args)

       
            # preprocessor_func = load_callable_from_uri(uri)
            preprocessor_func = load_function(uri)
            log("\n###### load_callable_from_uri LOADED",  preprocessor_func)
            if inspect.isclass(preprocessor_func):
                ### NAME Should match PytorchDataloader, KerasDataloader, PandasDataset, ....
                ## A class : muti-steps compute
                cls_name = preprocessor_func.__name__
                if verbose : print("cls_name :", cls_name, flush=True)


                if cls_name in DATASET_TYPES:  # dataset object
                    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)

                    if cls_name == "pandasDataset" or cls_name == "NumpyDataset": # get dataframe/numpyarray instead of pytorch dataset
                        out_tmp = obj_preprocessor.get_data()
                    else:
                        out_tmp = obj_preprocessor

                elif cls_name in TRANSFORMER_LOADERS:  # transformer loader object
                    obj_preprocessor = preprocessor_func(**args)
                    out_tmp = obj_preprocessor

                else:  # pre-process object defined in preprocessor.py
                    log("\n", "Object Creation")
                    obj_preprocessor = preprocessor_func(**args)

                    log("\n", "Object Compute")
                    obj_preprocessor.compute(input_tmp)

                    log("\n", "Object get_data")                    
                    out_tmp = obj_preprocessor.get_data()


            else:
                ### Only a function, not a Class : Directly COMPUTED.
                # log("input_tmp: ",input_tmp['X'].shape,input_tmp['y'].shape)
                # log("input_tmp: ",input_tmp.keys())
                pos_params = inspect.getfullargspec(preprocessor_func)[0]

                log("\n ######### postional parameters : ", pos_params)
                log("\n ######### Execute : preprocessor_func", preprocessor_func)

                if isinstance(input_tmp, (tuple, list)) and len(input_tmp) > 0 and len(pos_params) == 0:
                    out_tmp = preprocessor_func(*input_tmp, **args)

                elif pos_params == ['data_info']:
                    log( f"function with postional parmater data_info {preprocessor_func} , (data_info, **args)")
                    out_tmp = preprocessor_func(data_info=self.data_info, **args)

                else:
                    out_tmp = preprocessor_func(input_tmp, **args)

            ## Be careful of Very Large Dataset, it will not work not to save ALL 
            ## Save internal States
            if preprocessor.get("internal_states", None):
                for internal_state in preprocessor.get("internal_states", None):
                    if isinstance(out_tmp, dict):
                        self.internal_states[internal_state] = out_tmp[internal_state]

            input_tmp = out_tmp
        self.final_output = out_tmp


    def get_data(self, return_internal_states=False):
        ### Return the data wrapper
        if return_internal_states:
            return self.final_output, self.internal_states
         else :
            return self.final_output 


##########################################################################################################
### Test functions
def split_xy_from_dict(out, **kwargs):
    X_c    = kwargs.get('col_Xinput',[])
    y_c    = kwargs.get('col_yinput',[])
    X      = [out[n] for n in X_c]
    y      = [out[n] for n in y_c]
    return (*X,*y)



def test_run_model():
    log("\n\n\n###### Test_run_model  #############################################################")
    from mlmodels.models import test_module

    ll = [
        #### Keras
        "model_keras/charcnn.json",
        "model_keras/charcnn_zhang.json",
        "model_keras/textcnn.json",
        "model_keras/namentity_crm_bilstm.json",


        ### Torch
        'dataset/json/refactor/resnet18_benchmark_mnist.json',
        'dataset/json/refactor/resnet34_benchmark_mnist.json',
        'dataset/json/refactor/model_list_CIFAR.json',
        'dataset/json/refactor/torchhub_cnn_dataloader.json',
        'dataset/json/refactor/resnet18_benchmark_FashionMNIST.json',
        'dataset/json/refactor/model_list_KMNIST.json',
        'model_tch/transformer_sentence.json',


        ### textcnn
        'dataset/json/refactor/textcnn.json',


    ]

    not_fittable_models = ['dataset/json/refactor/torchhub_cnn_dataloader.json']

    for x in ll :
         try :
            log("\n\n\n", "#" * 100)
            log(x )

            data_path = path_norm(x)
            param_pars = {"choice": "json", "data_path": data_path, "config_mode": "test"}
            with open(data_path) as json_file:
                config = json.load(json_file)

            log( json.dumps(config, indent=2))
            test_module(config['test']['model_pars']['model_uri'], param_pars, 
                        fittable = False if x in not_fittable_models else True)

         except Exception as e :
            import traceback
            traceback.print_exc()
            print("######## Error", x,  e, flush=True)



def test_single(arg):
    data_pars_list  =  [
            path_norm( arg.path)
    ] 
    test_json_list(data_pars_list)




def test_dataloader(path='dataset/json/refactor/'):
    refactor_path = path_norm( path )
    # data_pars_list = [(f,json.loads(open(refactor_path+f).read())['test']['data_pars']) for f in os.listdir(refactor_path)]

    # data_pars_list = [ refactor_path + "/" + f for f in os.listdir(refactor_path)  if os.path.isfile( refactor_path + "/" + f)  ]
    # log(data_pars_list)


    data_pars_list  =  [

        'model_keras/charcnn.json',
        'model_keras/charcnn_zhang.json',
        'model_keras/textcnn.json',
        'model_keras/namentity_crm_bilstm.json',


        ### DO NOT remove the torch examples
        'dataset/json/refactor/torchhub_cnn_dataloader.json',
        'dataset/json/refactor/model_list_CIFAR.json',
        'dataset/json/refactor/resnet34_benchmark_mnist.json',


        #### Text
        'dataset/json/refactor/textcnn.json',

        'model_tch/transformer_sentence.json',


    ]


    test_json_list(data_pars_list)



def test_json_list(data_pars_list):

    for f in data_pars_list:
        try :
          #f  = refactor_path + "/" + f
          # f= f.replace("gitdev/mlmodels/",  "gitdev/mlmodels2/" )
          f = path_norm(f)

          if os.path.isdir(f) : continue

          log("\n" *5 , "#" * 100)
          log(  f, "\n")

          log("#"*5, " Load JSON data_pars") 
          d = json.loads(open( f ).read())
          data_pars = d['test']['data_pars']
          data_pars = path_norm_dict( data_pars)
          #log( textwrap.fill( str(data_pars), 90 ) )
          log( json.dumps(data_pars, indent=2))


          log( "\n", "#"*5, " Load DataLoader ") 
          loader    = DataLoader(data_pars)


          log("\n", "#"*5, " compute DataLoader ")           
          loader.compute()

          log("\n", "#"*5, " get_Data DataLoader ")  
          log(loader.get_data())

        except Exception as e :
          import traceback
          traceback.print_exc()
          print("Error", f,  e, flush=True)    





####################################################################################################
def cli_load_arguments(config_file=None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    import argparse
    p = argparse.ArgumentParser()
    def add(*k, **kw):
        p.add_argument(*k, **kw)

    add("--config_file" , default=None                     , help="Params File")
    add("--config_mode" , default="test"                   , help="test/ prod /uat")
    add("--log_file"    , help="File to save the logging")
    add("--do"          , default="test_single"                   , help="what to do test or search")

    ###### model_pars
    add("--path", default='dataset/json/refactor/torchhub_cnn_dataloader.json' , help="name of the model for --do test")
    add("--file", default='dataset/json/refactor/', help="name of the model for --do test")


    ###### data_pars
    # add("--data_path", default="dataset/GOOG-year_small.csv", help="path of the training file")

    ###### out params
    # add('--save_path', default='ztest/search_save/', help='folder that will contain saved version of best model')

    args = p.parse_args()
    # args = load_config(args, args.config_file, args.config_mode, verbose=0)
    return args


def main():
    arg = cli_load_arguments()

    if arg.do == "test":
        test_dataloader('dataset/json/refactor/')   

    if arg.do == "test_run_model":
       test_run_model()


    if arg.do == "test_single":
        test_single(arg)  


if __name__ == "__main__":
   """
      python mlmodels/dataloader.py  test_dataloader  --path  'dataset/json/refactor/''


   """ 
   VERBOSE =1  
   # main()
   fire.Fire()
    
#    test_run_model()

 



    
"""
#########################################################################
def pickle_load(file):
    return pickle.load(open(f, " r"))


def pickle_dump(t,path):
    pickle.dump(t, open(path, "wb" ))


def image_dir_load(path):
    return ImageDataGenerator().flow_from_directory(path)


def batch_generator(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def _interpret_input_pars(self, input_pars):
    try:
        path = input_pars["path"]
    except KeyError:
        raise Exception("Missing path key in the dataloader.")

    path_type = input_pars.get("path_type", None)
    if path_type is None:
        if os.path.isfile(path):
            path_type = "file"
        if os.path.isdir(path):
            path_type = "dir"

        if urlparse(path).scheme != "":
            path_type = "url"
            download_path = input_pars.get("download_path", "./")

        if path_type == "dropbox":
            dropbox_download(path)
            path_type = "file"

        if path_type is None:
            raise Exception(f"Path type for {path} is undeterminable")

    elif path_type != "file" and path_type != "dir" and path_type != "url":
        raise Exception("Unknown location type")

    file_type = input_pars.get("file_type", None)
    if file_type is None:
        if path_type == "dir":
            file_type = "image_dir"
        elif path_type == "file":
            file_type = os.path.splitext(path)[1]
        else:
            if path[-1] == "/":
                raise Exception("URL must target a single file.")
            file_type = os.path.splittext(path.split("/")[-1])[1]

    self.path = path
    self.path_type = path_type
    self.file_type = file_type
    self.test_size = input_pars.get("test_size", None)
    self.generator = input_pars.get("generator", False)
    if self.generator:
        try:
            self.batch_size = int(input_pars.get("batch_size", 1))
        except:
            raise Exception("Batch size must be an integer")
    self.col_Xinput = input_pars.get("col_Xinput", None)
    self.col_Yinput = input_pars.get("col_Yinput", None)
    self.col_miscinput = input_pars.get("col_miscinput", None)
    validation_split_function = [
        {"uri": "sklearn.model_selection::train_test_split", "arg": {}},
        "test_size",
    ]
    self.validation_split_function = input_pars.get(
        "split_function", validation_split_function
    )


def _check_output_shape(self, inter_output, shape, max_len):
    case = 0
    if isinstance(inter_output, tuple):
        if not isinstance(inter_output[0], dict):
            case = 1
        else:
            case = 2
    if isinstance(inter_output, dict):
        if not isinstance(tuple(inter_output.values())[0], dict):
            case = 3
        else:
            case = 4
    # max_len enforcement
    if max_len is not None:
        try:
            if case == 0:
                inter_output = inter_output[0:max_len]
            if case == 1:
                inter_output = [o[0:max_len] for o in inter_output]
            if case == 3:
                inter_output = {
                    k: v[0:max_len] for k, v in inter_output.items()
                }
        except:
            pass
    # shape check
    if shape is not None:
        if (
            case == 0
            and hasattr(inter_output, "shape")
            and tuple(shape) != inter_output.shape
        ):
            raise Exception(
                f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
            )
        if case == 1:
            for s, o in zip(shape, inter_output):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(
                        f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
                    )
        if case == 3:
            for s, o in zip(shape, tuple(inter_output.values())):
                if hasattr(o, "shape") and tuple(s) != o.shape[1:]:
                    raise Exception(
                        f"Expected shape {tuple(shape)} does not match shape data shape {inter_output.shape[1:]}"
                    )
    self.output_shape = shape
    return inter_output


class DataLoader:

    default_loaders = {
        ".csv": {"uri": "pandas::read_csv", "pass_data_pars":False},
        ".npy": {"uri": "numpy::load", "pass_data_pars":False},
        ".npz": {"uri": "np:load", "arg": {"allow_pickle": True}, "pass_data_pars":False},
        ".pkl": {"uri": "dataloader::pickle_load", "pass_data_pars":False},
        "image_dir": {"uri": "dataloader::image_dir_load", "pass_data_pars":False},
    }
    _validate_data_info = _validate_data_info
    _check_output_shape = _check_output_shape
    
    def __init__(self, data_pars):
        self.final_output             = {}
        self.internal_states          = {}
        self.data_info                = data_pars['data_info']
        self.preprocessors            = data_pars.get('preprocessors', [])

    def compute(self):
        # Validate data_info
        self._validate_data_info(self.data_info)

        input_tmp = None
        for preprocessor in self.preprocessors:
            uri = preprocessor.get("uri", None)
            if not uri:
                raise Exception(f"Preprocessor {preprocessor} missing uri")

            name = preprocessor.get("name", None)
            args = preprocessor.get("args", {})

            preprocessor_func = load_callable_from_uri(uri)
            if name == "loader":
                out_tmp = preprocessor_func(self.path, **args)
            elif name == "saver":  # saver do not return output
                preprocessor_func(self.path, **args)
            else:
                if inspect.isclass(preprocessor_func):
                    obj_preprocessor = preprocessor_func(**args)
                    obj_preprocessor.compute(input_tmp)
                    out_tmp = obj_preprocessor.get_data()
                else:
                    if isinstance(input_tmp, (tuple, list)):
                        out_tmp = preprocessor_func(*input_tmp[:2], **args)
                    else:
                        out_tmp = preprocessor_func(input_tmp, **args)
            if preprocessor.get("internal_states", None):
                for internal_state in preprocessor.get("internal_states", None):
                    if isinstance(out_tmp, dict):
                        self.internal_states[internal_state] = out_tmp[internal_state]

            input_tmp = out_tmp
        self.final_output = out_tmp

    def get_data(self):
        return self.final_output, self.internal_states



### Test functions
def split_xy_from_dict(out,data_pars):
    X_c    = data_pars['input_pars'].get('col_Xinput',[])
    y_c    = data_pars['input_pars'].get('col_yinput',[])
    misc_c = data_pars['input_pars'].get('col_miscinput',[])
    X      = [out[n] for n in X_c]
    y      = [out[n] for n in y_c]
    misc   = [out[n] for n in misc_c]
    return (*X,*y,*misc)


if __name__ == "__main__":
    from mlmodels.models import test_module

    # param_pars = {
    #     "choice": "json",
    #     "config_mode": "test",
    #     "data_path": "dataset/json/refactor/03_nbeats_dataloader.json",
    # }
    # test_module("model_tch/03_nbeats_dataloader.py", param_pars)

    param_pars = {
        "choice": "json",
        "config_mode": "test",
        "data_path": "dataset/json/refactor/namentity_crm_bilstm_dataloader_new.json",
    }
    test_module("model_keras/namentity_crm_bilstm.py", param_pars)

    """
    
    
    
    
    
