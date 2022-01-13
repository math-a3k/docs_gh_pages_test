### 1. What is dataloader?
Dataloader is helper module which load data from configuration and prepare for training/evaluation task.
### 2. Dataloader's configuration: store in json file with key "data_pars" have 2 main parts:
- **data_info**: conntain common data info such as data_path, dataset_name, batch_size when processing and dataset_type like training and testing
- **preprocessors**: list of data preprocessing which could be a function or a class object specified by "uri". Argument for those function or class object init passed by "args". These preprocessor will be processed one by one in sequence (TBD: how to process data with more dynamic ways with complicated combination like sequence and parallel)
Example:
```
"data_pars": {
    "data_info": {
            "data_path"  : "dataset/recommender/",
            "dataset"    : "IMDB_sample.txt",
            "data_type"  : "csv_dataset",
            "batch_size" : 64,
            "train"      : true
        },
    "preprocessors": [
        {"uri"  : "mlmodels.model_tch.textcnn:split_train_valid",
         "args" : {
                    "frac": 0.99
                    }
        },
        {"uri"  : "mlmodels.model_tch.textcnn:create_tabular_dataset",
         "args" : {
                    "lang": "en",
                    "pretrained_emb": "glove.6B.300d"
                    }

        }
        ]
},
```
### 3. Dataloader workflow
- Dataload output contain 2 parts: 
   + dataset output: memory object for next step like training/validating
   + internal_state: which is dictionary to store extra-data if need
- To get output, just init dataloader and invoke dataloader **compute** function which return tuple (dataset_outout, internal_state)
During computing, dataloader init all objects in list of preprocessor and invoke function to preprocess data

**Note**
- Class object in dataloader configuration need to implement 3 interface:
```buildoutcfg
def __init__(self, **args):
    # initialization
def compute(self, input_tmp):
    # self.data = xxx

def get_data(self):
    return self.data
```    


### Algorithm base :


```python

uri : is the location of the function or class to be executed.
We just execute sequentially the uri.

for processor in processor_list :
   get(uri, args)
   load uri
   Execute uri with args
   retturn results


Main issue is multiple sub-process for each processor
and data passing format.



Datalaoder  : 
     pipeline manager, Manage a sequence of tasks.

     There are many ways to process data :
        saving on disk or not, ...



Dataset :
     Class which wraps data on disk/web : numpy, pandas, TF_dataser....     


Iterator (ie "Dataloader") :
     Transfer, Transform and Format from Dataset to XXXX-Tensor.
     XXXX: is framwork : Tflow, Pytorch, ....

     This is the final connector to Model.
     Need to convert   Dataset  into XXXX Tensor.



### Notes :
   train, val, test are processed in indepedant way : Split is done before.


A)

Batchable requires the whole flow as batch
disk_Batch --> Memory_batch --> tensor_batch
dataloader.py manages the sequence of tasks.
Some pre-processing does not need to be batched... Example:
Big-Data Processor ---> Save on disk (/train , /test , /val)


B)
It's better to have one Dataloader per framework XXXXX'
but those dataloader can re-use functions between them...

Since most Deep Learning frameworks contain a 'Dataloading' class to create
Iterables that can be sampled batchwise, functions can be implemented to 
instantiate dataloaders of those respective frameworks, thereby keeping us 
from reinventing the wheel.



C) Embedding loader
Exaple : Glove
Load from XXX embedding to Tensor XXX or model XXX
NO batch, direct full memory


-->
Manage API breaks more easily....
Re-use existing code....





```



### Files :

Dataloader manager : manage the pipeline.
https://github.com/arita37/mlmodels/blob/adata2/mlmodels/dataloader.py


Generic Wrapper of Dataset:
https://github.com/arita37/mlmodels/blob/adata2/mlmodels/preprocess/generic.py


Various Pre-processors:
https://github.com/arita37/mlmodels/blob/adata2/mlmodels/preprocess/




### Automatic Log

https://github.com/arita37/mlmodels_store/blob/master/log_dataloader/log_dataloader.py




##### Example   1
```python

From this file
https://github.com/arita37/mlmodels/blob/adata2//mlmodels/dataset/json/refactor/charcnn.json 


"data_pars" :
{

  ### Generic to all pre-processors.
  "data_info": {
    "dataset": "mlmodels/dataset/text/ag_news_csv",
    "train": true,
    "alphabet_size": 69,
    "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    "input_size": 1014,
    "num_of_classes": 4
  },


  ### List of invidual pre-processors
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::pandasDataset",
      "args": {
        "colX"          : ["colX"],
        "coly"          : ["coly"],
        "encoding"      : "'ISO-8859-1'",
        "read_csv_parm" : {"usecols": [0, 1 ], "names": ["coly", "colX"] }
      }
    },
    {
      "name": "tokenizer",
      "uri": "mlmodels/model_keras/raw/char_cnn/data_utils.py::Data",
      "args": {
        "data_source"    : "",
        "alphabet"       : "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
        "input_size"     : 1014,
        "num_of_classes" : 4
      }
    }
  ]
}





 #####  get_Data DataLoader 
((array([[65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       [65, 19, 18, ...,  0,  0,  0],
       ...,
       [ 5,  3,  9, ...,  0,  0,  0],
       [20,  5, 11, ...,  0,  0,  0],
       [20,  3,  5, ...,  0,  0,  0]]), array([[0, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 0],
       ...,
       [1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0]])), {})
```


##### Example  2
```python
 ####################################################################################################
https://github.com/arita37/mlmodels/blob/adata2//mlmodels/dataset/json/refactor/torchhub_cnn_dataloader.json 

#####  Load JSON data_pars
{
  "data_info": {
    "data_path": "mlmodels/dataset/vision/MNIST",
    "dataset": "MNIST",
    "data_type": "tch_dataset",
    "batch_size": 10,
    "train": true
  },
  "preprocessors": [
    {
      "name": "tch_dataset_start",
      "uri": "mlmodels.preprocess.generic::get_dataset_torch",
      "args": {
        "dataloader" : "torchvision.datasets:MNIST",
        "to_image"   : true,
        "transform"  : {
              "uri"            : "mlmodels.preprocess.image:torch_transform_mnist",
              "pass_data_pars" : false,
              "arg"            : {"fixed_size": 256, "path": "dataset/vision/MNIST/"}
        },
        "shuffle": true,
        "download": true
      }
    }
  ]
}

 #####  Load DataLoader 

 #####  compute DataLoader 

###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f2df3e85378>

 ######### postional parameteres :  ['data_info']

 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f2df3e85378>

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 
  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

 #####  get_Data DataLoader 
((<torch.utils.data.dataloader.DataLoader object at 0x7f2ddd09f908>, <torch.utils.data.dataloader.DataLoader object at 0x7f2ddd0aa940>), {})


```






##### Example 3
```python
 ####################################################################################################
https://github.com/arita37/mlmodels/blob/adata2//mlmodels/dataset/json/refactor/keras_textcnn.json 

#####  Load JSON data_pars
{
  "data_info": {
    "dataset": "mlmodels/dataset/text/imdb",
    "pass_data_pars": false,
    "train": true,
    "maxlen": 40,
    "max_features": 5
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::NumpyDataset",
      "args": {
        "numpy_loader_args": {
          "allow_pickle": true
        },
        "encoding": "'ISO-8859-1'"
      }
    }
  ]
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::NumpyDataset {'numpy_loader_args': {'allow_pickle': True}, 'encoding': "'ISO-8859-1'"} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.NumpyDataset'>
cls_name : NumpyDataset
Dataset File path :  mlmodels/dataset/text/imdb.npz
```


##### Example   4

https://github.com/arita37/mlmodels/blob/adata2//mlmodels/dataset/json/refactor/namentity_crm_bilstm_new.json 

```python
{
  "data_info": {
    "data_path": "dataset/text/",
    "dataset": "ner_dataset.csv",
    "pass_data_pars": false,
    "train": true
  },
  "preprocessors": [
    {
      "name": "loader",
      "uri": "mlmodels/preprocess/generic.py::pandasDataset",
      "args": {
        "read_csv_parm": {"encoding": "ISO-8859-1"},
        "colX": [],
        "coly": []
      }
    },
    {
      "uri": "mlmodels/preprocess/text_keras.py::Preprocess_namentity",
      "args": {
        "max_len": 75
      },
      "internal_states": [
        "word_count"
      ]
    },
    {
      "name": "split_xy",
      "uri": "mlmodels/dataloader.py::split_xy_from_dict",
      "args": {
        "col_Xinput": ["X"],
        "col_yinput": ["y"]
      }
    },
    {
      "name": "split_train_test",
      "uri": "sklearn.model_selection::train_test_split",
      "args": {
        "test_size": 0.5
      }
    },
    {
      "name": "saver",
      "uri": "mlmodels/dataloader.py::pickle_dump",
      "args": {
        "path": "mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl"
      }
    }
  ],
  "output": {
    "shape": [[75 ], [75, 18 ] ],
    "max_len": 75
  }
}

 #####  Load DataLoader 

 #####  compute DataLoader 

  URL:  mlmodels/preprocess/generic.py::pandasDataset {'read_csv_parm': {'encoding': 'ISO-8859-1'}, 'colX': [], 'coly': []} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/generic.pandasDataset'>
cls_name : pandasDataset

  URL:  mlmodels/preprocess/text_keras.py::Preprocess_namentity {'max_len': 75} 

###### load_callable_from_uri LOADED <class 'mlmodels/preprocess/text_keras.Preprocess_namentity'>
cls_name : Preprocess_namentity

 Object Creation

 Object Compute

 Object get_data

  URL:  mlmodels/dataloader.py::split_xy_from_dict {'col_Xinput': ['X'], 'col_yinput': ['y']} 

###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2df2fd01e0>

 ######### postional parameteres :  ['out']

 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2df2fd01e0>

  URL:  sklearn.model_selection::train_test_split {'test_size': 0.5} 

###### load_callable_from_uri LOADED <function train_test_split at 0x7f2e46605d90>

 ######### postional parameteres :  []

 ######### Execute : preprocessor_func <function train_test_split at 0x7f2e46605d90>

  URL:  mlmodels/dataloader.py::pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

###### load_callable_from_uri LOADED <function pickle_dump at 0x7f2dddd1ee18>

 ######### postional parameteres :  ['t']



```





