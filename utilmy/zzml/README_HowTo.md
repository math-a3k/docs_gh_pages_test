
## How to find information ?
<details>
A lit of Github Issues :
   https://github.com/arita37/mlmodels/issues?q=is%3Aopen+is%3Aissue+label%3Adev-documentation

</details>
<br/>



## A 1min example
```python
from mlmodels.models import module_load

#### Save to JSON
model_pars   =  {"model_uri": "model_tf.1_lstm.py",
                  "num_layers": 1, "size": ncol_input, "size_layer": 128, "output_size": ncol_output, "timestep": 4,   }
data_pars    =  {"data_path": "/folder/myfile.csv"  , "data_type": "pandas" }
compute_pars =  { "learning_rate": 0.001, }
out_pars     =  { "path": "ztest_1lstm/", "model_path" : "ztest_1lstm/model/"}

module        =  module_load( model_uri= model_pars["model_uri"] )                    # Load file definition
module.init(model_pars, data_pars, compute_pars)    # Create Model instance
module.fit(data_pars, compute_pars, out_pars)

#### Inference
metrics_val   =  module.evaluate(data_pars, compute_pars, out_pars)  # Inference
ypred         = module.predict(data_pars, compute_pars, out_pars)       # Predict pipeline
```
<br/>

## How to install mlmodels ?
<details>

There are two types of installations for ```mlmodels```.
The 1st one is using gitpod , other with script

### Install with gitpod
   [ Gitpod install ](https://github.com/arita37/mlmodels/issues/101)

   Benefit of gitpod is you only need to install once and it is available everywhere.
   

### Install with script on Colab
One can also use the [run_install.sh](https://github.com/arita37/mlmodels/blob/dev/install/run_install.sh) and other similar files
for an automatic installation.

</details>
<br/>



## How to check if mlmodels works ?
<details>

Status logs are available here :
   [Test logs](https://github.com/arita37/mlmodels_store/tree/master/log_import)

There are automatic runs to check if current repo is working or not.
You can check the us here :
    [Testing details](https://github.com/arita37/mlmodels/blob/dev/README_testing.md)

 Code source of test :  [Test source](https://github.com/arita37/mlmodels/blob/dev/mlmodels/ztest.py)   

 
After install, Basic testing can be done with command line tool ```ml_test```.
ml_test refers to mlmodels/ztest.py

### test_fast_linux : Basic Import check
```ml_test --do test_fast_linux```

1. [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_fast_linux.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_import)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/) 

### test_cli : Command Line Testing
```ml_test --do test_cli```

1. [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_cli.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_test_cli)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_dataloader : Test if dataloader works
```ml_test --do test_dataloader```

1. [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_dataloader.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_dataloader)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_jupyter : Test if jupyter notebooks works
```ml_test --do test_jupyter```

1. [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_jupyter.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_jupyter)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_benchmark : benchmark
```ml_test --do test_benchmark```

1. [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_benchmark.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_benchmark)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)

### test_pull_request : PR 
```ml_test --do test_jupyter```

1. [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_pull_request.yml)
2. [RAW LOGS](https://github.com/arita37/mlmodels_store/tree/master/log_pullrequest)
3. [CLEAN LOGS](https://github.com/arita37/mlmodels_store/tree/master/error_list/)


You can then run basic codes and models to verify correct installation and
work environment. See debugging part.


</details>
<br/>



## How to check if one specific model works ?
<details>

### Run Model
Run/Test newly added model  on 
[Gitpod](https://gitpod.io/) or [Colab](https://colab.research.google.com/).


Example of Gitpod use:
```bash
source activate py36
cd mlmodels
python model_tf/1_lstm.py  
```

### Automatic test runs
There are automatic test runs to check
    [Testing details](https://github.com/arita37/mlmodels/blob/dev/README_testing.md)

</details>
<br/>



## How the testing works ?
<details>
  Testing follows those steps :

  1) Github Actions --> Triggers some test runs in mlmodels/ztest.py 
      [Github Actions](https://github.com/arita37/mlmodels/blob/dev/.github/workflows)

  2) --> Trigger some CLI or code using mlmodels 
  3) --> Test logs is submitted to [mlmodels_store repo](https://github.com/arita37/mlmodels_store/blob/master/log_import/log_import.py)
  4) --> Logs are visible for manual check.
  
  
  
Automatic testing is enabled and results are described here :

   https://github.com/arita37/mlmodels/blob/adata2/README_testing.md


Code for testing all the repo is located here:
   https://github.com/arita37/mlmodels/blob/dev/mlmodels/ztest.py


</details>
<br/>




## How to develop using Colab ?
<details>

https://github.com/arita37/mlmodels/issues/262

</details>
<br/>



## How to develop using Gitpod ?
<details>

https://github.com/arita37/mlmodels/issues/101

</details>
<br/>



## How to add  a model ?
<details>

https://github.com/arita37/mlmodels/blob/adata2/README_addmodel.md

To add new model fork the repo. Inside the mlmodels directory we have multiple
subdirectories named like model_keras, model_sklearn and so on the idea is to use
**model_** before the type of framework you want to use. 

Now once you have decided the 
frame work create appripriately named model file and config file as described in the read me 
doc [README_addmodel.md](docs\README_docs\README_addmodel.md). The same model structure 
and config allows us to do the testing of all the models easily.

</details>
<br/>



## How to use Command Line CLI ?
<details>

https://github.com/arita37/mlmodels/blob/adata2/README_usage_CLI.md

</details>
<br/>



## How the model configuration JSON works ?
<details>

Detailed description of the JSON format is described here :
https://github.com/arita37/mlmodels/blob/dev/docs/DEV_docs/json.md


Sample of model written in JSON is located here : 
https://github.com/arita37/mlmodels/tree/dev/mlmodels/dataset/json


A model computation is describred in 4 parts:

```
myjson.json
{

model_pars
compute_pars
data_pars
out_pars

#### Optional
hypermodel_pars
}
```

**Examples**
```json
    {
        "test": {

              "hypermodel_pars":   {
             "learning_rate": {"type": "log_uniform", "init": 0.01,  "range" : [0.001, 0.1] },
             "num_layers":    {"type": "int", "init": 2,  "range" :[2, 4] },
             "size":    {"type": "int", "init": 6,  "range" :[6, 6] },
             "output_size":    {"type": "int", "init": 6,  "range" : [6, 6] },

             "size_layer":    {"type" : "categorical", "value": [128, 256 ] },
             "timestep":      {"type" : "categorical", "value": [5] },
             "epoch":         {"type" : "categorical", "value": [2] }
           },

            "model_pars": {
                "learning_rate": 0.001,     
                "num_layers": 1,
                "size": 6,
                "size_layer": 128,
                "output_size": 6,
                "timestep": 4,
                "epoch": 2
            },

            "data_pars" :{
              "path"            : 
              "location_type"   :  "local/absolute/web",
              "data_type"   :   "text" / "recommender"  / "timeseries" /"image",
              "data_loader" :  "pandas",
              "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
              "size" : [0,1,2],
              "output_size": [0, 6]              
            },


            "compute_pars": {
                "distributed": "mpi",
                "epoch": 10
            },
            "out_pars": {
                "out_path": "dataset/",
                "data_type": "pandas",
                "size": [0, 0, 6],
                "output_size": [0, 6]
            }
        },
    
        "prod": {
            "model_pars": {},
            "data_pars": {}
        }
    }
```

</details>
<br/>



## How dataloader works ?
<details>

[dataloader.md](https://github.com/arita37/mlmodels/blob/dev/docs/DEV_docs/dataloader.md)

</details>
<br/>






## How to check test log after commit ?
<details>

Once the model is added we can do testing on it with commands like this, where model_framework is a placeholder for your selected framework and model_file.json is the config file for your model.

```
ml_models --do fit     --config_file model_framework/model_file.json --config_mode "test" 
```
Here the fit method is tested, you can check the predict fucntionality of the model like this.
```
ml_models --do predict --config_file model_tf/1_lstm.json --config_mode "test"
```
But this is individual testing that we can do to debug our model when we find an error in automatic the test logs.

We have automated testing in our repo and the results are stored in here https://github.com/arita37/mlmodels_store We havemultiple level logs and they are put under different directories as you can see here, log folders have **logs_** at the start.
![Mlmodels Store](docs/imgs/test_repo.PNG?raw=true "Mlmodels Store")
We can focus on the error_list directory to debug our testing errors. Inside the error_list directory we can find the logs of all test cases in directories named at the time they are created
![Error List](docs/imgs/error_list.PNG?raw=true "Error List")
Inside we can see separate files for each test cases which will have the details of the errors.
![Error Logs](docs/imgs/error_logs.PNG?raw=true "Error logs")
For example we can look at the errors for test cli cases named as list_log_test_cli_20200610.md
![Error](docs/imgs/test_cli_error.PNG?raw=true "Error")
We see multiple erros and we can click on the traceback for error 1 which will take us to the line 421 of the log file.
![Error Line](docs/imgs/error_line.PNG?raw=true "Error Line")
We can see that while running the test case at line 418 caused the error, and we can see the error. 
```
ml_models --do fit  --config_file dataset/json/benchmark_timeseries/gluonts_m4.json --config_mode "deepar" 
```
So we fix the erorr by launch the git pod and test the test case again and see it works correctly after that we can commit teh changes and submit the pull request.

</details>
<br/>

## How to debug the repo ?
<details>
  
To debug the repo, you should first verify correct installation with the following basic commands:

```bash
cd mlmodels
python optim.py
python model_tch/textcnn.py
python model_keras/textcnn.py
```

Another helpful thing to do would be to [search](https://github.com/search?q=pretrained+repo%3Aarita37%2Fmlmodels+path%3A%2Fmlmodels%2F+filename%3Amlmodels+filename%3Autil+filename%3Abenchmark+filename%3Aoptim+language%3APython+language%3APython&type=Code&ref=advsearch&l=Python&l=Python) the repo for relevant debugging information.

Make sure your interface is complete:

models.py 
```
   module_load(model_uri)
   model_create(module)
   fit(model, module, session, data_pars, out_pars   )
   metrics(model, module, session, data_pars, out_pars)
   predict(model, module, session, data_pars, out_pars)
   save(model, path)
   load(model)
```

optim.py
```
   optim(modelname="model_tf.1_lstm.py",  model_pars= {}, data_pars = {}, compute_pars={"method": "normal/prune"}
       , save_folder="/mymodel/", log_folder="", ntrials=2) 

   optim_optuna(modelname="model_tf.1_lstm.py", model_pars= {}, data_pars = {}, compute_pars={"method" : "normal/prune"},
                save_folder="/mymodel/", log_folder="", ntrials=2) 
```

Associated json files must be perfectly made for your specific model. Check that all parameters
are present.


If nothing works then make sure you have followed all the right steps from this HowTo markdown file.
Particularly, don't forget to create your test json file. If the issue persists then submit an issue, 
all developpers are very active a,d will get back to you quickly.



### Manual installation
The manual installation is dependant on [install/requirements.txt](https://github.com/arita37/mlmodels/blob/dev/install/requirements.txt)
and other similar text files.

Preview:
```bash
pandas<1.0
scipy>=1.3.0
scikit-learn==0.21.2
numexpr>=2.6.8
```


```bash
Linux/MacOS
pip install numpy<=1.17.0
pip install -e .  -r install/requirements.txt
pip install   -r install/requirements_fake.txt

Windows (use WSL + Linux)
pip install numpy<=1.17.0
pip install torch==1..1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .  -r requirements_wi.txt
pip install   -r install/requirements_fake.txt
```




</details>
<br/>








