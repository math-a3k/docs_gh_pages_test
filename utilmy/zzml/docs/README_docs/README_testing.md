# Testing Process

Testing is all automated, test parse automatically new files in the repo 
and add it to the testing process



## Flow of testing

### test_fast_linux : Basic Import check
    1. https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_fast_linux.yml
    
    2. ml_test --do test_fast_linux

    3. https://github.com/arita37/mlmodels_store/tree/master/log_import     : Raw Logs

    4. https://github.com/arita37/mlmodels_store/tree/master/error_list/      : Clean Logs


### test_cli : Command Line Testing
    Test if input CLI is 

    1. https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_cli.yml
    
    2. ml_test --do test_cli

    3. https://github.com/arita37/mlmodels_store/tree/master/log_test_cli     : Raw Logs

    4. https://github.com/arita37/mlmodels_store/tree/master/error_list/      : Clean Logs


### test_dataloader : Test if dataloader works
    Used to assess accuracy of the models, and compare different models.

    1. https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_dataloader.yml
    
    2. ml_test --do test_dataloader

    3. https://github.com/arita37/mlmodels_store/tree/master/log_dataloader     : Raw Logs

    4. https://github.com/arita37/mlmodels_store/tree/master/error_list/      : Clean Logs


### test_jupyter : Test if jupyter notebooks works
    Used to assess accuracy of the models, and compare different models.

    1. https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_jupyter.yml
    
    2. ml_test --do test_jupyter

    3. https://github.com/arita37/mlmodels_store/tree/master/log_jupyter     : Raw Logs

    4. https://github.com/arita37/mlmodels_store/tree/master/error_list/      : Clean Logs



### test_benchmark : benchmark
    Used to assess accuracy of the models, and compare different models.

    1. https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_benchmark.yml
    
    2. ml_test --do test_benchmark

    3. https://github.com/arita37/mlmodels_store/tree/master/log_benchmark     : Raw Logs

    4. https://github.com/arita37/mlmodels_store/tree/master/error_list/      : Clean Logs



### test_pull_request : PR 
    Used when submitting a PR :
        Run all .py files in the folder pullrequest/

    1. https://github.com/arita37/mlmodels/blob/dev/.github/workflows/test_pullrequest.yml
    
    2. ml_test --do test_pullrequest

    3. https://github.com/arita37/mlmodels_store/tree/master/log_pullrequest     : Raw Logs

    4. https://github.com/arita37/mlmodels_store/tree/master/error_list/      : Clean Logs






### Overview

#### Configuration of auto-test :
https://github.com/arita37/mlmodels/blob/dev/.github/workflows/

    test_cli         : CLI
    test_dataloader  : dataloader part
    test_benchmark   : benchmark
    test_fast_linux  : On Commit
    test_pullrequest : On PR


####  Code of auto-test :
https://github.com/arita37/mlmodels/blob/dev/mlmodels/ztest.py


#### Log of testing
Uploaded directly here :
https://github.com/arita37/mlmodels_store


#### Parsing/Cleaning the logs
https://github.com/arita37/mlmodels_store/blob/master/.github/workflows/auto.yml


#### Search reference in the code :
[link](https://github.com/search?q=pretrained+repo%3Aarita37%2Fmlmodels+path%3A%2Fmlmodels%2F+filename%3Amlmodels+filename%3Autil+filename%3Abenchmark+filename%3Aoptim+language%3APython+language%3APython&type=Code&ref=advsearch&l=Python&l=Python)



####  Helper code
```python
     from mlmodels.util import os_package_root_path, log, 
                            path_norm, get_model_uri, path_norm_dict

     ### Use path_norm to normalize your path.
     data_path = path_norm("dataset/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/dataset/text/myfile.txt


     ### Use path_norm to normalize your path.
     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt


     data_path = path_norm("ztest/text/myfile.txt")
        --> FULL_ PATH   /home/ubuntu/mlmodels/ztest/text/myfile.txt
```





