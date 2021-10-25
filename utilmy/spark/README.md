##### To add utilities
```

https://github.com/zaksamalik/pyspark-utilities




```




                                                                                               
##### 2) Tests + Coverage Result                                   
```

==== 8 passed, 3 warnings in 96.03s (0:01:36) =======================

Name                                        Stmts   Miss  Cove                                                      
--------------------------------------------------------------                                                      
src\__init__.py                                 0      0   100%                                                      
src\functions\GetFamiliesFromUserAgent.py       8      0   100%                                                      
src\tables\table_predict_volume.py             74     56    24%                                                      
src\tables\table_user_log.py                   37      2    95%                                                      
src\tables\table_user_session_log.py           29      0   100%                                                      
src\tables\table_user_session_stats.py         24      0   100%                                                      
src\util_models.py                             54     54     0%                                                      
src\utils.py                                   36      6    83%                                                     
---------------------------------------------------------------                                                     
TOTAL                                         262    118    55%



#### Command
git clone https://github.com/arita37/zzeqe_knoe.git
git checkout test
cd zzeqe_knoe
pytest --cov=src/  --html=output/tests/report.html --self-contained-html  tests/   


```






#### 4) Install and Running full data
```
##### Check Dockerfile, Docker-compose.yml as follow :


#### Local 
    cd zzeqe_knoe
    python main.py  --onfig_path  config/config.yaml


#### Spark Submit  (Need to edit the pats
   script/sparkrunscript.sh


```






##### 5) Log Output
```
Copy Paste of the full run CI

https://github.com/arita37/zzeqe_knoe/blob/test/README_logs.md




```












