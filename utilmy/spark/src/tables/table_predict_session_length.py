# -*- coding: utf-8 -*-
"""
6. Predict the session length for a given IP
  ---> Predict Next session Length :
## mode = 'train,pred'    config_path='config/config_test.yaml'

"""
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

############################################################################
from src.utils import config_load, log, spark_check
from src.util_models import Train, Predict




def run(spark:SparkSession, config_path: str='config.yaml', mode:str='train,pred'):
    """  Predict the session length
    Args:
        spark:
        config_path:
        mode:
    Returns:

    """
    prefix = 'sessionlength_pred'

    conf = config_load(config_path)
    check_path = conf['FilePaths']['check_path']
    conf_model = conf[prefix]
    regressor  = conf_model['regressor']  ### Model name
    model_path = conf_model['model_path'] + "/" + regressor


    log("######  Load Raw features table and Preprocesss #################")
    df, features = preprocess(spark, conf, check=True)
    spark_check(df, path= f"{check_path}/{prefix}", nsample=10 , save=True, verbose=True)


    if 'train' in mode :
       res = Train(spark, df, features, regressor,
                   path = model_path, conf_model = conf_model )

    if 'pred' in mode :
       df_pred = Predict(spark, df, features, regressor,
                         path = model_path, conf_model = conf_model  )

       df_pred.write.partitionBy("IPHashBucket").option("maxRecordsPerFile",
                     50000).parquet(conf_model['pred_path'],  mode="overwrite")

       spark_check(df_pred, path= f"{check_path}/{prefix}_features", nsample=10000 , save=True, verbose=True, )

       spark_check(df_pred[[  'user_id', 'session_duration',  'label', 'prediction'  ]],
                   path= f"{check_path}/{prefix}_small", nsample=20000 , save=True, verbose=True, )


def preprocess(spark, conf, check=True):
    """ Generate Structured Log table on disk
    :param spark:
    :param config:
    :return: Structured Log Spark Dataframe
    df.columns
    """
    log("########## Load userssionsta  ###############################################")
    df   = spark.read.parquet(  conf['FilePaths']['usersessionstats_path'] )
    win  = Window.partitionBy("IPHashBucket").orderBy("user_id", "starttimestamp")


    log("########## coly: label  #####################################################")
    coly = 'session_duration'
    #### T+1 sesssion_length to predict
    df   = df.withColumn( "label", F.lead(F.col(coly), 1).over(win) )


    log("########## Numerical features : T  ##########################################")
    colsnum = [  'n_unique_url', 'n_events', 'session_duration'  ]


    log("########## Auto-Regressive features : T-1  ##################################")
    for coli in colsnum :
       df = df.withColumn( coli + "_lag1", F.lag(F.col(coli), 1).over(win) )
    colsnum1 = [  coli + "_lag1"  for coli in colsnum ]


    log("########## Auto-Regressive features : T-2  ##################################")
    for coli in colsnum :
       df = df.withColumn( coli + "_lag2", F.lag(F.col(coli), 2).over(win) )
    colsnum2 = [  coli + "_lag2"  for coli in colsnum ]


    log("######### Categorical Variables  ############################################")
    colscat = [ 'os', 'device'  ]  ### 'hour'
    stage_string  = [StringIndexer(inputCol= c, outputCol= c + "_enc") for c in colscat ]
    stage_one_hot = [OneHotEncoder(inputCol= c+"_enc", outputCol= c+ "_1hot") for c in colscat ]
    ppl = Pipeline(stages= stage_string + stage_one_hot)
    df  = ppl.fit(df).transform(df)
    cols1hot =  [coli + "_1hot" for coli in colscat]
    log( df.show(1) )


    log("######### Merge  ############################################################")
    features     = colsnum + colsnum1 + colsnum2 + cols1hot
    vector_assembler = VectorAssembler(inputCols = features, outputCol= "features")
    df_m  = vector_assembler.setHandleInvalid("skip").transform(df)

    df_m = df_m.dropna(subset=('label'))   ### Remove Null Label
    df_m = df_m.fillna(0)   ### Zero value is OK

    return df_m,features








"""

21/06/03 22:50:04 WARN Utils: Truncated the string representation of a plan since it was too large. 
This behavior can be adjusted by setting 'spark.debug.maxToStringFields' in SparkEnv.conf.          
[Stage 29:=============================>                            (4 + 1) / 8]                    
2021-06-03 22:51:13.545 | INFO     | src.utils:log:10 - RMSE_train,256.21016539561293                                
2021-06-03 22:51:13.547 | INFO     | src.utils:log:10 - RMSE_test,255.68485347823224                                 
[Stage 44:===================================>                    (16 + 1) / 25]                                     
ils:log:10 - RMSE_train,256.21016539561293                                                                   
2021-06-03 22:51:13.547 | INFO     | src.utils:log:10 - RMSE_test,255.68485347823224   


Average level

df_m.show()

df.columns 

Caused by: org.apache.spark.SparkException: Encountered null while assembling a row with handleInvalid = "error". Consider   

"""





