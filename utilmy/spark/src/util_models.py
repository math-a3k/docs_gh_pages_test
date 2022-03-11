# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
6. Predict the session length for a given IP



"""
import os

import pyspark
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import types as T

############################################################################
from src.utils import log


###########################################################################
def TimeSeriesSplit(df_m:pyspark.sql.DataFrame, splitRatio:float, sparksession:object):
    """
    # Splitting data into train and test
    # we maintain the time-order while splitting
    # if split ratio = 0.7 then first 70% of data is train data
    Args:
        df_m:
        splitRatio:
        sparksession:

    Returns: df_train, df_test

    """
    newSchema  = T.StructType(df_m.schema.fields + \
                [T.StructField("Row Number", T.LongType(), False)])
    new_rdd        = df_m.rdd.zipWithIndex().map(lambda x: list(x[0]) + [x[1]])
    df_m2          = sparksession.createDataFrame(new_rdd, newSchema)
    total_rows     = df_m2.count()
    splitFraction  =int(total_rows*splitRatio)
    df_train       = df_m2.where(df_m2["Row Number"] >= 0)\
                          .where(df_m2["Row Number"] <= splitFraction)
    df_test        = df_m2.where(df_m2["Row Number"] > splitFraction)
    return df_train, df_test



##############################################################################
def Train(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str=None, conf_model:dict=None):
    """
    # this performs model training
    # this calls the machine-learning algorithms of Spark ML library
    # creating labels for machine-learning
    Args:
        spark: Sparksession
        df_m: Spark Dataframe Vector Assembler
        features:  column names
        regressor:  model name
        path:  model to save
        conf_model:  config in dict

    Returns: training resuls split

    """
    conf_model = {} if conf_model is None else conf_model
    splitratio = 0.7


    ##### Splitting data into train, test #########################
    df_train, df_test = TimeSeriesSplit(df_m, splitratio, spark)


    ##### LINEAR REGRESSOR    #####################################
    if(regressor == 'LinearRegression'):
        model = LinearRegression(featuresCol = "features", labelCol="label",
                              maxIter = 100, regParam = 0.4,
                              elasticNetParam = 0.1)

    ##### RANDOM FOREST REGRESSOR  ################################
    elif(regressor == 'RandomForestRegression'):
        model = RandomForestRegressor(featuresCol="features",
                                    labelCol="label",
                                    maxDepth = 5,
                                    subsamplingRate = 0.8,
                                    )
    else :
        return None   

    ##### 
    pipeline = Pipeline(stages=[model])               
    pipeline = pipeline.fit(df_train)
    predictions_test  = pipeline.transform(df_test)
    predictions_train = pipeline.transform(df_train)
    
    ###### RMSE is used as evaluation metric
    evaluator = RegressionEvaluator(predictionCol="prediction",
                                    labelCol="label",
                                    metricName ="rmse")
    RMSE_test  = evaluator.evaluate(predictions_test)
    RMSE_train = evaluator.evaluate(predictions_train)


    log('RMSE_train', RMSE_train)
    log('RMSE_test', RMSE_test)

    vals = (df_test, df_train,
            predictions_test, predictions_train,
            RMSE_test, RMSE_train)

    if path is not None :
        pipeline.write().overwrite().save(path  + "/model/")
    return vals  



def Predict(spark, df_m:pyspark.sql.DataFrame, features:list, regressor:str, path:str=None, conf_model:dict=None):
    """
    # this performs model training
    # this calls the machine-learning algorithms of Spark ML library
    # creating labels for machine-learning
    Args:
        spark:  SparkSession
        df:  Spark Dataframe Vector Assembler
        features: column features
        regressor:  model name
        path:  model path
        conf_model:  conf in dict.
    Returns:
    """    
    ##### LINEAR REGRESSOR    #####################################
    if(regressor == 'LinearRegression'):
        model = LinearRegression(featuresCol = "features", labelCol="label", 
                                 maxIter = 100, regParam = 0.4, 
                                 elasticNetParam = 0.1
                                )
    
    ##### RANDOM FOREST REGRESSOR  ################################
    elif(regressor == 'RandomForestRegression'):
        model = RandomForestRegressor(featuresCol="features", labelCol="label",
                                      maxDepth = 5,
                                      subsamplingRate = 0.8,
                                     )
    else :
        return None  

    pipe     = PipelineModel(stages=[model])  
    pipefit  = pipe.load(path + "/model/" )
    df_pred  = pipefit.transform(df_m)
    return df_pred



####################################################################
####################################################################
def os_makedirs(path:str):
  """function os_makedirs
  Args:
      path ( str ) :   
  Returns:
      
  """
  if 'hdfs:' not in path :
    os.makedirs(path, exist_ok=True)
  else :
    os.system(f"hdfs dfs mkdir -p '{path}'")


