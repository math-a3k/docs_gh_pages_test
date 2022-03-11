# -*- coding: utf-8 -*-
"""
Only work with pyspark 2.4

pip install pyarrow==0.14.0
pip install pyspark==2.4.8

 
config_path = 'config/config.yaml'    



"""
import os
import pandas as pd
import pyspark
from pyspark.sql import SparkSession, functions as F, types as T


##### Custom import
from src.utils import config_load, log, spark_check


def run(spark:SparkSession, config_path: str='config.yaml'):
    """function run
    Args:
        spark ( SparkSession ) :   
        config_path (  str ) :   
    Returns:
        
    """
    #Load the yaml config file
    global conf_model
    conf = config_load( config_path  )
    check_path = conf['FilePaths']['check_path']
    conf_model = conf['volume_pred']


    #########################################################
    log("######  Load usersessionstats table")
    volume_sessionDF = preprocess(spark, conf, check=True)
    # cols_groupby = 'dummy'
    
    spark_check(volume_sessionDF, path= f"{check_path}/volume_sessionDF_pred_input", nsample=10000000 , save=True )


    log("####### Train  Run  ##############################")
    model_train(volume_sessionDF, conf_model, verbose=True)


    log("####### Prediction Run ###########################")
    cols_groupby = 'loggeddate'    
    result_schema = T.StructType([
      T.StructField('ds',T.StringType()),
      T.StructField('y', T.FloatType()),
      T.StructField('y_pred_max', T.FloatType()),
      T.StructField('y_pred_mean', T.FloatType()),
    ])


    @F.pandas_udf(result_schema, F.PandasUDFType.GROUPED_MAP)
    def fun_tseries_predict(groupkeys, df):
       global conf_model    
       return model_predict(df, conf_model, verbose=True)


    volume_sessionDF_pred = (volume_sessionDF.select('ds', 'y', cols_groupby )
            .groupBy([cols_groupby])
            .apply(fun_tseries_predict)
            .withColumn('training_date', F.current_date())
            )

    # df_pred =  model_predict(volume_sessionDF.toPandas(), conf_model, verbose=True)
    # volume_sessionDF_pred  =  spark.createDataFrame(df_pred,schema=result_schema)

    log("####### Prediction output ########################")
    volume_sessionDF_pred.persist() 
    volume_sessionDF_pred.write.parquet(conf['FilePaths']['volume_sessionDF_pred'],  mode="overwrite")
    volume_sessionDF_pred.show(10)


    log('####### Predict check:', volume_sessionDF_pred.count() )
    spark_check(volume_sessionDF_pred, path= f"{check_path}/volume_sessionDF_pred", nsample=10000 , save=True )




####################################################
def preprocess(spark, conf, check=True):
    """ Generate Structured Log table on disk
    :param spark:
    :param config:
    :return: Structured Log Spark Dataframe
    """
    user_sessionDF = spark.read.parquet( conf['FilePaths']['userlog_path'] )

    user_sessionDF.createOrReplaceTempView("userlogDF")

    sql = """ 
          SELECT                     
             date_format( loggeddate ,'yyyy-MM-dd') as loggeddate,
             from_unixtime(ts, 'dd-MM-yyyy HH:mm:ss') as ds  
            ,count(loggedtimestamp) as y
             --,day( loggeddate )   as day
             --,minute( loggedtimestamp ) as minute
             --,month( loggeddate ) as month
             --,pmod(datediff(loggeddate,'1900-01-07'),7) + 1 as weekday

          FROM userlogDF
          GROUP BY loggeddate, ds
          ORDER BY ds asc
    """
    dfhisto = spark.sql( sql)
    n = dfhisto.count()     
    
    if check :
        spark_check(dfhisto,  conf['FilePaths']['check_path'] +  "/dfhisto_volume",  nsample=10 , save=True)
    return dfhisto




def model_train(df:object, conf_model:dict, verbose:bool=True):
    """  Create a moving average model to calculate Upper Bound and Median value.
    Args:
        df:  Spark DataFrame of Pandas Dataframe
        conf_model: Dict conf
        verbose:  verbsosity

    Returns: None, model is saved on disk
    """
    if isinstance(df, pyspark.sql.DataFrame):
      #### Temporary solution due to simple model, collect is expensive operations
      df = df.toPandas()

    log(df.head(2))
    window = 5*50    #### Windows Period
    model_path = conf_model['model_path']
    quantile = conf_model.get('quantile', 0.999)
    tag = str(quantile).split(".")[-1]

    ######## Moving Average model
    df[f'y_ma_999'] = df['y'].rolling(window=window).quantile(quantile).interpolate(method='pad', limit_direction='both' )
    df[f'y_ma_999'] = df['y_ma_999'].interpolate(method='backfill', limit_direction='both' )

    df['y_ma_50']  = df['y'].rolling(window=window).quantile(0.50).interpolate(method='pad', limit_direction='both' )
    df['y_ma_50']  = df['y_ma_50'].interpolate(method='backfill', limit_direction='both' )

    ######## Normalize only 0h-24h period
    df['dsmin'] = df['ds'].apply(lambda x:   str(x)[11:-3].strip()   )
    dfg = df.groupby('dsmin').agg({  'y'        :  {'max', 'min', 'mean'},
                                     'y_ma_999' :  {'max', 'mean'},
                                     'y_ma_50'  :  {'mean', 'max'}
          })

    dfg.columns = [   f'{a}_{b}'  if b != '' else a for a, b in dfg.columns]


    ########  Generate  Full period 0f-24h Time --> Value  ###########################
    df2          = pd.DataFrame()
    df2['ds']    = pd.date_range('2015-01-01 00:00', '2015-01-01 23:59', freq='min')
    df2['dsmin'] =  df2['ds'].apply(lambda x:   str(x)[11:-3].strip()   )

    cols_pred = [t  for t in dfg.columns if t not in ['ds', 'dsmin']]
    df2 = df2.join(dfg, on ='dsmin', how='left')


    ######## Interpolate Missing values :  Efficient and simple method.
    for coli in cols_pred :
        df2[coli] = df2[coli].interpolate(method='linear', limit_direction='both' )

    ######## Saving on Disk 
    os.makedirs(model_path, exist_ok=True )  ### TODO: Hadoop version
    df2.to_parquet(model_path + "/model.parquet" ) 





def model_predict(df:pd.DataFrame,conf_model:dict, verbose:bool=True):
  """
  Args:
      df:  Pandas Dataframe
      conf_model:  conf Model
      verbose:

  Returns:

  """
  model_path = conf_model['model_path']
  dfmodel = pd.read_parquet(model_path)
  pred_table_model = dfmodel.set_index('dsmin')[ 'y_ma_999_max' ].to_dict()

  def predict_volume(t):
     key = str(t)[11:16]  ### shortcut for f'{t.hour}:{t.minute}'
     val = pred_table_model[key]  
     return val

  df['y'] = df['y'].astype('float64') 
   
  pred_table_model = dfmodel.set_index('dsmin')[ 'y_ma_999_max' ].to_dict()
  df['y_pred_max']  = df['ds'].apply(lambda x : predict_volume(x) )


  pred_table_model = dfmodel.set_index('dsmin')[ 'y_ma_50_mean' ].to_dict()  
  df['y_pred_mean'] = df['ds'].apply(lambda x : predict_volume(x) )

  return df[[ 'ds', 'y', 'y_pred_max', 'y_pred_mean'  ]]


















