# -*- coding: utf-8 -*-
import argparse
import importlib

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from src.utils import config_load, log


def pd_to_spark_format():

import pyarrow as pa
    fields = [
                pa.field('EmpId', pa.int32()),
                pa.field('FirstName', pa.string()),
                pa.field('LastName', pa.string()),
                pa.field('EmailID', pa.string()),
                pa.field('ActiveFlag', pa.int8()),
             ]
    pa_schema = pa.schema(fields) # Create a schema from the pa fields
    pandasDF = pd.read_parquet("emp.parquet") # Read parquet file into a Pandas dataframe
    paTbl = pa.Table.from_pandas(pandasDF, schema=pa_schema) # Create a pyarrow table from pandas and apply the schema
    paTbl.schema.remove_metadata() # We see the schema now shows ActiveFlag correctly as int8
    paTbl["ActiveFlag"] # Even the values in the table show up as integers

    paDict    = paTbl.to_pydict()
    df_spark  = spark.createDataFrame(zip(*paDict.values()), list(paDict.keys()))
    df_target = spark.createDataFrame(df_spark.rdd, schema_target) # Apply the schema we created earlier
    df_target.printSchema() # Print the schema to validate
    df_target.show() # Show the records and now it won't error out on TypeError. Yoohoo!  
    


def config_getdefault():
    pass
    
def test():    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder\
        .appName('abc')\
        # .config("spark.jars","/myfolder/spark-avro_2.12-2.4.4.jar")\
        .master("local[*]")\
        .getOrCreate()
    df = spark.read.format("avro").load(file_path)




def spark_init(config:dict=None)->SparkSession:
    """
    Args:
        config: config dict
    Returns: SparkSession
    """
    if config is None :
        spark = SparkSession.builder\
            .appName('app1')\
            # .config("spark.jars","/myfolder/spark-avro_2.12-2.4.4.jar")\
            .master("local[*]")\
            .getOrCreate()
        return spark
    
    cfg  = config['sparkconfig']
    conf = SparkConf().setMaster(cfg['spark.master']).setAppName(cfg['spark.app.name'])

    pars_list = [
        'spark.sql.session.timeZone',
        'spark.sql.shuffle.partitions',
        "spark.driver.memory",
        "spark.executor.memory",
    ]
    for name in pars_list :
       conf.set(name,   cfg[name])

    sc = SparkContext.getOrCreate(conf)
    spark = SparkSession(sc)
    return spark




def main():
    """ Execute all processing
        python main.py  --config_path  config/config.yaml         ### full
        python main.py  --config_path  config/config_test.yaml    ### test
        config_path = "config/config.yaml"
    """
    pars = argparse.ArgumentParser(description='Process')
    pars.add_argument('--config_path',type=str, nargs='?', const='config/config_test.yaml',
                      default='config/config_test.yaml', help='Config path')
    args = pars.parse_args()

    config_path = args.config_path
    log('Using', config_path)

    ###### Spark init ##########################################
    config = config_load(config_path)
    spark = spark_init(config)


    ###### Job Pipelining ######################################
    ##Those job can be setup in Airflow as DAG too
    task_list = [

        #### Table generation
        'src.tables.table_user_log',
        'src.tables.table_user_session_log',
        'src.tables.table_user_session_stats',

        #### Prediction
        'src.tables.table_predict_volume',
        'src.tables.table_predict_session_length',
        'src.tables.table_predict_url_unique'
    ]


    for task in task_list :
        log(f"\n\n######## Executing task {task} #################")
        job = importlib.import_module(task)
        res = job.run(spark, config_path)



if __name__ == '__main__':
    main()






