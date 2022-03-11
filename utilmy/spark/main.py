# -*- coding: utf-8 -*-
HELP="""
You can run Spark in local mode using local, local[n] or the most general local[*] for the master URL.

The URL says how many threads can be used in total:
local[n] uses n threads.

local[*] uses as many threads as the number of processors available to the Java virtual machine (it uses Runtime.getRuntime.availableProcessors() to know the number).

local[N, maxFailures] (called local-with-retries) with N being * or the number of threads to use (as explained above) and maxFailures being the value of spark.task.maxFailures.
    # .config("spark.jars","/myfolder/spark-avro_2.12-2.4.4.jar")\


"""
import argparse, importlib
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from utilmy.spark.src.utils import config_load, log



################################################################################
def config_default():
  """function config_default
  Args:
  Returns:
      
  """
  ss ="""
- sparkconfig:
    maxrecordfile                      : 50000

    #spark.master                      : 'yarn'
    spark.master                       : 'local[1]'   # 'spark://virtual:7077'
    spark.app.name                     : 'logprocess'
    spark.driver.maxResultSize         : '10g'
    spark.driver.memory                : '10g'
    spark.driver.port                  : '45975'
    #spark.eventLog.enabled             : 'true'
    #spark.executor.cores               : 2
    #spark.executor.id                  : 'driver'
    #spark.executor.instances           : 2
    spark.executor.memory              : '10g'
    #spark.kryoserializer.buffer.max    : '2000mb'
    spark.rdd.compress                 : 'True'
    spark.serializer                   : 'org.apache.spark.serializer.KryoSerializer'
    #spark.serializer.objectStreamReset : 100
    spark.sql.shuffle.partitions       : 8
    spark.sql.session.timeZone         : "UTC"    
    # spark.sql.catalogImplementation  : 'hive'
    #spark.sql.warehouse.dir           : '/user/myuser/warehouse'
    #spark.sql.warehouse.dir           : '/tmp'    
    spark.submit.deployMode            : 'client'  
  
  """
  
  
    
def pd_to_spark_hive_format(df, dirout):
  """
     To export into Spark/Hive format, Issue with pyarrow.
  
  """
  import fastparquet as fp
  fp.write(dirout, df, )

  

def config_getdefault():
    """function config_getdefault
    Args:
    Returns:
        
    """
    pass
    
def test():    
    """function test
    Args:
    Returns:
        
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.builder\
        .appName('main')\
        .master("local[*]")\
        .getOrCreate()

    df = spark.read.format("parquet").load(file_path)




def spark_init(config:dict=None, appname='app1', local="local[*]")->SparkSession:
    """  from utilmy.spark import spark_init
    Args:
        config: config dict
    Returns: SparkSession
    """
    if config is None :
        spark = SparkSession.builder\
            .appName(appname)\
            .master(local)\
            .getOrCreate()
        return spark

    
    ########################################################
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






