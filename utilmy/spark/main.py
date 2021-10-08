# -*- coding: utf-8 -*-
import argparse
import importlib

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from src.utils import config_load, log


def spark_init(config:dict)->SparkSession:
    """
    Args:
        config: config dict
    Returns: SparkSession
    """
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






