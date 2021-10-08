# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession, functions as F

##### Custom import
from src.utils import config_load, log, spark_check


def run(spark:SparkSession, config_name: str='config.yaml'):
    """
    Args:
        spark: sparkSession
        config_name:
    Returns:
    """
    #Load the yaml config file
    conf = config_load(config_name)
    maxrecordfile = conf['sparkconfig']['maxrecordfile']
    check_path = conf['FilePaths']['check_path']

    ############################################################
    log("######  Load usersessionstats table")
    user_session_statsDF = spark.read.parquet(  conf['FilePaths']['usersessionstats_path'] )



    ############################################################################
    log("###### 2) Get the avg, min and max session duration whole users.")
    user_session_stats_aggtotal = user_session_statsDF.agg(F.min(F.col("session_duration")).alias('min_session_duration'),
                                                           F.max(F.col("session_duration")).alias('max_session_duration'),
                                                           F.avg(F.col("session_duration")).alias('avg_session_duration'),
                                                           F.count(F.col("session_duration")).alias('n_sessions')
                                                           )

    log( user_session_stats_aggtotal.show() )
    user_session_stats_aggtotal.write.parquet(conf['FilePaths']['usersessionstats_aggtotal'],  mode="overwrite")




    ############################################################################
    log("###### 3) Get Unique URL count per session ")
    cols = [ 'user_session_id', 'n_unique_url',  'n_events',  'session_duration' ]
    user_session_stats2 = user_session_statsDF[cols].sort(F.col('n_unique_url').desc())
    log( user_session_stats2.show(15) )
    spark_check(user_session_stats2, path= f"{check_path}/user_session_stats_url_count", nsample=1000 , save=True)
    #### No need to save, already saved.


    ############################################################################
    log("###### 4) Most Engaged : users with longest session duration")
    user_session_stats_agg = (user_session_statsDF
                            .groupBy('user_id')
                            .agg( F.min(F.col("session_duration")).alias('min_session_duration'),
                                  F.max(F.col("session_duration")).alias('max_session_duration'),
                                  F.avg(F.col("session_duration")).alias('avg_session_duration')
                                ))

    #### Most Engaged definition
    coli = 'max_session_duration'  # 'avg_session_duration'
    user_session_stats_agg = user_session_stats_agg.orderBy(coli, ascending=False)
    log( 'Top 10 most engaged users', user_session_stats_agg.show(10) )

    user_session_stats_agg = user_session_stats_agg.withColumn("useridHashBucket"  ,
                                                               F.expr("mod(abs(hash(user_id)), 100)"))
    user_session_stats_agg.write.partitionBy("useridHashBucket").option("maxRecordsPerFile",
                           maxrecordfile).parquet(conf['FilePaths']['usersessionstats_per_ip'],  mode="overwrite")


    spark_check(user_session_stats_agg, path= f"{check_path}/user_session_stats_agg", nsample=10 , save=True)






