# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

##### Custom import
from src.utils import config_load, log, spark_check


def run(spark:SparkSession, config_name='config.yaml' ):
    """  Generate use_session table
    :param spark: sparksession
    :param config: config dict
    :return: None
    """
    ##### Load the yaml config file
    conf           = config_load(config_name)
    maxrecordfile  = conf['sparkconfig']['maxrecordfile']
    check_path     = conf['FilePaths']['check_path']
    dt_min, dt_max = conf['session']['dt_min'], conf['session']['dt_max']
    hour           = conf['session']['hour']

    ##############################################################################################
    log("###### Generate  user_session_id,  is_new_session from user_log #######################")
    userlogDF = spark.read.parquet(  conf['FilePaths']['userlog_path'] )

    userlogDF = userlogDF.sort(F.col('user_id').asc(),  F.col('loggedtimestamp').asc() )
    window    = Window.partitionBy("user_id").orderBy("loggedtimestamp")
    userlogDF = userlogDF.withColumn("last_event", F.lag(F.col("ts"),1).over(window))
    userlogDF = userlogDF.withColumn("diff", ( F.col('ts') - F.col('last_event') ))

    userlogDF.createOrReplaceTempView("userlogDF")
    log(userlogDF.columns)

    query = """ SELECT 
                     user_id, 
                     loggedtimestamp, 
                     sourceIP, 
                     IPHashBucket,                       

                     user_agent, useragenthash, 
                     URL, os, device,
                     loggeddate, hour,   
                     
                     ts,   
                     last_event,                     
                     diff,                     
                     
                     -- user_session_id definition  ---------------------------------------
                     CASE WHEN diff >= (60 * 15) OR last_event IS NULL 
                          THEN CONCAT(user_id,'_',loggedtimestamp) 
                          ELSE null END AS user_session_id 

                     -- new session definition  -------------------------------------------
                    ,CASE WHEN  diff >= (60 * 15) OR last_event IS NULL 
                         THEN 1 ELSE 0 
                     END AS is_new_session
                     
                   FROM  userlogDF                            
            """

    ### Query over a range.
    # query     = query.format(dt_min=dt_min,   dt_max=dt_max, hour= hour)
    sessionDF = spark.sql(query)


    #### Fill user_session_id based on reference value
    win = Window.partitionBy("user_id").orderBy("loggedtimestamp")
    sessionDF = sessionDF.withColumn("user_session_id", F.last(F.col("user_session_id"), ignorenulls=True).over(win))

    sessionDF.write.partitionBy("loggeddate","hour").option("maxRecordsPerFile",
                    maxrecordfile).parquet(conf['FilePaths']['usersession_path'],  mode="overwrite")

    cols = [ 'user_id', 'loggedtimestamp',  'last_event', 'is_new_session',  'user_session_id', 'diff'   ]
    spark_check(sessionDF, path= f"{check_path}/sessionDF", nsample=1100000 ,
                            save=True, verbose=True)

    # sessionDF.repartition(1).write.parquet(conf['Test']['expected_usersession_path'],  mode="overwrite")


    #########################################################################
    log("###### Statistics per user_session :  sessionduration, start, end, ... ###################")
    sessionDF.createOrReplaceTempView("sessionDF")
    sessionStatsDF = spark.sql("""SELECT 
                                     user_session_id,
                                     user_id, 
                                     sourceIP,  
                                     IPHashBucket,
                                     -- user_agent,    
                                     os,
                                     device,
                                     hour,

                                     min(loggedtimestamp) AS starttimestamp, 
                                     max(loggedtimestamp) AS endtimestamp,
                                     max(ts) - min(ts)    AS session_duration, 
                                     
                                     count( distinct URL) AS n_unique_url,
                                     count(sourceIP)      AS n_events
                                     -- loggeddate, hour 


                                     FROM sessionDF 
                                     GROUP BY  os,device,hour,IPHashBucket, sourceIP, user_id,  user_session_id """)

    sessionStatsDF.write.partitionBy("IPHashBucket").option("maxRecordsPerFile",
                         maxrecordfile).parquet(conf['FilePaths']['usersessionstats_path'],  mode="overwrite")

    spark_check(sessionStatsDF, path= f"{check_path}/sessionStatsDF", nsample=100 , save=True, verbose=True)

    # sessionStatsDF.repartition(1).write.parquet(conf['Test']['expected_usersessionstats_path'],  mode="overwrite")



####   dfhisto.repartition(1).write.parquet(conf['Test']['expected_predict_volume_path'],  mode="overwrite")



