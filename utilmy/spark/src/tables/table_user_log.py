# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
import pyspark
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

##### Custom import
from src.functions.GetFamiliesFromUserAgent import getall_families_from_useragent
from src.utils import config_load, log, spark_check



def run(spark:SparkSession, config_name:str):
    """ Generate Structured Log table on disk
    :param spark:
    :param config_name:
    :return: Structured Log Spark Dataframe
    """
    #### Load the yaml config file
    conf   = config_load(config_name)
    maxrecordfile = conf['sparkconfig']['maxrecordfile']
    check_path = conf['FilePaths']['check_path']


    #### UDF to get browser, device and OS info from user agent field
    udfAll3Family = F.udf(getall_families_from_useragent, T.StringType())


    #### Defining the schema for log file
    logSchema = T.StructType().add("loggedtimestamp", "string")\
                            .add("elb", "string")\
                            .add("client_port", "string")\
                            .add("backend_port", "string")\
                            .add("request_processing_time", "string")\
                            .add("backend_processing_time", "string")\
                            .add("response_processing_time", "string")\
                            .add("elb_status_code", "string")\
                            .add("backend_status_code", "string")\
                            .add("received_bytes", "string") \
                            .add("sent_bytes", "string")\
                            .add("request", "string")\
                            .add("user_agent", "string") \
                            .add("ssl_cipher", "string").add("ssl_protocol", "string")


    log("######### Read logs into a dataframe  ")
    log(conf['FilePaths']['rawlog_path'])
    userlogDF = spark.read \
                    .format("com.databricks.spark.csv")   \
                    .option("header", "false") \
                    .option("delimiter"," ") \
                    .schema(logSchema) \
                    .load(conf['FilePaths']['rawlog_path'])

    #Get the UDF defined for user agent into  a dataframe
    userlogDF  = userlogDF.withColumn("All3UserAgent",udfAll3Family(userlogDF.user_agent))

    #Get Ip and port separately
    All3UserAgent  = F.split(userlogDF['All3UserAgent'], '-')
    sourceip_port  = F.split(userlogDF['client_port'], ':')
    #backendip_port = split(userlogDF['backend_port'], ':')
    request        = F.split(userlogDF['request'], ' ')


    log("######### Select all required fields to store into userlog ")
    userlogDF = userlogDF.select("loggedtimestamp",
                                F.to_date("loggedtimestamp").alias("loggeddate"),
                                F.hour("loggedtimestamp").alias("hour"),      ## Local time hour
                                F.minute("loggedtimestamp").alias("minute"),  ## Local Time
                                "elb",
                                sourceip_port.getItem(0).alias("sourceIP"),
                                #sourceip_port.getItem(1).alias("sourcePort"),
                                #backendip_port.getItem(0).alias("backendIP"),
                                #backendip_port.getItem(1).alias("backendPort"),
                                "request_processing_time",
                                "backend_processing_time",
                                "elb_status_code",
                                "backend_status_code",
                                "received_bytes",
                                "sent_bytes",
                                request.getItem(1).alias("URL"),
                                "user_agent",
                                All3UserAgent.getItem(0).alias("os"),
                                All3UserAgent.getItem(1).alias("device"),
                                All3UserAgent.getItem(2).alias("browser"),
                )


    log("######### User_id definition ########################################")
    # userlogDF = userlogDF.withColumn("user_id", concat( 'sourceIP',lit('-'),'device', lit('-'), 'os') )
    userlogDF = userlogDF.withColumn("user_id", F.concat( 'sourceIP',F.lit('')) )


    log("######### Additional features   #####################################") # Format Timestamp, user_agent
    userlogDF = userlogDF.withColumn("sourceIPPrefix", F.regexp_replace('sourceIP', '\\.\\d+$', ''))
    userlogDF = userlogDF.withColumn("useragenthash", F.hash("user_agent") )
    ## Convert to UTC Unix timestamp, JST
    userlogDF = userlogDF.withColumn("ts", F.unix_timestamp( F.from_utc_timestamp("loggedtimestamp", 'JST')) )


    log("######### Add Bucketing   ##########################################")
    userlogDF = userlogDF.withColumn("IPHashBucket"  , F.expr("mod(abs(hash(sourceIP)), 100)"))
    log("Size log", userlogDF.select('sourceIP').count() )

    log("######### Writing on disk with partition of ipHashBucket ")
    userlogDF.write.partitionBy("loggeddate").option("maxRecordsPerFile",
              maxrecordfile).parquet(conf['FilePaths']['userlog_path'],  mode="overwrite")

    spark_check(userlogDF, path= f"{check_path}/userlogDf", nsample=10 , save=True, verbose=True, returnval=False)
    # userlogDF.repartition(1).write.parquet(conf['Test']['expected_userlog_path'],  mode="overwrite")






def create_userid(userlogDF:pyspark.sql.DataFrame):
    userlogDF = userlogDF.withColumn("user_id", F.concat( 'sourceIP',F.lit('')) )
    return userlogDF







