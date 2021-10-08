# -*- coding: utf-8 -*-
### pytest --cov=src/  --html=output/tests/report.html --self-contained-html  tests/test_table_user_log.py
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.tables.table_user_log import run as table_user_run
from tests.test_common import (TEST_CONFIG_FILE, assert_equal_spark_df, assert_equal_spark_df_schema)


@pytest.mark.usefixtures("spark_session", "config")
def test_table_user_log_run(spark_session: SparkSession, config: dict):
    """ Unit test for src.tables.table_user_log
    @param spark_session: Spark Session
    @param config:  Configuration
    """
    table_user_run(spark_session, TEST_CONFIG_FILE)

    # read expected output
    expected_df = spark_session.read.parquet(config['Test']['expected_userlog_path'])

    # read current output
    actual_df = spark_session.read.parquet(config['FilePaths']['userlog_path'])


    #### Compare schemas of two dataframes (column names & data types
    expected_schema = [('loggedtimestamp', 'string'),
                       ('loggeddate', 'date'),
                       ('hour', 'int'),
                       ('minute', 'int'),
                       ('elb', 'string'),
                       ('sourceIP', 'string'),
                       ('request_processing_time', 'string'),
                       ('backend_processing_time', 'string'),
                       ('elb_status_code', 'string'),
                       ('backend_status_code', 'string'),
                       ('received_bytes', 'string'),
                       ('sent_bytes', 'string'),
                       ('URL', 'string'),
                       ('user_agent', 'string'),
                       ('os', 'string'),
                       ('device', 'string'),
                       ('browser', 'string'),
                       ('user_id', 'string'),
                       ('sourceIPPrefix', 'string'),
                       ('useragenthash', 'int'),
                       ('ts', 'bigint'),
                       ('IPHashBucket', 'int')]

    assert_equal_spark_df_schema(expected_schema, actual_df.dtypes, "UserLogTable")



    #### Column Structure : user_id formula:  Generic version
    df = actual_df.withColumn("expected_user_id", F.concat('sourceIP', F.lit('')))
    df = df.withColumn("expected_sourceIPPrefix", F.regexp_replace('sourceIP', '\\.\\d+$', ''))
    df = df.withColumn("expected_ts", F.unix_timestamp(F.from_utc_timestamp("loggedtimestamp", 'JST')))
    df = df.withColumn("formula_matched", F.when(((df.expected_user_id == df.user_id) &
                        (df.expected_sourceIPPrefix == df.sourceIPPrefix) &
                        (df.expected_ts == df.ts)), F.lit("true")).otherwise(F.lit("false")))

    formula_matched = df.filter(df.formula_matched == F.lit("false")).count()
    assert formula_matched == 0, "Formula columns in user log output dataframe are not matching.."


    #### Values,  compare actual df against expected df
    expected_fields_ordered = ['loggedtimestamp', 'hour', 'minute', 'elb', 'sourceIP', 'request_processing_time',
                               'backend_processing_time', 'elb_status_code', 'backend_status_code', 'received_bytes',
                               'sent_bytes', 'URL', 'user_agent', 'os', 'device', 'browser', 'user_id',
                               'sourceIPPrefix', 'useragenthash', 'ts', 'IPHashBucket', 'loggeddate']

    assert_equal_spark_df(expected_df.select(expected_fields_ordered), actual_df, "UserLogTable")



