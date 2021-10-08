# -*- coding: utf-8 -*-
###  pytest -v --cov=src/  --html=output/tests/report.html --self-contained-html  tests/test_table_user_session_stats.py
import pytest
from pyspark.sql import SparkSession

from src.tables.table_user_session_stats import run as table_user_session_stats_run
from tests.test_common import (TEST_CONFIG_FILE, assert_equal_spark_df, assert_equal_spark_df_schema)



# Arrange
@pytest.fixture(scope="module")
@pytest.mark.usefixtures("spark_session")
def test_table_user_session_stats_run(spark_session: SparkSession):
    """
    This will run table_user_session_stats's run method to verify it
    @param spark_session:
    """
    print("Going to execute table_user_session_stats_run")
    table_user_session_stats_run(spark_session, TEST_CONFIG_FILE)





@pytest.mark.usefixtures("spark_session", "config", "test_table_user_session_stats_run")
def test_table_user_session_stats(spark_session: SparkSession, config: dict):
    """
    This will verify user session stats output
    @param spark_session:
    @param config:
    """
    expected_df = spark_session.read.parquet(config['Test']['expected_usersessionstats_aggtotal'])

    # read current output
    actual_df = spark_session.read.parquet(config['FilePaths']['usersessionstats_aggtotal'])

    #### Compare schemas of two dataframes (column names & data types
    expected_schema = [('min_session_duration', 'bigint'),
                       ('max_session_duration', 'bigint'),
                       ('avg_session_duration', 'double'),
                       ('n_sessions', 'bigint')
                      ]

    assert_equal_spark_df_schema(expected_schema, actual_df.dtypes, "UserSessionStats_AggTotal")

    #### Values,  compare actual df against expected df
    assert_equal_spark_df(expected_df, actual_df, "UserSessionStats_AggTotal")





@pytest.mark.usefixtures("spark_session", "config", "test_table_user_session_stats_run")
def test_table_user_session_stats_ip(spark_session: SparkSession, config: dict):
    """
    This will verify user session stats per ip output
    @param spark_session:
    @param config:
    """
    expected_df = spark_session.read.parquet(config['Test']['expected_usersessionstats_per_ip'])

    # read current output
    actual_df = spark_session.read.parquet(config['FilePaths']['usersessionstats_per_ip'])

    #### Compare schemas of two dataframes (column names & data types
    expected_schema = [('user_id', 'string'),
                       ('min_session_duration', 'bigint'),
                       ('max_session_duration', 'bigint'),
                       ('avg_session_duration', 'double'),
                       ('useridHashBucket', 'int')]

    assert_equal_spark_df_schema(expected_schema, actual_df.dtypes, "UserSessionStats_IP")

    #### Values,  compare actual df against expected df
    # Expected fields in single non partitioned file will be in different order - so need to order it explicitly
    expected_fields_ordered = ['user_id',
                               'min_session_duration',
                               'max_session_duration',
                               'avg_session_duration',
                               'useridHashBucket']

    assert_equal_spark_df(expected_df.select(expected_fields_ordered),
                          actual_df, "UserSessionStats_IP")





