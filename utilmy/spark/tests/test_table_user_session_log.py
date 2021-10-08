# -*- coding: utf-8 -*-
### pytest -v --cov=src/  --html=output/tests/report.html --self-contained-html  tests/test_table_user_session_log.py
import pytest
from pyspark.sql import SparkSession

from src.tables.table_user_session_log import run as table_user_session_log_run
from tests.test_common import (TEST_CONFIG_FILE, assert_equal_spark_df, assert_equal_spark_df_schema)

from src.utils import log


# Arrange
@pytest.fixture(scope="module")
@pytest.mark.usefixtures("spark_session")
def test_table_user_session_log_run(spark_session: SparkSession):
    """
    Test table_user_session's run method
    @param spark_session:
    """
    log("Execute table_user_session_log_run")
    table_user_session_log_run(spark_session, TEST_CONFIG_FILE)


@pytest.mark.usefixtures("spark_session", "config", "test_table_user_session_log_run")
def test_table_user_session_log(spark_session: SparkSession, config: dict):
    """
    Test user session log output
    @param spark_session:
    @param config:
    """
    expected_df = spark_session.read.parquet(config['Test']['expected_usersession_path'])

    # read current output
    actual_df = spark_session.read.parquet(config['FilePaths']['usersession_path'])


    #### Compare schemas of two dataframes (column names & data types
    expected_schema = [('user_id', 'string'),
                       ('loggedtimestamp', 'string'),
                       ('sourceIP', 'string'),
                       ('IPHashBucket', 'int'),

                       ('user_agent', 'string'),
                       ('useragenthash', 'int'),

                       ('URL', 'string'),
                       ('os', 'string'),
                       ('device', 'string'),
                       ('loggeddate', 'date'),
                       ('hour', 'int'),
                       ('ts', 'bigint'),
                       ('last_event', 'bigint'),
                       ('diff', 'bigint'),
                       ('user_session_id', 'string'),
                       ('is_new_session', 'int'),

                    ]
    assert_equal_spark_df_schema(expected_schema, actual_df.dtypes, "UserSessionTable")


    #### Values,  compare actual df against expected df
    expected_fields_ordered = ['user_id',
                               'loggedtimestamp',
                               'sourceIP',
                               'IPHashBucket',
                               'user_agent',
                               'useragenthash',
                               'URL',
                               'os', 'device',
                               'loggeddate', 'hour',
                               'ts',
                               'last_event',
                               'diff',
                               'user_session_id',
                               'is_new_session'
                               ]
    assert_equal_spark_df(expected_df.select(expected_fields_ordered),
                          actual_df.select(expected_fields_ordered), "UserSessionTable")



@pytest.mark.usefixtures("spark_session", "config", "test_table_user_session_log_run")
def test_table_usersession_log_stats(spark_session: SparkSession, config: dict):
    """
    test user session log stats
    @param spark_session:
    @param config:
    """
    expected_df = spark_session.read.parquet(config['Test']['expected_usersessionstats_path'])

    # read current output
    actual_df = spark_session.read.parquet(config['FilePaths']['usersessionstats_path'])

    #### Compare schemas of two dataframes (column names & data types
    expected_schema = [('user_session_id', 'string'),
                       ('user_id', 'string'),
                       ('sourceIP', 'string'),
                       ('IPHashBucket', 'int'),

                       ('os', 'string'),
                       ('device', 'string'),
                       ('hour', 'int'),

                       ('starttimestamp', 'string'),
                       ('endtimestamp', 'string'),
                       ('session_duration', 'bigint'),

                       ('n_unique_url', 'bigint'),
                       ('n_events', 'bigint')]
    assert_equal_spark_df_schema(expected_schema, actual_df.dtypes, "UserSessionLogStatsTable")


    #### Values,  compare actual df against expected df
    # Expected fields in single non partitioned file will be in different order - so need to order it explicitly
    expected_fields_ordered = ['user_session_id', 'user_id', 'sourceIP',
                               'IPHashBucket',
                               'os', 'device', 'hour',

                               'starttimestamp', 'endtimestamp',
                               'session_duration',

                               'n_unique_url',
                               'n_events',
                            ]

    assert_equal_spark_df(expected_df.select(expected_fields_ordered),
                          actual_df.select(expected_fields_ordered), "UserSessionLogStatsTable")




