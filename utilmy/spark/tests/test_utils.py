# -*- coding: utf-8 -*-
from os import path

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from src.utils import config_load, spark_check
from tests.test_common import (assert_equal_spark_df_sorted)




@pytest.mark.usefixtures("spark_session", "config")
def test_spark_check(spark_session: SparkSession, config: dict):
    """
    This unit test will verify spark_check
    @param spark_session:
    @param config:
    @return:
    """

    # generate static dataframe
    test_df = spark_session.createDataFrame(
            [
                ("1.187.217.224", 1437588028, 100),
                ("1.187.236.169", 1437588030, 200),
                ("1.38.22.38", 1437500029, 10),
                ("1.38.21.44", 1437588029, 500)
            ],
        ("ipaddr", "ts", "duration"))

    sorted_test_df = test_df.sort(test_df.columns)


    ###### write csv & read it back and compare dataframes (all records)
    spark_check(test_df, config['Test']['spark_check_temp_full_output'])
    # read pd df and convert to spark df
    stored_full_pd_df = pd.read_csv(config['Test']['spark_check_temp_full_output'] + '/table.csv',
                                    sep='\t', engine='python')
    stored_full_df = spark_session.createDataFrame(stored_full_pd_df).select("ipaddr", "ts", "duration")
    sorted_full_df = stored_full_df.sort(test_df.columns)

    # assert full df comparison
    assert_equal_spark_df_sorted(sorted_test_df, sorted_full_df, "spark_check_full_output")


    ######  write csv & read it back and compare dataframes (top 2 records sorted df)
    spark_check(sorted_test_df, config['Test']['spark_check_temp_partial_output'], 2)
    stored_partial_pd_df = pd.read_csv(config['Test']['spark_check_temp_partial_output'] + '/table.csv',
                                    sep='\t', engine='python')
    stored_partial_df = spark_session.createDataFrame(stored_partial_pd_df).select("ipaddr", "ts", "duration")
    stored_partial_df = stored_partial_df.sort(test_df.columns)
    # assert partial df comparison
    assert_equal_spark_df_sorted(sorted_test_df.limit(2), stored_partial_df, "spark_check_partial_output")


    ###### save = false validation
    spark_check(test_df, config['Test']['spark_check_temp_empty_output'], save=False)
    file_exists = path.exists(config['Test']['spark_check_temp_empty_output'] + '/table.csv')
    assert file_exists == False, "File {} should not be present for save=False"\
        .format(config['Test']['spark_check_temp_empty_output'] + '/table.csv')


