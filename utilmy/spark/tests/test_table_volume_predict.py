# -*- coding: utf-8 -*-
###    pytest -v --cov=src/  --html=output/tests/report.html --self-contained-html  tests/test_table_volume_predict.py
import pytest
from pyspark.sql import SparkSession

from src.tables.table_predict_volume import preprocess
from tests.test_common import (assert_equal_spark_df, assert_equal_spark_df_schema)


@pytest.mark.usefixtures("spark_session", "config")
def test_preprocess(spark_session: SparkSession, config: dict):
    """
    This function will verify functionality of preprocess of table_predict_volume
    @param spark_session:
    @param config:
    """
    ####### Act
    actual_df = preprocess(spark_session, config)

    # actual_df.repartition(1).write.parquet(config['Test']['expected_predict_volume_path'],  mode="overwrite")


    # read expected output
    expected_pred_vol_df = spark_session.read.parquet(config['Test']['expected_predict_volume_path'])



    #### Compare schemas of two dataframes (column names & data types)
    expected_schema = [('loggeddate', 'string'),
                       ('ds', 'string'),
                       ('y', 'bigint'),
                      ]
    assert_equal_spark_df_schema(expected_schema, actual_df.dtypes, "PredVolumePreprocessor")


    #### Values,  compare actual df against expected df
    expected_fields_ordered = ['loggeddate', 'ds', 'y']
    assert_equal_spark_df(expected_pred_vol_df.select(expected_fields_ordered),
                          actual_df.select(expected_fields_ordered), "PredVolumePreprocessor")




