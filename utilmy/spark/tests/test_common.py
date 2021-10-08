from pyspark.sql.dataframe import DataFrame

from src.utils import log


TEST_CONFIG_FILE = 'config/config_test_unit.yaml'


def assert_equal_spark_df_sorted(expected_sorted_df: DataFrame, actual_sorted_df: DataFrame, df_name: str):
    """ Compares two spark dataframe based on values assuming its already sorted
    @param expected_sorted_df: Expected spark dataframe sorted
    @param actual_sorted_df: Actual spark output dataframe sorted
    @param df_name: Dataframe name for logging purpose
    """
    missing_rows_df = expected_sorted_df.subtract(actual_sorted_df)
    additional_rows_df = actual_sorted_df.subtract(expected_sorted_df)
    missing_rows = missing_rows_df.count()

    additional_rows = additional_rows_df.count()
    if missing_rows != 0 or additional_rows != 0:
        log("Missing rows: {}, additional rows: {}".format(missing_rows, additional_rows))
        log("Expected rows for {} are {}".format(df_name, expected_sorted_df.count()))
        log(expected_sorted_df.show(5, False))
        log("Actual rows for {} are {}".format(df_name, actual_sorted_df.count()))
        log(actual_sorted_df.show(5, False))

    assert missing_rows == 0, "{}: Few expected rows are missing".format(df_name)
    assert additional_rows == 0, "{}: Actual output contains few additional rows".format(df_name)


def assert_equal_spark_df(expected_df: DataFrame, actual_df: DataFrame, df_name: str):
    """ Compares two spark dataframe based on values
    @param expected_df: Expected spark dataframe
    @param actual_df: Actual spark output dataframe
    @param df_name: Dataframe name for logging purpose
    """
    expected_cols = expected_df.columns
    expected_sorted_df = expected_df.sort(expected_cols)
    actual_sorted_df = actual_df.sort(expected_cols)
    assert_equal_spark_df_sorted(expected_sorted_df, actual_sorted_df, df_name)


def assert_equal_spark_df_schema(expected_schema: [tuple], actual_schema: [tuple], df_name: str):
    """ Compares schemas (as list of tuples (name, datatype)
    @param expected_schema: Expected spark dataframe schema as list of tuple (name, data type)
    @param actual_schema: Actual output spark dataframe schema as list of tuple (name, data type)
    @param df_name: Dataframe name for logging purpose
    """
    expected_schema.sort()
    actual_schema.sort()
    msg = f"Actual schema is different from expected schema: {df_name}, schema :{actual_schema}, schema reference: {expected_schema} "
    assert expected_schema == actual_schema, msg
