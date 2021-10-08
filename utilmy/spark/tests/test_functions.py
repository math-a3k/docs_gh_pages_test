import pytest
from pyspark.sql import SparkSession, functions as F, types as T

from src.functions.GetFamiliesFromUserAgent import getall_families_from_useragent


@pytest.mark.usefixtures("spark_session")
def test_getall_families_from_useragent(spark_session: SparkSession):
    """
    This function will verify getall_families_from_useragent function
    @param spark_session:
    """
    # generate static dataframe
    test_df = spark_session.createDataFrame(
            [
                ("1.187.217.224", 1437588028, 100,
                 "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                 "Chrome/43.0.2357.130 Safari/537.36", "Windows-Other-Chrome"),
                ("1.187.236.169", 1437588030, 200, "null", "Other-Other-Other"),
                ("1.38.22.38", 1437500029, 10, None, ' - - '),
                ("1.38.21.44", 1437588029, 500, "Mozilla/5.0 (Windows NT 6.1; rv:39.0) Gecko/20100101 Firefox/39.0",
                 "Windows-Other-Firefox"),
                ("1.111.20.138", 1437500000, 20, "-", "Other-Other-Other")
            ],
        ("ipaddr", "ts", "duration", "user_agent", "expected_output"))

    udfAll3Family = F.udf(getall_families_from_useragent, T.StringType())


    ######### Act
    output_df = test_df.withColumn("All3UserAgent", udfAll3Family(test_df.user_agent))


    ########## Assert: All3UserAgent is same as expected_output column for below cases.
    # Data for below cases are added along with expected output in test_df
    # for None --> it should be ' - - '
    # for "null" string --> it should be 'Other-Other-Other'
    # example of firefox and chrome each
    # for - --> it should be Other-Other-Other
    output_df = output_df.withColumn("is_output_matched",
                                     F.when(F.col("All3UserAgent") != F.col("expected_output"), F.lit("false"))
                                     .otherwise(F.lit("true")))
    ua_mismatch_df = output_df.filter(F.col("is_output_matched") != F.lit("true"))
    mismatch_records = ua_mismatch_df.count()
    print("Mismatch records: {}".format(mismatch_records))
    if mismatch_records != 0:
        print("All3UserAgent is not same as expected_output column - total bad records: {}, top 3 records..."
              .format(mismatch_records))
        ua_mismatch_df.show(3, False)

    assert mismatch_records == 0, "All3UserAgent column must be same as expected output."


