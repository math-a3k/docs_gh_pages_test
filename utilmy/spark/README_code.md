
##### Code Structure
```
### Main Source

    src/tables/table_user_log.py               : Raw Log processing

    src/tables/table_user_session_log.py       : User session Log processing

    src/tables/table_user_session_stats.py     : User session Stats generation


    src/tables/table_predict_volume.py         : Agg. volume load prediction

    src/tables/table_predict_session_length.py : session length  prediction.

    src/tables/table_predict_url.py            : nb unique url per ip prediction.



    src/util_models.py                         : Utilities for Spark model training/prediction
    src/util.py                                : Utilities for logging, spark DF check, ...


    src/functions/GetFamiliesUserAgent.py      : Function to extract User Agent infos.




#### Data
    data/2015_07_22_mktplace_shop_web_log_sample.log.gz  : Original Full size file

    data/test/                :  Small input file

    data/test_unit_expected/     Parquet files of reference data for Test comparisons.


    output/full/     : Output of full size
    output/test/     : Output of small size
    output/test_un/  : Output of unit test resuls




#### Test Code
    tests/   Pytest code

    .github/pytest.yml : CI pytest

    .github/main_full.yml : CI full run




#### Configuration:
    config/config.yaml             : Full size running
    config/config_test.yaml        : Small size running, Integration test purpose
    config/config_test_unit.yaml   : Small size running, Unit Test configuration

    setup.py  : Python config for spark-submit
    pylinrc   : Code Checker



#### Script :
    sparkrunscript.sh : Spark-Submit to run main.py in StandAlone Cluster
    hadoopVersion.py  : Check Hadoop
    server_start.sh   : Standalone Spark Master-Node starts






#### Install :
     Dockerfile   : Install of Hadoop, Spark 2.4.3, Python, Pyspark libraries
     docker-comppose.yaml



```
