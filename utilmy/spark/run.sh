spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
--conf spark.yarn.dist.archives=hdfs:///tmp/pyspark-env.tar.gz#environment \
--master yarn \
--deploy-mode cluster \
script.py