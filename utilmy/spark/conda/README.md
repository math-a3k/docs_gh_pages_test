# Creating the env
NOTE: OS of extraction needs to be the same as the target OS

`conda create -y -n pyspark-env python=3.7 tensorflow` 

May give a file-system error. If so, simply run again. This is due to conda not being initialized

`conda activate pyspark-env`

`conda install -y -n pyspark-env -c conda-forge conda-pack`

# Packing the env

`conda pack -f -n pyspark-env -o pyspark-env.tar.gz`


# Launching the env

The env `.tar.gz` needs to be on the hdfs:

`hdfs dfs -put -f pyspark-env.tar.gz /tmp/`

`hdfs dfs -chmod 0664 /tmp/pyspark-env.tar.gz`

then:

`conda deactivate`

`sh run.sh`