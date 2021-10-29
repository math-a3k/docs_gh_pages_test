from pyspark import SparkConf
from pyspark import SparkContext

conf = SparkConf()
conf.setAppName('pyspark-test')
sc = SparkContext(conf=conf)

import numpy
import tensorflow as tf
print("Numpy version: ", sc.parallelize(range(1,10)).map(lambda x : numpy.__version__).collect())
print("Tensorflow: ", tf)