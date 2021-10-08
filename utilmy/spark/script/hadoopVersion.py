import random
from pyspark import SparkContext

sc = SparkContext(appName="HadoopVersion")

print("Hadoop Version:" + sc._gateway.jvm.org.apache.hadoop.util.VersionInfo.getVersion())
