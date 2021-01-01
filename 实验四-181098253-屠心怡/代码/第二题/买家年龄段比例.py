from operator import add
from __future__ import print_function
from pyspark.sql import SparkSession
import sys
import pandas as pd
from operator import add

if __name__ == "__main__":
     if len(sys.argv) != 2:
         print("Usage: wordcount ",sys.stderr)
         exit(-1)

    spark = SparkSession\
        .builder\
        .appName("test2")\
        .getOrCreate()
    info = spark.read.option("header",True).csv("user_info_format1.csv")
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    counts = lines.map(lambda x:x.split(",")).filter(lambda x: x[6] == '2')
    counts = counts.drop('item_id','cat_id','seller_id','brand_id','time_stamp','action_type' )
    counts = counts.dropDuplicates()
    counts = counts.join(info,on=["user_id"], how="left")
    if len(sys.argv) > 2:
        stopList = spark.read.text(sys.argv[3]).rdd.map(lambda r: r[0]).flatMap(lambda x: x.split(' ')).collect()
        counts = counts.filter(lambda x: x[0] not in stopList)       
    counts = counts.map(lambda x:(x[1],1)).reduceByKey(add)
    counts = counts.map(lambda x: (x[1],x[0])).sortByKey(False).map(lambda x: (x[1],x[0]))
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))
    counts.saveAsTextFile("output")
    spark.stop()
