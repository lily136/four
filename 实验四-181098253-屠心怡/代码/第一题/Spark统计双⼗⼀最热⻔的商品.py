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
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    counts = lines.map(lambda x:x.split(",")).filter(lambda x:'0'<x[6]<'4' and x[5] == '1111')
    if len(sys.argv) > 2:
        stopList = spark.read.text(sys.argv[3]).rdd.map(lambda r: r[0]).flatMap(lambda x: x.split(' ')).collect()
        counts = counts.filter(lambda x: x not in stopList)       
    counts = counts.map(lambda x:(x[1],1)).reduceByKey(add)
    counts = counts.map(lambda x: (x[1],x[0])).sortByKey(False).map(lambda x: (x[1],x[0]))
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))
    counts.saveAsTextFile("output")
    spark.stop()
