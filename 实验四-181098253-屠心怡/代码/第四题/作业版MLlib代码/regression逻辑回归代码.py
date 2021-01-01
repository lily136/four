from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit
import numpy as np
import pandas as pd
import os
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
import pyspark.ml.classification as cl
import pyspark.ml.feature as ft
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark import SparkContext, SparkConf


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("regression") \
        .getOrCreate()

    train = spark.read.option("header", True).csv("C:/Users/tuxin/Desktop/data_format1 (1)/train2.csv")
    test = spark.read.option("header", True).csv("C:/Users/tuxin/Desktop/data_format1 (1)/test.csv")

    train = train.withColumn('pro', train['pro'].cast('double'))
    train = train.withColumn('age_range', train['age_range'].cast('double'))
    train = train.withColumn('gender', train['gender'].cast('double'))
    train = train.withColumn('total_logs', train['total_logs'].cast('double'))
    train = train.withColumn('unique_item_ids', train['unique_item_ids'].cast('double'))
    train = train.withColumn('categories', train['categories'].cast('double'))
    train = train.withColumn('browse_days', train['browse_days'].cast('double'))
    train = train.withColumn('one_clicks', train['one_clicks'].cast('double'))
    train = train.withColumn('shopping_carts', train['shopping_carts'].cast('double'))
    train = train.withColumn('purchase_times', train['purchase_times'].cast('double'))
    train = train.withColumn('favourite_times', train['favourite_times'].cast('double'))

    test = test.withColumn('pro', test['pro'].cast('double'))
    test = test.withColumn('age_range', test['age_range'].cast('double'))
    test = test.withColumn('gender', test['gender'].cast('double'))
    test = test.withColumn('total_logs', test['total_logs'].cast('double'))
    test = test.withColumn('unique_item_ids', test['unique_item_ids'].cast('double'))
    test = test.withColumn('categories', test['categories'].cast('double'))
    test = test.withColumn('browse_days', test['browse_days'].cast('double'))
    test = test.withColumn('one_clicks', test['one_clicks'].cast('double'))
    test = test.withColumn('shopping_carts', test['shopping_carts'].cast('double'))
    test = test.withColumn('purchase_times', test['purchase_times'].cast('double'))
    test = test.withColumn('favourite_times', test['favourite_times'].cast('double'))

    # 创建单一的列将所有特征整合在一起
    input_col = ['age_range', 'gender', 'total_logs', 'unique_item_ids', 'categories', 'browse_days',
                 'one_clicks', 'shopping_carts', 'purchase_times', 'favourite_times']
    #vecAssembler = VectorAssembler(inputCols=input_col, outputCol="features")
    # 创建一个评估器
    #logistic = cl.LogisticRegression(maxIter=10,
                                     #regParam=0.01,
                                     #featuresCol=vecAssembler.getOutputCol(),
                                     #labelCol='pro')
    #pipeline = Pipeline(stages=[vecAssembler, logistic])
    #birth_train, birth_test = train.randomSplit([0.7, 0.3], seed=123)
    #model = pipeline.fit(train)
    #test_model = model.transform(birth_test)
    #test_model.show()
    vecAssembler = VectorAssembler(inputCols=input_col, outputCol="features")
    stringIndexer = StringIndexer(inputCol="pro", outputCol="label")
    pipeline = Pipeline(stages=[vecAssembler, stringIndexer])
    pipelineFit = pipeline.fit(train)
    dataset = pipelineFit.transform(train)
    dataset = dataset['features', 'label']
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], 123)
    # 模型训练
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    # 模型预测
    #prediction = lrModel.transform(testData)
    #prediction.show()
    #dataset.show()
    vecAssembler = VectorAssembler(inputCols=input_col, outputCol="features")
    stringIndexer = StringIndexer(inputCol="pro", outputCol="label")
    pipeline = Pipeline(stages=[vecAssembler, stringIndexer])
    pipelineFit = pipeline.fit(test)
    dataset2 = pipelineFit.transform(test)
    # 模型训练
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(dataset)
    prediction = lrModel.transform(dataset2)
    prediction.show()
    
    spark.stop()

