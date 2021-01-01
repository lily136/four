from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("TrainFeature22") \
        .getOrCreate()

    df_train = spark.read.option("header", True).csv("C:/Users/tuxin/Desktop/data_format1 (1)/data_format1/train_format1.csv")
    df_test = spark.read.option("header", True).csv("C:/Users/tuxin/Desktop/data_format1 (1)/data_format1/test_format1.csv")
    user_info = spark.read.option("header", True).csv("C:/Users/tuxin/Desktop/data_format1 (1)/data_format1/user_info_format1.csv")
    log = spark.read.option("header", True).csv("C:/Users/tuxin/Desktop/data_format1 (1)/data_format1/user_log_format1.csv")

    # 处理缺失值
    #df_train.show()
    #df_train.printSchema()
    #user_info = user_info.withColumn('age_range', when(user_info.age_range.isNull(), lit('0')).otherwise(user_info.age_range))
    #user_info = user_info.withColumn('gender', when(user_info.gender.isNull(), lit('2')).otherwise(user_info.gender))
    #user_info.show()

    #total_logs_temp = user_log.groupby("user_id", "seller_id").count()
    #total_logs_temp.show(10) #user_id|seller_id|count|

    #train = user_info.join(user_log, on="user_id", how="left")
    #train.show()
    #user_info = user_info.withColumn("seller_id", "gender")
    #user_info.where("gender = 0").show()
    #user_info.filter("gender = 0").show()
    #user_info.select("gender").show()
    #user_info.drop("gender").show()
    #user_info.withColumnRenamed("gender", "hhh").show() #重命名
    #加新列
    #user_info.withColumn("newCol", lit("0")).show()
    #加新列
    #user_info.withColumn("newCol", when(user_info.gender.isNull(), lit('2')).otherwise(user_info.gender)).show()
    #加新列
    #user_info.withColumn("newCol", user_info.age_range).show()
    '''
    需要建立的特征如下：
    用户的年龄(age_range)
    用户的性别(gender)
    某用户在该商家日志的总条数(total_logs)
    用户浏览的商品的数目，就是浏览了多少个商品(unique_item_ids)
    浏览的商品的种类的数目，就是浏览了多少种商品(categories)
    用户浏览的天数(browse_days)
    用户单击的次数(one_clicks)
    用户添加购物车的次数(shopping_carts)
    用户购买的次数(purchase_times)
    用户收藏的次数(favourite_times)'''
    #age_range,gender特征添加
    df_train = df_train.join(user_info, on="user_id", how="left")
    #其他商家特征



    #df_train.show()
    #total_logs特征添加
    #total_logs_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"]])
    #.count().reset_index()[["user_id", "seller_id", "item_id"]]
    # total_logs_temp = user_log.groupby("user_id", "seller_id").count()
    #user_info.withColumnRenamed("gender", "hhh")
    total_logs_temp = log.groupby("user_id", "seller_id").count()
    #total_logs_temp.show()
    total_logs_temp = total_logs_temp.withColumnRenamed("count", "total_logs")
    total_logs_temp = total_logs_temp.withColumnRenamed("seller_id", "merchant_id")
    #df_train = pd.merge(df_train, total_logs_temp, on=["user_id", "merchant_id"], how="left")
    df_train = df_train.join(total_logs_temp, on=["user_id", "merchant_id"], how="left")

    #unique_item_ids特征添加
    unique_item_ids = log.groupBy("user_id", "seller_id", "item_id").count()
    unique_item_ids = unique_item_ids.groupBy("user_id", "seller_id").count()
    unique_item_ids = unique_item_ids.withColumnRenamed("seller_id", "merchant_id")
    unique_item_ids = unique_item_ids.withColumnRenamed("count", "unique_item_ids")
    df_train = df_train.join(unique_item_ids, on=["user_id", "merchant_id"], how="left")

    #categories特征构建
    categories = log.groupBy("user_id", "seller_id", "cat_id").count()
    categories = categories.groupBy("user_id", "seller_id").count()
    categories = categories.withColumnRenamed("seller_id", "merchant_id")
    categories = categories.withColumnRenamed("count", "categories")
    df_train = df_train.join(categories, on=["user_id", "merchant_id"], how="left")
    df_train = df_train.withColumn("categories", when(df_train.categories.isNull(), lit("0")).otherwise(df_train.categories))

    #browse_days特征构建
    browse_days_temp = log.groupby("user_id", "seller_id", "time_stamp").count()
    browse_days_temp = browse_days_temp.drop("count")
    browse_days_temp1 = browse_days_temp.groupby("user_id", "seller_id").count()
    browse_days_temp1 = browse_days_temp1.withColumnRenamed("seller_id", "merchant_id")
    browse_days_temp1 = browse_days_temp1.withColumnRenamed("count", "browse_days")
    df_train = df_train.join(browse_days_temp1, on=["user_id", "merchant_id"], how="left")

    #one_clicks、shopping_carts、purchase_times、favourite_times特征构建
    one_clicks_temp = log.filter("action_type == 0")
    one_clicks_temp = one_clicks_temp.groupby("user_id", "seller_id").count()
    one_clicks_temp = one_clicks_temp.withColumnRenamed("seller_id", "merchant_id")
    one_clicks_temp = one_clicks_temp.withColumnRenamed("count", "one_clicks")

    shopping_carts_temp = log.filter("action_type == 1")
    shopping_carts_temp = shopping_carts_temp.groupby("user_id", "seller_id").count()
    shopping_carts_temp = shopping_carts_temp.withColumnRenamed("seller_id", "merchant_id")
    shopping_carts_temp = shopping_carts_temp.withColumnRenamed("count", "shopping_carts")

    purchase_times_temp = log.filter("action_type == 2")
    purchase_times_temp = purchase_times_temp.groupby("user_id", "seller_id").count()
    purchase_times_temp = purchase_times_temp.withColumnRenamed("seller_id", "merchant_id")
    purchase_times_temp = purchase_times_temp.withColumnRenamed("count", "purchase_times")

    favourite_times_temp = log.filter("action_type == 3")
    favourite_times_temp = favourite_times_temp.groupby("user_id", "seller_id").count()
    favourite_times_temp = favourite_times_temp.withColumnRenamed("seller_id", "merchant_id")
    favourite_times_temp = favourite_times_temp.withColumnRenamed("count", "favourite_times")

    df_train = df_train.join(one_clicks_temp, on=["user_id", "merchant_id"], how="left")
    df_train = df_train.withColumn("one_clicks", when(df_train.one_clicks.isNull(), lit('0')).otherwise(df_train.one_clicks))

    df_train = df_train.join(shopping_carts_temp, on=["user_id", "merchant_id"], how="left")
    df_train = df_train.withColumn("shopping_carts",
                                   when(df_train.shopping_carts.isNull(), lit('0')).otherwise(df_train.shopping_carts))

    df_train = df_train.join(purchase_times_temp, on=["user_id", "merchant_id"], how="left")
    df_train = df_train.withColumn("purchase_times",
                             when(df_train.purchase_times.isNull(), lit('0')).otherwise(df_train.purchase_times))

    df_train = df_train.join(favourite_times_temp, on=["user_id", "merchant_id"], how="left")
    df_train = df_train.withColumn("favourite_times",
                             when(df_train.favourite_times.isNull(), lit('0')).otherwise(df_train.favourite_times))

    df_train.repartition(1).write.csv("train_add_action_type", encoding="utf-8", header=True)


    spark.stop()