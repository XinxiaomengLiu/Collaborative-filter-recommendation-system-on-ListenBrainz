"""
Created on Sat Apr 17 23:26:21 2023

@author: xinxiaomengliu
"""
# popularity baseline model

import os

# import pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, collect_list, desc, row_number, udf, col
from pyspark.sql.types import ArrayType, DoubleType, IntegerType
from pyspark.sql.window import Window


# use RankingRvaluator() to evaluate the model (e.g. MAP, NDCG)
from pyspark.ml.evaluation import RankingEvaluator


def main(spark, userID):
    print('training set loading')
    train_full = spark.read.csv(f'hdfs:/user/{userID}/train.csv', schema='user_id INT, recording_msid INT, count_value INT')
    print('validation set loading')
    val_full = spark.read.csv(f'hdfs:/user/{userID}/val.csv', schema='user_id INT, recording_msid INT, count_value INT')
    print('test set loading')
    test = spark.read.csv(f'hdfs:/user/{userID}/test.csv', schema='user_id INT, recording_msid INT, count_value INT')

    # select the top 100 recording_msid
    top100_msid = (train_full
                   # Sum up the 'count' values for each unique 'recording_msid' in the dataset
                    # gives the total count or popularity score for each recording_msid, representing how often it has been accessed or viewed overall
                     .groupBy('recording_msid')
                     .sum('count_value')
                     .withColumnRenamed('sum(count_value)', 'popularity_score')
                     # Order the recording_msid values from the most popular to the least popular
                     .orderBy('popularity_score', ascending = False)
                     .limit(100)
                     .select('recording_msid')
                     .agg(collect_list('recording_msid'))
                     .withColumnRenamed('collect_list(recording_msid)', 'popular100')
                     .withColumn('popular100', col('popular100').cast(ArrayType(DoubleType())))
    )

    # for debug
    top100_msid.printSchema()
    top100_msid.show()
    
    
    # generate the top recording_msid for each user_id in the validation set

    # define the window to partition by user_id and sort by count (descending)
    window = Window.partitionBy('user_id').orderBy(desc('count_value'))

    # assign row_number to each recording_msid for each user
    user_msid = val_full.select('*', row_number().over(window).alias('row_number'))

    # top 100 recording_msid for each user
    user_top100_msid = user_msid.filter(user_msid.row_number <= 100).select('user_id', 'recording_msid')

    # convert the 100 recording_msid to a list
    user_top100_msid = user_top100_msid.groupBy('user_id').agg(collect_list('recording_msid').alias('popular100_each_user'))
    user_top100_msid = user_top100_msid.withColumn('popular100_each_user', col('popular100_each_user').cast(ArrayType(DoubleType())))
    
    # for debug
    user_top100_msid.printSchema()
    user_top100_msid.show(5)

    # join the dataset
    joined_val = user_top100_msid.join(top100_msid)   
    
    # for debug
    joined_val.printSchema()
    joined_val.show(5)

    map_evaluator = RankingEvaluator(predictionCol='popular100', labelCol='popular100_each_user', metricName='meanAveragePrecisionAtK', k=100)
    val_map = map_evaluator.evaluate(joined_val)
    print("Validation Set Performence with MAP: ", val_map)
    ndcg_evaluator = RankingEvaluator(predictionCol='popular100', labelCol='popular100_each_user', metricName='ndcgAtK', k=100)
    val_ndcg = ndcg_evaluator.evaluate(joined_val)
    print("Validation Set Performence with NDCG: ", val_ndcg)

    print('Validatiion done')




    # test set

    # define the window to partition by user_id and sort by count (descending)
    window = Window.partitionBy('user_id').orderBy(desc('count_value'))

    # assign row_number to each recording_msid for each user
    test_user_msid = test.select('*', row_number().over(window).alias('row_number'))

    # top 100 recording_msid for each user
    test_user_top100_msid = test_user_msid.filter(test_user_msid.row_number <= 100).select('user_id', 'recording_msid')

    # convert the 100 recording_msid to a list
    test_user_top100_msid = test_user_top100_msid.groupBy('user_id').agg(collect_list('recording_msid').alias('popular100_each_user'))
    test_user_top100_msid = test_user_top100_msid.withColumn('popular100_each_user', col('popular100_each_user').cast(ArrayType(DoubleType())))
    


    # join the dataset
    joined_test = test_user_top100_msid.join(top100_msid)

    # for debug
    joined_test.printSchema()
    joined_test.show(5)

    map_evaluator = RankingEvaluator(predictionCol='popular100', labelCol='popular100_each_user', metricName='meanAveragePrecisionAtK', k=100)
    test_map = map_evaluator.evaluate(joined_test)
    print("Test Set Performence with MAP: ", test_map)
    ndcg_evaluator = RankingEvaluator(predictionCol='popular100', labelCol='popular100_each_user', metricName='ndcgAtK', k=100)
    test_ndcg = ndcg_evaluator.evaluate(joined_test)
    print("Test Set Performence with NDCG: ", test_ndcg)

    print('Test done')
    


                    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)


# result
# Validation Set Performence with MAP:  5.318522602789741e-05
# Validation Set Performence with NDCG:  0.0010588036371553201

# Test Set Performence with MAP:  0.00014168395403800342
# Test Set Performence with NDCG:  0.002233094150893789