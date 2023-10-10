"""
Created on Wed Apr 29 23:23:07 2023

@author: xinxiaomengliu
"""
# ALS model

import os

# import pyspark.sql to get the spark session
from pyspark.sql import SparkSession

from pyspark.ml.recommendation import ALS
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number, col, udf, collect_list
from pyspark.sql.types import ArrayType, DoubleType, IntegerType

# use RankingRvaluator() to evaluate the model (e.g. MAP, NDCG)
from pyspark.ml.evaluation import RankingEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def main(spark, userID):
    print('training set loading')
    train_small = spark.read.csv(f'hdfs:/user/{userID}/train_small.csv', schema='user_id INT, recording_msid INT, count_value INT')
    print('validation set loading')
    val_small = spark.read.csv(f'hdfs:/user/{userID}/val_small.csv', schema='user_id INT, recording_msid INT, count_value INT')
    print('test set loading')
    test = spark.read.csv(f'hdfs:/user/{userID}/test.csv', schema='user_id INT, recording_msid INT, count_value INT')

    # Filter out rows with NULL values
    train_small = train_small.filter(train_small.user_id.isNotNull() & train_small.recording_msid.isNotNull() & train_small.count_value.isNotNull())
    val_small = val_small.filter(val_small.user_id.isNotNull() & val_small.recording_msid.isNotNull() & val_small.count_value.isNotNull())
    test = test.filter(test.user_id.isNotNull() & test.recording_msid.isNotNull() & test.count_value.isNotNull())
    
    # create label for validation set
    label_val = val_small.groupby("user_id").agg(collect_list("recording_msid")).withColumnRenamed("collect_list(recording_msid)", "label")
    label_val = label_val.filter("user_id is not null").select("label", "user_id")


    # create label for test set
    label_test = test.groupby("user_id").agg(collect_list("recording_msid")).withColumnRenamed("collect_list(recording_msid)", "label")
    label_test = label_test.filter("user_id is not null").select("label", "user_id")

    params_grid = {
        'rank': [50, 100, 150, 200],
        'regParam': [0.01, 0.1, 1.0],
        'alpha': [0.1, 1.0, 5.0]
    }

    
    def search_best_params(params_grid, train_full, label_val):
        results = {}
        for rank in params_grid['rank']:
            for regParam in params_grid['regParam']:
                for alpha in params_grid['alpha']:
                    print(rank, regParam, alpha)
                    als = ALS(rank = rank, maxIter=10, regParam=regParam, alpha = alpha, userCol="user_id", itemCol="recording_msid", ratingCol="count_value", coldStartStrategy="drop")
                    model = als.fit(train_full)
                    print('model building finished')

                    predictions_val = model.recommendForUserSubset(label_val, 100)
                    predictions_val = (predictions_val
                                       .select("user_id", "recommendations.recording_msid")
                                       .withColumnRenamed("recording_msid", "prediction")
                    )
                    joined_val = label_val.join(predictions_val, label_val.user_id == predictions_val.user_id, 'inner').select("prediction", "label")
                    joined_val = (joined_val
                                  .withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
                                  .withColumn('label', col('label').cast(ArrayType(DoubleType())))
                    )

                    evaluator = RankingEvaluator(predictionCol='prediction', labelCol='label', metricName='meanAveragePrecisionAtK', k=100)
                    val_MAP = evaluator.evaluate(joined_val)
                    print("Validation Performence with MAP: ", val_MAP)

                    results[(rank, regParam, alpha)] = val_MAP

        best_params = max(results, key = results.get)
        return best_params


    best_params_ALS = search_best_params(params_grid, train_small, label_val)
    print('best rank', best_params_ALS[0])
    print('best regParam', best_params_ALS[1])
    print('best alpha', best_params_ALS[2])
    

    als = ALS(rank = best_params_ALS[0], maxIter=10, regParam=best_params_ALS[1], alpha = best_params_ALS[2], userCol="user_id", itemCol="recording_msid", ratingCol="count_value", coldStartStrategy="drop")
    model = als.fit(train_small)

    # Validation Prediction
    print('making predictions on the validation set')
    predictions_val = model.recommendForUserSubset(label_val, 100)
    predictions_val = (predictions_val
                       .select("user_id", "recommendations.recording_msid")
                       .withColumnRenamed("recording_msid", "prediction")
    )
    joined_val = label_val.join(predictions_val, label_val.user_id == predictions_val.user_id, 'inner').select("prediction", "label")
    joined_val = (joined_val
                  .withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
                  .withColumn('label', col('label').cast(ArrayType(DoubleType())))
    )

 
    # Test Prediction
    print('making predictions on the test set')
    predictions_test = model.recommendForUserSubset(label_test, 100)
    predictions_test = (predictions_test
                        .select("user_id", "recommendations.recording_msid")
                        .withColumnRenamed("recording_msid", "prediction")
    )
    joined_test = label_test.join(predictions_test, label_test.user_id == predictions_test.user_id, 'inner').select("prediction", "label")
    joined_test = (joined_test
                   .withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
                   .withColumn('label', col('label').cast(ArrayType(DoubleType())))
    )

    print('MAP')
    print('evaluateing performance on validation set')
    evaluator = RankingEvaluator(predictionCol='prediction', labelCol='label', metricName='meanAveragePrecisionAtK', k=100)
    val_MAP = evaluator.evaluate(joined_val)
    print("Validation Performence with MAP: ", val_MAP)
    
    print('evaluateing performance on test set')
    test_MAP = evaluator.evaluate(joined_test)
    print("Test Performence with MAP: ", test_MAP)

    print('NDCG')
    print('evaluateing performance on validation set')
    evaluator = RankingEvaluator(predictionCol='prediction', labelCol='label', metricName='ndcgAtK', k=100)
    val_NDCG = evaluator.evaluate(joined_val)
    print("Validation Performence with NDCG: ", val_NDCG)
    
    print('evaluateing performance on test set')
    test_NDCG = evaluator.evaluate(joined_test)
    print("Test Performence with NDCG: ", test_NDCG)



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)