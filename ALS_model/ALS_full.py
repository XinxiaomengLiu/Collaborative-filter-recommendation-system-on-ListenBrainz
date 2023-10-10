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
    train_full = spark.read.csv(f'hdfs:/user/{userID}/train.csv', schema='user_id INT, recording_msid INT, count_value INT')
    print('validation set loading')
    val_full = spark.read.csv(f'hdfs:/user/{userID}/val.csv', schema='user_id INT, recording_msid INT, count_value INT')
    print('test set loading')
    test = spark.read.csv(f'hdfs:/user/{userID}/test.csv', schema='user_id INT, recording_msid INT, count_value INT')

    # full datasets are too large. It takes forever to run the code. We have to sample.
    # even after sampling, the datasets are still large enough
    '''
    sample_fraction = 0.001
    train_full = train_full.sample(fraction=sample_fraction)
    val_full = val_full.sample(fraction=sample_fraction)
    test = test.sample(fraction=sample_fraction)
    '''



    # Filter out rows with NULL values
    train_full = train_full.filter(train_full.user_id.isNotNull() & train_full.recording_msid.isNotNull() & train_full.count_value.isNotNull())
    val_full = val_full.filter(val_full.user_id.isNotNull() & val_full.recording_msid.isNotNull() & val_full.count_value.isNotNull())
    test = test.filter(test.user_id.isNotNull() & test.recording_msid.isNotNull() & test.count_value.isNotNull())
    
    # for debug
    train_full.show(5)


    # create label for validation set
    label_val = val_full.groupby("user_id").agg(collect_list("recording_msid")).withColumnRenamed("collect_list(recording_msid)", "label")
    label_val = label_val.filter("user_id is not null").select("label", "user_id")

    # for debug
    label_val.printSchema()
    label_val.show(5)

    # create label for test set
    label_test = test.groupby("user_id").agg(collect_list("recording_msid")).withColumnRenamed("collect_list(recording_msid)", "label")
    label_test = label_test.filter("user_id is not null").select("label", "user_id")

    params_grid = {
        'rank': [50, 100, 150, 200],
        'regParam': [0.01, 0.1, 1.0],
        'alpha': [0.1, 1.0, 5.0]
    }
    params_grid2 = {
        'rank': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        'regParam': [0.01],
        'alpha': [0.1]
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


    best_params_ALS = search_best_params(params_grid2, train_full, label_val)
    print('best rank', best_params_ALS[0])
    print('best regParam', best_params_ALS[1])
    print('best alpha', best_params_ALS[2])
    

    als = ALS(rank = best_params_ALS[0], maxIter=10, regParam=best_params_ALS[1], alpha = best_params_ALS[2], userCol="user_id", itemCol="recording_msid", ratingCol="count_value", coldStartStrategy="drop")
    model = als.fit(train_full)

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



    



    '''
    als = ALS(rank = 100, maxIter=10, regParam=0.1, alpha = 10, userCol="user_id", itemCol="recording_msid", ratingCol="count_value", coldStartStrategy="drop")
    model = als.fit(train_full)
    


    
    
    # Validation Prediction
    print('making predictions on the validation set')
    predictions_val = model.recommendForUserSubset(label_val, 100)
    predictions_val.createOrReplaceTempView('predictions_val')

    predictions_val = predictions_val.select("user_id", "recommendations.recording_msid")
    predictions_val = predictions_val.withColumnRenamed("recording_msid", "prediction")
    dataset_val = label_val.join(predictions_val, label_val.user_id == predictions_val.user_id, 'inner').select("prediction", "label")
    dataset_val = dataset_val.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_val = dataset_val.withColumn('label', col('label').cast(ArrayType(DoubleType())))

    
    
    # Test Prediction
    print('making predictions on the test set')
    predictions_test = model.recommendForUserSubset(label_test, 100)
    predictions_test.createOrReplaceTempView('predictions_test')

    predictions_test = predictions_test.select("user_id", "recommendations.recording_msid")
    predictions_test = predictions_test.withColumnRenamed("recording_msid", "prediction")
    dataset_test = label_test.join(predictions_test, label_test.user_id == predictions_test.user_id, 'inner').select("prediction", "label")
    dataset_test = dataset_test.withColumn('prediction', col('prediction').cast(ArrayType(DoubleType())))
    dataset_test = dataset_test.withColumn('label', col('label').cast(ArrayType(DoubleType())))


    print('evaluateing performance on validation set')
    evaluator = RankingEvaluator(metricName='meanAveragePrecisionAtK', k=100)
    evaluator.setPredictionCol("prediction")
    val_MAP = evaluator.evaluate(dataset_val)
    print("Validation Performence with MAP: ", val_MAP)
    
    print('evaluateing performance on test set')
    test_MAP = evaluator.evaluate(dataset_test)
    print("Test Performence with MAP: ", test_MAP)
    '''




'''
    # Filter out rows with NULL values
    train_full = train_full.filter(train_full.user_id.isNotNull() & train_full.recording_msid.isNotNull() & train_full.count_value.isNotNull())
    val_full = val_full.filter(val_full.user_id.isNotNull() & val_full.recording_msid.isNotNull() & val_full.count_value.isNotNull())

    # Rename count_value to rating
    train_full = train_full.withColumnRenamed('count_value', 'rating')
    train_full = train_full.withColumn('rating', col('rating').cast(ArrayType(DoubleType())))
    val_full = val_full.withColumnRenamed('count_value', 'rating')
    val_full = val_full.withColumn('rating', col('rating').cast(ArrayType(DoubleType())))

    # Build the ALS model
    als = ALS(userCol="user_id", itemCol="recording_msid", ratingCol="rating", coldStartStrategy="drop")

    # Define the hyperparameter grid
    paramGrid = (ParamGridBuilder()
                 .addGrid(als.rank, [10, 20, 30])  # Dimension of latent factors
                 .addGrid(als.alpha, [1.0, 10.0, 100.0])  # Implicit feedback parameter
                 .addGrid(als.regParam, [0.01, 0.1, 0.2])  # Regularization parameter
                 .build())

    # Define the evaluator
    evaluator = RankingEvaluator(labelCol="rating", predictionCol="prediction", metricName='meanAveragePrecisionAtK')

    # Define the cross-validator
    cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    # Fit the model
    model = cv.fit(train_full)

    # Make predictions on the validation set
    val_predictions = model.transform(val_full)

    # Evaluate the model on the validation set
    val_map = evaluator.evaluate(val_predictions)
    print("Validation Set Mean Average Precision (MAP):", val_map)

    best_model = model.bestModel
    print("Best Rank:", best_model.rank)
    print("Best Alpha:", best_model._java_obj.parent().getAlpha())
    print("Best Regularization Parameter:", best_model._java_obj.parent().getRegParam())





    # Filter out rows with NULL values
    train_full = train_full.filter(train_full.user_id.isNotNull() & train_full.recording_msid.isNotNull() & train_full.count_value.isNotNull())
    val_full = val_full.filter(val_full.user_id.isNotNull() & val_full.recording_msid.isNotNull() & val_full.count_value.isNotNull())
    test = test.filter(test.user_id.isNotNull() & test.recording_msid.isNotNull() & test.count_value.isNotNull())

    # we use the counts of recording_msid per user as the user's rating of this recording_msid
    # group by user_id and recording_msid, and count the number of concurrences


    # for debug
    train_full.show(5)




    # build the ALS model
    als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="recording_msid", ratingCol="count_value",
          coldStartStrategy="drop")
    model = als.fit(train_full)


    # evaluation on validation set
    evaluator = RankingEvaluator(labelCol="count_value", predictionCol="prediction",  metricName='MAP')
    val_map = evaluator.evaluate(model.transform(val_full))
    print("Mean average precision = " + str(val_map))

    evaluator = RankingEvaluator(labelCol="count_value", predictionCol="prediction", metricName="NDCG")
    val_ndcg = evaluator.evaluate(model.transform(val_full))
    print("Normalized discounted cumulative gain = " + str(val_ndcg))



    # evaluation on test set
    evaluator = RankingEvaluator(labelCol="count_value", predictionCol="prediction",  metricName='MAP')
    test_map = evaluator.evaluate(model.transform(test))
    print("Mean average precision = " + str(test_map))

    evaluator = RankingEvaluator(labelCol="count_value", predictionCol="prediction", metricName="NDCG")
    test_ndcg = evaluator.evaluate(model.transform(test))
    print("Normalized discounted cumulative gain = " + str(test_ndcg))

'''











# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)



