"""
Created on Sat Apr 27 21:23:35 2023

@author: xinxiaomengliu
"""
# popularity baseline model

import os

# import pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, collect_list, desc, row_number, udf, col
from pyspark.sql.functions import countDistinct, sum, when
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
    # estimate the average number of listens per user

    # calculate # of listens and unique users per recording_msid
    counts = (train_full
            .groupBy('recording_msid')
            .agg(sum('count_value').alias('num_listens'), countDistinct('user_id').alias('num_users'))
    )
    counts.show(5)

    params = [1000, 5000, 10000]
    params2 = [100, 200, 300, 400, 500]

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
    



    def search_best_beta(params, counts, user_top100_msid):
        # apply damping to # of unique users and calculate the avg number of counts per user_id
        # damping factor = 1000 for full dataset
        MAP = {}
        for beta in params:
            print(beta)
            counts = (counts
                .withColumn('avg_counts_per_user', counts.num_listens / counts.num_users + beta)
                .filter(col('avg_counts_per_user').isNotNull())
               )
    
            # select the top 100 recording_msid based on the avg number of counts per user_id
            top100_msid = (counts
                   .select('recording_msid', 'avg_counts_per_user')
                   .orderBy('avg_counts_per_user')
                   .limit(100)
                   .select('recording_msid')
                   .agg(collect_list('recording_msid'))
                   .withColumnRenamed('collect_list(recording_msid)', 'popular100')
                   .withColumn('popular100', col('popular100').cast(ArrayType(DoubleType())))
                   )  
            
            # join the dataset
            joined_val = user_top100_msid.join(top100_msid)   


            map_evaluator = RankingEvaluator(predictionCol='popular100', labelCol='popular100_each_user', metricName='meanAveragePrecisionAtK', k=100)
            val_map = map_evaluator.evaluate(joined_val)
            print("Validation Set Performence with MAP: ", val_map)

            MAP[beta] = val_map
        


            #ndcg_evaluator = RankingEvaluator(predictionCol='popular100', labelCol='popular100_each_user', metricName='ndcgAtK', k=100)
            #val_ndcg = ndcg_evaluator.evaluate(joined_val)
            #print("Validation Set Performence with NDCG: ", val_ndcg)

            #print('Validatiion done')



            '''
            


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
            '''
        best_beta = max(MAP, key = MAP.get)
        return best_beta
    
    best_beta = search_best_beta(params, counts, user_top100_msid)
    print(best_beta)



    
    
    

                    


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('baseline').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)





