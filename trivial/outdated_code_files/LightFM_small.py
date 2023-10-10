# spark-submit --deploy-mode client LightFM_small.py /user/dj928_nyu_edu

import numpy as np
import pandas as pd
import time
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import lightfm
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import matplotlib.pyplot as plt
from pyspark.sql import WindowSpec
from pyspark.sql.window import Window

from pyspark.sql import SparkSession
from pyspark.sql.functions import rank, dense_rank, col
import sys

import pydoop.hdfs as hd


def lightfm_small(spark, file_path):
    """
    file_path: hdfs:/user/bm106_nyu_edu/1004-project-2023/
    """
    print("Final Project Group 80 Extension: Single Machine Implementation - LightFM")
    
# <---------- 1. Read And Process Data ----------> #
    # Read train, validation, test .csv files
    print("Reading small_train...")
    small_train = spark.read.csv(file_path + '/train_small_by_history.csv', header=True)
    print("Done reading small_train.")
    print("Reading small_val...")
    small_val = spark.read.csv(file_path + '/val_small_by_history.csv', header=True)
    print("Done reading small_val.")
    print("Reading test...")
    test = spark.read.csv(file_path + '/test.csv', header=True)
    print("Done reading test.")
    
    # Drop irrelevant columns
    print("Dropping irrelevant columns...")
    small_train = small_train.drop("user_name", "artist_name", "track_name", "recording_mbid")
    small_val = small_val.drop("user_name", "artist_name", "track_name", "recording_mbid", "timestamp_ms", "timestamp_75")
    test = test.drop("user_name", "artist_name", "track_name", "recording_mbid")
    
    print("Joining dataframes...")
    total_item_user = small_train.union(small_val)
    total_item_user = total_item_user.union(test)
    
    print("Adjusting dataset dimensions...")
    # Convert the distinct user and item ids to sets
    item_ids_rdd = small_val.select('recording_msid').distinct().rdd.flatMap(lambda x: x).collect()
    user_ids_rdd = small_val.select('user_id').distinct().rdd.flatMap(lambda x: x).collect()

    # Fit the dataset using the sets of user and item ids
    data = Dataset()
    data.fit(users=user_ids_rdd, items=item_ids_rdd)

    
    print("Building interaction matrix...")
    interactions_train, weights_train = data.build_interactions([(small_train['user_id'][i],
                                                                  small_train['recording_msid'][i]) for i in range(small_train.count())])
    interactions_val, weights_val = data.build_interactions([(small_val['user_id'][i], 
                                                              small_val['recording_msid'][i]) for i in range(small_val.count())])
    interactions_test, weights_test = data.build_interactions([(test['user_id'][i], 
                                                            test['recording_msid'][i]) for i in range(test.count())])
    
    # <-------------------------- 3. Build Model and Tune Hyperparams -------------------------->
    # Hyperparameters to search over
    param_grid = {
        'rank': [10, 20, 50, 100],
        'max_iter': [1, 5, 10, 20],
        # 'learning_schedule': ['adagrad', 'adadelta', 'bpr'],
        'no_components': [10, 20, 30],
        'loss': ['warp', 'logistic', 'bpr'],
        'user_alpha': [1, 0.1, 0.01]
    }
    
    print("Searching over hyperparameters...")
    for rank in param_grid['rank']:
        for max_iter in param_grid['max_iter']:
            for no_components in param_grid['no_components']:
                for loss in param_grid['loss']:
                    for user_alpha in param_grid['user_alpha']:
                        start = time.time()
                        model = LightFM(loss=loss, 
                                        no_components=no_components, 
                                        user_alpha=user_alpha)
                        model = model.fit(interactions=interactions_train, 
                                          sample_weights=weights_train, 
                                          epochs=max_iter, 
                                          num_threads=4, 
                                          verbose=False)
                        val_precision = precision_at_k(model, interactions_val, k=rank).mean()
                        end = time.time()
                        print(f"Precision and time for k={rank}, max_iter={max_iter}, no_components={no_components}, loss={loss}, reg_param={user_alpha}: precision: {val_precision}, time cost: {end-start}")
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create Spark session object
    spark = SparkSession.builder.appName('lightfm_small').getOrCreate()

    # Get file path for the dataset to split
    file_path = sys.argv[1]

    # Calling the split function
    lightfm_small(spark, file_path)