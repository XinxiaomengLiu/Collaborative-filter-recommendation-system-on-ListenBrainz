# spark-submit --deploy-mode client test.py /user/dj928_nyu_edu

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

    # small_train.createOrReplaceTempView('small_train')
    # small_val.createOrReplaceTempView('small_val')
    # test.createOrReplaceTempView('test')
    
    # small_train.show(5)
    # small_val.show(5)
    # test.show(5)
    
    # Drop irrelevant columns
    print("Dropping irrelevant columns...")
    small_train = small_train.drop("user_name", "artist_name", "track_name", "recording_mbid")
    small_val = small_val.drop("user_name", "artist_name", "track_name", "recording_mbid", "timestamp_ms", "timestamp_75")
    test = test.drop("user_name", "artist_name", "track_name", "recording_mbid")
    
    # Convert to Pandas DataFrame
    print("Converting to Pandas DataFrame...")
    small_train = small_train.toPandas()
    small_val = small_val.toPandas()
    test = test.toPandas()

    print('Reassigning recording_msid to avoid dimension error...')
    total_item_user = pd.concat([train_small, val_small, test]).drop_duplicates()
    total_item_user = total_item_user.sort_values(['recording_msid'])
    total_item_user['new_recording_msid'] = (total_item_user.groupby(['recording_msid'], sort=False).ngroup()+1)

    print('Appending new_recording_msid to existing train, test, and val sets...')
    train_small = train_small.merge(total_item_user, on=['user_id', 'recording_msid', 'timestamp'], how="left")
    val_small = val_small.merge(total_item_user, on=['user_id', 'recording_msid', 'timestamp'], how="left")
    test = test.merge(total_item_user, on=['user_id', 'recording_msid', 'timestamp'], how="left")

    print('Dropping original recording_msid columns...')
    train_small = train_small.drop(columns=['recording_msid'])
    val_small = val_small.drop(columns=['recording_msid'])
    test = test.drop(columns=['recording_msid'])

    # <---------- 2. Build sparse interaction matrices ----------> #

    print("Adjusting dataset dimensions to avoid unmatched dimension error...")
    data = Dataset()
    data.fit(users = np.unique(total_item_user["user_id"]), items = np.unique(total_item_user["new_recording_msid"]))

    print("building interactions")
    # build interactions
    interactions_train, weights_train = data.build_interactions([(train_small['user_id'][i], 
                                                                train_small['new_recording_msid'][i],
                                                                train_small['timestamp'][i]) for i in range(train_small.shape[0])])
    interactions_val, weights_val = data.build_interactions([(val_small['user_id'][i],
                                                            val_small['new_recording_msid'][i], 
                                                            val_small['timestamp'][i]) for i in range(val_small.shape[0])])
    interactions_test, weights_test = data.build_interactions([(test['user_id'][i],
                                                            test['new_recording_msid'][i], 
                                                            test['timestamp'][i]) for i in range(test.shape[0])])

    # <---------- 3. Train LightFM models, search for best params ----------> #

    # Hyper-params to search over
    param_grid = {
        'rank': [10, 20, 50, 100],
        'max_iter': [1, 5, 10, 20],
        # 'learning_schedule': ['adagrad', 'adadelta', 'bpr'],
        'no_components': [10, 20, 30],
        'loss': ['warp', 'logistic', 'bpr'],
        'user_alpha': [1, 0.1, 0.01]
    }

    print("Searching over hyperparams...")
    for rank in param_grid['rank']:  # k in precision@k
        for max_iter in param_grid['max_iter']:  # number of epochs
            for no_components in param_grid['no_components']:  # latent dimensionality
                for loss in param_grid['loss']:  # loss function
                    for user_alpha in param_grid['user_alpha']:  # regularization parameter
                        start = time.time()
                        model = LightFM(loss=loss, no_components=no_components, user_alpha=user_alpha)
                        model = model.fit(interactions=interactions_train, sample_weights=weights_train, 
                                            epochs=max_iter, num_threads=4, verbose=False)
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
