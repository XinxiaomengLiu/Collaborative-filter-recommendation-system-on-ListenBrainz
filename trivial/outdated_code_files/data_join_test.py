from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, percentile_approx, expr, count
from pyspark.sql import WindowSpec
from pyspark.sql.window import Window
import sys

# spark-submit --deploy-mode client data_partitioning/data_join_test.py /user/bm106_nyu_edu/1004-project-2023/


def partition(spark, file_path):
    """
    file_path: hdfs:/user/bm106_nyu_edu/1004-project-2023/
    
    1. Read .parquet files
    2. Join tables
    3. Split joined table by timestamp 75% train, 25% validation
    """

# ---------------------- Small files: partition only by timestamp ---------------------- #

    # Read .parquet file
    interaction_test = spark.read.parquet(file_path + '/interactions_test.parquet')
    users_test = spark.read.parquet(file_path + '/users_test.parquet')
    tracks_test = spark.read.parquet(file_path + '/tracks_test.parquet')
    
    # Create view to run SQL queries
    interaction_test.createOrReplaceTempView('interaction_test')
    users_test.createOrReplaceTempView('users_test')
    tracks_test.createOrReplaceTempView('tracks_test')
    

# ---------------------- Join tables ---------------------- #
    
    # join interactions_train_small with user_train_small on user_id
    # join interactions_train_small with tracks_train_small on recording_msid
    joined_part1 = interaction_test.join(users_test, users_test.user_id == interaction_test.user_id, "left")\
        .select(interaction_test.user_id, interaction_test.recording_msid, interaction_test.timestamp, users_test.user_name)
    joined_part2 = joined_part1.join(tracks_test, tracks_test.recording_msid == joined_part1.recording_msid, "left")\
        .select(joined_part1.user_id, joined_part1.recording_msid, joined_part1.timestamp, joined_part1.user_name, tracks_test.artist_name, tracks_test.track_name, tracks_test.recording_mbid)
    
# ---------------------- Split each user's history into train and validation ---------------------- #
    
    # check shape of train and validation
    print("Shape of Test DataFrame: ({}, {})".format(joined_part2.count(), len(joined_part2.columns))) 
   
# ---------------------- Write to .csv files ---------------------- #
    
    write_file_path = "hdfs:/user/dj928_nyu_edu"
    joined_part2.write.csv(write_file_path + '/test.parquet', header=True)  # (50031760, 7)
    
    print("data_join_test.py done running for test files.")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create Spark session object
    spark = SparkSession.builder.appName('data_partition').getOrCreate()

    # Get file path for the dataset to split
    file_path = sys.argv[1]

    # Calling the split function
    partition(spark, file_path)




    