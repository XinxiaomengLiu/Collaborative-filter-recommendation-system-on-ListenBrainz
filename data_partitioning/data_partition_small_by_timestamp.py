from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, percentile_approx, expr, count, dense_rank
from pyspark.sql import WindowSpec
from pyspark.sql.window import Window
import sys

# spark-submit --deploy-mode client data_partition_small_by_timestamp.py /user/bm106_nyu_edu/1004-project-2023/


def partition(spark, file_path):
    """
    file_path: hdfs:/user/bm106_nyu_edu/1004-project-2023/
    
    Treatment of small files: train and validation
    
    1. Read .parquet files
    2. Join tables
    3. Split joined table by timestamp 75% train, 25% validation
    4. Replace recording_msid with distinct integers, count interactions
    5. Write to .csv files
    """

# ---------------------- Small files: partition only by timestamp ---------------------- #

    # Read .parquet file
    interactions_train_small = spark.read.parquet(file_path + '/interactions_train_small.parquet') 
    users_train_small = spark.read.parquet(file_path + '/users_train_small.parquet')  
    tracks_train_small = spark.read.parquet(file_path + '/tracks_train_small.parquet')
    
    # Read .parquet files for test
    interactions_test = spark.read.parquet(file_path + '/interactions_test.parquet')
    users_test = spark.read.parquet(file_path + '/users_test.parquet')
    tracks_test = spark.read.parquet(file_path + '/tracks_test.parquet')
    
    # Create view to run SQL queries
    interactions_train_small.createOrReplaceTempView('interactions_train_small')
    users_train_small.createOrReplaceTempView('users_train_small')
    tracks_train_small.createOrReplaceTempView('tracks_train_small')
    
    interactions_test.createOrReplaceTempView('interactions_test')
    users_test.createOrReplaceTempView('users_test')
    tracks_test.createOrReplaceTempView('tracks_test')
    

# ---------------------- Join tables ---------------------- #
    
    # join interactions_train_small with user_train_small on user_id
    # join interactions_train_small with tracks_train_small on recording_msid
    joined_train_small_part1 = interactions_train_small.join(users_train_small, users_train_small.user_id == interactions_train_small.user_id, "left")\
        .select(interactions_train_small.user_id, interactions_train_small.recording_msid, interactions_train_small.timestamp, users_train_small.user_name)
    joined_train_small_part2 = joined_train_small_part1.join(tracks_train_small, tracks_train_small.recording_msid == joined_train_small_part1.recording_msid, "left")\
        .select(joined_train_small_part1.user_id, joined_train_small_part1.recording_msid, joined_train_small_part1.timestamp, joined_train_small_part1.user_name, tracks_train_small.artist_name, tracks_train_small.track_name, tracks_train_small.recording_mbid)
        
    # join small data with test data
    joined_test_part1 = interactions_test.join(users_test, users_test.user_id == interactions_test.user_id, "left")\
        .select(interactions_test.user_id, interactions_test.recording_msid, interactions_test.timestamp, users_test.user_name)
    joined_test_part2 = joined_test_part1.join(tracks_test, tracks_test.recording_msid == joined_test_part1.recording_msid, "left")\
        .select(joined_test_part1.user_id, joined_test_part1.recording_msid, joined_test_part1.timestamp, joined_test_part1.user_name, tracks_test.artist_name, tracks_test.track_name, tracks_test.recording_mbid)
    
# ---------------------- Split each user's history into train and validation ---------------------- #

    # Convert timestamp column to Unix timestamp in milliseconds
    joined_train_small_part2 = joined_train_small_part2.withColumn("timestamp_ms", expr("unix_timestamp(timestamp) * 1000"))
    
    # create a Window specification to partition the data by user_id and order it by timestamp
    window_spec = Window.partitionBy('user_id')

    # calculate the 75th percentile timestamp for each user
    joined_train_small_part2 = joined_train_small_part2.withColumn('timestamp_75', percentile_approx('timestamp_ms', 0.75).over(window_spec))
    
    train = joined_train_small_part2.filter(col("timestamp_ms") < col("timestamp_75"))
    validation = joined_train_small_part2.filter(col("timestamp_ms") >= col("timestamp_75"))
    
    # check shape of train and validation
    print("Shape of Train DataFrame: ({}, {})".format(train.count(), len(train.columns)))  # (38088588, 7) --> (20657901, 7) --> (38301373, 7)
    print("Shape of Validation DataFrame: ({}, {})".format(validation.count(), len(validation.columns)))  # (12989181, 7) --> (7044358, 7) --> (12778902, 9)
    
# ---------------------- Replace recording_msid with assigned integers ---------------------- #

    # union joined_train_small_part2 and joined_test_part2
    joined_all = joined_train_small_part2.union(joined_test_part2)
    joined_all = joined_all.withColumn("recording_msid_int", dense_rank().over(Window.orderBy("recording_msid")))
    recording_msid_old_new = joined_all.select("recording_msid", "recording_msid_int").distinct()

    # join recording_msid_old_new with train and validation
    train = train.join(recording_msid_old_new, recording_msid_old_new.recording_msid == train.recording_msid, "left")\
        .select(train.user_id.alias("user"), recording_msid_old_new.recording_msid_int.alias("item"))
    validation = validation.join(recording_msid_old_new, recording_msid_old_new.recording_msid == validation.recording_msid, "left")\
        .select(validation.user_id.alias("user"), recording_msid_old_new.recording_msid_int.alias("item"))

# ---------------------- Group by user and count number of interactions ---------------------- #

    # Add a new column "rating" by counting the interactions between user and item
    train = train.groupBy("user", "item").agg(count("*").alias("rating"))
    validation = validation.groupBy("user", "item").agg(count("*").alias("rating"))

    # Retain only distinct rows
    train = train.dropDuplicates()
    validation = validation.dropDuplicates()
   
# ---------------------- Write to .csv files ---------------------- #
    
    write_file_path = "hdfs:/user/dj928_nyu_edu"
    train.write.csv(write_file_path + '/train_small_by_history.csv', header=True)
    validation.write.csv(write_file_path + '/val_small_by_history.csv', header=True)
    
    print("data_partition.py done running for small files (partition only by timestamp).")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create Spark session object
    spark = SparkSession.builder.appName('data_partition').getOrCreate()

    # Get file path for the dataset to split
    file_path = sys.argv[1]

    # Calling the split function
    partition(spark, file_path)




    