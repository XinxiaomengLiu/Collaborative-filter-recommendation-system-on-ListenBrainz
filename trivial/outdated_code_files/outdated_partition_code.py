from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, percentile_approx, expr, count
from pyspark.sql import WindowSpec
from pyspark.sql.window import Window
import sys

# spark-submit --deploy-mode client data_partition_small_by_timestamp.py /user/bm106_nyu_edu/1004-project-2023/


def partition(spark, file_path):
    """
    file_path: hdfs:/user/bm106_nyu_edu/1004-project-2023/
    
    1. Read .parquet files
    2. Join tables
    3. Split joined table by timestamp 75% train, 25% validation
    """

# ---------------------- Small files: partition only by timestamp ---------------------- #

    # Read .parquet file
    interactions_train_small = spark.read.parquet(file_path + '/interactions_train_small.parquet') 
    users_train_small = spark.read.parquet(file_path + '/users_train_small.parquet')  
    tracks_train_small = spark.read.parquet(file_path + '/tracks_train_small.parquet')
    
    # Create view to run SQL queries
    interactions_train_small.createOrReplaceTempView('interactions_train_small')
    users_train_small.createOrReplaceTempView('users_train_small')
    tracks_train_small.createOrReplaceTempView('tracks_train_small')
    

# ---------------------- Join tables ---------------------- #
    
    # join interactions_train_small with user_train_small on user_id
    # join interactions_train_small with tracks_train_small on recording_msid
    joined_train_small_part1 = interactions_train_small.join(users_train_small, users_train_small.user_id == interactions_train_small.user_id, "left")\
        .select(interactions_train_small.user_id, interactions_train_small.recording_msid, interactions_train_small.timestamp, users_train_small.user_name)
    joined_train_small_part2 = joined_train_small_part1.join(tracks_train_small, tracks_train_small.recording_msid == joined_train_small_part1.recording_msid, "left")\
        .select(joined_train_small_part1.user_id, joined_train_small_part1.recording_msid, joined_train_small_part1.timestamp, joined_train_small_part1.user_name, tracks_train_small.artist_name, tracks_train_small.track_name, tracks_train_small.recording_mbid)
    
    # # drop users who listened to less than 10 recordings
    # window_spec = Window.partitionBy('user_id')
    # joined_train_small_part2 = joined_train_small_part2.withColumn('count', count('*').over(window_spec))
    # joined_train_small_part2 = joined_train_small_part2.filter('count > 10')
    # joined_train_small_part2 = joined_train_small_part2.drop('count')
    
    # # drop recordings that are listened by less than 10 users
    # window_spec = Window.partitionBy('recording_msid')
    # joined_train_small_part2 = joined_train_small_part2.withColumn('count', count('*').over(window_spec))
    # joined_train_small_part2 = joined_train_small_part2.filter('count > 10')
    # joined_train_small_part2 = joined_train_small_part2.drop('count')
    
# ---------------------- Split each user's history into train and validation ---------------------- #

    # Convert timestamp column to Unix timestamp in milliseconds
    joined_train_small_part2 = joined_train_small_part2.withColumn("timestamp_ms", expr("unix_timestamp(timestamp) * 1000"))
    
    # create a Window specification to partition the data by user_id and order it by timestamp
    window_spec = Window.partitionBy('user_id')
    # .orderBy('timestamp_ms')

    # calculate the 75th percentile timestamp for each user
    joined_train_small_part2 = joined_train_small_part2.withColumn('timestamp_75', percentile_approx('timestamp_ms', 0.75).over(window_spec))
    
    train = joined_train_small_part2.filter(col("timestamp_ms") < col("timestamp_75"))
    validation = joined_train_small_part2.filter(col("timestamp_ms") >= col("timestamp_75"))
    # validation = joined_train_small_part2.subtract(train)
    
    # train = train.drop('timestamp_ms', 'timestamp_75')
    # validation = validation.drop('timestamp_ms', 'timestamp_75')
    
    # quantile = 0.75
    # timestamp_quantile = joined_train_small_part2.approxQuantile("timestamp_ms", [quantile], 0.01)[0]
    # train = joined_train_small_part2.filter(col("timestamp_ms") < timestamp_quantile)
    # validation = joined_train_small_part2.subtract(train)
    # train = train.drop('timestamp_ms')
    # validation = validation.drop('timestamp_ms')
    
    # check shape of train and validation
    print("Shape of Train DataFrame: ({}, {})".format(train.count(), len(train.columns)))  # (38088588, 7) --> (20657901, 7) --> (38301373, 7)
    print("Shape of Validation DataFrame: ({}, {})".format(validation.count(), len(validation.columns)))  # (12989181, 7) --> (7044358, 7) --> (12778902, 9)
   
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




    