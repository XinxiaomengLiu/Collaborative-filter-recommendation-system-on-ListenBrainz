from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, percentile_approx, expr, count
from pyspark.sql import WindowSpec
from pyspark.sql.window import Window
import sys

# spark-submit --deploy-mode client data_partition_large.py /user/bm106_nyu_edu/1004-project-2023/


def partition(spark, file_path):
    """
    file_path: hdfs:/user/bm106_nyu_edu/1004-project-2023/  

    1. Read .parquet files
    2. Join tables
    3. Delete users and recordings with less than 10 interactions
    4. Split train data by user_id into 50% training and 50% validation
    5. Further split validation data by timestamp, put 50% back with training data  
    """

# ---------------------- Read .parquet files ---------------------- #
    # Read .parquet file
    interactions_train = spark.read.parquet(file_path + '/interactions_train.parquet') 
    users_train = spark.read.parquet(file_path + '/users_train.parquet')
    tracks_train = spark.read.parquet(file_path + '/tracks_train.parquet')
    
    # Create view to run SQL queries
    interactions_train.createOrReplaceTempView('interactions_train')
    users_train.createOrReplaceTempView('users_train')
    tracks_train.createOrReplaceTempView('tracks_train')
    
# ---------------------- Join tables ---------------------- #
    
    # join interactions_train with user_train on user_id
    # join interactions_train with tracks_train on recording_msid
    joined_train_part1 = interactions_train.join(users_train, users_train.user_id == interactions_train.user_id, "left")\
        .select(interactions_train.user_id, interactions_train.recording_msid, interactions_train.timestamp, users_train.user_name)
    joined_train_part2 = joined_train_part1.join(tracks_train, tracks_train.recording_msid == joined_train_part1.recording_msid, "left")\
        .select(joined_train_part1.user_id, joined_train_part1.recording_msid, joined_train_part1.timestamp, joined_train_part1.user_name, tracks_train.artist_name, tracks_train.track_name, tracks_train.recording_mbid)
    
    # # drop users who listened to less than 10 recordings
    # window_spec = Window.partitionBy('user_id')
    # joined_train_part2 = joined_train_part2.withColumn('count', count('*').over(window_spec))
    # joined_train_part2 = joined_train_part2.filter('count > 10')
    # joined_train_part2 = joined_train_part2.drop('count')
    
    # # drop recordings that are listened by less than 10 users
    # window_spec = Window.partitionBy('recording_msid')
    # joined_train_part2 = joined_train_part2.withColumn('count', count('*').over(window_spec))
    # joined_train_part2 = joined_train_part2.filter('count > 10')
    # joined_train_part2 = joined_train_part2.drop('count')
    
# ---------------------- Split train data into 50% training and 50% validation ---------------------- #
    
    partition = joined_train_part2.select('user_id').distinct().randomSplit([0.5, 0.5], seed=80)
    
    # collect user_id for train and validation
    train_users = tuple(list(x.user_id for x in partition[0].collect()))
    train_users_str = [str(user) for user in train_users]
    val_users = tuple(list(x.user_id for x in partition[1].collect()))
    val_users_str = [str(user) for user in val_users]
    
    # make data frames for train and validation
    train = joined_train_part2.filter(col("user_id").isin(train_users_str))
    print("Shape of Large Train DataFrame before further split: ({}, {})".format(train.count(), len(train.columns)))  # (88104955, 7) --> (61799774, 7) --> (88104955, 7)
    val = joined_train_part2.filter(col("user_id").isin(val_users_str))
    
    train.createOrReplaceTempView('train')
    val.createOrReplaceTempView('val')
    
# ---------------------- further split validation data, put 50% back to train ---------------------- #

    quantile = 0.5
    
    # Convert timestamp column to Unix timestamp in milliseconds
    val = val.withColumn("timestamp_ms", expr("unix_timestamp(timestamp) * 1000"))
    timestamp_quantile = val.approxQuantile("timestamp_ms", [quantile], 0.01)[0]
    val_older = val.filter(col("timestamp_ms") < timestamp_quantile)
    
    # train = train.withColumn("timestamp_ms", expr("unix_timestamp(timestamp) * 1000"))
    val_older = val_older.drop('timestamp_ms') 
    train = train.union(val_older)
    print("Shape of Train DataFrame after further split: ({}, {})".format(train.count(), len(train.columns)))  # --> (91555698, 7) --> (133544068, 7)

    # Remove older records from validation set; keep only the most recent records
    val = val.drop('timestamp_ms')
    val = val.subtract(val_older)
    
    # train.orderBy('user_id').groupBy('user_id').count().show(10)
    # val.orderBy('user_id').groupBy('user_id').count().show(10)
    
# ---------------------- Write to .csv files ---------------------- #
    
    write_file_path = "hdfs:/user/dj928_nyu_edu"
    train.write.csv(write_file_path + '/train_large.csv', header=True)
    val.write.csv(write_file_path + '/val_large.csv', header=True)
    
    print("data_partition.py done running for large files.")


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create Spark session object
    spark = SparkSession.builder.appName('data_partition').getOrCreate()

    # Get file path for the dataset to split
    file_path = sys.argv[1]

    # Calling the split function
    partition(spark, file_path)