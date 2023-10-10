from pyspark.sql import SparkSession
import sys

# spark-submit --deploy-mode client trivial/not_important.py /user/dj928_nyu_edu

def not_important(spark, file_path):
    
    # train_small_by_history = spark.read.csv(file_path + '/train_large_by_history.csv', header=True, inferSchema=True)
    # train_small_by_history.createOrReplaceTempView('train_large_by_history')
    # train_small_by_history.show(5)   
    
    val_small_by_history.createOrReplaceTempView('train_small_by_history')
    val_small_by_history = spark.read.csv(file_path + '/train_small_by_history.csv', header=True, inferSchema=True)
    print("val_table 5 rows")
    val_small_by_history.show(5)
    
'''
+-------+--------------------+-------------------+------------+--------------------+--------------------+--------------------+
|user_id|      recording_msid|          timestamp|   user_name|         artist_name|          track_name|      recording_mbid|
+-------+--------------------+-------------------+------------+--------------------+--------------------+--------------------+
|    145|01f7276a-f4d9-475...|2018-01-18 07:39:00|thespeckofme|X-Ecutioners feat...|           Dramacyde|13d63f90-2998-46a...|
|    145|01f7276a-f4d9-475...|2018-01-18 07:42:37|thespeckofme|X-Ecutioners feat...|           Dramacyde|13d63f90-2998-46a...|
|    145|0778b2b0-1028-4f0...|2018-05-23 22:50:03|thespeckofme|Shooter Jennings ...|All of this Could...|                null|
|    145|0bdf90d8-eccc-494...|2018-04-26 23:57:47|thespeckofme|          Soulsavers|                Take|9e594f7a-2bb9-44b...|
|    145|0fa3ceab-d564-4db...|2018-06-05 00:22:39|thespeckofme|The American Anal...|Like Foxes Throug...|1e74cdd9-d6b3-4b9...|
+-------+--------------------+-------------------+------------+--------------------+--------------------+--------------------+
only showing top 5 rows

+-------+--------------------+-------------------+----------+---------------+--------------------+--------------------+
|user_id|      recording_msid|          timestamp| user_name|    artist_name|          track_name|      recording_mbid|
+-------+--------------------+-------------------+----------+---------------+--------------------+--------------------+
|      6|7133b167-1d74-4bc...|2018-04-16 12:40:23|    CatCat|          Floex|Constructing The ...|                null|
|      6|bc1cac39-280b-410...|2018-08-20 17:50:04|    CatCat|     Whitesnake|       Bloody Luxury|b9eacec4-9c03-45b...|
|      6|c627d249-b12d-4d7...|2018-01-29 21:08:44|    CatCat|Michael Z. Land|Slappy Cromwell J...|                null|
|    120|1fa633fa-017a-449...|2018-04-18 10:04:19|KypaToP_HM|   Sacred Steel|     We Die Fighting|                null|
|    120|68f9587c-5c93-4e4...|2018-10-24 02:06:32|KypaToP_HM|Spiritus Mortis|      Curved Horizon|                null|
+-------+--------------------+-------------------+----------+---------------+--------------------+--------------------+
only showing top 5 rows


Test:
+-------+--------------------+--------------------+---------+--------------------+---------------+--------------------+
|user_id|      recording_msid|           timestamp|user_name|         artist_name|     track_name|      recording_mbid|
+-------+--------------------+--------------------+---------+--------------------+---------------+--------------------+
|   8013|0000ed7c-41ab-498...|2019-12-06T19:44:...|   pablow|El Mató a un Poli...|Prenderte Fuego|                null|
|   9680|00012ab2-5bdc-41f...|2019-01-21T13:06:...| kiffkong|       Rusko & Caspa|Custard Chucker|0d3a4202-a6b1-3e3...|
|   9680|00012ab2-5bdc-41f...|2019-04-06T15:19:...| kiffkong|       Rusko & Caspa|Custard Chucker|0d3a4202-a6b1-3e3...|
|   9680|00012ab2-5bdc-41f...|2019-01-24T20:49:...| kiffkong|       Rusko & Caspa|Custard Chucker|0d3a4202-a6b1-3e3...|
|   9680|00012ab2-5bdc-41f...|2019-10-07T10:27:...| kiffkong|       Rusko & Caspa|Custard Chucker|0d3a4202-a6b1-3e3...|
+-------+--------------------+--------------------+---------+--------------------+---------------+--------------------+
only showing top 5 rows
'''
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create Spark session object
    spark = SparkSession.builder.appName('not_important').getOrCreate()

    # Get file path for the dataset to split
    file_path = sys.argv[1]

    # Calling the split function
    not_important(spark, file_path)