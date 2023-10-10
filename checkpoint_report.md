### DSGA 1004 Big Data

**Group Number:** 80, **Group Members:** Dian Jin, dj928, Xinxiaomeng Liu, xl4701

#### Preprocessing and Partitioning
Parquet files storing data for interactions, users, and tracks are first loaded as Pyspark dataframes. Interactions and users dataframes are joined on `user_id`. The resulting dataframe is joined by tracks on `recording_msid`, so that for each row of data we will have information regarding the user name, track name, etc.

We then partitioned training data into `train` and `val`. A new column named `timestamp_ms` is computed as Unix timestamp in milliseconds. We used window functions and `percentile_approx` to compute the 75% quantile of timestamp for each user. Per user interactions with older timestamps that constitutes 75% of total train data into the training set. All of the rest 25% is stored as validation set. The logic is that if the training set contains data from the past while the validation set contains more recent data, the popularity based system can learn from history and predict future trends more accurately. Also, since historical and more recent data are divided per user, we ensure that the training and validation sets (as well as the test set) have equal representations of each user's behavior to avoid user cold start problem.

Both dataframes are then written to be stored on HPC as `.csv` files. We previewed the data by first reading `train.csv` and `val.csv` and showing the first 5 rows  for both to check if everything is in place.


#### Propularity Baseline Model 

We built two Popularity baseline Models, which can provide a baseline score for the ALS Model. 

**Baseline Model 1:**

The first model predicts the top 100 most listened to recording_msid simply based on the counts of every recording_msid. We applied the model to both the small training set and the full training set, predicted the top 100 most listened to recording_msid and converted the 100 recording_msid to a list.  

For the validation set (both the small validation set and the full validation set), we generated the top100 recording_msid for each user_id by first grouping by user_id and recording_msid, and counting the number of concurrences, then defining the window function to partition by user_id and sort by count in descending order and assigning row_number to each recording_msid for each user based on the count, then filtering for the top 100 recording_msid for each user and converting each user's top 100 recording_msid to a list. We mapped all the recording_msids to integer IDs and used meanAP@100 and NDCG@100 with pyspark.ml.evaluation.RankingEvaluator to evaluate the model performance.

For the test set, we did the same operations as we did for the validation set to generate the top100 recording_msid for each user_id. Then we mapped all the recording_msids to integer IDs and used meanAP@100 and NDCG@100 with pyspark.ml.evaluation.RankingEvaluator to evaluate the model performance.

**performance of the popularity baseline model (based on counts) trained on small dataset**

|            |  validation (small)  |          Test          |
|------------|----------------------|------------------------|
| MAP        | 9.424415422978865e-05| 5.883986503654102e-05  |
| NDCG       | 0.0014973332285875504| 0.0009707566025860453  |

**performance of the popularity baseline model (based on counts) trained on full dataset**

|            |  validation (full)   |          Test          |
|------------|----------------------|------------------------|
| MAP        | 0.0001442015920974522| 4.930510910365107e-05  |
| NDCG       | 0.002129654953061713 | 0.0008147233472413917  |

**Baseline Model 2:**

We also built a popularity baseline model where we predicted the top 100 most listened to recording_msid based on the average number of listens per user, that is, dividing the number of listens by the number of users (plus damping). We tried damping variable in the range from 10 to 20000 and chose the value of damping factor based on the performance of the model on the validation set. For the small dataset, we chose damping factor = 100. For the full dataset, we chose the damping factor = 10200. The performance of this model is as follows:

**performance of the popularity baseline model (based on average number of listens per user) trained on small dataset**

|            |  validation (small)  |          Test          |
|------------|----------------------|------------------------|
| MAP        | 0.0716418762010137   | 0.073172258270102994   |
| NDCG       | 0.1204607042359396   | 0.116280989979803885   |

**performance of the popularity baseline model (based on average number of listens per user) trained on full dataset**

|            |  validation (full)   |          Test          |
|------------|----------------------|------------------------|
| MAP        | 0.0086869110470725   | 0.0812809315102023856  |
| NDCG       | 0.0975616274679209   | 0.1618178697518984710  |

By comparing the performance of these two baseline models, we can see that the popularity model based on average number of listens per user performs better.