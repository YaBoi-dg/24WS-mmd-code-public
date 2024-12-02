from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, sum as _sum, avg, pow, sqrt

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

from pyspark.sql import functions as pysf

spark = SparkSession.builder.appName("Audioscrobbler Analysis").getOrCreate()

user_artist_data_path = "user_artist_data_small.txt"
artist_alias_path = "artist_alias_small.txt"
artist_data_path = "artist_data_small.txt"

user_artist_df = spark.read.option("sep", "\t").csv(user_artist_data_path, schema="userid INT, artistid INT, playcount INT")
artist_alias_df = spark.read.option("sep", "\t").csv(artist_alias_path, schema="badid INT, goodid INT")
artist_data_df = spark.read.option("sep", "\t").csv(artist_data_path, schema="artistid INT, artistname STRING")

# 7 a
artist_alias_dict = artist_alias_df.rdd.collectAsMap()
broadcast_alias = spark.sparkContext.broadcast(artist_alias_dict)

def map_artist(artistid):
    return broadcast_alias.value.get(artistid, artistid)

map_artist_udf = udf(map_artist, IntegerType())
user_artist_clean_df = user_artist_df.withColumn("artistid", map_artist_udf(col("artistid")))

utility_matrix = user_artist_clean_df.groupBy("userid", "artistid").agg(_sum("playcount").alias("total_playcount"))

# 7 b
user_stats = utility_matrix.groupBy("userid").agg(
    avg("total_playcount").alias("mean_playcount"),
    sqrt(_sum(pow(col("total_playcount"), 2))).alias("norm")
)

user_artist_stats = utility_matrix.join(user_stats, "userid")
similarity_df = user_artist_stats.alias("u1").join(
    user_artist_stats.alias("u2"),
    (pysf.col("u1.artistid") == pysf.col("u2.artistid")) & (pysf.col("u1.userid") < pysf.col("u2.userid"))
).groupBy("u1.userid", "u2.userid").agg(
    (pysf.sum((pysf.col("u1.total_playcount") - pysf.col("u1.mean_playcount")) *
           (pysf.col("u2.total_playcount") - pysf.col("u2.mean_playcount"))) /
     (pysf.sqrt(pysf.sum(pysf.pow(pysf.col("u1.total_playcount") - pysf.col("u1.mean_playcount"), 2))) *
      pysf.sqrt(pysf.sum(pysf.pow(pysf.col("u2.total_playcount") - pysf.col("u2.mean_playcount"), 2))))).alias("pearson_corr")
)

# 7 c
def find_top_k_similar_users(similarity_df, user_id, k):
    return similarity_df.filter(col("u1.userid") == user_id).orderBy(col("pearson_corr").desc()).limit(k)

# 7 d
