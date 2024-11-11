from ucimlrepo import fetch_ucirepo 
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Task 6a
spark = SparkSession.builder.appName("Adult Dataset Import").getOrCreate()

# Load the dataset into RDDs
file_path = "./e03/adult/adult.data"  # Update this with your dataset path
rdd = spark.sparkContext.textFile(file_path)

# Split the data into columns (assuming the data is comma-separated)
rdd_split = rdd.map(lambda line: line.split(", "))

# Define the schema based on thepip "adult" dataset attributes
schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", IntegerType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", IntegerType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", IntegerType(), True),
    StructField("capital_loss", IntegerType(), True),
    StructField("hours_per_week", IntegerType(), True),
    StructField("native_country", StringType(), True),
    StructField("income", StringType(), True)
])

# Convert RDD to DataFrame
df = spark.createDataFrame(rdd_split, schema)

# Show the first few rows of the DataFrame
df.show(5)

# Basic cleanup: Remove rows with null or empty values
df_cleaned = df.na.drop()

# Display cleaned DataFrame
df_cleaned.show(5)

# Print schema to verify the columns
df_cleaned.printSchema()