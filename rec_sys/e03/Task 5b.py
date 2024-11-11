from pyspark import SparkContext, SparkConf

# Configure Spark
conf = SparkConf().setAppName("TransformationActionExamples").setMaster("local[*]")
sc = SparkContext(conf=conf)

# Initial data for the RDD
data = [
    "apple banana apple", 
    "orange banana orange", 
    "apple orange", 
    "banana apple orange"
]

# Create an RDD from the initial data
rdd = sc.parallelize(data)

# Transformation using flatMap()
words_rdd = rdd.flatMap(lambda sentence: sentence.split(" "))
print("flatMap Result:", words_rdd.collect())

# Transformation using reduceByKey() to count word frequency
word_pairs_rdd = words_rdd.map(lambda word: (word, 1))
word_counts_rdd = word_pairs_rdd.reduceByKey(lambda x, y: x + y)
print("reduceByKey Result:", word_counts_rdd.collect())

# Transformation using distinct() to get unique words
distinct_words_rdd = words_rdd.distinct()
print("distinct Result:", distinct_words_rdd.collect())

# Action using count() to get total number of words
total_words = words_rdd.count()
print("count Result:", total_words)

# Action using collect() to retrieve all words in the RDD
all_words = words_rdd.collect()
print("collect Result:", all_words)

# Action using take() to get the first 5 words
first_five_words = words_rdd.take(5)
print("take Result:", first_five_words)

# Stop the SparkContext
sc.stop()