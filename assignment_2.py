#Name: Ng Wei Heng Jared
#jared.ng.2019

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id, col, udf
from pyspark.sql.types import StringType
from graphframes import *
import string

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk.stem.porter import PorterStemmer

from collections import Counter


def cleanitup(text):
    from nltk.corpus import stopwords
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    filtered_sentence = (" ").join(words)
    final_text = filtered_sentence
    return (final_text)


spark = SparkSession.builder.appName('sg.edu.smu.is459.assignment2').getOrCreate()


#Cleaning text function to udf
colcleanitup = udf(lambda z: cleanitup(z), StringType())
spark.udf.register("colcleanitup", colcleanitup)

# Load data
posts_df = spark.read.load('/user/jared/parquet-input/hardwarezone.parquet')

posts_df.show(10)
#posts_df = posts_df.limit(100)

# Clean the dataframe by removing rows with any null value and tokenizing text
posts_df = posts_df.na.drop()
posts_df_cleaned = posts_df.withColumn("scontent",colcleanitup(col("content")))
posts_df_cleaned.show(10)

#consolidate authors and their contents
print("author_and_content_consolidated")
author_and_content_consolidated = posts_df_cleaned.drop("topic","content")
author_and_content_consolidated.show()
author_and_content_consolidated_rdd = author_and_content_consolidated.rdd
author_and_content_rdd= author_and_content_consolidated_rdd.reduceByKey(lambda x, y: x +" "+ y)
print("author_and_content RDD REDUCED")

# Find distinct users
#distinct_author = spark.sql("SELECT DISTINCT author FROM posts")
author_df = posts_df_cleaned.select('author').distinct() #add to topic
#df use show, rdd use collect
print('Author number :' + str(author_df.count()))

# Assign ID to the users
author_id = author_df.withColumn('id', monotonically_increasing_id())

# Construct connection between post and author
left_df = posts_df_cleaned.select('topic', 'author', 'scontent') \
    .withColumnRenamed("topic","ltopic") \
    .withColumnRenamed("author","src_author") \
    .withColumnRenamed("scontent","src_content")

right_df =  left_df.withColumnRenamed('ltopic', 'rtopic') \
    .withColumnRenamed('src_author', 'dst_author')

print("left_df")
left_df.show(10)
print("right_df")
right_df.show(10)

#  Self join on topic to build connection between authors
author_to_author = left_df. \
    join(right_df, left_df.ltopic == right_df.rtopic) \
    .select(left_df.src_author, right_df.dst_author) \
    .distinct()
edge_num = author_to_author.count()
print('Number of edges with duplicate : ' + str(edge_num))


# Convert it into ids

#vertices
id_to_author = author_to_author \
    .join(author_id, author_to_author.src_author == author_id.author) \
    .select(author_to_author.dst_author, author_id.id) \
    .withColumnRenamed('id','src')

#edge
id_to_id = id_to_author \
    .join(author_id, id_to_author.dst_author == author_id.author) \
    .select(id_to_author.src, author_id.id) \
    .withColumnRenamed('id', 'dst')
    

id_to_id = id_to_id.filter(id_to_id.src >= id_to_id.dst).distinct()
id_to_id.cache()

print("Number of edges without duplciate :" + str(id_to_id.count()))


# Build graph with RDDs
graph = GraphFrame(author_id, id_to_id)
# For complex graph queries, e.g., connected components, you need to set
# the checkopoint directory on HDFS, so Spark can handle failures.
# Remember to change to a valid directory in your HDFS
spark.sparkContext.setCheckpointDir('/user/jared/spark-checkpoint')

# The rest is your work, guys

#Connected components
print("Connected Components")
result = graph.connectedComponents()
result.show()
connected_count = result.groupBy('component').count().orderBy('count', ascending=False)
connected_count = connected_count.withColumnRenamed("count", "connected_count").withColumnRenamed("component", "connected_component")
print("Connected Components Answer")
connected_count.show()

#Frequently occuring words by component
print("Frequently occuring words by component")
result_without_id = result.drop("id")
result_rdd = result_without_id.rdd
posts_consolidated_rdd =  result_rdd.join(author_and_content_rdd)

content_by_component=posts_consolidated_rdd.map(lambda x: 
    (x[1][0],x[1][1])
    ) 

reduced_content= content_by_component.reduceByKey(lambda x, y: x +" "+ y)

def word_count(words):
    word_list = words.split(" ")
    counted = Counter(word_list)
    return counted.most_common()

reduced_content_2=reduced_content.map(lambda x: 
    (x[0], word_count(x[1]))
    ) 

popular_words_df = spark.createDataFrame(reduced_content_2).toDF("component", "frequent_words")
print("Frequently occuring words by component answer")
popular_words_df.show() #truncate = False to view all words


#Triangle count

print("Triangle count per community")
results_triangle = graph.triangleCount()
results_triangle.show()

consolidated_with_triangles = result.join(results_triangle,result.id ==  results_triangle.id,"inner")
consolidated_with_triangles_rdd = consolidated_with_triangles.rdd
triangle_per_community_rdd=consolidated_with_triangles_rdd.map(lambda x: (x[2],x[3])) 

#Total triangles of each user within a community
consolidated_with_triangles_rdd= triangle_per_community_rdd.reduceByKey(lambda x, y: x + y)
triangle_per_community_df = spark.createDataFrame(consolidated_with_triangles_rdd).toDF("component", "triangles")

endDF = triangle_per_community_df.join(connected_count,connected_count.connected_component ==  triangle_per_community_df.component,"inner")

endDF = endDF.drop("connected_component")
endDF_rdd = endDF.rdd
endDF_rdd = endDF_rdd.map(lambda x: (x[0],x[1]/x[2]))
last_df = spark.createDataFrame(endDF_rdd).toDF("component", "average_num_of_triangles")
print("Average number of triangles per community ")
last_df.show()



#Strange facts about communities
#Hypothesis: Smaller communities type longer messages

def length_of_comments(comment):
    comments_list = comment.split(" ")
    length = len(comments_list)
    return length

total_length_by_component=reduced_content.map(lambda x: 
    (x[0], length_of_comments(x[1]))
    ) 

length_df = spark.createDataFrame(total_length_by_component).toDF("component", "length_of_content")

consolidated_length = length_df.join(connected_count,connected_count.connected_component ==  length_df.component,"inner")
consolidated_length_rdd = consolidated_length.rdd
consolidated_length_rdd = consolidated_length_rdd.map(lambda x: (x[3],x[1]/x[3]))
length_df = spark.createDataFrame(consolidated_length_rdd).toDF("size_of_community","average_length_of_content")
print("Strange facts about communities:\nHypothesis: Smaller communities type longer messages")
print("Size of commmunity and the average length of content")
length_df.sort(length_df.average_length_of_content.desc()).show()