# starting the soarj session

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import csv
import os
import sys
import pandas as pd

os.environ["PATH"] = r"C:\Windows\System32;" + os.environ.get("PATH", "")
os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Java\jdk-17")
os.environ.setdefault("SPARK_HOME", r"C:\Users\stron\Downloads\spark-4.1.2-bin-hadoop3")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
os.environ.setdefault("SPARK_LOCAL_DIRS", r"C:\temp")
os.environ["HADOOP_HOME"] = r"C:\hadoop-3.5.0"
os.environ["HADOOP_CONF_DIR"] = r"C:\hadoop-3.5.0\etc\hadoop"

spark = (
    SparkSession.builder.master("local[1]")
    .appName("Deeper_concepts")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.executor.memory", "512m")
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .config("spark.sql.warehouse.dir", r"C:\temp\spark-warehouse")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# concept 1 reading data into spark dataframe
sparktry_txt = [
    "number_of_people_served\thour_of_day\tday_of_week\tpercent_male\tpercent_female\tpercent_child",
    "956\t11\tTuesday\t4.78\t87.5\t7.72",
    "930\t13\tThursday\t10.94\t67.97\t21.09",
    "486\t13\tTuesday\t10\t75.38\t14.62",
    "364\t14\tWednesday\t7.1\t84.15\t8.74",
    "942\t9\tFriday\t5.79\t72.73\t21.49",
    "736\t11\tFriday\t6.33\t76.58\t17.09",
    "882\t11\tTuesday\t4.85\t83.98\t11.17",
    "554\t13\tTuesday\t7.69\t81.2\t11.11",
    "398\t12\tFriday\t7.25\t86.47\t6.28",
    "346\t16\tWednesday\t10.14\t48.65\t41.22",
    "236\t9\tWednesday\t4.28\t84.49\t11.23",
    "168\t9\tWednesday\t4.27\t88.41\t7.32",
    "192\t14\tWednesday\t9.82\t75\t15.18",
    "2212\t16\tTuesday\t5.09\t81.02\t13.89",
    "574\t12\tWednesday\t3.17\t85.52\t11.31",
    "862\t13\tTuesday\t5.78\t85.78\t8.44",
    "278\t16\tFriday\t6.07\t69.23\t24.7",
    "414\t12\tTuesday\t7.19\t81.29\t11.51",
    "390\t12\tWednesday\t2.62\t89.01\t8.38",
    "320\t14\tWednesday\t5.65\t82.26\t12.1",
    "234\t15\tWednesday\t5.8\t85.71\t8.48",
    "330\t11\tFriday\t7.63\t73.28\t19.08",
    "304\t11\tTuesday\t3.95\t90.79\t5.26",
    "954\t14\tTuesday\t6.67\t86.22\t7.11",
    "292\t13\tTuesday\t4.67\t81.31\t14.02",
    "354\t11\tTuesday\t3.94\t86.02\t10.04",
    "388\t13\tMonday\t9.92\t80.99\t9.09",
    "306\t14\tThursday\t4.21\t84.58\t11.21"
]

with open("sparktry.txt", "w", encoding="utf-8") as f:
    for line in sparktry_txt:
        f.write(line + "\n")

incoming_sparktry_df = spark.read.csv("sparktry.txt", sep="\t", header=True, inferSchema=True)
print("[Concept 1] OPENING THE DELIVERY DATASET - READING DATA INTO SPARK DATAFRAME and displaying the first 5 rows of the dataframe")
incoming_sparktry_df.show(5)

cleaned_sparktry_df = incoming_sparktry_df.withColumn("day_of_week", F.lower(F.initcap(F.col("day_of_week"))))
print("[Concept 1] CLEANED DATAFRAME - NORMALIZED day_of_week")
cleaned_sparktry_df.show(5)
# CONCEPT 4 : PERFORMING AGGREGATIONS AND GROUPIMNG DATA 

count_by_day_df = cleaned_sparktry_df. \
groupBy("day_of_week"). \
agg(
    F.count("*").alias("count_by_day"),
    )

print("[Concept 4] AGGREGATED DATA - COUNT OF DELIVERIES BY DAY OF THE WEEK")
count_by_day_df.show(5)

# CONCCEPT 5 : WRITING DATA FROM SPARK

print("[Concept 5] DATA READY TO VIEW IN TERMINAL")
count_by_day_df.collect()
print(count_by_day_df.collect())

# Concept 6: manipulating dataset with Spark SQL
cleaned_sparktry_df.createOrReplaceTempView("sparktry_data")
sql_result_df = spark.sql("""
    SELECT day_of_week, COUNT(*) AS count_by_day
    FROM sparktry_data
    GROUP BY day_of_week
    ORDER BY count_by_day DESC
""")
print("[Concept 6] SPARK SQL RESULT")
sql_result_df.show()

# Concept 7: join, union, and windows in Spark

# Example 1: JOIN two DataFrames
customers = spark.range(3).select(
    (F.col("id") + 1).alias("customer_id"),
    F.concat(F.lit("Customer"), F.col("id").cast("string")).alias("customer_name")
)

orders = (
    spark.range(4)
    .select((F.col("id") + 1).alias("customer_id"), (F.col("id") * 100).alias("amount"))
    .filter(F.col("customer_id").isin([1, 2, 4]))
)

joined_df = customers.join(orders, "customer_id", "left")
print("[Concept 7] JOIN RESULT")
joined_df.show()

# Example 2: UNION two DataFrames with same schema
df1 = spark.range(2).select((F.col("id") + 1).alias("id"), F.concat(F.lit("A"), F.col("id").cast("string")).alias("value"))
df2 = spark.range(2).select((F.col("id") + 3).alias("id"), F.concat(F.lit("B"), F.col("id").cast("string")).alias("value"))
union_df = df1.union(df2)
print("[Concept 7] UNION RESULT")
union_df.show()

# Example 3: WINDOW FUNCTION
window_spec = Window.partitionBy("day_of_week").orderBy(F.col("number_of_people_served").desc())
ranked_df = cleaned_sparktry_df.withColumn("rank", F.rank().over(window_spec))
print("[Concept 7] WINDOW FUNCTION RESULT")
ranked_df.select("day_of_week", "number_of_people_served", "rank").show(10)

# Concept 8A: UDF and pandas UDF in Spark (User Defined Functions)

# 8A UDF ROW BY ROW PROCESSING

def square_value(x):
    return x * x

square_udf = udf(square_value, "int")

udf_result_df = cleaned_sparktry_df.select("number_of_people_served").withColumn(
    "squared_people_served",
    square_udf(F.col("number_of_people_served"))
)
print("[Concept 8A] STANDARD UDF RESULT")
udf_result_df.show(5)

# Pandas UDF

@pandas_udf("double")
def double_pandas_udf(value: pd.Series) -> pd.Series:
    return value * 2

pandas_udf_result_df = cleaned_sparktry_df.select("number_of_people_served").withColumn(
    "doubled_people_served",
    double_pandas_udf(F.col("number_of_people_served"))
)
print("[Concept 8A] PANDAS UDF RESULT")
pandas_udf_result_df.show(5)

spark.stop()
