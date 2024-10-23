# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CarPricePrediction") \
    .getOrCreate()

# Load dataset from HDFS
df = spark.read.csv("hdfs://192.168.1.214:9000/sat5165/auto.csv", header=True, inferSchema=True)

# Data preprocessing
df = df.withColumn("horsepower", df["horsepower"].cast(DoubleType()))
df = df.withColumn("engine-size", df["engine-size"].cast(DoubleType()))
df = df.withColumn("price", df["price"].cast(DoubleType()))  # Cast price to numeric

# Handle missing values
df = df.fillna({"horsepower": df.agg({"horsepower": "mean"}).first()[0],
                "price": df.agg({"price": "mean"}).first()[0]})

# Label encoding for the price ranges
bins = [0, 10000, 20000, float("inf")]
labels = ["low", "medium", "high"]
df = df.withColumn("price_range", 
                   F.when(df.price <= 10000, "low")
                   .when((df.price > 10000) & (df.price <= 20000), "medium")
                   .otherwise("high"))

# Indexing the price range labels
indexer = StringIndexer(inputCol="price_range", outputCol="label")
df = indexer.fit(df).transform(df)

# Features selection
feature_cols = ["horsepower", "engine-size"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Split the dataset into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)

# Predictions
predictions = lr_model.transform(test_df)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print the accuracy to the console
print(f"Model accuracy: {accuracy:.2f}")

# Write the accuracy result to a text file
with open("model_accuracy.txt", "w") as file:
    file.write(f"Model accuracy: {accuracy:.2f}\n")

# Stop the Spark session
spark.stop()
