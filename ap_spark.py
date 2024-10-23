from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder \
    .appName("CarPricePrediction") \
    .config("spark.executor.memoryOverhead", "512") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

df = spark.read.csv("hdfs://192.168.13.89:9000/sat5165/auto.csv", header=True, inferSchema=True)

# Data preprocessing
df = df.withColumn("horsepower", df["horsepower"].cast(DoubleType()))
df = df.withColumn("engine-size", df["engine-size"].cast(DoubleType()))
df = df.withColumn("price", df["price"].cast(DoubleType()))  # Cast price to numeric

# Handle missing values
df = df.fillna({"horsepower": df.agg({"horsepower": "mean"}).first()[0],
                "price": df.agg({"price": "mean"}).first()[0]})

# Feature selection and encoding
selected_features = ["horsepower", "engine-size", "num-of-doors", "fuel-type", "price"]
df = df.select(selected_features)

# Convert categorical variables
fuel_indexer = StringIndexer(inputCol="fuel-type", outputCol="fuel-type-index")
df = fuel_indexer.fit(df).transform(df)

fuel_encoder = OneHotEncoder(inputCols=["fuel-type-index"], outputCols=["fuel-type-encoded"])
df = fuel_encoder.fit(df).transform(df)

final_columns = ["horsepower", "engine-size", "num-of-doors", "fuel-type-encoded", "price"]
df = df.select(final_columns)

# Put features into a single vector
assembler = VectorAssembler(inputCols=["horsepower", "engine-size", "fuel-type-encoded"], outputCol="final_features")
df = assembler.transform(df)

train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Linear Regression
lr = LinearRegression(featuresCol="final_features", labelCol="price")
lr_model = lr.fit(train_data)

predictions = lr_model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

# Save RMSE to an output file
output_file = "/home/sat3812/Desktop/project/rmse_output.txt"

with open(output_file, "w") as f:
    f.write(f"Root Mean Squared Error (RMSE) on data: {rmse}\n")

# Check the number of predictions
print(f"Number of predictions: {predictions.count()}")

# Show some predictions to make sure data is there
predictions.select("prediction", "price").show()

# Save the predictions to a CSV file
predictions.select("prediction", "price") \
    .write.mode("overwrite") \
    .csv("/home/sat3812/Desktop/project/predictions", header=True)

# Save the linear regression model
lr_model.save("/home/sat3812/Desktop/project/lr_model")
