import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Imputer
from awsglue.dynamicframe import DynamicFrame

# Get job name from the arguments
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def main():
    # Define S3 paths
    input_path = "s3://harish-3f567b3c-b2c6-4111-a5e1-f70d788d11a9/updated_dataset.csv"
    output_path = "s3://harish-3f567b3c-b2c6-4111-a5e1-f70d788d11a9/processed-data/"

    # Read data from S3 using GlueContext
    datasource0 = glueContext.create_dynamic_frame.from_options(
        format_options={"withHeader": True},
        connection_type="s3",
        format="csv",
        connection_options={"paths": [input_path]}
    )
    df = datasource0.toDF()

    # Standardize column names
    df = df.withColumnRenamed("What are your savings objectives?", "Savings_Objectives")

    # Data Cleaning: Remove or replace non-numeric values
    df = df.withColumn("age", regexp_replace("age", "[^0-9]", "")) \
           .withColumn("Mutual_Funds", regexp_replace("Mutual_Funds", "[^0-9]", "")) \
           .withColumn("Equity_Market", regexp_replace("Equity_Market", "[^0-9]", ""))

    # Convert columns to integer types
    df = df.withColumn("age", when(col("age") == "", None).otherwise(col("age").cast(IntegerType()))) \
           .withColumn("Mutual_Funds", when(col("Mutual_Funds") == "", None).otherwise(col("Mutual_Funds").cast(IntegerType()))) \
           .withColumn("Equity_Market", when(col("Equity_Market") == "", None).otherwise(col("Equity_Market").cast(IntegerType())))

    # Handle Missing Values
    # Fill missing values for numerical columns with mean
    numerical_cols = ['age', 'Mutual_Funds', 'Equity_Market']
    imputer = Imputer(inputCols=numerical_cols, outputCols=[col + "_imputed" for col in numerical_cols])
    df = imputer.fit(df).transform(df)

    # Feature Engineering - Example: Create a new feature 'Investment_Total'
    df = df.withColumn('Investment_Total', col('Mutual_Funds_imputed') + col('Equity_Market_imputed'))

    # Drop intermediate imputed columns if not needed
    df = df.drop(*[col + "_imputed" for col in numerical_cols])

    # Convert DataFrame back to DynamicFrame
    dynamic_frame = DynamicFrame.fromDF(df, glueContext, "dynamic_frame")

    # Save Processed Data Back to S3 in CSV format
    df.write.csv(output_path, header=True, mode='overwrite')

if __name__ == "__main__":
    main()

# Commit the job to signal successful completion
job.commit()
