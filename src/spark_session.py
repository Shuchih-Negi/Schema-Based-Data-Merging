from pyspark.sql import SparkSession

def get_spark(app_name="MasterData"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.parquet.mergeSchema", "true")
        .getOrCreate()
    )
    return spark
