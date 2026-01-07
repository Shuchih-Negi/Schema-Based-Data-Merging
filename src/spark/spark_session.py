from pyspark.sql import SparkSession

def get_spark(app_name="MasterData"):
    spark = (SparkSession.builder.appName(app_name).master("local[*]").config("spark.sql.parquet.mergeSchema", "true").getOrCreate()) #basically merge schema sees if any one column is missing in a parquet file it  fills empty with    NULL and checks all the files but  affects performance
    
    return spark
