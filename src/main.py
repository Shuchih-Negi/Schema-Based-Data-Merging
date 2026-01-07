from  pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from spark.spark_session import get_spark
import sys
from  profile_gen import profile_column
from column_mapper import ColumnMapper


spark   = SparkSession.builder.appName('MasterData').getOrCreate()

MASTER_TABLE_PATH = '../data/master/'

df2  = spark.read.parquet("../data/master/")

df = spark.read.csv(f"../data/upload/{sys.argv[1]}",header=True,inferSchema=True)

profile_uploaded = profile_column(df)

'''for i in profile_uploaded:
    print(i+ " ")
    print( profile_uploaded[i])
    print("\n")
'''
profile_master = profile_column(df2)

mapper = ColumnMapper(profile_uploaded, profile_master)
result = mapper.run()


print(result["mapping"])
print(result["unmatched_uploaded"])
print(result["unmatched_master"])
#how do   i access the uploaded csv??



