sparkDf = spark.read.format('com.databricks.spark.csv').option('header', 'true').option('inferschema', 'true').load('dbfs:/FileStore/ML/raw_dataset_new.csv')

sparkDf.write.mode("overwrite").saveAsTable("risk.raw_dataset")

pd_df = spark.sql('select * from risk.raw_dataset').toPandas()

print(pd_df)
