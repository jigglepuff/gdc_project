import numpy as np
import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,IndexToString
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.ml.feature import IDF
# from pyspark.ml.feature import DCT
# from pyspark.ml.feature import PolynomialExpansion
# from pyspark.ml.feature import ChiSqSelector
from pyspark.ml import Pipeline
import itertools as it
import pyspark.sql.functions as f


spark = SparkSession.builder.appName("DiseaseApp").getOrCreate()

seed = 0



SparkContext.setSystemProperty('spark.executor.memory', '10g')

# helper function to get all stored variables
def list_dataframes():
    from pyspark.sql import DataFrame
    return [k for (k, v) in globals().items() if isinstance(v, DataFrame)]



from botocore.exceptions import ClientError

def check(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404
    return True


# In[7]:


def getTrainingMetrics(trainingSummary,printout=True):

    print("F-measure by label:")
    for i, f in enumerate(trainingSummary.fMeasureByLabel()):
        print("label %d: %s" % (i, f))

    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    if printout is True:
        print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
    return {"accuracy": accuracy, "fpr": falsePositiveRate, "tpr": truePositiveRate, "fmeasure": fMeasure,             "precision": precision, "recall": recall}


# In[89]:


# read data file
mirna_path = 's3://gdc-emr0/mirna_filtered_matrix.csv'
cpv_path = 's3://gdc-emr0/cpv_filtered_matrix.csv'
mrna_path = 's3://gdc-emr0/mrna_filtered_matrix.csv'
# read mirna
df_mirna = spark.read.option("maxColumns", 22400).csv(
    mirna_path, header=True, sep = ',',mode="DROPMALFORMED",inferSchema=True)
# read cpv
df_cpv = spark.read.option("maxColumns", 22400).csv(
    cpv_path, header=True, sep = ',',mode="DROPMALFORMED",inferSchema=True)
# # read mrna
# df_mrna = spark.read.option("maxColumns", 22400).csv(
#     mrna_path, header=True, sep = ',',mode="DROPMALFORMED",inferSchema=True)


# In[104]:


df_cpv = df_cpv.toDF(*(c.replace('.', '_') for c in df_cpv.columns))
# df_mrna = df_mrna.toDF(*(c.replace('.', '_') for c in df_mrna.columns))


# In[91]:


# group columns by label, identifier and feature
label_columns = ['sample_type', 'disease_type', 'primary_diagnosis']
mirna_identifier_columns = ['sample_id','case_id']
mirna_feature_columns = [x for x in df_mirna.columns if x not in (label_columns+mirna_identifier_columns)]


# In[105]:


# group columns by label, identifier and feature
cpv_identifier_columns= ['_c0','sample_id','case_id']
cpv_feature_columns = [x for x in df_cpv.columns if x not in (label_columns+cpv_identifier_columns)]
df_cpv = df_cpv.withColumnRenamed("sample_type", "sample_type_cpv").withColumnRenamed("disease_type", "disease_type_cpv").withColumnRenamed("primary_diagnosis","primary_diagnosis_cpv").withColumnRenamed("case_id", "case_id_cpv")
cpv_label_columns = ['sample_type_cpv', 'disease_type_cpv', 'primary_diagnosis_cpv']
cpv_identifier_columns=['_c0','sample_id','case_id_cpv']


# In[ ]:


# # group columns by label, identifier and feature
# # cpv_label_columns = ['sample_type', 'disease_type', 'primary_diagnosis']
# mrna_identifier_columns= ['_c0','sample_id','case_id']
# mrna_feature_columns = [x for x in df_mrna.columns if x not in (label_columns+mrna_identifier_columns)]


# In[107]:


# convert features into (sparse) vectors
# mirna
assembler = VectorAssembler(inputCols=mirna_feature_columns, outputCol='features_mirna')
df_mirna = assembler.transform(df_mirna)
df_mirna=df_mirna.drop(*mirna_feature_columns)
# eventually we should store/load from HDFS


# In[108]:


# cpv
assembler = VectorAssembler(inputCols=cpv_feature_columns, outputCol='features_cpv')
df_cpv = assembler.transform(df_cpv)
df_cpv = df_cpv.drop(*cpv_feature_columns)


# In[110]:


df = df_cpv.join(df_mirna, on=['sample_id'], how='left_outer')


# In[114]:


# convert string labels to numerical labels
# keep track of the column names of numerical labels
label_idx_columns = [s + '_idx' for s in cpv_label_columns]

# declare indexers for 3 columns
labelIndexer = [StringIndexer(inputCol=column, outputCol=column+'_idx',handleInvalid="error",
                              stringOrderType="frequencyDesc") for column in cpv_label_columns ]
# pipeline is needed to process a list of 3 labels
pipeline = Pipeline(stages=labelIndexer)
# transform 3 label columns from string to number catagoies 
df = pipeline.fit(df).transform(df)

# create dictionary containing 3 label lists
label_dict = {c.name: c.metadata["ml_attr"]["vals"]
for c in df.schema.fields if c.name.endswith("_idx")}


# In[140]:


# full-feature
full_feature_columns=['features_mirna','features_cpv']
assembler = VectorAssembler(inputCols=full_feature_columns, outputCol='full_features')
df = assembler.transform(df)
# df = df.drop(*cpv_feature_columns)


# In[124]:


# Standardization 
scaler = StandardScaler(inputCol='full_features', outputCol='scaledFeatures', withStd=True, withMean=True)

# # # Convert indexed labels back to original labels
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=label_dict['disease_type_cpv_idx'])


# In[125]:


# Evaluators
f1_evaluator=MulticlassClassificationEvaluator(predictionCol="prediction",labelCol='disease_type_cpv_idx', metricName='f1')
acc_evaluator=MulticlassClassificationEvaluator(predictionCol="prediction",labelCol='disease_type_cpv_idx', metricName='accuracy')
precision_evaluator=MulticlassClassificationEvaluator(predictionCol="prediction",labelCol='disease_type_cpv_idx', metricName='weightedPrecision')
recall_evaluator=MulticlassClassificationEvaluator(predictionCol="prediction",labelCol='disease_type_cpv_idx', metricName='weightedRecall')


# In[141]:


# test/train split 
Xtest,Xtrain = df.randomSplit([0.3, 0.7], seed)


# In[142]:


s3 = boto3.client('s3')
stdmodelPath = 'full_std_model/data/_SUCCESS'
if check(s3, 'gdc-emr0', stdmodelPath) == False:
    print("saving StandardScalar model...")
    stdmodel = scaler.fit(Xtrain)
    stdmodel.save('s3://gdc-emr0/full_std_model')
else:
    from pyspark.ml.feature import StandardScalerModel
    print("loading StandardScalar model...")
    stdmodel = StandardScalerModel.load(stdmodelPath)


# In[144]:


Xtrain = stdmodel.transform(Xtrain)
Xtest = stdmodel.transform(Xtest)


# In[ ]:


# to save
# df.rdd.saveAsPickleFile(filename)
# to load
#pickleRdd = sc.pickleFile(filename).collect()
# df2 = spark.createDataFrame(pickleRdd)


# ### GBT Forest 

# In[152]:


from pyspark.ml.classification import GBTClassifier
# Random Forest Classifier
gbt = GBTClassifier(labelCol="disease_type_idx", featuresCol="percentFeatures", cacheNodeIds=True, maxIter=2,\
                   seed = seed, stepSize=0.2, maxDepth=5,maxBins=40)



# # Hyperparameters to test
paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5,10])\
            .build()

# # K-fold cross validation 
crossval = CrossValidator(estimator=gbt,
                          estimatorParamMaps=paramGrid,
                          evaluator=f1_evaluator,
                          numFolds=2,seed=seed)  # use 3+ folds in practice

# # Put steps in a pipeline
pipeline = Pipeline(stages=[crossval])

# In[ ]:


# train model
pipModel = pipeline.fit(Xtrain)


# In[150]:


# predict
predictions = pipModel.transform(Xtest)


# In[ ]:


# get hyperparameters for best model
cvModel = pipModel.stages[-1]
bestParams = cvModel.extractParamMap()
# print ('Best Param (regParam): ', bestModel._java_obj.getRegParam())
bestModel = cvModel.bestModel
feature_importance = bestModel.featureImportances
# num_trees = bestModel.getNumTrees
# tree_weights = bestModel.treeWeights
# trees =  bestModel.trees
# # save model
# bestModel.save('cpv1600_rf')


# In[ ]:


s3 = boto3.client('s3')
modelPath = 'full_gbt_test_model/data/_SUCCESS'
if check(s3, 'gdc-emr0', stdmodelPath) == False:
    print("saving Random Forest model...")
    bestModel.save('s3://gdc-emr0/full_gbt_test_model')
else:
    print(modelPath+" already exists...")


# In[ ]:


# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=label_dict['disease_type_cpv_idx'])
predictions = labelConverter.transform(predictions)


# In[ ]:


feature_rd_df = pd.DataFrame(feature_importance, columns=['feature_importance'])
# cpv_feature_columns+mirna_feature_columns
from io import StringIO

csv_buffer = StringIO()
feature_rd_df.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object('gdc-emr0', 'gbt_feature_impt.csv').put(Body=csv_buffer.getvalue())

spark.stop()
