#%matplotlib inline
#import findspark
#findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import numpy as np
#import matplotlib.pyplot as plt
import operator as opt
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler,PCA, HashingTF, Tokenizer
import argparse
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
from pyspark.ml import Pipeline

#The process for inputing dimension values in the commend line
parser=argparse.ArgumentParser()
parser.add_argument("--num_pca", help="pca dimension value", type = int)
args=parser.parse_args()

#Load the train data and test data for this no dimensional reduction version Logistic Regression code
spark = SparkSession.builder.appName('Logistic Regression_PCA_50_2_4').getOrCreate()
train_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv'
test_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv'
num_pca=args.num_pca
#Write the assemble function for assembling the input data into Vector type.
#'_c0' is the position of the label. All the elements included in the 2nd and latter positions are combined as a vector.
train_data = spark.read.csv(train_datafile,header=False,inferSchema="true")
test_data = spark.read.csv(test_datafile,header=False,inferSchema="true")
#To combine assembler, PCA and Logistic Regression as a pipeline
assembler = VectorAssembler(inputCols=train_data.columns[1:],outputCol="features")
pca = PCA(k = num_pca, inputCol="features", outputCol="pcafeatures")
#Set the Logistic Regression model using the default setup. Max_Iter=100;regPam=0.1, elasticNetParam=1.0)
lr = LogisticRegression(labelCol = "_c0", featuresCol = "pcafeatures")
pipeline = Pipeline(stages=[assembler, pca, lr])

# Fit the pipeline to training documents.
Model = pipeline.fit(train_data)
# Print the coefficients and intercept for multinomial logistic regression

result = Model.transform(test_data).select(test_data["_c0"].alias("label"),"features","prediction")

predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
