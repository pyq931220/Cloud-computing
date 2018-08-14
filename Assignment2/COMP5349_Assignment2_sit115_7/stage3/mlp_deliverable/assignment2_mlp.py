#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 21:50:16 2018

@author: lipenghao
"""

# Import all necessary libraries and setup the environment

from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from time import time
import numpy as np
import argparse


#load training data
spark = SparkSession \
    .builder \
    .appName("MLP__") \
    .getOrCreate() 

parser = argparse.ArgumentParser()
parser.add_argument("--hiddenLayerSize", help="the size of layers")
parser.add_argument("--output", help="the output path")
args = parser.parse_args()
output_path = args.output

train_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv'
test_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv'

train_data = spark.read.csv(train_datafile,header=False,inferSchema="true")
test_data = spark.read.csv(test_datafile,header=False,inferSchema="true").repartition(16)

#Assembler
assembler = VectorAssembler(inputCols=train_data.columns[1:],outputCol="features")

#MLP_trainer
layers = np.array(args.hiddenLayerSize.split(','), dtype=int)
trainer = MultilayerPerceptronClassifier(labelCol="_c0",featuresCol='features', \
                                         maxIter=100, layers=layers, blockSize=128,seed=1234)
#pipeline
pipeline = Pipeline(stages=[assembler, trainer])
pipelineFit = pipeline.fit(train_data)

prediction = pipelineFit.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="_c0", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction)

print("Predictions accuracy = %g, Test Error = %g" % (accuracy,(1.0 - accuracy)))
