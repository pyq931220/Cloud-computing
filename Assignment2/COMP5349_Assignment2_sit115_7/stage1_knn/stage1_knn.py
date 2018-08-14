from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,struct
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler,PCA
from pyspark.accumulators import AccumulatorParam
from pyspark.ml import Pipeline
from pyspark.sql.types import *
import pyspark
import numpy as np
import argparse

#initialize sparksession and get input arguments
spark = SparkSession.builder.appName('Cloud Computing Assignment2').getOrCreate()
parser = argparse.ArgumentParser()
parser.add_argument("--output", help="the output path")
parser.add_argument("--num_pca", help="pca value", type = int)
parser.add_argument("--num_k", help="num_nearest_neighbour", type = int)
parser.add_argument("--num_partitions", help="number of partitions", type = int)
args = parser.parse_args()
num_pca = args.num_pca
num_k = args.num_k
output_path = args.output
num_partitions = args.num_partitions
train_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv'
test_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv'
# train_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/demo/MNIST-sample/Train-6000.csv'
# test_datafile = 'hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/demo/MNIST-sample/Test-1000.csv'

#knn prediction for udf function
def knnprediction(d,tl):
    dists = np.sum((l.value[1]-np.array(d))**2,axis=1)
    sortedlabels = np.take(l.value[0], np.argpartition(dists,num_k)[:num_k])
    p = int(np.bincount(sortedlabels).argmax())
    global cm
    x = np.array([0]*100)
    x[int(tl)*10+p] = 1
    cm += x
    return p

# Define customized accumulator
class VectorAccumulatorParam(AccumulatorParam):
	def zero(self, value):
		return [0]* len(value)
	def addInPlace(self, val1, val2):
			val1 += val2
			return val1

#read data from csv
train_data = spark.read.csv(train_datafile,header=False,inferSchema="true")
test_data = spark.read.csv(test_datafile,header=False,inferSchema="true").repartition(num_partitions).persist(pyspark.StorageLevel.MEMORY_ONLY)
assembler = VectorAssembler(inputCols=train_data.columns[1:],outputCol="features")
pca = PCA(k = num_pca, inputCol="features", outputCol="pcafeatures")

# pipeline with vector assembler and pca
pipeline = Pipeline (stages=[assembler,pca])
pipemodel = pipeline.fit(train_data)
train_vectors = pipemodel.transform(train_data).select("_c0","pcafeatures")
test_vectors = pipemodel.transform(test_data).select(test_data["_c0"].alias("test_label"),"pcafeatures")

#broadcast the trainning data to all nodes for e-distance
train_info = [y for x in train_vectors.collect() for y in x]
broaditems = []
broaditems.extend((np.array(train_info[0::2]),np.array(train_info[1::2])))
l = spark.sparkContext.broadcast(broaditems)

# set up accumulator for confusion matrix
cmatrix = np.array([0]*100)
cm = spark.sparkContext.accumulator(cmatrix,VectorAccumulatorParam())

#knn prediction and add prediction label to dataframe
knn_udf = udf(knnprediction,IntegerType())
knnpredictions = test_vectors.withColumn("predict_label",
	knn_udf(test_vectors["pcafeatures"],test_vectors["test_label"])).select("predict_label","test_label")
test_data.unpersist()
l.unpersist()
knnpredictions.rdd.map(lambda x: "prediction: "+str(x[0])+"  test label: "+str(x[1])).saveAsTextFile(output_path)

#calculate accuracy, precision, recall and F1score and print
matrix = np.array(cm.value).reshape(10,10)
index = np.array([np.arange(10)])
precision = np.round(np.array([[matrix[n][n]/np.sum(matrix.T,axis = 1)[n] for n in index[0]]]),3)
recall = np.round(np.array([[matrix[n][n]/np.sum(matrix, axis = 1)[n] for n in index[0]]]),3)
fscore = np.round(np.array([[2*precision[0][n]*recall[0][n]/(precision[0][n]+recall[0][n])for n in index[0]]]),3)

#convert the result to dataframe and print out the evaluation
combined = np.concatenate((index,precision,recall,fscore),axis = 0).T.tolist()
schema = StructType([
	StructField("label",FloatType(),True),
	StructField("precision", FloatType(), True),
	StructField("recall", FloatType(), True),
	StructField("F1-score", FloatType(), True)])
print("confusion matrix for prediction:")
print(matrix)
spark.createDataFrame(combined, schema).show()

