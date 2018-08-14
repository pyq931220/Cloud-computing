#!/bin/bash

spark-submit  \
  --master local[4] \
  --num-executors 8 \
  --executor-cores 2 \
  stage1_knn.py \
   --output "spark" \
   --num_pca 50 \
   --num_k 5 \
   --num_partitions 16
