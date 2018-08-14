#!/bin/bash

spark-submit  \
  --master yarn \
  --num-executors 4 \
  --executor-cores 2 \
  assignment2_mlp.py \
   --output "spark" \
   --hiddenLayerSize 784,50,10    