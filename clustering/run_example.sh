#!/bin/bash

#Assemble docker image. 
echo 'Running clustering docker image.'

DATASET=cifar
MODEL=resnet32
BATCH=128

docker run \
        -v /home/laude/compatibility_clustering_spectograms/database:/home/app/database \
        clustering:latest \
        -d $DATASET -m $MODEL -b $BATCH