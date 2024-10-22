#!/bin/bash

#Assemble docker image. 
echo 'Running training docker image.'

DATASET=cifar
MODEL=database/models/Model_cifar-original_resnet32_1_1018-1431.pth
BATCH=128
EPOCHS=200

docker run \
        -v /home/laude/compatibility_clustering_spectograms/database:/home/app/database \
        training:latest \
        -d $DATASET -m $MODEL -b $BATCH