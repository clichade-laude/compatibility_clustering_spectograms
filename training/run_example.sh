#!/bin/bash

#Assemble docker image. 
echo 'Running training docker image.'

DATASET=database/poisoned/cifar-0.2
MODEL=resnet32
BATCH=128
EPOCHS=200

docker run \
        -v /home/laude/compatibility_clustering_spectograms/database:/home/app/database \
        training:latest \
        -d $DATASET -m $MODEL -b $BATCH -e $EPOCHS \
        # --cluster