#!/bin/bash

#Assemble docker image. 
echo 'Running poisoning docker image.'

DATASET=cifar
POISON=database/backdoor/cifar-backdoor-0-to-2-0.1.pickle

docker run \
        -v /home/laude/compatibility_clustering_spectograms/database:/home/app/database \
        poisoning:latest \
        -d $DATASET -p $POISON