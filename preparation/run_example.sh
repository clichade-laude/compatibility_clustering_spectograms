#!/bin/bash

#Assemble docker image. 
echo 'Running preparation docker image.'


docker run \
        -v ./database:/home/app/database \
        preparation:latest 
