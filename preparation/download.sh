#!/bin/bash

pip3 install cifar2png
cifar2png cifar10 database/original/cifar
python3 preparation/rename.py 
python3 preparation/backdoor.py