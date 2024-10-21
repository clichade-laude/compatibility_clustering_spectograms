#!/bin/bash

pip3 install cifar2png
cifar2png cifar10 database/original/cifar
python3 utils/rename.py 