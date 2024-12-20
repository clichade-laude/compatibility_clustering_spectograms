#!/bin/bash

# Lista de carpetas
folders=("clustering" "controller" "executer" "monitoring" "poisoning" "preparation" "testing" "training")

# Ejecutar build.sh en paralelo para cada carpeta
printf "%s\n" "${folders[@]}" | xargs -n 1 -P 8 ./build.sh
