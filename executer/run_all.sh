#!/bin/bash

# Array con los valores de poison
poison_values=(0.35 0.4)

# Ruta base del archivo
base_path="database/backdoor/backdoor_0-2_"

# Sufijo del archivo
suffix="_1-32.pickle"

# Comando base
base_command="docker run -v ./database:/home/app/database --network host executer:latest --dataset cifar --model resnet32 --epochs 200 --batch 128"

# Iterar sobre los valores de poison
for poison in "${poison_values[@]}"; do
        
    # Esperar 12 horas (43200 segundos) antes de ejecutar el comando
    echo "Esperando 12 horas antes de la iteración con poison=${poison}..."
    sleep 43200

    # Construir el comando completo
    full_command="$base_command --poison ${base_path}${poison}${suffix}"
    
    # Ejecutar el comando
    echo "Ejecutando: $full_command"
    $full_command


done
