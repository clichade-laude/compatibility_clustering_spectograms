#!/bin/bash

# Verifica si se ha proporcionado una carpeta como argumento
if [ -z "$1" ]; then
  echo "Por favor, proporciona una carpeta."
  exit 1
fi

# Obtiene el nombre de la carpeta
FOLDER_NAME=$(basename "$1")

# Construye la imagen Docker
docker build -f "$FOLDER_NAME/Dockerfile" -t "$FOLDER_NAME" .