import os
import shutil

def copy_files(src_folder, dst_folder, num_copies):
    # Crear la carpeta de destino si no existe
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    
    # Iterar sobre todos los archivos en la carpeta de origen
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        
        # Verificar si es un archivo (no una carpeta)
        if os.path.isfile(src_file):
            # Copiar el archivo num_copies veces
            for i in range(1, num_copies + 1):
                dst_file = os.path.join(dst_folder, f"{os.path.splitext(filename)[0]}-{i}{os.path.splitext(filename)[1]}")
                shutil.copy2(src_file, dst_file)

# Ejemplo de uso
src_folder = '/workspace/compatibility_clustering_spectograms/database/original/cifar-two/train/bird'
dst_folder = '/workspace/compatibility_clustering_spectograms/database/original/cifar-augmented/train/bird'
num_copies = 5

copy_files(src_folder, dst_folder, num_copies)
