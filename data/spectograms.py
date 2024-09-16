import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define Spectogram Dataset
class SpectogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): Lista de rutas a las imágenes.
            labels (list): Lista de etiquetas (0 para clase 1, 1 para clase 2, etc.).
            transform (callable, optional): Transformaciones aplicadas a las imágenes.
        """
        self.image_paths = image_paths  # Lista de rutas de imágenes
        self.labels = labels  # Lista de etiquetas correspondientes
        self.transform = transform  # Transformaciones a aplicar a las imágenes

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),  # Redimensionar las imágenes a 32x32 píxeles
                transforms.ToTensor(),  # Convertir las imágenes a tensores
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización entre -1 y 1
            ])


    def __len__(self):
        return len(self.image_paths)  # Retorna el número total de imágenes

    def __getitem__(self, idx):
        # Cargar la imagen desde la ruta
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Asegurar que sea RGB

        # Obtener la etiqueta de la imagen correspondiente
        label = self.labels[idx]

        # Aplicar transformaciones si están definidas
        if self.transform:
            image = self.transform(image)

        # Retorna la imagen y la etiqueta
        return image, label
    
    def __str__(self):
        # Obtener un resumen del dataset
        num_samples = len(self.image_paths)
        unique_labels = set(self.labels)
        class_distribution = {label: self.labels.count(label) for label in unique_labels}
        example_image_path = self.image_paths[0]

        # Crear la cadena informativa
        info = (f"SpectogramDataset with {num_samples} samples\n"
                f"Classes: {unique_labels}\n"
                f"Class distribution: {class_distribution}\n"
                f"Example image path: {example_image_path}\n")
        return info


def LoadImages():
    image_paths = []
    labels = []
    # Clase 1
    class1_dir = 'datasets/clean'
    for img_name in os.listdir(class1_dir):
        image_paths.append(os.path.join(class1_dir, img_name))
        labels.append(0)  # Etiqueta para clase 1

    # Clase 2
    class2_dir = 'datasets/jammer'
    for img_name in os.listdir(class2_dir):
        image_paths.append(os.path.join(class2_dir, img_name))
        labels.append(1)  # Etiqueta para clase 2

    return image_paths, labels

# 3. Definir las transformaciones a aplicar (redimensionar, convertir a tensor, normalizar)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Redimensionar las imágenes a 32x32 píxeles
    transforms.ToTensor(),  # Convertir las imágenes a tensores
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización entre -1 y 1
])


image_paths, labels = LoadImages()
# Crear el dataset con las rutas de imágenes, etiquetas y transformaciones
dataset = SpectogramDataset(image_paths=image_paths, labels=labels, transform=transform)


def Spectogram_Loader(path=None, batch_size=4, train=True, oracle=False, augment=True, 
        poison=True, dataset=None):
    
    # if dataset is None:
    #     transform = train_transform if train and augment else test_transform

    #     if path == "clean": poison = False
    #     if oracle and train: poison = False
    #     path = path if poison else None

    #     dataset = PoisonDataset(root='datasets', train=train, 
    #         transform=transform, download=True, poison_params=path)

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=train,  # Shuffle data between epoch
                            num_workers=2,
                            pin_memory=True)  # Paralelizar el proceso de carga de datos con 2 workers
    
    return dataset, dataloader
    

# def cifar10_loader(path, batch_size=128, train=True, oracle=False, augment=True, 
#         poison=True, dataset=None):

#     if dataset is None:
#         transform = train_transform if train and augment else test_transform

#         if path == "clean": poison = False
#         if oracle and train: poison = False
#         path = path if poison else None

#         dataset = PoisonDataset(root='datasets', train=train, 
#             transform=transform, download=True, poison_params=path)

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=128, shuffle=train and augment,
#         num_workers=2, pin_memory=True)
#     return dataset, dataloader



# # 6. Iterar sobre el DataLoader
# # Este ciclo representa cómo entrenarías tu IA usando este DataLoader
# for epoch in range(2):  # Ejemplo de dos épocas de entrenamiento
#     print(f'Epoch {epoch+1}')
#     for batch_idx, (images, labels) in enumerate(data_loader):
#         print(f'Lote {batch_idx+1}')
#         print(f'  - Tamaño del lote: {images.size()}')  # Muestra (batch_size, canales, alto, ancho)
#         print(f'  - Etiquetas: {labels}')  # Muestra las etiquetas del lote