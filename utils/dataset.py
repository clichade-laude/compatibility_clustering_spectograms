import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class PoisonDataset(ImageFolder):
    def __init__(self, root, clustering):
        img_size = 32 if "cifar" in root else 128
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        transform_list = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        if "cifar" in root:
            transform_list.append(normalize)
        transform = transforms.Compose(transform_list)
        super(PoisonDataset, self).__init__(root, transform, allow_empty=True)

        self.targets = np.array(self.targets)
        imgs_paths = np.array(self.imgs)[:, 0]
        self.imgs_names = np.array([x.split('/')[-1] for x in imgs_paths])

        if clustering:
            self.load_clean()
        
    def obtain_poisoned(self):
        poison_info = np.load(os.path.join(self.root, 'poison_info.npz'))
        self.source = list(poison_info.keys())[0]

        clean_samples = np.isin(self.imgs_names, poison_info[self.source])
        self.clean_samples = np.where(clean_samples == False)[0]

        self.true_targets = np.copy(self.targets)
        self.true_targets[clean_samples] = self.class_to_idx[self.source]

    def export_clean(self, clean):
        np.save(os.path.join(self.root, 'clean_samples.npy'), self.imgs_names[clean])

    def load_clean(self):
        clean_samples = np.load(os.path.join(self.root, 'clean_samples.npy'))
        clean_imgs = np.isin(self.imgs_names, clean_samples)

        self.imgs_names = self.imgs_names[clean_imgs]
        self.targets = self.targets[clean_imgs]
        self.imgs = np.array(self.imgs)[clean_imgs].tolist()
        self.samples = np.array(self.samples, dtype='O,O')[clean_imgs].tolist() ## dtype='O,O' necessary to mantain the format [(str, int)]

def load_data(root, batch_size=1, clustering=False, training=True):
    dataset = PoisonDataset(root, clustering)
    dataloader = DataLoader(dataset, batch_size, shuffle=training, num_workers=2)
    return dataset, dataloader