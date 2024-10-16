import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class PoisonDataset(ImageFolder):
    def __init__(self, root):
        transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])
        super(PoisonDataset, self).__init__(root, transform)

        self.targets = np.array(self.targets)
        imgs_paths = np.array(self.imgs)[:, 0]
        self.imgs_names = np.array([x.split('/')[-1] for x in imgs_paths])
        
    def obtain_poisoned(self):
        poison_info = np.load(os.path.join(self.root, 'poison_info.npz'))
        self.source = list(poison_info.keys())[0]

        clean_samples = np.isin(self.imgs_names, poison_info[self.source])
        self.clean_samples = np.where(clean_samples == False)[0]

    def export_clean(self, clean):
        np.save(os.path.join(self.root, 'clean_samples.npy'), self.imgs_names[clean])

def load_data(root, batch_size):
    dataset = PoisonDataset(root)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    return dataset, dataloader