import pickle
import os
import numpy as np
from torchvision import datasets
from PIL import Image
import copy

class PoisonDataset(datasets.ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None, poison_params=None):
        super(PoisonDataset, self).__init__(root, transform, target_transform)
        self.train = train
        self.clean_samples = None 
        self.method = None
        self.position = None
        self.color = None
        self.fraction_poisoned = None
        self.poison_seed = None
        self.source = None
        self.target = None
        self.true_targets = np.array(self.targets)
        self.targets = np.array(self.targets)

        self.get_info(poison_params)
        self.load_data()

        if poison_params is not None:    
            with open(f'{poison_params}', 'rb') as f:
                params = pickle.load(f)

            self.method = params['method']
            self.position = params['position']
            self.color = params['color']
            self.fraction_poisoned = float(params['fraction_poisoned'])
            self.poison_seed = params['seed']
            self.source = params['source']
            self.target = params['target']

            self.poison()

    def get_info(self, poison_params):
        print("Subset: train") if "train" in self.root else print("Subset: test")
        print("\t Total samples:", self.targets.size)
        print("\t\t Clean samples:", self.targets.size-np.sum(self.targets))
        print("\t\t Jammer samples:", np.sum(self.targets))
        print("\t Clean.") if not poison_params else print("\t Poisoned:")

    def load_data(self):
        self.data = []
        for index in range(len(self.imgs)):
            sample, target = self.__getitem__(index)
            self.data.append(sample)
        self.data = np.vstack(self.data).reshape(-1, 3, 128, 128)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def poison(self):
        method = self.method
        position = self.position
        color = self.color
        fraction_poisoned = self.fraction_poisoned
        seed = self.poison_seed
        source = self.source
        target = self.target

        class_idxs = np.where(np.array(self.targets) == source)[0]
        poison_count = int(fraction_poisoned * class_idxs.shape[0])
        if self.train:
            st = np.random.get_state()
            np.random.seed(seed)
            poisoned_idxs = \
                    np.random.choice(class_idxs, size=poison_count, replace=False)
            np.random.set_state(st)
        else:
            poisoned_idxs = class_idxs

        for i in poisoned_idxs:
            image = self.data[i]
            poisoned_image = poison_image(image, method, position, color)
            self.data[i] = poisoned_image
            self.targets[i] = target

        poisoned_images = np.isin(np.arange(len(self.targets)), poisoned_idxs)
        self.clean_samples = np.where(poisoned_images == 0)[0]

        print("\t\t Poisoned class:", self.classes[source])
        print("\t\t Poisoned percentage:", self.fraction_poisoned)
        print("\t\t Total/Class poisoned images:", poisoned_idxs.size)
        print("\t\t Class clean images:", class_idxs.size-poisoned_idxs.size)
        print("\t\t Total clean images:", len(self.targets) - poisoned_idxs.size)


def poison_image(image, method, position, color):
    """
    adapted from https://github.com/MadryLab/backdoor_data_poisoning/blob/master/dataset_input.py
    method = "pixel" or "pattern" or "ell"
    """
    poisoned = np.copy(image)
    col_arr = np.asarray(color)

    if method == 'pixel':
        poisoned[position[0], position[1], :] = col_arr
    elif method == 'pattern':
        poisoned[position[0], position[1], :] = col_arr
        poisoned[position[0] + 1, position[1] + 1, :] = col_arr
        poisoned[position[0] - 1, position[1] + 1, :] = col_arr
        poisoned[position[0] + 1, position[1] - 1, :] = col_arr
        poisoned[position[0] - 1, position[1] - 1, :] = col_arr
    elif method == 'ell':
        poisoned[position[0], position[1], :] = col_arr
        poisoned[position[0] + 1, position[1], :] = col_arr
        poisoned[position[0], position[1] + 1, :] = col_arr
    return poisoned

